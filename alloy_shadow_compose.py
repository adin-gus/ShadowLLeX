#!/usr/bin/env python3
"""
alloy_shadow_compose.py — Shadow LLM Composer (zstd version)

Assembles a new GGUF model from records in the shadow library.
No source models required at compose time — everything comes from
the library built by alloy_shadow_extract.py.

This is the inverse of extraction:
    Extract:  GGUF → Shadow Library (decompose)
    Compose:  Shadow Library → GGUF (recompose)

The composer takes a composition spec — a description of what
you want the output model to contain — queries the library,
retrieves exact tensor bytes, and assembles them into a valid GGUF
using a base model as the structural template (metadata, architecture).

Composition spec format:
    {
      "name": "my_custom_model",
      "base_model": "path/to/base.gguf",   # provides architecture/metadata
      "tiers": {
        "KNOWLEDGE": {
          "source_model": "RYS-Qwen-27B",
          "depth_min": 0.25,
          "depth_max": 0.75,
          "ila_max": 0.3
        },
        "REASONING": {
          "source_model": "RYS-Qwen-27B",
          "depth_min": 0.10,
          "depth_max": 0.60
        }
        // tiers not specified → taken from base model unchanged
      }
    }

Usage:
    python alloy_shadow_compose.py ^
        --library  shadow_library ^
        --spec     compose_spec.json ^
        --out      my_custom_model.gguf

Or reassemble a model (sanity test):
    python alloy_shadow_compose.py ^
        --library  shadow_library ^
        --reassemble Vital-Mistral7B ^
        --base     vital-mistral7b.gguf ^
        --out      rebuilt.gguf

Or interactively:
    python alloy_shadow_compose.py ^
        --library  shadow_library ^
        --interactive ^
        --base     Mistral-7B.gguf ^
        --out      my_custom_model.gguf
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from gguf import GGMLQuantizationType

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from alloy_shadow_extract import (
    ShadowLibrary, ShadowRecord, Tier,
    classify_function, extract_layer_idx, get_relative_depth,
    TIER_DESCRIPTIONS,
)


# ============================================================
# Composition plan
# ============================================================

@dataclass
class TierSpec:
    """What to pull from the library for one tier."""
    tier:         str
    source_model: Optional[str]  = None   # None = any model
    depth_min:    Optional[float] = None
    depth_max:    Optional[float] = None
    ila_min:      Optional[float] = None
    ila_max:      Optional[float] = None
    kurt_min:     Optional[float] = None
    kurt_max:     Optional[float] = None
    function:     Optional[str]  = None   # attention/mlp/norm/etc


@dataclass 
class CompositionPlan:
    """Full composition: what goes where."""
    name:       str
    base_gguf:  str              # structural template
    tier_specs: Dict[str, TierSpec]
    # Resolved at plan time:
    selected:   Dict[str, List[ShadowRecord]] = None  # tier → records


def resolve_plan(
    library:    ShadowLibrary,
    base_gguf:  str,
    tier_specs: Dict[str, TierSpec],
    n_layers:   int,
    name:       str = "composed_model",
) -> CompositionPlan:
    """
    Query the library for each tier spec and build the composition plan.
    For each base layer, determine which record fills it.
    """
    selected: Dict[str, List[ShadowRecord]] = {}

    for tier_name, spec in tier_specs.items():
        records = library.query(
            tier      = tier_name,
            model     = spec.source_model,
            function  = spec.function,
            depth_min = spec.depth_min,
            depth_max = spec.depth_max,
            ila_min   = spec.ila_min,
            ila_max   = spec.ila_max,
            kurt_min  = spec.kurt_min,
            kurt_max  = spec.kurt_max,
        )
        selected[tier_name] = records
        print(f"  {tier_name:<14}: {len(records)} records from library")

    return CompositionPlan(
        name       = name,
        base_gguf  = base_gguf,
        tier_specs = tier_specs,
        selected   = selected,
    )


# ============================================================
# Tensor routing (with shape validation)
# ============================================================

def build_tensor_route(
    plan:          CompositionPlan,
    base_tensors:  Dict[str, object],  # name → GGUFTensor
    n_layers:      int,
) -> Dict[str, Optional[ShadowRecord]]:
    """
    For each tensor in the base model, decide whether to replace it
    with a library record or keep the base tensor.

    Returns dict of tensor_name → ShadowRecord (or None = use base).
    """
    # Build lookup: (tier, layer_depth_bucket) → best record
    # We match by function class and approximate depth
    tier_records: Dict[str, List[ShadowRecord]] = plan.selected or {}

    # For each tier, build depth-sorted list for nearest-match lookup
    tier_by_depth: Dict[str, List[ShadowRecord]] = {}
    for tier_name, records in tier_records.items():
        tier_by_depth[tier_name] = sorted(
            records, key=lambda r: r.layer_depth
        )

    route: Dict[str, Optional[ShadowRecord]] = {}

    for tensor_name, tensor in base_tensors.items():
        layer_idx   = extract_layer_idx(tensor_name, n_layers)
        layer_depth = get_relative_depth(layer_idx, n_layers)
        fn_class    = classify_function(tensor_name)

        # Determine which tier this tensor belongs to in the base model
        tensor_tier = _infer_tier(tensor_name, fn_class, layer_depth)

        # Check if we have library records for this tier
        candidates = tier_by_depth.get(tensor_tier, [])
        if not candidates:
            route[tensor_name] = None  # use base
            continue

        # Find closest depth match with matching function class
        best = _find_best_match(
            tensor_name, fn_class, layer_depth, candidates
        )
        route[tensor_name] = best

    return route


def _infer_tier(tensor_name: str, fn_class: str, depth: float) -> str:
    """Infer which tier a base model tensor belongs to."""
    if fn_class in ("embedding", "output"):
        return "CROWN"
    if fn_class in ("norm", "rope", "ssm"):
        return "FOUNDATION"
    if depth < 0.0:
        return "FOUNDATION"
    if fn_class == "mlp":
        return "KNOWLEDGE"
    if fn_class == "attention":
        if "attn_k" in tensor_name or "attn_v" in tensor_name:
            return "CONTEXT"
        if depth < 0.65:
            return "REASONING"
        return "STYLE"
    return "REASONING"


def _find_best_match(
    tensor_name: str,
    fn_class:    str,
    depth:       float,
    candidates:  List[ShadowRecord],
) -> Optional[ShadowRecord]:
    """
    Find the library record that best matches this tensor's
    function class and relative depth position.
    """
    # Filter to matching function class
    fn_matches = [r for r in candidates if r.function_class == fn_class]
    if not fn_matches:
        fn_matches = candidates  # relax constraint if no fn match

    # Find nearest depth
    best = min(fn_matches, key=lambda r: abs(r.layer_depth - depth))
    return best


def _shapes_match(record_shape: tuple, base_shape: tuple) -> bool:
    """Check if two tensor shapes are compatible for replacement."""
    if record_shape == base_shape:
        return True
    
    # Allow transposition? No - GGUF is strict
    # Allow broadcasting? No - would break model
    
    return False


# ============================================================
# GGUF assembly
# ============================================================

def _to_byte_shape(shape, tensor_type):
    """Convert element-count shape to byte-count shape for GGUFWriter.add_tensor."""
    try:
        from gguf.quants import GGML_QUANT_SIZES
        from gguf import GGMLQuantizationType
        qt = GGMLQuantizationType(int(tensor_type))
        if qt in GGML_QUANT_SIZES:
            block_size, type_size = GGML_QUANT_SIZES[qt]
            return list(shape[:-1]) + [(shape[-1] // block_size) * type_size]
    except Exception:
        pass
    return list(shape)


def run_compose(
    library_dir: str,
    base_gguf:   str,
    tier_specs:  Dict[str, TierSpec],
    out_path:    str,
    model_name:  str = "shadow_composed",
):
    """
    Compose a new GGUF from shadow library records + base model template.

    Base model provides:
      - Architecture metadata (KV fields)
      - Structural template for tensor shapes/types
      - Any tensors not covered by tier specs

    Shadow library provides:
      - Tensor bytes for specified tiers
      - Exact lossless data from source models (zstd compressed)
    """
    try:
        from gguf import GGUFReader, GGUFWriter
    except ImportError:
        print("[ERROR] pip install gguf")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  SHADOW LLM COMPOSER (zstd)")
    print("  Assembling model from shadow library")
    print("=" * 60)
    print(f"\n  Base:    {base_gguf}")
    print(f"  Library: {library_dir}")
    print(f"  Output:  {out_path}")
    print(f"  Name:    {model_name}")

    # Load library
    print(f"\n  Loading shadow library...")
    lib = ShadowLibrary(library_dir)
    stats = lib.stats()
    print(f"  Models in library: {len(stats['models'])}")
    print(f"  Total records:     {stats['total_records']}")
    if stats['models']:
        print(f"  Available:         {', '.join(stats['models'])}")

    # Open base model
    print(f"\n  Opening base model...")
    reader_b    = GGUFReader(base_gguf)
    base_tensors = {t.name: t for t in reader_b.tensors}

    # Get layer count from base
    n_layers = 0
    for k, field in reader_b.fields.items():
        if k.endswith("block_count"):
            try:
                v = field.parts[-1]
                n_layers = int(v[0]) if hasattr(v, "__len__") else int(v)
                break
            except Exception:
                pass
    print(f"  Base tensors: {len(base_tensors)}  Layers: {n_layers}")

    # Resolve tier specs against library
    print(f"\n  Querying library for tier specs...")
    plan = resolve_plan(
        library    = lib,
        base_gguf  = base_gguf,
        tier_specs = tier_specs,
        n_layers   = n_layers,
        name       = model_name,
    )

    # Build tensor routing table
    print(f"\n  Building tensor route...")
    route = build_tensor_route(plan, base_tensors, n_layers)

    n_from_library = sum(1 for r in route.values() if r is not None)
    n_from_base    = sum(1 for r in route.values() if r is None)
    print(f"  From library: {n_from_library} tensors")
    print(f"  From base:    {n_from_base} tensors")

    # Confirm
    print()
    try:
        confirm = input("  Proceed with assembly? (y/n) [y]: ").strip().lower()
    except EOFError:
        confirm = "y"
    if confirm == "n":
        print("  Aborted.")
        return

    # Write output GGUF
    print(f"\n  Assembling...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # Get arch string
    arch = "llama"
    for k, field in reader_b.fields.items():
        if k == "general.architecture":
            try:
                arch = bytes(field.parts[-1]).decode("utf-8").rstrip("\x00")
            except Exception:
                pass
            break

    writer = GGUFWriter(out_path, arch=arch, use_temp_file=True)

    # Copy metadata from base
    for key in reader_b.fields:
        if key in ("general.architecture", "general.file_type"):
            continue
        if key.startswith("GGUF."):
            continue
        try:
            field = reader_b.fields[key]
            writer.add_key_value(key, field.parts[-1], field.types[-1])
        except Exception:
            pass

    # Write tensors in base model order (critical for llama.cpp)
    from tqdm import tqdm
    n_library = 0
    n_base    = 0
    n_failed  = 0
    n_shape_mismatch = 0

    for tensor_b in tqdm(reader_b.tensors, desc="  Tensors"):
        name   = tensor_b.name
        record = route.get(name)

        # Get base tensor shape (GGUF stores shape reversed)
        base_shape = tuple(reversed(tensor_b.shape))

        if record is not None:
            # Get data from library (zstd decompressed)
            data = lib.decode(record)

            if data is not None:
                # Check shape compatibility FIRST
                record_shape = tuple(record.shape)
                
                if not _shapes_match(record_shape, base_shape):
                    # Shape mismatch - cannot use this record
                    n_shape_mismatch += 1
                    # Fall through to base tensor
                elif len(data) == len(bytes(tensor_b.data)):
                    # Shape matches AND byte size matches - use library tensor
                    try:
                        dtype = GGMLQuantizationType(int(record.tensor_type))
                        bshape = _to_byte_shape(list(reversed(tensor_b.shape)), 
                                                int(tensor_b.tensor_type))
                        writer.add_tensor(
                            name,
                            np.frombuffer(data, dtype=np.uint8),
                            raw_shape=bshape,
                            raw_dtype=GGMLQuantizationType(int(tensor_b.tensor_type)),
                        )
                        n_written += 1
                    except Exception as e:
                        print(f"\n  [WARN] Failed to write {name}: {e}")
                        n_failed += 1
                        continue
                # else: byte size mismatch - use base

        # Use base tensor (either no record, shape mismatch, or size mismatch)
        try:
            base_data  = bytes(tensor_b.data)
            dtype      = GGMLQuantizationType(int(tensor_b.tensor_type))
            bshape     = _to_byte_shape(list(reversed(tensor_b.shape)),
                                        int(tensor_b.tensor_type))
            writer.add_tensor(
                name,
                np.frombuffer(base_data, dtype=np.uint8),
                raw_shape=bshape,
                raw_dtype=dtype,
            )
            n_base += 1
        except Exception as e:
            print(f"\n  [WARN] Failed to write {name}: {e}")
            n_failed += 1

    # Finalize
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size_gb = os.path.getsize(out_path) / 1024**3

    print(f"\n  From library:      {n_library} tensors")
    print(f"  From base:         {n_base} tensors")
    if n_shape_mismatch:
        print(f"  Shape mismatches:  {n_shape_mismatch} (used base instead)")
    if n_failed:
        print(f"  Failed:            {n_failed} tensors")
    print(f"  Output: {out_path} ({size_gb:.2f} GB)")

    # Write composition report
    report = {
        "model_name":     model_name,
        "base_gguf":      base_gguf,
        "library_dir":    library_dir,
        "output":         out_path,
        "size_gb":        round(size_gb, 2),
        "n_from_library": n_library,
        "n_from_base":    n_base,
        "n_shape_mismatch": n_shape_mismatch,
        "tier_specs":     {k: {
            "source_model": v.source_model,
            "depth_min":    v.depth_min,
            "depth_max":    v.depth_max,
            "ila_min":      v.ila_min,
            "ila_max":      v.ila_max,
        } for k, v in tier_specs.items()},
        "tensor_route": {
            name: {
                "source":       "library",
                "record_id":    r.record_id,
                "source_model": r.source_model,
                "tier":         r.tier,
                "depth":        r.layer_depth,
                "shape":        r.shape,
            } if r else {"source": "base"}
            for name, r in route.items()
        }
    }

    report_path = out_path.replace(".gguf", "_compose_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report: {report_path}")
    print()
    print("=" * 60)
    print("  COMPOSITION COMPLETE")
    print("=" * 60)


# ============================================================
# Reassemble mode (sanity test)
# ============================================================

def run_reassemble(
    library_dir: str,
    model_name:  str,
    base_gguf:   str,
    out_path:    str,
):
    """
    Reassemble a model from the library using all its own tiers.
    This is a sanity test - the output should be identical to the original.
    """
    print()
    print("=" * 60)
    print("  SHADOW LIBRARY REASSEMBLE (Sanity Test)")
    print("  Reassembling model from its own extracted tensors")
    print("=" * 60)
    print(f"\n  Model:   {model_name}")
    print(f"  Library: {library_dir}")
    print(f"  Base:    {base_gguf}")
    print(f"  Output:  {out_path}")

    # Build a spec that uses all tiers from this model
    tier_specs = {}
    for tier in Tier:
        tier_specs[tier.value] = TierSpec(
            tier=tier.value,
            source_model=model_name,
        )

    print(f"\n  Reassembling using all {len(tier_specs)} tiers from {model_name}")
    
    # Run normal compose with the auto-generated spec
    run_compose(
        library_dir=library_dir,
        base_gguf=base_gguf,
        tier_specs=tier_specs,
        out_path=out_path,
        model_name=f"{model_name}_reassembled",
    )


# ============================================================
# Interactive mode
# ============================================================

def run_interactive(
    library_dir: str,
    base_gguf:   str,
    out_path:    str,
    model_name:  str = "shadow_composed",
):
    """Interactive tier spec builder."""
    lib   = ShadowLibrary(library_dir)
    stats = lib.stats()

    print()
    print("  Shadow library contents:")
    for model in stats["models"]:
        tiers = lib.tiers(model)
        print(f"  {model}:")
        for tier, count in tiers.items():
            print(f"    {tier:<14}: {count} tensors")
    print()

    tier_specs: Dict[str, TierSpec] = {}

    print("  For each tier, specify what to pull from the library.")
    print("  Press Enter to skip a tier (uses base model).")
    print()

    for tier in Tier:
        desc = TIER_DESCRIPTIONS[tier]
        print(f"  [{tier.value}] {desc}")
        try:
            use = input(f"  Pull from library? (y/n) [n]: ").strip().lower()
        except EOFError:
            use = "n"

        if use != "y":
            print()
            continue

        try:
            src = input(f"  Source model (Enter for any): ").strip() or None
            dlo = input(f"  Depth min (Enter for none): ").strip()
            dhi = input(f"  Depth max (Enter for none): ").strip()
            ila_min = input(f"  ILA min (Enter for none): ").strip()
            ila_max = input(f"  ILA max (Enter for none): ").strip()
        except EOFError:
            src = dlo = dhi = ila_min = ila_max = ""

        spec = TierSpec(
            tier         = tier.value,
            source_model = src or None,
            depth_min    = float(dlo) if dlo else None,
            depth_max    = float(dhi) if dhi else None,
            ila_min      = float(ila_min) if ila_min else None,
            ila_max      = float(ila_max) if ila_max else None,
        )
        tier_specs[tier.value] = spec

        # Show what we'd get
        preview = lib.query(
            tier      = tier.value,
            model     = spec.source_model,
            depth_min = spec.depth_min,
            depth_max = spec.depth_max,
            ila_min   = spec.ila_min,
            ila_max   = spec.ila_max,
        )
        print(f"  → {len(preview)} matching records")
        print()

    if not tier_specs:
        print("  No tiers selected — output would be identical to base. Aborting.")
        return

    run_compose(
        library_dir = library_dir,
        base_gguf   = base_gguf,
        tier_specs  = tier_specs,
        out_path    = out_path,
        model_name  = model_name,
    )


def compose_from_blueprint(
    library_dir: str,
    model_name: str,
    out_path: str,
):
    """
    Compose a GGUF using only the library blueprint (no base model).
    """
    try:
        from gguf import GGUFWriter
    except ImportError:
        print("[ERROR] pip install gguf")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("  SHADOW LLM COMPOSER (Blueprint Mode)")
    print("  No base model required")
    print("=" * 60)
    
    # Load library
    lib = ShadowLibrary(library_dir)
    
    # Load blueprint
    model_dir = os.path.join(library_dir, model_name)
    blueprint_path = os.path.join(model_dir, "blueprint.json")
    
    if not os.path.exists(blueprint_path):
        print(f"[ERROR] Blueprint not found: {blueprint_path}")
        sys.exit(1)
    
    with open(blueprint_path) as f:
        blueprint = json.load(f)
    
    print(f"\n  Model: {model_name}")
    print(f"  Architecture: {blueprint['architecture']}")
    print(f"  Tensors: {len(blueprint['tensor_order'])}")
    
    # Build tensor lookup from library
    # source_model in records may differ from model_name if extraction used
    # a different name. Use blueprint's stored model_name as authoritative.
    blueprint_model_name = blueprint.get("model_name", model_name)
    
    # Try exact match first, then partial match
    all_records = {r.tensor_name: r for r in lib._index 
                   if r.source_model == blueprint_model_name}
    
    if not all_records:
        # Try matching by directory name (safe_name version)
        def _safe_name(n):
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in n)
        safe_model = _safe_name(blueprint_model_name)
        all_records = {r.tensor_name: r for r in lib._index
                       if _safe_name(r.source_model) == safe_model
                       or r.source_model == safe_model}
    
    if not all_records:
        # Last resort: any records from this model directory
        # Read source_model from first record file directly
        for tier in ["KNOWLEDGE", "REASONING", "CROWN", "FOUNDATION"]:
            rpath = os.path.join(model_dir, tier, f"{tier}_records.json")
            if os.path.exists(rpath):
                try:
                    with open(rpath) as f:
                        recs = json.load(f)
                    if recs:
                        actual_name = recs[0].get("source_model", "")
                        if actual_name:
                            all_records = {r.tensor_name: r for r in lib._index
                                           if r.source_model == actual_name}
                            if all_records:
                                print(f"  Using source_model: {actual_name!r}")
                                break
                except Exception:
                    pass
    
    print(f"  Records found: {len(all_records)}")
    if not all_records:
        print(f"  [WARN] No records found for model {blueprint_model_name!r}")
        print(f"  Available models: {lib.models()}")
    
    # Create writer — handle arch stored as list of bytes or string
    raw_arch = blueprint["architecture"]
    if isinstance(raw_arch, list):
        try:
            arch = bytes(raw_arch).decode("utf-8").rstrip("\x00")
        except Exception:
            arch = "llama"
    elif isinstance(raw_arch, str) and raw_arch and raw_arch != "unknown":
        arch = raw_arch
    else:
        # Fall back: check config dict for architecture
        cfg_arch = blueprint.get("config", {}).get("general.architecture", "")
        if isinstance(cfg_arch, list):
            try:
                arch = bytes(cfg_arch).decode("utf-8").rstrip("\x00")
            except Exception:
                arch = "llama"
        elif isinstance(cfg_arch, str) and cfg_arch:
            arch = cfg_arch
        else:
            arch = "llama"
    print(f"  Architecture: {arch}")
    writer = GGUFWriter(out_path, arch=arch, use_temp_file=True)
    
    # Write config from blueprint
    for key, value in blueprint["config"].items():
        if key.startswith("GGUF."):
            continue
        
        # Handle different types for GGUFWriter
        # GGUF spec uses uint32 for virtually all integer hyperparameters
        if key in ("general.architecture",):
            continue  # already set by GGUFWriter constructor
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            # Try uint32 first (correct for almost all GGUF int fields)
            # Fall back to uint64 if value is too large
            try:
                if value < 0:
                    writer.add_int32(key, value)
                elif value <= 0xFFFFFFFF:
                    writer.add_uint32(key, value)
                else:
                    writer.add_uint64(key, value)
            except Exception:
                writer.add_int64(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_string(key, str(value))
    
    # Write tensors in blueprint order
    from tqdm import tqdm
    n_written = 0
    n_missing = 0
    n_failed = 0
    
    for tensor_info in tqdm(blueprint["tensor_manifest"], desc="  Tensors"):
        name = tensor_info["name"]
        record = all_records.get(name)
        
        if record is None:
            n_missing += 1
            continue
        
        data = lib.decode(record)
        if data is None:
            n_missing += 1
            continue
        
        try:
            dtype  = GGMLQuantizationType(int(record.tensor_type))
            bshape = _to_byte_shape(list(tensor_info["shape"]),
                                    int(record.tensor_type))
            writer.add_tensor(
                name,
                np.frombuffer(data, dtype=np.uint8),
                raw_shape=bshape,
                raw_dtype=dtype,
            )
            n_written += 1
        except Exception as e:
            print(f"\n  [WARN] Failed to write {name}: {e}")
            n_failed += 1
    
    # Finalize
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    
    size_gb = os.path.getsize(out_path) / 1024**3
    print(f"\n  Written: {n_written} tensors")
    print(f"  Missing: {n_missing} tensors")
    if n_failed:
        print(f"  Failed: {n_failed} tensors")
    print(f"  Output: {out_path} ({size_gb:.2f} GB)")
    
    # Save composition report
    report = {
        "mode": "blueprint",
        "source_model": model_name,
        "output": out_path,
        "size_gb": round(size_gb, 2),
        "n_tensors_written": n_written,
        "n_missing": n_missing,
        "n_failed": n_failed,
    }
    
    report_path = out_path.replace(".gguf", "_compose_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  Report: {report_path}")
    print()
    print("=" * 60)
    print("  COMPOSITION COMPLETE")
    print("=" * 60)
# ============================================================
# Spec file mode
# ============================================================

def load_spec(spec_path: str) -> Tuple[str, Dict[str, TierSpec], str]:
    """Load composition spec from JSON file."""
    with open(spec_path) as f:
        spec = json.load(f)

    base_gguf  = spec["base_model"]
    model_name = spec.get("name", "shadow_composed")
    tier_specs: Dict[str, TierSpec] = {}

    for tier_name, cfg in spec.get("tiers", {}).items():
        tier_specs[tier_name] = TierSpec(
            tier         = tier_name,
            source_model = cfg.get("source_model"),
            depth_min    = cfg.get("depth_min"),
            depth_max    = cfg.get("depth_max"),
            ila_min      = cfg.get("ila_min"),
            ila_max      = cfg.get("ila_max"),
            kurt_min     = cfg.get("kurt_min"),
            kurt_max     = cfg.get("kurt_max"),
            function     = cfg.get("function"),
        )

    return base_gguf, tier_specs, model_name


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Shadow LLM Composer (zstd) — assemble GGUF from shadow library"
    )
    p.add_argument("--library",     required=True,  help="Shadow library directory")
    p.add_argument("--out",         required=True,  help="Output GGUF path")
    p.add_argument("--spec",        default=None,   help="Composition spec JSON")
    p.add_argument("--base",        default=None,   help="Base model GGUF (structural template)")
    p.add_argument("--name",        default=None,   help="Output model name")
    p.add_argument("--interactive", action="store_true", help="Interactive tier selection")
    p.add_argument("--list",        action="store_true", help="List library contents and exit")
    p.add_argument("--blueprint", default=None, help="Use blueprint from this model (no base needed)")
    p.add_argument("--reassemble",  default=None,   help="Reassemble a model from library (sanity test)")
    
    args = p.parse_args()
# ===== NEW: Blueprint/Reassemble mode (add this block) =====
    if args.reassemble:
        compose_from_blueprint(
            library_dir=args.library,
            model_name=args.reassemble,
            out_path=args.out,
        )
        return
    
    if args.blueprint:
        compose_from_blueprint(
            library_dir=args.library,
            model_name=args.blueprint,
            out_path=args.out,
        )
        return
    # ===== END NEW BLOCK ====

    # List mode
    if args.list:
        lib   = ShadowLibrary(args.library)
        stats = lib.stats()
        print(f"\nShadow library: {args.library}")
        print(f"Total records:  {stats['total_records']}")
        print(f"Models: {len(stats['models'])}")
        for model in stats["models"]:
            print(f"\n  {model}:")
            tiers = lib.tiers(model)
            for tier, count in tiers.items():
                print(f"    {tier:<14}: {count:>4} tensors")
        return

    # Reassemble mode (sanity test)
    if args.reassemble:
        if not args.base:
            print("[ERROR] --base required for reassemble mode (path to original .gguf)")
            sys.exit(1)
        if not os.path.exists(args.base):
            print(f"[ERROR] Base model not found: {args.base}")
            sys.exit(1)
        run_reassemble(
            library_dir = args.library,
            model_name  = args.reassemble,
            base_gguf   = args.base,
            out_path    = args.out,
        )
        return

    # Spec file mode
    if args.spec:
        if not os.path.exists(args.spec):
            print(f"[ERROR] Spec not found: {args.spec}")
            sys.exit(1)
        base_gguf, tier_specs, model_name = load_spec(args.spec)
        model_name = args.name or model_name
        run_compose(
            library_dir = args.library,
            base_gguf   = base_gguf,
            tier_specs  = tier_specs,
            out_path    = args.out,
            model_name  = model_name,
        )
        return

    # Interactive mode
    if args.interactive:
        if not args.base:
            print("[ERROR] --base required for interactive mode")
            sys.exit(1)
        if not os.path.exists(args.base):
            print(f"[ERROR] Base model not found: {args.base}")
            sys.exit(1)
        run_interactive(
            library_dir = args.library,
            base_gguf   = args.base,
            out_path    = args.out,
            model_name  = args.name or "shadow_composed",
        )
        return

    print("[ERROR] Specify --spec, --interactive, --reassemble, or --list")
    p.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()