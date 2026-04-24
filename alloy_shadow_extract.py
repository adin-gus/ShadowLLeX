#!/usr/bin/env python3
"""
alloy_shadow_extract.py — Clean zstd-based shadow library
No Memvid, no OpenViking. Just simple, fast, lossless storage with zstd.

Includes full classification + ShadowLibrary for query/compose.
"""

import os
import sys
import json
import struct
import hashlib
import datetime
import zstandard as zstd
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Tier definitions — the quaternion hierarchy
# ============================================================

class Tier(Enum):
    CROWN       = "CROWN"        # +magnitude: embeddings, identity
    INSTRUCTION = "INSTRUCTION"  # +phase:     ILA-sensitive
    REASONING   = "REASONING"    # -phase:     early-mid, winning ticket
    KNOWLEDGE   = "KNOWLEDGE"    # -magnitude: mid MLP, facts
    CONTEXT     = "CONTEXT"      # +rotation:  K/V, sequence memory
    STYLE       = "STYLE"        # -rotation:  late, output shaping
    FOUNDATION  = "FOUNDATION"   # +comp:      norms, rope, structure
    VOID        = "VOID"         # -comp:      holographic-quiet


TIER_DESCRIPTIONS = {
    Tier.CROWN:       "Embeddings and output projection — model's world interface",
    Tier.INSTRUCTION: "ILA-sensitive layers — model's self-knowledge and direction-following",
    Tier.REASONING:   "Early-mid attention, high kurtosis — where the model thinks",
    Tier.KNOWLEDGE:   "Mid MLP layers — factual associations and domain expertise",
    Tier.CONTEXT:     "K/V attention — how the model holds context across tokens",
    Tier.STYLE:       "Late layers — tone, verbosity, output formatting",
    Tier.FOUNDATION:  "Norms, rope, positional — structural skeleton beneath everything",
    Tier.VOID:        "Holographic-quiet — VSA superposition, distributed whispers",
}


# ============================================================
# Helper: convert numpy types to JSON serializable
# ============================================================

def make_json_safe(obj):
    """Recursively convert numpy/GGUF types to Python native types."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(i) for i in obj]
    return obj


# ============================================================
# Shadow record — one per tensor
# ============================================================

@dataclass
class ShadowRecord:
    """A classified, indexed tensor record."""
    # Identity
    record_id:      str
    source_model:   str
    tensor_name:    str
    extraction_date: str

    # Classification
    tier:           str
    function_class: str
    layer_idx:      int
    layer_depth:    float
    
    # Signal quality
    ila_score:      float
    kurtosis:       float
    super_scale_mean: float
    
    # Geometry
    shape:          List[int]
    tensor_type:    int
    tensor_type_name: str
    n_params:       int
    
    # Storage (zstd)
    data_path:      str


def make_record_id(model_name: str, tensor_name: str) -> str:
    h = hashlib.sha256(f"{model_name}:{tensor_name}".encode()).hexdigest()
    return h[:16]


# ============================================================
# Tier classification
# ============================================================

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16", 22: "IQ4_XS",
}


def classify_function(tensor_name: str) -> str:
    n = tensor_name.lower()
    if "token_embd" in n or "embed_tokens" in n:
        return "embedding"
    if "output.weight" in n and "norm" not in n:
        return "output"
    if "output_norm" in n or "norm" in n:
        return "norm"
    if "rope" in n or "freq" in n:
        return "rope"
    if "attn_q" in n or "attn_k" in n or "attn_v" in n or "attn_output" in n or "attn_qkv" in n:
        return "attention"
    if "ffn" in n or "fwd" in n or "mlp" in n or "gate" in n:
        return "mlp"
    if "ssm" in n or "mamba" in n:
        return "ssm"
    return "unknown"


def extract_layer_idx(tensor_name: str, n_layers: int) -> int:
    parts = tensor_name.split(".")
    if len(parts) >= 2 and parts[0] == "blk":
        try:
            return int(parts[1])
        except ValueError:
            pass
    return -1


def get_relative_depth(layer_idx: int, n_layers: int) -> float:
    if layer_idx < 0 or n_layers <= 0:
        return -1.0
    return layer_idx / max(n_layers - 1, 1)


def extract_scale_envelope(data: bytes, tensor_type: int) -> float:
    """Fast super_scale mean from first 32 superblocks."""
    SUPERBLOCK_SIZES = {10: 110, 11: 110, 12: 144, 13: 176, 14: 210}
    sb_size = SUPERBLOCK_SIZES.get(tensor_type, 0)
    if sb_size == 0:
        return 0.0
    try:
        n_sbs   = min(32, len(data) // sb_size)
        scales  = []
        for i in range(n_sbs):
            ss = abs(struct.unpack_from('<e', data, i * sb_size)[0])
            if ss > 1e-8:
                scales.append(ss)
        return float(np.mean(scales)) if scales else 0.0
    except Exception:
        return 0.0


def classify_tier(
    tensor_name:  str,
    data:         bytes,
    tensor_type:  int,
    shape:        tuple,
    layer_idx:    int,
    layer_depth:  float,
    n_layers:     int,
    ila_scores:   Dict[int, float],
    kurtosis_map: Dict[int, float],
) -> Tier:
    """
    Classify a tensor into one of the eight Shadow tiers.
    Maps to quaternion octahedron axes top to bottom.
    """
    fn = classify_function(tensor_name)

    # CROWN — world interface
    if fn in ("embedding", "output"):
        return Tier.CROWN

    # FOUNDATION — structural skeleton
    if fn in ("norm", "rope", "ssm"):
        return Tier.FOUNDATION

    # Non-layer tensors that aren't embedding/norm/rope
    if layer_idx < 0:
        return Tier.FOUNDATION

    ila     = ila_scores.get(layer_idx, 0.0)
    kurt    = kurtosis_map.get(layer_idx, 1.0)
    scale   = extract_scale_envelope(data, tensor_type)

    # INSTRUCTION — ILA-sensitive
    if ila > 0.4:
        return Tier.INSTRUCTION

    # VOID — holographic-quiet (low kurtosis, low scale)
    scale_threshold = 0.003
    if kurt < 0.6 and scale < scale_threshold and layer_depth > 0.0:
        return Tier.VOID

    # KNOWLEDGE — mid MLP
    if fn == "mlp" and 0.2 < layer_depth < 0.8:
        return Tier.KNOWLEDGE

    # CONTEXT — K/V attention (sequence memory)
    if fn == "attention" and (
        "attn_k" in tensor_name or "attn_v" in tensor_name
    ):
        return Tier.CONTEXT

    # REASONING — early-mid attention, high kurtosis
    if fn == "attention" and layer_depth < 0.65 and kurt >= 1.0:
        return Tier.REASONING

    # STYLE — late layers
    if layer_depth >= 0.65:
        return Tier.STYLE

    # Default to REASONING for unclassified attention/mlp
    return Tier.REASONING


# ============================================================
# SimpleTierWriter (zstd version)
# ============================================================

class SimpleTierWriter:
    """One tier = one folder with zstd-compressed tensors + clean JSON index"""
    def __init__(self, tier: Tier, tier_dir: str):
        self.tier = tier
        self.tier_dir = tier_dir
        self.tensors_dir = os.path.join(tier_dir, "tensors")
        os.makedirs(self.tensors_dir, exist_ok=True)
        self.records: List[ShadowRecord] = []
        # Uniform zstd — always .bin.zst, no extension ambiguity
        # Level 1: quantized types (incompressible — frame wrap only, ~free)
        # Level 3: float types (real redundancy, worth compressing)
        self.compressor_fast = zstd.ZstdCompressor(level=1)
        self.compressor_slow = zstd.ZstdCompressor(level=3)
        self.decompressor    = zstd.ZstdDecompressor()
        self.FLOAT_TYPES     = {0, 1, 30}  # F32, F16, BF16

    def add_tensor(self, record: ShadowRecord, raw_bytes: bytes):
        """Compress and store — always .bin.zst for archive uniformity."""
        compressor = (self.compressor_slow
                      if record.tensor_type in self.FLOAT_TYPES
                      else self.compressor_fast)
        compressed = compressor.compress(raw_bytes)
        data_path  = os.path.join(self.tensors_dir, f"{record.record_id}.bin.zst")
        with open(data_path, "wb") as f:
            f.write(compressed)
        record.data_path = data_path
        self.records.append(record)

    def finalize(self):
        """Write records index"""
        index_path = os.path.join(self.tier_dir, f"{self.tier.value}_records.json")
        with open(index_path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2, default=make_json_safe)


# ============================================================
# ShadowExtractHandler (classification + zstd storage)
# ============================================================

class ShadowExtractHandler:
    """
    Stream handler for alloy_stream_core.
    One pass through a GGUF — classifies, stores, AND builds blueprint.
    """

    def __init__(
        self,
        model_name:  str,
        out_dir:     str,
        n_layers:    int,
        ila_scores:  Optional[Dict[int, float]] = None,
        kurtosis_map: Optional[Dict[int, float]] = None,
    ):
        self.model_name   = model_name
        self.out_dir      = out_dir
        self.n_layers     = n_layers
        self.ila_scores   = ila_scores   or {}
        self.kurtosis_map = kurtosis_map or {}

        # One writer per tier
        model_dir = os.path.join(out_dir, _safe_name(model_name))
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

        self.writers: Dict[Tier, SimpleTierWriter] = {}
        for tier in Tier:
            tier_dir = os.path.join(model_dir, tier.value)
            os.makedirs(tier_dir, exist_ok=True)
            self.writers[tier] = SimpleTierWriter(tier, tier_dir)

        self._n_processed = 0
        self._stats: Dict[str, int] = {t.value: 0 for t in Tier}
        
        # BLUEPRINT ACCUMULATION
        self._blueprint_config: Dict[str, Any] = {}
        self._blueprint_tensors: List[Dict] = []
        self._tensor_order: List[str] = []

    def add_field(self, key: str, value: Any, field_type: int = 0) -> None:
        """Add a GGUF field to blueprint config."""
        import numpy as np
        # field.contents() returns proper Python types — just make JSON safe
        # Handle residual numpy arrays (fallback path from field.parts[-1])
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            try:
                decoded = bytes(value).decode("utf-8").rstrip("\x00")
                value = decoded if (decoded.isprintable() and len(decoded) < 512) else value.tolist()
            except Exception:
                value = make_json_safe(value)
        else:
            value = make_json_safe(value)
        self._blueprint_config[key] = value

    def accepts(self, tensor_name: str, tensor_type: int) -> bool:
        return True  # extract everything

    def process(
        self,
        name:        str,
        data:        bytes,
        tensor_type: int,
        shape:       tuple,
    ) -> None:
        layer_idx   = extract_layer_idx(name, self.n_layers)
        layer_depth = get_relative_depth(layer_idx, self.n_layers)
        fn_class    = classify_function(name)
        ila         = self.ila_scores.get(layer_idx, 0.0)
        kurt        = self.kurtosis_map.get(layer_idx, 1.0)
        scale       = extract_scale_envelope(data, tensor_type)
        n_params    = int(np.prod(shape)) if shape else 0

        tier = classify_tier(
            tensor_name  = name,
            data         = data,
            tensor_type  = tensor_type,
            shape        = shape,
            layer_idx    = layer_idx,
            layer_depth  = layer_depth,
            n_layers     = self.n_layers,
            ila_scores   = self.ila_scores,
            kurtosis_map = self.kurtosis_map,
        )

        record = ShadowRecord(
            record_id        = make_record_id(self.model_name, name),
            source_model     = self.model_name,
            tensor_name      = name,
            extraction_date  = datetime.datetime.now().isoformat(),
            tier             = tier.value,
            function_class   = fn_class,
            layer_idx        = int(layer_idx),
            layer_depth      = round(float(layer_depth), 4),
            ila_score        = round(float(ila), 4),
            kurtosis         = round(float(kurt), 4),
            super_scale_mean = round(float(scale), 8),
            shape            = [int(x) for x in shape],
            tensor_type      = int(tensor_type),
            tensor_type_name = QUANT_NAMES.get(int(tensor_type), f"TYPE_{tensor_type}"),
            n_params         = int(n_params),
            data_path        = "",
        )

        self.writers[tier].add_tensor(record, data)
        self._stats[tier.value] += 1
        self._n_processed += 1
        
        # BLUEPRINT: Record tensor manifest
        self._blueprint_tensors.append({
            "name": name,
            "tier": tier.value,
            "function": fn_class,
            "depth": round(layer_depth, 4),
            "shape": list(shape),
            "tensor_type": int(tensor_type),
            "tensor_type_name": QUANT_NAMES.get(int(tensor_type), f"TYPE_{tensor_type}"),
            "order": len(self._blueprint_tensors),
        })
        self._tensor_order.append(name)

    def save_blueprint(self, reader=None) -> Dict:
        """Save blueprint.json after all tensors processed."""
        
        # If reader provided, extract any missing architecture fields
        if reader:
            arch_fields = [
                "general.architecture",
                "llama.block_count", "llama.context_length", "llama.embedding_length",
                "llama.feed_forward_length", "llama.head_count", "llama.head_count_kv",
                "llama.attention.layer_norm_rms_epsilon", "llama.rope.dimension_count",
                "llama.rope.freq_base", "vocab_size",
            ]

            for key, field in reader.fields.items():
                if key not in self._blueprint_config:
                    if any(arch_field in key for arch_field in arch_fields) or key.startswith("llama."):
                        try:
                            try:
                                value = field.contents()
                            except Exception:
                                value = field.parts[-1]
                            value = make_json_safe(value)
                            self._blueprint_config[key] = value
                        except Exception:
                            pass
        
        # Make sure config is fully JSON safe
        safe_config = make_json_safe(self._blueprint_config)
        
        # Make sure tensor manifest is JSON safe
        safe_manifest = make_json_safe(self._blueprint_tensors)
        
        blueprint = {
            "model_name": self.model_name,
            "architecture": (
                lambda a: bytes(a).decode("utf-8").rstrip("\x00") 
                    if isinstance(a, (list, bytes)) 
                    else str(a) if a else "unknown"
            )(safe_config.get("general.architecture", "unknown")),
            "n_layers": self.n_layers,
            "config": safe_config,
            "tensor_order": self._tensor_order,
            "tensor_manifest": safe_manifest,
            "extraction_date": datetime.datetime.now().isoformat(),
            "total_tensors": self._n_processed,
        }
        
        blueprint_path = os.path.join(self.model_dir, "blueprint.json")
        with open(blueprint_path, "w") as f:
            json.dump(blueprint, f, indent=2, default=make_json_safe)
        
        print(f"  Blueprint saved: {blueprint_path}")
        return blueprint

    def result(self) -> Dict:
        return {
            "model_name":  self.model_name,
            "model_dir":   self.model_dir,
            "n_processed": self._n_processed,
            "tier_counts": self._stats,
            "writers":     self.writers,
        }


# ============================================================
# ShadowLibrary — Query interface for compose
# ============================================================

class ShadowLibrary:
    """
    Query interface for an extracted shadow library (zstd version).
    
    Usage:
        lib = ShadowLibrary("shadow_library")
        
        # List all models
        lib.models()
        
        # Query by tier and properties
        records = lib.query(
            tier        = "KNOWLEDGE",
            model       = "RYS-Qwen-27B",
            depth_min   = 0.25,
            depth_max   = 0.75,
            ila_max     = 0.3,
        )
        
        # Decode tensor bytes (zstd decompression)
        data = lib.decode(records[0])   # → raw bytes (Q4_K etc)
    """

    def __init__(self, library_dir: str):
        self.library_dir = library_dir
        self.decompressor = zstd.ZstdDecompressor()
        self._index: List[ShadowRecord] = []
        self._load_index()

    def _load_index(self):
        """Load all record indices from library."""
        self._index = []
        if not os.path.exists(self.library_dir):
            return
        for model_dir in Path(self.library_dir).iterdir():
            if not model_dir.is_dir():
                continue
            for tier in Tier:
                records_path = model_dir / tier.value / f"{tier.value}_records.json"
                if records_path.exists():
                    try:
                        with open(records_path) as f:
                            records = json.load(f)
                        for r in records:
                            try:
                                # Fix relative data_path
                                if not os.path.isabs(r.get("data_path", "")):
                                    r["data_path"] = str(model_dir / tier.value / r.get("data_path",""))
                                # Only pass fields that exist in ShadowRecord
                                import dataclasses
                                valid_fields = {f.name for f in dataclasses.fields(ShadowRecord)}
                                safe_r = {k: v for k, v in r.items() if k in valid_fields}
                                # Fill any missing required fields with defaults
                                defaults = {
                                    "extraction_date": "",
                                    "kurtosis": 1.0,
                                    "super_scale_mean": 0.0,
                                    "ila_score": 0.0,
                                    "layer_depth": -1.0,
                                    "layer_idx": -1,
                                    "function_class": "unknown",
                                    "tensor_type_name": "",
                                    "n_params": 0,
                                }
                                for k, v in defaults.items():
                                    safe_r.setdefault(k, v)
                                self._index.append(ShadowRecord(**safe_r))
                            except Exception as re:
                                pass  # skip malformed records
                    except Exception as e:
                        print(f"  [WARN] Failed to load {records_path}: {e}")

    def models(self) -> List[str]:
        return sorted(set(r.source_model for r in self._index))

    def tiers(self, model: Optional[str] = None) -> Dict[str, int]:
        records = self._index
        if model:
            records = [r for r in records if r.source_model == model]
        counts = {t.value: 0 for t in Tier}
        for r in records:
            counts[r.tier] = counts.get(r.tier, 0) + 1
        return {k: v for k, v in counts.items() if v > 0}

    def query(
        self,
        tier:       Optional[str]   = None,
        model:      Optional[str]   = None,
        function:   Optional[str]   = None,
        depth_min:  Optional[float] = None,
        depth_max:  Optional[float] = None,
        ila_min:    Optional[float] = None,
        ila_max:    Optional[float] = None,
        kurt_min:   Optional[float] = None,
        kurt_max:   Optional[float] = None,
        limit:      Optional[int]   = None,
    ) -> List[ShadowRecord]:
        results = self._index

        if tier:
            results = [r for r in results if r.tier == tier]
        if model:
            results = [r for r in results if model.lower() in r.source_model.lower()]
        if function:
            results = [r for r in results if r.function_class == function]
        if depth_min is not None:
            results = [r for r in results if r.layer_depth >= depth_min]
        if depth_max is not None:
            results = [r for r in results if r.layer_depth <= depth_max]
        if ila_min is not None:
            results = [r for r in results if r.ila_score >= ila_min]
        if ila_max is not None:
            results = [r for r in results if r.ila_score <= ila_max]
        if kurt_min is not None:
            results = [r for r in results if r.kurtosis >= kurt_min]
        if kurt_max is not None:
            results = [r for r in results if r.kurtosis <= kurt_max]

        # Sort by depth for natural layer ordering
        results = sorted(results, key=lambda r: (r.source_model, r.layer_depth))

        if limit:
            results = results[:limit]

        return results

    def get_record(self, record_id: str) -> Optional[ShadowRecord]:
        """Get a specific record by ID."""
        for r in self._index:
            if r.record_id == record_id:
                return r
        return None

    def decode(self, record: ShadowRecord) -> Optional[bytes]:
        """Retrieve tensor bytes — always zstd compressed."""
        if not record.data_path or not os.path.exists(record.data_path):
            return None
        try:
            with open(record.data_path, "rb") as f:
                data = f.read()
            return self.decompressor.decompress(data)
        except Exception as e:
            print(f"  [WARN] Failed to decompress {record.data_path}: {e}")
            return None

    def decode_float32(self, record: ShadowRecord) -> Optional[np.ndarray]:
        """Decode tensor to float32 numpy array (if dequant available)."""
        data = self.decode(record)
        if data is None:
            return None
        try:
            # Try to import dequant function if available
            from alloy_project import dequant_for_projection
            shape = tuple(record.shape)
            return dequant_for_projection(data, record.tensor_type, shape)
        except ImportError:
            # Return raw bytes as uint8 array if dequant not available
            return np.frombuffer(data, dtype=np.uint8)
        except Exception:
            return None

    def stats(self) -> Dict:
        total_params = sum(r.n_params for r in self._index)
        return {
            "total_records":   len(self._index),
            "models":          self.models(),
            "tiers":           self.tiers(),
            "total_params":    total_params,
            "total_params_gb": total_params / 1e9 if total_params else 0,
        }


# ============================================================
# Utility
# ============================================================

def _safe_name(name: str) -> str:
    """Filesystem-safe model name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# ============================================================
# Main extraction pipeline
# ============================================================

def run_extraction(
    gguf_path:    str,
    out_dir:      str,
    model_name:   Optional[str]        = None,
    ila_map_path: Optional[str]        = None,
    kurtosis_path: Optional[str]       = None,
):
    """
    Full extraction pipeline: GGUF → Shadow Library.
    
    One sequential pass through the model.
    Every tensor is classified, compressed (zstd), and indexed by tier.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        print("[ERROR] pip install gguf")
        sys.exit(1)

    if model_name is None:
        model_name = Path(gguf_path).stem

    print()
    print("=" * 60)
    print("  SHADOW LLM EXTRACTOR (zstd)")
    print("  Decomposing model to quaternion tier library")
    print("=" * 60)
    print(f"\n  Model:   {gguf_path}")
    print(f"  Name:    {model_name}")
    print(f"  Library: {out_dir}")

    # Load ILA scores
    ila_scores: Dict[int, float] = {}
    if ila_map_path and os.path.exists(ila_map_path):
        try:
            with open(ila_map_path) as f:
                d = json.load(f)
            divergence = d.get("divergence", {})
            for layer_str, div in divergence.items():
                ila_scores[int(layer_str)] = float(div)
            print(f"  ILA map: {len(ila_scores)} layers loaded")
        except Exception as e:
            print(f"  [WARN] Could not load ILA map: {e}")

    # Load kurtosis map if available
    kurtosis_map: Dict[int, float] = {}
    if kurtosis_path and os.path.exists(kurtosis_path):
        try:
            with open(kurtosis_path) as f:
                kurtosis_map = {int(k): float(v) for k, v in json.load(f).items()}
            print(f"  Kurtosis map: {len(kurtosis_map)} layers loaded")
        except Exception as e:
            print(f"  [WARN] Could not load kurtosis map: {e}")

    # Open GGUF and get layer count
    print(f"\n  Scanning GGUF...")
    reader   = GGUFReader(gguf_path)
    n_layers = 0
    for k, field in reader.fields.items():
        if k.endswith("block_count"):
            try:
                try:
                    v = field.contents()
                    n_layers = int(v) if not hasattr(v, "__len__") else int(v[0]) if v else 0
                except Exception:
                    v = field.parts[-1]
                    n_layers = int(v[0]) if hasattr(v, "__len__") else int(v)
                break
            except Exception:
                pass

    n_tensors = len(reader.tensors)
    print(f"  Tensors: {n_tensors}  Layers: {n_layers}")

    # Build handler
    handler = ShadowExtractHandler(
        model_name   = model_name,
        out_dir      = out_dir,
        n_layers     = n_layers,
        ila_scores   = ila_scores,
        kurtosis_map = kurtosis_map,
    )
    
    # Add all fields to handler for blueprint
    for key, field in reader.fields.items():
        try:
            try:
                value = field.contents()  # correctly handles arrays, strings, scalars
            except Exception:
                value = field.parts[-1]   # fallback
            handler.add_field(key, value, field.types[-1] if field.types else 0)
        except Exception:
            pass
    
    # One pass — process every tensor
    print(f"\n  Extracting to shadow library (zstd compression)...")
    print(f"  Tier distribution:")

    from tqdm import tqdm
    for tensor in tqdm(reader.tensors, desc="  Tensors"):
        try:
            data = bytes(tensor.data)
            handler.process(
                name        = tensor.name,
                data        = data,
                tensor_type = int(tensor.tensor_type),
                shape       = tuple(reversed(tensor.shape)),
            )
        except Exception as e:
            print(f"\n  [WARN] Skipping {tensor.name}: {e}")

    # Save blueprint
    blueprint = handler.save_blueprint(reader)
    
    # Finalize all tiers
    print(f"\n  Finalizing tier libraries...")
    result    = handler.result()
    tier_info = {}

    for tier, writer in result["writers"].items():
        writer.finalize()
        tier_info[tier.value] = {
            "tier": tier.value,
            "n_tensors": len(writer.records),
            "records_index": os.path.join(result["model_dir"], tier.value, f"{tier.value}_records.json")
        }
        if len(writer.records) > 0:
            print(f"    {tier.value:<14}: {len(writer.records):>4} tensors")

    # Write manifest
    manifest = {
        "model_name":     model_name,
        "source_gguf":    str(Path(gguf_path).resolve()),
        "extraction_date": datetime.datetime.now().isoformat(),
        "n_layers":       n_layers,
        "n_tensors":      result["n_processed"],
        "tier_counts":    result["tier_counts"],
        "tier_info":      tier_info,
        "library_dir":    str(Path(out_dir).resolve()),
        "compression":    "zstd (level 19)",
        "tiers": {
            t.value: TIER_DESCRIPTIONS[t] for t in Tier
        }
    }

    manifest_path = os.path.join(result["model_dir"], "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=make_json_safe)
    print(f"\n  Manifest: {manifest_path}")

    # Update global index
    global_idx_path = os.path.join(out_dir, "global_index.json")
    global_idx = {}
    if os.path.exists(global_idx_path):
        try:
            with open(global_idx_path) as f:
                global_idx = json.load(f)
        except Exception:
            pass

    global_idx[model_name] = {
        "manifest":    manifest_path,
        "model_dir":   result["model_dir"],
        "n_tensors":   result["n_processed"],
        "tier_counts": result["tier_counts"],
        "added":       datetime.datetime.now().isoformat(),
        "compression": "zstd",
    }

    with open(global_idx_path, "w") as f:
        json.dump(global_idx, f, indent=2, default=make_json_safe)

    print(f"  Global index: {global_idx_path}")
    print()
    print("=" * 60)
    print("  EXTRACTION COMPLETE")
    print(f"  {result['n_processed']} tensors decomposed into 8 tiers")
    print("  All tensors stored as .bin.zst (zstd compressed, lossless)")
    print("=" * 60)

    return manifest


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Shadow LLM Extractor (zstd) — decompose GGUF to quaternion tier library"
    )
    p.add_argument("--model",    required=True,  help="GGUF model to extract")
    p.add_argument("--out-dir",  required=True,  help="Shadow library output directory")
    p.add_argument("--name",     default=None,   help="Model name (default: filename stem)")
    p.add_argument("--ila-map",  default=None,   help="instruction_map.json from alloy_probe")
    p.add_argument("--kurtosis", default=None,   help="kurtosis_map.json if available")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    run_extraction(
        gguf_path     = args.model,
        out_dir       = args.out_dir,
        model_name    = args.name,
        ila_map_path  = args.ila_map,
        kurtosis_path = args.kurtosis,
    )