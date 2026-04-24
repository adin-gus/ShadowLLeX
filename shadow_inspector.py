#!/usr/bin/env python3
"""
shadow_inspector.py — Shadow Library Inspector

Scans a shadow library and reports:
- What models are available
- What tiers each model has
- Statistics per tier (depth, ILA, kurtosis ranges)
- What profiles can be built
- Suggestions for composition
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from alloy_shadow_extract import ShadowLibrary, Tier, TIER_DESCRIPTIONS


@dataclass
class TierStats:
    """Statistics for a single tier in a model."""
    count: int
    depth_min: float
    depth_max: float
    depth_mean: float
    ila_min: float
    ila_max: float
    ila_mean: float
    kurt_min: float
    kurt_max: float
    kurt_mean: float
    functions: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelReport:
    """Full report for a model."""
    name: str
    total_tensors: int
    tiers: Dict[str, TierStats]


class ShadowInspector:
    def __init__(self, library_dir: str):
        self.lib = ShadowLibrary(library_dir)
        self.library_dir = library_dir
        
    def generate_report(self, model_name: Optional[str] = None) -> Dict[str, ModelReport]:
        """Generate stats for all models or a specific one."""
        reports = {}
        
        models = [model_name] if model_name else self.lib.models()
        
        for model in models:
            records = [r for r in self.lib._index if r.source_model == model]
            if not records:
                continue
                
            tier_stats = {}
            for tier in Tier:
                tier_records = [r for r in records if r.tier == tier.value]
                if not tier_records:
                    continue
                    
                depths = [r.layer_depth for r in tier_records if r.layer_depth >= 0]
                ilas = [r.ila_score for r in tier_records]
                kurts = [r.kurtosis for r in tier_records]
                
                # Function distribution
                func_counts = {}
                for r in tier_records:
                    func_counts[r.function_class] = func_counts.get(r.function_class, 0) + 1
                
                tier_stats[tier.value] = TierStats(
                    count=len(tier_records),
                    depth_min=min(depths) if depths else -1,
                    depth_max=max(depths) if depths else -1,
                    depth_mean=sum(depths)/len(depths) if depths else -1,
                    ila_min=min(ilas) if ilas else -1,
                    ila_max=max(ilas) if ilas else -1,
                    ila_mean=sum(ilas)/len(ilas) if ilas else -1,
                    kurt_min=min(kurts) if kurts else -1,
                    kurt_max=max(kurts) if kurts else -1,
                    kurt_mean=sum(kurts)/len(kurts) if kurts else -1,
                    functions=func_counts,
                )
            
            reports[model] = ModelReport(
                name=model,
                total_tensors=len(records),
                tiers=tier_stats,
            )
        
        return reports
    
    def suggest_profiles(self, reports: Dict[str, ModelReport]) -> List[Dict]:
        """Suggest what profiles each model can support."""
        suggestions = []
        
        # Define profile requirements
        profiles = {
            "reasoning": {
                "required_tiers": ["REASONING", "INSTRUCTION"],
                "min_count": 50,
                "ila_min": 0.3,
            },
            "expert": {
                "required_tiers": ["KNOWLEDGE", "CROWN"],
                "min_count": 100,
            },
            "agent": {
                "required_tiers": ["INSTRUCTION", "CONTEXT", "STYLE"],
                "min_count": 30,
                "ila_min": 0.35,
            },
            "creative": {
                "required_tiers": ["VOID", "STYLE"],
                "min_count": 10,
            },
            "balanced": {
                "required_tiers": ["REASONING", "KNOWLEDGE", "CONTEXT", "STYLE"],
                "min_count": 50,
            },
        }
        
        for model_name, report in reports.items():
            available_tiers = set(report.tiers.keys())
            
            for profile_name, requirements in profiles.items():
                required = set(requirements["required_tiers"])
                if required.issubset(available_tiers):
                    # Check counts
                    meets_count = all(
                        report.tiers[tier].count >= requirements.get("min_count", 0)
                        for tier in required
                    )
                    
                    # Check ILA if specified
                    meets_ila = True
                    if "ila_min" in requirements:
                        for tier in required.intersection(["INSTRUCTION"]):
                            if report.tiers[tier].ila_max < requirements["ila_min"]:
                                meets_ila = False
                    
                    if meets_count and meets_ila:
                        suggestions.append({
                            "model": model_name,
                            "profile": profile_name,
                            "tiers_used": list(required),
                        })
        
        return suggestions


def print_report(reports: Dict[str, ModelReport], suggestions: List[Dict]):
    """Pretty print the inspection report."""
    print()
    print("═" * 70)
    print("  SHADOW LIBRARY INSPECTOR")
    print("═" * 70)
    
    for model_name, report in reports.items():
        print(f"\n📦 {model_name}")
        print(f"   Total tensors: {report.total_tensors:,}")
        print()
        print(f"   {'Tier':<14} {'Count':>6} {'Depth range':>14} {'ILA range':>14} {'Kurtosis range':>16}")
        print(f"   {'─'*14} {'─'*6} {'─'*14} {'─'*14} {'─'*16}")
        
        for tier in Tier:
            if tier.value not in report.tiers:
                continue
            s = report.tiers[tier.value]
            
            depth_str = f"{s.depth_min:.2f}-{s.depth_max:.2f}" if s.depth_min >= 0 else "N/A"
            ila_str = f"{s.ila_min:.2f}-{s.ila_max:.2f}" if s.ila_min >= 0 else "N/A"
            kurt_str = f"{s.kurt_min:.2f}-{s.kurt_max:.2f}" if s.kurt_min >= 0 else "N/A"
            
            print(f"   {tier.value:<14} {s.count:>6}   {depth_str:>12}   {ila_str:>12}   {kurt_str:>14}")
        
        # Function breakdown for key tiers
        print()
        print(f"   Function breakdown (REASONING + KNOWLEDGE):")
        for tier in ["REASONING", "KNOWLEDGE"]:
            if tier in report.tiers:
                funcs = report.tiers[tier].functions
                func_str = ", ".join(f"{k}:{v}" for k, v in list(funcs.items())[:3])
                print(f"     {tier:<12} → {func_str}")
    
    # Suggestions
    if suggestions:
        print()
        print("═" * 70)
        print("  WHAT YOU CAN BUILD")
        print("═" * 70)
        
        # Group by profile
        by_profile = {}
        for s in suggestions:
            by_profile.setdefault(s["profile"], []).append(s["model"])
        
        for profile, models in by_profile.items():
            print(f"\n  ✓ {profile.upper()}")
            print(f"    Source models: {', '.join(models[:3])}")
            if profile == "reasoning":
                print(f"    → Use REASONING+INSTRUCTION layers with ILA>0.3")
            elif profile == "expert":
                print(f"    → Use KNOWLEDGE+CROWN for factual density")
            elif profile == "agent":
                print(f"    → Use INSTRUCTION+CONTEXT+STYLE for conversation")
            elif profile == "creative":
                print(f"    → Use VOID+STYLE (low kurtosis layers)")
    
    print()
    print("═" * 70)


def interactive_builder(reports: Dict[str, ModelReport]):
    """Launch interactive profile builder."""
    print()
    print("═" * 70)
    print("  INTERACTIVE PROFILE BUILDER")
    print("═" * 70)
    
    # Select model
    models = list(reports.keys())
    print("\nAvailable models:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")
    
    try:
        choice = input(f"\nSelect model [1-{len(models)}]: ").strip()
        model_idx = int(choice) - 1
        model_name = models[model_idx]
    except:
        print("Invalid selection")
        return
    
    report = reports[model_name]
    
    print(f"\nBuilding profile for: {model_name}")
    print()
    
    # Select tiers
    print("Which tiers to include?")
    selected_tiers = []
    for tier in Tier:
        if tier.value in report.tiers:
            s = report.tiers[tier.value]
            print(f"  {tier.value:<14} ({s.count} tensors, ILA {s.ila_min:.2f}-{s.ila_max:.2f})")
            inc = input(f"  Include? (y/n) [n]: ").strip().lower()
            if inc == 'y':
                selected_tiers.append(tier.value)
    
    # Depth range
    depth_min = input(f"\nMin depth (0-1, Enter for all): ").strip()
    depth_max = input(f"Max depth (0-1, Enter for all): ").strip()
    
    # ILA filter
    ila_max = input(f"\nMax ILA score (0-1, Enter for all): ").strip()
    
    # Generate command
    print()
    print("═" * 70)
    print("  COMPOSITION COMMAND")
    print("═" * 70)
    
    # Build spec JSON
    spec = {
        "name": f"{model_name}_custom",
        "base_model": "path/to/base.gguf",  # User must fill
        "tiers": {}
    }
    
    for tier in selected_tiers:
        spec["tiers"][tier] = {
            "source_model": model_name,
        }
        if depth_min:
            spec["tiers"][tier]["depth_min"] = float(depth_min)
        if depth_max:
            spec["tiers"][tier]["depth_max"] = float(depth_max)
        if ila_max:
            spec["tiers"][tier]["ila_max"] = float(ila_max)
    
    print(f"\nSave this spec to a file, then run:")
    print(f"\n  python alloy_shadow_compose.py --library shadow_library --spec custom_spec.json --out custom_model.gguf")
    
    # Option to save
    save = input(f"\nSave spec to file? (y/n) [n]: ").strip().lower()
    if save == 'y':
        spec_path = f"{model_name}_profile.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"  Saved to: {spec_path}")


def main():
    parser = argparse.ArgumentParser(description="Shadow Library Inspector")
    parser.add_argument("--library", required=True, help="Path to shadow library")
    parser.add_argument("--model", help="Specific model to inspect")
    parser.add_argument("--interactive", action="store_true", help="Interactive profile builder")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.library):
        print(f"[ERROR] Library not found: {args.library}")
        sys.exit(1)
    
    inspector = ShadowInspector(args.library)
    reports = inspector.generate_report(args.model)
    suggestions = inspector.suggest_profiles(reports)
    
    if args.json:
        # Convert to serializable format
        output = {}
        for name, report in reports.items():
            output[name] = {
                "total_tensors": report.total_tensors,
                "tiers": {
                    tier: {
                        "count": s.count,
                        "depth_min": s.depth_min,
                        "depth_max": s.depth_max,
                        "ila_min": s.ila_min,
                        "ila_max": s.ila_max,
                        "kurt_min": s.kurt_min,
                        "kurt_max": s.kurt_max,
                    }
                    for tier, s in report.tiers.items()
                }
            }
        print(json.dumps(output, indent=2))
    elif args.interactive:
        interactive_builder(reports)
    else:
        print_report(reports, suggestions)


if __name__ == "__main__":
    main()