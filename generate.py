#!/usr/bin/env python3
"""
DevData Practice — Generate realistic development economics datasets.

Usage:
    python generate.py                      # Generate all datasets
    python generate.py household_survey     # Generate a single dataset
    python generate.py --list               # List available datasets
    python generate.py --rows 50000         # Override default size
    python generate.py --seed 99            # Set random seed
    python generate.py --format parquet     # Output as parquet
    python generate.py --output ./mydata    # Custom output directory
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

GENERATORS = {
    "household_survey": {
        "module": "generators.household_survey",
        "size_param": "n_households",
        "default_size": 15000,
        "description": "LSMS-style household survey (demographics, consumption, assets)",
    },
    "rct_experiment": {
        "module": "generators.rct_experiment",
        "size_param": "n_individuals",
        "default_size": 25000,
        "description": "Multi-arm RCT with compliance, attrition, and spillovers",
    },
    "panel_data": {
        "module": "generators.panel_data",
        "size_param": None,
        "default_size": None,
        "description": "Country-level panel (25 countries × 25 years × 20 indicators)",
    },
    "agriculture": {
        "module": "generators.agriculture",
        "size_param": "n_households",
        "default_size": 15000,
        "description": "Plot-level agricultural survey (yields, inputs, weather)",
    },
    "health_nutrition": {
        "module": "generators.health_nutrition",
        "size_param": "n_children",
        "default_size": 35000,
        "description": "DHS-style health survey (anthropometrics, vaccination, maternal)",
    },
    "education": {
        "module": "generators.education",
        "size_param": "n_schools",
        "default_size": 500,
        "description": "School + student-level data (test scores, attendance, resources)",
    },
    "labor_market": {
        "module": "generators.labor_market",
        "size_param": "n_individuals",
        "default_size": 40000,
        "description": "Labor force survey (employment, wages, formality, migration)",
    },
    "microfinance": {
        "module": "generators.microfinance",
        "size_param": "n_loans",
        "default_size": 30000,
        "description": "MFI loan records (group lending, repayment, default)",
    },
    "targeting": {
        "module": "generators.targeting",
        "size_param": "n_households",
        "default_size": 20000,
        "description": "PMT targeting data (true vs. predicted poverty, errors)",
    },
    "trade_markets": {
        "module": "generators.trade_markets",
        "size_param": "n_markets",
        "default_size": 80,
        "description": "Weekly market prices (80 markets × 104 weeks × 8 commodities)",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic development economics practice datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", nargs="*", default=[],
                        help="Dataset name(s) to generate. Omit for all.")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit.")
    parser.add_argument("--rows", type=int, default=None,
                        help="Override default dataset size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default varies by dataset).")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="Output format (default: csv).")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: ./output).")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in GENERATORS.items():
            size_str = f"(default ~{info['default_size']:,} units)" if info["default_size"] else "(fixed size)"
            print(f"  {name:<22} {info['description']}")
            print(f"  {'':22} {size_str}\n")
        return

    datasets = args.dataset if args.dataset else list(GENERATORS.keys())

    # Validate names
    for name in datasets:
        if name not in GENERATORS:
            print(f"Error: Unknown dataset '{name}'.")
            print(f"Available: {', '.join(GENERATORS.keys())}")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    print(f"\n{'='*60}")
    print(f"  DevData Practice — Dataset Generator")
    print(f"{'='*60}\n")

    for name in datasets:
        info = GENERATORS[name]
        print(f"  Generating: {name}...", end=" ", flush=True)
        t0 = time.time()

        mod = importlib.import_module(info["module"])

        kwargs = {}
        if info["size_param"] and args.rows:
            kwargs[info["size_param"]] = args.rows
        if args.seed is not None:
            kwargs["seed"] = args.seed

        df = mod.generate(**kwargs)

        # Save
        ext = args.format
        out_path = output_dir / f"{name}.{ext}"
        if ext == "csv":
            df.to_csv(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)

        elapsed = time.time() - t0
        rows, cols = df.shape
        total_rows += rows
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"{rows:>8,} rows × {cols:>3} cols  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")

    print(f"\n  Total: {total_rows:,} rows across {len(datasets)} datasets")
    print(f"  Output: {output_dir.resolve()}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
