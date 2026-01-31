from __future__ import annotations

import argparse
import json

from .dataset_profiling import profile_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced Benchmark Tool")
    p.add_argument("--input", "-i", required=True, help="Input CSV path")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = profile_dataset(args.input)
    if args.json:
        print(json.dumps(profile, indent=2, default=str))
    else:
        print("shape:", profile["shape"])
        print("columns:", profile["columns"])
        print("null_counts:", profile["null_counts"])
    return 0
