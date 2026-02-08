"""
Quick single-file runner for relation extraction.

Usage:
  python scripts/run_single_relation.py --csv path/to/doc.csv \
      --version v2 --max-concurrent 20 --window-size 1

Outputs a relation_results.json alongside the CSV (unless --output is set).
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_extract_and_fuse import (  # noqa: E402
    extract_relations_for_csv,
    extract_relations_for_csv_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run relation extraction for a single CSV.")
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument(
        "--output",
        help="Output JSON path (default: <csv_stem>_relation_results.json beside CSV).",
    )
    parser.add_argument(
        "--error-log",
        help="Error log path (default: <csv_stem>_errors.log beside CSV).",
    )
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="Extractor version: v1 (page-first) or v2 (brand->series->product staged).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent LLM calls.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Sliding window size (pages before/after current page).",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Disable sliding window context (window size forced to 0).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_relation_results.json")
    )
    error_log = (
        Path(args.error_log).expanduser().resolve()
        if args.error_log
        else csv_path.with_name(f"{csv_path.stem}_errors.log")
    )

    use_sliding = not args.no_sliding_window

    runner = extract_relations_for_csv_v2 if args.version == "v2" else extract_relations_for_csv

    print(f"[run-single] CSV: {csv_path}")
    print(f"[run-single] Output: {output_path}")
    print(f"[run-single] Errors: {error_log}")
    print(f"[run-single] Version: {args.version}, sliding_window={use_sliding}, window={args.window_size}")

    runner(
        csv_path=csv_path,
        output_path=output_path,
        error_log_path=error_log,
        max_concurrent=args.max_concurrent,
        use_sliding_window=use_sliding,
        window_size=args.window_size,
    )

    print(f"[run-single] Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
