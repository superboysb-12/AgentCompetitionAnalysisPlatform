"""
Quick Markdown tester for the staged (v2) relation extractor.

Usage:
  python LLMRelationExtracter_v2/test_md_extract.py

Adjust INPUT_PATH and MODE below.
"""

from pathlib import Path
import json
import sys
from typing import Optional, Tuple

# Project root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LLMRelationExtracter_v2 import extract_relations_multistage  # noqa: E402

# ---- configure here -------------------------------------------------------
# single_dir: treat INPUT_PATH directory as one document task
# children_dirs: iterate each first-level subdirectory under INPUT_PATH as one task
MODE = "single_dir"  # single_dir | children_dirs

# Example single-dir:
# INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data\20260213_GMV ES")
# Example children-dirs:
# INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data")
INPUT_PATH = Path(r"D:\AgentCompetitionAnalysisPlatform\data\20260213_GMV ES")

# If None, outputs are written beside each target directory.
OUTPUT_DIR = None  # set to Path(r"...") to override

MAX_CONCURRENT = 100
WINDOW_SIZE = 1
USE_SLIDING_WINDOW = True
SHOW_PROGRESS = True
# --------------------------------------------------------------------------


def _has_markdown_files(directory: Path) -> bool:
    return any(path.is_file() for path in directory.rglob("*_result.md"))


def _resolve_targets(input_path: Path, mode: str) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_path}")

    if mode == "single_dir":
        if not _has_markdown_files(input_path):
            raise FileNotFoundError(f"No *_result.md found under: {input_path}")
        return [input_path]

    if mode == "children_dirs":
        targets = [
            child
            for child in sorted(input_path.iterdir())
            if child.is_dir() and _has_markdown_files(child)
        ]
        if not targets:
            raise FileNotFoundError(
                f"No first-level subdirectory with *_result.md found under: {input_path}"
            )
        return targets

    raise ValueError(f"Unsupported MODE: {mode}. Use 'single_dir' or 'children_dirs'.")


def _build_output_paths(target_dir: Path, output_dir: Optional[Path]) -> Tuple[Path, Path]:
    base_dir = output_dir if output_dir is not None else target_dir.parent
    output_path = base_dir / f"{target_dir.name}_relation_results.json"
    error_log_path = base_dir / f"{target_dir.name}_errors.log"
    return output_path, error_log_path


def _count_products(output_path: Path) -> Tuple[int, Optional[dict]]:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    products = []
    if isinstance(data, list):
        for batch in data:
            if isinstance(batch, dict):
                products.extend(batch.get("results", []) or [])
    return len(products), (products[0] if products else None)


def main() -> None:
    input_path = Path(INPUT_PATH).expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else None
    targets = _resolve_targets(input_path, MODE)

    print(f"[v2 md test] Mode: {MODE}")
    print(f"[v2 md test] Input: {input_path}")
    print(f"[v2 md test] Targets: {len(targets)}")
    print(
        f"[v2 md test] Concurrency={MAX_CONCURRENT}, window_size={WINDOW_SIZE}, "
        f"sliding={USE_SLIDING_WINDOW}"
    )

    total_products = 0
    for idx, target_dir in enumerate(targets, start=1):
        output_path, error_log_path = _build_output_paths(target_dir, output_dir)
        print(f"\n[v2 md test] ({idx}/{len(targets)}) Target: {target_dir}")
        print(f"[v2 md test] Output: {output_path}")
        print(f"[v2 md test] Errors: {error_log_path}")

        extract_relations_multistage(
            csv_path=target_dir,
            output_path=output_path,
            error_log_path=error_log_path,
            max_concurrent=MAX_CONCURRENT,
            window_size=WINDOW_SIZE,
            use_sliding_window=USE_SLIDING_WINDOW,
            show_progress=SHOW_PROGRESS,
        )

        product_count, sample_product = _count_products(output_path)
        total_products += product_count
        print(f"[v2 md test] Products: {product_count}")
        if sample_product:
            print(
                "[v2 md test] Sample product: "
                f"{json.dumps(sample_product, ensure_ascii=False, indent=2)}"
            )

    print(f"\n[v2 md test] Done. Total products: {total_products}")


if __name__ == "__main__":
    main()
