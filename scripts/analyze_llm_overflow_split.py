"""
Analyze LLM length/context overflow from retry logs and suggest split plans.

Usage:
  python scripts/analyze_llm_overflow_split.py
  python scripts/analyze_llm_overflow_split.py --retry-log logs/llm_retry_events.jsonl
  python scripts/analyze_llm_overflow_split.py --json-out logs/llm_overflow_plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.settings import RELATION_EXTRACTOR_CONFIG

OVERFLOW_HINTS = (
    "lengthfinishreasonerror",
    "length limit was reached",
    "context_length_exceeded",
    "maximum context length",
    "context length exceeded",
    "token limit",
)

TOKEN_PATTERNS = {
    "prompt_tokens": re.compile(r"prompt_tokens=(\d+)"),
    "completion_tokens": re.compile(r"completion_tokens=(\d+)"),
    "total_tokens": re.compile(r"total_tokens=(\d+)"),
}


@dataclass
class SchemaGuide:
    stage: str
    split_axis: List[str]
    knobs: List[str]
    code_path: str


SCHEMA_GUIDE: Dict[str, SchemaGuide] = {
    "brand_schema": SchemaGuide(
        stage="Stage A",
        split_axis=[
            "按页/聚类分组调用（减少每次聚类页数）",
            "减少每页注入内容（尤其 rows_json/table_data）",
        ],
        knobs=["brand_cluster_size", "table_rows_per_block", "rows_json_content_clip"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "brand_filter_schema": SchemaGuide(
        stage="Stage A",
        split_axis=["按候选品牌列表分批 review/filter"],
        knobs=["(建议新增) brand_filter_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "brand_canon_schema": SchemaGuide(
        stage="Stage A",
        split_axis=["按候选品牌列表分批 canonical，再合并"],
        knobs=["(建议新增) brand_canon_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "series_schema": SchemaGuide(
        stage="Stage B",
        split_axis=["按文本长度切块（_split_text）", "减少 brand evidence 跟页数"],
        knobs=["max_chars_per_call", "series_brand_follow_pages", "series_context_follow_pages"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "series_feature_schema": SchemaGuide(
        stage="Stage B/C",
        split_axis=[
            "按页面 chunk 分批（2/3 页）",
            "按 series_names 列表再分批（避免单次输出过多）",
        ],
        knobs=[
            "series_brand_chunk_pages",
            "series_context_follow_pages",
            "(建议新增) series_feature_series_batch_size",
        ],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "series_review_schema": SchemaGuide(
        stage="Stage B",
        split_axis=["按系列候选列表分批 review"],
        knobs=["(建议新增) series_review_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "series_canon_schema": SchemaGuide(
        stage="Stage B",
        split_axis=["按系列候选列表分批 canonical"],
        knobs=["(建议新增) series_canon_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "model_schema": SchemaGuide(
        stage="Stage C",
        split_axis=["按文本长度切块（_split_text）", "减少系列上下文跟页数"],
        knobs=["max_chars_per_call", "series_model_follow_pages", "model_context_follow_pages"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "model_review_schema": SchemaGuide(
        stage="Stage C",
        split_axis=["按模型候选列表分批 review"],
        knobs=["model_review_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "product_schema": SchemaGuide(
        stage="Stage D",
        split_axis=["按文本长度切块（_split_text）", "减少型号上下文跟页数"],
        knobs=["max_chars_per_call", "product_model_follow_pages", "model_context_follow_pages"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
    "product_review_schema": SchemaGuide(
        stage="Stage D",
        split_axis=["按产品候选列表分批 review"],
        knobs=["product_review_max_items"],
        code_path="LLMRelationExtracter_v2/staged_extractor.py",
    ),
}


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _collect_error_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(str(record.get("error_type") or ""))
    parts.append(str(record.get("error") or ""))
    diagnostics = record.get("diagnostics")
    if isinstance(diagnostics, dict):
        parts.append(str(diagnostics.get("root_error_type") or ""))
        parts.append(str(diagnostics.get("root_error") or ""))
        chain = diagnostics.get("error_chain")
        if isinstance(chain, list):
            for item in chain:
                if isinstance(item, dict):
                    parts.append(str(item.get("type") or ""))
                    parts.append(str(item.get("message") or ""))
    return " | ".join(parts)


def _is_overflow_record(record: Dict[str, Any]) -> bool:
    text = _collect_error_text(record).lower()
    return any(token in text for token in OVERFLOW_HINTS)


def _extract_tokens(record: Dict[str, Any]) -> Dict[str, int]:
    source = _collect_error_text(record)
    out: Dict[str, int] = {}
    for key, pat in TOKEN_PATTERNS.items():
        m = pat.search(source)
        if m:
            out[key] = int(m.group(1))
    return out


def _compute_split_factor(
    *,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    prompt_target: int,
    completion_target: int,
    total_target: int,
) -> int:
    factor = 1
    if prompt_tokens:
        factor = max(factor, int(math.ceil(prompt_tokens / max(1, prompt_target))))
    if completion_tokens:
        factor = max(
            factor,
            int(math.ceil(completion_tokens / max(1, completion_target))),
        )
    if total_tokens:
        factor = max(factor, int(math.ceil(total_tokens / max(1, total_target))))
    return max(1, factor)


def _percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    rank = max(0, min(len(values) - 1, int(round((len(values) - 1) * pct))))
    return sorted(values)[rank]


def _schema_order_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
    schema, stat = item
    return (-int(stat.get("count", 0)), schema)


def analyze_overflow(
    rows: Iterable[Dict[str, Any]],
    *,
    prompt_target: int,
    completion_target: int,
    total_target: int,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "hosts": set(),
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": [],
            "split_factors": [],
            "samples": [],
        }
    )
    total_rows = 0
    overflow_rows = 0
    for row in rows:
        total_rows += 1
        if not _is_overflow_record(row):
            continue
        overflow_rows += 1
        schema = str(row.get("schema") or "unknown_schema").strip() or "unknown_schema"
        stat = grouped[schema]
        stat["count"] += 1
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, dict):
            host = str(diagnostics.get("request_host") or "").strip()
            if host:
                stat["hosts"].add(host)
        tokens = _extract_tokens(row)
        prompt_tokens = tokens.get("prompt_tokens")
        completion_tokens = tokens.get("completion_tokens")
        total_tokens = tokens.get("total_tokens")
        if prompt_tokens is not None:
            stat["prompt_tokens"].append(prompt_tokens)
        if completion_tokens is not None:
            stat["completion_tokens"].append(completion_tokens)
        if total_tokens is not None:
            stat["total_tokens"].append(total_tokens)
        split_factor = _compute_split_factor(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_target=prompt_target,
            completion_target=completion_target,
            total_target=total_target,
        )
        stat["split_factors"].append(split_factor)
        if len(stat["samples"]) < 3:
            stat["samples"].append(
                {
                    "timestamp": row.get("timestamp"),
                    "phase": row.get("phase"),
                    "attempt": row.get("attempt"),
                    "error_type": row.get("error_type"),
                    "error": str(row.get("error") or "")[:240],
                    "split_factor": split_factor,
                }
            )

    rendered: Dict[str, Any] = {}
    for schema, stat in grouped.items():
        prompt_tokens = [int(v) for v in stat["prompt_tokens"] if isinstance(v, int)]
        completion_tokens = [int(v) for v in stat["completion_tokens"] if isinstance(v, int)]
        total_tokens = [int(v) for v in stat["total_tokens"] if isinstance(v, int)]
        split_factors = [int(v) for v in stat["split_factors"] if isinstance(v, int)]
        rendered[schema] = {
            "count": int(stat["count"]),
            "hosts": sorted(stat["hosts"]),
            "prompt_tokens": {
                "max": max(prompt_tokens) if prompt_tokens else 0,
                "p95": _percentile(prompt_tokens, 0.95) if prompt_tokens else 0,
                "avg": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else 0.0,
            },
            "completion_tokens": {
                "max": max(completion_tokens) if completion_tokens else 0,
                "p95": _percentile(completion_tokens, 0.95) if completion_tokens else 0,
                "avg": round(statistics.mean(completion_tokens), 2) if completion_tokens else 0.0,
            },
            "total_tokens": {
                "max": max(total_tokens) if total_tokens else 0,
                "p95": _percentile(total_tokens, 0.95) if total_tokens else 0,
                "avg": round(statistics.mean(total_tokens), 2) if total_tokens else 0.0,
            },
            "suggested_split_factor": max(split_factors) if split_factors else 1,
            "samples": stat["samples"],
        }

    return {
        "total_retry_rows": total_rows,
        "overflow_rows": overflow_rows,
        "schemas": dict(sorted(rendered.items(), key=_schema_order_key)),
    }


def _pick_recommended_env(summary: Dict[str, Any]) -> Dict[str, Any]:
    cfg = RELATION_EXTRACTOR_CONFIG
    schemas = summary.get("schemas", {})
    text_split_factor = 1
    for schema_name in ("series_schema", "model_schema", "product_schema"):
        item = schemas.get(schema_name, {})
        text_split_factor = max(text_split_factor, int(item.get("suggested_split_factor", 1)))
    review_split_factor = 1
    for schema_name in ("model_review_schema", "product_review_schema"):
        item = schemas.get(schema_name, {})
        review_split_factor = max(review_split_factor, int(item.get("suggested_split_factor", 1)))

    max_chars_base = int(cfg.get("max_chars_per_call", 8000))
    model_review_base = int(cfg.get("model_review_max_items", 80))
    product_review_base = int(cfg.get("product_review_max_items", 50))

    env: Dict[str, Any] = {}
    if text_split_factor > 1:
        env["MAX_CHARS_PER_CALL"] = max(1200, int(max_chars_base / text_split_factor))
    if review_split_factor > 1:
        env["MODEL_REVIEW_MAX_ITEMS"] = max(12, int(model_review_base / review_split_factor))
        env["PRODUCT_REVIEW_MAX_ITEMS"] = max(10, int(product_review_base / review_split_factor))

    series_feature_item = schemas.get("series_feature_schema", {})
    series_feature_factor = int(series_feature_item.get("suggested_split_factor", 1))
    if series_feature_factor > 1:
        env["SERIES_BRAND_CHUNK_PAGES"] = 2
        env["SERIES_CONTEXT_FOLLOW_PAGES"] = max(
            0,
            int(cfg.get("series_context_follow_pages", 2)) - 1,
        )
        env["SERIES_FEATURE_SERIES_BATCH_SIZE"] = max(3, int(12 / series_feature_factor))

    return env


def _render_console(summary: Dict[str, Any], *, show_samples: bool) -> str:
    lines: List[str] = []
    lines.append("== LLM Overflow Split Plan ==")
    lines.append(f"retry_rows={summary.get('total_retry_rows', 0)}")
    lines.append(f"overflow_rows={summary.get('overflow_rows', 0)}")
    schemas = summary.get("schemas", {})
    if not schemas:
        lines.append("No overflow records found.")
        return "\n".join(lines)

    for schema, item in schemas.items():
        guide = SCHEMA_GUIDE.get(schema)
        stage = guide.stage if guide else "unknown_stage"
        lines.append("")
        lines.append(f"[{schema}] stage={stage} count={item.get('count', 0)}")
        lines.append(
            "tokens "
            f"prompt(max/p95/avg)={item['prompt_tokens']['max']}/{item['prompt_tokens']['p95']}/{item['prompt_tokens']['avg']} "
            f"completion(max/p95/avg)={item['completion_tokens']['max']}/{item['completion_tokens']['p95']}/{item['completion_tokens']['avg']} "
            f"total(max/p95/avg)={item['total_tokens']['max']}/{item['total_tokens']['p95']}/{item['total_tokens']['avg']}"
        )
        lines.append(f"suggested_split_factor={item.get('suggested_split_factor', 1)}")
        if item.get("hosts"):
            lines.append(f"hosts={','.join(item['hosts'])}")
        if guide:
            lines.append(f"split_axis={'; '.join(guide.split_axis)}")
            lines.append(f"knobs={', '.join(guide.knobs)}")
            lines.append(f"code={guide.code_path}")
        if show_samples:
            for sample in item.get("samples", []):
                lines.append(
                    "sample "
                    f"time={sample.get('timestamp')} phase={sample.get('phase')} "
                    f"attempt={sample.get('attempt')} factor={sample.get('split_factor')} "
                    f"type={sample.get('error_type')}"
                )

    env = _pick_recommended_env(summary)
    lines.append("")
    lines.append("== Suggested Env Overrides ==")
    if env:
        for key, value in env.items():
            lines.append(f"{key}={value}")
    else:
        lines.append("No override needed from current log sample.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze LLM overflow retries and suggest split strategy per schema.",
    )
    parser.add_argument(
        "--retry-log",
        type=Path,
        default=Path("logs/llm_retry_events.jsonl"),
        help="Path to retry events jsonl.",
    )
    parser.add_argument(
        "--prompt-target",
        type=int,
        default=12000,
        help="Target prompt tokens per request.",
    )
    parser.add_argument(
        "--completion-target",
        type=int,
        default=12000,
        help="Target completion tokens per request.",
    )
    parser.add_argument(
        "--total-target",
        type=int,
        default=24000,
        help="Target total tokens per request.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print sample overflow rows for each schema.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save machine-readable summary JSON.",
    )
    args = parser.parse_args()

    if not args.retry_log.exists():
        print(f"Retry log not found: {args.retry_log}")
        return 2

    rows = list(_read_jsonl(args.retry_log))
    summary = analyze_overflow(
        rows,
        prompt_target=max(1, int(args.prompt_target)),
        completion_target=max(1, int(args.completion_target)),
        total_target=max(1, int(args.total_target)),
    )
    print(_render_console(summary, show_samples=bool(args.show_samples)))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON written: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
