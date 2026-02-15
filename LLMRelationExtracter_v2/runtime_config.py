"""
Shared runtime config builders for staged extractor entry scripts.
"""

from __future__ import annotations

from typing import Dict, Optional

from backend.settings import RELATION_EXTRACTOR_CONFIG


def build_staged_runtime_config(
    *,
    max_concurrent: Optional[int] = None,
    llm_global_concurrency: Optional[int] = None,
    max_retries: Optional[int] = None,
    llm_call_hard_timeout: Optional[float] = None,
) -> Dict:
    """
    Build a normalized runtime config for v2 staged extraction.
    """
    cfg = dict(RELATION_EXTRACTOR_CONFIG)
    if max_concurrent is not None:
        safe_max_concurrent = max(1, int(max_concurrent))
        cfg["max_concurrent"] = safe_max_concurrent
        cfg["global_concurrency"] = safe_max_concurrent
    if llm_global_concurrency is not None:
        cfg["llm_global_concurrency"] = max(1, int(llm_global_concurrency))
    if max_retries is not None:
        cfg["max_retries"] = max(0, int(max_retries))
    if llm_call_hard_timeout is not None:
        cfg["llm_call_hard_timeout"] = max(1.0, float(llm_call_hard_timeout))
    cfg.setdefault("max_chars_per_call", 8000)
    return cfg

