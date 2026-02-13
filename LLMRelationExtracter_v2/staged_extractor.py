"""
Staged (brand → series → product) relation extractor.

- 输入: CSV 与 v1 相同
- 输出: [{"results": [product...]}]，字段遵循 v1 RELATION_SCHEMA
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.settings import RELATION_EXTRACTOR_CONFIG
from LLMRelationExtracter import (  # v1 复用的通用模块
    correct_all_categories,
    deduplicate_results,
    filter_empty_products,
    load_pages_with_context,
)
from LLMRelationExtracter.md_processor import (
    load_pages_with_context_from_md,
    load_pages_with_context_from_md_directory,
)
from sklearn.cluster import AgglomerativeClustering
from .staged_prompts import (
    BRAND_CANON_SCHEMA,
    BRAND_SCHEMA,
    BRAND_FILTER_SCHEMA,
    MODEL_REVIEW_SCHEMA,
    MODEL_SCHEMA,
    PRODUCT_REVIEW_SCHEMA,
    PRODUCT_SCHEMA,
    SERIES_CANON_SCHEMA,
    SERIES_FEATURE_SCHEMA,
    SERIES_SCHEMA,
    SERIES_REVIEW_SCHEMA,
    build_brand_canon_messages,
    build_brand_filter_messages,
    build_brand_global_filter_messages,
    build_brand_primary_messages,
    build_brand_messages,
    build_model_messages,
    build_model_review_messages,
    build_product_messages,
    build_product_review_messages,
    build_series_canon_messages,
    build_series_feature_messages,
    build_series_review_messages,
    build_series_messages,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_pages_with_context_v2(
    source_path: str, window_size: int = 1, known_models: Optional[List[str]] = None
) -> Iterable[Tuple[str, Dict]]:
    """Load pages from CSV, Markdown file, or Markdown directory."""
    path = Path(source_path)
    if path.is_dir():
        yield from load_pages_with_context_from_md_directory(
            source_path,
            window_size=window_size,
            known_models=known_models,
        )
        return

    if path.suffix.lower() == ".md":
        yield from load_pages_with_context_from_md(
            source_path,
            window_size=window_size,
            known_models=known_models,
        )
        return

    yield from load_pages_with_context(
        source_path,
        window_size=window_size,
        known_models=known_models,
    )


def _normalize_name(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _dedup_list(seq: List) -> List:
    """Order-preserving dedup for hashable items."""
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_brand_key(text: str) -> str:
    """Stronger normalization for brand merge: remove spaces/punct and lowercase."""
    return re.sub(r"[^a-z0-9一-龥]+", "", (text or "").lower())


def _normalize_series_key(text: str) -> str:
    """Normalization for series merge/canonical matching."""
    normalized = re.sub(r"[^a-z0-9一-龥]+", "", (text or "").lower())
    return normalized.replace("系列", "")


_SERIES_META_KEYWORDS = (
    "上一代",
    "下一代",
    "上一代产品",
    "下一代产品",
    "参数",
    "对比",
    "说明",
    "介绍",
    "通讯",
    "通信",
    "协议",
    "网络",
    "网关",
    "控制",
    "方案",
    "最大框体",
)
_SERIES_PROTOCOL_KEYWORDS = (
    "can网络",
    "can总线",
    "bacnet",
    "modbus",
    "knx",
    "lonworks",
    "协议",
    "通讯",
    "通信",
    "网关",
)
_SERIES_GENERIC_NON_PRODUCT = (
    "中央空调",
    "空调",
    "多联机",
    "热泵",
    "室内机",
    "室外机",
)
_HVAC_SERIES_TAILS = (
    "室内机",
    "室外机",
    "风管式",
    "壁挂式",
    "天井式",
    "座吊式",
    "多联机",
    "机组",
    "热泵",
    "盘管",
    "新风",
)


_SERIES_NARRATIVE_HINTS = (
    "延续",
    "继承",
    "推出",
    "先推",
    "采用",
    "相比",
    "说明",
    "引领",
    "技术发展",
    "发展方向",
)


def _clean_series_name_text(text: str) -> str:
    """Normalize/clean noisy series candidate display text."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""

    s = re.sub(r"^\[title\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^[#*·\-\s]+", "", s)

    # Drop bracketed meta notes (e.g., "(上一代产品)").
    def _strip_meta_brackets(src: str) -> str:
        out = src
        bracket_patterns = [
            r"\([^)]*\)",
            r"（[^）]*）",
            r"\[[^\]]*\]",
            r"【[^】]*】",
        ]
        for pat in bracket_patterns:
            while True:
                m = re.search(pat, out)
                if not m:
                    break
                token = m.group(0)
                if any(k in token for k in _SERIES_META_KEYWORDS):
                    out = out[: m.start()] + out[m.end() :]
                else:
                    break
        return out

    s = _strip_meta_brackets(s)

    # Keep the first segment before enum-like separators: ① ② / 1) / 2.
    s = re.split(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]", s)[0]
    s = re.split(r"\s+\d+\s*[).、．。:：]\s*", s)[0]

    # Trim trailing explanatory words that are not part of a product series name.
    s = re.sub(r"(使用|说明|参数|介绍|配置|功能|对比|方案|系统)\s*$", "", s)
    s = s.strip(" ,，;；|/\\-:_")
    return s


def _looks_like_series_description_text(name: str) -> bool:
    """
    Detect long narrative/explanatory text that should not be used as canonical series name.
    """
    clean = _clean_series_name_text(name)
    if not clean:
        return False
    compact = re.sub(r"\s+", "", clean)
    if len(compact) < 10:
        return False
    if "..." in clean or "…" in clean:
        return True
    if re.search(r"[,，。;；:：!！?？]", clean):
        return True
    if len(compact) >= 22 and any(token in compact for token in _SERIES_NARRATIVE_HINTS):
        return True
    if len(compact) >= 30 and ("系列" in compact or "series" in compact.lower()):
        if any(t in compact for t in _HVAC_SERIES_TAILS):
            return True
    return False


def _is_hvac_series_candidate(name: str) -> bool:
    """Check whether a candidate looks like a real HVAC product series/line."""
    clean = _clean_series_name_text(name)
    if not clean or len(clean) < 2 or len(clean) > 64:
        return False

    compact = re.sub(r"\s+", "", clean)
    lower = compact.lower()
    has_series_word = ("系列" in compact) or ("series" in lower)
    has_code_prefix = bool(re.match(r"^[A-Za-z][A-Za-z0-9+\-]{1,23}", compact))
    has_hvac_tail = any(t in compact for t in _HVAC_SERIES_TAILS)

    if compact in _SERIES_GENERIC_NON_PRODUCT:
        return False

    if any(k in lower for k in _SERIES_PROTOCOL_KEYWORDS):
        return False

    # Meta/description-heavy text should not become canonical series names.
    if any(k in compact for k in _SERIES_META_KEYWORDS):
        if not has_code_prefix:
            return False
        if not has_series_word and not has_hvac_tail:
            return False

    if has_series_word:
        return True
    if has_code_prefix and has_hvac_tail:
        return True
    return False


def _series_merge_key(text: str) -> str:
    """
    Key used for Stage-B alias merging.
    It intentionally collapses variants like `SDZD` / `SDZD系列` / `SDZD系列...`.
    """
    cleaned = _clean_series_name_text(text)
    raw = re.sub(r"[^a-z0-9一-龥]+", "", cleaned.lower())
    if not raw:
        return ""
    if "系列" in raw:
        prefix = raw.split("系列", 1)[0].strip()
        if len(prefix) >= 2:
            return prefix
    code_prefix = re.match(r"^([a-z0-9+\-]{2,20})(?=[一-龥])", raw)
    if code_prefix:
        return code_prefix.group(1).strip("+-")
    if raw.endswith("series") and len(raw) > len("series"):
        return raw[: -len("series")]
    return raw.replace("系列", "")


def _choose_series_canonical_name(names: Sequence[str]) -> str:
    """Pick a stable canonical display name from aliases of one merged series key."""
    candidates = _dedup_list(
        [
            _clean_series_name_text(name)
            for name in names
            if _clean_series_name_text(name) and _is_hvac_series_candidate(name)
        ]
    )
    if not candidates:
        return ""
    stable_candidates = [n for n in candidates if not _looks_like_series_description_text(n)]
    if stable_candidates:
        candidates = stable_candidates

    def _score(name: str) -> Tuple[int, int, int, int, int, int]:
        compact = re.sub(r"\s+", "", name)
        lower = compact.lower()
        has_series_word = int("系列" in compact or "series" in lower)
        has_hvac_tail = int(any(t in compact for t in _HVAC_SERIES_TAILS))
        noise_penalty = -sum(1 for k in _SERIES_META_KEYWORDS if k in compact)
        description_penalty = -int(_looks_like_series_description_text(name))
        has_chinese = int(bool(re.search(r"[一-龥]", compact)))
        length_score = -abs(len(compact) - 14)
        return (
            has_series_word,
            has_hvac_tail,
            noise_penalty,
            description_penalty,
            has_chinese,
            length_score,
        )

    return max(candidates, key=lambda n: (_score(n), n))


def _split_series_name_candidates(name: str) -> List[str]:
    """
    Split noisy combined Stage-B series text into atomic series candidates.
    Example:
    "SDB系列... SDK系列... SDTS系列..." -> ["SDB系列...", "SDK系列...", "SDTS系列..."]
    """
    text = _clean_series_name_text(name)
    text = text.strip(" ,;|/")
    if not text:
        return []

    start_pattern = re.compile(r"(?:[A-Za-z0-9+\-]{2,20}|[一-龥]{1,8})\s*系列")
    code_pattern = re.compile(r"(?:(?<=^)|(?<=[\s,;|/\\]))[A-Za-z0-9+\-]{2,20}(?:\s*系列)?(?=[一-龥])")
    starts = sorted(
        set([m.start() for m in start_pattern.finditer(text)] + [m.start() for m in code_pattern.finditer(text)])
    )
    if len(starts) <= 1:
        return [text]

    starts.append(len(text))
    parts: List[str] = []
    for idx in range(len(starts) - 1):
        chunk = text[starts[idx] : starts[idx + 1]].strip(" ,;|/")
        if len(chunk) >= 2:
            parts.append(chunk)
    return _dedup_list(parts) or [text]


def _looks_like_code_series_name(name: str) -> bool:
    """
    Heuristic safety-net for Stage-B review:
    preserve series candidates that clearly look like coded HVAC series labels.
    """
    compact = re.sub(r"\s+", "", _clean_series_name_text(name))
    if len(compact) < 2:
        return False

    # e.g. SDTS系列..., GMV9系列..., SDC+系列...
    if "系列" in compact:
        prefix = compact.split("系列", 1)[0]
        if 2 <= len(prefix) <= 24 and re.search(r"[A-Za-z]", prefix):
            if re.fullmatch(r"[A-Za-z0-9+\-]+", prefix) and _is_hvac_series_candidate(compact):
                return True

    # e.g. SDTS?????????? (without explicit "系列")
    if re.match(r"^[A-Za-z][A-Za-z0-9+\-]{1,23}[一-龥]", compact) and _is_hvac_series_candidate(compact):
        return True

    return False


def _extract_code_series_from_text(text: str) -> List[str]:
    """
    Recover coded series names from noisy evidence snippets.
    Used as a Stage-B supplement when LLM misses aliases in long rows.
    """
    source = re.sub(r"\s+", " ", str(text or ""))
    if not source:
        return []

    patterns = [
        re.compile(
            r"[A-Za-z][A-Za-z0-9+\-]{1,23}\s*系列[^\n,，;；|]{0,28}?"
            r"(?:室内机|室外机|盘管|机组|多联机)"
        ),
        re.compile(
            r"[A-Za-z][A-Za-z0-9+\-]{1,23}[^\n,，;；|]{0,22}?"
            r"(?:室内机|室外机|盘管|机组|多联机)"
        ),
    ]

    found: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(source):
            candidate = match.group(0).strip(" ,，;；|/\\")
            if len(candidate) < 2 or len(candidate) > 56:
                continue
            for part in _split_series_name_candidates(candidate):
                clean = _clean_series_name_text(part.strip(" ,，;；|/\\"))
                if _looks_like_code_series_name(clean) and _is_hvac_series_candidate(clean):
                    found.append(clean)
    return _dedup_list(found)


def _sanitize_series_items(series_items: List[Dict]) -> List[Dict]:
    """Keep only HVAC product-series-like items and normalize their display names."""
    out: List[Dict] = []
    for item in series_items or []:
        name = _clean_series_name_text(item.get("name") or "")
        if not _is_hvac_series_candidate(name):
            continue
        cloned = dict(item or {})
        cloned["name"] = name
        cloned["evidence"] = _dedup_list(cloned.get("evidence", []))
        cloned["pages"] = _dedup_list(cloned.get("pages", []))
        out.append(cloned)
    return out


def _match_series_name(series_guess: str, brand: str, alias_map: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    "Return canonical series name within a brand if guess matches alias/canonical."
    guess_key = _series_merge_key(series_guess) or _normalize_series_key(series_guess)
    if not guess_key:
        return None
    brand_map = alias_map.get(brand, {}) if alias_map else {}
    for canonical, aliases in brand_map.items():
        for name in [canonical] + aliases:
            name_key = _series_merge_key(name) or _normalize_series_key(name)
            if name_key == guess_key:
                return canonical
    return None


_MODEL_CHAR_TRANSLATION = str.maketrans(
    {
        "＋": "+",
        "－": "-",
        "—": "-",
        "–": "-",
        "／": "/",
        "（": "(",
        "）": ")",
        "，": ",",
        "、": ",",
        "；": ";",
        "：": ":",
        "　": " ",
    }
)


def _canonicalize_model_name(text: str) -> str:
    """
    Canonicalize model display format while keeping semantic content:
    - normalize full-width punctuations to ASCII
    - trim spaces around separators
    - uppercase latin letters
    - normalize combo models joined by '+'
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    norm = raw.translate(_MODEL_CHAR_TRANSLATION)
    norm = re.sub(r"\s+", "", norm)
    norm = re.sub(r"\++", "+", norm)
    norm = re.sub(r",+", ",", norm)

    if "+" in norm:
        parts = [p for p in norm.split("+") if p]
        norm = "+".join(parts)

    return norm.upper()


def _normalize_model_key(text: str) -> str:
    """Normalization for model matching and dedup."""
    return _canonicalize_model_name(text)


def _looks_like_non_model_value(text: str) -> bool:
    """
    Guardrail for Stage C:
    drop numeric capacity values/ranges mistakenly extracted as model names.
    """
    token = str(text or "").strip()
    if not token:
        return True

    lower = token.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", lower):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[/~\-]\d+(?:\.\d+)?)+", lower):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:kw|hp|匹)", lower):
        return True
    return False


def _pair_key(brand: str, series: str) -> str:
    return f"{brand}|||{series}"


def _split_pair_key(key: str) -> Tuple[str, str]:
    parts = (key or "").split("|||", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key or "", ""


def _model_node_key(brand: str, series: str, model_name: str) -> str:
    return f"{brand}|||{series}|||{_normalize_model_key(model_name)}"


@lru_cache(maxsize=2)
def _get_embedder(model_name: str, device: str):
    return SentenceTransformer(model_name, device=device)


def _get_concurrency(config: Dict, key: str, default: int) -> int:
    """
    Use one global concurrency for all LLM calls.
    """
    if "global_concurrency" in config and config["global_concurrency"] is not None:
        return int(config["global_concurrency"])
    if "max_concurrent" in config and config["max_concurrent"] is not None:
        return int(config["max_concurrent"])
    return int(default)


def _embed_texts(texts: List[str], model_name: str, device: str = "cpu") -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    model = _get_embedder(model_name, device)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _merge_brand_candidates(brands: List[Dict]) -> List[Dict]:
    """
    Merge brand candidates with simple normalization & containment heuristics.
    Keeps evidence/pages merged.
    """
    merged: Dict[str, Dict] = {}
    for b in brands:
        name = b.get("name", "")
        key = _normalize_brand_key(name)
        if not key:
            continue

        # try exact or containment merge
        target_key = None
        for existing_key in merged.keys():
            if key == existing_key:
                target_key = existing_key
                break
            # containment heuristic for short aliases (e.g., "haier" vs "haierac")
            if (
                key in existing_key or existing_key in key
            ) and min(len(key), len(existing_key)) >= 3:
                target_key = existing_key
                break

        if target_key is None:
            merged[key] = {
                "name": name,
                "evidence": list(b.get("evidence", [])),
                "pages": list(b.get("pages", [])),
            }
        else:
            merged[target_key]["evidence"].extend(b.get("evidence", []))
            merged[target_key]["pages"].extend(b.get("pages", []))

    merged_list = []
    for v in merged.values():
        v["evidence"] = _dedup_list(v.get("evidence", []))
        v["pages"] = _dedup_list(v.get("pages", []))
        merged_list.append(v)
    return merged_list


def _merge_series_candidates(series_items: List[Dict]) -> List[Dict]:
    """Merge exact/near-exact series candidates by normalized key."""
    series_items = _sanitize_series_items(series_items)
    merged: Dict[str, Dict] = {}
    for item in series_items:
        raw_name = (item.get("name") or "").strip()
        for name in _split_series_name_candidates(raw_name):
            clean_name = _clean_series_name_text(name)
            if not _is_hvac_series_candidate(clean_name):
                continue
            key = _series_merge_key(clean_name) or _normalize_series_key(clean_name)
            if not key:
                continue
            if key not in merged:
                merged[key] = {
                    "names": [clean_name],
                    "evidence": list(item.get("evidence", [])),
                    "pages": list(item.get("pages", [])),
                }
            else:
                merged[key]["names"].append(clean_name)
                merged[key]["evidence"].extend(item.get("evidence", []))
                merged[key]["pages"].extend(item.get("pages", []))
    out: List[Dict] = []
    for value in merged.values():
        candidate_names = _dedup_list(value.get("names", []))
        canonical_name = _choose_series_canonical_name(candidate_names)
        value["name"] = canonical_name or (candidate_names[0] if candidate_names else "")
        value.pop("names", None)
        value["evidence"] = _dedup_list(value.get("evidence", []))
        value["pages"] = _dedup_list(value.get("pages", []))
        out.append(value)
    return out


def _merge_model_candidates(model_items: List[Dict]) -> List[Dict]:
    """Merge model candidates by normalized model key."""
    merged: Dict[str, Dict] = {}
    for item in model_items:
        name = _canonicalize_model_name(item.get("name") or "")
        key = _normalize_model_key(name)
        if not key:
            continue
        if key not in merged:
            base = {
                "name": name,
                "evidence": list(item.get("evidence", [])),
                "pages": list(item.get("pages", [])),
            }
            for parent_field in ("brand", "series", "pair_key", "model_key"):
                if item.get(parent_field):
                    base[parent_field] = item.get(parent_field)
            merged[key] = base
        else:
            merged[key]["evidence"].extend(item.get("evidence", []))
            merged[key]["pages"].extend(item.get("pages", []))
            for parent_field in ("brand", "series", "pair_key", "model_key"):
                if item.get(parent_field) and not merged[key].get(parent_field):
                    merged[key][parent_field] = item.get(parent_field)
    out: List[Dict] = []
    for value in merged.values():
        value["evidence"] = _dedup_list(value.get("evidence", []))
        value["pages"] = _dedup_list(value.get("pages", []))
        out.append(value)
    return out


def _merge_series_semantic(
    series_items: List[Dict],
    model_name: str,
    distance_threshold: float = 0.2,
    device: str = "cpu",
) -> List[Dict]:
    """Merge semantically close series aliases via multilingual embeddings."""
    if len(series_items) <= 1:
        return series_items
    names = [s.get("name", "") for s in series_items]
    embeddings = _embed_texts(names, model_name, device=device)

    clusterer = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clusterer.fit_predict(embeddings)
    grouped: Dict[int, List[Dict]] = {}
    for idx, cid in enumerate(labels):
        grouped.setdefault(cid, []).append(series_items[idx])

    merged: List[Dict] = []
    for group in grouped.values():
        representative = sorted(
            group,
            key=lambda s: len(_dedup_list(s.get("pages", []))),
            reverse=True,
        )[0]
        merged_ev: List[str] = []
        merged_pages: List = []
        for item in group:
            merged_ev.extend(item.get("evidence", []))
            merged_pages.extend(item.get("pages", []))
        merged_item = dict(representative)
        merged_item["evidence"] = _dedup_list(merged_item.get("evidence", []) + merged_ev)
        merged_item["pages"] = _dedup_list(merged_item.get("pages", []) + merged_pages)
        merged.append(merged_item)
    return merged


def _is_plausible_brand(name: str) -> bool:
    """
    Heuristic filter to drop obvious non-brand strings (model lists, org names, parameters).
    Keeps this lightweight to avoid hurting recall in truly multi-brand documents.
    """
    if not name:
        return False
    n = name.strip()
    if len(n) < 2:
        return False
    if len(n) > 24:  # excessively long tends to be org names or sentences
        return False
    if re.search(r"\d", n):
        return False  # model-like token
    if "/" in n or "|" in n:
        return False  # batch of models or alternatives
    noise_keywords = ["系列", "型号", "产品", "室内机", "室外机", "参数", "风管", "冷水机", "机组", "控制器"]
    if any(k in n for k in noise_keywords):
        return False
    return True


def _prune_brands(
    brands: List[Dict], min_page_ratio: float = 0.15, max_candidates: int = 12
) -> List[Dict]:
    """
    Keep only the strongest brand candidates to avoid exploding Stage B/C.
    Criteria:
      - drop implausible names (heuristic)
      - require page coverage within min_page_ratio of the top brand
      - cap total candidates to max_candidates (by page coverage)
    """
    if not brands:
        return []

    scored = []
    for b in brands:
        pages = _dedup_list(b.get("pages", []))
        page_count = len(pages)
        scored.append((page_count, b))

    max_pages = max(page for page, _ in scored)
    min_pages = max(1, int(max_pages * min_page_ratio))

    filtered = [
        b for page, b in scored if page >= min_pages and _is_plausible_brand(b.get("name", ""))
    ]
    if not filtered:
        # Fallback: keep top ones even if heuristics filtered everything
        filtered = [b for _, b in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]

    filtered = sorted(filtered, key=lambda b: len(_dedup_list(b.get("pages", []))), reverse=True)
    return filtered[:max_candidates]


def _merge_translated_brands(
    brands: List[Dict],
    model_name: str,
    distance_threshold: float = 0.25,
    device: str = "cpu",
) -> List[Dict]:
    """
    Merge bilingual / alias brand names using multilingual embeddings.
    """
    if len(brands) <= 1:
        return brands
    names = [b.get("name", "") for b in brands]
    embeddings = _embed_texts(names, model_name, device=device)

    clusterer = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clusterer.fit_predict(embeddings)
    cluster_map: Dict[int, List[Dict]] = {}
    for idx, cid in enumerate(labels):
        cluster_map.setdefault(cid, []).append(brands[idx])

    merged: List[Dict] = []
    for items in cluster_map.values():
        # choose representative with most pages; merge evidence/pages
        rep = sorted(items, key=lambda b: len(_dedup_list(b.get("pages", []))), reverse=True)[0]
        ev: List[str] = []
        pages: List = []
        for b in items:
            ev.extend(b.get("evidence", []))
            pages.extend(b.get("pages", []))
        rep2 = dict(rep)
        rep2["evidence"] = _dedup_list(rep2.get("evidence", []) + ev)
        rep2["pages"] = _dedup_list(rep2.get("pages", []) + pages)
        merged.append(rep2)
    return merged

def _chunk_sequence(seq: Sequence, chunk_size: int) -> List[Sequence]:
    if chunk_size is None or chunk_size <= 0:
        return [seq]
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def _split_text(text: str, max_chars: int = 8000) -> List[str]:
    """
    Split long text into reasonably sized chunks to keep prompts short.
    Tries to split on paragraph boundaries; falls back to raw slicing.
    """
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    paragraphs = text.split("\n\n")
    buf = []
    buf_len = 0
    for para in paragraphs:
        if buf_len + len(para) + 2 <= max_chars:
            buf.append(para)
            buf_len += len(para) + 2
        else:
            if buf:
                parts.append("\n\n".join(buf))
            buf = [para]
            buf_len = len(para)
            if buf_len > max_chars:
                # Hard split this overlong paragraph
                chunk = para
                while len(chunk) > max_chars:
                    parts.append(chunk[:max_chars])
                    chunk = chunk[max_chars:]
                if chunk:
                    buf = [chunk]
                    buf_len = len(chunk)
                else:
                    buf = []
                    buf_len = 0
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def _chunk_preview(text: str, max_chars: int = 64) -> str:
    """Single-line preview for chunk-level progress bars."""
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _looks_like_series_feature(model_text: str) -> bool:
    """Lightweight heuristic to catch series-level headings/ranges when LLM misses them."""
    t = str(model_text or "")
    lower = t.lower()
    if "系列" in t or "series" in lower:
        return True
    if re.search(r"[A-Za-z0-9]{2,}\\s*[~\\-]\\s*[A-Za-z0-9]{2,}", t):
        return True
    return False


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    t = str(text or "").lower()
    for kw in keywords or []:
        if kw and kw.lower() in t:
            return True
    return False


def _default_product_template() -> Dict:
    return {
        "brand": "",
        "category": "",
        "series": "",
        "product_model": "",
        "manufacturer": "",
        "refrigerant": "",
        "energy_efficiency_grade": "",
        "features": [],
        "key_components": [],
        "performance_specs": [],
        "fact_text": [],
        "evidence": [],
    }


def _append_error(error_log: Path, stage: str, meta: Dict, exc: Exception) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"stage": stage, "metadata": meta, "error": str(exc)},
                ensure_ascii=False,
            )
            + "\n"
        )


def _snapshot_brand_candidates(brands: Sequence[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for item in brands or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_brand_key(name) or _normalize_name(name)
        if not key:
            continue
        bucket = merged.setdefault(
            key,
            {
                "name": name,
                "evidence": [],
                "pages": [],
            },
        )
        bucket["evidence"].extend(item.get("evidence", []))
        bucket["pages"].extend(item.get("pages", []))
        if len(_dedup_list(item.get("pages", []))) > len(_dedup_list(bucket.get("pages", []))):
            bucket["name"] = name

    snapshots: List[Dict] = []
    for bucket in merged.values():
        snapshots.append(
            {
                "name": bucket.get("name", ""),
                "evidence": _dedup_list(bucket.get("evidence", [])),
                "pages": _dedup_list(bucket.get("pages", [])),
            }
        )
    snapshots.sort(key=lambda x: len(_dedup_list(x.get("pages", []))), reverse=True)
    return snapshots


def _build_dropped_brand_cache(
    all_candidates: Sequence[Dict],
    kept_brands: Sequence[Dict],
    brand_alias_map: Dict[str, List[str]],
) -> List[Dict]:
    keep_keys = set()
    for item in kept_brands or []:
        key = _normalize_brand_key(item.get("name", ""))
        if key:
            keep_keys.add(key)
    for canonical, aliases in (brand_alias_map or {}).items():
        canonical_key = _normalize_brand_key(canonical)
        if canonical_key:
            keep_keys.add(canonical_key)
        for alias in aliases or []:
            alias_key = _normalize_brand_key(alias)
            if alias_key:
                keep_keys.add(alias_key)

    primary_brand = ""
    if len(kept_brands or []) == 1:
        primary_brand = str((kept_brands or [])[0].get("name") or "").strip()

    dropped: List[Dict] = []
    for item in _snapshot_brand_candidates(all_candidates):
        key = _normalize_brand_key(item.get("name", ""))
        if key and key in keep_keys:
            continue
        dropped.append(
            {
                "name": item.get("name", ""),
                "evidence": _dedup_list(item.get("evidence", [])),
                "pages": _dedup_list(item.get("pages", [])),
                "drop_reason": "not_selected_in_stage_a_final_brands",
                "selected_primary_brand": primary_brand,
            }
        )
    return dropped


def _enforce_single_brand_series_scope(
    brands: Sequence[Dict],
    brand_to_series: Dict[str, List[Dict]],
    series_alias_map: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, List[str]]], bool]:
    normalized_series = _enrich_series_by_brand(brand_to_series)
    alias_map = dict(series_alias_map or {})
    if len(brands or []) != 1:
        return normalized_series, alias_map, False

    primary_brand = str((brands or [])[0].get("name") or "").strip()
    if not primary_brand:
        return normalized_series, alias_map, False

    changed = set(normalized_series.keys()) != {primary_brand}
    flattened: List[Dict] = []
    for brand, series_list in normalized_series.items():
        if brand != primary_brand:
            changed = True
        for series_item in series_list or []:
            entry = dict(series_item or {})
            series_name = (entry.get("name") or "").strip()
            if not series_name:
                continue
            if str(entry.get("brand") or "").strip() != primary_brand:
                changed = True
            entry["name"] = series_name
            entry["brand"] = primary_brand
            entry["pair_key"] = _pair_key(primary_brand, series_name)
            entry["series_key"] = _normalize_series_key(series_name)
            entry["evidence"] = _dedup_list(entry.get("evidence", []))
            entry["pages"] = _dedup_list(entry.get("pages", []))
            flattened.append(entry)

    if not flattened:
        collapsed_series = {primary_brand: []}
    else:
        merged_series = _merge_series_candidates(flattened)
        grouped: Dict[str, Dict] = {}
        for item in merged_series:
            series_name = (item.get("name") or "").strip()
            if not series_name:
                continue
            key = _normalize_series_key(series_name) or _normalize_name(series_name)
            bucket = grouped.get(key)
            if bucket is None:
                grouped[key] = dict(item)
                continue
            bucket["name"] = (
                _choose_series_canonical_name([bucket.get("name", ""), series_name])
                or bucket.get("name", "")
            )
            bucket["evidence"] = _dedup_list(bucket.get("evidence", []) + item.get("evidence", []))
            bucket["pages"] = _dedup_list(bucket.get("pages", []) + item.get("pages", []))
        collapsed_series = {primary_brand: list(grouped.values())}

    alias_grouped: Dict[str, List[str]] = {}
    for brand, mapping in alias_map.items():
        if brand != primary_brand and mapping:
            changed = True
        if not isinstance(mapping, dict):
            continue
        for canonical, aliases in mapping.items():
            canonical_name = str(canonical or "").strip()
            if not canonical_name:
                continue
            key = _normalize_series_key(canonical_name) or _normalize_name(canonical_name)
            slot = alias_grouped.setdefault(key, [])
            slot.append(canonical_name)
            for alias in aliases or []:
                alias_name = str(alias or "").strip()
                if alias_name:
                    slot.append(alias_name)

    collapsed_alias_inner: Dict[str, List[str]] = {}
    for series_item in collapsed_series.get(primary_brand, []):
        series_name = str(series_item.get("name") or "").strip()
        if not series_name:
            continue
        key = _normalize_series_key(series_name) or _normalize_name(series_name)
        aliases = _dedup_list(alias_grouped.get(key, []) + [series_name])
        collapsed_alias_inner[series_name] = aliases

    collapsed_alias_map = {primary_brand: collapsed_alias_inner}
    if alias_map != collapsed_alias_map:
        changed = True

    return collapsed_series, collapsed_alias_map, changed


def _enforce_single_brand_models_scope(
    brands: Sequence[Dict],
    models_by_pair: Dict[str, List[Dict]],
) -> Tuple[Dict[str, List[Dict]], bool]:
    normalized_models = _enrich_models_by_pair(models_by_pair)
    if len(brands or []) != 1:
        return normalized_models, False

    primary_brand = str((brands or [])[0].get("name") or "").strip()
    if not primary_brand:
        return normalized_models, False

    changed = False
    collapsed: Dict[str, List[Dict]] = {}
    for pair_key, model_list in normalized_models.items():
        _, series_name = _split_pair_key(pair_key)
        target_pair_key = _pair_key(primary_brand, series_name)
        if pair_key != target_pair_key:
            changed = True
        bucket = collapsed.setdefault(target_pair_key, [])
        for model_item in model_list or []:
            entry = dict(model_item or {})
            model_name = _canonicalize_model_name(entry.get("name") or "")
            if not model_name:
                continue
            if str(entry.get("brand") or "").strip() != primary_brand:
                changed = True
            entry["name"] = model_name
            entry["brand"] = primary_brand
            entry["series"] = series_name
            entry["pair_key"] = target_pair_key
            entry["model_key"] = _model_node_key(primary_brand, series_name, model_name)
            entry["evidence"] = _dedup_list(entry.get("evidence", []))
            entry["pages"] = _dedup_list(entry.get("pages", []))
            bucket.append(entry)

    for pair_key in list(collapsed.keys()):
        collapsed[pair_key] = _merge_model_candidates(collapsed[pair_key])

    return _enrich_models_by_pair(collapsed), changed


def _remap_model_conflicts_to_single_brand(
    brands: Sequence[Dict],
    model_conflicts: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], bool]:
    if len(brands or []) != 1:
        return model_conflicts, False
    if not model_conflicts:
        return model_conflicts, False

    primary_brand = str((brands or [])[0].get("name") or "").strip()
    if not primary_brand:
        return model_conflicts, False

    changed = False
    remapped: Dict[str, Dict] = {}
    for conflict_id, payload in model_conflicts.items():
        entry = dict(payload or {})
        if str(entry.get("brand") or "").strip() != primary_brand:
            changed = True
        entry["brand"] = primary_brand

        winner_pair_key = str(entry.get("winner_pair_key") or "")
        if winner_pair_key:
            _, winner_series = _split_pair_key(winner_pair_key)
            winner_target = _pair_key(primary_brand, winner_series)
            if winner_target != winner_pair_key:
                changed = True
            entry["winner_pair_key"] = winner_target

        remapped_candidates: List[Dict] = []
        for candidate in entry.get("candidates") or []:
            cand = dict(candidate or {})
            pair_key = str(cand.get("pair_key") or "")
            if pair_key:
                _, cand_series = _split_pair_key(pair_key)
                target_pair_key = _pair_key(primary_brand, cand_series)
                if target_pair_key != pair_key:
                    changed = True
                cand["pair_key"] = target_pair_key
            remapped_candidates.append(cand)
        entry["candidates"] = remapped_candidates

        model_key = str(entry.get("model_key") or "")
        target_conflict_id = f"{primary_brand}::{model_key}" if model_key else str(conflict_id)
        if target_conflict_id != str(conflict_id):
            changed = True
        remapped[target_conflict_id] = entry

    return remapped, changed


def _enforce_single_brand_products_scope(
    brands: Sequence[Dict],
    products: Sequence[Dict],
) -> Tuple[List[Dict], bool]:
    if len(brands or []) != 1:
        return list(products or []), False

    primary_brand = str((brands or [])[0].get("name") or "").strip()
    if not primary_brand:
        return list(products or []), False

    changed = False
    normalized: List[Dict] = []
    for product in products or []:
        entry = dict(product or {})
        if str(entry.get("brand") or "").strip() != primary_brand:
            changed = True
        entry["brand"] = primary_brand
        normalized.append(entry)
    return normalized, changed


def _enforce_single_brand_series_feature_scope(
    brands: Sequence[Dict],
    series_feature_map: Dict[str, Dict[str, List[Dict]]],
) -> Tuple[Dict[str, Dict[str, List[Dict]]], bool]:
    if len(brands or []) != 1:
        return series_feature_map, False

    primary_brand = str((brands or [])[0].get("name") or "").strip()
    if not primary_brand:
        return series_feature_map, False

    changed = False
    collapsed: Dict[str, List[Dict]] = {}
    for brand, series_map in (series_feature_map or {}).items():
        if brand != primary_brand and series_map:
            changed = True
        for series_name, features in (series_map or {}).items():
            bucket = collapsed.setdefault(series_name, [])
            bucket.extend(features or [])

    deduped: Dict[str, List[Dict]] = {}
    for series_name, features in collapsed.items():
        seen = set()
        unique_features: List[Dict] = []
        for feature in features or []:
            key = json.dumps(feature or {}, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique_features.append(feature)
        deduped[series_name] = unique_features

    result = {primary_brand: deduped}
    if result != (series_feature_map or {}):
        changed = True
    return result, changed


# --------------------------------------------------------------------------- #
# Core extractor
# --------------------------------------------------------------------------- #


class StagedRelationExtractor:
    def __init__(self, config: Optional[Dict] = None, log_name: str = "relation_extractor_v2") -> None:
        self.config = dict(RELATION_EXTRACTOR_CONFIG)
        if config:
            self.config.update(config)

        self.lc_llm = ChatOpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
            model=self.config["model"],
            temperature=0,
            timeout=self.config.get("timeout", 300),
        )
        self.semaphore = asyncio.Semaphore(_get_concurrency(self.config, "max_concurrent", 5))
        self.logger = self._setup_logger(log_name)
        self.brand_alias_map: Dict[str, List[str]] = {}
        self.brand_candidates_all: List[Dict] = []
        self.brand_dropped: List[Dict] = []
        self.series_alias_map: Dict[str, Dict[str, List[str]]] = {}
        self.series_feature_map: Dict[str, Dict[str, List[Dict]]] = {}
        self.series_component_map: Dict[str, Dict[str, List[str]]] = {}
        self.models_by_pair: Dict[str, List[Dict]] = {}
        self.model_page_stats: Dict[str, List] = {}
        self.model_conflicts: Dict[str, Dict] = {}
        self.model_redirects: List[Dict] = []

    def _retrieve_pages(
        self, keywords: List[str], pages: Sequence[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """
        Keyword-only retrieval for stage evidence recall.
        BM25/embedding paths are intentionally disabled.
        """
        return _select_pages_by_keywords(
            pages,
            keywords,
            top_k=int(
                self.config.get(
                    "keyword_retrieval_top_k",
                    self.config.get("retrieval_top_k", 0),
                )
            ),
            min_hits=int(self.config.get("keyword_retrieval_min_hits", 1)),
            match_mode=str(self.config.get("keyword_retrieval_match_mode", "any")),
        )

    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        level = getattr(logging, str(self.config.get("log_level", "INFO")).upper(), logging.INFO)
        logger.setLevel(level)
        logger.propagate = False

        log_file = Path(self.config.get("log_file", "logs/relation_extractor.log"))
        log_file = log_file.with_name("relation_extractor_v2.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == log_file
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

        return logger

    # -------------------------- low-level LLM call ------------------------- #
    async def _acall(
        self,
        messages: List[Dict],
        schema: Dict,
        schema_name: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        return await self._acall_langchain(messages, schema, schema_name, metadata)

    async def _acall_langchain(
        self,
        messages: List[Dict],
        schema: Dict,
        schema_name: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        del metadata

        lc_messages = []
        for msg in messages or []:
            role = str((msg or {}).get("role") or "").strip().lower()
            content = str((msg or {}).get("content") or "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        llm = self.lc_llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        )
        async with self.semaphore:
            response = await llm.ainvoke(lc_messages)

        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, str):
            content = raw_content.strip()
        elif isinstance(raw_content, list):
            text_parts: List[str] = []
            for part in raw_content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts).strip()
        else:
            content = str(raw_content or "").strip()
        if not content:
            raise ValueError(f"Empty response for {schema_name}")
        try:
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            preview = content[:200]
            raise ValueError(f"Invalid JSON for {schema_name}: {preview}") from exc

    # -------------------------- Stage A: brand ----------------------------- #
    async def _extract_brands_single(
        self, text: str, metadata: Dict, error_log: Path
    ) -> List[Dict]:
        try:
            payload = await self._acall(
                build_brand_messages(text, str(metadata.get("page", ""))),
                BRAND_SCHEMA,
                "brand_schema",
                metadata,
            )
            brands = []
            for item in payload.get("brands", []):
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                brands.append(
                    {
                        "name": name,
                        "evidence": item.get("evidence", []),
                        "pages": item.get("pages") or [metadata.get("page")],
                    }
                )
            return brands
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "brand_extract", metadata, exc)
            self.logger.error("Brand extract failed on page %s: %s", metadata.get("page"), exc)
            return []

    async def _extract_brands_cluster(
        self, text: str, page_labels: List, error_log: Path
    ) -> List[Dict]:
        """Secondary pass: run brand extraction on clustered multi-page text."""
        try:
            payload = await self._acall(
                build_brand_messages(text, f"pages {page_labels}"),
                BRAND_SCHEMA,
                "brand_schema",
                {"pages": page_labels},
            )
            brands = []
            for item in payload.get("brands", []):
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                brands.append(
                    {
                        "name": name,
                        "evidence": item.get("evidence", []),
                        "pages": item.get("pages") or page_labels,
                    }
                )
            return brands
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "brand_extract_cluster", {"pages": page_labels}, exc)
            self.logger.error("Brand extract (cluster) failed on pages %s: %s", page_labels, exc)
            return []

    def _cluster_pages_for_brands(
        self, pages: Sequence[Tuple[str, Dict]], cluster_size: int
    ) -> List[Tuple[str, List]]:
        """Group pages into fixed-size clusters to re-scan brands on aggregated text."""
        clusters: List[Tuple[str, List]] = []
        buffer: List[Tuple[str, Dict]] = []
        for text, meta in pages:
            buffer.append((text, meta))
            if len(buffer) >= cluster_size:
                clusters.append(self._combine_cluster(buffer))
                buffer = []
        if buffer:
            clusters.append(self._combine_cluster(buffer))
        return clusters

    def _cluster_pages_for_brands_embed(
        self,
        pages: Sequence[Tuple[str, Dict]],
        model_name: str,
        distance_threshold: float = 0.25,
        device: str = "cpu",
    ) -> List[Tuple[str, List]]:
        """
        Cluster page texts using embeddings + agglomerative clustering (cosine).
        Returns list of (combined_text, page_labels).
        """
        if not pages:
            return []
        if len(pages) == 1:
            text, meta = pages[0]
            return [self._combine_cluster([(text, meta)])]
        texts = [text for text, _ in pages]
        labels = [meta.get("page") for _, meta in pages]
        embeddings = _embed_texts(texts, model_name, device=device)

        clusterer = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
        clusters_idx = clusterer.fit_predict(embeddings)

        cluster_map: Dict[int, List[Tuple[str, Dict]]] = {}
        for idx, cid in enumerate(clusters_idx):
            cluster_map.setdefault(cid, []).append((texts[idx], {"page": labels[idx]}))

        clusters: List[Tuple[str, List]] = []
        for items in cluster_map.values():
            clusters.append(self._combine_cluster(items))
        return clusters

    def _combine_cluster(self, buffer: List[Tuple[str, Dict]]) -> Tuple[str, List]:
        page_labels = [item[1].get("page") for item in buffer]
        combined_text = "\n\n".join(
            [f"<<PAGE {meta.get('page', '')}>>\n{text}" for text, meta in buffer]
        )
        return combined_text, page_labels

    def _cluster_brand_names(
        self,
        brands: List[Dict],
        model_name: str,
        distance_threshold: float = 0.25,
        device: str = "cpu",
    ) -> List[List[Dict]]:
        """
        Cluster brand names using embeddings to group aliases/variants.
        """
        if len(brands) <= 1:
            return [brands] if brands else []
        texts = [b.get("name", "") for b in brands]
        embeddings = _embed_texts(texts, model_name, device=device)

        clusterer = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
        labels = clusterer.fit_predict(embeddings)
        cluster_map: Dict[int, List[Dict]] = {}
        for idx, cid in enumerate(labels):
            cluster_map.setdefault(cid, []).append(brands[idx])
        return list(cluster_map.values())

    async def _refine_brands(self, brands: List[Dict], error_log: Path) -> List[Dict]:
        """
        Optional LLM-assisted brand pruning after recall-heavy extraction.
        Goal: reduce to canonical brands, not to fill gaps.
        """
        if not brands:
            return []

        refined: List[Dict] = list(brands)
        min_refine_count = int(self.config.get("brand_refine_min_count", 3))
        if len(brands) >= min_refine_count:
            # prepare clusters by name similarity
            clusters = self._cluster_brand_names(
                brands,
                model_name=self.config.get("brand_refine_embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
                distance_threshold=float(self.config.get("brand_refine_threshold", 0.35)),
                device=self.config.get("brand_refine_device", "cpu"),
            )

            refined = []
            pbar = None
            if getattr(self, "_show_progress", False):
                pbar = tqdm(total=len(clusters), desc="Stage A: brand refine", unit="cluster")
            for cluster in clusters:
                # rank candidates by page coverage
                cluster_sorted = sorted(
                    cluster, key=lambda b: len(_dedup_list(b.get("pages", []))), reverse=True
                )
                candidate_payload = [
                    {
                        "name": b.get("name", ""),
                        "pages": len(_dedup_list(b.get("pages", []))),
                        "evidence": (b.get("evidence") or [])[:3],
                    }
                    for b in cluster_sorted[:5]
                ]
                try:
                    result = await self._acall(
                        build_brand_filter_messages(candidate_payload),
                        BRAND_FILTER_SCHEMA,
                        "brand_filter_schema",
                        {"cluster_size": len(cluster_sorted)},
                    )
                    keep_names = [n.strip() for n in result.get("keep", []) if n and n.strip()]
                except Exception as exc:  # noqa: BLE001
                    _append_error(error_log, "brand_refine", {"cluster": candidate_payload}, exc)
                    self.logger.error("Brand refine failed on cluster %s: %s", candidate_payload, exc)
                    keep_names = []

                # choose canonical brand: prefer LLM keep, else top by pages
                chosen = None
                if keep_names:
                    norm_map = {_normalize_brand_key(b.get("name", "")): b for b in cluster_sorted}
                    for name in keep_names:
                        key = _normalize_brand_key(name)
                        if key in norm_map:
                            chosen = norm_map[key]
                            break
                if chosen is None:
                    chosen = cluster_sorted[0]

                # merge evidence/pages from entire cluster into chosen
                merged_ev: List[str] = []
                merged_pages: List = []
                for b in cluster:
                    merged_ev.extend(b.get("evidence", []))
                    merged_pages.extend(b.get("pages", []))
                chosen_merged = dict(chosen)
                chosen_merged["evidence"] = _dedup_list(chosen_merged.get("evidence", []) + merged_ev)
                chosen_merged["pages"] = _dedup_list(chosen_merged.get("pages", []) + merged_pages)

                # avoid duplicates in refined list
                if all(_normalize_brand_key(chosen_merged["name"]) != _normalize_brand_key(x["name"]) for x in refined):
                    refined.append(chosen_merged)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

        # final cap to avoid long tail
        max_candidates = int(self.config.get("brand_max_candidates", 12))
        refined = sorted(
            refined, key=lambda b: len(_dedup_list(b.get("pages", []))), reverse=True
        )[:max_candidates]

        # bilingual / alias merge using multilingual embeddings
        if len(refined) > 1:
            refined = _merge_translated_brands(
                refined,
                model_name=self.config.get(
                    "brand_translate_embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ),
                distance_threshold=float(self.config.get("brand_translate_threshold", 0.25)),
                device=self.config.get("brand_translate_device", "cpu"),
            )

        # global LLM pass to drop residual series/model/non-brand items
        pre_global_refined = list(refined)
        try:
            candidate_payload = [
                {
                    "name": b.get("name", ""),
                    "pages": len(_dedup_list(b.get("pages", []))),
                    "evidence": (b.get("evidence") or [])[:3],
                }
                for b in refined
            ]
            result = await self._acall(
                build_brand_global_filter_messages(candidate_payload),
                BRAND_FILTER_SCHEMA,
                "brand_filter_schema",
                {"stage": "brand_global_filter"},
            )
            keep_names = { _normalize_brand_key(n) for n in result.get("keep", []) if n }
            refined = [b for b in refined if _normalize_brand_key(b.get("name", "")) in keep_names]
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "brand_global_filter", {}, exc)
            self.logger.error("Brand global filter failed: %s", exc)

        if not refined and pre_global_refined:
            refined = sorted(
                pre_global_refined,
                key=lambda b: len(_dedup_list(b.get("pages", []))),
                reverse=True,
            )[:1]

        # canonicalize to Chinese brand form
        refined, alias_map = await self._canonicalize_brands(refined, error_log)
        if bool(self.config.get("single_brand_per_document", True)) and len(refined) > 1:
            refined = await self._select_primary_brand(refined, error_log)
            keep_keys = {_normalize_brand_key(item.get("name", "")) for item in refined}
            alias_map = {
                canonical: aliases
                for canonical, aliases in alias_map.items()
                if _normalize_brand_key(canonical) in keep_keys
            }
        self.brand_alias_map = alias_map
        return refined

    async def _select_primary_brand(self, brands: List[Dict], error_log: Path) -> List[Dict]:
        """
        Document-level brand decision: keep one primary manufacturer brand.
        """
        if not brands:
            return []
        if len(brands) == 1:
            return brands

        candidates = sorted(
            brands, key=lambda b: len(_dedup_list(b.get("pages", []))), reverse=True
        )
        payload = [
            {
                "name": item.get("name", ""),
                "pages": len(_dedup_list(item.get("pages", []))),
                "evidence": (item.get("evidence") or [])[:3],
            }
            for item in candidates[:12]
        ]
        try:
            result = await self._acall(
                build_brand_primary_messages(payload),
                BRAND_FILTER_SCHEMA,
                "brand_primary_schema",
                {"stage": "brand_primary"},
            )
            keep_names = [_normalize_brand_key(name) for name in result.get("keep", []) if name]
            if keep_names:
                kept = [item for item in candidates if _normalize_brand_key(item.get("name", "")) in set(keep_names)]
                if kept:
                    return [kept[0]]
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "brand_primary", {}, exc)
            self.logger.error("Brand primary selection failed: %s", exc)

        # deterministic fallback by page coverage if LLM selection fails.
        return [candidates[0]]

    async def _canonicalize_brands(
        self, brands: List[Dict], error_log: Path
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        if not brands:
            return brands, {}
        payload = [
            {"name": b.get("name", ""), "evidence": (b.get("evidence") or [])[:3]}
            for b in brands
        ]
        try:
            result = await self._acall(
                build_brand_canon_messages(payload),
                BRAND_CANON_SCHEMA,
                "brand_canon_schema",
                {"count": len(brands)},
            )
            mapping = {
                _normalize_brand_key(item.get("original", "")): item.get("canonical_cn", "").strip()
                for item in result.get("items", [])
            }
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "brand_canon", {}, exc)
            self.logger.error("Brand canonicalization failed: %s", exc)
            return brands, {}

        merged: Dict[str, Dict] = {}
        alias_map: Dict[str, List[str]] = {}
        for b in brands:
            orig_key = _normalize_brand_key(b.get("name", ""))
            canon = mapping.get(orig_key) or b.get("name", "")
            canon_key = _normalize_brand_key(canon)
            if canon_key not in merged:
                merged[canon_key] = dict(b)
                merged[canon_key]["name"] = canon
                alias_map[canon] = [b.get("name", "")]
            else:
                merged[canon_key]["evidence"].extend(b.get("evidence", []))
                merged[canon_key]["pages"].extend(b.get("pages", []))
                alias_map.setdefault(canon, []).append(b.get("name", ""))
        # dedup fields
        for v in merged.values():
            v["evidence"] = _dedup_list(v.get("evidence", []))
            v["pages"] = _dedup_list(v.get("pages", []))
        for k in alias_map:
            alias_map[k] = _dedup_list(alias_map[k])
        return list(merged.values()), alias_map

    async def extract_brands(self, pages: Sequence[Tuple[str, Dict]], error_log: Path) -> List[Dict]:
        self.brand_candidates_all = []
        self.brand_dropped = []
        tasks = [
            asyncio.create_task(self._extract_brands_single(text, meta, error_log))
            for text, meta in pages
        ]
        if getattr(self, "_show_progress", False):
            results = []
            with tqdm(total=len(tasks), desc="Stage A: brands", unit="page") as pbar:
                for coro in asyncio.as_completed(tasks):
                    res = await coro
                    results.append(res)
                    pbar.update(1)
        else:
            results = await asyncio.gather(*tasks)

        merged: Dict[str, Dict] = {}
        for brand_list in results:
            for b in brand_list:
                key = _normalize_name(b["name"])
                if key not in merged:
                    merged[key] = b
                else:
                    merged[key]["evidence"].extend(b.get("evidence", []))
                    merged[key]["pages"].extend(b.get("pages", []))
        # second-pass merge for near-duplicates/aliases
        merged_list = list(merged.values())
        merged_list = _merge_brand_candidates(merged_list)

        # clustered re-scan to boost recall
        cluster_size = int(self.config.get("brand_cluster_size", 5))
        if cluster_size > 1:
            cluster_mode = self.config.get("brand_cluster_mode", "fixed")
            clusters: List[Tuple[str, List]]
            if cluster_mode == "embed":
                clusters = self._cluster_pages_for_brands_embed(
                    pages,
                    model_name=self.config.get(
                        "brand_cluster_embed_model", "sentence-transformers/all-MiniLM-L6-v2"
                    ),
                    distance_threshold=float(self.config.get("brand_cluster_threshold", 0.25)),
                    device=self.config.get("brand_cluster_device", "cpu"),
                )
            else:
                clusters = self._cluster_pages_for_brands(pages, cluster_size=cluster_size)
            if clusters:
                c_tasks = [
                    asyncio.create_task(self._extract_brands_cluster(text, labels, error_log))
                    for text, labels in clusters
                ]
                if getattr(self, "_show_progress", False):
                    cluster_results = []
                    with tqdm(
                        total=len(clusters),
                        desc="Stage A: brand clusters",
                        unit="cluster",
                    ) as pbar:
                        for coro in asyncio.as_completed(c_tasks):
                            res = await coro
                            cluster_results.append(res)
                            pbar.update(1)
                else:
                    cluster_results = await asyncio.gather(*c_tasks)
                for brand_list in cluster_results:
                    merged_list.extend(brand_list)
                merged_list = _merge_brand_candidates(merged_list)
        all_candidates = _snapshot_brand_candidates(merged_list)
        # Optional lightweight pre-prune before LLM refinement.
        if bool(self.config.get("brand_use_pre_prune", False)):
            merged_list = _prune_brands(
                merged_list,
                min_page_ratio=float(self.config.get("brand_min_page_ratio", 0.15)),
                max_candidates=int(self.config.get("brand_max_candidates", 12)),
            )
        refined_brands = await self._refine_brands(merged_list, error_log)
        self.brand_candidates_all = all_candidates
        self.brand_dropped = _build_dropped_brand_cache(
            all_candidates,
            refined_brands,
            self.brand_alias_map,
        )
        return refined_brands

    # -------------------------- Pipeline facades -------------------------- #
    async def run_brand_stage(self, pages: Sequence[Tuple[str, Dict]], error_log: Path) -> List[Dict]:
        return await self.extract_brands(pages, error_log)

    async def run_series_stage(
        self, brands: List[Dict], pages: Sequence[Tuple[str, Dict]], error_log: Path
    ) -> Dict[str, List[Dict]]:
        return await self.extract_series(brands, pages, error_log)

    async def run_model_stage(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> Dict[str, List[Dict]]:
        model_map, model_pages = await self.extract_models(brand_to_series, pages, error_log)
        self.models_by_pair = model_map
        self.model_page_stats = model_pages
        return model_map

    async def run_product_stage(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> List[Dict]:
        return await self.extract_products(brand_to_series, pages, error_log)

    # -------------------------- Stage B: series ---------------------------- #
    async def _extract_series_for_brand(
        self,
        brand: str,
        combined_text: str,
        error_log: Path,
        page_refs: Optional[List] = None,
        stage_a_pages: Optional[List] = None,
        show_chunk_progress: bool = True,
    ) -> List[Dict]:
        chunk_pbar = None
        try:
            chunks = _split_text(combined_text, self.config.get("max_chars_per_call", 8000))
            merged = {}
            if getattr(self, "_show_progress", False) and show_chunk_progress:
                brand_label = (brand or "unknown")[:16]
                chunk_pbar = tqdm(
                    total=len(chunks),
                    desc=f"Stage B chunks[{brand_label}]",
                    unit="chunk",
                    leave=False,
                )
            for chunk in chunks:
                if chunk_pbar:
                    chunk_pbar.set_postfix_str(
                        _chunk_preview(chunk, int(self.config.get("chunk_progress_preview_chars", 64)))
                    )
                payload = await self._acall(
                    build_series_messages(
                        brand,
                        chunk,
                        stage_a_pages=stage_a_pages,
                        chunk_pages=page_refs,
                    ),
                    SERIES_SCHEMA,
                    "series_schema",
                    {"brand": brand, "stage_a_pages": stage_a_pages or [], "chunk_pages": page_refs or []},
                )
                if chunk_pbar:
                    chunk_pbar.update(1)
                for item in payload.get("series", []):
                    name = (item.get("name") or "").strip()
                    if not name:
                        continue
                    key = _normalize_name(name)
                    if key not in merged:
                        merged[key] = {
                            "name": name,
                            "evidence": [],
                            "pages": [],
                        }
                    merged[key]["evidence"].extend(item.get("evidence", []))
                    merged[key]["pages"].extend(item.get("pages", []))
            if chunk_pbar:
                chunk_pbar.close()
            merged_list = []
            for v in merged.values():
                v["evidence"] = _dedup_list(v.get("evidence", []))
                v["pages"] = _dedup_list(v.get("pages", []))
                merged_list.append(v)
            if page_refs:
                chunk_pages = _dedup_list([p for p in page_refs if p is not None and str(p) != ""])
                for item in merged_list:
                    merged_pages = item.get("pages", [])
                    if not merged_pages:
                        item["pages"] = chunk_pages
            return merged_list
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "series_extract", {"brand": brand}, exc)
            self.logger.error("Series extract failed for brand %s: %s", brand, exc)
            return []
        finally:
            if chunk_pbar:
                chunk_pbar.close()

    async def _extract_series_single(
        self,
        brand: str,
        text: str,
        metadata: Dict,
        error_log: Path,
    ) -> List[Dict]:
        page = metadata.get("page")
        return await self._extract_series_for_brand(
            brand,
            text,
            error_log,
            page_refs=[page] if page is not None and str(page) != "" else None,
            stage_a_pages=[],
            show_chunk_progress=False,
        )

    async def _extract_series_cluster(
        self,
        brand: str,
        text: str,
        page_labels: List,
        error_log: Path,
    ) -> List[Dict]:
        return await self._extract_series_for_brand(
            brand,
            text,
            error_log,
            page_refs=_dedup_list([p for p in (page_labels or []) if p is not None and str(p) != ""]),
            stage_a_pages=[],
            show_chunk_progress=False,
        )

    async def _extract_series_features_for_brand(
        self,
        brand: str,
        series_names: List[str],
        relevant_pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
        source_category: str = "series_feature_stage_b",
    ) -> None:
        """
        Stage-B series_feature extraction:
        produce series-level features/components before Stage C/D.
        """
        if not brand or not series_names or not relevant_pages:
            return

        chunk_size = self._resolve_series_chunk_size()
        page_chunks = _chunk_sequence(relevant_pages, chunk_size)
        for chunk_pages in page_chunks:
            combined = _combine_pages(chunk_pages, self.config)
            chunk_refs = _dedup_list(
                [
                    meta.get("page")
                    for _, meta in chunk_pages
                    if meta.get("page") is not None and str(meta.get("page")) != ""
                ]
            )
            try:
                payload = await self._acall(
                    build_series_feature_messages(
                        brand,
                        series_names,
                        combined,
                        chunk_pages=chunk_refs,
                    ),
                    SERIES_FEATURE_SCHEMA,
                    "series_feature_schema",
                    {"brand": brand, "series_count": len(series_names), "chunk_pages": chunk_refs},
                )
            except Exception as exc:  # noqa: BLE001
                _append_error(
                    error_log,
                    "series_feature_extract",
                    {"brand": brand, "chunk_pages": chunk_refs},
                    exc,
                )
                self.logger.warning(
                    "Stage B series_feature extract failed for brand=%s pages=%s: %s",
                    brand,
                    chunk_refs,
                    exc,
                )
                continue

            alias_map = self.series_alias_map.get(brand, {})
            for item in payload.get("items", []) or []:
                series_guess = str(item.get("series") or "").strip()
                canonical = _match_series_name(series_guess, brand, alias_map)
                if not canonical:
                    # fallback: direct hit on current series list
                    for s in series_names:
                        if _series_merge_key(s) == _series_merge_key(series_guess):
                            canonical = s
                            break
                if not canonical:
                    continue

                entry = {
                    "title": str(item.get("title") or "").strip() or canonical,
                    "fact_text": _dedup_list([str(x).strip() for x in (item.get("fact_text") or []) if str(x).strip()]),
                    "features": _dedup_list([str(x).strip() for x in (item.get("features") or []) if str(x).strip()]),
                    "key_components": _dedup_list([str(x).strip() for x in (item.get("key_components") or []) if str(x).strip()]),
                    "performance_specs": [],
                    "evidence": _dedup_list([str(x).strip() for x in (item.get("evidence") or []) if str(x).strip()]),
                    "source_category": source_category,
                    "series_feature_flag": True,
                }
                pages_field = item.get("pages") or []
                if pages_field:
                    entry["pages"] = _dedup_list(pages_field)
                self._store_series_feature_entry(brand, canonical, entry)
                if entry.get("key_components"):
                    self._store_series_components(brand, canonical, entry.get("key_components") or [])

    def _store_series_feature_entry(
        self,
        brand: str,
        series: str,
        entry: Dict,
    ) -> None:
        brand_bucket = self.series_feature_map.setdefault(brand, {})
        feature_list = brand_bucket.setdefault(series or "", [])
        key = json.dumps(entry or {}, ensure_ascii=False, sort_keys=True)
        for existing in feature_list:
            if json.dumps(existing or {}, ensure_ascii=False, sort_keys=True) == key:
                return
        feature_list.append(entry)

    def _resolve_series_chunk_size(self) -> int:
        """
        Stage-B pages are grouped into 2 or 3 pages per LLM call.
        """
        raw = int(
            self.config.get(
                "series_brand_chunk_pages",
                self.config.get("series_page_chunk_size", 2),
            )
        )
        return 2 if raw <= 2 else 3

    def _collect_series_context_pages(
        self,
        brand_item: Dict,
        pages: Sequence[Tuple[str, Dict]],
        keywords: List[str],
    ) -> Tuple[List[Tuple[str, Dict]], List]:
        """
        Prefer Stage-A brand evidence pages + following N pages for Stage-B context.
        Fall back to keyword retrieval when evidence pages are unavailable.
        """
        stage_a_pages = _dedup_list(
            [
                p
                for p in (brand_item.get("pages", []) or [])
                if p is not None and str(p) != ""
            ]
        )
        use_brand_evidence = bool(self.config.get("series_from_brand_evidence", True))

        if use_brand_evidence and stage_a_pages:
            follow_after = int(self.config.get("series_brand_follow_pages", 1))
            from_stage_a = _select_pages_by_numbers_with_following(
                pages,
                stage_a_pages,
                follow_after=follow_after,
            )
            if from_stage_a:
                return from_stage_a, stage_a_pages

        return self._retrieve_pages(keywords, pages), stage_a_pages

    async def _review_series_for_brand(
        self, brand: str, candidates: List[Dict], error_log: Path
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        LLM-based review/canonicalization for Stage B candidates.
        Mirrors Stage A's refine+canon flow while preserving high recall with guarded fallback.
        """
        if not candidates:
            return [], {}
        candidates = _sanitize_series_items(candidates)
        if not candidates:
            return [], {}

        payload = [
            {
                "name": s.get("name", ""),
                "pages": len(_dedup_list(s.get("pages", []))),
                "evidence": (s.get("evidence") or [])[:2],
            }
            for s in candidates
        ]
        try:
            result = await self._acall(
                build_series_review_messages(brand, payload),
                SERIES_REVIEW_SCHEMA,
                "series_review_schema",
                {"brand": brand, "count": len(payload)},
            )
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "series_review", {"brand": brand}, exc)
            self.logger.error("Series review failed for brand %s: %s", brand, exc)
            return candidates, {}

        # map by normalized original name for robust matching
        review_map: Dict[str, Dict] = {}
        for item in result.get("items", []):
            original_name = item.get("original", "")
            for key in (
                _series_merge_key(original_name),
                _normalize_series_key(original_name),
            ):
                if key:
                    review_map[key] = item

        merged: Dict[str, Dict] = {}
        alias_map: Dict[str, List[str]] = {}
        dropped = 0
        rescued = 0
        for series in candidates:
            original = _clean_series_name_text(series.get("name") or "")
            if not _is_hvac_series_candidate(original):
                dropped += 1
                continue
            original_key = _series_merge_key(original) or _normalize_series_key(original)
            review = review_map.get(original_key) or review_map.get(_normalize_series_key(original))

            # Keep unmatched outputs to avoid accidental recall collapse.
            if review is None:
                canonical = original
                keep_flag = True
            else:
                keep_flag = bool(review.get("keep")) and review.get("kind") == "series"
                canonical = _clean_series_name_text((review.get("canonical") or original).strip())
                if not canonical:
                    canonical = original

            if not keep_flag and (_looks_like_code_series_name(original) or _looks_like_code_series_name(canonical)):
                keep_flag = True
                rescued += 1

            if not keep_flag:
                dropped += 1
                continue

            canonical_key = _series_merge_key(canonical) or _normalize_series_key(canonical)
            if not canonical_key:
                canonical = original
                canonical_key = original_key
            if canonical_key not in merged:
                merged[canonical_key] = {
                    "names": [canonical, original],
                    "evidence": list(series.get("evidence", [])),
                    "pages": list(series.get("pages", [])),
                }
            else:
                merged[canonical_key]["names"].extend([canonical, original])
                merged[canonical_key]["evidence"].extend(series.get("evidence", []))
                merged[canonical_key]["pages"].extend(series.get("pages", []))

        # Evidence-level supplement: recover coded sub-series hidden in long rows.
        for series in candidates:
            pages = list(series.get("pages", []))
            for snippet in (series.get("evidence") or []):
                for derived in _extract_code_series_from_text(str(snippet)):
                    derived = _clean_series_name_text(derived)
                    if not _is_hvac_series_candidate(derived):
                        continue
                    derived_key = _series_merge_key(derived) or _normalize_series_key(derived)
                    if not derived_key:
                        continue
                    if derived_key not in merged:
                        merged[derived_key] = {
                            "names": [derived],
                            "evidence": [str(snippet)],
                            "pages": pages,
                        }
                    else:
                        merged[derived_key]["names"].append(derived)
                        merged[derived_key]["evidence"].append(str(snippet))
                        merged[derived_key]["pages"].extend(pages)

        reviewed: List[Dict] = []
        for value in merged.values():
            candidate_names = _dedup_list(
                [
                    _clean_series_name_text(name)
                    for name in value.get("names", [])
                    if _is_hvac_series_candidate(name)
                ]
            )
            canonical_name = _choose_series_canonical_name(candidate_names)
            if not canonical_name:
                canonical_name = candidate_names[0] if candidate_names else ""
            if not _is_hvac_series_candidate(canonical_name):
                continue
            item = {
                "name": canonical_name,
                "evidence": _dedup_list(value.get("evidence", [])),
                "pages": _dedup_list(value.get("pages", [])),
            }
            reviewed.append(item)
            if canonical_name:
                alias_map[canonical_name] = candidate_names

        self.logger.info(
            "Series review for brand=%s: candidates=%s dropped=%s rescued=%s kept=%s",
            brand,
            len(candidates),
            dropped,
            rescued,
            len(reviewed),
        )

        reviewed, alias_map = await self._canonicalize_series_groups_llm(
            brand,
            reviewed,
            alias_map,
            error_log,
        )

        return reviewed, alias_map

    async def _canonicalize_series_groups_llm(
        self,
        brand: str,
        reviewed: List[Dict],
        alias_map: Dict[str, List[str]],
        error_log: Path,
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Stage-B second-pass canonicalization:
        choose one stable canonical name per merged alias group via LLM.
        """
        if not reviewed:
            return reviewed, alias_map or {}

        group_meta: Dict[str, Dict] = {}
        for item in reviewed:
            name = _clean_series_name_text(item.get("name") or "")
            if not _is_hvac_series_candidate(name):
                continue
            group_key = _series_merge_key(name) or _normalize_series_key(name)
            if not group_key:
                continue

            aliases = [name] + list((alias_map or {}).get(name, []))
            aliases = _dedup_list(
                [
                    _clean_series_name_text(alias)
                    for alias in aliases
                    if _clean_series_name_text(alias) and _is_hvac_series_candidate(alias)
                ]
            )
            if not aliases:
                aliases = [name]

            evidence = _dedup_list(item.get("evidence", []))
            pages = _dedup_list(item.get("pages", []))

            if group_key not in group_meta:
                group_meta[group_key] = {
                    "aliases": aliases,
                    "evidence": evidence,
                    "pages": pages,
                    "fallback_name": name,
                }
            else:
                group_meta[group_key]["aliases"].extend(aliases)
                group_meta[group_key]["evidence"].extend(evidence)
                group_meta[group_key]["pages"].extend(pages)

        if not group_meta:
            return reviewed, alias_map or {}

        groups_payload = [
            {
                "group_key": group_key,
                "aliases": _dedup_list(meta.get("aliases", []))[:12],
                "evidence": _dedup_list(meta.get("evidence", []))[:2],
            }
            for group_key, meta in group_meta.items()
        ]

        try:
            result = await self._acall(
                build_series_canon_messages(brand, groups_payload),
                SERIES_CANON_SCHEMA,
                "series_canon_schema",
                {"brand": brand, "groups": len(groups_payload)},
            )
            canon_items = result.get("items", []) if isinstance(result, dict) else []
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "series_canon", {"brand": brand}, exc)
            self.logger.warning("Series canonicalization failed for brand %s: %s", brand, exc)
            return reviewed, alias_map or {}

        mapping: Dict[str, str] = {}
        for row in canon_items:
            group_key = _series_merge_key(row.get("group_key", "")) or _normalize_series_key(
                row.get("group_key", "")
            )
            canonical = _clean_series_name_text(row.get("canonical", ""))
            if group_key and canonical:
                mapping[group_key] = canonical

        merged: Dict[str, Dict] = {}
        new_alias_map: Dict[str, List[str]] = {}

        for group_key, meta in group_meta.items():
            aliases = _dedup_list(meta.get("aliases", []))
            canonical = mapping.get(group_key, "") or meta.get("fallback_name", "")
            if canonical not in aliases:
                canonical = meta.get("fallback_name", "") or canonical

            canonical = _clean_series_name_text(canonical)
            if (
                not canonical
                or not _is_hvac_series_candidate(canonical)
                or _looks_like_series_description_text(canonical)
            ):
                canonical = _choose_series_canonical_name(aliases) or meta.get("fallback_name", "")
            if not canonical or not _is_hvac_series_candidate(canonical):
                continue

            canonical_key = _series_merge_key(canonical) or _normalize_series_key(canonical) or group_key
            bucket = merged.setdefault(
                canonical_key,
                {"name": canonical, "evidence": [], "pages": [], "aliases": []},
            )
            if bucket["name"] != canonical:
                bucket["name"] = _choose_series_canonical_name([bucket["name"], canonical]) or bucket["name"]
            bucket["evidence"].extend(meta.get("evidence", []))
            bucket["pages"].extend(meta.get("pages", []))
            bucket["aliases"].extend(aliases)

        canonicalized: List[Dict] = []
        for bucket in merged.values():
            name = _clean_series_name_text(bucket.get("name", ""))
            if not _is_hvac_series_candidate(name):
                continue
            evidence = _dedup_list(bucket.get("evidence", []))
            pages = _dedup_list(bucket.get("pages", []))
            canonicalized.append(
                {
                    "name": name,
                    "evidence": evidence,
                    "pages": pages,
                }
            )
            aliases = _dedup_list(
                [
                    _clean_series_name_text(alias)
                    for alias in bucket.get("aliases", [])
                    if _clean_series_name_text(alias) and _is_hvac_series_candidate(alias)
                ]
            )
            if name not in aliases:
                aliases.insert(0, name)
            new_alias_map[name] = aliases

        if not canonicalized:
            return reviewed, alias_map or {}
        return canonicalized, new_alias_map or (alias_map or {})

    async def extract_series(
        self, brands: List[Dict], pages: Sequence[Tuple[str, Dict]], error_log: Path
    ) -> Dict[str, List[Dict]]:
        brand_to_series: Dict[str, List[Dict]] = {}
        if not brands:
            return brand_to_series

        show_progress = bool(getattr(self, "_show_progress", False))
        for brand_item in brands:
            brand_name = brand_item["name"]
            raw_series_list: List[Dict] = []
            async def _run_page(text: str, meta: Dict) -> List[Dict]:
                return await self._extract_series_single(brand_name, text, meta, error_log)

            page_tasks = [asyncio.create_task(_run_page(text, meta)) for text, meta in pages]
            if page_tasks:
                if show_progress:
                    results = []
                    with tqdm(total=len(page_tasks), desc="Stage B: series", unit="page") as pbar:
                        for coro in asyncio.as_completed(page_tasks):
                            res = await coro
                            results.append(res)
                            pbar.update(1)
                else:
                    results = await asyncio.gather(*page_tasks)
                for res in results:
                    raw_series_list.extend(res or [])

            raw_series_list = _merge_series_candidates(raw_series_list)

            cluster_size = int(self.config.get("series_cluster_size", 2))
            cluster_mode = str(self.config.get("series_cluster_mode", "fixed")).strip().lower()
            use_embed_cluster = bool(self.config.get("series_use_embed_cluster", False))
            if use_embed_cluster:
                cluster_mode = "embed"
            clusters: List[Tuple[str, List]] = []
            if cluster_size > 1:
                if cluster_mode == "embed":
                    clusters = self._cluster_pages_for_brands_embed(
                        pages,
                        model_name=self.config.get(
                            "series_cluster_embed_model",
                            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        ),
                        distance_threshold=float(self.config.get("series_cluster_embed_threshold", 0.25)),
                        device=self.config.get("series_cluster_embed_device", "cpu"),
                    )
                else:
                    clusters = self._cluster_pages_for_brands(pages, cluster_size)

            if clusters:
                async def _run_cluster(cluster_text: str, page_labels: List) -> List[Dict]:
                    return await self._extract_series_cluster(
                        brand_name,
                        cluster_text,
                        page_labels,
                        error_log,
                    )

                cluster_tasks = [
                    asyncio.create_task(_run_cluster(cluster_text, page_labels))
                    for cluster_text, page_labels in clusters
                ]
                if show_progress:
                    cluster_results = []
                    with tqdm(
                        total=len(clusters),
                        desc="Stage B: series clusters",
                        unit="cluster",
                    ) as pbar:
                        for coro in asyncio.as_completed(cluster_tasks):
                            res = await coro
                            cluster_results.append(res)
                            pbar.update(1)
                else:
                    cluster_results = await asyncio.gather(*cluster_tasks)
                for res in cluster_results:
                    raw_series_list.extend(res or [])
                raw_series_list = _merge_series_candidates(raw_series_list)

            # Stage-B candidate consolidation (similar to Stage-A: merge -> semantic merge -> LLM review)
            raw_series_list = _merge_series_semantic(
                raw_series_list,
                model_name=self.config.get(
                    "series_merge_embed_model",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ),
                distance_threshold=float(self.config.get("series_merge_threshold", 0.2)),
                device=self.config.get("series_merge_device", "cpu"),
            )

            series_list, alias_map = await self._review_series_for_brand(
                brand_name,
                raw_series_list,
                error_log,
            )
            self.series_alias_map[brand_name] = alias_map

            # Stage-B writes series-level features/components before Stage C/D.
            feature_page_numbers: List = []
            for series_item in series_list:
                series_name = str(series_item.get("name") or "").strip()
                if not series_name:
                    continue
                feature_page_numbers.extend(series_item.get("pages", []))
                aliases = alias_map.get(series_name, [])
                feature_page_numbers.extend(
                    _collect_series_occurrence_pages(_dedup_list([series_name] + aliases), pages)
                )
            feature_page_numbers = _dedup_list(
                [p for p in feature_page_numbers if p is not None and str(p) != ""]
            )
            feature_follow = int(self.config.get("series_context_follow_pages", 1))
            if feature_page_numbers:
                relevant_pages = _select_pages_by_numbers_with_following(
                    pages,
                    feature_page_numbers,
                    follow_after=feature_follow,
                )
                if not relevant_pages:
                    relevant_pages = _select_pages_by_numbers_with_following(
                        pages,
                        feature_page_numbers,
                        follow_after=0,
                    )
            else:
                relevant_pages = list(pages)

            await self._extract_series_features_for_brand(
                brand_name,
                [str(item.get("name") or "").strip() for item in series_list if str(item.get("name") or "").strip()],
                relevant_pages,
                error_log,
                source_category="series_feature_stage_b",
            )

            brand_to_series[brand_name] = series_list
        return _enrich_series_by_brand(brand_to_series)

    # -------------------------- Stage C: models ---------------------------- #
    async def _extract_models_for_pair(
        self,
        brand: str,
        series: str,
        text_block: str,
        context_pages: Optional[List],
        error_log: Path,
    ) -> List[Dict]:
        chunk_pbar = None
        try:
            chunks = _split_text(text_block, self.config.get("max_chars_per_call", 8000))
            model_items: List[Dict] = []
            chunk_sem = asyncio.Semaphore(_get_concurrency(self.config, "model_chunk_concurrency", 8))
            if getattr(self, "_show_progress", False):
                pair_label = f"{(brand or 'unknown')[:10]}/{(series or 'all')[:12]}"
                chunk_pbar = tqdm(
                    total=len(chunks),
                    desc=f"Stage C model chunks[{pair_label}]",
                    unit="chunk",
                    leave=False,
                )

            async def _run_chunk(chunk_idx: int, chunk_text: str) -> Tuple[str, List[Dict]]:
                try:
                    async with chunk_sem:
                        payload = await self._acall(
                            build_model_messages(
                                brand,
                                series,
                                chunk_text,
                                context_pages=context_pages,
                            ),
                            MODEL_SCHEMA,
                            "model_schema",
                            {
                                "brand": brand,
                                "series": series,
                                "chunk_index": chunk_idx + 1,
                                "chunk_total": len(chunks),
                            },
                        )
                    return chunk_text, payload.get("models", [])
                except Exception as exc:  # noqa: BLE001
                    _append_error(
                        error_log,
                        "model_extract_chunk",
                        {"brand": brand, "series": series, "chunk_index": chunk_idx + 1},
                        exc,
                    )
                    self.logger.error(
                        "Model extract chunk failed for brand=%s series=%s chunk=%s/%s: %s",
                        brand,
                        series,
                        chunk_idx + 1,
                        len(chunks),
                        exc,
                    )
                    return chunk_text, []

            tasks = [
                asyncio.create_task(_run_chunk(chunk_idx, chunk_text))
                for chunk_idx, chunk_text in enumerate(chunks)
            ]

            for task in asyncio.as_completed(tasks):
                chunk_text, chunk_models = await task
                if chunk_pbar:
                    chunk_pbar.set_postfix_str(
                        _chunk_preview(
                            chunk_text,
                            int(self.config.get("chunk_progress_preview_chars", 64)),
                        )
                    )
                    chunk_pbar.update(1)
                model_items.extend(chunk_models or [])

            merged_models = _merge_model_candidates(model_items)
            if merged_models:
                reviewed = await self._review_models_llm(brand, series, merged_models, error_log)
                if reviewed:
                    merged_models = reviewed
            return merged_models
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "model_extract", {"brand": brand, "series": series}, exc)
            self.logger.error("Model extract failed for brand=%s series=%s: %s", brand, series, exc)
            return []
        finally:
            if chunk_pbar:
                chunk_pbar.close()

    async def _review_products_llm(
        self,
        brand: str,
        series: str,
        products: List[Dict],
        error_log: Path,
    ) -> List[Dict]:
        if not products:
            return []
        max_items = int(self.config.get("product_review_max_items", 50))
        payload = products[:max_items]
        untouched_tail = products[max_items:]
        try:
            result = await self._acall(
                build_product_review_messages(brand, series, payload),
                PRODUCT_REVIEW_SCHEMA,
                "product_review_schema",
                {"brand": brand, "series": series, "count": len(payload)},
            )
            verdicts = {item.get("product_model", "").strip(): item for item in result.get("products", [])}
            if not verdicts:
                return products
            reviewed: List[Dict] = []
            for prod in payload:
                model_key = (prod.get("product_model") or "").strip()
                decision = verdicts.get(model_key)
                if decision is None:
                    reviewed.append(prod)
                    continue

                role = (decision.get("role") or "").strip().lower()
                if role not in {"product", "series_feature", "accessory"}:
                    if decision.get("series_feature"):
                        role = "series_feature"
                    elif decision.get("is_accessory"):
                        role = "accessory"
                    else:
                        role = "product"

                if role == "series_feature":
                    self._store_series_feature(brand, series, prod, decision)
                    continue
                if role == "accessory":
                    self._store_series_components_from_product(brand, series, prod)
                    continue
                if decision.get("keep") and role == "product" and not decision.get("is_accessory"):
                    reviewed.append(prod)
            reviewed.extend(untouched_tail)
            return reviewed
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "product_review", {"brand": brand, "series": series}, exc)
            self.logger.warning("Product LLM review failed for %s/%s: %s", brand, series, exc)
            return products

    def _store_series_feature(
        self,
        brand: str,
        series: str,
        prod: Dict,
        decision: Optional[Dict] = None,
    ) -> None:
        """Persist series-level feature/benefit snippets for later reuse."""
        series_key = series or ""
        entry = {
            "title": prod.get("product_model", ""),
            "fact_text": prod.get("fact_text", []),
            "features": prod.get("features", []),
            "key_components": prod.get("key_components", []),
            "performance_specs": prod.get("performance_specs", []),
            "evidence": _dedup_list(prod.get("evidence", [])),
            "source_category": prod.get("category", ""),
        }
        if decision is not None:
            entry["series_feature_flag"] = bool(decision.get("series_feature", False))
        self._store_series_feature_entry(brand, series_key, entry)
        self._store_series_components(brand, series_key, entry.get("key_components") or [])

    def _store_series_components(self, brand: str, series: str, components: List[str]) -> None:
        if not brand:
            return
        normalized = _dedup_list([str(c).strip() for c in (components or []) if str(c).strip()])
        if not normalized:
            return
        brand_bucket = self.series_component_map.setdefault(brand, {})
        series_bucket = brand_bucket.setdefault(series or "", [])
        brand_bucket[series or ""] = _dedup_list(series_bucket + normalized)

    def _store_series_components_from_product(self, brand: str, series: str, prod: Dict) -> None:
        components: List[str] = []
        for key in ["key_components", "features"]:
            components.extend([str(x).strip() for x in (prod.get(key) or []) if str(x).strip()])
        model = _canonicalize_model_name(prod.get("product_model") or "")
        if model:
            components.append(model)
        if not components:
            return
        self._store_series_components(brand, series, components)

    def _merge_series_components_into_products(self, products: List[Dict]) -> List[Dict]:
        if not products:
            return products
        merged_products: List[Dict] = []
        for prod in products:
            brand = str(prod.get("brand") or "").strip()
            series = str(prod.get("series") or "").strip()
            mapped = (
                self.series_component_map.get(brand, {}).get(series, [])
                if brand and series
                else []
            )
            if mapped:
                prod = dict(prod)
                prod["key_components"] = _dedup_list(
                    [str(x).strip() for x in (prod.get("key_components") or []) if str(x).strip()]
                    + [str(x).strip() for x in mapped if str(x).strip()]
                )
            merged_products.append(prod)
        return merged_products

    def _merge_series_features_into_products(self, products: List[Dict]) -> List[Dict]:
        if not products:
            return products
        merged_products: List[Dict] = []
        for prod in products:
            brand = str(prod.get("brand") or "").strip()
            series = str(prod.get("series") or "").strip()
            entries = (
                self.series_feature_map.get(brand, {}).get(series, [])
                if brand and series
                else []
            )
            if not entries:
                merged_products.append(prod)
                continue

            merged = dict(prod)
            merged_features = [str(x).strip() for x in (merged.get("features") or []) if str(x).strip()]
            merged_fact_text = [str(x).strip() for x in (merged.get("fact_text") or []) if str(x).strip()]
            merged_key_components = [
                str(x).strip() for x in (merged.get("key_components") or []) if str(x).strip()
            ]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                merged_features.extend(
                    [str(x).strip() for x in (entry.get("features") or []) if str(x).strip()]
                )
                merged_fact_text.extend(
                    [str(x).strip() for x in (entry.get("fact_text") or []) if str(x).strip()]
                )
                merged_key_components.extend(
                    [str(x).strip() for x in (entry.get("key_components") or []) if str(x).strip()]
                )

            merged["features"] = _dedup_list(merged_features)
            merged["fact_text"] = _dedup_list(merged_fact_text)
            merged["key_components"] = _dedup_list(merged_key_components)
            merged_products.append(merged)
        return merged_products

    def _bootstrap_series_components_from_series_features(self) -> None:
        """Hydrate component map from cached series_features entries."""
        self.series_component_map = {}
        for brand, series_map in (self.series_feature_map or {}).items():
            if not isinstance(series_map, dict):
                continue
            for series, entries in series_map.items():
                if not isinstance(entries, list):
                    continue
                components: List[str] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    components.extend(entry.get("key_components") or [])
                self._store_series_components(brand, series, components)

    def _bind_products_to_known_models(
        self,
        brand: str,
        series: str,
        products: List[Dict],
        known_models: Optional[List[str]],
    ) -> List[Dict]:
        """
        Force Stage-D product_model to come from Stage-C models of the same (brand, series) pair.
        """
        if not products:
            return []

        known = _dedup_list(
            [
                _canonicalize_model_name(m)
                for m in (known_models or [])
                if _canonicalize_model_name(m)
            ]
        )
        if not known:
            return []

        norm_to_known: Dict[str, str] = {}
        for name in known:
            key = _normalize_model_key(name)
            if key and key not in norm_to_known:
                norm_to_known[key] = name
        if not norm_to_known:
            return []

        sorted_norm_keys = sorted(norm_to_known.keys(), key=len, reverse=True)
        multi_cap = max(1, int(self.config.get("product_model_multi_match_cap", 12)))

        def _match(text: str) -> List[str]:
            normalized = _normalize_model_key(text or "")
            if not normalized:
                return []
            if normalized in norm_to_known:
                return [norm_to_known[normalized]]
            hits = [norm_to_known[k] for k in sorted_norm_keys if k and k in normalized]
            return _dedup_list(hits)

        bound: List[Dict] = []
        dropped = 0
        expanded = 0
        for product in products:
            matches = _match(product.get("product_model", ""))

            if not matches:
                dropped += 1
                continue

            if len(matches) > multi_cap:
                matches = matches[:multi_cap]

            if len(matches) > 1:
                expanded += len(matches) - 1
                for model_name in matches:
                    cloned = dict(product)
                    cloned["product_model"] = model_name
                    bound.append(cloned)
                continue

            canonical = dict(product)
            canonical["product_model"] = matches[0]
            bound.append(canonical)

        if dropped or expanded:
            self.logger.info(
                "Stage D model binding brand=%s series=%s: in=%s out=%s dropped_unknown=%s expanded=%s known=%s",
                brand,
                series,
                len(products),
                len(bound),
                dropped,
                expanded,
                len(known),
            )
        return bound

    async def _review_models_llm(
        self,
        brand: str,
        series: str,
        candidates: List[Dict],
        error_log: Path,
    ) -> List[Dict]:
        if not candidates:
            return []
        max_items = int(self.config.get("model_review_max_items", 80))
        payload = candidates[:max_items]
        untouched_tail = candidates[max_items:]
        try:
            result = await self._acall(
                build_model_review_messages(brand, series, payload),
                MODEL_REVIEW_SCHEMA,
                "model_review_schema",
                {"brand": brand, "series": series, "count": len(payload)},
            )
            verdicts = {item.get("name", "").strip(): item for item in result.get("items", [])}
            if not verdicts:
                return candidates
            reviewed: List[Dict] = []
            alias_map = getattr(self, "series_alias_map", {})
            allow_redirect = True
            drop_mismatch = False
            redirect_conf = float(self.config.get("model_redirect_min_conf", 0.5))
            for cand in payload:
                name = (cand.get("name") or "").strip()
                decision = verdicts.get(name)
                if decision is None:
                    # Keep unmatched candidate to avoid accidental over-prune.
                    reviewed.append(cand)
                    continue
                if not (decision.get("keep") and decision.get("kind") == "model"):
                    continue

                series_guess = (decision.get("series_guess") or "").strip()
                redirect_to = (decision.get("redirect_to") or "").strip()
                conf = float(decision.get("confidence", 1.0) or 0)
                target_series = None
                if redirect_to:
                    target_series = _match_series_name(redirect_to, brand, alias_map) or redirect_to
                elif series_guess:
                    target_series = _match_series_name(series_guess, brand, alias_map)
                    if not target_series and _normalize_series_key(series_guess) == _normalize_series_key(series):
                        target_series = series
                if target_series and _normalize_series_key(target_series) != _normalize_series_key(series):
                    if allow_redirect and conf >= redirect_conf:
                        self.model_redirects.append(
                            {
                                "brand": brand,
                                "source_series": series,
                                "target_series": target_series,
                                "model": cand,
                            }
                        )
                        continue
                    if drop_mismatch:
                        continue
                elif not target_series and (redirect_to or series_guess) and drop_mismatch:
                    # series hinted but cannot map confidently -> drop to avoid wrong assignment
                    continue
                reviewed.append(cand)

            reviewed.extend(untouched_tail)
            return reviewed
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "model_review", {"brand": brand, "series": series}, exc)
            self.logger.warning("Model LLM review failed for %s/%s: %s", brand, series, exc)
            return candidates

    async def extract_models(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List]]:
        self.model_redirects = []
        models_by_pair: Dict[str, List[Dict]] = {}
        pairs: List[Tuple[str, str, List]] = []
        for brand, series_list in brand_to_series.items():
            targets = series_list or [{"name": "", "pages": []}]
            for series_item in targets:
                pairs.append(
                    (
                        brand,
                        series_item.get("name", ""),
                        _dedup_list(series_item.get("pages", [])),
                    )
                )

        pbar = None
        if getattr(self, "_show_progress", False):
            pbar = tqdm(total=len(pairs), desc="Stage C: models", unit="pair")

        pair_limit = max(1, _get_concurrency(self.config, "model_pair_concurrency", 6))
        pair_sem = asyncio.Semaphore(pair_limit)
        self.logger.info("Stage C model pair concurrency=%s, pairs=%s", pair_limit, len(pairs))

        async def _run_pair(
            brand: str,
            series_name: str,
            series_pages: List,
        ) -> Tuple[str, List[Dict]]:
            try:
                # Stage C context is constrained to series evidence pages + following pages.
                series_aliases = (
                    self.series_alias_map.get(brand, {}).get(series_name, [])
                    if hasattr(self, "series_alias_map")
                    else []
                )
                series_terms = _dedup_list([series_name] + series_aliases)
                occurrence_pages = _collect_series_occurrence_pages(series_terms, pages)
                base_pages = _dedup_list((series_pages or []) + occurrence_pages)
                follow_after = int(self.config.get("series_model_follow_pages", 1))
                relevant_pages = _select_pages_by_numbers_with_following(
                    pages,
                    base_pages,
                    follow_after=follow_after,
                )
                if not relevant_pages and base_pages:
                    # Retry strictly within declared series pages when occurrence search failed.
                    relevant_pages = _select_pages_by_numbers_with_following(
                        pages,
                        base_pages,
                        follow_after=0,
                    )
                if not relevant_pages:
                    self.logger.warning(
                        "Stage C: no context pages for brand=%s series=%s; skip pair to avoid noise",
                        brand,
                        series_name,
                    )
                    return _pair_key(brand, series_name), []

                context_page_refs = _dedup_list(
                    [
                        meta.get("page")
                        for _, meta in relevant_pages
                        if meta.get("page") is not None and str(meta.get("page")) != ""
                    ]
                )
                # Stage C also captures series-level features/components from the same strict context.
                if series_name:
                    await self._extract_series_features_for_brand(
                        brand,
                        [series_name],
                        relevant_pages,
                        error_log,
                        source_category="series_feature_stage_c",
                    )
                text_block = _combine_pages(relevant_pages, self.config)
                pair_models = await self._extract_models_for_pair(
                    brand,
                    series_name,
                    text_block,
                    context_page_refs,
                    error_log,
                )
                return _pair_key(brand, series_name), pair_models
            except Exception as exc:  # noqa: BLE001
                _append_error(
                    error_log,
                    "model_pair_extract",
                    {"brand": brand, "series": series_name, "series_pages": series_pages},
                    exc,
                )
                self.logger.error(
                    "Model pair extract failed for brand=%s series=%s: %s",
                    brand,
                    series_name,
                    exc,
                )
                return _pair_key(brand, series_name), []

        async def _run_pair_bounded(
            brand: str,
            series_name: str,
            series_pages: List,
        ) -> Tuple[str, List[Dict]]:
            async with pair_sem:
                return await _run_pair(brand, series_name, series_pages)

        tasks = [
            asyncio.create_task(_run_pair_bounded(brand, series_name, series_pages))
            for brand, series_name, series_pages in pairs
        ]
        for task in asyncio.as_completed(tasks):
            pair_key, pair_models = await task
            existing = models_by_pair.get(pair_key, [])
            merged = _merge_model_candidates(existing + pair_models)
            models_by_pair[pair_key] = merged
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

        # Redirect models to other series based on LLM series_guess signals.
        if self.model_redirects:
            for item in self.model_redirects:
                dest_key = _pair_key(item["brand"], item["target_series"])
                existing = models_by_pair.get(dest_key, [])
                existing.append(item["model"])
                models_by_pair[dest_key] = _merge_model_candidates(existing)
            self.logger.info("Model redirects applied: %s", len(self.model_redirects))
            self.model_redirects = []

        all_model_names: List[str] = []
        for model_list in models_by_pair.values():
            all_model_names.extend([(m.get("name") or "").strip() for m in model_list if m.get("name")])
        model_page_stats = _collect_model_occurrence_pages(all_model_names, pages)

        # Cross-series conflict resolution: same model under multiple series -> keep one owner.
        self.model_conflicts = {}
        resolved_map, conflicts = _resolve_models_across_series(
            models_by_pair,
            brand_to_series,
            model_page_stats,
        )
        if conflicts:
            self.logger.info(
                "Stage C model conflict resolved: conflicts=%s",
                len(conflicts),
            )
        models_by_pair = resolved_map
        self.model_conflicts = conflicts

        # write occurrence pages back into pair models for easier debugging/tracing
        for pair_key, model_list in models_by_pair.items():
            for model_item in model_list:
                name = (model_item.get("name") or "").strip()
                pages_hit = model_page_stats.get(name, [])
                if pages_hit:
                    model_item["pages"] = pages_hit
                else:
                    model_item["pages"] = _dedup_list(model_item.get("pages", []))

        models_by_pair = _enrich_models_by_pair(models_by_pair)
        return models_by_pair, model_page_stats

    # -------------------------- Stage D: products -------------------------- #
    async def _extract_products_for_pair(
        self,
        brand: str,
        series: str,
        text_block: str,
        page_refs: List,
        known_models: Optional[List[str]],
        error_log: Path,
        target_model: Optional[str] = None,
    ) -> List[Dict]:
        chunk_pbar = None
        try:
            chunks = _split_text(text_block, self.config.get("max_chars_per_call", 8000))
            products = []
            chunk_sem = asyncio.Semaphore(_get_concurrency(self.config, "product_chunk_concurrency", 8))
            if getattr(self, "_show_progress", False):
                pair_label = f"{(brand or 'unknown')[:10]}/{(series or 'all')[:12]}"
                if target_model:
                    pair_label = f"{pair_label}/{target_model[:14]}"
                chunk_pbar = tqdm(
                    total=len(chunks),
                    desc=f"Stage D chunks[{pair_label}]",
                    unit="chunk",
                    leave=False,
                )

            async def _run_chunk(chunk_idx: int, chunk_text: str) -> Tuple[str, List[Dict]]:
                try:
                    async with chunk_sem:
                        payload = await self._acall(
                            build_product_messages(
                                brand,
                                series,
                                chunk_text,
                                known_models=known_models,
                                target_model=target_model,
                            ),
                            PRODUCT_SCHEMA,
                            "product_schema",
                            {
                                "brand": brand,
                                "series": series,
                                "chunk_index": chunk_idx + 1,
                                "chunk_total": len(chunks),
                            },
                        )
                    return chunk_text, payload.get("results", [])
                except Exception as exc:  # noqa: BLE001
                    _append_error(
                        error_log,
                        "product_extract_chunk",
                        {"brand": brand, "series": series, "chunk_index": chunk_idx + 1},
                        exc,
                    )
                    self.logger.error(
                        "Product extract chunk failed for brand=%s series=%s chunk=%s/%s: %s",
                        brand,
                        series,
                        chunk_idx + 1,
                        len(chunks),
                        exc,
                    )
                    return chunk_text, []

            tasks = [
                asyncio.create_task(_run_chunk(chunk_idx, chunk_text))
                for chunk_idx, chunk_text in enumerate(chunks)
            ]

            for task in asyncio.as_completed(tasks):
                chunk_text, chunk_results = await task
                if chunk_pbar:
                    chunk_pbar.set_postfix_str(
                        _chunk_preview(
                            chunk_text,
                            int(self.config.get("chunk_progress_preview_chars", 64)),
                        )
                    )
                    chunk_pbar.update(1)
                for item in chunk_results:
                    merged = self._ensure_product_defaults(item, brand, series, page_refs)
                    # Stage-D hierarchy identity must inherit from upstream stages.
                    if brand:
                        merged["brand"] = str(brand).strip()
                    if series:
                        merged["series"] = str(series).strip()
                    if target_model:
                        merged["product_model"] = _canonicalize_model_name(target_model)
                    products.append(merged)

            # Merge across chunks for the same target model to form one coherent product output.
            if target_model:
                products = self._bind_products_to_known_models(
                    brand,
                    series,
                    products,
                    [target_model],
                )
            products = _deduplicate_products_by_model(products, self.config)
            return products
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "product_extract", {"brand": brand, "series": series}, exc)
            self.logger.error(
                "Product extract failed for brand=%s series=%s: %s", brand, series, exc
            )
            return []
        finally:
            if chunk_pbar:
                chunk_pbar.close()

    async def extract_products(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> List[Dict]:
        products: List[Dict] = []
        pairs: List[Tuple[str, str, List]] = []
        for brand, series_list in brand_to_series.items():
            targets = series_list or [{"name": "", "pages": []}]
            for series_item in targets:
                pairs.append(
                    (
                        brand,
                        series_item.get("name", ""),
                        _dedup_list(series_item.get("pages", [])),
                    )
                )

        pbar = None
        if getattr(self, "_show_progress", False):
            pbar = tqdm(total=len(pairs), desc="Stage D: products", unit="pair")

        pair_limit = max(1, _get_concurrency(self.config, "product_pair_concurrency", 6))
        pair_sem = asyncio.Semaphore(pair_limit)
        self.logger.info("Stage D pair concurrency=%s, pairs=%s", pair_limit, len(pairs))

        async def _run_pair(
            pair_index: int,
            brand: str,
            series_name: str,
            series_pages: List,
        ) -> Tuple[int, List[Dict]]:
            try:
                # Product stage context: model-hit pages + next N pages.
                pair_models = self.models_by_pair.get(_pair_key(brand, series_name), [])
                if not pair_models:
                    return pair_index, []
                known_models = _dedup_list(
                    [
                        _canonicalize_model_name(model_item.get("name") or "")
                        for model_item in pair_models
                        if _canonicalize_model_name(model_item.get("name") or "")
                    ]
                )
                if not known_models:
                    return pair_index, []
                follow_after = int(self.config.get("product_model_follow_pages", 0))
                model_to_pages: Dict[str, List] = {}
                for model_item in pair_models:
                    model_name = _canonicalize_model_name(model_item.get("name") or "")
                    if not model_name:
                        continue
                    pages_hit = model_to_pages.setdefault(model_name, [])
                    pages_hit.extend(self.model_page_stats.get(model_name, []))
                    pages_hit.extend(model_item.get("pages", []))
                for model_name in list(model_to_pages.keys()):
                    model_to_pages[model_name] = _dedup_list(
                        [p for p in model_to_pages[model_name] if p is not None and str(p) != ""]
                    )

                model_sem = asyncio.Semaphore(
                    max(1, _get_concurrency(self.config, "product_model_concurrency", 6))
                )

                async def _run_model(model_name: str) -> List[Dict]:
                    async with model_sem:
                        model_pages = model_to_pages.get(model_name, [])
                        if not model_pages:
                            return []
                        model_relevant_pages = _select_pages_by_numbers_with_following(
                            pages,
                            model_pages,
                            follow_after=follow_after,
                        )
                        if not model_relevant_pages:
                            return []
                        text_block = _combine_pages(
                            model_relevant_pages,
                            self.config,
                            target_model=model_name,
                        )
                        page_refs = [meta.get("page") for _, meta in model_relevant_pages]
                        extracted = await self._extract_products_for_pair(
                            brand,
                            series_name,
                            text_block,
                            page_refs,
                            [model_name],
                            error_log,
                            target_model=model_name,
                        )
                        processed_model = list(extracted)
                        if processed_model:
                            processed_model = await self._review_products_llm(
                                brand,
                                series_name,
                                processed_model,
                                error_log,
                            )
                        processed_model = self._bind_products_to_known_models(
                            brand,
                            series_name,
                            processed_model,
                            [model_name],
                        )
                        return _deduplicate_products_by_model(processed_model, self.config)

                model_tasks = [asyncio.create_task(_run_model(model_name)) for model_name in known_models]
                model_products: List[Dict] = []
                for mt in asyncio.as_completed(model_tasks):
                    model_products.extend(await mt)
                model_products = _deduplicate_products_by_model(model_products, self.config)
                model_products = self._bind_products_to_known_models(
                    brand,
                    series_name,
                    model_products,
                    known_models,
                )
                return pair_index, model_products
            except Exception as exc:  # noqa: BLE001
                _append_error(
                    error_log,
                    "product_pair_extract",
                    {"brand": brand, "series": series_name, "series_pages": series_pages},
                    exc,
                )
                self.logger.error(
                    "Product pair extract failed for brand=%s series=%s: %s",
                    brand,
                    series_name,
                    exc,
                )
                return pair_index, []

        async def _run_pair_bounded(
            pair_index: int,
            brand: str,
            series_name: str,
            series_pages: List,
        ) -> Tuple[int, List[Dict]]:
            async with pair_sem:
                return await _run_pair(pair_index, brand, series_name, series_pages)

        tasks = [
            asyncio.create_task(_run_pair_bounded(pair_index, brand, series_name, series_pages))
            for pair_index, (brand, series_name, series_pages) in enumerate(pairs)
        ]

        result_map: Dict[int, List[Dict]] = {}
        for task in asyncio.as_completed(tasks):
            pair_index, extracted = await task
            result_map[pair_index] = extracted
            if pbar:
                pbar.update(1)

        for pair_index in range(len(pairs)):
            products.extend(result_map.get(pair_index, []))
        products = self._merge_series_features_into_products(products)
        products = self._merge_series_components_into_products(products)
        products = _deduplicate_products_by_model(products, self.config)

        if pbar:
            pbar.close()
        return products

    # -------------------------- Public orchestration ---------------------- #
    async def extract_document(
        self, pages: Sequence[Tuple[str, Dict]], error_log: Path, show_progress: bool = False
    ) -> List[Dict]:
        # flag for inner methods
        self._show_progress = show_progress
        brands = await self.run_brand_stage(pages, error_log)
        if not brands:
            return []

        brand_to_series = await self.run_series_stage(brands, pages, error_log)
        await self.run_model_stage(brand_to_series, pages, error_log)
        products = await self.run_product_stage(brand_to_series, pages, error_log)
        return products

    # -------------------------- Post-processing helpers ------------------- #
    def _ensure_product_defaults(
        self, product: Dict, brand: str, series: str, page_refs: List
    ) -> Dict:
        merged = _default_product_template()
        merged.update(product or {})

        if brand and not merged.get("brand"):
            merged["brand"] = brand
        if series and not merged.get("series"):
            merged["series"] = series

        # Ensure list fields are lists
        for field in ["features", "key_components", "fact_text", "evidence"]:
            if merged.get(field) is None:
                merged[field] = []
        if merged.get("performance_specs") is None:
            merged["performance_specs"] = []

        if not merged.get("evidence"):
            merged["evidence"] = [f"pages: {page_refs}"]
        merged["evidence"] = _dedup_list(merged.get("evidence", []))

        # HVAC default category fallback
        if not merged.get("category"):
            merged["category"] = "暖通空调"

        # Strip whitespace
        for field in [
            "brand",
            "category",
            "series",
            "product_model",
            "manufacturer",
            "refrigerant",
            "energy_efficiency_grade",
        ]:
            if merged.get(field) is None:
                merged[field] = ""
            merged[field] = str(merged[field]).strip()

        # Canonicalize model format (e.g., full-width chars, spaces, combo '+').
        merged["product_model"] = _canonicalize_model_name(merged.get("product_model", ""))
        merged["performance_specs"] = _normalize_performance_specs(
            merged.get("performance_specs") or []
        )

        # Dedup any repeated page_refs in evidence string hints
        merged["evidence"] = _dedup_list(merged.get("evidence", []))

        return merged


# --------------------------------------------------------------------------- #
# Module-level runner with stage checkpoints
# --------------------------------------------------------------------------- #


def _select_pages_by_keywords(
    pages: Sequence[Tuple[str, Dict]],
    keywords: Sequence[str],
    top_k: int = 0,
    min_hits: int = 1,
    match_mode: str = "any",
) -> List[Tuple[str, Dict]]:
    """Keyword retrieval with hit-count ranking."""
    lowered = _dedup_list([str(k).strip().lower() for k in keywords if str(k).strip()])
    if not lowered:
        return list(pages)

    mode = str(match_mode or "any").strip().lower()
    required_hits = max(1, int(min_hits))
    ranked: List[Tuple[int, int, Tuple[str, Dict]]] = []

    for idx, page in enumerate(pages):
        text = str(page[0] or "").lower()
        hit_count = sum(1 for kw in lowered if kw in text)
        if mode == "all":
            is_match = hit_count == len(lowered)
        else:
            is_match = hit_count >= required_hits
        if is_match:
            ranked.append((hit_count, idx, page))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [page for _, _, page in ranked]
    if int(top_k) > 0:
        selected = selected[: int(top_k)]
    return selected


def _parse_bbox_like(value) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        nums = value
    else:
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    if len(nums) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _row_inside_table_bbox(
    row_bbox: Optional[Tuple[float, float, float, float]],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_cover_ratio: float = 0.85,
) -> bool:
    if not row_bbox or not table_bboxes:
        return False
    rx1, ry1, rx2, ry2 = row_bbox
    row_area = max((rx2 - rx1) * (ry2 - ry1), 1e-6)
    for tx1, ty1, tx2, ty2 in table_bboxes:
        ix1 = max(rx1, tx1)
        iy1 = max(ry1, ty1)
        ix2 = min(rx2, tx2)
        iy2 = min(ry2, ty2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        overlap = (ix2 - ix1) * (iy2 - iy1)
        if overlap / row_area >= min_cover_ratio:
            return True
    return False


def _normalize_rows_match_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _filter_rows_for_rows_json(rows_list: List[Dict], config: Optional[Dict]) -> List[Dict]:
    if not rows_list:
        return []
    return list(rows_list)


def _compact_row_for_rows_json(row: Dict, config: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(row, dict):
        return None

    clip_chars = int((config or {}).get("rows_json_content_clip", 2400))
    row_type = str(row.get("type", "")).lower()

    compact: Dict = {"type": row.get("type", "")}
    if row.get("bbox"):
        compact["bbox"] = row.get("bbox")
    if row.get("page") is not None and str(row.get("page")) != "":
        compact["page"] = row.get("page")

    content = str(row.get("content", "")).strip()
    table_data = str(row.get("table_data", "")).strip()

    if row_type == "table":
        if table_data:
            compact["table_data"] = table_data[:clip_chars] if clip_chars > 0 else table_data
        elif content:
            compact["content"] = content[:clip_chars] if clip_chars > 0 else content
    else:
        if content:
            compact["content"] = content[:clip_chars] if clip_chars > 0 else content
        if table_data and row_type not in {"title", "text", "list"}:
            compact["table_data"] = table_data[:clip_chars] if clip_chars > 0 else table_data

    if "content" not in compact and "table_data" not in compact:
        return None
    return compact


def _render_rows_blocks(rows, config: Optional[Dict]) -> List[str]:
    try:
        rows_list = list(rows) if isinstance(rows, (list, tuple)) else [rows]
    except Exception:
        rows_list = [rows]
    rows_list = _filter_rows_for_rows_json(rows_list, config)
    rows_per_block = max(1, int(config.get("table_rows_per_block", 10))) if config else 10
    max_cell_len = int(config.get("table_row_cell_clip", 0)) if config else 0
    blocks: List[str] = []
    for i in range(0, len(rows_list), rows_per_block):
        block_rows = rows_list[i : i + rows_per_block]
        rendered = []
        for r in block_rows:
            row_payload = _compact_row_for_rows_json(r, config)
            if row_payload is None:
                continue
            try:
                row_str = json.dumps(row_payload, ensure_ascii=False)
            except Exception:
                row_str = str(row_payload)
            if max_cell_len and len(row_str) > max_cell_len:
                row_str = row_str[:max_cell_len] + "..."
            rendered.append(row_str)
        if rendered:
            blocks.append("\n".join(rendered))
    return blocks


def _rows_to_prompt_text(rows_list: List[Dict]) -> str:
    prefix_map = {
        "title": "[title]",
        "text": "[text]",
        "list": "[text]",
        "table": "[table]",
    }
    parts: List[str] = []
    for row in rows_list or []:
        row_type = str(row.get("type", "")).lower()
        content = str(row.get("content", "")).strip()
        table_data = str(row.get("table_data", "")).strip()
        if row_type == "table" and table_data:
            table_lines = table_data.replace(" \\n ", "\n").replace("\\n", "\n")
            lines = [
                f"[table_row] {line.strip()}"
                for line in table_lines.splitlines()
                if line.strip()
            ]
            if lines:
                parts.append("\n".join(lines))
            continue
        if content:
            parts.append(f"{prefix_map.get(row_type, '[block]')} {content}")
    return "\n\n".join(parts)


def _slice_table_data_for_target_model(
    table_data: str,
    target_model: str,
    context_lines: int = 0,
) -> str:
    lines = [line.strip() for line in str(table_data or "").replace(" \\n ", "\n").replace("\\n", "\n").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    target_key = _normalize_model_key(target_model)
    if not target_key:
        return "\n".join(lines)

    normalized = [_normalize_model_key(line) for line in lines]
    hit_indices = [idx for idx, norm in enumerate(normalized) if target_key in norm]
    if not hit_indices:
        return "\n".join(lines)

    keep = set()
    # Keep top header rows and target rows with neighbors.
    for idx in range(min(2, len(lines))):
        keep.add(idx)
    for idx in hit_indices:
        left = max(0, idx - max(0, int(context_lines)))
        right = min(len(lines), idx + max(0, int(context_lines)) + 1)
        for j in range(left, right):
            keep.add(j)

    return "\n".join(lines[idx] for idx in sorted(keep))


def _filter_rows_for_target_model(
    rows_list: List[Dict],
    target_model: Optional[str],
    context_lines: int = 0,
) -> Tuple[List[Dict], bool]:
    if not rows_list or not target_model:
        return rows_list, False
    target_key = _normalize_model_key(target_model)
    if not target_key:
        return rows_list, False

    filtered: List[Dict] = []
    hit_any = False
    for row in rows_list:
        row_type = str(row.get("type", "")).lower()
        if row_type != "table":
            filtered.append(row)
            continue

        table_data = str(row.get("table_data", "")).strip()
        if not table_data:
            continue
        if target_key not in _normalize_model_key(table_data):
            continue

        sliced_table_data = _slice_table_data_for_target_model(
            table_data,
            target_model,
            context_lines=context_lines,
        )
        cloned = dict(row)
        cloned["table_data"] = sliced_table_data
        # Avoid re-injecting full-table flattened content.
        cloned["content"] = ""
        filtered.append(cloned)
        hit_any = True

    return (filtered, True) if hit_any else (rows_list, False)


def _combine_pages(
    pages: Sequence[Tuple[str, Dict]],
    config: Optional[Dict] = None,
    target_model: Optional[str] = None,
) -> str:
    chunks = []
    for text, meta in pages:
        page = meta.get("page", "")
        page_text = text
        chunk = [f"<<PAGE {page}>>"]
        rows = meta.get("rows")
        if rows:
            try:
                rows_list = list(rows) if isinstance(rows, (list, tuple)) else [rows]
            except Exception:
                rows_list = [rows]
            rows_payload = _filter_rows_for_rows_json(rows_list, config)
            target_hit = False
            if target_model:
                rows_payload, target_hit = _filter_rows_for_target_model(
                    rows_payload,
                    target_model=target_model,
                    context_lines=int((config or {}).get("target_model_table_context_lines", 0)),
                )
                if target_hit:
                    filtered_text = _rows_to_prompt_text(rows_payload)
                    if filtered_text:
                        page_text = filtered_text
            chunk.append(page_text)
            row_blocks: List[str] = _render_rows_blocks(rows_payload, config)
            for block in row_blocks:
                chunk.append("[rows_json]")
                chunk.append(block)
        else:
            chunk.append(page_text)
        chunks.append("\n".join(chunk))
    return "\n\n".join(chunks)


def _select_pages_by_numbers_with_following(
    pages: Sequence[Tuple[str, Dict]],
    base_pages: Sequence,
    follow_after: int = 2,
) -> List[Tuple[str, Dict]]:
    """Select base pages and the next N pages by document order."""
    if not pages:
        return []
    base_set = {str(p) for p in base_pages if p is not None and str(p) != ""}
    if not base_set:
        return []
    selected_indices = set()
    for idx, (_, meta) in enumerate(pages):
        page_id = str(meta.get("page", ""))
        if page_id in base_set:
            base_doc_group = str(meta.get("doc_group", ""))
            base_sample = str(meta.get("sample", ""))
            base_file = str(meta.get("file", ""))
            for offset in range(0, max(0, int(follow_after)) + 1):
                next_idx = idx + offset
                if next_idx < len(pages):
                    next_meta = pages[next_idx][1]
                    # Keep following-page window within the same source document.
                    next_doc_group = str(next_meta.get("doc_group", ""))
                    if base_doc_group and next_doc_group:
                        if next_doc_group != base_doc_group:
                            break
                    else:
                        if (
                            str(next_meta.get("sample", "")) != base_sample
                            or str(next_meta.get("file", "")) != base_file
                        ):
                            break
                    selected_indices.add(next_idx)
    return [pages[idx] for idx in sorted(selected_indices)]


def _collect_model_occurrence_pages(
    model_names: Sequence[str],
    pages: Sequence[Tuple[str, Dict]],
) -> Dict[str, List]:
    """Find all pages where each model appears in raw page text."""
    stats: Dict[str, List] = {}
    if not model_names or not pages:
        return stats
    normalized_pages = [(_normalize_model_key(text), meta.get("page")) for text, meta in pages]
    for model_name in _dedup_list([m for m in model_names if m]):
        key = _normalize_model_key(model_name)
        if not key:
            continue
        hit_pages = []
        for normalized_text, page_id in normalized_pages:
            if not normalized_text:
                continue
            if key in normalized_text:
                hit_pages.append(page_id)
        stats[model_name] = _dedup_list(hit_pages)
    return stats


def _collect_series_occurrence_pages(
    series_names: Sequence[str],
    pages: Sequence[Tuple[str, Dict]],
) -> List:
    """Find pages where any series/alias name appears in raw page text."""
    if not series_names or not pages:
        return []

    keys = []
    for name in _dedup_list([s for s in series_names if s]):
        key = _normalize_series_key(name)
        if len(key) >= 2:
            keys.append(key)
    if not keys:
        return []

    hit_pages: List = []
    for text, meta in pages:
        normalized_text = _normalize_series_key(text)
        if any(key in normalized_text for key in keys):
            hit_pages.append(meta.get("page"))
    return _dedup_list(hit_pages)


def _count_series_hits_in_evidence(series_name: str, evidence: Sequence[str]) -> int:
    key = _normalize_series_key(series_name)
    if len(key) < 2:
        return 0
    hits = 0
    for snippet in evidence or []:
        if key and key in _normalize_series_key(str(snippet or "")):
            hits += 1
    return hits


def _resolve_models_across_series(
    models_by_pair: Dict[str, List[Dict]],
    brand_to_series: Dict[str, List[Dict]],
    model_page_stats: Dict[str, List],
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """
    Resolve duplicate model names extracted under multiple series of the same brand.
    Keep one strongest (brand, series) owner per model to prevent downstream duplicates.
    """
    if not models_by_pair:
        return {}, {}

    pair_series_pages: Dict[str, set] = {}
    for brand, series_list in brand_to_series.items():
        targets = series_list or [{"name": "", "pages": []}]
        for series_item in targets:
            key = _pair_key(brand, series_item.get("name", ""))
            pair_series_pages[key] = {
                str(p)
                for p in _dedup_list(series_item.get("pages", []))
                if p is not None and str(p) != ""
            }

    candidate_buckets: Dict[Tuple[str, str], List[Dict]] = {}
    for pair_key, model_list in models_by_pair.items():
        brand, series_name = _split_pair_key(pair_key)
        for index, model_item in enumerate(model_list):
            model_name = _canonicalize_model_name(model_item.get("name") or "")
            model_key = _normalize_model_key(model_name)
            if not model_key:
                continue
            model_item["name"] = model_name
            bucket_key = (brand, model_key)
            candidate_buckets.setdefault(bucket_key, []).append(
                {
                    "pair_key": pair_key,
                    "series": series_name,
                    "index": index,
                    "name": model_name,
                    "item": model_item,
                }
            )

    keep_mask: Dict[str, List[bool]] = {
        pair_key: [True] * len(model_list) for pair_key, model_list in models_by_pair.items()
    }
    conflicts: Dict[str, Dict] = {}

    for (brand, model_key), candidates in candidate_buckets.items():
        if len(candidates) <= 1:
            continue

        scored: List[Dict] = []
        for cand in candidates:
            model_name = cand["name"]
            pair_key = cand["pair_key"]
            series_name = cand["series"]
            item = cand["item"]

            local_pages = {
                str(p)
                for p in _dedup_list(item.get("pages", []))
                if p is not None and str(p) != ""
            }
            global_pages = {
                str(p)
                for p in _dedup_list(model_page_stats.get(model_name, []))
                if p is not None and str(p) != ""
            }
            series_pages = pair_series_pages.get(pair_key, set())
            overlap_global = len(global_pages & series_pages)
            overlap_local = len(local_pages & series_pages)
            ev_hits = _count_series_hits_in_evidence(series_name, item.get("evidence", []))
            evidence_count = len(item.get("evidence", []))
            score_tuple = (
                overlap_global,
                overlap_local,
                ev_hits,
                len(local_pages),
                evidence_count,
                len(series_pages),
            )
            scored.append(
                {
                    "candidate": cand,
                    "score": score_tuple,
                    "detail": {
                        "pair_key": pair_key,
                        "series": series_name,
                        "model": model_name,
                        "overlap_global": overlap_global,
                        "overlap_local": overlap_local,
                        "series_hits_in_evidence": ev_hits,
                        "local_page_count": len(local_pages),
                        "evidence_count": evidence_count,
                        "series_page_count": len(series_pages),
                    },
                }
            )

        # Highest score wins; tie-break by pair_key for deterministic behavior.
        scored_sorted = sorted(
            scored,
            key=lambda x: (x["score"], x["candidate"]["pair_key"]),
            reverse=True,
        )
        winner = scored_sorted[0]["candidate"]
        winner_pair_key = winner["pair_key"]
        winner_index = int(winner["index"])

        for entry in scored_sorted[1:]:
            cand = entry["candidate"]
            keep_mask[cand["pair_key"]][int(cand["index"])] = False

        conflict_id = f"{brand}::{model_key}"
        conflicts[conflict_id] = {
            "brand": brand,
            "model_key": model_key,
            "winner_pair_key": winner_pair_key,
            "winner_series": winner.get("series", ""),
            "winner_index": winner_index,
            "candidates": [entry["detail"] for entry in scored_sorted],
        }

    resolved: Dict[str, List[Dict]] = {}
    for pair_key, model_list in models_by_pair.items():
        filtered = [
            item
            for idx, item in enumerate(model_list)
            if keep_mask.get(pair_key, [True] * len(model_list))[idx]
        ]
        resolved[pair_key] = filtered

    return resolved, conflicts


def _canonicalize_models_by_pair(models_by_pair: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for pair_key, model_list in (models_by_pair or {}).items():
        items: List[Dict] = []
        for model_item in model_list or []:
            cloned = dict(model_item or {})
            cloned["name"] = _canonicalize_model_name(cloned.get("name") or "")
            items.append(cloned)
        out[pair_key] = _merge_model_candidates(items)
    return _enrich_models_by_pair(out)


def _enrich_series_by_brand(brand_to_series: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for brand, series_list in (brand_to_series or {}).items():
        enriched: List[Dict] = []
        for item in series_list or []:
            entry = dict(item or {})
            series_name = (entry.get("name") or "").strip()
            if not series_name:
                continue
            entry["name"] = series_name
            entry["brand"] = brand
            entry["pair_key"] = _pair_key(brand, series_name)
            entry["series_key"] = _normalize_series_key(series_name)
            entry["evidence"] = _dedup_list(entry.get("evidence", []))
            entry["pages"] = _dedup_list(entry.get("pages", []))
            enriched.append(entry)
        out[brand] = enriched
    return out


def _enrich_models_by_pair(models_by_pair: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for pair_key, model_list in (models_by_pair or {}).items():
        brand, series = _split_pair_key(pair_key)
        enriched: List[Dict] = []
        for item in model_list or []:
            entry = dict(item or {})
            model_name = _canonicalize_model_name(entry.get("name") or "")
            if not model_name:
                continue
            entry["name"] = model_name
            entry["brand"] = brand
            entry["series"] = series
            entry["pair_key"] = pair_key
            entry["model_key"] = _model_node_key(brand, series, model_name)
            entry["evidence"] = _dedup_list(entry.get("evidence", []))
            entry["pages"] = _dedup_list(entry.get("pages", []))
            enriched.append(entry)
        out[pair_key] = _merge_model_candidates(enriched)
    return out


def _build_brand_series_relations(
    brands: Sequence[Dict],
    brand_to_series: Dict[str, List[Dict]],
) -> List[Dict]:
    brand_pages: Dict[str, List] = {}
    for item in brands or []:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        brand_pages[name] = _dedup_list(item.get("pages", []))

    edges: List[Dict] = []
    for brand, series_list in (brand_to_series or {}).items():
        for series_item in series_list or []:
            series_name = (series_item.get("name") or "").strip()
            if not series_name:
                continue
            edges.append(
                {
                    "brand": brand,
                    "brand_key": _normalize_brand_key(brand),
                    "brand_pages": brand_pages.get(brand, []),
                    "series": series_name,
                    "series_key": _normalize_series_key(series_name),
                    "pair_key": _pair_key(brand, series_name),
                    "evidence": _dedup_list(series_item.get("evidence", [])),
                    "pages": _dedup_list(series_item.get("pages", [])),
                }
            )
    return edges


def _build_series_model_relations(models_by_pair: Dict[str, List[Dict]]) -> List[Dict]:
    edges: List[Dict] = []
    for pair_key, model_list in (models_by_pair or {}).items():
        brand, series = _split_pair_key(pair_key)
        for model_item in model_list or []:
            model_name = _canonicalize_model_name(model_item.get("name") or "")
            if not model_name:
                continue
            edges.append(
                {
                    "brand": brand,
                    "series": series,
                    "pair_key": pair_key,
                    "model_name": model_name,
                    "model_key": _model_node_key(brand, series, model_name),
                    "evidence": _dedup_list(model_item.get("evidence", [])),
                    "pages": _dedup_list(model_item.get("pages", [])),
                }
            )
    return edges


def _models_explicit_view(models_by_pair: Dict[str, List[Dict]]) -> List[Dict]:
    rows: List[Dict] = []
    for pair_key, model_list in (models_by_pair or {}).items():
        for model_item in model_list or []:
            rows.append(dict(model_item or {}, pair_key=pair_key))
    return rows


def _build_model_specs_relations(products: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for product in products or []:
        brand = (product.get("brand") or "").strip()
        series = (product.get("series") or "").strip()
        model_name = _canonicalize_model_name(product.get("product_model") or "")
        if not model_name:
            continue
        model_key = _model_node_key(brand, series, model_name)
        entry = grouped.get(model_key)
        if entry is None:
            entry = {
                "brand": brand,
                "series": series,
                "pair_key": _pair_key(brand, series),
                "model_name": model_name,
                "model_key": model_key,
                "manufacturer": str(product.get("manufacturer") or "").strip(),
                "refrigerant": str(product.get("refrigerant") or "").strip(),
                "energy_efficiency_grade": str(product.get("energy_efficiency_grade") or "").strip(),
                "performance_specs": [],
                "evidence": [],
                "fact_text": [],
            }
            grouped[model_key] = entry

        entry["evidence"] = _dedup_list(entry.get("evidence", []) + (product.get("evidence") or []))
        entry["fact_text"] = _dedup_list(entry.get("fact_text", []) + (product.get("fact_text") or []))
        for spec in product.get("performance_specs") or []:
            key = (
                (spec.get("name") or "").strip(),
                (spec.get("value") or "").strip(),
                (spec.get("unit") or "").strip(),
            )
            if key == ("", "", ""):
                continue
            existing = {
                (
                    (s.get("name") or "").strip(),
                    (s.get("value") or "").strip(),
                    (s.get("unit") or "").strip(),
                )
                for s in entry["performance_specs"]
            }
            if key not in existing:
                entry["performance_specs"].append(spec)
    return list(grouped.values())


_SPEC_EN_NAME_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bstandard\s*static\s*pressure\b", flags=re.IGNORECASE), "标准静压"),
    (re.compile(r"\bstatic\s*pressure\s*range\b", flags=re.IGNORECASE), "静压范围"),
    (re.compile(r"\bstatic\s*pressure\b", flags=re.IGNORECASE), "静压"),
    (
        re.compile(
            r"\b(?:air\s*flow|airflow|circulating\s*air\s*volume|cycle\s*air\s*volume)\b",
            flags=re.IGNORECASE,
        ),
        "循环风量",
    ),
    (re.compile(r"\bcooling\s*capacity\b", flags=re.IGNORECASE), "制冷量"),
    (re.compile(r"\bheating\s*capacity\b", flags=re.IGNORECASE), "制热量"),
    (
        re.compile(r"\bcooling\s*(?:input\s*power|power\s*consumption|power)\b", flags=re.IGNORECASE),
        "制冷功率",
    ),
    (
        re.compile(r"\bheating\s*(?:input\s*power|power\s*consumption|power)\b", flags=re.IGNORECASE),
        "制热功率",
    ),
    (re.compile(r"\b(?:input\s*power|power\s*consumption)\b", flags=re.IGNORECASE), "耗电量"),
    (re.compile(r"\b(?:noise|sound\s*pressure)\b", flags=re.IGNORECASE), "噪音"),
    (re.compile(r"\b(?:power\s*supply|voltage)\b", flags=re.IGNORECASE), "电源"),
    (re.compile(r"\bfrequency\b", flags=re.IGNORECASE), "频率"),
    (re.compile(r"\bdrain\s*pipe\b", flags=re.IGNORECASE), "排水管"),
    (re.compile(r"\bdrainage\s*pipe\b", flags=re.IGNORECASE), "排水管"),
    (re.compile(r"\bnet\s*weight\b", flags=re.IGNORECASE), "净重"),
    (re.compile(r"\bgross\s*weight\b", flags=re.IGNORECASE), "毛重"),
    (re.compile(r"\boperating\s*weight\b", flags=re.IGNORECASE), "运行重量"),
    (re.compile(r"\bweight\b", flags=re.IGNORECASE), "重量"),
    (re.compile(r"\bdimension(?:s)?\b", flags=re.IGNORECASE), "外形尺寸"),
    (re.compile(r"\bconnection\s*method\b", flags=re.IGNORECASE), "连接方式"),
    (re.compile(r"\bprotection\s*(?:grade|level)\b", flags=re.IGNORECASE), "防护等级"),
    (re.compile(r"\bfuse\s*current\b", flags=re.IGNORECASE), "保险丝电流"),
    (re.compile(r"\bcircuit\s*breaker\s*capacity\b", flags=re.IGNORECASE), "断路器容量"),
    (re.compile(r"\bpanel\s*model\b", flags=re.IGNORECASE), "面板型号"),
    (re.compile(r"\bpanel\s*weight\b", flags=re.IGNORECASE), "面板重量"),
    (re.compile(r"\brefrigerant\s*charge\b", flags=re.IGNORECASE), "冷媒充注量"),
    (re.compile(r"\bpipe\s*diameter\b", flags=re.IGNORECASE), "管径"),
    (re.compile(r"\bcoil\s*water\s*flow\b", flags=re.IGNORECASE), "盘管水流量"),
    (re.compile(r"\brated\s*power\s*cooling\b", flags=re.IGNORECASE), "额定制冷功率"),
    (re.compile(r"\brated\s*power\s*heating\b", flags=re.IGNORECASE), "额定制热功率"),
    (re.compile(r"\bpower\s*input\b", flags=re.IGNORECASE), "输入功率"),
    (re.compile(r"^power$", flags=re.IGNORECASE), "功率"),
    (re.compile(r"^hp$", flags=re.IGNORECASE), "匹数"),
    (
        re.compile(
            r"\b(?:optional\s*)?(?:electric\s*)?auxiliary(?:\s*electric)?\s*heating\b",
            flags=re.IGNORECASE,
        ),
        "可选电辅热功率",
    ),
)

_SPEC_EN_GAS_PIPE_RULE = re.compile(
    r"(?:connection|connecting)?\s*pipe.*\bgas\b|\bgas\b.*(?:connection|connecting)?\s*pipe",
    flags=re.IGNORECASE,
)
_SPEC_EN_LIQUID_PIPE_RULE = re.compile(
    r"(?:connection|connecting)?\s*pipe.*\bliquid\b|\bliquid\b.*(?:connection|connecting)?\s*pipe",
    flags=re.IGNORECASE,
)
_SPEC_EN_APF_RULE = re.compile(r"\bapf\b", flags=re.IGNORECASE)
_SPEC_EN_COP_RULE = re.compile(r"\bcop\b", flags=re.IGNORECASE)
_SPEC_EN_EER_RULE = re.compile(r"\beer\b", flags=re.IGNORECASE)

_SPEC_CN_CANONICAL_MAP: Dict[str, str] = {
    "外机静压": "机外静压",
    "最高机外静压": "机外静压",
    "机外静压范围": "静压范围",
    "标准机外静压": "标准静压",
}

_SPEC_RAW_HINTS: Tuple[str, ...] = (
    "标准静压",
    "静压范围",
    "机外静压",
    "静压",
    "循环风量",
    "风量",
    "制冷量",
    "制热量",
    "制冷功率",
    "制热功率",
    "耗电量",
    "噪音",
    "电源",
    "连接管",
    "排水管",
    "外形尺寸",
    "净重",
    "毛重",
)

_SPEC_RAW_NAME_KEYWORDS: Tuple[str, ...] = (
    "压",
    "量",
    "功率",
    "电流",
    "风量",
    "噪音",
    "电源",
    "频率",
    "尺寸",
    "重量",
    "管",
    "能效",
    "温度",
)

_SPEC_RAW_NAME_STOPWORDS: Tuple[str, ...] = (
    "型号",
    "产品",
    "系列",
    "机组",
    "备注",
    "说明",
    "参数",
)


def _localize_hml_tokens(name: str) -> str:
    if not name:
        return ""
    out = name
    replacements = (
        (r"\(H/M/L\)", "(高/中/低)"),
        (r"\(H/ML\)", "(高/中低)"),
        (r"\(HML\)", "(高中低)"),
        (r"\(H/W/L\)", "(高/强/低)"),
        (r"\(H\)", "(高)"),
        (r"\(M\)", "(中)"),
        (r"\(L\)", "(低)"),
        (r"\bH/M/L\b", "高/中/低"),
        (r"\bH/ML\b", "高/中低"),
        (r"\bHigh\b", "高"),
        (r"\bMedium\b", "中"),
        (r"\bLow\b", "低"),
        (r"_High\b", "_高"),
        (r"_Medium\b", "_中"),
        (r"_Low\b", "_低"),
        (r"_H\b", "_高"),
        (r"_M\b", "_中"),
        (r"_L\b", "_低"),
    )
    for pattern, new in replacements:
        out = re.sub(pattern, new, out, flags=re.IGNORECASE)
    return out


def _canonicalize_cn_spec_name(name: str) -> str:
    clean = re.sub(r"\s+", "", str(name or ""))
    clean = _SPEC_CN_CANONICAL_MAP.get(clean, clean)
    clean = re.sub(r"(?<=[0-9一-龥)])x(?=[0-9一-龥(])", "×", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bW[×x]D[×x]H\b", "宽×深×高", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bW[×x]H[×x]D\b", "宽×高×深", clean, flags=re.IGNORECASE)
    if any(token in clean for token in ("风量", "噪音", "噪声", "规格")):
        clean = _localize_hml_tokens(clean)
    return clean


def _infer_cn_spec_name_from_raw(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)

    for hint in _SPEC_RAW_HINTS:
        if hint in compact:
            return _canonicalize_cn_spec_name(hint)

    matched = re.search(r"([一-龥]{2,16})\s*(?:[:：]|[0-9])", text)
    if not matched:
        return ""
    candidate = _canonicalize_cn_spec_name(matched.group(1))
    if any(token in candidate for token in _SPEC_RAW_NAME_STOPWORDS):
        return ""
    if any(token in candidate for token in _SPEC_RAW_NAME_KEYWORDS):
        return candidate
    return ""


def _canonicalize_spec_name(name: str, raw: str) -> str:
    original = str(name or "").strip()
    if not original:
        return _infer_cn_spec_name_from_raw(raw)

    clean = re.sub(r"\s+", " ", original).strip(" :：;；,，")
    if re.search(r"[一-龥]", clean):
        return _canonicalize_cn_spec_name(clean)

    normalized = re.sub(r"[_\-]+", " ", clean)
    lower = normalized.lower()

    if _SPEC_EN_GAS_PIPE_RULE.search(lower):
        return "连接管气管"
    if _SPEC_EN_LIQUID_PIPE_RULE.search(lower):
        return "连接管液管"
    if _SPEC_EN_APF_RULE.search(lower):
        return "APF"
    if _SPEC_EN_COP_RULE.search(lower):
        return "COP"
    if _SPEC_EN_EER_RULE.search(lower):
        return "EER"

    for pattern, canonical in _SPEC_EN_NAME_RULES:
        if pattern.search(lower):
            return canonical

    inferred = _infer_cn_spec_name_from_raw(raw)
    if inferred:
        return inferred
    return clean


def _normalize_performance_specs(specs: Sequence[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    seen = set()
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        name = _canonicalize_spec_name(spec.get("name", ""), spec.get("raw", ""))
        value = str(spec.get("value") or "").strip()
        unit = str(spec.get("unit") or "").strip()
        raw = str(spec.get("raw") or "").strip()
        if not (name or value or unit or raw):
            continue
        item = {
            "name": name,
            "value": value,
            "unit": unit,
            "raw": raw,
        }
        key = (name, value, unit, raw)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _normalize_spec_key(name: str) -> str:
    lowered = str(name or "").strip().lower()
    lowered = (
        lowered.replace("（", "(")
        .replace("）", ")")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", ";")
    )
    return re.sub(r"[\s,;:|/\\\-_()]+", "", lowered)


def _pick_best_spec_entry(entries: List[Dict]) -> Dict:
    """Select the most informative spec entry among duplicates."""
    def _score(spec: Dict) -> Tuple[int, int, int]:
        unit = 1 if spec.get("unit") else 0
        has_digit = 1 if re.search(r"\d", str(spec.get("value", ""))) else 0
        length = len(str(spec.get("value", "")))
        return (unit, has_digit, length)

    sorted_entries = sorted(entries, key=_score, reverse=True)
    return dict(sorted_entries[0]) if sorted_entries else {}


def _enforce_single_value_specs(product: Dict, config: Dict) -> Dict:
    targets = {
        _normalize_spec_key(name)
        for name in (config.get("single_value_performance_specs") or [])
        if _normalize_spec_key(name)
    }
    if not targets:
        return product
    specs = product.get("performance_specs") or []
    grouped: Dict[str, List[Dict]] = {}
    kept: List[Dict] = []
    for spec in specs:
        key = _normalize_spec_key(spec.get("name", ""))
        if key in targets:
            grouped.setdefault(key, []).append(spec)
        else:
            kept.append(spec)
    for entries in grouped.values():
        best = _pick_best_spec_entry(entries)
        if best:
            kept.append(best)
    product["performance_specs"] = kept
    return product


def _deduplicate_products_by_model(products: List[Dict], config: Optional[Dict] = None) -> List[Dict]:
    """
    Deduplicate products by normalized product_model to reduce repeats where
    category/fields differ across chunks for the same model.
    """
    if not products:
        return []
    grouped: Dict[str, List[Dict]] = {}
    for product in products:
        model = (product.get("product_model") or "").strip()
        if not model:
            grouped[f"_no_model_{id(product)}"] = [product]
            continue
        key = _normalize_model_key(model)
        grouped.setdefault(key, []).append(product)

    merged_products: List[Dict] = []
    for key, group in grouped.items():
        if len(group) == 1 or key.startswith("_no_model_"):
            merged_products.append(group[0])
            continue
        base = dict(group[0])
        for candidate in group[1:]:
            for field in [
                "brand",
                "category",
                "series",
                "product_model",
                "manufacturer",
                "refrigerant",
                "energy_efficiency_grade",
            ]:
                left = (base.get(field) or "").strip()
                right = (candidate.get(field) or "").strip()
                if len(right) > len(left):
                    base[field] = right
            for list_field in ["features", "key_components", "fact_text", "evidence"]:
                base[list_field] = _dedup_list((base.get(list_field) or []) + (candidate.get(list_field) or []))
            base_specs = base.get("performance_specs") or []
            cand_specs = candidate.get("performance_specs") or []
            merged_specs = {}
            for spec in base_specs + cand_specs:
                spec_key = (
                    (spec.get("name") or "").strip(),
                    (spec.get("value") or "").strip(),
                    (spec.get("unit") or "").strip(),
                )
                if spec_key == ("", "", ""):
                    continue
                if spec_key not in merged_specs:
                    merged_specs[spec_key] = spec
            base["performance_specs"] = _normalize_performance_specs(list(merged_specs.values()))
        base = _enforce_single_value_specs(base, config or {})
        merged_products.append(base)
    return merged_products


def extract_relations_multistage(
    csv_path: Path,
    output_path: Path,
    error_log_path: Path,
    max_concurrent: int = 100,
    window_size: int = 1,
    use_sliding_window: bool = True,
    show_progress: bool = False,
) -> None:
    """
    Orchestrate multi-stage extraction end-to-end. Output shape matches v1.
    """
    cfg = dict(RELATION_EXTRACTOR_CONFIG)
    cfg["max_concurrent"] = max_concurrent
    cfg["global_concurrency"] = max_concurrent
    cfg.setdefault("max_chars_per_call", 8000)

    logger = logging.getLogger("relation_extract_batch_v2")
    logger.setLevel(logging.INFO)

    effective_window = window_size if use_sliding_window else 0
    pages = list(
        load_pages_with_context_v2(
            str(csv_path), window_size=effective_window, known_models=None
        )
    )
    if not pages:
        output_path.write_text("[]", encoding="utf-8")
        return

    extractor = StagedRelationExtractor(cfg)
    # toggle tqdm progress bars inside stage methods
    extractor._show_progress = show_progress

    # stage checkpoint directory per CSV
    stage_dir = output_path.parent / f"{csv_path.stem}_v2_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    brand_file = stage_dir / "brands.json"
    brand_alias_file = stage_dir / "brand_aliases.json"
    brand_candidates_file = stage_dir / "brand_candidates.json"
    brand_dropped_file = stage_dir / "brand_dropped.json"
    series_file = stage_dir / "series.json"
    series_alias_file = stage_dir / "series_aliases.json"
    series_features_file = stage_dir / "series_features.json"
    models_file = stage_dir / "models.json"
    models_explicit_file = stage_dir / "models_explicit.json"
    model_pages_file = stage_dir / "model_pages.json"
    model_conflicts_file = stage_dir / "model_conflicts.json"
    brand_series_rel_file = stage_dir / "brand_series_relations.json"
    series_model_rel_file = stage_dir / "series_model_relations.json"
    model_specs_rel_file = stage_dir / "model_specs_relations.json"
    products_file = stage_dir / "products_raw.json"
    force_rerun = cfg.get("force_stage_rerun", False)

    async def _run() -> List[Dict]:
        # Stage A: brands
        if brand_file.exists() and not force_rerun:
            brands = json.loads(brand_file.read_text(encoding="utf-8"))
            if brand_alias_file.exists():
                extractor.brand_alias_map = json.loads(brand_alias_file.read_text(encoding="utf-8"))
            if brand_candidates_file.exists():
                extractor.brand_candidates_all = json.loads(
                    brand_candidates_file.read_text(encoding="utf-8")
                )
            if brand_dropped_file.exists():
                extractor.brand_dropped = json.loads(brand_dropped_file.read_text(encoding="utf-8"))
        else:
            brands = await extractor.run_brand_stage(pages, error_log_path)
        brand_file.write_text(json.dumps(brands, ensure_ascii=False, indent=2), encoding="utf-8")
        brand_alias_file.write_text(
            json.dumps(extractor.brand_alias_map or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        brand_candidates_file.write_text(
            json.dumps(extractor.brand_candidates_all or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        brand_dropped_file.write_text(
            json.dumps(extractor.brand_dropped or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not brands:
            return []

        # Stage B: series
        if series_file.exists() and not force_rerun:
            brand_to_series = json.loads(series_file.read_text(encoding="utf-8"))
            if series_alias_file.exists():
                extractor.series_alias_map = json.loads(series_alias_file.read_text(encoding="utf-8"))
        else:
            brand_to_series = await extractor.run_series_stage(brands, pages, error_log_path)
        brand_to_series, extractor.series_alias_map, _ = _enforce_single_brand_series_scope(
            brands,
            brand_to_series,
            extractor.series_alias_map,
        )
        series_file.write_text(json.dumps(brand_to_series, ensure_ascii=False, indent=2), encoding="utf-8")
        series_alias_file.write_text(
            json.dumps(extractor.series_alias_map or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Stage C: models
        if models_file.exists() and not force_rerun:
            extractor.models_by_pair = _canonicalize_models_by_pair(
                json.loads(models_file.read_text(encoding="utf-8"))
            )
            if model_pages_file.exists():
                extractor.model_page_stats = json.loads(model_pages_file.read_text(encoding="utf-8"))
            if model_conflicts_file.exists():
                extractor.model_conflicts = json.loads(model_conflicts_file.read_text(encoding="utf-8"))
        else:
            await extractor.run_model_stage(brand_to_series, pages, error_log_path)
            models_file.write_text(
                json.dumps(extractor.models_by_pair, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model_pages_file.write_text(
                json.dumps(extractor.model_page_stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if extractor.model_conflicts:
                model_conflicts_file.write_text(
                    json.dumps(extractor.model_conflicts, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        extractor.models_by_pair, _ = _enforce_single_brand_models_scope(
            brands,
            extractor.models_by_pair,
        )
        extractor.model_conflicts, _ = _remap_model_conflicts_to_single_brand(
            brands,
            extractor.model_conflicts,
        )
        models_file.write_text(
            json.dumps(extractor.models_by_pair, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_pages_file.write_text(
            json.dumps(extractor.model_page_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if extractor.model_conflicts or model_conflicts_file.exists():
            model_conflicts_file.write_text(
                json.dumps(extractor.model_conflicts or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if series_features_file.exists() and not force_rerun:
            extractor.series_feature_map = json.loads(series_features_file.read_text(encoding="utf-8"))
            extractor._bootstrap_series_components_from_series_features()

        # Stage D: products
        if products_file.exists() and not force_rerun:
            products = json.loads(products_file.read_text(encoding="utf-8"))
        else:
            products = await extractor.run_product_stage(brand_to_series, pages, error_log_path)
        products, _ = _enforce_single_brand_products_scope(brands, products)
        extractor.series_feature_map, _ = _enforce_single_brand_series_feature_scope(
            brands,
            extractor.series_feature_map,
        )
        products_file.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
        if extractor.series_feature_map or series_features_file.exists():
            series_features_file.write_text(
                json.dumps(extractor.series_feature_map or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Hierarchy relation views (always refresh to keep caches consistent).
        brand_series_relations = _build_brand_series_relations(brands, brand_to_series)
        series_model_relations = _build_series_model_relations(extractor.models_by_pair)
        model_specs_relations = _build_model_specs_relations(products)
        models_explicit = _models_explicit_view(extractor.models_by_pair)
        brand_series_rel_file.write_text(
            json.dumps(brand_series_relations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        series_model_rel_file.write_text(
            json.dumps(series_model_relations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model_specs_rel_file.write_text(
            json.dumps(model_specs_relations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        models_explicit_file.write_text(
            json.dumps(models_explicit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return products

    raw_products = asyncio.run(_run())

    # Dedup + category correction + filter (reuse v1 utilities)
    deduped = deduplicate_results([{"results": raw_products}])
    corrected = correct_all_categories(deduped)
    filtered = filter_empty_products(corrected)

    final_payload = [{"results": filtered}]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "v2 extraction finished: %s -> %s products (raw=%s, dedup=%s)",
        csv_path.name,
        len(filtered),
        len(raw_products),
        len(filtered),
    )
