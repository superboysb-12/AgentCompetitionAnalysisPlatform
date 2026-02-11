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
from openai import AsyncOpenAI
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - fallback handled later
    SentenceTransformer = None

from backend.settings import RELATION_EXTRACTOR_CONFIG
from LLMRelationExtracter import (  # v1 复用的通用模块
    correct_all_categories,
    deduplicate_results,
    filter_empty_products,
    load_pages_with_context,
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
    SERIES_SCHEMA,
    SERIES_REVIEW_SCHEMA,
    build_brand_canon_messages,
    build_brand_filter_messages,
    build_brand_global_filter_messages,
    build_brand_messages,
    build_model_messages,
    build_model_review_messages,
    build_product_messages,
    build_product_review_messages,
    build_series_review_messages,
    build_series_messages,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_pages_with_context_v2(
    csv_path: str, window_size: int = 1, known_models: Optional[List[str]] = None
) -> Iterable[Tuple[str, Dict]]:
    """Alias for v1 loader to保持 I/O 一致。"""
    yield from load_pages_with_context(csv_path, window_size=window_size, known_models=known_models)


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
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _normalize_series_key(text: str) -> str:
    """Normalization for series merge/canonical matching."""
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())
    return normalized.replace("系列", "")


def _match_series_name(series_guess: str, brand: str, alias_map: Dict[str, Dict[str, List[str]]]) -> Optional[str]:
    "Return canonical series name within a brand if guess matches alias/canonical."
    guess_key = _normalize_series_key(series_guess)
    if not guess_key:
        return None
    brand_map = alias_map.get(brand, {}) if alias_map else {}
    for canonical, aliases in brand_map.items():
        for name in [canonical] + aliases:
            if _normalize_series_key(name) == guess_key:
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
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed")
    return SentenceTransformer(model_name, device=device)


def _get_concurrency(config: Dict, key: str, default: int) -> int:
    """
    Unified concurrency resolver: prefer specific key, fallback to global_concurrency,
    then to provided default.
    """
    if key in config and config[key] is not None:
        return int(config[key])
    if "global_concurrency" in config and config["global_concurrency"] is not None:
        return int(config["global_concurrency"])
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
    merged: Dict[str, Dict] = {}
    for item in series_items:
        name = (item.get("name") or "").strip()
        key = _normalize_series_key(name)
        if not key:
            continue
        if key not in merged:
            merged[key] = {
                "name": name,
                "evidence": list(item.get("evidence", [])),
                "pages": list(item.get("pages", [])),
            }
        else:
            merged[key]["evidence"].extend(item.get("evidence", []))
            merged[key]["pages"].extend(item.get("pages", []))
    out: List[Dict] = []
    for value in merged.values():
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
    try:
        names = [s.get("name", "") for s in series_items]
        embeddings = _embed_texts(names, model_name, device=device)
    except ImportError:
        return series_items

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
    try:
        names = [b.get("name", "") for b in brands]
        embeddings = _embed_texts(names, model_name, device=device)
    except ImportError:
        return brands

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


# --------------------------------------------------------------------------- #
# Core extractor
# --------------------------------------------------------------------------- #


class StagedRelationExtractor:
    def __init__(self, config: Optional[Dict] = None, log_name: str = "relation_extractor_v2") -> None:
        self.config = dict(RELATION_EXTRACTOR_CONFIG)
        if config:
            self.config.update(config)

        self.client = AsyncOpenAI(
            api_key=self.config["api_key"], base_url=self.config["base_url"]
        )
        self.model = self.config["model"]
        self.semaphore = asyncio.Semaphore(_get_concurrency(self.config, "max_concurrent", 5))
        self.timeout = self.config.get("timeout", 300)
        self.logger = self._setup_logger(log_name)
        # feature toggles for composable pipeline
        self.enable_brand_stage = self.config.get("enable_brand_stage", True)
        self.enable_brand_cluster = self.config.get("enable_brand_cluster", True)
        self.enable_series_stage = self.config.get("enable_series_stage", True)
        self.enable_model_stage = self.config.get("enable_model_stage", True)
        self.enable_product_stage = self.config.get("enable_product_stage", True)
        self.brand_alias_map: Dict[str, List[str]] = {}
        self.series_alias_map: Dict[str, Dict[str, List[str]]] = {}
        self.series_feature_map: Dict[str, Dict[str, List[Dict]]] = {}
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
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
                timeout=self.timeout,
            )
        content = (response.choices[0].message.content or "").strip()
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
        try:
            texts = [b.get("name", "") for b in brands]
            embeddings = _embed_texts(texts, model_name, device=device)
        except ImportError:
            return [[b] for b in brands]

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
        if (
            not brands
            or not self.config.get("enable_brand_refine", True)
            or len(brands) < int(self.config.get("brand_refine_min_count", 3))
        ):
            return brands

        # prepare clusters by name similarity
        clusters = self._cluster_brand_names(
            brands,
            model_name=self.config.get("brand_refine_embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            distance_threshold=float(self.config.get("brand_refine_threshold", 0.35)),
            device=self.config.get("brand_refine_device", "cpu"),
        )

        refined: List[Dict] = []
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
        refined = _merge_translated_brands(
            refined,
            model_name=self.config.get(
                "brand_translate_embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            distance_threshold=float(self.config.get("brand_translate_threshold", 0.25)),
            device=self.config.get("brand_translate_device", "cpu"),
        )

        # global LLM pass to drop residual series/model/non-brand items
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

        # canonicalize to Chinese brand form
        if self.config.get("enable_brand_canon", True):
            refined, alias_map = await self._canonicalize_brands(refined, error_log)
            self.brand_alias_map = alias_map
        return refined

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

        # optional clustered re-scan to boost recall
        cluster_size = int(self.config.get("brand_cluster_size", 5))
        if self.enable_brand_cluster and cluster_size > 1:
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
        # prune long-tail noisy candidates to keep downstream stages tractable
        merged_list = _prune_brands(
            merged_list,
            min_page_ratio=float(self.config.get("brand_min_page_ratio", 0.15)),
            max_candidates=int(self.config.get("brand_max_candidates", 12)),
        )
        merged_list = await self._refine_brands(merged_list, error_log)
        return merged_list

    # -------------------------- Pipeline facades -------------------------- #
    async def run_brand_stage(self, pages: Sequence[Tuple[str, Dict]], error_log: Path) -> List[Dict]:
        if not self.enable_brand_stage:
            return []
        return await self.extract_brands(pages, error_log)

    async def run_series_stage(
        self, brands: List[Dict], pages: Sequence[Tuple[str, Dict]], error_log: Path
    ) -> Dict[str, List[Dict]]:
        if not self.enable_series_stage:
            return {b.get("name", ""): [] for b in brands}
        return await self.extract_series(brands, pages, error_log)

    async def run_model_stage(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> Dict[str, List[Dict]]:
        if not self.enable_model_stage:
            self.models_by_pair = {}
            self.model_page_stats = {}
            self.model_conflicts = {}
            return {}
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
        if not self.enable_product_stage:
            return []
        return await self.extract_products(brand_to_series, pages, error_log)

    # -------------------------- Stage B: series ---------------------------- #
    async def _extract_series_for_brand(
        self,
        brand: str,
        combined_text: str,
        error_log: Path,
        page_refs: Optional[List] = None,
        stage_a_pages: Optional[List] = None,
    ) -> List[Dict]:
        chunk_pbar = None
        try:
            chunks = _split_text(combined_text, self.config.get("max_chars_per_call", 8000))
            merged = {}
            if getattr(self, "_show_progress", False):
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
            key = _normalize_series_key(item.get("original", ""))
            if key:
                review_map[key] = item

        merged: Dict[str, Dict] = {}
        alias_map: Dict[str, List[str]] = {}
        dropped = 0
        for series in candidates:
            original = (series.get("name") or "").strip()
            original_key = _normalize_series_key(original)
            review = review_map.get(original_key)

            # Keep unmatched outputs to avoid accidental recall collapse.
            if review is None:
                canonical = original
                keep_flag = True
            else:
                keep_flag = bool(review.get("keep")) and review.get("kind") == "series"
                canonical = (review.get("canonical") or original).strip()

            if not keep_flag:
                dropped += 1
                continue

            canonical_key = _normalize_series_key(canonical)
            if not canonical_key:
                canonical = original
                canonical_key = original_key
            if canonical_key not in merged:
                merged[canonical_key] = {
                    "name": canonical,
                    "evidence": list(series.get("evidence", [])),
                    "pages": list(series.get("pages", [])),
                }
            else:
                merged[canonical_key]["evidence"].extend(series.get("evidence", []))
                merged[canonical_key]["pages"].extend(series.get("pages", []))
            alias_map.setdefault(canonical, []).append(original)

        reviewed: List[Dict] = []
        for value in merged.values():
            value["evidence"] = _dedup_list(value.get("evidence", []))
            value["pages"] = _dedup_list(value.get("pages", []))
            reviewed.append(value)
        for name in alias_map:
            alias_map[name] = _dedup_list(alias_map[name])

        self.logger.info(
            "Series review for brand=%s: candidates=%s dropped=%s kept=%s",
            brand,
            len(candidates),
            dropped,
            len(reviewed),
        )

        # Guardrail fallback when review is over-aggressive.
        min_ratio = float(self.config.get("series_filter_min_keep_ratio", 0.3))
        if candidates and (not reviewed or len(reviewed) < min_ratio * len(candidates)):
            self.logger.warning(
                "Series review kept %s/%s (< %.2f); fallback to pre-review candidates for brand=%s",
                len(reviewed),
                len(candidates),
                min_ratio,
                brand,
            )
            return candidates, {}

        return reviewed, alias_map

    async def extract_series(
        self, brands: List[Dict], pages: Sequence[Tuple[str, Dict]], error_log: Path
    ) -> Dict[str, List[Dict]]:
        brand_to_series: Dict[str, List[Dict]] = {}
        iterator = brands
        pbar = None
        if getattr(self, "_show_progress", False):
            pbar = tqdm(total=len(brands), desc="Stage B: series", unit="brand")
        for brand_item in iterator:
            brand_name = brand_item["name"]
            brand_aliases = self.brand_alias_map.get(brand_name, []) if hasattr(self, "brand_alias_map") else []
            series_keyword_boost = self.config.get("series_keyword_boost", [])
            if isinstance(series_keyword_boost, str):
                series_keyword_boost = [
                    k.strip() for k in series_keyword_boost.split(",") if k.strip()
                ]
            keywords = _dedup_list([brand_name] + brand_aliases + list(series_keyword_boost))
            relevant_pages, stage_a_pages = self._collect_series_context_pages(
                brand_item,
                pages,
                keywords,
            )

            # Stage-B page grouping: 2 or 3 pages per chunk.
            chunk_size = self._resolve_series_chunk_size()
            chunks = _chunk_sequence(relevant_pages, chunk_size)
            raw_series_list: List[Dict] = []

            async def _run_chunk(chunk_pages: Sequence[Tuple[str, Dict]]) -> List[Dict]:
                combined = _combine_pages(chunk_pages, self.config)
                chunk_refs = _dedup_list(
                    [
                        meta.get("page")
                        for _, meta in chunk_pages
                        if meta.get("page") is not None and str(meta.get("page")) != ""
                    ]
                )
                return await self._extract_series_for_brand(
                    brand_name,
                    combined,
                    error_log,
                    page_refs=chunk_refs,
                    stage_a_pages=stage_a_pages,
                )

            if len(chunks) == 1:
                raw_series_list = await _run_chunk(chunks[0])
            else:
                sem = asyncio.Semaphore(_get_concurrency(self.config, "series_chunk_concurrency", 6))

                async def _bounded(chunk_pages):
                    async with sem:
                        return await _run_chunk(chunk_pages)

                tasks = [asyncio.create_task(_bounded(c)) for c in chunks]
                results = await asyncio.gather(*tasks)
                for res in results:
                    raw_series_list.extend(res or [])

            # Stage-B candidate consolidation (similar to Stage-A: merge -> semantic merge -> LLM review)
            raw_series_list = _merge_series_candidates(raw_series_list)

            if self.config.get("enable_series_semantic_merge", True):
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

            brand_to_series[brand_name] = series_list
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        return _enrich_series_by_brand(brand_to_series)

    # -------------------------- Stage C: models ---------------------------- #
    async def _extract_models_for_pair(
        self,
        brand: str,
        series: str,
        text_block: str,
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
                            build_model_messages(brand, series, chunk_text),
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
            if self.config.get("enable_model_llm_review", True) and merged_models:
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
        try:
            result = await self._acall(
                build_product_review_messages(brand, series, payload),
                PRODUCT_REVIEW_SCHEMA,
                "product_review_schema",
                {"brand": brand, "series": series, "count": len(payload)},
            )
            verdicts = {item.get("product_model", "").strip(): item for item in result.get("products", [])}
            reviewed: List[Dict] = []
            enable_series_feature = bool(self.config.get("series_feature_to_series", False))
            for prod in payload:
                model_key = (prod.get("product_model") or "").strip()
                category_key = (prod.get("category") or "").strip()
                decision = verdicts.get(model_key)
                if decision is None:
                    reviewed.append(prod)
                    continue
                role = decision.get("role")
                if role == "series_feature" and enable_series_feature:
                    self._store_series_feature(brand, series, prod, decision)
                    continue
                if role == "accessory":
                    continue
                if decision.get("keep") and role in (None, "product") and not decision.get("is_accessory"):
                    reviewed.append(prod)
            return reviewed or products
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
        brand_bucket = self.series_feature_map.setdefault(brand, {})
        feature_list = brand_bucket.setdefault(series_key, [])
        entry = {
            "title": prod.get("product_model", ""),
            "fact_text": prod.get("fact_text", []),
            "features": prod.get("features", []),
            "performance_specs": prod.get("performance_specs", []),
            "evidence": _dedup_list(prod.get("evidence", [])),
            "source_category": prod.get("category", ""),
        }
        if decision is not None:
            entry["series_feature_flag"] = bool(decision.get("series_feature", False))
        title_norm = _normalize_name(entry.get("title", ""))
        if any(_normalize_name(item.get("title", "")) == title_norm for item in feature_list):
            return
        feature_list.append(entry)

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
        if not self.config.get("product_model_must_from_stage_c", True):
            return products

        known = _dedup_list(
            [
                _canonicalize_model_name(m)
                for m in (known_models or [])
                if _canonicalize_model_name(m)
            ]
        )
        if not known:
            return products

        norm_to_known: Dict[str, str] = {}
        for name in known:
            key = _normalize_model_key(name)
            if key and key not in norm_to_known:
                norm_to_known[key] = name
        if not norm_to_known:
            return products

        sorted_norm_keys = sorted(norm_to_known.keys(), key=len, reverse=True)
        drop_if_unknown = bool(self.config.get("product_model_drop_if_unknown", True))
        expand_multi = bool(self.config.get("product_model_expand_multi_match", True))
        match_in_evidence = bool(self.config.get("product_model_match_in_evidence", True))
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
            if not matches and match_in_evidence:
                evidence_text = " ".join(str(x) for x in (product.get("evidence") or []))
                fact_text = " ".join(str(x) for x in (product.get("fact_text") or []))
                for probe in (evidence_text, fact_text):
                    matches = _match(probe)
                    if matches:
                        break

            if not matches:
                if drop_if_unknown:
                    dropped += 1
                    continue
                bound.append(product)
                continue

            if len(matches) > multi_cap:
                matches = matches[:multi_cap]

            if len(matches) > 1 and expand_multi:
                expanded += len(matches) - 1
                for model_name in matches:
                    cloned = dict(product)
                    cloned["product_model"] = model_name
                    bound.append(cloned)
            else:
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
        try:
            result = await self._acall(
                build_model_review_messages(brand, series, payload),
                MODEL_REVIEW_SCHEMA,
                "model_review_schema",
                {"brand": brand, "series": series, "count": len(payload)},
            )
            verdicts = {item.get("name", "").strip(): item for item in result.get("items", [])}
            reviewed: List[Dict] = []
            alias_map = getattr(self, "series_alias_map", {})
            allow_redirect = bool(self.config.get("model_review_redirect", True))
            drop_mismatch = bool(self.config.get("model_review_drop_mismatch", False))
            redirect_conf = float(self.config.get("model_redirect_min_conf", 0.5))
            for cand in payload:
                name = (cand.get("name") or "").strip()
                decision = verdicts.get(name)
                if decision and decision.get("keep") and decision.get("kind") == "model":
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
            # fallback: avoid empty due to over-prune
            return reviewed or candidates
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
                follow_after = int(
                    self.config.get(
                        "series_context_follow_pages",
                        self.config.get("model_context_follow_pages", 2),
                    )
                )
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
                text_block = _combine_pages(relevant_pages, self.config)
                pair_models = await self._extract_models_for_pair(
                    brand,
                    series_name,
                    text_block,
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
        if self.config.get("enable_model_cross_series_resolve", True):
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
                # Product stage context: model-hit pages + next N pages, not BM25 retrieval.
                pair_models = self.models_by_pair.get(_pair_key(brand, series_name), [])
                if self.enable_model_stage and not pair_models:
                    return pair_index, []
                known_models = _dedup_list(
                    [
                        _canonicalize_model_name(model_item.get("name") or "")
                        for model_item in pair_models
                        if _canonicalize_model_name(model_item.get("name") or "")
                    ]
                )
                follow_after = int(self.config.get("model_context_follow_pages", 2))
                use_target_mode = bool(self.config.get("stage_d_target_model_mode", True))

                if use_target_mode and known_models:
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
                            if not model_pages and series_pages:
                                model_pages = _dedup_list(series_pages)
                            if not model_pages:
                                return []
                            model_relevant_pages = _select_pages_by_numbers_with_following(
                                pages,
                                model_pages,
                                follow_after=follow_after,
                            )
                            if not model_relevant_pages:
                                return []
                            text_block = _combine_pages(model_relevant_pages, self.config)
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
                            if self.config.get("enable_product_llm_review", True) and processed_model:
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

                # Fallback legacy pair-level extraction path.
                model_hit_pages: List = []
                for model_name in known_models:
                    model_hit_pages.extend(self.model_page_stats.get(model_name, []))
                if not model_hit_pages:
                    for model_item in pair_models:
                        model_hit_pages.extend(model_item.get("pages", []))
                if not model_hit_pages and series_pages:
                    model_hit_pages.extend(series_pages)

                relevant_pages = _select_pages_by_numbers_with_following(
                    pages,
                    _dedup_list(model_hit_pages),
                    follow_after=follow_after,
                )
                if not relevant_pages and series_pages:
                    relevant_pages = _select_pages_by_numbers_with_following(
                        pages,
                        _dedup_list(series_pages),
                        follow_after=follow_after,
                    )
                if not relevant_pages:
                    self.logger.warning(
                        "Stage D: no context pages for brand=%s series=%s; skip pair to avoid noise",
                        brand,
                        series_name,
                    )
                    return pair_index, []
                text_block = _combine_pages(relevant_pages, self.config)
                page_refs = [meta.get("page") for _, meta in relevant_pages]
                extracted = await self._extract_products_for_pair(
                    brand,
                    series_name,
                    text_block,
                    page_refs,
                    known_models,
                    error_log,
                    target_model=None,
                )
                processed = list(extracted)
                if self.config.get("enable_product_llm_review", True) and processed:
                    processed = await self._review_products_llm(
                        brand,
                        series_name,
                        processed,
                        error_log,
                    )
                processed = self._bind_products_to_known_models(
                    brand,
                    series_name,
                    processed,
                    known_models,
                )
                return pair_index, processed
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
            self.logger.warning("No brands found; fallback to whole-document extraction.")
            brands = [{"name": "", "evidence": [], "pages": []}]

        brand_to_series = await self.run_series_stage(brands, pages, error_log)
        await self.run_model_stage(brand_to_series, pages, error_log)
        products = await self.run_product_stage(brand_to_series, pages, error_log)

        if not products:
            self.logger.warning("No products extracted; emitting doc-level fallback.")
            fallback = self._ensure_product_defaults(_default_product_template(), "", "", [])
            fallback["product_model"] = "UNKNOWN_MODEL"
            products = [fallback]

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
    """Keyword retrieval with hit-count ranking; fallback to all pages when no match."""
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
        return list(pages)

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [page for _, _, page in ranked]
    if int(top_k) > 0:
        selected = selected[: int(top_k)]
    return selected


def _render_rows_blocks(rows, config: Optional[Dict]) -> List[str]:
    try:
        rows_list = list(rows) if isinstance(rows, (list, tuple)) else [rows]
    except Exception:
        rows_list = [rows]
    rows_per_block = max(1, int(config.get("table_rows_per_block", 10))) if config else 10
    max_cell_len = int(config.get("table_row_cell_clip", 0)) if config else 0
    blocks: List[str] = []
    for i in range(0, len(rows_list), rows_per_block):
        block_rows = rows_list[i : i + rows_per_block]
        rendered = []
        for r in block_rows:
            try:
                row_str = json.dumps(r, ensure_ascii=False)
            except Exception:
                row_str = str(r)
            if max_cell_len and len(row_str) > max_cell_len:
                row_str = row_str[:max_cell_len] + "..."
            rendered.append(row_str)
        if rendered:
            blocks.append("\n".join(rendered))
    return blocks


def _combine_pages(pages: Sequence[Tuple[str, Dict]], config: Optional[Dict] = None) -> str:
    chunks = []
    for text, meta in pages:
        page = meta.get("page", "")
        chunk = [f"<<PAGE {page}>>", text]
        rows = meta.get("rows")
        if rows:
            row_blocks: List[str] = []
            if config and config.get("table_row_exhaust_mode", False):
                row_blocks = _render_rows_blocks(rows, config)
            else:
                try:
                    clip_chars = (
                        int(config.get("table_rows_clip_chars", 4000)) if config else 4000
                    )
                except Exception:
                    clip_chars = 4000
                try:
                    raw_rows = json.dumps(rows, ensure_ascii=False)
                    if clip_chars and clip_chars > 0:
                        raw_rows = raw_rows[:clip_chars]
                    row_blocks = [raw_rows]
                except Exception:
                    row_blocks = []
            for block in row_blocks:
                chunk.append("[rows_json]")
                chunk.append(block)
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


def _normalize_spec_key(name: str) -> str:
    return re.sub(r"[\s：:|/\\-]+", "", (name or "")).lower()


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
            base["performance_specs"] = list(merged_specs.values())
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
    cfg.setdefault("max_chars_per_call", 8000)

    logger = logging.getLogger("relation_extract_batch_v2")
    logger.setLevel(logging.INFO)

    use_sliding_window = use_sliding_window and not cfg.get("v2_disable_loader_sliding_window", True)
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
        else:
            brands = await extractor.run_brand_stage(pages, error_log_path)
            brand_file.write_text(json.dumps(brands, ensure_ascii=False, indent=2), encoding="utf-8")
            if extractor.brand_alias_map:
                brand_alias_file.write_text(
                    json.dumps(extractor.brand_alias_map, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        if not brands:
            extractor.logger.warning("No brands found; fallback to whole-document extraction.")
            brands = [{"name": "", "evidence": [], "pages": []}]

        # Stage B: series
        if series_file.exists() and not force_rerun:
            brand_to_series = json.loads(series_file.read_text(encoding="utf-8"))
            if series_alias_file.exists():
                extractor.series_alias_map = json.loads(series_alias_file.read_text(encoding="utf-8"))
        else:
            brand_to_series = await extractor.run_series_stage(brands, pages, error_log_path)
            series_file.write_text(json.dumps(brand_to_series, ensure_ascii=False, indent=2), encoding="utf-8")
            if extractor.series_alias_map:
                series_alias_file.write_text(
                    json.dumps(extractor.series_alias_map, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        brand_to_series = _enrich_series_by_brand(brand_to_series)

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
        extractor.models_by_pair = _enrich_models_by_pair(extractor.models_by_pair)

        # Stage D: products
        if products_file.exists() and not force_rerun:
            products = json.loads(products_file.read_text(encoding="utf-8"))
            if series_features_file.exists():
                extractor.series_feature_map = json.loads(series_features_file.read_text(encoding="utf-8"))
        else:
            products = await extractor.run_product_stage(brand_to_series, pages, error_log_path)
            products_file.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
            if extractor.series_feature_map:
                series_features_file.write_text(
                    json.dumps(extractor.series_feature_map, ensure_ascii=False, indent=2),
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

        if not products:
            extractor.logger.warning("No products extracted; emitting doc-level fallback.")
            fallback = extractor._ensure_product_defaults(_default_product_template(), "", "", [])
            fallback["product_model"] = "UNKNOWN_MODEL"
            products = [fallback]

        return products

    raw_products = asyncio.run(_run())

    # Dedup + category correction + filter (reuse v1 utilities)
    deduped = deduplicate_results([{"results": raw_products}])
    corrected = correct_all_categories(deduped)
    filtered = filter_empty_products(corrected)

    if not filtered:
        fallback = _default_product_template()
        fallback["product_model"] = "UNKNOWN_MODEL"
        filtered = [fallback]

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
