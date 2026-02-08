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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

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
    PRODUCT_SCHEMA,
    SERIES_SCHEMA,
    build_brand_canon_messages,
    build_brand_filter_messages,
    build_brand_global_filter_messages,
    build_brand_messages,
    build_product_messages,
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


def _embed_texts(texts: List[str], model_name: str, device: str = "cpu") -> np.ndarray:
    """
    Embed a list of texts with a sentence-transformers model.
    Raises ImportError if sentence_transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "sentence-transformers not installed; install to enable embedding clustering."
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


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
        self.semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 5))
        self.timeout = self.config.get("timeout", 300)
        self.logger = self._setup_logger(log_name)
        # feature toggles for composable pipeline
        self.enable_brand_stage = self.config.get("enable_brand_stage", True)
        self.enable_brand_cluster = self.config.get("enable_brand_cluster", True)
        self.enable_series_stage = self.config.get("enable_series_stage", True)
        self.enable_product_stage = self.config.get("enable_product_stage", True)
        self.brand_alias_map: Dict[str, List[str]] = {}

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
        self, brand: str, combined_text: str, error_log: Path, page_refs: Optional[List] = None
    ) -> List[Dict]:
        try:
            chunks = _split_text(combined_text, self.config.get("max_chars_per_call", 8000))
            merged = {}
            for chunk in chunks:
                payload = await self._acall(
                    build_series_messages(brand, chunk),
                    SERIES_SCHEMA,
                    "series_schema",
                    {"brand": brand},
                )
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
            merged_list = []
            for v in merged.values():
                v["evidence"] = _dedup_list(v.get("evidence", []))
                v["pages"] = _dedup_list(v.get("pages", []))
                merged_list.append(v)
            return merged_list
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "series_extract", {"brand": brand}, exc)
            self.logger.error("Series extract failed for brand %s: %s", brand, exc)
            return []

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
            keywords = _dedup_list([brand_name] + brand_aliases)
            relevant_pages = _select_pages_by_keywords(pages, keywords)
            combined_text = _combine_pages(relevant_pages)
            series_list: List[Dict] = await self._extract_series_for_brand(
                brand_name, combined_text, error_log
            )
            brand_to_series[brand_name] = series_list
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        return brand_to_series

    # -------------------------- Stage C: products -------------------------- #
    async def _extract_products_for_pair(
        self,
        brand: str,
        series: str,
        text_block: str,
        page_refs: List,
        error_log: Path,
    ) -> List[Dict]:
        try:
            chunks = _split_text(text_block, self.config.get("max_chars_per_call", 8000))
            products = []
            for chunk in chunks:
                payload = await self._acall(
                    build_product_messages(brand, series, chunk),
                    PRODUCT_SCHEMA,
                    "product_schema",
                    {"brand": brand, "series": series},
                )
                for item in payload.get("results", []):
                    products.append(self._ensure_product_defaults(item, brand, series, page_refs))
            return products
        except Exception as exc:  # noqa: BLE001
            _append_error(error_log, "product_extract", {"brand": brand, "series": series}, exc)
            self.logger.error(
                "Product extract failed for brand=%s series=%s: %s", brand, series, exc
            )
            return []

    async def extract_products(
        self,
        brand_to_series: Dict[str, List[Dict]],
        pages: Sequence[Tuple[str, Dict]],
        error_log: Path,
    ) -> List[Dict]:
        products: List[Dict] = []
        pairs = []
        for brand, series_list in brand_to_series.items():
            targets = series_list or [{"name": ""}]
            for series_item in targets:
                pairs.append((brand, series_item.get("name", "")))

        pbar = None
        if getattr(self, "_show_progress", False):
            pbar = tqdm(total=len(pairs), desc="Stage C: products", unit="pair")

        for brand, series_name in pairs:
            brand_aliases = self.brand_alias_map.get(brand, []) if hasattr(self, "brand_alias_map") else []
            keywords = [brand] + brand_aliases
            if series_name:
                keywords.append(series_name)
            keywords = _dedup_list(keywords)
            relevant_pages = _select_pages_by_keywords(pages, keywords)
            text_block = _combine_pages(relevant_pages)
            page_refs = [meta.get("page") for _, meta in relevant_pages]
            extracted = await self._extract_products_for_pair(
                brand, series_name, text_block, page_refs, error_log
            )
            products.extend(extracted)
            if pbar:
                pbar.update(1)

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

        # Dedup any repeated page_refs in evidence string hints
        merged["evidence"] = _dedup_list(merged.get("evidence", []))

        return merged


# --------------------------------------------------------------------------- #
# Module-level runner with stage checkpoints
# --------------------------------------------------------------------------- #


def _select_pages_by_keywords(
    pages: Sequence[Tuple[str, Dict]], keywords: Sequence[str]
) -> List[Tuple[str, Dict]]:
    lowered = [k.lower() for k in keywords if k]
    if not lowered:
        return list(pages)

    selected = []
    for text, meta in pages:
        text_lower = text.lower()
        if all(k in text_lower for k in lowered):
            selected.append((text, meta))

    return selected or list(pages)


def _combine_pages(pages: Sequence[Tuple[str, Dict]]) -> str:
    chunks = []
    for text, meta in pages:
        page = meta.get("page", "")
        chunks.append(f"<<PAGE {page}>>\n{text}")
    return "\n\n".join(chunks)


def extract_relations_multistage(
    csv_path: Path,
    output_path: Path,
    error_log_path: Path,
    max_concurrent: int = 10,
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
        else:
            brand_to_series = await extractor.run_series_stage(brands, pages, error_log_path)
            series_file.write_text(json.dumps(brand_to_series, ensure_ascii=False, indent=2), encoding="utf-8")

        # Stage C: products
        if products_file.exists() and not force_rerun:
            products = json.loads(products_file.read_text(encoding="utf-8"))
        else:
            products = await extractor.run_product_stage(brand_to_series, pages, error_log_path)
            products_file.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")

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
