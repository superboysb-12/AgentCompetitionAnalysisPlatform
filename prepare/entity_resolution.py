"""
Entity cleaning and resolution pipeline for HVAC product data.

This script implements the approach outlined in the provided report:
- Unicode/encoding fixes via ftfy + NFKC normalization
- Company suffix stripping and symbol cleanup
- Multilingual sentence embeddings (SentenceTransformer) for semantic similarity
- FAISS ANN search to avoid O(N^2) comparisons
- Graph clustering with embedding similarity + RapidFuzz string ratio as a safety net
- Canonical value selection per cluster and re-application to every record

Usage (CPU-friendly defaults):
    python prepare/entity_resolution.py \
        --input fused_entities_all.json \
        --output cleaned_entities.jsonl \
        --cluster-report cluster_report.json

You can optionally limit records for a quick smoke test with --limit 500.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import faiss
import ftfy
import numpy as np
from cleanco import basename
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("entity_resolution")


TARGET_FIELDS = ["brand", "series", "category", "manufacturer"]
TRADEMARK_PATTERN = re.compile(r"[®©™]")
PUNCTUATION_PATTERN = re.compile(r"[，、|•·･·．/]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
CHINESE_COMPANY_SUFFIX = re.compile(
    r"(有限责任公司|有限公司|有限的公司|股份有限公司|集团有限公司|集团公司|控股有限公司)$"
)
EN_COMPANY_SUFFIX = re.compile(
    r"(co(\.|mpany)?|corp(\.|oration)?|inc(\.)?|ltd(\.)?|limited|holding(s)?)(\s+|$)",
    re.IGNORECASE,
)


@dataclass
class ResolveConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: Optional[str] = None
    batch_size: int = 256
    top_k: int = 20
    embed_threshold: float = 0.82
    fuzzy_threshold: int = 90
    min_value_len: int = 2
    # Optional per-field overrides to tighten loose clusters (e.g., manufacturer).
    field_overrides: Dict[str, Dict[str, object]] = None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def normalize_text(value: str) -> str:
    if not value:
        return ""
    text = str(value)
    text = ftfy.fix_text(text, normalization="NFKC")
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = TRADEMARK_PATTERN.sub(" ", text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def normalize_company(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    cleaned = basename(text).strip()
    cleaned = CHINESE_COMPANY_SUFFIX.sub("", cleaned).strip()
    cleaned = EN_COMPANY_SUFFIX.sub("", cleaned).strip()
    if not cleaned:
        cleaned = text
    return cleaned


def normalize_brand(value: str) -> str:
    text = normalize_text(value)
    return text.upper()


def normalize_series(value: str) -> str:
    text = normalize_text(value)
    return text.upper()


def normalize_category(value: str) -> str:
    text = normalize_text(value)
    return text.lower()


def normalize_model(value: str) -> str:
    text = normalize_text(value)
    text = text.replace(" ", "").upper()
    return text


def dedup_list(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    cleaned: List[str] = []
    for raw in values or []:
        val = normalize_text(str(raw))
        if not val or val in seen:
            continue
        seen.add(val)
        cleaned.append(val)
    return cleaned


def load_entities(path: Path, limit: Optional[int] = None) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    flattened: List[Dict] = []
    for batch_idx, batch in enumerate(data):
        for result_idx, result in enumerate(batch.get("results", [])):
            if not result:
                continue
            item = dict(result)
            item["_batch"] = batch_idx
            item["_result_idx"] = result_idx
            flattened.append(item)
            if limit and len(flattened) >= limit:
                return flattened
    return flattened


def clean_entity(record: MutableMapping[str, object]) -> MutableMapping[str, object]:
    record = dict(record)
    record["normalized_brand"] = normalize_brand(record.get("brand", ""))
    record["normalized_series"] = normalize_series(record.get("series", ""))
    record["normalized_category"] = normalize_category(record.get("category", ""))
    record["normalized_manufacturer"] = normalize_company(record.get("manufacturer", ""))
    record["normalized_product_model"] = normalize_model(record.get("product_model", ""))
    record["clean_features"] = dedup_list(record.get("features", []))
    record["clean_key_components"] = dedup_list(record.get("key_components", []))
    return record


def clean_entities(entities: List[MutableMapping[str, object]]) -> List[MutableMapping[str, object]]:
    return [clean_entity(e) for e in entities]


def collect_field_counts(
    entities: Sequence[Mapping[str, object]], field: str
) -> Counter:
    key = f"normalized_{field}"
    counter: Counter = Counter()
    for entity in entities:
        value = entity.get(key, "")
        if value:
            counter[str(value)] += 1
    return counter


def filter_rare_brand_entities(
    entities: List[MutableMapping[str, object]],
    min_count: int = 3,
    logger: Optional[logging.Logger] = None,
) -> List[MutableMapping[str, object]]:
    """
    Remove entities whose normalized_brand frequency is below min_count.
    Entities without a brand are kept (to avoid dropping UNKNOWN buckets).
    """
    brand_counts = collect_field_counts(entities, "brand")
    kept: List[MutableMapping[str, object]] = []
    dropped = 0
    for e in entities:
        brand = e.get("normalized_brand", "")
        if brand == "A品牌":
            dropped += 1
            continue
        if brand and brand_counts.get(brand, 0) < min_count:
            dropped += 1
            continue
        kept.append(e)
    if logger:
        logger.info("Filtered rare brands (<%s): dropped %s, kept %s", min_count, dropped, len(kept))
    return kept


def prepare_values_for_clustering(
    counts: Counter, min_len: int
) -> Tuple[List[str], Dict[str, str]]:
    cluster_values: List[str] = []
    passthrough: Dict[str, str] = {}
    for value in counts:
        if not value or value.upper() == "UNKNOWN" or len(value) < min_len:
            passthrough[value] = value
        else:
            cluster_values.append(value)
    cluster_values.sort()
    return cluster_values, passthrough


def embed_values(
    model: SentenceTransformer, values: Sequence[str], batch_size: int
) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype="float32")
    embeddings = model.encode(
        list(values),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_clusters(
    values: Sequence[str],
    counts: Mapping[str, int],
    embeddings: np.ndarray,
    top_k: int,
    embed_threshold: float,
    fuzzy_threshold: int,
    field: str = "",
    require_both: bool = False,
    min_token_overlap: int = 0,
    require_substring: bool = False,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if not values:
        return {}, {}

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    neighbors = min(top_k, len(values))
    sims, idxs = index.search(embeddings, neighbors)

    uf = UnionFind(len(values))
    def tokenize(text: str) -> List[str]:
        tokens = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", text).lower().split()
        return [t for t in tokens if t]

    token_cache = [tokenize(v) for v in values]

    for i in range(len(values)):
        for sim, j in zip(sims[i][1:], idxs[i][1:]):
            if j < 0:
                continue
            ratio = fuzz.token_sort_ratio(values[i], values[j])
            if require_both:
                similar = sim >= embed_threshold and ratio >= fuzzy_threshold
            else:
                similar = sim >= embed_threshold or ratio >= fuzzy_threshold

            if not similar:
                continue

            overlap = len(set(token_cache[i]) & set(token_cache[j]))
            if overlap < min_token_overlap:
                continue

            if require_substring:
                a = values[i].lower().replace(" ", "")
                b = values[j].lower().replace(" ", "")
                if a not in b and b not in a:
                    continue

            uf.union(i, j)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(values)):
        clusters[uf.find(idx)].append(idx)

    mapping: Dict[str, str] = {}
    cluster_members: Dict[str, List[str]] = {}
    for cid, member_idx in clusters.items():
        sorted_members = sorted(
            member_idx,
            key=lambda i: (-counts[values[i]], -len(values[i]), values[i]),
        )
        canonical = values[sorted_members[0]]
        members = [values[i] for i in sorted_members]
        cluster_members[canonical] = members
        for idx in member_idx:
            mapping[values[idx]] = canonical
    return mapping, cluster_members


def resolve_field(
    field: str, counts: Counter, config: ResolveConfig, model: SentenceTransformer
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    overrides = (config.field_overrides or {}).get(field, {})
    embed_threshold = overrides.get("embed_threshold", config.embed_threshold)
    fuzzy_threshold = overrides.get("fuzzy_threshold", config.fuzzy_threshold)
    min_value_len = overrides.get("min_value_len", config.min_value_len)
    require_both = overrides.get("require_both", False)
    min_token_overlap = overrides.get("min_token_overlap", 0)
    require_substring = overrides.get("require_substring", False)

    values, passthrough = prepare_values_for_clustering(counts, min_value_len)
    embeddings = embed_values(model, values, config.batch_size)
    mapping, cluster_members = build_clusters(
        values,
        counts,
        embeddings,
        top_k=overrides.get("top_k", config.top_k),
        embed_threshold=embed_threshold,
        fuzzy_threshold=fuzzy_threshold,
        field=field,
        require_both=require_both,
        min_token_overlap=min_token_overlap,
        require_substring=require_substring,
    )
    mapping.update(passthrough)
    for value in passthrough:
        cluster_members.setdefault(value, [value])
    return mapping, cluster_members


def apply_mappings(
    entities: Iterable[MutableMapping[str, object]],
    field_mappings: Mapping[str, Mapping[str, str]],
) -> List[MutableMapping[str, object]]:
    output: List[MutableMapping[str, object]] = []
    for entity in entities:
        updated = dict(entity)
        for field, mapping in field_mappings.items():
            key = f"normalized_{field}"
            normalized = updated.get(key, "")
            standard = mapping.get(normalized, normalized)
            updated[f"standard_{field}"] = standard
        output.append(updated)
    return output


def save_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_cluster_report(path: Path, clusters: Mapping[str, Mapping[str, List[str]]]) -> None:
    report = {}
    for field, mapping in clusters.items():
        report[field] = [
            {"canonical": canonical, "aliases": aliases}
            for canonical, aliases in sorted(mapping.items(), key=lambda x: x[0])
        ]
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    args: argparse.Namespace, config: ResolveConfig, logger: logging.Logger
) -> Dict[str, object]:
    """
    Execute full pipeline and return stage outputs for flexible dumping.
    """
    logger.info("Loading data from %s", args.input)
    raw_entities = load_entities(args.input, limit=args.limit)
    logger.info("Loaded %d records", len(raw_entities))

    logger.info("Cleaning fields")
    cleaned_entities = clean_entities(raw_entities)

    filtered_entities = filter_rare_brand_entities(
        cleaned_entities, min_count=args.min_brand_count, logger=logger
    )

    logger.info("Building embedding model %s", config.model_name)
    model = SentenceTransformer(config.model_name, device=config.device)

    field_mappings: Dict[str, Dict[str, str]] = {}
    cluster_report: Dict[str, Dict[str, List[str]]] = {}

    for field in TARGET_FIELDS:
        counts = collect_field_counts(filtered_entities, field)
        logger.info("Field %s: %d unique values", field, len(counts))
        mapping, clusters = resolve_field(field, counts, config, model)
        field_mappings[field] = mapping
        cluster_report[field] = clusters
        logger.info("Field %s: %d clusters", field, len(clusters))

    logger.info("Applying standardized values")
    standardized_entities = apply_mappings(filtered_entities, field_mappings)

    return {
        "raw": raw_entities,
        "cleaned": cleaned_entities,
        "filtered": filtered_entities,
        "standardized": standardized_entities,
        "cluster_report": cluster_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and resolve HVAC entities.")
    parser.add_argument("--input", type=Path, default=Path("prepare/fused_entities_all.json"))
    parser.add_argument("--output", type=Path, default=Path("prepare/cleaned_entities.jsonl"))
    parser.add_argument(
        "--cluster-report",
        type=Path,
        default=Path("prepare/cluster_report.json"),
        help="Where to store cluster membership per field.",
    )
    parser.add_argument("--model", type=str, default=ResolveConfig.model_name)
    parser.add_argument("--device", type=str, default=None, help="Set to cuda or cpu to override auto device.")
    parser.add_argument("--batch-size", type=int, default=ResolveConfig.batch_size)
    parser.add_argument("--top-k", type=int, default=ResolveConfig.top_k)
    parser.add_argument("--embed-threshold", type=float, default=ResolveConfig.embed_threshold)
    parser.add_argument("--fuzzy-threshold", type=int, default=ResolveConfig.fuzzy_threshold)
    parser.add_argument("--min-value-len", type=int, default=ResolveConfig.min_value_len)
    parser.add_argument("--min-brand-count", type=int, default=10, help="Drop brands with support below this count.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record cap for quick tests.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--precision-mode",
        action="store_true",
        help="Use stricter thresholds and lexical gates to avoid over-merging (slower, more precise).",
    )
    parser.add_argument("--dump-raw", type=Path, default=None, help="Dump raw flattened entities to JSONL.")
    parser.add_argument("--dump-cleaned", type=Path, default=None, help="Dump cleaned entities to JSONL.")
    parser.add_argument("--dump-filtered", type=Path, default=None, help="Dump filtered entities to JSONL.")
    parser.add_argument("--dump-standardized", type=Path, default=None, help="Dump standardized entities to JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Default conservative overrides
    base_overrides = {
        "brand": {
            "embed_threshold": max(args.embed_threshold, 0.88),
            "fuzzy_threshold": max(args.fuzzy_threshold, 90),
            "require_both": True,
            "min_value_len": max(args.min_value_len, 2),
            "min_token_overlap": 0,
        },
        "category": {
            "embed_threshold": max(args.embed_threshold, 0.9),
            "fuzzy_threshold": max(args.fuzzy_threshold, 92),
            "require_both": True,
            "min_value_len": max(args.min_value_len, 2),
            "min_token_overlap": 0,
        },
        "manufacturer": {
            "embed_threshold": max(args.embed_threshold, 0.9),
            "fuzzy_threshold": max(args.fuzzy_threshold, 92),
            "require_both": True,
            "min_value_len": max(args.min_value_len, 4),
            "min_token_overlap": 0,
        },
    }

    # Precision mode: tighter thresholds + lexical gates, slower but safer.
    if args.precision_mode:
        args.top_k = min(args.top_k, 10)  # shrink neighborhood for stricter blocking
        base_overrides["brand"].update(
            {
                "embed_threshold": max(base_overrides["brand"]["embed_threshold"], 0.92),
                "fuzzy_threshold": max(base_overrides["brand"]["fuzzy_threshold"], 94),
                "min_token_overlap": 1,
                "require_substring": False,
            }
        )
        base_overrides["category"].update(
            {
                "embed_threshold": max(base_overrides["category"]["embed_threshold"], 0.93),
                "fuzzy_threshold": max(base_overrides["category"]["fuzzy_threshold"], 95),
                "min_token_overlap": 1,
                "require_substring": False,
            }
        )
        base_overrides["manufacturer"].update(
            {
                "embed_threshold": max(base_overrides["manufacturer"]["embed_threshold"], 0.94),
                "fuzzy_threshold": max(base_overrides["manufacturer"]["fuzzy_threshold"], 96),
                "min_token_overlap": 1,
                "require_substring": False,
            }
        )

    config = ResolveConfig(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        embed_threshold=args.embed_threshold,
        fuzzy_threshold=args.fuzzy_threshold,
        min_value_len=args.min_value_len,
        field_overrides=base_overrides,
    )

    stages = run_pipeline(args, config, logger)

    if args.dump_raw:
        logger.info("Dumping raw entities to %s", args.dump_raw)
        save_jsonl(args.dump_raw, stages["raw"])
    if args.dump_cleaned:
        logger.info("Dumping cleaned entities to %s", args.dump_cleaned)
        save_jsonl(args.dump_cleaned, stages["cleaned"])
    if args.dump_filtered:
        logger.info("Dumping filtered entities to %s", args.dump_filtered)
        save_jsonl(args.dump_filtered, stages["filtered"])
    if args.dump_standardized:
        logger.info("Dumping standardized entities to %s", args.dump_standardized)
        save_jsonl(args.dump_standardized, stages["standardized"])

    logger.info("Writing cleaned entities to %s", args.output)
    save_jsonl(args.output, stages["standardized"])
    logger.info("Writing cluster report to %s", args.cluster_report)
    save_cluster_report(args.cluster_report, stages["cluster_report"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
