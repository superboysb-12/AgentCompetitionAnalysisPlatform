from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase


logger = logging.getLogger("graph_importer")


def load_settings_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    settings_path = repo_root / "backend_v2" / "settings.py"
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.py not found: {settings_path}")

    spec = importlib.util.spec_from_file_location("backend_v2_settings", settings_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load settings from: {settings_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json_as_dict(json_path: Path, encoding: str) -> Any:
    with json_path.open("r", encoding=encoding) as handle:
        content = handle.read()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback for JSON Lines (one JSON object per line) produced by the cleaner.
        records: List[Any] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive log
                logger.error("Failed to parse line in JSONL: %s", exc)
                raise
        return records


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    value = str(value).strip()
    return value or None


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    cleaned: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            item = item.strip()
            if item:
                cleaned.append(item)
            continue
        cleaned.append(str(item))
    return cleaned


def _compact_props(props: Dict[str, Any]) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, dict) and not value:
            continue
        compacted[key] = value
    return compacted


def iter_product_records(data: Any) -> Iterable[Tuple[Dict[str, Any], Dict[str, int]]]:
    if isinstance(data, list):
        for group_index, item in enumerate(data):
            if isinstance(item, dict) and isinstance(item.get("results"), list):
                for result_index, result in enumerate(item["results"]):
                    if isinstance(result, dict):
                        yield result, {"group_index": group_index, "result_index": result_index}
            elif isinstance(item, dict):
                yield item, {"group_index": group_index, "result_index": 0}
    elif isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            for result_index, result in enumerate(results):
                if isinstance(result, dict):
                    yield result, {"group_index": 0, "result_index": result_index}
        else:
            yield data, {"group_index": 0, "result_index": 0}


def build_product_id(record: Dict[str, Any], meta: Dict[str, int], index: int) -> str:
    brand = _as_text(record.get("brand"))
    product_model = _as_text(record.get("product_model"))
    series = _as_text(record.get("series"))
    category = _as_text(record.get("category"))

    if brand and product_model:
        return f"{brand}::{product_model}"
    if product_model:
        return product_model
    if brand and series:
        return f"{brand}::{series}"
    if brand and category:
        return f"{brand}::{category}"
    return f"record_{meta['group_index']}_{meta['result_index']}_{index}"


def normalize_specs(raw_specs: Any) -> List[Dict[str, Optional[str]]]:
    if not isinstance(raw_specs, list):
        return []

    specs: List[Dict[str, Optional[str]]] = []
    for item in raw_specs:
        if not isinstance(item, dict):
            continue
        name = _as_text(item.get("name"))
        value = _as_text(item.get("value"))
        unit = _as_text(item.get("unit"))
        raw = _as_text(item.get("raw"))
        if not name:
            continue
        specs.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "raw": raw,
            }
        )
    return specs


def create_constraints(session, labels: Dict[str, str]) -> None:
    product_label = labels.get("product")
    brand_label = labels.get("brand")
    manufacturer_label = labels.get("manufacturer")
    category_label = labels.get("category")
    series_label = labels.get("series")

    if product_label:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (p:{product_label}) REQUIRE p.product_id IS UNIQUE"
        )
    if brand_label:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{brand_label}) REQUIRE b.name IS UNIQUE"
        )
    if manufacturer_label:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (m:{manufacturer_label}) REQUIRE m.name IS UNIQUE"
        )
    if category_label:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{category_label}) REQUIRE c.name IS UNIQUE"
        )
    if series_label:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (s:{series_label}) REQUIRE s.name IS UNIQUE"
        )


def clear_database(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def import_record_tx(
    tx,
    record: Dict[str, Any],
    meta: Dict[str, int],
    index: int,
    schema: Dict[str, Any],
) -> None:
    labels = schema["labels"]
    relationships = schema["relationships"]

    product_id = build_product_id(record, meta, index)
    product_model = _as_text(record.get("product_model"))
    product_name = product_model or product_id
    specs = normalize_specs(record.get("performance_specs"))
    specs_json = json.dumps(specs, ensure_ascii=False) if specs else None

    product_props = _compact_props(
        {
            "name": product_name,
            "product_id": product_id,
            "product_model": product_model,
            "brand": _as_text(record.get("brand")),
            "category": _as_text(record.get("category")),
            "series": _as_text(record.get("series")),
            "manufacturer": _as_text(record.get("manufacturer")),
            "refrigerant": _as_text(record.get("refrigerant")),
            "energy_efficiency_grade": _as_text(record.get("energy_efficiency_grade")),
            "features": _as_list(record.get("features")),
            "key_components": _as_list(record.get("key_components")),
            "performance_specs": specs_json,
            "fact_text": _as_list(record.get("fact_text")),
        }
    )

    product_query = f"""
    MERGE (p:{labels['product']} {{product_id: $product_id}})
    SET p += $props
    """
    tx.run(product_query, product_id=product_id, props=product_props)

    brand = _as_text(record.get("brand"))
    if brand and labels.get("brand"):
        brand_query = f"""
        MERGE (p:{labels['product']} {{product_id: $product_id}})
        MERGE (b:{labels['brand']} {{name: $brand}})
        MERGE (p)-[:{relationships['brand']}]->(b)
        """
        tx.run(brand_query, product_id=product_id, brand=brand)

    manufacturer = _as_text(record.get("manufacturer"))
    if manufacturer and labels.get("manufacturer"):
        manufacturer_query = f"""
        MERGE (p:{labels['product']} {{product_id: $product_id}})
        MERGE (m:{labels['manufacturer']} {{name: $manufacturer}})
        MERGE (p)-[:{relationships['manufacturer']}]->(m)
        """
        tx.run(manufacturer_query, product_id=product_id, manufacturer=manufacturer)

    category = _as_text(record.get("category"))
    if category and labels.get("category"):
        category_query = f"""
        MERGE (p:{labels['product']} {{product_id: $product_id}})
        MERGE (c:{labels['category']} {{name: $category}})
        MERGE (p)-[:{relationships['category']}]->(c)
        """
        tx.run(category_query, product_id=product_id, category=category)

    series = _as_text(record.get("series"))
    if series and labels.get("series"):
        series_query = f"""
        MERGE (p:{labels['product']} {{product_id: $product_id}})
        MERGE (s:{labels['series']} {{name: $series}})
        MERGE (p)-[:{relationships['series']}]->(s)
        """
        tx.run(series_query, product_id=product_id, series=series)


def import_batch_tx(
    tx,
    batch: List[Tuple[Dict[str, Any], Dict[str, int], int]],
    schema: Dict[str, Any],
) -> None:
    for record, meta, index in batch:
        import_record_tx(tx, record, meta, index, schema)


def import_dict_to_graph(data: Any, config: Dict[str, Any]) -> int:
    schema = config["schema"]
    neo4j_conf = config["neo4j"]
    batch_size = max(1, int(config.get("batch_size", 100)))

    driver = GraphDatabase.driver(
        neo4j_conf["uri"],
        auth=(neo4j_conf["user"], neo4j_conf["password"]),
    )

    total = 0
    with driver:
        with driver.session(database=neo4j_conf.get("database")) as session:
            if config.get("clear_before_import", False):
                clear_database(session)
            if config.get("create_constraints", True):
                create_constraints(session, schema["labels"])

            buffer: List[Tuple[Dict[str, Any], Dict[str, int], int]] = []
            for index, (record, meta) in enumerate(iter_product_records(data)):
                buffer.append((record, meta, index))
                if len(buffer) >= batch_size:
                    session.execute_write(import_batch_tx, buffer, schema)
                    total += len(buffer)
                    buffer.clear()
                    if total % 100 == 0:
                        logger.info("Imported %s records", total)
            if buffer:
                session.execute_write(import_batch_tx, buffer, schema)
                total += len(buffer)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Import product JSON into Neo4j.")
    parser.add_argument("--json", dest="json_path", default=None, help="Override JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not import")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of records per transaction batch")
    parser.add_argument(
        "--clear-before-import",
        action="store_true",
        help="Clear database before import (DETACH DELETE all nodes/relationships).",
    )
    args = parser.parse_args()

    settings = load_settings_module()
    if not hasattr(settings, "GRAPH_IMPORTER_CONFIG"):
        raise RuntimeError("GRAPH_IMPORTER_CONFIG not found in backend_v2/settings.py")

    config = settings.GRAPH_IMPORTER_CONFIG
    if args.json_path:
        config["json_path"] = args.json_path
    if args.dry_run:
        config["dry_run"] = True
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.clear_before_import:
        config["clear_before_import"] = True

    json_path = Path(config["json_path"])
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = load_json_as_dict(json_path, config.get("json_encoding", "utf-8"))
    if config.get("dry_run", False):
        count = sum(1 for _ in iter_product_records(data))
        logger.info("Dry run OK, records parsed: %s", count)
        return

    total = import_dict_to_graph(data, config)
    logger.info("Import complete, total records: %s", total)


if __name__ == "__main__":
    main()
