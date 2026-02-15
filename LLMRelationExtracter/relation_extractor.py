"""
Relation extractor - extract structured product relations from plain text.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.settings import RELATION_EXTRACTOR_CONFIG
from LLMRelationExtracter.llm_client import (
    LangChainJsonLLMClient,
    get_shared_chat_openai,
)

# JSON schema for strict response validation (simplified, aligns with target table)
RELATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "brand": {"type": "string"},
                    "category": {"type": "string"},
                    "series": {"type": "string"},
                    "product_model": {"type": "string"},
                    "manufacturer": {"type": "string"},
                    "refrigerant": {"type": "string"},
                    "energy_efficiency_grade": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "key_components": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "performance_specs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                                "unit": {"type": "string"},
                                "raw": {"type": "string"},
                            },
                            # OpenAI strict schema requires required to include every property key
                            "required": ["name", "value", "unit", "raw"],
                        },
                    },
                    "fact_text": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "brand",
                    "category",
                    "series",
                    "product_model",
                    "manufacturer",
                    "refrigerant",
                    "energy_efficiency_grade",
                    "features",
                    "key_components",
                    "performance_specs",
                    "fact_text",
                    "evidence",
                ],
            },
        }
    },
    "required": ["results"],
}


class RelationExtractor:
    def __init__(self) -> None:
        self.logger = self._setup_logging()
        self.llm = get_shared_chat_openai(
            api_key=RELATION_EXTRACTOR_CONFIG["api_key"],
            base_url=RELATION_EXTRACTOR_CONFIG["base_url"],
            model=RELATION_EXTRACTOR_CONFIG["model"],
            temperature=0,
            timeout=RELATION_EXTRACTOR_CONFIG["timeout"],
        )
        self.semaphore = asyncio.Semaphore(RELATION_EXTRACTOR_CONFIG["max_concurrent"])
        self.llm_client = LangChainJsonLLMClient(
            self.llm,
            logger=self.logger,
            max_retries=int(RELATION_EXTRACTOR_CONFIG.get("max_retries", 8)),
            retry_delay=float(RELATION_EXTRACTOR_CONFIG.get("retry_delay", 2.0)),
            retry_backoff_factor=float(
                RELATION_EXTRACTOR_CONFIG.get("retry_backoff_factor", 2.5)
            ),
            retry_max_delay=float(RELATION_EXTRACTOR_CONFIG.get("retry_max_delay", 120.0)),
            hard_timeout=float(RELATION_EXTRACTOR_CONFIG.get("llm_call_hard_timeout", 180.0)),
            llm_global_concurrency=int(
                RELATION_EXTRACTOR_CONFIG.get("llm_global_concurrency", 10)
            ),
            print_call_counter=False,
            recycle_on_connection_error=bool(
                RELATION_EXTRACTOR_CONFIG.get(
                    "llm_socket_recycle_on_connection_error",
                    True,
                )
            ),
            recycle_after_calls=int(
                RELATION_EXTRACTOR_CONFIG.get(
                    "llm_socket_recycle_after_calls",
                    0,
                )
            ),
            recycle_min_interval=float(
                RELATION_EXTRACTOR_CONFIG.get(
                    "llm_socket_recycle_min_interval",
                    5.0,
                )
            ),
        )
        self.system_prompt = RELATION_EXTRACTOR_CONFIG.get("system_prompt", "")
        self.performance_units = RELATION_EXTRACTOR_CONFIG.get(
            "performance_param_units", {}
        )
        self._unit_tokens = self._build_unit_tokens(self.performance_units)
        self._phi_symbols = ("\u03a6", "\u03c6", "\u00d8", "\u00f8")

    def _setup_logging(self) -> logging.Logger:
        """Set up file logger and ensure log directory exists."""
        log_path = Path(RELATION_EXTRACTOR_CONFIG["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        level = getattr(
            logging,
            str(RELATION_EXTRACTOR_CONFIG["log_level"]).upper(),
            logging.INFO,
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    async def extract_relations_async(
        self, text: str, metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Extract product relations from text and return a list of relation objects.
        """
        async with self.semaphore:
            try:
                payload = await self.llm_client.call_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                self.system_prompt
                                or "You are a product information extraction expert. Only extract fields with evidence; if absent leave empty strings or empty arrays."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Extract product relations from the text below. "
                                "IMPORTANT INSTRUCTIONS:\n"
                                "- If you see multiple pages in the context, FOCUS ONLY on the page marked as 'CURRENT PAGE'\n"
                                "- Use context from other pages only to complete information for products on the current page\n"
                                "- If a table spans multiple pages, merge the information for products on the current page\n"
                                "- If multiple product models are present (e.g., table rows), output one result per model\n"
                                "- Do NOT merge specs across different models\n"
                                "- Do NOT extract products that only appear in context pages\n\n"
                                f"{text}"
                            ),
                        },
                    ],
                    RELATION_SCHEMA,
                    "relation_schema",
                )
                results = payload.get("results", [])
                processed = [self._post_process_item(item) for item in results]
                return processed
            except Exception as exc:  # noqa: BLE001
                preview = (text or "")[:200]
                self.logger.error(
                    "Extraction failed: %s",
                    exc,
                    exc_info=True,
                )
                self.logger.error("Input text preview: %s...", preview)
                if metadata:
                    self.logger.error("Metadata: %s", metadata)
                raise

    def _post_process_item(self, item: Dict) -> Dict:
        """Normalize fields, enforce units, and ensure required keys exist."""

        def as_str(value: Optional[str]) -> str:
            return "" if value is None else str(value).strip()

        def as_list_str(values: Optional[List]) -> List[str]:
            return [as_str(v) for v in values or [] if as_str(v)]

        features = as_list_str(item.get("features"))
        components = as_list_str(item.get("key_components"))

        features, feature_specs = self._extract_specs_from_lines(features)
        components, component_specs = self._extract_specs_from_lines(components)

        merged_specs = list(item.get("performance_specs") or [])
        merged_specs.extend(feature_specs)
        merged_specs.extend(component_specs)

        result: Dict[str, object] = {
            "brand": as_str(item.get("brand")),
            "category": as_str(item.get("category")),
            "series": as_str(item.get("series")),
            "product_model": as_str(item.get("product_model")),
            "manufacturer": as_str(item.get("manufacturer")),
            "refrigerant": as_str(item.get("refrigerant")),
            "energy_efficiency_grade": as_str(item.get("energy_efficiency_grade")),
            "features": features,
            "key_components": components,
            "performance_specs": self._filter_performance_specs(merged_specs),
            "fact_text": as_list_str(item.get("fact_text")),
            "evidence": as_list_str(item.get("evidence")),
        }
        return result

    def _filter_performance_specs(self, specs: List[Dict]) -> List[Dict]:
        """
        Keep only specs whose expected unit (when provided) appears in raw text.
        Missing name/raw are dropped. If unit is empty in config, allow any.
        """

        filtered: List[Dict] = []
        seen = set()
        for spec in specs:
            name = str(spec.get("name") or "").strip()
            raw = str(spec.get("raw") or "").strip()
            if not name or not raw:
                continue

            expected_unit = str(self.performance_units.get(name, "")).strip()
            unit = str(spec.get("unit") or "").strip()

            # Enforce expected unit presence when specified
            if expected_unit:
                # Require unit to appear in raw text to avoid hallucinated units
                if expected_unit.lower() not in raw.lower():
                    continue

            value = str(spec.get("value") or "").strip()
            unit = unit or expected_unit
            key = (name, value, unit, raw)
            if key in seen:
                continue
            seen.add(key)

            filtered.append(
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "raw": raw,
                }
            )

        return filtered

    def _build_unit_tokens(self, units_map: Dict[str, str]) -> List[str]:
        tokens = {str(unit).strip() for unit in units_map.values() if unit}
        tokens.update(
            {
                "mm",
                "cm",
                "m",
                "m3/h",
                "l/min",
                "ml/min",
                "kg",
                "g",
                "w",
                "kw",
                "a",
                "v",
                "hz",
                "pa",
                "kpa",
                "mpa",
                "%",
                "db",
                "db(a)",
                "rpm",
            }
        )
        return sorted({t for t in tokens if t}, key=len, reverse=True)

    def _extract_specs_from_lines(
        self, items: List[str]
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        remaining: List[str] = []
        specs: List[Dict[str, str]] = []
        for item in items:
            spec = self._parse_param_line(item)
            if spec:
                specs.append(spec)
            else:
                remaining.append(item)
        return remaining, specs

    def _parse_param_line(self, text: str) -> Optional[Dict[str, str]]:
        if not text:
            return None
        if not re.search(r"\d", text):
            return None

        name_raw = ""
        value_raw = ""

        match = re.match(r"^\s*([^:=：]+?)\s*[:=：]\s*(.+)$", text)
        if match:
            name_raw = match.group(1).strip()
            value_raw = match.group(2).strip()
        else:
            compact = self._split_compact_name_value(text)
            if not compact:
                return None
            name_raw, value_raw = compact

        if not name_raw or not value_raw:
            return None

        name, unit_from_name = self._extract_unit_from_name(name_raw)
        value, unit_from_value = self._extract_unit_from_value(value_raw)
        unit = unit_from_value or unit_from_name or ""

        return {
            "name": name or name_raw,
            "value": value,
            "unit": unit,
            "raw": text.strip(),
        }

    def _split_compact_name_value(self, text: str) -> Optional[Tuple[str, str]]:
        text = text.strip()
        if not text:
            return None

        split_index = None
        for index, char in enumerate(text):
            if char.isdigit() or char in self._phi_symbols:
                split_index = index
                break

        if split_index is None:
            return None

        name = text[:split_index].strip()
        value = text[split_index:].strip()
        if not name or not value:
            return None

        if not re.search(r"[A-Za-z\u4e00-\u9fff]", name):
            return None

        return name, value

    def _extract_unit_from_name(self, name: str) -> Tuple[str, str]:
        if not name:
            return "", ""
        unit = ""
        name_clean = name
        for match in re.finditer(r"\(([^()]*)\)", name):
            part = match.group(1).strip()
            part_normalized = part.replace("\u00b3", "3")
            if part_normalized and self._looks_like_unit(part_normalized):
                unit = part_normalized
                name_clean = name_clean.replace(match.group(0), "").strip()
                break

        if not unit:
            for token in self._unit_tokens:
                if name_clean.lower().endswith(token.lower()):
                    unit = token
                    name_clean = name_clean[: -len(token)].strip()
                    break

        return name_clean, unit

    def _extract_unit_from_value(self, value: str) -> Tuple[str, str]:
        if not value:
            return "", ""
        value_clean = value.strip().replace("\u00b3", "3")
        for token in self._unit_tokens:
            pattern = rf"^\s*(.+?)\s*{re.escape(token)}\s*$"
            match = re.match(pattern, value_clean, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip()
            if candidate and re.match(r"^[0-9.\-~xX*/\s]+$", candidate):
                return candidate, token
        return value_clean, ""

    def _looks_like_unit(self, text: str) -> bool:
        lowered = text.lower()
        return any(token.lower() in lowered for token in self._unit_tokens)
