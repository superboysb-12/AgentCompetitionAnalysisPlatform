"""
Relation extractor - extract structured product relations from plain text.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from backend_v2.settings import RELATION_EXTRACTOR_CONFIG

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
        self.client = AsyncOpenAI(
            api_key=RELATION_EXTRACTOR_CONFIG["api_key"],
            base_url=RELATION_EXTRACTOR_CONFIG["base_url"],
        )
        self.model = RELATION_EXTRACTOR_CONFIG["model"]
        self.semaphore = asyncio.Semaphore(RELATION_EXTRACTOR_CONFIG["max_concurrent"])
        self.logger = self._setup_logging()
        self.system_prompt = RELATION_EXTRACTOR_CONFIG.get("system_prompt", "")
        self.performance_units = RELATION_EXTRACTOR_CONFIG.get(
            "performance_param_units", {}
        )

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
            for attempt in range(RELATION_EXTRACTOR_CONFIG["max_retries"]):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
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
                                    "Extract product relations from the text. "
                                    "If multiple product models are present (e.g., table rows), output one result per model; "
                                    "do NOT merge specs across models.\n\n"
                                    f"{text}"
                                ),
                            },
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "relation_schema",
                                "schema": RELATION_SCHEMA,
                                "strict": True,
                            },
                        },
                        timeout=RELATION_EXTRACTOR_CONFIG["timeout"],
                    )

                    choices = getattr(response, "choices", None) or []
                    if not choices:
                        raise ValueError(f"Empty choices from model: {response}")

                    message = getattr(choices[0], "message", None)
                    content = getattr(message, "content", None)
                    if not content:
                        raise ValueError(f"Empty content from model: {response}")

                    data = json.loads(content)
                    results = data.get("results", [])
                    processed = [self._post_process_item(item) for item in results]
                    return processed

                except Exception as exc:  # noqa: BLE001
                    preview = (text or "")[:200]
                    self.logger.error(
                        "Extraction failed (attempt %s/%s): %s",
                        attempt + 1,
                        RELATION_EXTRACTOR_CONFIG["max_retries"],
                        exc,
                        exc_info=True,
                    )
                    self.logger.error("Input text preview: %s...", preview)
                    if metadata:
                        self.logger.error("Metadata: %s", metadata)

                    if attempt == RELATION_EXTRACTOR_CONFIG["max_retries"] - 1:
                        raise

                    await asyncio.sleep(RELATION_EXTRACTOR_CONFIG["retry_delay"])

        return []

    def _post_process_item(self, item: Dict) -> Dict:
        """Normalize fields, enforce units, and ensure required keys exist."""

        def as_str(value: Optional[str]) -> str:
            return "" if value is None else str(value).strip()

        def as_list_str(values: Optional[List]) -> List[str]:
            return [as_str(v) for v in values or [] if as_str(v)]

        result: Dict[str, object] = {
            "brand": as_str(item.get("brand")),
            "category": as_str(item.get("category")),
            "series": as_str(item.get("series")),
            "product_model": as_str(item.get("product_model")),
            "manufacturer": as_str(item.get("manufacturer")),
            "refrigerant": as_str(item.get("refrigerant")),
            "energy_efficiency_grade": as_str(item.get("energy_efficiency_grade")),
            "features": as_list_str(item.get("features")),
            "key_components": as_list_str(item.get("key_components")),
            "performance_specs": self._filter_performance_specs(
                item.get("performance_specs") or []
            ),
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

            filtered.append(
                {
                    "name": name,
                    "value": str(spec.get("value") or "").strip(),
                    "unit": unit or expected_unit,
                    "raw": raw,
                }
            )

        return filtered
