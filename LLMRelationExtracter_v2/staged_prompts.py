"""
Prompt builders and JSON schemas for the staged extractor.
"""

from typing import Dict, List, Optional

from LLMRelationExtracter.relation_extractor import RELATION_SCHEMA

# --- JSON Schemas ---------------------------------------------------------

# Brand schema: high-recall, lightweight
BRAND_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brands": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "evidence", "pages"],
            },
        }
    },
    "required": ["brands"],
}

# Series schema: conditioned on a brand
SERIES_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand": {"type": "string"},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "evidence", "pages"],
            },
        },
    },
    "required": ["brand", "series"],
}

# Product schema: reuse v1 relation schema to stay compatible with downstream
PRODUCT_SCHEMA: Dict = RELATION_SCHEMA
MODEL_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand": {"type": "string"},
        "series": {"type": "string"},
        "models": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "evidence", "pages"],
            },
        },
    },
    "required": ["brand", "series", "models"],
}
SERIES_FILTER_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}},
        "drop": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keep"],
}
SERIES_REVIEW_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "original": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "kind": {"type": "string", "enum": ["series", "non_series"]},
                    "canonical": {"type": "string"},
                },
                "required": ["original", "keep", "kind", "canonical"],
            },
        }
    },
    "required": ["items"],
}
BRAND_FILTER_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"keep": {"type": "array", "items": {"type": "string"}}},
    "required": ["keep"],
}
BRAND_CANON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "original": {"type": "string"},
                    "canonical_cn": {"type": "string"},
                },
                "required": ["original", "canonical_cn"],
            },
        }
    },
    "required": ["items"],
}


# --- Prompt helpers -------------------------------------------------------

def build_brand_messages(text: str, page_label: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a fast brand spotter for HVAC documents. "
                "Only output manufacturer/brand names (company or trademark). "
                "Do NOT output product series/lines/models such as GMV ES / GMV9 Flex / Ultra Heat / X-COOLING / Free / Mini-Ultra. "
                "If both Chinese and English names of the same brand appear, keep both entries is fine; never invent new ones. "
                "Return JSON per schema; if no brand is present, return an empty brands array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Page: {page_label}\n"
                "Extract brand names and evidence spans (short text snippets). "
                "Skip series/model-only mentions (e.g., GMV ES belongs to brand Gree/格力, but GMV ES itself is not a brand). "
                "If a logo/brand is only implied, skip it.\n\n"
                f"{text}"
            ),
        },
    ]


def build_series_messages(brand: str, combined_text: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are extracting product series for a given brand. "
                "Work only within the provided text. "
                "Return JSON with the exact brand echo and all series/line names plus evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand}\n"
                "Find series / line / sub-brand names under this brand. "
                "Include evidence text snippets. rows_json blocks contain raw CSV rows with table_data/bbox; you may cite them.\n"
                "If not found, return an empty array.\n\n"
                f"{combined_text}"
            ),
        },
    ]


def build_series_filter_messages(brand: str, candidates: List[Dict]) -> List[Dict]:
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(f"{idx}. name={item.get('name','')} | evidence={evid}")
    return [
        {
            "role": "system",
            "content": (
                "You are filtering series names for an HVAC brand. "
                "Keep only series/line/sub-brand names (e.g., GMV9, CoolAny, Free Match, Ultra Heat). "
                "Drop model numbers (contain full numeric/tonnage like 36K/280WM/S), single SKUs, parameter rows, or generic words. "
                "Do not invent names. Output keep/drop lists."
            ),
        },
        {
            "role": "user",
            "content": f"Brand: {brand}\nCandidates:\n" + "\n".join(lines),
        },
    ]


def build_series_review_messages(brand: str, candidates: List[Dict]) -> List[Dict]:
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(
            f"{idx}. name={item.get('name','')} | pages={item.get('pages',0)} | evidence={evid}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are reviewing Stage-B series candidates for an HVAC brand. "
                "For each candidate, classify whether it is a true product series/line name under this brand. "
                "Mark non-series for: single model/SKU, product type words, features/technologies, systems/platforms, "
                "projects/awards, unrelated categories (e.g., fridge/washing machine/water purifier/robot). "
                "When keep=true, output a short canonical series name without extra descriptors like "
                "'直流变频多联空调机组', capacity ranges, or long explanatory phrases. "
                "Do not invent names and do not merge across unrelated families."
            ),
        },
        {
            "role": "user",
            "content": f"Brand: {brand}\nCandidates:\n" + "\n".join(lines),
        },
    ]


def build_model_messages(brand: str, series: str, text_block: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are extracting HVAC product model names under a given brand/series. "
                "Output model identifiers (e.g., GMV-ND18PS/C, KFR-35GW/...) only. "
                "Do not output series names, technologies, categories, or generic words. "
                "Preserve exact model spelling from the text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand or '<unknown>'}\n"
                f"Series: {series or '<unknown>'}\n"
                "Extract model names with evidence snippets and pages. "
                "If none found, return empty models array.\n\n"
                f"{text_block}"
            ),
        },
    ]


def build_product_messages(
    brand: str,
    series: str,
    text_block: str,
    known_models: Optional[List[str]] = None,
) -> List[Dict]:
    model_hint = ""
    if known_models:
        model_hint = (
            "Known models for this pair (focus extraction on these and their nearby table rows):\n"
            + ", ".join(known_models[:50])
            + "\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a structured extractor focused on HVAC / air-conditioning products (VRF, chiller, AHU, FCU, packaged, split, heat pump, rooftop). "
                "Return products under the given brand/series using the strict JSON schema. "
                "Rules:\n"
                "- One row/line/model -> one product. Do NOT merge different models.\n"
                "- Capture refrigerant (R32/R410A/R22/etc.), energy efficiency metrics (SEER/EER/COP/IPLV/IEER/APF/HSPF), cooling/heating capacity (kW, RT/ton, BTU/h), airflow (m³/h), voltage/phase, indoor/outdoor unit model pairs.\n"
                "- If indoor+outdoor are paired, include both models in fact_text/evidence; use outdoor as product_model when unclear which is primary.\n"
                "- Preserve table row values; keep units; put any unmatched specs into performance_specs with raw text.\n"
                "- rows_json blocks contain raw CSV rows with table_data/bbox; use them to bind specs to the correct model row.\n"
                "- Only use information present in text; leave missing fields empty.\n"
                "- Fill brand/series with provided values when applicable."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand or '<unknown>'}\n"
                f"Series: {series or '<unknown>'}\n"
                f"{model_hint}"
                "Extract product models and fields. "
                "If multiple rows exist, output one result per model. "
                "Evidence should include short text fragments or table rows (you can quote rows_json entries).\n\n"
                f"{text_block}"
            ),
        },
    ]


def build_brand_filter_messages(candidates: List[Dict]) -> List[Dict]:
    """
    candidates: list of {"name": str, "pages": int, "evidence": [str]}
    """
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(
            f"{idx}. name={item.get('name','')} | pages={item.get('pages',0)} | evidence={evid}"
        )
    user_block = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": (
                "You are pruning brand candidates for an HVAC document. "
                "Only keep real brand names; drop model lists, series names, departments, research institutes, distributors, or unrelated companies. "
                "If multiple variants represent the same brand, keep ONE most standard/short form. "
                "Return JSON with a 'keep' array (0-1 items)."
            ),
        },
        {
            "role": "user",
            "content": (
                "Select at most ONE primary brand to keep from the following candidates in the same cluster. "
                "Prefer the canonical brand form; if none look like a brand, return an empty list.\n\n"
                f"{user_block}"
            ),
        },
    ]


def build_brand_global_filter_messages(candidates: List[Dict]) -> List[Dict]:
    """
    Global pass to drop non-brand items (series/model/org) from the candidate list.
    """
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(
            f"{idx}. name={item.get('name','')} | pages={item.get('pages',0)} | evidence={evid}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You receive candidate names from HVAC documents. "
                "Keep only true brands/manufacturers/trademarks. "
                "Drop: series/line/model tokens (e.g., GMV, GMV ES, MDV, VRV, MRV, VRF, Ultra Heat, Free, X-COOLING), "
                "institutes, departments, distributors, media, standards bodies. "
                "If uncertain, prefer dropping to keep precision. "
                "Return JSON with a 'keep' array of brand names to keep (can be empty)."
            ),
        },
        {
            "role": "user",
            "content": (
                "From the list below, choose the names that are actual brands/manufacturers/trademarks. "
                "If none qualify, return an empty array.\n\n"
                + "\n".join(lines)
            ),
        },
    ]


def build_brand_canon_messages(candidates: List[Dict]) -> List[Dict]:
    """
    Canonicalize brand names to concise Chinese form.
    """
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(f"{idx}. {item.get('name','')} | evidence={evid}")
    return [
        {
            "role": "system",
            "content": (
                "You are normalizing brand names for HVAC documents. "
                "For each input brand, output a concise Chinese canonical brand name. "
                "Rules:\n"
                "- If the brand is already Chinese, keep or shorten to its common brand form (e.g., “格力电器” -> “格力”).\n"
                "- If English brand has a well-known Chinese name, translate to that (e.g., Gree->格力, Midea->美的, Haier->海尔, Daikin->大金, Hitachi->日立, Panasonic->松下, Hisense->海信, Aux->奥克斯, TCL->TCL, Carrier->开利).\n"
                "- If no reliable Chinese name is known, keep the original spelling.\n"
                "- Do NOT create new brands beyond inputs; one output per input.\n"
                "Return JSON with an 'items' array of {original, canonical_cn}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Normalize each brand to Chinese canonical form. One output per input.\n\n"
                + "\n".join(lines)
            ),
        },
    ]
