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

# Series schema: document-level extraction, with brand as ownership echo.
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
MODEL_REVIEW_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "kind": {
                        "type": "string",
                        "enum": ["model", "series", "accessory", "other"],
                    },
                    "series_guess": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "redirect_to": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": [
                    "name",
                    "keep",
                    "kind",
                    "series_guess",
                    "redirect_to",
                    "confidence",
                ],
            },
        }
    },
    "required": ["items"],
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
                    "kind": {
                        "type": "string",
                        "enum": ["series", "product_type", "feature", "model_bucket", "other"],
                    },
                    "canonical": {"type": "string"},
                },
                "required": ["original", "keep", "kind", "canonical"],
            },
        }
    },
    "required": ["items"],
}
SERIES_CANON_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "group_key": {"type": "string"},
                    "canonical": {"type": "string"},
                },
                "required": ["group_key", "canonical"],
            },
        }
    },
    "required": ["items"],
}
SERIES_FEATURE_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "brand": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "series": {"type": "string"},
                    "title": {"type": "string"},
                    "fact_text": {"type": "array", "items": {"type": "string"}},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "key_components": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                },
                "required": [
                    "series",
                    "title",
                    "fact_text",
                    "features",
                    "key_components",
                    "evidence",
                    "pages",
                ],
            },
        },
    },
    "required": ["brand", "items"],
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
PRODUCT_REVIEW_SCHEMA: Dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "product_model": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "is_accessory": {"type": "boolean"},
                    "series_feature": {"type": "boolean"},
                    "role": {
                        "type": "string",
                        "enum": ["product", "series_feature", "accessory"],
                    },
                },
                "required": ["product_model", "keep", "is_accessory", "series_feature", "role"],
            },
        }
    },
    "required": ["products"],
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
                "Do NOT treat product line abbreviations as brands (e.g., GMV/VRV/MULTI-V/CITY MULTI are typically series lines). "
                "If both Chinese and English names of the same brand appear, keep both entries is fine; never invent new ones. "
                "Return JSON per schema; if no brand is present, return an empty brands array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Page: {page_label}\n"
                "Extract brand names and evidence spans (short text snippets). "
                "Skip any series/model-only mentions; only keep actual brand/manufacturer names. "
                "If a logo/brand is only implied, skip it.\n\n"
                f"{text}"
            ),
        },
    ]


def build_series_messages(
    brand: str,
    combined_text: str,
    stage_a_pages: Optional[List] = None,
    chunk_pages: Optional[List] = None,
) -> List[Dict]:
    stage_pages = [str(p) for p in (stage_a_pages or []) if p is not None and str(p) != ""][:80]
    chunk_page_labels = [str(p) for p in (chunk_pages or []) if p is not None and str(p) != ""]
    stage_hint = (
        f"Stage-A confirmed brand evidence pages: {', '.join(stage_pages)}\n"
        if stage_pages
        else "Stage-A confirmed brand evidence pages: <not provided>\n"
    )
    chunk_hint = (
        f"Current context chunk pages: {', '.join(chunk_page_labels)}\n"
        if chunk_page_labels
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a fast HVAC product-series spotter for a single-brand document. "
                "Work only within the provided text. "
                "Strict hierarchy separation is mandatory: "
                "brand != series != model != product_type != feature. "
                "Only output true series/line/sub-brand names present in this document. "
                "Do not treat the input brand as a retrieval constraint; it is only an ownership echo field. "
                "If one sentence/table row contains multiple series names, split them into separate series items. "
                "Do not concatenate multiple series into one name. "
                "Only keep HVAC product series. Do not output communication/network/protocol labels "
                "or generic control/system terms or version notes. "
                "Do not output brand names, model/SKU strings, product type/category words, or feature slogans. "
                "Use strict rule-based judgment from the text itself; do not follow example patterns. "
                "Return JSON with the exact brand echo and series names plus evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document primary brand (for output ownership echo only): {brand}\n"
                f"{stage_hint}"
                f"{chunk_hint}"
                "Please perform a document-level fast scan and extract series / line / sub-brand names. "
                "Include evidence text snippets. rows_json blocks contain raw CSV rows with table_data/bbox; you may cite them.\n"
                "If not found, return an empty array.\n\n"
                f"{combined_text}"
            ),
        },
    ]


def build_series_feature_messages(
    brand: str,
    series_names: List[str],
    combined_text: str,
    chunk_pages: Optional[List] = None,
) -> List[Dict]:
    chunk_hint = ""
    if chunk_pages:
        labels = [str(p) for p in chunk_pages if p is not None and str(p) != ""]
        if labels:
            chunk_hint = f"Current context chunk pages: {', '.join(labels)}\n"

    return [
        {
            "role": "system",
            "content": (
                "You are extracting SERIES-LEVEL features/components for HVAC documents. "
                "Do not output model-level specs as series features. "
                "Only output entries that belong to one provided series name. "
                "If content is accessory/controller/filter/optional module information for a series, "
                "put it into key_components. "
                "Prefer concise Chinese key_components terms when available in text. "
                "Do not invent data; return empty items when absent."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand}\n"
                f"Candidate series list (must map items to these names): {', '.join(series_names)}\n"
                f"{chunk_hint}"
                "Extract series-level items with fields: series/title/fact_text/features/key_components/evidence/pages.\n"
                "Rules:\n"
                "- series must be one of provided series names.\n"
                "- key_components should include accessory/controller/component names if present.\n"
                "- product_model rows should not be treated as independent products in this task.\n\n"
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
                "Keep only series/line/sub-brand names. "
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
                "Label each candidate with kind ∈ {series, product_type, feature, model_bucket, other}. "
                "- series: true product line/sub-brand. "
                "- product_type: installation/category words. "
                "- feature: technology/功能/卖点 slogans. "
                "- model_bucket: loose model range or pattern only. "
                "- other: anything else. "
                "Keep=true only when kind=series. "
                "For canonical: choose one stable canonical per same series alias group. "
                "Prefer concise, product-like canonical names (e.g., coded series labels) over narrative/marketing sentences. "
                "Drop non-product phrases such as protocol/network/control items (CAN/BACnet/Modbus), "
                "version notes (上一代/下一代), and explanatory descriptors. "
                "Use strict rule-based judgment; do not rely on example imitation. "
                "Do not invent names; rely only on provided evidence."
            ),
        },
        {
            "role": "user",
            "content": f"Brand: {brand}\nCandidates:\n" + "\n".join(lines),
        },
    ]


def build_series_canon_messages(brand: str, groups: List[Dict]) -> List[Dict]:
    lines = []
    for idx, item in enumerate(groups, 1):
        aliases = " | ".join(item.get("aliases", [])[:8])
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(
            f"{idx}. group_key={item.get('group_key','')} | aliases={aliases} | evidence={evid}"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are canonicalizing HVAC Stage-B series alias groups. "
                "For each group_key, choose exactly one canonical series name from the provided aliases only. "
                "Do not invent names. "
                "Prefer stable product-series labels, not explanatory/narrative sentences. "
                "Avoid long clause-like aliases with punctuation or ellipsis. "
                "Prefer concise canonical forms such as coded series names or coded series + product form."
            ),
        },
        {
            "role": "user",
            "content": f"Brand: {brand}\nGroups:\n" + "\n".join(lines),
        },
    ]


def build_model_messages(
    brand: str,
    series: str,
    text_block: str,
    context_pages: Optional[List] = None,
) -> List[Dict]:
    page_hint = ""
    if context_pages:
        labels = [str(p) for p in context_pages if p is not None and str(p) != ""]
        if labels:
            page_hint = f"Context pages (series page + next page scope): {', '.join(labels)}\n"
    return [
        {
            "role": "system",
            "content": (
                "You are extracting HVAC product model names under a given brand/series. "
                "Output concrete model identifiers only. "
                "Only keep models that belong to the current series in the provided page scope. "
                "Do not output series names, technologies, categories, or generic words. "
                "Do not output accessory/optional/controller/panel/filter/module model codes as product models. "
                "Preserve exact model spelling from the text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand or '<unknown>'}\n"
                f"Series: {series or '<unknown>'}\n"
                f"{page_hint}"
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
    target_model: Optional[str] = None,
) -> List[Dict]:
    model_hint = ""
    if known_models:
        model_hint = (
            "该品牌-系列下的已知型号（优先抽取这些型号及其邻近表格行）：\n"
            + ", ".join(known_models[:50])
            + "\n"
        )

    target_hint = ""
    target_rules = ""
    identity_rules = (
        "- brand 与 series 已由上游阶段确认，必须按输入原样继承，禁止改写或重识别。\n"
    )
    model_pairing_rule = (
        "- 若存在内外机配对，请在 fact_text/evidence 保留配对信息；无法判断时优先使用室外机型号作为 product_model。\n"
    )
    if target_model:
        target_hint = f"目标型号: {target_model}\n"
        target_rules = (
            f"- 仅允许抽取目标型号 '{target_model}' 的信息。\n"
            "- 其他型号的行/段落一律忽略。\n"
            "- 若当前文本块没有目标型号证据，返回空 results 数组。\n"
        )
        identity_rules += (
            f"- product_model 固定为 target_model '{target_model}'，禁止推断、替换或改写。\n"
        )
        model_pairing_rule = (
            "- 若存在内外机配对，可在 fact_text/evidence 保留配对信息，但 product_model 必须保持为 target_model。\n"
        )
    else:
        identity_rules += (
            "- 提供了 known_models 时，product_model 必须来自 known_models，禁止虚构新型号。\n"
        )

    return [
        {
            "role": "system",
            "content": (
                "你是暖通(HVAC)产品结构化抽取器。请按给定 JSON Schema 输出当前品牌/系列下的产品信息。\n"
                "规则：\n"
                "- 一行/一条/一个型号对应一个产品，禁止把不同型号合并。\n"
                f"{target_rules}"
                f"{identity_rules}"
                "- 尽量抽取：制冷剂、能效指标(SEER/EER/COP/IPLV/IEER/APF/HSPF)、制冷/制热量、风量、电源相数电压频率、内外机配对等。\n"
                f"{model_pairing_rule}"
                "- 保留表格中的原始数值和单位；无法归类的技术参数也放入 performance_specs，并保留 raw 片段。\n"
                "- 强制中文输出：performance_specs.name 必须是简洁中文参数名，不允许英文参数名。\n"
                "- 若原文参数名是英文，先翻译为中文标准参数名再输出；APF/COP/EER/IPLV/SEER/HSPF 可保留缩写。\n"
                "- rows_json 含原始 CSV 行与 table_data/bbox，可用于将参数绑定到正确型号行。\n"
                "- 仅使用文本中可证实的信息；缺失字段留空。\n"
                "- 重点是属性抽取（specs/features/evidence），不是层级识别。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"品牌: {brand or '<unknown>'}\n"
                f"系列: {series or '<unknown>'}\n"
                f"{target_hint}"
                f"{model_hint}"
                "请从以下文本块抽取产品字段。evidence 需给出短文本片段或表格行（可引用 rows_json）。\n\n"
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
                "Drop: series/line/model tokens and non-brand organization words, "
                "institutes, departments, distributors, media, standards bodies. "
                "Names that are mostly used as model prefixes or line abbreviations "
                "(e.g., GMV/VRV/MULTI-V/CITY MULTI) should be dropped unless clear evidence shows they are manufacturer names. "
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


def build_brand_primary_messages(candidates: List[Dict]) -> List[Dict]:
    """
    Pick the primary manufacturer brand for one document.
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
                "You are selecting the primary manufacturer brand for one HVAC document. "
                "Most single-product brochures have one manufacturer brand. "
                "Return at most one name in keep[]. "
                "Do not select series/line/model aliases as brand. "
                "If multiple true brands appear, choose the manufacturer that owns most product context in this document."
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose at most ONE primary manufacturer brand from candidates below. "
                "Output JSON with keep array (0 or 1 item).\n\n"
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
                "- If the brand is already Chinese, keep or shorten to its common brand form.\n"
                "- If an English brand has a well-known Chinese canonical form, translate to that.\n"
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


def build_model_review_messages(
    brand: str,
    series: str,
    candidates: List[Dict],
) -> List[Dict]:
    lines = []
    for idx, item in enumerate(candidates, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        lines.append(f"{idx}. name={item.get('name','')} | evidence={evid}")
    user_block = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": (
                "You are verifying HVAC product model candidates. "
                "For each name decide if it is a concrete product MODEL (keep=true) "
                "vs series/line/accessory/other (keep=false). "
                "Accessory means optional controller/panel/filter/module/kit model codes. "
                "Also guess the closest product series/line name if the text reveals it "
                "and leave it empty if uncertain. "
                "If it clearly belongs to another series, set redirect_to and a confidence score 0-1. "
                "Do not invent or rewrite names. Use evidence snippets only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand or '<unknown>'}\n"
                f"Series: {series or '<unknown>'}\n"
                "Label each candidate with kind in {model, series, accessory, other}, set keep=true only if it is a model, "
                "and fill series_guess/redirect_to/confidence when identifiable.\n\n"
                f"{user_block}"
            ),
        },
    ]


def build_product_review_messages(
    brand: str,
    series: str,
    products: List[Dict],
) -> List[Dict]:
    lines = []
    for idx, item in enumerate(products, 1):
        evid = "; ".join(item.get("evidence", [])[:2])
        facts = "; ".join(item.get("fact_text", [])[:1])
        lines.append(
            f"{idx}. model={item.get('product_model','')} | category={item.get('category','')} "
            f"| evidence={evid} | fact={facts}"
        )
    user_block = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": (
                "You are reviewing structured HVAC product candidates. "
                "Mark is_accessory=true only for accessories/optional controllers/parts; otherwise false. "
                "Set keep=true if evidence supports a real product row (unique model/SKU). "
                "Return role ∈ {product, series_feature, accessory}: "
                "- series_feature for series-level 功能/卖点/型号区间/标题; "
                "- accessory for controllers/filters/etc.; "
                "- product otherwise. "
                "If text is mainly describing optional accessory pairing (not an independent product row), use role=accessory. "
                "Align keep/is_accessory/series_feature with role."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand: {brand or '<unknown>'}\n"
                f"Series: {series or '<unknown>'}\n"
                "Return keep/is_accessory/series_feature/role for each product.\n\n"
                f"{user_block}"
            ),
        },
    ]
