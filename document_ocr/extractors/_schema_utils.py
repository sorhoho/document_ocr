"""Utilities to produce Ollama-compatible JSON schemas from Pydantic models.

Pydantic v2 generates schemas with $defs, $ref, anyOf, and complex Decimal
patterns. Ollama 0.24.x crashes when given schemas with $defs or unsupported
keywords. This module resolves references and simplifies the schema to a flat,
Ollama-safe form.

Simplification rules applied:
  - Inline all $ref/$defs recursively
  - anyOf [type, null] → type (Ollama infers optionality from defaults)
  - anyOf [string, number, null] → type: string (coerce after parsing)
  - Remove: title, description, pattern, exclusiveMinimum, allOf
  - Decimal fields become type: number (Pydantic serialises Decimal as str; we
    coerce in the validator)
"""
from __future__ import annotations

import copy
from typing import Any


def _inline_refs(schema: dict, defs: dict) -> dict:
    """Recursively resolve all $ref and inline $defs."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        resolved = copy.deepcopy(defs.get(ref_name, {}))
        resolved = _inline_refs(resolved, defs)
        return resolved

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("$defs", "$schema"):
            continue
        if isinstance(value, dict):
            result[key] = _inline_refs(value, defs)
        elif isinstance(value, list):
            result[key] = [
                _inline_refs(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _simplify_any_of(node: dict) -> dict:
    """Collapse anyOf[X, null] into just X (makes Ollama happy)."""
    if "anyOf" not in node:
        return node

    variants = node["anyOf"]
    non_null = [v for v in variants if v != {"type": "null"} and v.get("type") != "null"]

    if len(non_null) == 0:
        return {"type": "null"}

    if len(non_null) == 1:
        simplified = dict(non_null[0])
        # Propagate default if present
        if "default" in node:
            simplified["default"] = node["default"]
        return simplified

    # Multiple non-null variants: pick the most permissive type
    types = {v.get("type") for v in non_null if "type" in v}
    if "string" in types or "number" in types:
        result: dict[str, Any] = {"type": "string"}
        if "default" in node:
            result["default"] = node["default"]
        return result

    # Fall back to first variant
    simplified = dict(non_null[0])
    if "default" in node:
        simplified["default"] = node["default"]
    return simplified


# Strip keys that confuse Ollama or cause it to use schema defaults instead of extracting.
# "default" is stripped so Gemma doesn't fill fields with schema defaults instead of OCR text.
_STRIP_KEYS = {"title", "description", "pattern", "exclusiveMinimum", "examples", "$schema", "default"}


def _clean(node: Any) -> Any:
    """Remove noisy keys and simplify anyOf/allOf recursively."""
    if not isinstance(node, dict):
        return node

    # First simplify anyOf
    node = _simplify_any_of(node)

    # allOf with single item → unwrap
    if "allOf" in node and len(node["allOf"]) == 1:
        inner = dict(node["allOf"][0])
        node = {k: v for k, v in node.items() if k != "allOf"}
        node.update(inner)

    result: dict[str, Any] = {}
    for k, v in node.items():
        if k in _STRIP_KEYS:
            continue
        if isinstance(v, dict):
            result[k] = _clean(v)
        elif isinstance(v, list):
            result[k] = [_clean(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v

    return result


def ollama_schema(pydantic_cls: type) -> dict:
    """Return a flattened, Ollama-compatible JSON schema from a Pydantic model.

    Usage:
        format=ollama_schema(SupplierInvoice)
    """
    raw = pydantic_cls.model_json_schema()
    defs = raw.get("$defs", {})

    # Inline all defs recursively (defs themselves may reference other defs)
    resolved_defs: dict[str, dict] = {}
    for name, defn in defs.items():
        resolved_defs[name] = _inline_refs(defn, defs)

    inlined = _inline_refs(raw, resolved_defs)
    return _clean(inlined)
