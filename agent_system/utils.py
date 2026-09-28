import json
import re
from typing import Any, Dict, Optional


def ranked_products(final: Dict[str, Any], *, coerce_non_strings: bool = False) -> list[str]:
    """Extract complete recommendation identifiers, preserving order and repeats.

    Dictionary entries use ticker or symbol; strings are stripped and uppercased.
    A description of a stock option remains distinct from its underlying ticker.
    Metric callers skip non-string entries. Stock-presence analysis opts into
    string coercion so even malformed entries keep their saved cutoff positions.
    Deduplication and top-k truncation belong to the metric that consumes the list.
    """
    products = []
    for item in final.get("ranked_products") or []:
        if isinstance(item, dict):
            item = item.get("ticker") or item.get("symbol") or ""
        if coerce_non_strings:
            products.append(str(item).strip().upper())
        elif isinstance(item, str) and item:
            products.append(item.strip().upper())
    return products


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_markdown_fences(strip_think_blocks(text))
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = -1

    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    payload = cleaned[start:end]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # Some models emit literal newlines/tabs inside JSON string values.
        # Escape them and retry.
        repaired = re.sub(
            r'"(?:[^"\\]|\\.)*"',
            lambda m: m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"),
            payload,
            flags=re.DOTALL,
        )
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None
