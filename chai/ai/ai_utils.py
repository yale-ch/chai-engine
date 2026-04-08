import logging
import re
from typing import Optional, Union

import ujson as json
import yaml

logger = logging.getLogger("chai")


def extract_yaml(text: str):
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


def extract_json(text: str) -> Optional[Union[dict, list]]:
    """Extract JSON from LLM response, handling control chars, markdown fences and truncation."""

    text = text.strip()

    # 1. Clean control characters that might break standard JSON parsers
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # 2. Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

    # 3. Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Try fixing trailing commas
    cleaned_text = re.sub(r",\s*}", "}", text)
    cleaned_text = re.sub(r",\s*]", "]", cleaned_text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # 5. Try extracting the outermost JSON object/array
    match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 6. Truncated JSON recovery
    repaired = try_repair_truncated_json(text)
    if repaired is not None:
        return repaired

    logger.error(f"Failed to parse JSON from LLM response: {text[:500]}")
    return {}


def try_repair_truncated_json(text: str) -> Optional[Union[dict, list]]:
    """Attempt to repair truncated JSON by closing open structures."""
    match = re.search(r"[\{\[]", text)
    if not match:
        return None

    fragment = text[match.start() :]

    # Close any open string literal
    in_string = False
    escaped = False
    for ch in fragment:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        fragment += '"'

    # Count and close open brackets/braces
    opens = 0
    open_brackets = 0
    in_str = False
    esc = False
    for ch in fragment:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1

    fragment += "]" * max(open_brackets, 0)
    fragment += "}" * max(opens, 0)

    # Strip trailing comma before closing braces/brackets
    fragment = re.sub(r",\s*([}\]])", r"\1", fragment)

    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return None
