"""Parsing helpers for LLM responses: robust JSON/YAML extraction from messy model output."""

import logging
import re
from typing import Optional, Union

import ujson as json
import yaml

logger = logging.getLogger("chai")


def extract_yaml(text: str):
    """Parse *text* as YAML, returning ``None`` on any parse failure."""
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


# ---------------------------------------------------------------------- prompt slots

# The slots whose text changes from one call to the next. ``{step_name}`` and ``{input_length}`` are
# not listed: the first is fixed per component, the second a short number.
VARIABLE_SLOT_RE = re.compile(r"\{(text_input_\d+|ADDITIONAL_CONTEXT|first_input|last_input)\}")
TEXT_SLOT_RE = re.compile(r"\{text_input_(\d+)\}")

# How much fixed text may follow the first variable slot before the prompt is reported as defeating
# the provider's prompt (KV) cache. A short trailing line ("Return JSON only.") costs little; a page of
# rules after the variable part is re-read on every call.
FIXED_TAIL_LIMIT = 200


def describe_inputs(items) -> str:
    """What an AI component was given, for error messages: count and type of each item."""
    items = list(items)
    if not items:
        return "no items (an empty input)"
    kinds = []
    for item in items:
        typ = item.metadata.get("type") if hasattr(item, "metadata") else None
        kinds.append(typ or type(item).__name__)
    return f"{len(items)} item{'s' if len(items) != 1 else ''}: {', '.join(kinds)}"


def check_text_slots(component, prompt: str, format_vars: dict, items) -> None:
    """Raise ``ValueError`` when *prompt* names a ``{text_input_<i>}`` slot the input did not fill.

    Sending the literal ``{text_input_0}`` lets a model answer a question it was never asked (with no
    image either, when the input was empty, it invents a description), so the call is refused and the
    component's ``retries``/``on_error``/``error_steps`` apply instead.
    """
    named = sorted({int(i) for i in TEXT_SLOT_RE.findall(prompt)})
    missing = [f"{{text_input_{i}}}" for i in named if f"text_input_{i}" not in format_vars]
    if missing:
        raise ValueError(
            f"{component}: the prompt names {', '.join(missing)} but the input did not fill "
            f"{'it' if len(missing) == 1 else 'them'} (slot i takes the TEXT/DATA item at position i); "
            f"the input held {describe_inputs(items)}"
        )


def prompt_order_problem(prompt: str) -> Optional[str]:
    """Why *prompt* defeats prompt caching, or ``None``.

    The heuristic: after the first variable slot (``VARIABLE_SLOT_RE``), more than
    ``FIXED_TAIL_LIMIT`` characters of fixed text (the rest of the prompt with its slots removed,
    whitespace trimmed). Prompts with no variable slot are fine -- they are fixed throughout.
    """
    if not prompt:
        return None
    first = VARIABLE_SLOT_RE.search(prompt)
    if first is None:
        return None
    tail = VARIABLE_SLOT_RE.sub("", prompt[first.end():]).strip()
    if len(tail) <= FIXED_TAIL_LIMIT:
        return None
    return (
        f"the prompt puts {first.group(0)} before {len(tail)} characters of fixed text, so the cacheable "
        f"prefix ends at character {first.start()}; move standing rules into `system_instruction` (or "
        f"above the slot) and put the variable part at the end of the prompt"
    )


def warn_prompt_order(component) -> None:
    """Log ``prompt_order_problem`` for *component*'s prompt, once per component and prompt text."""
    prompt = getattr(component, "prompt_text", "") or ""
    if getattr(component, "_prompt_order_checked", None) == prompt:
        return
    component._prompt_order_checked = prompt
    problem = prompt_order_problem(prompt)
    if problem:
        logger.warning(f"{component}: {problem}")
