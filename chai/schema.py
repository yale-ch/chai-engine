"""Minimal JSON-Schema-style validation for structured component output.

Supports the subset that matters for LLM extraction: ``type`` (object, array,
string, number, integer, boolean, null), ``properties`` + ``required``,
``items``, and ``enum``. ``validate`` raises ``SchemaError`` with a path-rich
message -- raised from a component's ``_validate_output`` hook, that failure
flows into the standard retry/error policy, so ``retries: 2`` re-asks the
model when it emits an invalid record.
"""

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


class SchemaError(ValueError):
    """Raised when a value does not match its schema."""


def validate(value, schema, path="$"):
    """Validate *value* against *schema* (a dict); raises ``SchemaError`` on mismatch."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: schema must be a dict, got {type(schema).__name__}")

    expected = schema.get("type")
    if expected:
        py_type = _TYPES.get(expected)
        if py_type is None:
            raise SchemaError(f"{path}: unknown schema type {expected!r}")
        if expected == "number":
            ok = isinstance(value, py_type) and not isinstance(value, bool)
        elif expected == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, py_type)
        if not ok:
            raise SchemaError(f"{path}: expected {expected}, got {type(value).__name__} ({value!r:.80})")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value or value[key] in (None, ""):
                raise SchemaError(f"{path}.{key}: required field is missing or empty")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                validate(value[key], sub, path=f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], path=f"{path}[{i}]")

    return value
