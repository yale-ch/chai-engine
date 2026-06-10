"""chai: a component-based processing engine for LLM workflows."""

# Import core before result so the core<->result circular import always
# resolves in the same order, regardless of which submodule is imported first
# (e.g. ``import chai.result`` used to fail under unittest discovery).
from . import core  # noqa: F401
