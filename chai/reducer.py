"""Reducers: components that combine multiple results into a single one.

Only AI-backed variants exist so far (``GeminiReducer``, ...), generated from the backends in
``chai.ai``; they prompt a model with the collected inputs and return a single combined Result.
"""

from .ai import create_all_components
from .core import Component


class Reducer(Component):
    """Take multiple results and combine them to one.

    Abstract base for the reducer role: subclasses implement ``_process``, take a list-shaped Result
    (e.g. the per-entry outputs of an ``Iterator``) and return a single combined Result. When no prompt
    is configured, AI-backed variants use the workflow's default ``reduction`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("reduction", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Reducer))
