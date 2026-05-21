from .gemini import GeminiComponent
from .lm_studio import LMStudioComponent
from .ollama import OllamaComponent
from .transformers import TransformersComponent

__all__ = [GeminiComponent, LMStudioComponent, OllamaComponent]

__more_than_all__ = [TransformersComponent]


def create_all_components(base_class: type) -> dict:
    """Iterate through __all__ and create the components"""
    clss = {}
    for ai_class in __all__:
        name = f"{ai_class.__name__.replace('Component', '')}{base_class.__name__}"
        new = create_ai_component(name, base_class, ai_class)
        clss[name] = new
    return clss


def create_ai_component(name: str, base_class: type, ai_class: type) -> type:
    """
    Dynamically generates a composite AI component class (e.g. GeminiSegmenter).
    This avoids having to hand-craft boilerplate multiple inheritance classes.
    """

    def __init__(self, tree, workflow, parent=None):
        # Initialize the AI class first (this usually calls Component.__init__ via super)
        ai_class.__init__(self, tree, workflow, parent)

        # Initialize the base class if it has its own __init__ logic
        # (e.g., to set expected defaults or specific properties)
        # We avoid object.__init__ or Component.__init__ to prevent double-initialization
        if "Component" not in base_class.__name__ and base_class.__init__ is not object.__init__:
            try:
                base_class.__init__(self, tree, workflow, parent)
            except TypeError:
                pass

    def _process(self, input):
        # AI composite classes always delegate their core processing to the AI implementation
        return ai_class._process(self, input)

    return type(
        name,
        (base_class, ai_class),
        {"__init__": __init__, "_process": _process, "__module__": base_class.__module__},
    )
