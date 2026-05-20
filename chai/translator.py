from .ai import create_all_components
from .core import Component


class Translator(Component):
    """Takes linguistic content and translates it into one or more different languages"""

    pass


globals().update(create_all_components(Translator))
