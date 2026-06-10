"""The shared identity base for everything chai tracks: components and results alike."""

from typing import Optional


class BaseThing(object):
    """Minimal shared base for everything chai tracks: components and results alike.

    Carries the common identity attributes -- ``id`` (unique within a workflow run), a human-readable
    ``name``, a back-reference to the owning ``workflow``, and the raw ``input`` the thing was created
    from. ``Component`` and ``Result`` both extend this so results can point at their processors and
    workflows without circular imports.
    """

    id: str = ""
    name: str = ""
    workflow: Optional["BaseThing"] = None
    input = None
