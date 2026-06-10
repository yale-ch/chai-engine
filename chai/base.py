from typing import Optional


class BaseThing(object):
    id: str = ""
    name: str = ""
    workflow: Optional["BaseThing"] = None
    input = None
