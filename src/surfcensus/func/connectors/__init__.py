"""Connectors of preprocessing pipelines (``func.connectors.base`` has the protocol).

``CONNECTORS`` names the built ones in the order in which a folder is tried. A connector
module is imported when it is first needed.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.records import ConnectorMatch

__all__ = ["CONNECTORS", "Connector", "ConnectorError", "connector", "detect"]

CONNECTORS: tuple[str, ...] = ("fmriprep",)
"""Built connectors, by name; each is the module ``func.connectors.<name>`` with a function
``make() -> Connector``."""


def connector(name: str) -> Connector:
    """The connector of that name. Raises ``KeyError`` for a name that is not built."""
    if name not in CONNECTORS:
        raise KeyError(f"no connector {name!r}; built: {', '.join(CONNECTORS)}")
    made = import_module(f"surfcensus.func.connectors.{name}").make()
    if not isinstance(made, Connector):
        raise TypeError(f"func.connectors.{name}.make() does not make a Connector")
    return made


def detect(folder: Path) -> tuple[Connector, ConnectorMatch] | None:
    """The first connector of :data:`CONNECTORS` that recognises ``folder``, with its match;
    None when none does."""
    for name in CONNECTORS:
        made = connector(name)
        match = made.detect(Path(folder))
        if match is not None:
            return made, match
    return None
