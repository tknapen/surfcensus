"""``epi_session``: one session of a local folder packed into a directory, without a run
directory or a catalog; the function behind ``surfcensus epi-session`` (docs/epi_interface.md,
Step 1), the counterpart of ``render.three_plane``. Work package E1.Z.

Nothing is fetched and nothing is written into the folders that are read: the raw dataset, the
derivative and the FreeSurfer subjects directory are sources (``sources.local``'s refusal
applies to ``out``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surfcensus.config import EpiConfig
from surfcensus.func.model import SessionInfo
from surfcensus.records import ConnectorMatch


@dataclass(frozen=True)
class FuncFolders:
    """The three folders a session needs, as :func:`find_folders` found them."""

    deriv_root: Path
    match: ConnectorMatch
    raw_root: Path | None
    """None: the runs are listed with ``no raw series`` and the message names ``--raw``."""
    fs_subjects_dir: Path | None


def find_folders(path: Path, raw: Path | None = None) -> FuncFolders:
    """The derivative, the raw dataset and the FreeSurfer subjects directory of ``path`` by
    the rules of docs/epi_interface.md, "Which folders are found by rule", for a local folder.
    Raises ``SourceError("layout", ...)`` when ``path`` holds no derivative of a known
    pipeline."""
    raise NotImplementedError("work package E1.Z: func/api.py")


def epi_session(
    path: Path,
    subject: str,
    out: Path,
    session: str | None = None,
    raw: Path | None = None,
    config: EpiConfig | None = None,
    key: bytes | None = None,
) -> SessionInfo:
    """Pack one session of ``subject`` (``sub-01`` or ``01``) found under ``path`` into
    ``out``. ``session`` names one when the participant has several. ``key`` keys the session
    and run ids (``func.func_id``); a random one when None."""
    raise NotImplementedError("work package E1.Z: func/api.py")
