"""Where the functional part of a dataset of a run lies: the derivative, the raw BIDS dataset
and the FreeSurfer subjects directory, by the rules of docs/epi_interface.md, "Which folders
are found by rule" (step 2, work package E2.B).

One implementation of the rules, not two. Step 1 wrote them as ``func.api.find_folders`` (with
``FuncFolders``, ``refuse_out`` and the places that are read); E2.B moves those here unchanged
and leaves ``func.api`` importing them, so ``surfcensus epi-session`` and a run directory find
the same folders for the same path. What this module adds for a run:

* :func:`func_source_of` starts from the dataset record the surface source found (its root and
  ``fs_root``), never from a search of its own below ``derivatives/``: a stored root is not
  searched again (decision D35), and the derivative of a run is the one that holds the run's
  FreeSurfer subjects, else the folder itself, else the first folder of ``derivatives/`` that a
  connector recognises.
* :func:`read_places` and :func:`stored_read_places`: the derivative and the raw dataset as
  places that are only read, so that a run directory or an export there is refused like one
  in a FreeSurfer source (``sources.local.refuse_in_sources``; the rule of D35 extended).

Private side of a run, like ``inventory/subjects.py``: it handles roots. It lists folders and
reads ``dataset_description.json``; it opens no other file, fetches nothing and writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from surfcensus.records import FuncSourceRecord

if TYPE_CHECKING:
    from surfcensus.config import RunConfig
    from surfcensus.sources.base import DatasetRecord


def func_source_of(record: DatasetRecord, raw: Path | None = None) -> FuncSourceRecord | None:
    """The functional source of a local dataset of a run, or None when its folder has no
    functional part (no derivative of a known connector with ``sub-*/[ses-*/]func/``): then
    no functional table gets a row and nothing is said but one line at info level.

    ``raw`` is ``EpiConfig.raw`` (``--raw DIR``); without it the raw dataset is found by rule
    and may be None (its runs are listed as ``no raw series`` and the message names
    ``--raw``). ``deriv_identity`` is the resolved derivative root; the caller keeps a stored
    identity (``Catalog.upsert_func_source``). Raises ``SourceError`` as ``find_folders``
    does: ``unreadable`` for a folder that cannot be listed, ``layout`` or ``not found`` for a
    ``raw`` that holds no raw functional series or is no directory."""
    raise NotImplementedError("work package E2.B: func/discover.py")


def read_places(source: FuncSourceRecord) -> list[tuple[str, Path]]:
    """The derivative and the raw dataset of ``source`` and what a link at their top level
    leads to, as ``(what, directory)`` pairs for ``sources.local.refuse_in_sources(also=...)``.
    The FreeSurfer subjects directory is a place of the surface source already."""
    raise NotImplementedError("work package E2.B: func/discover.py")


def stored_read_places(config: RunConfig) -> list[tuple[str, Path]]:
    """:func:`read_places` of every local source of ``config``, found again from its stored
    root and ``EpiConfig.raw`` by listings alone (no catalog), for the commands that refuse a
    run directory from ``config.yaml`` (``sources.local.refuse_run_in_sources``: the rater
    app, ``finalize``, ``calibration freeze``, ``export``). A root that is not there gives no
    place."""
    raise NotImplementedError("work package E2.B: func/discover.py")
