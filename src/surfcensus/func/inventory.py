"""Phase 2f, the functional inventory of a run, and the lease of a session for the pack phase
(docs/epi_interface.md, "Step 2: the plan"; work package E2.B).

Private side of a run, like ``inventory/subjects.py``: it takes the private ``Catalog``, roots
and the run's key, and hands the pack phase nothing but a ``records.SessionWork`` inside a
lease. It reads listings, the raw sidecars, the summary and validation reports, the stored
transforms, the brain masks and the headers of the series (``func.session.plan_session`` and
the connector's ``raw_to_anat``, ``corrections``, ``repetition_time``); it reads no volume of a
series, fetches nothing and writes nothing outside ``catalog.sqlite``.

What :func:`run_func_inventory` does, per local dataset of the run with a functional source
(``func.discover.func_source_of``):

* stores the source (``Catalog.upsert_func_source``) and, per participant of the derivative,
  its ``participant_uid`` (``privacy.participant_uid`` under the run's key, over the source
  kind, the stored ``deriv_identity``, the connector and the label);
* per session, its ``session_uid`` and the ``run_uid`` of every run (``privacy.session_uid``,
  ``privacy.run_uid``); its ``inventory_hash`` (``records.func_signature`` of every listed file
  of its runs and of the session, the FreeSurfer files of the contours, and the extra lines
  ``EpiConfig.inventory_hash()``, the FreeSurfer subject's name and the run ids); a session
  whose hash is the stored one is skipped without opening a file (decision D21), otherwise it
  is planned (``func.session.plan_session``: the anchor among every run, the frame check, the
  grids, ``not selected`` from ``EpiConfig.runs``, ``not fetched`` from the git-annex
  placeholders) and stored with its runs (``Catalog.upsert_func_session``);
* participants, sessions and runs that the listing no longer has are marked absent
  (``Catalog.mark_func_absent``); their packs stay until the pack phase removes them.

The planning is that of step 1's ``func.api.epi_subject`` (the loop over the sessions, the
placeholders, the details of dangling links, the inputs of every packable run): E2.B moves it
out of ``func.api`` into this module or ``func.discover`` and has ``func.api`` call it, so that
one implementation serves both, and ``surfcensus epi-session`` keeps its behaviour and its
tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from surfcensus.records import FuncLeaseFacts, PhaseReport, SessionWork

if TYPE_CHECKING:
    from surfcensus.config import EpiConfig, RunConfig
    from surfcensus.privacy import HashKey
    from surfcensus.sources.base import Source
    from surfcensus.store import Catalog


def run_func_inventory(
    catalog: Catalog,
    sources: Sequence[Source],
    config: RunConfig,
    key: HashKey,
    *,
    force: bool = False,
) -> PhaseReport:
    """Phase 2f for the local sources of a run (module docstring). Returns a report of phase
    ``func_inventory`` whose units are sessions: done (stored now), skipped (hash unchanged),
    failed (a session whose plan could not be made: ``unreadable``). A dataset without a
    functional source, a run whose ``EpiConfig.enabled`` is false and a census run give an
    empty report and no row. ``force`` plans every session again. Raises ``SourceError`` for
    a derivative or a raw dataset that cannot be listed, before anything is recorded."""
    raise NotImplementedError("work package E2.B: func/inventory.py")


def session_work(facts: FuncLeaseFacts, config: EpiConfig) -> SessionWork:
    """The session of ``facts`` made again from its files, for the lease of the pack phase
    (``Run``'s ``func_lease``): the derivative and the raw dataset listed again for the
    participant, the listing compared with ``facts.inventory_hash`` (``SourceError("changed",
    ...)`` when it differs, so nothing is packed from files other than those inventoried),
    the plan (``func.session.plan_session`` with the stored ``run_uid``s), the inputs of every
    packable run (``records.RunPackInput``, with the anchor's transform and the session's
    inputs) and the contour job. The paths are those of the files in place: a local lease."""
    raise NotImplementedError("work package E2.B: func/inventory.py")


def func_fetch_lines(catalog: Catalog) -> list[str]:
    """The lines that name the functional runs with git-annex placeholders and the one
    command that fetches their files (``datalad -C <dataset> get ...``, else ``git -C <repo>
    annex get ...``), printed after the surface fetch line by ``review`` and ``inventory``
    (decision D35). A raw dataset and a derivative are often two DataLad datasets; the command
    names each. Empty when no present run has placeholders."""
    raise NotImplementedError("work package E2.B: func/inventory.py")
