"""The functional data of a run in ``surfcensus status`` (step 3 of docs/epi_interface.md,
"Step 3: the plan (E3.0)"; work package E3.B writes the bodies).

Three lines after the line ``render:``, printed only for a local run whose catalog has a
present functional source (``PublicCatalog.func_sources()`` not empty), each starting with one
of :data:`STATUS_PREFIXES`::

    func_inventory: 3 sessions of 2 participants, 19 runs: 11 can be packed, 8 cannot (...)
    pack: 11 done, 0 failed, 0 pending
    func checks: 2 runs flagged, 0 raw affines unchecked, 1 session with the anchor the outlier

The ``pack:`` line is ``func.batch.pack_status``: what ``surfcensus pack`` would find now (the
pack signature and the disk), not what is recorded (``func.batch.recorded_states``, which the
subject pages and the export say). ``status --failures`` adds :func:`failure_rows` to its
listing. Downstream of the inventory (decision D20): a ``RunContext`` and its
``PublicCatalog``, never the ``Run``. On a catalog of schema 1 every functional frame is
empty, so nothing is printed and nothing is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surfcensus.records import RunContext
    from surfcensus.store import PublicCatalog

_TODO = "work package E3.B: func/status.py"

STATUS_PREFIXES: tuple[str, ...] = ("func_inventory:", "pack:", "func checks:")
"""How the three lines of :func:`status_lines` begin, in the order they are printed."""

FailureRow = tuple[str, str, str, str | None]
"""``(unit, phase, reason, detail)``, the form of a row of the listing of ``status
--failures``."""


def status_lines(ctx: RunContext) -> list[str]:
    """The lines of :data:`STATUS_PREFIXES`, in that order; empty for a run without a present
    functional source, a census run and a catalog of schema 1. With the EPI interface off and
    rows stored, the second line is ``pack: off (--no-epi, stored)``."""
    raise NotImplementedError(_TODO)


def failure_rows(ctx: RunContext) -> list[FailureRow]:
    """What ``status --failures`` lists of the functional side: every run counted as failed
    by ``func.batch.pack_status`` (phase ``pack``, from ``func.batch.pack_failures``), every
    present run that cannot be packed (phase ``func_inventory``, with its reason and detail),
    and every session whose ``problem`` is not a fetch problem (phase ``func_inventory``,
    unit ``session:<session_uid>``). The ``work`` rows of phase ``epi_session`` are not among
    them: a whole-session failure is recorded on each of its runs (decision D41)."""
    raise NotImplementedError(_TODO)


def unit_names(public: PublicCatalog) -> dict[str, str]:
    """``run_uid`` -> ``sub-01 ses-1_task-rest_run-1`` and ``session:<session_uid>`` ->
    ``sub-01 ses-1`` in a local run, for the lines of ``pack``, ``status`` and ``review``
    (replaces ``cli._run_names``); a unit without a label is named by itself."""
    raise NotImplementedError(_TODO)
