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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from surfcensus.records import RunContext
    from surfcensus.store import PublicCatalog

STATUS_PREFIXES: tuple[str, ...] = ("func_inventory:", "pack:", "func checks:")
"""How the three lines of :func:`status_lines` begin, in the order they are printed."""

FailureRow = tuple[str, str, str, str | None]
"""``(unit, phase, reason, detail)``, the form of a row of the listing of ``status
--failures``."""

FUNC_INVENTORY_PHASE = "func_inventory"
"""The phase of the rows of :func:`failure_rows` that say what the functional inventory found:
a run that cannot be packed, a session that could not be planned."""

UNCHECKED_AFFINE = "header, unchecked"
"""``func_runs.raw_affine`` of a run whose raw affine no reference image could check."""

_SESSION_PREFIX = "session:"


def _true(value: object) -> bool:
    """A flag of a public frame (0 or 1, a bool, pandas' boolean or a missing value)."""
    if value is None:
        return False
    try:
        if value != value:  # NaN; pandas' NA raises in bool()
            return False
        return bool(value)
    except TypeError:
        return False


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


@dataclass(frozen=True)
class _Present:
    """The present functional rows of a local run, read once per call."""

    participants: Mapping[str, Mapping[str, Any]]
    sessions: Sequence[Mapping[str, Any]]
    runs: Sequence[Mapping[str, Any]]


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [] if len(frame) == 0 else frame.to_dict("records")


def _present(public: PublicCatalog) -> _Present | None:
    """The present participants, their present sessions and the present runs of those, as
    ``review`` counts them; None for a census run and for a run without a present functional
    source (also a catalog of schema 1, where every functional frame is empty)."""
    if public.mode() != "local":
        return None
    if not any(_true(flag) for flag in public.func_sources()["present"]):
        return None
    participants = {
        str(row["participant_uid"]): row
        for row in _rows(public.func_participants())
        if _true(row["present"])
    }
    sessions = [
        row
        for row in _rows(public.func_sessions())
        if _true(row["present"]) and row["participant_uid"] in participants
    ]
    live = {row["session_uid"] for row in sessions}
    runs = [
        row
        for row in _rows(public.func_runs())
        if _true(row["present"]) and row["session_uid"] in live
    ]
    return _Present(participants, sessions, runs)


def _planning_problem(problem: object) -> str | None:
    """The ``problem`` of a session that says it could not be planned; None for none and for
    a fetch problem (files without content), which is no failure (decision D35)."""
    from surfcensus.sources.patterns import is_unfetched_problem

    if not isinstance(problem, str) or not problem or is_unfetched_problem(problem):
        return None
    return problem


def _cannot(row: Mapping[str, Any]) -> str | None:
    """The reason a present run is not among those that can be packed (``not selected``
    included), or None for a run that can be packed."""
    from surfcensus.records import NOT_SELECTED

    if not _true(row["selected"]):
        return _text(row["reason"]) or NOT_SELECTED
    if not _true(row["packable"]):
        return _text(row["reason"]) or "no reason recorded"
    return None


def status_lines(ctx: RunContext) -> list[str]:
    """The lines of :data:`STATUS_PREFIXES`, in that order; empty for a run without a present
    functional source, a census run and a catalog of schema 1. With the EPI interface off and
    rows stored, the second line is ``pack: off (--no-epi, stored)``.

    ``func_inventory:`` counts the present sessions, participants and runs as the functional
    inventory recorded them, how many runs can be packed, and why the others cannot (the
    reasons sorted by name, each with its count; ``not selected`` for runs that ``--epi-runs``
    leaves out), then the sessions that could not be planned. ``pack:`` is
    ``func.batch.pack_status``: what ``surfcensus pack`` would find now (pack signature and
    disk), so a pack that is recorded and whose inputs changed since is pending here, while the
    subject pages and the export, which read what is recorded (``func.batch.recorded_states``),
    call it packed. ``func checks:`` counts runs flagged by the frame check, raw affines that no
    reference image checked, and sessions whose anchor is the outlier: statements about the
    raw data under each session's one registration (decision D36), not faults of the tool.
    Reads the catalog and lists the run directory; writes nothing."""
    from surfcensus.config import epi_settings
    from surfcensus.func.batch import pack_rewrites, pack_status

    rows = _present(ctx.public)
    if rows is None:
        return []
    reasons: dict[str, int] = {}
    packable = 0
    for row in rows.runs:
        why = _cannot(row)
        if why is None:
            packable += 1
        else:
            reasons[why] = reasons.get(why, 0) + 1
    cannot = sum(reasons.values())
    first = (
        f"func_inventory: {_plural(len(rows.sessions), 'session')} of "
        f"{_plural(len(rows.participants), 'participant')}, {_plural(len(rows.runs), 'run')}: "
        f"{packable} can be packed, {cannot} cannot"
    )
    if reasons:
        first += " (" + ", ".join(f"{why} {reasons[why]}" for why in sorted(reasons)) + ")"
    unplanned = sum(_planning_problem(row["problem"]) is not None for row in rows.sessions)
    if unplanned:
        first += f"; {_plural(unplanned, 'session')} could not be planned"
    if epi_settings(ctx.config).enabled:
        done, failed, pending = pack_status(ctx)
        second = f"pack: {done} done, {failed} failed, {pending} pending"
        rewrites = pack_rewrites(ctx)
        if rewrites:
            second += f"; {_plural(rewrites, 'session')} to write again"
        if failed:
            second += f" (tried again with: surfcensus pack --run-dir {ctx.run_dir} --retry-failed)"
    else:
        second = "pack: off (--no-epi, stored)"
    flagged = sum(_true(row["flagged"]) for row in rows.runs)
    unchecked = sum(row["raw_affine"] == UNCHECKED_AFFINE for row in rows.runs)
    outliers = sum(_true(row["anchor_is_outlier"]) for row in rows.sessions)
    third = (
        f"func checks: {_plural(flagged, 'run')} flagged, "
        f"{_plural(unchecked, 'raw affine')} unchecked, "
        f"{_plural(outliers, 'session')} with the anchor the outlier (statements about the raw "
        "data)"
    )
    return [first, second, third]


def failure_rows(ctx: RunContext) -> list[FailureRow]:
    """What ``status --failures`` lists of the functional side: every run counted as failed
    by ``func.batch.pack_status`` (phase ``pack``, from ``func.batch.pack_failures``), every
    present run that cannot be packed (phase ``func_inventory``, with its reason and detail),
    and every session whose ``problem`` is not a fetch problem (phase ``func_inventory``,
    unit ``session:<session_uid>``). The ``work`` rows of phase ``epi_session`` are not among
    them: a whole-session failure is recorded on each of its runs (decision D41).

    A run that ``--epi-runs`` leaves out (``not selected``) is not listed: it was left out on
    purpose and can be packed. Ordered by phase, then unit, as the listing sorts; a unit is a
    ``run_uid`` or ``session:<session_uid>``, to be named with :func:`unit_names`."""
    from surfcensus.config import epi_settings
    from surfcensus.func.batch import pack_failures
    from surfcensus.records import PACK_PHASE

    rows = _present(ctx.public)
    if rows is None:
        return []
    out: list[FailureRow] = []
    if epi_settings(ctx.config).enabled:
        out += [
            (uid, PACK_PHASE, reason, detail)
            for uid, (reason, detail) in pack_failures(ctx).items()
        ]
    for row in rows.runs:
        if _true(row["selected"]) and not _true(row["packable"]):
            reason = _text(row["reason"]) or "no reason recorded"
            out.append((str(row["run_uid"]), FUNC_INVENTORY_PHASE, reason, _text(row["detail"])))
    for row in rows.sessions:
        problem = _planning_problem(row["problem"])
        if problem is None:
            continue
        reason, colon, detail = problem.partition(": ")
        out.append(
            (
                f"{_SESSION_PREFIX}{row['session_uid']}",
                FUNC_INVENTORY_PHASE,
                reason,
                detail if colon else None,
            )
        )
    return sorted(out, key=lambda row: (row[1], row[0]))


def unit_names(public: PublicCatalog) -> dict[str, str]:
    """``run_uid`` -> ``sub-01 ses-1_task-rest_run-1`` and ``session:<session_uid>`` ->
    ``sub-01 ses-1`` in a local run, for the lines of ``pack``, ``status`` and ``review``
    (replaces ``cli._run_names``); a unit without a label is named by itself. Every run and
    session of the catalog is named, present or not; empty for a catalog without functional
    data."""
    from surfcensus.func.inventory import session_text

    participants = {
        str(row["participant_uid"]): _text(row["participant"])
        for row in _rows(public.func_participants())
    }
    owner: dict[str, str | None] = {}
    names: dict[str, str] = {}
    for row in _rows(public.func_sessions()):
        uid = str(row["session_uid"])
        who = participants.get(str(row["participant_uid"]))
        owner[uid] = who
        unit = f"{_SESSION_PREFIX}{uid}"
        names[unit] = unit if who is None else session_text(who, _text(row["session_label"]))
    for row in _rows(public.func_runs()):
        uid = str(row["run_uid"])
        who, name = owner.get(str(row["session_uid"])), _text(row["name"])
        names[uid] = f"{who} {name}" if who is not None and name is not None else uid
    return names
