"""``surfcensus review PATH``: inventory, metrics, selection, render and the rater app on a
local directory of FreeSurfer subjects, in one call. See docs/architecture.md.

The steps are those of the phase commands, in one process: the run is made or opened
(:func:`default_run_dir`), the subjects are inventoried and scored, every present subject
is selected, and the first subjects of the rating queue are rendered. Then the app starts,
and a second ``Run`` in a thread renders the others in queue order while it runs
(:func:`open_review`, decision D33). :func:`review` is that, the browser, and the wait for
Ctrl-C.

What the command says goes to stderr, one block per step. Subjects are named by label:
this is a local run, read on the terminal of whoever has the data.
"""

from __future__ import annotations

import math
import os
import signal
import sqlite3
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from surfcensus.app import APP_DB_FILE
from surfcensus.config import (
    ConfigError,
    load_config,
    new_local_config,
    with_changes,
    write_config,
)
from surfcensus.log import get_logger
from surfcensus.run import Run, other_root_hint
from surfcensus.sources.base import RENDER_ROLES, DatasetRecord, SourceError
from surfcensus.sources.local import (
    LAYOUT_WORDS,
    LocalSource,
    datalad_dataset_holding,
    dataset_holding,
    detect_layout,
    fetch_line,
    refuse_in_sources,
    subjects_held,
    where_subjects_lie,
)
from surfcensus.sources.patterns import ROLE_CANDIDATES, is_unfetched_problem, unfetched_said
from surfcensus.stopping import StopFlag, StopRequest
from surfcensus.store import queue_order, queue_scores

if TYPE_CHECKING:
    from surfcensus.app.server import ServerHandle
    from surfcensus.config import RunConfig, SourceConfig
    from surfcensus.records import PhaseReport

logger = get_logger()

DEFAULT_RUN_DIR = "surfcensus-run"
"""The name of the run directory that ``review`` makes beside the data when none is given."""

DEFAULT_SERVE_AFTER = 1
"""Renders the app waits for before it opens, when ``serve_after`` is not given (D33)."""

MAX_JOBS = 8
"""The worker processes of a review that names none: this, or the processors if fewer."""

SELECTION_REASON = "all"
"""The selection of a review: every present subject, with probability 1."""

_MAX_NAMES = 10  # subjects named on one line
_BY_SUBJECT = 3  # incomplete subjects that are each named with what they lack


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def default_jobs() -> int:
    """The worker processes of a review that names none."""
    return max(1, min(MAX_JOBS, os.cpu_count() or 1))


def default_run_dir(path: Path | str) -> Path:
    """Where ``review`` keeps the run of the data at ``path`` when no directory is given:
    ``PATH/../surfcensus-run`` when a run is there already or its parent can be written
    to, else ``./surfcensus-run``.

    A run directory may not lie inside the directory the subjects are read from. When
    ``path`` is one subject directory, its parent is that directory, and the run goes
    beside it: ``<subjects directory>/../surfcensus-run``, also when the subject directory
    is a link to another place.

    Subjects that lie below the ``derivatives/`` folder of a BIDS dataset
    (``sources.local.dataset_holding``): the run goes beside the dataset,
    ``<dataset>/../surfcensus-run``, never into it, whether ``path`` is the dataset or a
    folder inside it; a dataset, which may be a DataLad dataset, gets no files unless a run
    directory in it is asked for. A run of this data that an earlier version made inside
    the dataset, beside the folder that was named, is still found and used. The same holds
    when the place beside ``path`` lies inside a DataLad dataset
    (``sources.local.datalad_dataset_holding``, a directory with ``.git`` and ``.datalad``):
    the run goes beside the outermost such dataset, which gets no untracked files. When the
    place beside the dataset cannot be written to, the run goes to ``./surfcensus-run``,
    and when the current directory lies inside the dataset the call is refused
    (``ConfigError("invalid", ...)``), since only ``run_dir`` can then say where.

    A run of this root comes first, wherever of these places it is, and so does one in
    ``<name>-surfcensus-run`` beside the default (``<name>`` the dataset, or the folder
    named), the directory that :func:`refuse_foreign_default` tells a user to give when
    the default place holds the run of another dataset. When ``surfcensus-run`` holds the
    run of another root and no run of this one is found, it is returned all the same, and
    :func:`refuse_foreign_default` says what to give instead.

    Raises ``SourceError`` as :func:`surfcensus.sources.local.detect_layout` does.
    """
    record, implied = detect_layout(path, quiet=True)
    places = _run_places(Path(path), record, implied)
    for place in places.earlier:
        if _holds_run_of(place, record.root):
            return place
    if (places.default / "config.yaml").is_file():
        return places.default  # a run of another root: refuse_foreign_default speaks
    if places.dataset is None:
        if os.access(places.default.parent, os.W_OK | os.X_OK):
            return places.default
        return Path.cwd() / DEFAULT_RUN_DIR
    if os.access(places.default.parent, os.W_OK | os.X_OK):
        return places.default
    dataset = places.dataset
    here = Path.cwd()
    if here.resolve() == dataset or dataset in here.resolve().parents:
        raise ConfigError(
            "invalid",
            f"no place for the run directory: {dataset.parent} cannot be written to, and "
            f"the current directory lies inside the dataset {dataset}, which gets no files "
            "of a run unless asked; say where the run goes with --run-dir DIR",
        )
    return here / DEFAULT_RUN_DIR


@dataclass(frozen=True)
class _RunPlaces:
    """Where :func:`default_run_dir` looks for the run of a path."""

    default: Path
    """``surfcensus-run`` beside the path, or beside the dataset that holds it."""
    own: Path
    """``<name>-surfcensus-run`` beside the default, for a second dataset of one folder."""
    earlier: tuple[Path, ...]
    """Where a run of this root is looked for, in order: the default, the places where
    earlier versions put it, and ``own``."""
    dataset: Path | None
    """The BIDS or DataLad dataset that the run goes beside, if any."""


def _run_places(path: Path, record: DatasetRecord, implied: Sequence[str]) -> _RunPlaces:
    subjects_dir = Path(record.root) / record.fs_root
    if implied:
        # one subject directory, which may be a link: the run goes beside the directory that
        # lists it, which is the root the run stores, not beside what the link leads to
        beside = subjects_dir.parent
        name = subjects_dir.name
    else:
        beside, inside = path.resolve().parent, subjects_dir.resolve()
        name = path.resolve().name
        if beside == inside or inside in beside.parents:
            beside, name = inside.parent, inside.name
    dataset = dataset_holding(record, path) or _fmriprep_output_holding(subjects_dir, beside)
    datalad = datalad_dataset_holding(beside)
    if datalad is not None:
        # beside the path lies inside a DataLad dataset (a clone of an fMRIPrep output, say):
        # the run goes beside the outermost one, as beside a BIDS dataset
        datalad = datalad.resolve()
        if dataset is None or datalad == dataset or datalad in dataset.parents:
            dataset = datalad
    if dataset is None:
        default = beside / DEFAULT_RUN_DIR
        own = beside / f"{name}-{DEFAULT_RUN_DIR}"
        return _RunPlaces(default, own, (default, own), None)
    default = dataset.parent / DEFAULT_RUN_DIR
    own = dataset.parent / f"{dataset.name}-{DEFAULT_RUN_DIR}"
    # where the rule above put the run of this folder before datasets were known: beside
    # the path, beside the root, or (the older fMRIPrep layout named by its fmriprep
    # folder) in the root
    inside_runs = (beside, Path(record.root).parent, Path(record.root))
    earlier = (default, *(place / DEFAULT_RUN_DIR for place in inside_runs), own)
    return _RunPlaces(default, own, tuple(dict.fromkeys(earlier)), dataset)


def _fmriprep_output_holding(subjects_dir: Path, beside: Path) -> Path | None:
    """The fMRIPrep output that holds the subjects as ``sourcedata/freesurfer`` and has a
    ``dataset_description.json`` (a BIDS-derivative dataset, as fMRIPrep writes and
    OpenNeuroDerivatives publishes it), when the place beside the path lies inside it:
    ``review ds/sourcedata/freesurfer`` puts the run beside ``ds``, as for a BIDS dataset,
    also when ``ds`` was downloaded without DataLad. None otherwise."""
    try:
        inner = subjects_dir.resolve()
        output = inner.parent.parent
        if inner.parent.name != "sourcedata" or not (output / "dataset_description.json").is_file():
            return None
        place = beside.resolve()
    except OSError:
        return None
    return output if place == output or output in place.parents else None


def refuse_foreign_default(path: Path | str, run_dir: Path) -> None:
    """Raise ``ConfigError("other root", ...)`` when ``run_dir``, the default of
    :func:`default_run_dir` for ``path``, holds a local run of another root, with what to
    give instead: the stored root to go on with that run, ``--relocate`` when the stored
    root holds the same subjects in the same pipeline, and otherwise a ``--run-dir`` of
    this dataset's own (``<name>-surfcensus-run``, which :func:`default_run_dir` finds
    afterwards). Nothing happens when ``run_dir`` holds no run or a run of this root, or
    a census run, which ``review`` refuses for its own reason. The name of the default
    directory is open question O23."""
    if not (run_dir / "config.yaml").is_file():
        return
    try:
        stored = load_config(run_dir)
    except (ConfigError, OSError):
        return  # the run refuses itself, with its own reason
    roots = [s.root for s in stored.sources if s.kind == "local" and s.root is not None]
    record, implied = detect_layout(path, quiet=True)
    if stored.mode != "local" or not roots or record.root in roots:
        return
    places = _run_places(Path(path), record, implied)
    held = [subjects_held(root) for root in roots]
    here = (Path(record.root) / record.fs_root).resolve()
    old = ", ".join(roots)
    hint = other_root_hint(old, held, here, record.pipeline, f"--run-dir {places.own}")
    raise ConfigError("other root", f"{run_dir} holds the run of {old}, {hint}")


def _holds_run_of(run_dir: Path, root: str) -> bool:
    """Whether ``run_dir`` holds a local run with a source at ``root``."""
    if not (run_dir / "config.yaml").is_file():
        return False
    try:
        stored = load_config(run_dir)
    except (ConfigError, OSError):
        return False
    return stored.mode == "local" and _stored_source(stored, root) is not None


# --- the run ------------------------------------------------------------------------------------


def _n_submissions(run_dir: Path) -> int:
    """How many submissions the app database of the run holds; 0 without one."""
    path = run_dir / APP_DB_FILE
    if not path.is_file():
        return 0
    try:
        # read-write without creating: a read-only connection cannot roll back the journal
        # that a killed writer left
        uri = f"{path.resolve().as_uri()}?mode=rw"
        with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
            if "submissions" not in tables:
                return 0
            return int(db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0])
    except sqlite3.Error as exc:
        raise ConfigError(
            "invalid", f"{APP_DB_FILE} in {run_dir} cannot be read: {' '.join(str(exc).split())}"
        ) from exc


def _stored_source(stored: RunConfig, root: str) -> SourceConfig | None:
    for source in stored.sources:
        if source.kind == "local" and source.root == root:
            return source
    return None


def _open_run(
    path: Path,
    run_dir: Path,
    *,
    subjects: Sequence[str] | None,
    field_strength: float | None,
    blind: bool | None,
    require_marks: bool | None,
    video: bool | None,
    relocate: bool,
    force: bool,
    epi: bool | None = None,
    epi_runs: Sequence[str] | None = None,
    raw: Path | str | None = None,
) -> tuple[Run, bool]:
    """The run of a review, made or opened, with the options that were given written into
    its configuration; and whether it is new. An option that is None keeps what the run
    has; ``epi``, ``epi_runs`` and ``raw`` are those of ``run.epi_with``, and a glob of
    ``epi_runs`` that matches no functional run is refused before anything is stored."""
    from surfcensus.run import epi_run_stems, epi_with, refuse_unmatched_epi_runs

    if not (run_dir / "config.yaml").is_file():
        if relocate:
            raise ConfigError("missing", f"{run_dir} holds no run to relocate")
        source = LocalSource(path, subjects=subjects or (), field_strength=field_strength)
        # a glob that matches nothing stops here, before a run exists
        next(iter(source.subjects(source.dataset)), None)
        app: dict[str, object] = {}
        if blind is not None:
            app["blind"] = blind
        if require_marks is not None:
            app["require_marks"] = require_marks
        config = new_local_config(source.dataset.root, app=app, render_video=bool(video))
        config = epi_with(config, enabled=epi, runs=epi_runs, raw=raw)
        if epi_runs:
            from surfcensus.config import epi_settings

            stems = epi_run_stems([source], epi_settings(config).raw)
            refuse_unmatched_epi_runs(stems, epi_runs)
        return Run(run_dir, source, config=config), True

    stored = load_config(run_dir)
    if stored.mode != "local":
        raise ConfigError(
            "invalid",
            f"{run_dir} holds a census run; review works on a local run (surfcensus serve "
            "opens the app of a census run)",
        )
    record, implied = detect_layout(path, quiet=True)  # the source below says what it found
    known = _stored_source(stored, record.root)
    if known is None and relocate and len(stored.sources) == 1:
        known = stored.sources[0]
    globs = tuple(dict.fromkeys(subjects)) if subjects is not None else None
    # The source given is the truth for its globs and field strength (Run), so the stored
    # ones are handed back where none were given. One subject directory as path names its
    # own subject set.
    if globs is None and known is not None and not implied:
        globs = known.subjects
    if field_strength is None and known is not None:
        field_strength = known.field_strength
    source = LocalSource(path, subjects=globs or (), field_strength=field_strength)
    if subjects is not None:
        next(iter(source.subjects(source.dataset)), None)
    run = Run(run_dir, source, relocate=relocate)
    try:
        config = run.config
        app = config.app.model_dump()
        if blind is not None and blind != config.app.blind:
            rated = _n_submissions(run_dir)
            if rated and not force:
                was = "blind" if config.app.blind else "not blind"
                raise ConfigError(
                    "invalid",
                    f"this run is {was} and has {_plural(rated, 'submission')}: the order "
                    "of the queue and what a rater could see would change; pass --force "
                    "to change it all the same (the change is recorded)",
                )
            app["blind"] = blind
        if require_marks is not None:
            app["require_marks"] = require_marks
        changes: dict[str, object] = {"app": app}
        if video is not None:
            changes["render_video"] = video
        changed = with_changes(config, **changes)
        if changed != config:
            write_config(run_dir, changed)
            run.config = changed
            if changed.app.blind != config.app.blind:
                _record_blind(run, changed.app.blind)
        run.set_epi(enabled=epi, runs=epi_runs, raw=raw)
    except BaseException:
        run.close()
        raise
    return run, False


def _record_blind(run: Run, blind: bool) -> None:
    """Write the change of ``blind`` into the events of the app, when the run has an app
    database: which submissions were made under which setting must be on record."""
    if not (run.run_dir / APP_DB_FILE).is_file():
        return
    from sqlmodel import Session

    from surfcensus.app.db import app_db_mode, open_app_db
    from surfcensus.app.events import record_setting

    mode = app_db_mode(run.run_dir)
    if mode is None:
        return
    engine = open_app_db(run.run_dir, run_id=run.public().run_id(), mode=mode)
    try:
        with Session(engine) as session:
            record_setting(session, "blind", blind)
            session.commit()
    finally:
        engine.dispose()


def _refuse_in_fsaverage(run_dir: Path, fsaverage: Path | str) -> None:
    refuse_in_sources(run_dir, [], what="run", also=[("the fsaverage directory", Path(fsaverage))])


def _with_fsaverage(run: Run, fsaverage: Path | str | None) -> None:
    """The template of this call; not stored, since the run keeps the template itself. A
    run directory inside the directory given is refused: the template is only read."""
    if fsaverage is None:
        return
    _refuse_in_fsaverage(run.run_dir, fsaverage)
    metrics = run.config.metrics.model_dump() | {"fsaverage_dir": str(Path(fsaverage).resolve())}
    run.config = with_changes(run.config, metrics=metrics)


# --- what the steps say --------------------------------------------------------------------------


def _labels(run: Run) -> dict[str, str]:
    subjects = run.public().subjects()
    return {
        uid: label if isinstance(label, str) else uid
        for uid, label in zip(subjects["subject_uid"], subjects["subject_label"], strict=True)
    }


def _listed(names: Sequence[str]) -> str:
    more = f" and {len(names) - _MAX_NAMES} more" if len(names) > _MAX_NAMES else ""
    return ", ".join(names[:_MAX_NAMES]) + more


def _say_inventory(run: Run, path: Path | str | None = None) -> None:
    """``Found 48 subjects in sourcedata/freesurfer (fMRIPrep layout, FreeSurfer 7.3.2). 2
    cannot be rendered: ...``. Where the subjects lie is said relative to ``path``, the
    folder that was named, and left out when they lie in that folder itself."""
    public = run.public()
    subjects = public.subjects()
    present = subjects[subjects["present"].to_numpy(dtype=bool)]
    layouts = [LAYOUT_WORDS.get(str(name), str(name)) for name in public.datasets()["layout"]]
    versions = sorted({v for v in present["fs_version"] if isinstance(v, str)})
    described = list(dict.fromkeys(layouts))
    if versions:
        described.append("FreeSurfer " + ", ".join(versions))
    globs = [glob for source in run.config.sources for glob in source.subjects]
    if globs:
        # a subject set that is narrower than the directory is said, since it is kept
        described.append("of the subjects matching " + ", ".join(globs))
    where = where_subjects_lie(run.config, path)
    line = f"Found {_plural(len(present), 'subject')}{f' in {where}' if where else ''}"
    line += f" ({', '.join(described)})."
    lacking: list[tuple[str, str, bool]] = []  # label, what it lacks, whether not fetched
    unfetched: dict[str, object] = {}
    for row in present.sort_values("subject_label").itertuples(index=False):
        not_fetched = is_unfetched_problem(row.problem)
        if not_fetched:
            unfetched[str(row.subject_label)] = row.problem
        if bool(row.complete):
            continue
        if not_fetched:
            why = str(unfetched_said(row.problem))
        elif isinstance(row.problem, str):
            why = row.problem
        else:
            has = public.roles(str(row.subject_uid))
            why = ", ".join(
                f"no {ROLE_CANDIDATES[role][0]}" for role in RENDER_ROLES if role not in has
            )
        lacking.append((str(row.subject_label), why, not_fetched))
    if 0 < len(lacking) <= _BY_SUBJECT:
        named = ", ".join(f"{label} ({why})" for label, why, _ in lacking)
        line += f" {len(lacking)} cannot be rendered: {named}."
    elif lacking:
        # many: once per thing that is lacking, not once per subject
        by_reason: dict[str, list[str]] = {}
        for label, why, not_fetched in lacking:
            # each such subject names its own files; together they are one cause
            key = "files whose content is not fetched (git-annex)" if not_fetched else why
            by_reason.setdefault(key, []).append(label)
        grouped = "; ".join(
            f"{len(labels)} with {why} ({_listed(labels)})" for why, labels in by_reason.items()
        )
        line += f" {len(lacking)} cannot be rendered: {grouped}."
    _say(line)
    fetch = fetch_line(run.config, unfetched, path)
    if fetch is not None:
        # once, for all of them: one command, not one per subject
        named = sum(not_fetched for _, _, not_fetched in lacking)
        if len(unfetched) > named:  # complete subjects that lack Tier A or B files
            _say(
                f"{_plural(len(unfetched), 'subject')} lack files whose content is not "
                "fetched (git-annex); scores and provenance go without them until it is."
            )
        _say(fetch)


def _say_scores(run: Run) -> None:
    """One line on what the subjects were scored against."""
    from surfcensus.schemas import NO_REFERENCE

    scores = run.scores()
    if scores.empty:
        _say("Scores: none; the queue goes by subject.")
        return
    parts = []
    counts = scores.groupby(["score_kind", "reference_id"], sort=True).size()
    for key, count in counts.items():
        kind, reference = key if isinstance(key, tuple) else (key, NO_REFERENCE)
        if kind == "fallback_rank":
            parts.append(
                f"{count} ranked on ceiling fraction and occipital-to-control ratio, "
                "without a reference (decision D16)"
            )
        elif reference == NO_REFERENCE:
            parts.append(f"{count} without a z-map")
        else:
            parts.append(f"{count} against the reference {reference}")
    _say(f"Scores: {_plural(len(scores), 'subject')}; {', '.join(parts)}.")


def _say_failures(run: Run, report: PhaseReport, what: str) -> None:
    labels = _labels(run)
    by_reason: dict[str, list[str]] = {}
    for unit, reason in report.failures:
        by_reason.setdefault(reason, []).append(labels.get(unit, unit))
    for reason in sorted(by_reason):
        names = sorted(by_reason[reason])
        _say(f"  {what}, {reason}: {len(names)} ({_listed(names)})")
    for note in report.notes:
        _say(f"  {note}")


# --- the functional side (the EPI movie interface, step 2) ---------------------------------------


def _shown_place(place: Path, base: Path | None) -> str:
    """``place`` relative to ``base`` when it lies inside it (``.`` for ``base`` itself),
    else as it is."""
    if base is not None:
        if place == base:
            return "."
        if place.is_relative_to(base):
            return place.relative_to(base).as_posix()
    return str(place)


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    return f"{round(seconds / 60)} min"


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(zip(frame.columns, row, strict=True)) for row in frame.itertuples(index=False)]


def _true(value: object) -> bool:
    """A flag of a catalog frame (a bool, a numpy bool, or missing) as a bool."""
    if value is None or value is pd.NA:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def functional_lines(run: Run, path: Path | str | None = None) -> list[str]:
    """What ``review`` and ``inventory`` say about the functional runs of a run after the
    functional inventory: where the derivative and the raw dataset were found, how many runs
    of how many sessions and participants, how many can be packed and why the others cannot,
    which rule ordered the sessions of a participant with more than one (decision D42), the
    sessions that could not be planned, what the one registration of each session finds
    in the raw data (flagged runs, an anchor that is the outlier or is not packed: statements
    about the raw data, as ``epi-session`` makes them), and the commands that fetch functional
    files whose content is not there (after the surface fetch line).

    Empty for a run without functional data and for a census run: a folder that holds no
    functional data is told nothing (one line at info level). A run whose interface is off
    (``--no-epi``, stored) gets one line that says so when it has functional data."""
    from surfcensus.config import epi_settings
    from surfcensus.func.inventory import session_text
    from surfcensus.records import NOT_SELECTED

    if run.config.mode != "local":
        return []
    public = run.public()
    if not epi_settings(run.config).enabled:
        if public.has_func():
            return [
                "Functional: the EPI interface is off for this run (--no-epi, stored); "
                "--epi turns it on again."
            ]
        return []
    if not public.has_func():
        logger.info("no functional runs found (no fMRIPrep derivative with func/ series)")
        return []
    base = None if path is None else Path(path).resolve()
    participants = {
        row["participant_uid"]: row["participant"]
        for row in _records(public.func_participants())
        if _true(row["present"])
    }
    sessions = [
        row
        for row in _records(public.func_sessions())
        if _true(row["present"]) and row["participant_uid"] in participants
    ]
    session_of = {row["session_uid"]: row for row in sessions}
    runs = [
        row
        for row in _records(public.func_runs())
        if _true(row["present"]) and row["session_uid"] in session_of
    ]

    def who(row: dict[str, Any]) -> str:
        session = session_of[row["session_uid"]]
        return f"{participants[session['participant_uid']]} {row['name']}"

    places = []
    for folder in run.func_folders():
        deriv_root = Path(str(folder["deriv_root"]))
        raw_root = None if folder["raw_root"] is None else Path(str(folder["raw_root"]))
        # both folders named the same way: relative to the path when both lie in it, else
        # both as they are (review of step 2)
        common = base
        if base is not None and any(
            place is not None and not place.is_relative_to(base) for place in (deriv_root, raw_root)
        ):
            common = None
        deriv = _shown_place(deriv_root, common)
        version = folder["pipeline_version"]
        pipeline = "fMRIPrep" if folder["connector"] == "fmriprep" else str(folder["connector"])
        pipeline += f" {version}" if isinstance(version, str) else ""
        if raw_root is None:
            raw = "no raw dataset found (give it with --raw DIR)"
        else:
            raw = f"raw series in {_shown_place(raw_root, common)} (BIDS)"
        places.append(f"{deriv} ({pipeline}), {raw}")
    selected = [row for row in runs if _true(row["selected"])]
    packable = [row for row in selected if _true(row["packable"])]
    cannot = [row for row in selected if not _true(row["packable"])]
    left_out = len(runs) - len(selected)
    line = (
        f"Functional: {_plural(len(runs), 'run')} in {_plural(len(sessions), 'session')} of "
        f"{_plural(len(participants), 'participant')} in {'; '.join(places)}; "
        f"{len(packable)} can be packed (T1w space)"
    )
    if cannot:
        shown = ", ".join(f"{who(r)} ({r['reason']})" for r in cannot[:6])
        more = f" and {len(cannot) - 6} more" if len(cannot) > 6 else ""
        line += f", {len(cannot)} cannot: {shown}{more}"
    if left_out:
        line += f", {left_out} not selected (--epi-runs)"
    lines = [line + "."]
    lines += [f"  {line}" for line in run.func_order_lines()]
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_session.setdefault(row["session_uid"], []).append(row)
    for session in sessions:
        text = session_text(participants[session["participant_uid"]], session["session_label"])
        problem = session["problem"]
        if isinstance(problem, str) and problem.startswith("unreadable"):
            lines.append(f"  {text}: {problem}")
        mine = by_session.get(session["session_uid"], [])
        names = {r["run_uid"]: r["name"] for r in mine}
        anchor_uid = session["anchor_run_uid"]
        if not isinstance(anchor_uid, str):
            continue
        anchor = names.get(anchor_uid, anchor_uid)
        note = session["anchor_note"]
        if isinstance(note, str) and note:
            lines.append(f"  {text}: {note}.")
        anchor_row = next((r for r in mine if r["run_uid"] == anchor_uid), None)
        packed_any = any(_true(r["packable"]) and _true(r["selected"]) for r in mine)
        if (
            anchor_row is not None
            and packed_any
            and not (_true(anchor_row["packable"]) and _true(anchor_row["selected"]))
        ):
            reason = anchor_row["reason"] if isinstance(anchor_row["reason"], str) else NOT_SELECTED
            lines.append(
                f"  {text}: the anchor {anchor} is not packed ({reason}); the raw stream of "
                "every packed run is drawn through its stored registration."
            )
        limit = session["frame_limit_mm"]
        for r in mine:
            if not _true(r["flagged"]):
                continue
            lines.append(
                f"  {text}: flagged (raw data): {r['name']}, drawn through the registration of "
                f"the anchor {anchor}, lies up to {float(r['frame_max_mm']):.1f} mm (mean "
                f"{float(r['frame_mean_mm']):.1f} mm) from where its own stored registration "
                f"puts it, beyond the limit of {float(limit):g} mm."
            )
        if _true(session["anchor_is_outlier"]):
            lines.append(
                f"  {text}: the anchor {anchor} is the outlier (raw data): every other run is "
                "flagged against it and they agree with each other, so the anchor, or its "
                "stored registration, is the odd one."
            )
    lines += [f"  {line}" for line in run.func_fetch_lines()]
    return lines


def pack_counts(run: Run) -> tuple[int, int, int]:
    """Packable runs of the run that are packed, failed and pending now
    (``func.batch.pack_status``); zeros for a run without functional data or with the
    interface off, and for a census run."""
    from surfcensus.config import epi_settings
    from surfcensus.func.batch import pack_status

    if run.config.mode != "local" or not epi_settings(run.config).enabled:
        return 0, 0, 0
    if not run.public().has_func():
        return 0, 0, 0
    return pack_status(run.context())


def sessions_pending(run: Run) -> bool:
    """Whether a functional session is pending in the pack phase although every run's pack is
    current: a session whose inventory changed in a way that keeps its packs, such as its
    place in the stream (decision D42). Its ``session.json`` and the participant's
    ``stream.json`` are written again and nothing is packed; :func:`pack_counts`, which counts
    runs, does not see it. False where :func:`pack_counts` gives zeros."""
    from surfcensus.config import epi_settings
    from surfcensus.func.batch import pack_pending

    if run.config.mode != "local" or not epi_settings(run.config).enabled:
        return False
    if not run.public().has_func():
        return False
    return bool(pack_pending(run.context()))


def viewable(run: Run) -> int:
    """How many functional runs of the run are packed or can still be packed. With
    :func:`rateable` 0 as well, the run has nothing to show, which is when ``surfcensus
    review`` ends with exit code 1."""
    done, _, pending = pack_counts(run)
    return done + pending


def retry_hint(run: Run) -> str:
    """What tries failed packs again: a failed run is not tried again while its files stay
    as they are (the skip rule of D21)."""
    return f"tried again with: surfcensus pack --run-dir {run.run_dir} --retry-failed"


def pack_line(run: Run, *, behind: bool, after_renders: bool = False) -> str | None:
    """``EPI packs: 0 of 11 made; they are made while the app runs, after the renders (about
    20 s in one process, an estimate ...).``; None for a run without packable runs."""
    from surfcensus.func.api import RATE_S_PER_VOLUME, RATE_VOXELS
    from surfcensus.func.batch import pack_estimate

    done, failed, pending = pack_counts(run)
    total = done + failed + pending
    if total == 0:
        return None
    line = f"EPI packs: {done} of {_plural(total, 'run')} made"
    if failed:
        line += f", {failed} failed ({retry_hint(run)})"
    if pending:
        if behind:
            line += f"; {pending} are made while the app runs"
            line += ", after the renders" if after_renders else ""
        else:
            line += f"; {pending} to make"
        seconds = pack_estimate(run.context())
        if seconds > 0:
            line += (
                f" (about {_duration(seconds)} in one process, an estimate: "
                f"{RATE_S_PER_VOLUME} s per volume of both streams on {RATE_VOXELS:,} voxels, "
                "scaled by voxels)"
            )
    return line + "."


def _say_pack_failures(run: Run, report: PhaseReport) -> None:
    runs = {row["run_uid"]: row for row in _records(run.public().func_runs())}
    sessions = {row["session_uid"]: row for row in _records(run.public().func_sessions())}
    participants = {
        row["participant_uid"]: row["participant"]
        for row in _records(run.public().func_participants())
    }
    by_reason: dict[str, list[str]] = {}
    for unit, reason in report.failures:
        row = runs.get(unit)
        name = unit
        if row is not None and row["session_uid"] in sessions:
            who = participants.get(sessions[row["session_uid"]]["participant_uid"], "")
            name = f"{who} {row['name']}".strip()
        by_reason.setdefault(reason, []).append(name)
    for reason in sorted(by_reason):
        names = sorted(by_reason[reason])
        _say(f"  not packed, {reason}: {len(names)} ({_listed(names)})")
    for note in report.notes:
        _say(f"  {note}")


def _first_sessions(run: Run, wanted: int) -> list[str] | None:
    """The first ``wanted`` sessions to pack, in the order of the pack phase
    (``func.batch.session_order``), among those pending; None for all (``wanted`` 0)."""
    from surfcensus.func.batch import pack_pending, session_order

    if wanted == 0:
        return None
    context = run.context()
    pending = set(pack_pending(context))
    return [uid for uid in session_order(context) if uid in pending][:wanted]


# --- selection and render ------------------------------------------------------------------------


def _select_all(run: Run) -> list[str]:
    """Select every present subject (reason ``all``, probability 1, the seed of the
    queue); returns their uids. Incomplete subjects are in it: they get their ``missing
    surface`` render row and an unratable final label, and are never offered."""
    public = run.public()
    uids = list(public.subject_uids())
    rows = pd.DataFrame(
        {
            "subject_uid": pd.Series(uids, dtype=object),
            "reason": SELECTION_REASON,
            "stratum": SELECTION_REASON,
            "inclusion_prob": 1.0,
            "seed": run.config.app.queue_seed,
        }
    )
    stored = public.selection()
    stored = stored[stored["reason"] == SELECTION_REASON]
    same = list(stored["subject_uid"]) == sorted(uids) and bool(
        (stored["seed"] == run.config.app.queue_seed).all()
    )
    if not same:
        public.replace_selection(rows, [SELECTION_REASON])
    return uids


def _queue(run: Run, uids: Sequence[str]) -> list[str]:
    """``uids`` in the order of the rating queue, which is the order of the renders."""
    app = run.config.app
    return queue_order(uids, queue_scores(run.run_dir), blind=app.blind, seed=app.queue_seed)


def render_counts(run: Run) -> tuple[int, int, int]:
    """Renders of the run's selection that are done, failed and pending now
    (``render.batch.render_status``)."""
    from surfcensus.render.batch import render_status

    return render_status(run.context())


def rateable(run: Run) -> int:
    """How many subjects of the run have a render or can still get one. 0 means that the
    run has nothing to rate, which is when ``surfcensus review`` ends with exit code 1."""
    done, _, pending = render_counts(run)
    return done + pending


def _fetch_said(run: Run, path: Path | str | None) -> bool:
    """Whether :func:`_say_inventory` gave a command that fetches files of the run's
    present subjects whose content is not there (git-annex)."""
    subjects = run.public().subjects()
    present = subjects[subjects["present"].to_numpy(dtype=bool)]
    problems = {
        str(row.subject_label): row.problem
        for row in present.itertuples(index=False)
        if is_unfetched_problem(row.problem)
    }
    return bool(problems) and fetch_line(run.config, problems, path) is not None


def _render_first(run: Run, wanted: int, jobs: int) -> PhaseReport:
    """Render subjects in queue order until ``wanted`` renders of the selection are done or
    none is left to try; returns the last report. The incomplete subjects get their
    ``missing surface`` row in the first call, whatever it renders."""
    public = run.public()
    complete = set(public.subject_uids(complete=True))
    order = [uid for uid in _queue(run, list(public.subject_uids())) if uid in complete]
    report: PhaseReport | None = None
    position = 0
    while True:
        done, _, pending = render_counts(run)
        need = wanted - done
        if report is not None and (need <= 0 or pending == 0 or position >= len(order)):
            return report
        chunk = order[position : position + max(need, 0)]
        position += len(chunk)
        # one subject is rendered in this process: starting a worker costs more than it saves
        report = run.render(subjects=chunk, jobs=1 if len(chunk) < 2 else jobs, native=False)


# --- the app and the renderer behind it ----------------------------------------------------------


@dataclass
class ReviewApp:
    """The rater app of a review, running in a thread of this process, and the renderer
    that works behind it when subjects were still to render."""

    handle: ServerHandle
    run_dir: Path
    seconds_to_app: float
    """From the start of the call to the moment the app accepted connections."""
    epi: bool = False
    """Whether the run has functional runs that are packed or to pack: the app then offers
    the EPI index (``/epi``) beside the rating screen."""
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renderer: threading.Thread | None = field(default=None, repr=False)
    _problem: list[str] = field(default_factory=list, repr=False)
    _packing: bool = field(default=False, repr=False)

    @property
    def url(self) -> str:
        return self.handle.url

    @property
    def epi_url(self) -> str | None:
        """The address of the EPI index, when the run has functional runs to show."""
        return f"{self.handle.url.rstrip('/')}/epi" if self.epi else None

    @property
    def rendering(self) -> bool:
        """Whether the worker behind the app (the renders, then the EPI packs) is still at
        work."""
        return self._renderer is not None and self._renderer.is_alive()

    @property
    def render_problem(self) -> str | None:
        """Why the renderer behind the app gave up, if it did."""
        return self._problem[0] if self._problem else None

    def in_progress(self) -> str | None:
        """What the worker behind the app is at (``EPI packs`` or ``renders``), or None."""
        if not self.rendering:
            return None
        return "EPI packs" if self._packing else "renders"

    def wait_rendered(self, timeout: float | None = None) -> bool:
        """Wait for the renderer behind the app; false if it is still at work after
        ``timeout`` seconds."""
        if self._renderer is not None:
            self._renderer.join(timeout)
        return not self.rendering

    def stop(self) -> None:
        """Stop the server, then the renderer: the batch it is on is finished and
        recorded, the rest stays pending for the next call. A second call does nothing."""
        self.handle.stop()
        self._stop.set()
        if self.rendering:
            if self._packing:
                _say("Waiting for the EPI packs in progress; the others are made at the next call.")
            else:
                _say("Waiting for the renders in progress; the others are made at the next call.")
        if self._renderer is not None:
            self._renderer.join()


def _render_rest(
    run_dir: Path,
    jobs: int,
    fsaverage: Path | str | None,
    app: ReviewApp,
    total: int,
    render: bool = True,
    pack: bool = False,
) -> None:
    """The worker behind the app: a ``Run`` of its own renders what is pending, in queue
    order, a batch of ``jobs`` subjects at a time so that each is offered soon after it is
    made, and then packs the functional sessions that are pending, a batch of ``jobs``
    sessions at a time (the surfaces first, because a rater waits on them; D33 and the plan
    of step 2 of the EPI interface). The stop request of the app ends either between two
    batches. A failure of the run or the machine ends it and is said; the app goes on."""
    try:
        with Run(run_dir) as run:
            _with_fsaverage(run, fsaverage)
            cache = run.config.cache.model_dump() | {"batch_subjects": jobs}
            run.config = with_changes(run.config, cache=cache)
            if render:
                report = run.render(jobs=jobs, stop=app._stop.is_set)
                if app._stop.is_set():
                    return
                done, failed, _ = render_counts(run)
                _say(f"All renders are made: {done} of {_plural(total, 'subject')} can be rated.")
                if failed:
                    _say_failures(run, report, "not rendered")
            if pack:
                app._packing = True
                _pack_rest(run, jobs, app)
    except (ConfigError, SourceError, OSError, sqlite3.Error) as exc:
        problem = " ".join(str(exc).split())
        app._problem.append(problem)
        made = "the runs that are packed" if app._packing else "the subjects that are rendered"
        what = "Packing" if app._packing else "Rendering"
        _say(
            f"{what} stopped: {problem}. The app goes on with {made}; run the same "
            "command again once the cause is gone."
        )
    except Exception as exc:  # the thread must not die without a word
        app._problem.append(type(exc).__name__)
        logger.exception("the worker behind the app failed")
        what = "Packing" if app._packing else "Rendering"
        _say(f"{what} stopped on an unexpected error (see above); the app goes on.")


def _pack_rest(run: Run, jobs: int, app: ReviewApp) -> None:
    """The EPI packs behind the app, after the renders; the closing line when all are
    made."""
    report = run.pack(jobs=jobs, stop=app._stop.is_set)
    if app._stop.is_set():
        return
    done, failed, _ = pack_counts(run)
    _say(f"All EPI packs are made: {done} of {_plural(done + failed, 'run')} can be viewed.")
    if failed:
        _say_pack_failures(run, report)
        _say(f"  The failed runs are {retry_hint(run)}")


def _check(rater: str | None, jobs: int | None, serve_after: int | None, port: int) -> None:
    if rater is not None and (not rater or rater != rater.strip()):
        raise ConfigError("invalid", "a rater's name is not empty and has no space at its ends")
    if jobs is not None and jobs < 1:
        raise ConfigError("invalid", f"jobs is at least 1, not {jobs}")
    if serve_after is not None and serve_after < 0:
        raise ConfigError(
            "invalid", f"serve_after is 0 (every render first) or more, not {serve_after}"
        )
    if not 0 <= port <= 65535:
        raise ConfigError("invalid", f"a port lies between 0 and 65535, got {port}")


def open_review(
    path: Path | str,
    *,
    run_dir: Path | str | None = None,
    subjects: Sequence[str] | None = None,
    rater: str | None = None,
    blind: bool | None = None,
    require_marks: bool | None = None,
    field_strength: float | None = None,
    jobs: int | None = None,
    fsaverage: Path | str | None = None,
    video: bool | None = None,
    port: int = 8765,
    serve: bool = True,
    serve_after: int | None = None,
    relocate: bool = False,
    force: bool = False,
    epi: bool | None = None,
    epi_runs: Sequence[str] | None = None,
    raw: Path | str | None = None,
) -> tuple[Run, ReviewApp | None]:
    """Everything :func:`review` does up to the running app: the run, and the app with the
    worker behind it, which the caller stops (``ReviewApp.stop``) and whose run the caller
    closes. The app is None with ``serve=False`` and when the run has nothing to show: no
    subject can be rendered and no functional run packed (:func:`rateable` and
    :func:`viewable` are then 0). The options are those of :func:`review`.
    """
    started = time.perf_counter()
    _check(rater, jobs, serve_after, port)
    if serve:
        # before the work, not after it: an installation that lacks a package of the app
        # (a stripped or broken one) has no rater app to open
        import surfcensus.app.server  # noqa: F401
    if field_strength is not None and not 0 < field_strength < math.inf:
        raise ConfigError(
            "invalid", f"field_strength is in tesla, positive and finite, not {field_strength}"
        )
    path = Path(path)
    jobs = default_jobs() if jobs is None else jobs
    if run_dir is None:
        run_dir = default_run_dir(path)
        if not relocate:
            refuse_foreign_default(path, run_dir)  # before a run is opened there
    else:
        run_dir = Path(run_dir)
    if fsaverage is not None:
        _refuse_in_fsaverage(run_dir, fsaverage)  # before a run is made there
    run, new = _open_run(
        path,
        run_dir,
        subjects=subjects,
        field_strength=field_strength,
        blind=blind,
        require_marks=require_marks,
        video=video,
        relocate=relocate,
        force=force,
        epi=epi,
        epi_runs=epi_runs,
        raw=raw,
    )
    try:
        _say(f"Run directory: {run.run_dir} ({'new' if new else 'continued'})")
        report = run.inventory()
        _say_inventory(run, path)
        if report.failed:
            _say_failures(run, report, "not read")
        # the functional inventory reads listings and small files only (D33: the time to the
        # rating screen does not grow with the functional data); the packs come behind the app
        run.func_inventory()
        for line in functional_lines(run, path):
            _say(line)

        _with_fsaverage(run, fsaverage)
        report = run.metrics(jobs=jobs)
        _say_scores(run)
        if report.failed:
            _say_failures(run, report, "not scored")

        uids = _select_all(run)
        everything = not serve or serve_after == 0
        if everything:
            rendered = run.render(jobs=jobs)
        else:
            wanted = DEFAULT_SERVE_AFTER if serve_after is None else serve_after
            first = _render_first(run, wanted, jobs)
            done, _, pending = render_counts(run)
            # nothing left for a renderer behind the app: this call closes the phase
            rendered = run.render(jobs=jobs) if pending == 0 else first
        done, failed, pending = render_counts(run)
        line = f"Rendered: {done} of {_plural(len(uids), 'subject')}"
        if pending:
            line += f", {pending} to render while the app runs"
        if failed:
            line += f", {failed} cannot be rated"
        _say(line + ".")
        if failed:
            _say_failures(run, rendered, "not rendered")
        epi_done, _, epi_pending = pack_counts(run)
        shown = epi_done + epi_pending
        if done + pending == 0:
            line = "No subject of this run can be rendered, so there is nothing to rate."
            if _fetch_said(run, path):
                # the fetch line above is the remedy: say what comes after it
                line += " Fetch the files as said above, then run the same command again."
            if shown and serve:
                line += " The EPI movies of its functional runs are shown."
            elif shown:
                line += " Its functional runs are packed for the EPI movies."
            _say(line)
            if shown == 0:
                return run, None
            # nothing to rate: the app opens at the first packed session (serve_after counts
            # sessions then), and the rest is packed behind it
            first = (
                None
                if not serve
                else _first_sessions(
                    run, DEFAULT_SERVE_AFTER if serve_after is None else serve_after
                )
            )
            if first is None or first:
                report = run.pack(jobs=jobs, sessions=first)
                if report.failed:
                    _say_pack_failures(run, report)
        elif not serve and (epi_pending or sessions_pending(run)):
            report = run.pack(jobs=jobs)
            if report.failed:
                _say_pack_failures(run, report)
        epi_done, _, epi_pending = pack_counts(run)
        said = pack_line(run, behind=serve, after_renders=pending > 0)
        if said is not None:
            _say(said)
        if not serve:
            return run, None

        from surfcensus.app.server import start_server
        from surfcensus.protocol import install_protocol

        install_protocol(run.run_dir)
        handle = start_server(run.run_dir, port=port, rater=rater)
        app = ReviewApp(handle, run.run_dir, time.perf_counter() - started, epi=shown > 0)
        # a session can be pending with every pack current (its place in the stream moved,
        # D42): behind the app its session.json and stream.json are written again
        epi_behind = epi_pending > 0 or sessions_pending(run)
        if pending or epi_behind:
            thread = threading.Thread(
                target=_render_rest,
                args=(run.run_dir, jobs, fsaverage, app, len(uids), pending > 0, epi_behind),
                name="surfcensus-renderer",
                daemon=True,
            )
            app._renderer = thread
            thread.start()
        return run, app
    except BaseException:
        run.close()
        raise


def _progress_of(run_dir: Path, rater: str | None) -> tuple[int, int] | None:
    """Subjects the rater has rated and has still to rate, from the database of the app."""
    from sqlmodel import Session, col, func, select

    from surfcensus.app.db import app_db_mode, open_app_db
    from surfcensus.app.models import Assignment, Rater
    from surfcensus.app.queue import build_plan, subjects_left
    from surfcensus.store import PublicCatalog

    config = load_config(run_dir)
    mode = app_db_mode(run_dir)
    if mode is None:
        return None
    name = config.app.rater_name if rater is None else rater
    with PublicCatalog.open(run_dir) as public:
        plan = build_plan(
            public,
            queue_scores(run_dir),
            config.app,
            "local",
            render_config_hash=config.render.config_hash(),
        )
        engine = open_app_db(run_dir, run_id=public.run_id(), mode=mode)
    try:
        with Session(engine) as session:
            found = session.exec(select(Rater).where(Rater.name == name)).first()
            if found is None:
                return None
            rated = session.exec(
                select(func.count(func.distinct(Assignment.subject_uid))).where(
                    Assignment.rater_id == found.rater_id,
                    Assignment.kind == "main",
                    col(Assignment.completed_at).is_not(None),
                )
            ).one()
            return int(rated), subjects_left(session, found, plan)
    finally:
        engine.dispose()


def review(
    path: Path | str,
    *,
    run_dir: Path | str | None = None,
    subjects: Sequence[str] | None = None,
    rater: str | None = None,
    blind: bool | None = None,
    require_marks: bool | None = None,
    field_strength: float | None = None,
    jobs: int | None = None,
    fsaverage: Path | str | None = None,
    video: bool | None = None,
    port: int = 8765,
    open_browser: bool = True,
    serve: bool = True,
    serve_after: int | None = None,
    relocate: bool = False,
    force: bool = False,
    epi: bool | None = None,
    epi_runs: Sequence[str] | None = None,
    raw: Path | str | None = None,
) -> Run:
    """Inventory, score, select and render the subjects under ``path`` and open the rater
    app on them; returns the run.

    The functional runs of the folder (an fMRIPrep derivative with its raw BIDS dataset,
    found by rule, docs/epi_interface.md) are listed after the surface inventory and packed
    into ``epi/`` behind the app, after the remaining renders; the app then offers the EPI
    index at ``/epi``. ``epi`` (False: ``--no-epi``), ``epi_runs`` (``--epi-runs``) and
    ``raw`` (``--raw``) are stored and hold until given again. When no subject can be
    rendered and a run can be packed, the app opens when the first ``serve_after`` sessions
    are packed; ``serve=False`` packs every session as well.

    Every option that maps to a stored value is tri-state: None keeps what the run has, and
    the defaults (not blind, marks not required, no video) apply only when the run is
    created. ``rater`` is not stored: it names the rater of this server process.

    The app opens when the first ``serve_after`` subjects of the queue are rendered, and
    the others are rendered while it runs, in the order of the queue
    (``store.queue_order``). None is 1, so the rating screen comes up at the first render
    (decision D33); 0 renders every subject first. ``serve=False`` renders every subject
    and stops. ``force`` is needed to change ``blind`` once a rating exists. Running it
    again does the same steps with everything unchanged skipped.

    With ``serve`` the call returns after Ctrl-C or SIGTERM: the server is stopped, then
    the renderer, which finishes the batch it is on. When no subject can be rendered there
    is nothing to serve and the call returns at once; :func:`rateable` of the run is then
    0. Raises ``ConfigError`` and ``SourceError`` for what stops the phase commands.
    """
    run, app = open_review(
        path,
        run_dir=run_dir,
        subjects=subjects,
        rater=rater,
        blind=blind,
        require_marks=require_marks,
        field_strength=field_strength,
        jobs=jobs,
        fsaverage=fsaverage,
        video=video,
        port=port,
        serve=serve,
        serve_after=serve_after,
        relocate=relocate,
        force=force,
        epi=epi,
        epi_runs=epi_runs,
        raw=raw,
    )
    if app is None:
        return run
    # SIGTERM stops the app as Ctrl-C does, from before the app is announced until the
    # closing line is out: a second one (a process manager that signals the whole group)
    # must not end the call in the middle of stopping. The handler sets a flag and takes
    # no lock (``stopping.StopFlag``): a signal that arrives while a handler holds a lock
    # would wait for it for good. The handler that was there is put back when the call
    # returns, as a function owes its caller. A process that is about to end must not do
    # that (see :func:`serve_review`).
    stop = StopFlag()
    in_main = threading.current_thread() is threading.main_thread()
    earlier: Any = signal.signal(signal.SIGTERM, lambda *_: stop.set()) if in_main else None
    try:
        serve_review(run, app, rater=rater, open_browser=open_browser, stop=stop)
    finally:
        if in_main and earlier is not None:
            signal.signal(signal.SIGTERM, earlier)
    return run


def serve_review(
    run: Run,
    app: ReviewApp,
    *,
    rater: str | None = None,
    open_browser: bool = True,
    stop: StopRequest | None = None,
    stopped: Callable[[], None] | None = None,
) -> None:
    """The second half of :func:`review`, for the run and the running app that
    :func:`open_review` returned: name the address, open the browser, wait, stop the app
    and say how far the rater got.

    The wait ends at Ctrl-C, when the server ends, and when ``stop`` is set (a
    ``stopping.StopFlag``, which a signal handler can set without a lock, or a
    ``threading.Event`` of a caller). The call sets no signal handler: whoever wants
    SIGTERM to stop the app sets ``stop`` from a handler of its own, as :func:`review` does
    for the length of its call. The command line ignores SIGTERM from the first one on,
    and SIGINT too once this call has returned, because a SIGTERM that comes late (a
    process manager that signals a whole group, a wrapper that passes the signal on) would
    otherwise kill a command that had stopped in order (``stopping``). The server is
    stopped first, then the renderer, which finishes the batch it is on. ``stopped`` is
    called once both have stopped, before the closing line: the command line ignores both
    signals from there on, so that no Ctrl-C lands between the closing line and the end.
    """
    stop = StopFlag() if stop is None else stop
    try:
        _say(f"Rating app: {app.url} (Ctrl-C to stop)")
        if app.epi_url is not None:
            _say(f"EPI movies: {app.epi_url}")
        _say(f"  {app.seconds_to_app:.1f} s from the start of the command to the app.")
        # no subject to rate and EPI movies to view: the browser goes to the movies (review
        # of step 2)
        opened = app.url
        if app.epi_url is not None and rateable(run) == 0 and viewable(run) > 0:
            opened = app.epi_url
            _say("No subject can be rated; the browser opens the EPI movies.")
        if open_browser:
            with suppress(webbrowser.Error):
                webbrowser.open(opened)
        with suppress(KeyboardInterrupt):
            while app.handle.running and not stop.wait(0.2):
                pass
    finally:
        app.stop()
        if stopped is not None:
            stopped()  # a second Ctrl-C up to here ended the command (130, no closing line)
    try:
        progress = _progress_of(run.run_dir, rater)
    except Exception:  # a closing line is not worth a traceback after the work is saved
        logger.debug("no closing line", exc_info=True)
        progress = None
    if progress is not None:
        rated, left = progress
        _say(
            f"{rated} rated, {left} to go. Run the same command to continue. "
            f"Export: surfcensus export --run-dir {run.run_dir}"
        )
    if app.epi:
        try:
            done, failed, pending = pack_counts(run)
        except Exception:  # as above: the work is saved
            logger.debug("no EPI closing line", exc_info=True)
        else:
            line = f"EPI packs: {done} of {_plural(done + failed + pending, 'run')} made"
            if failed:
                line += f", {failed} failed ({retry_hint(run)})"
            if pending:
                line += f"; {pending} are made at the next call"
            _say(line + ".")
