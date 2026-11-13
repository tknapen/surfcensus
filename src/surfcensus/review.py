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
from collections.abc import Sequence
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
from surfcensus.run import Run
from surfcensus.sources.base import RENDER_ROLES, SourceError
from surfcensus.sources.local import LocalSource, detect_layout, refuse_in_sources
from surfcensus.sources.patterns import ROLE_CANDIDATES
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
_LAYOUTS = {
    "subjects_dir": "a subjects directory",
    "fmriprep": "fMRIPrep layout",
    "fmriprep_legacy": "older fMRIPrep layout",
    "bids_derivatives": "BIDS derivatives layout",
}


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
    is a link to another place. Raises ``SourceError`` as
    :func:`surfcensus.sources.local.detect_layout` does.
    """
    record, implied = detect_layout(path)
    subjects_dir = Path(record.root) / record.fs_root
    if implied:
        # one subject directory, which may be a link: the run goes beside the directory that
        # lists it, which is the root the run stores, not beside what the link leads to
        beside = subjects_dir.parent
    else:
        beside, inside = Path(path).resolve().parent, subjects_dir.resolve()
        if beside == inside or inside in beside.parents:
            beside = inside.parent
    candidate = beside / DEFAULT_RUN_DIR
    if (candidate / "config.yaml").is_file() or os.access(beside, os.W_OK | os.X_OK):
        return candidate
    return Path.cwd() / DEFAULT_RUN_DIR


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
) -> tuple[Run, bool]:
    """The run of a review, made or opened, with the options that were given written into
    its configuration; and whether it is new. An option that is None keeps what the run
    has."""
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
        return Run(run_dir, source, config=config), True

    stored = load_config(run_dir)
    if stored.mode != "local":
        raise ConfigError(
            "invalid",
            f"{run_dir} holds a census run; review works on a local run (surfcensus serve "
            "opens the app of a census run)",
        )
    record, implied = detect_layout(path)
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


def _say_inventory(run: Run) -> None:
    """``Found 48 subjects (fMRIPrep layout, FreeSurfer 7.3.2). 2 cannot be rendered: ...``"""
    public = run.public()
    subjects = public.subjects()
    present = subjects[subjects["present"].to_numpy(dtype=bool)]
    layouts = [_LAYOUTS.get(str(name), str(name)) for name in public.datasets()["layout"]]
    versions = sorted({v for v in present["fs_version"] if isinstance(v, str)})
    described = list(dict.fromkeys(layouts))
    if versions:
        described.append("FreeSurfer " + ", ".join(versions))
    globs = [glob for source in run.config.sources for glob in source.subjects]
    if globs:
        # a subject set that is narrower than the directory is said, since it is kept
        described.append("of the subjects matching " + ", ".join(globs))
    line = f"Found {_plural(len(present), 'subject')} ({', '.join(described)})."
    lacking: list[tuple[str, str]] = []
    for row in present.sort_values("subject_label").itertuples(index=False):
        if bool(row.complete):
            continue
        if isinstance(row.problem, str):
            why = row.problem
        else:
            has = public.roles(str(row.subject_uid))
            why = ", ".join(
                f"no {ROLE_CANDIDATES[role][0]}" for role in RENDER_ROLES if role not in has
            )
        lacking.append((str(row.subject_label), why))
    if 0 < len(lacking) <= _BY_SUBJECT:
        named = ", ".join(f"{label} ({why})" for label, why in lacking)
        line += f" {len(lacking)} cannot be rendered: {named}."
    elif lacking:
        # many: once per thing that is lacking, not once per subject
        by_reason: dict[str, list[str]] = {}
        for label, why in lacking:
            by_reason.setdefault(why, []).append(label)
        grouped = "; ".join(
            f"{len(labels)} with {why} ({_listed(labels)})" for why, labels in by_reason.items()
        )
        line += f" {len(lacking)} cannot be rendered: {grouped}."
    _say(line)


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
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renderer: threading.Thread | None = field(default=None, repr=False)
    _problem: list[str] = field(default_factory=list, repr=False)

    @property
    def url(self) -> str:
        return self.handle.url

    @property
    def rendering(self) -> bool:
        """Whether the renderer behind the app is still at work."""
        return self._renderer is not None and self._renderer.is_alive()

    @property
    def render_problem(self) -> str | None:
        """Why the renderer behind the app gave up, if it did."""
        return self._problem[0] if self._problem else None

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
            _say("Waiting for the renders in progress; the others are made at the next call.")
        if self._renderer is not None:
            self._renderer.join()


def _render_rest(
    run_dir: Path, jobs: int, fsaverage: Path | str | None, app: ReviewApp, total: int
) -> None:
    """The renderer behind the app: a ``Run`` of its own renders what is pending, in queue
    order, a batch of ``jobs`` subjects at a time so that each is offered soon after it is
    made. A failure of the run or the machine ends it and is said; the app goes on."""
    try:
        with Run(run_dir) as run:
            _with_fsaverage(run, fsaverage)
            cache = run.config.cache.model_dump() | {"batch_subjects": jobs}
            run.config = with_changes(run.config, cache=cache)
            report = run.render(jobs=jobs, stop=app._stop.is_set)
            if app._stop.is_set():
                return
            done, failed, _ = render_counts(run)
            _say(f"All renders are made: {done} of {_plural(total, 'subject')} can be rated.")
            if failed:
                _say_failures(run, report, "not rendered")
    except (ConfigError, SourceError, OSError, sqlite3.Error) as exc:
        problem = " ".join(str(exc).split())
        app._problem.append(problem)
        _say(
            f"Rendering stopped: {problem}. The app goes on with the subjects that are "
            "rendered; run the same command again once the cause is gone."
        )
    except Exception as exc:  # the thread must not die without a word
        app._problem.append(type(exc).__name__)
        logger.exception("the renderer behind the app failed")
        _say("Rendering stopped on an unexpected error (see above); the app goes on.")


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
) -> tuple[Run, ReviewApp | None]:
    """Everything :func:`review` does up to the running app: the run, and the app with the
    renderer behind it, which the caller stops (``ReviewApp.stop``) and whose run the
    caller closes. The app is None with ``serve=False`` and when no subject of the run can
    be rendered (:func:`rateable` is then 0). The options are those of :func:`review`.
    """
    started = time.perf_counter()
    _check(rater, jobs, serve_after, port)
    if serve:
        # before the work, not after it: a core install has no rater app to open
        import surfcensus.app.server  # noqa: F401
    if field_strength is not None and not 0 < field_strength < math.inf:
        raise ConfigError(
            "invalid", f"field_strength is in tesla, positive and finite, not {field_strength}"
        )
    path = Path(path)
    jobs = default_jobs() if jobs is None else jobs
    run_dir = default_run_dir(path) if run_dir is None else Path(run_dir)
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
    )
    try:
        _say(f"Run directory: {run.run_dir} ({'new' if new else 'continued'})")
        report = run.inventory()
        _say_inventory(run)
        if report.failed:
            _say_failures(run, report, "not read")

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
        if done + pending == 0:
            _say("No subject of this run can be rendered, so there is nothing to rate.")
            return run, None
        if not serve:
            return run, None

        from surfcensus.app.server import start_server
        from surfcensus.protocol import install_protocol

        install_protocol(run.run_dir)
        handle = start_server(run.run_dir, port=port, rater=rater)
        app = ReviewApp(handle, run.run_dir, time.perf_counter() - started)
        if pending:
            thread = threading.Thread(
                target=_render_rest,
                args=(run.run_dir, jobs, fsaverage, app, len(uids)),
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
) -> Run:
    """Inventory, score, select and render the subjects under ``path`` and open the rater
    app on them; returns the run.

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
    )
    if app is None:
        return run
    # SIGTERM stops the app as Ctrl-C does, from before the app is announced until the
    # closing line is out: a second one (a process manager that signals the whole group)
    # must not end the command in the middle of stopping.
    stop = threading.Event()
    in_main = threading.current_thread() is threading.main_thread()
    earlier: Any = signal.signal(signal.SIGTERM, lambda *_: stop.set()) if in_main else None
    try:
        try:
            _say(f"Rating app: {app.url} (Ctrl-C to stop)")
            _say(f"  {app.seconds_to_app:.1f} s from the start of the command to the app.")
            if open_browser:
                with suppress(webbrowser.Error):
                    webbrowser.open(app.url)
            with suppress(KeyboardInterrupt):
                while app.handle.running and not stop.wait(0.2):
                    pass
        finally:
            app.stop()
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
    finally:
        if in_main and earlier is not None:
            signal.signal(signal.SIGTERM, earlier)
    return run
