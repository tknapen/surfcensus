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
    CacheConfig,
    ConfigError,
    load_config,
    new_local_config,
    with_changes,
    write_config,
)
from surfcensus.log import get_logger
from surfcensus.remote import RemoteRef, parse_remote_path
from surfcensus.run import Run, other_root_hint, size_text
from surfcensus.sources.base import (
    RENDER_ROLES,
    DatasetRecord,
    FileRole,
    SourceError,
    SubjectRecord,
)
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
    import httpx

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
        _apply_options(run, blind=blind, require_marks=require_marks, video=video, force=force)
        run.set_epi(enabled=epi, runs=epi_runs, raw=raw)
    except BaseException:
        run.close()
        raise
    return run, False


def _apply_options(
    run: Run,
    *,
    blind: bool | None,
    require_marks: bool | None,
    video: bool | None,
    force: bool,
) -> None:
    """Write the options of a call that were given into the configuration of an existing
    run; None keeps what the run has. ``blind`` cannot change once the run has a submission
    unless ``force``, and a change is recorded in the events of the app."""
    config = run.config
    app = config.app.model_dump()
    if blind is not None and blind != config.app.blind:
        rated = _n_submissions(run.run_dir)
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
        write_config(run.run_dir, changed)
        run.config = changed
        if changed.app.blind != config.app.blind:
            _record_blind(run, changed.app.blind)


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


# --- a dataset of the OpenNeuroDerivatives bucket (docs/dev/remote_plan.md, section 4) ----------

REMOTE_RETRY_S = 30.0
"""The renderer behind the app tries a bucket that cannot be reached again after this many
seconds, for as long as the app runs."""

REMOTE_GUARD_BYTES = 2_000_000_000
"""A new review of a remote dataset that names no subjects (``--subjects``) stops when the
estimate of what it fetches before the first rating screen exceeds this, and asks for
``--subjects`` (decision D47, item 3): about 135 subjects of 14 MB."""

_RENDER_SPHERES = (FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)


def _refuse_for_remote(
    *,
    field_strength: float | None,
    relocate: bool,
    epi: bool | None,
    epi_runs: Sequence[str] | None,
    raw: Path | str | None,
) -> None:
    """The options that a review of a remote dataset does not take, each refused with one
    line (decision D47, item 7; since decision D50 ``--epi``, ``--no-epi`` and ``--epi-runs``
    are taken, and ``--raw`` alone is refused)."""
    if relocate:
        raise ConfigError("invalid", "a remote dataset does not move; --relocate is for local data")
    if raw is not None:
        raise ConfigError(
            "invalid",
            "--raw: a review of a remote dataset reads the raw series of its dataset from "
            "s3://openneuro.org (with --epi); a raw folder is not taken",
        )
    if field_strength is not None:
        raise ConfigError(
            "invalid",
            "--field-strength is not taken for a remote dataset; its field strength comes "
            "from the raw dataset once enrichment is built",
        )


@dataclass(frozen=True)
class RemoteEstimate:
    """What a review of a remote dataset will fetch, from the sizes of the listing alone.

    The figures are sums of listed sizes: the Tier A and Tier B files of every subject, the
    ``fsaverage`` next to the subjects when the run will take its template from there, and
    the render files (Tier C and, with a template, both ``sphere.reg`` files, which the
    metrics fetched once already: decision D47, item 6) of the subjects rendered before the
    app opens and of all. Files of this run's cache are left out."""

    subjects: int
    tier_a: int
    tier_b: int
    template: int
    first: int
    """Bytes of the renders before the first rating screen."""
    n_first: int
    renders: int
    """Bytes of every render."""
    peak: int
    """The source files on disk at once at most: the kept Tier A files and template, and the
    larger of one batch of the metrics and one of the renders."""
    everything: bool = False
    """The call renders every subject before it ends (no app, or ``--serve-after 0``)."""
    func: int = 0
    """Bytes of the functional files of the EPI interface (``--epi``, decision D50): fetched
    after the surface inventory and before the metrics, so before the first rating screen,
    and kept in the retained cache."""
    func_subjects: int = 0
    """The participants those files are of."""

    @property
    def before_app(self) -> int:
        return self.tier_a + self.tier_b + self.template + self.func + self.first

    @property
    def in_all(self) -> int:
        return self.tier_a + self.tier_b + self.template + self.func + self.renders


def _sum_roles(record: SubjectRecord, roles: Sequence[FileRole], skip: Callable[..., bool]) -> int:
    return sum(
        ref.size for role, ref in record.files.items() if role in roles and not skip(record, ref)
    )


def remote_estimate(
    source: Any,
    dataset: DatasetRecord,
    records: Sequence[SubjectRecord],
    *,
    template_needed: bool,
    n_first: int | None,
    jobs: int,
    batch_subjects: int,
    max_bytes: int,
    epi: bool = False,
) -> RemoteEstimate:
    """:class:`RemoteEstimate` of ``records``, the listed subjects of ``dataset`` of the
    remote ``source``, for a run that holds nothing yet. ``template_needed``: the run takes
    its template from the ``fsaverage`` next to the subjects; ``n_first``: the renders the
    app waits for (None: every one, a call that renders everything before it ends); ``epi``:
    the EPI interface is on, and the functional files of the selected participants are
    counted (``func_size`` of the source, which lists them)."""
    from surfcensus.sources.base import TIER_A, TIER_B, TIER_C

    cache = getattr(source, "cache", None)

    def kept(record: SubjectRecord, ref: Any) -> bool:
        # a retained copy of this version is not fetched again
        if cache is None:
            return False
        base = dataset.root.removeprefix(f"s3://{source.bucket}/")
        key = f"{base}{record.fs_dir}/{ref.location}"
        path = cache.path_for(source.bucket, key, ref.etag_or_mtime, retained=True)
        try:
            return path.is_file() and path.stat().st_size == ref.size
        except OSError:
            return False

    def never(record: SubjectRecord, ref: Any) -> bool:
        return False

    # with the EPI interface on, the FreeSurfer files of the contours are retained with the
    # functional files, and the render leases those copies without a request (decision D50)
    shared: set[str] = set()
    listing = getattr(source, "func_listing", None) if epi else None
    if listing is not None:
        found = listing(dataset)
        if found is not None:
            shared = {obj.key for kind, obj in found.files.values() if kind == "deriv"}

    def in_tree(record: SubjectRecord, ref: Any) -> bool:
        base = dataset.root.removeprefix(f"s3://{source.bucket}/")
        return f"{base}{record.fs_dir}/{ref.location}" in shared

    template = 0
    if template_needed:
        template = int(source.template_size(dataset) or 0)
    # a render leases both sphere.reg files whenever the run has a template: the one next
    # to the subjects, or one that the run holds or finds elsewhere (not needed from here)
    with_spheres = not template_needed or template > 0
    render_roles = (*TIER_C, *_RENDER_SPHERES) if with_spheres else TIER_C
    tier_a = sum(_sum_roles(r, TIER_A, kept) for r in records)
    tier_bs = [_sum_roles(r, TIER_B, never) for r in records]
    complete = [r for r in records if all(role in r.files for role in RENDER_ROLES)]
    renders = sorted((_sum_roles(r, render_roles, in_tree) for r in complete), reverse=True)
    first_n = len(renders) if n_first is None else min(n_first, len(renders))
    largest_b = max(tier_bs, default=0)
    batch = min(batch_subjects, max(1, max_bytes // largest_b)) if largest_b else batch_subjects
    peak_b = largest_b * min(batch, len(records))
    # the renders leased at once: a call that renders everything before it ends takes
    # batches of ``batch_subjects``; otherwise the renders before the app are one batch and
    # those behind it come ``jobs`` at a time; each is lowered to fit ``max_bytes``, as
    # ``Run`` lowers it
    largest_c = renders[0] if renders else 0
    fits = min(batch_subjects, max(1, max_bytes // largest_c)) if largest_c else batch_subjects
    at_once = fits if n_first is None else max(min(first_n, fits), min(jobs, fits))
    peak_c = largest_c * min(at_once, len(renders))
    func = func_subjects = 0
    if epi and getattr(source, "func_size", None) is not None:
        func, func_subjects = source.func_size(dataset)
    return RemoteEstimate(
        subjects=len(records),
        tier_a=tier_a,
        tier_b=sum(tier_bs),
        template=template,
        first=sum(renders[:first_n]),
        n_first=first_n,
        renders=sum(renders),
        peak=tier_a + template + func + max(peak_b, peak_c),
        everything=n_first is None,
        func=func,
        func_subjects=func_subjects,
    )


def estimate_line(estimate: RemoteEstimate, *, app: bool = True) -> str:
    """The line of a new run: ``To fetch: 108.2 MB before the first rating screen (...),
    148.0 MB in all, ...``; for a call without the app, what it fetches in all."""
    what = [f"Tier A and B of {_plural(estimate.subjects, 'subject')}"]
    if estimate.template:
        what.append("the fsaverage template")
    if estimate.func:
        what.append("the functional files")
    if estimate.n_first:
        what.append(_plural(estimate.n_first, "render"))
    if app:
        head = (
            f"To fetch: {size_text(estimate.before_app)} before the first rating screen "
            f"({', '.join(what)}), {size_text(estimate.in_all)} in all"
        )
    else:
        head = f"To fetch: {size_text(estimate.in_all)} in all ({', '.join(what)})"
    head += func_words(estimate.func, estimate.func_subjects, app=app)
    return (
        f"{head}, an estimate from the listed sizes; the files of each batch are removed after "
        f"it, and at most {size_text(estimate.peak)} of source files are on disk at once."
    )


def func_words(size: int, participants: int, *, app: bool) -> str:
    """``, 115.3 MB of it the functional files of 1 subject for the EPI movies, fetched
    before the metrics`` (decision D50); empty without functional bytes."""
    if not size:
        return ""
    when = ", fetched before the metrics, so before the first rating screen" if app else ""
    return (
        f", {size_text(size)} of it the functional files of "
        f"{_plural(participants, 'subject')} for the EPI movies{when}"
    )


def continued_line(
    run: Run,
    *,
    template_bytes: int,
    n_first: int | None,
    app: bool,
    func: tuple[int, int] = (0, 0),
) -> str:
    """The line of a continued run, said after its inventory: what ``metrics`` and
    ``render`` would lease now by their own plans (``Run.remote_pending``), so that a map or
    a render that a changed file or setting makes pending again is counted, and what the run
    holds is not. ``n_first``: the renders the app waits for in all (None: every one)."""
    maps, renders, exact = run.remote_pending()
    tier_b, in_renders = sum(maps.values()), sum(renders.values())
    func_bytes, func_subjects = func
    in_all = template_bytes + func_bytes + tier_b + in_renders
    if in_all == 0:
        return (
            "To fetch: nothing more; the maps and the renders of this run are those of the "
            "files listed and of its settings."
        )
    what = []
    if template_bytes:
        what.append("the fsaverage template")
    if func_bytes:
        what.append("the functional files")
    if maps:
        what.append(f"Tier B of {_plural(len(maps), 'subject')}")
    if renders:
        what.append(_plural(len(renders), "render"))
    line = f"To fetch: {'' if exact else 'at most '}{size_text(in_all)} in all ({', '.join(what)})"
    if app and renders:
        done = render_counts(run)[0]
        waited = len(renders) if n_first is None else min(max(n_first - done, 0), len(renders))
        first = sum(sorted(renders.values(), reverse=True)[:waited])
        before = template_bytes + func_bytes + tier_b + first
        line += f", {size_text(before)} of it before the rating screen"
    line += func_words(func_bytes, func_subjects, app=app)
    return line + ", an estimate from the listed sizes; what this run holds is not fetched again."


def _template_needed(run_dir: Path, stored: RunConfig | None, fsaverage: Path | str | None) -> bool:
    """Whether a review will take its template from the ``fsaverage`` next to the subjects:
    not when a directory is given, when ``$FREESURFER_HOME`` has one, or when the run holds
    its pinned template (``Run._find_template``)."""
    from surfcensus.sources.local import freesurfer_template_places
    from surfcensus.templates import TEMPLATES_DIR

    if fsaverage is not None:
        return False
    if any(place.is_dir() for _, place in freesurfer_template_places()):
        return False
    if stored is not None:
        found = [p for p in (run_dir / TEMPLATES_DIR).glob("*/template.json")]
        if found:
            return False
    return True


def _cache_bytes(run_dir: Path) -> int:
    """The bytes of the source files in the run's cache (its lock files left out)."""
    from surfcensus.remote.cache import CACHE_DIR, LOCK_FILE

    total = 0
    seen: set[tuple[int, int]] = set()  # the template's tree is hard links to retained files
    for directory, _, files in os.walk(run_dir / CACHE_DIR):
        for name in files:
            if name == LOCK_FILE:
                continue
            with suppress(OSError):
                found = os.lstat(os.path.join(directory, name))
                if (found.st_dev, found.st_ino) not in seen:
                    seen.add((found.st_dev, found.st_ino))
                    total += found.st_size
    return total


def _fetched_since(start: Any) -> tuple[int, int]:
    """Bytes and requests that this process sent to the buckets since ``start``
    (``remote.stats()`` at the start of the call)."""
    from surfcensus import remote

    now = remote.stats()
    return now.bytes - start.bytes, now.requests - start.requests


def _say_fetched(run: Run, start: Any) -> None:
    """The closing line of a review of a remote dataset."""
    fetched, requests = _fetched_since(start)
    _say(
        f"Fetched {size_text(fetched)} in {_plural(requests, 'request')} in all; "
        f"{size_text(_cache_bytes(run.run_dir))} of source files left in the cache."
    )


def _open_remote_run(
    ref: RemoteRef,
    run_dir: Path,
    *,
    subjects: Sequence[str] | None,
    blind: bool | None,
    require_marks: bool | None,
    video: bool | None,
    force: bool,
    transport: httpx.BaseTransport | None,
    serve: bool,
    serve_after: int | None,
    jobs: int,
    fsaverage: Path | str | None,
    epi: bool | None = None,
    epi_runs: Sequence[str] | None = None,
) -> tuple[Run, bool, int]:
    """The run of a review of the remote dataset ``ref``, made or opened, with the options
    given written into it, whether it is new, and the bytes of the ``fsaverage`` next to
    the subjects when the run will fetch it. The dataset and its subjects are listed first.
    For a new run, what will be fetched is said from the listed sizes before any file is
    read, and a run that names no subjects is refused when that exceeds
    :data:`REMOTE_GUARD_BYTES` (decision D47, item 3), before anything is written. For a
    continued run the caller says it after the inventory (:func:`continued_line`)."""
    from surfcensus.config import RemoteConfig, epi_settings
    from surfcensus.remote import RAW_BUCKET
    from surfcensus.remote.s3 import S3Client
    from surfcensus.run import remote_index
    from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives

    stored = load_config(run_dir) if (run_dir / "config.yaml").is_file() else None
    if stored is not None and stored.mode != "local":
        raise ConfigError(
            "invalid",
            f"{run_dir} holds a census run; review works on a local run (surfcensus serve "
            "opens the app of a census run)",
        )
    index = None if stored is None else remote_index(stored, ref.accession)
    if subjects is not None:
        globs = tuple(dict.fromkeys(subjects))
    elif stored is not None and index is not None:
        globs = stored.sources[index].subjects
    else:
        globs = ()
    settings = stored.remote if stored is not None else RemoteConfig()
    client = S3Client(ref.bucket, config=settings, transport=transport)
    # the raw bucket of the EPI interface (decision D50); asked only when it is on
    raw_client = S3Client(RAW_BUCKET, config=settings, transport=transport)
    source = OpenNeuroDerivatives(
        client=client, accession=ref.accession, subjects=globs, raw_client=raw_client
    )
    # the EPI interface of this call: as given, else as the run has it (off for a new run)
    epi_on = (
        bool(epi)
        if epi is not None
        else (stored is not None and index is not None and epi_settings(stored).enabled)
    )
    run: Run | None = None
    if stored is not None:
        # another source's run is "other root" here, before anything is listed
        run = Run(run_dir, source, transport=transport)
    try:
        if run is not None:
            _apply_options(run, blind=blind, require_marks=require_marks, video=video, force=force)
        if Path(ref.text).is_dir():
            # decision D47, item 8: the text is the remote dataset also where a folder has
            # that name
            _say(
                f"A folder named {ref.text} is here as well; this reads the dataset of the "
                f"bucket. To review the folder, give it as ./{ref.text}"
            )
        _say(f"Listing {ref.text} in s3://{ref.bucket}, anonymously and read-only ...")
        try:
            dataset = next(iter(source.datasets()))
            records = list(source.subjects(dataset))
        except SourceError as exc:
            if exc.reason != "unreachable":
                raise
            raise SourceError(
                "unreachable",
                f"the bucket could not be reached ({exc.detail}); nothing was changed. "
                + _offline_remedy(run, run_dir),
            ) from exc
        config = run.config if run is not None else None
        cache_settings = config.cache if config is not None else CacheConfig()
        new_run = run is None
        wanted = None if not serve or serve_after == 0 else serve_after or DEFAULT_SERVE_AFTER
        needed = _template_needed(run_dir, stored, fsaverage)
        estimate = remote_estimate(
            source,
            dataset,
            records,
            template_needed=needed,
            n_first=wanted,
            jobs=jobs,
            batch_subjects=cache_settings.batch_subjects,
            max_bytes=cache_settings.max_bytes,
            epi=epi_on and new_run,
        )
        new = run is None
        if epi_on and epi_runs:
            from surfcensus.run import refuse_unmatched_epi_runs

            refuse_unmatched_epi_runs(source.func_stems(dataset), epi_runs)
        if new and subjects is None and estimate.before_app > REMOTE_GUARD_BYTES:
            before = " before the first rating screen" if serve else ""
            raise ConfigError(
                "invalid",
                f"this dataset lists {_plural(estimate.subjects, 'subject')}, about "
                f"{size_text(estimate.before_app)} to fetch{before}, more "
                f"than {size_text(REMOTE_GUARD_BYTES)}: name the subjects to review with "
                "--subjects GLOB (for example --subjects 'sub-0[1-5]'), or --subjects '*' "
                "for all of them",
            )
        if run is None:
            app: dict[str, object] = {}
            if blind is not None:
                app["blind"] = blind
            if require_marks is not None:
                app["require_marks"] = require_marks
            base = new_local_config(None, app=app, render_video=bool(video))
            run = Run(run_dir, source, config=base, transport=transport)
        if epi is not None or epi_runs is not None:
            # stored as for a folder (decision D50): --epi turns the interface on, --no-epi
            # off again, --epi-runs selects what is packed; a glob that matches no
            # functional run of the selected participants is refused before it is stored
            run.set_epi(enabled=epi, runs=epi_runs)
        _say(f"Run directory: {run.run_dir} ({'new' if new else 'continued'})")
        if new:
            _say(estimate_line(estimate, app=serve))
        return run, new, estimate.template
    except BaseException:
        if run is not None:
            run.close()
        raise


def _offline_remedy(run: Run | None, run_dir: Path) -> str:
    """What to do when the bucket cannot be reached at the listing: the same command later,
    and, for a run that has renders, the app without the bucket."""
    again = "Run the same command again once it can be reached"
    if run is None:
        return again
    try:
        done = render_counts(run)[0]
    except Exception:  # the remedy must not hide the fault it is said for
        done = 0
    if not done:
        return again
    return (
        f"{again}; the {_plural(done, 'subject')} that {'is' if done == 1 else 'are'} rendered "
        f"can be rated without the bucket: surfcensus serve --run-dir {run_dir}"
    )


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


def _say_inventory(run: Run, path: Path | str | None = None, *, where: str | None = None) -> None:
    """``Found 48 subjects in sourcedata/freesurfer (fMRIPrep layout, FreeSurfer 7.3.2). 2
    cannot be rendered: ...``. Where the subjects lie is said relative to ``path``, the
    folder that was named, and left out when they lie in that folder itself; ``where``
    names it instead (``openneuro:ds000102``, as the user typed it)."""
    public = run.public()
    subjects = public.subjects()
    present = subjects[subjects["present"].to_numpy(dtype=bool)]
    layouts = [LAYOUT_WORDS.get(str(name), str(name)) for name in public.datasets()["layout"]]
    versions = sorted({v for v in present["fs_version"] if isinstance(v, str)})
    described = list(dict.fromkeys(layouts))
    if versions:
        described.append("FreeSurfer " + ", ".join(versions))
    globs = [glob for source in run.config.sources for glob in source.subjects]
    if globs and tuple(globs) != ("*",):
        # a subject set that is narrower than the directory is said, since it is kept
        described.append("of the subjects matching " + ", ".join(globs))
    if where is None:
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
        version = folder["pipeline_version"]
        pipeline = "fMRIPrep" if folder["connector"] == "fmriprep" else str(folder["connector"])
        pipeline += f" {version}" if isinstance(version, str) else ""
        if folder.get("source", "local") != "local":
            # a dataset of a bucket: named as on the command line, never by the tree in the
            # cache that the planner reads (decision D50)
            from surfcensus.remote import RAW_BUCKET, REMOTE_SCHEME

            places.append(
                f"{REMOTE_SCHEME}:{folder['accession']} ({pipeline}), raw series from "
                f"s3://{RAW_BUCKET}"
            )
            continue
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


def pack_rewrites(run: Run) -> int:
    """How many functional sessions the pack phase would write again now without packing a
    run of them (``func.batch.pack_rewrites``): their runs are counted as done by
    :func:`pack_counts`. 0 where :func:`pack_counts` gives zeros."""
    from surfcensus.config import epi_settings
    from surfcensus.func.batch import pack_rewrites as rewrites

    if run.config.mode != "local" or not epi_settings(run.config).enabled:
        return 0
    if not run.public().has_func():
        return 0
    return rewrites(run.context())


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
    again = pack_rewrites(run)
    if again:
        sessions = _plural(again, "session")
        line += (
            f"; {sessions} {'is' if again == 1 else 'are'} written again while the app runs"
            if behind
            else f"; {sessions} to write again"
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
    fetched_from: Any = None
    """For a remote dataset, ``remote.stats()`` at the start of the call, from which the
    closing line counts what was fetched; None otherwise."""
    fetched_at_app: tuple[int, int] | None = None
    """For a remote dataset, the bytes and requests fetched before the app opened."""
    transport: Any = None
    """The transport that the worker behind the app reaches a remote dataset with (None: the
    default of ``remote.default_transport``)."""
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renderer: threading.Thread | None = field(default=None, repr=False)
    _problem: list[str] = field(default_factory=list, repr=False)
    _paused: bool = field(default=False, repr=False)
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

    @property
    def render_paused(self) -> bool:
        """Whether the renderer behind the app waits for a bucket that cannot be reached
        (it tries again every :data:`REMOTE_RETRY_S` seconds)."""
        return self._paused and self.rendering

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
    transport: httpx.BaseTransport | None = None,
) -> None:
    """The worker behind the app: a ``Run`` of its own renders what is pending, in queue
    order, a batch of ``jobs`` subjects at a time so that each is offered soon after it is
    made, and then packs the functional sessions that are pending, a batch of ``jobs``
    sessions at a time (the surfaces first, because a rater waits on them; D33 and the plan
    of step 2 of the EPI interface). The stop request of the app ends either between two
    batches. A failure of the run or the machine ends it and is said; the app goes on."""
    try:
        paused = False
        while True:
            try:
                # a remote dataset is reached through a source and a cache of this Run's
                # own, which share the lease area of this process and its budget with those
                # of the caller
                with Run(run_dir, transport=transport) as run:
                    _with_fsaverage(run, fsaverage)
                    cache = run.config.cache.model_dump() | {"batch_subjects": jobs}
                    run.config = with_changes(run.config, cache=cache)
                    if render:
                        report = run.render(jobs=jobs, stop=app._stop.is_set)
                        if app._stop.is_set():
                            return
                        done, failed, _ = render_counts(run)
                        _say(
                            f"All renders are made: {done} of {_plural(total, 'subject')} "
                            "can be rated."
                        )
                        if failed:
                            _say_failures(run, report, "not rendered")
                    if pack:
                        app._packing = True
                        _pack_rest(run, jobs, app)
                return
            except SourceError as exc:
                # a bucket that cannot be reached is tried again while the app runs: the
                # rating screen says that the next subject opens by itself, and nothing of
                # the batch that stopped was recorded, so the next try goes on from it
                if exc.reason != "unreachable" or app._stop.is_set():
                    raise
                if not paused:
                    paused = True
                    _say(
                        f"Rendering paused: {' '.join(str(exc).split())}. It is tried again "
                        f"every {REMOTE_RETRY_S:g} s while the app runs; the subjects that "
                        "are rendered can be rated meanwhile."
                    )
                app._paused = True
                if app._stop.wait(REMOTE_RETRY_S):
                    return
                app._paused = False
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
    transport: httpx.BaseTransport | None = None,
) -> tuple[Run, ReviewApp | None]:
    """Everything :func:`review` does up to the running app: the run, and the app with the
    worker behind it, which the caller stops (``ReviewApp.stop``) and whose run the caller
    closes. The app is None with ``serve=False`` and when the run has nothing to show: no
    subject can be rendered and no functional run packed (:func:`rateable` and
    :func:`viewable` are then 0). The options are those of :func:`review`.
    """
    from surfcensus import remote

    started = time.perf_counter()
    fetched_from = remote.stats()
    _check(rater, jobs, serve_after, port)
    # openneuro:dsXXXXXX is read before any Path is made of the argument (decision D47,
    # item 8); any other text that starts with openneuro: is refused here
    ref = parse_remote_path(path)
    if ref is not None:
        _refuse_for_remote(
            field_strength=field_strength, relocate=relocate, epi=epi, epi_runs=epi_runs, raw=raw
        )
    if serve:
        # before the work, not after it: an installation that lacks a package of the app
        # (a stripped or broken one) has no rater app to open
        import surfcensus.app.server  # noqa: F401
    if field_strength is not None and not 0 < field_strength < math.inf:
        raise ConfigError(
            "invalid", f"field_strength is in tesla, positive and finite, not {field_strength}"
        )
    jobs = default_jobs() if jobs is None else jobs
    named: Path | None = None  # the folder that was named, for where things lie under it
    if ref is not None:
        # the default run directory is in the current directory (decision D47, item 2)
        run_dir = Path.cwd() / ref.run_dir_name if run_dir is None else Path(run_dir)
        if fsaverage is not None:
            _refuse_in_fsaverage(run_dir, fsaverage)  # before a run is made there
        run, new, template_bytes = _open_remote_run(
            ref,
            run_dir,
            subjects=subjects,
            blind=blind,
            require_marks=require_marks,
            video=video,
            force=force,
            transport=transport,
            serve=serve,
            serve_after=serve_after,
            jobs=jobs,
            fsaverage=fsaverage,
            epi=epi,
            epi_runs=epi_runs,
        )
    else:
        named = Path(path)
        if run_dir is None:
            run_dir = default_run_dir(named)
            if not relocate:
                refuse_foreign_default(named, run_dir)  # before a run is opened there
        else:
            run_dir = Path(run_dir)
        if fsaverage is not None:
            _refuse_in_fsaverage(run_dir, fsaverage)  # before a run is made there
        run, new = _open_run(
            named,
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
        if ref is None:
            _say(f"Run directory: {run.run_dir} ({'new' if new else 'continued'})")
        report = run.inventory()
        _say_inventory(run, named, where=None if ref is None else ref.text)
        if report.failed:
            _say_failures(run, report, "not read")
        if ref is not None:
            # before anything more is fetched: a subject that the cache's budget cannot hold
            run.refuse_over_budget()
            if not new:
                everything = not serve or serve_after == 0
                _say(
                    continued_line(
                        run,
                        template_bytes=template_bytes,
                        n_first=None if everything else serve_after or DEFAULT_SERVE_AFTER,
                        app=serve,
                        func=run.remote_func_pending(),
                    )
                )
        # the functional inventory reads listings and small files only (D33: the time to the
        # rating screen does not grow with the functional data); the packs come behind the app
        run.func_inventory()
        for line in functional_lines(run, named):
            _say(line)
        if run.func_left_out_note is not None:
            _say(f"  {run.func_left_out_note}.")

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
            if _fetch_said(run, named):
                # the fetch line above is the remedy: say what comes after it
                line += " Fetch the files as said above, then run the same command again."
            if shown and serve:
                line += " The EPI movies of its functional runs are shown."
            elif shown:
                line += " Its functional runs are packed for the EPI movies."
            _say(line)
            if shown == 0:
                if ref is not None:
                    _say_fetched(run, fetched_from)
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
            if ref is not None:
                _say_fetched(run, fetched_from)
            return run, None

        from surfcensus.app.server import start_server
        from surfcensus.protocol import install_protocol

        install_protocol(run.run_dir)
        handle = start_server(run.run_dir, port=port, rater=rater)
        app = ReviewApp(handle, run.run_dir, time.perf_counter() - started, epi=shown > 0)
        if ref is not None:
            app.fetched_from = fetched_from
            app.fetched_at_app = _fetched_since(fetched_from)
            app.transport = transport
        # a session can be pending with every pack current (its place in the stream moved,
        # D42): behind the app its session.json and stream.json are written again
        epi_behind = epi_pending > 0 or sessions_pending(run)
        if pending or epi_behind:
            thread = threading.Thread(
                target=_render_rest,
                args=(
                    run.run_dir,
                    jobs,
                    fsaverage,
                    app,
                    len(uids),
                    pending > 0,
                    epi_behind,
                    transport,
                ),
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
    transport: httpx.BaseTransport | None = None,
) -> Run:
    """Inventory, score, select and render the subjects under ``path`` and open the rater
    app on them; returns the run.

    ``path`` may name a dataset of the OpenNeuroDerivatives bucket, ``openneuro:dsXXXXXX``
    (docs/dev/remote_plan.md, section 4): its subjects are listed anonymously and read-only,
    what will be fetched is said from the listed sizes, and the files are fetched in batches
    into the run's cache and removed after use. The run directory is then
    ``./dsXXXXXX-surfcensus-run`` unless ``run_dir`` is given; ``relocate``, ``raw``, ``epi``,
    ``epi_runs`` and ``field_strength`` are refused; a new run that names no ``subjects`` is
    refused when the estimate before the first rating screen exceeds
    :data:`REMOTE_GUARD_BYTES`. ``transport`` is that of the bucket requests (a test's fake);
    None takes ``remote.default_transport``.

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
        transport=transport,
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
        took = f"  {app.seconds_to_app:.1f} s from the start of the command to the app"
        if app.fetched_at_app is not None:
            fetched, requests = app.fetched_at_app
            took += f"; {size_text(fetched)} fetched in {_plural(requests, 'request')}"
        _say(took + ".")
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
    if app.fetched_from is not None:
        try:
            _say_fetched(run, app.fetched_from)
        except Exception:  # as above: the work is saved
            logger.debug("no closing line of the fetches", exc_info=True)
    if app.epi:
        try:
            done, failed, pending = pack_counts(run)
            again = pack_rewrites(run)
        except Exception:  # as above: the work is saved
            logger.debug("no EPI closing line", exc_info=True)
        else:
            line = f"EPI packs: {done} of {_plural(done + failed + pending, 'run')} made"
            if failed:
                line += f", {failed} failed ({retry_hint(run)})"
            if pending:
                line += f"; {pending} are made at the next call"
            if again:
                line += (
                    f"; {_plural(again, 'session')} {'is' if again == 1 else 'are'} "
                    "written again at the next call"
                )
            _say(line + ".")
