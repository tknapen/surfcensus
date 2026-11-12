"""Phase 6 for a run: render the subjects of a run into ``media/<subject_uid>/``.

``run_render`` decides from the ``renders`` table what is pending, without opening a source
file, renders the pending subjects batch by batch (``batches.run_batches``) and records every
outcome. A subject is done when its row says so; the row is written after the files, so a
crash in between leaves a render without a row, which the next run finds complete on disk
and records.

This module holds the public side of a run only (``records.RunContext``): subjects by
``subject_uid``, datasets by pseudonym. The ``template`` argument of ``run_render`` is added
in milestone 3, when ``templates.py`` exists.
"""

from __future__ import annotations

import errno
import html
import math
import os
import tempfile
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from surfcensus._io import atomic_path
from surfcensus.batches import WorkerError, run_batches
from surfcensus.config import ConfigError
from surfcensus.records import PhaseReport, RenderResult, RenderRow, RunContext
from surfcensus.render import api, encode
from surfcensus.render.model import RenderConfig
from surfcensus.schemas import intensity_schema
from surfcensus.sources.base import RENDER_ROLES, TIER_C, FileRole, SourceError
from surfcensus.store import MEDIA_DIR, media_dir, read_table, table_path, upsert_part

PHASE = "render"
LOGS_DIR = "logs"
MISSING_SURFACE = "missing surface"
SOURCE_CHANGED = "source changed"
RENDER_FAILURE = "render failure"

# what a render reads; milestone 3 adds the two sphere_reg roles when the run has a template
_LEASED: tuple[FileRole, ...] = TIER_C


@dataclass
class _Plan:
    """What a call of ``run_render`` finds before it renders anything."""

    video: bool
    config_hash: str
    signatures: dict[str, str]
    """Input signature of every unit and of every incomplete subject."""
    pending: list[str]
    """Units to render, in order."""
    skipped: int
    """Units whose render is recorded as done and is on disk."""
    failures: dict[str, str]
    """``subject_uid`` -> reason, for the units recorded as failed that are not tried again
    and for every incomplete subject."""
    unrecorded: list[str]
    """Incomplete subjects that have no ``missing surface`` row for these inputs yet."""
    notes: tuple[str, ...]


def _plan(
    ctx: RunContext,
    *,
    datasets: Collection[str] | None,
    subjects: Sequence[str] | None,
    all_subjects: bool | None,
    video: bool | None,
    force: bool,
    retry_failed: bool,
) -> _Plan:
    """Decide from the ``renders`` table and the listing in the catalog what is pending.
    Opens no source file and writes nothing."""
    public, config = ctx.public, ctx.config
    video = config.render_video if video is None else video
    if all_subjects is None:
        all_subjects = public.mode() == "census" and config.select.render_scope == "all"
    present = public.subject_uids(datasets=datasets)
    complete = set(public.subject_uids(datasets=datasets, complete=True))
    units, notes = _units(ctx, present, complete, subjects, all_subjects)
    incomplete = [uid for uid in present if uid not in complete]
    config_hash = config.render.config_hash()
    signatures = {
        uid: public.input_signature(uid, _LEASED, _signature_extra(None))
        for uid in (*units, *incomplete)
    }

    def recorded(uid: str) -> RenderRow | None:
        """The subject's row if it was recorded for this config, these inputs and this
        ``video`` setting."""
        row = public.render(uid)
        same = (
            row is not None
            and row.config_hash == config_hash
            and row.input_signature == signatures[uid]
            and row.video == video
        )
        return row if same else None

    pending: list[str] = []
    failures: dict[str, str] = {}
    skipped = 0
    for uid in units:
        row = None if force else recorded(uid)
        if row is not None and row.status == "failed" and not retry_failed:
            failures[uid] = row.reason or RENDER_FAILURE
        elif row is not None and row.status == "done" and _render_json(ctx, uid).is_file():
            skipped += 1
        else:
            pending.append(uid)
    failures.update(dict.fromkeys(incomplete, MISSING_SURFACE))
    unrecorded = [
        uid
        for uid in incomplete
        if (row := recorded(uid)) is None or row.status != "failed" or row.reason != MISSING_SURFACE
    ]
    return _Plan(video, config_hash, signatures, pending, skipped, failures, unrecorded, notes)


def render_status(ctx: RunContext) -> tuple[int, int, int]:
    """``(done, failed, pending)``: what ``run_render(ctx)`` would find now, without
    rendering or recording anything.

    ``done`` are the subjects whose render is recorded for the run's present configuration
    and inputs and is on disk; ``failed`` those recorded as failed for them, and every
    present subject that lacks a render role; ``pending`` what a call would render.
    """
    plan = _status_plan(ctx)
    return plan.skipped, len(plan.failures), len(plan.pending)


def render_failures(ctx: RunContext) -> dict[str, tuple[str, str | None]]:
    """``subject_uid -> (reason, detail)`` for the subjects that ``render_status`` counts as
    failed.

    The detail is None where the failure is recorded: its row of the ``renders`` table has
    it, in the private catalog. An incomplete subject that has no ``missing surface`` row
    for its present inputs yet (``run_render`` writes it) comes with the detail that the
    row will have.
    """
    plan = _status_plan(ctx)
    details = _missing_details(ctx, plan.unrecorded)
    return {
        uid: (reason, ctx.scrub(details[uid]) if uid in details else None)
        for uid, reason in plan.failures.items()
    }


def _status_plan(ctx: RunContext) -> _Plan:
    """What ``run_render(ctx)`` would find now."""
    return _plan(
        ctx,
        datasets=None,
        subjects=None,
        all_subjects=None,
        video=None,
        force=False,
        retry_failed=False,
    )


def render_pending(
    ctx: RunContext,
    *,
    datasets: Collection[str] | None = None,
    subjects: Sequence[str] | None = None,
    all_subjects: bool | None = None,
    video: bool | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> int:
    """How many subjects ``run_render`` would render when called with the same arguments.
    Like the call itself it raises ``ConfigError`` for a subject the run does not have; it
    opens no source file and records nothing. Zero means that the call needs no source."""
    plan = _plan(
        ctx,
        datasets=datasets,
        subjects=subjects,
        all_subjects=all_subjects,
        video=video,
        force=force,
        retry_failed=retry_failed,
    )
    return len(plan.pending)


def run_render(
    ctx: RunContext,
    *,
    jobs: int = 1,
    datasets: Collection[str] | None = None,
    subjects: Sequence[str] | None = None,
    all_subjects: bool | None = None,
    video: bool | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> PhaseReport:
    """Render the pending subjects and record every outcome in the ``renders`` table.

    Which subjects: ``subjects`` (``subject_uid``s) in the order given; else every complete
    present subject when ``all_subjects``; else the subjects of the ``selection`` table;
    else, when that table is empty, every complete present subject. ``all_subjects=None``
    is ``select.render_scope == "all"`` in a census run and False in a local run.
    ``datasets`` (pseudonyms) restricts all of this. Without an order given, subjects go by
    pseudonym, then ``subject_uid``; the selected subjects of a local run that has scores
    and is not blind go in the order the rater app offers them. ``video=None`` is
    ``render_video`` of the run.

    A subject is skipped when its row is ``done`` for the same render config, the same
    input signature and the same ``video``, and ``media/<uid>/render.json`` exists. A
    ``failed`` row that matches in the same way is not tried again unless ``retry_failed``.
    ``force`` ignores the rows and renders over what is there.

    Every present subject that lacks a render role gets a ``failed`` row with the reason
    ``missing surface``, selected or not, without its files being touched. A subject that
    cannot be rendered is recorded with the reason of its ``RenderError``; one whose files
    changed since the inventory with ``source changed``; any other failure with ``render
    failure`` and the type of the exception. The batch goes on in each case.

    The report counts the subjects rendered now (``done``), skipped, and in the failed
    state when the call returns, whether tried now or not (``failed``, ``failures``).

    Raises ``ConfigError`` for what is wrong with the run or the machine and not with a
    subject: ``missing`` when a video is asked for and ffmpeg is absent or cannot encode
    (two small frames are encoded to find out), ``invalid`` for a video with an odd
    ``render.panel_px``, for a ``subject_uid`` that the run does not have, for a run
    directory inside a subject directory, and when the media cannot be written (a full or
    read-only disk, a quota, no permission under ``media/``). The first three are raised
    before anything is recorded; the last two end the call at the batch in which they occur,
    and nothing of that batch is recorded. ``KeyboardInterrupt`` ends the call as well; the
    batches before it are recorded.
    """
    started = time.perf_counter()
    public, config = ctx.public, ctx.config
    render_config = config.render
    if (config.render_video if video is None else video) and render_config.panel_px % 2:
        raise ConfigError(
            "invalid",
            f"a video needs an even render.panel_px, the run has {render_config.panel_px}",
        )
    plan = _plan(
        ctx,
        datasets=datasets,
        subjects=subjects,
        all_subjects=all_subjects,
        video=video,
        force=force,
        retry_failed=retry_failed,
    )
    video, config_hash, signatures = plan.video, plan.config_hash, plan.signatures
    pending, failures = plan.pending, plan.failures

    def failed_row(uid: str, reason: str, detail: str) -> RenderRow:
        """A failure of this config on these inputs; ``detail`` is scrubbed here."""
        return RenderRow(
            subject_uid=uid,
            config_hash=config_hash,
            input_signature=signatures[uid],
            video=video,
            status="failed",
            reason=reason,
            detail=ctx.scrub(detail),
        )

    if video and pending:
        _check_encoder(render_config)

    if plan.unrecorded:
        details = _missing_details(ctx, plan.unrecorded)
        public.record_renders(
            [failed_row(uid, MISSING_SURFACE, details[uid]) for uid in plan.unrecorded]
        )

    tile_hash = render_config.tile_hash()
    media_root = Path(ctx.run_dir) / MEDIA_DIR
    done = 0

    def commit(outcomes: Sequence[tuple[str, RenderResult | BaseException]]) -> None:
        nonlocal done
        unwritable = [
            o for _, o in outcomes if isinstance(o, Exception) and _is_unwritable(o, media_root)
        ]
        if unwritable:
            # A full or read-only disk fails every subject alike. Nothing of the batch is
            # recorded, so the next call does the work once there is room: the renders that
            # did succeed are found complete on disk.
            raise ConfigError(
                "invalid", f"the media cannot be written: {unwritable[0]}"
            ) from unwritable[0]
        refused = [o for _, o in outcomes if isinstance(o, api.OutputInSourceError)]
        if refused:
            # nothing can be rendered into this run directory, whatever the subject
            raise ConfigError(
                "invalid",
                "the run directory lies inside a directory that subjects are read from, "
                "which is only read; use a run directory outside the data",
            ) from refused[0]
        rows: list[RenderRow] = []
        results: list[tuple[str, RenderResult]] = []
        for uid, outcome in outcomes:
            if isinstance(outcome, BaseException):
                rows.append(failed_row(uid, *_failure(outcome)))
                continue
            info = outcome.info
            digests = {FileRole(role): entry.sha256 for role, entry in info.inputs.items()}
            public.set_file_digests(uid, digests)
            read = {role.value: digests[role] for role in TIER_C if role in digests}
            rows.append(
                RenderRow(
                    subject_uid=uid,
                    config_hash=config_hash,
                    input_signature=signatures[uid],
                    video=video,
                    status="done",
                    tile_hash=tile_hash,
                    render_key=ctx.render_key(tile_hash, read),
                    media_path=f"{MEDIA_DIR}/{uid}",
                    n_frames=len(info.modes["sync"].frames),
                )
            )
            results.append((uid, outcome))
        _write_extras(ctx, results)
        public.record_renders(rows)  # last: a row never stands for files that are not there
        done += len(results)
        failures.update(
            {
                row.subject_uid: row.reason or RENDER_FAILURE
                for row in rows
                if row.status == "failed"
            }
        )

    work = partial(
        _render_one,
        media_root=media_root,
        config=render_config,
        video=video,
        force=force,
    )
    run_batches(
        pending,
        lambda uid: ctx.lease(uid, _LEASED),
        work,
        commit,
        batch_subjects=config.cache.batch_subjects,
        jobs=jobs,
        progress=PHASE if pending else None,
    )
    return PhaseReport(
        phase=PHASE,
        done=done,
        skipped=plan.skipped,
        failed=len(failures),
        failures=tuple(sorted(failures.items())),
        notes=plan.notes,
        seconds=time.perf_counter() - started,
    )


def render_check_sample(ctx: RunContext, *, fraction: float = 0.02, seed: int) -> Path:
    """Draw a seeded sample of the done renders for the by-eye check of phase 6 and write it
    to ``logs/render-check-<seed>.txt`` and ``.html``; returns the text file.

    ``ceil(fraction x n)`` of the ``n`` done renders are drawn without replacement from the
    sorted ``subject_uid``s with ``numpy.random.default_rng(seed)``. The text file names the
    seed in its first line and then holds one ``subject_uid`` and poster path (relative to
    the run directory) per line, tab-separated, sorted; the page shows the same posters
    through relative links. ValueError for a fraction outside (0, 1] or a negative seed.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be above 0 and at most 1, not {fraction}")
    if seed < 0:
        raise ValueError(f"the seed must not be negative, got {seed}")
    renders = ctx.public.renders()
    done = sorted(renders.loc[renders["status"] == "done", "subject_uid"])
    n = math.ceil(fraction * len(done))
    drawn = np.random.default_rng(seed).choice(len(done), size=n, replace=False)
    sample = [done[position] for position in sorted(drawn.tolist())]
    posters = {uid: f"{MEDIA_DIR}/{uid}/{api.POSTER}" for uid in sample}
    summary = (
        f"render check: seed {seed}, {len(sample)} of {len(done)} done renders "
        f"(fraction {fraction:g})"
    )

    logs = Path(ctx.run_dir) / LOGS_DIR
    text = logs / f"render-check-{seed}.txt"
    lines = [f"# {summary}", *(f"{uid}\t{poster}" for uid, poster in posters.items())]
    with atomic_path(text) as tmp:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures = [
        f'<figure><img src="../{html.escape(poster)}" alt="" loading="lazy">'
        f"<figcaption>{html.escape(uid)}</figcaption></figure>"
        for uid, poster in posters.items()
    ]
    page = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Render check, seed {seed}</title>",
        "<style>body{background:#111;color:#ddd;font-family:sans-serif}"
        "figure{margin:1em 0}img{max-width:100%}</style>",
        "</head>",
        "<body>",
        f"<p>{html.escape(summary[0].upper() + summary[1:])}. Look for blank panels, missing "
        "contours, a wrong orientation and a misplaced pole.</p>",
        *figures,
        "</body>",
        "</html>",
    ]
    with atomic_path(text.with_suffix(".html")) as tmp:
        tmp.write_text("\n".join(page) + "\n", encoding="utf-8")
    return text


def _render_one(
    uid: str,
    files: Mapping[FileRole, Path],
    *,
    media_root: Path,
    config: RenderConfig,
    video: bool,
    force: bool,
) -> RenderResult:
    """What a worker does for one subject. Top-level, so that it can be sent to a process."""
    info = api.render_subject(files, media_root / uid, config, video=video, force=force)
    return RenderResult(info=info)


def _render_json(ctx: RunContext, uid: str) -> Path:
    return media_dir(ctx.run_dir, uid) / api.RENDER_JSON


def _signature_extra(template_id: str | None) -> tuple[str, ...]:
    """What the input signature of a render holds beside the files it reads."""
    return (f"template:{template_id or 'none'}",)


def _units(
    ctx: RunContext,
    present: Sequence[str],
    complete: Collection[str],
    subjects: Sequence[str] | None,
    all_subjects: bool,
) -> tuple[list[str], tuple[str, ...]]:
    """The subjects to bring up to date, in order, and notes for the user. ``present`` are
    the present subjects of the datasets asked for, by pseudonym, then ``subject_uid``."""
    public = ctx.public
    notes: tuple[str, ...] = ()
    if subjects is not None:
        named = list(dict.fromkeys([subjects] if isinstance(subjects, str) else subjects))
        known = set(public.subject_uids(present=None))
        unknown = [uid for uid in named if uid not in known]
        if unknown:
            more = f" and {len(unknown) - 3} more" if len(unknown) > 3 else ""
            raise ConfigError("invalid", f"the run has no subject {', '.join(unknown[:3])}{more}")
        wanted = set(present)
        left_out = sum(uid not in wanted for uid in named)
        if left_out:
            notes = (
                f"{left_out} of the subjects named are not rendered: they are no longer in "
                "the source, or not in the datasets asked for.",
            )
        candidates = named
    elif all_subjects:
        candidates = list(present)
    else:
        selected = set(public.selection()["subject_uid"])
        if selected:
            candidates = _queue_order(ctx, [uid for uid in present if uid in selected])
        else:
            candidates = list(present)
    return [uid for uid in candidates if uid in complete], notes


def _queue_order(ctx: RunContext, uids: Sequence[str]) -> list[str]:
    """``uids`` in the order the rater app of a local run offers them, when the run has
    scores and is not blind: first the subjects that could not be fully scored, by
    ``subject_uid``; then by rank, worst first; then the subjects without a score.
    Otherwise as given."""
    path = table_path(ctx.run_dir, "scores")
    if ctx.public.mode() != "local" or ctx.config.app.blind or not path.is_file():
        return list(uids)
    scores = read_table(path)
    position: dict[str, tuple[int, int]] = {}
    for uid, primary, rank, flags in zip(
        scores["subject_uid"], scores["primary"], scores["rank"], scores["flags"], strict=True
    ):
        if primary:
            position[uid] = (0, 0) if "partial_score" in flags.split(",") else (1, int(rank))
    return sorted(uids, key=lambda uid: (*position.get(uid, (2, 0)), uid))


def _missing_details(ctx: RunContext, uids: Sequence[str]) -> dict[str, str]:
    """For each subject, which render roles it lacks, in the words of ``RenderError``, and
    why its directory could not be listed if that is the cause."""
    subjects = ctx.public.subjects()
    problems = dict(zip(subjects["subject_uid"], subjects["problem"], strict=True))
    details = {}
    for uid in uids:
        has = ctx.public.roles(uid)
        detail = "no file for " + ", ".join(role.value for role in RENDER_ROLES if role not in has)
        problem = problems.get(uid)
        details[uid] = f"{detail} ({problem})" if isinstance(problem, str) and problem else detail
    return details


def _check_encoder(config: RenderConfig) -> None:
    """Find the ffmpeg binary and have it encode two small frames in a temporary directory.

    The encoder belongs to the environment: one that is absent, cannot be started or cannot
    encode (a build without libx264) fails every subject alike and is not recorded once per
    subject. Raises ``ConfigError("missing", ...)``.
    """
    try:
        encode.ffmpeg_exe()
        with tempfile.TemporaryDirectory(prefix="surfcensus-") as folder:
            frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 2
            encode.write_mp4(Path(folder) / "probe.mp4", frames, config.fps, config.crf)
    except (RuntimeError, OSError) as exc:
        raise ConfigError("missing", f"the video cannot be encoded: {exc}") from exc


_NO_ROOM = (errno.ENOSPC, errno.EDQUOT, errno.EROFS)
_NOT_ALLOWED = (errno.EACCES, errno.EPERM)


def _is_unwritable(exc: Exception, media_root: Path) -> bool:
    """Whether a failed render says that the media cannot be written, whatever the subject:
    the disk is full, over quota or read-only, or a path under ``media_root`` may not be
    written. Such a failure is of the run directory and is not recorded against a subject.

    The inputs are no cause of these: ``render_subject`` turns every error of reading them
    into a ``RenderError``. ffmpeg writes the video itself and reports a failed write in
    words, as a RuntimeError of ``encode.write_mp4``.
    """
    if isinstance(exc, OSError):
        if exc.errno in _NO_ROOM:
            return True
        name = exc.filename
        if exc.errno in _NOT_ALLOWED and isinstance(name, str | bytes | os.PathLike):
            return Path(os.fsdecode(name)).is_relative_to(media_root)
        return False
    if isinstance(exc, RuntimeError):
        return any(os.strerror(code) in str(exc) for code in _NO_ROOM)
    return False


def _failure(exc: BaseException) -> tuple[str, str]:
    """Reason and unscrubbed detail of the ``renders`` row for an exception of a lease or of
    a render."""
    if isinstance(exc, api.RenderError):
        return exc.reason, exc.detail
    if isinstance(exc, SourceError) and exc.reason == "changed":
        return SOURCE_CHANGED, exc.detail
    if isinstance(exc, WorkerError):  # carries the name of the class that was raised
        return RENDER_FAILURE, str(exc)
    return RENDER_FAILURE, f"{type(exc).__name__}: {exc}"


def _write_extras(ctx: RunContext, results: Sequence[tuple[str, RenderResult]]) -> None:
    """Rows that renders computed beside their files go to the part of each subject's dataset
    of the table they name, replacing the earlier rows of the subjects rendered."""
    if not any(result.extras for _, result in results):
        return
    by_dataset: dict[str, list[tuple[str, RenderResult]]] = {}
    for uid, result in results:
        by_dataset.setdefault(ctx.public.dataset_of(uid), []).append((uid, result))
    for dataset in sorted(by_dataset):
        members = by_dataset[dataset]
        for table in sorted({name for _, result in members for name in result.extras}):
            frames = [result.extras[table] for _, result in members if table in result.extras]
            schema = None
            if table == "intensity":  # its columns follow from the depths of this run
                schema = intensity_schema(len(ctx.config.intensity.depths))
            upsert_part(
                ctx.run_dir,
                table,
                dataset,
                pd.concat(frames, ignore_index=True),
                replace_uids=[uid for uid, _ in members],
                schema=schema,
            )
