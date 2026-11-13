"""Command-line entry point. Subcommands are specified in docs/architecture.md."""

from __future__ import annotations

import math
import re
import secrets
import signal
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from typer.core import TyperCommand

from surfcensus import __version__
from surfcensus.log import configure_cli_logging

if TYPE_CHECKING:
    from surfcensus.records import PhaseReport
    from surfcensus.run import Run

app = typer.Typer(
    no_args_is_help=True,
    help="FreeSurfer occipital surface QC: score, render, rate.",
)

_RUN_DIR = typer.Option(..., "--run-dir", metavar="DIR", help="The run directory.")
_DATASET = typer.Option(
    None,
    "--dataset",
    metavar="ID",
    help="Only this dataset, by pseudonym or accession. Repeatable.",
)
_FSAVERAGE = typer.Option(
    None,
    "--fsaverage",
    metavar="DIR",
    help="The fsaverage directory to take the template from (any FreeSurfer install has one "
    "in subjects/). The run keeps the template it first used.",
)
_MAX_NAMES = 10  # subjects named on one summary line
_RELOCATE_ANY = "*"  # --relocate given without a pseudonym


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also print what is being done."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    configure_cli_logging(verbose)


@app.command("render-subject")
def render_subject(
    subject_dir: Path = typer.Argument(..., help="A FreeSurfer subject directory."),
    out: str = typer.Option(
        ..., "--out", metavar="PATH", help="Media directory to write, or the path of an .mp4."
    ),
    force: bool = typer.Option(False, "--force", help="Render even if the output is up to date."),
    no_video: bool = typer.Option(False, "--no-video", help="Do not write anim.mp4."),
    fsaverage: Path | None = typer.Option(
        None,
        "--fsaverage",
        metavar="DIR",
        help="The fsaverage directory that click marks are mapped to. Without it: "
        "$FREESURFER_HOME/subjects/fsaverage, else fsaverage next to the subject.",
    ),
    no_template: bool = typer.Option(
        False, "--no-template", help="Render without fsaverage counterparts."
    ),
) -> None:
    """Render one subject without a run directory: tiles, poster, video, render.json."""
    # imports matplotlib
    from surfcensus.render import OutputInSourceError, RenderError, three_plane
    from surfcensus.render.model import PANEL_ORDER
    from surfcensus.templates import TemplateError, TemplateNotFound

    if not out.strip():  # an unset shell variable; as a path it would be the current directory
        raise typer.BadParameter("the output path is empty", param_hint="--out")
    target = Path(out)
    if no_video and target.suffix.lower() == ".mp4":
        raise typer.BadParameter("--no-video contradicts an --out that names a video")
    if no_template and fsaverage is not None:
        raise typer.BadParameter("--no-template contradicts --fsaverage")
    template = None if no_template else "auto" if fsaverage is None else fsaverage
    start = time.perf_counter()
    try:
        with _terminated_like_interrupted():
            info = three_plane(
                subject_dir, target, template=template, video=not no_video, force=force
            )
    except OutputInSourceError as exc:  # a call that cannot be meant, like an empty --out
        raise typer.BadParameter(str(exc), param_hint="--out") from exc
    except (TemplateNotFound, TemplateError) as exc:  # only for a directory that was given
        raise typer.BadParameter(_template_problem(exc), param_hint="--fsaverage") from exc
    except (RenderError, OSError) as exc:
        # OSError: the output cannot be written (not a directory, read-only, disk full)
        typer.echo(f"render failed: {' '.join(str(exc).split())}", err=True)
        raise typer.Exit(1) from exc
    counts = ", ".join(f"{len(info.panels[name].slices)} {name}" for name in PANEL_ORDER)
    typer.echo(f"{target}: {counts} slices, {time.perf_counter() - start:.1f} s")


@contextmanager
def _interrupted_is_exit_130() -> Iterator[None]:
    """A phase stopped by Ctrl-C ends with 130, as a shell reports it, not with the 1 that
    click gives an abort and that here means "some units failed"."""
    try:
        yield
    except KeyboardInterrupt:
        typer.echo("interrupted; what was recorded stays, and the next call goes on", err=True)
        raise typer.Exit(130) from None


@contextmanager
def _terminated_like_interrupted() -> Iterator[None]:
    """Turn SIGTERM into SystemExit(143) for the duration, so that a render stopped by a
    scheduler removes its temporary files as one stopped by Ctrl-C does."""
    if threading.current_thread() is not threading.main_thread():
        yield  # signal handlers can only be set from the main thread
        return

    def stop(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    before = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        if before is not None:  # None: a handler that was not set from Python
            signal.signal(signal.SIGTERM, before)


class _RelocateCommand(TyperCommand):
    """``--relocate`` takes an optional value, which typer cannot declare: a ``--relocate``
    that no value follows is given one that stands for "the run's one local source"."""

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        filled = []
        for position, arg in enumerate(args):
            bare = position + 1 == len(args) or args[position + 1].startswith("-")
            filled.append(f"--relocate={_RELOCATE_ANY}" if arg == "--relocate" and bare else arg)
        return super().parse_args(ctx, filled)


def _template_problem(exc: Exception) -> str:
    """One line for a template that was asked for and cannot be used."""
    looked_in = getattr(exc, "looked_in", None)
    if looked_in:
        return "not found: no usable fsaverage template; " + "; ".join(
            f"{where}: {state}" for _, where, state in looked_in
        )
    return " ".join(str(exc).split())


@contextmanager
def _stops_with_exit_2() -> Iterator[None]:
    """What is wrong with the call, the run, a source, the template or the machine, and not
    with a subject: one line ``reason: detail`` on stderr and exit code 2."""
    from surfcensus.config import ConfigError
    from surfcensus.sources.base import SourceError
    from surfcensus.templates import TemplateError, TemplateNotFound

    try:
        yield
    except (ConfigError, SourceError) as exc:
        typer.echo(" ".join(str(exc).split()), err=True)
        raise typer.Exit(2) from exc
    except (TemplateError, TemplateNotFound) as exc:
        typer.echo(_template_problem(exc), err=True)
        raise typer.Exit(2) from exc
    except BrokenPipeError:  # the reader of the output went away; click ends the command
        raise
    except (sqlite3.OperationalError, OSError) as exc:
        # The catalog or a file of the run could not be read or written: a read-only or full
        # disk, a run directory that cannot be made. Sources and renders turn their own
        # errors into SourceError and failed units, so what arrives here is the run's.
        text = " ".join(str(exc).split())
        typer.echo(f"invalid: the run directory cannot be used: {text}", err=True)
        raise typer.Exit(2) from exc


def _names(frame: Any, uids: Sequence[str]) -> str:
    """Subjects for a summary line: by label in a local run, where the public catalog has
    labels, else by ``subject_uid``; at most ``_MAX_NAMES`` of them."""
    labels = dict(zip(frame["subject_uid"], frame["subject_label"], strict=True))
    names = sorted(labels.get(uid) or uid for uid in uids)
    more = f" and {len(names) - _MAX_NAMES} more" if len(names) > _MAX_NAMES else ""
    return ", ".join(names[:_MAX_NAMES]) + more


def _print_report(run: Run, report: PhaseReport) -> None:
    """The closing lines of a phase command: counts, one line per failure reason, notes."""
    typer.echo(
        f"{report.phase}: {report.done} done, {report.skipped} skipped, {report.failed} failed, "
        f"{report.seconds:.1f} s"
    )
    subjects = run.public().subjects()
    by_reason: dict[str, list[str]] = {}
    for unit, reason in report.failures:
        by_reason.setdefault(reason, []).append(unit)
    for reason in sorted(by_reason):
        units = by_reason[reason]
        typer.echo(f"  {reason}: {len(units)} ({_names(subjects, units)})")
    for note in report.notes:
        typer.echo(f"  {note}")


def _open_for_inventory(
    run_dir: Path,
    source: Path | None,
    subjects: list[str] | None,
    field_strength: float | None,
    relocate: bool | str,
) -> Run:
    """The run that ``inventory`` works on: a new local run on ``--source``, or the stored
    one with the options that were given written into its configuration."""
    from surfcensus.config import CONFIG_FILE, ConfigError, load_config, with_changes
    from surfcensus.run import Run
    from surfcensus.sources.local import LocalSource, detect_layout

    if not (run_dir / CONFIG_FILE).is_file():
        if source is None:
            raise ConfigError(
                "missing", f"{run_dir} holds no run; pass --source PATH to create one"
            )
        if relocate:
            raise ConfigError("missing", f"{run_dir} holds no run to relocate")
        made = LocalSource(source, subjects=subjects or (), field_strength=field_strength)
        made.subjects(made.dataset)  # a glob that matches nothing stops here, before a run exists
        return Run(run_dir, made)

    if relocate and source is None:
        raise ConfigError("invalid", "--relocate needs --source PATH, where the data is now")
    stored = load_config(run_dir)
    if stored.mode != "local":
        # the roots of a census run are in its catalog: the run matches the source itself
        given = None
        if source is not None:
            given = LocalSource(source, subjects=subjects or (), field_strength=field_strength)
        elif subjects is not None or field_strength is not None:
            raise ConfigError("invalid", "in a census run these options need --source PATH")
        return Run(run_dir, given, relocate=relocate)

    # a local run: --source says which source is meant and never changes its subject set;
    # the options that were given replace the stored ones
    target = None
    if source is not None:
        root = detect_layout(source)[0].root
        roots = [s.root for s in stored.sources]
        if root not in roots:
            Run(run_dir, LocalSource(source), relocate=relocate).close()  # or "other root"
            stored = load_config(run_dir)
            roots = [s.root for s in stored.sources]
        target = roots.index(root)
    if subjects is None and field_strength is None:
        return Run(run_dir)
    if target is None:
        if len(stored.sources) != 1:
            raise ConfigError(
                "invalid", "this run has several sources; say which one is meant with --source"
            )
        target = 0
    described = [s.model_dump() for s in stored.sources]
    if subjects is not None:
        described[target]["subjects"] = tuple(dict.fromkeys(subjects))
    if field_strength is not None:
        described[target]["field_strength"] = field_strength
    config = with_changes(stored, sources=described)
    if config == stored:
        return Run(run_dir)
    changed = config.sources[target]
    if changed.root is not None and subjects is not None:
        # a glob that matches nothing stops here, before it is stored
        probe = LocalSource(changed.root, subjects=changed.subjects, identity=changed.identity)
        probe.subjects(probe.dataset)
    return Run(run_dir, config=config)


def _print_inventory(run: Run) -> None:
    """What the run now holds: the layout found, subject counts, FreeSurfer versions, and
    the subjects that lack something, grouped by what they lack."""
    public = run.public()
    for row in public.datasets().itertuples(index=False):
        version = f" {row.pipeline_version}" if isinstance(row.pipeline_version, str) else ""
        typer.echo(f"{row.pseudonym}: layout {row.layout}, pipeline {row.pipeline}{version}")
    subjects = public.subjects()
    present = subjects[subjects["present"]]
    typer.echo(
        f"subjects: {len(present)} found, {int(present['complete'].sum())} complete, "
        f"{int(present['tier_a'].sum())} with Tier A, {int(present['tier_b'].sum())} with Tier B"
    )
    versions: dict[str, int] = {}
    for version in present["fs_version"]:
        name = version if isinstance(version, str) else "unknown"
        versions[name] = versions.get(name, 0) + 1
    if versions:
        counts = ", ".join(f"{name} ({versions[name]})" for name in sorted(versions))
        typer.echo(f"FreeSurfer: {counts}")
    lacking: dict[str, list[str]] = {}
    for row in present.itertuples(index=False):
        if isinstance(row.problem, str):
            lacking.setdefault(row.problem, []).append(row.subject_uid)
        elif not (row.complete and row.tier_a and row.tier_b):
            what = f"lacks {row.missing_roles.replace(',', ', ')}"
            lacking.setdefault(what, []).append(row.subject_uid)
    for what in sorted(lacking):
        typer.echo(f"  {what}: {len(lacking[what])} ({_names(subjects, lacking[what])})")


@app.command("inventory", cls=_RelocateCommand)
def inventory(
    run_dir: Path = _RUN_DIR,
    source: Path | None = typer.Option(
        None,
        "--source",
        metavar="PATH",
        help="A SUBJECTS_DIR, an fMRIPrep output, a BIDS derivatives folder or one subject "
        "directory. Creates a local run when DIR holds none.",
    ),
    subjects: list[str] | None = typer.Option(
        None,
        "--subjects",
        metavar="GLOB",
        help="Only the subject directories that match. Repeatable; kept for later runs.",
    ),
    field_strength: float | None = typer.Option(
        None,
        "--field-strength",
        metavar="T",
        help="Field strength of the source's subjects in tesla; kept for later runs.",
    ),
    dataset: list[str] | None = _DATASET,
    force: bool = typer.Option(False, "--force", help="Read every subject again."),
    relocate: str | None = typer.Option(
        None,
        "--relocate",
        metavar="[PSEUDONYM]",
        help="The data moved to --source: keep the run, its subjects and their ratings. With "
        "several local sources, name the dataset that moved.",
    ),
) -> None:
    """List the subjects of a source and record what each has (phase 2)."""
    if relocate not in (None, _RELOCATE_ANY) and not re.fullmatch(r"D-[0-9a-f]+", relocate):
        raise typer.BadParameter(
            "takes a dataset pseudonym (D-xxxxxxxx) or nothing", param_hint="--relocate"
        )
    moved: bool | str = False
    if relocate is not None:
        moved = True if relocate == _RELOCATE_ANY else relocate
    with _stops_with_exit_2():
        if field_strength is not None and not 0 < field_strength < math.inf:
            from surfcensus.config import ConfigError

            raise ConfigError(
                "invalid",
                f"--field-strength is in tesla, positive and finite, not {field_strength}",
            )
        run = _open_for_inventory(run_dir, source, subjects, field_strength, moved)
        with run, _interrupted_is_exit_130(), _terminated_like_interrupted():
            report = run.inventory(datasets=dataset or None, force=force)
            _print_inventory(run)
            _print_report(run, report)
    raise typer.Exit(1 if report.failed else 0)


def _with_fsaverage(run: Run, fsaverage: Path | None) -> None:
    """Take the template of this call from ``--fsaverage``. The directory is not written to
    ``config.yaml``: the run keeps the template itself, by content, from its first use."""
    if fsaverage is None:
        return
    from surfcensus.config import with_changes

    metrics = run.config.metrics.model_dump() | {"fsaverage_dir": str(fsaverage.resolve())}
    run.config = with_changes(run.config, metrics=metrics)


def _print_scores(run: Run) -> None:
    """One line on what the subjects were scored against."""
    from surfcensus.schemas import NO_REFERENCE

    scores = run.scores()
    if scores.empty:
        return
    counts = scores.groupby(["score_kind", "reference_id"], sort=True).size()
    parts = []
    for key, count in counts.items():
        kind, reference = key if isinstance(key, tuple) else (key, NO_REFERENCE)
        if kind == "fallback_rank":
            parts.append(f"{count} ranked without a reference (decision D16)")
        elif reference == NO_REFERENCE:
            parts.append(f"{count} without a z-map")
        else:
            parts.append(f"{count} against {reference}")
    typer.echo(f"scores: {len(scores)} subjects; {', '.join(parts)}")


@app.command("metrics")
def metrics(
    run_dir: Path = _RUN_DIR,
    stage: list[str] | None = typer.Option(
        None,
        "--stage",
        metavar="NAME",
        help="Only this stage: regional, fsaverage, reference, score, native or merge. "
        "Repeatable. Without it the first four run.",
    ),
    dataset: list[str] | None = _DATASET,
    jobs: int = typer.Option(1, "--jobs", min=1, metavar="J", help="Worker processes."),
    fsaverage: Path | None = _FSAVERAGE,
    retemplate: bool = typer.Option(
        False,
        "--retemplate",
        help="Look for the template again and keep what is found, also when it differs "
        "from the one the run was scored with.",
    ),
    force: bool = typer.Option(False, "--force", help="Compute over what is there."),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Try the units again that failed with these inputs."
    ),
) -> None:
    """Compute the metrics and scores that are pending (phase 4)."""
    from surfcensus.run import Run

    with _stops_with_exit_2(), Run(run_dir) as run:
        _with_fsaverage(run, fsaverage)
        with _interrupted_is_exit_130(), _terminated_like_interrupted():
            report = run.metrics(
                datasets=dataset or None,
                stages=stage or None,
                jobs=jobs,
                force=force,
                retry_failed=retry_failed,
                retemplate=retemplate,
            )
        _print_report(run, report)
        _print_scores(run)
    raise typer.Exit(1 if report.failed else 0)


@app.command("render")
def render(
    run_dir: Path = _RUN_DIR,
    jobs: int = typer.Option(1, "--jobs", min=1, metavar="J", help="Worker processes."),
    dataset: list[str] | None = _DATASET,
    fsaverage: Path | None = _FSAVERAGE,
    all_subjects: bool = typer.Option(
        False, "--all", help="Every complete subject, whatever the selection."
    ),
    selected: bool = typer.Option(
        False, "--selected", help="The selected subjects only (all, if nothing is selected)."
    ),
    video: bool | None = typer.Option(
        None, "--video/--no-video", help="Write anim.mp4 as well; kept for later runs."
    ),
    force: bool = typer.Option(False, "--force", help="Render over what is there."),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Try the subjects again that failed with these inputs."
    ),
    check_sample: bool = typer.Option(
        False,
        "--check-sample",
        help="Render nothing: draw a sample of the done renders to look at.",
    ),
    fraction: float | None = typer.Option(
        None, "--fraction", metavar="F", help="With --check-sample: the share drawn (0.02)."
    ),
    seed: int | None = typer.Option(
        None, "--seed", min=0, metavar="S", help="With --check-sample: the seed of the draw."
    ),
) -> None:
    """Render the pending subjects of a run into media/ (phase 6)."""
    from surfcensus.config import load_config, with_changes
    from surfcensus.run import Run

    if all_subjects and selected:
        raise typer.BadParameter("--all contradicts --selected")
    rendering = all_subjects or selected or video is not None or force or retry_failed
    rendering = rendering or fsaverage is not None
    if check_sample and (rendering or dataset or jobs != 1):
        raise typer.BadParameter("--check-sample renders nothing and takes --fraction and --seed")
    if not check_sample and (fraction is not None or seed is not None):
        raise typer.BadParameter("--fraction and --seed belong to --check-sample")
    if fraction is not None and not 0 < fraction <= 1:
        raise typer.BadParameter("is above 0 and at most 1", param_hint="--fraction")

    with _stops_with_exit_2():
        stored = load_config(run_dir)
        config = None
        if video is not None and video != stored.render_video:
            config = with_changes(stored, render_video=video)
        with Run(run_dir, config=config) as run:
            if check_sample:
                from surfcensus.render.batch import render_check_sample

                drawn = secrets.randbelow(2**31) if seed is None else seed
                listing = render_check_sample(
                    run.context(), fraction=0.02 if fraction is None else fraction, seed=drawn
                )
                summary = listing.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
                typer.echo(f"{summary}: {listing} and {listing.with_suffix('.html').name}")
                raise typer.Exit()
            _with_fsaverage(run, fsaverage)
            with _interrupted_is_exit_130(), _terminated_like_interrupted():
                report = run.render(
                    jobs=jobs,
                    datasets=dataset or None,
                    all_subjects=True if all_subjects else False if selected else None,
                    force=force,
                    retry_failed=retry_failed,
                )
            _print_report(run, report)
    raise typer.Exit(1 if report.failed else 0)


@app.command("status")
def status(
    run_dir: Path = _RUN_DIR,
    failures: bool = typer.Option(
        False,
        "--failures",
        help="List every failed unit with its reason and detail. For the terminal, not for a log.",
    ),
) -> None:
    """Say where a run stands: one line per phase, from its catalog."""
    from surfcensus.records import ALL_METRIC_STAGES
    from surfcensus.render.batch import PHASE as RENDER
    from surfcensus.render.batch import render_failures, render_status
    from surfcensus.run import Run
    from surfcensus.store import CATALOG_FILE, Catalog

    with _stops_with_exit_2(), Run(run_dir) as run:
        context = run.context()
        public = run.public()
        subjects = public.subjects()
        present = subjects[subjects["present"]]
        # a dataset found in another pipeline's layout has a second row, the same pseudonym
        pseudonyms = ", ".join(dict.fromkeys(public.datasets()["pseudonym"])) or "none"
        typer.echo(f"{run.config.mode} run, datasets: {pseudonyms}")
        unreadable = int(present["problem"].notna().sum())
        absent = len(subjects) - len(present)
        gone = f" ({absent} no longer in the source)" if absent else ""
        typer.echo(f"inventory: {len(present) - unreadable} done, {unreadable} failed{gone}")
        for name in ALL_METRIC_STAGES:
            ledger = public.work(name)
            if len(ledger):  # a stage that never ran has no line
                failed = int((ledger["status"] == "failed").sum())
                typer.echo(f"metrics {name}: {len(ledger) - failed} done, {failed} failed")
        done, failed, pending = render_status(context)
        typer.echo(f"render: {done} done, {failed} failed, {pending} pending")
        if failures:
            with Catalog(run_dir / CATALOG_FILE, readonly=True) as catalog:
                frame = catalog.failures()
            # The renders listed are the ones counted on the line above. A recorded failure
            # that the count leaves out is not one now: its subject left the source, or is
            # pending again under another configuration or with other files. An incomplete
            # subject is counted before ``render`` has recorded its missing surface.
            counted = render_failures(context)
            recorded = {uid for uid, (_, detail) in counted.items() if detail is None}
            listed = [
                (row.unit, row.phase, row.reason, row.detail)
                for row in frame.itertuples(index=False)
                if row.phase != RENDER or row.unit in recorded
            ]
            listed += [
                (uid, RENDER, reason, detail)
                for uid, (reason, detail) in counted.items()
                if detail is not None
            ]
            for uid, phase, reason, detail in sorted(listed, key=lambda row: (row[1], row[0])):
                name = _names(subjects, [uid])
                unit = uid if name == uid else f"{uid} ({name})"
                text = f": {detail}" if isinstance(detail, str) else ""
                typer.echo(f"{phase} {unit}: {reason}{text}")


# --- the rater app (milestone 4) ------------------------------------------------------------------

_RATER = typer.Option(
    None,
    "--rater",
    metavar="NAME",
    help="Who rates in this session. Not stored: without it the rater is app.rater_name of "
    "the run (local).",
)
calibration_app = typer.Typer(no_args_is_help=True, help="The calibration set of a run.")
app.add_typer(calibration_app, name="calibration")


@contextmanager
def _app_stops_with_exit_2() -> Iterator[None]:
    """:func:`_stops_with_exit_2` for the commands of the rater app: also for a core
    install without the app's packages and for a database of the app that stays locked."""
    from surfcensus.config import ConfigError

    with _stops_with_exit_2():
        try:
            yield
        except ModuleNotFoundError as exc:
            if exc.name is None or exc.name.split(".")[0] not in _APP_PACKAGES:
                raise
            raise ConfigError(
                "missing",
                f"the rater app needs the package {exc.name}; install it with "
                'pip install "surfcensus[app]"',
            ) from exc
        except Exception as exc:
            # sqlalchemy is imported by the app only, so its error is named by its module
            if type(exc).__module__.split(".")[0] != "sqlalchemy":
                raise
            text = " ".join(str(getattr(exc, "orig", exc)).split())
            raise ConfigError("invalid", f"app.sqlite cannot be used: {text}") from exc


_APP_PACKAGES = ("fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "jinja2", "starlette")


def _checked_rater(name: str | None) -> str | None:
    if name is not None and (not name or name != name.strip()):
        raise typer.BadParameter(
            "a rater's name is not empty and has no space at its ends", param_hint="--rater"
        )
    return name


@app.command(
    "review",
    short_help="Score, render and rate the FreeSurfer subjects under PATH, in one go.",
)
def review_command(
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="A SUBJECTS_DIR, an fMRIPrep output, a BIDS derivatives folder or one subject "
        "directory. It is read and never changed.",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        metavar="DIR",
        help="The run directory. Default: surfcensus-run next to PATH, else in the current "
        "directory.",
    ),
    subjects: list[str] | None = typer.Option(
        None,
        "--subjects",
        metavar="GLOB",
        help="Only the subject directories that match. Repeatable; kept for later calls.",
    ),
    rater: str | None = _RATER,
    blind: bool | None = typer.Option(
        None,
        "--blind/--no-blind",
        help="Offer the subjects in a seeded random order and keep the review screen closed "
        "until all are rated; otherwise by score, worst first. Kept for later calls.",
    ),
    require_marks: bool | None = typer.Option(
        None,
        "--require-marks/--no-require-marks",
        help="A minor or major rating needs a click mark of its class. Kept for later calls.",
    ),
    field_strength: float | None = typer.Option(
        None,
        "--field-strength",
        metavar="T",
        help="Field strength of the subjects in tesla; kept for later calls.",
    ),
    jobs: int | None = typer.Option(
        None,
        "--jobs",
        min=1,
        metavar="J",
        help="Worker processes for metrics and renders. Default: the processors, at most 8.",
    ),
    fsaverage: Path | None = _FSAVERAGE,
    video: bool | None = typer.Option(
        None, "--video/--no-video", help="Write anim.mp4 as well; kept for later calls."
    ),
    port: int = typer.Option(
        8765, "--port", min=0, max=65535, metavar="N", help="The port of the app; 0: any free one."
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser."),
    no_serve: bool = typer.Option(
        False, "--no-serve", help="Prepare everything, render every subject, and stop."
    ),
    serve_after: int | None = typer.Option(
        None,
        "--serve-after",
        min=0,
        metavar="N",
        help="Open the app when N subjects are rendered and render the others while it runs. "
        "Default 1; 0 renders every subject first.",
    ),
    relocate: bool = typer.Option(
        False,
        "--relocate",
        help="The data moved to PATH: keep the run, its subjects and their ratings.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Change --blind although the run has ratings; it is recorded."
    ),
) -> None:
    """Score, render and rate the FreeSurfer subjects under PATH: inventory, metrics,
    render and the rating app in the browser, in one go. Run it again to continue."""
    from surfcensus import review as reviewing

    rater = _checked_rater(rater)
    with _app_stops_with_exit_2():
        if not no_serve:
            import surfcensus.app.server  # noqa: F401  (before the work, not after it)
        with _interrupted_is_exit_130(), _terminated_like_interrupted():
            run = reviewing.review(
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
                open_browser=not no_browser,
                serve=not no_serve,
                serve_after=serve_after,
                relocate=relocate,
                force=force,
            )
        with run:
            nothing = reviewing.rateable(run) == 0
            if no_serve and not nothing:
                typer.echo(f"Run directory: {run.run_dir}")
    raise typer.Exit(1 if nothing else 0)


@app.command("serve")
def serve_command(
    run_dir: Path = _RUN_DIR,
    host: str = typer.Option("127.0.0.1", "--host", metavar="H", help="The address to bind to."),
    port: int = typer.Option(
        8765, "--port", min=0, max=65535, metavar="N", help="The port; 0: any free one."
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        metavar="local|team",
        help="Default: local for a local run, team for a census run.",
    ),
    rater: str | None = _RATER,
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser."),
) -> None:
    """Open the rating app on a run that is rendered (phase 7). Renders nothing."""
    from surfcensus.config import load_config

    if mode not in (None, "local", "team"):
        raise typer.BadParameter("is local or team", param_hint="--mode")
    rater = _checked_rater(rater)
    with _app_stops_with_exit_2():
        from surfcensus.app.server import serve

        if mode is None:
            mode = "local" if load_config(run_dir).mode == "local" else "team"
        serve(
            run_dir,
            host=host,
            port=port,
            mode="local" if mode == "local" else "team",
            rater=rater,
            open_browser=not no_browser,
        )


@app.command(
    "finalize",
    short_help="Write the final labels and marks from the ratings of all raters.",
)
def finalize_command(run_dir: Path = _RUN_DIR) -> None:
    """Write the final labels and marks from the ratings of all raters: one rating as it
    is, raters within one scale step as their consensus, and a list of the subjects that
    need a decision."""
    with _app_stops_with_exit_2():
        from sqlmodel import Session, select

        from surfcensus.app.db import app_db_mode, open_app_db
        from surfcensus.app.finalize import finalize
        from surfcensus.app.models import Rater
        from surfcensus.app.queue import build_plan, subjects_left, subjects_skipped
        from surfcensus.config import ConfigError, load_config
        from surfcensus.export.tables import ratings_frame
        from surfcensus.protocol import load_protocol
        from surfcensus.store import PublicCatalog, queue_scores

        config = load_config(run_dir)
        with PublicCatalog.open(run_dir) as public:
            mode = app_db_mode(run_dir)
            if mode is None:
                raise ConfigError(
                    "missing", f"{run_dir} has no ratings yet: there is no app.sqlite to read"
                )
            protocol, _ = load_protocol(run_dir)
            plan = build_plan(
                public,
                queue_scores(run_dir),
                config.app,
                mode,
                render_config_hash=config.render.config_hash(),
            )
            subjects = public.subjects()
            engine = open_app_db(run_dir, run_id=public.run_id(), mode=mode)
            try:
                with Session(engine) as session:
                    report = finalize(session, plan, protocol)
                    waiting = []
                    if mode == "local":
                        for rater in session.exec(select(Rater).order_by(Rater.name)).all():
                            left = subjects_left(session, rater, plan)
                            if left:
                                skipped = subjects_skipped(session, rater, plan, None)
                                waiting.append((rater.name, left, skipped))
                    session.commit()
            finally:
                engine.dispose()
        typer.echo(
            f"final labels: {report.consensus} by consensus, {report.single} from one rating, "
            f"{report.unratable} unratable (render failed), {len(report.open)} open"
        )
        for name, left, skipped in waiting:
            noun = "subject" if left == 1 else "subjects"
            held = f" ({skipped} skipped)" if skipped else ""
            typer.echo(
                f"  {name} still has {left} {noun} to rate{held}; labels of those are from "
                "the other ratings only"
            )
        if report.open:
            typer.echo(
                "open (raters two or more steps apart, or unratable for one of them): "
                "decide together, then rate again or adjudicate"
            )
            ratings = ratings_frame(run_dir)
            ratings = ratings[ratings["kind"] == "main"]
            local = config.mode == "local"
            # by label where the run has labels: the order two raters look for
            named = sorted((_names(subjects, [uid]) if local else uid, uid) for uid in report.open)
            for name, uid in named:
                typer.echo(f"  {name}")
                of_subject = ratings[ratings["subject_uid"] == uid]
                latest = of_subject.sort_values("assignment_id").groupby("rater").tail(1)
                for row in latest.sort_values("rater").itertuples(index=False):
                    if isinstance(row.unratable_reason, str):
                        said = f"unratable ({row.unratable_reason})"
                    else:
                        values = {"A": row.A, "B": row.B, "C": row.C}
                        said = ", ".join(
                            f"{criterion} {int(value)}"
                            for criterion, value in values.items()
                            if not _is_missing(value)
                        )
                    typer.echo(f"    {row.rater}: {said}")


def _is_missing(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))  # pyrefly: ignore[no-matching-overload]


@calibration_app.command(
    "freeze", short_help="Write calibration.yaml from the final labels and marks of a run."
)
def calibration_freeze(
    run_dir: Path = _RUN_DIR,
    subject: list[str] | None = typer.Option(
        None,
        "--subject",
        metavar="UID",
        help="Only this subject, by subject_uid. Repeatable. Without it: every subject "
        "with a final label.",
    ),
    version: str = typer.Option("1", "--version", metavar="V", help="The version of the set."),
) -> None:
    """Write calibration.yaml from the final labels and marks of a run. The file is never
    overwritten; justifications and reference examples are then written into it by hand."""
    with _app_stops_with_exit_2():
        from surfcensus.app.finalize import freeze_calibration

        path = freeze_calibration(run_dir, subjects=subject or None, version=version)
        typer.echo(str(path))


@app.command("export", short_help="Write the tables for analysis and a manifest (phase 8).")
def export_command(
    run_dir: Path = _RUN_DIR,
    out: Path | None = typer.Option(
        None,
        "--out",
        metavar="DIR",
        help="Where to write. Default: export/ (release/ with --release) in the run "
        "directory. An earlier export there is replaced, also its files of the other format.",
    ),
    fmt: str = typer.Option(
        "csv", "--format", metavar="csv|parquet", help="CSV has no types: see the README."
    ),
    release: bool = typer.Option(
        False, "--release", help="The coarser tables a census publishes. Not for a local run."
    ),
) -> None:
    """Write the tables for analysis (phase 8): subjects, scores, selection, ratings,
    marks, final marks, clusters, datasets, behaviour, skips and a manifest."""
    if fmt not in ("csv", "parquet"):
        raise typer.BadParameter("is csv or parquet", param_hint="--format")
    with _stops_with_exit_2():
        from surfcensus.export.tables import export_run

        if out is not None:
            from surfcensus.config import load_config

            # a source is only read. The export itself refuses an output in a source of a
            # local run; the roots of a census run are in the catalog, which the run opens
            if load_config(run_dir).mode == "census":
                from surfcensus.run import Run

                with Run(run_dir) as run:
                    run.refuse_output_in_sources(out)
        paths = export_run(
            run_dir,
            out,
            fmt="csv" if fmt == "csv" else "parquet",
            profile="release" if release else "analysis",
        )
        for path in paths:
            typer.echo(str(path))
