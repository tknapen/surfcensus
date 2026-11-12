"""Command-line entry point. Subcommands are specified in docs/architecture.md."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from surfcensus import __version__

app = typer.Typer(
    no_args_is_help=True,
    help="FreeSurfer occipital surface QC: score, render, rate.",
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("render-subject")
def render_subject(
    subject_dir: Path = typer.Argument(..., help="A FreeSurfer subject directory."),
    out: str = typer.Option(
        ..., "--out", metavar="PATH", help="Media directory to write, or the path of an .mp4."
    ),
    force: bool = typer.Option(False, "--force", help="Render even if the output is up to date."),
    no_video: bool = typer.Option(False, "--no-video", help="Do not write anim.mp4."),
) -> None:
    """Render one subject without a run directory: tiles, poster, video, render.json."""
    from surfcensus.render import RenderError, three_plane  # imports matplotlib
    from surfcensus.render.model import PANEL_ORDER

    if not out.strip():  # an unset shell variable; as a path it would be the current directory
        raise typer.BadParameter("the output path is empty", param_hint="--out")
    target = Path(out)
    if no_video and target.suffix.lower() == ".mp4":
        raise typer.BadParameter("--no-video contradicts an --out that names a video")
    start = time.perf_counter()
    try:
        with _terminated_like_interrupted():
            info = three_plane(subject_dir, target, video=not no_video, force=force)
    except RenderError as exc:
        typer.echo(f"render failed: {' '.join(str(exc).split())}", err=True)
        raise typer.Exit(1) from exc
    except OSError as exc:  # the output cannot be written: not a directory, read-only, full
        typer.echo(f"render failed: {' '.join(str(exc).split())}", err=True)
        raise typer.Exit(1) from exc
    counts = ", ".join(f"{len(info.panels[name].slices)} {name}" for name in PANEL_ORDER)
    typer.echo(f"{target}: {counts} slices, {time.perf_counter() - start:.1f} s")


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
