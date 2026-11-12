"""Command-line entry point. Subcommands are specified in docs/architecture.md."""

from __future__ import annotations

import time
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
    out: Path = typer.Option(..., "--out", help="Media directory to write."),
    force: bool = typer.Option(False, "--force", help="Render even if the output is up to date."),
    no_video: bool = typer.Option(False, "--no-video", help="Do not write anim.mp4."),
) -> None:
    """Render one subject without a run directory: tiles, poster, video, render.json."""
    from surfcensus.render import RenderError, three_plane  # imports matplotlib
    from surfcensus.render.model import PANEL_ORDER

    if no_video and out.suffix.lower() == ".mp4":
        raise typer.BadParameter("--no-video contradicts an --out that names a video")
    start = time.perf_counter()
    try:
        info = three_plane(subject_dir, out, video=not no_video, force=force)
    except RenderError as exc:
        typer.echo(f"render failed: {' '.join(str(exc).split())}", err=True)
        raise typer.Exit(1) from exc
    counts = ", ".join(f"{len(info.panels[name].slices)} {name}" for name in PANEL_ORDER)
    typer.echo(f"{out}: {counts} slices, {time.perf_counter() - start:.1f} s")
