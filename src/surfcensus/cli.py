"""Command-line entry point. Subcommands are specified in docs/architecture.md."""

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
