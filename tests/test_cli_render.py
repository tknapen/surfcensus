"""Tests of ``surfcensus render-subject``."""

from __future__ import annotations

import json
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from synth import SyntheticSubject  # pyrefly: ignore[missing-import]
from typer import rich_utils
from typer.testing import CliRunner

from surfcensus.cli import app
from surfcensus.render import RenderConfig, RenderInfo, api


@pytest.fixture(params=["plain", "colour", "colour, 30 columns"])
def terminal(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """The ways typer writes a usage error or a help page, each of them in any environment.

    Typer hands these to Rich. It forces Rich's terminal output when ``GITHUB_ACTIONS``,
    ``FORCE_COLOR`` or ``PY_COLORS`` is set, as on every CI runner, and Rich then puts escape
    codes around and inside words (``--no-video`` becomes three coloured pieces). Rich also
    wraps to the width of the terminal that pytest runs in, which ``COLUMNS`` overrides.
    Typer reads its variables when it is imported, so its switch is set directly; the
    variables that Rich reads for itself are set or removed, so that each case is the same
    wherever the tests run.
    """
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", True if "colour" in request.param else None)
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", None)
    for name in ("FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "30" if "30" in request.param else "80")
    return request.param


def said(output: str) -> str:
    """The words of a usage error or a help page, whatever the terminal: without escape
    codes, without the frame Rich draws around them, and with every run of white space
    (Rich wraps lines) as one space."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    return " ".join(re.sub(r"[\u2500-\u257f]", " ", text).split())


def test_render_subject_end_to_end(synth_subject_pial_t1: SyntheticSubject, tmp_path: Path) -> None:
    out = tmp_path / "media"
    result = CliRunner().invoke(
        app, ["render-subject", str(synth_subject_pial_t1.path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    # one line: the output directory, the slice counts, the elapsed seconds
    line = re.escape(str(out)) + r": 61 sagittal, 54 coronal, 61 axial slices, \d+\.\d s\n"
    assert re.fullmatch(line, result.output)
    info = RenderInfo.model_validate_json((out / "render.json").read_text())
    assert info.config == RenderConfig()
    assert info.inputs["lh_pial"].name == "lh.pial.T1"
    assert len(list((out / "tiles").iterdir())) == 2 * 176
    assert {p.name for p in out.iterdir()} == {
        "tiles", "anim.mp4", "poster.webp", "vertex_map.npz", "render.json"
    }  # fmt: skip

    # a second run finds the render complete and leaves it alone
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*")}
    again = CliRunner().invoke(
        app, ["render-subject", str(synth_subject_pial_t1.path), "--out", str(out)]
    )
    assert again.exit_code == 0 and re.fullmatch(line, again.output), again.output
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*")} == stamps


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], {"video": True, "force": False}),
        (["--force"], {"video": True, "force": True}),
        (["--no-video"], {"video": False, "force": False}),
        (["--no-video", "--force"], {"video": False, "force": True}),
    ],
)
def test_flags_reach_three_plane(
    flags: list[str],
    expected: dict[str, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def record(subject_dir: Path, out: Path, **kwargs: bool) -> None:
        calls.append((subject_dir, out, kwargs))
        raise api.RenderError("render failure", "stop here")

    monkeypatch.setattr(api, "three_plane", record)
    result = CliRunner().invoke(
        app, ["render-subject", "some/subject", "--out", str(tmp_path), *flags]
    )
    assert calls == [(Path("some/subject"), tmp_path, expected)]
    assert result.exit_code == 1


def test_a_render_error_exits_with_1_and_one_line(tmp_path: Path, terminal: str) -> None:
    (tmp_path / "empty").mkdir()
    out = tmp_path / "media"
    result = CliRunner().invoke(app, ["render-subject", str(tmp_path / "empty"), "--out", str(out)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "render failed: missing surface: no file for t1, lh_white, rh_white, lh_pial, rh_pial\n"
    )
    assert not out.exists()


def test_a_missing_encoder_exits_with_1_and_one_line_before_anything_is_written(
    synth_subject_pial_t1: SyntheticSubject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "media"
    subject = str(synth_subject_pial_t1.path)
    done = CliRunner().invoke(app, ["render-subject", subject, "--out", str(out), "--no-video"])
    assert done.exit_code == 0, done.output
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(tmp_path / "no-ffmpeg"))
    result = CliRunner().invoke(app, ["render-subject", subject, "--out", str(out)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert re.fullmatch(
        r"render failed: render failure: the video cannot be encoded \(the ffmpeg binary "
        r"'.*no-ffmpeg' is missing or not executable\)\n",
        result.stderr,
    ), result.stderr
    # the render without video is still complete
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert (out / "render.json").is_file()


def test_no_video_contradicts_a_video_path(tmp_path: Path, terminal: str) -> None:
    result = CliRunner().invoke(
        app, ["render-subject", str(tmp_path), "--out", str(tmp_path / "clip.mp4"), "--no-video"]
    )
    assert result.exit_code == 2
    assert "--no-video contradicts an --out that names a video" in said(result.output)
    assert ("\x1b[" in result.output) == ("colour" in terminal)  # the fixture does what it says
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("out", ["", "  "])
def test_an_empty_out_is_refused(
    synth_subject_pial_t1: SyntheticSubject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    out: str,
    terminal: str,
) -> None:
    """``--out "$UNSET"`` would otherwise render into the current directory."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["render-subject", str(synth_subject_pial_t1.path), "--out", out, "--no-video"]
    )
    assert result.exit_code == 2
    assert "--out" in said(result.output) and "the output path is empty" in said(result.output)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("where", [".", "media", "mri", "surf/clip.mp4"])
def test_an_out_inside_the_subject_directory_is_refused(
    synth_subject_pial_t1: SyntheticSubject, tmp_path: Path, where: str, terminal: str
) -> None:
    """A source directory is only read. With ``--out .`` given from inside it, a render
    without video would also have removed an ``anim.mp4`` found there."""
    subject = Path(shutil.copytree(synth_subject_pial_t1.path, tmp_path / "sub-01"))
    (subject / "anim.mp4").write_text("my video")
    before = {p.relative_to(subject).as_posix() for p in subject.rglob("*")}
    flags = [] if where.endswith(".mp4") else ["--no-video"]
    result = CliRunner().invoke(
        app, ["render-subject", str(subject), "--out", str(subject / where), *flags]
    )
    assert result.exit_code == 2
    assert "--out" in said(result.output)
    assert "lies in a directory that the inputs are read from" in said(result.output)
    assert {p.relative_to(subject).as_posix() for p in subject.rglob("*")} == before
    assert (subject / "anim.mp4").read_text() == "my video"


def test_a_video_of_the_users_in_the_output_directory_is_kept(
    synth_subject_pial_t1: SyntheticSubject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out .`` in a directory that holds other things: ``anim.mp4`` is a name the
    renderer writes, and without a ``render.json`` that lists it the file is not its own."""
    (tmp_path / "anim.mp4").write_text("holiday film")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["render-subject", str(synth_subject_pial_t1.path), "--out", ".", "--no-video"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "anim.mp4").read_text() == "holiday film"
    assert json.loads((tmp_path / "render.json").read_text())["video"] is None


def test_an_output_that_cannot_be_written_exits_with_1_and_one_line(
    synth_subject_pial_t1: SyntheticSubject, tmp_path: Path, terminal: str
) -> None:
    out = tmp_path / "a-file"
    out.write_text("not a directory")
    result = CliRunner().invoke(
        app, ["render-subject", str(synth_subject_pial_t1.path), "--out", str(out), "--no-video"]
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert re.fullmatch(r"render failed: .*a-file.*\n", result.stderr), result.stderr
    assert out.read_text() == "not a directory"
    assert [p.name for p in tmp_path.iterdir()] == ["a-file"]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM cannot be caught on Windows")
def test_sigterm_removes_the_temporary_files(
    synth_subject_pial_t1: SyntheticSubject, tmp_path: Path
) -> None:
    """A scheduler stops a job with SIGTERM; like Ctrl-C it must not leave the hidden
    directory of a video render behind."""
    (tmp_path / "videos").mkdir()
    command = [sys.executable, "-c", "from surfcensus.cli import app; app()", "render-subject"]
    process = subprocess.Popen(
        [*command, str(synth_subject_pial_t1.path), "--out", str(tmp_path / "videos" / "clip.mp4")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not any((tmp_path / "videos").glob(".surfcensus-*/tiles/*.webp")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        output = process.communicate(timeout=60)
    finally:
        process.kill()
    assert process.returncode == 128 + signal.SIGTERM, output
    assert list((tmp_path / "videos").iterdir()) == []


def test_out_is_required_and_the_command_is_listed(terminal: str) -> None:
    missing = CliRunner().invoke(app, ["render-subject", "some/subject"])
    assert missing.exit_code == 2
    assert "--out" in said(missing.output)
    listed = CliRunner().invoke(app, ["--help"])
    assert listed.exit_code == 0
    assert "render-subject" in said(listed.output)


def test_said_reads_through_colour_and_frames() -> None:
    styled = (
        "\x1b[31m\u256d\u2500 Error \u2500\u256e\x1b[0m\n"
        "\x1b[31m\u2502\x1b[0m Invalid value: \x1b[1;36m-\x1b[0m\x1b[1;36m-no\x1b[0m"
        "\x1b[1;36m-video\x1b[0m     \x1b[31m\u2502\x1b[0m\n"
        "\x1b[31m\u2502\x1b[0m contradicts \x1b[31m\u2502\x1b[0m\n"
    )
    assert "--no-video" not in styled  # what the assertion on the raw output ran into on CI
    assert said(styled) == "Error Invalid value: --no-video contradicts"
    assert said("plain  text\n") == "plain text"
