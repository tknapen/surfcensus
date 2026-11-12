"""Tests of ``surfcensus render-subject``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from synth import SyntheticSubject  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner

from surfcensus.cli import app
from surfcensus.render import RenderConfig, RenderInfo, api


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


def test_a_render_error_exits_with_1_and_one_line(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    out = tmp_path / "media"
    result = CliRunner().invoke(app, ["render-subject", str(tmp_path / "empty"), "--out", str(out)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "render failed: missing surface: no file for t1, lh_white, rh_white, lh_pial, rh_pial\n"
    )
    assert not out.exists()


def test_no_video_contradicts_a_video_path(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["render-subject", str(tmp_path), "--out", str(tmp_path / "clip.mp4"), "--no-video"]
    )
    assert result.exit_code == 2
    assert "--no-video" in result.output
    assert list(tmp_path.iterdir()) == []


def test_out_is_required_and_the_command_is_listed() -> None:
    assert CliRunner().invoke(app, ["render-subject", "some/subject"]).exit_code == 2
    assert "render-subject" in CliRunner().invoke(app, ["--help"]).output
