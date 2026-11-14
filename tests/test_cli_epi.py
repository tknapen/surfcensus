"""Tests of ``surfcensus epi-session`` (docs/epi_interface.md, Step 1): one subject of a local
folder packed without a run directory, and, with ``--serve``, shown."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import test_cli_render  # pyrefly: ignore[missing-import]
from synth_annex import snapshot_links  # pyrefly: ignore[missing-import]
from synth_func import SyntheticFuncDataset, write_func_dataset  # pyrefly: ignore[missing-import]
from typer.testing import CliRunner

from surfcensus.cli import app
from surfcensus.func import STREAM_JSON
from surfcensus.func.model import StreamInfo, read_info

said = test_cli_render.said
terminal = test_cli_render.terminal
"""The terminals of the tests of ``render-subject``: plain, forced colour, 30 columns."""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("cli-epi") / "data")


def invoke(*args: object) -> object:
    return CliRunner().invoke(app, ["epi-session", *[str(a) for a in args]])


def test_it_packs_every_session_of_a_subject_and_says_so(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    before = snapshot_links(dataset.root)
    out = tmp_path / "packs"
    result = invoke(dataset.raw_root, "--subject", "01", "--out", out)
    assert result.exit_code == 0, result.output  # pyrefly: ignore[missing-attribute]
    text = result.output  # pyrefly: ignore[missing-attribute]
    assert "Functional: 7 runs in 2 sessions of sub-01 in derivatives/fmriprep" in text
    assert "EPI packs: 7 runs to pack, about" in text
    assert "sub-01 ses-1: ses-1_task-rest_run-2: 12 volumes," in text
    assert "sub-01 ses-2: 3 of 3 runs packed," in text
    assert f"EPI packs: 7 of 7 runs of sub-01 in {out}," in text
    stream = read_info(out / STREAM_JSON, StreamInfo)
    assert stream is not None and len(stream.sessions) == 2
    assert snapshot_links(dataset.root) == before


def test_one_session_and_the_runs_named(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    out = tmp_path / "packs"
    result = invoke(
        dataset.deriv_root, "--subject", "sub-02", "--out", out, "--epi-runs", "*nback*"
    )
    assert result.exit_code == 0, result.output  # pyrefly: ignore[missing-attribute]
    assert "EPI packs: 3 of 3 runs of sub-02" in result.output  # pyrefly: ignore[missing-attribute]
    result = invoke(dataset.raw_root, "--subject", "01", "--session", "ses-2", "--out", out / "x")
    assert result.exit_code == 0, result.output  # pyrefly: ignore[missing-attribute]
    assert "1 session of sub-01" in result.output  # pyrefly: ignore[missing-attribute]


def test_no_packable_run_exits_with_1(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    result = invoke(
        dataset.raw_root, "--subject", "02", "--out", tmp_path / "o", "--epi-runs", "*std*"
    )
    assert result.exit_code == 1, result.output  # pyrefly: ignore[missing-attribute]
    assert "No run of sub-02 can be packed" in result.output  # pyrefly: ignore[missing-attribute]
    assert (tmp_path / "o" / STREAM_JSON).is_file()  # the runs are listed with their reasons


@pytest.mark.parametrize(
    ("args", "words"),
    [
        (("--subject", "09"), "not found: sub-09 has no functional run"),
        (("--subject", "01", "--session", "7"), "invalid: sub-01 has no session 7"),
        (("--subject", "01", "--raw", "/nonexistent/raw"), "not found: --raw"),
    ],
)
def test_usage_and_configuration_errors_exit_with_2_and_one_line(
    dataset: SyntheticFuncDataset, tmp_path: Path, args: tuple[str, ...], words: str, terminal: str
) -> None:
    result = invoke(dataset.raw_root, *args, "--out", tmp_path / "o")
    assert result.exit_code == 2, result.output  # pyrefly: ignore[missing-attribute]
    lines = [x for x in result.output.splitlines() if x.strip()]  # pyrefly: ignore[missing-attribute]
    assert len(lines) == 1 and lines[0].startswith(words), lines
    assert not (tmp_path / "o").exists()


def test_an_out_in_a_source_is_refused(dataset: SyntheticFuncDataset, terminal: str) -> None:
    before = snapshot_links(dataset.root)
    result = invoke(dataset.raw_root, "--subject", "01", "--out", dataset.deriv_root / "packs")
    assert result.exit_code == 2
    assert said(result.output).startswith("invalid: the output directory")  # pyrefly: ignore[missing-attribute]
    assert snapshot_links(dataset.root) == before


def test_bad_parameters(dataset: SyntheticFuncDataset, tmp_path: Path, terminal: str) -> None:
    result = invoke(dataset.raw_root, "--subject", "01", "--out", " ")
    assert result.exit_code == 2 and "the output path is empty" in said(result.output)  # pyrefly: ignore[missing-attribute]
    result = invoke(dataset.raw_root, "--subject", "01", "--out", tmp_path, "--port", "9")
    assert result.exit_code == 2 and "--port and --no-browser go with --serve" in said(
        result.output
    )  # pyrefly: ignore[missing-attribute]
    result = invoke(dataset.raw_root, "--out", tmp_path)
    assert result.exit_code == 2 and "--subject" in said(result.output)  # pyrefly: ignore[missing-attribute]


def test_the_command_is_listed(terminal: str) -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert "epi-session" in said(result.output)  # pyrefly: ignore[missing-attribute]
    assert "without a run directory" in said(result.output)  # pyrefly: ignore[missing-attribute]


def test_serve_hands_the_packs_to_the_page(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import surfcensus.app.epi as epi

    called: list[tuple[Path, dict[str, object]]] = []

    def fake(out_dir: Path, **kwargs: object) -> None:
        called.append((Path(out_dir), kwargs))

    monkeypatch.setattr(epi, "serve_packs", fake)
    out = tmp_path / "packs"
    result = invoke(
        dataset.raw_root, "--subject", "02", "--out", out, "--serve", "--port", "0", "--no-browser"
    )
    assert result.exit_code == 0, result.output  # pyrefly: ignore[missing-attribute]
    ((where, kwargs),) = called
    assert where == out and kwargs["port"] == 0 and kwargs["open_browser"] is False
    assert kwargs["stop"] is not None
    assert "The EPI page has stopped." in result.output  # pyrefly: ignore[missing-attribute]


def test_serve_end_to_end_and_sigterm(dataset: SyntheticFuncDataset, tmp_path: Path) -> None:
    """The command as a process: the address, the stream of the page and one chunk, then a
    SIGTERM, which stops the page in order with exit code 0."""
    out = tmp_path / "packs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from surfcensus.cli import app; app()",
            "epi-session",
            str(dataset.raw_root),
            "--subject",
            "02",
            "--out",
            str(out),
            "--serve",
            "--port",
            "0",
            "--no-browser",
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = None
        lines = []
        deadline = time.monotonic() + 120
        assert process.stderr is not None
        while url is None and time.monotonic() < deadline:
            line = process.stderr.readline()
            if not line:
                break
            lines.append(line)
            if line.startswith("EPI page: "):
                url = line.split()[2]
        assert url is not None, "".join(lines)
        with urllib.request.urlopen(url + "api/epi/stream", timeout=10) as answer:
            shape = json.loads(answer.read())
        assert json.dumps(shape).count("sub-02") >= 1
        chunk = _first_chunk(shape)
        with urllib.request.urlopen(url.rstrip("/") + chunk, timeout=10) as answer:
            assert len(answer.read()) > 0
        process.send_signal(signal.SIGTERM)
        _, rest = process.communicate(timeout=60)
        assert process.returncode == 0, "".join(lines) + rest
        assert "The EPI page has stopped." in rest
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def _first_chunk(shape: object) -> str:
    """The URL of the first chunk of the first packed run of the first session."""
    for session in shape["sessions"]:  # pyrefly: ignore[bad-index]
        for run in session["runs"]:
            if run.get("pack"):
                return session["media_url"] + run["pack"]["streams"]["raw"]["chunks"][0]["file"]
    raise AssertionError("no packed run in the shape")


def test_ctrl_c_while_the_page_is_served_says_the_packs_are_complete(
    dataset: SyntheticFuncDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    """Page review 6: once the packs are complete, a Ctrl-C (or a SIGTERM that is taken as
    one) while the page is served says that the page was stopped, not that the packs are not
    complete."""
    import surfcensus.app.epi as epi

    def interrupted(out_dir: Path, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(epi, "serve_packs", interrupted)
    out = tmp_path / "packs"
    result = invoke(dataset.raw_root, "--subject", "02", "--out", out, "--serve", "--no-browser")
    assert result.exit_code == 130, result.output  # pyrefly: ignore[missing-attribute]
    text = said(result.output)  # pyrefly: ignore[missing-attribute]
    assert f"interrupted; the EPI page was stopped; the packs are complete in {out}" in text
    assert "not complete" not in text


def test_a_second_call_keeps_the_packs_and_force_packs_again(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    out = tmp_path / "packs"
    assert invoke(dataset.raw_root, "--subject", "02", "--out", out).exit_code == 0  # pyrefly: ignore[missing-attribute]
    result = invoke(dataset.raw_root, "--subject", "02", "--out", out)
    text = result.output  # pyrefly: ignore[missing-attribute]
    assert (
        result.exit_code == 0 and "EPI packs: 0 runs to pack, 4 kept from an earlier call" in text
    )  # pyrefly: ignore[missing-attribute]
    assert text.count("kept (its pack is current)") == 4
    result = invoke(dataset.raw_root, "--subject", "02", "--out", out, "--force")
    text = result.output  # pyrefly: ignore[missing-attribute]
    assert result.exit_code == 0 and text.count("volumes, packed,") == 4  # pyrefly: ignore[missing-attribute]


SLOW_SECOND_CALL = """
import sys, time
import surfcensus.func.pack as pack
real = pack.pack_run
def slow(*args, **kwargs):
    info = real(*args, **kwargs)
    print("ONE RUN PACKED", file=sys.stderr, flush=True)
    time.sleep(60)
    return info
pack.pack_run = slow
from surfcensus.cli import app
app()
"""


def test_a_second_call_stopped_by_sigterm_leaves_a_servable_output(
    dataset: SyntheticFuncDataset, tmp_path: Path
) -> None:
    """A second call into a complete --out, stopped by SIGTERM after it has replaced one pack:
    stream.json and session.json are those of the first call, and every pack they list is
    whole, so the output can still be served."""
    from surfcensus.func import SESSION_JSON, packread
    from surfcensus.func.model import SessionInfo
    from surfcensus.func.pack import content_digest

    out = tmp_path / "packs"
    assert invoke(dataset.raw_root, "--subject", "01", "--out", out).exit_code == 0  # pyrefly: ignore[missing-attribute]
    stream_before = (out / STREAM_JSON).read_bytes()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            SLOW_SECOND_CALL,
            "epi-session",
            str(dataset.raw_root),
            "--subject",
            "01",
            "--out",
            str(out),
            "--force",
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stderr is not None
        lines = []
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = process.stderr.readline()
            if not line:
                break
            lines.append(line)
            if line.startswith("ONE RUN PACKED"):
                break
        assert lines and lines[-1].startswith("ONE RUN PACKED"), "".join(lines)
        process.send_signal(signal.SIGTERM)
        _, rest = process.communicate(timeout=60)
        assert process.returncode == 143, "".join(lines) + rest
        assert "terminated; packing was stopped; the packs that are done stay" in rest
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert (out / STREAM_JSON).read_bytes() == stream_before
    stream = read_info(out / STREAM_JSON, StreamInfo)
    assert stream is not None
    n = 0
    for sid in stream.sessions:
        info = read_info(out / sid / SESSION_JSON, SessionInfo)
        assert info is not None
        for run in info.runs:
            if run.reason is None:
                pack = packread.read_pack(out / sid, run.run)
                assert pack is not None and packread.pack_on_disk(out / sid, run.run)
                assert content_digest(out / sid, pack) == pack.content_digest
                n += 1
        assert not [p.name for p in (out / sid).iterdir() if p.name.startswith(".")]
    assert n == 7
