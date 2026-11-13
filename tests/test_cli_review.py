"""Tests of ``surfcensus review``, ``serve``, ``finalize``, ``calibration freeze`` and
``export``: what they print, their exit codes, and that the app of a review stops at Ctrl-C
and at SIGTERM.

The runs are made by ``review`` on light subjects (16^3 voxels, 42 vertices per surface, no
template); the test that needs scores against a reference has its own.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import (  # pyrefly: ignore[missing-import]
    write_light_template,
    write_linked_tree,
    write_tree,
)
from typer.testing import CliRunner, Result

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus.app.server import create_app
from surfcensus.calibration import CALIBRATION_FILE
from surfcensus.cli import app
from surfcensus.config import ConfigError, load_config, with_changes, write_config
from surfcensus.export.tables import EXPORT_TABLES, MANIFEST, export_run
from surfcensus.privacy import HashKey
from surfcensus.review import DEFAULT_RUN_DIR
from surfcensus.run import Run
from surfcensus.sources.local import local
from surfcensus.store import PublicCatalog

BASE = "http://127.0.0.1"
LABELS = ("sub-01", "sub-02", "sub-03")


def cli(*args: object) -> Result:
    return CliRunner().invoke(app, [str(arg) for arg in args])


def said(output: str) -> str:
    """The words of a usage error, whatever the terminal: without escape codes, without the
    frame Rich draws, with every run of white space as one space."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    return " ".join(re.sub(r"[─-╿]", " ", text).split())


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """An fMRIPrep output with three light subjects; ``sub-03`` lacks ``surf/lh.pial``."""
    return write_tree(tmp_path / "data" / "study", "fmriprep", LABELS, incomplete=["sub-03"])


@pytest.fixture
def run_dir(study: Path) -> Path:
    """The run that ``surfcensus review study --no-serve`` makes beside the data."""
    result = cli("review", study, "--no-serve", "--jobs", 1)
    assert result.exit_code == 0, result.output
    return study.parent / DEFAULT_RUN_DIR


def uids(run_dir: Path) -> dict[str, str]:
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def rate_all(run_dir: Path, rater: str, values: dict[str, dict[str, int] | str]) -> None:
    """Rate every subject the app offers as ``rater``: ``values`` gives, per label, the
    three values or the reason the subject is unratable."""
    labels = {uid: label for label, uid in uids(run_dir).items()}
    with TestClient(create_app(run_dir, rater=rater), base_url=BASE) as client:
        version = client.get("/api/session").json()["protocol"]["version"]
        answer = client.get("/api/next").json()
        clock = 0
        while answer["status"] == "assignment":
            shown = answer["assignment"]
            given = values[labels[shown["subject_uid"]]]
            clock += 1
            body: dict[str, Any] = {
                "assignment_id": shown["assignment_id"],
                "protocol_version": version,
                "client_ts": f"2026-11-13T12:00:{clock:02d}.000Z",
                "ms_on_subject": 1000 * clock,
            }
            body |= {"unratable": given} if isinstance(given, str) else {"values": given}
            posted = client.post("/api/rating", json=body)
            assert posted.status_code == 200, posted.text
            answer = posted.json()["next"]
        assert answer["status"] == "done"


def app_rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(run_dir / "app.sqlite")
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# --- review ---------------------------------------------------------------------------------------


def test_the_command_table_has_one_line_per_command() -> None:
    """``surfcensus --help`` lists each command with the first paragraph of its help and keeps
    the line breaks of a docstring, so a summary of two source lines came out ragged
    (``... inventory, metrics,`` / ``render and ...``). Each has a summary of one line."""
    import inspect

    from typer.main import get_command

    def summaries(group: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
        for name, command in group.commands.items():
            text = command.short_help or inspect.cleandoc(command.help or "").split("\n\n")[0]
            yield prefix + name, text
            if hasattr(command, "commands"):
                yield from summaries(command, f"{prefix}{name} ")

    listed = dict(summaries(get_command(app)))
    assert {"review", "serve", "finalize", "export", "calibration freeze"} <= set(listed)
    assert all(text for text in listed.values())
    assert {name: text for name, text in listed.items() if "\n" in text} == {}


def test_review_without_the_app_prepares_the_run_and_says_what_it_did(study: Path) -> None:
    before = snapshot_tree(study)
    result = cli("review", study, "--no-serve", "--jobs", 1)
    run_dir = study.parent / DEFAULT_RUN_DIR
    assert result.exit_code == 0, result.output
    assert result.stdout == f"Run directory: {run_dir}\n"
    lines = result.stderr.splitlines()
    assert lines[0] == f"Run directory: {run_dir} (new)"
    assert lines[1] == (
        "Found 3 subjects (fMRIPrep layout). 1 cannot be rendered: sub-03 (no surf/lh.pial)."
    )
    assert "Scores: " in result.stderr
    assert "Rendered: 2 of 3 subjects, 1 cannot be rated." in lines
    assert "  not rendered, missing surface: 1 (sub-03)" in lines
    assert not any(line.startswith("Rating app") for line in lines)
    assert snapshot_tree(study) == before

    again = cli("review", study, "--no-serve", "--jobs", 1)
    assert again.exit_code == 0
    assert again.stderr.splitlines()[0] == f"Run directory: {run_dir} (continued)"


def test_review_ends_with_1_when_no_subject_can_be_rendered(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "subjects", "subjects_dir", LABELS, incomplete=LABELS)
    for options in (["--no-serve"], ["--no-browser", "--port", "0"]):
        result = cli("review", tree, "--run-dir", tmp_path / "run", *options)
        assert result.exit_code == 1, result.output
        assert result.stdout == ""
        assert "3 cannot be rendered: sub-01 (no surf/lh.pial)" in result.stderr
        assert result.stderr.splitlines()[-1] == (
            "No subject of this run can be rendered, so there is nothing to rate."
        )


def test_review_stops_with_exit_2_for_what_is_wrong_with_the_call(
    study: Path, run_dir: Path, tmp_path: Path
) -> None:
    def refused(*args: object) -> str:
        result = cli("review", *args, "--no-serve")
        assert result.exit_code == 2, result.output
        assert result.stdout == ""
        return said(result.stderr)

    assert refused(tmp_path / "nowhere").startswith("not found: ")
    assert refused(study, "--subjects", "sub-9*").startswith("not found: no subject matches")
    assert "a rater's name is not empty" in refused(study, "--rater", " ada")
    assert "a rater's name is not empty" in refused(study, "--rater", "")
    assert refused(study, "--field-strength", "-3").startswith("invalid: field_strength")
    assert "--jobs" in refused(study, "--jobs", 0)
    assert "--serve-after" in refused(study, "--serve-after", -1)
    assert "--port" in refused(study, "--port", 70000)
    moved = study.rename(study.parent / "moved")
    assert refused(moved, "--run-dir", run_dir).startswith("other root: ")
    result = cli("review", moved, "--run-dir", run_dir, "--relocate", "--no-serve")
    assert result.exit_code == 0, result.output


def test_review_refuses_to_change_blind_under_ratings_without_force(
    study: Path, run_dir: Path
) -> None:
    rate_all(run_dir, "local", {"sub-01": {"A": 0, "B": 0, "C": 0}, "sub-02": "render failure"})
    result = cli("review", study, "--no-serve", "--blind")
    assert result.exit_code == 2
    assert result.stderr.startswith("invalid: this run is not blind and has 2 submissions")
    assert result.stderr.count("\n") == 1 and "--force" in result.stderr
    assert not load_config(run_dir).app.blind
    forced = cli("review", study, "--no-serve", "--blind", "--force")
    assert forced.exit_code == 0 and load_config(run_dir).app.blind
    # and it holds until it is given again
    assert cli("review", study, "--no-serve").exit_code == 0 and load_config(run_dir).app.blind


def test_review_without_the_packages_of_the_app_says_how_to_get_them(
    study: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in [name for name in sys.modules if name.startswith("surfcensus.app.server")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    result = cli("review", study, "--no-browser")
    assert result.exit_code == 2
    assert result.stderr.strip() == (
        "missing: the rater app needs the package fastapi; install it with "
        'pip install "surfcensus[app]"'
    )
    # nothing was done before the refusal, and without the app the command works
    assert not (study.parent / DEFAULT_RUN_DIR).exists()
    assert cli("review", study, "--no-serve", "--jobs", 1).exit_code == 0


# --- review with the app, as a process ------------------------------------------------------------


def _default_sigint() -> None:
    # a test session started with SIGINT ignored (a job of a shell without job control)
    # would hand that on, and Python then installs no handler for Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@pytest.fixture
def review_process(study: Path) -> Iterator[subprocess.Popen[str]]:
    command = [
        sys.executable,
        "-c",
        "from surfcensus.cli import app; app()",
        "review",
        str(study),
        "--no-browser",
        "--port",
        "0",
        "--jobs",
        "1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ | {"PYTHONUNBUFFERED": "1"},
        preexec_fn=_default_sigint,
    )
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _until_serving(process: subprocess.Popen[str]) -> tuple[str, list[str]]:
    """The URL of the app and the lines of stderr up to the one that names it."""
    assert process.stderr is not None
    lines = []
    while True:
        line = process.stderr.readline()
        assert line, f"the command ended early:\n{''.join(lines)}"
        lines.append(line.rstrip("\n"))
        found = re.fullmatch(
            r"Rating app: (http://127\.0\.0\.1:\d+/) \(Ctrl-C to stop\)", lines[-1]
        )
        if found:
            return found.group(1), lines


@pytest.mark.skipif(os.name == "nt", reason="signals of a POSIX system")
@pytest.mark.parametrize("stop_with", [signal.SIGINT, signal.SIGTERM])
def test_the_app_of_a_review_stops_at_ctrl_c_and_at_sigterm(
    study: Path, review_process: subprocess.Popen[str], stop_with: signal.Signals
) -> None:
    url, lines = _until_serving(review_process)
    run_dir = study.parent / DEFAULT_RUN_DIR
    assert lines[0] == f"Run directory: {run_dir} (new)"
    # the app opens at the first render; the second subject is rendered behind it
    assert "Rendered: 1 of 3 subjects, 1 to render while the app runs, 1 cannot be rated." in lines
    with httpx.Client(base_url=url, trust_env=False) as client:
        assert client.get("/healthz").status_code == 200
        page = client.get("/")
        assert page.status_code == 200 and "<title>" in page.text
        deadline = time.monotonic() + 60
        while client.get("/api/next").json()["progress"]["rendered"] < 2:
            assert time.monotonic() < deadline, "the second subject was never rendered"
            time.sleep(0.1)
    review_process.send_signal(stop_with)
    if stop_with == signal.SIGTERM:
        # a process manager may signal a whole group, and a wrapper may pass the signal on:
        # the second one arrives while the command is stopping, and changes nothing
        time.sleep(0.05)
        review_process.send_signal(stop_with)
    # read on from the stream the first lines came from: it has read ahead
    assert review_process.stderr is not None and review_process.stdout is not None
    err = review_process.stderr.read()
    out = review_process.stdout.read()
    assert review_process.wait(timeout=60) == 0, err
    assert out == ""
    assert re.search(r"^  \d+\.\d s from the start of the command to the app\.$", err, re.M)
    assert err.splitlines()[-1] == (
        "0 rated, 2 to go. Run the same command to continue. "
        f"Export: surfcensus export --run-dir {run_dir}"
    )
    assert "Traceback" not in err
    with pytest.raises(httpx.TransportError), httpx.Client(trust_env=False) as client:
        client.get(url + "healthz")


# --- serve ----------------------------------------------------------------------------------------


def test_serve_refuses_what_it_cannot_serve(run_dir: Path, tmp_path: Path) -> None:
    def refused(*args: object) -> str:
        result = cli("serve", "--run-dir", run_dir, "--no-browser", *args)
        assert result.exit_code == 2, result.output
        return said(result.stderr)

    assert "is local or team" in refused("--mode", "census")
    assert refused("--mode", "team").startswith("invalid: team mode")
    assert refused("--host", "0.0.0.0").startswith("invalid: a local server listens on")
    assert "a rater's name is not empty" in refused("--rater", "ada ")
    missing = cli("serve", "--run-dir", tmp_path / "none", "--no-browser")
    assert missing.exit_code == 2 and missing.stderr.startswith("missing: ")
    # renders of another render config, the recorded failure of sub-03 among them: the app
    # would show nothing of them
    config = load_config(run_dir)
    write_config(
        run_dir, with_changes(config, render=config.render.model_dump() | {"panel_px": 128})
    )
    assert refused() == (
        f"invalid: 3 renders are out of date; run surfcensus render --run-dir {run_dir}"
    )
    assert not (run_dir / "app.sqlite").exists()


# --- finalize -------------------------------------------------------------------------------------


def test_finalize_needs_ratings(run_dir: Path, tmp_path: Path) -> None:
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 2
    assert (
        result.stderr == f"missing: {run_dir} has no ratings yet: there is no app.sqlite to read\n"
    )
    assert cli("finalize", "--run-dir", tmp_path / "none").exit_code == 2


def test_finalize_of_two_raters_prints_the_counts_and_the_open_subjects(
    tmp_path: Path,
) -> None:
    tree = write_tree(
        tmp_path / "subjects", "subjects_dir", ["sub-01", "sub-02", "sub-03", "sub-04"]
    )
    run_dir = tmp_path / "run"
    (tree / "sub-04" / "surf" / "lh.pial").unlink()
    assert cli("review", tree, "--run-dir", run_dir, "--no-serve", "--blind").exit_code == 0
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(
        run_dir,
        "ada",
        {"sub-01": {"A": 1, "B": 0, "C": 0}, "sub-02": {"A": 2, "B": 0, "C": 1}, "sub-03": none},
    )
    one = cli("finalize", "--run-dir", run_dir)
    assert one.exit_code == 0, one.output
    assert one.stdout.splitlines() == [
        "final labels: 0 by consensus, 3 from one rating, 1 unratable (render failed), 0 open"
    ]
    rate_all(
        run_dir,
        "ben",
        {
            "sub-01": {"A": 0, "B": 0, "C": 0},
            "sub-02": {"A": 0, "B": 0, "C": 1},
            "sub-03": "image too poor to judge",
        },
    )
    two = cli("finalize", "--run-dir", run_dir)
    assert two.exit_code == 0, two.output
    assert two.stdout.splitlines() == [
        "final labels: 1 by consensus, 0 from one rating, 1 unratable (render failed), 2 open",
        "open (raters two or more steps apart, or unratable for one of them): decide "
        "together, then rate again or adjudicate",
        "  sub-02",
        "    ada: A 2, B 0, C 1",
        "    ben: A 0, B 0, C 1",
        "  sub-03",
        "    ada: A 0, B 0, C 0",
        "    ben: unratable (image too poor to judge)",
    ]
    by_label = uids(run_dir)
    labels = app_rows(
        run_dir, "SELECT subject_uid, criterion, value, method FROM final_labels ORDER BY 1, 2"
    )
    assert [row for row in labels if row[0] == by_label["sub-01"]] == [
        (by_label["sub-01"], "A", 0, "consensus"),
        (by_label["sub-01"], "B", 0, "consensus"),
        (by_label["sub-01"], "C", 0, "consensus"),
    ]
    assert {row[0] for row in labels} == {by_label["sub-01"], by_label["sub-04"]}
    # a second call writes the same
    assert cli("finalize", "--run-dir", run_dir).stdout == two.stdout


def test_a_run_of_the_phase_commands_is_rated_and_finalized_like_one_of_review(
    study: Path, tmp_path: Path
) -> None:
    # `inventory` and `render` write no selection. The selection of a local run is then
    # every present subject, for the app and for `finalize` as for `render`
    run_dir = tmp_path / "run"
    assert cli("inventory", "--run-dir", run_dir, "--source", study).exit_code == 0
    rendered = cli("render", "--run-dir", run_dir, "--jobs", 1)
    assert rendered.exit_code == 1 and "render: 2 done, 0 skipped, 1 failed" in rendered.output
    with PublicCatalog.open(run_dir) as public:
        assert len(public.selection()) == 0
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(run_dir, "ada", {"sub-01": {"A": 1, "B": 0, "C": 0}, "sub-02": none})
    rate_all(run_dir, "ben", {"sub-01": none, "sub-02": none})
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "final labels: 2 by consensus, 0 from one rating, 1 unratable (render failed), 0 open"
    ]
    by_label = uids(run_dir)
    labels = app_rows(run_dir, "SELECT subject_uid, criterion, method FROM final_labels")
    assert sorted(labels) == sorted(
        [(by_label[label], criterion, "consensus") for label in LABELS[:2] for criterion in "ABC"]
        + [(by_label["sub-03"], "unratable", "single")]
    )


def test_finalize_names_a_rater_who_has_subjects_left(run_dir: Path) -> None:
    with TestClient(create_app(run_dir, rater="ada"), base_url=BASE) as client:
        version = client.get("/api/session").json()["protocol"]["version"]
        first = client.get("/api/next").json()["assignment"]
        skip = {
            "assignment_id": first["assignment_id"],
            "client_ts": "2026-11-13T12:00:00.000Z",
            "ms_on_subject": 700,
        }
        second = client.post("/api/skip", json=skip).json()["assignment"]
        rating = {
            "assignment_id": second["assignment_id"],
            "protocol_version": version,
            "values": {"A": 0, "B": 0, "C": 0},
            "client_ts": "2026-11-13T12:00:05.000Z",
            "ms_on_subject": 900,
        }
        assert client.post("/api/rating", json=rating).status_code == 200
    result = cli("finalize", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "final labels: 0 by consensus, 1 from one rating, 1 unratable (render failed), 0 open",
        "  ada still has 1 subject to rate (1 skipped); labels of those are from the other "
        "ratings only",
    ]


# --- calibration freeze ---------------------------------------------------------------------------


def test_calibration_freeze_writes_the_set_once(run_dir: Path) -> None:
    none = {"A": 0, "B": 0, "C": 0}
    refused = cli("calibration", "freeze", "--run-dir", run_dir)
    assert refused.exit_code == 2 and "no app.sqlite yet" in refused.stderr
    rate_all(run_dir, "local", {"sub-01": {"A": 2, "B": 0, "C": 0}, "sub-02": none})
    by_label = uids(run_dir)
    result = cli("calibration", "freeze", "--run-dir", run_dir, "--version", "pilot-1")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{run_dir / CALIBRATION_FILE}\n"
    written = yaml.safe_load((run_dir / CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert written["version"] == "pilot-1"
    # the two rated subjects; the one whose render failed is not a calibration subject
    assert sorted(subject["subject_uid"] for subject in written["subjects"]) == sorted(
        [by_label["sub-01"], by_label["sub-02"]]
    )
    again = cli("calibration", "freeze", "--run-dir", run_dir)
    assert again.exit_code == 2 and again.stderr.startswith("invalid: calibration.yaml exists")
    unknown = cli("calibration", "freeze", "--run-dir", run_dir.parent / "none")
    assert unknown.exit_code == 2


def test_calibration_freeze_of_named_subjects(run_dir: Path) -> None:
    none = {"A": 0, "B": 0, "C": 0}
    rate_all(run_dir, "local", {"sub-01": none, "sub-02": none})
    by_label = uids(run_dir)
    result = cli("calibration", "freeze", "--run-dir", run_dir, "--subject", by_label["sub-02"])
    assert result.exit_code == 0, result.output
    written = yaml.safe_load((run_dir / CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert [subject["subject_uid"] for subject in written["subjects"]] == [by_label["sub-02"]]


# --- export ---------------------------------------------------------------------------------------


def test_export_prints_the_files_it_wrote(run_dir: Path, tmp_path: Path) -> None:
    result = cli("export", "--run-dir", run_dir)
    assert result.exit_code == 0, result.output
    names = [f"{name}.csv" for name in EXPORT_TABLES] + [MANIFEST]
    assert result.stdout.splitlines() == [str(run_dir / "export" / name) for name in names]
    assert result.stderr == ""
    subjects = pd.read_csv(run_dir / "export" / "subjects.csv", dtype=str)
    assert sorted(subjects["subject_label"]) == list(LABELS)
    assert subjects.set_index("subject_label").loc["sub-03", "render_reason"] == "missing surface"
    manifest = json.loads((run_dir / "export" / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["profile"] == "analysis" and manifest["format"] == "csv"

    out = tmp_path / "elsewhere"
    parquet = cli("export", "--run-dir", run_dir, "--out", out, "--format", "parquet")
    assert parquet.exit_code == 0, parquet.output
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [f"{name}.parquet" for name in EXPORT_TABLES] + [MANIFEST]
    )
    assert len(pd.read_parquet(out / "subjects.parquet")) == 3


def test_export_stops_with_exit_2_for_what_it_cannot_do(run_dir: Path, tmp_path: Path) -> None:
    wrong = cli("export", "--run-dir", run_dir, "--format", "xlsx")
    assert wrong.exit_code == 2 and "is csv or parquet" in said(wrong.stderr)
    release = cli("export", "--run-dir", run_dir, "--release")
    assert release.exit_code == 2 and release.stderr.startswith("invalid: ")
    assert not (run_dir / "release").exists()
    missing = cli("export", "--run-dir", tmp_path / "none")
    assert missing.exit_code == 2 and missing.stderr.startswith("missing: ")
    taken = tmp_path / "a-file"
    taken.write_text("")
    blocked = cli("export", "--run-dir", run_dir, "--out", taken)
    assert blocked.exit_code == 2
    assert blocked.stderr.startswith("invalid: the run directory cannot be used: ")


@pytest.mark.parametrize("where", ["subjects directory", "subject", "below a subject", "link"])
def test_export_refuses_an_output_in_the_subjects_directory(
    run_dir: Path, study: Path, tmp_path: Path, where: str
) -> None:
    subjects_dir = study / "sourcedata" / "freesurfer"
    out = {
        "subjects directory": subjects_dir,
        "subject": subjects_dir / "sub-01",
        "below a subject": subjects_dir / "sub-01" / "tables",
        "link": tmp_path / "link",
    }[where]
    if where == "link":
        out.symlink_to(subjects_dir / "sub-02")
    before = snapshot_tree(study)
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stdout == "" and len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("invalid: ")
    assert "lies inside the subjects directory" in result.stderr
    # the same from Python
    with pytest.raises(ConfigError, match="lies inside the subjects directory") as raised:
        export_run(run_dir, out)
    assert raised.value.reason == "invalid"
    assert snapshot_tree(study) == before
    # beside the subjects directory, inside the fMRIPrep output, is a place like any other
    beside = cli("export", "--run-dir", run_dir, "--out", study / "tables")
    assert beside.exit_code == 0, beside.output
    assert (study / "tables" / MANIFEST).is_file()


@pytest.fixture
def linked(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A subjects directory whose two subjects and whose template are links into a store,
    and the run that ``review`` makes of it: ``(subjects directory, store, run)``."""
    store = tmp_path / "store"
    root = write_linked_tree(tmp_path / "linked" / "study", store, ["sub-01", "sub-02"])
    result = cli("review", root, "--no-serve", "--jobs", 1)
    assert result.exit_code == 0, result.output
    # the template comes after the review: a light one has too few vertices to score with
    write_light_template(store / "fsaverage")
    (root / "fsaverage").symlink_to(store / "fsaverage", target_is_directory=True)
    return root, store, tmp_path / "linked" / DEFAULT_RUN_DIR


@pytest.mark.parametrize(
    "where", ["link", "below the link", "target", "below the target", "template", "alias"]
)
def test_export_refuses_an_output_in_what_a_subject_links_to(
    linked: tuple[Path, Path, Path], tmp_path: Path, where: str
) -> None:
    """A subject directory that is a link resolves outside the subjects directory. What it
    leads to is read as a source all the same, and so is what the template link leads to."""
    root, store, run_dir = linked
    out = {
        "link": root / "sub-02",
        "below the link": root / "sub-01" / "exp",
        "target": store / "sub-02",
        "below the target": store / "sub-01" / "exp",
        "template": store / "fsaverage",
        "alias": tmp_path / "alias",
    }[where]
    if where == "alias":
        out.symlink_to(store / "sub-01" / "surf")
    before = snapshot_tree(store), snapshot_tree(root)
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stdout == "" and len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("invalid: the output directory ")
    assert "which is only read" in result.stderr
    with pytest.raises(ConfigError, match="which is only read") as raised:
        export_run(run_dir, out)
    assert raised.value.reason == "invalid"
    assert (snapshot_tree(store), snapshot_tree(root)) == before
    # beside what the links lead to is a place like any other
    beside = cli("export", "--run-dir", run_dir, "--out", store / "tables")
    assert beside.exit_code == 0, beside.output


def test_a_run_directory_in_what_a_subject_links_to_is_refused_before_it_is_made(
    linked: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    root, store, _run_dir = linked
    before = snapshot_tree(store), snapshot_tree(root)
    for place in (store / "sub-01" / "run", root / "sub-02" / "run", store / "fsaverage" / "r"):
        for command in (
            ("review", root, "--run-dir", place, "--no-serve", "--jobs", 1),
            ("inventory", "--run-dir", place, "--source", root),
        ):
            result = cli(*command)
            assert result.exit_code == 2, result.output
            assert result.stderr.splitlines()[-1].startswith("invalid: the run directory ")
            assert "which is only read" in result.stderr
    # a run directory in the fsaverage directory that the call names
    given = tmp_path / "templates" / "fsaverage"
    result = cli("review", root, "--run-dir", given / "run", "--fsaverage", given, "--no-serve")
    assert result.exit_code == 2 and "lies inside the fsaverage directory" in result.stderr
    assert not given.exists()
    assert (snapshot_tree(store), snapshot_tree(root)) == before


def test_a_run_that_was_moved_into_its_source_is_not_written_to(run_dir: Path, study: Path) -> None:
    """``export`` without ``--out``, ``serve``, ``finalize`` and ``calibration freeze`` open a
    run without building its sources. Nothing the package writes goes into a source."""
    rate_all(run_dir, "ada", {"sub-01": dict(A=0, B=0, C=0), "sub-02": dict(A=1, B=0, C=0)})
    moved = study / "sourcedata" / "freesurfer" / "run-moved"
    run_dir.rename(moved)
    before = snapshot_tree(study)
    for command in (
        ("export", "--run-dir", moved),
        ("export", "--run-dir", moved, "--format", "parquet"),
        ("serve", "--run-dir", moved, "--no-browser", "--port", 0),
        ("finalize", "--run-dir", moved),
        ("calibration", "freeze", "--run-dir", moved),
        ("review", study, "--run-dir", moved, "--no-serve"),
    ):
        result = cli(*command)
        assert result.exit_code == 2, (command, result.output)
        assert result.stdout == "" and len(result.stderr.splitlines()) == 1, command
        assert "lies inside the subjects directory" in result.stderr, command
    with pytest.raises(ConfigError, match="lies inside the subjects directory"):
        create_app(moved)
    with pytest.raises(ConfigError, match="lies inside the subjects directory"):
        export_run(moved)
    assert snapshot_tree(study) == before
    assert cli("status", "--run-dir", moved).exit_code == 0  # reading is no writing


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_export_refuses_an_output_among_the_files_of_the_run(
    run_dir: Path, study: Path, fmt: str
) -> None:
    """``clusters.parquet`` and ``scores.parquet`` are names of the run's own tables and of
    an export: an export into ``tables/`` used to leave a run that no phase could read."""
    before = snapshot_tree(run_dir)
    for out in (run_dir / "tables", run_dir, run_dir / "media" / "x", run_dir / "mine"):
        result = cli("export", "--run-dir", run_dir, "--format", fmt, "--out", out)
        assert result.exit_code == 2, result.output
        assert result.stdout == "" and len(result.stderr.splitlines()) == 1
        assert "lies inside the run directory" in result.stderr
        assert "export/ or release/" in result.stderr
        with pytest.raises(ConfigError, match="lies inside the run directory"):
            export_run(run_dir, out, fmt="csv" if fmt == "csv" else "parquet")
    assert snapshot_tree(run_dir) == before
    # the two directories of an export, also when they are named, and places below them
    for out in (run_dir / "export", run_dir / "release", run_dir / "export" / "second"):
        result = cli("export", "--run-dir", run_dir, "--format", fmt, "--out", out)
        assert result.exit_code == 0, result.output
        assert (out / MANIFEST).is_file()
    # the run is as usable as before
    again = cli("review", study, "--no-serve", "--jobs", 1)
    assert again.exit_code == 0, again.output


def test_export_refuses_an_output_where_the_subjects_were(
    run_dir: Path, study: Path, tmp_path: Path
) -> None:
    study.rename(tmp_path / "moved")  # the data is not mounted: the root is what is known
    result = cli("export", "--run-dir", run_dir, "--out", study / "sourcedata" / "tables")
    assert result.exit_code == 2 and "lies inside the subjects directory" in result.stderr
    assert not study.exists()
    assert cli("export", "--run-dir", run_dir).exit_code == 0


def test_export_of_a_census_run_refuses_an_output_in_a_local_source(
    study: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "census"
    Run(run_dir, local(study), key=HashKey(bytes(range(1, 33)), "dataset-key")).close()
    assert load_config(run_dir).sources[0].root is None  # the root is in the catalog only
    before = snapshot_tree(study)
    out = study / "sourcedata" / "freesurfer" / "sub-01"
    result = cli("export", "--run-dir", run_dir, "--out", out)
    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("invalid: ")
    assert "lies inside the subjects directory" in result.stderr
    assert snapshot_tree(study) == before
    elsewhere = cli("export", "--run-dir", run_dir, "--out", tmp_path / "out")
    assert elsewhere.exit_code == 0, elsewhere.output
    # the run itself moved into the source: its own export/ is an output there
    moved = study / "sourcedata" / "freesurfer" / "census"
    run_dir.rename(moved)
    before = snapshot_tree(study)
    result = cli("export", "--run-dir", moved)
    assert result.exit_code == 2, result.output
    assert "lies inside the subjects directory" in result.stderr
    # the roots of a census run are read from its catalog, whose write-ahead log is opened
    assert not (moved / "export").exists()
    assert {name: at for name, at in snapshot_tree(study).items() if "/census" not in name} == {
        name: at for name, at in before.items() if "/census" not in name
    }


def test_the_commands_are_listed_in_the_help() -> None:
    result = cli("--help")
    listed = said(result.output)
    for command in ("review", "serve", "finalize", "calibration", "export"):
        assert f" {command} " in listed
    assert "freeze" in said(cli("calibration", "--help").output)
