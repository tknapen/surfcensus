"""Tests of surfcensus/review.py: ``review`` on directories of synthetic subjects, what it
stores between calls, when the app opens, and that a source is only read.

``prepared`` is one ``review(..., serve=False)`` on the three subjects of
``small_run_cohort`` (16^3 voxels, a template of the real size next to them); the tests
that change a run work on a copy of its run directory, which still reads the cohort.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]
from synth_tree import write_tree  # pyrefly: ignore[missing-import]

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus import review as reviewing
from surfcensus.app.server import create_app
from surfcensus.config import (
    CONFIG_FILE,
    ConfigError,
    MetricsConfig,
    load_config,
    new_local_config,
    with_changes,
)
from surfcensus.export.tables import EXPORT_TABLES, MANIFEST, export_run
from surfcensus.privacy import HashKey
from surfcensus.protocol import PROTOCOL_FILE
from surfcensus.render import api
from surfcensus.review import (
    DEFAULT_RUN_DIR,
    ReviewApp,
    default_run_dir,
    open_review,
    rateable,
    render_counts,
    review,
)
from surfcensus.run import Run
from surfcensus.sources.base import SourceError
from surfcensus.sources.local import local
from surfcensus.store import PublicCatalog, primary_scores, queue_order

BASE = "http://127.0.0.1"
LABELS = ("sub-000", "sub-001", "sub-002")
WITHIN_THE_SET = MetricsConfig(min_reference_n=3, min_vertex_n=2)
"""With three subjects in the reference a run of the cohort has z-maps and clusters."""


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort) -> Path:
    """The run directory of ``review(cohort, serve=False)``: inventoried, scored, selected,
    every subject rendered at the default 320 px. Read-only."""
    run_dir = tmp_path_factory.mktemp("review") / "run"
    review(small_run_cohort.root, run_dir=run_dir, serve=False, jobs=1).close()
    return run_dir


@pytest.fixture
def run_dir(prepared: Path, tmp_path: Path) -> Path:
    """A copy of ``prepared`` that a test may change."""
    return Path(shutil.copytree(prepared, tmp_path / "run"))


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The media directories that ``render_subject`` is called for, from now on."""
    calls: list[Path] = []
    original = api.render_subject

    def recording(files: object, out: Path, *args: object, **kwargs: object) -> object:
        calls.append(Path(out))
        return original(files, out, *args, **kwargs)

    monkeypatch.setattr(api, "render_subject", recording)
    return calls


def uids_by_label(run_dir: Path) -> dict[str, str]:
    with PublicCatalog.open(run_dir) as public:
        subjects = public.subjects()
    return dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))


def queue_of(run_dir: Path) -> list[str]:
    """The rating queue of the run, computed apart from ``review``."""
    config = load_config(run_dir)
    with PublicCatalog.open(run_dir) as public:
        uids = list(public.subject_uids())
    return queue_order(
        uids, primary_scores(run_dir), blind=config.app.blind, seed=config.app.queue_seed
    )


def rate_one(run_dir: Path) -> str:
    """Rate the first subject of the queue through the app; returns its uid."""
    with TestClient(create_app(run_dir), base_url=BASE) as client:
        shown = client.get("/api/next").json()["assignment"]
        body = {
            "assignment_id": shown["assignment_id"],
            "protocol_version": client.get("/api/session").json()["protocol"]["version"],
            "values": {"A": 0, "B": 1, "C": 0},
            "client_ts": "2026-11-13T12:00:00.000Z",
            "ms_on_subject": 1500,
        }
        answer = client.post("/api/rating", json=body)
        assert answer.status_code == 200, answer.text
    return str(shown["subject_uid"])


def app_rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(run_dir / "app.sqlite")
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# --- one call does every phase --------------------------------------------------------------------


def test_review_inventories_scores_selects_and_renders(
    prepared: Path, small_run_cohort: SyntheticCohort, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(prepared)
    assert config.mode == "local" and not config.app.blind and not config.render_video
    assert config.app.require_marks is False
    assert config.sources[0].root == str(small_run_cohort.root.resolve())
    with Run(prepared) as run:
        public = run.public()
        subjects = public.subjects()
        assert sorted(subjects["subject_label"]) == list(LABELS)
        assert bool(subjects["complete"].all())
        scores = run.scores()
        assert len(scores) == 3 and scores["primary_rank"].tolist() == [1, 2, 3]
        selection = public.selection()
        assert sorted(selection["subject_uid"]) == sorted(subjects["subject_uid"])
        assert set(selection["reason"]) == {"all"} and set(selection["inclusion_prob"]) == {1.0}
        assert set(selection["seed"]) == {config.app.queue_seed}
        assert render_counts(run) == (3, 0, 0) and rateable(run) == 3
        renders = public.renders()
        assert set(renders["status"]) == {"done"} and renders["template_id"].notna().all()
        for uid in subjects["subject_uid"]:
            assert (run.media_dir(uid) / "render.json").is_file()
            assert not (run.media_dir(uid) / "anim.mp4").exists()
    # a review without the app installs no protocol and makes no database
    assert not (prepared / "app.sqlite").exists()


def test_a_second_review_changes_nothing_and_renders_nothing(
    run_dir: Path,
    small_run_cohort: SyntheticCohort,
    rendered: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = snapshot_tree(small_run_cohort.root)
    config_bytes = (run_dir / CONFIG_FILE).read_bytes()
    tables = {path.name: path.read_bytes() for path in (run_dir / "tables").glob("*.parquet")}
    review(small_run_cohort.root, run_dir=run_dir, serve=False, jobs=1).close()
    assert rendered == []
    assert (run_dir / CONFIG_FILE).read_bytes() == config_bytes
    assert tables == {
        path.name: path.read_bytes() for path in (run_dir / "tables").glob("*.parquet")
    }
    said = capsys.readouterr().err
    assert f"Run directory: {run_dir} (continued)" in said
    assert "Found 3 subjects (a subjects directory, FreeSurfer 6.0.1)." in said
    assert "Rendered: 3 of 3 subjects." in said
    assert snapshot_tree(small_run_cohort.root) == before


def test_what_review_says_goes_to_stderr_and_names_the_reference(
    tmp_path: Path, small_cohort: SyntheticCohort, capsys: pytest.CaptureFixture[str]
) -> None:
    review(small_cohort.root, run_dir=tmp_path / "run", serve=False, jobs=1).close()
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == f"Run directory: {tmp_path / 'run'} (new)"
    assert lines[1] == "Found 5 subjects (a subjects directory, FreeSurfer 6.0.1)."
    assert lines[2] == (
        "Scores: 5 subjects; 5 ranked on ceiling fraction and occipital-to-control ratio, "
        "without a reference (decision D16)."
    )
    assert lines[3] == "Rendered: 5 of 5 subjects."


def test_the_options_are_stored_and_hold_until_they_are_given_again(
    run_dir: Path, small_run_cohort: SyntheticCohort, rendered: list[Path]
) -> None:
    root = small_run_cohort.root
    review(root, run_dir=run_dir, serve=False, jobs=1, blind=True, require_marks=True).close()
    stored = load_config(run_dir)
    assert stored.app.blind and stored.app.require_marks is True
    review(root, run_dir=run_dir, serve=False, jobs=1).close()
    assert load_config(run_dir) == stored
    review(root, run_dir=run_dir, serve=False, jobs=1, blind=False).close()
    assert not load_config(run_dir).app.blind and load_config(run_dir).app.require_marks is True
    # none of this is a setting of a tile
    assert rendered == []


def test_video_once_asked_for_stays_and_nothing_is_rendered_twice(
    run_dir: Path, small_run_cohort: SyntheticCohort, rendered: list[Path]
) -> None:
    root = small_run_cohort.root
    review(root, run_dir=run_dir, serve=False, jobs=1, video=True).close()
    assert load_config(run_dir).render_video
    assert len(rendered) == 3  # the tiles are current; the video is the one thing made
    uids = uids_by_label(run_dir)
    assert all((run_dir / "media" / uid / "anim.mp4").is_file() for uid in uids.values())
    rendered.clear()
    review(root, run_dir=run_dir, serve=False, jobs=1).close()
    assert rendered == [] and load_config(run_dir).render_video


def test_a_new_run_takes_its_options_at_creation(
    tmp_path: Path, small_run_cohort: SyntheticCohort
) -> None:
    run_dir = tmp_path / "run"
    with review(
        small_run_cohort.root,
        run_dir=run_dir,
        serve=False,
        jobs=1,
        blind=True,
        require_marks=True,
        subjects=["sub-00[01]"],
        field_strength=7.0,
    ) as run:
        config = load_config(run_dir)
        assert config.app.blind and config.app.require_marks is True
        assert config.sources[0].subjects == ("sub-00[01]",)
        assert config.sources[0].field_strength == 7.0
        subjects = run.public().subjects()
        assert sorted(subjects["subject_label"]) == ["sub-000", "sub-001"]
        assert set(subjects["field_strength"]) == {7.0}
    # the subject set is kept by a call that names none
    with review(small_run_cohort.root, run_dir=run_dir, serve=False, jobs=1) as run:
        assert len(run.public().subjects()) == 2
        assert load_config(run_dir) == config


# --- blind and ratings ----------------------------------------------------------------------------


def test_blind_cannot_change_once_a_rating_exists_unless_forced_and_then_it_is_recorded(
    run_dir: Path, small_run_cohort: SyntheticCohort
) -> None:
    root = small_run_cohort.root
    rate_one(run_dir)
    config_bytes = (run_dir / CONFIG_FILE).read_bytes()
    with pytest.raises(ConfigError, match=r"not blind and has 1 submission.*--force") as refused:
        review(root, run_dir=run_dir, serve=False, jobs=1, blind=True)
    assert refused.value.reason == "invalid"
    assert (run_dir / CONFIG_FILE).read_bytes() == config_bytes
    # the value the run has is no change
    review(root, run_dir=run_dir, serve=False, jobs=1, blind=False).close()

    review(root, run_dir=run_dir, serve=False, jobs=1, blind=True, force=True).close()
    assert load_config(run_dir).app.blind
    changes = app_rows(
        run_dir,
        "SELECT payload_json FROM events WHERE assignment_id IS NULL "
        "AND type = 'config_change' AND payload_json LIKE '%\"blind\"%' ORDER BY event_id",
    )
    assert [json.loads(text) for (text,) in changes] == [
        {"setting": "blind", "old": None, "new": False},  # the server, when it first started
        {"setting": "blind", "old": False, "new": True},
    ]
    # the server that starts next finds the change on record and adds none
    with TestClient(create_app(run_dir), base_url=BASE):
        pass
    assert len(app_rows(run_dir, "SELECT 1 FROM events WHERE type = 'config_change'")) == 4


def test_relative_paths_work_also_for_a_run_that_has_ratings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tree(tmp_path / "subjects", "subjects_dir", ["sub-01", "sub-02"])
    monkeypatch.chdir(tmp_path)
    with review("subjects", run_dir="runs/mine", serve=False, jobs=1) as run:
        assert run.run_dir == Path("runs/mine") and render_counts(run) == (2, 0, 0)
    rate_one(Path("runs/mine"))
    with pytest.raises(ConfigError, match="has 1 submission"):
        review("subjects", run_dir="runs/mine", serve=False, jobs=1, blind=True)
    review("subjects", run_dir="runs/mine", serve=False, jobs=1, blind=True, force=True).close()
    assert load_config("runs/mine").app.blind
    # without a run directory given, a relative path finds the same default
    with review("subjects", serve=False, jobs=1) as run:
        assert run.run_dir == tmp_path / DEFAULT_RUN_DIR


def test_export_after_review_and_a_rating(run_dir: Path, small_run_cohort: SyntheticCohort) -> None:
    rated = rate_one(run_dir)
    paths = export_run(run_dir)
    assert [path.name for path in paths] == [f"{name}.csv" for name in EXPORT_TABLES] + [MANIFEST]
    subjects = pd.read_csv(run_dir / "export" / "subjects.csv", dtype={"subject_uid": str})
    assert sorted(subjects["subject_label"]) == list(LABELS)
    assert subjects["render_status"].tolist() == ["done"] * 3
    row = subjects.set_index("subject_uid").loc[rated]
    assert (row["final_A"], row["final_B"], row["final_C"]) == (0, 1, 0)
    assert row["final_method"] == "single" and row["n_ratings"] == 1
    ratings = pd.read_csv(run_dir / "export" / "ratings.csv", dtype={"subject_uid": str})
    assert ratings["subject_uid"].tolist() == [rated] and ratings["ms_on_subject"].tolist() == [
        1500
    ]


# --- when the app opens ---------------------------------------------------------------------------


@pytest.fixture
def opened(tmp_path: Path) -> Iterator[list[tuple[Run, ReviewApp | None]]]:
    """What a test opened with ``open_review``, stopped and closed afterwards."""
    made: list[tuple[Run, ReviewApp | None]] = []
    yield made
    for run, app in made:
        if app is not None:
            app.stop()
        run.close()


def test_the_app_opens_at_the_first_render_and_the_others_follow(
    tmp_path: Path,
    small_run_cohort: SyntheticCohort,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer behind the app is held until the test has looked: at that moment one
    subject is rendered, it is the first of the queue, and the app offers it."""
    gate = threading.Event()
    original = reviewing._render_rest

    def held(*args: Any, **kwargs: Any) -> None:
        gate.wait(60)
        original(*args, **kwargs)

    monkeypatch.setattr(reviewing, "_render_rest", held)
    run_dir = tmp_path / "run"
    # scored within the set, so that the run has clusters for the native stage to place
    root = small_run_cohort.root
    Run(run_dir, local(root), config=new_local_config(root, metrics=WITHIN_THE_SET)).close()
    run, app = open_review(root, run_dir=run_dir, jobs=1, port=0)
    opened.append((run, app))
    assert app is not None and app.rendering and app.handle.running
    assert app.url == f"http://127.0.0.1:{app.handle.port}/" and app.seconds_to_app > 0
    assert (run_dir / PROTOCOL_FILE).is_file()
    queue = queue_of(run_dir)
    assert render_counts(run) == (1, 0, 2)
    done = [uid for uid in queue if (run_dir / "media" / uid / "render.json").is_file()]
    assert done == queue[:1]

    with httpx.Client(base_url=app.url, trust_env=False) as client:
        first = client.get("/api/next").json()
        assert first["status"] == "assignment"
        assert first["assignment"]["subject_uid"] == queue[0]
        assert first["progress"]["rendered"] == 1 and first["progress"]["n_subjects"] == 3
        assert first["prefetch"] == []

        gate.set()
        assert app.wait_rendered(120) and app.render_problem is None
        assert render_counts(run) == (3, 0, 0)
        again = client.get("/api/next").json()
        assert again["assignment"]["subject_uid"] == queue[0]
        assert again["progress"]["rendered"] == 3
        assert [entry["media_base"] for entry in again["prefetch"]] == [
            f"/media/{uid}/" for uid in queue[1:]
        ]
    # the renderer ended with the native stage, as a plain render does
    with Run(run_dir) as fresh:
        assert len(fresh.public().work("native")) == 1
        clusters = fresh.clusters()
        placed = clusters[clusters["native_vertex"].notna()]
        assert len(placed) > 0 and set(placed["subject_uid"]) <= set(queue)
    app.stop()
    assert not app.handle.running
    app.stop()  # a second call does nothing


def test_serve_after_names_how_many_renders_the_app_waits_for(
    tmp_path: Path,
    small_run_cohort: SyntheticCohort,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reviewing, "_render_rest", lambda *args, **kwargs: None)
    run, app = open_review(
        small_run_cohort.root, run_dir=tmp_path / "two", jobs=1, port=0, serve_after=2
    )
    opened.append((run, app))
    assert app is not None and render_counts(run) == (2, 0, 1)
    queue = queue_of(tmp_path / "two")
    assert [(tmp_path / "two" / "media" / uid / "render.json").is_file() for uid in queue] == [
        True,
        True,
        False,
    ]
    app.stop()

    # 0: every render first, and no renderer behind the app
    run, app = open_review(
        small_run_cohort.root, run_dir=tmp_path / "all", jobs=1, port=0, serve_after=0
    )
    opened.append((run, app))
    assert app is not None and not app.rendering and render_counts(run) == (3, 0, 0)


def test_a_blind_review_renders_in_the_shuffled_order_of_its_queue(
    tmp_path: Path,
    small_run_cohort: SyntheticCohort,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reviewing, "_render_rest", lambda *args, **kwargs: None)
    seen = set()
    for attempt in range(4):
        run_dir = tmp_path / f"run{attempt}"
        run, app = open_review(small_run_cohort.root, run_dir=run_dir, jobs=1, port=0, blind=True)
        opened.append((run, app))
        assert app is not None
        queue = queue_of(run_dir)
        done = [uid for uid in queue if (run_dir / "media" / uid / "render.json").is_file()]
        assert done == queue[:1]
        labels = {uid: label for label, uid in uids_by_label(run_dir).items()}
        seen.add(labels[queue[0]])
        app.stop()
    # four runs with four seeds: the first subject is not always the same one
    assert len(seen) > 1


def test_a_stopped_renderer_leaves_the_rest_pending_and_the_next_call_makes_it(
    run_dir: Path, small_run_cohort: SyntheticCohort
) -> None:
    """``Run.render(stop=...)``, which is how the app's renderer is ended: asked before each
    batch, so the batch that runs is recorded and no later one is started."""
    shutil.rmtree(run_dir / "media")
    asked: list[int] = []

    def stop() -> bool:
        asked.append(1)
        return len(asked) > 1

    with Run(run_dir) as run:
        run.config = with_changes(
            run.config, cache=run.config.cache.model_dump() | {"batch_subjects": 1}
        )
        native_before = run.public().work("native")["finished_at"].tolist()
        report = run.render(stop=stop)
        assert (report.done, report.failed) == (1, 0)
        assert render_counts(run) == (1, 0, 2)
        # a stopped call leaves the native stage to the call that finishes
        assert run.public().work("native")["finished_at"].tolist() == native_before
        report = run.render()
        assert report.done == 2 and render_counts(run) == (3, 0, 0)


# --- subjects that cannot be rendered -------------------------------------------------------------


def test_an_incomplete_subject_is_named_selected_and_never_offered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01", "sub-02"], incomplete=["sub-02"])
    before = snapshot_tree(tree)
    run_dir = tmp_path / "run"
    with review(tree, run_dir=run_dir, serve=False, jobs=1) as run:
        said = capsys.readouterr().err
        assert "Found 2 subjects (fMRIPrep layout). 1 cannot be rendered: sub-02 (no " in said
        assert "surf/lh.pial" in said
        assert "Rendered: 1 of 2 subjects, 1 cannot be rated." in said
        assert "  not rendered, missing surface: 1 (sub-02)" in said
        assert render_counts(run) == (1, 1, 0) and rateable(run) == 1
        uids = uids_by_label(run_dir)
        assert sorted(run.public().selection()["subject_uid"]) == sorted(uids.values())
    with TestClient(create_app(run_dir), base_url=BASE) as client:
        first = client.get("/api/next").json()
        assert first["assignment"]["subject_uid"] == uids["sub-01"]
        assert first["progress"]["n_subjects"] == 1 and first["progress"]["remaining"] == 1
    labels = app_rows(run_dir, "SELECT subject_uid, criterion, method FROM final_labels")
    assert labels == [(uids["sub-02"], "unratable", "single")]
    assert snapshot_tree(tree) == before


def test_when_no_subject_can_be_rendered_there_is_no_app(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = ["sub-01", "sub-02"]
    tree = write_tree(tmp_path / "study", "subjects_dir", labels, incomplete=labels)
    run, app = open_review(tree, run_dir=tmp_path / "run", jobs=1, port=0)
    with run:
        assert app is None and rateable(run) == 0 and render_counts(run) == (0, 2, 0)
    said = capsys.readouterr().err
    assert "2 cannot be rendered: sub-01 (no surf/lh.pial), sub-02 (no surf/lh.pial)." in said
    assert "No subject of this run can be rendered, so there is nothing to rate." in said
    assert not (tmp_path / "run" / "app.sqlite").exists()
    # review itself returns the run as well, and at once
    with review(tree, run_dir=tmp_path / "run", jobs=1, port=0, open_browser=False) as again:
        assert rateable(again) == 0


def test_review_without_a_template_ranks_and_renders_without_counterparts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-02"])
    before = snapshot_tree(tree)
    with caplog.at_level("WARNING"), review(tree, serve=False, jobs=1) as run:
        assert run.run_dir == tmp_path / DEFAULT_RUN_DIR
        assert run.template() is None
        assert render_counts(run) == (2, 0, 0)
        assert run.public().renders()["template_id"].isna().all()
    assert "no fsaverage template" in caplog.text.lower() or "--fsaverage" in caplog.text
    assert "Found 2 subjects (a subjects directory)." in capsys.readouterr().err
    assert snapshot_tree(tree) == before


# --- where the run goes, and which run it is ------------------------------------------------------


def test_the_default_run_directory_is_beside_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = write_tree(tmp_path / "a" / "subjects", "subjects_dir", ["sub-01"])
    assert default_run_dir(plain) == tmp_path / "a" / DEFAULT_RUN_DIR
    # one subject as path: beside the subjects directory, never inside it
    assert default_run_dir(plain / "sub-01") == tmp_path / "a" / DEFAULT_RUN_DIR
    study = write_tree(tmp_path / "b" / "fmriprep-out", "fmriprep", ["sub-01"])
    assert default_run_dir(study) == tmp_path / "b" / DEFAULT_RUN_DIR
    legacy = write_tree(tmp_path / "c" / "out", "fmriprep_legacy", ["sub-01"])
    assert default_run_dir(legacy) == tmp_path / "c" / DEFAULT_RUN_DIR
    # given as its fmriprep folder, the subjects lie beside that folder: one level up
    assert default_run_dir(legacy / "fmriprep") == tmp_path / "c" / "out" / DEFAULT_RUN_DIR

    # a parent that cannot be written to: the current directory
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    real_access = os.access
    monkeypatch.setattr(
        os, "access", lambda path, mode: Path(path) != tmp_path / "a" and real_access(path, mode)
    )
    assert default_run_dir(plain) == elsewhere / DEFAULT_RUN_DIR
    # unless the run is there already: then it is the run of that data
    (tmp_path / "a" / DEFAULT_RUN_DIR).mkdir()
    (tmp_path / "a" / DEFAULT_RUN_DIR / CONFIG_FILE).write_text("")
    assert default_run_dir(plain) == tmp_path / "a" / DEFAULT_RUN_DIR


def test_one_subject_directory_as_path(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "subjects", "subjects_dir", ["sub-01", "sub-02"])
    with review(tree / "sub-02", serve=False, jobs=1) as run:
        assert run.run_dir == tmp_path / DEFAULT_RUN_DIR
        assert run.public().subjects()["subject_label"].tolist() == ["sub-02"]
        assert render_counts(run) == (1, 0, 0)
    # the other subject as path: the run takes it, and the first one is no longer listed
    with review(tree / "sub-01", serve=False, jobs=1) as run:
        subjects = run.public().subjects().set_index("subject_label")
        assert bool(subjects.loc["sub-01", "present"]) and not bool(
            subjects.loc["sub-02", "present"]
        )


def test_data_that_moved_needs_relocate_and_keeps_its_subjects(
    tmp_path: Path, rendered: list[Path]
) -> None:
    tree = write_tree(tmp_path / "old", "subjects_dir", ["sub-01", "sub-02"])
    run_dir = tmp_path / "run"
    review(tree, run_dir=run_dir, serve=False, jobs=1).close()
    before = uids_by_label(run_dir)
    moved = tree.rename(tmp_path / "new")
    rendered.clear()
    with pytest.raises(ConfigError) as refused:
        review(moved, run_dir=run_dir, serve=False, jobs=1)
    assert refused.value.reason == "other root"
    review(moved, run_dir=run_dir, serve=False, jobs=1, relocate=True).close()
    assert uids_by_label(run_dir) == before and rendered == []
    assert load_config(run_dir).sources[0].root == str(moved.resolve())


def test_what_review_refuses(tmp_path: Path, small_run_cohort: SyntheticCohort) -> None:
    tree = write_tree(tmp_path / "subjects", "subjects_dir", ["sub-01"])
    run_dir = tmp_path / "run"
    for options, message in [
        ({"rater": ""}, "a rater's name"),
        ({"rater": " ada"}, "a rater's name"),
        ({"jobs": 0}, "jobs is at least 1"),
        ({"serve_after": -1}, "serve_after"),
        ({"port": 70000}, "a port lies between"),
        ({"field_strength": 0.0}, "tesla"),
        ({"relocate": True}, "holds no run to relocate"),
    ]:
        with pytest.raises(ConfigError, match=message):
            review(tree, run_dir=run_dir, serve=False, **options)  # type: ignore[arg-type]
    # a glob that matches nothing, and a path without subjects, stop before a run exists
    with pytest.raises(SourceError, match="no subject matches"):
        review(tree, run_dir=run_dir, serve=False, subjects=["sub-9*"])
    with pytest.raises(SourceError):
        review(tmp_path / "nowhere", run_dir=run_dir, serve=False)
    assert not run_dir.exists()

    key = HashKey(bytes(range(1, 33)), "dataset-key")
    census = tmp_path / "census"
    Run(census, local(tree), key=key).close()
    with pytest.raises(ConfigError, match="holds a census run"):
        review(tree, run_dir=census, serve=False)


def test_a_rater_name_reaches_the_server_and_is_not_stored(
    run_dir: Path,
    small_run_cohort: SyntheticCohort,
    opened: list[tuple[Run, ReviewApp | None]],
) -> None:
    config_bytes = (run_dir / CONFIG_FILE).read_bytes()
    run, app = open_review(small_run_cohort.root, run_dir=run_dir, jobs=1, port=0, rater="ada")
    opened.append((run, app))
    assert app is not None and not app.rendering
    assert app_rows(run_dir, "SELECT name, role FROM raters") == [("ada", "lead")]
    assert (run_dir / CONFIG_FILE).read_bytes() == config_bytes


def test_the_closing_line_counts_what_the_rater_did(
    run_dir: Path, small_run_cohort: SyntheticCohort
) -> None:
    assert reviewing._progress_of(run_dir, None) is None  # no database yet
    rate_one(run_dir)
    assert reviewing._progress_of(run_dir, None) == (1, 2)
    assert reviewing._progress_of(run_dir, "someone else") is None


def test_review_returns_when_the_server_is_stopped(
    run_dir: Path,
    small_run_cohort: SyntheticCohort,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``review`` with the app: in place of Ctrl-C, the test stops the server from another
    thread once it answers, and the call ends with its closing line."""
    started: list[ReviewApp] = []
    original = reviewing.open_review

    def recording(*args: Any, **kwargs: Any) -> tuple[Run, ReviewApp | None]:
        run, app = original(*args, **kwargs)
        assert app is not None
        started.append(app)
        return run, app

    monkeypatch.setattr(reviewing, "open_review", recording)
    opened_urls: list[str] = []
    monkeypatch.setattr(reviewing.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    def stop_soon() -> None:
        deadline = time.monotonic() + 60
        while not started and time.monotonic() < deadline:
            time.sleep(0.05)
        rate_through = started[0]
        with httpx.Client(base_url=rate_through.url, trust_env=False) as client:
            assert client.get("/healthz").status_code == 200
        rate_through.handle.stop()

    stopper = threading.Thread(target=stop_soon)
    stopper.start()
    review(small_run_cohort.root, run_dir=run_dir, jobs=1, port=0).close()
    stopper.join()
    said = capsys.readouterr().err
    assert f"Rating app: {started[0].url} (Ctrl-C to stop)" in said
    assert "s from the start of the command to the app." in said
    assert opened_urls == [started[0].url]
    assert (
        "0 rated, 3 to go. Run the same command to continue. "
        f"Export: surfcensus export --run-dir {run_dir}"
    ) in said
