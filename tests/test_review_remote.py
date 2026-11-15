"""Tests of ``review openneuro:dsXXXXXX`` from Python, on the fake bucket of
``tests/fakes3.py`` (docs/dev/remote_plan.md, sections 4, 5 and 7).

The tests of ``tests/test_review.py``, ``tests/test_pipeline.py`` and
``tests/test_inventory.py`` that section 7 names, on a remote source; the equality of a
local and a remote review of the same files; the template of a remote dataset and its lazy
lookup; the disk bound of a batch; the pruning of the cache; a fault of the bucket behind
the running app. Nothing here touches the network: the bucket is the fake, reached through
``FakeS3.transport()``.

``prepared`` is one ``review("openneuro:ds999999", serve=False)`` on the three subjects of
``small_run_cohort`` laid out as a bucket dataset, with the template next to them; the tests
that change a run work on a copy of its run directory.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from fakes3 import FAKE_ACCESSION, FakeS3, bucket_of, key_of  # pyrefly: ignore[missing-import]
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]
from synth_tree import write_tree  # pyrefly: ignore[missing-import]

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from surfcensus import privacy, remote
from surfcensus import review as reviewing
from surfcensus import run as run_module
from surfcensus.app.server import create_app
from surfcensus.config import (
    CONFIG_FILE,
    ConfigError,
    MetricsConfig,
    RemoteConfig,
    load_config,
    new_local_config,
    run_key,
    with_changes,
)
from surfcensus.remote.cache import CACHE_DIR, LEASES, RETAINED, Cache
from surfcensus.remote.s3 import S3Client
from surfcensus.render import api
from surfcensus.review import ReviewApp, open_review, rateable, render_counts, review
from surfcensus.run import Run
from surfcensus.sources.base import TIER_A, SourceError
from surfcensus.sources.local import local
from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives
from surfcensus.store import PublicCatalog, primary_scores, queue_order, read_table, table_path

BUCKET = "openneuro-derivatives"
TEXT = f"openneuro:{FAKE_ACCESSION}"
BASE = "http://127.0.0.1"
LABELS = ("sub-000", "sub-001", "sub-002")
WITHIN_THE_SET = MetricsConfig(min_reference_n=3, min_vertex_n=2)
"""With three subjects in the reference a run of the cohort has z-maps and clusters."""


@pytest.fixture(autouse=True)
def no_freesurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed FreeSurfer would give the template, and none would be looked for next
    to the subjects."""
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


@pytest.fixture(scope="module")
def bucket(tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort) -> Path:
    """The cohort of ``small_run`` laid out as the dataset ``ds999999`` of the bucket, with
    the template next to its subjects. Read-only."""
    return bucket_of(small_run_cohort, root=tmp_path_factory.mktemp("bucket"))


@pytest.fixture
def fake(bucket: Path) -> FakeS3:
    return FakeS3({BUCKET: bucket}, page_size=7)


def remote_review(fake: FakeS3, run_dir: Path, **options: Any) -> Run:
    """``review(openneuro:ds999999, serve=False)`` over ``fake``."""
    options = {"serve": False, "jobs": 1} | options
    return review(TEXT, run_dir=run_dir, transport=fake.transport(), **options)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory, bucket: Path) -> Path:
    """The run directory of one remote review of the cohort: inventoried, scored, selected,
    every subject rendered at the default 320 px. Read-only."""
    run_dir = tmp_path_factory.mktemp("remote") / f"{FAKE_ACCESSION}-surfcensus-run"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FREESURFER_HOME", raising=False)
        remote_review(FakeS3({BUCKET: bucket}), run_dir).close()
    return run_dir


@pytest.fixture
def run_dir(prepared: Path, tmp_path: Path) -> Path:
    """A copy of ``prepared`` that a test may change."""
    return Path(shutil.copytree(prepared, tmp_path / f"{FAKE_ACCESSION}-surfcensus-run"))


@pytest.fixture(scope="module")
def local_review(
    tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort
) -> Path:
    """The run directory of a local review of the same files. Read-only."""
    run_dir = tmp_path_factory.mktemp("local") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("FREESURFER_HOME", raising=False)
        review(small_run_cohort.root, run_dir=run_dir, serve=False, jobs=1).close()
    return run_dir


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
    config = load_config(run_dir)
    with PublicCatalog.open(run_dir) as public:
        uids = list(public.subject_uids())
    return queue_order(
        uids, primary_scores(run_dir), blind=config.app.blind, seed=config.app.queue_seed
    )


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []


def keys_asked(fake: FakeS3) -> list[str]:
    return [entry.key or "" for entry in fake.log]


# --- one call does every phase --------------------------------------------------------------------


def test_review_inventories_scores_selects_and_renders(prepared: Path) -> None:
    config = load_config(prepared)
    assert config.mode == "local" and not config.app.blind and not config.render_video
    (stored,) = config.sources
    assert (stored.kind, stored.bucket, stored.prefix, stored.subjects) == (
        "openneuro-derivatives",
        BUCKET,
        "fmriprep/",
        (),
    )
    assert stored.dataset == privacy.dataset_pseudonym(
        run_key(config), "openneuro-derivatives", FAKE_ACCESSION
    )
    assert FAKE_ACCESSION not in (prepared / CONFIG_FILE).read_text()
    with Run(prepared) as run:
        public = run.public()
        subjects = public.subjects()
        assert sorted(subjects["subject_label"]) == list(LABELS)
        assert bool(subjects["complete"].all())
        assert len(run.scores()) == 3
        selection = public.selection()
        assert sorted(selection["subject_uid"]) == sorted(subjects["subject_uid"])
        assert render_counts(run) == (3, 0, 0) and rateable(run) == 3
        renders = public.renders()
        assert set(renders["status"]) == {"done"} and renders["template_id"].notna().all()
        for uid in subjects["subject_uid"]:
            assert (run.media_dir(uid) / "render.json").is_file()
            assert not (run.media_dir(uid) / "anim.mp4").exists()
    assert not (prepared / "app.sqlite").exists()
    # nothing of a lease is left, and only Tier A files stay in the cache
    assert files_under(prepared / CACHE_DIR / LEASES) == []
    names = {path.name for path in files_under(prepared / CACHE_DIR / RETAINED)}
    assert names == {
        "build-stamp.txt",
        "recon-all.log",
        "recon-all-lh.log",
        "recon-all-rh.log",
        "lh.aparc.stats",
        "rh.aparc.stats",
        "aseg.stats",
    }
    assert len(files_under(prepared / CACHE_DIR / RETAINED)) == 3 * 7


def test_what_a_remote_review_says(
    tmp_path: Path, fake: FakeS3, bucket: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = snapshot_tree(bucket)
    run_dir = tmp_path / "run"
    remote_review(fake, run_dir).close()
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == f"Listing {TEXT} in s3://{BUCKET}, anonymously and read-only ..."
    assert lines[1] == f"Run directory: {run_dir} (new)"
    assert (
        lines[2].startswith("To fetch: ")
        and " before the first rating screen (Tier A and" in lines[2]
    )
    assert "of 3 subjects, the fsaverage template, 3 renders)" in lines[2]
    assert "an estimate from the listed sizes" in lines[2]
    assert lines[3] == f"Found 3 subjects in {TEXT} (fMRIPrep layout, FreeSurfer 6.0.1)."
    assert "Rendered: 3 of 3 subjects." in lines
    assert lines[-1].startswith("Fetched ") and "of source files left in the cache." in lines[-1]
    # the requests: GET and listings (GET) only, anonymous, at most 8 at once
    assert fake.methods <= {"GET", "LIST"} and fake.signed == []
    assert 1 <= fake.high_water <= remote.MAX_CONCURRENCY
    assert snapshot_tree(bucket) == before


def test_the_estimate_is_the_sum_of_the_listed_sizes(tmp_path: Path, fake: FakeS3) -> None:
    """Recomputed apart from ``review``, from a listing of the fake bucket."""
    from surfcensus.sources.base import RENDER_ROLES, TIER_B, TIER_C, FileRole

    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    source = OpenNeuroDerivatives(client=client, accession=FAKE_ACCESSION)
    dataset = next(iter(source.datasets()))
    records = list(source.subjects(dataset))
    found = reviewing.remote_estimate(
        source,
        dataset,
        records,
        template_needed=True,
        n_first=1,
        jobs=2,
        batch_subjects=32,
        max_bytes=4_000_000_000,
    )
    sizes = {(r.subject_label, role): ref.size for r in records for role, ref in r.files.items()}
    tier_a = sum(size for (_, role), size in sizes.items() if role in TIER_A)
    tier_b = sum(size for (_, role), size in sizes.items() if role in TIER_B)
    spheres = (FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)
    renders = sorted(
        (
            sum(sizes.get((r.subject_label, role), 0) for role in (*TIER_C, *spheres))
            for r in records
            if all(role in r.files for role in RENDER_ROLES)
        ),
        reverse=True,
    )
    template = source.template_size(dataset)
    assert template is not None and template > 30_000_000  # a template of the real size
    assert (found.tier_a, found.tier_b, found.template) == (tier_a, tier_b, template)
    assert (found.first, found.n_first, found.renders) == (renders[0], 1, sum(renders))
    assert found.before_app == tier_a + tier_b + template + renders[0]
    assert found.in_all == tier_a + tier_b + template + sum(renders)
    # listing is all it took
    assert {entry.method for entry in fake.log} <= {"LIST", "GET"}
    assert all(
        entry.key.endswith("dataset_description.json")
        for entry in fake.log
        if entry.method == "GET" and entry.key
    )


def test_the_peak_of_the_estimate_follows_the_batches_that_are_leased(fake: FakeS3) -> None:
    """A call that renders everything before it ends leases ``batch_subjects`` renders at
    once, one that opens the app first its first renders and then ``jobs`` at a time; a
    render leases both ``sphere.reg`` files when the run has a template from anywhere."""
    from surfcensus.sources.base import RENDER_ROLES, TIER_B, TIER_C, FileRole

    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    source = OpenNeuroDerivatives(client=client, accession=FAKE_ACCESSION)
    dataset = next(iter(source.datasets()))
    records = list(source.subjects(dataset))
    spheres = (FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)

    def of(record: Any, roles: Any) -> int:
        return sum(ref.size for role, ref in record.files.items() if role in roles)

    complete = [r for r in records if all(role in r.files for role in RENDER_ROLES)]
    assert len(complete) == 3
    with_spheres = max(of(r, (*TIER_C, *spheres)) for r in complete)
    without = max(of(r, TIER_C) for r in complete)
    tier_a = sum(of(r, TIER_A) for r in records)
    largest_b = max(of(r, TIER_B) for r in records)

    def estimate(**changes: Any) -> reviewing.RemoteEstimate:
        settings: dict[str, Any] = {
            "template_needed": False,
            "n_first": None,
            "jobs": 1,
            "batch_subjects": 32,
            "max_bytes": 4_000_000_000,
        }
        return reviewing.remote_estimate(source, dataset, records, **(settings | changes))

    # everything before the call ends: one batch of all three renders, although jobs is 1
    assert with_spheres * 3 > largest_b * 3
    assert estimate().peak == tier_a + with_spheres * 3
    assert estimate(batch_subjects=2).peak == tier_a + max(with_spheres * 2, largest_b * 2)
    # a budget that holds one render at a time lowers both batches, as Run lowers them
    tight = estimate(max_bytes=with_spheres)
    assert with_spheres // largest_b < 3
    assert tight.peak == tier_a + max(with_spheres, largest_b * (with_spheres // largest_b))
    # the app after one render, the others two at a time behind it
    assert estimate(n_first=1, jobs=2).peak == tier_a + max(with_spheres * 2, largest_b * 3)
    assert estimate(n_first=1, jobs=1).peak == tier_a + max(with_spheres, largest_b * 3)
    # a template that the run holds or finds elsewhere: nothing fetched for it, and the
    # renders lease the sphere.reg files all the same
    found = estimate(n_first=1)
    assert found.template == 0 and found.first == with_spheres
    assert with_spheres > without
    # the template next to the subjects: its bytes are in the peak
    needed = estimate(template_needed=True)
    assert needed.template > 0
    assert needed.peak == tier_a + needed.template + with_spheres * 3
    # a continued run: the maps and the renders that it holds are left out, and the app
    # waits for one render in all, which the run has
    labels = [r.subject_label for r in records]
    held = estimate(n_first=1, mapped=labels[:2], rendered=labels[:1])
    assert held.tier_b == of(records[2], TIER_B)
    assert held.renders == sum(of(r, (*TIER_C, *spheres)) for r in records[1:])
    assert (held.first, held.n_first) == (0, 0)
    everything = estimate(mapped=labels, rendered=labels)
    assert (everything.tier_b, everything.renders, everything.first) == (0, 0, 0)
    assert reviewing.estimate_line(everything, new=False).startswith("To fetch: at most ")
    nothing = reviewing.RemoteEstimate(
        subjects=3, tier_a=0, tier_b=0, template=0, first=0, n_first=0, renders=0, peak=0
    )
    assert reviewing.estimate_line(nothing, new=False).startswith("To fetch: nothing but the")


def test_a_second_review_changes_nothing_and_fetches_nothing_but_the_listing(
    run_dir: Path, fake: FakeS3, rendered: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config_bytes = (run_dir / CONFIG_FILE).read_bytes()
    tables = {path.name: path.read_bytes() for path in (run_dir / "tables").glob("*.parquet")}
    remote_review(fake, run_dir).close()
    assert rendered == []
    assert (run_dir / CONFIG_FILE).read_bytes() == config_bytes
    assert tables == {
        path.name: path.read_bytes() for path in (run_dir / "tables").glob("*.parquet")
    }
    said = capsys.readouterr().err
    assert f"Run directory: {run_dir} (continued)" in said
    # the run holds its template, every map and every render: the estimate says so
    assert "To fetch: nothing but the listing; this run holds" in said
    assert "Rendered: 3 of 3 subjects." in said
    # the listing, and the description it reads the fMRIPrep version from; no other file
    gets = [entry.key for entry in fake.log if entry.method == "GET"]
    assert gets == [f"fmriprep/{FAKE_ACCESSION}-fmriprep/dataset_description.json"]
    # the pinned template is read from templates/: no file of fsaverage is fetched (its
    # prefix is listed with the dataset, which says whether it has a template)
    assert not any(
        "/fsaverage/" in (entry.key or "") for entry in fake.log if entry.method == "GET"
    )


def test_the_template_of_a_remote_dataset_is_that_of_a_local_review(
    prepared: Path, local_review: Path
) -> None:
    with (
        PublicCatalog.open(prepared) as remote_public,
        PublicCatalog.open(local_review) as local_public,
    ):
        assert remote_public.meta("template_id") == local_public.meta("template_id")
        assert remote_public.meta("template_id") is not None


def test_a_local_and_a_remote_review_of_the_same_files_give_the_same_outputs(
    prepared: Path, local_review: Path
) -> None:
    """Matched by subject label, since the salts and so the uids differ."""
    remote_uids, local_uids = uids_by_label(prepared), uids_by_label(local_review)
    assert sorted(remote_uids) == sorted(local_uids) == list(LABELS)
    for label in LABELS:
        here = prepared / "media" / remote_uids[label]
        there = local_review / "media" / local_uids[label]
        names = [p.relative_to(there) for p in files_under(there) if p.name != "render.json"]
        assert names and names == [
            p.relative_to(here) for p in files_under(here) if p.name != "render.json"
        ]
        for name in names:
            assert (here / name).read_bytes() == (there / name).read_bytes(), (label, name)
        assert (prepared / "tables" / "fsavg" / f"{remote_uids[label]}.npz").read_bytes() == (
            local_review / "tables" / "fsavg" / f"{local_uids[label]}.npz"
        ).read_bytes()

    def rows(run_dir: Path, name: str, uids: dict[str, str]) -> pd.DataFrame:
        frame = read_table(table_path(run_dir, name))
        label = {uid: lab for lab, uid in uids.items()}
        frame = frame.assign(subject_uid=frame["subject_uid"].map(label))
        keys = [c for c in ("subject_uid", "hemi", "parcel") if c in frame.columns]
        return frame.sort_values(keys).reset_index(drop=True)

    for name in ("regional", "vertex_summary"):
        pd.testing.assert_frame_equal(
            rows(prepared, name, remote_uids), rows(local_review, name, local_uids)
        )


# --- when the app opens ---------------------------------------------------------------------------


@pytest.fixture
def opened() -> Iterator[list[tuple[Run, ReviewApp | None]]]:
    made: list[tuple[Run, ReviewApp | None]] = []
    yield made
    for run, app in made:
        if app is not None:
            app.stop()
        run.close()


def new_remote_run(fake: FakeS3, run_dir: Path, **overrides: Any) -> None:
    """A remote run made with a configuration of its own, as a user's earlier call made it."""
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    source = OpenNeuroDerivatives(client=client, accession=FAKE_ACCESSION)
    config = new_local_config(None, **overrides)
    Run(run_dir, source, config=config, transport=fake.transport()).close()


def test_the_app_opens_at_the_first_render_and_the_others_follow(
    tmp_path: Path,
    fake: FakeS3,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = threading.Event()
    original = reviewing._render_rest

    def held(*args: Any, **kwargs: Any) -> None:
        gate.wait(60)
        original(*args, **kwargs)

    monkeypatch.setattr(reviewing, "_render_rest", held)
    run_dir = tmp_path / "run"
    new_remote_run(fake, run_dir, metrics=WITHIN_THE_SET)
    run, app = open_review(TEXT, run_dir=run_dir, jobs=1, port=0, transport=fake.transport())
    opened.append((run, app))
    assert app is not None and app.rendering and app.handle.running
    assert app.fetched_at_app is not None and app.fetched_at_app[1] > 0
    queue = queue_of(run_dir)
    assert render_counts(run) == (1, 0, 2)
    done = [uid for uid in queue if (run_dir / "media" / uid / "render.json").is_file()]
    assert done == queue[:1]
    before_behind = len(fake.log)
    with httpx.Client(base_url=app.url, trust_env=False) as client:
        first = client.get("/api/next").json()
        assert first["assignment"]["subject_uid"] == queue[0]
        assert first["progress"]["rendered"] == 1
        gate.set()
        assert app.wait_rendered(120) and app.render_problem is None
        assert render_counts(run) == (3, 0, 0)
        again = client.get("/api/next").json()
        assert again["progress"]["rendered"] == 3
    # the renderer behind the app fetched the render files of the two others
    behind = [entry.key or "" for entry in fake.log[before_behind:] if entry.method == "GET"]
    assert behind and all("/surf/" in key or "/mri/" in key for key in behind)
    with Run(run_dir) as fresh:
        assert len(fresh.public().work("native")) == 1
        clusters = fresh.clusters()
        assert len(clusters[clusters["native_vertex"].notna()]) > 0
    stop = threading.Event()
    stop.set()
    reviewing.serve_review(run, app, open_browser=False, stop=stop)  # stops at once
    said = capsys.readouterr().err
    assert "fetched in" in said and said.splitlines()[-1].startswith("Fetched ")
    assert files_under(run_dir / CACHE_DIR / LEASES) == []


def test_a_fault_of_the_bucket_behind_the_app_is_said_and_the_next_call_goes_on(
    tmp_path: Path,
    fake: FakeS3,
    opened: list[tuple[Run, ReviewApp | None]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bucket becomes unreachable while the renderer behind the app works: one line
    says so, the app keeps serving what is rendered, nothing of the batch is recorded, and
    the next call renders the rest."""
    gate = threading.Event()
    original = reviewing._render_rest

    def held(*args: Any, **kwargs: Any) -> None:
        gate.wait(60)
        original(*args, **kwargs)

    monkeypatch.setattr(reviewing, "_render_rest", held)
    run_dir = tmp_path / "run"
    # no retries, so that a reset is "unreachable" at once
    new_remote_run(fake, run_dir, remote=RemoteConfig(max_retries=0, max_concurrency=4))
    run, app = open_review(TEXT, run_dir=run_dir, jobs=1, port=0, transport=fake.transport())
    opened.append((run, app))
    assert app is not None and render_counts(run) == (1, 0, 2)
    rendered_first = set(queue_of(run_dir)[:1])
    fake.fault("*/mri/T1.mgz", "reset", times=None)
    gate.set()
    assert app.wait_rendered(120)
    assert app.render_problem is not None and app.render_problem.startswith("unreachable:")
    said = capsys.readouterr().err
    assert "Rendering stopped: unreachable: " in said
    assert "The app goes on with the subjects that are rendered" in said
    assert FAKE_ACCESSION not in app.render_problem
    with httpx.Client(base_url=app.url, trust_env=False) as client:
        answer = client.get("/api/next").json()
        assert answer["status"] == "assignment"
        assert answer["assignment"]["subject_uid"] in rendered_first
    assert render_counts(run) == (1, 0, 2)  # nothing of the stopped batch is recorded
    app.stop()
    run.close()
    opened.clear()
    fake.clear()
    with remote_review(fake, run_dir) as again:
        assert render_counts(again) == (3, 0, 0)
    assert files_under(run_dir / CACHE_DIR / LEASES) == []


def test_a_stopped_renderer_leaves_the_rest_pending_and_the_next_call_makes_it(
    run_dir: Path, fake: FakeS3
) -> None:
    shutil.rmtree(run_dir / "media")
    asked: list[int] = []

    def stop() -> bool:
        asked.append(1)
        return len(asked) > 1

    with Run(run_dir, transport=fake.transport()) as run:
        run.config = with_changes(
            run.config, cache=run.config.cache.model_dump() | {"batch_subjects": 1}
        )
        report = run.render(stop=stop)
        assert (report.done, report.failed) == (1, 0)
        assert render_counts(run) == (1, 0, 2)
        report = run.render()
        assert report.done == 2 and render_counts(run) == (3, 0, 0)
    assert files_under(run_dir / CACHE_DIR / LEASES) == []


# --- subjects that cannot be rendered, and a bucket without a template ----------------------------


def test_an_incomplete_subject_is_named_selected_and_never_offered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = write_tree(tmp_path / "study", "fmriprep", ["sub-01", "sub-02"], incomplete=["sub-02"])
    fake = FakeS3({BUCKET: bucket_of(tree / "sourcedata" / "freesurfer", root=tmp_path / "bucket")})
    run_dir = tmp_path / "run"
    with remote_review(fake, run_dir) as run:
        said = capsys.readouterr().err
        assert (
            f"Found 2 subjects in {TEXT} (fMRIPrep layout). 1 cannot be rendered: sub-02 (no "
            in said
        )
        assert "Rendered: 1 of 2 subjects, 1 cannot be rated." in said
        assert "  not rendered, missing surface: 1 (sub-02)" in said
        assert render_counts(run) == (1, 1, 0) and rateable(run) == 1
    uids = uids_by_label(run_dir)
    with TestClient(create_app(run_dir), base_url=BASE) as client:
        first = client.get("/api/next").json()
        assert first["assignment"]["subject_uid"] == uids["sub-01"]
        assert first["progress"]["n_subjects"] == 1


def test_review_without_a_template_ranks_and_renders_without_counterparts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = write_tree(tmp_path / "study", "subjects_dir", ["sub-01", "sub-02"])
    fake = FakeS3({BUCKET: bucket_of(tree, root=tmp_path / "bucket")})
    with caplog.at_level("WARNING"), remote_review(fake, tmp_path / "run") as run:
        assert run.template() is None
        assert render_counts(run) == (2, 0, 0)
        assert run.public().renders()["template_id"].isna().all()
    assert "--fsaverage" in caplog.text
    said = capsys.readouterr().err
    assert "the fsaverage template" not in said  # nothing of a template was estimated
    assert FAKE_ACCESSION not in caplog.text


# --- the phases on faults of single files (tests/test_pipeline.py) --------------------------------


def test_a_file_that_changed_since_the_inventory_is_source_changed(
    tmp_path: Path, fake: FakeS3
) -> None:
    run_dir = tmp_path / "run"
    new_remote_run(fake, run_dir)
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    source = OpenNeuroDerivatives(client=client, accession=FAKE_ACCESSION)
    # the object changes once the inventory has listed it: the listing gives the earlier
    # ETag, and the lease of the metrics gets another (412, as the bucket answers If-Match)
    fake.mutate_after_list(BUCKET, key_of(FAKE_ACCESSION, "sub-001", "label/lh.aparc.annot"))
    with Run(run_dir, source, transport=fake.transport()) as run:
        run.inventory()
        uids = {label: uid for uid, label in reviewing._labels(run).items()}
        report = run.metrics()
        assert report.failures == ((uids["sub-001"], "fsaverage: source changed"),)
        assert (report.failed, report.done) == (1, 3 + 2 + 2)


def test_an_unreadable_file_fails_its_subject_and_no_other(tmp_path: Path, fake: FakeS3) -> None:
    run_dir = tmp_path / "run"
    new_remote_run(fake, run_dir)
    fake.fault(key_of(FAKE_ACCESSION, "sub-000", "surf/lh.thickness"), "403", times=None)
    with remote_review(fake, run_dir) as run:
        uids = {label: uid for uid, label in reviewing._labels(run).items()}
        failures = run._catalog.failures()
        rows = [
            (row.unit, row.phase, row.reason, row.detail)
            for row in failures.itertuples(index=False)
            if row.phase == "fsaverage"
        ]
        assert [(unit, phase, reason) for unit, phase, reason, _ in rows] == [
            (uids["sub-000"], "fsaverage", "unreadable")
        ]
        detail = rows[0][3]
        assert detail.startswith("surf/lh.thickness") and FAKE_ACCESSION not in detail
        # the other subjects have their maps and their renders
        assert render_counts(run) == (3, 0, 0)


# --- what a remote review refuses -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("option", "words"),
    [
        ({"relocate": True}, "does not move"),
        ({"raw": "/tmp"}, "--raw"),
        ({"epi": True}, "no EPI interface"),
        ({"epi": False}, "no EPI interface"),
        ({"epi_runs": ["*rest*"]}, "--epi-runs"),
        ({"field_strength": 3.0}, "--field-strength"),
    ],
)
def test_options_of_a_local_review_are_refused_before_anything_is_listed(
    tmp_path: Path, fake: FakeS3, option: dict[str, Any], words: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        remote_review(fake, tmp_path / "run", **option)
    assert caught.value.reason == "invalid" and words in str(caught.value)
    assert fake.log == [] and not (tmp_path / "run").exists()


def test_a_text_that_names_no_dataset_is_refused(tmp_path: Path, fake: FakeS3) -> None:
    with pytest.raises(SourceError) as caught:
        review("openneuro:ds12", run_dir=tmp_path / "run", serve=False, transport=fake.transport())
    assert caught.value.reason == "invalid" and "openneuro:ds" in str(caught.value)
    assert fake.log == []


def test_a_new_review_that_would_fetch_too_much_asks_for_subjects(
    tmp_path: Path, fake: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reviewing, "REMOTE_GUARD_BYTES", 1_000_000)
    run_dir = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        remote_review(fake, run_dir)
    said = str(caught.value)
    assert caught.value.reason == "invalid" and "--subjects" in said and "'*'" in said
    assert "lists 3 subjects" in said and FAKE_ACCESSION not in said
    assert not run_dir.exists()  # refused before anything was written
    # only listings and the description were asked for: no file of a subject
    assert all(
        entry.method == "LIST" or (entry.key or "").endswith("dataset_description.json")
        for entry in fake.log
    )
    # --subjects '*' asks for all of them
    with remote_review(fake, run_dir, subjects=["*"]) as run:
        assert run.config.sources[0].subjects == ("*",)
    # and a continued run is not asked again
    remote_review(fake, run_dir).close()


def test_a_glob_that_matches_nothing_names_the_subjects(tmp_path: Path, fake: FakeS3) -> None:
    with pytest.raises(SourceError) as caught:
        remote_review(fake, tmp_path / "run", subjects=["sub-9*"])
    assert caught.value.reason == "not found" and "sub-000" in str(caught.value)
    assert not (tmp_path / "run").exists()


def test_another_dataset_or_a_local_run_is_another_root(
    tmp_path: Path, run_dir: Path, fake: FakeS3, small_run_cohort: SyntheticCohort
) -> None:
    other = FakeS3({BUCKET: bucket_of(small_run_cohort, "ds999998", root=tmp_path / "other")})
    with pytest.raises(ConfigError) as caught:
        review("openneuro:ds999998", run_dir=run_dir, serve=False, transport=other.transport())
    assert caught.value.reason == "other root" and "--run-dir" in str(caught.value)
    assert "ds999998" not in str(caught.value) and FAKE_ACCESSION not in str(caught.value)
    assert other.log == []  # refused before anything was listed
    local_run = tmp_path / "local"
    Run(local_run, local(small_run_cohort.root)).close()
    with pytest.raises(ConfigError) as caught:
        remote_review(fake, local_run)
    assert caught.value.reason == "other root"


def test_the_default_run_directory_is_named_after_the_dataset_in_the_current_directory(
    tmp_path: Path, fake: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with review(TEXT, serve=False, jobs=1, transport=fake.transport()) as run:
        assert run.run_dir == tmp_path / f"{FAKE_ACCESSION}-surfcensus-run"


# --- the disk bound of a batch, and the cache ----------------------------------------------------


def test_a_batch_larger_than_the_budget_is_made_smaller(
    run_dir: Path, fake: FakeS3, caplog: pytest.LogCaptureFixture
) -> None:
    shutil.rmtree(run_dir / "media")
    with Run(run_dir, transport=fake.transport()) as run:
        largest = max(run._remote_roles_bytes(run_module.RENDER_LEASED))
        settings = run.config.cache.model_dump() | {"max_bytes": largest + 1, "batch_subjects": 3}
        run.config = with_changes(run.config, cache=settings)
        with caplog.at_level(logging.INFO, logger="surfcensus"):
            report = run.render()
        assert (report.done, report.failed) == (3, 0)
        assert "batches of 1 subjects for the renders" in caplog.text
        assert run.config.cache.batch_subjects == 3  # for that call only


def test_a_subject_larger_than_the_budget_is_refused(run_dir: Path, fake: FakeS3) -> None:
    shutil.rmtree(run_dir / "media")
    with Run(run_dir, transport=fake.transport()) as run:
        settings = run.config.cache.model_dump() | {"max_bytes": 1000}
        run.config = with_changes(run.config, cache=settings)
        with pytest.raises(ConfigError) as caught:
            run.render()
        assert "cache.max_bytes" in str(caught.value) and caught.value.reason == "invalid"
        assert render_counts(run) == (0, 0, 3)


def test_a_disk_without_room_for_a_batch_is_refused(
    run_dir: Path, fake: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.rmtree(run_dir / "media")
    monkeypatch.setattr(run_module, "free_bytes", lambda path: 1_000_000)
    with Run(run_dir, transport=fake.transport()) as run:
        with pytest.raises(ConfigError) as caught:
            run.render()
        assert "free" in str(caught.value) and "make room" in str(caught.value)
        assert render_counts(run) == (0, 0, 3)
        assert fake.log == []  # refused before any lease


def test_the_cache_keeps_what_the_present_subjects_name_and_drops_the_rest(
    run_dir: Path, fake: FakeS3
) -> None:
    with Run(run_dir, transport=fake.transport()) as run:
        cache = Cache(run_dir / CACHE_DIR, max_bytes=10**9, secret=run._catalog.run_secret())
        # an earlier version of a file, and a template tree that the pin makes needless
        stale = cache.path_for(BUCKET, "fmriprep/x/stats/lh.aparc.stats", "0" * 32, retained=True)
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"old")
        tree = cache.path_for(BUCKET, "fmriprep/x/fsaverage", "1" * 64, retained=True)
        (tree / "surf").mkdir(parents=True)
        (tree / "surf" / "lh.white").write_bytes(b"w")
        kept = sorted(files_under(run_dir / CACHE_DIR / RETAINED))
        assert run.prune_cache() == 2
        assert not stale.exists() and not tree.exists()
        assert sorted(files_under(run_dir / CACHE_DIR / RETAINED)) == [
            p for p in kept if p != stale and not p.is_relative_to(tree)
        ]
        assert len(files_under(run_dir / CACHE_DIR / RETAINED)) == 3 * len(TIER_A)
        # a subject that left the dataset: its Tier A files go
        uid = sorted(run.public().subject_uids())[0]
        with run._catalog.write() as conn:
            conn.execute("UPDATE subjects SET present = 0 WHERE subject_uid = ?", (uid,))
        assert run.prune_cache() == len(TIER_A)


def test_without_a_pinned_template_nothing_unknown_is_dropped(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "templates")
    with Run(run_dir) as run:
        before = files_under(run_dir / CACHE_DIR / RETAINED)
        assert run.prune_cache() == 0
        assert files_under(run_dir / CACHE_DIR / RETAINED) == before


def test_what_a_killed_process_left_in_the_cache_is_pruned_by_the_next(
    run_dir: Path, fake: FakeS3
) -> None:
    dead = run_dir / CACHE_DIR / LEASES / f"{reviewing_host()}-999999999"
    (dead / "ab").mkdir(parents=True)
    (dead / ".lock").write_bytes(b"")
    (dead / "ab" / "lh.white").write_bytes(b"partial")
    with Run(run_dir, transport=fake.transport()) as run:
        run._source(0)  # building the remote source prunes
        assert not dead.exists()


def reviewing_host() -> str:
    from surfcensus.remote.cache import _host

    return _host()


# --- the inventory of a census run on the bucket (the census fixture of tests/test_inventory.py) --

DATASET_KEY = privacy.HashKey(bytes(range(32)), "dataset-key")
OTHER = "ds999998"


@pytest.fixture(scope="module")
def two_datasets(
    tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort
) -> Path:
    """``ds999999`` with ``sub-000`` and ``sub-001`` and ``ds999998`` with ``sub-002`` of the
    cohort, as links."""
    root = tmp_path_factory.mktemp("two_datasets")
    cohort = small_run_cohort.root
    for accession, labels in ((FAKE_ACCESSION, LABELS[:2]), (OTHER, LABELS[2:])):
        dataset = root / "fmriprep" / f"{accession}-fmriprep"
        subjects = dataset / "sourcedata" / "freesurfer"
        subjects.mkdir(parents=True)
        (dataset / "dataset_description.json").write_text(
            '{"GeneratedBy": [{"Name": "fMRIPrep", "Version": "23.0.2"}]}'
        )
        for label in labels:
            (subjects / label).symlink_to((cohort / label).resolve(), target_is_directory=True)
    return root


def census_run(fake: FakeS3, run_dir: Path) -> Run:
    client = S3Client(
        BUCKET, config=RemoteConfig(max_concurrency=4, max_retries=0), transport=fake.transport()
    )
    source = OpenNeuroDerivatives(client=client, prefix="fmriprep/")
    return Run(run_dir, source, key=DATASET_KEY, transport=fake.transport())


def test_a_census_inventory_fetches_tier_a_once_and_a_second_one_reads_no_file(
    tmp_path: Path, two_datasets: Path
) -> None:
    fake = FakeS3({BUCKET: two_datasets}, page_size=7)
    with census_run(fake, tmp_path / "census") as run:
        report = run.inventory()
        assert (report.done, report.skipped, report.failed) == (3, 0, 0)
        gets = [entry.key or "" for entry in fake.log if entry.method == "GET"]
        objects = [key for key in gets if not key.endswith("dataset_description.json")]
        # every Tier A file of every subject once, and nothing else
        assert len(objects) == len(set(objects)) == 3 * len(TIER_A)
        public = run.public().subjects()
        assert len(public) == 3 and public["subject_label"].isna().all()
        fake.clear()
        again = run.inventory()
        assert (again.done, again.skipped) == (0, 3)
        assert {entry.method for entry in fake.log} <= {"LIST", "GET"}
        assert all(
            (entry.key or "").endswith("dataset_description.json")
            for entry in fake.log
            if entry.method == "GET"
        )


def test_the_filter_of_a_census_inventory_lists_that_dataset_alone(
    tmp_path: Path, two_datasets: Path
) -> None:
    fake = FakeS3({BUCKET: two_datasets}, page_size=7)
    with census_run(fake, tmp_path / "census") as run:
        run.inventory()  # the datasets are known once listed (discovery is not built)
    fake.clear()
    with census_run(fake, tmp_path / "census") as run:
        report = run.inventory(datasets=[OTHER], force=True)
        assert (report.done, report.failed) == (1, 0)
        asked = [entry.key or "" for entry in fake.log]
        assert not any(FAKE_ACCESSION in key for key in asked)
        pseudonym = privacy.dataset_pseudonym(DATASET_KEY, "openneuro-derivatives", FAKE_ACCESSION)
        report = run.inventory(datasets=[pseudonym], force=True)
        assert (report.done, report.failed) == (2, 0)


def test_stats_that_cannot_be_fetched_are_fetched_by_the_next_inventory(
    tmp_path: Path, two_datasets: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeS3({BUCKET: two_datasets}, page_size=7)
    fake.fault("*/stats/lh.aparc.stats", "403", times=None)
    with census_run(fake, tmp_path / "census") as run:
        with caplog.at_level(logging.INFO, logger="surfcensus"):
            report = run.inventory()
        assert (report.done, report.failed) == (3, 0)
        said = "\n".join([caplog.text, repr(report)])
        for canary in (FAKE_ACCESSION, OTHER, *LABELS):
            assert canary not in said, canary
        fake.clear()
        healed = run.inventory()
        assert healed.done == 3
        gets = [entry.key or "" for entry in fake.log if entry.method == "GET"]
        assert sum(key.endswith("lh.aparc.stats") for key in gets) == 3
        assert run.inventory().skipped == 3
