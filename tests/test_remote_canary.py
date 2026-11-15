"""The canary of decision D20 on remote data (docs/dev/remote_plan.md, section 7).

A census run made from Python over a fake bucket with two datasets goes through inventory,
metrics and render; then every file of the run directory but ``catalog.sqlite`` (with its
``-wal`` and ``-shm``) is searched, name and content, for what must stay in the catalog: the
accessions, the subject labels, every ETag of the bucket, the dataset key in hex and the
bucket keys. The content of ``cache/`` is the source files themselves and is not searched;
its names are. The same search, with the labels allowed as in any local run, is made on a
local run on a remote dataset (``review openneuro:dsXXXXXX``) and its export. Nothing here
touches the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes3 import FakeS3  # pyrefly: ignore[missing-import]
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

pytest.importorskip("fastapi")

from surfcensus.config import RemoteConfig
from surfcensus.export.tables import export_run
from surfcensus.privacy import HashKey
from surfcensus.remote.cache import CACHE_DIR
from surfcensus.remote.s3 import S3Client
from surfcensus.review import review
from surfcensus.run import Run
from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives

BUCKET = "openneuro-derivatives"
FIRST, SECOND = "ds999999", "ds999998"
LABELS = {FIRST: ("sub-CANARY7", "sub-02"), SECOND: ("sub-CANARY8",)}
DATASET_KEY = HashKey(bytes(range(32, 64)), "dataset-key")


@pytest.fixture(autouse=True)
def no_freesurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREESURFER_HOME", raising=False)


@pytest.fixture(scope="module")
def bucket(tmp_path_factory: pytest.TempPathFactory, small_run_cohort: SyntheticCohort) -> Path:
    """Two datasets of the bucket whose subjects are links to those of the cohort under the
    canary labels, with the template next to them. The cohort's logs carry ``ds999999``."""
    root = tmp_path_factory.mktemp("canary_bucket")
    cohort = small_run_cohort.root
    sources = iter(sorted(p for p in cohort.iterdir() if p.name.startswith("sub-")))
    for accession, labels in LABELS.items():
        dataset = root / "fmriprep" / f"{accession}-fmriprep"
        subjects = dataset / "sourcedata" / "freesurfer"
        subjects.mkdir(parents=True)
        (dataset / "dataset_description.json").write_text(
            '{"Name": "fMRIPrep", "GeneratedBy": [{"Name": "fMRIPrep", "Version": "23.0.2"}]}'
        )
        for label in labels:
            (subjects / label).symlink_to(next(sources).resolve(), target_is_directory=True)
        (subjects / "fsaverage").symlink_to(
            (cohort / "fsaverage").resolve(), target_is_directory=True
        )
    return root


def canaries(fake: FakeS3, *, labels: bool) -> list[str]:
    """What no file of the run but the catalog may hold."""
    found = [FIRST, SECOND, DATASET_KEY.secret.hex(), "fmriprep/ds", "s3://"]
    keys = fake.keys(BUCKET)
    found += keys
    for key in keys:
        path = fake.buckets[BUCKET].joinpath(*key.split("/"))
        found.append(hashlib.md5(path.read_bytes()).hexdigest())  # the ETag
    if labels:
        found += [label for names in LABELS.values() for label in names]
    return found


def run_files(run_dir: Path) -> Iterator[tuple[Path, bytes | None]]:
    """Every file of the run but the catalog, with its content; ``None`` for a file of
    ``cache/``, whose content is a source file and whose name alone is searched."""
    for path in sorted(run_dir.rglob("*")):
        if path.name.startswith("catalog.sqlite"):
            continue
        relative = path.relative_to(run_dir)
        if relative.parts[0] == CACHE_DIR or not path.is_file():
            yield relative, None
        else:
            yield relative, path.read_bytes()


def leaks(run_dir: Path, words: list[str]) -> list[tuple[str, str]]:
    found = []
    for relative, content in run_files(run_dir):
        for word in words:
            if word in relative.as_posix():
                found.append((relative.as_posix(), f"name holds {word}"))
            if content is not None and word.encode() in content:
                found.append((relative.as_posix(), f"content holds {word}"))
    return found


def test_a_census_run_on_the_bucket_keeps_every_identifier_in_its_catalog(
    tmp_path: Path, bucket: Path
) -> None:
    fake = FakeS3({BUCKET: bucket}, page_size=7)
    client = S3Client(BUCKET, config=RemoteConfig(max_concurrency=4), transport=fake.transport())
    source = OpenNeuroDerivatives(client=client, prefix="fmriprep/")
    run_dir = tmp_path / "census"
    with Run(run_dir, source, key=DATASET_KEY, transport=fake.transport()) as run:
        report = run.inventory()
        assert (report.done, report.failed) == (3, 0)
        assert run.metrics().failed == 0
        assert run.template() is not None  # the fsaverage next to the subjects, downloaded
        report = run.render()
        assert (report.done, report.failed) == (3, 0)
        rows = run._catalog.query("SELECT accession FROM datasets ORDER BY accession")
        assert [row["accession"] for row in rows] == [SECOND, FIRST]
    # the search finds what it is for: the catalog holds the accession
    assert FIRST.encode() in (run_dir / "catalog.sqlite").read_bytes()
    assert leaks(run_dir, canaries(fake, labels=True)) == []
    assert (run_dir / "media").is_dir() and any((run_dir / "media").iterdir())
    assert fake.methods <= {"GET", "LIST"} and fake.signed == []


def test_a_local_review_of_a_remote_dataset_keeps_its_accession_in_the_catalog(
    tmp_path: Path, bucket: Path
) -> None:
    fake = FakeS3({BUCKET: bucket}, page_size=7)
    run_dir = tmp_path / "run"
    review(
        f"openneuro:{FIRST}", run_dir=run_dir, serve=False, jobs=1, transport=fake.transport()
    ).close()
    export_run(run_dir)
    assert (run_dir / "export").is_dir()
    assert FIRST.encode() in (run_dir / "catalog.sqlite").read_bytes()
    # labels are allowed in a local run, as for any local folder; the rest is not
    assert leaks(run_dir, canaries(fake, labels=False)) == []


def test_the_search_finds_a_leak(tmp_path: Path) -> None:
    """The check itself: a file that names an accession, and a cache name with a label."""
    (tmp_path / "notes.txt").write_text(f"made from {FIRST}\n")
    (tmp_path / CACHE_DIR / "sub-CANARY7").mkdir(parents=True)
    found = leaks(tmp_path, [FIRST, "sub-CANARY7"])
    assert ("notes.txt", f"content holds {FIRST}") in found
    assert (f"{CACHE_DIR}/sub-CANARY7", "name holds sub-CANARY7") in found
