"""The OpenNeuroDerivatives bucket as a source (work package RE), on the fake bucket.

Every test runs against ``tests/fakes3.py``: the one-dataset source and the source of every
dataset of ``fmriprep/``, the two FreeSurfer roots, the layouts and versions that a local
directory with the same prefixes gives, the subjects with their roles, variants, ETags and
times, the globs, ``read_bytes`` whole and ranged with the Tier A files of a subject fetched
together, the lease through the cache, the template directory, ``config()`` through
``run.from_config``, an object that changed after its listing, and an inventory through the
remote source that equals one through ``LocalSource`` on the same files.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from fakes3 import (  # pyrefly: ignore[missing-import]
    FAKE_ACCESSION,
    FakeS3,
    bucket_of,
    key_of,
)
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

from surfcensus import templates
from surfcensus.config import (
    ConfigError,
    InventoryConfig,
    MetricsConfig,
    RemoteConfig,
    new_local_config,
    remote_dataset_config,
    with_changes,
)
from surfcensus.inventory.subjects import inventory_subject
from surfcensus.remote.cache import CACHE_DIR, LOCK_FILE, Cache
from surfcensus.remote.s3 import S3Client
from surfcensus.run import from_config
from surfcensus.sources.base import TIER_A, FileRole, SourceError, SubjectRecord
from surfcensus.sources.local import LocalSource
from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives, dataset_root

BUCKET = "openneuro-derivatives"
OTHER = "ds999998"
SECRET = bytes(range(32))
ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")

SUBJECT_FILES = (
    "mri/T1.mgz",
    "mri/brainmask.mgz",
    "mri/transforms/talairach.xfm",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
    "stats/aseg.stats",
    "scripts/recon-all.log",
    "scripts/build-stamp.txt",
    "scripts/recon-all-lh.log",
    "scripts/recon-all-rh.log",
)
TEMPLATE = (
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "surf/lh.white",
    "surf/rh.white",
    "label/lh.cortex.label",
    "label/rh.cortex.label",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
)
TEMPLATE_EXTRA = (
    "surf/lh.white.avg.area.mgh",
    "surf/rh.white.avg.area.mgh",
    "scripts/build-stamp.txt",
)
DESCRIPTION = (
    b'{"Name": "fMRIPrep", "DatasetType": "derivative",'
    b' "GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.2"}]}'
)


def write(root: Path, relative: str, content: bytes) -> Path:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def content(label: str, name: str) -> bytes:
    """Distinct bytes per file, of different sizes."""
    return f"{label}:{name}\n".encode() * (3 + len(name) % 5)


def subject(root: Path, label: str, names: tuple[str, ...] = SUBJECT_FILES) -> None:
    for name in names:
        write(root / label, name, content(label, name))


def dataset(
    bucket: Path,
    accession: str = FAKE_ACCESSION,
    *,
    fs_root: str = "sourcedata/freesurfer",
    labels: tuple[str, ...] = ("sub-01", "sub-02"),
    template: tuple[str, ...] = TEMPLATE + TEMPLATE_EXTRA,
    description: str | None = "dataset_description.json",
) -> Path:
    """A dataset of the fake bucket: ``fmriprep/<accession>-fmriprep/`` with subjects and an
    ``fsaverage`` below ``fs_root``. Returns the dataset's directory."""
    base = bucket / "fmriprep" / f"{accession}-fmriprep"
    for label in labels:
        subject(base / fs_root, label)
    for name in template:
        write(base / fs_root / "fsaverage", name, content("fsaverage", name))
    if description is not None:
        write(base, description, DESCRIPTION)
    return base


def client(fake: FakeS3, *, concurrency: int = 4, retries: int = 1) -> S3Client:
    return S3Client(
        BUCKET,
        config=RemoteConfig(max_concurrency=concurrency, max_retries=retries),
        transport=fake.transport(),
        sleep=lambda _: None,
    )


def source(
    fake: FakeS3,
    *,
    accession: str | None = FAKE_ACCESSION,
    subjects: tuple[str, ...] = (),
    cache: Cache | None = None,
    concurrency: int = 4,
    retries: int = 1,
) -> OpenNeuroDerivatives:
    return OpenNeuroDerivatives(
        client=client(fake, concurrency=concurrency, retries=retries),
        cache=cache,
        accession=accession,
        subjects=subjects,
    )


def cache_at(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "run" / CACHE_DIR, max_bytes=10**9, secret=SECRET)


def gets(fake: FakeS3) -> list[str]:
    return [entry.key or "" for entry in fake.log if entry.method == "GET"]


def lists(fake: FakeS3) -> list[str]:
    return [entry.key or "" for entry in fake.log if entry.method == "LIST"]


def assert_private(detail: str) -> None:
    """No accession, bucket key, prefix, URL or ETag in a text that may reach a log."""
    for text in (FAKE_ACCESSION, OTHER, "fmriprep/", "s3://", "http", "amazonaws"):
        assert text not in detail, detail
    assert not re.search(r"[0-9a-f]{32}", detail), detail


def only_reads(fake: FakeS3) -> None:
    assert fake.methods <= {"GET", "HEAD", "LIST"}
    assert fake.signed == []
    assert fake.high_water <= 4


# --- datasets ------------------------------------------------------------------------------------


def test_the_record_of_one_dataset(tmp_path: Path) -> None:
    dataset(tmp_path / "b")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    found = list(source(fake).datasets())
    assert len(found) == 1
    record = found[0]
    assert record.source == "openneuro-derivatives"
    assert record.identity == record.accession == FAKE_ACCESSION
    assert record.root == dataset_root(BUCKET, FAKE_ACCESSION)
    assert record.root == f"s3://{BUCKET}/fmriprep/{FAKE_ACCESSION}-fmriprep/"
    assert (record.fs_root, record.layout, record.pipeline) == (
        "sourcedata/freesurfer",
        "fmriprep",
        "fmriprep",
    )
    assert record.pipeline_version == "21.0.2"
    assert record.template_location == "sourcedata/freesurfer/fsaverage"
    # the top, the subjects' folder, one subject's top, the template: four listings, one GET
    assert len(lists(fake)) == 4 and gets(fake) == [
        f"fmriprep/{FAKE_ACCESSION}-fmriprep/dataset_description.json"
    ]
    only_reads(fake)


def local_record(directory: Path) -> tuple[object, ...]:
    record = LocalSource(directory).dataset
    return (record.fs_root, record.layout, record.pipeline, record.pipeline_version)


def remote_record(fake: FakeS3) -> tuple[object, ...]:
    (record,) = source(fake).datasets()
    return (record.fs_root, record.layout, record.pipeline, record.pipeline_version)


@pytest.mark.parametrize(
    ("fs_root", "description", "expected"),
    [
        (
            "sourcedata/freesurfer",
            "dataset_description.json",
            ("sourcedata/freesurfer", "fmriprep", "fmriprep", "21.0.2"),
        ),
        (
            "freesurfer",
            "fmriprep/dataset_description.json",
            ("freesurfer", "fmriprep_legacy", "fmriprep", "21.0.2"),
        ),
        (
            "freesurfer",
            "dataset_description.json",
            ("freesurfer", "bids_derivatives", "freesurfer", None),
        ),
        ("sourcedata/freesurfer", None, ("sourcedata/freesurfer", "fmriprep", "fmriprep", None)),
    ],
)
def test_layout_and_pipeline_are_those_of_a_local_directory(
    tmp_path: Path, fs_root: str, description: str | None, expected: tuple[object, ...]
) -> None:
    base = dataset(tmp_path / "b", fs_root=fs_root, description=description)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    assert remote_record(fake) == expected
    assert local_record(base) == expected


def test_sourcedata_comes_before_the_older_root(tmp_path: Path) -> None:
    base = dataset(tmp_path / "b")
    subject(base / "freesurfer", "sub-99")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    assert remote_record(fake)[0] == "sourcedata/freesurfer"
    assert local_record(base)[0] == "sourcedata/freesurfer"


def test_a_root_without_a_subject_candidate_is_passed_over(tmp_path: Path) -> None:
    base = dataset(tmp_path / "b", fs_root="freesurfer")
    # sourcedata/freesurfer holds only a template and a folder without mri/
    write(base / "sourcedata/freesurfer/fsaverage", "surf/lh.white", b"x")
    write(base / "sourcedata/freesurfer/sub-01", "surf/lh.white", b"x")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    assert remote_record(fake)[:2] == ("freesurfer", "bids_derivatives")
    assert local_record(base)[:2] == ("freesurfer", "bids_derivatives")


def test_a_dataset_without_freesurfer(tmp_path: Path) -> None:
    base = tmp_path / "b" / "fmriprep" / f"{FAKE_ACCESSION}-fmriprep"
    write(base, "dataset_description.json", DESCRIPTION)
    write(base, "sub-01/func/sub-01_bold.nii.gz", b"bold")
    dataset(tmp_path / "b", OTHER)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    with pytest.raises(SourceError) as raised:
        list(source(fake).datasets())
    assert raised.value.reason == "layout"
    assert_private(raised.value.detail)
    with pytest.raises(SourceError, match="layout"):
        LocalSource(base)
    # the source of every dataset passes it over
    assert [record.accession for record in source(fake, accession=None).datasets()] == [OTHER]


def test_a_dataset_that_is_not_in_the_bucket(tmp_path: Path) -> None:
    dataset(tmp_path / "b", OTHER)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    with pytest.raises(SourceError) as raised:
        list(source(fake).datasets())
    assert raised.value.reason == "not found"
    assert_private(raised.value.detail)


def test_the_source_of_every_dataset(tmp_path: Path) -> None:
    dataset(tmp_path / "b", OTHER, fs_root="freesurfer", labels=("sub-a",))
    dataset(tmp_path / "b")
    write(tmp_path / "b", "fmriprep/README", b"x")
    write(tmp_path / "b", "fmriprep/notes/x.txt", b"x")
    write(tmp_path / "b", "fmriprep/ds12345-fmriprep/x.txt", b"x")  # not an accession
    write(tmp_path / "b", "mriqc/ds000001-mriqc/x.txt", b"x")
    fake = FakeS3({BUCKET: tmp_path / "b"}, page_size=2)
    census = source(fake, accession=None)
    records = list(census.datasets())
    assert [(r.accession, r.identity, r.fs_root) for r in records] == [
        (OTHER, OTHER, "freesurfer"),
        (FAKE_ACCESSION, FAKE_ACCESSION, "sourcedata/freesurfer"),
    ]
    assert lists(fake)[0] == "fmriprep/"
    before = len(fake.log)
    assert [r.accession for r in census.datasets({FAKE_ACCESSION})] == [FAKE_ACCESSION]
    assert list(census.datasets(set())) == []
    assert len(fake.log) == before  # kept for the life of the source
    one = source(fake)
    assert list(one.datasets({OTHER})) == []
    only_reads(fake)


def test_a_template_that_lacks_a_file_is_not_recorded(tmp_path: Path) -> None:
    dataset(tmp_path / "b", template=TEMPLATE[1:])
    fake = FakeS3({BUCKET: tmp_path / "b"})
    (record,) = source(fake).datasets()
    assert record.template_location is None
    assert source(fake, cache=cache_at(tmp_path)).template_dir(record) is None


# --- subjects ------------------------------------------------------------------------------------


def test_subjects_in_label_order_with_their_files(tmp_path: Path) -> None:
    base = dataset(tmp_path / "b", labels=("sub-10", "sub-02", "sub-1"))
    subject(base / "sourcedata/freesurfer", "fsaverage5")
    subject(base / "sourcedata/freesurfer", "cvs_avg35")
    write(base / "sourcedata/freesurfer/sub-03", "surf/lh.white", b"no mri")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake)
    (record,) = remote.datasets()
    found = list(remote.subjects(record))
    assert [s.subject_label for s in found] == ["sub-02", "sub-1", "sub-10"]
    first = found[0]
    assert first.fs_dir == "sourcedata/freesurfer/sub-02" and first.dataset == record
    assert first.problem is None and dict(first.unfetched) == {}
    assert list(first.files) == [role for role in FileRole]  # in FileRole order
    for role, ref in first.files.items():
        data = content("sub-02", ref.location)
        assert ref.role == role
        assert ref.size == len(data)
        assert ref.etag_or_mtime == hashlib.md5(data).hexdigest()  # without quotes
        assert ref.last_modified is not None and ISO.fullmatch(ref.last_modified)
    assert first.files[FileRole.LH_PIAL].variant == "lh.pial"
    assert first.files[FileRole.LH_WHITE].variant is None
    # no file was opened
    assert gets(fake) == [f"fmriprep/{FAKE_ACCESSION}-fmriprep/dataset_description.json"]
    only_reads(fake)


def test_pial_t1_alone_gives_that_variant_and_an_incomplete_subject_is_listed(
    tmp_path: Path,
) -> None:
    names = tuple(
        name.replace("lh.pial", "lh.pial.T1") for name in SUBJECT_FILES if name != "surf/rh.pial"
    )
    base = dataset(tmp_path / "b", labels=("sub-01",))
    subject(base / "sourcedata/freesurfer", "sub-07", names)
    write(base / "sourcedata/freesurfer/sub-07", "surf/rh.pial.T2", b"t2")
    write(base / "sourcedata/freesurfer/sub-07", "surf/rh.pial.T1", b"t1")
    subject(base / "sourcedata/freesurfer", "sub-08", ("mri/orig.mgz", "surf/lh.orig"))
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake)
    (record,) = remote.datasets()
    by_label = {s.subject_label: s for s in remote.subjects(record)}
    pial = by_label["sub-07"].files[FileRole.LH_PIAL]
    assert (pial.location, pial.variant) == ("surf/lh.pial.T1", "lh.pial.T1")
    rh = by_label["sub-07"].files[FileRole.RH_PIAL]
    assert (rh.location, rh.variant) == ("surf/rh.pial.T2", "lh.pial.T2".replace("lh", "rh"))
    assert by_label["sub-08"].files == {}  # mri/ and surf/ without the render files
    local = {s.subject_label: s for s in LocalSource(base).subjects(LocalSource(base).dataset)}
    for label in by_label:
        assert {r: (f.location, f.size, f.variant) for r, f in by_label[label].files.items()} == {
            r: (f.location, f.size, f.variant) for r, f in local[label].files.items()
        }


def test_globs_list_only_the_matching_subject_prefixes(tmp_path: Path) -> None:
    labels = tuple(f"sub-{n:02d}" for n in range(14))
    dataset(tmp_path / "b", labels=labels)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, subjects=("sub-0[12]",))
    (record,) = remote.datasets()
    fake.clear()
    found = list(remote.subjects(record))
    assert [s.subject_label for s in found] == ["sub-01", "sub-02"]
    prefix = f"fmriprep/{FAKE_ACCESSION}-fmriprep/sourcedata/freesurfer/"
    assert sorted(lists(fake)) == [prefix + "sub-01/", prefix + "sub-02/"]
    # kept: the same call again makes no request
    fake.clear()
    assert [s.subject_label for s in remote.subjects(record)] == ["sub-01", "sub-02"]
    assert fake.log == []


def test_a_glob_that_matches_nothing_names_ten_subjects(tmp_path: Path) -> None:
    labels = tuple(f"sub-{n:02d}" for n in range(14))
    dataset(tmp_path / "b", labels=labels)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, subjects=("sub-01", "sub-9*"))
    (record,) = remote.datasets()
    fake.clear()
    with pytest.raises(SourceError) as raised:
        remote.subjects(record)
    assert raised.value.reason == "not found"
    detail = raised.value.detail
    assert "'sub-9*'" in detail and "sub-00, sub-01" in detail and "sub-09" in detail
    assert "sub-10" not in detail and detail.endswith("and 4 more")
    assert_private(detail)
    assert lists(fake) == []  # refused before any subject was listed


def test_a_subject_prefix_that_cannot_be_listed(tmp_path: Path) -> None:
    dataset(tmp_path / "b")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, retries=0)
    (record,) = remote.datasets()
    fake.fault("*sub-02/", "500", times=None, methods=("LIST",))
    with pytest.raises(SourceError) as raised:
        remote.subjects(record)
    assert raised.value.reason == "unreachable"
    assert_private(raised.value.detail)


# --- read_bytes ----------------------------------------------------------------------------------


@pytest.fixture
def one(tmp_path: Path) -> tuple[FakeS3, OpenNeuroDerivatives, SubjectRecord]:
    dataset(tmp_path / "b", labels=("sub-01",))
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, cache=cache_at(tmp_path))
    (record,) = remote.datasets()
    (first,) = remote.subjects(record)
    fake.clear()
    return fake, remote, first


def test_the_first_read_fetches_tier_a_together_and_keeps_it(
    one: tuple[FakeS3, OpenNeuroDerivatives, SubjectRecord],
) -> None:
    fake, remote, first = one
    stamp = remote.read_bytes(first, FileRole.BUILD_STAMP)
    assert stamp == content("sub-01", "scripts/build-stamp.txt")
    tier_a = {key_of(FAKE_ACCESSION, "sub-01", first.files[r].location) for r in TIER_A}
    assert sorted(gets(fake)) == sorted(tier_a)
    assert all(entry.if_match is not None for entry in fake.log)
    fake.clear()
    for role in TIER_A:
        assert remote.read_bytes(first, role) == content("sub-01", first.files[role].location)
    assert (
        remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(2, 5))
        == (content("sub-01", "scripts/recon-all.log")[2:6])
    )
    assert fake.log == []
    only_reads(fake)


def test_ranges_are_clamped_and_asked_with_if_match(
    one: tuple[FakeS3, OpenNeuroDerivatives, SubjectRecord],
) -> None:
    fake, remote, first = one
    data = content("sub-01", "scripts/recon-all.log")
    # before the file is retained a range is one ranged GET
    assert remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(0, 9)) == data[:10]
    (entry,) = fake.log
    ref = first.files[FileRole.RECON_LOG]
    assert (entry.range, entry.if_match) == ("bytes=0-9", f'"{ref.etag_or_mtime}"')
    fake.clear()
    assert remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(5, 10**9)) == data[5:]
    assert fake.log[0].range == f"bytes=5-{len(data) - 1}"
    fake.clear()
    assert (
        remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(len(data), len(data) + 9)) == b""
    )
    assert fake.log == []
    with pytest.raises(ValueError):
        remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(5, 4))
    # a role outside Tier A is read from the bucket and not kept
    assert remote.read_bytes(first, FileRole.TALAIRACH) == content(
        "sub-01", "mri/transforms/talairach.xfm"
    )
    remote.read_bytes(first, FileRole.TALAIRACH)
    assert len(gets(fake)) == 2


def test_a_role_the_subject_lacks_is_not_found(
    one: tuple[FakeS3, OpenNeuroDerivatives, SubjectRecord],
) -> None:
    _, remote, first = one
    import dataclasses

    without = dataclasses.replace(
        first, files={r: f for r, f in first.files.items() if r is not FileRole.ASEG_STATS}
    )
    with pytest.raises(SourceError) as raised:
        remote.read_bytes(without, FileRole.ASEG_STATS)
    assert raised.value.reason == "not found"
    # the others are still read, with the six Tier A files it has
    assert remote.read_bytes(without, FileRole.LH_STATS) == content(
        "sub-01", "stats/lh.aparc.stats"
    )


def test_an_unbound_source_reads_from_the_bucket(tmp_path: Path) -> None:
    dataset(tmp_path / "b", labels=("sub-01",))
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake)
    (record,) = remote.datasets()
    (first,) = remote.subjects(record)
    fake.clear()
    assert remote.read_bytes(first, FileRole.BUILD_STAMP) == content(
        "sub-01", "scripts/build-stamp.txt"
    )
    assert len(gets(fake)) == 1
    with pytest.raises(ConfigError):
        remote.lease(first, [FileRole.LH_WHITE])
    with pytest.raises(ConfigError):
        remote.template_dir(record)


def test_a_changed_object_is_changed(tmp_path: Path) -> None:
    dataset(tmp_path / "b", labels=("sub-01",))
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, cache=cache_at(tmp_path))
    for location in ("scripts/recon-all.log", "surf/lh.white", "stats/lh.aparc.stats"):
        fake.mutate_after_list(BUCKET, key_of(FAKE_ACCESSION, "sub-01", location))
    (record,) = remote.datasets()
    (first,) = remote.subjects(record)
    with pytest.raises(SourceError) as raised:
        remote.read_bytes(first, FileRole.RECON_LOG)
    assert raised.value.reason == "changed"
    assert raised.value.detail.startswith("scripts/recon-all.log: ")
    assert_private(raised.value.detail)
    # the other Tier A files of the round were fetched and are read
    assert remote.read_bytes(first, FileRole.BUILD_STAMP)
    with pytest.raises(SourceError) as again:
        remote.read_bytes(first, FileRole.RECON_LOG, byte_range=(0, 3))
    assert again.value.reason == "changed"
    for roles, place in (
        ([FileRole.LH_WHITE, FileRole.RH_WHITE], "surf/lh.white: "),
        ([FileRole.LH_STATS], "stats/lh.aparc.stats: "),
    ):
        with pytest.raises(SourceError) as leased:
            with remote.lease(first, roles):
                pass
        assert leased.value.reason == "changed"
        assert leased.value.detail.startswith(place), leased.value.detail
        assert_private(leased.value.detail)
    only_reads(fake)


# --- lease ---------------------------------------------------------------------------------------


def test_a_lease_through_the_cache(
    tmp_path: Path, one: tuple[FakeS3, OpenNeuroDerivatives, SubjectRecord]
) -> None:
    fake, remote, first = one
    assert remote.cache is not None
    roles = [FileRole.LH_WHITE, FileRole.RH_THICKNESS, FileRole.LH_STATS]
    with remote.lease(first, [*roles, FileRole.TALAIRACH, FileRole.ASEG_STATS]) as paths:
        assert list(paths) == [
            FileRole.LH_WHITE,
            FileRole.RH_THICKNESS,
            FileRole.LH_STATS,
            FileRole.ASEG_STATS,
            FileRole.TALAIRACH,
        ]
        for role, path in paths.items():
            assert path.read_bytes() == content("sub-01", first.files[role].location)
            assert path.name == Path(first.files[role].location).name
    assert len(gets(fake)) == 5
    area = remote.cache.area
    assert [p.name for p in area.iterdir()] == [LOCK_FILE]
    # Tier A was kept: the stats are read and leased again without a request
    fake.clear()
    assert remote.read_bytes(first, FileRole.LH_STATS) == content("sub-01", "stats/lh.aparc.stats")
    assert fake.log == []
    # a Tier A file that is not kept starts the round, which fetches the five the lease did not
    assert remote.read_bytes(first, FileRole.RECON_LOG) == content(
        "sub-01", "scripts/recon-all.log"
    )
    assert len(gets(fake)) == 5
    fake.clear()
    with remote.lease(first, [FileRole.LH_STATS, FileRole.ASEG_STATS, FileRole.RECON_LOG]):
        pass
    assert fake.log == []
    with remote.lease(first, []) as nothing:
        assert dict(nothing) == {}
    only_reads(fake)


def test_tier_a_is_fetched_at_most_seven_at_once(tmp_path: Path) -> None:
    dataset(tmp_path / "b", labels=("sub-01",))
    fake = FakeS3({BUCKET: tmp_path / "b"}, latency_s=0.02)
    remote = source(fake, cache=cache_at(tmp_path), concurrency=8)
    (record,) = remote.datasets()
    (first,) = remote.subjects(record)
    fake.clear()
    remote.read_bytes(first, FileRole.RECON_LOG_RH)
    assert len(gets(fake)) == 7 and fake.high_water <= 7


# --- the template --------------------------------------------------------------------------------


def test_the_template_directory(tmp_path: Path) -> None:
    dataset(tmp_path / "b")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    cache = cache_at(tmp_path)
    remote = source(fake, cache=cache)
    (record,) = remote.datasets()
    fake.clear()
    found = remote.template_dir(record)
    assert found is not None and found.name == "fsaverage"
    assert found.is_relative_to(cache.root / "retained")
    shown = str(found.relative_to(tmp_path))
    assert FAKE_ACCESSION not in shown and "sourcedata" not in shown
    for name in TEMPLATE + TEMPLATE_EXTRA:
        assert (found / name).read_bytes() == content("fsaverage", name)
    assert len(gets(fake)) == 11
    assert [p.name for p in cache.area.iterdir()] == [LOCK_FILE]
    fake.clear()
    assert remote.template_dir(record) == found
    assert fake.log == []
    # a source made afterwards finds the tree with one listing of the template
    later = source(fake, cache=cache_at(tmp_path))
    fake.clear()
    assert later.template_dir(record) == found
    assert gets(fake) == [] and len(lists(fake)) == 1
    only_reads(fake)


def test_a_template_without_the_optional_files(tmp_path: Path) -> None:
    dataset(tmp_path / "b", template=TEMPLATE)
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, cache=cache_at(tmp_path))
    (record,) = remote.datasets()
    found = remote.template_dir(record)
    assert found is not None
    files = sorted(p.relative_to(found).as_posix() for p in found.rglob("*") if p.is_file())
    assert files == sorted(TEMPLATE)


def test_a_template_file_that_changed_leaves_no_tree(tmp_path: Path) -> None:
    dataset(tmp_path / "b")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    cache = cache_at(tmp_path)
    remote = source(fake, cache=cache)
    fake.mutate_after_list(
        BUCKET, f"fmriprep/{FAKE_ACCESSION}-fmriprep/sourcedata/freesurfer/fsaverage/surf/rh.white"
    )
    # the listing of datasets() names the changed file; its GET then gets other content
    (record,) = remote.datasets()
    with pytest.raises(SourceError) as raised:
        remote.template_dir(record)
    assert raised.value.reason == "changed"
    assert raised.value.detail.startswith("surf/rh.white: ")
    assert_private(raised.value.detail)
    assert not list((cache.root / "retained").rglob("fsaverage"))
    assert [p.name for p in cache.area.iterdir()] == [LOCK_FILE]


def test_a_template_from_the_bucket_is_the_template_of_the_directory(
    tmp_path: Path, small_run_cohort: SyntheticCohort
) -> None:
    root = bucket_of(small_run_cohort, root=tmp_path / "bucket")
    fake = FakeS3({BUCKET: root})
    remote = source(fake, cache=cache_at(tmp_path))
    (record,) = remote.datasets()
    found = remote.template_dir(record)
    assert found is not None
    local = LocalSource(root / "fmriprep" / f"{FAKE_ACCESSION}-fmriprep")
    here = local.template_dir(local.dataset)
    assert here is not None
    built = templates.build_template(found, "adjacent", mesh="fsaverage4")
    expected = templates.build_template(here, "adjacent", mesh="fsaverage4")
    assert built.template_id == expected.template_id
    assert built.files == expected.files and built.build_stamp == expected.build_stamp
    assert MetricsConfig().mesh == "fsaverage6"  # the default mesh is a larger one


# --- config --------------------------------------------------------------------------------------


def test_config_round_trips_through_from_config(tmp_path: Path) -> None:
    dataset(tmp_path / "b")
    fake = FakeS3({BUCKET: tmp_path / "b"})
    base = new_local_config(None)
    stored = remote_dataset_config(base, FAKE_ACCESSION, subjects=("sub-0*",))
    run_config = with_changes(base, sources=(stored,))
    remote = source(fake, subjects=("sub-0*",))
    with pytest.raises(ConfigError):
        remote.config()
    remote.bind(cache=cache_at(tmp_path), pseudonym=stored.dataset)
    assert remote.config() == stored
    rebuilt = from_config(
        stored,
        run_config,
        tmp_path / "run",
        identity=FAKE_ACCESSION,
        transport=fake.transport(),
        secret=SECRET,
    )
    assert isinstance(rebuilt, OpenNeuroDerivatives)
    assert rebuilt.config() == stored
    assert (rebuilt.accession, rebuilt.globs) == (FAKE_ACCESSION, ("sub-0*",))
    assert list(rebuilt.datasets()) == list(remote.datasets())
    (record,) = rebuilt.datasets()
    assert [s.subject_label for s in rebuilt.subjects(record)] == ["sub-01", "sub-02"]
    census = source(fake, accession=None)
    assert census.config().prefix == "fmriprep/" and census.config().dataset is None


# --- the same files as a local directory ---------------------------------------------------------


def without_listing_facts(record: SubjectRecord) -> dict[FileRole, tuple[object, ...]]:
    return {r: (f.role, f.location, f.size, f.variant) for r, f in record.files.items()}


@pytest.mark.parametrize("fs_root", ["sourcedata/freesurfer", "freesurfer"])
def test_an_inventory_through_the_bucket_equals_one_through_the_directory(
    tmp_path: Path, small_run_cohort: SyntheticCohort, fs_root: str
) -> None:
    root = bucket_of(small_run_cohort, root=tmp_path / "bucket", fs_root=fs_root)
    fake = FakeS3({BUCKET: root}, page_size=7)
    remote = source(fake, cache=cache_at(tmp_path))
    local = LocalSource(root / "fmriprep" / f"{FAKE_ACCESSION}-fmriprep")
    (record,) = remote.datasets()
    here = local.dataset
    assert (record.fs_root, record.layout, record.pipeline, record.pipeline_version) == (
        here.fs_root,
        here.layout,
        here.pipeline,
        here.pipeline_version,
    )
    assert record.template_location == here.template_location
    remote_subjects = list(remote.subjects(record))
    local_subjects = list(local.subjects(here))
    assert [s.subject_label for s in remote_subjects] == [s.subject_label for s in local_subjects]
    assert len(remote_subjects) == 3
    config = InventoryConfig()
    for there, ours in zip(remote_subjects, local_subjects, strict=True):
        assert there.fs_dir == ours.fs_dir
        assert without_listing_facts(there) == without_listing_facts(ours)
        assert set(there.files) >= set(TIER_A)
        assert inventory_subject(remote, there, config) == inventory_subject(local, ours, config)
        for role in TIER_A:
            assert remote.read_bytes(there, role) == local.read_bytes(ours, role)
    only_reads(fake)
    # no request but the listings and one GET per Tier A file, and the description of an
    # fMRIPrep layout
    described = 1 if record.layout == "fmriprep" else 0
    assert len(gets(fake)) == described + 3 * len(TIER_A)


def test_no_request_but_get_and_head_and_no_private_text_in_errors(tmp_path: Path) -> None:
    dataset(tmp_path / "b", labels=("sub-01",))
    fake = FakeS3({BUCKET: tmp_path / "b"})
    remote = source(fake, cache=cache_at(tmp_path), retries=0)
    (record,) = remote.datasets()
    (first,) = remote.subjects(record)
    fake.fault("*lh.thickness", "500", times=None)
    fake.fault("*aseg.stats", "500", times=None)
    errors: list[SourceError] = []
    for call in (
        lambda: remote.read_bytes(first, FileRole.ASEG_STATS),
        lambda: remote.lease(first, [FileRole.LH_THICKNESS]).__enter__(),
    ):
        with pytest.raises(SourceError) as raised:
            call()
        errors.append(raised.value)
    assert [e.reason for e in errors] == ["unreachable", "unreachable"]
    assert errors[0].detail.startswith("stats/aseg.stats: ")
    assert errors[1].detail.startswith("surf/lh.thickness: ")
    for error in errors:
        assert_private(error.detail)
    only_reads(fake)
