"""inventory/subjects: phase 2 on a catalog and sources built by hand, without a ``Run``.

The recon-all logs are the excerpts of two real subjects in ``tests/data/recon_logs/``
(``fs6_*``: ds000102 sub-01, FreeSurfer 6.0.1; ``fs7_*``: ds003020 sub-UTS01, FreeSurfer
7.3.2). One test reads the whole logs when the pytest cache has them, which is after a run
of the network tests and never in CI (docs/decisions.md, D29).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]
from synth_tree import write_tree  # pyrefly: ignore[missing-import]

from surfcensus import privacy
from surfcensus.config import (
    ConfigError,
    InventoryConfig,
    RunConfig,
    SourceConfig,
    new_census_config,
    new_local_config,
    run_key,
    stage_hashes,
)
from surfcensus.inventory.provenance import decode_text, subject_provenance
from surfcensus.inventory.subjects import (
    PROVENANCE_ROLES,
    STATS_ROLES,
    inventory_subject,
    run_inventory,
)
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport
from surfcensus.render import RenderConfig
from surfcensus.sources.base import (
    TIER_A,
    TIER_B,
    DatasetRecord,
    FileRole,
    Source,
    SourceError,
    SubjectRecord,
    input_signature,
)
from surfcensus.sources.local import LocalSource
from surfcensus.sources.patterns import POINTER_MAX_BYTES, POINTER_PROBE_BYTES, ROLE_CANDIDATES
from surfcensus.store import Catalog, PublicCatalog

DATA = Path(__file__).parent / "data" / "recon_logs"
LABELS = ("sub-01", "sub-02", "sub-03")
FS_ROOT = "sourcedata/freesurfer"
STAMPS = {
    "fs6": "freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a",
    "fs7": "freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275",
}
LOGS = {
    FileRole.RECON_LOG: "recon-all.log",
    FileRole.RECON_LOG_LH: "recon-all-lh.log",
    FileRole.RECON_LOG_RH: "recon-all-rh.log",
}
DATASET_KEY = HashKey(bytes(range(1, 33)), "dataset-key")
ACCESSION = "ds999999"
BUCKET_ROOT = f"s3://openneuro-derivatives/fmriprep/{ACCESSION}-fmriprep/"


# ---------------------------------------------------------------------------------------
# helpers


def add_tier_a(subject_dir: Path, prefix: str = "fs6") -> None:
    """The build stamp, the three logs (excerpts of a real subject) and the stats files."""
    scripts = subject_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "build-stamp.txt").write_text(STAMPS[prefix] + "\n")
    for name in LOGS.values():
        shutil.copyfile(DATA / f"{prefix}_{name}", scripts / name)
    for role in STATS_ROLES:
        path = subject_dir / ROLE_CANDIDATES[role][0]
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Measure Cortex, NumVert, Number of Vertices, 148332, unitless\n")


def add_tier_b(subject_dir: Path) -> None:
    """Files with the names of Tier B. The inventory lists them and opens none."""
    for role in TIER_B:
        path = subject_dir / ROLE_CANDIDATES[role][0]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"not read by the inventory")


def touch(path: Path) -> None:
    """Move the modification time on by a second, content unchanged."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


class Spy:
    """A source that hands everything to a ``LocalSource``, records what it is asked, and
    fails where a test says so. With ``dataset`` it poses as another dataset, for example
    a remote one."""

    def __init__(self, inner: LocalSource, dataset: DatasetRecord | None = None) -> None:
        self.inner = inner
        self.dataset = dataset if dataset is not None else inner.dataset
        self.kind = self.dataset.source
        self.only: list[Collection[str] | None] = []
        self.reads: list[tuple[str, FileRole, tuple[int, int] | None]] = []
        self.leases: list[tuple[str, tuple[FileRole, ...]]] = []
        self.fail_read: dict[tuple[str, FileRole], BaseException] = {}
        self.fail_lease: SourceError | None = None
        self.fail_listing: SourceError | None = None
        self.listed_before_failing = 0
        self.problems: dict[str, str] = {}
        self.after_listing: Callable[[], None] | None = None

    @property
    def tier_a_reads(self) -> list[tuple[str, FileRole, tuple[int, int] | None]]:
        """The reads of the build stamp and the logs; the others are the heads of small
        files, read to tell git-annex pointer files (see the test of what is read)."""
        return [read for read in self.reads if read[1] in PROVENANCE_ROLES]

    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]:
        self.only.append(only)
        yield self.dataset

    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]:
        records = list(self.inner.subjects(dataset))
        if self.after_listing is not None:
            self.after_listing()
        for index, record in enumerate(records):
            if self.fail_listing is not None and index == self.listed_before_failing:
                raise self.fail_listing
            if record.subject_label in self.problems:
                record = SubjectRecord(
                    dataset,
                    record.subject_label,
                    record.fs_dir,
                    {},
                    self.problems[record.subject_label],
                )
            yield record

    def read_bytes(
        self,
        subject: SubjectRecord,
        role: FileRole,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        self.reads.append((subject.subject_label, role, byte_range))
        failure = self.fail_read.get((subject.subject_label, role))
        if failure is not None:
            raise failure
        return self.inner.read_bytes(subject, role, byte_range=byte_range)

    def lease(
        self, subject: SubjectRecord, roles: Collection[FileRole]
    ) -> AbstractContextManager[Mapping[FileRole, Path]]:
        self.leases.append((subject.subject_label, tuple(roles)))
        if self.fail_lease is not None:
            raise self.fail_lease
        return self.inner.lease(subject, roles)

    def template_dir(self, dataset: DatasetRecord) -> Path | None:
        return self.inner.template_dir(dataset)

    def config(self) -> SourceConfig:
        if self.kind == "local":
            return self.inner.config()
        return SourceConfig(kind=self.kind, bucket="openneuro-derivatives", prefix="fmriprep/")


@dataclasses.dataclass
class Bench:
    """A tree of three subjects in the fMRIPrep layout and an empty local catalog for it.

    ``sub-01`` has Tier A (FreeSurfer 6 logs) and Tier B, ``sub-02`` Tier A (FreeSurfer 7
    logs), ``sub-03`` only what a render needs.
    """

    tree: Path
    config: RunConfig
    key: HashKey
    catalog: Catalog

    @property
    def subjects_dir(self) -> Path:
        return self.tree / FS_ROOT

    def source(self, **options: object) -> LocalSource:
        return LocalSource(self.tree, **options)

    def run(
        self, *sources: Source, config: RunConfig | None = None, **options: object
    ) -> PhaseReport:
        return run_inventory(
            self.catalog,
            list(sources) or [self.source()],
            config or self.config,
            self.key,
            **options,
        )

    def rows(self, sql: str, *params: object) -> list[dict[str, object]]:
        return [dict(row) for row in self.catalog.query(sql, params)]

    def subject(self, label: str) -> dict[str, object]:
        (row,) = self.rows("SELECT * FROM subjects WHERE subject_label = ?", label)
        return row

    def uid(self, label: str, source: Source | None = None) -> str:
        dataset = next(iter((source or self.source()).datasets()))
        return privacy.subject_uid(
            self.key, dataset.source, dataset.identity, dataset.pipeline, label
        )

    def pseudonym(self) -> str:
        dataset = self.source().dataset
        return privacy.dataset_pseudonym(self.key, dataset.source, dataset.identity)

    def dump(self) -> dict[str, list[dict[str, object]]]:
        """Everything the inventory writes, row by row."""
        return {
            "datasets": self.rows("SELECT * FROM datasets ORDER BY dataset_id"),
            "subjects": self.rows("SELECT * FROM subjects ORDER BY subject_uid"),
            "files": self.rows("SELECT * FROM files ORDER BY subject_uid, role"),
        }


@pytest.fixture
def bench(tmp_path: Path) -> Iterator[Bench]:
    tree = write_tree(tmp_path / "study", "fmriprep", LABELS)
    add_tier_a(tree / FS_ROOT / "sub-01", "fs6")
    add_tier_b(tree / FS_ROOT / "sub-01")
    add_tier_a(tree / FS_ROOT / "sub-02", "fs7")
    config = new_local_config(tree)
    key = run_key(config)
    catalog = Catalog.create(tmp_path / "run" / "catalog.sqlite", mode="local", key=key)
    yield Bench(tree, config, key, catalog)
    catalog.close()


@pytest.fixture
def upserted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Labels of the subjects that the inventory stores, in order, from now on."""
    labels: list[str] = []
    original = Catalog.upsert_subject

    def recording(
        self: Catalog,
        uid: str,
        dataset_id: int,
        record: SubjectRecord,
        *rest: object,
        **more: object,
    ) -> None:
        labels.append(record.subject_label)
        original(self, uid, dataset_id, record, *rest, **more)

    monkeypatch.setattr(Catalog, "upsert_subject", recording)
    return labels


def refuse_stat(monkeypatch: pytest.MonkeyPatch, *refused: Path) -> None:
    """``stat()`` of the given paths fails as in a directory without search permission.
    Replacing ``Path.stat`` lets the test run as root, whom modes do not stop."""
    stat = Path.stat

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self in refused:
            raise PermissionError(13, "Permission denied", str(self))
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", refuse)


# ---------------------------------------------------------------------------------------
# the first inventory: what is stored


def test_the_first_inventory_stores_the_dataset_and_every_subject(bench: Bench) -> None:
    source = bench.source()
    report = bench.run(source)

    assert (report.phase, report.done, report.skipped, report.failed) == ("inventory", 3, 0, 0)
    assert report.failures == () and report.notes == ()
    assert report.seconds > 0

    (dataset,) = bench.rows("SELECT * FROM datasets")
    assert dataset["pseudonym"] == privacy.dataset_pseudonym(
        bench.key, "local", str(bench.tree.resolve())
    )
    assert dataset["n_subjects_listed"] == 3
    assert (dataset["layout"], dataset["pipeline"], dataset["pipeline_version"]) == (
        "fmriprep",
        "fmriprep",
        "23.2.0",
    )
    assert bench.catalog.dataset_record(dataset["dataset_id"]) == source.dataset

    listed = {record.subject_label: record for record in source.subjects(source.dataset)}
    assert sorted(listed) == list(LABELS)
    stage = stage_hashes(bench.config)["inventory"]
    for label, record in listed.items():
        uid = privacy.subject_uid(bench.key, "local", str(bench.tree.resolve()), "fmriprep", label)
        row = bench.subject(label)
        assert row["subject_uid"] == uid
        assert (row["present"], row["complete"], row["problem"]) == (1, 1, None)
        assert row["fs_dir"] == f"{FS_ROOT}/{label}"
        # what every later phase starts from, in place of a new listing
        assert bench.catalog.subject_record(uid) == record
        assert bench.catalog.inventory_hash(uid) == input_signature(record.files, None, (stage,))
    assert [(row["tier_a"], row["tier_b"]) for row in map(bench.subject, LABELS)] == [
        (1, 1),
        (1, 0),
        (0, 0),
    ]


def test_provenance_of_the_logs_is_stored(bench: Bench) -> None:
    bench.run()

    fs6 = bench.subject("sub-01")
    assert (fs6["fs_build_stamp"], fs6["fs_version"], fs6["fs_major"]) == (
        STAMPS["fs6"],
        "6.0.1",
        6,
    )
    assert (fs6["n_invocations"], fs6["recon_finished"]) == (8, 1)  # 4 + 2 + 2
    assert (fs6["holes_lh"], fs6["holes_rh"]) == (41, 26)
    assert [fs6[f"trunc_{name}"] for name in ("lh_1", "lh_2", "rh_1", "rh_2")] == [39, 225, 93, 434]
    assert [
        fs6[name] for name in ("t2pial", "flairpial", "hires", "expert", "xopts", "edited")
    ] == [0] * 6
    assert (fs6["pial_variant"], fs6["participant"], fs6["longitudinal"]) == ("pial", "sub-01", 0)
    flags = json.loads(fs6["recon_flags"])
    assert "-autorecon1" in flags and flags == sorted(flags)
    assert all(
        flag.startswith("-") and "/" not in flag and "ds999999" not in flag for flag in flags
    )

    fs7 = bench.subject("sub-02")
    assert (fs7["fs_version"], fs7["fs_major"], fs7["n_invocations"]) == ("7.3.2", 7, 9)
    # fMRIPrep passes -noT2pial to FreeSurfer 7: read as False, not as a T2 refinement
    assert "-noT2pial" in json.loads(fs7["recon_flags"])
    assert (fs7["t2pial"], fs7["trunc_lh_1"], fs7["trunc_lh_2"]) == (0, 22, 230)

    # no stamp and no log: unknown, which is not the same as False or as no flags
    bare = bench.subject("sub-03")
    for name in ("fs_build_stamp", "fs_version", "fs_major", "recon_flags", "n_invocations"):
        assert bare[name] is None, name
    for name in ("recon_finished", "t2pial", "edited", "holes_lh", "trunc_rh_2"):
        assert bare[name] is None, name
    assert (bare["pial_variant"], bare["participant"]) == ("pial", "sub-03")


def digests(bench: Bench, label: str) -> dict[object, object]:
    """role -> ``files.sha256`` of the roles that the inventory reads, for one subject."""
    rows = bench.rows(
        "SELECT role, sha256 FROM files WHERE subject_uid = ? ORDER BY role", bench.uid(label)
    )
    read = {role.value for role in PROVENANCE_ROLES}
    return {row["role"]: row["sha256"] for row in rows if row["role"] in read}


def file_digest(bench: Bench, label: str, role: FileRole) -> str:
    path = bench.subjects_dir / label / ROLE_CANDIDATES[role][0]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_files_that_the_inventory_reads_whole_get_their_digest(bench: Bench) -> None:
    # recon_flags, n_invocations, holes and truncations come from these bytes; for a local
    # file size and modification time do not say which bytes they were
    bench.run()
    for label in ("sub-01", "sub-02"):
        assert digests(bench, label) == {
            role.value: file_digest(bench, label, role) for role in PROVENANCE_ROLES
        }
    assert digests(bench, "sub-03") == {}
    # nothing else was opened, so nothing else has a digest
    others = bench.rows(
        "SELECT DISTINCT sha256 FROM files WHERE role NOT IN (?, ?, ?, ?)",
        *(role.value for role in PROVENANCE_ROLES),
    )
    assert others == [{"sha256": None}]

    # a changed file is another file: the digest follows the bytes that were parsed
    stamp = bench.subjects_dir / "sub-01" / "scripts" / "build-stamp.txt"
    before = digests(bench, "sub-01")
    stamp.write_text(STAMPS["fs7"] + "\n")
    touch(stamp)
    bench.run()
    after = digests(bench, "sub-01")
    assert after["build_stamp"] == file_digest(bench, "sub-01", FileRole.BUILD_STAMP)
    assert after["build_stamp"] != before["build_stamp"]
    assert {role: after[role] for role in after if role != "build_stamp"} == {
        role: before[role] for role in before if role != "build_stamp"
    }


def test_a_file_that_was_not_read_whole_has_no_digest(bench: Bench) -> None:
    sizes = {name: (DATA / f"fs6_{name}").stat().st_size for name in LOGS.values()}
    assert sizes["recon-all-lh.log"] < 2000 < sizes["recon-all.log"]
    head = bench.config.model_copy(
        update={"inventory": InventoryConfig(logs="head", head_bytes=2000)}
    )
    spy = Spy(bench.source())
    spy.fail_read[("sub-02", FileRole.RECON_LOG_LH)] = SourceError("unreadable", "no")
    bench.run(spy, config=head)
    whole = {role.value: file_digest(bench, "sub-01", role) for role in PROVENANCE_ROLES}
    # the digest of a head would be the digest of no file
    assert digests(bench, "sub-01") == whole | {"recon_log": None}
    assert digests(bench, "sub-02")["recon_log_lh"] is None
    assert digests(bench, "sub-02")["build_stamp"] == file_digest(
        bench, "sub-02", FileRole.BUILD_STAMP
    )

    # read whole, the log gets its digest; a later head of the same file leaves it
    bench.run()
    assert digests(bench, "sub-01") == whole
    bench.run(config=head)
    assert digests(bench, "sub-01") == whole


def test_an_interruption_leaves_no_stored_subject_without_its_digests(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a subject stored without them would be skipped by every later inventory and never
    # get them, so they are written with the subject, not by a second write after it
    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(PublicCatalog, "set_file_digests", interrupt)
        try:
            bench.run()
        except KeyboardInterrupt:
            pass
    bench.run()
    for label in ("sub-01", "sub-02"):
        assert digests(bench, label) == {
            role.value: file_digest(bench, label, role) for role in PROVENANCE_ROLES
        }


def test_only_the_stamp_the_logs_and_heads_of_small_files_are_read_and_nothing_is_leased(
    bench: Bench,
) -> None:
    spy = Spy(bench.source())
    bench.run(spy)
    whole = [(label, role, None) for label in ("sub-01", "sub-02") for role in PROVENANCE_ROLES]
    # and the head of every other file small enough to be a git-annex pointer file, which
    # says that its content is not there (patterns.annex_pointer)
    source = bench.source()
    heads = [
        (record.subject_label, role, (0, POINTER_PROBE_BYTES - 1))
        for record in source.subjects(source.dataset)
        for role, ref in record.files.items()
        if role not in PROVENANCE_ROLES and ref.size <= POINTER_MAX_BYTES
    ]
    assert heads  # the light subjects are small
    assert sorted(spy.reads, key=str) == sorted(whole + heads, key=str)
    assert spy.leases == []
    assert spy.only == [None]
    # a subject that the skip test passes over is not opened at all
    spy = Spy(bench.source())
    bench.run(spy)
    assert spy.reads == []


def test_inventory_subject_parses_what_the_source_reads(bench: Bench) -> None:
    source = bench.source()
    records = {record.subject_label: record for record in source.subjects(source.dataset)}
    record = records["sub-01"]
    texts = {
        role: decode_text((bench.subjects_dir / "sub-01" / ROLE_CANDIDATES[role][0]).read_bytes())
        for role in PROVENANCE_ROLES
    }
    found = inventory_subject(source, record, InventoryConfig())
    assert found == subject_provenance(texts, record.files, "sub-01")
    assert (found.fs_version, found.n_invocations, found.finished) == ("6.0.1", 8, True)
    assert found.truncations == {"lh": (39, 225), "rh": (93, 434)}

    # a subject without Tier A files: nothing is read and nothing is known
    spy = Spy(source)
    bare = inventory_subject(spy, records["sub-03"], InventoryConfig())
    assert spy.tier_a_reads == []
    assert (bare.build_stamp, bare.fs_version, bare.n_invocations, bare.finished) == (None,) * 4
    assert (bare.t2pial, bare.edited, bare.holes) == (None, None, None)
    assert (bare.flags, bare.pial_variant, bare.participant) == ((), "pial", "sub-03")


def test_a_longitudinal_and_a_session_directory_are_ordinary_subjects(tmp_path: Path) -> None:
    labels = ["sub-01.long.base", "sub-01_ses-02"]
    tree = write_tree(tmp_path / "study", "subjects_dir", labels)
    config = new_local_config(tree)
    key = run_key(config)
    with Catalog.create(tmp_path / "run" / "catalog.sqlite", mode="local", key=key) as catalog:
        report = run_inventory(catalog, [LocalSource(tree)], config, key)
        rows = catalog.query(
            "SELECT subject_label, participant, longitudinal FROM subjects ORDER BY subject_label"
        )
    assert report.done == 2
    assert [tuple(row) for row in rows] == [
        ("sub-01.long.base", "sub-01", 1),
        ("sub-01_ses-02", "sub-01", 0),
    ]


# ---------------------------------------------------------------------------------------
# the second inventory: the skip test


def test_a_second_inventory_reads_no_file_and_changes_nothing(
    bench: Bench, upserted: list[str]
) -> None:
    first = Spy(bench.source())
    assert bench.run(first).done == 3
    assert len(first.tier_a_reads) == 8 and upserted == list(LABELS)
    before = bench.dump()

    second = Spy(bench.source())
    report = bench.run(second)
    assert (report.done, report.skipped, report.failed, report.notes) == (0, 3, 0, ())
    assert second.reads == [] and second.leases == []
    assert upserted == list(LABELS)  # nothing stored again
    assert bench.dump() == before


def test_force_reads_every_subject_again(bench: Bench, upserted: list[str]) -> None:
    bench.run()
    spy = Spy(bench.source())
    report = bench.run(spy, force=True)
    assert (report.done, report.skipped) == (3, 0)
    assert len(spy.tier_a_reads) == 8
    assert upserted == [*LABELS, *LABELS]


def test_a_touched_file_inventories_that_subject_only(bench: Bench, upserted: list[str]) -> None:
    bench.run()
    uids = {label: bench.subject(label)["subject_uid"] for label in LABELS}
    volume = bench.subjects_dir / "sub-02" / "mri" / "T1.mgz"
    touch(volume)
    del upserted[:]

    spy = Spy(bench.source())
    report = bench.run(spy)
    assert (report.done, report.skipped) == (1, 2)
    assert upserted == ["sub-02"]
    assert {label for label, _, _ in spy.reads} == {"sub-02"}
    assert {label: bench.subject(label)["subject_uid"] for label in LABELS} == uids
    (t1,) = bench.rows(
        "SELECT etag_or_mtime FROM files WHERE subject_uid = ? AND role = 't1'", uids["sub-02"]
    )
    assert t1["etag_or_mtime"] == str(volume.stat().st_mtime_ns)

    assert bench.run().skipped == 3


def test_a_changed_log_is_read_again(bench: Bench) -> None:
    bench.run()
    stamp = bench.subjects_dir / "sub-01" / "scripts" / "build-stamp.txt"
    stamp.write_text(STAMPS["fs7"] + "\n")
    touch(stamp)  # a rewrite within the clock's resolution would not show otherwise
    report = bench.run()
    assert (report.done, report.skipped) == (1, 2)
    assert bench.subject("sub-01")["fs_version"] == "7.3.2"


def test_a_pial_link_that_leads_to_another_variant_of_equal_size_and_time_is_read_again(
    bench: Bench,
) -> None:
    # FreeSurfer 7: lh.pial is a link to lh.pial.T1, or to lh.pial.T2 after a T2 refinement.
    # The variants of one subject have one size (one topology), and a copy that keeps the
    # modification time (cp -p) leaves location, size and time of the role as they were.
    surf = bench.subjects_dir / "sub-01" / "surf"
    for hemi in ("lh", "rh"):
        os.replace(surf / f"{hemi}.pial", surf / f"{hemi}.pial.T1")
        (surf / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T1")
    bench.run()
    row = bench.subject("sub-01")
    assert (row["pial_variant"], row["t2pial"]) == ("pial.T1", 0)
    uid = str(row["subject_uid"])
    # as after a render, which stores the digest of every file it read
    was_read = hashlib.sha256((surf / "lh.pial.T1").read_bytes()).hexdigest()
    bench.catalog.public().set_file_digests(uid, {FileRole.LH_PIAL: was_read})
    listed = bench.rows(
        "SELECT location, size, etag_or_mtime FROM files "
        "WHERE subject_uid = ? AND role LIKE ? ORDER BY role",
        uid,
        "%pial",
    )

    for hemi in ("lh", "rh"):
        first, second = surf / f"{hemi}.pial.T1", surf / f"{hemi}.pial.T2"
        second.write_bytes(first.read_bytes()[::-1])  # other content of the same size
        os.utime(second, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))
        (surf / f"{hemi}.pial").unlink()
        (surf / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T2")

    report = bench.run()
    assert (report.done, report.skipped) == (1, 2)
    row = bench.subject("sub-01")
    assert (row["pial_variant"], row["t2pial"]) == ("pial.T2", 1)
    stored = bench.rows(
        "SELECT location, size, etag_or_mtime, variant, sha256 FROM files "
        "WHERE subject_uid = ? AND role LIKE ? ORDER BY role",
        uid,
        "%pial",
    )
    # nothing but the variant tells the listing apart, and the digest was of the other file
    assert [{k: r[k] for k in ("location", "size", "etag_or_mtime")} for r in stored] == listed
    assert [(r["variant"], r["sha256"]) for r in stored] == [
        ("lh.pial.T2", None),
        ("rh.pial.T2", None),
    ]
    assert bench.run().skipped == 3


def test_a_changed_inventory_setting_reads_everything_again_and_another_setting_nothing(
    bench: Bench, upserted: list[str]
) -> None:
    bench.run()
    other_render = bench.config.model_copy(update={"render": RenderConfig(panel_px=64)})
    assert bench.run(config=other_render).skipped == 3

    head = bench.config.model_copy(update={"inventory": InventoryConfig(logs="head")})
    assert stage_hashes(head)["inventory"] != stage_hashes(bench.config)["inventory"]
    report = bench.run(config=head)
    assert (report.done, report.skipped) == (3, 0)
    assert upserted == [*LABELS, *LABELS]
    assert bench.run(config=head).skipped == 3


def test_an_interrupted_inventory_is_completed_by_the_next(bench: Bench) -> None:
    spy = Spy(bench.source())
    spy.fail_read[("sub-02", FileRole.BUILD_STAMP)] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        bench.run(spy)
    assert [row["subject_label"] for row in bench.rows("SELECT subject_label FROM subjects")] == [
        "sub-01"
    ]

    report = bench.run()
    assert (report.done, report.skipped) == (2, 1)
    assert len(bench.rows("SELECT 1 FROM subjects WHERE present = 1")) == 3


# ---------------------------------------------------------------------------------------
# head mode


def test_head_mode_reads_the_first_bytes_of_a_long_log_and_short_files_whole(
    bench: Bench,
) -> None:
    sizes = {name: (DATA / f"fs6_{name}").stat().st_size for name in LOGS.values()}
    assert sizes["recon-all-lh.log"] < 2000 < sizes["recon-all.log"]
    config = bench.config.model_copy(
        update={"inventory": InventoryConfig(logs="head", head_bytes=2000)}
    )
    spy = Spy(bench.source())
    bench.run(spy, config=config)
    assert [
        (role, byte_range) for label, role, byte_range in spy.tier_a_reads if label == "sub-01"
    ] == [
        (FileRole.BUILD_STAMP, None),
        (FileRole.RECON_LOG, (0, 1999)),
        (FileRole.RECON_LOG_LH, None),
        (FileRole.RECON_LOG_RH, None),
    ]

    row = bench.subject("sub-01")
    assert row["fs_version"] == "6.0.1"
    # the head cannot say how the run ended, nor that a flag was never given
    assert (row["recon_finished"], row["t2pial"], row["edited"]) == (None, None, None)
    head = decode_text((DATA / "fs6_recon-all.log").read_bytes()[:2000])
    texts = {
        role: decode_text((DATA / f"fs6_{name}").read_bytes()) for role, name in LOGS.items()
    } | {FileRole.RECON_LOG: head, FileRole.BUILD_STAMP: STAMPS["fs6"] + "\n"}
    expected = subject_provenance(texts, {}, "sub-01", partial=True)
    assert 0 < row["n_invocations"] == expected.n_invocations < 8
    assert json.loads(row["recon_flags"]) == list(expected.flags)


def test_a_log_that_fits_in_the_head_is_not_partial(bench: Bench) -> None:
    longest = max((DATA / f"fs6_{name}").stat().st_size for name in LOGS.values())
    source = bench.source()
    (record,) = [r for r in source.subjects(source.dataset) if r.subject_label == "sub-01"]
    spy = Spy(source)
    found = inventory_subject(spy, record, InventoryConfig(logs="head", head_bytes=longest))
    assert all(byte_range is None for _, _, byte_range in spy.tier_a_reads)
    assert found == inventory_subject(source, record, InventoryConfig())
    assert found.finished is True and found.t2pial is False

    shorter = inventory_subject(
        source, record, InventoryConfig(logs="head", head_bytes=longest - 1)
    )
    assert shorter.finished is None and shorter.t2pial is None


# ---------------------------------------------------------------------------------------
# incomplete, unreadable and vanished subjects


def test_incomplete_subjects_are_stored_and_counted_not_dropped(tmp_path: Path) -> None:
    tree = write_tree(
        tmp_path / "study", "subjects_dir", ["sub-01", "sub-02"], incomplete=["sub-02"]
    )
    for folder in ("mri", "surf"):  # what a recon-all run that crashed at once leaves
        (tree / "sub-03" / folder).mkdir(parents=True)
    config = new_local_config(tree)
    key = run_key(config)
    with Catalog.create(tmp_path / "run" / "catalog.sqlite", mode="local", key=key) as catalog:
        report = run_inventory(catalog, [LocalSource(tree)], config, key)
        rows = {row["subject_label"]: dict(row) for row in catalog.query("SELECT * FROM subjects")}
        n_files = {
            label: len(
                catalog.query("SELECT 1 FROM files WHERE subject_uid = ?", (row["subject_uid"],))
            )
            for label, row in rows.items()
        }
        (n_listed,) = catalog.query("SELECT n_subjects_listed FROM datasets")[0]
    assert (report.done, report.failed, report.failures) == (3, 0, ())
    assert n_listed == 3
    assert {label: row["complete"] for label, row in rows.items()} == {
        "sub-01": 1,
        "sub-02": 0,
        "sub-03": 0,
    }
    assert rows["sub-02"]["missing_roles"].split(",")[0] == "lh_pial"
    assert {"t1", "lh_white", "rh_pial"} <= set(rows["sub-03"]["missing_roles"].split(","))
    assert all(row["present"] == 1 and row["problem"] is None for row in rows.values())
    assert n_files["sub-03"] == 0 < n_files["sub-02"] < n_files["sub-01"]


def test_a_directory_that_cannot_be_listed_is_a_failed_unit_until_it_can(
    bench: Bench, monkeypatch: pytest.MonkeyPatch, upserted: list[str]
) -> None:
    uid = bench.uid("sub-02")
    with monkeypatch.context() as patch:
        refuse_stat(patch, bench.subjects_dir / "sub-02" / "mri")
        first = bench.run()
        assert (first.done, first.skipped, first.failed) == (3, 0, 1)
        assert first.failures == ((uid, "unreadable"),)
        row = bench.subject("sub-02")
        assert (row["problem"], row["complete"], row["present"]) == (
            "unreadable: Permission denied",
            0,
            1,
        )
        assert bench.rows("SELECT 1 FROM files WHERE subject_uid = ?", uid) == []
        failures = bench.catalog.failures("inventory")
        assert failures.to_dict("records") == [
            {
                "unit": uid,
                "phase": "inventory",
                "reason": "unreadable",
                "detail": "Permission denied",
            }
        ]

        # not retried, still failed: the same count, and so the same exit code
        second = bench.run()
        assert (second.done, second.skipped, second.failed) == (0, 3, 1)
        assert second.failures == first.failures
        assert upserted == list(LABELS)

    third = bench.run()
    assert (third.done, third.skipped, third.failed, third.failures) == (1, 2, 0, ())
    row = bench.subject("sub-02")
    assert (row["problem"], row["complete"], row["fs_version"]) == (None, 1, "7.3.2")
    assert bench.catalog.failures("inventory").empty


def test_a_changed_problem_is_stored(bench: Bench) -> None:
    spy = Spy(bench.source())
    spy.problems["sub-03"] = "unreadable: Permission denied"
    assert bench.run(spy).failed == 1
    spy.problems["sub-03"] = "unreadable: Input/output error"
    report = bench.run(spy)
    assert (report.done, report.skipped, report.failed) == (1, 2, 1)
    assert bench.subject("sub-03")["problem"] == "unreadable: Input/output error"


def test_a_vanished_subject_is_marked_absent_and_comes_back(
    bench: Bench, tmp_path: Path, upserted: list[str]
) -> None:
    bench.run()
    uid = bench.subject("sub-03")["subject_uid"]
    files = bench.rows("SELECT * FROM files WHERE subject_uid = ?", uid)
    shutil.move(bench.subjects_dir / "sub-03", tmp_path / "elsewhere")

    report = bench.run()
    assert (report.done, report.skipped, report.failed) == (0, 2, 0)
    assert len(report.notes) == 1
    assert report.notes[0].startswith(f"{bench.pseudonym()}: 1 subject(s) are no longer listed")
    assert "sub-03" not in report.notes[0]
    assert {label: bench.subject(label)["present"] for label in LABELS} == {
        "sub-01": 1,
        "sub-02": 1,
        "sub-03": 0,
    }
    # the row and its files stay: renders and ratings hang on the uid
    assert bench.rows("SELECT * FROM files WHERE subject_uid = ?", uid) == files
    assert bench.rows("SELECT n_subjects_listed FROM datasets") == [{"n_subjects_listed": 2}]
    assert bench.run().notes == ()  # said once

    shutil.move(tmp_path / "elsewhere", bench.subjects_dir / "sub-03")
    back = bench.run()
    assert (back.done, back.skipped, back.notes) == (0, 3, ())
    assert bench.subject("sub-03")["present"] == 1
    assert upserted == list(LABELS)  # unchanged files: present again without a new read


def test_globs_define_the_subject_set(bench: Bench) -> None:
    bench.run()
    report = bench.run(bench.source(subjects=("sub-0[12]",)))
    assert (report.done, report.skipped) == (0, 2)
    assert [
        row["subject_label"] for row in bench.rows("SELECT * FROM subjects WHERE present = 0")
    ] == ["sub-03"]
    assert bench.run(bench.source(subjects=("sub-01", "*3"))).skipped == 2
    assert [
        row["subject_label"] for row in bench.rows("SELECT * FROM subjects WHERE present = 0")
    ] == ["sub-02"]


def test_a_glob_that_matches_nothing_fails_the_dataset_and_marks_nobody_absent(
    bench: Bench, caplog: pytest.LogCaptureFixture
) -> None:
    bench.run()
    before = bench.dump()
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        report = bench.run(bench.source(subjects=("nobody-*",)))
    pseudonym = bench.pseudonym()
    assert (report.done, report.skipped, report.failed) == (0, 0, 1)
    assert report.failures == ((f"dataset:{pseudonym}", "not found"),)
    (note,) = report.notes
    assert note.startswith(f"{pseudonym}: the subjects could not be listed (not found: ")
    assert "'nobody-*'" in note and "sub-01, sub-02, sub-03" in note
    # the text of the exception is scrubbed: no absolute path, here or in the log
    assert str(bench.tree.resolve()) not in note
    assert str(bench.tree.resolve()) not in caplog.text and "'nobody-*'" in caplog.text
    assert bench.dump() == before


def test_a_listing_that_stops_half_way_keeps_what_it_stored_and_marks_nobody_absent(
    bench: Bench,
) -> None:
    spy = Spy(bench.source())
    spy.fail_listing = SourceError("unreachable", "the listing was cut off")
    spy.listed_before_failing = 2
    report = bench.run(spy)
    pseudonym = bench.pseudonym()
    assert (report.done, report.skipped, report.failed) == (2, 0, 1)
    assert report.failures == ((f"dataset:{pseudonym}", "unreachable"),)
    assert report.notes == (
        f"{pseudonym}: the subjects could not be listed (unreachable: the listing was cut off)",
    )
    assert [row["subject_label"] for row in bench.rows("SELECT * FROM subjects ORDER BY 3")] == [
        "sub-01",
        "sub-02",
    ]
    assert bench.rows("SELECT n_subjects_listed FROM datasets") == [{"n_subjects_listed": None}]

    assert bench.run().done == 1
    # a full catalog, then half a listing: sub-03 is not marked absent
    again = bench.run(spy)
    assert (again.done, again.skipped, again.failed) == (0, 2, 1)
    assert bench.subject("sub-03")["present"] == 1
    assert bench.rows("SELECT n_subjects_listed FROM datasets") == [{"n_subjects_listed": 3}]


# ---------------------------------------------------------------------------------------
# Tier A files that cannot be read


def test_an_unreadable_log_gives_empty_fields_is_no_failure_and_is_read_again(
    bench: Bench, caplog: pytest.LogCaptureFixture, upserted: list[str]
) -> None:
    spy = Spy(bench.source())
    spy.fail_read[("sub-01", FileRole.RECON_LOG)] = SourceError(
        "unreadable", "scripts/recon-all.log: Permission denied"
    )
    with caplog.at_level(logging.WARNING, logger="surfcensus"):
        report = bench.run(spy)
    assert (report.done, report.skipped, report.failed, report.failures) == (3, 0, 0, ())
    (note,) = report.notes
    assert note.startswith(f"{bench.pseudonym()}: 1 subject(s) with a build stamp, log or stats")
    assert "unreadable: 1" in note and "sub-01" not in note
    uid = bench.uid("sub-01")
    assert (
        f"{uid}: recon_log was not read: unreadable: scripts/recon-all.log: Permission denied"
        in caplog.text
    )
    assert "sub-01" not in caplog.text  # the log names subjects by uid

    row = bench.subject("sub-01")
    # the stamp and the two other logs were read; what the missing log decides is unknown
    assert (row["fs_version"], row["n_invocations"]) == ("6.0.1", 4)
    assert (row["recon_finished"], row["t2pial"], row["holes_lh"]) == (None, None, None)
    assert (row["trunc_lh_1"], row["trunc_rh_2"]) == (39, 434)
    record = bench.catalog.subject_record(uid)
    plain = input_signature(record.files, None, (stage_hashes(bench.config)["inventory"],))
    assert row["inventory_hash"] != plain

    # still unreadable: that subject again, and only that one
    del upserted[:]
    again = bench.run(spy)
    assert (again.done, again.skipped, again.failed) == (1, 2, 0)
    assert again.notes == report.notes and upserted == ["sub-01"]

    spy.fail_read.clear()
    healed = bench.run(spy)
    assert (healed.done, healed.skipped, healed.notes) == (1, 2, ())
    row = bench.subject("sub-01")
    assert (row["n_invocations"], row["recon_finished"], row["t2pial"], row["holes_lh"]) == (
        8,
        1,
        0,
        41,
    )
    assert row["inventory_hash"] == plain
    assert bench.run(spy).skipped == 3


def test_inventory_subject_does_not_raise_for_an_unreadable_file(bench: Bench) -> None:
    source = bench.source()
    (record,) = [r for r in source.subjects(source.dataset) if r.subject_label == "sub-01"]
    spy = Spy(source)
    spy.fail_read[("sub-01", FileRole.BUILD_STAMP)] = SourceError("unreadable", "no")
    spy.fail_read[("sub-01", FileRole.RECON_LOG_RH)] = SourceError("unreachable", "no")
    found = inventory_subject(spy, record, InventoryConfig())
    whole = inventory_subject(source, record, InventoryConfig())
    # the version then comes from the stamp line of recon-all.log
    assert (found.fs_version, found.build_stamp) == ("6.0.1", STAMPS["fs6"])
    assert found.truncations == {"lh": (39, 225), "rh": None}
    assert (found.n_invocations, found.t2pial) == (6, None)
    assert (whole.n_invocations, whole.t2pial) == (8, False)


def test_a_file_that_changes_between_listing_and_reading_is_picked_up_by_the_next_inventory(
    bench: Bench,
) -> None:
    log = bench.subjects_dir / "sub-01" / "scripts" / "recon-all.log"
    spy = Spy(bench.source())

    def append() -> None:
        with log.open("a") as stream:
            stream.write("one more line\n")

    spy.after_listing = append
    report = bench.run(spy)  # the real LocalSource refuses the file: SourceError("changed")
    assert (report.done, report.failed) == (3, 0)
    assert len(report.notes) == 1 and "changed: 1" in report.notes[0]
    assert bench.subject("sub-01")["n_invocations"] == 4

    report = bench.run()
    assert (report.done, report.skipped, report.notes) == (1, 2, ())
    row = bench.subject("sub-01")
    assert (row["n_invocations"], row["recon_finished"]) == (8, 1)
    (stored,) = bench.rows(
        "SELECT size FROM files WHERE subject_uid = ? AND role = 'recon_log'", row["subject_uid"]
    )
    assert stored["size"] == log.stat().st_size


# ---------------------------------------------------------------------------------------
# field strength


def test_the_field_strength_of_a_source_is_written_to_its_listed_subjects(
    bench: Bench, upserted: list[str]
) -> None:
    bench.run(bench.source(field_strength=7.0, subjects=("sub-0[12]",)))
    assert [bench.subject(label)["field_strength"] for label in ("sub-01", "sub-02")] == [7.0, 7.0]
    assert len(bench.rows("SELECT 1 FROM subjects")) == 2

    # not set: what is stored stays, as do the other columns of phase 3
    with bench.catalog.write() as conn:
        conn.execute("UPDATE subjects SET age = 31.5, sex = 'F', manufacturer = 'Philips'")
    bench.run(force=True)
    assert [bench.subject(label)["field_strength"] for label in LABELS] == [7.0, 7.0, None]
    row = bench.subject("sub-01")
    assert (row["age"], row["sex"], row["manufacturer"]) == (31.5, "F", "Philips")

    # changed: written also to subjects that are skipped
    del upserted[:]
    report = bench.run(bench.source(field_strength=3.0))
    assert (report.done, report.skipped) == (0, 3) and upserted == []
    assert [bench.subject(label)["field_strength"] for label in LABELS] == [3.0, 3.0, 3.0]


# ---------------------------------------------------------------------------------------
# several datasets, the --dataset filter, keys


def _other_tree(tmp_path: Path) -> LocalSource:
    return LocalSource(write_tree(tmp_path / "other", "subjects_dir", ["sub-x", "sub-y"]))


def test_two_sources_are_two_datasets_and_the_filter_takes_pseudonyms(
    bench: Bench, tmp_path: Path
) -> None:
    first, second = bench.source(), _other_tree(tmp_path)
    other = privacy.dataset_pseudonym(bench.key, "local", second.dataset.identity)
    assert other != bench.pseudonym()

    report = bench.run(first, second, datasets=[other])
    assert (report.done, report.failed) == (2, 0)
    assert [row["pseudonym"] for row in bench.rows("SELECT pseudonym FROM datasets")] == [other]

    report = bench.run(first, second)
    assert (report.done, report.skipped) == (3, 2)
    assert sorted(
        row["pseudonym"] for row in bench.rows("SELECT pseudonym FROM datasets")
    ) == sorted([other, bench.pseudonym()])
    counts = bench.rows(
        "SELECT d.pseudonym, COUNT(*) AS n FROM subjects s JOIN datasets d USING (dataset_id) "
        "GROUP BY d.pseudonym"
    )
    assert {row["pseudonym"]: row["n"] for row in counts} == {other: 2, bench.pseudonym(): 3}
    assert bench.subject("sub-x")["subject_uid"] == bench.uid("sub-x", second)

    report = bench.run(first, second, datasets=[bench.pseudonym(), other], force=True)
    assert report.done == 5


def test_a_dataset_that_no_source_has_is_a_failed_unit(bench: Bench) -> None:
    report = bench.run(datasets=["D-0000abcd", bench.pseudonym()])
    assert (report.done, report.failed) == (3, 1)
    assert report.failures == (("dataset:D-0000abcd", "not found"),)
    assert report.notes == ("D-0000abcd: no source of this run has this dataset",)
    assert len(bench.rows("SELECT 1 FROM datasets")) == 1


def test_failures_and_notes_are_ordered_by_unit_and_pseudonym_not_by_source_order(
    bench: Bench, tmp_path: Path
) -> None:
    first = Spy(bench.source())
    second = Spy(_other_tree(tmp_path))
    for spy in (first, second):
        spy.fail_listing = SourceError("unreachable", "no answer")
    forward = bench.run(first, second)
    backward = bench.run(second, first)
    pseudonyms = sorted(
        privacy.dataset_pseudonym(bench.key, "local", spy.dataset.identity)
        for spy in (first, second)
    )
    assert forward.failures == tuple((f"dataset:{p}", "unreachable") for p in pseudonyms)
    assert [note.split(":")[0] for note in forward.notes] == pseudonyms
    assert (backward.failures, backward.notes) == (forward.failures, forward.notes)


def test_two_sources_on_one_dataset_are_refused(bench: Bench) -> None:
    with pytest.raises(ConfigError) as caught:
        bench.run(bench.source(), bench.source(subjects=("sub-01",)))
    assert caught.value.reason == "invalid"
    assert bench.pseudonym() in caught.value.detail


def test_another_key_than_the_catalogs_is_refused_before_anything_is_written(
    bench: Bench,
) -> None:
    other = HashKey(bytes(32), "local-salt")
    with pytest.raises(ConfigError) as caught:
        run_inventory(bench.catalog, [bench.source()], bench.config, other)
    assert caught.value.reason == "key"
    assert bench.dump() == {"datasets": [], "subjects": [], "files": []}


def test_a_stored_pseudonym_that_the_key_does_not_give_is_refused(bench: Bench) -> None:
    bench.run()
    with bench.catalog.write() as conn:
        conn.execute("UPDATE datasets SET pseudonym = 'D-00000000'")
    with pytest.raises(ConfigError) as caught:
        bench.run()
    assert caught.value.reason == "key"


def test_two_subjects_with_one_uid_stop_the_inventory(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(privacy, "subject_uid", lambda *parts: "0123456789abcdef")
    with pytest.raises(ConfigError) as caught:
        bench.run()
    assert caught.value.reason == "invalid"
    assert "two subjects with one identifier" in caught.value.detail
    assert len(bench.rows("SELECT 1 FROM subjects")) == 1


# ---------------------------------------------------------------------------------------
# a census run: a remote source and what must not leak


@dataclasses.dataclass
class Census:
    catalog: Catalog
    config: RunConfig
    remote: Spy
    local: LocalSource

    def run(self, *sources: Source, **options: object) -> PhaseReport:
        return run_inventory(
            self.catalog,
            list(sources) or [self.remote, self.local],
            self.config,
            DATASET_KEY,
            **options,
        )

    @property
    def pseudonym(self) -> str:
        return privacy.dataset_pseudonym(DATASET_KEY, "openneuro-derivatives", ACCESSION)


@pytest.fixture
def census(tmp_path: Path) -> Iterator[Census]:
    """A census catalog, a source that poses as a bucket dataset ``ds999999`` with subjects
    ``sub-CANARY7`` and ``sub-02``, and a local source under a path with ``CANARYROOT``."""
    bucket = write_tree(tmp_path / "bucket", "fmriprep", ["sub-CANARY7", "sub-02"])
    for label in ("sub-CANARY7", "sub-02"):
        add_tier_a(bucket / FS_ROOT / label)
    posing = dataclasses.replace(
        LocalSource(bucket).dataset,
        source="openneuro-derivatives",
        identity=ACCESSION,
        accession=ACCESSION,
        root=BUCKET_ROOT,
    )
    local = LocalSource(write_tree(tmp_path / "CANARYROOT" / "pilot", "subjects_dir", ["sub-7T01"]))
    catalog = Catalog.create(tmp_path / "run" / "catalog.sqlite", mode="census", key=DATASET_KEY)
    yield Census(catalog, new_census_config(DATASET_KEY), Spy(LocalSource(bucket), posing), local)
    catalog.close()


def test_a_remote_source_has_its_stats_files_fetched_once(census: Census) -> None:
    report = census.run()
    assert (report.done, report.skipped, report.failed) == (3, 0, 0)
    # one lease per subject for the cache to keep; the local source of the run is not asked
    assert census.remote.leases == [
        ("sub-02", STATS_ROLES),
        ("sub-CANARY7", STATS_ROLES),
    ]
    assert len(census.remote.reads) == 8

    (remote,) = [
        dict(row)
        for row in census.catalog.query("SELECT * FROM datasets WHERE accession IS NOT NULL")
    ]
    assert (remote["pseudonym"], remote["source"], remote["identity"], remote["root"]) == (
        census.pseudonym,
        "openneuro-derivatives",
        ACCESSION,
        BUCKET_ROOT,
    )
    uid = privacy.subject_uid(
        DATASET_KEY, "openneuro-derivatives", ACCESSION, "fmriprep", "sub-CANARY7"
    )
    (label,) = census.catalog.query(
        "SELECT subject_label FROM subjects WHERE subject_uid = ?", (uid,)
    )
    assert label["subject_label"] == "sub-CANARY7"
    public = census.catalog.public().subjects()
    assert len(public) == 3 and public["subject_label"].isna().all()

    del census.remote.leases[:], census.remote.reads[:]
    again = census.run()
    assert (again.done, again.skipped) == (0, 3)
    assert census.remote.leases == [] and census.remote.reads == []


def test_a_subject_without_stats_files_is_not_leased(census: Census, tmp_path: Path) -> None:
    for role in STATS_ROLES:
        (tmp_path / "bucket" / FS_ROOT / "sub-02" / ROLE_CANDIDATES[role][0]).unlink()
    (tmp_path / "bucket" / FS_ROOT / "sub-CANARY7" / "stats" / "aseg.stats").unlink()
    census.run(census.remote)
    assert census.remote.leases == [
        ("sub-CANARY7", (FileRole.LH_STATS, FileRole.RH_STATS)),
    ]


def test_stats_that_cannot_be_fetched_are_fetched_by_the_next_inventory(census: Census) -> None:
    census.remote.fail_lease = SourceError("unreachable", f"GET {BUCKET_ROOT}x: 503")
    report = census.run(census.remote)
    assert (report.done, report.failed) == (2, 0)
    (note,) = report.notes
    assert note.startswith(f"{census.pseudonym}: 2 subject(s)") and "unreachable: 2" in note
    rows = census.catalog.query("SELECT fs_version FROM subjects")
    assert [row["fs_version"] for row in rows] == ["6.0.1", "6.0.1"]  # what was read is kept

    census.remote.fail_lease = None
    del census.remote.leases[:]
    healed = census.run(census.remote)
    assert (healed.done, healed.notes) == (2, ())
    assert len(census.remote.leases) == 2
    assert census.run(census.remote).skipped == 2


def test_the_filter_reaches_a_remote_source_as_accessions(census: Census) -> None:
    census.run()
    assert census.remote.only == [None]
    report = census.run(datasets=[census.pseudonym])
    assert census.remote.only[-1] == {ACCESSION}
    assert (report.done, report.skipped, report.failed) == (0, 2, 0)

    local = privacy.dataset_pseudonym(DATASET_KEY, "local", census.local.dataset.identity)
    report = census.run(datasets=[local])
    assert census.remote.only[-1] == set()  # a local dataset has no accession
    assert (report.done, report.skipped, report.failed) == (0, 1, 0)


def test_a_pseudonym_lengthened_at_discovery_is_kept(census: Census) -> None:
    long = privacy.dataset_pseudonym(DATASET_KEY, "openneuro-derivatives", ACCESSION, length=12)
    census.catalog.upsert_dataset(census.remote.dataset, long, None)
    report = census.run(census.remote, datasets=[long])
    assert (report.done, report.failed) == (2, 0)
    assert [row["pseudonym"] for row in census.catalog.query("SELECT pseudonym FROM datasets")] == [
        long
    ]


def test_nothing_in_the_report_or_the_log_of_a_census_run_names_a_dataset_or_a_subject(
    census: Census, caplog: pytest.LogCaptureFixture
) -> None:
    canaries = (ACCESSION, "CANARY7", "CANARYROOT", "sub-7T01", "sub-02")
    with caplog.at_level(logging.INFO, logger="surfcensus"):
        census.run()
        census.remote.fail_read[("sub-CANARY7", FileRole.RECON_LOG)] = SourceError(
            "unreachable",
            f"GET {BUCKET_ROOT}{FS_ROOT}/sub-CANARY7/scripts/recon-all.log of {ACCESSION}: 503",
        )
        reread = census.run(census.remote, force=True)
        census.remote.fail_listing = SourceError(
            "unreachable", f"listing {BUCKET_ROOT}{FS_ROOT}/sub-CANARY7/ of {ACCESSION}: 503"
        )
        shutil.rmtree(census.local.dataset.root)  # "<root> is no longer a directory"
        failed = census.run()

    assert reread.failed == 0 and len(reread.notes) == 1
    local = privacy.dataset_pseudonym(DATASET_KEY, "local", census.local.dataset.identity)
    assert failed.failures == tuple(
        sorted([(f"dataset:{local}", "not found"), (f"dataset:{census.pseudonym}", "unreachable")])
    )
    # the accession in the exception's text comes out as the pseudonym
    assert any(f"of {census.pseudonym}: 503" in note for note in failed.notes)
    assert "was not read: unreachable:" in caplog.text
    assert "the subjects could not be listed" in caplog.text
    said = "\n".join([caplog.text, repr(reread), repr(failed)])
    for canary in canaries:
        assert canary not in said, canary


# ---------------------------------------------------------------------------------------
# the source is read, never written


def test_the_source_tree_is_unchanged_by_every_kind_of_inventory(
    bench: Bench, tmp_path: Path
) -> None:
    before = snapshot_tree(bench.tree)
    bench.run()
    bench.run()
    bench.run(force=True)
    bench.run(
        bench.source(subjects=("sub-01",), field_strength=3.0),
        config=bench.config.model_copy(
            update={"inventory": InventoryConfig(logs="head", head_bytes=100)}
        ),
    )
    assert snapshot_tree(bench.tree) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run", "study"]
    assert [path.name for path in (tmp_path / "run").iterdir()] == ["catalog.sqlite"]


# ---------------------------------------------------------------------------------------
# the whole logs of the two real subjects, when the pytest cache has them


@pytest.mark.parametrize(
    ("prefix", "version", "n_main", "n_all", "truncations", "holes"),
    [
        ("fs6", "6.0.1", 4, 8, [39, 225, 93, 434], (41, 26)),
        ("fs7", "7.3.2", 5, 9, [22, 230, 23, 325], (4, 15)),
    ],
)
def test_inventory_of_a_subject_with_the_whole_logs_of_a_real_one(
    cached_subjects: Path,
    bench: Bench,
    prefix: str,
    version: str,
    n_main: int,
    n_all: int,
    truncations: list[int],
    holes: tuple[int, int],
) -> None:
    subject = {"fs6": "ds000102-sub-01", "fs7": "ds003020-sub-UTS01"}[prefix]
    logs = [cached_subjects / subject / "scripts" / name for name in LOGS.values()]
    if not all(path.is_file() for path in logs):
        pytest.skip("the whole logs are not in the pytest cache (the network tests fetch them)")
    scripts = bench.subjects_dir / "sub-03" / "scripts"
    scripts.mkdir()
    for path in logs:
        shutil.copyfile(path, scripts / path.name)

    bench.run()
    row = bench.subject("sub-03")
    # no build-stamp.txt here: the version is that of the stamp line of recon-all.log
    assert (row["fs_version"], row["fs_major"], row["fs_build_stamp"]) == (
        version,
        int(version[0]),
        STAMPS[prefix],
    )
    assert (row["n_invocations"], row["recon_finished"]) == (n_all, 1)
    assert (row["holes_lh"], row["holes_rh"]) == holes
    assert [row[f"trunc_{name}"] for name in ("lh_1", "lh_2", "rh_1", "rh_2")] == truncations
    assert [
        row[name] for name in ("t2pial", "flairpial", "hires", "expert", "xopts", "edited")
    ] == [0] * 6
    flags = json.loads(row["recon_flags"])
    assert flags and all("/" not in flag and "ds00" not in flag for flag in flags)

    # the first 64 kB, what Tier A fetched before the logs were read whole: fewer
    # invocations, and no verdict
    head = bench.config.model_copy(update={"inventory": InventoryConfig(logs="head")})
    spy = Spy(bench.source())
    bench.run(spy, config=head)
    assert (FileRole.RECON_LOG, (0, 65535)) in [
        (role, r) for label, role, r in spy.reads if label == "sub-03"
    ]
    row = bench.subject("sub-03")
    assert row["fs_version"] == version
    assert row["n_invocations"] < n_all and row["recon_finished"] is None
    assert n_main > 2  # the head of recon-all.log holds two of them


def test_the_roles_read_and_fetched_are_tier_a() -> None:
    assert set(PROVENANCE_ROLES) | set(STATS_ROLES) == set(TIER_A)
    assert not set(PROVENANCE_ROLES) & set(STATS_ROLES)


def test_files_without_content_are_named_without_their_size_in_a_census_run(
    census: Census,
) -> None:
    from synth_annex import unfetch  # pyrefly: ignore[missing-import]

    subject = Path(census.local.subjects_dir) / "sub-7T01"
    unfetch(census.local.subjects_dir, [subject / "mri" / "T1.mgz"])
    unfetch(census.local.subjects_dir, [subject / "surf" / "lh.white"], "pointer")
    census.run(census.local)
    (row,) = [
        dict(row)
        for row in census.catalog.query(
            "SELECT problem, complete, missing_roles FROM subjects WHERE subject_label = ?",
            ("sub-7T01",),
        )
    ]
    # the sizes of local files stay in no public column of a census run (decision D20)
    assert row["problem"] == (
        "not fetched: 2 files are git-annex links and pointer files whose content is not "
        "here: mri/T1.mgz, surf/lh.white"
    )
    assert row["complete"] == 0 and row["missing_roles"].startswith("t1,lh_white")
    assert census.catalog.failures().empty  # no failure of the inventory
