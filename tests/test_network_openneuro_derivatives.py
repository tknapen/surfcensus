"""The OpenNeuroDerivatives source on the real bucket (work package RE), marked ``network``.

A small fixed set of requests, anonymous and read-only, at most four at once: the listing of
``ds000102`` sub-01 and ``ds003020`` sub-UTS01, their roles against the files that the network
fixtures keep in the pytest cache, the Tier A files of sub-01 whole and a range of its log, a
lease of two of its files, and the template next to the subjects of ``ds000102`` against the
cached copy. Run with ``-s`` to see the requests, bytes and times of each.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from surfcensus import remote, templates
from surfcensus.config import RemoteConfig
from surfcensus.remote.cache import CACHE_DIR, LOCK_FILE, Cache
from surfcensus.remote.s3 import S3Client
from surfcensus.sources.base import TIER_A, FileRole, SubjectRecord
from surfcensus.sources.local import subject_file_refs
from surfcensus.sources.openneuro_derivatives import OpenNeuroDerivatives

pytestmark = pytest.mark.network


@pytest.fixture
def client() -> Iterator[S3Client]:
    made = S3Client(remote.DERIVATIVES_BUCKET, config=RemoteConfig(max_concurrency=4))
    yield made
    made.close()


def cache_at(tmp_path: Path) -> Cache:
    return Cache(tmp_path / CACHE_DIR, max_bytes=10**9, secret=secrets.token_bytes(32))


def said(what: str, start: float) -> None:
    stats = remote.stats()
    print(
        f"{what}: {stats.requests} requests, {stats.bytes} bytes, high water "
        f"{stats.high_water}, {time.monotonic() - start:.2f} s"
    )
    assert stats.high_water <= 4


def listed(
    client: S3Client, accession: str, label: str, cache: Cache | None = None
) -> tuple[OpenNeuroDerivatives, SubjectRecord]:
    remote.reset_stats()
    start = time.monotonic()
    source = OpenNeuroDerivatives(
        client=client, cache=cache, accession=accession, subjects=(label,)
    )
    (record,) = source.datasets()
    (subject,) = source.subjects(record)
    said(f"{accession} {label}: datasets() and subjects()", start)
    print(
        f"  record: fs_root={record.fs_root} layout={record.layout} pipeline={record.pipeline} "
        f"version={record.pipeline_version} template={record.template_location}"
    )
    assert record.identity == record.accession == accession
    assert record.root == f"s3://openneuro-derivatives/fmriprep/{accession}-fmriprep/"
    assert (record.fs_root, record.layout, record.pipeline) == (
        "sourcedata/freesurfer",
        "fmriprep",
        "fmriprep",
    )
    assert subject.subject_label == label and subject.problem is None
    return source, subject


def equal_to_the_cache(subject: SubjectRecord, cached: Path) -> None:
    """Every role of the cached files has the same location, size and variant in the bucket's
    listing; the ETag and the time are those of the bucket."""
    local = subject_file_refs(cached)
    assert local
    for role, ref in local.items():
        there = subject.files[role]
        assert (there.location, there.size, there.variant) == (ref.location, ref.size, ref.variant)
        assert len(there.etag_or_mtime) == 32 and '"' not in there.etag_or_mtime
        assert there.last_modified
    extra = sorted(role.value for role in subject.files if role not in local)
    print(f"  {len(local)} roles equal to the cache; in the bucket only: {extra}")


def test_ds000102_sub_01(client: S3Client, real_subject: Path, tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    source, subject = listed(client, "ds000102", "sub-01", cache)
    assert subject.dataset.pipeline_version == "21.0.2"
    equal_to_the_cache(subject, real_subject)

    # a range of the log before it is retained: one ranged GET with If-Match
    remote.reset_stats()
    start = time.monotonic()
    head = source.read_bytes(subject, FileRole.RECON_LOG, byte_range=(0, 65535))
    said("first 64 kB of recon-all.log", start)
    assert head == (real_subject / "scripts/recon-all.log").read_bytes()[:65536]

    remote.reset_stats()
    start = time.monotonic()
    for role in TIER_A:
        data = source.read_bytes(subject, role)
        assert data == (real_subject / subject.files[role].location).read_bytes(), role
    said("Tier A whole", start)
    assert remote.stats().requests == len(TIER_A)

    remote.reset_stats()
    start = time.monotonic()
    with source.lease(subject, [FileRole.LH_THICKNESS, FileRole.LH_WHITE]) as paths:
        for role, path in paths.items():
            assert path.read_bytes() == (real_subject / subject.files[role].location).read_bytes()
    said("lease of lh.thickness and lh.white", start)
    assert sorted(path.name for path in cache.area.iterdir()) == [LOCK_FILE]


def test_ds003020_sub_uts01(client: S3Client, real_subject_fs7: Path) -> None:
    _, subject = listed(client, "ds003020", "sub-UTS01")
    assert subject.dataset.pipeline_version == "23.0.1"
    pial = subject.files[FileRole.LH_PIAL]
    assert (pial.location, pial.variant) == ("surf/lh.pial.T1", "lh.pial.T1")
    equal_to_the_cache(subject, real_subject_fs7)


def test_the_template_of_ds000102(client: S3Client, real_template: Path, tmp_path: Path) -> None:
    cache = cache_at(tmp_path)
    source = OpenNeuroDerivatives(client=client, cache=cache, accession="ds000102")
    remote.reset_stats()
    start = time.monotonic()
    (record,) = source.datasets()
    said("ds000102: datasets()", start)
    remote.reset_stats()
    start = time.monotonic()
    found = source.template_dir(record)
    said("template_dir", start)
    assert found is not None
    built = templates.build_template(found, "adjacent")
    expected = templates.build_template(real_template, "adjacent")
    print(f"  template id {built.template_id} (cache {expected.template_id})")
    assert built.template_id == expected.template_id
    assert built.files == expected.files and built.build_stamp == expected.build_stamp
    remote.reset_stats()
    assert source.template_dir(record) == found and remote.stats().requests == 0
