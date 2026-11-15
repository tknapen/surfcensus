"""The S3 client against the real OpenNeuroDerivatives bucket, one request each.

Inside one subject of ``ds000102`` only: the listing of the subject (one page), a ranged GET of
the head of its ``recon-all.log``, a GET with an ``If-Match`` that cannot match, a HEAD, and
one whole GET of ``surf/lh.white`` with its MD5 checked. About five requests in all, through
``default_transport()`` and the proxy of the environment, anonymous and read-only. What each
answer says is printed (run with ``-s``), since the plan lists these facts as not verified.
"""

from __future__ import annotations

import hashlib
import re
import time

import pytest

from surfcensus import remote
from surfcensus.config import RemoteConfig
from surfcensus.remote.s3 import S3Client, S3Object
from surfcensus.sources.base import SourceError

pytestmark = pytest.mark.network

PREFIX = "fmriprep/ds000102-fmriprep/sourcedata/freesurfer/sub-01/"


@pytest.fixture(scope="module")
def client() -> S3Client:
    # one retry at most, so that a fault costs at most one request more
    return S3Client(
        remote.DERIVATIVES_BUCKET, config=RemoteConfig(max_concurrency=4, max_retries=1)
    )


@pytest.fixture(scope="module")
def listing(client: S3Client) -> dict[str, S3Object]:
    before = remote.stats().requests
    objects = [o for o in client.list(PREFIX) if isinstance(o, S3Object)]
    requests = remote.stats().requests - before
    print(f"\nlisting: {len(objects)} keys in {requests} request(s)")
    print(f"listing: ETag forms {sorted({_form(o.etag) for o in objects})}")
    return {o.key[len(PREFIX) :]: o for o in objects}


def _form(etag: str) -> str:
    if re.fullmatch(r"[0-9a-f]{32}", etag):
        return "md5"
    if re.fullmatch(r"[0-9a-f]{32}-\d+", etag):
        return "multipart"
    return "other"


def test_the_listing_of_one_subject_is_one_page_of_309_keys(listing: dict[str, S3Object]) -> None:
    assert len(listing) == 309
    assert "mri/T1.mgz" in listing and "surf/lh.white" in listing


def test_a_ranged_get_of_the_log_head_is_206(
    client: S3Client, listing: dict[str, S3Object]
) -> None:
    log = listing["scripts/recon-all.log"]
    body, head = client.get(
        log.key, byte_range=(0, 65_535), if_match=log.etag, location="scripts/recon-all.log"
    )
    print(f"\nrange: {len(body)} bytes, whole size {head.size} (listed {log.size})")
    assert len(body) == 65_536 and head.size == log.size


def test_a_wrong_if_match_is_refused(client: S3Client, listing: dict[str, S3Object]) -> None:
    stamp = listing["scripts/build-stamp.txt"]
    with pytest.raises(SourceError) as caught:
        client.get(stamp.key, if_match="0" * 32, location="scripts/build-stamp.txt")
    print(f"\nwrong If-Match: {caught.value.reason}: {caught.value.detail}")
    assert caught.value.reason == "changed"
    assert "412" in caught.value.detail


def test_a_head_says_whether_a_version_id_is_sent(
    client: S3Client, listing: dict[str, S3Object]
) -> None:
    stats = listing["stats/lh.aparc.stats"]
    head = client.head(stats.key, location="stats/lh.aparc.stats")
    print(
        f"\nHEAD: size {head.size}, ETag form {_form(head.etag)}, equal to the listed ETag "
        f"{head.etag == stats.etag}, version id sent {head.version_id is not None}, "
        f"Last-Modified {head.last_modified!r}"
    )
    assert head.size == stats.size and head.etag == stats.etag


def test_a_whole_get_of_lh_white_is_checked_against_its_md5(
    client: S3Client, listing: dict[str, S3Object]
) -> None:
    white = listing["surf/lh.white"]
    start = time.monotonic()
    body, head = client.get(white.key, if_match=white.etag, location="surf/lh.white")
    seconds = time.monotonic() - start
    print(
        f"\nlh.white: {len(body)} bytes in {seconds:.2f} s, ETag form {_form(white.etag)} "
        f"(an MD5 ETag is checked against the body by get); ETag is the MD5 of the body "
        f"{hashlib.md5(body).hexdigest() == white.etag}"
    )
    assert len(body) == white.size == head.size
