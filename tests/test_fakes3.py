"""The fake bucket of ``tests/fakes3.py``: what it answers, in the process and over a socket.

The remote layer is tested against this fake, so the fake is tested on its own: the listing
and its pages, ranges, preconditions, HEAD, the refusal of other methods and of signed
requests, the faults, an object that changes after it was listed, and the two ways in.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest
from fakes3 import (  # pyrefly: ignore[missing-import]
    FAKE_ACCESSION,
    NAMESPACE,
    FakeS3,
    bucket_of,
    key_of,
)
from synth_tierb import SyntheticCohort  # pyrefly: ignore[missing-import]

BUCKET = "openneuro-derivatives"
NS = {"s3": NAMESPACE}


def write(root: Path, key: str, content: bytes) -> Path:
    path = root.joinpath(*key.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "bucket"
    for index in range(25):
        write(root, f"fmriprep/ds999999-fmriprep/sub-{index:02d}/surf/lh.white", b"w" * index)
    write(root, "fmriprep/ds999999-fmriprep/dataset_description.json", b"{}")
    write(root, "fmriprep/ds999998-fmriprep/x.txt", b"x")
    write(root, "top.txt", b"0123456789")
    return root


def client(fake: FakeS3) -> httpx.Client:
    return httpx.Client(transport=fake.transport(), trust_env=False)


def list_all(http: httpx.Client, url: str, **params: str) -> tuple[list[str], list[str], int]:
    """Keys, common prefixes and the number of pages of a ListObjectsV2 walk."""
    keys: list[str] = []
    prefixes: list[str] = []
    pages = 0
    token = None
    while True:
        query = {"list-type": "2", **params}
        if token is not None:
            query["continuation-token"] = token
        answer = http.get(url, params=query)
        assert answer.status_code == 200
        pages += 1
        tree = ET.fromstring(answer.content)
        keys += [e.text or "" for e in tree.findall("s3:Contents/s3:Key", NS)]
        prefixes += [e.text or "" for e in tree.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        count = int(tree.findtext("s3:KeyCount", "0", NS))
        assert count == len(tree.findall("s3:Contents", NS)) + len(
            tree.findall("s3:CommonPrefixes", NS)
        )
        if tree.findtext("s3:IsTruncated", None, NS) != "true":
            return keys, prefixes, pages
        token = tree.findtext("s3:NextContinuationToken", None, NS)
        assert token


VIRTUAL = f"https://{BUCKET}.s3.amazonaws.com/"
PATH = f"https://s3.amazonaws.com/{BUCKET}/"


@pytest.mark.parametrize("page_size", [1000, 7, 2])
@pytest.mark.parametrize("url", [VIRTUAL, PATH])
def test_a_listing_comes_in_pages_in_binary_order(tree: Path, page_size: int, url: str) -> None:
    fake = FakeS3({BUCKET: tree}, page_size=page_size)
    with client(fake) as http:
        keys, prefixes, pages = list_all(http, url, prefix="fmriprep/ds999999-fmriprep/")
    assert prefixes == []
    assert keys == sorted(keys, key=str.encode) and len(keys) == 26
    assert pages == -(-26 // page_size)
    assert fake.methods == {"LIST"}


def test_a_listing_with_a_delimiter_gives_common_prefixes(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree}, page_size=2)
    with client(fake) as http:
        keys, prefixes, _ = list_all(http, VIRTUAL, prefix="fmriprep/", delimiter="/")
        assert keys == []
        assert prefixes == ["fmriprep/ds999998-fmriprep/", "fmriprep/ds999999-fmriprep/"]
        keys, prefixes, _ = list_all(http, VIRTUAL, delimiter="/")
        assert (keys, prefixes) == (["top.txt"], ["fmriprep/"])


def test_a_listing_names_size_etag_and_time(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree})
    with client(fake) as http:
        answer = http.get(VIRTUAL, params={"list-type": "2", "prefix": "top"})
    contents = ET.fromstring(answer.content).find("s3:Contents", NS)
    assert contents is not None
    assert contents.findtext("s3:Size", None, NS) == "10"
    assert contents.findtext("s3:ETag", None, NS) == f'"{hashlib.md5(b"0123456789").hexdigest()}"'
    stamp = contents.findtext("s3:LastModified", "", NS)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", stamp)


def test_get_head_range_and_if_match(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree}, version_ids=True)
    etag = f'"{hashlib.md5(b"0123456789").hexdigest()}"'
    with client(fake) as http:
        whole = http.get(VIRTUAL + "top.txt")
        assert (whole.status_code, whole.content) == (200, b"0123456789")
        assert whole.headers["etag"] == etag and whole.headers["last-modified"].endswith("GMT")
        assert whole.headers["x-amz-version-id"].startswith("v")
        head = http.head(PATH + "top.txt")
        assert (head.status_code, head.content, head.headers["content-length"]) == (200, b"", "10")
        part = http.get(VIRTUAL + "top.txt", headers={"Range": "bytes=2-5"})
        assert (part.status_code, part.content) == (206, b"2345")
        assert part.headers["content-range"] == "bytes 2-5/10"
        clamped = http.get(VIRTUAL + "top.txt", headers={"Range": "bytes=8-99"})
        assert (clamped.content, clamped.headers["content-range"]) == (b"89", "bytes 8-9/10")
        suffix = http.get(VIRTUAL + "top.txt", headers={"Range": "bytes=-3"})
        assert suffix.content == b"789"
        beyond = http.get(VIRTUAL + "top.txt", headers={"Range": "bytes=10-12"})
        assert (beyond.status_code, beyond.headers["content-range"]) == (416, "bytes */10")
        assert http.get(VIRTUAL + "top.txt", headers={"If-Match": etag}).status_code == 200
        wrong = http.get(VIRTUAL + "top.txt", headers={"If-Match": '"0"'})
        assert wrong.status_code == 412 and b"PreconditionFailed" in wrong.content
        assert http.get(VIRTUAL + "no/such/key").status_code == 404
        assert http.get("https://other.s3.amazonaws.com/top.txt").status_code == 404


def test_other_methods_and_signed_requests_are_refused_and_logged(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree})
    with client(fake) as http:
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            assert http.request(method, VIRTUAL + "top.txt").status_code == 405
        signed = http.get(VIRTUAL + "top.txt", headers={"Authorization": "AWS4-HMAC-SHA256 x"})
        assert signed.status_code == 403
        presigned = http.get(VIRTUAL + "top.txt", params={"X-Amz-Signature": "abc"})
        assert presigned.status_code == 403
    assert len(fake.signed) == 2 and all(entry.status == 403 for entry in fake.signed)
    assert fake.methods == {"POST", "PUT", "DELETE", "PATCH", "OPTIONS", "GET"}


def test_faults_hit_their_keys_as_often_as_asked(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree})
    fake.fault("top.txt", "500", times=2)
    fake.fault("*lh.white", "slowdown", times=None, retry_after=1.5)
    fake.fault("fmriprep/ds999998*", "503", methods=("LIST",))
    with client(fake) as http:
        statuses = [http.get(VIRTUAL + "top.txt").status_code for _ in range(3)]
        assert statuses == [500, 500, 200]
        slow = http.get(VIRTUAL + "fmriprep/ds999999-fmriprep/sub-01/surf/lh.white")
        assert slow.status_code == 503 and b"<Code>SlowDown</Code>" in slow.content
        assert slow.headers["retry-after"] == "1.5"
        listing = http.get(VIRTUAL, params={"list-type": "2", "prefix": "fmriprep/ds999998"})
        assert listing.status_code == 503
        assert http.get(VIRTUAL + "fmriprep/ds999998-fmriprep/x.txt").status_code == 200


def test_a_truncated_body_and_a_reset_fail_the_read(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree})
    fake.fault("top.txt", "truncate")
    fake.fault("fmriprep/ds999998-fmriprep/x.txt", "reset")
    with client(fake) as http:
        with pytest.raises(httpx.RemoteProtocolError):
            http.get(VIRTUAL + "top.txt")
        with pytest.raises(httpx.ReadError):
            http.get(VIRTUAL + "fmriprep/ds999998-fmriprep/x.txt")
        assert http.get(VIRTUAL + "top.txt").content == b"0123456789"


def test_an_object_changes_once_a_listing_named_it(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree})
    fake.mutate_after_list(BUCKET, "top.txt")
    before = (tree / "top.txt").stat()
    with client(fake) as http:
        assert http.get(VIRTUAL + "top.txt").content == b"0123456789"  # not listed yet
        listing = http.get(VIRTUAL, params={"list-type": "2", "prefix": "top"})
        listed = ET.fromstring(listing.content).findtext("s3:Contents/s3:ETag", "", NS)
        old = f'"{hashlib.md5(b"0123456789").hexdigest()}"'
        assert listed == old  # the listing saw the object as it was
        after = http.get(VIRTUAL + "top.txt")
        assert after.content == b"0123456789\n" and after.headers["etag"] != listed
        assert http.get(VIRTUAL + "top.txt", headers={"If-Match": old}).status_code == 412
        again = http.get(VIRTUAL, params={"list-type": "2", "prefix": "top"})
        assert (
            ET.fromstring(again.content).findtext("s3:Contents/s3:ETag", "", NS)
            == (after.headers["etag"])
        )
    assert (tree / "top.txt").stat() == before


def test_concurrent_requests_are_counted(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree}, latency_s=0.2)
    with client(fake) as http:
        threads = [threading.Thread(target=http.get, args=(VIRTUAL + "top.txt",)) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert fake.high_water == 5 and len(fake.log) == 5


def test_a_cohort_as_a_bucket(tmp_path: Path) -> None:
    cohort = tmp_path / "subjects"
    write(cohort, "sub-01/surf/lh.white", b"white")
    write(cohort, "fsaverage/surf/lh.sphere.reg", b"sphere")
    root = bucket_of(cohort, root=tmp_path / "bucket")
    fake = FakeS3({BUCKET: root})
    keys = fake.keys(BUCKET)
    prefix = f"fmriprep/{FAKE_ACCESSION}-fmriprep/"
    assert keys == [
        prefix + "dataset_description.json",
        prefix + "sourcedata/freesurfer/fsaverage/surf/lh.sphere.reg",
        prefix + "sourcedata/freesurfer/sub-01/surf/lh.white",
    ]
    assert key_of(FAKE_ACCESSION, "sub-01", "surf/lh.white") == keys[2]
    with client(fake) as http:
        assert http.get(VIRTUAL + keys[2]).content == b"white"  # through the link


def files_of(root: Path) -> list[tuple[str, int, int]]:
    return sorted(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    )


def test_over_a_socket(tree: Path) -> None:
    fake = FakeS3({BUCKET: tree}, page_size=7)
    fake.fault("top.txt", "truncate", times=1)
    fake.fault("fmriprep/ds999998-fmriprep/x.txt", "reset", times=1)
    before = files_of(tree)
    with fake.serve() as endpoint:
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+", endpoint)
        with httpx.Client(trust_env=False) as http:
            url = f"{endpoint}/{BUCKET}/"
            keys, _, pages = list_all(http, url, prefix="fmriprep/ds999999-fmriprep/")
            assert len(keys) == 26 and pages == 4
            with pytest.raises(httpx.RemoteProtocolError):
                http.get(url + "top.txt")
            with pytest.raises(httpx.TransportError):
                http.get(url + "fmriprep/ds999998-fmriprep/x.txt")
            assert http.get(url + "top.txt").content == b"0123456789"
            head = http.head(url + "top.txt")
            assert (head.content, head.headers["content-length"]) == (b"", "10")
            part = http.get(url + "top.txt", headers={"Range": "bytes=0-0"})
            assert (part.status_code, part.content) == (206, b"0")
            assert http.post(url + "top.txt", content=b"x").status_code == 405
            virtual = http.get(
                url[: -len(BUCKET) - 1] + "top.txt", headers={"Host": f"{BUCKET}.s3.amazonaws.com"}
            )
            assert virtual.content == b"0123456789"
    assert files_of(tree) == before
    assert "POST" in fake.methods and fake.signed == []


def test_the_fake_writes_nothing(tree: Path) -> None:
    before = files_of(tree)
    fake = FakeS3({BUCKET: tree})
    fake.mutate_after_list(BUCKET, "top.txt", b"other")
    with client(fake) as http:
        list_all(http, VIRTUAL)
        http.get(VIRTUAL + "top.txt")
    assert files_of(tree) == before
    assert not any(name.startswith(".") for name in os.listdir(tree))


def test_a_synthetic_cohort_as_a_bucket(tmp_path: Path, small_run_cohort: SyntheticCohort) -> None:
    root = bucket_of(small_run_cohort, root=tmp_path / "bucket", fs_root="freesurfer")
    fake = FakeS3({BUCKET: root})
    keys = fake.keys(BUCKET)
    prefix = f"fmriprep/{FAKE_ACCESSION}-fmriprep/freesurfer/"
    labels = {key[len(prefix) :].split("/")[0] for key in keys if key.startswith(prefix)}
    assert labels == {"fsaverage", *(s.path.name for s in small_run_cohort.subjects)}
    first = small_run_cohort.subjects[0].path
    key = key_of(FAKE_ACCESSION, first.name, "surf/lh.white", fs_root="freesurfer")
    with client(fake) as http:
        assert http.get(VIRTUAL + key).content == (first / "surf" / "lh.white").read_bytes()
