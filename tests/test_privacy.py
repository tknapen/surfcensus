"""Tests of surfcensus/privacy.py (milestone 2 part): keyed identifiers and scrubbing."""

from __future__ import annotations

import dataclasses
import pickle
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from surfcensus import privacy
from surfcensus.config import new_local_config, run_key
from surfcensus.privacy import (
    HashKey,
    cache_name,
    dataset_pseudonym,
    local_salt_from_hex,
    new_local_salt,
    render_key,
    scrub,
    scrub_with,
    subject_uid,
)
from surfcensus.sources.base import DatasetRecord
from surfcensus.store import Catalog

SECRET = bytes(range(32))
KEY = HashKey(SECRET, "dataset-key")
OTHER = HashKey(bytes(range(1, 33)), "dataset-key")
# The values below were computed with `openssl dgst -sha256 -mac HMAC` on the messages that
# the docstrings of privacy.py give, with the key 00 01 .. 1f, not with this package.
GOLDEN_PSEUDONYM = "D-bd75ed90"
GOLDEN_PSEUDONYM_12 = "D-bd75ed90ac55"
GOLDEN_UID = "13ce4e5d1ba667fa"
GOLDEN_RENDER_KEY = "1dba075d1cf89335"
GOLDEN_CACHE_NAME = "f17d1f2193153c0e9325e2af1ce98cd0"
GOLDEN_FINGERPRINT = "630dcd2966c4"


# --- HashKey ------------------------------------------------------------------------------


def test_repr_of_a_key_shows_kind_and_fingerprint_and_no_secret() -> None:
    assert KEY.fingerprint == GOLDEN_FINGERPRINT
    for text in (repr(KEY), str(KEY), f"{KEY}", f"{KEY!r}", repr([KEY]), repr({"key": KEY})):
        assert "dataset-key" in text and GOLDEN_FINGERPRINT in text
        assert SECRET.hex() not in text and repr(SECRET) not in text and "secret" not in text


def test_a_key_is_32_bytes_and_frozen() -> None:
    for length in (0, 31, 33, 64):
        with pytest.raises(ValueError, match="32 bytes"):
            HashKey(b"x" * length, "local-salt")
    with pytest.raises(dataclasses.FrozenInstanceError):
        KEY.secret = b"y" * 32  # pyrefly: ignore[read-only]
    assert KEY == HashKey(bytes(range(32)), "dataset-key")
    assert KEY != HashKey(bytes(range(32)), "local-salt")
    assert pickle.loads(pickle.dumps(KEY)) == KEY  # a key crosses into worker processes


def test_new_local_salts_are_random_and_of_their_kind() -> None:
    one, two = new_local_salt(), new_local_salt()
    assert one.kind == two.kind == "local-salt"
    assert len(one.secret) == 32 and one.secret != two.secret
    assert one.fingerprint != two.fingerprint


def test_the_salt_round_trips_through_hex_and_through_a_config(tmp_path) -> None:
    salt = new_local_salt()
    assert local_salt_from_hex(salt.secret.hex()) == salt
    assert local_salt_from_hex(salt.secret.hex().upper()) == salt
    config = new_local_config(tmp_path)
    key = run_key(config)
    assert key.kind == "local-salt"
    assert key.secret.hex() == config.identity.local_salt
    assert key.fingerprint == config.identity.key_fingerprint
    for bad in ("", "abc", "g" * 64, "0" * 63, "0" * 65, " " + "0" * 63):
        with pytest.raises(ValueError, match="64 hex"):
            local_salt_from_hex(bad)


# --- pseudonyms and subject identifiers --------------------------------------------------


def test_the_pseudonym_is_the_formula_of_decision_d10() -> None:
    assert dataset_pseudonym(KEY, "openneuro-derivatives", "ds000102") == GOLDEN_PSEUDONYM
    assert (
        dataset_pseudonym(KEY, "openneuro-derivatives", "ds000102", length=12)
        == GOLDEN_PSEUDONYM_12
    )
    assert re.fullmatch(r"D-[0-9a-f]{8}", GOLDEN_PSEUDONYM)
    different = {
        dataset_pseudonym(KEY, "openneuro-derivatives", "ds000102"),
        dataset_pseudonym(OTHER, "openneuro-derivatives", "ds000102"),
        dataset_pseudonym(KEY, "openneuro-embedded", "ds000102"),
        dataset_pseudonym(KEY, "openneuro-derivatives", "ds000103"),
        dataset_pseudonym(KEY, "local", "/data/SUBJECTS_DIR"),
    }
    assert len(different) == 5
    for length in (0, 65):
        with pytest.raises(ValueError, match="1 to 64"):
            dataset_pseudonym(KEY, "local", "x", length=length)


def test_the_subject_uid_is_16_hex_and_differs_by_every_part_and_by_the_key() -> None:
    parts = ("openneuro-derivatives", "ds000102", "fmriprep", "sub-01")
    uid = subject_uid(KEY, *parts)
    assert uid == GOLDEN_UID
    assert re.fullmatch(r"[0-9a-f]{16}", uid)
    assert subject_uid(KEY, *parts) == uid
    others = [
        subject_uid(OTHER, *parts),
        subject_uid(KEY, "openneuro-embedded", "ds000102", "fmriprep", "sub-01"),
        subject_uid(KEY, "openneuro-derivatives", "ds000103", "fmriprep", "sub-01"),
        subject_uid(KEY, "openneuro-derivatives", "ds000102", "freesurfer", "sub-01"),
        subject_uid(KEY, "openneuro-derivatives", "ds000102", "fmriprep", "sub-02"),
        # the parts cannot be shifted into one another
        subject_uid(KEY, "openneuro-derivatives", "ds000102fmriprep", "", "sub-01"),
        subject_uid(KEY, "openneuro-derivatives", "ds000102", "fmriprepsub-01", ""),
    ]
    assert len({uid, *others}) == len(others) + 1
    # and it is not the pseudonym's hash under another name
    assert uid[:8] != GOLDEN_PSEUDONYM[2:]


# --- render keys and cache names ---------------------------------------------------------


def test_the_render_key_changes_with_the_secret_the_tile_hash_and_a_digest_only() -> None:
    digests = {"t1": "aa", "lh_pial": "bb"}
    key = render_key(SECRET, "0123456789ab", digests)
    assert key == GOLDEN_RENDER_KEY
    assert re.fullmatch(r"[0-9a-f]{16}", key)
    # not with the order of the mapping
    assert render_key(SECRET, "0123456789ab", {"lh_pial": "bb", "t1": "aa"}) == key
    changed = [
        render_key(bytes(range(1, 33)), "0123456789ab", digests),
        render_key(SECRET, "0123456789ac", digests),
        render_key(SECRET, "0123456789ab", {"t1": "ab", "lh_pial": "bb"}),
        render_key(SECRET, "0123456789ab", {"t1": "aa"}),
        render_key(SECRET, "0123456789ab", {"t1": "aa", "rh_pial": "bb"}),
        render_key(SECRET, "0123456789ab", {"t1": "bb", "lh_pial": "aa"}),
    ]
    assert len({key, *changed}) == len(changed) + 1


def test_the_cache_name_changes_with_the_secret_bucket_key_and_etag_only() -> None:
    name = cache_name(SECRET, "openneuro-derivatives", "fmriprep/ds000102-fmriprep/x", "abc")
    assert name == GOLDEN_CACHE_NAME
    assert re.fullmatch(r"[0-9a-f]{32}", name)
    changed = [
        cache_name(
            bytes(range(1, 33)), "openneuro-derivatives", "fmriprep/ds000102-fmriprep/x", "abc"
        ),
        cache_name(SECRET, "openneuro.org", "fmriprep/ds000102-fmriprep/x", "abc"),
        cache_name(SECRET, "openneuro-derivatives", "fmriprep/ds000102-fmriprep/y", "abc"),
        cache_name(SECRET, "openneuro-derivatives", "fmriprep/ds000102-fmriprep/x", "abd"),
        cache_name(SECRET, "openneuro-derivatives", "fmriprep/ds000102-fmriprep/xabc", ""),
    ]
    assert len({name, *changed}) == len(changed) + 1
    # a render key and a cache name over the same words are different hashes
    assert render_key(SECRET, "abc", {})[:16] != cache_name(SECRET, "abc", "", "")[:16]


# --- scrub --------------------------------------------------------------------------------

PSEUDONYMS = {"ds000102": "D-3f9a1c07"}
SUBJECT_PATH = "/data/ds000102-fmriprep/sourcedata/freesurfer/sub-01"


def test_scrub_reduces_the_two_real_messages_to_the_file_name() -> None:
    # the texts nibabel gave for an empty and for a missing .mgz in the milestone 1 review
    empty = f"Empty file: '{SUBJECT_PATH}/mri/T1.mgz'"
    missing = f"No such file or no access: '{SUBJECT_PATH}/surf/lh.pial'"
    assert scrub(empty) == "Empty file: 'T1.mgz'"
    assert scrub(missing) == "No such file or no access: 'lh.pial'"
    assert scrub(empty, pseudonyms=PSEUDONYMS) == "Empty file: 'T1.mgz'"
    assert scrub(missing, pseudonyms=PSEUDONYMS) == "No such file or no access: 'lh.pial'"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cannot read /data/study/sub-01/mri/T1.mgz: empty", "cannot read T1.mgz: empty"),
        ('"/data/My Study/sub 01/surf/lh.white" is short', '"lh.white" is short'),
        ("PosixPath('/a/b/c.mgz') and (/d/e, /f/g)", "PosixPath('c.mgz') and (e, g)"),
        (r"bad header in C:\Users\me\data\sub-01\mri\T1.mgz", "bad header in T1.mgz"),
        (
            r'bad header in "C:\My Data\sub-01\mri\T1.mgz" (3 bytes)',
            'bad header in "T1.mgz" (3 bytes)',
        ),
        (r"\\server\share\study\surf\lh.white is short", "lh.white is short"),
        ("c:/data/study/lh.pial", "lh.pial"),
        (
            "GET https://openneuro-derivatives.s3.amazonaws.com/fmriprep/x/mri/T1.mgz gave 503",
            "GET T1.mgz gave 503",
        ),
        ("listing s3://openneuro-derivatives/fmriprep/ failed", "listing fmriprep failed"),
        ("a directory /data/study/ and the root /", "a directory study and the root /"),
    ],
)
def test_scrub_replaces_an_absolute_path_by_its_last_component(text: str, expected: str) -> None:
    assert scrub(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "white and pial of surf/lh.pial differ: 42 and 43 vertices",
        "0 of 12 subjects complete, and/or 5 / 3 = 1.67 (N/A)",
        "relative ./mri/T1.mgz, ../surf/lh.white and ~/data stay",
        "render failure: box outside the volume",
        "x = -30.0 to +30.0 mm; ratio 0.811; 'quoted words'",
        "",
    ],
)
def test_scrub_leaves_text_without_paths_or_identifiers_unchanged(text: str) -> None:
    assert scrub(text) == text
    assert scrub(text, pseudonyms=PSEUDONYMS, roots=["/data/study"]) == text


def test_scrub_turns_an_accession_into_its_pseudonym() -> None:
    text = "ds000102 has 26 subjects; ds000102-fmriprep and ds003020 were listed"
    assert scrub(text, pseudonyms=PSEUDONYMS) == (
        "D-3f9a1c07 has 26 subjects; D-3f9a1c07-fmriprep and D-? were listed"
    )
    # without a mapping every accession is masked
    assert scrub(text) == "D-? has 26 subjects; D-?-fmriprep and D-? were listed"
    # only whole accessions: six digits, not part of a longer word or number
    for other in ("ds00010", "ds0001021", "xds000102", "Ds000102", "ads000102"):
        assert scrub(other, pseudonyms=PSEUDONYMS) == other


def test_scrub_masks_subject_labels_in_a_census_run_and_keeps_them_in_a_local_run() -> None:
    text = "sub-01 lacks lh_pial; sub-UTS01_ses-2 and sub-01.long.base are complete"
    assert scrub(text) == text
    assert scrub(text, pseudonyms={}) == (
        "sub-? lacks lh_pial; sub-?_ses-2 and sub-?.long.base are complete"
    )
    assert scrub(f"unreadable: {SUBJECT_PATH}", pseudonyms=PSEUDONYMS) == "unreadable: sub-?"
    assert scrub(f"unreadable: {SUBJECT_PATH}") == "unreadable: sub-01"
    # not inside another word
    assert scrub("resub-01 and sub- alone", pseudonyms={}) == "resub-01 and sub- alone"


def test_scrub_replaces_roots_and_what_lies_under_them() -> None:
    roots = ["/data/study", "/data/study/derivatives", "my-identity"]
    assert scrub("the root /data/study is gone", roots=roots) == "the root <root> is gone"
    assert scrub("in /data/study/derivatives now", roots=roots) == "in <root> now"
    assert scrub("/data/study/sub-01/mri/T1.mgz is empty", roots=roots) == "T1.mgz is empty"
    assert scrub("dataset my-identity moved", roots=roots) == "dataset <root> moved"
    assert scrub("nothing here", roots=["", "/data/study"]) == "nothing here"
    # a root that is an accession is left to the pseudonym, which says more
    assert scrub("ds000102 moved", pseudonyms=PSEUDONYMS, roots=["ds000102"]) == "D-3f9a1c07 moved"
    assert scrub("ds003020 moved", roots=["ds003020"]) == "D-? moved"
    remote = "s3://openneuro-derivatives/fmriprep/ds000102-fmriprep/"
    assert scrub(f"{remote}sub-01/mri/T1.mgz changed", pseudonyms={}, roots=[remote]) == (
        "T1.mgz changed"
    )
    assert scrub(f"{remote} lists nothing", pseudonyms={}, roots=[remote]) == (
        "<root> lists nothing"
    )


def test_scrub_collapses_white_space() -> None:
    assert scrub("  two\tlines\n  of   text \r\n") == "two lines of text"


def test_scrub_hides_the_canaries_of_a_census_failure() -> None:
    text = (
        "Traceback: ImageFileError(\"Empty file: '/tmp/CANARYROOT/freesurfer/sub-CANARY7/mri/"
        "T1.mgz'\") while reading s3://bucket/fmriprep/ds999999-fmriprep/sourcedata/freesurfer/"
        "sub-CANARY7/mri/T1.mgz for ds999999 sub-CANARY7"
    )
    cleaned = scrub(text, pseudonyms={"ds999999": "D-0badcafe"}, roots=["/tmp/CANARYROOT"])
    for canary in ("ds999999", "CANARY7", "CANARYROOT", "bucket", "/tmp"):
        assert canary not in cleaned
    assert "T1.mgz" in cleaned and "D-0badcafe" in cleaned and "sub-?" in cleaned


@given(
    st.text(),
    st.one_of(st.none(), st.dictionaries(st.from_regex(r"ds\d{6}", fullmatch=True), st.text())),
    st.lists(st.text(), max_size=3),
)
def test_scrub_never_raises_and_leaves_single_spaces_and_no_unmapped_accession(
    text: str, pseudonyms: dict[str, str] | None, roots: list[str]
) -> None:
    cleaned = scrub(text, pseudonyms=pseudonyms, roots=roots)
    assert cleaned == " ".join(cleaned.split())
    if not pseudonyms:
        assert not privacy._ACCESSION.search(cleaned)


@given(st.lists(st.from_regex(r"[A-Za-z0-9_.-]{1,12}", fullmatch=True), min_size=2, max_size=6))
def test_scrub_keeps_only_the_last_component_of_any_posix_path(parts: list[str]) -> None:
    path = "/" + "/".join(parts)
    cleaned = scrub(f"cannot open '{path}' today")
    assert cleaned == f"cannot open '{parts[-1]}' today"
    assert scrub(f"cannot open {path}") == f"cannot open {parts[-1]}"


# --- scrub with what a catalog holds ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["local", "census"])
def test_scrub_with_a_catalog_uses_its_roots_and_in_a_census_run_its_pseudonyms(
    tmp_path, mode: str
) -> None:
    key = KEY if mode == "census" else new_local_salt()
    remote = DatasetRecord(
        source="openneuro-derivatives",
        identity="ds999999",
        accession="ds999999",
        root="s3://openneuro-derivatives/fmriprep/ds999999-fmriprep/",
        fs_root="sourcedata/freesurfer",
        layout="fmriprep",
        pipeline="fmriprep",
    )
    text = "ds999999: cannot list /data/CANARYROOT for sub-CANARY7"
    with Catalog.create(tmp_path / "catalog.sqlite", mode=mode, key=key) as catalog:
        before = scrub_with(catalog, text)
        local = DatasetRecord(
            source="local",
            identity="/data/CANARYROOT",
            accession=None,
            root="/data/CANARYROOT",
            fs_root="freesurfer",
            layout="bids_derivatives",
            pipeline="freesurfer",
        )
        catalog.upsert_dataset(local, "D-00000001", None)
        if mode == "census":
            catalog.upsert_dataset(remote, "D-0badcafe", None)
        after = scrub_with(catalog, text)  # with what the catalog holds now
    if mode == "local":  # the labels of a local run are the user's own
        assert before == "D-?: cannot list CANARYROOT for sub-CANARY7"
        assert after == "D-?: cannot list <root> for sub-CANARY7"
    else:
        assert before == "D-?: cannot list CANARYROOT for sub-?"
        assert after == "D-0badcafe: cannot list <root> for sub-?"
