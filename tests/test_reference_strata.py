"""Names of strata and of references (reference/strata.py)."""

from __future__ import annotations

import hashlib
import json

import pytest

from surfcensus.reference import strata
from surfcensus.reference.strata import (
    ALL_STRATUM,
    FIELD_BINS,
    dataset_stratum,
    field_bin,
    field_version_stratum,
    members_sha256,
    set_reference_id,
    version_stratum,
)

DIGEST_A, DIGEST_B = "a" * 64, "b" * 64


@pytest.mark.parametrize(
    ("tesla", "expected"),
    [
        (1.5, "1.5"),
        (1.494, "1.5"),
        (0.55, "1.5"),
        (2.89, "3"),  # what a scanner sold as 3 T reports
        (3.0, "3"),
        (3, "3"),
        (4.7, "3"),
        (6.98, "7"),
        (7.0, "7"),
        (9.4, "7"),
        (2.25, "1.5"),  # half-way: the lower bin
        (5.0, "3"),
        (2.2500001, "3"),
    ],
)
def test_field_strength_goes_to_the_nearest_bin(tesla: float, expected: str) -> None:
    assert field_bin(tesla) == expected


@pytest.mark.parametrize("tesla", [None, 0.0, -3.0, float("nan"), float("inf")])
def test_an_unusable_field_strength_has_no_bin(tesla: float | None) -> None:
    assert field_bin(tesla) is None


def test_the_bins_and_the_names() -> None:
    assert FIELD_BINS == (1.5, 3.0, 7.0)
    assert ALL_STRATUM == "all"
    assert dataset_stratum("D-3f9a1c07") == "dataset:D-3f9a1c07"
    assert version_stratum(6) == "fs:6" and version_stratum(None) is None
    assert field_version_stratum(2.89, 7) == "field:3|fs:7"
    assert field_version_stratum(1.5, 6) == "field:1.5|fs:6"
    assert field_version_stratum(None, 6) is None
    assert field_version_stratum(3.0, None) is None
    assert field_version_stratum(float("nan"), 6) is None
    # the middle name ends with the name of its version alone
    middle, version = field_version_stratum(7.0, 6), version_stratum(6)
    assert middle is not None and version is not None and middle.endswith("|" + version)


def test_members_digest_is_over_sorted_lines_of_uid_and_map_digest() -> None:
    members = {"uid2": DIGEST_B, "uid1": DIGEST_A}
    expected = hashlib.sha256(f"uid1\t{DIGEST_A}\nuid2\t{DIGEST_B}".encode()).hexdigest()
    assert members_sha256(members) == expected
    assert members_sha256(dict(reversed(members.items()))) == expected
    assert members_sha256({}) == hashlib.sha256(b"").hexdigest()
    # a member whose map changed has the same uid and another digest
    assert members_sha256({"uid1": DIGEST_A, "uid2": DIGEST_A}) != expected
    assert members_sha256({"uid1": DIGEST_A}) != expected


def test_the_id_of_a_set_reference() -> None:
    members = {"all": members_sha256({"uid1": DIGEST_A}), "dataset:D-1": "c" * 64}
    reference_id = set_reference_id("0123456789ab", members)
    payload = json.dumps(
        {"config": "0123456789ab", "members": members}, sort_keys=True, separators=(",", ":")
    )
    assert reference_id == "set-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
    assert reference_id.startswith(strata.SET_PREFIX) and len(reference_id) == 16
    assert set_reference_id("0123456789ab", dict(reversed(members.items()))) == reference_id
    # another setting, another member map and another set of strata each give another id
    assert set_reference_id("0123456789ac", members) != reference_id
    changed = members | {"all": members_sha256({"uid1": DIGEST_B})}
    assert set_reference_id("0123456789ab", changed) != reference_id
    assert set_reference_id("0123456789ab", {"all": members["all"]}) != reference_id


def test_the_prefixes_tell_the_kinds_apart() -> None:
    prefixes = {strata.SET_PREFIX, strata.SHIPPED_PREFIX, strata.RATED_PREFIX}
    assert prefixes == {"set-", "shipped-", "rated-"}
