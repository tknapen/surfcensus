"""A reference on disk (reference/load.py): what is written, what is read back, what is
refused, which stratum applies to a subject, and the reference shipped with the package."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from surfcensus import _io
from surfcensus.reference import load
from surfcensus.reference.load import (
    INDEX_JSON,
    MEMBERS_PARQUET,
    REFERENCE_SCHEMA,
    Reference,
    ReferenceIndex,
    ReferenceMaps,
    applicable_stratum,
    read_index,
    read_reference,
    shipped_reference,
    stratum_size,
    write_reference,
)
from surfcensus.reference.strata import members_sha256, set_reference_id
from surfcensus.sources.base import HEMIS, Hemi

V = 42  # vertices of an icosahedron of order 1
STAGE_HASH = "0123456789ab"
SPHERES: dict[Hemi, str] = {"lh": "1" * 64, "rh": "2" * 64}
UIDS = tuple(f"{value:016x}" for value in range(0xA0, 0xAA))
DATASET, FIELD_FS, ALL = "dataset:D-3f9a1c07", "field:3|fs:7", "all"
SET_MEMBERS = {ALL: UIDS, DATASET: UIDS[:5], FIELD_FS: UIDS[:8]}
MAP_KEYS = [f"{hemi}_{field}" for hemi in HEMIS for field in ("median", "mad", "n")]


def some_maps(seed: int, n_subjects: int, v: int = V) -> dict[Hemi, ReferenceMaps]:
    """Maps as ``build_reference`` makes them: NaN where too few subjects have a value."""
    rng = np.random.default_rng(seed)
    maps: dict[Hemi, ReferenceMaps] = {}
    for hemi in HEMIS:
        n = rng.integers(0, n_subjects + 1, v).astype(np.int32)
        n[0], n[1] = n_subjects, 0
        median = rng.normal(0.0, 0.3, v).astype(np.float32)
        mad = (0.1 + rng.random(v)).astype(np.float32)
        median[n < 2] = np.nan
        mad[n < 2] = np.nan
        maps[hemi] = ReferenceMaps(median=median, mad=mad, n=n)
    return maps


def map_digests(uids: tuple[str, ...], version: str = "") -> dict[str, str]:
    """A stand-in for the SHA-256 of each member's map file."""
    return {uid: hashlib.sha256(f"map of {uid}{version}".encode()).hexdigest() for uid in uids}


def set_reference(
    members: Mapping[str, tuple[str, ...]] = SET_MEMBERS, version: str = ""
) -> tuple[Reference, dict[str, str]]:
    """A reference built within a run, and the digests its id was made from."""
    digests = {name: members_sha256(map_digests(uids, version)) for name, uids in members.items()}
    ref = Reference(
        reference_id=set_reference_id(STAGE_HASH, digests),
        kind="set",
        mesh="fsaverage6",
        template_id="5e1f7a9c0b2d",
        sphere_sha256=SPHERES,
        config_hash=STAGE_HASH,
        strata={
            name: some_maps(seed, len(uids)) for seed, (name, uids) in enumerate(members.items())
        },
        members=dict(members),
        metric_norms={},
        parcel_norms={},
    )
    return ref, digests


def grid(low: float, high: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(low, high, 101))


def shipped(strata: tuple[str, ...] = ("field:3|fs:7", "fs:7", "all")) -> Reference:
    return Reference(
        reference_id="shipped-2027.1",
        kind="shipped",
        mesh="fsaverage6",
        template_id=None,
        sphere_sha256=SPHERES,
        config_hash=STAGE_HASH,
        strata={name: some_maps(10 + seed, 400) for seed, name in enumerate(strata)},
        members={},
        metric_norms={
            name: {"mass_medial": grid(0.0, 80.0), "ceiling_frac": (0.0,) * 60 + grid(0, 0.2)[60:]}
            for name in strata
        },
        parcel_norms={
            strata[0]: {"lateraloccipital": (-0.31, 0.12), "cuneus": (-0.6200000047683716, 0.0)}
        },
    )


def planted(record: object, **fields: object) -> object:
    """A copy of a frozen record with fields replaced, without the checks of its
    constructor: what code that runs without the type-checking hook can hand over."""
    changed = copy.copy(record)
    for name, value in fields.items():
        object.__setattr__(changed, name, value)
    return changed


def with_maps(ref: Reference, stratum: str, hemi: Hemi, **fields: object) -> Reference:
    maps = dict(ref.strata[stratum])
    maps[hemi] = planted(maps[hemi], **fields)
    changed = planted(ref, strata={**ref.strata, stratum: maps})
    assert isinstance(changed, Reference)
    return changed


def assert_same(read: Reference, written: Reference) -> None:
    for name in ("reference_id", "kind", "mesh", "template_id", "config_hash"):
        assert getattr(read, name) == getattr(written, name)
    assert dict(read.sphere_sha256) == dict(written.sphere_sha256)
    assert dict(read.members) == {name: uids for name, uids in written.members.items() if uids}
    assert read.metric_norms == written.metric_norms
    assert read.parcel_norms == written.parcel_norms
    assert set(read.strata) == set(written.strata)
    for stratum, by_hemi in written.strata.items():
        assert set(read.strata[stratum]) == set(HEMIS)
        for hemi in HEMIS:
            for field, dtype in (("median", np.float32), ("mad", np.float32), ("n", np.int32)):
                got = getattr(read.strata[stratum][hemi], field)
                want = getattr(by_hemi[hemi], field)
                assert got.dtype == dtype
                assert got.tobytes() == want.tobytes(), (stratum, hemi, field)


def files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def edit_index(directory: Path, change: Callable[[dict], None]) -> None:
    data = json.loads((directory / INDEX_JSON).read_text())
    change(data)
    (directory / INDEX_JSON).write_text(json.dumps(data))


@pytest.fixture
def set_dir(tmp_path: Path) -> Path:
    ref, digests = set_reference()
    directory = tmp_path / "tables" / "reference" / ref.reference_id
    write_reference(directory, ref, members_sha256=digests)
    return directory


# --- round trip ----------------------------------------------------------------------------


def test_a_set_reference_comes_back_as_it_was_written(tmp_path: Path) -> None:
    ref, digests = set_reference()
    directory = tmp_path / "tables" / "reference" / ref.reference_id
    write_reference(directory, ref, members_sha256=digests)
    read = read_reference(directory)
    assert_same(read, ref)
    assert read.members == SET_MEMBERS and read.metric_norms == {} and read.parcel_norms == {}
    assert all(type(uids) is tuple for uids in read.members.values())
    assert np.isnan(read.strata[ALL]["lh"].median[1]) and np.isnan(read.strata[ALL]["rh"].mad[1])


def test_a_shipped_reference_comes_back_with_its_norms_and_without_members(
    tmp_path: Path,
) -> None:
    ref = shipped()
    write_reference(tmp_path / "ref", ref)
    read = read_reference(tmp_path / "ref")
    assert_same(read, ref)
    assert read.members == {} and read.template_id is None and read.kind == "shipped"
    norms = read.metric_norms["fs:7"]["ceiling_frac"]
    assert type(norms) is tuple and len(norms) == 101 and all(type(x) is float for x in norms)
    assert norms == ref.metric_norms["fs:7"]["ceiling_frac"]
    # a float that has no short decimal form survives the JSON
    assert read.parcel_norms["field:3|fs:7"]["cuneus"] == (-0.6200000047683716, 0.0)
    assert set(read.parcel_norms) == {"field:3|fs:7"}


def test_a_stratum_without_members_is_left_out_on_reading(tmp_path: Path) -> None:
    ref, _ = set_reference()
    rated = planted(
        ref, kind="rated", reference_id="rated-0123456789ab", members={ALL: UIDS, DATASET: ()}
    )
    assert isinstance(rated, Reference)
    write_reference(tmp_path / "ref", rated)
    read = read_reference(tmp_path / "ref")
    assert read.members == {ALL: UIDS}
    index = read_index(tmp_path / "ref")
    # without members the size is the largest count of the maps
    assert index.strata[ALL].n == 10 and index.strata[DATASET].n == 5 == stratum_size(read, DATASET)


def test_the_size_of_a_stratum_with_members_is_their_number(tmp_path: Path) -> None:
    # a member that lacks a hemisphere, or cortex somewhere, is counted at no vertex in full
    ref, digests = set_reference()
    sparse = ref
    for hemi in HEMIS:
        sparse = with_maps(sparse, DATASET, hemi, n=np.minimum(ref.strata[DATASET][hemi].n, 4))
    write_reference(tmp_path / "ref", sparse, members_sha256=digests)
    assert read_index(tmp_path / "ref").strata[DATASET].n == 5
    read = read_reference(tmp_path / "ref")
    assert_same(read, sparse)
    assert int(read.strata[DATASET]["lh"].n.max()) == 4 and stratum_size(read, DATASET) == 5


def test_big_endian_maps_are_stored_little_endian_and_read_equal(tmp_path: Path) -> None:
    ref = shipped(("all",))
    maps = ref.strata["all"]["lh"]
    swapped = with_maps(
        ref,
        "all",
        "lh",
        median=maps.median.astype(">f4"),
        mad=maps.mad.astype(">f4"),
        n=maps.n.astype(">i4"),
    )
    write_reference(tmp_path / "swapped", swapped)
    write_reference(tmp_path / "plain", ref)
    assert files(tmp_path / "swapped") == files(tmp_path / "plain")
    assert_same(read_reference(tmp_path / "swapped"), ref)


# --- what is on disk -------------------------------------------------------------------------


def test_the_files_of_a_reference(set_dir: Path) -> None:
    ref, digests = set_reference()
    assert sorted(path.name for path in set_dir.iterdir()) == [
        "index.json",
        "members.parquet",
        "stratum_000.npz",
        "stratum_001.npz",
        "stratum_002.npz",
    ]
    index = json.loads((set_dir / INDEX_JSON).read_text())
    assert sorted(index) == [
        "config_hash",
        "kind",
        "mesh",
        "metric_norms",
        "parcel_norms",
        "reference_id",
        "schema_version",
        "sphere_sha256",
        "strata",
        "template_id",
    ]
    assert index["schema_version"] == REFERENCE_SCHEMA == 1
    assert (index["reference_id"], index["kind"]) == (ref.reference_id, "set")
    assert (index["mesh"], index["template_id"]) == ("fsaverage6", "5e1f7a9c0b2d")
    assert index["sphere_sha256"] == SPHERES and index["config_hash"] == STAGE_HASH
    # numbered in the order of the names, whatever the order of the mapping
    assert index["strata"] == {
        ALL: {"file": "stratum_000.npz", "n": 10, "members_sha256": digests[ALL]},
        DATASET: {"file": "stratum_001.npz", "n": 5, "members_sha256": digests[DATASET]},
        FIELD_FS: {"file": "stratum_002.npz", "n": 8, "members_sha256": digests[FIELD_FS]},
    }

    arrays = _io.read_npz(set_dir / "stratum_001.npz")
    assert list(arrays) == MAP_KEYS
    assert [arrays[key].dtype.str for key in MAP_KEYS] == ["<f4", "<f4", "<i4"] * 2
    for hemi in HEMIS:
        maps = ref.strata[DATASET][hemi]
        assert np.array_equal(arrays[f"{hemi}_median"], maps.median, equal_nan=True)
        assert np.array_equal(arrays[f"{hemi}_mad"], maps.mad, equal_nan=True)
        assert np.array_equal(arrays[f"{hemi}_n"], maps.n)

    table = pq.read_table(set_dir / MEMBERS_PARQUET)
    assert table.schema.names == ["stratum", "subject_uid"]
    assert [str(field.type) for field in table.schema] == ["string", "string"]
    rows = list(zip(*(table.column(name).to_pylist() for name in table.schema.names), strict=True))
    assert rows == [(name, uid) for name in sorted(SET_MEMBERS) for uid in SET_MEMBERS[name]]
    assert table.schema.metadata is None


def test_the_index_is_a_frozen_model_and_holds_what_was_recorded(set_dir: Path) -> None:
    ref, digests = set_reference()
    index = read_index(set_dir)
    assert isinstance(index, ReferenceIndex) and index.reference_id == ref.reference_id
    assert {name: entry.members_sha256 for name, entry in index.strata.items()} == digests
    assert (set_dir / INDEX_JSON).read_text() == index.to_json()
    with pytest.raises(ValueError, match="frozen"):
        index.mesh = "fsaverage"
    with pytest.raises(ValueError, match="surprise"):
        ReferenceIndex.model_validate(index.model_dump() | {"surprise": 1})


def test_without_the_digests_the_index_says_so(tmp_path: Path) -> None:
    ref, _ = set_reference()
    write_reference(tmp_path / "ref", ref)
    index = read_index(tmp_path / "ref")
    assert [entry.members_sha256 for entry in index.strata.values()] == [None, None, None]
    assert_same(read_reference(tmp_path / "ref"), ref)


def test_the_same_reference_gives_the_same_bytes_and_names_no_place(tmp_path: Path) -> None:
    ref, digests = set_reference()
    write_reference(tmp_path / "one", ref, members_sha256=digests)
    backwards = planted(
        ref,
        strata=dict(reversed(list(ref.strata.items()))),
        members=dict(reversed(list(ref.members.items()))),
    )
    assert isinstance(backwards, Reference)
    write_reference(tmp_path / "deep" / "two", backwards, members_sha256=digests)
    write_reference(tmp_path / "one", ref, members_sha256=digests)  # over itself
    one, two = files(tmp_path / "one"), files(tmp_path / "deep" / "two")
    assert one == two and len(one) == 5
    for name, content in one.items():
        assert str(tmp_path).encode() not in content, name
    text = one[INDEX_JSON].decode()
    assert text.endswith("}\n") and not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert '"/' not in text


def test_a_stratum_of_fsaverage6_takes_just_under_a_megabyte(tmp_path: Path) -> None:
    ref = Reference(
        reference_id="shipped-size",
        kind="shipped",
        mesh="fsaverage6",
        template_id=None,
        sphere_sha256=SPHERES,
        config_hash=STAGE_HASH,
        strata={"all": some_maps(3, 500, v=40_962)},
        members={},
        metric_norms={},
        parcel_norms={},
    )
    write_reference(tmp_path / "ref", ref)
    size = (tmp_path / "ref" / "stratum_000.npz").stat().st_size
    assert 6 * 4 * 40_962 < size < 6 * 4 * 40_962 + 2_000  # 983,088 bytes of maps
    assert_same(read_reference(tmp_path / "ref"), ref)


# --- writing over what is there ---------------------------------------------------------------


def test_stratum_files_of_an_earlier_reference_go_and_other_files_stay(tmp_path: Path) -> None:
    directory = tmp_path / "ref"
    write_reference(directory, shipped())
    (directory / "README.md").write_text("kept")
    (directory / "stratum_notes.npz").write_text("kept")
    (directory / "stratum_0007.npz").write_text("not a name the writer gives")
    (directory / "stratum_\u0660\u0660\u0667.npz").write_text("not ASCII digits")
    (directory / "stratum_009.npz").mkdir()
    (directory / "stratum_1000.npz").write_text("stale")
    one = shipped(("all",))
    write_reference(directory, one)
    assert sorted(path.name for path in directory.iterdir()) == [
        "README.md",
        "index.json",
        "members.parquet",
        "stratum_000.npz",
        "stratum_0007.npz",
        "stratum_009.npz",
        "stratum_notes.npz",
        "stratum_\u0660\u0660\u0667.npz",
    ]
    assert_same(read_reference(directory), one)


def test_the_index_goes_first_and_comes_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "ref"
    write_reference(directory, shipped())
    seen: list[bool] = []

    def full_disk(*args: object, **kwargs: object) -> None:
        seen.append((directory / INDEX_JSON).exists())
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(load, "write_table", full_disk)
        with pytest.raises(OSError, match="No space left"):
            write_reference(directory, shipped(("all",)))
    assert seen == [False]
    # what is left is not taken for a reference
    assert not (directory / INDEX_JSON).exists()
    with pytest.raises(FileNotFoundError):
        read_reference(directory)
    assert not [path.name for path in directory.iterdir() if path.name.endswith(".tmp")]
    write_reference(directory, shipped(("all",)))
    assert_same(read_reference(directory), shipped(("all",)))


def test_the_maps_are_written_before_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_npz, real_table = load.write_npz, load.write_table

    def npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        order.append(Path(path).name)
        assert not (tmp_path / "ref" / INDEX_JSON).exists()
        real_npz(path, arrays)

    def table(path: Path, *args: object, **kwargs: object) -> None:
        order.append(Path(path).name)
        assert not (tmp_path / "ref" / INDEX_JSON).exists()
        real_table(path, *args, **kwargs)

    monkeypatch.setattr(load, "write_npz", npz)
    monkeypatch.setattr(load, "write_table", table)
    write_reference(tmp_path / "ref", shipped(("fs:7", "all")))
    assert order == ["stratum_000.npz", "stratum_001.npz", "members.parquet"]
    assert (tmp_path / "ref" / INDEX_JSON).is_file()


# --- what is refused on writing ---------------------------------------------------------------


def _set() -> Reference:
    return set_reference()[0]


def _float64(ref: Reference) -> Reference:
    return with_maps(ref, ALL, "rh", mad=ref.strata[ALL]["rh"].mad.astype(np.float64))


def _int64(ref: Reference) -> Reference:
    return with_maps(ref, ALL, "lh", n=ref.strata[ALL]["lh"].n.astype(np.int64))


def _unsigned(ref: Reference) -> Reference:
    return with_maps(ref, ALL, "lh", n=ref.strata[ALL]["lh"].n.astype(np.uint32))


def _two_axes(ref: Reference) -> Reference:
    return with_maps(ref, ALL, "lh", median=ref.strata[ALL]["lh"].median.reshape(2, -1))


def _short_hemi(ref: Reference) -> Reference:
    maps = ref.strata[DATASET]["rh"]
    return with_maps(ref, DATASET, "rh", median=maps.median[:-1], mad=maps.mad[:-1], n=maps.n[:-1])


def _short_field(ref: Reference) -> Reference:
    return with_maps(ref, DATASET, "rh", n=ref.strata[DATASET]["rh"].n[:-2])


def _other_mesh(ref: Reference) -> Reference:
    changed = planted(ref, strata={**ref.strata, FIELD_FS: some_maps(5, 8, v=162)})
    assert isinstance(changed, Reference)
    return changed


def _one_hemi(ref: Reference) -> Reference:
    changed = planted(ref, strata={**ref.strata, ALL: {"lh": ref.strata[ALL]["lh"]}})
    assert isinstance(changed, Reference)
    return changed


def _negative_mad(ref: Reference) -> Reference:
    mad = ref.strata[ALL]["lh"].mad.copy()
    mad[5] = -0.01
    return with_maps(ref, ALL, "lh", mad=mad)


def _infinite(ref: Reference) -> Reference:
    median = ref.strata[ALL]["rh"].median.copy()
    median[5] = np.inf
    return with_maps(ref, ALL, "rh", median=median)


def _negative_count(ref: Reference) -> Reference:
    n = ref.strata[ALL]["lh"].n.copy()
    n[5] = -1
    return with_maps(ref, ALL, "lh", n=n)


def _more_than_members(ref: Reference) -> Reference:
    n = ref.strata[DATASET]["rh"].n.copy()
    n[7] = 6
    return with_maps(ref, DATASET, "rh", n=n)


def _replace(**fields: object) -> Callable[[Reference], Reference]:
    def change(ref: Reference) -> Reference:
        changed = planted(ref, **fields)
        assert isinstance(changed, Reference)
        return changed

    return change


def _unnamed_stratum(ref: Reference) -> Reference:
    return _replace(strata={"": ref.strata["all"]}, metric_norms={}, parcel_norms={})(ref)


REFUSED: dict[str, tuple[Callable[[], Reference], Callable[[Reference], Reference], str]] = {
    "float64 map": (_set, _float64, "stratum all: rh mad must be one axis of float32, got 1 of fl"),
    "int64 count": (_set, _int64, "stratum all: lh n must be one axis of int32, got 1 of int64"),
    "unsigned count": (_set, _unsigned, "lh n must be one axis of int32, got 1 of uint32"),
    "two axes": (_set, _two_axes, "lh median must be one axis of float32, got 2 of float32"),
    "hemispheres differ": (_set, _short_hemi, "on one mesh, got lengths 41, 42"),
    "fields differ": (_set, _short_field, "on one mesh, got lengths 40, 42"),
    "strata differ": (_set, _other_mesh, "on one mesh, got lengths 42, 162"),
    "one hemisphere": (_set, _one_hemi, "stratum all: maps for lh and rh are needed, got"),
    "negative MAD": (_set, _negative_mad, "stratum all: lh holds a negative MAD"),
    "infinite median": (_set, _infinite, "stratum all: rh holds an infinite value"),
    "negative count": (_set, _negative_count, "stratum all: lh holds a negative count"),
    "count above members": (
        _set,
        _more_than_members,
        "stratum dataset:D-3f9a1c07: a vertex counts 6 subjects, the stratum has 5 members",
    ),
    "members of an unknown stratum": (
        _set,
        _replace(members={**SET_MEMBERS, "fs:7": UIDS}),
        "members of strata the reference lacks: fs:7",
    ),
    "members not sorted": (
        _set,
        _replace(members={**SET_MEMBERS, DATASET: (UIDS[1], UIDS[0], *UIDS[2:5])}),
        "stratum dataset:D-3f9a1c07: the members are not sorted, or one is there twice",
    ),
    "member twice": (
        _set,
        _replace(members={**SET_MEMBERS, DATASET: (*UIDS[:5], UIDS[4])}),
        "the members are not sorted, or one is there twice",
    ),
    "empty member": (
        _set,
        _replace(members={**SET_MEMBERS, DATASET: ("", *UIDS[:4])}),
        "stratum dataset:D-3f9a1c07: a member is not a subject_uid",
    ),
    "set stratum without members": (
        _set,
        _replace(members={ALL: UIDS, FIELD_FS: UIDS[:8], DATASET: ()}),
        "a set reference needs the members of dataset:D-3f9a1c07",
    ),
    "set without template": (
        _set,
        _replace(template_id=None),
        "a set reference needs the template_id of its run",
    ),
    "kind and id disagree": (
        _set,
        _replace(reference_id="shipped-1"),
        "the id of a set reference is set-<name>, got 'shipped-1'",
    ),
    "id with a path": (
        shipped,
        _replace(reference_id="shipped-../1"),
        "the id of a shipped reference is shipped-<name>",
    ),
    "id without a name": (shipped, _replace(reference_id="shipped-"), "shipped-<name>"),
    "unknown kind": (shipped, _replace(kind="pilot"), "kind: Input should be 'shipped', 'set'"),
    "shipped with members": (
        shipped,
        _replace(members={"all": UIDS}),
        "a shipped reference names no members",
    ),
    "shipped with template": (
        shipped,
        _replace(template_id="5e1f7a9c0b2d"),
        "a shipped reference has no template_id",
    ),
    "one sphere digest": (
        shipped,
        _replace(sphere_sha256={"lh": "1" * 64}),
        "sphere_sha256 holds one SHA-256 in lower-case hex for lh and for rh",
    ),
    "short sphere digest": (
        shipped,
        _replace(sphere_sha256={"lh": "1" * 64, "rh": "2" * 63}),
        "sphere_sha256 holds one SHA-256",
    ),
    "upper-case sphere digest": (
        shipped,
        _replace(sphere_sha256={"lh": "1" * 64, "rh": "A" * 64}),
        "sphere_sha256 holds one SHA-256",
    ),
    "no config hash": (shipped, _replace(config_hash=""), "config_hash: String should have at"),
    "no mesh": (shipped, _replace(mesh=""), "mesh: String should have at least 1 character"),
    "unnamed stratum": (
        shipped,
        _unnamed_stratum,
        "a stratum has no name",
    ),
    "norms of an unknown stratum": (
        shipped,
        _replace(metric_norms={"fs:6": {"mass_medial": grid(0, 1)}}),
        "metric_norms names strata the reference lacks: fs:6",
    ),
    "parcel norms of an unknown stratum": (
        shipped,
        _replace(parcel_norms={"fs:6": {"cuneus": (0.0, 0.1)}}),
        "parcel_norms names strata the reference lacks: fs:6",
    ),
    "100 percentiles": (
        shipped,
        _replace(metric_norms={"all": {"mass_medial": grid(0, 1)[:-1]}}),
        "metric_norms of mass_medial in all: not 101 percentiles that do not decrease",
    ),
    "percentiles that decrease": (
        shipped,
        _replace(metric_norms={"all": {"mass_medial": (*grid(0, 1)[:-1], 0.5)}}),
        "metric_norms of mass_medial in all: not 101 percentiles that do not decrease",
    ),
    "NaN percentile": (
        shipped,
        _replace(metric_norms={"all": {"mass_medial": (*grid(0, 1)[:-1], float("nan"))}}),
        "the norms of mass_medial in all hold a value that is not finite",
    ),
    "infinite parcel norm": (
        shipped,
        _replace(parcel_norms={"all": {"cuneus": (float("-inf"), 0.1)}}),
        "the norms of cuneus in all hold a value that is not finite",
    ),
    "negative parcel MAD": (
        shipped,
        _replace(parcel_norms={"all": {"cuneus": (0.2, -0.1)}}),
        "parcel_norms of cuneus in all: a negative MAD",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_reference_that_cannot_be_right_is_refused_before_anything_is_written(
    case: str, tmp_path: Path
) -> None:
    make, change, message = REFUSED[case]
    good = make()
    write_reference(tmp_path / "there", good)
    before = files(tmp_path / "there")
    bad = change(good)
    for directory in (tmp_path / "new" / "ref", tmp_path / "there"):
        with pytest.raises(ValueError, match=re.escape(message)) as caught:
            write_reference(directory, bad)
        assert str(tmp_path) not in str(caught.value)
    assert not (tmp_path / "new").exists()
    assert files(tmp_path / "there") == before
    write_reference(tmp_path / "new" / "ref", good)  # the unchanged one is accepted


def test_a_set_reference_is_written_only_under_the_id_of_its_digests(tmp_path: Path) -> None:
    ref, digests = set_reference()
    other = {**digests, DATASET: "f" * 64}
    with pytest.raises(ValueError, match="is not that of this config and these members' maps"):
        write_reference(tmp_path / "ref", ref, members_sha256=other)
    other_config = planted(ref, config_hash="ba9876543210")
    assert isinstance(other_config, Reference)
    with pytest.raises(ValueError, match=f"the id {ref.reference_id} is not that of this config"):
        write_reference(tmp_path / "ref", other_config, members_sha256=digests)
    for wrong in ({ALL: digests[ALL]}, {**digests, "fs:7": "f" * 64}):
        with pytest.raises(ValueError, match="must name exactly the strata of the reference"):
            write_reference(tmp_path / "ref", ref, members_sha256=wrong)
    with pytest.raises(ValueError, match="members_sha256 is not a SHA-256 in lower-case hex"):
        write_reference(tmp_path / "ref", ref, members_sha256={**digests, ALL: "abc"})
    assert not (tmp_path / "ref").exists()
    write_reference(tmp_path / "ref", ref, members_sha256=digests)
    # for a kind without a rule for its id the digests are only recorded
    rated = planted(ref, kind="rated", reference_id="rated-0123456789ab")
    assert isinstance(rated, Reference)
    write_reference(tmp_path / "rated", rated, members_sha256=other)
    assert read_index(tmp_path / "rated").strata[DATASET].members_sha256 == "f" * 64


# --- the id of a set reference follows the members' maps -----------------------------------------


def test_the_id_changes_when_a_members_map_changes_and_its_uid_does_not(tmp_path: Path) -> None:
    maps_dir = tmp_path / "tables" / "fsavg"
    rng = np.random.default_rng(11)
    for uid in UIDS:
        thickness = rng.normal(2.5, 0.4, (2, V)).astype("<f2")
        _io.write_npz(
            maps_dir / f"{uid}.npz", {"lh_thickness": thickness[0], "rh_thickness": thickness[1]}
        )

    def reference_now() -> tuple[Reference, dict[str, str]]:
        file_digests = {uid: _io.sha256_file(maps_dir / f"{uid}.npz") for uid in UIDS}
        digests = {
            name: members_sha256({uid: file_digests[uid] for uid in uids})
            for name, uids in SET_MEMBERS.items()
        }
        ref, _ = set_reference()
        named = planted(ref, reference_id=set_reference_id(STAGE_HASH, digests))
        assert isinstance(named, Reference)
        return named, digests

    before, digests_before = reference_now()
    assert reference_now()[0].reference_id == before.reference_id
    # the surfaces of one member changed: its map is another file under the same name
    changed_uid = UIDS[6]
    arrays = _io.read_npz(maps_dir / f"{changed_uid}.npz")
    arrays["lh_thickness"] = arrays["lh_thickness"].copy()
    arrays["lh_thickness"][3] += np.float16(0.25)
    _io.write_npz(maps_dir / f"{changed_uid}.npz", arrays)
    after, digests_after = reference_now()

    assert after.members == before.members
    assert after.reference_id != before.reference_id
    assert re.fullmatch(r"set-[0-9a-f]{12}", after.reference_id)
    # only the strata that hold the member have another digest
    assert digests_after[DATASET] == digests_before[DATASET]
    assert digests_after[FIELD_FS] != digests_before[FIELD_FS]
    assert digests_after[ALL] != digests_before[ALL]

    root = tmp_path / "tables" / "reference"
    write_reference(root / before.reference_id, before, members_sha256=digests_before)
    write_reference(root / after.reference_id, after, members_sha256=digests_after)
    assert sorted(path.name for path in root.iterdir()) == sorted(
        [before.reference_id, after.reference_id]
    )
    for ref, digests in ((before, digests_before), (after, digests_after)):
        assert read_reference(root / ref.reference_id).reference_id == ref.reference_id
        recorded = read_index(root / ref.reference_id).strata
        assert {name: entry.members_sha256 for name, entry in recorded.items()} == digests
    # the old id cannot be written with the new maps, nor the new id with the old ones
    with pytest.raises(ValueError, match="is not that of this config and these members' maps"):
        write_reference(tmp_path / "wrong", before, members_sha256=digests_after)
    with pytest.raises(ValueError, match="is not that of this config and these members' maps"):
        write_reference(tmp_path / "wrong", after, members_sha256=digests_before)


# --- what is refused on reading -----------------------------------------------------------------


def test_a_directory_without_an_index_holds_no_reference(tmp_path: Path, set_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_reference(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        read_index(tmp_path)
    (set_dir / INDEX_JSON).unlink()
    with pytest.raises(FileNotFoundError):
        read_reference(set_dir)


def _drop(key: str) -> Callable[[dict], None]:
    def change(data: dict) -> None:
        del data[key]

    return change


def _drop_stratum(name: str) -> Callable[[dict], None]:
    def change(data: dict) -> None:
        del data["strata"][name]

    return change


def _another_stratum(n: int, **header: object) -> Callable[[dict], None]:
    """A fourth stratum, ``fs:7``, on a copy of the maps of ``all``."""

    def change(data: dict) -> None:
        data["strata"]["fs:7"] = {"file": "stratum_003.npz", "n": n, "members_sha256": None}
        data.update(header)

    return change


def _put(key: str, value: object) -> Callable[[dict], None]:
    return lambda data: data.update({key: value})


def _another_header(**fields: object) -> Callable[[dict], None]:
    return lambda data: data.update(fields)


def _stratum(name: str, key: str, value: object) -> Callable[[dict], None]:
    return lambda data: data["strata"][name].update({key: value})


def _forget_digests(data: dict) -> None:
    for entry in data["strata"].values():
        entry["members_sha256"] = None


BAD_INDEX: dict[str, tuple[Callable[[dict], None], str]] = {
    "newer schema": (
        _put("schema_version", 2),
        "index.json has schema version 2; this surfcensus reads 1",
    ),
    "no schema": (_drop("schema_version"), "index.json has schema version None"),
    "field missing": (_drop("mesh"), "index.json: mesh: Field required"),
    "extra field": (_put("created", "2026-11-13"), "index.json: created: Extra inputs are not"),
    "other id": (
        _put("reference_id", "set-000000000000"),
        "index.json: the id set-000000000000 is not that of this config and these members' maps",
    ),
    "other config": (_put("config_hash", "ba9876543210"), "is not that of this config"),
    "other digest": (
        _stratum(ALL, "members_sha256", "e" * 64),
        "is not that of this config and these members' maps",
    ),
    "file outside": (
        _stratum(ALL, "file", "../stratum_000.npz"),
        "index.json: strata.all: '../stratum_000.npz' is not the name of a stratum file",
    ),
    "file twice": (_stratum(ALL, "file", "stratum_001.npz"), "two strata name one file"),
    "file missing": (
        _stratum(ALL, "file", "stratum_017.npz"),
        "index.json lists stratum_017.npz, which is missing",
    ),
    "fewer subjects than the maps count": (
        _another_stratum(9, kind="rated", reference_id="rated-1"),
        "stratum_003.npz: a vertex counts 10 subjects, index.json says 9",
    ),
    "other member count": (
        _stratum(DATASET, "n", 6),
        "members.parquet has 5 members of dataset:D-3f9a1c07, index.json says 6",
    ),
    "negative count": (_stratum(DATASET, "n", -1), "strata.dataset:D-3f9a1c07.n: Input should"),
    # the id of a set reference covers its strata, so one that is taken out shows there
    "stratum gone": (_drop_stratum(DATASET), "is not that of this config and these members' maps"),
    "stratum gone, no digests": (
        _drop_stratum(DATASET),
        "members.parquet names strata that index.json lacks: ['dataset:D-3f9a1c07']",
    ),
    "set stratum without members": (
        _another_stratum(10),
        "members.parquet lacks the members of a stratum of a set reference",
    ),
    "shipped with members": (
        _another_header(kind="shipped", reference_id="shipped-1", template_id=None),
        "a shipped reference names no members",
    ),
    "NaN norm": (
        _put("parcel_norms", {ALL: {"cuneus": [float("nan"), 0.1]}}),
        "index.json: parcel_norms.all.cuneus.0: Input should be a finite number",
    ),
}


WITH_A_FOURTH_FILE = {"fewer subjects than the maps count", "set stratum without members"}
WITHOUT_DIGESTS = {"stratum gone, no digests", "set stratum without members"}
"""Cases on a reference written without the members' digests, as ``write_reference(dir,
ref)`` leaves it: with them, the id no longer fits and that is what is reported."""


@pytest.mark.parametrize("case", sorted(BAD_INDEX))
def test_an_index_that_does_not_fit_the_directory_is_refused(case: str, set_dir: Path) -> None:
    change, message = BAD_INDEX[case]
    if case in WITH_A_FOURTH_FILE:
        (set_dir / "stratum_003.npz").write_bytes((set_dir / "stratum_000.npz").read_bytes())
    if case in WITHOUT_DIGESTS:
        edit_index(set_dir, _forget_digests)
    edit_index(set_dir, change)
    with pytest.raises(ValueError, match=re.escape(message)) as caught:
        read_reference(set_dir)
    assert not isinstance(caught.value, FileNotFoundError)
    assert str(set_dir) not in str(caught.value)


def test_an_index_that_is_not_json_or_not_an_object_is_refused(set_dir: Path) -> None:
    (set_dir / INDEX_JSON).write_text('{"schema_version": 1,')
    with pytest.raises(ValueError, match=r"index\.json is not JSON"):
        read_reference(set_dir)
    (set_dir / INDEX_JSON).write_text("[1]")
    with pytest.raises(ValueError, match=r"index\.json has schema version None"):
        read_index(set_dir)


def rewrite_stratum(path: Path, **changed: np.ndarray | None) -> None:
    """Write the archive again with arrays replaced, added, or (None) left out."""
    arrays = _io.read_npz(path) | changed
    _io.write_npz(path, {key: array for key, array in arrays.items() if array is not None})


def test_stratum_files_that_are_not_what_the_index_lists_are_refused(set_dir: Path) -> None:
    path = set_dir / "stratum_001.npz"
    good = path.read_bytes()

    arrays = _io.read_npz(path)
    rewrite_stratum(path, rh_n=None)
    with pytest.raises(ValueError, match=r"stratum_001\.npz holds .*, not the six maps"):
        read_reference(set_dir)
    path.write_bytes(good)
    rewrite_stratum(path, lh_z=arrays["lh_mad"])
    with pytest.raises(ValueError, match=r"stratum_001\.npz holds .*, not the six maps"):
        read_reference(set_dir)
    path.write_bytes(good)
    rewrite_stratum(path, lh_median=arrays["lh_median"].astype("<f8"))
    with pytest.raises(ValueError, match=r"stratum_001\.npz: lh_median is not one axis of float32"):
        read_reference(set_dir)
    path.write_bytes(good)
    rewrite_stratum(path, rh_n=arrays["rh_n"].astype("<f4"))
    with pytest.raises(ValueError, match=r"stratum_001\.npz: rh_n is not one axis of int32"):
        read_reference(set_dir)
    path.write_bytes(good)
    rewrite_stratum(path, **{key: arrays[key][:-1] for key in MAP_KEYS})
    with pytest.raises(ValueError, match="on one mesh, got lengths 41, 42"):
        read_reference(set_dir)
    for damaged in (good[: len(good) // 2], b"not an archive", b""):
        path.write_bytes(damaged)
        with pytest.raises(ValueError, match=r"stratum_001\.npz cannot be read as an archive"):
            read_reference(set_dir)
    path.write_bytes(good)
    assert_same(read_reference(set_dir), set_reference()[0])


def test_a_member_table_that_is_missing_or_damaged_is_refused(set_dir: Path) -> None:
    path = set_dir / MEMBERS_PARQUET
    good = path.read_bytes()
    path.write_bytes(b"PAR1 not a table")
    with pytest.raises(ValueError, match=r"members\.parquet is not the member table"):
        read_reference(set_dir)
    path.write_bytes(good)
    renamed = pq.read_table(path).rename_columns(["stratum", "uid"])
    pq.write_table(renamed, path)
    with pytest.raises(ValueError, match=r"members\.parquet is not the member table"):
        read_reference(set_dir)
    path.unlink()
    with pytest.raises(ValueError, match=r"index\.json is there and members\.parquet is missing"):
        read_reference(set_dir)
    path.write_bytes(good)
    assert_same(read_reference(set_dir), set_reference()[0])


def test_a_member_named_twice_is_refused(set_dir: Path) -> None:
    path = set_dir / MEMBERS_PARQUET
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[-1] = rows[-2]
    pq.write_table(table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match=r"members\.parquet names a member of .* twice"):
        read_reference(set_dir)


# --- which stratum applies ----------------------------------------------------------------------


def test_the_applicable_stratum_is_the_most_specific_the_reference_has() -> None:
    full = shipped(("field:3|fs:7", "field:7|fs:7", "fs:7", "fs:6", "all"))
    assert applicable_stratum(full, 3.0, 7) == "field:3|fs:7"
    assert applicable_stratum(full, 2.89, 7) == "field:3|fs:7"  # binned as the strata are
    assert applicable_stratum(full, 7.0, 7) == "field:7|fs:7"
    assert applicable_stratum(full, 1.5, 7) == "fs:7"  # no 1.5 T stratum of that version
    assert applicable_stratum(full, 3.0, 6) == "fs:6"
    assert applicable_stratum(full, None, 7) == "fs:7"  # a local run without --field-strength
    assert applicable_stratum(full, float("nan"), 7) == "fs:7"
    assert applicable_stratum(full, 3.0, 5) == "all"
    assert applicable_stratum(full, 3.0, None) == "all"  # the version decides both upper levels
    assert applicable_stratum(full, None, None) == "all"


def test_a_reference_without_a_matching_stratum_does_not_apply() -> None:
    narrow = shipped(("field:3|fs:7", "fs:6"))
    assert applicable_stratum(narrow, 3.0, 7) == "field:3|fs:7"
    assert applicable_stratum(narrow, 7.0, 7) is None
    assert applicable_stratum(narrow, None, 7) is None
    assert applicable_stratum(narrow, 7.0, 6) == "fs:6"
    assert applicable_stratum(narrow, None, None) is None
    only_all = shipped(("all",))
    assert applicable_stratum(only_all, 3.0, 7) == "all"
    # a dataset stratum is never what a subject of another run is compared with
    ref, _ = set_reference({DATASET: UIDS[:5], FIELD_FS: UIDS[:8]})
    assert applicable_stratum(ref, 3.0, 7) == FIELD_FS
    assert applicable_stratum(ref, 3.0, 6) is None


def test_the_size_of_a_stratum() -> None:
    ref, _ = set_reference()
    assert [stratum_size(ref, name) for name in (ALL, DATASET, FIELD_FS)] == [10, 5, 8]
    assert stratum_size(shipped(), "fs:7") == 400  # no members: the largest count of the maps
    lopsided = with_maps(shipped(), "fs:7", "lh", n=np.zeros(V, np.int32))
    assert stratum_size(lopsided, "fs:7") == 400
    with pytest.raises(KeyError):
        stratum_size(ref, "fs:7")


# --- the reference shipped with the package ------------------------------------------------------


def test_no_reference_ships_today() -> None:
    assert shipped_reference() is None
    assert load.SHIPPED_DIR == Path(load.__file__).parent.parent / "data" / "reference"
    assert [path.name for path in load.SHIPPED_DIR.iterdir()] == ["README.md"]


def test_a_reference_in_the_package_data_is_the_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "SHIPPED_DIR", tmp_path / "data" / "reference")
    assert shipped_reference() is None  # the directory need not exist
    write_reference(load.SHIPPED_DIR, shipped())
    (load.SHIPPED_DIR / "README.md").write_text("what this is")
    found = shipped_reference()
    assert found is not None
    assert_same(found, shipped())
    assert applicable_stratum(found, None, 7) == "fs:7"


def test_package_data_of_another_kind_is_not_taken_for_the_shipped_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "SHIPPED_DIR", tmp_path)
    ref, digests = set_reference()
    write_reference(tmp_path, ref, members_sha256=digests)
    with pytest.raises(ValueError, match="of kind set, not shipped"):
        shipped_reference()
