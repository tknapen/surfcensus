"""A normative reference in memory and on disk: per-vertex median and MAD maps per stratum.

On disk a reference is a directory (``tables/reference/<reference_id>/`` of a run, and
``surfcensus/data/reference/`` for the one shipped with the package):

* ``index.json`` (:class:`ReferenceIndex`): ``schema_version``, ``reference_id``, ``kind``,
  ``mesh``, ``template_id``, ``sphere_sha256``, ``config_hash``, ``metric_norms``,
  ``parcel_norms``, and ``strata``: name ->
  ``{"file": "stratum_000.npz", "n": ..., "members_sha256": ...}``;
* one ``stratum_<nnn>.npz`` per stratum, numbered in the order of the stratum names, with
  ``{lh,rh}_median`` and ``{lh,rh}_mad`` (``<f4``) and ``{lh,rh}_n`` (``<i4``);
* ``members.parquet`` with the columns ``stratum`` and ``subject_uid``, sorted.

``index.json`` is removed before anything else is written and written last, so its presence
marks a complete reference. No file holds a path or a time, and the same reference gives the
same bytes (for the parquet file: with the same pyarrow).

Stratum names and reference ids are made by ``reference.strata``.

A reference that cannot be written, and a directory that does not hold what its index
lists, is a ``ValueError`` that names the file and never its path. A directory without
``index.json`` is a ``FileNotFoundError``: there is no reference there.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from surfcensus._io import atomic_path, read_npz, write_npz
from surfcensus.reference.strata import (
    ALL_STRATUM,
    RATED_PREFIX,
    SET_PREFIX,
    SHIPPED_PREFIX,
    field_version_stratum,
    set_reference_id,
    version_stratum,
)
from surfcensus.sources.base import HEMIS, Hemi
from surfcensus.store import read_table, write_table

REFERENCE_SCHEMA = 1
"""Version of the layout on disk; ``index.json`` of another version is not read."""

INDEX_JSON = "index.json"
MEMBERS_PARQUET = "members.parquet"

SHIPPED_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
"""Where the reference that ships with the package lies."""

N_PERCENTILES = 101
"""Length of a grid in ``metric_norms``: the 0th to the 100th percentile."""

MEMBERS_SCHEMA = pa.schema(
    [
        pa.field("stratum", pa.string(), nullable=False),
        pa.field("subject_uid", pa.string(), nullable=False),
    ]
)

ReferenceKind = Literal["shipped", "set", "rated"]

_PREFIX: dict[str, str] = {"shipped": SHIPPED_PREFIX, "set": SET_PREFIX, "rated": RATED_PREFIX}
_ID_REST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
# three digits, or more without a leading zero: the names _stratum_file gives, and no other
_STRATUM_FILE = re.compile(r"stratum_(?:[0-9]{3}|[1-9][0-9]{3,})\.npz", re.ASCII)
_MAP_DTYPES: dict[str, str] = {"median": "<f4", "mad": "<f4", "n": "<i4"}


@dataclass(frozen=True)
class ReferenceMaps:
    """The reference of one stratum on one hemisphere."""

    median: Float[np.ndarray, " v"]
    """float32, of thickness minus the subject's global median; NaN where there is no
    reference."""
    mad: Float[np.ndarray, " v"]
    """float32, smoothed on the mesh and floored."""
    n: Int[np.ndarray, " v"]
    """int32, subjects with a value at the vertex."""


@dataclass(frozen=True)
class Reference:
    reference_id: str
    """``shipped-<version>``, ``set-<12 hex>``, later ``rated-<12 hex>``."""
    kind: ReferenceKind
    mesh: str
    template_id: str | None
    """None for a shipped reference."""
    sphere_sha256: Mapping[Hemi, str]
    """Of the template's source ``?h.sphere.reg``. A reference applies to a run only when
    these equal the run template's."""
    config_hash: str
    """The hash of the ``reference`` stage that built it; with the members' digests it gives
    the id of a set reference (``strata.set_reference_id``)."""
    strata: Mapping[str, Mapping[Hemi, ReferenceMaps]]
    members: Mapping[str, tuple[str, ...]]
    """Stratum -> sorted ``subject_uid``s; empty for a shipped reference."""
    metric_norms: Mapping[str, Mapping[str, tuple[float, ...]]]
    """Stratum -> score component -> its 101 percentiles (0th to 100th) in the reference
    population; shipped and rated references only."""
    parcel_norms: Mapping[str, Mapping[str, tuple[float, float]]]
    """Stratum -> parcel -> (median, MAD) of ``ThickAvg`` minus the subject's mean
    thickness; shipped and rated references only."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class StratumEntry(_Frozen):
    """One stratum in ``index.json``."""

    file: str
    n: int = Field(ge=0)
    """Subjects the stratum's maps were built from: its members where they are recorded,
    else the largest per-vertex count of its maps."""
    members_sha256: str | None = None
    """``strata.members_sha256`` of the members' map files; None when the writer was not
    given the digests."""

    @model_validator(mode="after")
    def _names(self) -> StratumEntry:
        if not _STRATUM_FILE.fullmatch(self.file):
            raise ValueError(f"{self.file!r} is not the name of a stratum file")
        if self.members_sha256 is not None and not _SHA256.fullmatch(self.members_sha256):
            raise ValueError("members_sha256 is not a SHA-256 in lower-case hex")
        return self


class ReferenceIndex(_Frozen):
    """The content of ``index.json``: everything about a reference but its maps and the
    names of its members."""

    schema_version: int
    reference_id: str
    kind: ReferenceKind
    mesh: str = Field(min_length=1)
    template_id: str | None
    sphere_sha256: dict[Hemi, str]
    config_hash: str = Field(min_length=1)
    metric_norms: dict[str, dict[str, tuple[float, ...]]]
    parcel_norms: dict[str, dict[str, tuple[float, float]]]
    strata: dict[str, StratumEntry]

    @model_validator(mode="after")
    def _consistent(self) -> ReferenceIndex:
        prefix = _PREFIX[self.kind]
        rest = self.reference_id.removeprefix(prefix)
        if not self.reference_id.startswith(prefix) or not _ID_REST.fullmatch(rest):
            raise ValueError(
                f"the id of a {self.kind} reference is {prefix}<name>, got {self.reference_id!r}"
            )
        if self.kind == "shipped" and self.template_id is not None:
            raise ValueError("a shipped reference has no template_id")
        if self.kind != "shipped" and not self.template_id:
            raise ValueError(f"a {self.kind} reference needs the template_id of its run")
        if set(self.sphere_sha256) != set(HEMIS) or not all(
            _SHA256.fullmatch(digest) for digest in self.sphere_sha256.values()
        ):
            raise ValueError("sphere_sha256 holds one SHA-256 in lower-case hex for lh and for rh")
        if "" in self.strata:
            raise ValueError("a stratum has no name")
        files = [entry.file for entry in self.strata.values()]
        if len(set(files)) != len(files):
            raise ValueError("two strata name one file")
        for what, norms in (
            ("metric_norms", self.metric_norms),
            ("parcel_norms", self.parcel_norms),
        ):
            unknown = sorted(set(norms) - set(self.strata))
            if unknown:
                raise ValueError(f"{what} names strata the reference lacks: {', '.join(unknown)}")
        for stratum, components in self.metric_norms.items():
            for component, grid in components.items():
                if len(grid) != N_PERCENTILES or any(b < a for a, b in pairwise(grid)):
                    raise ValueError(
                        f"metric_norms of {component} in {stratum}: not {N_PERCENTILES} "
                        "percentiles that do not decrease"
                    )
        for stratum, parcels in self.parcel_norms.items():
            for parcel, (_, mad) in parcels.items():
                if mad < 0:
                    raise ValueError(f"parcel_norms of {parcel} in {stratum}: a negative MAD")
        digests = {name: entry.members_sha256 for name, entry in self.strata.items()}
        known = {name: digest for name, digest in digests.items() if digest is not None}
        if self.kind == "set" and known and len(known) == len(digests):
            expected = set_reference_id(self.config_hash, known)
            if self.reference_id != expected:
                raise ValueError(
                    f"the id {self.reference_id} is not that of this config and these members' "
                    f"maps ({expected})"
                )
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"


def stratum_size(ref: Reference, stratum: str) -> int:
    """Subjects that the maps of ``stratum`` were built from: the number of its members
    where the reference records them, else (a shipped reference) the largest per-vertex
    count of its maps. ``KeyError`` for a stratum the reference lacks."""
    members = ref.members.get(stratum, ())
    if members:
        return len(members)
    counts = [maps.n for maps in ref.strata[stratum].values() if maps.n.size]
    return max((int(n.max()) for n in counts), default=0)


def shipped_reference() -> Reference | None:
    """The reference in ``surfcensus/data/reference/``; None while it has no ``index.json``."""
    if not (SHIPPED_DIR / INDEX_JSON).is_file():
        return None
    ref = read_reference(SHIPPED_DIR)
    if ref.kind != "shipped":
        raise ValueError(f"the reference in the package data is of kind {ref.kind}, not shipped")
    return ref


def applicable_stratum(
    ref: Reference, field_strength: float | None, fs_major: int | None
) -> str | None:
    """The stratum of ``ref`` that a subject is compared with: the first that the reference
    has of ``field:<T>|fs:<major>``, ``fs:<major>``, ``all`` (names from
    ``reference.strata``); None when it has none of them."""
    candidates = (
        field_version_stratum(field_strength, fs_major),
        version_stratum(fs_major),
        ALL_STRATUM,
    )
    return next((name for name in candidates if name is not None and name in ref.strata), None)


def write_reference(
    directory: Path, ref: Reference, *, members_sha256: Mapping[str, str] | None = None
) -> None:
    """Write ``ref`` into ``directory`` in the layout of the module docstring.

    ``members_sha256`` maps every stratum to the digest of its members' map files
    (``strata.members_sha256``) and is recorded in the index; :func:`read_index` returns it.
    Given for a set reference, it must be what the id was made from.

    Everything is checked before anything is written (ValueError): the id against the kind
    and, for a set reference, against the digests; maps of float32 and int32, one length for
    both hemispheres of every stratum; members sorted, of known strata, at least as many as
    the largest count of the stratum's maps, present for every stratum of a set reference
    and for none of a shipped one; finite norms. A stratum without members may be left out
    of ``ref.members`` or have an empty tuple; :func:`read_reference` leaves it out.

    Stratum files that an earlier reference left in ``directory`` are removed; other files
    are left alone.
    """
    directory = Path(directory)
    names = sorted(ref.strata)
    files = {name: _stratum_file(position) for position, name in enumerate(names)}
    members = _checked_members(ref)
    arrays = {name: _checked_arrays(name, ref.strata[name]) for name in names}
    _same_length(arrays)
    for name, uids in members.items():
        largest = max(int(arrays[name][f"{hemi}_n"].max(initial=0)) for hemi in HEMIS)
        if largest > len(uids):
            raise ValueError(
                f"stratum {name}: a vertex counts {largest} subjects, the stratum has "
                f"{len(uids)} members"
            )
    if members_sha256 is not None and set(members_sha256) != set(names):
        raise ValueError("members_sha256 must name exactly the strata of the reference")
    index = _validated(
        {
            "schema_version": REFERENCE_SCHEMA,
            "reference_id": ref.reference_id,
            "kind": ref.kind,
            "mesh": ref.mesh,
            "template_id": ref.template_id,
            "sphere_sha256": dict(ref.sphere_sha256),
            "config_hash": ref.config_hash,
            "metric_norms": _plain(ref.metric_norms),
            "parcel_norms": _plain(ref.parcel_norms),
            "strata": {
                name: {
                    "file": files[name],
                    "n": _size(members.get(name, ()), arrays[name]),
                    "members_sha256": None if members_sha256 is None else members_sha256[name],
                }
                for name in names
            },
        },
        "the reference",
    )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / INDEX_JSON).unlink(missing_ok=True)
    for name in names:
        write_npz(directory / files[name], arrays[name])
    rows = [(name, uid) for name in sorted(members) for uid in members[name]]
    write_table(
        directory / MEMBERS_PARQUET,
        _members_frame(rows),
        MEMBERS_SCHEMA,
        sort_by=("stratum", "subject_uid"),
    )
    kept = set(files.values())
    for path in sorted(directory.iterdir()):
        stale = _STRATUM_FILE.fullmatch(path.name) and path.name not in kept
        if stale and (path.is_file() or path.is_symlink()):
            path.unlink()
    with atomic_path(directory / INDEX_JSON) as tmp:
        tmp.write_bytes(index.to_json().encode())


def read_index(directory: Path) -> ReferenceIndex:
    """``index.json`` of the reference in ``directory``, without its maps: what
    :func:`write_reference` recorded, the members' digests included."""
    raw = (Path(directory) / INDEX_JSON).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{INDEX_JSON} is not JSON: {exc}") from exc
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version != REFERENCE_SCHEMA:
        raise ValueError(
            f"{INDEX_JSON} has schema version {version!r}; this surfcensus reads {REFERENCE_SCHEMA}"
        )
    return _validated(data, INDEX_JSON)


def read_reference(directory: Path) -> Reference:
    """Read a reference written by :func:`write_reference`; equal to the one written.

    Maps come back as float32 and int32, norms as tuples of floats, and ``members`` holds
    the strata that have members. What the directory holds is checked against the index.
    """
    directory = Path(directory)
    index = read_index(directory)
    members = _read_members(directory, index)
    strata: dict[str, dict[Hemi, ReferenceMaps]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(index.strata):
        entry = index.strata[name]
        arrays[name] = _read_stratum(directory / entry.file)
        largest = max(int(arrays[name][f"{hemi}_n"].max(initial=0)) for hemi in HEMIS)
        if largest > entry.n:
            raise ValueError(
                f"{entry.file}: a vertex counts {largest} subjects, {INDEX_JSON} says {entry.n}"
            )
        strata[name] = {
            hemi: ReferenceMaps(
                median=arrays[name][f"{hemi}_median"].astype(np.float32, copy=False),
                mad=arrays[name][f"{hemi}_mad"].astype(np.float32, copy=False),
                n=arrays[name][f"{hemi}_n"].astype(np.int32, copy=False),
            )
            for hemi in HEMIS
        }
    _same_length(arrays)
    return Reference(
        reference_id=index.reference_id,
        kind=index.kind,
        mesh=index.mesh,
        template_id=index.template_id,
        sphere_sha256=dict(index.sphere_sha256),
        config_hash=index.config_hash,
        strata=strata,
        members=members,
        metric_norms=index.metric_norms,
        parcel_norms=index.parcel_norms,
    )


def _stratum_file(position: int) -> str:
    return f"stratum_{position:03d}.npz"


def _validated(data: object, what: str) -> ReferenceIndex:
    try:
        return ReferenceIndex.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        text = first["msg"].removeprefix("Value error, ")
        raise ValueError(f"{what}: {where}: {text}" if where else f"{what}: {text}") from None


def _plain(norms: Mapping[str, Mapping[str, tuple[float, ...]]]) -> dict[str, dict[str, object]]:
    """Norms as plain floats; a value that is not finite is refused here, by its name."""
    plain: dict[str, dict[str, object]] = {}
    for stratum, by_name in norms.items():
        plain[stratum] = {}
        for name, values in by_name.items():
            floats = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in floats):
                raise ValueError(
                    f"the norms of {name} in {stratum} hold a value that is not finite"
                )
            plain[stratum][name] = floats
    return plain


def _checked_members(ref: Reference) -> dict[str, tuple[str, ...]]:
    """The strata of ``ref`` that have members, each with its ``subject_uid``s."""
    unknown = sorted(set(ref.members) - set(ref.strata))
    if unknown:
        raise ValueError(f"members of strata the reference lacks: {', '.join(unknown)}")
    members = {name: tuple(uids) for name, uids in ref.members.items() if len(uids)}
    for name, uids in members.items():
        if not all(isinstance(uid, str) and uid for uid in uids):
            raise ValueError(f"stratum {name}: a member is not a subject_uid")
        if any(b <= a for a, b in pairwise(uids)):
            raise ValueError(f"stratum {name}: the members are not sorted, or one is there twice")
    if ref.kind == "shipped" and members:
        raise ValueError("a shipped reference names no members")
    if ref.kind == "set":
        without = sorted(set(ref.strata) - set(members))
        if without:
            raise ValueError(f"a set reference needs the members of {', '.join(without)}")
    return members


def _checked_arrays(name: str, maps: Mapping[Hemi, ReferenceMaps]) -> dict[str, np.ndarray]:
    """The six arrays of one stratum under their names in the archive."""
    if set(maps) != set(HEMIS):
        raise ValueError(f"stratum {name}: maps for lh and rh are needed, got {sorted(maps)}")
    arrays: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        for field, dtype in _MAP_DTYPES.items():
            array = np.asarray(getattr(maps[hemi], field))
            if array.ndim != 1 or (array.dtype.kind, array.dtype.itemsize) != (dtype[1], 4):
                raise ValueError(
                    f"stratum {name}: {hemi} {field} must be one axis of {np.dtype(dtype).name}, "
                    f"got {array.ndim} of {array.dtype.name}"
                )
            arrays[f"{hemi}_{field}"] = array
        if np.isinf(arrays[f"{hemi}_median"]).any() or np.isinf(arrays[f"{hemi}_mad"]).any():
            raise ValueError(f"stratum {name}: {hemi} holds an infinite value")
        if (arrays[f"{hemi}_mad"] < 0).any():
            raise ValueError(f"stratum {name}: {hemi} holds a negative MAD")
        if (arrays[f"{hemi}_n"] < 0).any():
            raise ValueError(f"stratum {name}: {hemi} holds a negative count")
    return arrays


def _same_length(arrays: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    lengths = {array.shape[0] for stratum in arrays.values() for array in stratum.values()}
    if len(lengths) > 1:
        raise ValueError(
            "the maps of a reference are on one mesh, got lengths "
            + ", ".join(str(length) for length in sorted(lengths))
        )


def _size(members: tuple[str, ...], arrays: Mapping[str, np.ndarray]) -> int:
    if members:
        return len(members)
    return max(int(arrays[f"{hemi}_n"].max(initial=0)) for hemi in HEMIS)


def _members_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stratum": pd.Series([stratum for stratum, _ in rows], dtype=object),
            "subject_uid": pd.Series([uid for _, uid in rows], dtype=object),
        }
    )


def _read_members(directory: Path, index: ReferenceIndex) -> dict[str, tuple[str, ...]]:
    path = directory / MEMBERS_PARQUET
    if not path.is_file():
        raise ValueError(f"{INDEX_JSON} is there and {MEMBERS_PARQUET} is missing")
    try:
        frame = read_table(path, MEMBERS_SCHEMA)
    except (ValueError, pa.ArrowException, OSError) as exc:
        raise ValueError(f"{MEMBERS_PARQUET} is not the member table of a reference") from exc
    members: dict[str, tuple[str, ...]] = {}
    for stratum, uids in frame.groupby("stratum", sort=True)["subject_uid"]:
        members[str(stratum)] = tuple(sorted(str(uid) for uid in uids))
    unknown = sorted(set(members) - set(index.strata))
    if unknown:
        raise ValueError(f"{MEMBERS_PARQUET} names strata that {INDEX_JSON} lacks: {unknown}")
    for name, uids in members.items():
        if len(set(uids)) != len(uids):
            raise ValueError(f"{MEMBERS_PARQUET} names a member of {name} twice")
        if len(uids) != index.strata[name].n:
            raise ValueError(
                f"{MEMBERS_PARQUET} has {len(uids)} members of {name}, {INDEX_JSON} says "
                f"{index.strata[name].n}"
            )
    if index.kind == "shipped" and members:
        raise ValueError("a shipped reference names no members")
    if index.kind == "set" and set(members) != set(index.strata):
        raise ValueError(f"{MEMBERS_PARQUET} lacks the members of a stratum of a set reference")
    return members


def _read_stratum(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"{INDEX_JSON} lists {path.name}, which is missing")
    try:
        arrays = read_npz(path)
    except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path.name} cannot be read as an archive of arrays") from exc
    expected = {f"{hemi}_{field}": dtype for hemi in HEMIS for field, dtype in _MAP_DTYPES.items()}
    if set(arrays) != set(expected):
        raise ValueError(f"{path.name} holds {sorted(arrays)}, not the six maps of a stratum")
    for key, dtype in expected.items():
        array = arrays[key]
        if array.ndim != 1 or (array.dtype.kind, array.dtype.itemsize) != (dtype[1], 4):
            raise ValueError(f"{path.name}: {key} is not one axis of {np.dtype(dtype).name}")
    return arrays
