"""The template: everything the metrics need on the target mesh, derived at run time from a
FreeSurfer ``fsaverage`` directory. See docs/metrics.md and decision D14.

Nothing derived from fsaverage ships with the package. A template is built from a directory
that holds the files of ``sources.patterns.TEMPLATE_FILES`` (found in a FreeSurfer install,
next to the subjects, or given explicitly), identified by the content of those files, and
cached in the run directory under ``templates/<template_id>/``.

The target mesh is an icosahedral mesh of lower or equal order than the directory's own.
Its vertices are the first vertices of the full mesh, so every per-vertex file is cut to
length; its faces are not a subset of the full faces and are derived by
:func:`coarsen_faces`.

Checked on the ``fsaverage`` next to the subjects of ``ds000102-fmriprep`` (2026-11-13): the
sphere and the white surface of a hemisphere have the same 327,680 faces; both annotations
carry the same 36 names in the same order; ``surf/?h.white.avg.area.mgh`` is an MGH volume
of shape (163842, 1, 1), float32; ``scripts/build-stamp.txt`` is one line.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from stat import S_ISDIR
from typing import Literal

import nibabel as nib
import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy import sparse

from surfcensus._io import atomic_path, canonical_json, read_npz, sha256_file, write_npz
from surfcensus.config import MeshName, MetricsConfig, RegionConfig
from surfcensus.fsio.annot import read_annotation, read_label
from surfcensus.fsio.registration import unit_sphere
from surfcensus.fsio.surfaces import read_surface
from surfcensus.log import get_logger
from surfcensus.sources.base import HEMIS, Hemi
from surfcensus.sources.patterns import TEMPLATE_FILES, TEMPLATE_OPTIONAL_FILES, is_file

__all__ = [
    "MESH_ORDER",
    "REGION_NAMES",
    "TEMPLATES_DIR",
    "TEMPLATE_FILES",
    "TEMPLATE_JSON",
    "TEMPLATE_NPZ",
    "TEMPLATE_OPTIONAL_FILES",
    "TEMPLATE_SCHEMA",
    "Origin",
    "RegionName",
    "Template",
    "TemplateError",
    "TemplateHemi",
    "TemplateNotFound",
    "build_template",
    "coarsen_faces",
    "find_template",
    "icosahedron_vertices",
    "load_template",
    "read_template",
    "store_template",
    "template_candidates",
    "write_template",
]

TEMPLATE_SCHEMA = 1
"""Hashed into every ``template_id``: raised when the derivation changes for equal files."""
TEMPLATE_JSON = "template.json"
TEMPLATES_DIR = "templates"
"""The template cache of a run directory: one directory per ``template_id``."""
TEMPLATE_NPZ = "template.npz"
"""The two files of a cached template, in ``templates/<template_id>/``."""

RegionName = Literal["search", "medial", "inferior", "lateral", "control"]
REGION_NAMES: tuple[RegionName, ...] = ("search", "medial", "inferior", "lateral", "control")
"""``search`` is the occipital search region; ``medial``, ``inferior`` and ``lateral`` are
its three parts; ``control`` is the frontal control region."""

Origin = Literal["explicit", "freesurfer_home", "adjacent", "bundled"]
"""Where a template directory was found: given by the user, ``$FREESURFER_HOME/subjects``,
next to the subjects, or shipped with the package (decision D14)."""

MESH_ORDER: dict[MeshName, int] = {
    "fsaverage4": 4,
    "fsaverage5": 5,
    "fsaverage6": 6,
    "fsaverage": 7,
}
"""Icosahedral order of each mesh name."""

_DEFAULT_REGIONS = RegionConfig()

_SPHERE, _WHITE, _CORTEX, _ANNOT = TEMPLATE_FILES
(_AVG_AREA,) = TEMPLATE_OPTIONAL_FILES
_BUILD_STAMP = "scripts/build-stamp.txt"
_REQUIRED: tuple[str, ...] = tuple(name.format(h=hemi) for name in TEMPLATE_FILES for hemi in HEMIS)
_BUNDLED_DIR = Path(__file__).parent / "data" / "fsaverage"
"""Where a copy shipped with the package would lie. None ships (decision D14)."""
_NOT_GIVEN: dict[Origin, tuple[str, str]] = {
    "freesurfer_home": ("$FREESURFER_HOME/subjects/fsaverage", "not set"),
    "adjacent": ("next to the subjects", "absent"),
    "bundled": ("bundled copy", "not shipped"),
}
"""``(where, state)`` in ``TemplateNotFound.looked_in`` for an origin that has no directory."""
_ID = re.compile(r"[0-9a-f]{12}")
_MAX_ORDER = 12
_GONE = (
    "the template this run was scored with is no longer available; "
    "pass --fsaverage DIR or --retemplate"
)


def icosahedron_vertices(order: int) -> int:
    """Vertices of the icosahedral mesh of ``order``: ``10 * 4**order + 2`` (40,962 for
    order 6, 163,842 for order 7)."""
    if order < 0:
        raise ValueError(f"an icosahedral order is not negative, got {order}")
    return 10 * 4**order + 2


@dataclass(frozen=True)
class TemplateHemi:
    """One hemisphere of the template on the target mesh; ``v`` vertices, ``f`` faces.

    The arrays are read-only: a template is shared by everything that is computed with it.
    """

    sphere: Float[np.ndarray, "v 3"]
    """Unit vectors, float64."""
    faces: Int[np.ndarray, "f 3"]
    """int32, wound outward (81,920 for fsaverage6)."""
    white: Float[np.ndarray, "v 3"]
    """float32, the template's white surface in its surface RAS."""
    cortex: Bool[np.ndarray, " v"]
    parcel: Int[np.ndarray, " v"]
    """int16 index into ``Template.parcel_names``; -1 for a vertex without a parcel."""
    area: Float[np.ndarray, " v"]
    """float32, mm2 per vertex; see ``Template.area_source``."""
    regions: Mapping[RegionName, Bool[np.ndarray, " v"]]
    """One mask per name of :data:`REGION_NAMES`, each a subset of ``cortex``."""
    pole_vertex: int
    """The cortex vertex with the smallest white y: the occipital pole of decision D12."""


@dataclass(frozen=True)
class Template:
    mesh: MeshName
    order: int
    """``MESH_ORDER[mesh]``."""
    template_id: str
    """12 hex characters over the schema, the mesh, the region config and the SHA-256 of the
    source files that were read. Content only: the same fsaverage found in two places is the
    same template."""
    origin: Origin
    files: Mapping[str, tuple[int, str]]
    """Relative path -> (size, SHA-256) of the source files that were read: the eight of
    ``TEMPLATE_FILES`` and, when the areas come from them, the two average-area files."""
    area_source: Literal["white.avg.area", "white mesh"]
    """``white.avg.area``: FreeSurfer's average of subject vertex areas, summed onto the
    target mesh. ``white mesh``: a third of the incident face areas of the template's own
    white surface, which is smaller than real cortex by a factor that varies by region."""
    build_stamp: str | None
    """``scripts/build-stamp.txt`` of the source directory, if any."""
    parcel_names: tuple[str, ...]
    regions: RegionConfig
    hemis: Mapping[Hemi, TemplateHemi]

    def adjacency(self, hemi: Hemi) -> sparse.csr_matrix:
        """Vertex adjacency of the mesh: symmetric, 0/1 as float64, no diagonal, column
        indices sorted. Computed once per hemisphere; the matrix is shared, not a copy."""
        # a frozen dataclass refuses attribute assignment, not a write to its __dict__
        cache: dict[str, sparse.csr_matrix] = self.__dict__.setdefault("_adjacency", {})
        if hemi not in cache:
            half = self.hemis[hemi]
            cache[hemi] = _adjacency(half.faces, len(half.sphere))
        return cache[hemi]

    def __getstate__(self) -> dict[str, object]:
        # the cached matrices are not sent to a worker process with the template
        return {name: value for name, value in self.__dict__.items() if name != "_adjacency"}


class TemplateError(Exception):
    """A template directory cannot be used, or is not the one a run was scored with.

    ``reason`` is one of

    * ``not icosahedral``: the sphere is not an icosahedral mesh of at least the order
      asked for, or the white surface is another mesh;
    * ``vertex count mismatch``: a per-vertex file does not have the length of the sphere;
    * ``unreadable``: a file cannot be opened or is damaged;
    * ``missing parcel``: the annotation lacks a parcel that the region config names;
    * ``changed``: the template differs from the one the run is pinned to.

    ``detail`` names files relative to the template directory, never a path.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[TemplateError], tuple[str, str]]:
        return type(self), (self.reason, self.detail)


class TemplateNotFound(Exception):
    """No usable fsaverage directory was found.

    ``looked_in`` holds one ``(origin, where, state)`` per place that was looked at, in the
    order of the lookup. ``where`` is a path or a description such as ``bundled copy``;
    ``state`` is ``not set``, ``absent``, ``incomplete: <files>``, ``not shipped``, or
    ``unreadable: <error>`` for a directory that cannot be examined.
    """

    def __init__(self, looked_in: tuple[tuple[str, str, str], ...]) -> None:
        super().__init__("no fsaverage template found")
        self.looked_in = tuple(looked_in)

    def __reduce__(
        self,
    ) -> tuple[type[TemplateNotFound], tuple[tuple[tuple[str, str, str], ...]]]:
        return type(self), (self.looked_in,)


def coarsen_faces(faces: Int[np.ndarray, "f 3"], n_coarse: int) -> Int[np.ndarray, "g 3"]:
    """The faces of the icosahedral mesh one order below, from the faces of a mesh whose
    first ``n_coarse`` vertices are that mesh's vertices.

    Every vertex with index ``>= n_coarse`` must have exactly two neighbours below
    ``n_coarse`` (its parents), and the faces made of three such vertices must number
    ``2 * n_coarse - 4``; otherwise ``TemplateError("not icosahedral", ...)``. For such a
    face ``(m0, m1, m2)``, in input order, the coarse face is ``(a, b, c)`` with ``a`` the
    parent shared by ``m2`` and ``m0``, ``b`` by ``m0`` and ``m1``, ``c`` by ``m1`` and
    ``m2``. The result has the dtype of ``faces`` and the winding of its faces.

    Checked 2026-11-13: applied twice to the faces of the real fsaverage ``lh.sphere.reg``
    it gives the faces of ``fsaverage5/surf/lh.sphere.reg`` exactly, in their order.
    """
    return _coarsen(faces, n_coarse)[0]


def _coarsen(
    faces: Int[np.ndarray, "f 3"], n_coarse: int
) -> tuple[Int[np.ndarray, "g 3"], Int[np.ndarray, "m 2"]]:
    """:func:`coarsen_faces`, and the two parents of every vertex from ``n_coarse`` on
    (row ``i`` for vertex ``n_coarse + i``, the lower parent first)."""
    f = np.asarray(faces).astype(np.int64)
    if n_coarse < 3:
        raise TemplateError("not icosahedral", f"no mesh has {n_coarse} vertices")
    if f.size == 0 or int(f.min()) < 0:
        raise TemplateError("not icosahedral", "the mesh has no faces or a negative vertex index")
    n_fine = int(f.max()) + 1 - n_coarse
    if n_fine <= 0:
        raise TemplateError(
            "not icosahedral", f"the mesh has no vertex beyond the first {n_coarse}"
        )

    # every edge in both directions; kept where it leads from a new vertex to an old one
    start = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    end = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    down = (start >= n_coarse) & (end < n_coarse)
    pairs = np.unique((start[down] - n_coarse) * n_coarse + end[down])
    child, parent = np.divmod(pairs, n_coarse)
    n_parents = np.bincount(child, minlength=n_fine)
    if not (n_parents == 2).all():
        raise TemplateError(
            "not icosahedral",
            f"{int((n_parents != 2).sum())} of the {n_fine} vertices from {n_coarse} on do "
            "not have two neighbours below it",
        )
    parents = parent.reshape(n_fine, 2)  # sorted by child, then by parent

    middle = f[(f >= n_coarse).all(axis=1)]
    if len(middle) != 2 * n_coarse - 4:
        raise TemplateError(
            "not icosahedral",
            f"{len(middle)} faces lie between vertices from {n_coarse} on; a mesh of "
            f"{n_coarse} vertices has {2 * n_coarse - 4} faces",
        )
    p0, p1, p2 = (parents[middle[:, corner] - n_coarse] for corner in range(3))
    corners = []
    for x, y in ((p2, p0), (p0, p1), (p1, p2)):
        first = (x[:, 0] == y[:, 0]) | (x[:, 0] == y[:, 1])
        second = (x[:, 1] == y[:, 0]) | (x[:, 1] == y[:, 1])
        if not (first ^ second).all():
            raise TemplateError(
                "not icosahedral",
                f"{int((~(first ^ second)).sum())} faces have two corners that do not "
                "share exactly one parent",
            )
        corners.append(np.where(first, x[:, 0], x[:, 1]))
    coarse = np.column_stack(corners)
    if (
        (coarse[:, 0] == coarse[:, 1])
        | (coarse[:, 1] == coarse[:, 2])
        | (coarse[:, 2] == coarse[:, 0])
    ).any():
        raise TemplateError("not icosahedral", "a coarse face would have a corner twice")
    return coarse.astype(np.asarray(faces).dtype), parents


def template_candidates(
    *, adjacent: Path | None, env: Mapping[str, str] = os.environ
) -> list[tuple[Origin, Path | None]]:
    """The places a template is looked for when none is given, in the order of decision
    D14: ``$FREESURFER_HOME/subjects/fsaverage`` (None when the variable is not set or
    empty), ``adjacent``, the copy bundled with the package (None while none ships).

    ``adjacent`` is the ``fsaverage`` directory next to the subjects itself, not the
    subjects directory.
    """
    home = env.get("FREESURFER_HOME")
    return [
        ("freesurfer_home", Path(home) / "subjects" / "fsaverage" if home else None),
        ("adjacent", None if adjacent is None else Path(adjacent)),
        ("bundled", _BUNDLED_DIR if _BUNDLED_DIR.is_dir() else None),
    ]


def build_template(
    directory: Path,
    origin: Origin,
    *,
    mesh: MeshName = "fsaverage6",
    regions: RegionConfig = _DEFAULT_REGIONS,
) -> Template:
    """Derive the template on ``mesh`` from an fsaverage directory.

    Per hemisphere, from the files of ``TEMPLATE_FILES``:

    * ``sphere``: ``sphere.reg`` scaled to unit length, its first vertices; ``faces``:
      :func:`coarsen_faces` applied once per order between the file's mesh and ``mesh``;
    * ``white``, ``parcel`` (``aparc.annot``): the first entries; ``cortex``: the vertices
      that ``cortex.label`` lists, below the vertex count;
    * ``pole_vertex``: the cortex vertex with the smallest white y, lowest index on a tie;
    * regions, all inside ``cortex``: ``search`` is the ``occipital`` parcels plus the
      part of the ``posterior`` parcels whose white y is at most ``cut_mm`` in front of
      the pole's; ``medial`` and ``inferior`` are ``search`` within those parcel lists,
      ``lateral`` is the rest of ``search``; ``control`` is the ``control`` parcels;
    * ``area``: when both hemispheres have ``surf/?h.white.avg.area.mgh``, its values
      brought to the target mesh one order at a time: a vertex keeps its own value and
      gets half the value of every vertex that was added between it and a neighbour. The
      total is the file's. Otherwise a third of the areas of the faces around the vertex
      on the white surface of the target mesh.

    Why halves. A vertex of the full mesh that the target mesh lacks lies between two
    target vertices, and its patch of cortex belongs to both. Giving all of it to the
    nearer one, as a nearest-vertex rule does, decides by a difference in distance of half
    a percent (median on the real fsaverage; 2.9% at most) and leaves a target vertex with
    one to seven patches where four are its share: on the real left hemisphere the areas
    then range from 0.21 to 7.65 mm2 and differ between neighbours by a factor of 1.67
    (one SD), against 0.66 to 5.59 mm2 and a factor of 1.09 with halves (measured
    2026-11-13). Over several orders the halves multiply out to the weights of linear
    interpolation in the target face that holds the vertex.

    Parcel indices of both hemispheres refer to one list of names: those of the left
    annotation, followed by any that only the right one has.

    Raises ``TemplateNotFound`` for a directory that lacks a required file, and
    ``TemplateError`` with the reasons listed there for one that cannot be used.
    """
    directory = Path(directory)
    state = _state(directory)
    if state is not None:
        raise TemplateNotFound(((origin, _where(origin, directory), state),))
    return _build(directory, origin, mesh, regions, _source_files(directory))


def write_template(directory: Path, template: Template) -> None:
    """Write ``template.json`` and ``template.npz`` into ``directory``. Neither holds a path.

    ``template.json`` is removed first and written last, so its presence marks a whole
    template. The bytes of both files depend on the template alone.
    """
    directory = Path(directory)
    arrays: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        half = template.hemis[hemi]
        arrays[f"{hemi}_sphere"] = half.sphere.astype("<f8")
        arrays[f"{hemi}_faces"] = half.faces.astype("<i4")
        arrays[f"{hemi}_white"] = half.white.astype("<f4")
        arrays[f"{hemi}_cortex"] = half.cortex.astype("u1")
        arrays[f"{hemi}_parcel"] = half.parcel.astype("<i2")
        arrays[f"{hemi}_area"] = half.area.astype("<f4")
        for name in REGION_NAMES:
            arrays[f"{hemi}_region_{name}"] = half.regions[name].astype("u1")
    info = _TemplateFile.model_validate(
        {
            "schema": TEMPLATE_SCHEMA,
            "template_id": template.template_id,
            "mesh": template.mesh,
            "order": template.order,
            "origin": template.origin,
            "area_source": template.area_source,
            "build_stamp": template.build_stamp,
            "parcel_names": template.parcel_names,
            "regions": template.regions,
            "files": {
                name: {"size": size, "sha256": digest}
                for name, (size, digest) in template.files.items()
            },
            "pole_vertex": {hemi: template.hemis[hemi].pole_vertex for hemi in HEMIS},
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TEMPLATE_JSON).unlink(missing_ok=True)
    write_npz(directory / TEMPLATE_NPZ, arrays)
    with atomic_path(directory / TEMPLATE_JSON) as tmp:
        tmp.write_bytes(info.to_json().encode())


def read_template(directory: Path) -> Template:
    """Read a template written by :func:`write_template`; equal to the one written.

    Raises ``OSError`` when one of the two files is absent or cannot be opened, and
    ``ValueError`` when they are not a template of this schema: damaged, written by another
    version, a ``template.json`` whose content does not hash to its ``template_id``, or
    arrays that do not have the shapes of its mesh.
    """
    directory = Path(directory)
    raw = (directory / TEMPLATE_JSON).read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{TEMPLATE_JSON} is not JSON: {exc}") from exc
    version = data.get("schema") if isinstance(data, dict) else None
    if version != TEMPLATE_SCHEMA:
        raise ValueError(
            f"{TEMPLATE_JSON} has schema {version!r}; this surfcensus reads {TEMPLATE_SCHEMA}"
        )
    try:
        info = _TemplateFile.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ValueError(f"{TEMPLATE_JSON}: {where}: {first['msg']}") from None
    files = {name: (entry.size, entry.sha256) for name, entry in info.files.items()}
    expected = _template_id(info.mesh, info.regions, files)
    if info.order != MESH_ORDER[info.mesh] or info.template_id != expected:
        raise ValueError(
            f"{TEMPLATE_JSON}: its content is not that of the template {info.template_id}"
        )
    try:
        arrays = read_npz(directory / TEMPLATE_NPZ)
    except OSError:
        raise
    except Exception as exc:  # zipfile and numpy raise several kinds of error
        raise ValueError(f"{TEMPLATE_NPZ} cannot be read: {exc}") from exc

    n, n_faces = icosahedron_vertices(info.order), 20 * 4**info.order
    layout: dict[str, tuple[str, tuple[int, ...]]] = {}
    for hemi in HEMIS:
        layout[f"{hemi}_sphere"] = ("f8", (n, 3))
        layout[f"{hemi}_faces"] = ("i4", (n_faces, 3))
        layout[f"{hemi}_white"] = ("f4", (n, 3))
        layout[f"{hemi}_cortex"] = ("u1", (n,))
        layout[f"{hemi}_parcel"] = ("i2", (n,))
        layout[f"{hemi}_area"] = ("f4", (n,))
        for name in REGION_NAMES:
            layout[f"{hemi}_region_{name}"] = ("u1", (n,))
    if set(arrays) != set(layout):
        raise ValueError(f"{TEMPLATE_NPZ} does not hold the arrays of a template")
    for key, (kind, shape) in layout.items():
        found = arrays[key]
        if found.dtype.str.lstrip("<>|=") != kind or found.shape != shape:
            raise ValueError(
                f"{TEMPLATE_NPZ}: {key} is {found.dtype.str} {found.shape}, not {kind} {shape}"
            )
    hemis: dict[Hemi, TemplateHemi] = {}
    for hemi in HEMIS:
        pole = info.pole_vertex[hemi]
        if not 0 <= pole < n:
            raise ValueError(f"{TEMPLATE_JSON}: the {hemi} pole vertex {pole} is not on the mesh")
        hemis[hemi] = _frozen_hemi(
            sphere=arrays[f"{hemi}_sphere"].astype(np.float64),
            faces=arrays[f"{hemi}_faces"].astype(np.int32),
            white=arrays[f"{hemi}_white"].astype(np.float32),
            cortex=arrays[f"{hemi}_cortex"].astype(bool),
            parcel=arrays[f"{hemi}_parcel"].astype(np.int16),
            area=arrays[f"{hemi}_area"].astype(np.float32),
            regions={name: arrays[f"{hemi}_region_{name}"].astype(bool) for name in REGION_NAMES},
            pole_vertex=pole,
        )
    return Template(
        mesh=info.mesh,
        order=info.order,
        template_id=info.template_id,
        origin=info.origin,
        files=files,
        area_source=info.area_source,
        build_stamp=info.build_stamp,
        parcel_names=info.parcel_names,
        regions=info.regions,
        hemis=hemis,
    )


def store_template(cache_dir: Path | str, template: Template) -> Path:
    """The directory of ``template`` in a cache (``<cache_dir>/<template_id>``), written
    first if it is not there. Worker processes read the template from it."""
    directory = Path(cache_dir) / template.template_id
    if not (directory / TEMPLATE_JSON).is_file():
        write_template(directory, template)
    return directory


@functools.lru_cache(maxsize=2)
def load_template(directory: str) -> Template:
    """:func:`read_template`, once per process and directory: a worker is sent the
    directory of the run's template, not its 6 MB with every subject. The directory is
    named by the template's id, so what it holds does not change."""
    return read_template(Path(directory))


def find_template(
    *,
    config: MetricsConfig,
    adjacent: Path | None,
    cache_dir: Path | None = None,
    pinned: str | None = None,
    retemplate: bool = False,
    env: Mapping[str, str] = os.environ,
) -> Template:
    """The template of a run, on ``config.mesh`` with ``config.regions``.

    Nothing pinned, or ``retemplate``: ``config.fsaverage_dir`` when it is set, without
    falling through when it cannot be used; else the first usable directory of
    :func:`template_candidates`, where usable means that it holds every file of
    ``TEMPLATE_FILES``. ``TemplateNotFound`` lists every place that was looked at.

    ``pinned`` (the id a run was first scored with) and no ``retemplate``: the template in
    ``cache_dir/<pinned>/``, whatever the environment looks like now. When that cannot be
    read, the template is built again from the explicit directory or the first candidate
    whose files give the pinned id. ``TemplateError("changed", ...)`` when

    * ``config.fsaverage_dir`` is usable and its files give another id;
    * the cached template is on another mesh or has other regions than ``config``;
    * the cache is gone and no directory gives the pinned id: a run that has scores never
      loses its template silently.

    An explicit directory that cannot be used while a template is pinned is passed over
    with one warning. Nothing is written: the caller stores the template and the pin.
    """
    explicit = Path(config.fsaverage_dir) if config.fsaverage_dir else None
    if pinned is not None and not retemplate:
        return _find_pinned(pinned, config, explicit, adjacent, cache_dir, env)
    if explicit is not None:
        return build_template(explicit, "explicit", mesh=config.mesh, regions=config.regions)
    looked_in: list[tuple[str, str, str]] = []
    for origin, directory in template_candidates(adjacent=adjacent, env=env):
        if directory is None:
            looked_in.append((origin, *_NOT_GIVEN[origin]))
            continue
        state = _state(directory)
        if state is None:
            return build_template(directory, origin, mesh=config.mesh, regions=config.regions)
        looked_in.append((origin, _where(origin, directory), state))
    raise TemplateNotFound(tuple(looked_in))


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class _SourceFile(_Model):
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _TemplateFile(_Model):
    """``template.json``."""

    schema_version: int = Field(alias="schema")
    template_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    mesh: MeshName
    order: int
    origin: Origin
    area_source: Literal["white.avg.area", "white mesh"]
    build_stamp: str | None
    parcel_names: tuple[str, ...]
    regions: RegionConfig
    files: dict[str, _SourceFile]
    pole_vertex: dict[Hemi, int]

    def to_json(self) -> str:
        return (
            json.dumps(self.model_dump(mode="json", by_alias=True), indent=1, sort_keys=True) + "\n"
        )


def _is_dir(path: Path) -> bool:
    """Whether ``path`` is a directory or a link to one; a dangling link is not. Raises
    OSError when the path cannot be examined, on every Python version."""
    try:
        return S_ISDIR(path.stat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _state(directory: Path) -> str | None:
    """None for a directory that holds every required file, else what is wrong with it, in
    the words of ``TemplateNotFound.looked_in``."""
    try:
        if not _is_dir(directory):
            return "absent"
        missing = [name for name in _REQUIRED if not is_file(directory / name)]
    except OSError as exc:
        return f"unreadable: {exc.strerror or type(exc).__name__}"
    return "incomplete: " + ", ".join(missing) if missing else None


def _where(origin: Origin, directory: Path) -> str:
    return _NOT_GIVEN[origin][0] if origin == "bundled" else str(directory)


@contextmanager
def _reading(directory: Path, name: str) -> Iterator[None]:
    """Turn any failure of reading ``directory/name`` into ``TemplateError("unreadable")``
    that names the file relative to the directory, on one line and without a path."""
    try:
        yield
    except TemplateError:
        raise
    except Exception as exc:  # nibabel raises many kinds of error on a damaged file
        if isinstance(exc, OSError) and exc.strerror:
            problem = exc.strerror
        else:
            problem = str(exc)
            for path in (directory / name, directory):
                problem = problem.replace(str(path), path.name)
            # the readers of fsio start their messages with the file's name
            problem = " ".join(problem.removeprefix(f"{Path(name).name}: ").split())
            if not isinstance(exc, OSError | ValueError):
                problem = f"{type(exc).__name__}: {problem}" if problem else type(exc).__name__
        raise TemplateError("unreadable", f"{name}: {problem}") from exc


def _source_files(directory: Path) -> dict[str, tuple[int, str]]:
    """``(size, SHA-256)`` of the files a template is built from: the required ones, and
    the average-area files when both hemispheres have one."""
    names = list(_REQUIRED)
    optional = [_AVG_AREA.format(h=hemi) for hemi in HEMIS]
    with _reading(directory, optional[0]):
        if all(is_file(directory / name) for name in optional):
            names += optional
    files: dict[str, tuple[int, str]] = {}
    for name in names:
        with _reading(directory, name):
            digest = sha256_file(directory / name)
            files[name] = ((directory / name).stat().st_size, digest)
    return files


def _template_id(
    mesh: MeshName, regions: RegionConfig, files: Mapping[str, tuple[int, str]]
) -> str:
    payload = {
        "schema": TEMPLATE_SCHEMA,
        "mesh": mesh,
        "order": MESH_ORDER[mesh],
        "regions": regions.model_dump(mode="json"),
        "files": {name: digest for name, (_, digest) in files.items()},
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]


def _build(
    directory: Path,
    origin: Origin,
    mesh: MeshName,
    regions: RegionConfig,
    files: Mapping[str, tuple[int, str]],
) -> Template:
    with_area = all(_AVG_AREA.format(h=hemi) in files for hemi in HEMIS)
    built = {hemi: _build_hemi(directory, hemi, mesh, regions, with_area) for hemi in HEMIS}
    names = list(built["lh"][1])
    names += [name for name in built["rh"][1] if name not in names]
    hemis: dict[Hemi, TemplateHemi] = {}
    for hemi in HEMIS:
        half, own = built[hemi]
        if tuple(names[: len(own)]) != own:
            # the right annotation orders its names differently or has others
            to_shared = np.array([names.index(name) for name in own], dtype=np.int16)
            parcel = np.where(half.parcel >= 0, to_shared[half.parcel], -1).astype(np.int16)
            parcel.setflags(write=False)
            half = replace(half, parcel=parcel)
        hemis[hemi] = half
    return Template(
        mesh=mesh,
        order=MESH_ORDER[mesh],
        template_id=_template_id(mesh, regions, files),
        origin=origin,
        files=dict(files),
        area_source="white.avg.area" if with_area else "white mesh",
        build_stamp=_read_build_stamp(directory),
        parcel_names=tuple(names),
        regions=regions,
        hemis=hemis,
    )


def _read_build_stamp(directory: Path) -> str | None:
    """The first line of ``scripts/build-stamp.txt``; None when there is no such file or it
    is empty."""
    with _reading(directory, _BUILD_STAMP):
        if not is_file(directory / _BUILD_STAMP):
            return None
        text = (directory / _BUILD_STAMP).read_bytes().decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else None


def _icosahedral_order(n_vertices: int) -> int | None:
    return next((k for k in range(_MAX_ORDER + 1) if icosahedron_vertices(k) == n_vertices), None)


def _build_hemi(
    directory: Path, hemi: Hemi, mesh: MeshName, regions: RegionConfig, with_area: bool
) -> tuple[TemplateHemi, tuple[str, ...]]:
    """One hemisphere on the target mesh, and the parcel names its ``parcel`` indexes."""
    order = MESH_ORDER[mesh]
    n = icosahedron_vertices(order)
    sphere_name, white_name = _SPHERE.format(h=hemi), _WHITE.format(h=hemi)
    cortex_name, annot_name = _CORTEX.format(h=hemi), _ANNOT.format(h=hemi)

    with _reading(directory, sphere_name):
        sphere = read_surface(directory / sphere_name)
    n_full = len(sphere.vertices)
    full_order = _icosahedral_order(n_full)
    if full_order is None or full_order < order:
        raise TemplateError(
            "not icosahedral",
            f"{sphere_name} has {n_full} vertices; {mesh} needs 10 * 4**k + 2 with k >= {order}",
        )
    faces = sphere.faces
    if len(faces) != 20 * 4**full_order or faces.min() < 0 or faces.max() >= n_full:
        raise TemplateError(
            "not icosahedral",
            f"{sphere_name} has {len(faces)} faces for {n_full} vertices, or a face that "
            f"names no vertex; an icosahedral mesh has {20 * 4**full_order}",
        )
    steps: list[np.ndarray] = []  # the parents of the vertices dropped at each step
    try:
        unit = np.ascontiguousarray(unit_sphere(sphere.vertices)[:n])
        if full_order == order:
            # no step to take: one is taken and dropped, which shows that the mesh nests
            coarsen_faces(faces, icosahedron_vertices(order - 1))
        for step in range(full_order, order, -1):
            faces, parents = _coarsen(faces, icosahedron_vertices(step - 1))
            steps.append(parents)
    except (TemplateError, ValueError) as exc:
        detail = exc.detail if isinstance(exc, TemplateError) else str(exc)
        raise TemplateError("not icosahedral", f"{sphere_name}: {detail}") from None
    faces = _wound_outward(unit, faces.astype(np.int32))

    with _reading(directory, white_name):
        white_surface = read_surface(directory / white_name)
    if len(white_surface.vertices) != n_full:
        raise TemplateError(
            "vertex count mismatch",
            f"{white_name} has {len(white_surface.vertices)} vertices and {sphere_name} {n_full}",
        )
    if not np.array_equal(white_surface.faces, sphere.faces):
        raise TemplateError(
            "not icosahedral", f"{white_name} does not have the faces of {sphere_name}"
        )
    white = white_surface.vertices[:n].astype(np.float32)
    if not np.isfinite(white).all():
        raise TemplateError("unreadable", f"{white_name}: a coordinate is not finite")

    with _reading(directory, cortex_name):
        listed = read_label(directory / cortex_name)
    if listed.size and int(listed[-1]) >= n_full:
        raise TemplateError(
            "vertex count mismatch",
            f"{cortex_name} lists vertex {int(listed[-1])} and {sphere_name} has {n_full}",
        )
    cortex = np.zeros(n, dtype=bool)
    cortex[listed[listed < n]] = True
    if not cortex.any():
        raise TemplateError("unreadable", f"{cortex_name}: lists no vertex of {mesh}")

    with _reading(directory, annot_name):
        annot = read_annotation(directory / annot_name)
    if len(annot.labels) != n_full:
        raise TemplateError(
            "vertex count mismatch",
            f"{annot_name} has {len(annot.labels)} labels and {sphere_name} {n_full} vertices",
        )
    wanted = (
        *regions.occipital,
        *regions.posterior,
        *regions.medial,
        *regions.inferior,
        *regions.control,
    )
    missing = [name for name in dict.fromkeys(wanted) if name not in annot.names]
    if missing:
        raise TemplateError("missing parcel", f"{annot_name} has no parcel {', '.join(missing)}")
    parcel = annot.labels[:n].astype(np.int16)

    def named(names: tuple[str, ...]) -> np.ndarray:
        return np.isin(parcel, [annot.names.index(name) for name in names]) & cortex

    white_y = white[:, 1].astype(np.float64)
    in_cortex = np.flatnonzero(cortex)
    pole = int(in_cortex[np.argmin(white_y[in_cortex])])
    search = named(regions.occipital) | (
        named(regions.posterior) & (white_y <= white_y[pole] + regions.cut_mm)
    )
    medial = search & named(regions.medial)
    inferior = search & named(regions.inferior)
    region_masks: dict[RegionName, np.ndarray] = {
        "search": search,
        "medial": medial,
        "inferior": inferior,
        "lateral": search & ~medial & ~inferior,
        "control": named(regions.control),
    }

    if with_area:
        area_name = _AVG_AREA.format(h=hemi)
        with _reading(directory, area_name):
            image = nib.load(str(directory / area_name))
            values = np.asarray(image.dataobj, dtype=np.float64).ravel()
        if len(values) != n_full:
            raise TemplateError(
                "vertex count mismatch",
                f"{area_name} has {len(values)} values and {sphere_name} {n_full} vertices",
            )
        if not (np.isfinite(values).all() and (values >= 0).all()):
            raise TemplateError("unreadable", f"{area_name}: an area is negative or not finite")
        area = values
        for parents in steps:
            kept = len(area) - len(parents)
            half = area[kept:] / 2.0
            area = (
                area[:kept]
                + np.bincount(parents[:, 0], half, minlength=kept)
                + np.bincount(parents[:, 1], half, minlength=kept)
            )
    else:
        area = _vertex_areas(white.astype(np.float64), faces)

    half = _frozen_hemi(
        sphere=unit,
        faces=faces,
        white=white,
        cortex=cortex,
        parcel=parcel,
        area=area.astype(np.float32),
        regions=region_masks,
        pole_vertex=pole,
    )
    return half, annot.names


def _frozen_hemi(
    *,
    sphere: np.ndarray,
    faces: np.ndarray,
    white: np.ndarray,
    cortex: np.ndarray,
    parcel: np.ndarray,
    area: np.ndarray,
    regions: Mapping[RegionName, np.ndarray],
    pole_vertex: int,
) -> TemplateHemi:
    """A :class:`TemplateHemi` whose arrays are contiguous and read-only."""

    def fixed(array: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(array)
        array.setflags(write=False)
        return array

    return TemplateHemi(
        sphere=fixed(sphere),
        faces=fixed(faces),
        white=fixed(white),
        cortex=fixed(cortex),
        parcel=fixed(parcel),
        area=fixed(area),
        regions={name: fixed(regions[name]) for name in REGION_NAMES},
        pole_vertex=pole_vertex,
    )


def _wound_outward(
    unit: Float[np.ndarray, "v 3"], faces: Int[np.ndarray, "f 3"]
) -> Int[np.ndarray, "f 3"]:
    """``faces``, with every face reversed when most of them are wound inward. FreeSurfer's
    spheres are wound outward; a sphere written by another tool may not be."""
    a, b, c = (unit[faces[:, corner]] for corner in range(3))
    inward = int(((a * np.cross(b, c)).sum(axis=1) < 0).sum())
    return faces[:, [0, 2, 1]] if 2 * inward > len(faces) else faces


def _vertex_areas(
    vertices: Float[np.ndarray, "v 3"], faces: Int[np.ndarray, "f 3"]
) -> Float[np.ndarray, " v"]:
    """Per vertex, a third of the areas of the faces around it."""
    a, b, c = (vertices[faces[:, corner]] for corner in range(3))
    third = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 6.0
    area = np.zeros(len(vertices))
    for corner in range(3):
        area += np.bincount(faces[:, corner], third, minlength=len(vertices))
    return area


def _adjacency(faces: Int[np.ndarray, "f 3"], n_vertices: int) -> sparse.csr_matrix:
    f = faces.astype(np.int64)
    start = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    end = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    pairs = np.unique(start[start != end] * n_vertices + end[start != end])
    rows, columns = np.divmod(pairs, n_vertices)
    matrix = sparse.csr_matrix(
        (np.ones(len(pairs)), (rows, columns)), shape=(n_vertices, n_vertices)
    )
    matrix.sort_indices()
    return matrix


def _find_pinned(
    pinned: str,
    config: MetricsConfig,
    explicit: Path | None,
    adjacent: Path | None,
    cache_dir: Path | None,
    env: Mapping[str, str],
) -> Template:
    """The template a run is pinned to; see :func:`find_template`. A directory is compared
    with the pin by the digests of its files, so nothing is built to find out."""
    cached = _cached(cache_dir, pinned)
    if cached is not None and (cached.mesh != config.mesh or cached.regions != config.regions):
        differs = [f"the mesh ({cached.mesh}, now {config.mesh})"] * (cached.mesh != config.mesh)
        differs += ["the regions"] * (cached.regions != config.regions)
        raise TemplateError(
            "changed",
            f"the configuration no longer matches the run's template {pinned} in "
            f"{' and '.join(differs)}; pass --retemplate",
        )
    if explicit is not None:
        state = _state(explicit)
        if state is None:
            files = _source_files(explicit)
            found = _template_id(config.mesh, config.regions, files)
            if found != pinned:
                raise TemplateError("changed", f"{pinned} -> {found}; pass --retemplate")
            if cached is None:
                return _build(explicit, "explicit", config.mesh, config.regions, files)
        else:
            get_logger().warning(
                "The fsaverage directory that was given cannot be used (%s); %s",
                state,
                f"the run keeps its template {pinned}."
                if cached is not None
                else f"looking for the run's template {pinned} elsewhere.",
            )
    if cached is not None:
        return cached
    for origin, directory in template_candidates(adjacent=adjacent, env=env):
        if directory is None or _state(directory) is not None:
            continue
        files = _source_files(directory)
        if _template_id(config.mesh, config.regions, files) == pinned:
            return _build(directory, origin, config.mesh, config.regions, files)
    raise TemplateError("changed", _GONE)


def _cached(cache_dir: Path | None, template_id: str) -> Template | None:
    """The template ``template_id`` of the cache; None when it is not there or cannot be
    read, in which case it is built again from its source."""
    if cache_dir is None or not _ID.fullmatch(template_id):
        return None
    try:
        template = read_template(Path(cache_dir) / template_id)
    except (OSError, ValueError):
        return None
    return template if template.template_id == template_id else None
