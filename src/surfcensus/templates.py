"""The template: everything the metrics need on the target mesh, derived at run time from a
FreeSurfer ``fsaverage`` directory. See docs/metrics.md and decision D14.

Nothing derived from fsaverage ships with the package. A template is built from a directory
that holds the files of ``sources.patterns.TEMPLATE_FILES`` (found in a FreeSurfer install,
next to the subjects, or given explicitly), identified by the content of those files, and
cached in the run directory under ``templates/<template_id>/``.

Interface of milestone 3. The types, constants and error classes are final; the functions
are written by package 3.F.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy import sparse

from surfcensus.config import MeshName, MetricsConfig, RegionConfig
from surfcensus.sources.base import Hemi
from surfcensus.sources.patterns import TEMPLATE_FILES, TEMPLATE_OPTIONAL_FILES

__all__ = [
    "MESH_ORDER",
    "REGION_NAMES",
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
    "read_template",
    "template_candidates",
    "write_template",
]

_OWNER = "package 3.F (template)"

TEMPLATE_SCHEMA = 1
"""Hashed into every ``template_id``: raised when the derivation changes for equal files."""
TEMPLATE_JSON = "template.json"
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


def icosahedron_vertices(order: int) -> int:
    """Vertices of the icosahedral mesh of ``order``: ``10 * 4**order + 2`` (40,962 for
    order 6, 163,842 for order 7)."""
    if order < 0:
        raise ValueError(f"an icosahedral order is not negative, got {order}")
    return 10 * 4**order + 2


@dataclass(frozen=True)
class TemplateHemi:
    """One hemisphere of the template on the target mesh; ``v`` vertices, ``f`` faces."""

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
    """Relative path -> (size, SHA-256) of the source files that were read."""
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
        """Vertex adjacency of the mesh: symmetric, 0/1, no diagonal. Computed once per
        hemisphere."""
        raise NotImplementedError(f"templates.Template.adjacency: {_OWNER}")


class TemplateError(Exception):
    """A template directory cannot be used, or is not the one a run was scored with.

    ``reason`` is one of ``not icosahedral``, ``missing parcel``, ``changed``.
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
    ``state`` is ``not set``, ``absent``, ``incomplete: <files>`` or ``not shipped``.
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
    ``m2``.
    """
    raise NotImplementedError(f"templates.coarsen_faces: {_OWNER}")


def template_candidates(
    *, adjacent: Path | None, env: Mapping[str, str] = os.environ
) -> list[tuple[Origin, Path | None]]:
    """The places a template is looked for when none is given, in the order of decision
    D14: ``$FREESURFER_HOME/subjects/fsaverage`` (None when the variable is not set),
    ``adjacent``, the copy bundled with the package (None while none ships)."""
    raise NotImplementedError(f"templates.template_candidates: {_OWNER}")


def build_template(
    directory: Path,
    origin: Origin,
    *,
    mesh: MeshName = "fsaverage6",
    regions: RegionConfig = _DEFAULT_REGIONS,
) -> Template:
    """Derive the template on ``mesh`` from an fsaverage directory.

    Raises ``TemplateNotFound`` for a directory that lacks a required file,
    ``TemplateError("not icosahedral", ...)`` for a sphere that is not an icosahedral mesh
    of at least the order of ``mesh``, and ``TemplateError("missing parcel", ...)`` when the
    annotation lacks a parcel that ``regions`` names.
    """
    raise NotImplementedError(f"templates.build_template: {_OWNER}")


def write_template(directory: Path, template: Template) -> None:
    """Write ``template.json`` and ``template.npz`` into ``directory``. Neither holds a path."""
    raise NotImplementedError(f"templates.write_template: {_OWNER}")


def read_template(directory: Path) -> Template:
    """Read a template written by :func:`write_template`; equal to the one written."""
    raise NotImplementedError(f"templates.read_template: {_OWNER}")


def find_template(
    *,
    config: MetricsConfig,
    adjacent: Path | None,
    cache_dir: Path | None = None,
    pinned: str | None = None,
    retemplate: bool = False,
    env: Mapping[str, str] = os.environ,
) -> Template:
    """The template of a run.

    In order: the template with id ``pinned`` in ``cache_dir``; ``config.fsaverage_dir``
    when nothing is pinned or ``retemplate``, without falling through when it cannot be
    used; then :func:`template_candidates`. A pinned template is kept whatever the
    environment looks like later: an explicit directory that builds another id raises
    ``TemplateError("changed", ...)`` unless ``retemplate``.

    Raises ``TemplateNotFound`` with every place looked at, and ``TemplateError``.
    """
    raise NotImplementedError(f"templates.find_template: {_OWNER}")
