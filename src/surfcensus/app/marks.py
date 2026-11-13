"""Click marks: from a point on a tile to voxels, tkr RAS, scanner RAS and vertices. See
docs/rater_app.md, "Marking errors by clicking".

Pure functions on a ``RenderInfo`` and a vertex map: no database, no web framework. The
shapes ``MarkIn`` and ``ResolvedMark`` are in ``app.schemas``.

The chain, all of it computed here from what the page posts (panel, slice, tile pixel):

* tile pixel to continuous voxel coordinates, through the panel's ``pix2vox`` and the voxel
  index of the slice (``render.geometry.pixel_to_voxel``);
* voxel to tkr RAS and to scanner RAS, through the two affines of ``render.json``;
* tkr RAS to the nearest native vertex of ``vertex_map.npz``, and from that vertex to its
  template vertex.

Distances are computed here, in float64, from the positions of the vertex map; the KD-trees
only find candidates. So a vertex and its distance do not depend on how the tree was built,
and equally near vertices are ordered by rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial import cKDTree

from surfcensus.app.schemas import MarkIn, ResolvedMark, Surface
from surfcensus.render import geometry
from surfcensus.render.model import PANEL_ORDER, RenderInfo, Vec3
from surfcensus.sources.base import HEMIS, Hemi

MarkField = Literal["panel", "slice", "px", "py", "frame"]
MARK_FIELDS: tuple[MarkField, ...] = ("panel", "slice", "px", "py", "frame")

SURFACES: tuple[Surface, Surface] = ("pial", "white")

_SLACK = 1e-9
"""A tree is asked for a ball this fraction (and this many mm) wider than the one that is
wanted, and the exact distances decide: the tree's own distance of a vertex can differ from
the one computed here in the last bits."""


class MarkError(ValueError):
    """A mark that does not fit the render it was made on. ``field`` names the part of the
    mark that is wrong, ``message`` says how; the server answers 422 with both."""

    def __init__(self, field: MarkField, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message

    def __reduce__(self) -> tuple[type[MarkError], tuple[str, str]]:
        return type(self), (self.field, self.message)


@dataclass(frozen=True)
class _HemiIndex:
    """The vertices of one hemisphere in a vertex map, row by row."""

    native: Int[np.ndarray, " k"]
    fsavg: Int[np.ndarray, " k"]
    positions: Mapping[Surface, Float[np.ndarray, "k 3"]]
    trees: Mapping[Surface, cKDTree]
    """Empty for a hemisphere without vertices."""

    def search(
        self, surface: Surface, point: Float[np.ndarray, " 3"], radius: float | None = None
    ) -> tuple[Int[np.ndarray, " r"], Float[np.ndarray, " r"]]:
        """Rows within ``radius`` of ``point`` (boundary included) and their distances, by
        distance and then by native index. Without a radius: the nearest rows, which is
        more than one row where several are equally near."""
        tree = self.trees.get(surface)
        if tree is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        # the tree names one of the nearest vertices, at about that distance
        reach = float(tree.query(point)[0]) if radius is None else radius
        found = tree.query_ball_point(point, reach * (1.0 + _SLACK) + _SLACK)
        rows = np.asarray(found, dtype=np.int64)
        # written out per axis: the sum of three squares in one fixed order
        d = self.positions[surface][rows] - point
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        keep = dist <= (dist.min() if radius is None else radius)
        rows, dist = rows[keep], dist[keep]
        order = np.lexsort((self.native[rows], dist))
        return rows[order], dist[order]


class VertexIndex:
    """KD-trees over the pial and the white positions of one ``vertex_map.npz``, per
    hemisphere. A result is ``(hemi, native vertex, distance in mm, template vertex or -1)``.

    The object is not changed after it is made, so several threads can ask it at once.
    """

    def __init__(self, vmap: Mapping[str, np.ndarray]) -> None:
        """``vmap``: the arrays of ``render.vertexmap.read_vertex_map``. ValueError when an
        array is missing, has another shape than its place in a vertex map, or holds a
        position that is not finite."""
        hemis: dict[Hemi, _HemiIndex] = {}
        for hemi in HEMIS:
            arrays = {}
            for field in ("index", "pial", "white", "fsavg6"):
                key = f"{hemi}_{field}"
                if key not in vmap:
                    raise ValueError(f"the vertex map has no array {key}")
                arrays[field] = np.asarray(vmap[key])
            native, fsavg = arrays["index"], arrays["fsavg6"]
            n = native.shape[0] if native.ndim == 1 else -1
            for field in ("index", "fsavg6"):
                if arrays[field].shape != (n,) or arrays[field].dtype.kind not in "iu":
                    raise ValueError(f"{hemi}_{field} of the vertex map is no list of vertices")
            positions: dict[Surface, np.ndarray] = {}
            for surface in SURFACES:
                xyz = arrays[surface]
                if xyz.shape != (n, 3) or xyz.dtype.kind != "f" or not np.isfinite(xyz).all():
                    raise ValueError(
                        f"{hemi}_{surface} of the vertex map does not hold one finite "
                        f"position for each of its {max(n, 0)} vertices"
                    )
                # float32 in the file, as in the surface files: exact in float64
                positions[surface] = np.ascontiguousarray(xyz, dtype=np.float64)
                positions[surface].setflags(write=False)
            hemis[hemi] = _HemiIndex(
                native=native.astype(np.int64),
                fsavg=fsavg.astype(np.int64),
                positions=positions,
                trees={s: cKDTree(positions[s]) for s in SURFACES} if n else {},
            )
        self._hemis = hemis

    @property
    def n_vertices(self) -> dict[Hemi, int]:
        """How many vertices of each hemisphere the vertex map holds."""
        return {hemi: len(half.native) for hemi, half in self._hemis.items()}

    @property
    def has_counterparts(self) -> bool:
        """Whether any vertex has a template vertex (a render with a template)."""
        return any(bool((half.fsavg >= 0).any()) for half in self._hemis.values())

    def nearest(
        self, surface: Surface, tkr: Vec3, max_dist_mm: float, *, hemi: Hemi | None = None
    ) -> tuple[Hemi, int, float, int] | None:
        """The vertex of ``surface`` nearest ``tkr`` within ``max_dist_mm`` (boundary
        included), or None. ``hemi=None`` searches both hemispheres: the nearer wins, ``lh``
        on an exact tie. Within a hemisphere the lowest native index wins a tie."""
        point, radius = _point(tkr), _radius(max_dist_mm)
        best: tuple[Hemi, int, float, int] | None = None
        for name in HEMIS if hemi is None else (hemi,):
            half = self._hemis[name]
            rows, dists = half.search(surface, point)
            if not len(rows):
                continue
            row, dist = int(rows[0]), float(dists[0])
            if dist <= radius and (best is None or dist < best[2]):
                best = (name, int(half.native[row]), dist, int(half.fsavg[row]))
        return best

    def within(
        self, surface: Surface, tkr: Vec3, radius_mm: float, *, hemi: Hemi | None = None
    ) -> list[tuple[Hemi, int, float, int]]:
        """Every vertex of ``surface`` within ``radius_mm`` of ``tkr`` (boundary included),
        by ascending distance, then ``lh`` before ``rh``, then native index."""
        point, radius = _point(tkr), _radius(radius_mm)
        out: list[tuple[Hemi, int, float, int]] = []
        for name in HEMIS if hemi is None else (hemi,):
            half = self._hemis[name]
            rows, dist = half.search(surface, point, radius)
            out.extend(
                (name, int(half.native[row]), float(d), int(half.fsavg[row]))
                for row, d in zip(rows.tolist(), dist.tolist(), strict=True)
            )
        # within a hemisphere the rows are in order already, and the sort is stable
        out.sort(key=lambda found: (found[2], HEMIS.index(found[0])))
        return out

    def _near_any(
        self, surface: Surface, hemi: Hemi, points: Float[np.ndarray, "p 3"], radius: float
    ) -> Int[np.ndarray, " k"]:
        """Native vertices of one hemisphere within ``radius`` of any of the points."""
        half = self._hemis[hemi]
        near = np.zeros(len(half.native), dtype=bool)
        for point in points:
            rows, _ = half.search(surface, point, radius)
            near[rows] = True
        return np.unique(half.native[near]).astype(np.int32)


def _point(tkr: Sequence[float]) -> Float[np.ndarray, " 3"]:
    point = np.array([float(x) for x in tkr], dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"a point is three finite coordinates, got {tuple(tkr)}")
    return point


def _radius(radius_mm: float) -> float:
    radius = float(radius_mm)
    if not (math.isfinite(radius) and radius >= 0):
        raise ValueError(f"a distance in mm is finite and not negative, got {radius_mm}")
    return radius


def check_snap_radius(max_dist_mm: float, vertex_map_margin_mm: float) -> None:
    """ValueError unless a mark that snaps over ``max_dist_mm`` finds, in a vertex map made
    with ``vertex_map_margin_mm`` (``RenderConfig``), the vertex that the full surface would
    give: the map reaches that margin beyond what the tiles show, and no farther
    (docs/rendering.md). The server asks this once at start, for ``app.mark_max_dist_mm``
    and the render config of the run; :func:`resolve_mark` asks it for every mark."""
    radius = _radius(max_dist_mm)
    if not radius <= vertex_map_margin_mm:
        raise ValueError(
            f"the snap radius of marks ({radius:g} mm) exceeds the margin of the vertex map "
            f"(vertex_map_margin_mm = {vertex_map_margin_mm:g} mm); beyond that margin the "
            "vertex map does not hold the vertex that the surface would give"
        )


def validate_mark(info: RenderInfo, mark: MarkIn) -> None:
    """Raise :class:`MarkError` unless the mark lies on a tile of the render:
    ``0 <= slice < len(panel.slices)``, ``0 <= px <= width``, ``0 <= py <= height``, and,
    when ``frame`` and ``mode`` are both given, that frame of that mode exists and shows
    ``slice`` in the mark's panel. The first of these that fails is the one reported, in
    the order of ``MARK_FIELDS``."""
    panel = info.panels.get(mark.panel)
    if panel is None:
        raise MarkError("panel", f"the render has no {mark.panel} panel")
    if not 0 <= mark.slice < len(panel.slices):
        raise MarkError(
            "slice", f"the {mark.panel} panel has {len(panel.slices)} slices, got {mark.slice}"
        )
    # written so that a NaN fails as well
    if not 0 <= mark.px <= panel.width:
        raise MarkError("px", f"px lies between 0 and {panel.width}, got {mark.px:g}")
    if not 0 <= mark.py <= panel.height:
        raise MarkError("py", f"py lies between 0 and {panel.height}, got {mark.py:g}")
    if mark.frame is None or mark.mode is None:
        return
    mode = info.modes.get(mark.mode)
    if mode is None:
        raise MarkError("frame", f"the render has no {mark.mode} mode")
    if not 0 <= mark.frame < len(mode.frames):
        raise MarkError(
            "frame", f"the {mark.mode} mode has {len(mode.frames)} frames, got {mark.frame}"
        )
    shown = mode.frames[mark.frame][PANEL_ORDER.index(mark.panel)]
    if shown != mark.slice:
        raise MarkError(
            "frame",
            f"frame {mark.frame} of the {mark.mode} mode shows slice {shown} of the "
            f"{mark.panel} panel, not slice {mark.slice}",
        )


def _vec3(xyz: np.ndarray) -> Vec3:
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _box_distance(info: RenderInfo, tkr: Vec3) -> float:
    """Distance from a point to the view box of the render; 0 inside and on its faces."""
    outside = [
        max(low - x, 0.0, x - high)
        for x, low, high in zip(tkr, info.box.low, info.box.high, strict=True)
    ]
    return math.sqrt(outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2])


def _check_index(info: RenderInfo, index: VertexIndex) -> None:
    """ValueError when the index cannot be that of the render: a vertex of another
    subject's map would be stored as this subject's."""
    listed = info.vertex_map
    if listed is None:
        raise ValueError("the render lists no vertex map, and a vertex index was given")
    counts = index.n_vertices
    if (counts["lh"], counts["rh"]) != (listed.n_lh, listed.n_rh):
        raise ValueError(
            f"the vertex index holds {counts['lh']} and {counts['rh']} vertices, the vertex "
            f"map of the render {listed.n_lh} and {listed.n_rh}: it is not that of this render"
        )
    if listed.fsaverage is None and index.has_counterparts:
        raise ValueError(
            "the vertex index has fsaverage counterparts and the render names no template: "
            "it is not that of this render"
        )


def resolve_mark(
    info: RenderInfo, index: VertexIndex | None, mark: MarkIn, *, max_dist_mm: float = 5.0
) -> ResolvedMark:
    """Everything the server stores for a valid mark.

    ``vox = geometry.pixel_to_voxel(panel, slice, px, py)``, ``tkr`` and ``ras`` through the
    two affines of the render. One hemisphere per mark: the mark's own surface is the white
    surface for class C and the pial surface for A, B and X; its nearest vertex within
    ``max_dist_mm`` over both hemispheres sets ``hemi``, and the other surface is then
    searched in that hemisphere only. When the mark's own surface has no vertex in range,
    the other surface is searched over both hemispheres and sets ``hemi``.
    ``fsavg6_vertex`` is the counterpart of the vertex on the mark's own surface, else of
    the other one, and ``vertex_surface`` says which; both are None when that counterpart
    is -1 (a render without a template) or no vertex is in range. A mark may resolve to a
    vertex that is not ``visible``. ``index=None`` (no vertex map) gives coordinates and
    ``box_dist_mm`` only.

    The mark is validated first (:func:`validate_mark`, :class:`MarkError`): a slice of -1
    would otherwise be read as the last one. ValueError, which is the caller's error and
    not the rater's: ``max_dist_mm`` exceeds the ``vertex_map_margin_mm`` of the render
    (:func:`check_snap_radius`), or ``index`` does not have the vertex counts that the
    render lists for its vertex map.
    """
    validate_mark(info, mark)
    check_snap_radius(max_dist_mm, info.config.vertex_map_margin_mm)
    panel = info.panels[mark.panel]
    voxel = geometry.pixel_to_voxel(panel, mark.slice, mark.px, mark.py)
    vox = _vec3(voxel)
    tkr = _vec3(geometry.apply_affine(np.array(info.volume.vox2ras_tkr, dtype=float), voxel))
    ras = _vec3(geometry.apply_affine(np.array(info.volume.vox2ras, dtype=float), voxel))
    box_dist_mm = _box_distance(info, tkr)
    if index is None:
        return ResolvedMark(vox=vox, tkr=tkr, ras=ras, box_dist_mm=box_dist_mm)
    _check_index(info, index)

    own: Surface = "white" if mark.mark_class == "C" else "pial"
    other: Surface = "pial" if own == "white" else "white"
    found: dict[Surface, tuple[Hemi, int, float, int] | None] = {
        own: index.nearest(own, tkr, max_dist_mm)
    }
    first = found[own]
    # the other surface in the hemisphere of the first, so that both vertices are of one
    # hemisphere; over both hemispheres only when the mark's own surface is out of range
    found[other] = index.nearest(other, tkr, max_dist_mm, hemi=None if first is None else first[0])
    counterpart, vertex_surface = (first, own) if first is not None else (found[other], other)
    pial, white = found["pial"], found["white"]
    has_fsavg = counterpart is not None and counterpart[3] >= 0
    return ResolvedMark(
        vox=vox,
        tkr=tkr,
        ras=ras,
        hemi=None if counterpart is None else counterpart[0],
        pial_vertex=None if pial is None else pial[1],
        pial_dist_mm=None if pial is None else pial[2],
        white_vertex=None if white is None else white[1],
        white_dist_mm=None if white is None else white[2],
        fsavg6_vertex=counterpart[3] if counterpart is not None and has_fsavg else None,
        vertex_surface=vertex_surface if has_fsavg else None,
        box_dist_mm=box_dist_mm,
    )


def marked_vertices(
    index: VertexIndex,
    points: Sequence[Vec3],
    tolerance_mm: float,
    *,
    surface: Surface = "pial",
) -> dict[Hemi, Int[np.ndarray, " k"]]:
    """Native vertex indices of ``surface`` within ``tolerance_mm`` of any of the points
    (tkr RAS, boundary included), ascending, per hemisphere; an empty array for a hemisphere
    without any. For error maps and vertex-level positives: a single nearest vertex is no
    stable summary of a click.

    The vertex map holds the vertices within ``vertex_map_margin_mm`` of what the tiles
    show, so for a mark on a tile the result is that of the full surface for a tolerance up
    to that margin, and can lack vertices beyond it.
    """
    radius = _radius(tolerance_mm)
    xyz = np.array([_point(p) for p in points], dtype=np.float64).reshape(-1, 3)
    return {hemi: index._near_any(surface, hemi, xyz, radius) for hemi in HEMIS}
