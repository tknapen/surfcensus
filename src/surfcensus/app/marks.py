"""Click marks: from a point on a tile to voxels, tkr RAS, scanner RAS and vertices. See
docs/rater_app.md, "Marking errors by clicking".

Pure functions on a ``RenderInfo`` and a vertex map: no database, no web framework. The
shapes ``MarkIn`` and ``ResolvedMark`` are in ``app.schemas``.

Interface of milestone 4. ``MarkError`` is final; the class and the functions are written
by package 4.A.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from jaxtyping import Int

from surfcensus.app.schemas import MarkIn, ResolvedMark, Surface
from surfcensus.render.model import RenderInfo, Vec3
from surfcensus.sources.base import Hemi

_OWNER = "package 4.A (marks)"

MarkField = Literal["panel", "slice", "px", "py", "frame"]
MARK_FIELDS: tuple[MarkField, ...] = ("panel", "slice", "px", "py", "frame")


class MarkError(ValueError):
    """A mark that does not fit the render it was made on. ``field`` names the part of the
    mark that is wrong, ``message`` says how; the server answers 422 with both."""

    def __init__(self, field: MarkField, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message

    def __reduce__(self) -> tuple[type[MarkError], tuple[str, str]]:
        return type(self), (self.field, self.message)


class VertexIndex:
    """KD-trees over the pial and the white positions of one ``vertex_map.npz``, per
    hemisphere. A result is ``(hemi, native vertex, distance in mm, template vertex or -1)``."""

    def __init__(self, vmap: Mapping[str, np.ndarray]) -> None:
        """``vmap``: the arrays of ``render.vertexmap.read_vertex_map``."""
        raise NotImplementedError(f"app.marks.VertexIndex: {_OWNER}")

    def nearest(
        self, surface: Surface, tkr: Vec3, max_dist_mm: float, *, hemi: Hemi | None = None
    ) -> tuple[Hemi, int, float, int] | None:
        """The vertex of ``surface`` nearest ``tkr`` within ``max_dist_mm`` (boundary
        included), or None. ``hemi=None`` searches both hemispheres: the nearer wins, ``lh``
        on an exact tie. Within a hemisphere the lowest native index wins a tie."""
        raise NotImplementedError(f"app.marks.VertexIndex.nearest: {_OWNER}")

    def within(
        self, surface: Surface, tkr: Vec3, radius_mm: float, *, hemi: Hemi | None = None
    ) -> list[tuple[Hemi, int, float, int]]:
        """Every vertex of ``surface`` within ``radius_mm`` of ``tkr``, by ascending
        distance, then ``lh`` before ``rh``, then native index."""
        raise NotImplementedError(f"app.marks.VertexIndex.within: {_OWNER}")


def validate_mark(info: RenderInfo, mark: MarkIn) -> None:
    """Raise :class:`MarkError` unless the mark lies on a tile of the render:
    ``0 <= slice < len(panel.slices)``, ``0 <= px <= width``, ``0 <= py <= height``, and,
    when ``frame`` and ``mode`` are both given, that frame of that mode exists and shows
    ``slice`` in the mark's panel."""
    raise NotImplementedError(f"app.marks.validate_mark: {_OWNER}")


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
    ``box_dist_mm`` only. ``max_dist_mm`` must not exceed the ``vertex_map_margin_mm`` of
    the render (ValueError): beyond it the vertex map does not give the vertex that the
    full surface would.
    """
    raise NotImplementedError(f"app.marks.resolve_mark: {_OWNER}")


def marked_vertices(
    index: VertexIndex,
    points: Sequence[Vec3],
    tolerance_mm: float,
    *,
    surface: Surface = "pial",
) -> dict[Hemi, Int[np.ndarray, " k"]]:
    """Native vertex indices of ``surface`` within ``tolerance_mm`` of any of the points
    (tkr RAS), ascending, per hemisphere; an empty array for a hemisphere without any. For
    error maps and vertex-level positives: a single nearest vertex is no stable summary of
    a click."""
    raise NotImplementedError(f"app.marks.marked_vertices: {_OWNER}")
