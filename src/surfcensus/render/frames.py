"""Tiles and composite strips: grayscale slice tiles, contour tiles, and their composition.

Conventions (docs/rendering.md): tile arrays are indexed ``[row, column]`` with row 0 at the
top; the pixel in column ``c`` and row ``r`` is sampled at panel pixel ``(c + 0.5, r + 0.5)``.
Every coordinate conversion goes through :mod:`surfcensus.render.geometry`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from types import TracebackType

import numpy as np
from jaxtyping import Float, UInt8
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.path import Path
from PIL import Image, ImageDraw, ImageFont

from surfcensus.render import geometry
from surfcensus.render.model import PANEL_ORDER, Frame, PanelInfo, PanelName, RenderConfig
from surfcensus.render.slicing import MeshSlicer

# drawing order of the contour layers: white surfaces first, pial over them
SURFACE_ORDER = ("lh_white", "rh_white", "lh_pial", "rh_pial")

_LETTER_COLOUR = (255, 255, 255)
_LETTER_OUTLINE = (0, 0, 0)
_LETTER_PAD_PX = 3
_CHAIN_GRID = 1024  # endpoints closer than 1/1024 px count as the same point when chaining


def gray_tile(
    data: np.ndarray, panel: PanelInfo, slice_pos: int, window: tuple[float, float]
) -> UInt8[np.ndarray, "h w"]:
    """Grayscale tile of slice ``panel.slices[slice_pos]`` of a 3D volume.

    Nearest neighbour at every pixel centre: the voxel is ``floor(v + 0.5)`` of the continuous
    voxel coordinate ``v``. Pixels outside the volume are 0. ``window`` is ``(lo, hi)`` with
    ``hi > lo``: lo maps to 0 and hi to 255, linearly, clipped and rounded to nearest.
    """
    lo, hi = window
    if data.ndim != 3 or not hi > lo:
        raise ValueError(f"expected a 3D volume and hi > lo, got shape {data.shape}, {window}")
    h_axis, v_axis, s_axis = panel.h_voxel_axis, panel.v_voxel_axis, panel.slice_voxel_axis
    along_h = geometry.pixel_to_voxel(panel, slice_pos, np.arange(panel.width) + 0.5, 0.5)
    along_v = geometry.pixel_to_voxel(panel, slice_pos, 0.5, np.arange(panel.height) + 0.5)
    cols = np.floor(along_h[:, h_axis] + 0.5).astype(np.intp)
    rows = np.floor(along_v[:, v_axis] + 0.5).astype(np.intp)
    # tile columns and rows that fall inside the volume: one contiguous run each
    c_in = np.flatnonzero((cols >= 0) & (cols < data.shape[h_axis]))
    r_in = np.flatnonzero((rows >= 0) & (rows < data.shape[v_axis]))
    s = panel.slices[slice_pos]
    tile = np.zeros((panel.height, panel.width), dtype=np.uint8)
    if not 0 <= s < data.shape[s_axis] or c_in.size == 0 or r_in.size == 0:
        return tile
    plane = np.moveaxis(data, s_axis, 0)[s]  # the two in-plane voxel axes, in ascending order
    if v_axis > h_axis:
        plane = plane.T
    cols, rows = cols[c_in], rows[r_in]
    # window the block of voxels the tile shows, then repeat voxels into pixels
    block = plane[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].astype(np.float64)
    block = np.nan_to_num((block - lo) * (255.0 / (hi - lo)), nan=0.0)
    block = np.rint(np.clip(block, 0.0, 255.0)).astype(np.uint8)
    tile[r_in[0] : r_in[-1] + 1, c_in[0] : c_in[-1] + 1] = block.take(
        rows - rows.min(), axis=0
    ).take(cols - cols.min(), axis=1)
    return tile


class ContourRenderer:
    """Draws line segments onto a transparent RGBA tile with one persistent Agg figure.

    The figure is ``width`` x ``height`` pixels and its coordinates are panel pixels, y down.
    It is built on the Agg canvas directly, without pyplot, so it touches no global state and
    is safe in worker processes. Use it as a context manager, or call :meth:`close`.
    """

    def __init__(self, width: int, height: int, line_width_px: float) -> None:
        if width < 1 or height < 1 or not line_width_px > 0:
            raise ValueError(f"invalid tile size or line width: {width}, {height}, {line_width_px}")
        self.width, self.height = width, height
        # at 1 dpi a figure inch is a pixel (so the size is exact) and a point is 1/72 px
        self._line_width_pt = 72.0 * line_width_px
        self._figure = Figure(
            figsize=(width, height), dpi=1, facecolor="none", frameon=False, layout="none"
        )
        self._canvas: FigureCanvasAgg | None = FigureCanvasAgg(self._figure)
        self._axes = self._figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
        self._axes.set_axis_off()
        self._axes.set_xlim(0, width)
        self._axes.set_ylim(height, 0)
        self._collections: list[PathCollection] = []

    def render(
        self, layers: Sequence[tuple[Float[np.ndarray, "_ 2 2"], str]]
    ) -> UInt8[np.ndarray, "h w 4"]:
        """Draw ``(segments, "#RRGGBB")`` layers in the given order; return straight RGBA.

        Segments are ``(start, end)`` points in panel pixel coordinates. Segments that share
        endpoints are joined into polylines first, so the line keeps its width where they
        meet. Antialiased, round caps and joins. Every pixel with alpha 0 has RGB 0.
        """
        if self._canvas is None:
            raise RuntimeError("the renderer is closed")
        while len(self._collections) < len(layers):
            collection = PathCollection(
                [],
                facecolors="none",
                linewidths=self._line_width_pt,
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
            collection.set_snap(False)  # snapping would move axis-parallel lines to pixel centres
            self._axes.add_collection(collection, autolim=False)
            self._collections.append(collection)
        for i, collection in enumerate(self._collections):
            collection.set_visible(i < len(layers))
            if i < len(layers):
                segments, colour = layers[i]
                collection.set_paths(_chain(segments))
                collection.set_edgecolor([tuple(c / 255 for c in _rgb(colour))])
        self._canvas.draw()  # clears the buffer first
        rgba = np.array(self._canvas.buffer_rgba())  # Agg keeps straight alpha
        if rgba.shape != (self.height, self.width, 4):
            raise RuntimeError(f"Agg canvas is {rgba.shape}, expected {self.height} x {self.width}")
        rgba *= rgba[..., 3:] > 0
        return rgba

    def close(self) -> None:
        if self._canvas is not None:
            self._figure.clear()
            self._collections = []
            self._canvas = None

    def __enter__(self) -> ContourRenderer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def contour_layers(
    panel: PanelInfo,
    slice_pos: int,
    vox2ras_tkr: Float[np.ndarray, "4 4"],
    slicers: Mapping[str, MeshSlicer],
    config: RenderConfig,
) -> list[tuple[Float[np.ndarray, "_ 2 2"], str]]:
    """Contours of one tile as layers for :meth:`ContourRenderer.render`.

    ``slicers`` maps any of ``lh_white``, ``rh_white``, ``lh_pial``, ``rh_pial`` to the slicer
    of that surface (vertices in tkr RAS). Each is cut by the plane of the slice; segments
    whose bounding box misses the tile are dropped, and so are surfaces left without segments.
    White surfaces come first, then pial.
    """
    if unknown := sorted(set(slicers) - set(SURFACE_ORDER)):
        raise ValueError(f"unknown surfaces {unknown}; expected a subset of {SURFACE_ORDER}")
    pad = config.line_width_px
    layers: list[tuple[np.ndarray, str]] = []
    for name in SURFACE_ORDER:
        if name not in slicers:
            continue
        xyz = slicers[name].segments(panel.world_axis, panel.slice_mm[slice_pos])
        px, py = geometry.tkr_to_pixel(panel, vox2ras_tkr, xyz)
        visible = (
            (px.max(axis=1) >= -pad)
            & (px.min(axis=1) <= panel.width + pad)
            & (py.max(axis=1) >= -pad)
            & (py.min(axis=1) <= panel.height + pad)
        )
        if visible.any():
            colour = config.white_colour if name.endswith("white") else config.pial_colour
            layers.append((np.stack([px[visible], py[visible]], axis=-1), colour))
    return layers


def blend(
    gray: UInt8[np.ndarray, "h w"], contours: UInt8[np.ndarray, "h w 4"]
) -> UInt8[np.ndarray, "h w 3"]:
    """Contours (straight RGBA) alpha-composited over a grayscale tile."""
    alpha = contours[..., 3:].astype(np.uint16)  # the weighted sum is at most 255 * 255
    mixed = contours[..., :3] * alpha + gray[..., None] * (255 - alpha)
    return ((mixed + 127) // 255).astype(np.uint8)


def composite_strip(
    tiles: Mapping[PanelName, UInt8[np.ndarray, "h w 3"]],
    panels: Mapping[PanelName, PanelInfo],
    frame: Frame,
    config: RenderConfig,
) -> UInt8[np.ndarray, "h w3 3"]:
    """One video frame: sagittal, coronal, axial tiles from left to right.

    On each panel, a 1 px vertical and a 1 px horizontal crosshair line through the pixel that
    holds the current position of the other two planes, blended with ``crosshair_alpha``, then
    the four edge letters in Pillow's built-in bitmap font (no system fonts involved).
    """
    cross = geometry.crosshair_pixels(dict(panels), frame)
    colour = np.array(_rgb(config.crosshair_colour), dtype=np.float64)
    alpha = config.crosshair_alpha
    parts: list[np.ndarray] = []
    for name in PANEL_ORDER:
        tile = tiles[name].copy()
        height, width = tile.shape[:2]
        # the pixel that holds the point; one on a pixel boundary must not hang on the last bit
        col, row = (int(np.floor(a + 1e-6)) for a in cross[name])
        on = np.zeros((height, width), dtype=bool)
        if 0 <= col < width:
            on[:, col] = True
        if 0 <= row < height:
            on[row, :] = True
        tile[on] = np.rint(tile[on] * (1 - alpha) + colour * alpha).astype(np.uint8)
        parts.append(tile)
    image = Image.fromarray(np.concatenate(parts, axis=1))
    draw = ImageDraw.Draw(image)
    font = _font()
    x0 = 0
    for name, part in zip(PANEL_ORDER, parts, strict=True):
        height, width = part.shape[:2]
        edges = panels[name].edges
        for letter, fx, fy in (
            (edges.left, 0.0, 0.5),
            (edges.right, 1.0, 0.5),
            (edges.top, 0.5, 0.0),
            (edges.bottom, 0.5, 1.0),
        ):
            left, top, right, bottom = font.getbbox(letter)
            room_x = width - 2 * _LETTER_PAD_PX - (right - left)
            room_y = height - 2 * _LETTER_PAD_PX - (bottom - top)
            x = x0 + _LETTER_PAD_PX + round(fx * room_x) - left
            y = _LETTER_PAD_PX + round(fy * room_y) - top
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                draw.text((x + dx, y + dy), letter, font=font, fill=_LETTER_OUTLINE)
            draw.text((x, y), letter, font=font, fill=_LETTER_COLOUR)
        x0 += width
    return np.array(image)


def _chain(segments: np.ndarray) -> list[Path]:
    """Join ``(start, end)`` segments into open and closed polylines.

    A segment follows another when it starts where that one ends; segments of no length are
    dropped.
    """
    if segments.shape[0] == 0:
        return []
    grid = np.rint(segments.reshape(-1, 2) * _CHAIN_GRID).astype(np.int64)
    grid -= grid.min(axis=0)
    _, point = np.unique(grid[:, 0] * (grid[:, 1].max() + 1) + grid[:, 1], return_inverse=True)
    start, end = point.reshape(-1, 2).T
    # a plane through a vertex gives zero-length segments, which would be drawn as dots
    keep = start != end
    segments, start, end = segments[keep], start[keep], end[keep]
    n = segments.shape[0]
    if n == 0:
        return []
    # pair the k-th segment ending at a point with the k-th segment starting there
    by_start, by_end = np.argsort(start, kind="stable"), np.argsort(end, kind="stable")
    s_sorted, e_sorted = start[by_start], end[by_end]
    s_key = s_sorted * n + np.arange(n) - np.searchsorted(s_sorted, s_sorted)
    e_key = e_sorted * n + np.arange(n) - np.searchsorted(e_sorted, e_sorted)
    at = np.minimum(np.searchsorted(s_key, e_key), n - 1)
    matched = s_key[at] == e_key
    successor = np.full(n, -1)
    successor[by_end[matched]] = by_start[at[matched]]
    is_head = np.ones(n, dtype=bool)
    is_head[successor[successor >= 0]] = False

    follow = successor.tolist()
    seen = [False] * n
    paths: list[Path] = []
    for first in (*np.flatnonzero(is_head).tolist(), *range(n)):  # open chains, then loops
        members: list[int] = []
        i = first
        while i >= 0 and not seen[i]:
            seen[i] = True
            members.append(i)
            i = follow[i]
        if members and i == first:  # the last vertex of a closed path is a placeholder
            paths.append(Path(segments[[*members, first], 0], closed=True))
        elif members:
            paths.append(Path(np.concatenate([segments[members, 0], segments[members[-1:], 1]])))
    return paths


def _rgb(colour: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
        raise ValueError(f"colour must be '#RRGGBB', got {colour!r}")
    return (int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16))


@lru_cache(maxsize=1)
def _font() -> ImageFont.ImageFont:
    """Pillow's built-in bitmap font: the same glyphs everywhere, with or without FreeType."""
    load = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)
    return load()
