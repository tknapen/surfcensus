"""Data contract for rendering: the render configuration and the contents of render.json.

The prose specification is docs/rendering.md. Everything a client needs to turn a click on a
tile into anatomy, and to play the animation, is in :class:`RenderInfo`.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RENDER_SCHEMA_VERSION = 2
"""Version of ``render.json`` and of the files it lists. 2: the vertex map is selected by the
clickable region of the tiles and carries ``{hemi}_visible``; 1 selected it by the view box."""

TILE_FIELDS: tuple[str, ...] = (
    "box_x_mm",
    "box_y_mm",
    "box_z_mm",
    "margin_mm",
    "slice_step_mm",
    "panel_px",
    "radiological",
    "sagittal_anterior_left",
    "pial_colour",
    "white_colour",
    "line_width_px",
    "window_percentiles",
    "brainmask_dilation",
    "webp_quality",
)
"""Fields of :class:`RenderConfig` that change a tile's pixels or its geometry."""

NON_TILE_FIELDS: tuple[str, ...] = (
    "crosshair_colour",
    "crosshair_alpha",
    "fps",
    "crf",
    "vertex_map_margin_mm",
)
"""The other fields: they change the video, the poster or the vertex map only. Every field of
:class:`RenderConfig` is in one of the two tuples."""

PanelName = Literal["sagittal", "coronal", "axial"]
PANEL_ORDER: tuple[PanelName, ...] = ("sagittal", "coronal", "axial")

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Vec3 = tuple[float, float, float]
Frame = tuple[int, int, int]
"""Slice numbers shown in the sagittal, coronal and axial panel, in that order."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RenderConfig(_Frozen):
    """Everything that changes the rendered output. Hashed into ``config_hash``."""

    box_x_mm: tuple[float, float] = (-30.0, 30.0)
    """Low and high end relative to the box centre, low at or below high; likewise
    ``box_y_mm`` and ``box_z_mm``. One of the three, or ``margin_mm``, must have an extent."""
    box_y_mm: tuple[float, float] = (-8.0, 45.0)
    box_z_mm: tuple[float, float] = (-35.0, 25.0)
    margin_mm: float = Field(default=10.0, ge=0)
    slice_step_mm: float = Field(default=1.0, gt=0)
    panel_px: int = Field(default=320, ge=16)

    radiological: bool = True
    """Subject's left on the right of the coronal and axial panels (decision D13)."""
    sagittal_anterior_left: bool = False
    """Anterior at the left of the sagittal panel. Freeview shows anterior at the right."""

    pial_colour: str = "#FFD400"
    """``#RRGGBB``, stored in upper case; likewise ``white_colour`` and ``crosshair_colour``."""
    white_colour: str = "#FF3B30"
    line_width_px: float = Field(default=1.25, gt=0)
    """At most ``panel_px / 8``."""
    crosshair_colour: str = "#33C7FF"
    crosshair_alpha: float = Field(default=0.45, ge=0, le=1)

    window_percentiles: tuple[float, float] = (1.0, 99.5)
    brainmask_dilation: int = Field(default=3, ge=0)

    webp_quality: int = Field(default=80, ge=1, le=100)
    fps: int = Field(default=12, ge=1)
    crf: int = Field(default=23, ge=0, le=51)

    vertex_map_margin_mm: float = Field(default=5.0, ge=0)
    """How far beyond the region that the tiles show ``vertex_map.npz`` reaches. A mark
    resolves through the vertex map to the same nearest vertex as on the full surfaces for
    any snap radius up to this margin; the default is the snap radius of docs/rater_app.md."""

    @field_validator("pial_colour", "white_colour", "crosshair_colour")
    @classmethod
    def _hex_colour(cls, colour: str) -> str:
        # the one form the renderer draws (frames._rgb), in one spelling, so that two configs
        # that render the same bytes have the same hash
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            raise ValueError(f"colour must be '#RRGGBB', got {colour!r}")
        return colour.upper()

    @model_validator(mode="after")
    def _box_can_be_shown(self) -> RenderConfig:
        # a box that no panel can show would otherwise be refused once per subject, as a
        # failure of the subject (geometry.field_of_view, geometry.build_panels)
        extents = []
        for name in ("box_x_mm", "box_y_mm", "box_z_mm"):
            low, high = getattr(self, name)
            if not low <= high:
                raise ValueError(
                    f"{name} must run from its low to its high end, got {low:g}, {high:g}"
                )
            extents.append(high - low)
        if not max(extents) + 2 * self.margin_mm > 0:
            raise ValueError(
                "box_x_mm, box_y_mm and box_z_mm without extent and a margin_mm of 0 "
                "leave the panels nothing to show"
            )
        return self

    @model_validator(mode="after")
    def _line_narrower_than_panel(self) -> RenderConfig:
        # a contour tile that its lines cover entirely cannot be stored with an alpha channel
        # (encode.write_rgba_webp); the bound keeps lines far from that, it is not a proof
        if self.line_width_px > self.panel_px / 8:
            raise ValueError(
                f"line_width_px must be at most panel_px / 8 = {self.panel_px / 8:g}, "
                f"got {self.line_width_px:g}"
            )
        return self

    def box_offsets(self) -> np.ndarray:
        """Box corners relative to the centre, shape (2, 3): row 0 low, row 1 high."""
        return np.array([self.box_x_mm, self.box_y_mm, self.box_z_mm], dtype=float).T

    def config_hash(self) -> str:
        payload = json.dumps(
            {"schema": RENDER_SCHEMA_VERSION, "config": self.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def tile_hash(self) -> str:
        """Identity of the tiles alone: 12 hex characters over :data:`TILE_FIELDS`.

        Unlike :meth:`config_hash` it has no schema number and no video, crosshair or
        vertex-map setting in it, so it stays the same when a render is only re-encoded or
        its vertex map rewritten. A rating belongs to the tiles it was made on.
        """
        payload = json.dumps(
            self.model_dump(mode="json", include=set(TILE_FIELDS)),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class VolumeInfo(_Frozen):
    shape: tuple[int, int, int]
    voxel_size_mm: Vec3
    vox2ras_tkr: Matrix4
    vox2ras: Matrix4


class InputFile(_Frozen):
    name: str
    """The name FreeSurfer gives the file (for example ``lh.pial.T1``): never a path, and
    never a name the file has in some other store."""
    size: int
    sha256: str

    @field_validator("name")
    @classmethod
    def _no_path(cls, name: str) -> str:
        # render.json is served to raters, and a path could name the dataset
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"a file name without directories is required, got {name!r}")
        return name


class PoleInfo(_Frozen):
    vertex: int
    tkr: Vec3


class BoxInfo(_Frozen):
    centre: Vec3
    low: Vec3
    high: Vec3


class PanelEdges(_Frozen):
    """Anatomical letter (L, R, A, P, S, I) at each edge of the panel."""

    left: str
    right: str
    top: str
    bottom: str


class PanelInfo(_Frozen):
    name: PanelName
    world_axis: int
    """tkr axis normal to the slices: 0 x (right), 1 y (anterior), 2 z (superior)."""
    slice_voxel_axis: int
    """Voxel axis the slices step along."""
    h_voxel_axis: int
    """Voxel axis that varies along the panel's horizontal."""
    v_voxel_axis: int
    """Voxel axis that varies along the panel's vertical."""
    slices: tuple[int, ...]
    """Voxel index along ``slice_voxel_axis`` of each tile, from low to high tkr coordinate."""
    slice_mm: tuple[float, ...]
    """tkr coordinate along ``world_axis`` of each slice plane."""
    centre_slice: int
    """Position in ``slices`` nearest the box centre."""
    width: int
    height: int
    pix2vox: Matrix3
    """(px, py, 1) -> (voxel coordinate along h_voxel_axis, along v_voxel_axis, 1)."""
    vox2pix: Matrix3
    edges: PanelEdges

    def pix2vox_array(self) -> np.ndarray:
        return np.array(self.pix2vox, dtype=float)

    def vox2pix_array(self) -> np.ndarray:
        return np.array(self.vox2pix, dtype=float)


class ModeInfo(_Frozen):
    frames: tuple[Frame, ...]
    """One full ping-pong loop."""


class TileFiles(_Frozen):
    gray: str = "tiles/{panel}_{index:03d}.webp"
    contours: str = "tiles/{panel}_{index:03d}_c.webp"

    def gray_path(self, panel: str, index: int) -> str:
        return self.gray.format(panel=panel, index=index)

    def contours_path(self, panel: str, index: int) -> str:
        return self.contours.format(panel=panel, index=index)


class VideoInfo(_Frozen):
    file: str = "anim.mp4"
    fps: int
    n_frames: int
    width: int
    height: int


class VertexMapInfo(_Frozen):
    file: str = "vertex_map.npz"
    n_lh: int
    """Number of left-hemisphere vertices in the file; likewise ``n_rh``."""
    n_rh: int
    n_lh_visible: int
    """How many of them have their pial position in the region that the tiles show
    (``lh_visible`` in the file); likewise ``n_rh_visible``."""
    n_rh_visible: int
    fsaverage: str | None = None
    """Name of the fsaverage mesh the counterparts refer to; None while they are unset (-1)."""


class RenderInfo(_Frozen):
    """Contents of render.json."""

    schema_version: int = RENDER_SCHEMA_VERSION
    surfcensus_version: str
    config: RenderConfig
    config_hash: str
    volume: VolumeInfo
    inputs: dict[str, InputFile]
    """Keyed by file role (``t1``, ``lh_pial``, ...)."""
    poles: dict[Literal["lh", "rh"], PoleInfo]
    box: BoxInfo
    window: tuple[float, float]
    panels: dict[PanelName, PanelInfo]
    modes: dict[Literal["sync", "seq"], ModeInfo]
    tiles: TileFiles = TileFiles()
    video: VideoInfo | None = None
    poster: str | None = "poster.webp"
    vertex_map: VertexMapInfo | None = None
    versions: dict[str, str]

    def to_json(self) -> str:
        """Canonical serialization: stable key order, trailing newline."""
        return json.dumps(self.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
