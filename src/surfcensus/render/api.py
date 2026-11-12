"""Render one subject into a media directory (docs/rendering.md, "Output per subject").

``render.json`` is written last, so its presence marks a complete render. Source files are
only read; everything written lies inside the output directory.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np

import surfcensus
from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.fsio.volumes import intensity_window, read_volume
from surfcensus.render import encode, frames, geometry, vertexmap
from surfcensus.render.model import (
    PANEL_ORDER,
    RENDER_SCHEMA_VERSION,
    Frame,
    InputFile,
    PanelName,
    RenderConfig,
    RenderInfo,
    VertexMapInfo,
    VideoInfo,
    VolumeInfo,
)
from surfcensus.render.slicing import MeshSlicer
from surfcensus.sources.base import RENDER_ROLES, FileRole
from surfcensus.sources.patterns import resolve_subject_files

RENDER_JSON = "render.json"
POSTER = "poster.webp"
_SURFACE_ROLES = (FileRole.LH_WHITE, FileRole.RH_WHITE, FileRole.LH_PIAL, FileRole.RH_PIAL)
_VERSIONED = ("numpy", "matplotlib", "pillow", "nibabel")


class RenderError(Exception):
    """A subject that cannot be rendered.

    ``reason`` is one of the unratable reasons of ``data/protocol.yaml``: ``missing surface``,
    ``wrong orientation`` or ``render failure``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def render_subject(
    files: Mapping[FileRole, Path],
    out_dir: Path | str,
    config: RenderConfig | None = None,
    *,
    video: bool = True,
    force: bool = False,
) -> RenderInfo:
    """Write tiles, poster, video, vertex map and ``render.json`` of one subject to ``out_dir``.

    ``files`` maps roles to local paths; ``t1`` and the four surfaces are required, and
    ``brainmask`` is used when present. If ``out_dir`` already holds a complete render of the
    same inputs (size and SHA-256), config and ``video`` setting, its ``render.json`` is
    returned and nothing is written, unless ``force``. Otherwise the directory is rendered
    again, and tiles and video of an earlier render that the new one does not list are removed.

    Raises :class:`RenderError` for a subject that cannot be rendered, before anything is
    written.
    """
    config = config or RenderConfig()
    out_dir = Path(out_dir)
    if video and config.panel_px % 2:
        raise ValueError(f"the video needs an even panel_px, got {config.panel_px}")
    paths = {
        role: Path(files[role])
        for role in (*RENDER_ROLES, FileRole.BRAINMASK)
        if role in files and Path(files[role]).is_file()
    }
    if missing := [role.value for role in RENDER_ROLES if role not in paths]:
        raise RenderError("missing surface", "no file for " + ", ".join(missing))
    inputs = {role.value: _input_file(path) for role, path in paths.items()}
    if not force and (done := _complete_render(out_dir, config, inputs, video)) is not None:
        return done

    try:
        t1 = read_volume(paths[FileRole.T1])
        mask = read_volume(paths[FileRole.BRAINMASK]) if FileRole.BRAINMASK in paths else None
        surfaces = {role.value: read_surface(paths[role]) for role in _SURFACE_ROLES}
    except Exception as exc:  # nibabel raises many kinds of error on a damaged file
        raise RenderError("render failure", f"unreadable input ({exc})") from exc
    _check_meshes(surfaces)
    try:
        amap = geometry.axis_map(t1.vox2ras_tkr, t1.shape)
    except geometry.NotAxisAlignedError as exc:
        raise RenderError("wrong orientation", str(exc)) from exc
    # nibabel's vox2ras-tkr has FreeSurfer's fixed axes (LIA) whatever the direction cosines
    # of the file; tkr RAS is anatomical RAS only if the scanner affine has the same axes
    tkr, scanner = t1.vox2ras_tkr[:3, :3], t1.vox2ras[:3, :3]
    if not np.abs(scanner - tkr).max() <= 1e-4 * np.abs(tkr).max():
        raise RenderError(
            "wrong orientation",
            "the volume is not in FreeSurfer's conformed orientation (LIA), "
            "so its tkr axes are not the anatomical ones",
        )
    lh_pial, rh_pial = surfaces["lh_pial"].vertices, surfaces["rh_pial"].vertices
    poles = geometry.find_poles(lh_pial, rh_pial)
    box = geometry.compute_box(lh_pial, rh_pial, poles, config)
    try:
        panels = geometry.build_panels(box, amap, config)
        window = intensity_window(t1, mask, config.window_percentiles, config.brainmask_dilation)
    except ValueError as exc:  # box outside the volume, brainmask on another grid
        raise RenderError("render failure", str(exc)) from exc
    modes = geometry.frame_tables(panels)
    sync = modes["sync"].frames
    vmap = vertexmap.build_vertex_map(surfaces, box, config.vertex_map_margin_mm)
    vertex_map = VertexMapInfo(n_lh=len(vmap["lh_index"]), n_rh=len(vmap["rh_index"]))
    width, height = 3 * config.panel_px, config.panel_px
    info = RenderInfo(
        surfcensus_version=surfcensus.__version__,
        config=config,
        config_hash=config.config_hash(),
        volume=VolumeInfo.model_validate(
            {
                "shape": t1.shape,
                "voxel_size_mm": t1.voxel_size,
                "vox2ras_tkr": t1.vox2ras_tkr.tolist(),
                "vox2ras": t1.vox2ras.tolist(),
            }
        ),
        inputs=inputs,
        poles=poles,
        box=box,
        window=window,
        panels=panels,
        modes=modes,
        video=VideoInfo(fps=config.fps, n_frames=len(sync), width=width, height=height)
        if video
        else None,
        poster=POSTER,
        vertex_map=vertex_map,
        versions={name: version(name) for name in _VERSIONED},
    )

    (out_dir / RENDER_JSON).unlink(missing_ok=True)  # incomplete from here on
    slicers = {name: MeshSlicer(s.vertices, s.faces) for name, s in surfaces.items()}
    blended: dict[PanelName, list[np.ndarray]] = {name: [] for name in PANEL_ORDER}
    with frames.ContourRenderer(config.panel_px, config.panel_px, config.line_width_px) as pen:
        for name, panel in panels.items():
            for pos in range(len(panel.slices)):
                gray = frames.gray_tile(t1.data, panel, pos, window)
                layers = frames.contour_layers(panel, pos, t1.vox2ras_tkr, slicers, config)
                contours = pen.render(layers)
                encode.write_gray_webp(
                    out_dir / info.tiles.gray_path(name, pos), gray, config.webp_quality
                )
                encode.write_rgba_webp(out_dir / info.tiles.contours_path(name, pos), contours)
                blended[name].append(frames.blend(gray, contours))

    def strip(frame: Frame) -> np.ndarray:
        tiles = {name: blended[name][pos] for name, pos in zip(PANEL_ORDER, frame, strict=True)}
        return frames.composite_strip(tiles, panels, frame, config)

    n_forward = max(len(panel.slices) for panel in panels.values())
    poster = strip(sync[(n_forward - 1) // 2])
    encode.write_rgb_webp(out_dir / POSTER, poster, config.webp_quality)
    if info.video is not None:
        strips = (strip(frame) for frame in sync)  # composed one at a time, as ffmpeg takes them
        encode.write_mp4(out_dir / info.video.file, strips, config.fps, config.crf)
    vertexmap.write_vertex_map(out_dir / vertex_map.file, vmap)
    _remove_stale(out_dir, info)
    with encode.atomic_path(out_dir / RENDER_JSON) as tmp:
        tmp.write_bytes(info.to_json().encode())
    return info


def three_plane(
    subject_dir: Path | str, out: Path | str, config: RenderConfig | None = None, **kwargs: bool
) -> RenderInfo:
    """Render one FreeSurfer subject directory, without a run directory.

    ``out`` is the media directory to write, or, if it ends in ``.mp4``, the path of the video
    alone. ``kwargs`` are the ``video`` and ``force`` switches of :func:`render_subject`.
    """
    files = resolve_subject_files(subject_dir)
    out = Path(out)
    if out.suffix.lower() != ".mp4":
        return render_subject(files, out, config, **kwargs)
    if not kwargs.get("video", True):
        raise ValueError(f"video=False contradicts the video output {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # rendered next to the destination, so that the video moves into place by a rename
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".surfcensus-") as tmp:
        info = render_subject(files, Path(tmp), config, **kwargs)
        assert info.video is not None
        os.replace(Path(tmp) / info.video.file, out)
    return info


def _input_file(path: Path) -> InputFile:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # a link inside its own directory (FreeSurfer 7: lh.pial -> lh.pial.T1) names the variant
    target = path.resolve()
    name = target.name if target.parent == path.parent.resolve() else path.name
    return InputFile(name=name, size=target.stat().st_size, sha256=digest)


def _check_meshes(surfaces: Mapping[str, Surface]) -> None:
    for name, surface in surfaces.items():
        n = len(surface.vertices)
        if n == 0 or len(surface.faces) == 0:
            problem = "has no vertices or no faces"
        elif not np.isfinite(surface.vertices).all():
            problem = "has non-finite vertex coordinates"
        elif surface.faces.min() < 0 or surface.faces.max() >= n:
            problem = "has face indices out of range"
        elif n != len(surfaces[name[:3] + "pial"].vertices):
            problem = "differs from the pial surface in vertex count"
        else:
            continue
        raise RenderError("render failure", f"degenerate mesh: {name} {problem}")


def _listed_files(info: RenderInfo) -> list[str]:
    """Every file of a render except ``render.json``, relative to the media directory."""
    names = [
        path(name, pos)
        for name, panel in info.panels.items()
        for pos in range(len(panel.slices))
        for path in (info.tiles.gray_path, info.tiles.contours_path)
    ]
    if info.poster is not None:
        names.append(info.poster)
    if info.video is not None:
        names.append(info.video.file)
    if info.vertex_map is not None:
        names.append(info.vertex_map.file)
    return names


def _complete_render(
    out_dir: Path, config: RenderConfig, inputs: Mapping[str, InputFile], video: bool
) -> RenderInfo | None:
    """The render already in ``out_dir`` if it is complete and matches, else None."""
    try:
        info = RenderInfo.model_validate_json((out_dir / RENDER_JSON).read_bytes())
    except (OSError, ValueError):  # absent, or not a render.json of this schema
        return None
    same = (
        info.schema_version == RENDER_SCHEMA_VERSION
        and info.config_hash == config.config_hash()
        and info.inputs == inputs
        and (info.video is not None) == video
        and all((out_dir / name).is_file() for name in _listed_files(info))
    )
    return info if same else None


def _remove_stale(out_dir: Path, info: RenderInfo) -> None:
    """Remove tiles and video of an earlier render that ``info`` does not list, and temporary
    files that an interrupted writer left behind."""
    keep = {out_dir / name for name in _listed_files(info)}
    patterns = ("anim.mp4", "tiles/*.webp", ".*.tmp", "tiles/.*.tmp")
    for path in (path for pattern in patterns for path in out_dir.glob(pattern)):
        if path not in keep:
            path.unlink()
