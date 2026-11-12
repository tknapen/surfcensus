"""Render one subject into a media directory (docs/rendering.md, "Output per subject").

``render.json`` is written last, so its presence marks a complete render. Source files are
only read; everything written lies inside the output directory. Renders into one directory
run one after the other (``_exclusive``).
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

import surfcensus
from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.fsio.volumes import Volume, intensity_window, read_volume
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
from surfcensus.sources.patterns import ROLE_CANDIDATES, is_file, resolve_subject_files

try:
    import fcntl
except ImportError:  # Windows has no flock: renders into one directory are not serialized there
    fcntl = None

RENDER_JSON = "render.json"
POSTER = "poster.webp"
VIDEO = "anim.mp4"
VERTEX_MAP = "vertex_map.npz"
_SURFACE_ROLES = (FileRole.LH_WHITE, FileRole.RH_WHITE, FileRole.LH_PIAL, FileRole.RH_PIAL)
_VERSIONED = ("numpy", "matplotlib", "pillow", "nibabel")
# the names this module writes, which are the only ones it ever removes (_remove_stale)
_TILE = re.compile(rf"(?:{'|'.join(PANEL_ORDER)})_\d{{3,}}(?:_c)?\.webp")
_TOP_LEVEL = re.compile("|".join(map(re.escape, (RENDER_JSON, POSTER, VIDEO, VERTEX_MAP))))
_TEMPORARY = re.compile(r"\.(.+)\.[0-9a-f]{16}\.tmp")  # encode.atomic_path, group 1: final name


class RenderError(Exception):
    """A subject that cannot be rendered.

    ``reason`` is one of the unratable reasons of ``data/protocol.yaml``: ``missing surface``,
    ``wrong orientation`` or ``render failure``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class _Prepared:
    """A subject that passed every check, and what its render will write. Nothing has been
    written when one exists."""

    info: RenderInfo
    t1: Volume
    surfaces: dict[str, Surface]
    vertex_map: dict[str, np.ndarray]


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
    While another render writes to ``out_dir``, this one waits for it.

    Raises :class:`RenderError` for a subject that cannot be rendered, or for a video that
    cannot be encoded because there is no ffmpeg binary, and ValueError for a config that
    cannot be used, before anything is written.
    """
    config = _checked_config(config, video)
    out_dir = Path(out_dir)
    paths, inputs = _locate(files)
    if not force and (done := _complete_render(out_dir, config, inputs, video)) is not None:
        return done
    prepared = _prepare(paths, inputs, config, video)
    with _exclusive(out_dir) as locked:
        # the render this one waited for may have left just what is asked for
        if not force and (done := _complete_render(out_dir, config, inputs, video)) is not None:
            return done
        _write(out_dir, prepared, remove_temporaries=locked)
    return prepared.info


def three_plane(
    subject_dir: Path | str,
    out: Path | str,
    config: RenderConfig | None = None,
    *,
    video: bool = True,
    force: bool = False,
) -> RenderInfo:
    """Render one FreeSurfer subject directory, without a run directory.

    ``out`` is the media directory to write, or, if it ends in ``.mp4``, the path of the video
    alone. ``video`` and ``force`` are the switches of :func:`render_subject`; a video alone is
    always rendered.
    """
    try:
        files = resolve_subject_files(subject_dir)
    except OSError as exc:
        raise _unreadable(exc) from exc
    out = Path(out)
    if out.suffix.lower() != ".mp4":
        return render_subject(files, out, config, video=video, force=force)
    if not video:
        raise ValueError(f"video=False contradicts the video output {out}")
    # every refusal comes first, so that a refused subject leaves no directory behind
    config = _checked_config(config, video=True)
    prepared = _prepare(*_locate(files), config, video=True)
    created = [folder for folder in (out.parent, *out.parent.parents) if not folder.exists()]
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        # rendered next to the destination, so that the video moves into place by a rename
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".surfcensus-") as tmp:
            _write(Path(tmp), prepared, remove_temporaries=False)
            os.replace(Path(tmp) / VIDEO, out)
    except BaseException:
        for folder in created:  # innermost first; one that holds something stays
            with suppress(OSError):
                folder.rmdir()
        raise
    return prepared.info


def _checked_config(config: RenderConfig | None, video: bool) -> RenderConfig:
    """``config`` or the default, validated again: ``model_copy(update=...)`` does not validate,
    and a value that fails while drawing would fail after files have been replaced."""
    config = RenderConfig.model_validate((config or RenderConfig()).model_dump())
    if video and config.panel_px % 2:
        raise ValueError(f"the video needs an even panel_px, got {config.panel_px}")
    return config


def _locate(files: Mapping[FileRole, Path]) -> tuple[dict[FileRole, Path], dict[str, InputFile]]:
    """The files of the render roles that exist, and their identity (name, size, SHA-256)."""
    try:
        paths = {
            role: Path(files[role])
            for role in (*RENDER_ROLES, FileRole.BRAINMASK)
            if role in files and is_file(files[role])
        }
        if missing := [role.value for role in RENDER_ROLES if role not in paths]:
            raise RenderError("missing surface", "no file for " + ", ".join(missing))
        return paths, {role.value: _input_file(role, path) for role, path in paths.items()}
    except OSError as exc:
        raise _unreadable(exc) from exc


def _unreadable(exc: OSError) -> RenderError:
    # the file's name and the error, not its path, which could name the dataset
    named = isinstance(exc.filename, str | bytes | os.PathLike)
    name = Path(os.fsdecode(exc.filename)).name if named else "a file"
    return RenderError("render failure", f"unreadable input ({name}: {exc.strerror or exc})")


def _prepare(
    paths: Mapping[FileRole, Path],
    inputs: Mapping[str, InputFile],
    config: RenderConfig,
    video: bool,
) -> _Prepared:
    """Read the inputs and work out the whole of ``render.json``. Every :class:`RenderError`
    is raised here or in :func:`_locate`; nothing is written."""
    if video:
        # found now: write_mp4 comes after render.json is removed and every tile is written
        try:
            encode.ffmpeg_exe()
        except RuntimeError as exc:
            raise RenderError("render failure", f"the video cannot be encoded ({exc})") from exc
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
    # everything a click on a tile can name, grown by the margin: a mark then finds the same
    # nearest vertex in the map as on the full surface, for any snap radius up to the margin
    region = geometry.clickable_region(box, panels, config)
    vmap = vertexmap.build_vertex_map(surfaces, region, config.vertex_map_margin_mm)
    vertex_map = VertexMapInfo(
        file=VERTEX_MAP,
        n_lh=len(vmap["lh_index"]),
        n_rh=len(vmap["rh_index"]),
        n_lh_visible=int(vmap["lh_visible"].sum()),
        n_rh_visible=int(vmap["rh_visible"].sum()),
    )
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
        inputs=dict(inputs),
        poles=poles,
        box=box,
        window=window,
        panels=panels,
        modes=modes,
        video=VideoInfo(
            file=VIDEO,
            fps=config.fps,
            n_frames=len(modes["sync"].frames),
            width=width,
            height=height,
        )
        if video
        else None,
        poster=POSTER,
        vertex_map=vertex_map,
        versions={name: version(name) for name in _VERSIONED},
    )
    return _Prepared(info=info, t1=t1, surfaces=surfaces, vertex_map=vmap)


def _write(out_dir: Path, prepared: _Prepared, *, remove_temporaries: bool) -> None:
    """Write every file of the render to ``out_dir``, ``render.json`` last."""
    info, t1, surfaces = prepared.info, prepared.t1, prepared.surfaces
    config, panels, window = info.config, info.panels, info.window
    sync = info.modes["sync"].frames
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
    assert info.vertex_map is not None
    vertexmap.write_vertex_map(out_dir / info.vertex_map.file, prepared.vertex_map)
    _remove_stale(out_dir, info, temporaries=remove_temporaries)
    with encode.atomic_path(out_dir / RENDER_JSON) as tmp:
        tmp.write_bytes(info.to_json().encode())


@contextmanager
def _exclusive(out_dir: Path) -> Iterator[bool]:
    """Create ``out_dir`` and hold it for one render; yields whether a lock is held.

    The lock is ``flock`` on the directory itself, so it adds no file, and it ends when the
    descriptor is closed or the process dies, however that happens. A render that finds the
    directory locked waits here. Without ``fcntl`` (Windows), or on a file system that refuses
    the lock, there is none and the render goes ahead as it did before there was a lock.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    try:
        locked = False
        if fcntl is not None:
            try:
                descriptor = os.open(out_dir, os.O_RDONLY)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            except OSError:
                pass
        yield locked
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _input_file(role: FileRole, path: Path) -> InputFile:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # render.json is served to raters, so only a name that FreeSurfer gives a file of this role
    # is recorded, never one chosen by whoever stored the data. A link inside its own directory
    # (FreeSurfer 7: lh.pial -> lh.pial.T1) names the variant.
    names = [Path(candidate).name for candidate in ROLE_CANDIDATES[role]]
    target = path.resolve()
    if target.parent == path.parent.resolve() and target.name in names:
        name = target.name
    else:
        name = path.name if path.name in names else names[0]
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


def _remove_stale(out_dir: Path, info: RenderInfo, *, temporaries: bool) -> None:
    """Remove tiles and video of an earlier render that ``info`` does not list and, with
    ``temporaries``, temporary files that an interrupted writer left behind.

    Only names that a render writes are removed, whatever else the directory holds. A
    temporary file is known to be abandoned only while the directory is locked; without the
    lock it may belong to a render that is still running.
    """
    keep = {out_dir / name for name in _listed_files(info)}
    tiles = out_dir / "tiles"
    stale = [out_dir / VIDEO]
    stale += [path for path in tiles.glob("*.webp") if _TILE.fullmatch(path.name)]
    if temporaries:
        for folder, written in ((out_dir, _TOP_LEVEL), (tiles, _TILE)):
            for path in folder.glob(".*.tmp"):
                match = _TEMPORARY.fullmatch(path.name)
                if match is not None and written.fullmatch(match.group(1)):
                    stale.append(path)
    for path in stale:
        if path not in keep:
            path.unlink(missing_ok=True)
