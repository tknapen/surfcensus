"""Render one subject into a media directory (docs/rendering.md, "Output per subject").

``render.json`` is written last, so its presence marks a complete render. Source files are
only read; everything written lies inside the output directory, which may not lie inside a
source directory. Renders into one directory run one after the other (``_exclusive``).

With a template, the vertex map carries for every native vertex its counterpart on the
template mesh, and ``render.json`` the template's occipital pole on the native surface
(decision D12). The tiles, the poster and the video do not depend on the template: a
directory whose tiles are current is brought up to date with another template, or with
none, by writing ``vertex_map.npz`` and ``render.json`` alone (``_update_counterparts``).
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np

import surfcensus
from surfcensus._io import sha256_file
from surfcensus.config import MetricsConfig
from surfcensus.fsio.registration import SphereMap, build_sphere_map
from surfcensus.fsio.surfaces import Surface, read_surface
from surfcensus.fsio.volumes import Volume, intensity_window, read_volume
from surfcensus.log import get_logger
from surfcensus.render import encode, frames, geometry, vertexmap
from surfcensus.render.model import (
    PANEL_ORDER,
    RENDER_JSON,
    RENDER_SCHEMA_VERSION,
    Frame,
    FsaveragePole,
    InputFile,
    PanelName,
    PoleInfo,
    RenderConfig,
    RenderInfo,
    TemplateInfo,
    VertexMapInfo,
    VideoInfo,
    VolumeInfo,
    read_render,
)
from surfcensus.render.slicing import MeshSlicer
from surfcensus.sources.base import HEMIS, RENDER_ROLES, FileRole, Hemi
from surfcensus.sources.patterns import (
    ROLE_CANDIDATES,
    TEMPLATE_FILES,
    is_file,
    resolve_subject_files,
    variant_name,
)
from surfcensus.templates import (
    Template,
    TemplateError,
    TemplateNotFound,
    build_template,
    find_template,
    template_candidates,
)

try:
    import fcntl
except ImportError:  # Windows has no flock: renders into one directory are not serialized there
    fcntl = None

POSTER = "poster.webp"
VIDEO = "anim.mp4"
VERTEX_MAP = "vertex_map.npz"
_SURFACE_ROLES = (FileRole.LH_WHITE, FileRole.RH_WHITE, FileRole.LH_PIAL, FileRole.RH_PIAL)
_SPHERE_ROLES: dict[Hemi, FileRole] = {"lh": FileRole.LH_SPHERE_REG, "rh": FileRole.RH_SPHERE_REG}
_PIAL_ROLES: dict[Hemi, FileRole] = {"lh": FileRole.LH_PIAL, "rh": FileRole.RH_PIAL}
_TEMPLATE_SPHERE = TEMPLATE_FILES[0]  # surf/{h}.sphere.reg
_NO_COUNTERPARTS = "click marks will not map to fsaverage"
_VERSIONED = ("numpy", "matplotlib", "pillow", "nibabel")
# The names this module writes, which are the only ones it ever removes (_write,
# _remove_stale). A tile number is what "{index:03d}" writes: ASCII digits, three of them, or
# more without a leading zero.
_TILE = re.compile(rf"(?:{'|'.join(PANEL_ORDER)})_(?:[0-9]{{3}}|[1-9][0-9]{{3,}})(?:_c)?\.webp")
_TOP_LEVEL = re.compile("|".join(map(re.escape, (RENDER_JSON, POSTER, VIDEO, VERTEX_MAP))))
_TEMPORARY = re.compile(r"\.(.+)\.[0-9a-f]{16}\.tmp")  # encode.atomic_path, group 1: final name


class RenderError(Exception):
    """A subject that cannot be rendered.

    ``reason`` is one of the unratable reasons of ``data/protocol.yaml``: ``missing surface``,
    ``wrong orientation`` or ``render failure``; ``detail`` says what was found.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[RenderError], tuple[str, str]]:
        # pickle would call __init__ with the one argument of Exception.args
        return type(self), (self.reason, self.detail)


class OutputInSourceError(ValueError):
    """An output location in a directory that the inputs are read from. Such a directory is
    only read (hard rule in ``CLAUDE.md``), so nothing is rendered into it."""


@dataclass(frozen=True)
class _Located:
    """The input files of a render that exist, and their identity (name, size, SHA-256)."""

    paths: dict[FileRole, Path]
    inputs: dict[str, InputFile]
    no_sphere: dict[Hemi, str]
    """Per hemisphere without a ``sphere.reg`` among ``paths``, why; empty when the render
    has no template, which reads none."""


@dataclass(frozen=True)
class _Counterparts:
    """What a template adds to a render. All empty without a template, and when no
    hemisphere has a ``sphere.reg`` that can be used."""

    nearest_target: dict[Hemi, np.ndarray]
    """Per hemisphere that could be mapped, the template vertex nearest each native one."""
    poles: dict[Hemi, FsaveragePole]
    template: TemplateInfo | None


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
    template: Template | None = None,
    video: bool = True,
    force: bool = False,
) -> RenderInfo:
    """Write tiles, poster, video, vertex map and ``render.json`` of one subject to ``out_dir``.

    ``files`` maps roles to local paths; ``t1`` and the four surfaces are required, and
    ``brainmask`` is used when present. If ``out_dir`` already holds a complete render of the
    same inputs (size and SHA-256), config, ``video`` setting and template, its
    ``render.json`` is returned and nothing is written, unless ``force``. Otherwise the
    directory is rendered again, and what an earlier render left and the new one does not
    list is removed: its tiles, and its video if the ``render.json`` in the directory lists
    one. While another render writes to ``out_dir``, this one waits for it.

    With a ``template``, the two ``sphere_reg`` files are read as well, where ``files`` has
    them. For each hemisphere whose ``sphere.reg`` has the vertices of its pial surface,
    ``{hemi}_fsavg6`` of the vertex map holds the nearest template vertex of every vertex
    (``SphereMap.nearest_target``), and ``render.json`` gets the template's occipital pole
    on the native surface (``fsaverage_poles``). ``template`` and ``vertex_map.fsaverage``
    of ``render.json`` are set when at least one hemisphere was mapped. A hemisphere
    without a ``sphere.reg``, or with one that cannot be read, has another vertex count or
    is no sphere, keeps -1 and gets no pole; that is logged as a warning and is no failure
    of the render. The ``sphere.reg`` files that were read are among the ``inputs`` in
    either case.

    The tiles, the poster and the video do not depend on the template. When ``out_dir``
    holds a render that is whole and differs from the one asked for only in the template
    or in the ``sphere.reg`` files, ``vertex_map.npz`` and ``render.json`` are written and
    nothing else is touched, unless ``force``.

    Raises :class:`RenderError` for a subject that cannot be rendered, or for a video that
    cannot be encoded because there is no ffmpeg binary, and ValueError for a config that
    cannot be used or for an ``out_dir`` that is, or lies inside, a directory the inputs are
    read from (:class:`OutputInSourceError`, see :func:`_source_dirs`), before anything is
    written.
    """
    config = _checked_config(config, video)
    out_dir = Path(out_dir)
    located = _locate(files, spheres=template is not None)
    _refuse_output_inside(out_dir, _source_dirs(located.paths))
    wanted = None if template is None else _template_info(template)
    current = None if force else _current_tiles(out_dir, config, located.inputs, video)
    if current is not None and _up_to_date(current, located.inputs, wanted):
        return current
    # with current tiles there is nothing to refuse and, as a rule, nothing to draw
    prepared = None if current is not None else _prepare(located, config, video, template)
    with _exclusive(out_dir) as locked:
        if not force:
            # the render this one waited for may have left just what is asked for
            current = _current_tiles(out_dir, config, located.inputs, video)
            if current is not None:
                if _up_to_date(current, located.inputs, wanted):
                    return current
                updated = _update_counterparts(out_dir, current, located, template)
                if updated is not None:
                    return updated
        if prepared is None:
            prepared = _prepare(located, config, video, template)
        _write(out_dir, prepared, remove_temporaries=locked)
    return prepared.info


def three_plane(
    subject_dir: Path | str,
    out: Path | str,
    config: RenderConfig | None = None,
    *,
    template: Template | Path | Literal["auto"] | None = "auto",
    video: bool = True,
    force: bool = False,
) -> RenderInfo:
    """Render one FreeSurfer subject directory, without a run directory.

    ``out`` is the media directory to write, or, if it ends in ``.mp4``, the path of the video
    alone. ``video`` and ``force`` are the switches of :func:`render_subject`; a video alone is
    always rendered. An ``out`` that is the subject directory or lies inside it is refused
    with :class:`OutputInSourceError`: a source directory is only read.

    ``template`` says where the fsaverage counterparts of the vertex map come from:

    * ``"auto"``: the template that ``templates.find_template`` finds with the default
      metrics config: ``$FREESURFER_HOME/subjects/fsaverage``, else an ``fsaverage``
      directory next to the subject directory. When there is none, or the one found cannot
      be used, the subject is rendered without counterparts, and once it is rendered a
      warning says so and where a template was looked for;
    * a ``Path``: that fsaverage directory, built with the default metrics config.
      ``TemplateNotFound`` or ``TemplateError`` when it cannot be used, before anything is
      written;
    * a ``Template``: as given;
    * None: no counterparts.

    The template that was used is recorded in ``render.json``. An ``out`` in the fsaverage
    directory that the template was read from is refused like one in the subject directory.
    """
    try:
        files = resolve_subject_files(subject_dir)
    except OSError as exc:
        raise _unreadable(exc) from exc
    out = Path(out)
    to_video = out.suffix.lower() == ".mp4"
    if to_video and not video:
        raise ValueError(f"video=False contradicts the video output {out}")
    found, found_in, none_found = _resolve_template(Path(subject_dir), template)
    if found_in is not None:
        _refuse_output_inside(out, [found_in])
    if not to_video:
        info = render_subject(files, out, config, template=found, video=video, force=force)
        if none_found is not None:
            get_logger().warning(none_found)
        return info
    # A video has no marks, so nothing is said here when no template was found.
    # Every refusal comes first, so that a refused subject leaves no directory behind.
    config = _checked_config(config, video=True)
    located = _locate(files, spheres=found is not None)
    _refuse_output_inside(out, _source_dirs(located.paths))
    prepared = _prepare(located, config, True, found)
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


def _source_dirs(paths: Mapping[FileRole, Path]) -> set[Path]:
    """The directories that the inputs are read from, into which nothing is written.

    For a file at its FreeSurfer place in a subject directory (``<subject>/mri/T1.mgz``) that
    is the subject directory, for any other file the directory that holds it; and for a link,
    the directory that holds its target as well.
    """
    folders: set[Path] = set()
    for role, path in paths.items():
        folder = path.parent
        for candidate in ROLE_CANDIDATES[role]:
            parts = Path(candidate).parts
            if path.parts[-len(parts) :] == parts:
                folder = path.parents[len(parts) - 1]
        folders |= {folder, path.resolve().parent}
    return folders


def _refuse_output_inside(out: Path, sources: Iterable[Path]) -> None:
    """Raise :class:`OutputInSourceError` if ``out`` is one of ``sources`` or lies inside one.

    Links are followed, and directories are compared as the file system identifies them, so
    another spelling of a source (a link, another letter case on a file system that ignores
    it) is found too. ``out`` need not exist: its parents that do exist decide. A directory
    that holds a source is not inside it and is accepted.
    """
    target = out.resolve()
    for folder in (target, *target.parents):
        for source in sources:
            try:
                inside = folder.samefile(source)
            except OSError:  # a part of ``out`` that does not exist yet
                continue
            if inside:
                raise OutputInSourceError(
                    f"the output {out} lies in a directory that the inputs are read from; "
                    "such a directory is only read, choose a location outside it"
                )


def _resolve_template(
    subject_dir: Path, template: Template | Path | Literal["auto"] | None
) -> tuple[Template | None, Path | None, str | None]:
    """The template of a call of :func:`three_plane`; the directory it was read from, when
    that is known; and, when ``"auto"`` found none that can be used, what to tell the user
    once the subject is rendered. A subject that cannot be rendered gets its own message
    and not this one."""
    if template is None or isinstance(template, Template):
        return template, None, None
    defaults = MetricsConfig()
    if template != "auto":
        directory = Path(template)
        built = build_template(directory, "explicit", mesh=defaults.mesh, regions=defaults.regions)
        return built, directory, None
    # next to the subject directory under the name it was given by, as a local source looks
    named = subject_dir if subject_dir.name not in ("", "..") else subject_dir.resolve()
    adjacent = named.parent / "fsaverage"
    try:
        found = find_template(config=defaults, adjacent=adjacent)
    except TemplateNotFound as exc:
        looked_in = "; ".join(f"{where}: {state}" for _, where, state in exc.looked_in)
        none_found = (
            f"no fsaverage template found: {_NO_COUNTERPARTS}. Looked in {looked_in}. Give "
            "an fsaverage directory as the template (--fsaverage DIR on the command line), "
            "set FREESURFER_HOME, or put one next to the subject directory."
        )
        return None, None, none_found
    except TemplateError as exc:
        unusable = f"the fsaverage template that was found cannot be used ({exc})"
        return None, None, f"{unusable}: {_NO_COUNTERPARTS}"
    return found, dict(template_candidates(adjacent=adjacent)).get(found.origin), None


def _template_info(template: Template) -> TemplateInfo:
    return TemplateInfo(
        mesh=template.mesh,
        template_id=template.template_id,
        sphere_sha256={hemi: template.files[_TEMPLATE_SPHERE.format(h=hemi)][1] for hemi in HEMIS},
    )


def _locate(files: Mapping[FileRole, Path], *, spheres: bool = False) -> _Located:
    """The files of the render roles that exist and, with ``spheres``, the ``sphere.reg``
    files. A ``sphere.reg`` that is absent or cannot be examined is left out with its
    reason: the render does not depend on it."""
    try:
        paths = {
            role: Path(files[role])
            for role in (*RENDER_ROLES, FileRole.BRAINMASK)
            if role in files and is_file(files[role])
        }
        if missing := [role.value for role in RENDER_ROLES if role not in paths]:
            raise RenderError("missing surface", "no file for " + ", ".join(missing))
        inputs = {role.value: _input_file(role, path) for role, path in paths.items()}
    except OSError as exc:
        raise _unreadable(exc) from exc
    no_sphere: dict[Hemi, str] = {}
    for hemi, role in _SPHERE_ROLES.items() if spheres else ():
        try:
            if role not in files or not is_file(files[role]):
                no_sphere[hemi] = f"no {hemi}.sphere.reg"
                continue
            inputs[role.value] = _input_file(role, Path(files[role]))
            paths[role] = Path(files[role])
        except OSError as exc:
            no_sphere[hemi] = _unreadable(exc, Path(files[role])).detail
    return _Located(paths=paths, inputs=inputs, no_sphere=no_sphere)


def _unreadable(exc: Exception, path: Path | None = None) -> RenderError:
    """The failure for an input that cannot be read: ``path``, or else the file that an
    OSError names. The message has the file's name and the error, never the path, which could
    name the dataset; nibabel puts the path into some of its messages (an empty file)."""
    if path is None and isinstance(exc, OSError):
        if isinstance(exc.filename, str | bytes | os.PathLike):
            path = Path(os.fsdecode(exc.filename))
    name = path.name if path is not None else "a file"
    if isinstance(exc, OSError) and exc.strerror:
        problem = exc.strerror
    else:
        problem = str(exc) if path is None else str(exc).replace(str(path), name)
    return RenderError("render failure", f"unreadable input ({name}: {problem})")


def _map_sphere(located: _Located, hemi: Hemi, n_pial: int, template: Template) -> SphereMap | str:
    """The mapping between one hemisphere's ``sphere.reg`` and the template sphere, or why
    there is none."""
    if hemi in located.no_sphere:
        return located.no_sphere[hemi]
    path = located.paths[_SPHERE_ROLES[hemi]]
    try:
        sphere = read_surface(path)
    except Exception as exc:  # nibabel raises many kinds of error on a damaged file
        return _unreadable(exc, path).detail
    if len(sphere.vertices) != n_pial:
        return (
            f"{hemi}.sphere.reg has {len(sphere.vertices)} vertices and the pial surface {n_pial}"
        )
    try:
        return build_sphere_map(sphere, template.hemis[hemi].sphere)
    except ValueError as exc:  # no sphere: radii that differ, are zero or are not finite
        return f"{hemi}.sphere.reg: {exc}"


def _counterparts(
    located: _Located,
    pial: Mapping[Hemi, np.ndarray],
    geometric: Mapping[str, PoleInfo],
    template: Template | None,
) -> _Counterparts:
    """Map each hemisphere's ``sphere.reg`` onto the template sphere.

    ``pial`` holds the pial vertices of every hemisphere that has a ``sphere.reg`` among the
    located files, and ``geometric`` the geometric poles. A hemisphere that cannot be
    mapped is left out with a warning; nothing raises. Why a hemisphere fails does not
    depend on the template: a file that cannot be read, another vertex count than the pial
    surface, or vertices that do not lie on a sphere.
    """
    nearest_target: dict[Hemi, np.ndarray] = {}
    poles: dict[Hemi, FsaveragePole] = {}
    if template is None:
        return _Counterparts(nearest_target, poles, None)
    for hemi in HEMIS:
        mapped = _map_sphere(located, hemi, len(pial[hemi]) if hemi in pial else 0, template)
        if isinstance(mapped, str):
            get_logger().warning("no fsaverage counterparts for %s: %s", hemi, mapped)
            continue
        template_vertex = template.hemis[hemi].pole_vertex
        native_vertex = int(mapped.nearest_source[template_vertex])
        there = pial[hemi][native_vertex]
        pole = pial[hemi][geometric[hemi].vertex]
        # squared and summed in Python: the digits then do not depend on numpy's linear algebra
        squares = [(float(there[axis]) - float(pole[axis])) ** 2 for axis in range(3)]
        nearest_target[hemi] = mapped.nearest_target
        poles[hemi] = FsaveragePole(
            template_vertex=template_vertex,
            native_vertex=native_vertex,
            tkr=(float(there[0]), float(there[1]), float(there[2])),
            distance_mm=math.sqrt(squares[0] + squares[1] + squares[2]),
        )
    return _Counterparts(nearest_target, poles, _template_info(template) if poles else None)


def _prepare(
    located: _Located, config: RenderConfig, video: bool, template: Template | None = None
) -> _Prepared:
    """Read the inputs and work out the whole of ``render.json``. Every :class:`RenderError`
    is raised here or in :func:`_locate`; nothing is written."""
    paths, inputs = located.paths, located.inputs
    if video:
        # found now: write_mp4 comes after render.json is removed and every tile is written
        try:
            encode.ffmpeg_exe()
        except RuntimeError as exc:
            raise RenderError("render failure", f"the video cannot be encoded ({exc})") from exc
    reading = paths[FileRole.T1]
    try:
        t1 = read_volume(reading)
        mask = None
        if FileRole.BRAINMASK in paths:
            reading = paths[FileRole.BRAINMASK]
            mask = read_volume(reading)
        surfaces: dict[str, Surface] = {}
        for role in _SURFACE_ROLES:
            reading = paths[role]
            surfaces[role.value] = read_surface(reading)
    except Exception as exc:  # nibabel raises many kinds of error on a damaged file
        raise _unreadable(exc, reading) from exc
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
    found = _counterparts(
        located, {hemi: surfaces[f"{hemi}_pial"].vertices for hemi in HEMIS}, poles, template
    )
    vmap = vertexmap.fill_fsaverage(vmap, found.nearest_target)
    vertex_map = VertexMapInfo(
        file=VERTEX_MAP,
        n_lh=len(vmap["lh_index"]),
        n_rh=len(vmap["rh_index"]),
        n_lh_visible=int(vmap["lh_visible"].sum()),
        n_rh_visible=int(vmap["rh_visible"].sum()),
        fsaverage=None if found.template is None else found.template.mesh,
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
        template=found.template,
        fsaverage_poles=dict(found.poles) or None,
        versions={name: version(name) for name in _VERSIONED},
    )
    return _Prepared(info=info, t1=t1, surfaces=surfaces, vertex_map=vmap)


def _update_counterparts(
    out_dir: Path, stored: RenderInfo, located: _Located, template: Template | None
) -> RenderInfo | None:
    """Bring a render whose tiles are current up to date with ``template`` and the
    ``sphere.reg`` files, by writing ``vertex_map.npz`` and ``render.json`` alone.

    ``stored`` is the ``render.json`` in ``out_dir``, of the same config and the same tile
    inputs. Everything in it that describes the tiles stays, the library versions included.
    ``render.json`` is removed first and written last, as in a full render. Returns None,
    with nothing written, when the vertex map in the directory is not the one that
    ``stored`` describes or a pial surface cannot be read: the caller then renders in full.
    """
    sizes = stored.vertex_map
    if sizes is None:
        return None
    try:
        vmap = vertexmap.read_vertex_map(out_dir / sizes.file)
        pial = {
            hemi: read_surface(located.paths[_PIAL_ROLES[hemi]]).vertices
            for hemi, role in _SPHERE_ROLES.items()
            if role in located.paths
        }
    except Exception:  # absent, damaged, or without the arrays of a vertex map
        return None
    for hemi, n, n_visible in (
        ("lh", sizes.n_lh, sizes.n_lh_visible),
        ("rh", sizes.n_rh, sizes.n_rh_visible),
    ):
        shapes = {field: vmap[f"{hemi}_{field}"].shape for field in ("index", "pial", "white")}
        visible = vmap[f"{hemi}_visible"]
        if shapes != {"index": (n,), "pial": (n, 3), "white": (n, 3)}:
            return None
        if visible.shape != (n,) or int(np.count_nonzero(visible)) != n_visible:
            return None
        if hemi in pial and not 0 <= stored.poles[hemi].vertex < len(pial[hemi]):
            return None
    found = _counterparts(located, pial, stored.poles, template)
    try:
        vmap = vertexmap.fill_fsaverage(vmap, found.nearest_target)
    except ValueError:  # the map names a vertex that the surface does not have
        return None
    info = RenderInfo.model_validate(
        stored.model_copy(
            update={
                "inputs": dict(located.inputs),
                "vertex_map": sizes.model_copy(
                    update={"fsaverage": None if found.template is None else found.template.mesh}
                ),
                "template": found.template,
                "fsaverage_poles": dict(found.poles) or None,
            }
        ).model_dump()
    )
    (out_dir / RENDER_JSON).unlink(missing_ok=True)  # incomplete from here on
    vertexmap.write_vertex_map(out_dir / sizes.file, vmap)
    with encode.atomic_path(out_dir / RENDER_JSON) as tmp:
        tmp.write_bytes(info.to_json().encode())
    return info


def _write(out_dir: Path, prepared: _Prepared, *, remove_temporaries: bool) -> None:
    """Write every file of the render to ``out_dir``, ``render.json`` last."""
    info, t1, surfaces = prepared.info, prepared.t1, prepared.surfaces
    config, panels, window = info.config, info.panels, info.window
    sync = info.modes["sync"].frames
    # The video of the render that render.json describes goes with it, unless this render
    # writes one. It goes first: if the two removals are parted, render.json lists a video
    # that is absent, which is an incomplete render, and not a video that nothing lists.
    if info.video is None and _lists_video(out_dir):
        _unlink_file(out_dir / VIDEO)
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
    digest = sha256_file(path)
    # render.json lies with the media of a run, so only a name that FreeSurfer gives a file of
    # this role is recorded, never one chosen by whoever stored the data (patterns.variant_name)
    name = variant_name(role, path)
    return InputFile(name=name, size=path.stat().st_size, sha256=digest)


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


def _stored_render(out_dir: Path, config: RenderConfig, video: bool) -> RenderInfo | None:
    """The ``render.json`` in ``out_dir`` if it is of this schema, of ``config`` and of this
    video setting, else None. Says nothing about the other files."""
    info = read_render(out_dir)
    if info is None:
        return None
    same = (
        info.schema_version == RENDER_SCHEMA_VERSION
        and info.config_hash == config.config_hash()
        and (info.video is not None) == video
    )
    return info if same else None


def _tile_inputs(inputs: Mapping[str, InputFile]) -> dict[str, InputFile]:
    """The inputs that the tiles are made from: all but the ``sphere.reg`` files."""
    spheres = {role.value for role in _SPHERE_ROLES.values()}
    return {role: entry for role, entry in inputs.items() if role not in spheres}


def _current_tiles(
    out_dir: Path, config: RenderConfig, inputs: Mapping[str, InputFile], video: bool
) -> RenderInfo | None:
    """The render already in ``out_dir`` if it is whole and its tiles are those asked for:
    this schema, config and video setting, the same tile inputs, every listed file there.
    Else None. Says nothing about the template."""
    info = _stored_render(out_dir, config, video)
    if info is None or _tile_inputs(info.inputs) != _tile_inputs(inputs):
        return None
    return info if all((out_dir / name).is_file() for name in _listed_files(info)) else None


def _up_to_date(
    current: RenderInfo, inputs: Mapping[str, InputFile], wanted: TemplateInfo | None
) -> bool:
    """Whether a render with current tiles also has the counterparts asked for: the same
    ``sphere.reg`` files among its inputs and the template ``wanted``.

    A render that read these ``sphere.reg`` files and names no template could map neither
    hemisphere. That does not depend on the template (:func:`_counterparts`), so it is up
    to date with any.
    """
    return current.inputs == inputs and current.template in (None, wanted)


def render_on_disk(out_dir: Path | str, config: RenderConfig, *, video: bool) -> bool:
    """Whether ``out_dir`` holds a whole render of ``config`` with this video setting: a
    ``render.json`` that says so, and something under every name it lists.

    This is the test for skipping a render that is recorded as done. It opens no source
    file and does not compare the inputs, which the caller knows from its listing. The
    names are looked up in a listing of each directory, two per render, and not one by one:
    a census asks this of every subject each time ``render`` or ``status`` runs.
    """
    out_dir = Path(out_dir)
    info = _stored_render(out_dir, config, video)
    if info is None:
        return False
    listed: dict[str, set[str]] = {}
    for name in _listed_files(info):
        parent, _, base = name.rpartition("/")
        listed.setdefault(parent, set()).add(base)
    for parent, names in listed.items():
        try:
            present = os.listdir(out_dir / parent)
        except OSError:
            return False
        if not names.issubset(present):
            return False
    return True


def _lists_video(out_dir: Path) -> bool:
    """Whether the ``render.json`` in ``out_dir`` lists ``anim.mp4``, which is then the video
    of a render and not a file that someone else put there.

    Read as plain JSON: a ``render.json`` of an earlier schema lists its video too, and does
    not load as a :class:`RenderInfo`.
    """
    try:
        doc = json.loads((out_dir / RENDER_JSON).read_bytes())
    except (OSError, ValueError):  # absent, unreadable, or not JSON
        return False
    video = doc.get("video") if isinstance(doc, dict) else None
    return isinstance(video, dict) and video.get("file") == VIDEO


def _unlink_file(path: Path) -> None:
    """Remove a file or a link. A directory, or anything else, is not what a render wrote and
    stays; ``unlink`` on a directory would also raise in the middle of a render."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)


def _remove_stale(out_dir: Path, info: RenderInfo, *, temporaries: bool) -> None:
    """Remove tiles of an earlier render that ``info`` does not list and, with
    ``temporaries``, temporary files that an interrupted writer left behind.

    Only names that a render writes are removed, and only files, whatever else the directory
    holds. A temporary file is known to be abandoned only while the directory is locked;
    without the lock it may belong to a render that is still running. The video of an earlier
    render is removed in :func:`_write`, while the ``render.json`` that lists it is there.
    """
    keep = {out_dir / name for name in _listed_files(info)}
    tiles = out_dir / "tiles"
    stale = [path for path in tiles.glob("*.webp") if _TILE.fullmatch(path.name)]
    if temporaries:
        for folder, written in ((out_dir, _TOP_LEVEL), (tiles, _TILE)):
            for path in folder.glob(".*.tmp"):
                match = _TEMPORARY.fullmatch(path.name)
                if match is not None and written.fullmatch(match.group(1)):
                    stale.append(path)
    for path in stale:
        if path not in keep:
            _unlink_file(path)
