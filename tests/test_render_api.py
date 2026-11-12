"""End-to-end tests of render/api.py on the synthetic subjects (docs/rendering.md)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files as package_files
from pathlib import Path

import imageio_ffmpeg
import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy import ndimage
from scipy.spatial import cKDTree
from synth import (  # pyrefly: ignore[missing-import]
    T1_GRAY,
    T1_HEAD,
    T1_WHITE,
    SyntheticSubject,
    file_modes_apply,
    write_synthetic_subject,
)

import surfcensus
from surfcensus.fsio.surfaces import read_surface
from surfcensus.fsio.volumes import Volume, intensity_window, read_volume
from surfcensus.render import (
    RenderError,
    api,
    encode,
    frames,
    geometry,
    render_subject,
    three_plane,
)
from surfcensus.render.encode import read_image
from surfcensus.render.model import PANEL_ORDER, InputFile, RenderConfig, RenderInfo
from surfcensus.render.slicing import MeshSlicer, plane_segments
from surfcensus.render.vertexmap import read_vertex_map
from surfcensus.sources.base import FileRole
from surfcensus.sources.patterns import resolve_subject_files

# a small box inside the left hemisphere: 13 slices per panel, tiles of 64 px
SMALL = RenderConfig(
    panel_px=64, box_x_mm=(-30.0, -18.0), box_y_mm=(0.0, 12.0), box_z_mm=(-6.0, 6.0)
)
NARROW = SMALL.model_copy(update={"box_x_mm": (-30.0, -24.0)})  # 7 sagittal slices
SURFACES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
OLD_NS = 1_000_000_000_000_000_000  # an mtime in 2001


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def source_state(root: Path) -> dict[str, tuple[int, int, str]]:
    """Relative path -> (size, mtime in ns, SHA-256 or link target) of everything under root."""
    state = {}
    for p in sorted(root.rglob("*")):
        content = os.readlink(p) if p.is_symlink() else ""
        if p.is_file() and not p.is_symlink():
            content = hashlib.sha256(p.read_bytes()).hexdigest()
        state[p.relative_to(root).as_posix()] = (p.lstat().st_size, p.lstat().st_mtime_ns, content)
    return state


def expected_files(counts: dict[str, int], video: bool = True) -> set[str]:
    """The output set of docs/rendering.md for the given slice counts."""
    names = {"poster.webp", "vertex_map.npz", "render.json"} | ({"anim.mp4"} if video else set())
    for panel, n in counts.items():
        names |= {f"tiles/{panel}_{i:03d}.webp" for i in range(n)}
        names |= {f"tiles/{panel}_{i:03d}_c.webp" for i in range(n)}
    return names


def age(root: Path) -> None:
    for p in root.rglob("*"):
        os.utime(p, ns=(OLD_NS, OLD_NS))


def aged(root: Path) -> dict[str, bool]:
    return {
        p.relative_to(root).as_posix(): p.stat().st_mtime_ns == OLD_NS
        for p in root.rglob("*")
        if p.is_file()
    }


def video_frames(path: Path) -> tuple[dict, list[np.ndarray]]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    return meta, [np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3) for f in reader]


@dataclass(frozen=True)
class Rendered:
    subject: SyntheticSubject
    out: Path
    info: RenderInfo
    source_before: dict[str, tuple[int, int, str]]
    source_after: dict[str, tuple[int, int, str]]


def _render(subject: SyntheticSubject, out: Path, **kwargs: bool) -> Rendered:
    before = source_state(subject.path)
    info = render_subject(resolve_subject_files(subject.path), out, **kwargs)
    return Rendered(subject, out, info, before, source_state(subject.path))


@pytest.fixture(scope="module")
def default_render(
    synth_subject: SyntheticSubject, tmp_path_factory: pytest.TempPathFactory
) -> Rendered:
    """The 1 mm subject with the default config."""
    return _render(synth_subject, tmp_path_factory.mktemp("media") / "rendered-here")


@pytest.fixture(scope="module")
def hires_render(
    synth_subject_hires: SyntheticSubject, tmp_path_factory: pytest.TempPathFactory
) -> Rendered:
    """The 0.8 mm subject with the default config, without video."""
    return _render(
        synth_subject_hires, tmp_path_factory.mktemp("media_hires") / "rendered-here", video=False
    )


@pytest.fixture(scope="module")
def coarse_subject(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    """642 vertices per surface; tests copy it before changing anything."""
    return write_synthetic_subject(tmp_path_factory.mktemp("coarse"), subdivisions=3)


@pytest.fixture
def subject_copy(coarse_subject: SyntheticSubject, tmp_path: Path) -> Path:
    return Path(shutil.copytree(coarse_subject.path, tmp_path / "copy" / "sub-copy"))


@pytest.fixture(scope="module")
def small_render(
    coarse_subject: SyntheticSubject, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, RenderInfo]:
    """A pristine render of the coarse subject with SMALL; tests copy the directory."""
    out = tmp_path_factory.mktemp("small") / "media"
    return out, render_subject(resolve_subject_files(coarse_subject.path), out, SMALL)


@pytest.fixture
def small_copy(small_render: tuple[Path, RenderInfo], tmp_path: Path) -> Path:
    return Path(shutil.copytree(small_render[0], tmp_path / "media"))


# ---------------------------------------------------------------- the output set


def test_output_set_and_counts(default_render: Rendered) -> None:
    out, info = default_render.out, default_render.info
    counts = {name: len(info.panels[name].slices) for name in PANEL_ORDER}
    assert counts == {"sagittal": 61, "coronal": 54, "axial": 61}
    written = tree(out)
    assert set(written) == expected_files(counts)
    assert len(written) == 2 * 176 + 4
    assert len(info.modes["sync"].frames) == 120 and len(info.modes["seq"].frames) == 346
    assert read_image(out / "tiles/coronal_053.webp").shape == (320, 320, 3)
    assert read_image(out / "tiles/coronal_053_c.webp").shape == (320, 320, 4)
    assert read_image(out / "poster.webp").shape == (320, 960, 3)
    meta, video = video_frames(out / "anim.mp4")
    assert (meta["size"], meta["fps"], len(video)) == ((960, 320), 12, 120)
    assert info.video is not None
    assert (info.video.file, info.video.n_frames, info.video.fps) == ("anim.mp4", 120, 12)
    assert (info.video.width, info.video.height) == (960, 320)


def test_render_json_is_the_returned_info(default_render: Rendered) -> None:
    out, info, subject = default_render.out, default_render.info, default_render.subject
    text = (out / "render.json").read_text()
    assert text == info.to_json()
    assert RenderInfo.model_validate_json(text) == info
    assert info.schema_version == 2 and info.surfcensus_version == surfcensus.__version__
    assert info.config == RenderConfig() and info.config_hash == RenderConfig().config_hash()
    assert info.versions == {
        name: version(name) for name in ("numpy", "matplotlib", "pillow", "nibabel")
    }

    header = nib.load(subject.path / "mri" / "T1.mgz").header
    np.testing.assert_array_equal(np.array(info.volume.vox2ras_tkr), header.get_vox2ras_tkr())
    np.testing.assert_array_equal(np.array(info.volume.vox2ras), header.get_vox2ras())
    assert info.volume.shape == subject.shape
    assert info.volume.voxel_size_mm == (1.0, 1.0, 1.0)
    assert info.poles["lh"].tkr == (-34.0, -70.5, 5.0)
    assert info.poles["rh"].tkr == (26.0, -72.5, 5.0)
    assert info.box.centre[1:] == (-71.5, 5.0)
    assert info.window == (float(T1_HEAD), 110.0)

    names = {
        "t1": "mri/T1.mgz",
        "brainmask": "mri/brainmask.mgz",
        **{f"{h}_{s}": f"surf/{h}.{s}" for h in ("lh", "rh") for s in ("white", "pial")},
    }
    assert set(info.inputs) == set(names)
    for role, relative in names.items():
        content = (subject.path / relative).read_bytes()
        used = info.inputs[role]
        assert used.name == Path(relative).name
        assert (used.size, used.sha256) == (len(content), hashlib.sha256(content).hexdigest())

    vmap = read_vertex_map(out / "vertex_map.npz")
    assert info.vertex_map is not None and info.vertex_map.fsaverage is None
    assert (info.vertex_map.n_lh, info.vertex_map.n_rh) == (
        len(vmap["lh_index"]),
        len(vmap["rh_index"]),
    )
    assert (info.vertex_map.n_lh_visible, info.vertex_map.n_rh_visible) == (
        int(vmap["lh_visible"].sum()),
        int(vmap["rh_visible"].sum()),
    )
    assert info.vertex_map.n_lh > info.vertex_map.n_lh_visible > 500
    assert info.vertex_map.n_rh > info.vertex_map.n_rh_visible > 500
    assert set(vmap["lh_fsavg6"].tolist()) == {-1}


def test_render_json_describes_a_volume_that_is_not_a_1_mm_cube(
    hires_render: Rendered, tmp_path: Path
) -> None:
    """Shape, voxel size and pole vertices where a wrong value shows: the synthetic subjects
    are cubes whose two hemispheres have their pole at the same vertex number."""
    assert hires_render.info.volume.voxel_size_mm == pytest.approx((0.8, 0.8, 0.8), abs=1e-6)
    assert hires_render.info.volume.shape == (200, 200, 200)

    # three different dimensions, and the right pole moved to another vertex
    subject = write_synthetic_subject(tmp_path, shape=(160, 176, 168), subdivisions=3)
    path = subject.path / "surf" / "rh.pial"
    vertices, faces, meta, stamp = fs.read_geometry(str(path), read_metadata=True, read_stamp=True)
    vertices[5, 1] = vertices[:, 1].min() - 1.0
    fs.write_geometry(str(path), vertices, faces, stamp, meta)
    info = three_plane(subject.path, tmp_path / "media", SMALL, video=False)
    assert info.volume.shape == (160, 176, 168)
    assert info.volume.voxel_size_mm == (1.0, 1.0, 1.0)
    for hemi in ("lh", "rh"):
        pial = read_surface(subject.path / "surf" / f"{hemi}.pial").vertices
        assert info.poles[hemi].vertex == int(np.argmin(pial[:, 1]))
        assert info.poles[hemi].tkr == tuple(float(a) for a in pial[info.poles[hemi].vertex])
    assert info.poles["rh"].vertex == 5 != info.poles["lh"].vertex


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "/abs/T1.mgz", "a\\b", "..\\T1.mgz"])
def test_an_input_is_recorded_by_file_name_only(name: str) -> None:
    """render.json is served to raters, and a path could name the dataset."""
    with pytest.raises(ValidationError):
        InputFile(name=name, size=1, sha256="0" * 64)
    assert InputFile(name="lh.pial.T1", size=1, sha256="0" * 64).name == "lh.pial.T1"


def test_inputs_larger_than_a_megabyte_are_hashed_whole(tmp_path: Path) -> None:
    """A real surface takes 5 MB; the other synthetic inputs are all smaller than 0.4 MB."""
    subject = write_synthetic_subject(tmp_path, subdivisions=6).path  # 40,962 vertices
    info = three_plane(subject, tmp_path / "media", SMALL, video=False)
    for role in SURFACES:
        content = (subject / "surf" / role.replace("_", ".")).read_bytes()
        assert len(content) > 1 << 20
        used = info.inputs[role]
        assert (used.size, used.sha256) == (len(content), hashlib.sha256(content).hexdigest())


def test_render_json_names_no_directory_and_no_subject(default_render: Rendered) -> None:
    out, subject = default_render.out, default_render.subject
    text = (out / "render.json").read_text()
    for path in (out, subject.path):
        named = [path.name, path.parent.name, *(str(p) for p in path.resolve().parents[:-1])]
        assert [name for name in named if name in text] == []

    def strings(node: object) -> list[str]:
        if isinstance(node, dict):
            return [s for k, v in node.items() for s in (k, *strings(v))]
        if isinstance(node, list):
            return [s for v in node for s in strings(v)]
        return [node] if isinstance(node, str) else []

    with_separator = {s for s in strings(json.loads(text)) if "/" in s or "\\" in s}
    assert with_separator == {"tiles/{panel}_{index:03d}.webp", "tiles/{panel}_{index:03d}_c.webp"}


def test_every_tile_shows_its_slice(default_render: Rendered) -> None:
    """Tile ``i`` of a panel is slice ``i`` of ``render.json``: grayscale within the loss of
    WebP, contours exactly."""
    out, info, subject = default_render.out, default_render.info, default_render.subject
    t1 = read_volume(subject.path / "mri" / "T1.mgz")
    slicers = {}
    for name in SURFACES:
        surface = read_surface(subject.path / "surf" / name.replace("_", "."))
        slicers[name] = MeshSlicer(surface.vertices, surface.faces)
    empty = 0
    with frames.ContourRenderer(320, 320, info.config.line_width_px) as pen:
        for name, panel in info.panels.items():
            for pos in range(len(panel.slices)):
                gray = read_image(out / info.tiles.gray_path(name, pos))
                assert np.ptp(gray.astype(int), axis=2).max() <= 1
                expected = frames.gray_tile(t1.data, panel, pos, info.window)
                assert np.abs(gray[..., 0].astype(int) - expected).mean() < 2.0, (name, pos)
                layers = frames.contour_layers(panel, pos, t1.vox2ras_tkr, slicers, info.config)
                contours = read_image(out / info.tiles.contours_path(name, pos))
                np.testing.assert_array_equal(contours, pen.render(layers), str((name, pos)))
                empty += not contours[..., 3].any()
    # no contour between the hemispheres (x from -5.5 to -2.5) and behind the poles (y < -72.5)
    assert empty == sum(-5.5 <= x <= -2.5 for x in info.panels["sagittal"].slice_mm) + sum(
        y <= -72.5 for y in info.panels["coronal"].slice_mm
    )
    assert empty == 10
    between = info.panels["sagittal"].slice_mm.index(-4.0)
    assert not read_image(out / info.tiles.contours_path("sagittal", between)).any()
    # lines are opaque at their core, in the two configured colours
    tile = read_image(out / info.tiles.contours_path("coronal", 30))
    core = {tuple(int(c) for c in rgb) for rgb in tile[tile[..., 3] == 255][:, :3]}
    assert {(0xFF, 0xD4, 0x00), (0xFF, 0x3B, 0x30)} <= core


def test_poster_and_video_follow_the_synchronized_frame_table(default_render: Rendered) -> None:
    out, info = default_render.out, default_render.info
    blended = {
        name: [
            frames.blend(
                read_image(out / info.tiles.gray_path(name, pos))[..., 0],
                read_image(out / info.tiles.contours_path(name, pos)),
            )
            for pos in range(len(panel.slices))
        ]
        for name, panel in info.panels.items()
    }
    sync = info.modes["sync"].frames
    forward = [
        frames.composite_strip(
            {name: blended[name][pos] for name, pos in zip(PANEL_ORDER, frame, strict=True)},
            info.panels,
            frame,
            info.config,
        ).astype(int)
        for frame in sync[:61]
    ]

    def best_match(image: np.ndarray) -> tuple[int, float]:
        errors = [float(np.abs(image.astype(int) - strip).mean()) for strip in forward]
        return int(np.argmin(errors)), min(errors)

    # the poster is the middle forward frame
    assert best_match(read_image(out / "poster.webp"))[0] == 30
    assert best_match(read_image(out / "poster.webp"))[1] < 3.0
    # video frame k shows table entry k: forward for k <= 60, then back without the end frames
    _, video = video_frames(out / "anim.mp4")
    for k in (0, 1, 17, 30, 59, 60, 61, 62, 90, 118, 119):
        f, error = best_match(video[k])
        assert sync[k] == sync[f] and f == (k if k <= 60 else 120 - k), k
        assert error < 4.0, (k, error)
    # the contours are in the picture: yellow where the pial line is fully opaque
    for image, frame in ((read_image(out / "poster.webp"), sync[30]), (video[17], sync[17])):
        layers = [
            read_image(out / info.tiles.contours_path(name, pos))
            for name, pos in zip(PANEL_ORDER, frame, strict=True)
        ]
        pial = np.concatenate([(t[..., 3] == 255) & (t[..., 1] >= 180) for t in layers], axis=1)
        red_over_blue = image[pial][:, 0].astype(int) - image[pial][:, 2]
        assert pial.sum() > 100 and red_over_blue.mean() > 80, red_over_blue.mean()


def test_poster_and_video_show_the_crosshair_and_letters_of_their_frame(
    default_render: Rendered,
) -> None:
    """On every strip the two crosshair lines of each panel are where the other two planes of
    that frame are, and the edge letters are there (docs/rendering.md, "What a rater sees").
    The lines and letters are too few pixels to show in an error over the whole strip, so the
    strip is compared on those pixels alone, with and without them drawn."""
    out, info = default_render.out, default_render.info
    sync = info.modes["sync"].frames
    size = info.config.panel_px
    _, video = video_frames(out / "anim.mp4")
    shown = [("poster", read_image(out / "poster.webp"), sync[30])]
    shown += [(f"video frame {k}", video[k], sync[k]) for k in (0, 17, 45, 60, 90)]
    luma = np.array([0.2126, 0.7152, 0.0722])  # at full resolution in 4:2:0, unlike chroma
    for label, image, frame in shown:
        tiles = {
            name: frames.blend(
                read_image(out / info.tiles.gray_path(name, pos))[..., 0],
                read_image(out / info.tiles.contours_path(name, pos)),
            )
            for name, pos in zip(PANEL_ORDER, frame, strict=True)
        }
        plain = np.concatenate([tiles[name] for name in PANEL_ORDER], axis=1).astype(int)
        strip = frames.composite_strip(tiles, info.panels, frame, info.config).astype(int)
        cross = geometry.crosshair_pixels(dict(info.panels), frame)
        masks: dict[str, np.ndarray] = {}
        lines = np.zeros(plain.shape[:2], dtype=bool)
        for i, name in enumerate(PANEL_ORDER):
            col, row = (int(np.floor(a + 1e-6)) for a in cross[name])
            assert 0 <= col < size and 0 <= row < size
            vertical, horizontal = np.zeros_like(lines), np.zeros_like(lines)
            vertical[:, i * size + col] = True
            horizontal[row, i * size : (i + 1) * size] = True
            masks[f"{name} vertical line"], masks[f"{name} horizontal line"] = vertical, horizontal
            lines |= vertical | horizontal
        masks["letters"] = (strip != plain).any(axis=2) & ~lines
        assert masks["letters"].sum() > 150  # twelve letters with their outlines
        for key, mask in masks.items():
            with_it = float(np.abs((image[mask].astype(int) - strip[mask]) @ luma).mean())
            without = float(np.abs((image[mask].astype(int) - plain[mask]) @ luma).mean())
            assert with_it < 0.25 * without, (label, key, with_it, without)


def test_source_directory_is_untouched(default_render: Rendered, hires_render: Rendered) -> None:
    for rendered in (default_render, hires_render):
        assert rendered.source_before == rendered.source_after
        assert rendered.source_after == source_state(rendered.subject.path)
        assert len(rendered.source_after) == 8  # mri/, surf/ and their six files


# ---------------------------------------------------------------- determinism, idempotence


def test_two_renders_are_byte_identical(default_render: Rendered, tmp_path: Path) -> None:
    again = _render(default_render.subject, tmp_path / "elsewhere" / "b")
    assert again.info == default_render.info
    first, second = tree(default_render.out), tree(again.out)
    assert set(first) == set(second)
    assert [name for name in first if first[name] != second[name]] == []


def test_second_call_writes_nothing_and_force_rewrites(
    coarse_subject: SyntheticSubject, small_render: tuple[Path, RenderInfo], small_copy: Path
) -> None:
    files = resolve_subject_files(coarse_subject.path)
    before = tree(small_copy)
    age(small_copy)
    info = render_subject(files, small_copy, SMALL)
    assert info == small_render[1]
    assert all(aged(small_copy).values())
    assert tree(small_copy) == before

    forced = render_subject(files, small_copy, SMALL, force=True)
    assert forced == info
    assert not any(aged(small_copy).values())
    assert tree(small_copy) == before


@pytest.mark.parametrize(
    "damage", ["tile", "contour tile", "poster", "video", "vertex map", "no json", "bad json"]
)
def test_an_incomplete_directory_is_rendered_again(
    coarse_subject: SyntheticSubject, small_copy: Path, damage: str
) -> None:
    before = tree(small_copy)
    victim = {
        "tile": "tiles/axial_012.webp",
        "contour tile": "tiles/sagittal_000_c.webp",
        "poster": "poster.webp",
        "video": "anim.mp4",
        "vertex map": "vertex_map.npz",
        "no json": "render.json",
    }.get(damage)
    if victim is None:
        (small_copy / "render.json").write_text('{"schema_version": 1}')
    else:
        (small_copy / victim).unlink()
    age(small_copy)
    render_subject(resolve_subject_files(coarse_subject.path), small_copy, SMALL)
    assert not any(aged(small_copy).values())
    assert tree(small_copy) == before


@pytest.mark.parametrize(
    "old", ["as version 1 wrote it", "version 1 that loads", "version alone", "hash alone"]
)
def test_a_render_of_schema_version_1_is_rendered_again(
    coarse_subject: SyntheticSubject, small_copy: Path, old: str
) -> None:
    """Schema 1 selected the vertex map by the view box and had no ``visible`` arrays. Its
    ``vertex_map_margin_mm`` meant another thing under the same name, so a render of that
    version must not pass for a complete render of the same config.

    A ``render.json`` as version 1 wrote it lacks two required fields and does not load, so it
    is rendered again whatever its version says. The other cases load under the present
    model: the version number and the config hash of version 1 together, and each of the two
    alone, since each is a reason of its own to render again."""
    before = tree(small_copy)
    doc = json.loads(before["render.json"])
    assert doc["schema_version"] == 2
    if old != "hash alone":
        doc["schema_version"] = 1
    if old != "version alone":  # the hash that version 1 gave the very same config
        payload = json.dumps(
            {"schema": 1, "config": doc["config"]}, sort_keys=True, separators=(",", ":")
        )
        doc["config_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:12]
        assert doc["config_hash"] != SMALL.config_hash()
    if old == "as version 1 wrote it":
        del doc["vertex_map"]["n_lh_visible"], doc["vertex_map"]["n_rh_visible"]
        vmap = read_vertex_map(small_copy / "vertex_map.npz")
        with (small_copy / "vertex_map.npz").open("wb") as stream:
            np.savez(stream, **{key: a for key, a in vmap.items() if not key.endswith("visible")})
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    (small_copy / "render.json").write_text(text)
    # the file is a render.json of the present model in every case but the first
    if old == "as version 1 wrote it":
        with pytest.raises(ValidationError):
            RenderInfo.model_validate_json(text)
    else:
        assert (
            RenderInfo.model_validate_json(text).inputs
            == RenderInfo.model_validate_json(before["render.json"]).inputs
        )
    age(small_copy)
    info = render_subject(resolve_subject_files(coarse_subject.path), small_copy, SMALL)
    assert info.schema_version == 2 and info.config_hash == SMALL.config_hash()
    assert not any(aged(small_copy).values())
    assert tree(small_copy) == before


def test_a_changed_config_renders_again_and_removes_stale_files(
    coarse_subject: SyntheticSubject, small_copy: Path
) -> None:
    files = resolve_subject_files(coarse_subject.path)
    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    assert set(tree(small_copy)) == expected_files(counts)
    # leftovers of an interrupted writer, and a file that is not ours
    (small_copy / ".render.json.0123456789abcdef.tmp").write_text("partial")
    (small_copy / "tiles" / ".axial_000.webp.0123456789abcdef.tmp").write_text("partial")
    (small_copy / "notes.txt").write_text("kept")

    info = render_subject(files, small_copy, NARROW, video=False)
    assert info.config == NARROW and info.config_hash != SMALL.config_hash()
    assert info.video is None
    counts["sagittal"] = 7
    assert set(tree(small_copy)) == expected_files(counts, video=False) | {"notes.txt"}
    assert RenderInfo.model_validate_json((small_copy / "render.json").read_text()) == info

    # asking for the video is a different render, too
    age(small_copy)
    with_video = render_subject(files, small_copy, NARROW)
    assert with_video.video is not None and with_video.video.n_frames == 24
    assert set(tree(small_copy)) == expected_files(counts) | {"notes.txt"}
    assert not any(aged(small_copy)[name] for name in expected_files(counts))


@pytest.mark.parametrize("change", ["force", "config", "no video"])
def test_rendering_again_removes_only_what_a_render_writes(
    coarse_subject: SyntheticSubject, small_copy: Path, change: str
) -> None:
    others = [
        ".user.tmp",
        ".notes.0123456789abcdef.tmp",  # the temporary name of a file that is not a render's
        ".poster.webp.tmp",
        "tiles/.x.tmp",
        "tiles/.mine.webp.0123456789abcdef.tmp",
        "tiles/mine.webp",
        "tiles/sagittal_13.webp",  # tiles have at least three digits
        "tiles/axial_0001.webp",  # and a fourth digit only from tile 1000 on
        "tiles/axial_000_d.webp",
        "tiles/axial_\u0660\u0660\u0660.webp",  # Arabic-Indic digits: not what "{:03d}" writes
        "tiles/sagittal_\uff10\uff10\uff11_c.webp",  # full-width digits
        "notes.txt",
        "anim.mp4.txt",
    ]
    # directories under the names of a stale tile and of an abandoned temporary file
    folders = ["tiles/coronal_998.webp", "tiles/.axial_001.webp.0123456789abcdef.tmp"]
    # what a killed render leaves behind, and tiles of an earlier, wider box
    abandoned = [
        ".poster.webp.0123456789abcdef.tmp",
        ".anim.mp4.0123456789abcdef.tmp",
        ".vertex_map.npz.0123456789abcdef.tmp",
        ".render.json.0123456789abcdef.tmp",
        "tiles/.axial_000.webp.0123456789abcdef.tmp",
        "tiles/.coronal_012_c.webp.fedcba9876543210.tmp",
        "tiles/sagittal_013.webp",
        "tiles/sagittal_013_c.webp",
        "tiles/coronal_1000.webp",
    ]
    for name in others + abandoned:
        (small_copy / name).write_text("not a file of this render")
    for name in folders:
        (small_copy / name).mkdir()
        (small_copy / name / "kept.txt").write_text("not a file of this render")
        others.append(name + "/kept.txt")

    files = resolve_subject_files(coarse_subject.path)
    if change == "force":
        render_subject(files, small_copy, SMALL, force=True)
    elif change == "config":
        render_subject(files, small_copy, SMALL.model_copy(update={"pial_colour": "#00FF00"}))
    else:
        render_subject(files, small_copy, SMALL, video=False)

    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    assert set(tree(small_copy)) == expected_files(counts, video=change != "no video") | set(others)
    assert {(small_copy / name).read_text() for name in others} == {"not a file of this render"}


@pytest.mark.parametrize("earlier", ["none", "interrupted", "not json", "without video"])
def test_a_video_that_no_render_json_lists_is_left_alone(
    coarse_subject: SyntheticSubject, small_copy: Path, earlier: str
) -> None:
    """``anim.mp4`` is a name a render writes, and a name a user may give a file of their own
    in a directory they render into. A render without video removes it only if the
    ``render.json`` in the directory says that a render wrote it."""
    files = resolve_subject_files(coarse_subject.path)
    video = small_copy / "anim.mp4"
    if earlier == "none":  # a directory that was never rendered into
        shutil.rmtree(small_copy / "tiles")
        for name in ("render.json", "poster.webp", "vertex_map.npz"):
            (small_copy / name).unlink()
    elif earlier == "interrupted":  # a render that died: no render.json says whose video it is
        (small_copy / "render.json").unlink()
    elif earlier == "not json":
        (small_copy / "render.json").write_text("my notes on rendering")
    else:
        render_subject(files, small_copy, SMALL, video=False)
        assert not video.exists()
    video.write_text("holiday film")

    info = render_subject(files, small_copy, SMALL, video=False, force=True)
    assert info.video is None and video.read_text() == "holiday film"
    # and again, now next to the render.json of a render without video
    render_subject(files, small_copy, SMALL.model_copy(update={"fps": 6}), video=False)
    assert video.read_text() == "holiday film"
    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    assert set(tree(small_copy)) == expected_files(counts)  # the render's files and the film


@pytest.mark.parametrize("earlier", ["this schema", "schema 1", "listed and absent"])
def test_the_video_of_an_earlier_render_is_removed_by_a_render_without_video(
    coarse_subject: SyntheticSubject, small_copy: Path, earlier: str
) -> None:
    files = resolve_subject_files(coarse_subject.path)
    doc = json.loads((small_copy / "render.json").read_text())
    assert doc["video"]["file"] == "anim.mp4" and (small_copy / "anim.mp4").is_file()
    if earlier == "schema 1":  # does not load as a RenderInfo, and lists its video all the same
        doc["schema_version"] = 1
        del doc["vertex_map"]["n_lh_visible"], doc["vertex_map"]["n_rh_visible"]
        (small_copy / "render.json").write_text(json.dumps(doc))
    elif earlier == "listed and absent":
        (small_copy / "anim.mp4").unlink()
    info = render_subject(files, small_copy, SMALL, video=False)
    assert info.video is None
    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    assert set(tree(small_copy)) == expected_files(counts, video=False)


def test_the_video_goes_before_the_render_json_that_lists_it(
    coarse_subject: SyntheticSubject, small_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render without video that dies at its first tile has removed the earlier video
    together with ``render.json``. Were the video left for the end of the render, it would
    stay behind with no ``render.json`` to say whose it is, and no later render would remove
    it."""
    files = resolve_subject_files(coarse_subject.path)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(api.encode, "write_gray_webp", fail)
        with pytest.raises(OSError, match="disk full"):
            render_subject(files, small_copy, SMALL, video=False)
    assert not (small_copy / "render.json").exists() and not (small_copy / "anim.mp4").exists()
    render_subject(files, small_copy, SMALL, video=False)
    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    assert set(tree(small_copy)) == expected_files(counts, video=False)


def test_a_directory_under_the_name_of_the_video_is_left_alone(
    coarse_subject: SyntheticSubject, small_copy: Path
) -> None:
    # render.json lists a video, and what is there under its name is no file of a render
    (small_copy / "anim.mp4").unlink()
    (small_copy / "anim.mp4").mkdir()
    (small_copy / "anim.mp4" / "kept.txt").write_text("kept")
    render_subject(resolve_subject_files(coarse_subject.path), small_copy, SMALL, video=False)
    assert (small_copy / "anim.mp4" / "kept.txt").read_text() == "kept"
    assert (small_copy / "render.json").is_file()


# ---------------------------------------------------------------- the source is only read


def _into_the_subject(subject: Path, tmp_path: Path, where: str) -> Path:
    """An output location that is the subject directory or lies inside it."""
    if where == "by a link":
        (tmp_path / "link").symlink_to(subject, target_is_directory=True)
        return tmp_path / "link" / "media"
    if where == "by a relative path":
        return subject / "surf" / ".." / ".." / subject.name / "media"
    return {
        "itself": subject,
        "inside": subject / "media",
        "deep inside": subject / "qc" / "render" / "media",
        "mri": subject / "mri",
        "surf": subject / "surf",
        "a video inside": subject / "clip.mp4",
        "a video in mri": subject / "mri" / "clip.mp4",
    }[where]


INSIDE = ["itself", "inside", "deep inside", "mri", "surf", "by a link", "by a relative path"]


@pytest.mark.parametrize(
    ("call", "where"),
    [("three_plane", where) for where in [*INSIDE, "a video inside", "a video in mri"]]
    + [("render_subject", where) for where in INSIDE],  # which writes a directory only
)
def test_an_output_inside_the_subject_directory_is_refused(
    subject_copy: Path, tmp_path: Path, where: str, call: str
) -> None:
    """Local sources are read in place and never modified (hard rule in CLAUDE.md)."""
    (subject_copy / "anim.mp4").write_text("my video")
    (subject_copy / ".notes.tmp").write_text("mine")
    out = _into_the_subject(subject_copy, tmp_path, where)
    before = source_state(subject_copy)
    with pytest.raises(api.OutputInSourceError, match="only read") as raised:
        if call == "three_plane":
            three_plane(subject_copy, out, SMALL, video=out.suffix == ".mp4")
        else:
            render_subject(resolve_subject_files(subject_copy), out, SMALL, video=False)
    assert isinstance(raised.value, ValueError)
    assert source_state(subject_copy) == before


def test_the_subject_may_be_given_by_a_link_or_a_relative_path(
    subject_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal compares directories, not spellings: the subject as a link, the output as
    a path into the directory the link leads to, and both relative to the working directory."""
    (tmp_path / "link").symlink_to(subject_copy, target_is_directory=True)
    before = source_state(subject_copy)
    with pytest.raises(api.OutputInSourceError):
        three_plane(tmp_path / "link", subject_copy / "media", SMALL, video=False)
    monkeypatch.chdir(subject_copy)
    for out in (".", "media", "mri/../media"):
        with pytest.raises(api.OutputInSourceError):
            three_plane(".", out, SMALL, video=False)
    monkeypatch.chdir(subject_copy / "surf")
    with pytest.raises(api.OutputInSourceError):
        three_plane("..", ".", SMALL, video=False)
    assert source_state(subject_copy) == before


def test_an_output_next_to_or_around_the_subject_is_accepted(
    subject_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the subject directory is a source. The directory that holds it is not, also when
    its name starts like the subject's, and neither is a directory that an input links from."""
    before = source_state(subject_copy)
    parent = subject_copy.parent
    counts = {"sagittal": 13, "coronal": 13, "axial": 13}
    for out in (parent / (subject_copy.name + "-media"), parent / "sub"):
        three_plane(subject_copy, out, SMALL, video=False)
        assert set(tree(out)) == expected_files(counts, video=False)
    monkeypatch.chdir(parent)
    three_plane(subject_copy.name, ".", SMALL, video=False)  # around the subject
    assert (parent / "render.json").is_file() and (parent / "tiles").is_dir()
    assert source_state(subject_copy) == before


def test_an_output_where_a_linked_input_is_stored_is_refused(
    subject_copy: Path, tmp_path: Path
) -> None:
    """Inputs handed over by role need not sit in a subject directory (a download cache). The
    directory that holds a file is then the source, and so is the one a link leads to."""
    store = tmp_path / "store"
    store.mkdir()
    shutil.move(subject_copy / "mri" / "T1.mgz", store / "0a1b2c")
    (subject_copy / "mri" / "T1.mgz").symlink_to(store / "0a1b2c")
    flat = tmp_path / "flat"
    flat.mkdir()
    files = resolve_subject_files(subject_copy)
    for role in SURFACES:
        files[FileRole(role)] = Path(shutil.copy(files[FileRole(role)], flat / role))
    for out in (store, store / "media", flat, flat / "media" / "a"):
        with pytest.raises(api.OutputInSourceError):
            render_subject(files, out, SMALL, video=False)
    assert sorted(p.name for p in store.iterdir()) == ["0a1b2c"]
    assert sorted(p.name for p in flat.iterdir()) == sorted(SURFACES)
    info = render_subject(files, tmp_path / "media", SMALL, video=False)
    assert info.inputs["t1"].name == "T1.mgz"


def test_changed_inputs_render_again(subject_copy: Path, tmp_path: Path) -> None:
    out = tmp_path / "media"
    first = render_subject(resolve_subject_files(subject_copy), out, SMALL, video=False)
    assert "brainmask" in first.inputs

    # same size, other content: the white surface moved by 1 mm
    path = subject_copy / "surf" / "lh.white"
    vertices, faces, meta, stamp = fs.read_geometry(str(path), read_metadata=True, read_stamp=True)
    fs.write_geometry(str(path), vertices + np.array([0.0, 0.0, 1.0]), faces, stamp, meta)
    age(out)
    second = render_subject(resolve_subject_files(subject_copy), out, SMALL, video=False)
    assert second.inputs["lh_white"].size == first.inputs["lh_white"].size
    assert second.inputs["lh_white"].sha256 != first.inputs["lh_white"].sha256
    assert second.inputs["lh_pial"] == first.inputs["lh_pial"]
    assert not any(aged(out).values())

    # without the brainmask the window comes from the non-zero voxels of the T1
    (subject_copy / "mri" / "brainmask.mgz").unlink()
    third = render_subject(resolve_subject_files(subject_copy), out, SMALL, video=False)
    assert "brainmask" not in third.inputs and set(third.inputs) == set(first.inputs) - {
        "brainmask"
    }
    t1 = read_volume(subject_copy / "mri" / "T1.mgz")
    assert third.window == intensity_window(t1, None) == (float(T1_HEAD), 110.0)


def test_an_interrupted_render_is_not_complete(
    coarse_subject: SyntheticSubject, small_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render.json goes first and comes back last: a render that dies on its last file before
    it leaves a directory that the next call renders again."""
    files = resolve_subject_files(coarse_subject.path)
    before = tree(small_copy)

    def fail(*args: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(api.vertexmap, "write_vertex_map", fail)
        with pytest.raises(OSError, match="disk full"):
            render_subject(files, small_copy, SMALL, force=True)
    assert not (small_copy / "render.json").exists()
    assert set(before) - set(tree(small_copy)) == {"render.json"}
    age(small_copy)
    render_subject(files, small_copy, SMALL)
    assert tree(small_copy) == before and not any(aged(small_copy).values())


def lock_is_free(out: Path) -> bool:
    """Whether no render holds ``out``: an exclusive ``flock`` on it succeeds at once."""
    import fcntl

    descriptor = os.open(out, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)
    return True


@pytest.mark.parametrize("second", ["same config", "other config"])
def test_renders_into_one_directory_run_one_after_the_other(
    coarse_subject: SyntheticSubject,
    small_render: tuple[Path, RenderInfo],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second: str,
) -> None:
    """Render A is held just before it writes its poster, and render B starts in the same
    directory meanwhile. B has to wait for A. Then it finds A's render complete and writes
    nothing, or, with another config, replaces all of it: either way the directory ends up
    as the render that ``render.json`` describes would have left it on its own."""
    pytest.importorskip("fcntl")
    files = resolve_subject_files(coarse_subject.path)
    config_b, reference = SMALL, tree(small_render[0])
    if second == "other config":
        config_b = SMALL.model_copy(update={"pial_colour": "#00FF00"})
        render_subject(files, tmp_path / "alone", config_b)
        reference = tree(tmp_path / "alone")
        assert reference != tree(small_render[0])

    out = tmp_path / "media"
    a_at_poster, release_a = threading.Event(), threading.Event()
    write_poster = encode.write_rgb_webp
    posters: list[str] = []

    def held_poster(path: Path, image: np.ndarray, quality: int) -> None:
        name = threading.current_thread().name
        posters.append(name)
        if name == "A":
            a_at_poster.set()
            assert release_a.wait(60)
        write_poster(path, image, quality)

    monkeypatch.setattr(encode, "write_rgb_webp", held_poster)
    results: dict[str, RenderInfo | BaseException] = {}

    def run(config: RenderConfig) -> None:
        try:
            results[threading.current_thread().name] = render_subject(files, out, config)
        except BaseException as exc:  # shown by the assertions below
            results[threading.current_thread().name] = exc

    a = threading.Thread(target=run, args=(SMALL,), name="A")
    b = threading.Thread(target=run, args=(config_b,), name="B")
    a.start()
    assert a_at_poster.wait(60)
    held = not lock_is_free(out)
    b.start()
    b.join(1.5)  # a whole render takes about 0.3 s: without a lock, B is through by now
    b_waited = b.is_alive()
    release_a.set()
    a.join(60)
    b.join(60)
    assert not a.is_alive() and not b.is_alive()

    assert held and b_waited
    assert isinstance(results["A"], RenderInfo) and isinstance(results["B"], RenderInfo)
    assert results["A"].config == SMALL and results["B"].config == config_b
    assert posters == (["A"] if second == "same config" else ["A", "B"])
    assert RenderInfo.model_validate_json((out / "render.json").read_bytes()) == results["B"]
    assert tree(out) == reference
    assert lock_is_free(out)


def test_the_directory_is_released_when_a_render_fails(
    coarse_subject: SyntheticSubject, small_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fcntl")
    files = resolve_subject_files(coarse_subject.path)
    seen: list[bool] = []

    def fail(*args: object) -> None:
        seen.append(lock_is_free(small_copy))
        raise OSError("disk full")

    monkeypatch.setattr(api.vertexmap, "write_vertex_map", fail)
    with pytest.raises(OSError, match="disk full"):
        render_subject(files, small_copy, SMALL, force=True)
    assert seen == [False]  # held while writing
    assert lock_is_free(small_copy)
    with pytest.raises(RenderError):  # a refusal comes before the lock
        render_subject({}, small_copy, SMALL)
    assert lock_is_free(small_copy)


@pytest.mark.parametrize("why", ["no fcntl", "lock refused"])
def test_without_a_lock_the_render_goes_ahead(
    coarse_subject: SyntheticSubject,
    small_render: tuple[Path, RenderInfo],
    small_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
    why: str,
) -> None:
    """Windows has no ``fcntl``, and a file system may refuse ``flock``. The render is the
    same; only temporary files stay, since one may belong to a render that is still running."""

    class Refusing:
        LOCK_EX = 2

        @staticmethod
        def flock(descriptor: int, operation: int) -> None:
            raise OSError(37, "No locks available")

    monkeypatch.setattr(api, "fcntl", None if why == "no fcntl" else Refusing)
    temporary = ".poster.webp.0123456789abcdef.tmp"
    (small_copy / temporary).write_text("of another render")
    (small_copy / "tiles" / "sagittal_013.webp").write_text("stale")
    age(small_copy)
    info = render_subject(resolve_subject_files(coarse_subject.path), small_copy, SMALL, force=True)
    assert info == small_render[1]
    assert tree(small_copy) == {**tree(small_render[0]), temporary: b"of another render"}
    assert not any(stays for name, stays in aged(small_copy).items() if name != temporary)


def test_config_reaches_the_vertex_map_and_the_window(subject_copy: Path, tmp_path: Path) -> None:
    surfaces = {
        name: read_surface(subject_copy / "surf" / name.replace("_", ".")) for name in SURFACES
    }
    # a brainmask of white matter only: undilated it selects a single intensity
    t1 = nib.load(subject_copy / "mri" / "T1.mgz")
    mask = (np.asanyarray(t1.dataobj) == T1_WHITE).astype(np.uint8)
    nib.save(nib.MGHImage(mask, t1.affine), str(subject_copy / "mri" / "brainmask.mgz"))
    files = resolve_subject_files(subject_copy)
    # (vertex map margin, dilation, percentiles) -> window: its low end is the darkest tissue
    # that the dilated mask reaches, or the median, which lies in the white matter
    cases = {
        (5.0, 3, (1.0, 99.5)): (T1_HEAD, T1_WHITE),
        (0.0, 0, (1.0, 99.5)): (T1_WHITE, T1_WHITE + 1),
        (3.5, 1, (1.0, 99.5)): (T1_GRAY, T1_WHITE),
        (3.5, 3, (50.0, 99.5)): (T1_WHITE, T1_WHITE + 1),
    }
    sizes = set()
    for (margin, dilation, percentiles), window in cases.items():
        config = SMALL.model_copy(
            update={
                "vertex_map_margin_mm": margin,
                "brainmask_dilation": dilation,
                "window_percentiles": percentiles,
            }
        )
        info = render_subject(files, tmp_path / "media", config, video=False)
        assert info.window == window
        # the margin is counted from what the tiles show, not from the box
        region = tile_region(json.loads((tmp_path / "media" / "render.json").read_text()))
        vmap = read_vertex_map(tmp_path / "media" / "vertex_map.npz")
        for hemi in ("lh", "rh"):
            near = np.zeros(len(surfaces[f"{hemi}_pial"].vertices), dtype=bool)
            for name in ("pial", "white"):
                near |= in_boxes(surfaces[f"{hemi}_{name}"].vertices, region, margin)
            assert vmap[f"{hemi}_index"].tolist() == np.flatnonzero(near).tolist()
        assert 0 < len(vmap["lh_index"]) < len(surfaces["lh_pial"].vertices)
        sizes.add(len(vmap["lh_index"]))
    assert len(sizes) == 3  # one per margin


def test_config_reaches_the_encoders_and_the_pen(
    coarse_subject: SyntheticSubject, small_render: tuple[Path, RenderInfo], tmp_path: Path
) -> None:
    config = SMALL.model_copy(
        update={"webp_quality": 30, "fps": 24, "crf": 40, "line_width_px": 3.0}
    )
    info = three_plane(coarse_subject.path, tmp_path / "media", config)
    ours, reference = tree(tmp_path / "media"), tree(small_render[0])
    assert set(ours) == set(reference)
    for name in ("tiles/axial_006.webp", "poster.webp", "anim.mp4"):
        assert len(ours[name]) < 0.8 * len(reference[name]), name
    assert info.video is not None and info.video.fps == 24
    assert video_frames(tmp_path / "media" / "anim.mp4")[0]["fps"] == 24
    drawn = [
        (read_image(root / "tiles/axial_006_c.webp")[..., 3] == 255).sum()
        for root in (tmp_path / "media", small_render[0])
    ]
    assert drawn[0] > 2 * drawn[1] > 0


# ---------------------------------------------------------------- registration and variants


@pytest.mark.parametrize("which", ["default_render", "hires_render"])
def test_contours_are_registered_to_the_image(which: str, request: pytest.FixtureRequest) -> None:
    """On the written tiles, the pial line runs along the edge of the tissue: within about
    half a voxel everywhere, and no shift of the line by whole pixels fits better."""
    rendered: Rendered = request.getfixturevalue(which)
    out, info = rendered.out, rendered.info
    lo, hi = info.window
    threshold = 255 * ((T1_GRAY + T1_HEAD) / 2 - lo) / (hi - lo)  # between cortex and head
    voxel_px = rendered.subject.voxel_size * info.config.panel_px / 80.0
    shifts = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    tested = 0
    for name, panel in info.panels.items():
        for pos in range(1, len(panel.slices), 6):
            tissue = read_image(out / info.tiles.gray_path(name, pos))[..., 0] > threshold
            contours = read_image(out / info.tiles.contours_path(name, pos))
            line = (contours[..., 3] >= 200) & (contours[..., 1] >= 180)  # the yellow pial line
            if line.sum() < 200:
                continue
            # distance of every pixel centre to the edge of the tissue
            edge = (
                np.where(
                    tissue,
                    ndimage.distance_transform_edt(tissue),
                    ndimage.distance_transform_edt(~tissue),
                )
                - 0.5
            )
            rows, cols = np.nonzero(line)
            keep = (rows >= 3) & (rows < 317) & (cols >= 3) & (cols < 317)
            rows, cols = rows[keep], cols[keep]
            assert edge[rows, cols].mean() < 0.3 * voxel_px + 0.5, (name, pos)
            assert edge[rows, cols].max() < 0.75 * voxel_px + 1.5, (name, pos)
            score = {s: edge[rows + s[1], cols + s[0]].mean() for s in shifts}
            assert max(np.abs(min(score, key=lambda s: score[s]))) <= 1, (name, pos)
            tested += 1
    assert tested >= 20


def test_pial_variant_is_recorded(synth_subject_pial_t1: SyntheticSubject, tmp_path: Path) -> None:
    before = source_state(synth_subject_pial_t1.path)
    info = three_plane(synth_subject_pial_t1.path, tmp_path / "media", SMALL, video=False)
    assert info.inputs["lh_pial"].name == "lh.pial.T1"
    assert info.inputs["rh_pial"].name == "rh.pial.T1"
    assert info.inputs["lh_white"].name == "lh.white"
    assert read_image(tmp_path / "media" / "tiles/coronal_006_c.webp")[..., 3].any()
    assert source_state(synth_subject_pial_t1.path) == before


def test_links_record_the_variant_but_never_another_directory(tmp_path: Path) -> None:
    subject = write_synthetic_subject(tmp_path, subdivisions=3, pial_variant="pial.T1").path
    for hemi in ("lh", "rh"):  # the FreeSurfer 7 layout: ?h.pial is a link to ?h.pial.T1
        (subject / "surf" / f"{hemi}.pial").symlink_to(f"{hemi}.pial.T1")
    elsewhere = tmp_path / "store" / "sub-secret_T1.mgz"
    elsewhere.parent.mkdir()
    (subject / "mri" / "T1.mgz").rename(elsewhere)
    (subject / "mri" / "T1.mgz").symlink_to(elsewhere)

    files = resolve_subject_files(subject)
    assert files[FileRole.LH_PIAL].name == "lh.pial"
    info = render_subject(files, tmp_path / "media", SMALL, video=False)
    assert info.inputs["lh_pial"].name == "lh.pial.T1"
    assert info.inputs["rh_pial"].name == "rh.pial.T1"
    assert info.inputs["t1"].name == "T1.mgz"
    assert info.inputs["t1"].size == elsewhere.stat().st_size
    assert "secret" not in (tmp_path / "media" / "render.json").read_text()


def test_only_freesurfer_names_are_recorded(subject_copy: Path, tmp_path: Path) -> None:
    """render.json is served to raters: a name given by whoever stored the data stays out of
    it, whether it is the target of a link beside the file or the file handed over itself."""
    for folder, name, foreign in (
        ("mri", "T1.mgz", "ds000999_sub-SECRET01_T1.mgz"),
        ("surf", "lh.white", "ds000999.lh.white"),
        ("surf", "rh.pial", "ds000999.rh.pial"),
    ):
        (subject_copy / folder / name).rename(subject_copy / folder / foreign)
        (subject_copy / folder / name).symlink_to(foreign)
    # the FreeSurfer 7 link is still recorded under the variant it names
    (subject_copy / "surf" / "lh.pial").rename(subject_copy / "surf" / "lh.pial.T1")
    (subject_copy / "surf" / "lh.pial").symlink_to("lh.pial.T1")
    expected = {
        "t1": "T1.mgz",
        "brainmask": "brainmask.mgz",
        "lh_white": "lh.white",
        "rh_white": "rh.white",
        "lh_pial": "lh.pial.T1",
        "rh_pial": "rh.pial",
    }
    info = three_plane(subject_copy, tmp_path / "media", SMALL, video=False)
    assert {role: entry.name for role, entry in info.inputs.items()} == expected
    text = (tmp_path / "media" / "render.json").read_text()
    assert "ds000999" not in text and "SECRET01" not in text

    # files handed over under names of a store, as a download cache may do
    files = resolve_subject_files(subject_copy)
    files[FileRole.T1] = subject_copy / "mri" / "ds000999_sub-SECRET01_T1.mgz"
    files[FileRole.RH_PIAL] = subject_copy / "surf" / "ds000999.rh.pial"
    files[FileRole.LH_PIAL] = subject_copy / "surf" / "lh.pial.T1"
    direct = render_subject(files, tmp_path / "direct", SMALL, video=False)
    assert direct.inputs == info.inputs
    text = (tmp_path / "direct" / "render.json").read_text()
    assert "ds000999" not in text and "SECRET01" not in text


# ---------------------------------------------------------------- failures


def protocol_reasons() -> list[str]:
    text = package_files("surfcensus").joinpath("data/protocol.yaml").read_text()
    return yaml.safe_load(text)["unratable_reasons"]


def _oblique_t1(subject: Path, angle: float = np.deg2rad(5.0)) -> None:
    """Turn the scanner affine of the T1 by ``angle`` (radians) about the scanner z axis."""
    path = subject / "mri" / "T1.mgz"
    image = nib.load(path)
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    nib.save(nib.MGHImage(np.asanyarray(image.dataobj), turn @ image.affine), str(path))


def _ras_t1(subject: Path) -> None:
    """Axis-aligned, but with voxel axes along R, A, S instead of FreeSurfer's L, I, A."""
    path = subject / "mri" / "T1.mgz"
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = -80.0
    nib.save(nib.MGHImage(np.asanyarray(nib.load(path).dataobj), affine), str(path))


def _rewrite(subject: Path, name: str, change: str) -> None:
    path = subject / "surf" / name
    vertices, faces = fs.read_geometry(str(path))
    if change == "no faces":
        faces = faces[:0]
    elif change == "non-finite":
        vertices[7, 1] = np.nan
    elif change == "face index":
        faces[3, 2] = len(vertices)
    elif change == "fewer vertices":
        vertices, faces = vertices[:-1], faces[(faces < len(vertices) - 1).all(axis=1)]
    fs.write_geometry(str(path), vertices, faces)


def _other_grid_brainmask(subject: Path) -> None:
    path = subject / "mri" / "brainmask.mgz"
    nib.save(nib.MGHImage(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), str(path))


DAMAGE = {
    "missing white": (lambda s: (s / "surf" / "rh.white").unlink(), "missing surface", "rh_white"),
    "missing t1": (lambda s: (s / "mri" / "T1.mgz").unlink(), "missing surface", "t1"),
    "oblique": (_oblique_t1, "wrong orientation", "conformed orientation"),
    "not LIA": (_ras_t1, "wrong orientation", "conformed orientation"),
    "no faces": (lambda s: _rewrite(s, "lh.pial", "no faces"), "render failure", "lh_pial"),
    "non-finite": (lambda s: _rewrite(s, "rh.white", "non-finite"), "render failure", "rh_white"),
    "face index": (lambda s: _rewrite(s, "rh.pial", "face index"), "render failure", "rh_pial"),
    "fewer vertices": (
        lambda s: _rewrite(s, "lh.white", "fewer vertices"),
        "render failure",
        "lh_white",
    ),
    "truncated surface": (
        lambda s: (s / "surf" / "lh.white").write_bytes(b"\xff\xff\xfe"),
        "render failure",
        r"unreadable input \(lh\.white: ",
    ),
    "not an mgz": (
        lambda s: (s / "mri" / "T1.mgz").write_bytes(b"not a volume"),
        "render failure",
        r"unreadable input \(T1\.mgz: ",
    ),
    # nibabel names an empty file by its whole path, which the message must not repeat
    "empty t1": (
        lambda s: (s / "mri" / "T1.mgz").write_bytes(b""),
        "render failure",
        r"unreadable input \(T1\.mgz: Empty file: 'T1\.mgz'\)",
    ),
    "empty brainmask": (
        lambda s: (s / "mri" / "brainmask.mgz").write_bytes(b""),
        "render failure",
        r"unreadable input \(brainmask\.mgz: Empty file: 'brainmask\.mgz'\)",
    ),
    "brainmask on another grid": (_other_grid_brainmask, "render failure", "brainmask"),
}


@pytest.mark.parametrize("damage", list(DAMAGE))
def test_unrenderable_subjects_raise_with_a_protocol_reason(
    subject_copy: Path, tmp_path: Path, damage: str
) -> None:
    change, reason, hint = DAMAGE[damage]
    change(subject_copy)
    out = tmp_path / "media"
    with pytest.raises(RenderError, match=hint) as raised:
        render_subject(resolve_subject_files(subject_copy), out, SMALL)
    assert raised.value.reason == reason
    assert str(raised.value).startswith(reason + ": ")
    assert reason in protocol_reasons()
    # the message may be logged and shown: no directory in it, which could name the dataset
    assert str(tmp_path) not in str(raised.value) and subject_copy.name not in str(raised.value)
    assert not out.exists()


@pytest.mark.parametrize(("angle", "refused"), [(5e-5, False), (2e-4, True)])
def test_orientation_tolerance(
    subject_copy: Path,
    small_render: tuple[Path, RenderInfo],
    tmp_path: Path,
    angle: float,
    refused: bool,
) -> None:
    """docs/rendering.md, "Axis alignment": the linear parts of vox2ras and vox2ras-tkr may
    differ by 1e-4, relative. For a scanner affine turned by a small angle that difference is
    the sine of the angle."""
    _oblique_t1(subject_copy, angle)
    out = tmp_path / "media"
    if refused:
        with pytest.raises(RenderError, match="conformed orientation") as raised:
            three_plane(subject_copy, out, SMALL)
        assert raised.value.reason == "wrong orientation"
        assert not out.exists()
    else:
        info = three_plane(subject_copy, out, SMALL)
        # image and contours are both placed through vox2ras-tkr, which has not changed: the
        # same files as for the subject as it was, but for the T1 header that render.json holds
        assert info.volume.vox2ras != small_render[1].volume.vox2ras
        assert info.inputs["t1"].sha256 != small_render[1].inputs["t1"].sha256
        written, plain = tree(out), tree(small_render[0])
        assert set(written) == set(plain)
        assert {name for name in plain if written[name] != plain[name]} == {"render.json"}


def test_a_tkr_affine_off_the_voxel_axes_is_a_wrong_orientation(
    coarse_subject: SyntheticSubject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nibabel never returns such a vox2ras-tkr, so the reader is replaced."""

    def tilted(path: Path) -> Volume:
        volume = read_volume(path)
        turn = np.eye(4)
        turn[:2, :2] = [[np.cos(0.01), -np.sin(0.01)], [np.sin(0.01), np.cos(0.01)]]
        return dataclasses.replace(
            volume, vox2ras_tkr=turn @ volume.vox2ras_tkr, vox2ras=turn @ volume.vox2ras
        )

    monkeypatch.setattr(api, "read_volume", tilted)
    with pytest.raises(RenderError, match="not parallel") as raised:
        three_plane(coarse_subject.path, tmp_path / "media", SMALL)
    assert raised.value.reason == "wrong orientation"
    assert not (tmp_path / "media").exists()


def test_missing_roles_are_all_named(tmp_path: Path, subject_copy: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(RenderError) as raised:
        three_plane(tmp_path / "empty", tmp_path / "media")
    assert raised.value.reason == "missing surface"
    assert str(raised.value).endswith("t1, lh_white, rh_white, lh_pial, rh_pial")
    # a role whose file has gone counts as missing
    files = resolve_subject_files(subject_copy)
    files[FileRole.LH_PIAL].unlink()
    with pytest.raises(RenderError, match=r"no file for lh_pial$"):
        render_subject(files, tmp_path / "media")
    assert not (tmp_path / "media").exists()


def test_a_box_outside_the_volume_is_a_render_failure(
    coarse_subject: SyntheticSubject, tmp_path: Path
) -> None:
    config = SMALL.model_copy(update={"box_z_mm": (300.0, 310.0)})
    with pytest.raises(RenderError, match="outside the volume") as raised:
        three_plane(coarse_subject.path, tmp_path / "media", config)
    assert raised.value.reason == "render failure"
    assert not (tmp_path / "media").exists()


def test_a_refused_subject_leaves_an_earlier_render_as_it_was(
    subject_copy: Path, small_copy: Path
) -> None:
    before = tree(small_copy)
    age(small_copy)
    _rewrite(subject_copy, "lh.pial", "non-finite")
    with pytest.raises(RenderError):
        render_subject(resolve_subject_files(subject_copy), small_copy, SMALL)
    assert tree(small_copy) == before and all(aged(small_copy).values())


@pytest.mark.parametrize(
    ("target", "mode"),
    [
        ("mri/brainmask.mgz", 0),
        ("mri/T1.mgz", 0),
        ("surf/lh.pial", 0),
        ("surf", 0),
        ("mri", 0o444),
        (".", 0),
    ],
)
def test_an_input_without_permission_is_a_render_failure(
    subject_copy: Path, tmp_path: Path, target: str, mode: int
) -> None:
    if not file_modes_apply(tmp_path):
        pytest.skip("file modes do not restrict this user")
    (subject_copy / target).chmod(mode)
    try:
        with pytest.raises(RenderError, match="unreadable input") as raised:
            three_plane(subject_copy, tmp_path / "media", SMALL, video=False)
    finally:
        (subject_copy / target).chmod(0o755)
    assert raised.value.reason == "render failure"
    assert "Permission denied" in str(raised.value)
    assert str(tmp_path) not in str(raised.value) and "sub-copy" not in str(raised.value)
    assert not (tmp_path / "media").exists()


@pytest.mark.parametrize("call", ["open", "stat"])
@pytest.mark.parametrize("name", ["brainmask.mgz", "T1.mgz", "rh.white"])
def test_an_input_that_cannot_be_read_is_a_render_failure(
    subject_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: str, name: str
) -> None:
    """As above, with the error raised by ``Path.open`` (a file that may not be read) or by
    ``Path.stat`` (a directory that may not be searched), so that it also runs as root."""
    original = getattr(Path, call)

    def refuse(self: Path, *args: object, **kwargs: object) -> object:
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, *args, **kwargs)

    for through_three_plane in (True, False):
        with monkeypatch.context() as patch, pytest.raises(RenderError) as raised:
            if through_three_plane:
                patch.setattr(Path, call, refuse)
                three_plane(subject_copy, tmp_path / "media", SMALL, video=False)
            else:
                files = resolve_subject_files(subject_copy)
                patch.setattr(Path, call, refuse)
                render_subject(files, tmp_path / "media", SMALL, video=False)
        assert raised.value.reason == "render failure"
        assert str(raised.value) == f"render failure: unreadable input ({name}: Permission denied)"
        assert not (tmp_path / "media").exists()


BAD_CONFIG = [
    {"pial_colour": "yellow"},
    {"pial_colour": ""},
    {"white_colour": "#F00"},
    {"white_colour": "FF3B30"},
    {"crosshair_colour": "cyan"},
    {"crosshair_colour": "#33C7FF\n"},
    {"crosshair_colour": "#33c7ff80"},
    {"line_width_px": 400.0},
    {"line_width_px": 8.5},  # more than panel_px / 8
    {"box_x_mm": (30.0, -30.0)},  # high end first
    {"box_z_mm": (6.0, float("nan"))},
    # no extent along any axis and no margin: panels 0 mm wide
    {"box_x_mm": (0.0, 0.0), "box_y_mm": (5.0, 5.0), "box_z_mm": (0.0, 0.0), "margin_mm": 0.0},
]


@pytest.mark.parametrize("update", BAD_CONFIG)
def test_an_invalid_config_is_rejected_when_it_is_made(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RenderConfig(**{**SMALL.model_dump(), **update})


def test_colours_and_line_widths_that_can_be_drawn_are_accepted() -> None:
    config = RenderConfig(pial_colour="#ffd400", white_colour="#00Ff00", crosshair_colour="#000000")
    # stored in one spelling, so that equal colours give equal config hashes
    assert (config.pial_colour, config.white_colour) == ("#FFD400", "#00FF00")
    lower = RenderConfig(pial_colour="#ffd400", white_colour="#ff3b30", crosshair_colour="#33c7ff")
    assert lower == RenderConfig() and lower.config_hash() == RenderConfig().config_hash()
    assert RenderConfig.model_validate_json(lower.model_dump_json()) == RenderConfig()
    assert RenderConfig(panel_px=16).line_width_px == 1.25  # the default line on the smallest panel
    assert RenderConfig(panel_px=64, line_width_px=8.0).line_width_px == 8.0
    assert RenderConfig(line_width_px=40.0).panel_px == 320
    # a box may be flat along any axis as long as the panels have an extent
    flat = RenderConfig(box_x_mm=(0, 0), box_y_mm=(5, 5), box_z_mm=(0, 0))
    assert (flat.box_y_mm, flat.margin_mm) == ((5, 5), 10)
    assert (
        RenderConfig(box_x_mm=(0, 0), box_y_mm=(5, 5), box_z_mm=(0, 1), margin_mm=0).margin_mm == 0
    )


@pytest.mark.parametrize("update", BAD_CONFIG)
def test_an_invalid_config_leaves_an_earlier_render_as_it_was(
    coarse_subject: SyntheticSubject, small_copy: Path, tmp_path: Path, update: dict[str, object]
) -> None:
    before = tree(small_copy)
    age(small_copy)
    bad = SMALL.model_copy(update=update)  # model_copy does not validate
    for video in (True, False):
        with pytest.raises(ValueError, match=next(iter(update))):
            three_plane(coarse_subject.path, small_copy, bad, video=video)
    assert tree(small_copy) == before and all(aged(small_copy).values())
    for target in ("fresh", "videos/clip.mp4"):
        with pytest.raises(ValueError, match=next(iter(update))):
            three_plane(coarse_subject.path, tmp_path / target, bad)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["media"]


def test_the_spelling_of_a_colour_does_not_render_again(
    coarse_subject: SyntheticSubject, small_copy: Path
) -> None:
    """Lower-case colours draw the same bytes; they must not look like another config, also
    when they come in through ``model_copy``, which does not validate."""
    before = tree(small_copy)
    age(small_copy)
    lower = {name: getattr(SMALL, name).lower() for name in ("pial_colour", "white_colour")}
    assert lower == {"pial_colour": "#ffd400", "white_colour": "#ff3b30"}
    for config in (SMALL.model_copy(update=lower), RenderConfig(**{**SMALL.model_dump(), **lower})):
        info = three_plane(coarse_subject.path, small_copy, config)
        assert info.config == SMALL and info.config_hash == SMALL.config_hash()
    assert tree(small_copy) == before and all(aged(small_copy).values())


@pytest.mark.parametrize("problem", ["absent", "not executable", "none found"])
def test_a_missing_encoder_is_found_before_anything_is_written(
    coarse_subject: SyntheticSubject,
    small_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    """The video is encoded after ``render.json`` is removed and every tile is written; a
    render that cannot encode it must not start."""
    before = tree(small_copy)
    age(small_copy)
    if problem == "none found":

        def none_found() -> str:
            raise RuntimeError("No ffmpeg exe could be found.")

        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", none_found)
    else:
        exe = tmp_path / "ffmpeg-here"
        if problem == "not executable":
            exe.write_bytes(b"data")
        monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
    other = SMALL.model_copy(update={"fps": 10})
    for config, force in ((other, False), (SMALL, True)):
        with pytest.raises(RenderError, match="the video cannot be encoded") as raised:
            three_plane(coarse_subject.path, small_copy, config, force=force)
        assert raised.value.reason == "render failure"
        assert isinstance(raised.value.__cause__, RuntimeError)
    assert tree(small_copy) == before and all(aged(small_copy).values())
    for target in ("fresh", "videos/clip.mp4"):
        with pytest.raises(RenderError, match="the video cannot be encoded"):
            three_plane(coarse_subject.path, tmp_path / target, SMALL)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "ffmpeg-here") == ["media"]
    # a render that is up to date encodes nothing, and one without video needs no encoder
    assert three_plane(coarse_subject.path, small_copy, SMALL).video is not None
    assert all(aged(small_copy).values())
    info = three_plane(coarse_subject.path, tmp_path / "fresh", SMALL, video=False)
    assert info.video is None and (tmp_path / "fresh" / "render.json").is_file()


def test_video_needs_an_even_panel(coarse_subject: SyntheticSubject, tmp_path: Path) -> None:
    odd = SMALL.model_copy(update={"panel_px": 65})
    with pytest.raises(ValueError, match="even"):
        three_plane(coarse_subject.path, tmp_path / "media", odd)
    assert not (tmp_path / "media").exists()
    info = three_plane(coarse_subject.path, tmp_path / "media", odd, video=False)
    assert read_image(tmp_path / "media" / "poster.webp").shape == (65, 195, 3)
    assert info.video is None


# ---------------------------------------------------------------- three_plane, package import


def test_three_plane_with_an_mp4_path_writes_only_the_video(
    coarse_subject: SyntheticSubject, small_render: tuple[Path, RenderInfo], tmp_path: Path
) -> None:
    before = source_state(coarse_subject.path)
    target = tmp_path / "videos" / "clip.mp4"
    info = three_plane(str(coarse_subject.path), str(target), SMALL)
    assert [p.name for p in target.parent.iterdir()] == ["clip.mp4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["videos"]
    assert target.read_bytes() == (small_render[0] / "anim.mp4").read_bytes()
    assert info == small_render[1]
    assert source_state(coarse_subject.path) == before
    with pytest.raises(ValueError, match="video"):
        three_plane(coarse_subject.path, tmp_path / "other" / "clip.mp4", SMALL, video=False)
    assert not (tmp_path / "other").exists()


def test_a_failed_mp4_render_leaves_nothing_behind(subject_copy: Path, tmp_path: Path) -> None:
    (subject_copy / "surf" / "lh.white").unlink()
    (tmp_path / "videos").mkdir()
    with pytest.raises(RenderError):
        three_plane(subject_copy, tmp_path / "videos" / "clip.mp4", SMALL)
    assert list((tmp_path / "videos").iterdir()) == []


def test_a_refused_mp4_render_touches_nothing(subject_copy: Path, tmp_path: Path) -> None:
    """No directory is made on the way to the video, and none is made and removed inside an
    existing one, which would show in its modification time."""
    (subject_copy / "surf" / "lh.white").unlink()
    (tmp_path / "videos").mkdir()
    os.utime(tmp_path / "videos", ns=(OLD_NS, OLD_NS))
    for target in ("deep/er/clip.mp4", "videos/clip.mp4"):
        with pytest.raises(RenderError, match="lh_white"):
            three_plane(subject_copy, tmp_path / target, SMALL)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy", "videos"]
    assert (tmp_path / "videos").stat().st_mtime_ns == OLD_NS


def test_an_mp4_render_that_fails_removes_the_directories_it_made(
    coarse_subject: SyntheticSubject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path, *args: object) -> int:
        if "shared" in path.parts:  # someone else writes to a directory this call made
            (tmp_path / "shared" / "theirs.txt").write_text("kept")
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(encode, "write_mp4", fail)
    (tmp_path / "videos").mkdir()
    for target in ("deep/er/clip.mp4", "videos/clip.mp4", "shared/new/clip.mp4"):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            three_plane(coarse_subject.path, tmp_path / target, SMALL)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [
        "shared",
        "shared/theirs.txt",
        "videos",
    ]


def test_three_plane_with_a_directory_is_render_subject(
    coarse_subject: SyntheticSubject, small_render: tuple[Path, RenderInfo], tmp_path: Path
) -> None:
    info = three_plane(coarse_subject.path, tmp_path / "media", SMALL)
    assert info == small_render[1]
    assert tree(tmp_path / "media") == tree(small_render[0])


def test_importing_surfcensus_does_not_import_matplotlib() -> None:
    code = (
        "import sys, surfcensus as sc\n"
        "from surfcensus.render import RenderConfig, RenderInfo\n"
        "import surfcensus.cli\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported early'\n"
        "sc.render.three_plane, sc.render.render_subject, sc.render.RenderError\n"
        "assert 'matplotlib' in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert surfcensus.render.three_plane is three_plane
    assert set(surfcensus.render.__all__) == {
        "OutputInSourceError",
        "RenderConfig",
        "RenderError",
        "RenderInfo",
        "render_subject",
        "three_plane",
    }
    with pytest.raises(AttributeError):
        _ = surfcensus.render.no_such_name


# ---------------------------------------------------------------- click round trip


@pytest.mark.parametrize("which", ["default_render", "hires_render"])
def test_click_round_trip_through_render_json(which: str, request: pytest.FixtureRequest) -> None:
    """A native vertex is projected onto the nearest tile of each panel, clicked at the centre
    of the pixel that holds it, and resolved again using render.json and vertex_map.npz only:
    the click lands within half a slice spacing of the vertex (half a voxel on the 1 mm
    subject), and the vertex is its nearest neighbour in the vertex map.
    """
    rendered: Rendered = request.getfixturevalue(which)
    doc = json.loads((rendered.out / "render.json").read_text())
    vmap = read_vertex_map(rendered.out / doc["vertex_map"]["file"])
    vox2tkr = np.array(doc["volume"]["vox2ras_tkr"])
    tkr2vox = np.linalg.inv(vox2tkr)
    # slices are 1 mm apart and snapped to voxel centres: 1 mm apart on the 1 mm grid, 0.8 or
    # 1.6 mm apart on the 0.8 mm grid, where every fifth voxel plane has no tile
    spacing = max(np.diff(panel["slice_mm"]).max() for panel in doc["panels"].values())
    assert spacing == pytest.approx(1.0 if which == "default_render" else 1.6)
    half_voxel = spacing / 2
    low, high = np.array(doc["box"]["low"]), np.array(doc["box"]["high"])

    stored = np.concatenate([vmap["lh_pial"], vmap["rh_pial"]]).astype(float)
    search = cKDTree(stored)
    # the reach of a click: half a voxel across the slice, half a pixel (1/8 mm) within it;
    # a vertex is a fair target when no other vertex is within twice that
    reach = float(np.hypot(half_voxel, np.hypot(0.125, 0.125)))
    lonely = search.query(stored, k=2)[0][:, 1] > 2 * reach + 0.01
    inside = ((stored > low + 1) & (stored < high - 1)).all(axis=1)
    candidates = np.flatnonzero(lonely & inside)
    assert len(candidates) > 200
    rng = np.random.default_rng(20261112)
    n_lh = len(vmap["lh_index"])
    for row in rng.choice(candidates, size=60, replace=False).tolist():
        hemi, local = ("lh", row) if row < n_lh else ("rh", row - n_lh)
        native = int(vmap[f"{hemi}_index"][local])
        surface = read_surface(rendered.subject.path / "surf" / f"{hemi}.pial")
        xyz = surface.vertices[native]
        np.testing.assert_array_equal(xyz.astype(np.float32), vmap[f"{hemi}_pial"][local])
        ijk = tkr2vox[:3, :3] @ xyz + tkr2vox[:3, 3]
        for name in ("sagittal", "coronal", "axial"):
            panel = doc["panels"][name]
            s_axis, h_axis, v_axis = (
                panel[k] for k in ("slice_voxel_axis", "h_voxel_axis", "v_voxel_axis")
            )
            pos = int(np.argmin(np.abs(np.array(panel["slices"]) - ijk[s_axis])))
            px, py, _ = np.array(panel["vox2pix"]) @ [ijk[h_axis], ijk[v_axis], 1.0]
            assert 0 <= px < panel["width"] and 0 <= py < panel["height"]
            click = [np.floor(px) + 0.5, np.floor(py) + 0.5, 1.0]

            back = np.empty(3)
            back[h_axis], back[v_axis], _ = np.array(panel["pix2vox"]) @ click
            back[s_axis] = panel["slices"][pos]
            tkr = vox2tkr[:3, :3] @ back + vox2tkr[:3, 3]
            assert np.abs(tkr - xyz).max() <= half_voxel + 1e-6, (name, native)
            assert tkr[panel["world_axis"]] == pytest.approx(panel["slice_mm"][pos], abs=1e-9)
            assert search.query(tkr)[1] == row, (name, native)


# ---------------------------------------------------------------- the vertex map and the tiles

SNAP_MM = 5.0  # docs/rater_app.md: a mark goes to the nearest vertex within 5 mm


def tile_region(doc: dict) -> np.ndarray:
    """The clickable region of docs/rendering.md, from render.json alone: for each panel the
    low and the high corner, in tkr RAS, of the box its tiles sweep. Along the slice axis it
    runs from half a slice step before the first slice plane to half a step past the last,
    within the plane from tile edge to tile edge."""
    vox2tkr = np.array(doc["volume"]["vox2ras_tkr"])
    half_step = doc["config"]["slice_step_mm"] / 2
    boxes = []
    for name in PANEL_ORDER:
        panel = doc["panels"][name]
        corners = np.array([[0.0, 0.0, 1.0], [panel["width"], panel["height"], 1.0]])
        in_plane = corners @ np.array(panel["pix2vox"]).T
        ijk = np.zeros((2, 3))
        ijk[:, panel["h_voxel_axis"]] = in_plane[:, 0]
        ijk[:, panel["v_voxel_axis"]] = in_plane[:, 1]
        tkr = ijk @ vox2tkr[:3, :3].T + vox2tkr[:3, 3]
        tkr[:, panel["world_axis"]] = [
            min(panel["slice_mm"]) - half_step,
            max(panel["slice_mm"]) + half_step,
        ]
        boxes.append([tkr.min(axis=0), tkr.max(axis=0)])
    return np.array(boxes)


def in_boxes(xyz: np.ndarray, boxes: np.ndarray, grow: float) -> np.ndarray:
    inside = np.zeros(len(xyz), dtype=bool)
    for low, high in boxes:
        inside |= ((xyz >= low - grow) & (xyz <= high + grow)).all(axis=1)
    return inside


def tile_points(
    doc: dict, surfaces: dict[str, tuple[np.ndarray, np.ndarray]], clicks: bool = True
) -> np.ndarray:
    """tkr RAS of points on every tile: three on each contour segment of the given meshes,
    where they fall inside the tile, and, with ``clicks``, a grid of clicks that includes the
    tile's edges."""
    vox2tkr = np.array(doc["volume"]["vox2ras_tkr"])
    tkr2vox = np.linalg.inv(vox2tkr)
    points = []
    for name in PANEL_ORDER:
        panel = doc["panels"][name]
        s_axis, h_axis, v_axis = (
            panel[k] for k in ("slice_voxel_axis", "h_voxel_axis", "v_voxel_axis")
        )
        size = panel["width"]
        assert panel["height"] == size
        edge = np.append(np.arange(0.0, size, 8.0), [size - 0.5, size])
        px, py = (grid.ravel() for grid in np.meshgrid(edge, edge))
        grid_h, grid_v, _ = np.array(panel["pix2vox"]) @ [px, py, np.ones_like(px)]
        for index, plane in zip(panel["slices"], panel["slice_mm"], strict=True):
            for vertices, faces in surfaces.values():
                ends = plane_segments(vertices, faces, panel["world_axis"], plane)
                on_contour = np.concatenate([ends[:, 0], ends.mean(axis=1), ends[:, 1]])
                ijk = on_contour @ tkr2vox[:3, :3].T + tkr2vox[:3, 3]
                cx, cy, _ = np.array(panel["vox2pix"]) @ [
                    ijk[:, h_axis],
                    ijk[:, v_axis],
                    np.ones(len(ijk)),
                ]
                shown = (cx >= 0) & (cx <= size) & (cy >= 0) & (cy <= size)
                points.append(on_contour[shown])
            if not clicks:
                continue
            ijk = np.empty((len(px), 3))
            ijk[:, h_axis], ijk[:, v_axis], ijk[:, s_axis] = grid_h, grid_v, index
            points.append(ijk @ vox2tkr[:3, :3].T + vox2tkr[:3, 3])
    return np.concatenate(points)


def check_marks_resolve_as_on_the_full_surfaces(subject: Path, out: Path, snap_mm: float) -> None:
    doc = json.loads((out / "render.json").read_text())
    vmap = read_vertex_map(out / doc["vertex_map"]["file"])
    full = {
        f"{hemi}_{name}": read_surface(subject / "surf" / f"{hemi}.{name}")
        for hemi in ("lh", "rh")
        for name in ("pial", "white")
    }
    for name in ("pial", "white"):
        meshes = {
            hemi: (full[f"{hemi}_{name}"].vertices, full[f"{hemi}_{name}"].faces)
            for hemi in ("lh", "rh")
        }
        marks = tile_points(doc, meshes)
        # the stored coordinates are float32, so the full surface is compared at that precision
        everything = np.concatenate([v for v, _ in meshes.values()]).astype(np.float32)
        stored = np.concatenate([vmap[f"lh_{name}"], vmap[f"rh_{name}"]])
        d_full, _ = cKDTree(everything).query(marks, distance_upper_bound=snap_mm)
        d_map, _ = cKDTree(stored).query(marks, distance_upper_bound=snap_mm)
        assert np.isfinite(d_full).sum() > 5000  # the tiles do show the surface
        # the same distance is the same vertex, or an exact tie; inf where there is none in reach
        wrong = d_map != d_full
        assert not wrong.any(), (
            f"{name}: {wrong.sum()} of {np.isfinite(d_full).sum()} marks with a vertex within "
            f"{snap_mm} mm resolve to another vertex, or to none, through vertex_map.npz"
        )


@pytest.mark.parametrize("which", ["default_render", "hires_render"])
def test_a_mark_on_a_tile_resolves_through_the_vertex_map_as_on_the_full_surfaces(
    which: str, request: pytest.FixtureRequest
) -> None:
    """vertex_map.npz stands in for the surfaces once they are deleted (docs/rendering.md).
    For points on every contour that a tile shows, and for clicks all over every tile, the
    nearest vertex within the snap radius is the same in the vertex map as on the full surface.
    The tiles reach 13.5 mm beyond the default box anteriorly and 10 mm on the other sides."""
    rendered: Rendered = request.getfixturevalue(which)
    assert rendered.info.config.vertex_map_margin_mm == SNAP_MM
    check_marks_resolve_as_on_the_full_surfaces(rendered.subject.path, rendered.out, SNAP_MM)


@pytest.mark.parametrize(
    ("update", "snap_mm"),
    [
        ({"margin_mm": 20.0}, SNAP_MM),  # tiles that show 20 mm and more around the box
        ({"box_z_mm": (-10.0, 10.0)}, SNAP_MM),  # a flat box: 20 axial slices, tiles 80 mm high
        ({"vertex_map_margin_mm": 2.5, "slice_step_mm": 2.0}, 2.5),  # exact up to the margin
    ],
    ids=["wide-panel-margin", "flat-box", "small-vertex-map-margin"],
)
def test_marks_resolve_whatever_the_panel_margin_the_box_and_the_vertex_map_margin(
    synth_subject: SyntheticSubject, tmp_path: Path, update: dict[str, object], snap_mm: float
) -> None:
    config = RenderConfig(panel_px=64).model_copy(update=update)
    render_subject(
        resolve_subject_files(synth_subject.path), tmp_path / "media", config, video=False
    )
    check_marks_resolve_as_on_the_full_surfaces(synth_subject.path, tmp_path / "media", snap_mm)


@pytest.mark.parametrize("which", ["default_render", "hires_render"])
def test_vertex_map_selection_and_visible_flag_follow_the_tiles(
    which: str, request: pytest.FixtureRequest
) -> None:
    """A vertex is in the map if its pial or white position lies in the clickable region
    grown by the margin, and visible if its pial position lies in the region itself."""
    rendered: Rendered = request.getfixturevalue(which)
    doc = json.loads((rendered.out / "render.json").read_text())
    vmap = read_vertex_map(rendered.out / doc["vertex_map"]["file"])
    region, margin = tile_region(doc), doc["config"]["vertex_map_margin_mm"]
    assert margin == SNAP_MM
    for hemi in ("lh", "rh"):
        pial = read_surface(rendered.subject.path / "surf" / f"{hemi}.pial").vertices
        white = read_surface(rendered.subject.path / "surf" / f"{hemi}.white").vertices
        index, visible = vmap[f"{hemi}_index"], vmap[f"{hemi}_visible"]
        chosen = np.zeros(len(pial), dtype=bool)
        chosen[index] = True
        seen = np.zeros(len(pial), dtype=bool)
        seen[index[visible]] = True
        # region recomputed from render.json: 1e-6 mm of slack either way at its faces
        for flags, grow, xyzs in ((chosen, margin, (pial, white)), (seen, 0.0, (pial,))):
            surely = np.any([in_boxes(xyz, region, grow - 1e-6) for xyz in xyzs], axis=0)
            at_most = np.any([in_boxes(xyz, region, grow + 1e-6) for xyz in xyzs], axis=0)
            assert (surely <= flags).all() and (flags <= at_most).all()
        assert visible.dtype == np.bool_ and not seen[~chosen].any()
        assert doc["vertex_map"][f"n_{hemi}"] == len(index)
        assert doc["vertex_map"][f"n_{hemi}_visible"] == int(visible.sum())
        # more than the box holds, fewer than the map: the tiles show 10 mm and more around
        # the box, and the map reaches 5 mm beyond the tiles
        low, high = np.array(doc["box"]["low"]), np.array(doc["box"]["high"])
        in_box = ((pial >= low) & (pial <= high)).all(axis=1)
        assert in_box.sum() < visible.sum() < len(index) < len(pial)
        assert seen[in_box].all()


@pytest.mark.parametrize("which", ["default_render", "hires_render"])
def test_a_mark_on_a_drawn_pial_contour_may_resolve_to_a_vertex_just_outside_the_region(
    which: str, request: pytest.FixtureRequest
) -> None:
    """``visible`` says where a vertex is, not what a mark can resolve to (docs/rendering.md).
    A drawn contour point lies in the clickable region, on a face of the mesh, and the vertex
    nearest to it can lie outside the region: past half a slice step at the ends of a sweep,
    or past the tile edge. It is then no farther outside than the point is from the nearest
    corner of its face, which is at most the longest edge divided by the square root of 3.
    The analysis must not take a mark's vertex for a visible one (docs/analysis_plan.md)."""
    rendered: Rendered = request.getfixturevalue(which)
    doc = json.loads((rendered.out / "render.json").read_text())
    vmap = read_vertex_map(rendered.out / doc["vertex_map"]["file"])
    region = tile_region(doc)
    meshes, longest = {}, 0.0
    for hemi in ("lh", "rh"):
        pial = read_surface(rendered.subject.path / "surf" / f"{hemi}.pial")
        meshes[hemi] = (pial.vertices, pial.faces)
        corners = pial.vertices[pial.faces]
        longest = max(longest, np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max())
    marks = tile_points(doc, meshes, clicks=False)
    assert len(marks) > 50_000 and in_boxes(marks, region, 1e-6).all()
    stored = np.concatenate([vmap["lh_pial"], vmap["rh_pial"]]).astype(float)
    visible = np.concatenate([vmap["lh_visible"], vmap["rh_visible"]])
    _, nearest = cKDTree(stored).query(marks)
    beyond = ~visible[nearest]
    # measured: 346 of 101,596 points on the 1 mm subject, 373 of 101,104 on the 0.8 mm one
    assert 0.001 < beyond.mean() < 0.01
    assert in_boxes(stored[nearest[beyond]], region, longest / np.sqrt(3)).all()
    assert not in_boxes(stored[nearest[beyond]], region, 0.0).any()
