"""End-to-end tests of render/api.py on the synthetic subjects (docs/rendering.md)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
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
from scipy import ndimage
from scipy.spatial import cKDTree
from synth import (  # pyrefly: ignore[missing-import]
    T1_GRAY,
    T1_HEAD,
    T1_WHITE,
    SyntheticSubject,
    write_synthetic_subject,
)

import surfcensus
from surfcensus.fsio.surfaces import read_surface
from surfcensus.fsio.volumes import Volume, intensity_window, read_volume
from surfcensus.render import RenderError, api, frames, render_subject, three_plane
from surfcensus.render.encode import read_image
from surfcensus.render.model import PANEL_ORDER, RenderConfig, RenderInfo
from surfcensus.render.slicing import MeshSlicer
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
    assert info.schema_version == 1 and info.surfcensus_version == surfcensus.__version__
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
    assert info.vertex_map.n_lh > 500 and info.vertex_map.n_rh > 500
    assert set(vmap["lh_fsavg6"].tolist()) == {-1}


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
        (10.0, 3, (1.0, 99.5)): (T1_HEAD, T1_WHITE),
        (0.0, 0, (1.0, 99.5)): (T1_WHITE, T1_WHITE + 1),
        (3.5, 1, (1.0, 99.5)): (T1_GRAY, T1_WHITE),
        (3.5, 3, (50.0, 99.5)): (T1_WHITE, T1_WHITE + 1),
    }
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
        low, high = np.array(info.box.low) - margin, np.array(info.box.high) + margin
        vmap = read_vertex_map(tmp_path / "media" / "vertex_map.npz")
        for hemi in ("lh", "rh"):
            near = np.zeros(len(surfaces[f"{hemi}_pial"].vertices), dtype=bool)
            for name in ("pial", "white"):
                xyz = surfaces[f"{hemi}_{name}"].vertices
                near |= ((xyz >= low) & (xyz <= high)).all(axis=1)
            assert vmap[f"{hemi}_index"].tolist() == np.flatnonzero(near).tolist()
        assert 0 < len(vmap["lh_index"]) < len(surfaces["lh_pial"].vertices)


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


# ---------------------------------------------------------------- failures


def protocol_reasons() -> list[str]:
    text = package_files("surfcensus").joinpath("data/protocol.yaml").read_text()
    return yaml.safe_load(text)["unratable_reasons"]


def _oblique_t1(subject: Path) -> None:
    path = subject / "mri" / "T1.mgz"
    image = nib.load(path)
    angle = np.deg2rad(5.0)
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
        "unreadable",
    ),
    "not an mgz": (
        lambda s: (s / "mri" / "T1.mgz").write_bytes(b"not a volume"),
        "render failure",
        "unreadable",
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
    assert not out.exists()


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
