"""Tile, poster and video writers: round trips, determinism, atomic writes, failure paths."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from surfcensus.render import encode
from surfcensus.render.encode import (
    read_image,
    write_gray_webp,
    write_mp4,
    write_rgb_webp,
    write_rgba_webp,
)

PIAL, WHITE = (255, 212, 0), (255, 59, 48)


def gray_tile(h: int = 320, w: int = 320, seed: int = 0, noise: float = 4.0) -> np.ndarray:
    """Blurred blobs on a dark background with a little noise, roughly like a T1 slice."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    image = np.zeros((h, w))
    for _ in range(12):
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.05, 0.3) * min(h, w)
        image += rng.uniform(30, 90) / (1 + np.exp((np.hypot(yy - cy, xx - cx) - r) / 2))
    image += rng.normal(0, noise, image.shape)
    return image.clip(0, 255).astype(np.uint8)


def contour_tile(h: int = 320, w: int = 320, seed: int = 0) -> np.ndarray:
    """Antialiased one-pixel circles in two colours on full transparency; RGB 0 where alpha 0."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    for k in range(6):
        cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0.1, 0.6) * min(h, w)
        coverage = (1.1 - np.abs(np.hypot(yy - cy, xx - cx) - r)).clip(0, 1)
        alpha = np.round(255 * coverage).astype(np.uint8)
        over = alpha > tile[..., 3]
        tile[over, :3] = PIAL if k % 2 else WHITE
        tile[over, 3] = alpha[over]
    assert not tile[tile[..., 3] == 0].any()
    return tile


def rgb_frame(i: int, h: int = 96, w: int = 128) -> np.ndarray:
    """Smooth drifting gray pattern with one coloured stripe, different for every ``i``."""
    yy, xx = np.mgrid[:h, :w]
    gray = (110 + 60 * np.sin((xx + 3 * i) / 17) * np.cos(yy / 23)).astype(np.uint8)
    frame = np.repeat(gray[..., None], 3, axis=2)
    frame[:, (10 + 4 * i) % w : (10 + 4 * i) % w + 6] = PIAL
    return frame


def read_video(path: Path) -> tuple[dict, list[np.ndarray]]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    return meta, [np.frombuffer(b, dtype=np.uint8).reshape(h, w, 3) for b in reader]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- WebP round trips


@pytest.mark.parametrize("seed", range(4))
def test_rgba_round_trip_is_exact(tmp_path: Path, seed: int) -> None:
    tile = contour_tile(seed=seed)
    write_rgba_webp(tmp_path / "c.webp", tile)
    decoded = read_image(tmp_path / "c.webp")
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, tile)


@pytest.mark.parametrize("shape", [(320, 320), (1, 1), (7, 301)])
def test_rgba_all_transparent_round_trip(tmp_path: Path, shape: tuple[int, int]) -> None:
    tile = np.zeros((*shape, 4), dtype=np.uint8)
    write_rgba_webp(tmp_path / "c.webp", tile)
    assert np.array_equal(read_image(tmp_path / "c.webp"), tile)


def test_rgba_round_trip_keeps_colour_under_zero_alpha(tmp_path: Path) -> None:
    """Stricter than the caller's guarantee: arbitrary RGBA, including RGB where alpha is 0."""
    rng = np.random.default_rng(5)
    tile = rng.integers(0, 256, (40, 56, 4), dtype=np.uint8)
    tile[:8, :, 3] = 0
    assert tile[:8, :, :3].any()
    write_rgba_webp(tmp_path / "c.webp", tile)
    assert np.array_equal(read_image(tmp_path / "c.webp"), tile)


def test_rgba_without_transparency_is_refused(tmp_path: Path) -> None:
    tile = np.full((16, 16, 4), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="alpha"):
        write_rgba_webp(tmp_path / "c.webp", tile)
    assert list(tmp_path.iterdir()) == []


def test_pillow_drops_alpha_of_an_opaque_tile(tmp_path: Path) -> None:
    """The library behaviour that the refusal above guards against."""
    tile = np.full((16, 16, 4), 255, dtype=np.uint8)
    Image.fromarray(tile).save(tmp_path / "c.webp", lossless=True, exact=True)
    assert read_image(tmp_path / "c.webp").shape == (16, 16, 3)


def test_gray_round_trip(tmp_path: Path) -> None:
    tile = gray_tile()
    write_gray_webp(tmp_path / "g.webp", tile, 80)
    decoded = read_image(tmp_path / "g.webp")
    assert decoded.dtype == np.uint8
    assert decoded.shape == (320, 320, 3)
    assert np.abs(decoded.astype(int) - decoded[..., :1]).max() <= 1
    error = np.abs(decoded.astype(int) - tile[..., None])
    assert error.mean() < 4
    assert error.max() < 24

    write_gray_webp(tmp_path / "g100.webp", tile, 100)
    assert np.abs(read_image(tmp_path / "g100.webp").astype(int) - tile[..., None]).max() <= 4


def test_gray_flat_tiles_are_reproduced_within_one_level(tmp_path: Path) -> None:
    for level in (0, 1, 110, 254, 255):
        write_gray_webp(tmp_path / "g.webp", np.full((32, 48), level, dtype=np.uint8), 80)
        assert np.abs(read_image(tmp_path / "g.webp").astype(int) - level).max() <= 1


def test_rgb_round_trip(tmp_path: Path) -> None:
    image = rgb_frame(3, 320, 960)
    write_rgb_webp(tmp_path / "poster.webp", image, 80)
    decoded = read_image(tmp_path / "poster.webp")
    assert decoded.dtype == np.uint8
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(int) - image).mean() < 3


@pytest.mark.parametrize("kind", ["gray", "rgb"])
def test_quality_increases_file_size(tmp_path: Path, kind: str) -> None:
    gray = gray_tile(noise=8.0)
    sizes = []
    for quality in (10, 30, 50, 70, 80, 90, 100):
        path = tmp_path / f"{quality}.webp"
        if kind == "gray":
            write_gray_webp(path, gray, quality)
        else:
            write_rgb_webp(path, np.stack([gray, gray[::-1], gray[:, ::-1]], axis=2), quality)
        sizes.append(path.stat().st_size)
    assert sizes == sorted(set(sizes))


@pytest.mark.parametrize("quality", [-1, 101])
def test_quality_out_of_range_raises(tmp_path: Path, quality: int) -> None:
    with pytest.raises(ValueError, match="quality"):
        write_gray_webp(tmp_path / "g.webp", gray_tile(16, 16), quality)
    with pytest.raises(ValueError, match="quality"):
        write_rgb_webp(tmp_path / "p.webp", rgb_frame(0), quality)
    assert list(tmp_path.iterdir()) == []


def test_non_contiguous_input_and_inputs_unchanged(tmp_path: Path) -> None:
    tile = contour_tile(64, 80)
    view = np.asfortranarray(tile)[::-1, ::2]
    before = view.copy()
    write_rgba_webp(tmp_path / "c.webp", view)
    assert np.array_equal(read_image(tmp_path / "c.webp"), before)
    assert np.array_equal(view, before)

    gray = gray_tile(64, 80)
    write_gray_webp(tmp_path / "a.webp", gray.T, 80)
    write_gray_webp(tmp_path / "b.webp", np.ascontiguousarray(gray.T), 80)
    assert digest(tmp_path / "a.webp") == digest(tmp_path / "b.webp")


# --- read_image


def test_read_image_png_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    for shape in ((9, 13), (9, 13, 3), (9, 13, 4)):
        array = rng.integers(0, 256, shape, dtype=np.uint8)
        Image.fromarray(array).save(tmp_path / "x.png")
        decoded = read_image(tmp_path / "x.png")
        assert decoded.dtype == np.uint8
        assert np.array_equal(decoded, array)
        decoded[0, 0] = 0  # a writable copy, not a view of the file


def test_read_image_refuses_other_pixel_formats(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).convert("P").save(tmp_path / "p.png")
    with pytest.raises(ValueError, match="pixel format"):
        read_image(tmp_path / "p.png")


# --- video


def test_mp4_read_back(tmp_path: Path) -> None:
    frames = [rgb_frame(i) for i in range(30)]
    path = tmp_path / "anim.mp4"
    assert write_mp4(path, frames, fps=12, crf=23) == 30

    meta, decoded = read_video(path)
    assert meta["size"] == (128, 96)
    assert meta["fps"] == 12
    assert meta["codec"].startswith("h264")
    # limited range, and matrix, primaries and transfer all BT.709 (ffmpeg prints the three
    # as a/b/c when they differ): tagged, not left to the player
    assert meta["pix_fmt"].startswith("yuv420p(tv, bt709") and "/" not in meta["pix_fmt"]
    assert meta["duration"] == pytest.approx(30 / 12, abs=0.01)
    assert len(decoded) == 30
    assert imageio_ffmpeg.count_frames_and_secs(str(path))[0] == 30
    errors = [np.abs(d.astype(int) - f).mean() for d, f in zip(decoded, frames, strict=True)]
    assert max(errors) < 4
    # frames are distinct and in order: each decoded frame is nearest to its own input
    for i in (0, 7, 29):
        nearest = np.argmin([np.abs(decoded[i].astype(int) - f).mean() for f in frames])
        assert nearest == i


def test_mp4_encoder_settings_are_the_documented_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/rendering.md, "Video": H.264 with preset veryfast and the given CRF and frame rate,
    yuv420p in limited-range BT.709, a single thread and the bit-exact flags. Most of these
    do not show in a file made and read back on one machine, so the command is looked at."""
    commands: list[list[str]] = []
    popen = subprocess.Popen

    def spy(command: list[str], **kwargs: int) -> subprocess.Popen[bytes]:
        commands.append(list(command))
        return popen(command, **kwargs)

    monkeypatch.setattr(encode.subprocess, "Popen", spy)
    write_mp4(tmp_path / "anim.mp4", [rgb_frame(i) for i in range(3)], fps=12, crf=27)
    (command,) = commands
    split = command.index("-i") + 2  # options before the input describe it, the rest the output
    of_input, of_output = command[:split], command[split:]

    def option(flag: str, options: list[str] = of_output) -> str:
        assert options.count(flag) == 1, flag
        return options[options.index(flag) + 1]

    assert option("-framerate", of_input) == "12"
    assert (option("-c:v"), option("-preset"), option("-crf")) == ("libx264", "veryfast", "27")
    assert option("-pix_fmt") == "yuv420p"
    assert (option("-threads"), option("-filter_threads")) == ("1", "1")
    assert (option("-fflags"), option("-flags:v")) == ("+bitexact", "+bitexact")
    for part in ("accurate_rnd+bitexact", "out_color_matrix=bt709", "out_range=tv", "yuv420p"):
        assert part in option("-vf"), part


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="needs a way to hide processors from ffmpeg, and more than one of them",
)
def test_mp4_bytes_do_not_depend_on_the_number_of_processors(tmp_path: Path) -> None:
    """x264 takes its number of threads from the processors it sees, and its output depends
    on that number; with one thread the file is the same on any machine."""
    frames = [rgb_frame(i, 320, 960) for i in range(24)]
    write_mp4(tmp_path / "all.mp4", frames, fps=12, crf=23)
    allowed = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {min(allowed)})
        write_mp4(tmp_path / "one.mp4", frames, fps=12, crf=23)
    finally:
        os.sched_setaffinity(0, allowed)
    assert digest(tmp_path / "all.mp4") == digest(tmp_path / "one.mp4")


def test_mp4_crf_changes_size(tmp_path: Path) -> None:
    sizes = []
    for crf in (35, 23, 12):
        write_mp4(tmp_path / f"{crf}.mp4", (rgb_frame(i) for i in range(12)), fps=12, crf=crf)
        sizes.append((tmp_path / f"{crf}.mp4").stat().st_size)
    assert sizes == sorted(set(sizes))


def test_mp4_generator_is_consumed_lazily(tmp_path: Path) -> None:
    """One buffer is reused for every frame: a writer that collected the frames first would
    encode the last one n times."""
    levels = [20, 80, 140, 200, 240, 60]
    buffer = np.zeros((48, 64, 3), dtype=np.uint8)
    pulled = []

    def frames() -> Iterator[np.ndarray]:
        for level in levels:
            buffer[:] = level
            pulled.append(level)
            yield buffer

    assert write_mp4(tmp_path / "anim.mp4", frames(), fps=12, crf=18) == len(levels)
    assert pulled == levels
    _, decoded = read_video(tmp_path / "anim.mp4")
    assert [round(float(d.mean())) for d in decoded] == pytest.approx(levels, abs=2)


def test_mp4_accepts_non_contiguous_frames(tmp_path: Path) -> None:
    frames = [rgb_frame(i) for i in range(5)]
    write_mp4(tmp_path / "a.mp4", frames, fps=12, crf=23)
    write_mp4(tmp_path / "b.mp4", (np.asfortranarray(f[:, ::-1])[:, ::-1] for f in frames), 12, 23)
    assert digest(tmp_path / "a.mp4") == digest(tmp_path / "b.mp4")


@pytest.mark.parametrize("shape", [(48, 65, 3), (47, 64, 3), (47, 65, 3)])
def test_mp4_odd_dimensions_raise(tmp_path: Path, shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="even"):
        write_mp4(tmp_path / "anim.mp4", [np.zeros(shape, dtype=np.uint8)] * 3, fps=12, crf=23)
    assert list(tmp_path.iterdir()) == []


def test_mp4_no_frames_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_mp4(tmp_path / "anim.mp4", [], fps=12, crf=23)
    with pytest.raises(ValueError, match="no frames"):
        write_mp4(tmp_path / "anim.mp4", (rgb_frame(i) for i in range(0)), fps=12, crf=23)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("fps", "crf"), [(0, 23), (12, -1), (12, 52)])
def test_mp4_parameters_out_of_range_raise(tmp_path: Path, fps: int, crf: int) -> None:
    with pytest.raises(ValueError, match="fps"):
        write_mp4(tmp_path / "anim.mp4", [rgb_frame(0)], fps=fps, crf=crf)
    assert list(tmp_path.iterdir()) == []


def test_mp4_inconsistent_frames_raise(tmp_path: Path) -> None:
    def changing_shape() -> Iterator[np.ndarray]:
        yield rgb_frame(0)
        yield rgb_frame(1)[:, :64]

    def changing_dtype() -> Iterator[np.ndarray]:
        yield rgb_frame(0)
        yield rgb_frame(1).astype(np.float32)

    for frames in (changing_shape(), changing_dtype()):
        with pytest.raises(ValueError, match="frame 1"):
            write_mp4(tmp_path / "anim.mp4", frames, fps=12, crf=23)
    assert list(tmp_path.iterdir()) == []


# --- determinism


def test_every_writer_is_byte_identical_across_calls(tmp_path: Path) -> None:
    gray, contours, poster = gray_tile(), contour_tile(), rgb_frame(2, 320, 960)
    for run in ("a", "b"):
        out = tmp_path / run
        write_gray_webp(out / "gray.webp", gray, 80)
        write_rgba_webp(out / "contours.webp", contours)
        write_rgb_webp(out / "poster.webp", poster, 80)
        write_mp4(out / "anim.mp4", (rgb_frame(i) for i in range(24)), fps=12, crf=23)
    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == ["anim.mp4", "contours.webp", "gray.webp", "poster.webp"]
    for name in names:
        assert digest(tmp_path / "a" / name) == digest(tmp_path / "b" / name), name


def test_mp4_carries_no_run_dependent_metadata(tmp_path: Path) -> None:
    write_mp4(tmp_path / "anim.mp4", [rgb_frame(0)] * 2, fps=12, crf=23)
    data = (tmp_path / "anim.mp4").read_bytes()
    assert b"Lavf" not in data  # muxer version string
    mvhd = data.index(b"mvhd")
    assert data[mvhd + 8 : mvhd + 16] == bytes(8)  # creation and modification time
    assert data.index(b"moov") < data.index(b"mdat")  # faststart


# --- atomic writes


def test_parent_directories_are_created_and_no_temporary_files_remain(tmp_path: Path) -> None:
    out = tmp_path / "media" / "uid" / "tiles"
    write_gray_webp(out / "sagittal_000.webp", gray_tile(32, 32), 80)
    write_rgba_webp(out / "sagittal_000_c.webp", contour_tile(32, 32))
    write_rgb_webp(out.parent / "poster.webp", rgb_frame(0), 80)
    write_mp4(out.parent / "anim.mp4", [rgb_frame(0)] * 3, fps=12, crf=23)
    found = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert found == [
        "media/uid/anim.mp4",
        "media/uid/poster.webp",
        "media/uid/tiles/sagittal_000.webp",
        "media/uid/tiles/sagittal_000_c.webp",
    ]


def test_mp4_failure_in_the_frame_generator_leaves_nothing(tmp_path: Path) -> None:
    def frames() -> Iterator[np.ndarray]:
        for i in range(5):
            yield rgb_frame(i)
        raise KeyError("render failed")

    with pytest.raises(KeyError, match="render failed"):
        write_mp4(tmp_path / "anim.mp4", frames(), fps=12, crf=23)
    assert list(tmp_path.iterdir()) == []


def test_mp4_failure_keeps_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "anim.mp4"
    write_mp4(path, [rgb_frame(0)] * 3, fps=12, crf=23)
    before = path.read_bytes()

    def frames() -> Iterator[np.ndarray]:
        yield rgb_frame(1)
        raise BrokenPipeError("raised by the caller's generator, not by the pipe")

    with pytest.raises(BrokenPipeError, match="caller"):
        write_mp4(path, frames(), fps=12, crf=23)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_mp4_ffmpeg_failure_raises_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "_VIDEO_FILTER", "no_such_filter")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        write_mp4(tmp_path / "one.mp4", [rgb_frame(0)], fps=12, crf=23)
    with pytest.raises(RuntimeError, match="no_such_filter"):
        write_mp4(tmp_path / "many.mp4", (rgb_frame(i, 320, 960) for i in range(60)), 12, 23)
    assert list(tmp_path.iterdir()) == []


def test_ffmpeg_exe_is_the_binary_of_imageio_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    assert encode.ffmpeg_exe() == imageio_ffmpeg.get_ffmpeg_exe()
    assert Path(encode.ffmpeg_exe()).is_file()


def test_no_ffmpeg_binary_is_a_runtime_error_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """imageio-ffmpeg hands back ``IMAGEIO_FFMPEG_EXE`` unchecked, and raises RuntimeError
    itself when it finds no binary at all."""
    out = tmp_path / "out"
    (tmp_path / "ffmpeg-data").write_bytes(b"not a program")  # a file without execute permission
    for exe in (tmp_path / "no-such-ffmpeg", tmp_path / "ffmpeg-data", tmp_path):
        monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
        with pytest.raises(RuntimeError, match="missing or not executable"):
            encode.ffmpeg_exe()
        with pytest.raises(RuntimeError, match="missing or not executable"):
            write_mp4(out / "a.mp4", [rgb_frame(0)], fps=12, crf=23)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE")

    def none_found() -> str:
        raise RuntimeError("No ffmpeg exe could be found.")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", none_found)
    with pytest.raises(RuntimeError, match="No ffmpeg exe"):
        write_mp4(out / "a.mp4", [rgb_frame(0)], fps=12, crf=23)
    assert not out.exists()  # not even the directory the video would go to


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-ins for ffmpeg are shell scripts")
def test_an_ffmpeg_that_does_not_start_or_writes_nothing_is_a_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    path = out / "anim.mp4"
    write_mp4(path, [rgb_frame(0)] * 3, fps=12, crf=23)
    before = path.read_bytes()
    stand_ins = {
        "not a program\n": "failed to start",  # executable, but the system cannot run it
        "#!/bin/sh\ncat > /dev/null\n": "wrote no video",  # takes every frame, exits with 0
        # the same, and leaves an empty file where the video should be (the last argument)
        '#!/bin/sh\ncat > /dev/null\nfor last; do :; done\n: > "${last#file:}"\n': "wrote no video",
        "#!/bin/sh\ncat > /dev/null\necho no codec >&2\nexit 3\n": "status 3: no codec",
    }
    for number, (script, message) in enumerate(stand_ins.items()):
        exe = tmp_path / f"ffmpeg-{number}"
        exe.write_text(script)
        exe.chmod(0o755)
        monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(exe))
        assert encode.ffmpeg_exe() == str(exe)
        with pytest.raises(RuntimeError, match=message):
            write_mp4(path, (rgb_frame(i) for i in range(5)), fps=12, crf=23)
        assert path.read_bytes() == before
        assert list(out.iterdir()) == [path]  # no temporary file


def test_webp_failure_while_writing_leaves_nothing_and_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "g.webp"
    write_gray_webp(path, gray_tile(32, 32), 80)
    before = path.read_bytes()
    partial = []

    def failing_save(self: Image.Image, fp: Path, **params: object) -> None:
        Path(fp).write_bytes(b"RIFF partial")
        partial.append(Path(fp))
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", failing_save)
    for write in (
        lambda: write_gray_webp(path, gray_tile(32, 32, seed=1), 80),
        lambda: write_rgb_webp(tmp_path / "p.webp", rgb_frame(0), 80),
        lambda: write_rgba_webp(tmp_path / "c.webp", contour_tile(32, 32)),
    ):
        with pytest.raises(OSError, match="disk full"):
            write()
    assert len(partial) == 3
    assert all(
        p.parent == tmp_path and p.name not in ("g.webp", "p.webp", "c.webp") for p in partial
    )
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
