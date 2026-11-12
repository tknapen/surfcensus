"""Writers for tiles, poster and video (docs/rendering.md, "Output per subject" and "Video").

Conventions:

* Images are uint8 arrays indexed (row, column[, channel]), row 0 at the top; channels are RGB
  or RGBA with straight (not premultiplied) alpha.
* Every writer creates the parent directories, writes to a temporary file in the destination
  directory and renames it into place, so the final name never holds a partial file.
* Every encoder parameter that changes the output is fixed here, so output is byte-identical
  between runs for the same input within one locked environment (Pillow, libwebp, ffmpeg).
* WebP has no single-channel mode: a grayscale tile is stored as RGB and reads back as
  (h w 3), with channels that differ by at most 1.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from jaxtyping import UInt8
from PIL import Image

from surfcensus._io import atomic_path

_LOSSY = {"lossless": False, "method": 4}
# exact=True keeps RGB under alpha 0 as given instead of zeroing it, so any tile round-trips.
_LOSSLESS = {"lossless": True, "exact": True, "method": 2, "quality": 75}
# RGB to limited-range BT.709 4:2:0 on swscale's C path (the default SIMD path gives other
# bytes), tagged so that players do not have to guess the matrix.
_VIDEO_FILTER = (
    "scale=flags=bicubic+accurate_rnd+bitexact:out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709,format=yuv420p"
)


def _write_webp(path: Path | str, image: Image.Image, **params: bool | int) -> None:
    with atomic_path(path) as tmp:
        image.save(tmp, format="WEBP", **params)


def _checked_quality(quality: int) -> int:
    if not 0 <= quality <= 100:
        raise ValueError(f"WebP quality must be in 0..100, got {quality}")
    return quality


def write_gray_webp(path: Path | str, tile: UInt8[np.ndarray, "h w"], quality: int) -> None:
    """Lossy WebP of a grayscale tile, stored as RGB with equal channels."""
    image = Image.fromarray(tile).convert("RGB")
    _write_webp(path, image, quality=_checked_quality(quality), **_LOSSY)


def write_rgb_webp(path: Path | str, image: UInt8[np.ndarray, "h w 3"], quality: int) -> None:
    """Lossy WebP of an RGB image (the poster)."""
    _write_webp(path, Image.fromarray(image), quality=_checked_quality(quality), **_LOSSY)


def write_rgba_webp(path: Path | str, tile: UInt8[np.ndarray, "h w 4"]) -> None:
    """Lossless WebP with alpha; :func:`read_image` returns ``tile`` exactly.

    A tile with no pixel of alpha below 255 is refused: libwebp would store it without an
    alpha channel and it would read back as (h w 3).
    """
    if not (tile[..., 3] < 255).any():
        raise ValueError("tile has no transparent pixel; WebP would drop its alpha channel")
    _write_webp(path, Image.fromarray(tile), **_LOSSLESS)


def read_image(path: Path | str) -> UInt8[np.ndarray, "h w *channels"]:
    """Decode a WebP or PNG as stored: (h w) gray, (h w 3) RGB or (h w 4) RGBA."""
    with Image.open(path) as image:
        if image.mode not in ("L", "RGB", "RGBA"):
            raise ValueError(f"{path}: unsupported pixel format {image.mode!r}")
        return np.array(image)


def ffmpeg_exe() -> str:
    """The ffmpeg binary that :func:`write_mp4` runs, as imageio-ffmpeg finds it: the one named
    by ``IMAGEIO_FFMPEG_EXE``, else the one in its wheel, else one that is installed.

    Raises RuntimeError if there is none, or if the one named is not a file that can be
    executed (imageio-ffmpeg returns ``IMAGEIO_FFMPEG_EXE`` unchecked). Nothing is started.
    """
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if shutil.which(exe) is None:
        raise RuntimeError(f"the ffmpeg binary {exe!r} is missing or not executable")
    return exe


def write_mp4(
    path: Path | str, frames: Iterable[UInt8[np.ndarray, "h w 3"]], fps: int, crf: int
) -> int:
    """H.264 video (yuv420p, preset veryfast, one thread) from RGB frames; returns their number.

    Frames are consumed one at a time and piped to the ffmpeg binary of imageio-ffmpeg. Raises
    ValueError for no frames, an odd width or height, frames that differ in shape, or fps and
    crf out of range, and RuntimeError when ffmpeg fails: no binary, one that cannot be
    started, a non-zero exit status, or no output file.
    """
    if fps < 1 or not 0 <= crf <= 51:
        raise ValueError(f"fps must be at least 1 and crf in 0..51, got fps={fps}, crf={crf}")
    rest = iter(frames)
    first = next(rest, None)
    if first is None:
        raise ValueError("no frames to encode")
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"frames must be uint8 (h w 3), got {first.dtype} {first.shape}")
    height, width = first.shape[:2]
    if height % 2 or width % 2 or not height or not width:
        raise ValueError(f"yuv420p needs an even, non-zero width and height, got {width}x{height}")

    n, broken = 0, False
    exe = ffmpeg_exe()
    with atomic_path(path) as tmp:
        # fmt: off
        command = [
            exe, "-y", "-nostats", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-vf", _VIDEO_FILTER, "-filter_threads", "1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-threads", "1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", f"file:{tmp}",
        ]
        # fmt: on
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as exc:  # not a program for this machine, or gone since it was found
            raise RuntimeError(f"ffmpeg failed to start: {exc}") from exc
        stdin, stderr = process.stdin, process.stderr
        assert stdin is not None and stderr is not None
        log: list[bytes] = []
        # drained on a thread so that ffmpeg can never block on a full stderr pipe
        drain = threading.Thread(target=lambda: log.append(stderr.read()))
        drain.start()
        try:
            for frame in chain((first,), rest):
                if frame.shape != first.shape or frame.dtype != np.uint8:
                    raise ValueError(
                        f"frame {n} is {frame.dtype} {frame.shape}, expected uint8 {first.shape}"
                    )
                try:
                    stdin.write(np.ascontiguousarray(frame).data)
                except BrokenPipeError:  # ffmpeg has exited; reported below
                    broken = True
                    break
                n += 1
        except BaseException:
            process.kill()
            raise
        finally:
            try:
                stdin.close()
            except OSError:
                broken = True
            process.wait()
            drain.join()
            stderr.close()
        if broken or process.returncode != 0:
            message = b"".join(log).decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with status {process.returncode}: {message}")
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("ffmpeg failed: it exited with status 0 and wrote no video")
    return n
