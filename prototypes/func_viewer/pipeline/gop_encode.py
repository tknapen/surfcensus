"""H.264 with a chosen key-frame interval, for the playback measurements (README, "Fast
playback and scrubbing").

`surfcensus.render.encode.write_mp4` has no option for the key-frame interval, and the package
is not changed for a prototype. This is its ffmpeg command line as it stood on 2026-11-13 with
one flag added, `-g KEYINT`: at most KEYINT frames from one key frame to the next, so KEYINT = 1
makes every frame a key frame (all-intra). With KEYINT = 250, the default of x264, the file
should equal the one that write_mp4 makes; run_playback.sh checks that, so a change of the
package's command line that is not copied here is noticed.
"""

import subprocess
from contextlib import suppress
from itertools import chain
from pathlib import Path

import numpy as np

from surfcensus.render.encode import ffmpeg_exe

# as surfcensus.render.encode._VIDEO_FILTER
VIDEO_FILTER = (
    "scale=flags=bicubic+accurate_rnd+bitexact:out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709,format=yuv420p"
)


def write_mp4_keyint(path, frames, fps, crf, keyint):
    """RGB uint8 frames (h, w, 3) to an H.264 mp4 at `path`; returns the number of frames."""
    path = Path(path)
    rest = iter(frames)
    first = next(rest)
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"yuv420p needs an even width and height, got {width}x{height}")
    tmp = path.with_name(f".{path.name}.part")
    # fmt: off
    command = [
        ffmpeg_exe(), "-y", "-nostats", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-vf", VIDEO_FILTER, "-filter_threads", "1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-g", str(keyint),
        "-threads", "1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", f"file:{tmp}",
    ]
    # fmt: on
    n = 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin, stderr = process.stdin, process.stderr
    assert stdin is not None and stderr is not None
    try:
        for frame in chain((first,), rest):  # one frame at a time, never the whole film
            stdin.write(np.ascontiguousarray(frame).data)
            n += 1
    except BrokenPipeError:  # ffmpeg has exited; its message is reported below
        pass
    with suppress(OSError):
        stdin.close()
    # ffmpeg runs with -loglevel error, so what it writes to stderr is short
    message = stderr.read().decode(errors="replace").strip()
    stderr.close()
    status = process.wait()
    if status != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed with status {status}: {message}")
    tmp.replace(path)
    return n
