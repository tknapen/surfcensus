"""Decode every video again: frame count, size, key frames, and how far frames are from the PNG
frames they were made from.

usage: check_videos.py DIR    (every *.mp4 in DIR; PNG frames <stem>_tNNNN.png beside them)
"""

import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
    sys.exit(2)
exe = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path(sys.argv[1])


def key_frames(mp4):
    """Indices of the key frames, from ffmpeg's showinfo filter (the frames a decoder can start
    at)."""
    cmd = [exe, "-hide_banner", "-i", str(mp4), "-vf", "showinfo", "-f", "null", "-"]
    info = subprocess.run(cmd, capture_output=True, text=True).stderr
    return [int(n) for n, key in re.findall(r"\] n:\s*(\d+) .*? iskey:(\d)", info) if key == "1"]


for mp4 in sorted(OUT.glob("*.mp4")):
    info = subprocess.run([exe, "-hide_banner", "-i", str(mp4)], capture_output=True, text=True)
    m = re.search(r"Stream.*Video: (\w+).*?, (\d+)x(\d+).*?([\d.]+) fps", info.stderr)
    dur = re.search(r"Duration: ([\d:.]+)", info.stderr).group(1)
    codec, w, h, fps = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    cmd = [exe, "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    size = mp4.stat().st_size
    line = (
        f"{mp4.name}: {codec} {w}x{h} {fps:g} fps, {len(frames)} frames, duration {dur},"
        f" {size / 1e6:.2f} MB, {size * 8 / 1000 / (len(frames) / fps):.0f} kbit/s"
    )
    stem = re.sub(r"_(24|60)fps.*$", "", mp4.stem)
    errs = []
    for png in sorted(OUT.glob(f"{stem}_t*.png")):
        t = int(png.stem[-4:])
        src = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
        d = frames[t].astype(np.int16) - src
        gray = (src[..., 0] == src[..., 1]) & (src[..., 1] == src[..., 2])
        col = ~gray
        errs.append(
            (
                t,
                np.abs(d).mean(),
                10 * np.log10(255**2 / np.mean(d.astype(float) ** 2)),
                np.abs(d[gray]).mean(),
                np.abs(d[col]).mean() if col.any() else float("nan"),
            )
        )
    if errs:
        e = np.array(errs)
        line += (
            f"; vs {len(errs)} PNG frames: mean |err| {e[:, 1].mean():.2f},"
            f" PSNR {e[:, 2].min():.1f}..{e[:, 2].max():.1f} dB, gray pixels {e[:, 3].mean():.2f},"
            f" coloured (contour/marker) pixels {e[:, 4].mean():.1f}"
        )
    # is the motion still in the decoded video? frame-to-frame change of decoded frames
    body = frames[:, 48 : h - 200 if h > 500 else h - 14].astype(np.int16)
    fd = np.abs(np.diff(body, axis=0)).mean(axis=(1, 2, 3))
    line += (
        f"; decoded frame-to-frame mean |diff| median {np.median(fd):.2f},"
        f" largest at frames {np.argsort(-fd)[:4] + 1}"
    )
    # how far a decoder has to go back from a frame to the key frame it must start at
    keys = np.array(key_frames(mp4))
    back = np.arange(len(frames)) - keys[np.searchsorted(keys, np.arange(len(frames)), "right") - 1]
    line += (
        f"; {len(keys)} key frames, frames to decode before the one sought:"
        f" mean {back.mean():.1f}, most {back.max()}"
    )
    print(line)
