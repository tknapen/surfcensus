"""How much of the frame-to-frame change of the EPI survives H.264: decoded video against the
frames it was made from.

usage: fidelity.py DATA VIDEO_DIR [NAME ...]

Each NAME.mp4 in VIDEO_DIR is a side-by-side video without contours at 2 px per mm
(make_videos.py --video C --no-contours); its left half, run 1 of the raw-rigid stream, is
compared with the frames computed again here. Without names, the four videos that run_all.sh
makes: c_nocont_crf12, c_nocont_crf18, c_nocont_crf23 and c_nocont_crf28.
"""

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from epiview import Panels
from geom import read_itk_ras, resample

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
    sys.exit(2)
D, W = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = sys.argv[3:] or ["c_nocont_crf12", "c_nocont_crf18", "c_nocont_crf23", "c_nocont_crf28"]
exe = imageio_ffmpeg.get_ffmpeg_exe()
run = 1
raw = nib.load(D / f"raw/run-{run}_bold.nii.gz")
rd = np.asarray(raw.dataobj, dtype=np.float32)
pre = nib.load(D / f"deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz")
B = np.asarray(pre.dataobj, dtype=np.float32)
masks = [
    np.asarray(nib.load(D / f"deriv/run-{r}_space-T1w_desc-brain_mask.nii.gz").dataobj) > 0
    for r in (1, 2)
]
mask = masks[0]
hi = float(np.percentile(B[..., ::10][mask], 99.5))
com = np.array(np.nonzero(masks[0] | masks[1])).mean(axis=1)
P = Panels(B.shape[:3], pre.affine, 2.0, np.round(com + np.array([4, 0, 1])).astype(int))
m = read_itk_ras(D / f"deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt")
n = B.shape[3]
src = np.stack(
    [P.strip(resample(rd[..., t], raw.affine, B.shape[:3], pre.affine, m), 0, hi) for t in range(n)]
).astype(np.int16)
head = src.mean(axis=0) > 40
ds = np.diff(src, axis=0)[:, head]
print(
    "source frames (raw-rigid, run 1): frame-to-frame mean |diff| in the head"
    f" {np.abs(ds).mean():.2f} gray levels"
)
for name in NAMES:
    mp4 = W / f"{name}.mp4"
    cmd = [exe, "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True).stdout
    decoded = np.frombuffer(out, np.uint8).reshape(-1, 612, 1920)
    fr = decoded[:n, 48 : 48 + P.height, : P.width].astype(np.int16)
    dd = np.diff(fr, axis=0)[:, head]
    err = (fr - src)[:, head]
    r = np.corrcoef(ds.ravel(), dd.ravel())[0, 1]
    # temporal standard deviation per pixel, video against source
    sd_s, sd_v = src[:, head].std(axis=0), fr[:, head].std(axis=0)
    print(
        f"{name}: {mp4.stat().st_size / 1e6:.2f} MB;"
        f" decoded frame-to-frame mean |diff| {np.abs(dd).mean():.2f};"
        f" r(source diff, decoded diff) = {r:.3f};"
        f" mean |decoded - source| {np.abs(err).mean():.2f};"
        f" temporal SD per pixel: source median {np.median(sd_s):.2f},"
        f" video {np.median(sd_v):.2f}"
    )
