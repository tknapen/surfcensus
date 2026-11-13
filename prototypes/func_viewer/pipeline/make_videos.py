"""Videos of the EPI time series of one subject, all runs in one go.

  A  raw volumes, each resampled with the ONE rigid scanner-to-T1w transform of its run
     (no motion correction, no distortion correction)
  B  fMRIPrep's space-T1w preprocessed series
  C  A and B side by side, framewise displacement and DVARS under them, a cursor on the traces

usage: make_videos.py DATA FS_SUBJECT_DIR OUT --video A|B|C [--fps 12] [--scale 2] [--crf 23]
                      [--no-contours] [--png T,T,...] [--name NAME] [--one-core] [--no-video]
                      [--keyint N]

One frame per volume, so --fps is volumes per second. --keyint N encodes with at most N frames
between key frames through gop_encode.py (1 makes every frame a key frame); without it the
video is written by surfcensus.render.encode.write_mp4. A line of timings and sizes is printed
and appended to OUT/timings.jsonl. --one-core needs Linux.
"""

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("fs_dir")
ap.add_argument("out")
ap.add_argument("--video", choices="ABC", required=True)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--crf", type=int, default=23)
ap.add_argument("--no-contours", action="store_true")
ap.add_argument("--png", default="")
ap.add_argument("--name", default="")
ap.add_argument("--one-core", action="store_true")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--contour-alpha", type=float, default=0.65)
ap.add_argument("--keyint", type=int, default=None)
args = ap.parse_args()
if args.one_core:
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

T0 = time.perf_counter()
# The imports below come after the thread limits, which numpy's libraries read when they load,
# and after T0, so that the reported set-up time includes them. Hence E402 on each.
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from surfcensus.render.encode import write_mp4  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from epiview import Panels, compose, font  # noqa: E402
from geom import read_itk_ras, resample, surfaces_in_t1w  # noqa: E402
from gop_encode import write_mp4_keyint  # noqa: E402

D, FS, OUT = Path(args.data), Path(args.fs_dir), Path(args.out)
RUNS = (1, 2)
FD_LIMIT = 0.5
timing = {"load_s": 0.0, "resample_s": 0.0, "draw_s": 0.0}
need_a, need_b = args.video in "AC", args.video in "BC"

# ---- per run: grids, transform, window, confounds (small files only) -------------------------
runs = []
for r in RUNS:
    ref = nib.load(D / f"deriv/run-{r}_space-T1w_boldref.nii.gz")
    mask_img = nib.load(D / f"deriv/run-{r}_space-T1w_desc-brain_mask.nii.gz")
    conf = pd.read_csv(D / f"deriv/run-{r}_desc-confounds_timeseries.tsv", sep="\t")
    sidecar = json.loads((D / f"deriv/run-{r}_space-T1w_desc-preproc_bold.json").read_text())
    runs.append(
        dict(
            run=r,
            shape=ref.shape,
            affine=ref.affine,
            mask=np.asarray(mask_img.dataobj) > 0,
            tr=sidecar["RepetitionTime"],
            fd=conf["framewise_displacement"].to_numpy(float),
            dvars=conf["std_dvars"].to_numpy(float),
            t1w_to_scanner=read_itk_ras(
                D / f"deriv/run-{r}_from-scanner_to-T1w_mode-image_xfm.txt"
            ),
        )
    )
assert all(
    q["shape"] == runs[0]["shape"] and np.allclose(q["affine"], runs[0]["affine"]) for q in runs
)
shape, affine = runs[0]["shape"], runs[0]["affine"]
union = np.any([q["mask"] for q in runs], axis=0)
com = np.array(np.nonzero(union)).mean(axis=1)
# sagittal 12 mm off the midline, through one hemisphere
slices = np.round(com + np.array([4, 0, 1])).astype(int)
P = Panels(shape, affine, args.scale, slices)
overlay = None
if not args.no_contours:
    surfs, _ = surfaces_in_t1w(FS, D / "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt")
    overlay = P.contours(surfs, 1.0)
    overlay[..., 3] = (overlay[..., 3] * args.contour_alpha + 0.5).astype(np.uint8)
timing["setup_s"] = time.perf_counter() - T0


def load(path):
    t = time.perf_counter()
    img = nib.load(path)
    # int16 as stored; slope and intercept are applied per volume
    a = img.dataobj.get_unscaled()
    timing["load_s"] += time.perf_counter() - t
    return a, img.affine, float(img.dataobj.slope), float(img.dataobj.inter)


def stream(q):
    """Per volume of one run: (raw-rigid volume or None, preprocessed volume or None) on the
    T1w grid."""
    raw = None
    if need_a:
        raw, raw_aff, rs, ri = load(D / f"raw/run-{q['run']}_bold.nii.gz")
    # the preprocessed series is read for A alone as well: the window of a run comes from it
    pre, _, ps, pi = load(D / f"deriv/run-{q['run']}_space-T1w_desc-preproc_bold.nii.gz")
    # one window per run, from the preprocessed series: 0 .. 99.5th percentile within the brain
    # mask
    sub = pre[..., ::10].astype(np.float32) * ps + pi
    q["hi"] = float(np.percentile(sub[q["mask"]], 99.5))
    q["n"] = pre.shape[3]
    for t in range(pre.shape[3]):
        a = None
        if need_a:
            t1 = time.perf_counter()
            vol = raw[..., t].astype(np.float32) * rs + ri
            a = resample(vol, raw_aff, shape, affine, q["t1w_to_scanner"], order=3)
            timing["resample_s"] += time.perf_counter() - t1
        yield a, (pre[..., t].astype(np.float32) * ps + pi if need_b else None)


# ---- layout ---------------------------------------------------------------------------------
n_streams = 2 if args.video == "C" else 1
W = P.width * n_streams
BAR, TITLE, PROG = 26, 22, 14
TRACE_H = 2 * 96 if args.video == "C" else 0
H = BAR + TITLE + P.height + (TRACE_H if args.video == "C" else PROG)
H += H % 2
TITLES = {
    "A": "raw EPI, one rigid transform to T1w space per run (no motion or distortion correction)",
    "B": "preprocessed by fMRIPrep, as shared, in T1w space",
}
SHORT = {"A": "raw, rigid only", "B": "preprocessed"}
N_PER = [len(q["fd"]) for q in runs]
N = sum(N_PER)
START = np.concatenate([[0], np.cumsum(N_PER)])
fd_all = np.concatenate([q["fd"] for q in runs])
dv_all = np.concatenate([q["dvars"] for q in runs])
f_bar, f_title = font(15), font(13)


def traces():
    """FD and DVARS of the whole experiment, drawn once; returns the RGB image and the pixel x
    of every volume."""
    fig = Figure(figsize=(W / 100, TRACE_H / 100), dpi=100, facecolor="black")
    FigureCanvasAgg(fig)
    left, width = 62 / W, (W - 62 - 14) / W
    axes = []
    series = ((fd_all, "FD (mm)", "#4FC3F7"), (dv_all, "DVARS (std)", "#AED581"))
    for i, (y, lab, col) in enumerate(series):
        ax = fig.add_axes((left, 0.5 * (1 - i) + 0.09, width, 0.5 - 0.13), facecolor="black")
        ax.plot(np.arange(N), y, color=col, lw=1.0)
        ax.set_xlim(-0.5, N - 0.5)
        if i == 0:
            ax.axhline(FD_LIMIT, color="#FF5252", lw=0.8, ls="--")
            bad = np.flatnonzero(y > FD_LIMIT)
            ax.plot(bad, y[bad], "o", color="#FF5252", ms=4)
            ax.set_ylim(0, max(np.nanmax(y) * 1.3, FD_LIMIT * 1.2))
        for s in START[1:-1]:
            ax.axvline(s - 0.5, color="white", lw=1.0)
        for k, s in enumerate(START[:-1]):
            if i == 0:
                top = ax.get_ylim()[1] * 0.97
                ax.text(s + 1, top, f"run {k + 1}", color="white", fontsize=8, va="top")
        ax.set_ylabel(lab, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7, length=2)
        ax.set_xticks([] if i == 0 else [*START[:-1], N - 1])
        for sp in ax.spines.values():
            sp.set_color("#888888")
        axes.append(ax)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., :3].copy()
    x = (left + width * (np.arange(N) + 0.5) / N) * W
    return img, x


trace_img, trace_x = traces() if args.video == "C" else (None, None)


def frame(g, a, b, q):
    """One RGB frame for global volume index g."""
    t0 = time.perf_counter()
    k = g - START[q["run"] - 1]
    strips = [compose(P.strip(vol, 0.0, q["hi"]), overlay) for vol in (a, b) if vol is not None]
    img = Image.new("RGB", (W, H), (0, 0, 0))
    y_strip = BAR + TITLE
    img.paste(Image.fromarray(np.concatenate(strips, axis=1)), (0, y_strip))
    for s in range(n_streams):  # letters on every stream
        sub = img.crop((s * P.width, y_strip, (s + 1) * P.width, y_strip + P.height))
        img.paste(P.letters(sub), (s * P.width, y_strip))
    d = ImageDraw.Draw(img)
    fd = q["fd"][k]
    starting = k < 6
    d.rectangle((0, 0, W, BAR - 1), fill=(0, 105, 120) if starting else (28, 28, 28))
    txt = f"run {q['run']} of {len(runs)}   volume {k + 1:3d} / {q['n']}   {k * q['tr']:5.0f} s"
    if starting:
        txt = f"RUN {q['run']} STARTS   " + txt
    d.text((8, 4), txt, font=f_bar, fill=(255, 255, 255))
    fd_txt = "FD  n/a (first volume)" if not np.isfinite(fd) else f"FD {fd:4.2f} mm"
    bad = np.isfinite(fd) and fd > FD_LIMIT
    if bad:
        fd_txt += f"  > {FD_LIMIT} mm"
        d.rectangle((W - 230, 0, W, BAR - 1), fill=(200, 30, 30))
        box = (0, y_strip, W - 1, y_strip + P.height - 1)
        d.rectangle(box, outline=(255, 60, 60), width=4)
    d.text((W - 222, 4), fd_txt, font=f_bar, fill=(255, 255, 255))
    for s, key in enumerate("AB" if args.video == "C" else args.video):
        title = TITLES[key] if P.width >= 900 else SHORT[key]
        d.text((s * P.width + 8, BAR + 3), title, font=f_title, fill=(220, 220, 220))
        if s:
            d.line((s * P.width, BAR, s * P.width, y_strip + P.height), fill=(90, 90, 90))
    if args.video == "C":
        y0 = y_strip + P.height
        img.paste(Image.fromarray(trace_img), (0, y0))
        x = float(trace_x[g])
        d.line((x, y0 + 2, x, y0 + TRACE_H - 2), fill=(255, 255, 255), width=1)
    else:  # a progress bar with the run boundaries
        y0 = y_strip + P.height + 3
        d.rectangle((0, y0, W - 1, y0 + 7), fill=(45, 45, 45))
        d.rectangle((0, y0, int(W * (g + 1) / N), y0 + 7), fill=(150, 150, 150))
        for s in START[1:-1]:
            xs = int(W * s / N)
            d.rectangle((xs - 1, y0 - 3, xs + 1, y0 + 10), fill=(0, 200, 220))
        for j in np.flatnonzero(fd_all > FD_LIMIT):
            xs = int(W * (j + 0.5) / N)
            d.line((xs, y0, xs, y0 + 7), fill=(255, 60, 60))
    out = np.asarray(img)
    timing["draw_s"] += time.perf_counter() - t0
    return out


wanted = {int(x) for x in args.png.split(",") if x}
DEFAULT_NAMES = {"A": "example_raw_rigid", "B": "example_preproc", "C": "example_side_by_side"}
name = args.name or DEFAULT_NAMES[args.video]


def frames():
    g = 0
    for q in runs:
        for a, b in stream(q):
            f = frame(g, a, b, q)
            if g in wanted:
                Image.fromarray(f).save(OUT / f"{name}_t{g:04d}.png")
            g += 1
            yield f


child_peak = [0.0]
stop = threading.Event()


def watch_children():
    """Peak resident memory of the ffmpeg child, from /proc (ru_maxrss of a child starts at the
    parent's size)."""
    me = str(os.getpid())
    while not stop.wait(0.1):
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                txt = Path(f"/proc/{pid}/status").read_text()
            except OSError:
                continue
            f = dict(ln.split(":", 1) for ln in txt.splitlines() if ":" in ln)
            if f.get("PPid", "").strip() == me and "ffmpeg" in f.get("Name", "") and "VmHWM" in f:
                child_peak[0] = max(child_peak[0], int(f["VmHWM"].split()[0]) / 1024)


threading.Thread(target=watch_children, daemon=True).start()
t_enc = time.perf_counter()
if args.no_video:
    n = sum(1 for _ in frames())
    size = 0
else:
    path = OUT / f"{name}.mp4"
    if args.keyint is None:
        n = write_mp4(path, frames(), fps=args.fps, crf=args.crf)
    else:
        n = write_mp4_keyint(path, frames(), fps=args.fps, crf=args.crf, keyint=args.keyint)
    size = path.stat().st_size
stop.set()
total = time.perf_counter() - T0
rep = dict(
    video=args.video,
    name=name,
    fps=args.fps,
    scale=args.scale,
    crf=args.crf,
    frames=n,
    width=W,
    height=H,
    bytes=size,
    total_s=round(total, 2),
    frames_and_encode_s=round(time.perf_counter() - t_enc, 2),
    **{k: round(v, 2) for k, v in timing.items()},
    maxrss_python_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    peak_ffmpeg_mb=round(child_peak[0], 1),
    cpu_python_s=round(sum(os.times()[:2]), 2),
    cpu_children_s=round(sum(os.times()[2:4]), 2),
    contours=not args.no_contours,
    one_core=args.one_core,
    slices=[int(s) for s in slices],
    window_hi=[round(q["hi"], 1) for q in runs],
)
if args.keyint is not None:
    rep["keyint"] = args.keyint
print(json.dumps(rep))
with open(OUT / "timings.jsonl", "a") as f:
    f.write(json.dumps(rep) + "\n")
