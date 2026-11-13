"""What still images would cost instead of video.

usage: tile_sizes.py DATA [--panels] [--fs-dir FS_SUBJECT_DIR] [--one-core]

Always: one 8-bit mosaic of all axial slices per volume (native voxels) of run 1 of the
preprocessed series, in several encodings. This is the part the feasibility study ran.

With --panels (added for the playback question, README "Fast playback and scrubbing"): one
image per volume and panel, as the surface rater app plays its slice tiles from a frame table.
The three panels are those of the videos (same slices, same window, 2 px per mm), for both
streams and both runs. The lossy tiles are written by surfcensus.render.encode.write_gray_webp
at quality 80, the setting of the surface tiles, into a temporary directory that is removed
afterwards. With --fs-dir the contour layer, which does not change over time, is written once
per panel as lossless RGBA.
"""

import argparse
import gzip
import io
import os
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--panels", action="store_true")
ap.add_argument("--fs-dir", default=None)
ap.add_argument("--one-core", action="store_true")
args = ap.parse_args()
if args.one_core:
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

# after the thread limits, which numpy's libraries read when they load; hence E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

D = Path(args.data)
ENCODINGS = (
    ("webp lossless", dict(format="WEBP", lossless=True)),
    ("webp q90", dict(format="WEBP", quality=90)),
    ("webp q80", dict(format="WEBP", quality=80)),
    ("png", dict(format="PNG", optimize=True)),
)


def encoded_size(image, **params):
    b = io.BytesIO()
    image.save(b, **params)
    return b.tell()


def mosaic(v):  # 6 x 6 axial slices
    rows = [
        np.concatenate([v[:, :, r * 6 + c].T[::-1] for c in range(6)], axis=1) for r in range(6)
    ]
    return np.concatenate(rows, axis=0)


def mosaics():
    pre = nib.load(D / "deriv/run-1_space-T1w_desc-preproc_bold.nii.gz")
    B = np.asarray(pre.dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(D / "deriv/run-1_space-T1w_desc-brain_mask.nii.gz").dataobj) > 0
    hi = float(np.percentile(B[..., ::10][mask], 99.5))
    u8 = np.clip(B / hi * 255 + 0.5, 0, 255).astype(np.uint8)  # (49, 62, 36, 146)
    nt = u8.shape[3]
    sizes = {k: [] for k, _ in ENCODINGS}
    t0 = time.perf_counter()
    for t in range(0, nt, 5):
        im = Image.fromarray(mosaic(u8[..., t]))
        for k, kw in ENCODINGS:
            sizes[k].append(encoded_size(im, **kw))
    n = len(sizes["png"])
    print("mosaic", mosaic(u8[..., 0]).shape, "pixels; per volume, mean bytes over", n, "volumes:")
    for k, v in sizes.items():
        per_run = np.mean(v) * nt / 1e6
        print(f"  {k:14s} {np.mean(v) / 1e3:6.1f} kB  -> {per_run:5.1f} MB per run of {nt}")
    raw = u8.tobytes()
    print(
        f"  whole run as one uint8 array: {len(raw) / 1e6:.1f} MB raw,"
        f" {len(gzip.compress(raw, 6)) / 1e6:.1f} MB gzip;"
        f" time for the encodings above {time.perf_counter() - t0:.1f} s"
    )


def panel_tiles():
    sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
    from epiview import Panels
    from geom import read_itk_ras, resample, surfaces_in_t1w

    from surfcensus.render.encode import write_gray_webp, write_rgba_webp

    runs = (1, 2)
    masks = [
        np.asarray(nib.load(D / f"deriv/run-{r}_space-T1w_desc-brain_mask.nii.gz").dataobj) > 0
        for r in runs
    ]
    ref = nib.load(D / "deriv/run-1_space-T1w_boldref.nii.gz")
    com = np.array(np.nonzero(np.any(masks, axis=0))).mean(axis=1)
    # the slices, scale and window of make_videos.py
    P = Panels(ref.shape, ref.affine, 2.0, np.round(com + np.array([4, 0, 1])).astype(int))
    q80 = {}  # (stream, run) -> bytes of the tiles written by the package's writer
    other = {"webp lossless": 0, "webp q90": 0}  # every 5th volume, encoded in memory
    n_other = n_vol = 0
    t_q80 = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for r, mask in zip(runs, masks, strict=True):
            raw = nib.load(D / f"raw/run-{r}_bold.nii.gz")
            pre = nib.load(D / f"deriv/run-{r}_space-T1w_desc-preproc_bold.nii.gz")
            m = read_itk_ras(D / f"deriv/run-{r}_from-scanner_to-T1w_mode-image_xfm.txt")
            A = np.asarray(raw.dataobj, dtype=np.float32)
            B = np.asarray(pre.dataobj, dtype=np.float32)
            hi = float(np.percentile(B[..., ::10][mask], 99.5))
            k = 255.0 / hi
            nt = B.shape[3]
            n_vol += nt
            for t in range(nt):
                a = resample(A[..., t], raw.affine, B.shape[:3], pre.affine, m, order=3)
                for stream, vol in (("raw_rigid", a), ("preproc", B[..., t])):
                    for p in P.panels:
                        # the windowing of Panels.strip, one panel at a time
                        g = np.clip(p.cut(vol).astype(np.float32) * k + 0.5, 0, 255)
                        g = g.astype(np.uint8)
                        path = tmp / f"{stream}_run-{r}_{p.name}_{t:04d}.webp"
                        t0 = time.perf_counter()
                        write_gray_webp(path, g, 80)
                        t_q80 += time.perf_counter() - t0
                        q80[stream, r] = q80.get((stream, r), 0) + path.stat().st_size
                        if t % 5 == 0:
                            im = Image.fromarray(g)
                            other["webp lossless"] += encoded_size(im, format="WEBP", lossless=True)
                            other["webp q90"] += encoded_size(im, format="WEBP", quality=90)
                            n_other += 1
        n_tiles = len(list(tmp.glob("*.webp")))
        contour_bytes = None
        if args.fs_dir:
            xfm = D / "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt"
            surfs, _ = surfaces_in_t1w(args.fs_dir, xfm)
            overlay = P.contours(surfs, 1.0)
            contour_bytes, x0 = 0, 0
            for p in P.panels:
                y0 = (P.height - p.height) // 2
                path = tmp / f"contours_{p.name}.webp"
                write_rgba_webp(path, overlay[y0 : y0 + p.height, x0 : x0 + p.width])
                contour_bytes += path.stat().st_size
                x0 += p.width
    sizes = ", ".join(f"{p.name} {p.width} x {p.height}" for p in P.panels)
    total = sum(q80.values())
    print(f"panel tiles ({sizes} px), one per volume, panel and stream; {n_vol} volumes, 2 runs:")
    print(
        f"  webp q80, written by surfcensus.render.encode.write_gray_webp: {n_tiles} tiles,"
        f" {total / 1e6:.2f} MB, mean {total / n_tiles / 1e3:.2f} kB per tile,"
        f" {t_q80:.1f} s to encode and write"
    )
    for (stream, r), b in sorted(q80.items()):
        print(f"    {stream:9s} run {r}: {b / 1e6:.2f} MB")
    for k, b in other.items():
        est = b / n_other * n_tiles
        print(
            f"  {k}, from every 5th volume ({n_other} tiles): mean {b / n_other / 1e3:.2f} kB"
            f" per tile -> {est / 1e6:.1f} MB for the {n_tiles} tiles"
        )
    if contour_bytes is not None:
        print(
            "  contour layer, lossless RGBA, once per panel for the whole experiment:"
            f" {contour_bytes / 1e3:.1f} kB for 3 panels"
        )


mosaics()
if args.panels:
    panel_tiles()
