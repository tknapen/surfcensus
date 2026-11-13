"""Step 1: prove the direction of the transforms on one subject. Numbers only; the pictures are
made by overlays.py.

usage: verify_geometry.py DATA FS_SUBJECT_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from geom import (
    read_itk_ras,
    resample,
    sample,
    surfaces_in_t1w,
    vertex_normals,
)

if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
    sys.exit(2)
D, FS = Path(sys.argv[1]), Path(sys.argv[2])
np.set_printoptions(precision=4, suppress=True)
res = {}
SHIFTS = (-3, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 3)


def corr(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def shifted(axis, mm):
    sh = np.eye(4)
    sh[axis, 3] = mm
    return sh


for run in (1, 2):
    raw = nib.load(D / f"raw/run-{run}_bold.nii.gz")
    data = np.asarray(raw.dataobj, dtype=np.float32)
    mean_raw = data.mean(axis=3)
    ref_b = nib.load(D / f"deriv/run-{run}_boldref.nii.gz")
    ref_b_data = np.asarray(ref_b.dataobj, dtype=np.float32)
    ref_t = nib.load(D / f"deriv/run-{run}_space-T1w_boldref.nii.gz")
    mask_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz")
    mask = np.asarray(mask_img.dataobj) > 0
    target = np.asarray(ref_t.dataobj, dtype=np.float32)
    s2t = read_itk_ras(D / f"deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt")
    t2s = read_itk_ras(D / f"deriv/run-{run}_from-T1w_to-scanner_mode-image_xfm.txt")
    print(f"run {run}: from-scanner_to-T1w as RAS point map (T1w -> scanner)\n{s2t}")
    print(
        "  product with the from-T1w_to-scanner file (should be identity): max |dev| =",
        float(np.abs(s2t @ t2s - np.eye(4)).max()),
    )
    r = {}
    for label, m in [
        ("file as fixed->moving (used)", s2t),
        ("inverse (wrong direction)", np.linalg.inv(s2t)),
        ("no transform", np.eye(4)),
    ]:
        a = resample(mean_raw, raw.affine, target.shape, ref_t.affine, m)
        b = resample(ref_b_data, ref_b.affine, target.shape, ref_t.affine, m)
        r[label] = dict(
            raw_mean_all=corr(a, target),
            raw_mean_mask=corr(a, target, mask),
            boldref_all=corr(b, target),
            boldref_mask=corr(b, target, mask),
        )
        print(
            f"  {label:32s} raw mean vs space-T1w_boldref:"
            f" r = {r[label]['raw_mean_all']:.4f} (all voxels),"
            f" {r[label]['raw_mean_mask']:.4f} (brain mask);"
            f" BOLD-space boldref resampled:"
            f" r = {r[label]['boldref_all']:.4f}, {r[label]['boldref_mask']:.4f}"
        )
    # sensitivity of the correlation to a shift: how sharp is the optimum
    for d in (1.0, 3.0):
        worst = []
        for ax in range(3):
            for s in (-d, d):
                m = s2t @ shifted(ax, s)
                b = resample(ref_b_data, ref_b.affine, target.shape, ref_t.affine, m)
                worst.append(corr(b, target, mask))
        print(
            f"  boldref resampled with an extra shift of {d} mm along one axis:"
            f" r (mask) = {min(worst):.4f} .. {max(worst):.4f}"
        )
        r[f"shift_{d}mm"] = [min(worst), max(worst)]
    # intensity scale of raw against preprocessed
    pre = nib.load(D / f"deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz")
    pm = np.asarray(pre.dataobj[..., :20], dtype=np.float32).mean(axis=3)
    a = resample(data[..., :20].mean(axis=3), raw.affine, target.shape, ref_t.affine, s2t)
    print(
        f"  first 20 volumes: median in mask raw-rigid {np.median(a[mask]):.1f},"
        f" preprocessed {np.median(pm[mask]):.1f}; r = {corr(a, pm, mask):.4f}"
    )
    r["scale"] = [float(np.median(a[mask])), float(np.median(pm[mask]))]
    r["first20_mask"] = corr(a, pm, mask)
    r["first20_all"] = corr(a, pm)
    # whole series: temporal mean of the raw series under the one transform against the mean
    # of the preprocessed series
    pmean = np.asarray(pre.dataobj, dtype=np.float32).mean(axis=3)
    amean = resample(mean_raw, raw.affine, target.shape, ref_t.affine, s2t)
    r["mean_mask"] = corr(amean, pmean, mask)
    r["mean_all"] = corr(amean, pmean)
    print(
        f"  temporal means of all 146 volumes: r = {r['mean_all']:.4f} (all),"
        f" {r['mean_mask']:.4f} (mask); space-T1w_boldref vs preprocessed mean:"
        f" r = {corr(target, pmean, mask):.4f} (mask)"
    )
    r["boldref_vs_preproc_mean_mask"] = corr(target, pmean, mask)
    prof = {}
    for ax, lab in enumerate("xyz"):
        row = []
        for sft in SHIFTS:
            m = s2t @ shifted(ax, sft)
            a = resample(mean_raw, raw.affine, target.shape, ref_t.affine, m)
            row.append(corr(a, pmean, mask))
        prof[lab] = row
        cells = " ".join(f"{sft:+.2f}:{c:.4f}" for sft, c in zip(SHIFTS, row, strict=True))
        print("    extra shift along", lab, cells)
    r["shift_profile_mean"] = prof
    res[f"run{run}"] = r

# surfaces against the T1w and the EPI: contrast across the white and the pial surface
surfs, m = surfaces_in_t1w(FS, D / "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt")
print("tkr RAS -> T1w world:\n", m)
alt = surfaces_in_t1w(FS, D / "deriv/anat_from-fsnative_to-T1w_mode-image_xfm.txt")[0]
worst = max(np.linalg.norm(surfs[k][0] - alt[k][0], axis=1).max() for k in surfs)
print(
    "vertex displacement if the fsnative transform were used the other way round:"
    f" max {worst:.3f} mm"
)
t1w = nib.load(D / "deriv/anat_desc-preproc_T1w.nii.gz")
t1 = np.asarray(t1w.dataobj, dtype=np.float32)
NO_SHIFT = np.zeros(3)


def contrast(vol, aff, shift=NO_SHIFT, d=1.0, name="white"):
    """Median over vertices of (inside - outside) / mean, d mm either side along the normal;
    and the fraction of vertices where it is positive."""
    out = []
    for hemi in ("lh", "rh"):
        v, f = surfs[hemi, name]
        n = vertex_normals(v, f)
        a = sample(vol, aff, v + shift - d * n)
        b = sample(vol, aff, v + shift + d * n)
        ok = np.isfinite(a) & np.isfinite(b) & (a + b > 0)
        out.append(200 * (a[ok] - b[ok]) / (a[ok] + b[ok]))
    c = np.concatenate(out)
    return float(np.median(c)), float((c > 0).mean())


def axis_shift(axis, mm):
    sh = np.zeros(3)
    sh[axis] = mm
    return sh


med, pos = contrast(t1, t1w.affine)
print(
    "T1w, across the white surface (1 mm in minus 1 mm out, percent of mean):"
    f" median {med:.1f}, fraction positive {pos:.3f}"
)
med, pos = contrast(t1, t1w.affine, name="pial")
print(f"T1w, across the pial surface: median {med:.1f}, fraction positive {pos:.3f}")
res["t1w_white"] = contrast(t1, t1w.affine)
res["t1w_pial"] = contrast(t1, t1w.affine, name="pial")
print("T1w white-surface contrast with the surfaces shifted (mm):")
T1W_SHIFTS = (-4, -2, -1, 0, 1, 2, 4)
for ax, lab in enumerate("xyz"):
    row = [contrast(t1, t1w.affine, axis_shift(ax, s))[0] for s in T1W_SHIFTS]
    print("  ", lab, " ".join(f"{s:+d}:{c:6.2f}" for s, c in zip(T1W_SHIFTS, row, strict=True)))
    res[f"t1w_shift_{lab}"] = row
EPI_SHIFTS = (-6, -3, 0, 3, 6)
for run in (1, 2):
    ref_t = nib.load(D / f"deriv/run-{run}_space-T1w_boldref.nii.gz")
    e = np.asarray(ref_t.dataobj, dtype=np.float32)
    # bbregister's contrast: gray (outside white) brighter than white in T2*: sample 2 mm out,
    # 2 mm in
    med, pos = contrast(e, ref_t.affine, d=2.0)
    print(
        f"EPI run {run} (space-T1w_boldref), across the white surface, 2 mm"
        f" (in - out, % of mean): median {med:.2f}, fraction negative {1 - pos:.3f}"
    )
    for ax, lab in enumerate("xyz"):
        row = [contrast(e, ref_t.affine, axis_shift(ax, s), d=2.0)[0] for s in EPI_SHIFTS]
        print("  ", lab, " ".join(f"{s:+d}:{c:6.2f}" for s, c in zip(EPI_SHIFTS, row, strict=True)))
        res[f"epi{run}_shift_{lab}"] = row
Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
