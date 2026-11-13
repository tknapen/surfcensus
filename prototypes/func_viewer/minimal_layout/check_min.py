"""Can the stored fits of a 'minimal' fMRIPrep derivative (23.2.0) be applied outside fMRIPrep?
One run of one subject. Uses the conventions of pipeline/geom.py (an ITK text transform is an
LPS point map fixed -> moving).

usage: check_min.py DATA [--save NPY]
       (DATA as fetch_min.py wrote it; --save keeps the reference image in T1w space)
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))  # also under -I
import geom

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--save", default=None)
args = ap.parse_args()
D = Path(args.data)
FLIP = geom.FLIP


def read_many(path):
    """Every transform of an ITK text file that holds one per volume, as RAS point maps."""
    out, cur = [], {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("Parameters:"):
            cur["p"] = np.array(line.split(":")[1].split(), float)
        elif line.startswith("FixedParameters:"):
            c = np.array(line.split(":")[1].split(), float)
            p = cur["p"]
            m = np.eye(4)
            m[:3, :3] = p[:9].reshape(3, 3)
            m[:3, 3] = p[9:] + c - m[:3, :3] @ c
            out.append(FLIP @ m @ FLIP)
    return out


def rotation_deg(m):
    return np.degrees(np.arccos(np.clip((np.trace(m[:3, :3]) - 1) / 2, -1, 1)))


def r(a, b, m):
    return np.corrcoef(a[m], b[m])[0, 1]


raw = nib.load(D / "raw_bold.nii.gz")
hb = nib.load(D / "hmc_boldref.nii.gz")
cb = nib.load(D / "coreg_boldref.nii.gz")
t1 = nib.load(D / "T1w.nii.gz")
mask = nib.load(D / "T1w_mask.nii.gz")
dseg = nib.load(D / "dseg.nii.gz")
print("raw", raw.shape, raw.header.get_zooms(), nib.aff2axcodes(raw.affine), raw.get_data_dtype())
print(
    "hmc_boldref",
    hb.shape,
    nib.aff2axcodes(hb.affine),
    "same grid as raw:",
    np.allclose(hb.affine, raw.affine, atol=1e-3),
    "max |aff diff|",
    np.abs(hb.affine - raw.affine).max(),
)
print(
    "coreg_boldref",
    cb.shape,
    "same grid as hmc_boldref:",
    np.allclose(cb.affine, hb.affine, atol=1e-3),
)
print("T1w", t1.shape, t1.header.get_zooms(), nib.aff2axcodes(t1.affine))
hmc = read_many(D / "hmc_xfm.txt")
coreg = geom.read_itk_ras(D / "coreg_xfm.txt")
print("hmc transforms:", len(hmc), "raw volumes:", raw.shape[3])
R = coreg[:3, :3]
print(
    f"coreg: max|R^T R - I| {np.abs(R.T @ R - np.eye(3)).max():.2e},"
    f" rot deg {np.degrees(np.arccos((np.trace(R) - 1) / 2)):.2f},"
    f" |t| {np.linalg.norm(coreg[:3, 3]):.2f} mm"
)
x = raw.get_fdata(dtype=np.float32)
hbd = hb.get_fdata(dtype=np.float32)
cbd = cb.get_fdata(dtype=np.float32)
print(
    f"corr(hmc_boldref, coreg_boldref) {np.corrcoef(hbd.ravel(), cbd.ravel())[0, 1]:.4f};"
    f" corr(hmc_boldref, raw mean) {np.corrcoef(hbd.ravel(), x.mean(3).ravel())[0, 1]:.4f}"
)
# head-motion parameters from the transforms
tr = np.array([m[:3, 3] for m in hmc])
ang = np.array([rotation_deg(m) for m in hmc])
print(f"hmc translations: max |t| {np.abs(tr).max():.2f} mm, max rotation {ang.max():.2f} deg")
# (c) head motion: each raw volume against hmc_boldref, as is / transform as stored / inverse
bm = hbd > 0.3 * np.percentile(hbd, 99)
bm = ndimage.binary_erosion(bm, iterations=1)
t0 = time.time()
res = {"none": [], "stored": [], "inverse": []}
corr = np.empty_like(x)
for t in range(x.shape[3]):
    v = x[..., t]
    res["none"].append(r(v, hbd, bm))
    a = geom.resample(v, raw.affine, hb.shape, hb.affine, hmc[t], order=3)
    corr[..., t] = a
    res["stored"].append(r(a, hbd, bm))
    inv = geom.resample(v, raw.affine, hb.shape, hb.affine, np.linalg.inv(hmc[t]), order=3)
    res["inverse"].append(r(inv, hbd, bm))
print(f"hmc resampling of {x.shape[3]} volumes x2: {time.time() - t0:.1f} s")
for k, v in res.items():
    print(
        f"  r(volume, hmc_boldref) in brain, {k:<8s} median {np.median(v):.5f}"
        f"  min {np.min(v):.5f}  mean {np.mean(v):.5f}"
    )
n = np.array(res["none"])
s = np.array(res["stored"])
i = np.array(res["inverse"])
print(
    f"  volumes where 'stored' beats 'none': {(s > n).sum()} of {len(n)};"
    f" 'inverse' beats 'none': {(i > n).sum()}"
)
size = np.linalg.norm(tr, axis=1)
big = np.argsort(-size)[:10]
print(
    f"  10 volumes with largest translation ({size[big].min():.2f}-{size[big].max():.2f} mm):"
    f" none {n[big].mean():.5f} stored {s[big].mean():.5f} inverse {i[big].mean():.5f}"
)
edge = ndimage.binary_dilation(bm, iterations=2) & ~ndimage.binary_erosion(bm, iterations=2)
inner = bm & ~edge


def rel_sd(series, where):
    return np.median(series.std(3)[where] / series.mean(3)[where])


print(
    f"  temporal SD / mean at the brain edge: raw {rel_sd(x, edge):.4f},"
    f" after stored hmc {rel_sd(corr, edge):.4f};"
    f" inside: raw {rel_sd(x, inner):.4f}, after {rel_sd(corr, inner):.4f}"
)
print(
    f"  corr(mean of hmc-corrected series, hmc_boldref) {r(corr.mean(3), hbd, bm):.5f};"
    f" corr(raw mean, hmc_boldref) {r(x.mean(3), hbd, bm):.5f}"
)
# (d) surfaces: fMRIPrep's GIFTI white surface against the T1w, and FreeSurfer's lh.white
# brought over
t1d = t1.get_fdata(dtype=np.float32)
g = nib.load(D / "L_white.surf.gii")
gv, gf = g.darrays[0].data.astype(float), g.darrays[1].data.astype(np.intp)


def contrast(vol, aff, v, f, d, shift=(0, 0, 0)):
    """(inside - outside) over their mean, percent, d mm either side of the surface along the
    normal; and the fraction of vertices where inside is the brighter."""
    nrm = geom.vertex_normals(v, f)
    v = v + np.array(shift)
    a = geom.sample(vol, aff, v - d * nrm)
    b = geom.sample(vol, aff, v + d * nrm)
    ok = np.isfinite(a) & np.isfinite(b)
    return 100 * np.mean(a[ok] - b[ok]) / np.mean((a[ok] + b[ok]) / 2), np.mean(a[ok] > b[ok])


c, pos = contrast(t1d, t1.affine, gv, gf, 1)
print(
    f"GIFTI L white on T1w: contrast (1 mm in - 1 mm out) {c:.1f}%, {pos:.3f} of vertices positive"
)
for sh in [(1, 0, 0), (0, 2, 0), (0, 0, 4)]:
    c, pos = contrast(t1d, t1.affine, gv, gf, 1, sh)
    print("   shifted", sh, f"{c:.1f}% {pos:.3f}")
fv, ff = nib.freesurfer.read_geometry(str(D / "fs_lh.white"))
T1 = nib.load(D / "fs_T1.mgz")
tkr2scan = T1.affine @ np.linalg.inv(T1.header.get_vox2ras_tkr())
A = geom.read_itk_ras(D / "T1w_to_fsnative.txt")
Bm = geom.read_itk_ras(D / "fsnative_to_T1w.txt")
print(
    f"fsnative<->T1w: translation {np.round(Bm[:3, 3], 3)} mm,"
    f" product deviates from identity by {np.abs(A @ Bm - np.eye(4)).max():.1e}"
)
for name, m in [
    ("tkr->scanner only", tkr2scan),
    ("then point map of T1w_to_fsnative file", A @ tkr2scan),
    ("then point map of fsnative_to_T1w file", Bm @ tkr2scan),
]:
    v = fv @ m[:3, :3].T + m[:3, 3]
    same = len(v) == len(gv)
    dist = np.abs(v - gv).max() if same else np.nan
    c = contrast(t1d, t1.affine, v, ff.astype(np.intp), 1)[0]
    print(
        f"  FS lh.white, {name:<40s}: max dist to GIFTI vertex {dist:.4f} mm"
        f" (same count: {same}); contrast {c:.1f}%"
    )
# (b) coregistration: coreg_boldref onto a 2 mm grid in T1w space; white-surface contrast on
# the EPI and overlap with the T1w brain mask
z = np.array(t1.header.get_zooms()[:3])
f2 = 2.0 / z
shape2 = tuple(np.ceil(np.array(t1.shape) / f2).astype(int))
aff2 = t1.affine.copy()
aff2[:3, :3] = t1.affine[:3, :3] * f2
aff2[:3, 3] = t1.affine[:3, 3] + t1.affine[:3, :3] @ ((f2 - 1) / 2)
m2 = geom.resample(mask.get_fdata(dtype=np.float32), mask.affine, shape2, aff2, order=1) > 0.5
ds2 = geom.resample((dseg.get_fdata(dtype=np.float32)), dseg.affine, shape2, aff2, order=0)
print("dseg labels:", np.unique(ds2)[:6])


def score(m, label):
    e = geom.resample(cbd, cb.affine, shape2, aff2, m, order=3)
    thr = 0.3 * np.percentile(cbd, 99)
    em = e > thr
    dice = 2 * (em & m2).sum() / max(em.sum() + m2.sum(), 1)
    c = contrast(e, aff2, gv, gf, 2)
    # correlation ratio: variance of EPI explained by tissue class inside the mask
    inside = m2 & (e > 0)
    lab = ds2[inside]
    val = e[inside]
    tot = val.var()
    eta = np.nan
    if len(val):
        within = sum(val[lab == k].var() * (lab == k).sum() for k in np.unique(lab))
        eta = 1 - within / (tot * len(val))
    print(
        f"  {label:<34s} nonzero {(e > 0).sum():7d}  Dice(EPI>thr, T1w mask) {dice:.3f}"
        f"  white-surface contrast {c[0]:.1f}% ({1 - c[1]:.2f} gray>white)"
        f"  eta2(tissue) {eta:.3f}"
    )
    return e


print("coreg_boldref -> T1w space:")
e = score(coreg, "as stored (T1w -> boldref)")
score(np.linalg.inv(coreg), "inverse")
score(np.eye(4), "identity (headers only)")
for sh in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0), (0, 3, 0), (0, 0, 3)]:
    m = coreg.copy()
    m[:3, 3] += sh
    score(m, f"as stored + shift {sh} mm")
# (e) whole chain for one frame: raw volume -> hmc -> coreg in one interpolation, vs boldref
# resampled. The point map goes T1w world -> boldref world (coreg) -> orig world (hmc[t]).
t = int(big[0])
v1 = geom.resample(x[..., t], raw.affine, shape2, aff2, hmc[t] @ coreg, order=3)
v0 = geom.resample(x[..., t], raw.affine, shape2, aff2, coreg, order=3)
print(
    f"volume {t} (largest motion): r with boldref in T1w space, rigid only {r(v0, e, m2):.5f},"
    f" hmc+coreg in one step {r(v1, e, m2):.5f}"
)
if args.save:
    np.save(args.save, e)
