"""Two stills of one raw volume of a minimal-layout run in T1w space, with the derivative's own
white and pial surfaces as contours. Three rows each: the volume with the stored
coregistration, the same volume with the headers alone, and the T1w.

usage: still_min.py DATA OUT_DIR [--volume 60]
       (DATA as fetch_min.py wrote it; writes OUT_DIR/minimal_raw_rigid_a.png and _b.png;
        OUT_DIR is made if it is not there and should not be DATA)
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))  # also under -I
import epiview
import geom


def gifti_surfaces(data):
    """fMRIPrep's surfaces, which are in T1w space already, keyed like geom.surfaces_in_t1w."""
    surf = {}
    for h, hemi in (("lh", "L"), ("rh", "R")):
        for n in ("white", "pial"):
            g = nib.load(data / f"{hemi}_{n}.surf.gii")
            surf[h, n] = (g.darrays[0].data.astype(float), g.darrays[1].data.astype(np.intp))
    return surf


def display_grid(surf):
    """An axis-aligned grid of 3 x 3 x 3.5 mm around the surfaces: shape, affine, centre voxel."""
    allv = np.concatenate([surf[k][0] for k in surf])
    lo = allv.min(0) - 22
    hi = allv.max(0) + 14
    size = np.array([3.0, 3.0, 3.5])
    shape = np.ceil((hi - lo) / size).astype(int)
    aff = np.diag([*size, 1.0])
    aff[:3, 3] = lo + size / 2
    c = np.round((allv.mean(0) - aff[:3, 3]) / size).astype(int)
    return shape, aff, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--volume", type=int, default=60)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    surf = gifti_surfaces(data)
    t1 = nib.load(data / "T1w.nii.gz")
    raw = nib.load(data / "raw_bold.nii.gz")
    print("T1w affine\n", np.round(t1.affine, 3))
    shape, aff, c = display_grid(surf)
    coreg = geom.read_itk_ras(data / "coreg_xfm.txt")
    x = raw.dataobj[..., args.volume].astype(np.float32)
    e = geom.resample(x, raw.affine, shape, aff, coreg, order=3)
    e_id = geom.resample(x, raw.affine, shape, aff, None, order=3)
    t = geom.resample(t1.get_fdata(dtype=np.float32), t1.affine, shape, aff, None, order=1)
    hi_e = np.percentile(e[e > 0], 99.5)
    rows = []
    for sl in ((c[0] - 4, c[1] - 6, c[2] + 3), (c[0] + 6, c[1] + 8, c[2] - 6)):
        P = epiview.Panels(shape, aff, 3.0, sl)
        ov = P.contours(surf, 1.0)
        for vol, hi in ((e, hi_e), (e_id, hi_e), (t, np.percentile(t, 99.8))):
            image = Image.fromarray(epiview.compose(P.strip(vol, 0, hi), ov))
            rows.append(np.asarray(P.letters(image)))
    Image.fromarray(np.concatenate(rows[:3], 0)).save(out / "minimal_raw_rigid_a.png")
    Image.fromarray(np.concatenate(rows[3:], 0)).save(out / "minimal_raw_rigid_b.png")
    print(
        f"rows: raw volume {args.volume} with the stored coregistration; the same volume with"
        " headers only; T1w. size",
        rows[0].shape,
    )


if __name__ == "__main__":
    main()
