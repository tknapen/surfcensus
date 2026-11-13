"""Step 1, the pictures: surfaces on the T1w and on the EPI, and raw-rigid against fMRIPrep's
reference.

usage: overlays.py DATA FS_SUBJECT_DIR OUT_DIR
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from epiview import Panels, compose
from geom import read_itk_ras, resample, surfaces_in_t1w

if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
    sys.exit(2)
D, FS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
surfs, _ = surfaces_in_t1w(FS, D / "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt")


def save(arr, name):
    Image.fromarray(arr).save(OUT / name)
    print(name, arr.shape)


def world(affine, voxel):
    return affine[:3, :3] @ voxel + affine[:3, 3]


# T1w with contours, 2 px per mm, three positions
t1w = nib.load(D / "deriv/anat_desc-preproc_T1w.nii.gz")
t1 = np.asarray(t1w.dataobj, dtype=np.float32)
bm = np.asarray(nib.load(D / "deriv/anat_desc-brain_mask.nii.gz").dataobj) > 0
hi = np.percentile(t1[bm], 99.5)
com = np.array(np.nonzero(bm)).mean(axis=1)
print("T1w brain mask centre (voxel)", com, "world", world(t1w.affine, com))
for tag, off in (("a", (12, 0, 5)), ("b", (-30, -45, -25)), ("c", (38, 35, 30))):
    sl = np.round(com + np.array(off)).astype(int)
    P = Panels(t1.shape, t1w.affine, 2.0, sl)
    g = P.strip(t1, 0, hi)
    both = np.concatenate([compose(g), compose(g, P.contours(surfs, 1.25))], axis=0)
    save(np.asarray(P.letters(Image.fromarray(both))), f"check_t1w_{tag}.png")

for run in (1, 2):
    ref_t = nib.load(D / f"deriv/run-{run}_space-T1w_boldref.nii.gz")
    e = np.asarray(ref_t.dataobj, dtype=np.float32)
    mask_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz")
    mask = np.asarray(mask_img.dataobj) > 0
    raw = nib.load(D / f"raw/run-{run}_bold.nii.gz")
    mean_raw = np.asarray(raw.dataobj, dtype=np.float32).mean(axis=3)
    s2t = read_itk_ras(D / f"deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt")
    a = resample(mean_raw, raw.affine, e.shape, ref_t.affine, s2t)
    pre_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz")
    pre = np.asarray(pre_img.dataobj, dtype=np.float32).mean(axis=3)
    comv = np.array(np.nonzero(mask)).mean(axis=1)
    print(
        f"run {run}: EPI mask centre (voxel)",
        comv,
        "world",
        world(ref_t.affine, comv),
        "mask voxels",
        int(mask.sum()),
    )
    for tag, off in (("a", (4, 0, 1)), ("b", (-9, -12, -6))):
        sl = np.round(comv + np.array(off)).astype(int)
        P = Panels(e.shape, ref_t.affine, 4.0, sl)
        ov = P.contours(surfs, 1.25)
        hi_e, hi_a = np.percentile(e[mask], 99.5), np.percentile(pre[mask], 99.5)
        rows = [
            compose(P.strip(e, 0, hi_e), ov),  # fMRIPrep's space-T1w boldref + contours
            compose(P.strip(a, 0, hi_a), ov),  # raw mean, one rigid transform + contours
            compose(P.strip(pre, 0, hi_a), ov),  # mean of the preprocessed series + contours
            # |raw-rigid mean - preprocessed mean| x 5
            compose(P.strip(np.abs(a - pre) * 5, 0, hi_a)),
        ]
        image = Image.fromarray(np.concatenate(rows, axis=0))
        save(np.asarray(P.letters(image)), f"check_epi_run{run}_{tag}.png")
    # wrong direction, for the record
    w = resample(mean_raw, raw.affine, e.shape, ref_t.affine, np.linalg.inv(s2t))
    print(f"run {run}: wrong direction leaves {int((w > 0).sum())} non-zero voxels of {w.size}")
