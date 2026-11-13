"""One blind trial of whether a coregistration error can be seen: four rows of the same raw
volume of a minimal-layout run in T1w space with contours, in shuffled order. One row uses the
stored coregistration; the others are shifted by 3 mm along y, by -3 mm along z and by 2 mm
along x. The key is written beside the picture, or into KEY_DIR, so that whoever is shown
the picture's folder does not find the answer in it.

usage: blind_shift.py DATA OUT_DIR SEED [--volume 60] [--key-dir KEY_DIR]
       (DATA as fetch_min.py wrote it; writes OUT_DIR/blind_shift.png and blind_shift_key.txt;
        both directories are made if they are not there)

The feasibility study ran this once (critique of the science). It is the seed of the
shift-detection test that README.md lists among the next steps, not that test.
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))  # still_min, also under -I
sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
import epiview
import geom
from still_min import display_grid, gifti_surfaces

CONDITIONS = [("stored", (0, 0, 0)), ("y+3", (0, 3, 0)), ("z-3", (0, 0, -3)), ("x+2", (2, 0, 0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("seed", type=int)
    ap.add_argument("--volume", type=int, default=60)
    ap.add_argument("--key-dir", default=None)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    key_dir = Path(args.key_dir) if args.key_dir else out
    out.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    surf = gifti_surfaces(data)
    raw = nib.load(data / "raw_bold.nii.gz")
    shape, aff, c = display_grid(surf)
    coreg = geom.read_itk_ras(data / "coreg_xfm.txt")
    x = raw.dataobj[..., args.volume].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(CONDITIONS))
    rows = []
    key = []
    P = epiview.Panels(shape, aff, 3.0, (c[0] - 4, c[1] - 6, c[2] + 3))
    ov = P.contours(surf, 1.0)
    for i in order:
        name, sh = CONDITIONS[i]
        m = coreg.copy()
        m[:3, 3] += sh
        e = geom.resample(x, raw.affine, shape, aff, m, order=3)
        strip = P.strip(e, 0, np.percentile(e[e > 0], 99.5))
        rows.append(np.asarray(Image.fromarray(epiview.compose(strip, ov))))
        key.append(name)
    Image.fromarray(np.concatenate(rows, 0)).save(out / "blind_shift.png")
    (key_dir / "blind_shift_key.txt").write_text("\n".join(key) + "\n")


if __name__ == "__main__":
    main()
