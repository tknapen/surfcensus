"""How long is an experiment? The header of every T1w-space preprocessed BOLD series of the
first subject of the named datasets: a ranged GET of the first 4 kB of each file, anonymous, one
request at a time. Prints matrix, voxel size, TR and the number of volumes; no names.

usage: headers.py INDEX [INDEX ...]

INDEX is the position of a dataset in the key-ordered list of prefixes under fmriprep/. The
feasibility study read indices 0, 30, 120 and 154 of the 308 prefixes listed on 2026-11-13
(four derivatives that hold the series), and at most the first 8 runs of each.
"""

import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import DERIV, get, list_prefix


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    _, prefixes = list_prefix(DERIV, "fmriprep/", "/")
    datasets = [x for x in prefixes if re.fullmatch(r"fmriprep/ds\d{6}-fmriprep/", x)]
    for idx in map(int, sys.argv[1:]):
        _, pres = list_prefix(DERIV, datasets[idx], "/")
        sub = sorted(p for p in pres if "/sub-" in p)[0]
        objs, _ = list_prefix(DERIV, sub)
        bolds = [(k, s) for k, s, _, _ in objs if k.endswith("space-T1w_desc-preproc_bold.nii.gz")]
        tot_vol = 0
        tot_s = 0.0
        shapes = set()
        for k, _ in bolds[:8]:
            raw = get(f"{DERIV}/{k}", headers={"Range": "bytes=0-4095"}).content
            h = zlib.decompressobj(31).decompress(raw)[:348]
            dim = struct.unpack("<8h", h[40:56])
            pix = struct.unpack("<8f", h[76:108])
            shapes.add((dim[1:4], tuple(round(p, 2) for p in pix[1:4]), round(pix[4], 3)))
            tot_vol += dim[4]
            tot_s += dim[4] * pix[4]
        print(
            f"index {idx}: {len(bolds)} T1w-space runs; first {min(len(bolds), 8)} read:"
            f" {tot_vol} volumes, {tot_s / 60:.1f} min of scanning;"
            f" (matrix, voxel mm, TR s): {sorted(shapes)};"
            f" mean MB per run {sum(s for _, s in bolds) / len(bolds) / 1e6:.0f}"
        )


if __name__ == "__main__":
    main()
