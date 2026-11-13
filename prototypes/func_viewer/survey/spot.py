"""Independent spot check of the layout split: list func/ of the first subject of 12 datasets
drawn at random (seed 20261113) from those outside the survey's even sample of 30.

usage: spot.py OUT    (writes OUT/spot.txt)

Listing and dataset_description.json only, anonymous, at most 4 requests at a time (the
feasibility study used 6). OUT/spot.txt has one line per dataset with its prefix, so it holds
accessions and stays out of the repository. The feasibility study printed these lines; the
script was given OUT on 2026-11-13 so that running it without an argument sends no request
and puts no dataset name on the terminal.
"""

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import DERIV, MAX_REQUESTS, get, list_prefix, sample_indices

SEED = 20261113
N_PICK = 12


def one(prefix):
    try:
        v = get(f"{DERIV}/{prefix}dataset_description.json").json()
        ver = [
            g.get("Version")
            for g in v.get("GeneratedBy", [])
            if g.get("Name", "").lower() == "fmriprep"
        ]
    except Exception as e:
        ver = [repr(e)]
    _, subs = list_prefix(DERIV, prefix, "/")
    subs = [s for s in subs if re.search(r"/sub-[^/]+/$", s)]
    if not subs:
        return prefix, ver, None
    keys, _ = list_prefix(DERIV, subs[0])
    f = [(k, s) for k, s, _, _ in keys if "/func/" in k]

    def n(pat):
        return sum(1 for k, _ in f if re.search(pat, k))

    spaces = n(r"space-[A-Za-z0-9]+(_res-\d)?_desc-preproc_bold\.nii\.gz$")
    return (
        prefix,
        ver,
        dict(
            sub=subs[0].split("/")[-2],
            nsub=len(subs),
            func_keys=len(f),
            func_MB=round(sum(s for _, s in f) / 1e6, 1),
            T1w_preproc=n(r"space-T1w_desc-preproc_bold\.nii\.gz$"),
            bold_preproc=n(r"(?<!res-2)_desc-preproc_bold\.nii\.gz$") - spaces,
            confounds=n(r"confounds_timeseries\.tsv$"),
            scanner_to_T1w=n(r"from-scanner_to-T1w.*xfm\.txt$"),
            coreg_xfm=n(r"from-boldref_to-T1w.*xfm\.txt$"),
            hmc_xfm=n(r"from-orig_to-boldref.*xfm\.txt$"),
        ),
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # no argument: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    _, ds = list_prefix(DERIV, "fmriprep/", "/")
    lines = [f"{len(ds)} dataset prefixes"]
    sampled = set(sample_indices(len(ds)))
    rng = random.Random(SEED)
    pick = sorted(rng.sample([i for i in range(len(ds)) if i not in sampled], N_PICK))
    with ThreadPoolExecutor(MAX_REQUESTS) as ex:
        for prefix, ver, r in ex.map(one, [ds[i] for i in pick]):
            lines.append(f"{prefix} {ver} {json.dumps(r)}")
    (out_dir / "spot.txt").write_text("\n".join(lines) + "\n")
    print(len(ds), "dataset prefixes;", len(pick), "listed -> spot.txt")


if __name__ == "__main__":
    main()
