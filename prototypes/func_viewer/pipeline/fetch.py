"""Fetch the files of the worked example, anonymously and read-only, at most 4 requests at a time.

usage: fetch.py OUT

The worked example is ds000102 sub-01 (two runs of the flanker task): the raw series from the
OpenNeuro bucket, and from the OpenNeuroDerivatives bucket the T1w-space preprocessed series,
reference images, masks, confounds and transforms of fMRIPrep 21.0.2. 29 files, 133 MB. A file
that is already in OUT is not fetched again. OUT/fetch_log.json lists the files and their sizes.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MAX_REQUESTS = 4
D = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/sub-01/"
R = "https://s3.amazonaws.com/openneuro.org/ds000102/"
S = "sub-01"
PER_RUN = (
    "space-T1w_desc-preproc_bold.nii.gz",
    "space-T1w_desc-preproc_bold.json",
    "space-T1w_boldref.nii.gz",
    "space-T1w_desc-brain_mask.nii.gz",
    "boldref.nii.gz",
    "desc-brain_mask.nii.gz",
    "desc-confounds_timeseries.tsv",
    "from-scanner_to-T1w_mode-image_xfm.txt",
    "from-T1w_to-scanner_mode-image_xfm.txt",
)
ANAT = (
    "desc-preproc_T1w.nii.gz",
    "desc-brain_mask.nii.gz",
    "from-fsnative_to-T1w_mode-image_xfm.txt",
    "from-T1w_to-fsnative_mode-image_xfm.txt",
)


def file_list():
    """Local name (relative to OUT) -> URL."""
    files = {}
    for run in (1, 2):
        p = f"{S}_task-flanker_run-{run}_"
        files[f"raw/run-{run}_bold.nii.gz"] = R + f"{S}/func/{p}bold.nii.gz"
        files[f"raw/run-{run}_events.tsv"] = R + f"{S}/func/{p}events.tsv"
        for name in PER_RUN:
            files[f"deriv/run-{run}_{name}"] = D + f"func/{p}{name}"
    for name in ANAT:
        files[f"deriv/anat_{name}"] = D + f"anat/{S}_{name}"
    files["raw/task_bold.json"] = R + "task-flanker_bold.json"
    files["raw/dataset_description.json"] = R + "dataset_description.json"
    files["deriv/dataset_description.json"] = D.rsplit("sub-01/", 1)[0] + "dataset_description.json"
    return files


def get(out, item):
    local, url = item
    dst = out / local
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return local, dst.stat().st_size, 0.0, "cached"
    t = time.time()
    with (
        httpx.Client(timeout=120, headers={"User-Agent": "research-prototype"}) as c,
        c.stream("GET", url) as r,
    ):
        if r.status_code != 200:
            return local, 0, time.time() - t, f"HTTP {r.status_code}"
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
        tmp.rename(dst)
    return local, dst.stat().st_size, time.time() - t, "ok"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out = Path(sys.argv[1])
    files = file_list()
    t0 = time.time()
    with ThreadPoolExecutor(MAX_REQUESTS) as ex:
        res = list(ex.map(lambda item: get(out, item), files.items()))
    tot = sum(r[1] for r in res)
    for r in sorted(res):
        print(f"{r[1]:>10d} {r[2]:6.1f}s {r[3]:8s} {r[0]}")
    wall = time.time() - t0
    print(f"total {tot} bytes ({tot / 1e6:.1f} MB) in {wall:.1f} s wall, {len(res)} files")
    log = {"files": {r[0]: r[1] for r in res}, "total_bytes": tot, "wall_s": time.time() - t0}
    (out / "fetch_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
