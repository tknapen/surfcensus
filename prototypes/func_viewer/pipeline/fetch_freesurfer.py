"""Fetch the FreeSurfer files of the worked example that the contours need: T1.mgz and the white
and pial surfaces of both hemispheres of ds000102 sub-01 (5 files, 27 MB), from
sourcedata/freesurfer/ of the fMRIPrep derivative. Anonymous, read-only, at most 4 requests at
a time. A file that is already there is not fetched again.

usage: fetch_freesurfer.py OUT    (writes OUT/sub-01/mri/T1.mgz and OUT/sub-01/surf/?h.*)

Not part of the feasibility study, which read these files from the pytest cache of the package's
network tests; added so that a fresh clone has a FreeSurfer subject directory to pass on.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MAX_REQUESTS = 4
BASE = (
    "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/"
    "sourcedata/freesurfer/sub-01/"
)
FILES = ("mri/T1.mgz", "surf/lh.white", "surf/lh.pial", "surf/rh.white", "surf/rh.pial")


def get(out, rel):
    dst = out / "sub-01" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return rel, dst.stat().st_size, "cached"
    with (
        httpx.Client(timeout=120, headers={"User-Agent": "research-prototype"}) as c,
        c.stream("GET", BASE + rel) as r,
    ):
        if r.status_code != 200:
            return rel, 0, f"HTTP {r.status_code}"
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
        tmp.rename(dst)
    return rel, dst.stat().st_size, "ok"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out = Path(sys.argv[1])
    t0 = time.time()
    with ThreadPoolExecutor(MAX_REQUESTS) as ex:
        res = list(ex.map(lambda rel: get(out, rel), FILES))
    for rel, size, status in res:
        print(f"{size:>10d} {status:8s} {rel}")
    tot = sum(r[1] for r in res)
    print(f"total {tot} bytes ({tot / 1e6:.1f} MB) in {time.time() - t0:.1f} s wall")
    if any(r[2] not in ("ok", "cached") for r in res):
        sys.exit(1)


if __name__ == "__main__":
    main()
