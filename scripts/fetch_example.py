"""Fetch a small real example for surfcensus: ds000102 sub-01, laid out as a BIDS dataset.

usage: python scripts/fetch_example.py OUT

Writes OUT/ds000102/ with the raw functional series of sub-01 (two runs of the flanker
task), and below derivatives/fmriprep/ the fMRIPrep 21.0.2 derivative of those runs
(T1w-space series, masks, reference images, confounds, transforms, summary reports) and
the FreeSurfer 6.0.1 subject in sourcedata/freesurfer/. About 160 MB in 40 files, read
anonymously from the public OpenNeuro and OpenNeuroDerivatives buckets, at most 4 requests
at a time. A file that is already there is not fetched again.

Then, for both interfaces:

    surfcensus review OUT/ds000102

or for the EPI movies alone:

    surfcensus epi-session OUT/ds000102 --subject 01 --out OUT/epi-sub-01 --serve
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MAX_REQUESTS = 4
RAW = "https://s3.amazonaws.com/openneuro.org/ds000102/"
DERIV = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/"
SUBJECT = "sub-01"
STEM = SUBJECT + "_task-flanker_run-{n}"
PER_RUN = (
    "space-T1w_desc-preproc_bold.nii.gz",
    "space-T1w_desc-preproc_bold.json",
    "space-T1w_desc-brain_mask.nii.gz",
    "boldref.nii.gz",
    "desc-confounds_timeseries.tsv",
    "from-scanner_to-T1w_mode-image_xfm.txt",
)
ANAT = (
    "from-T1w_to-fsnative_mode-image_xfm.txt",
    "from-fsnative_to-T1w_mode-image_xfm.txt",
)
FREESURFER = (
    "mri/T1.mgz",
    "mri/brainmask.mgz",
    "surf/lh.white",
    "surf/lh.pial",
    "surf/rh.white",
    "surf/rh.pial",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "surf/lh.sphere.reg",
    "surf/rh.sphere.reg",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
    "stats/aseg.stats",
    "scripts/build-stamp.txt",
    "scripts/recon-all.log",
    "scripts/recon-all-lh.log",
    "scripts/recon-all-rh.log",
)


def file_list() -> dict[str, str]:
    """Path below OUT/ds000102 -> URL."""
    d = "derivatives/fmriprep/"
    files = {
        "dataset_description.json": RAW + "dataset_description.json",
        "task-flanker_bold.json": RAW + "task-flanker_bold.json",
        d + "dataset_description.json": DERIV + "dataset_description.json",
    }
    for n in (1, 2):
        stem = STEM.format(n=n)
        raw = f"{SUBJECT}/func/{stem}_bold.nii.gz"
        files[raw] = RAW + raw
        for suffix in PER_RUN:
            name = f"{SUBJECT}/func/{stem}_{suffix}"
            files[d + name] = DERIV + name
        report = f"{SUBJECT}/figures/{stem}_desc-summary_bold.html"
        files[d + report] = DERIV + report
    for suffix in ANAT:
        name = f"{SUBJECT}/anat/{SUBJECT}_{suffix}"
        files[d + name] = DERIV + name
    for name in FREESURFER:
        path = f"sourcedata/freesurfer/{SUBJECT}/{name}"
        files[d + path] = DERIV + path
    return files


def fetch(root: Path, item: tuple[str, str]) -> tuple[str, int, str]:
    local, url = item
    path = root / local
    if path.is_file() and path.stat().st_size > 0:
        return local, path.stat().st_size, "there"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with (
        httpx.Client(timeout=300.0, follow_redirects=False) as client,
        client.stream("GET", url) as response,
    ):
        if response.status_code != 200:
            return local, 0, f"HTTP {response.status_code}"
        with open(part, "wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    part.replace(path)
    return local, path.stat().st_size, "fetched"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    root = out / "ds000102"
    with ThreadPoolExecutor(MAX_REQUESTS) as pool:
        results = list(pool.map(lambda item: fetch(root, item), file_list().items()))
    failed = [r for r in results if r[2] not in ("there", "fetched")]
    total = sum(size for _, size, _ in results)
    n_new = sum(1 for r in results if r[2] == "fetched")
    print(f"{len(results)} files, {total / 1e6:.0f} MB in {root} ({n_new} fetched now)")
    for local, _, why in failed:
        print(f"not fetched: {local} ({why})", file=sys.stderr)
    if failed:
        return 1
    print(f"Both interfaces:  surfcensus review {root}")
    print(
        f"EPI movies alone: surfcensus epi-session {root} --subject 01 "
        f"--out {out / 'epi-sub-01'} --serve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
