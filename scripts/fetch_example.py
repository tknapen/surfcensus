"""Fetch one subject of a public OpenNeuro dataset for surfcensus, laid out as a BIDS dataset.

usage: python scripts/fetch_example.py OUT [--dataset dsNNNNNN] [--subject LABEL]

Without options this is a small real example, ds000102 sub-01 (two runs of the flanker
task, fMRIPrep 21.0.2, FreeSurfer 6.0.1; 40 files, about 160 MB).

Writes OUT/<dataset>/ with the raw functional series of the subject and their sidecars,
the subject's sessions.tsv and the scans.tsv of its sessions when the dataset has them (the
EPI interface orders sessions by the acquisition dates they hold), and below
derivatives/fmriprep/ what surfcensus reads of the fMRIPrep derivative that
OpenNeuroDerivatives publishes for the dataset: the T1w-space preprocessed series, their
masks, reference images, confounds tables, transforms and summary reports, and the
FreeSurfer subject in sourcedata/freesurfer/. The two public buckets are listed and read
anonymously, at most 4 requests at a time; nothing is written to them. A file that is
already there with the listed size is not fetched again.

Only derivatives with T1w-space series can be shown by the EPI interface (fMRIPrep below
23.2 in OpenNeuroDerivatives); the script says so when it finds none.

Then, for both interfaces:

    surfcensus review OUT/<dataset>

or for the EPI movies alone:

    surfcensus epi-session OUT/<dataset> --subject LABEL --out OUT/epi-sub-LABEL --serve
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MAX_REQUESTS = 4
RAW_BUCKET = "https://s3.amazonaws.com/openneuro.org/"
DERIV_BUCKET = "https://openneuro-derivatives.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

FUNC_SUFFIXES = (
    "_space-T1w_desc-preproc_bold.nii.gz",
    "_space-T1w_desc-preproc_bold.json",
    "_space-T1w_desc-brain_mask.nii.gz",
    "_desc-confounds_timeseries.tsv",
    "_from-scanner_to-T1w_mode-image_xfm.txt",
)
ANAT_SUFFIXES = (
    "_from-T1w_to-fsnative_mode-image_xfm.txt",
    "_from-fsnative_to-T1w_mode-image_xfm.txt",
)
FREESURFER = {
    "mri/T1.mgz",
    "mri/brainmask.mgz",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.pial.T1",
    "surf/rh.pial.T1",
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
}


def list_keys(client: httpx.Client, bucket: str, prefix: str, *, flat: bool = False) -> dict:
    """Key -> size of every object under ``prefix`` (ListObjectsV2); with ``flat`` only the
    objects directly in it."""
    keys: dict[str, int] = {}
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if flat:
            params["delimiter"] = "/"
        if token:
            params["continuation-token"] = token
        response = client.get(bucket, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.iter(NS + "Contents"):
            keys[item.findtext(NS + "Key") or ""] = int(item.findtext(NS + "Size") or 0)
        token = root.findtext(NS + "NextContinuationToken")
        if not token:
            return keys


def wanted(dataset: str, subject: str) -> tuple[dict[str, tuple[str, int]], int]:
    """Path below OUT/<dataset> -> (URL, size), and the number of T1w-space series found."""
    sub = f"sub-{subject}"
    raw_root = f"{dataset}/"
    deriv_root = f"fmriprep/{dataset}-fmriprep/"
    out: dict[str, tuple[str, int]] = {}
    with httpx.Client(timeout=120.0) as client:
        for key, size in list_keys(client, RAW_BUCKET, raw_root, flat=True).items():
            name = key[len(raw_root) :]
            if name == "dataset_description.json" or name.endswith("_bold.json"):
                out[name] = (RAW_BUCKET + key, size)
        for key, size in list_keys(client, RAW_BUCKET, f"{raw_root}{sub}/").items():
            name = key[len(raw_root) :]
            series = name.endswith(("_bold.nii.gz", "_bold.nii")) and "/func/" in name
            # the tables the acquisition dates of sessions are read from (decision D42)
            dates = name == f"{sub}/{sub}_sessions.tsv" or (
                name.startswith(f"{sub}/ses-") and name.endswith("_scans.tsv")
            )
            if series or dates or name.endswith("_bold.json"):
                out[name] = (RAW_BUCKET + key, size)
        d = "derivatives/fmriprep/"
        description = deriv_root + "dataset_description.json"
        for key, size in list_keys(client, DERIV_BUCKET, description).items():
            out[d + "dataset_description.json"] = (DERIV_BUCKET + key, size)
        n_series = 0
        for key, size in list_keys(client, DERIV_BUCKET, f"{deriv_root}{sub}/").items():
            name = key[len(deriv_root) :]
            base = name.rsplit("/", 1)[-1]
            if "/func/" in name:
                reference = base.endswith("_boldref.nii.gz") and "_space-" not in base
                keep = base.endswith(FUNC_SUFFIXES) or reference
                n_series += base.endswith(FUNC_SUFFIXES[0])
            elif "/figures/" in name:
                keep = base.endswith("_desc-summary_bold.html")
            elif "/anat/" in name:
                keep = base.endswith(ANAT_SUFFIXES)
            else:
                keep = False
            if keep:
                out[d + name] = (DERIV_BUCKET + key, size)
        fs = f"{deriv_root}sourcedata/freesurfer/{sub}/"
        for key, size in list_keys(client, DERIV_BUCKET, fs).items():
            if key[len(fs) :] in FREESURFER:
                out[d + key[len(deriv_root) :]] = (DERIV_BUCKET + key, size)
    return out, n_series


def fetch(root: Path, item: tuple[str, tuple[str, int]]) -> tuple[str, str]:
    local, (url, size) = item
    path = root / local
    if path.is_file() and path.stat().st_size == size:
        return local, "there"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with (
        httpx.Client(timeout=300.0) as client,
        client.stream("GET", url) as response,
    ):
        if response.status_code != 200:
            return local, f"HTTP {response.status_code}"
        with open(part, "wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    if part.stat().st_size != size:
        return local, f"{part.stat().st_size} bytes, the listing says {size}"
    part.replace(path)
    return local, "fetched"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the folder that gets <dataset>/")
    parser.add_argument("--dataset", default="ds000102", help="OpenNeuro accession")
    parser.add_argument("--subject", default="01", help="subject label, without sub-")
    args = parser.parse_args()
    subject = args.subject.removeprefix("sub-")
    root = args.out / args.dataset

    files, n_series = wanted(args.dataset, subject)
    total = sum(size for _, size in files.values())
    if not files:
        print(f"nothing found for sub-{subject} of {args.dataset}", file=sys.stderr)
        return 1
    print(f"sub-{subject} of {args.dataset}: {len(files)} files, {total / 1e6:.0f} MB, into {root}")
    if n_series == 0:
        print(
            "No T1w-space preprocessed series in the derivative of this subject: the surface "
            "interface can be used, the EPI movies cannot.",
            file=sys.stderr,
        )
    with ThreadPoolExecutor(MAX_REQUESTS) as pool:
        results = list(pool.map(lambda item: fetch(root, item), files.items()))
    failed = [(local, why) for local, why in results if why not in ("there", "fetched")]
    n_new = sum(1 for _, why in results if why == "fetched")
    print(f"{n_new} fetched now, {len(results) - n_new - len(failed)} were there")
    for local, why in failed:
        print(f"not fetched: {local} ({why})", file=sys.stderr)
    if failed:
        return 1
    print(f"Both interfaces:  surfcensus review {root}")
    print(
        f"EPI movies alone: surfcensus epi-session {root} --subject {subject} "
        f"--out {args.out / ('epi-sub-' + subject)} --serve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
