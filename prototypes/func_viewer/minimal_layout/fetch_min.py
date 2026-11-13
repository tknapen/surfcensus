"""Fetch one run of one subject of a derivative in the minimal layout (fMRIPrep 23.2.0 or
newer: reference images and transforms, no preprocessed series), with the raw series, the
anatomy, the derivative's own surfaces and the FreeSurfer files needed to check directions.
Anonymous, read-only, at most 4 requests at a time. A file that is already in OUT is kept.

usage: fetch_min.py OUT ACCESSION SUBJECT --task TASK [--session SES] [--run RUN]

  ACCESSION  as dsNNNNNN; SUBJECT with its sub- prefix; TASK, SES and RUN are the labels as
             they stand in the file names (run "01" is not run "1")

The feasibility study fetched one run this way: 23 files, 96.8 MB (the run had no JSON sidecar
of its own in the raw bucket, which is reported as HTTP 404 and is no error). Which dataset
that was is not written in this repository; README.md says how it was chosen.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1] / "survey"))  # also under -I
from survey import DERIV, RAW, list_prefix

MAX_REQUESTS = 4


def file_list(accession, subject, task, session, run):
    """Local name -> URL."""
    d = f"{DERIV}/fmriprep/{accession}-fmriprep/"
    r = f"{RAW}/{accession}/"
    s = subject
    ents = "".join(
        f"_{name}-{label}"
        for name, label in (("ses", session), ("task", task), ("run", run))
        if label
    )
    folder = f"{s}/ses-{session}/func/" if session else f"{s}/func/"
    f = f"{folder}{s}{ents}_"
    files = {
        "raw_bold.nii.gz": r + f + "bold.nii.gz",
        "raw_bold.json": r + f + "bold.json",
        "task_bold.json": r + f"task-{task}_bold.json",
        "hmc_boldref.nii.gz": d + f + "desc-hmc_boldref.nii.gz",
        "coreg_boldref.nii.gz": d + f + "desc-coreg_boldref.nii.gz",
        "hmc_boldref.json": d + f + "desc-hmc_boldref.json",
        "coreg_boldref.json": d + f + "desc-coreg_boldref.json",
        "coreg_xfm.txt": d + f + "from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt",
        "coreg_xfm.json": d + f + "from-boldref_to-T1w_mode-image_desc-coreg_xfm.json",
        "hmc_xfm.txt": d + f + "from-orig_to-boldref_mode-image_desc-hmc_xfm.txt",
        "hmc_xfm.json": d + f + "from-orig_to-boldref_mode-image_desc-hmc_xfm.json",
        "T1w.nii.gz": d + f"{s}/anat/{s}_desc-preproc_T1w.nii.gz",
        "T1w_mask.nii.gz": d + f"{s}/anat/{s}_desc-brain_mask.nii.gz",
        "dseg.nii.gz": d + f"{s}/anat/{s}_dseg.nii.gz",
        "L_white.surf.gii": d + f"{s}/anat/{s}_hemi-L_white.surf.gii",
        "L_pial.surf.gii": d + f"{s}/anat/{s}_hemi-L_pial.surf.gii",
        "R_white.surf.gii": d + f"{s}/anat/{s}_hemi-R_white.surf.gii",
        "R_pial.surf.gii": d + f"{s}/anat/{s}_hemi-R_pial.surf.gii",
        "fsnative_to_T1w.txt": d + f"{s}/anat/{s}_from-fsnative_to-T1w_mode-image_xfm.txt",
        "T1w_to_fsnative.txt": d + f"{s}/anat/{s}_from-T1w_to-fsnative_mode-image_xfm.txt",
        "fs_lh.white": d + f"sourcedata/freesurfer/{s}/surf/lh.white",
        "fs_T1.mgz": d + f"sourcedata/freesurfer/{s}/mri/T1.mgz",
        "dataset_description.json": d + "dataset_description.json",
    }
    # fMRIPrep's own record of its settings (output level among them), in a folder named after
    # the time of the run
    logs, _ = list_prefix(DERIV, f"fmriprep/{accession}-fmriprep/{s}/log/")
    tomls = sorted(k for k, *_ in logs if k.endswith("/fmriprep.toml"))
    if tomls:
        files["fmriprep.toml"] = f"{DERIV}/{tomls[0]}"
    return files


def get(out, item):
    local, url = item
    dst = out / local
    if dst.exists():
        return local, dst.stat().st_size, "cached"
    with (
        httpx.Client(timeout=180, headers={"User-Agent": "research-prototype"}) as c,
        c.stream("GET", url) as r,
    ):
        if r.status_code != 200:
            return local, 0, f"HTTP {r.status_code}"
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as fh:
            for ch in r.iter_bytes(1 << 20):
                fh.write(ch)
        tmp.rename(dst)
    return local, dst.stat().st_size, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("accession")
    ap.add_argument("subject")
    ap.add_argument("--task", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = file_list(args.accession, args.subject, args.task, args.session, args.run)
    t = time.time()
    with ThreadPoolExecutor(MAX_REQUESTS) as ex:
        res = list(ex.map(lambda item: get(out, item), files.items()))
    for r in res:
        print(f"{r[1]:>10d} {r[2]:7s} {r[0]}")
    print(sum(r[1] for r in res), "bytes in", round(time.time() - t, 1), "s")


if __name__ == "__main__":
    main()
