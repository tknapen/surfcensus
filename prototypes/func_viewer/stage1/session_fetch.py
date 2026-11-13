"""Fetch what the session measurements need of one subject: small files, and means on request.

usage: session_fetch.py OUT ACCESSION SUBJECT [--means]     (SUBJECT with its sub- prefix)

Anonymous, read-only, at most 4 requests at a time. For every run of the subject that has a
from-scanner_to-T1w transform in the fMRIPrep derivative, into OUT/deriv: that transform, the
T1w-space brain mask, the T1w-space reference image and the reference image in the run's own
space (text and images of a few hundred kB). Into OUT/raw: the JSON sidecar of the raw run where
there is one, and shape and affine of the raw series from the first 4 kB of the file. Files
keep the derivative's names, which name the subject and the task: OUT stays out of the
repository.

--means also reads the two series of every run, one run at a time: the raw series and the
T1w-space preprocessed series go to OUT/tmp, their temporal means to OUT/means as 3D images, and
the series are deleted before the next run is fetched. A run whose two series together exceed
300 MB is left out and named.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
sys.path.append(str(Path(__file__).resolve().parent.parent / "survey"))  # the listing functions
from session import parse_name, run_key, session_of
from session_frames import MASK, XFM, fetch, raw_grid
from survey import DERIV, MAX_REQUESTS, RAW, client, list_prefix

SMALL = (XFM, MASK, "_space-T1w_boldref.nii.gz", "_boldref.nii.gz")
PREPROC = "_space-T1w_desc-preproc_bold.nii.gz"
MAX_SERIES_BYTES = 300e6


def download(url, dst):
    """A large object, streamed to dst."""
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dst)


def temporal_mean(path, dst):
    """The temporal mean of a series as a 3D float32 image with the affine of the series."""
    img = nib.load(path)
    mean = np.asarray(img.dataobj, dtype=np.float32).mean(axis=3)
    nib.save(nib.Nifti1Image(mean, img.affine), dst)


def main():
    args = [a for a in sys.argv[1:] if a != "--means"]
    if len(args) < 3 or args[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out, accession, subject = Path(args[0]), args[1], args[2]
    deriv_keys, _ = list_prefix(DERIV, f"fmriprep/{accession}-fmriprep/{subject}/")
    raw_keys, _ = list_prefix(RAW, f"{accession}/{subject}/")
    sizes = {k: s for k, s, _, _ in deriv_keys}
    stems = sorted(k[: -len(XFM)] for k in sizes if k.endswith(XFM) and "/func/" in k)
    raw = {}
    for key, size, _, _ in raw_keys:
        ent = parse_name(key)
        if "/func/" in key and "echo" not in ent:
            for kind in ("_bold.json", "_bold.nii.gz"):
                if key.endswith(kind):
                    raw[session_of(ent), run_key(ent), kind] = (key, size)
    jobs = []
    for stem in stems:
        name = stem.rsplit("/", 1)[-1]
        jobs += [(DERIV, stem + s, out / "deriv" / (name + s)) for s in SMALL if stem + s in sizes]
        ent = parse_name(name)
        side = raw.get((session_of(ent), run_key(ent), "_bold.json"))
        if side:
            jobs.append((RAW, side[0], out / "raw" / (name + "_bold.json")))
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        done = list(pool.map(lambda job: fetch(*job), jobs))
        series = {}
        for stem in stems:
            name = stem.rsplit("/", 1)[-1]
            ent = parse_name(name)
            series[name] = raw.get((session_of(ent), run_key(ent), "_bold.nii.gz"))
        grids = list(
            pool.map(lambda n: raw_grid(series[n] and series[n][0], out, n), sorted(series))
        )
    n_side = sum(1 for _, _, dst in jobs if dst.name.endswith("_bold.json"))
    print(
        f"{len(stems)} runs with a transform; {sum(done)} of {len(jobs)} small files are in"
        f" place, {n_side} of them raw sidecars; {sum(g is not None for g in grids)} raw headers"
    )
    if "--means" not in sys.argv[1:]:
        return
    for stem in stems:
        name = stem.rsplit("/", 1)[-1]
        raw_mean = out / "means" / (name + "_raw_mean.nii.gz")
        pre_mean = out / "means" / (name + "_preproc_mean.nii.gz")
        if raw_mean.exists() and pre_mean.exists():
            continue
        source = series[name]
        if source is None or stem + PREPROC not in sizes:
            print(f"{name}: no raw series or no T1w-space series; no means")
            continue
        total = source[1] + sizes[stem + PREPROC]
        if total > MAX_SERIES_BYTES:
            print(f"{name}: the two series are {total / 1e6:.0f} MB; left out")
            continue
        tmp = out / "tmp"
        pairs = [
            (f"{RAW}/{source[0]}", tmp / "raw_bold.nii.gz"),
            (f"{DERIV}/{stem + PREPROC}", tmp / "preproc_bold.nii.gz"),
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda pair: download(*pair), pairs))
        raw_mean.parent.mkdir(parents=True, exist_ok=True)
        temporal_mean(pairs[0][1], raw_mean)
        temporal_mean(pairs[1][1], pre_mean)
        for _, path in pairs:
            path.unlink()
        print(f"means of {total / 1e6:.0f} MB of series written")


if __name__ == "__main__":
    main()
