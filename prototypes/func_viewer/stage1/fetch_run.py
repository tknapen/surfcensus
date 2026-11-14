"""Fetch what one selected session needs for pack_run.py, anonymously and read-only.

usage: fetch_run.py LISTING SELECTION ID OUT [--max-runs N] [--runs N,N,...] [--plan]

LISTING    the directory of list_runs.py (runs.parquet and listings/)
SELECTION  the directory of select_runs.py (selection.parquet)
ID         a neutral id of the selection: H01 to H06, Q01, Q02
OUT        a directory for the files; not in the repository. It gets:
             deriv/   the T1w-space series, brain masks, scanner-to-T1w transforms, confounds
                      tables, BOLD-space reference images, derivative sidecars and fMRIPrep's
                      summary reports (figures/<stem>_desc-summary_bold.html, a few kB) of the
                      runs, and the fsnative transform
             raw/     the raw series of the runs and their own sidecars, when they have one
             fs/      mri/T1.mgz and surf/{lh,rh}.{white,pial} (a pial variant such as
                      lh.pial.T1 is stored as lh.pial, the name pack_run.py reads)
             fetch_log.json   every file with key, size, ETag, MD5 check and time; it names
                      the dataset, so OUT stays outside the repository
--max-runs N  the runs of the session that are fetched, at most (default 4): the anchor, the
              selected run, then the others in BIDS name order
--runs N,...  fetch exactly these runs, by their number in the session (1 is <ID>-01); the
              anchor and the selected run must be among them
--plan        print which runs would be fetched and how many bytes; fetch only the sidecars,
              which the anchor rule needs

The anchor is chosen as session.pick_anchor chooses it: by AcquisitionTime of the raw sidecars
when every run of the session has its own, else the first in BIDS name order. The sidecars are
therefore fetched first (they are a few kB).

At most 4 requests at a time, no credential, nothing signed, nothing written to a bucket. Every
file is compared with the listing: its size, and its MD5 against the ETag where the ETag is an
MD5 (not for a multipart upload, whose ETag has a dash). A file that is in OUT with the size of
the listing and a matching MD5 in fetch_log.json is not fetched again; a partial download
(<name>.part) is continued with a range request. Exit status 0 when every file is in place and
checked, 1 when one is not, 2 for usage.

A run whose BOLD-space reference image or summary report the listing lacks is named on
standard error: without the reference image pack_run.py takes the raw header's img.affine
unchecked, without the report it records distortion and slice-timing correction as unknown (or
distortion correction as applied when the listing has the run's distortion-correction figure,
figures/<stem>_desc-sdc_bold.svg). fetch_log.json records, under "_session", which runs were not
fetched and, per run, whether that figure is listed; pack_run.py reads both.

The last line on standard output is JSON with the counts and names nothing: runs, files, bytes
transferred in this call, requests and seconds.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))  # the sibling modules, also under -I
from list_table import read_listing  # noqa: E402
from select_runs import read_selection  # noqa: E402
from session import pick_anchor, sidecar_time  # noqa: E402

MAX_REQUESTS = 4
ATTEMPTS = 7
RETRY_STATUS = (500, 502, 503, 504)
DERIV = "https://openneuro-derivatives.s3.amazonaws.com"
RAW = "https://s3.amazonaws.com/openneuro.org"
PREPROC = "_space-T1w_desc-preproc_bold.nii.gz"
SUMMARY = "_desc-summary_bold.html"
SDC_FIGURE = "_desc-sdc_bold.svg"
BLOCK = 1 << 20

_slots = threading.BoundedSemaphore(MAX_REQUESTS)
_lock = threading.Lock()
_count = {"requests": 0, "bytes": 0}
_client = httpx.Client(timeout=120.0, headers={"User-Agent": "listing-survey/0.4"})


def _add(what, n):
    with _lock:
        _count[what] += n


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(BLOCK):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size, etag):
    """Stream url to dest through dest.part, continuing a partial file. Returns a log entry."""
    part = dest.with_name(dest.name + ".part")
    md5_check = bool(etag) and "-" not in etag
    t0 = time.perf_counter()
    last = None
    for attempt in range(ATTEMPTS):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if 0 < have < size else {}
        if have < size or size == 0:
            with _slots:
                _add("requests", 1)
                try:
                    with _client.stream("GET", url, headers=headers) as r:
                        if r.status_code in RETRY_STATUS:
                            last = f"HTTP {r.status_code}"
                        elif r.status_code not in (200, 206):
                            return {"ok": False, "error": f"HTTP {r.status_code}"}
                        else:
                            mode = "ab" if r.status_code == 206 else "wb"
                            with open(part, mode) as f:
                                for block in r.iter_bytes(BLOCK):
                                    f.write(block)
                                    _add("bytes", len(block))
                            last = None
                except httpx.HTTPError as exc:
                    last = repr(exc)
            if last is not None:
                time.sleep(min(1.5 * 2**attempt, 30.0))  # outside the semaphore
                continue
        got = part.stat().st_size
        if got != size:
            last = f"size {got} against {size} in the listing"
            part.unlink()
            continue
        md5 = _md5_file(part) if md5_check else None
        if md5_check and md5 != etag:
            part.unlink()
            last = "MD5 differs from the ETag of the listing"
            continue
        part.replace(dest)
        return {"ok": True, "md5": md5, "seconds": round(time.perf_counter() - t0, 2)}
    return {"ok": False, "error": str(last)}


def figures_index(listing):
    """{file name: (key, size, etag)} of the derivative's figures/ folders in the listing."""
    out = {}
    for key, size, etag in listing.get("deriv", []):
        parts = key.rsplit("/", 2)
        if len(parts) == 3 and parts[1] == "figures":
            out[parts[2]] = (key, size, etag)
    return out


def run_stem(preproc_key):
    return preproc_key.rsplit("/", 1)[1][: -len(PREPROC)]


def missing_per_run(rows, listing):
    """[(run_id, what)] of the reference images and summary reports that the listing lacks."""
    keys = {key for key, _, _ in listing.get("deriv", [])}
    figures = figures_index(listing)
    out = []
    for row in rows.itertuples():
        if row.preproc_key[: -len(PREPROC)] + "_boldref.nii.gz" not in keys:
            out.append((row.run_id, "no BOLD-space reference image"))
        if run_stem(row.preproc_key) + SUMMARY not in figures:
            out.append((row.run_id, "no summary report"))
    return out


def sdc_figures_listed(rows, listing):
    """{stem: whether figures/<stem>_desc-sdc_bold.svg is in the listing} for every row."""
    figures = figures_index(listing)
    return {run_stem(k): run_stem(k) + SDC_FIGURE in figures for k in rows.preproc_key}


def plan_files(rows, listing, fs_dir_name="fs"):
    """{local path relative to OUT: (base, key, size, etag)} for the given rows of one session."""
    files = {}
    deriv_index = {key: (size, etag) for key, size, etag in listing.get("deriv", [])}
    figures = figures_index(listing)
    for row in rows.itertuples():
        for kind in ("preproc", "mask", "xfm", "confounds"):
            key = getattr(row, f"{kind}_key")
            files["deriv/" + key.rsplit("/", 1)[1]] = (
                DERIV,
                key,
                int(getattr(row, f"{kind}_size")),
                getattr(row, f"{kind}_etag"),
            )
        sidecar = row.preproc_key[: -len(".nii.gz")] + ".json"
        # the BOLD-space reference image: its affine is the raw frame of the stored transform
        # (session.raw_affine); the derivative's sidecar holds RepetitionTime
        boldref = row.preproc_key[: -len(PREPROC)] + "_boldref.nii.gz"
        for key in (sidecar, boldref):
            if key in deriv_index:
                size, etag = deriv_index[key]
                files["deriv/" + key.rsplit("/", 1)[1]] = (DERIV, key, size, etag)
        # fMRIPrep's summary report of the run: whether distortion and slice-timing correction
        # were applied to the preprocessed series (pack_inputs.preprocessing_of)
        summary = run_stem(row.preproc_key) + SUMMARY
        if summary in figures:
            key, size, etag = figures[summary]
            files["deriv/" + summary] = (DERIV, key, int(size), etag)
        files["raw/" + row.raw_key.rsplit("/", 1)[1]] = (
            RAW,
            row.raw_key,
            int(row.raw_size),
            row.raw_etag,
        )
    first = rows.iloc[0]
    files["deriv/" + first.fsnative_xfm_key.rsplit("/", 1)[1]] = (
        DERIV,
        first.fsnative_xfm_key,
        int(first.fsnative_xfm_size),
        first.fsnative_xfm_etag,
    )
    for col, local in (
        ("fs_t1", "mri/T1.mgz"),
        ("fs_lh_white", "surf/lh.white"),
        ("fs_rh_white", "surf/rh.white"),
        ("fs_lh_pial", "surf/lh.pial"),
        ("fs_rh_pial", "surf/rh.pial"),
    ):
        files[f"{fs_dir_name}/{local}"] = (
            DERIV,
            first[f"{col}_key"],
            int(first[f"{col}_size"]),
            first[f"{col}_etag"],
        )
    return files


def sidecar_files(rows):
    """The raw runs' own sidecars, as plan_files gives files."""
    out = {}
    for row in rows.itertuples():
        if row.raw_json_key is not None and not _na(row.raw_json_key):
            out["raw/" + row.raw_json_key.rsplit("/", 1)[1]] = (
                RAW,
                row.raw_json_key,
                int(row.raw_json_size),
                row.raw_json_etag,
            )
    return out


def _na(value):
    try:
        return bool(value != value) or str(value) == "<NA>"
    except TypeError:
        return True


def fetch_all(out, files, log):
    """Fetch every file that is not in place; update log; return the local paths that failed."""

    def one(item):
        local, (base, key, size, etag) = item
        dest = out / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = log.get(local)
        if (
            dest.exists()
            and dest.stat().st_size == size
            and done
            and done.get("ok")
            and done.get("key") == key
            and done.get("etag") == etag
        ):
            return local, done, True
        entry = download(f"{base}/{key}", dest, size, etag)
        entry.update({"key": key, "size": size, "etag": etag})
        return local, entry, False

    failed = []
    with ThreadPoolExecutor(MAX_REQUESTS) as pool:
        for local, entry, _skipped in pool.map(one, sorted(files.items())):
            log[local] = entry
            if not entry.get("ok"):
                failed.append(local)
    return failed


def choose_runs(rows, anchor_stem, max_runs, numbers):
    """The rows to fetch: anchor, selected, then the others in name order (rows are in it)."""
    stems = [k.rsplit("/", 1)[1][: -len(PREPROC)] for k in rows.preproc_key]
    anchor = stems.index(anchor_stem)
    selected = [i for i, s in enumerate(rows.selected) if s]
    if numbers:
        take = [n - 1 for n in numbers]
        missing = [i for i in [anchor, *selected] if i not in take]
        if missing:
            raise SystemExit("--runs must include the anchor and the selected run")
    else:
        take = [anchor] + [i for i in selected if i != anchor]
        for i in range(len(stems)):
            if len(take) >= max_runs:
                break
            if i not in take:
                take.append(i)
    take = sorted(set(take))
    return rows.iloc[take], [i for i in range(len(stems)) if i not in take]


def main():
    args = sys.argv[1:]
    options = {"--max-runs": "4", "--runs": None}
    plan_only = False
    positional = []
    while args:
        arg = args.pop(0)
        if arg in options and args:
            options[arg] = args.pop(0)
        elif arg == "--plan":
            plan_only = True
        elif arg.startswith("--"):
            positional = []
            break
        else:
            positional.append(arg)
    if len(positional) != 4:
        print(__doc__.strip().split("\n\nThe anchor")[0], file=sys.stderr)
        sys.exit(2)
    listing_dir, selection_dir, pack_id, out = (Path(p) for p in positional)
    pack_id = str(pack_id)
    if (HERE.parents[2] / ".git").exists() and Path(out).resolve().is_relative_to(HERE.parents[2]):
        sys.exit("OUT names a dataset in fetch_log.json: it may not lie inside the repository")
    t_start = time.perf_counter()
    sel = read_selection(selection_dir / "selection.parquet")
    rows = sel[sel["id"] == pack_id].reset_index(drop=True)
    if rows.empty:
        sys.exit(f"{pack_id} is not an id of the selection")
    _, listing = read_listing(listing_dir, rows["dataset"].iloc[0])
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    failed = fetch_all(out, sidecar_files(rows), log)
    stems = [k.rsplit("/", 1)[1][: -len(PREPROC)] for k in rows.preproc_key]
    times = []
    for row in rows.itertuples():
        path = None
        if not _na(row.raw_json_key):
            path = out / "raw" / row.raw_json_key.rsplit("/", 1)[1]
        times.append(sidecar_time(path) if path is not None and path.exists() else None)
    anchor = pick_anchor(list(zip(stems, times, strict=True)))
    numbers = [int(n) for n in options["--runs"].split(",")] if options["--runs"] else None
    take, left = choose_runs(rows, anchor.name, int(options["--max-runs"]), numbers)
    files = plan_files(take, listing)
    total = sum(f[2] for f in files.values())
    run_ids = list(take.run_id)
    print(
        f"{pack_id}: session of {len(rows)} runs; anchor {rows.run_id[stems.index(anchor.name)]}"
        f" by {anchor.rule}; fetching {', '.join(run_ids)}"
        f"{'; not fetched ' + ', '.join(rows.run_id[i] for i in left) if left else ''};"
        f" {len(files)} files, {total / 1e6:.1f} MB"
    )
    for run_id, what in missing_per_run(take, listing):
        print(f"{run_id}: the listing has {what} of this run", file=sys.stderr)
    if not plan_only:
        failed += fetch_all(out, files, log)
    log["_session"] = {
        "id": pack_id,
        "anchor_run_id": rows.run_id[stems.index(anchor.name)],
        "anchor_rule": anchor.rule,
        "fetched_run_ids": run_ids,
        "left_out_run_ids": [rows.run_id[i] for i in left],
        "pial_variants": [rows.fs_lh_pial_variant[0], rows.fs_rh_pial_variant[0]],
        "sdc_figure_listed": sdc_figures_listed(rows, listing),
        "missing_per_run": [list(x) for x in missing_per_run(take, listing)],
    }
    log_path.write_text(json.dumps(log, indent=1, sort_keys=True))
    summary = {
        "id": pack_id,
        "runs_fetched": len(run_ids),
        "runs_left_out": len(left),
        "files": len(files),
        "bytes_planned": total,
        "bytes_transferred": _count["bytes"],
        "requests": _count["requests"],
        "seconds": round(time.perf_counter() - t_start, 1),
        "failed": len(failed),
    }
    for local in failed:
        print(f"not in place: {local}: {log[local].get('error')}", file=sys.stderr)
    print(json.dumps(summary, sort_keys=True))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
