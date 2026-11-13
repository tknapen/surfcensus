"""List every subject, session and functional run of the fMRIPrep derivatives that were made with
fMRIPrep 23.0.2 or older, and say for each run whether it has what the two streams need.

usage: list_runs.py OUT [--max-datasets N] [--table-only]

Step 1 of stage 1 (README.md of the prototype, "Next steps"). Listing only: anonymous, read-only,
at most 4 requests at a time, no file of a dataset is opened but dataset_description.json of
each derivative, which names the fMRIPrep version. What it lists:

  the prefixes under fmriprep/ of the OpenNeuroDerivatives bucket, and the version of each;
  for each derivative of 23.0.2 or older: its top level, the folders under its FreeSurfer
  folder (sourcedata/freesurfer/, else freesurfer/), the top level of the raw dataset in the
  raw bucket, and then for every subject folder of the derivative, in full, the subject in the
  derivative, its FreeSurfer folder and the subject in the raw dataset.

Written to OUT, which names datasets and stays out of the repository:

  prefixes.json, versions.json    the prefixes and their versions, as first listed
  listings/<accession>.tsv.gz     every key that was listed, with size, ETag and date
  listings/<accession>.json       what was found at the top levels; written last
  runs.parquet                    one row per functional run (list_table.py has the columns)
  datasets.csv                    one row per derivative of 23.0.2 or older
  summary.txt                     the counts; it names no dataset
  invocations.jsonl               requests and seconds of each call of this script

The listing is resumable per dataset: a dataset whose listings/<accession>.json exists is not
listed again, in this call or a later one, and prefixes.json and versions.json are read when
they are there. To list anew, give another OUT. --max-datasets N stops after N more datasets
(the table is then of the datasets listed so far). --table-only sends no request and makes the
table, datasets.csv and summary.txt again from the stored listings.

check_listing.py OUT checks the table against its own rules and against the stored listings.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
sys.path.append(str(Path(__file__).resolve().parents[1] / "survey"))
from list_table import (
    FS_ROOTS,
    QUALIFY,
    build_rows,
    listing_paths,
    raw_run_id,
    raw_run_ids,
    read_listing,
    to_frame,
    write_listing,
)

NEWEST = (23, 0, 2)  # derivatives of this version or older hold a preprocessed series
SURVEY = "survey of 2026-11-13, first subjects only: 72 older derivatives, 71 with (a)"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def version_tuple(version: str | None) -> tuple[int, int, int] | None:
    """(major, minor, patch) of 23.0.2 or 22.1.0+0.gce344b39.dirty; None when it does not read."""
    parts = (version or "").split("+")[0].split(".")
    numbers = []
    for part in parts[:3]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            return None
        numbers.append(int(digits))
    if len(numbers) < 2:
        return None
    return (numbers[0], numbers[1], numbers[2] if len(numbers) > 2 else 0)


def accession_of(prefix: str) -> str:
    """fmriprep/<accession>-fmriprep/ -> <accession>."""
    return prefix.split("/")[1].rsplit("-fmriprep", 1)[0]


def read_version(prefix: str) -> dict:
    import list_s3 as s3
    from survey import fmriprep_version

    r = s3.get(f"{s3.DERIV}/{prefix}dataset_description.json")
    out = {"prefix": prefix, "status": r.status_code, "version": None}
    if r.status_code == 200:
        try:
            out["version"] = fmriprep_version(json.loads(r.content))
        except ValueError:
            pass
    return out


def prefixes_and_versions(out: Path) -> tuple[list[str], dict[str, str | None]]:
    """The dataset prefixes and the version each names; listed once and then read from OUT."""
    import list_s3 as s3

    path = out / "prefixes.json"
    if path.exists():
        prefixes = json.loads(path.read_text())["prefixes"]
    else:
        _, found = s3.list_prefix(s3.DERIV, "fmriprep/", "/")
        prefixes = sorted(set(found))
        path.write_text(json.dumps({"listed_at": now(), "prefixes": prefixes}, indent=0))
    path = out / "versions.json"
    if path.exists():
        versions = json.loads(path.read_text())["versions"]
    else:
        with ThreadPoolExecutor(s3.MAX_REQUESTS) as pool:
            versions = list(pool.map(read_version, prefixes))
        path.write_text(json.dumps({"read_at": now(), "versions": versions}, indent=0))
    return prefixes, {v["prefix"]: v["version"] for v in versions}


def list_dataset(prefix: str, version: str | None) -> tuple[dict, dict[str, list]]:
    """Everything that is listed of one derivative and of its raw dataset."""
    import list_s3 as s3

    started, t0, r0 = now(), time.time(), s3.requests_made()
    accession = accession_of(prefix)
    listing: dict[str, list] = {}
    listing["deriv_top"], top = s3.list_prefix(s3.DERIV, prefix, "/")
    subjects = sorted(p[len(prefix) :] for p in top if p[len(prefix) :].startswith("sub-"))
    fs_root, fs_folders = None, []
    for root in FS_ROOTS:
        _, found = s3.list_prefix(s3.DERIV, prefix + root, "/")
        if found:
            fs_root, fs_folders = root, sorted(p[len(prefix + root) :] for p in found)
            break
    listing["raw_top"], raw_top = s3.list_prefix(s3.RAW, f"{accession}/", "/")
    raw_folders = sorted(p[len(accession) + 1 :] for p in raw_top)

    tasks = []
    for subject in subjects:  # "sub-01/"
        tasks.append(("deriv", s3.DERIV, prefix + subject))
        if subject in fs_folders:
            tasks.append(("fs", s3.DERIV, prefix + fs_root + subject))
        if subject in raw_folders:
            tasks.append(("raw", s3.RAW, f"{accession}/{subject}"))
    with ThreadPoolExecutor(s3.MAX_REQUESTS) as pool:
        results = list(pool.map(lambda t: s3.list_prefix(t[1], t[2])[0], tasks))
    for source in ("deriv", "fs", "raw"):
        listing[source] = []
    for (source, _, _), keys in zip(tasks, results, strict=True):
        listing[source].extend(keys)
    meta = {
        "accession": accession,
        "prefix": prefix,
        "version": version,
        "fs_root": fs_root,
        "deriv_subjects": [s.rstrip("/") for s in subjects],
        "fs_folders": [s.rstrip("/") for s in fs_folders],
        "raw_listed": bool(listing["raw_top"] or raw_top),
        "raw_subjects": [s.rstrip("/") for s in raw_folders if s.startswith("sub-")],
        "n_keys": {source: len(keys) for source, keys in listing.items()},
        "listed_at": started,
        "requests": s3.requests_made() - r0,
        "seconds": round(time.time() - t0, 2),
    }
    return meta, listing


# --------------------------------------------------------------------------------- the table


def dataset_row(index: int, meta: dict, listing: dict, rows: list[dict]) -> dict:
    subjects = set(meta["deriv_subjects"])
    sessions = {(r["subject"], r["session"]) for r in rows}
    raw_runs = raw_run_ids(meta["accession"], listing)
    found = {raw_run_id(r["raw_key"]) for r in rows if r["raw_key"]}
    return {
        "dataset": meta["accession"],
        "prefix_index": index,
        "fmriprep_version": meta["version"],
        "fs_root": meta["fs_root"],
        "raw_listed": meta["raw_listed"],
        "n_subject_folders": len(subjects),
        "n_subjects_with_fs_folder": len(subjects & set(meta["fs_folders"])),
        "n_subjects_in_raw": len(subjects & set(meta["raw_subjects"])),
        "n_raw_subject_folders": len(meta["raw_subjects"]),
        "n_subjects_with_runs": len({r["subject"] for r in rows}),
        "n_sessions": len(sessions),
        "n_runs": len(rows),
        **{f"n_{flag}": sum(r[flag] for r in rows) for flag in QUALIFY},
        "n_has_preproc_boldspace": sum(r["has_preproc_boldspace"] for r in rows),
        "n_qualifying": sum(r["qualifies"] for r in rows),
        "n_raw_runs": len(raw_runs),  # of the subjects that the derivative has
        "n_raw_runs_without_derivative_run": len(raw_runs - found),
        "n_keys_deriv": meta["n_keys"]["deriv"],
        "n_keys_fs": meta["n_keys"]["fs"],
        "n_keys_raw": meta["n_keys"]["raw"],
        "requests": meta["requests"],
        "seconds": meta["seconds"],
        "listed_at": meta["listed_at"],
    }


def quartiles(values) -> str:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return "none"
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return "min {:g}, quartiles {:g} / {:g} / {:g}, max {:g}".format(*q)


def histogram(values, edges=(1, 2, 3, 4, 5, 10, 20, 50)) -> str:
    """Counts of whole numbers in the classes 1, 2, 3, 4, 5 to 9, 10 to 19, 20 to 49, 50+."""
    v = np.asarray(values)
    parts = []
    for lo, hi in zip(edges, (*edges[1:], None), strict=True):
        n = int(((v >= lo) & (v < hi)).sum()) if hi else int((v >= lo).sum())
        label = str(lo) if hi == lo + 1 else (f"{lo} to {hi - 1}" if hi else f"{lo} or more")
        parts.append(f"{label}: {n}")
    return "; ".join(parts)


def mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def summary(
    runs: pd.DataFrame, datasets: pd.DataFrame, versions: dict, out: Path, n_older: int
) -> str:
    """The counts of the listing, as text that names no dataset."""
    tuples = {p: version_tuple(v) for p, v in versions.items()}
    unread = sum(t is None for t in tuples.values())
    older = Counter(versions[p] for p, t in tuples.items() if t is not None and t <= NEWEST)
    q = runs[runs["qualifies"]]
    session_key = ["dataset", "subject", "session"]
    subject_key = ["dataset", "subject"]
    per_session = runs.groupby(session_key, dropna=False).size()
    per_subject = runs.drop_duplicates(session_key).groupby(subject_key).size()
    q_per_session = q.groupby(session_key, dropna=False).size()
    q_per_subject = q.drop_duplicates(session_key).groupby(subject_key).size()
    with_a = int((datasets["n_has_preproc"] > 0).sum())
    lines = [
        "Functional runs in the fMRIPrep derivatives of fMRIPrep 23.0.2 or older",
        "(OpenNeuroDerivatives bucket, fmriprep/; raw runs in the OpenNeuro bucket).",
        "Listings only. This text names no dataset; datasets.csv and runs.parquet do.",
        f"Datasets listed between {datasets['listed_at'].min()} and {datasets['listed_at'].max()}"
        if len(datasets)
        else "No dataset listed yet.",
        f"Table written {now()}.",
        "",
        f"Dataset prefixes under fmriprep/: {len(versions)}",
        f"  naming fMRIPrep 23.0.2 or older: {n_older}"
        f" ({', '.join(f'{v} ({n})' for v, n in sorted(older.items()))})",
        f"  naming a newer version: {len(versions) - n_older - unread};"
        f" without a version that reads: {unread}",
        f"  listed in full so far: {len(datasets)} of {n_older}",
        "",
        "Datasets",
        f"  with at least one functional run: {int((datasets['n_runs'] > 0).sum())}",
        f"  with at least one run that has (a), the T1w-space series: {with_a}",
        f"  with at least one qualifying run: {int((datasets['n_qualifying'] > 0).sum())}",
        f"  with a FreeSurfer folder: {int(datasets['fs_root'].notna().sum())}"
        f" (at {', '.join(sorted(set(datasets['fs_root'].dropna())))})",
        f"  whose raw dataset is in the raw bucket: {int(datasets['raw_listed'].sum())}",
        f"  ({SURVEY})",
        "",
        "Subjects, sessions, runs (a dataset without ses- labels has one session per subject)",
        f"  subject folders in the derivatives: {int(datasets['n_subject_folders'].sum())}",
        f"    with a FreeSurfer folder: {int(datasets['n_subjects_with_fs_folder'].sum())}",
        f"    with a folder in the raw dataset: {int(datasets['n_subjects_in_raw'].sum())}",
        f"  subjects with at least one functional run: {len(runs.drop_duplicates(subject_key))}",
        f"  sessions: {len(per_session)}",
        f"  runs: {len(runs)}",
        f"  qualifying runs: {len(q)}, in {q['dataset'].nunique()} datasets,"
        f" {len(q_per_subject)} subjects and {len(q_per_session)} sessions",
        "",
        "Runs that have each item",
    ]
    names = {
        "has_preproc": "(a) T1w-space preprocessed series",
        "has_confounds": "(b) confounds table",
        "has_xfm": "(c) from-scanner_to-T1w transform of the run",
        "has_fsnative_xfm": "(d) from-T1w_to-fsnative transform of the subject",
        "has_fs": "(e) T1.mgz, white and pial of both hemispheres",
        "has_mask": "(f) brain mask of the run in T1w space",
        "has_raw": "(g) raw run in the raw bucket",
    }
    for flag, name in names.items():
        lines.append(f"  {name}: {int(runs[flag].sum())} of {len(runs)}")
    first_missing = Counter()
    for flags in runs.loc[~runs["qualifies"], list(QUALIFY)].itertuples(index=False):
        missing = [names[f][:3] for f, have in zip(QUALIFY, flags, strict=True) if not have]
        first_missing[" ".join(missing) or "run number not unique as a number"] += 1
    lines.append(f"Runs that do not qualify: {len(runs) - len(q)}, by what they lack")
    for what, n in first_missing.most_common():
        lines.append(f"  {what}: {n}")
    no_raw = runs[~runs["has_raw"]]
    only_inherited = q["raw_json_key"].isna() & q["raw_json_inherited_key"].notna()
    both = q["preproc_size"] + q["raw_size_all"]
    bold_only = runs[runs["has_preproc_boldspace"] & ~runs["has_preproc"]]
    own_all = q["raw_json_key"].notna().groupby([q[k] for k in session_key], dropna=False).all()
    where_subject = runs.drop_duplicates(subject_key)["fsnative_xfm_where"].value_counts()
    where = runs["fsnative_xfm_where"].value_counts()
    variants = runs.drop_duplicates(subject_key)["fs_lh_pial_variant"].value_counts()
    lines += [
        f"Runs without (g): {len(no_raw)}; of them the raw dataset has no folder for the subject"
        f" in {int((~no_raw['raw_subject_listed']).sum())}, and several raw run labels and none"
        f" equal as text in {int((no_raw['raw_match'] == 'ambiguous').sum())}",
        f"Raw run found by the same label: {int((runs['raw_match'] == 'exact').sum())};"
        f" by the same number under another label: {int((runs['raw_match'] == 'numeric').sum())}",
        f"Raw functional runs of the subjects that the derivatives have: "
        f"{int(datasets['n_raw_runs'].sum())}; of them without a run in the derivative: "
        f"{int(datasets['n_raw_runs_without_derivative_run'].sum())}, in "
        f"{int((datasets['n_raw_runs_without_derivative_run'] > 0).sum())} datasets",
        f"Runs with more than one raw echo: {int((runs['raw_n_echoes'] > 1).sum())}"
        f" (qualifying: {int((q['raw_n_echoes'] > 1).sum())})",
        f"Qualifying runs with a JSON sidecar beside the raw series: "
        f"{int(q['raw_json_key'].notna().sum())} of {len(q)}; without one but with a sidecar"
        f" higher up: {int(only_inherited.sum())}",
        f"Sessions in which every qualifying run has such a sidecar: {int(own_all.sum())} of"
        f" {len(own_all)}; among sessions with more than one qualifying run:"
        f" {int(own_all[q_per_session > 1].sum())} of {int((q_per_session > 1).sum())}",
        "Where (d) was found, by run: "
        + (", ".join(f"{k} level {int(n)}" for k, n in where.items()) or "nowhere")
        + "; by subject: "
        + (", ".join(f"{k} level {int(n)}" for k, n in where_subject.items()) or "nowhere"),
        "Pial surface name (left hemisphere), by subject with runs: "
        + (", ".join(f"{k} {int(n)}" for k, n in variants.items()) or "none"),
        f"Runs whose number is not unique as a number: {int(runs['dup_numeric_run'].sum())}",
        f"Runs without a run entity: {int(runs['run'].isna().sum())}; with a run label that is"
        f" not a number: {int((runs['run'].isna() & runs['run_label'].notna()).sum())}",
        f"Runs whose derivative names carry an echo entity: {int((runs['n_echo_keys'] > 0).sum())}",
        "Qualifying runs whose ETag is not an MD5 (multipart upload), T1w-space series: "
        f"{int(q['preproc_etag'].str.contains('-').sum())}, raw series: "
        f"{int(q['raw_etag'].str.contains('-').sum())}",
        f"Runs with a preprocessed series in BOLD space and none in T1w space: {len(bold_only)},"
        f" in {bold_only['dataset'].nunique()} datasets and"
        f" {len(bold_only.drop_duplicates(subject_key))} subjects",
        "Qualifying runs per dataset that has one: "
        + quartiles(datasets.loc[datasets["n_qualifying"] > 0, "n_qualifying"]),
        "Subjects with a qualifying run, per dataset that has one: "
        + quartiles(q.drop_duplicates(subject_key).groupby("dataset").size()),
        "",
        "Runs per session, all runs: " + quartiles(per_session),
        "  sessions by number of runs: " + histogram(per_session),
        "Sessions per subject, all runs: " + quartiles(per_subject),
        "  subjects by number of sessions: " + histogram(per_subject),
        "Qualifying runs per session that has one: " + quartiles(q_per_session),
        "  sessions by number of qualifying runs: " + histogram(q_per_session),
        f"  sessions with more than one qualifying run: {int((q_per_session > 1).sum())}"
        f" of {len(q_per_session)}",
        "Sessions with a qualifying run, per subject that has one: " + quartiles(q_per_subject),
        "  subjects by number of such sessions: " + histogram(q_per_subject),
        f"  subjects with more than one such session: {int((q_per_subject > 1).sum())}"
        f" of {len(q_per_subject)}",
        "",
        f"Confounds tables of the qualifying runs: {mb(float(q['confounds_size'].sum()))}"
        f" in {len(q)} files ({quartiles(q['confounds_size'] / 1e6)} MB)",
        "Size of a qualifying run's two series together (T1w-space series plus raw series, all"
        " echoes of the raw run), MB: " + quartiles(both / 1e6),
        f"  all qualifying runs together: {mb(float(both.sum()))}"
        f" (T1w-space {mb(float(q['preproc_size'].sum()))},"
        f" raw {mb(float(q['raw_size_all'].sum()))})",
        "",
    ]
    calls = [json.loads(line) for line in (out / "invocations.jsonl").read_text().splitlines()]
    lines += [
        f"Requests sent: {sum(c['requests'] for c in calls)} in {len(calls)} call(s) of the"
        f" script, {sum(c['seconds'] for c in calls):.0f} s in all, at most 4 at a time",
        f"  of them per dataset: {int(datasets['requests'].sum())} requests,"
        f" {datasets['seconds'].sum():.0f} s ({quartiles(datasets['requests'])} per dataset)",
        f"Keys listed: {int(datasets['n_keys_deriv'].sum())} under derivative subjects,"
        f" {int(datasets['n_keys_fs'].sum())} under FreeSurfer folders,"
        f" {int(datasets['n_keys_raw'].sum())} under raw subjects",
    ]
    return "\n".join(lines) + "\n"


def write_tables(out: Path, prefixes: list[str], versions: dict, older: list[str]) -> None:
    rows, dataset_rows = [], []
    for prefix in older:
        accession = accession_of(prefix)
        if not listing_paths(out, accession)[1].exists():
            continue
        meta, listing = read_listing(out, accession)
        found = build_rows(meta, listing)
        rows.extend(found)
        dataset_rows.append(dataset_row(prefixes.index(prefix), meta, listing, found))
    runs = to_frame(rows)
    runs.to_parquet(out / "runs.parquet", index=False)
    datasets = pd.DataFrame(dataset_rows)
    datasets.to_csv(out / "datasets.csv", index=False)
    text = summary(runs, datasets, versions, out, len(older)) if len(datasets) else ""
    (out / "summary.txt").write_text(text)
    print(text, end="")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help") or args[0].startswith("--"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out = Path(args[0])
    table_only = "--table-only" in args
    limit = int(args[args.index("--max-datasets") + 1]) if "--max-datasets" in args else None
    out.mkdir(parents=True, exist_ok=True)
    if table_only and not (out / "versions.json").exists():
        sys.exit(f"--table-only: {out} holds no listing yet")

    t0 = time.time()
    prefixes, versions = prefixes_and_versions(out)
    tuples = {p: version_tuple(versions[p]) for p in prefixes}
    older = [p for p in prefixes if tuples[p] is not None and tuples[p] <= NEWEST]
    listed = 0
    if not table_only:
        import list_s3 as s3

        done: set[str] = set()  # no dataset is listed twice in one call
        for i, prefix in enumerate(older):
            accession = accession_of(prefix)
            if accession in done or listing_paths(out, accession)[1].exists():
                done.add(accession)
                continue
            if limit is not None and listed >= limit:
                break
            done.add(accession)
            meta, listing = list_dataset(prefix, versions[prefix])
            write_listing(out, meta, listing)
            listed += 1
            print(
                f"dataset {i + 1} of {len(older)}: {len(meta['deriv_subjects'])} subject folders,"
                f" {sum(meta['n_keys'].values())} keys, {meta['requests']} requests,"
                f" {meta['seconds']:.0f} s",
                file=sys.stderr,
                flush=True,
            )
        with open(out / "invocations.jsonl", "a") as f:
            call = {
                "ended_at": now(),
                "requests": s3.requests_made(),
                "seconds": round(time.time() - t0, 1),
                "datasets_listed": listed,
            }
            f.write(json.dumps(call) + "\n")
    write_tables(out, prefixes, versions, older)


if __name__ == "__main__":
    main()
