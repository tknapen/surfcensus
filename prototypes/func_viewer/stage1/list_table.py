"""The table of functional runs that list_runs.py writes: its columns, how a run is told from
the names in a listing, and the reader that the later steps of stage 1 use.

A helper module, not a script. Other scripts of this folder read the table with

    sys.path.append(str(Path(__file__).resolve().parent))
    from list_table import read_runs
    runs = read_runs(OUT / "runs.parquet")

read_runs() returns a pandas frame with the column types of COLUMNS, whatever the parquet
reader made of them, so a run number is an integer (pandas Int64, missing where the name has no
run entity) and a subject label such as "01" is text.

What a row is. One functional run of one derivative: the files in a func/ folder of the
derivative whose names share sub, ses, task, acq, ce, rec, dir and run. The echo and part
entities do not make a run of their own. Keys are full keys of their bucket: preproc, confounds,
xfm, mask, fsnative_xfm and fs_* in the derivatives bucket, raw and raw_json in the raw bucket.

Run numbers are compared as numbers: run-1 of the derivative and run-01 of the raw dataset are
the same run. run_label keeps the text of the derivative's name and raw_run_label that of the
raw name, and raw_match says whether the two were equal as text ("exact") or only as numbers
("numeric").

A run QUALIFIES when it has all of: (a) the T1w-space preprocessed series, (b) the confounds
table, (c) its own from-scanner_to-T1w transform, (d) the subject's from-T1w_to-fsnative
transform, (e) T1.mgz and the white and pial surfaces of both hemispheres under the FreeSurfer
folder of the derivative, (f) the brain mask of the run in T1w space, (g) the raw run in the raw
bucket. A file of size 0 counts as absent. A run whose number is not unique as a number among
the derivative's runs (run-1 beside run-01) does not qualify.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

# The entities that tell one run from another, in the order of a BIDS name.
RUN_ENTITIES = ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run")
OTHER_ENTITIES = ("acq", "ce", "rec", "dir")

# The pial surface has several names since FreeSurfer 7, and the link ?h.pial that recon-all
# makes is absent on S3 (docs/data_sources.md of the package). Same order as the package's
# resolver: the plain name, then the T2 and FLAIR refinements, then the T1 surface.
PIAL_VARIANTS = ("pial", "pial.T2", "pial.FLAIR", "pial.T1")
FS_ROOTS = ("sourcedata/freesurfer/", "freesurfer/")  # fMRIPrep's place, then the older one

# The files of a run whose key, size and ETag are recorded: column prefix -> what it is.
FILES = {
    "preproc": "(a) space-T1w_desc-preproc_bold.nii.gz of the run",
    "confounds": "(b) desc-confounds_timeseries.tsv (or _regressors.tsv) of the run",
    "xfm": "(c) from-scanner_to-T1w_mode-image_xfm.txt of the run",
    "fsnative_xfm": "(d) from-T1w_to-fsnative_mode-image_xfm.txt of the subject",
    "fs_t1": "(e) mri/T1.mgz of the subject's FreeSurfer folder",
    "fs_lh_white": "(e) surf/lh.white",
    "fs_rh_white": "(e) surf/rh.white",
    "fs_lh_pial": "(e) surf/lh.pial, or the variant named in fs_lh_pial_variant",
    "fs_rh_pial": "(e) surf/rh.pial, or the variant named in fs_rh_pial_variant",
    "mask": "(f) space-T1w_desc-brain_mask.nii.gz of the run",
    "raw": "(g) the raw series (the first echo when there are several)",
    "raw_json": "the JSON sidecar beside the raw series, when the listing shows one",
}
FLAGS = ("has_preproc", "has_confounds", "has_xfm", "has_fsnative_xfm", "has_fs", "has_mask")
QUALIFY = (*FLAGS, "has_raw")

# column -> pandas dtype. The order is the order of the table.
COLUMNS: dict[str, str] = {
    "dataset": "string",  # accession; the table names datasets and stays out of the repository
    "fmriprep_version": "string",
    "subject": "string",  # label without "sub-"
    "session": "string",  # label without "ses-"; missing when the name has none
    "task": "string",
    "run": "Int64",  # the run index as a number; missing when the name has none
    "run_label": "string",  # the run index as the derivative writes it
    "other_entities": "string",  # acq, ce, rec, dir of the name, as "acq-x_dir-y"; "" for none
    "run_stem": "string",  # the derivative's name up to the run entity
    "qualifies": "bool",
    **dict.fromkeys(QUALIFY, "bool"),
    "has_preproc_boldspace": "bool",  # a preprocessed series without a space entity
    "dup_numeric_run": "bool",  # another run of the derivative has the same number
    "n_func_keys": "Int64",  # keys of the run in the derivative's func/ folder
    "n_echo_keys": "Int64",  # of those, keys with an echo entity
    **{f"{f}_{part}": t for f in FILES for part, t in (("key", "string"), ("size", "Int64"))},
    **{f"{f}_etag": "string" for f in FILES},
    "fsnative_xfm_where": "string",  # "subject" (sub-X/anat/) or "session" (sub-X/ses-Y/anat/)
    "fsnative_xfm_n": "Int64",  # how many such transforms the subject has
    "fs_lh_pial_variant": "string",
    "fs_rh_pial_variant": "string",
    "raw_subject_listed": "bool",  # the raw dataset has a folder for the subject
    "raw_run_label": "string",
    "raw_match": "string",  # "exact", "numeric", "ambiguous" (several raw labels, no exact) or NA
    "raw_n_echoes": "Int64",  # raw magnitude series of the run (1 unless multi-echo)
    "raw_size_all": "Int64",  # bytes of all of them
    "raw_json_inherited_key": "string",  # the nearest sidecar higher up that applies to the run
}
KEY = ("dataset", "subject", "session", "task", "other_entities", "run_label")

# The parts of a dataset's stored listing (OUT/listings/<accession>.tsv.gz): the keys under the
# subject folders of the derivative, under their FreeSurfer folders, under the same subjects of
# the raw dataset, and at the top level of the derivative and of the raw dataset.
SOURCES = ("deriv", "fs", "raw", "deriv_top", "raw_top")


def listing_paths(out: Path, accession: str) -> tuple[Path, Path]:
    """(the listing, its meta file). The meta file is written last and marks the dataset done."""
    return out / "listings" / f"{accession}.tsv.gz", out / "listings" / f"{accession}.json"


def write_listing(out: Path, meta: dict, listing: dict[str, list]) -> None:
    """Store one dataset's listing: the keys first, the meta file last, each by rename."""
    path, meta_path = listing_paths(out, meta["accession"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as f:
        for source in SOURCES:
            for key, size, etag, modified in listing[source]:
                f.write(f"{source}\t{key}\t{size}\t{etag}\t{modified}\n")
    os.replace(tmp, path)
    tmp = meta_path.with_name(meta_path.name + ".part")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, meta_path)


def read_listing(out: Path, accession: str) -> tuple[dict, dict[str, list]]:
    """(meta, listing) of a dataset that list_runs.py has listed; the listing has, per source,
    (key, size, etag) and under "raw_subjects" the subject folders of the raw dataset."""
    path, meta_path = listing_paths(out, accession)
    meta = json.loads(meta_path.read_text())
    listing: dict[str, list] = {source: [] for source in SOURCES}
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            source, key, size, etag, _ = line.rstrip("\n").split("\t")
            listing[source].append((key, int(size), etag))
    listing["raw_subjects"] = meta["raw_subjects"]
    return meta, listing


def parse_name(name: str) -> tuple[dict[str, str], str, str]:
    """(entities, suffix, extension) of a BIDS file name; the extension starts at the first dot."""
    stem, dot, ext = name.partition(".")
    entities: dict[str, str] = {}
    suffix = ""
    for part in stem.split("_"):
        key, dash, value = part.partition("-")
        if dash:
            entities.setdefault(key, value)
        else:
            suffix = part
    return entities, suffix, dot + ext


def run_number(label: str | None) -> int | None:
    """The run index as a number; None when the name has none or it is not made of digits."""
    if label is not None and label.isascii() and label.isdigit():
        return int(label)
    return None


def text_id(entities: dict[str, str]) -> tuple:
    """What tells a run from another, with the run index as it is written."""
    return tuple(entities.get(e) for e in RUN_ENTITIES)


def numeric_id(entities: dict[str, str]) -> tuple:
    """The same with the run index as a number, so that run-1 and run-01 are one run."""
    label = entities.get("run")
    number = run_number(label)
    return (*text_id(entities)[:-1], label if number is None else number)


def _is(entities, suffix, ext, want_suffix, want_ext, **want) -> bool:
    return (
        suffix in want_suffix
        and ext in want_ext
        and all(entities.get(k) == v for k, v in want.items())
    )


def kind_of(entities: dict[str, str], suffix: str, ext: str) -> str | None:
    """Which of the recorded files of a run a name in a derivative's func/ folder is."""
    if _is(entities, suffix, ext, ("bold",), (".nii.gz",), desc="preproc"):
        if entities.get("space") == "T1w":
            return "preproc"
        return "preproc_boldspace" if "space" not in entities else None
    if _is(entities, suffix, ext, ("timeseries", "regressors"), (".tsv",), desc="confounds"):
        return "confounds"
    if _is(entities, suffix, ext, ("xfm",), (".txt",), to="T1w", mode="image"):
        return "xfm" if entities.get("from") == "scanner" else None
    if _is(entities, suffix, ext, ("mask",), (".nii.gz",), space="T1w", desc="brain"):
        return "mask"
    return None


def _first(candidates: list) -> tuple | None:
    """Of several files of one kind: the one without an echo entity, else the lowest echo."""
    ok = [c for c in candidates if c[1] > 0]  # a file of size 0 counts as absent
    if not ok:
        return None
    return min(ok, key=lambda c: (c[3] is not None, run_number(c[3]) or 0, c[0]))


def _set_file(row: dict, name: str, found: tuple | None) -> None:
    row[f"{name}_key"], row[f"{name}_size"], row[f"{name}_etag"] = (
        found[:3] if found else (None, None, None)
    )


def _raw_index(accession: str, raw: list) -> tuple[dict, dict, dict]:
    """Raw series by numeric run id and text label; every raw JSON key with size and ETag; the
    bold sidecars of each folder, parsed."""
    series: dict[tuple, dict[str | None, list]] = defaultdict(lambda: defaultdict(list))
    sidecars: dict[str, tuple[int, str]] = {}
    bold_json: dict[str, list] = defaultdict(list)
    for key, size, etag in raw:
        folder, _, name = key.rpartition("/")
        entities, suffix, ext = parse_name(name)
        if ext == ".json":
            sidecars[key] = (size, etag)
            if suffix == "bold" and size > 0:
                bold_json[folder].append((key, entities))
            continue
        if not folder.endswith("/func") or suffix != "bold" or ext not in (".nii.gz", ".nii"):
            continue
        if "task" not in entities or entities.get("part") not in (None, "mag"):
            continue  # a phase series is not the series that was preprocessed
        series[numeric_id(entities)][entities.get("run")].append(
            (key, size, etag, entities.get("echo"))
        )
    return series, sidecars, bold_json


def _inherited_sidecar(accession: str, entities: dict, bold_json: dict, own: str | None):
    """The nearest JSON higher up that applies to a run by the inheritance rule of BIDS: every
    entity it names is one of the run's, with the run index compared as a number."""
    ours = dict(zip(RUN_ENTITIES, numeric_id(entities), strict=True))
    folders = [accession, f"{accession}/sub-{entities['sub']}"]
    if "ses" in entities:
        folders.append(f"{folders[-1]}/ses-{entities['ses']}")
    folders.append(f"{folders[-1]}/func")
    for depth in range(len(folders) - 1, -1, -1):  # the nearest folder first
        fitting = []
        for key, found in bold_json.get(folders[depth], []):
            theirs = dict(zip(RUN_ENTITIES, numeric_id(found), strict=True))
            if key == own or set(found) - set(RUN_ENTITIES) - {"echo", "part"}:
                continue
            if theirs == ours:
                continue  # a sidecar of this very run (of another echo), not one higher up
            if all(v is None or ours[k] == v for k, v in theirs.items()):
                fitting.append((len(found), key))
        if fitting:
            return max(fitting)[1]  # the one that names the most entities
    return None


def raw_run_ids(accession: str, listing: dict[str, list]) -> set[tuple]:
    """The functional runs of the raw subjects in a listing, each by its numeric id."""
    return set(_raw_index(accession, listing["raw"])[0])


def raw_run_id(raw_key: str) -> tuple:
    """The numeric id of the raw run that a raw key belongs to."""
    return numeric_id(parse_name(raw_key.rsplit("/", 1)[-1])[0])


def build_rows(meta: dict, listing: dict[str, list]) -> list[dict]:
    """The rows of one dataset, from its listing alone.

    meta has accession, prefix (of the derivative), version and fs_root; listing is what
    read_listing() returns.
    """
    accession, prefix = meta["accession"], meta["prefix"]
    runs: dict[tuple, list] = defaultdict(list)
    fsnative: dict[str, list] = defaultdict(list)
    for key, size, etag in listing["deriv"]:
        parts = key[len(prefix) :].split("/")
        if len(parts) < 3:
            continue
        entities, suffix, ext = parse_name(parts[-1])
        if parts[-2] == "func" and "task" in entities and "sub" in entities:
            runs[text_id(entities)].append((key, size, etag, entities, suffix, ext))
        elif parts[-2] == "anat" and _is(
            entities, suffix, ext, ("xfm",), (".txt",), to="fsnative", mode="image"
        ):
            if entities.get("from") == "T1w" and size > 0:
                where = "subject" if len(parts) == 3 else "session"
                fsnative[parts[0]].append((key, size, etag, where, entities.get("ses")))
    fs_files: dict[str, dict[str, tuple]] = defaultdict(dict)
    fs_prefix = prefix + (meta.get("fs_root") or FS_ROOTS[0])
    for key, size, etag in listing["fs"]:
        subject, _, rel = key[len(fs_prefix) :].partition("/")
        if size > 0:
            fs_files[subject][rel] = (key, size, etag)
    raw_series, sidecars, bold_json = _raw_index(
        accession, [*listing["raw"], *listing.get("raw_top", [])]
    )
    raw_subjects = set(listing["raw_subjects"])
    numbers: dict[tuple, int] = defaultdict(int)
    for tid in runs:
        numbers[numeric_id(dict(zip(RUN_ENTITIES, tid, strict=True)))] += 1

    rows = []
    for tid, files in runs.items():
        entities = {e: v for e, v in zip(RUN_ENTITIES, tid, strict=True) if v is not None}
        subject = "sub-" + entities["sub"]
        row: dict = {
            "dataset": accession,
            "fmriprep_version": meta.get("version"),
            "subject": entities["sub"],
            "session": entities.get("ses"),
            "task": entities["task"],
            "run": run_number(entities.get("run")),
            "run_label": entities.get("run"),
            "other_entities": "_".join(
                f"{e}-{entities[e]}" for e in OTHER_ENTITIES if e in entities
            ),
            "run_stem": "_".join(f"{e}-{entities[e]}" for e in RUN_ENTITIES if e in entities),
            "dup_numeric_run": numbers[numeric_id(entities)] > 1,
            "n_func_keys": len(files),
            "n_echo_keys": sum("echo" in f[3] for f in files),
        }
        kinds: dict[str, list] = defaultdict(list)
        for key, size, etag, found, suffix, ext in files:
            kind = kind_of(found, suffix, ext)
            if kind:
                kinds[kind].append((key, size, etag, found.get("echo")))
        for name in ("preproc", "confounds", "xfm", "mask"):
            _set_file(row, name, _first(kinds[name]))
        row["has_preproc_boldspace"] = _first(kinds["preproc_boldspace"]) is not None

        # (d): at subject level, else in the anat folder of the run's session, else any session
        candidates = sorted(
            fsnative.get(subject, []),
            key=lambda c: (c[3] != "subject", c[4] != entities.get("ses"), c[0]),
        )
        _set_file(row, "fsnative_xfm", candidates[0] if candidates else None)
        row["fsnative_xfm_where"] = candidates[0][3] if candidates else None
        row["fsnative_xfm_n"] = len(candidates)

        # (e)
        have = fs_files.get(subject, {})
        _set_file(row, "fs_t1", have.get("mri/T1.mgz"))
        for hemi in ("lh", "rh"):
            _set_file(row, f"fs_{hemi}_white", have.get(f"surf/{hemi}.white"))
            variant = next((v for v in PIAL_VARIANTS if f"surf/{hemi}.{v}" in have), None)
            _set_file(row, f"fs_{hemi}_pial", have.get(f"surf/{hemi}.{variant}"))
            row[f"fs_{hemi}_pial_variant"] = variant

        # (g): the raw run with the same entities, the run index compared as a number
        row["raw_subject_listed"] = subject in raw_subjects
        by_label = raw_series.get(numeric_id(entities), {})
        label = entities.get("run")
        if label in by_label:
            match = "exact"
        elif len(by_label) == 1:
            label, match = next(iter(by_label)), "numeric"
        else:
            match = "ambiguous" if by_label else None
        series = sorted(
            (s for s in by_label.get(label, []) if s[1] > 0) if match != "ambiguous" else [],
            key=lambda s: (run_number(s[3]) or 0, s[0]),
        )
        _set_file(row, "raw", series[0] if series else None)
        row["raw_run_label"] = label if series else None
        row["raw_match"] = match if series or match == "ambiguous" else None
        row["raw_n_echoes"] = len(series) if series else None
        row["raw_size_all"] = sum(s[1] for s in series) if series else None
        own = None
        if series:
            own = series[0][0].removesuffix(".gz").removesuffix(".nii") + ".json"
            own = own if sidecars.get(own, (0, ""))[0] > 0 else None
        _set_file(row, "raw_json", (own, *sidecars[own]) if own else None)
        row["raw_json_inherited_key"] = (
            _inherited_sidecar(accession, entities, bold_json, own) if series else None
        )

        row["has_preproc"] = row["preproc_key"] is not None
        row["has_confounds"] = row["confounds_key"] is not None
        row["has_xfm"] = row["xfm_key"] is not None
        row["has_fsnative_xfm"] = row["fsnative_xfm_key"] is not None
        row["has_fs"] = all(
            row[f"fs_{f}_key"] is not None
            for f in ("t1", "lh_white", "rh_white", "lh_pial", "rh_pial")
        )
        row["has_mask"] = row["mask_key"] is not None
        row["has_raw"] = row["raw_key"] is not None
        row["qualifies"] = all(row[f] for f in QUALIFY) and not row["dup_numeric_run"]
        rows.append(row)
    return rows


def sort_key(row: dict) -> tuple:
    """Table order: dataset, subject, session, task, other entities, run as a number."""
    run = row["run"]
    return (
        row["dataset"],
        row["subject"],
        row["session"] or "",
        row["task"],
        row["other_entities"],
        -1 if run is None or pd.isna(run) else int(run),
        row["run_label"] or "",
    )


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as a frame with the columns and types of COLUMNS, in table order."""
    rows = sorted(rows, key=sort_key)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    return frame.astype(COLUMNS)


def read_runs(path: str | Path) -> pd.DataFrame:
    """The table of runs with the column types of COLUMNS; refuses a table of other columns."""
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(COLUMNS):
        missing = sorted(set(COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(COLUMNS))
        raise ValueError(
            f"{Path(path).name}: not the table of list_runs.py (missing {missing}, extra {extra})"
        )
    return frame.astype(COLUMNS)
