"""Check the table of list_runs.py against its own rules, and against the listings it came from.

usage: check_listing.py OUT [--no-listings] [--example DATA FS] [--head N SEED]

OUT is the directory that list_runs.py wrote. No request is sent unless --head is given. Exit
status 0 when every check holds, 1 when one fails (each failure is printed with up to five
rows), 2 for usage.

The table alone:
  the columns and their types are those of list_table.COLUMNS, and the run number is stored
  as an integer, not as text; no run is listed twice; a run number is the number of its label;
  within a session the runs are in the order of their numbers (2 before 10); a qualifying run
  has every one of the files (a) to (g), each with a key, a size above 0 and an ETag, and a run
  that has them all qualifies unless its number is not unique; every key carries the subject,
  session, task and run number of its row (read from the key with regular expressions, not
  with the parser that made the table), and a raw run has the number of its row as a number,
  whatever its label; the labels of a "numeric" match differ as text and are equal as numbers.
  Then the rules themselves, on a made-up listing: run-1 finds run-01, run-1 does not find
  run-10, a file of size 0 is absent, run-1 beside run-01 in a derivative qualifies neither.

Against OUT/listings (skipped with --no-listings):
  every key of the table is in the listing of its dataset with the same size and ETag; the
  table has one row for each run name that a regular expression finds in the func/ folders;
  (a), (b), (c) and (f) of a row are what the listing shows under the name built from the row
  (with or without an echo entity before the ending);
  no dataset is in datasets.csv twice and none is missing from it.

--head N SEED: N qualifying runs are drawn with the seed, and for the files of each (the two
series, confounds, both transforms, the mask, T1.mgz, the raw sidecar when there is one) a HEAD
request is sent, anonymously and at most 4 at a time. Each must answer 200 with the size and
the ETag of the table, which is what a later step relies on to tell a changed file.

--example DATA FS: the worked example (ds000102 sub-01, as pipeline/fetch.py and
pipeline/fetch_freesurfer.py fetched it into DATA and FS). Its two runs qualify, and every
file of theirs that is on disk has the size of the table and, where the ETag is an MD5 (no
dash in it), that MD5.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from list_table import (
    COLUMNS,
    FILES,
    KEY,
    QUALIFY,
    build_rows,
    listing_paths,
    read_listing,
    read_runs,
    to_frame,
)

FS_FILES = ("fs_t1", "fs_lh_white", "fs_rh_white", "fs_lh_pial", "fs_rh_pial")
PER_FLAG = {
    "has_preproc": ("preproc",),
    "has_confounds": ("confounds",),
    "has_xfm": ("xfm",),
    "has_fsnative_xfm": ("fsnative_xfm",),
    "has_fs": FS_FILES,
    "has_mask": ("mask",),
    "has_raw": ("raw",),
}
ENDINGS = {
    "preproc": ("_space-T1w_desc-preproc_bold.nii.gz",),
    "confounds": ("_desc-confounds_timeseries.tsv", "_desc-confounds_regressors.tsv"),
    "xfm": ("_from-scanner_to-T1w_mode-image_xfm.txt",),
    "mask": ("_space-T1w_desc-brain_mask.nii.gz",),
    "fsnative_xfm": ("_from-T1w_to-fsnative_mode-image_xfm.txt",),
    "raw": ("_bold.nii.gz", "_bold.nii"),
    "raw_json": ("_bold.json",),
}
HEAD_FILES = ("preproc", "confounds", "xfm", "mask", "fsnative_xfm", "fs_t1", "raw", "raw_json")
RUN_FILES = ("preproc", "confounds", "xfm", "mask", "raw", "raw_json")  # named after the run
STEM = re.compile(
    r"^(sub-[^_]+(?:_ses-[^_]+)?_task-[^_]+(?:_acq-[^_]+)?(?:_ce-[^_]+)?(?:_rec-[^_]+)?"
    r"(?:_dir-[^_]+)?(?:_run-[^_]+)?)_"
)

failures: list[str] = []


def check(name: str, bad, n: int | None = None) -> None:
    """Record one check: bad is a frame or list of offending rows, or a truth value."""
    count = len(bad) if hasattr(bad, "__len__") else int(bool(bad))
    of = "" if n is None else f" of {n}"
    print(f"{'ok  ' if count == 0 else 'FAIL'} {name}" + ("" if count == 0 else f": {count}{of}"))
    if count:
        failures.append(name)
        shown = bad.head(5).to_string() if isinstance(bad, pd.DataFrame) else str(list(bad)[:5])
        print("     " + shown.replace("\n", "\n     "))


def run_of(key: str) -> tuple[bool, int | None]:
    """(the name has a run entity, its number) read from a key with a regular expression."""
    m = re.search(r"_run-([^_/]+)_[^/]*$", key)
    if not m:
        return False, None
    return True, int(m.group(1)) if re.fullmatch(r"[0-9]+", m.group(1)) else None


def table_rules(stored: pd.DataFrame, runs: pd.DataFrame) -> None:
    check("columns are those of list_table.COLUMNS", list(stored.columns) != list(COLUMNS))
    types = {c: str(stored[c].dtype) for c in stored.columns if c in COLUMNS}
    wrong = [f"{c}: {t}" for c, t in types.items() if t.lower() != COLUMNS[c].lower()]
    check("column types as stored are those of list_table.COLUMNS", wrong)
    check(
        "the run number is stored as an integer",
        not pd.api.types.is_integer_dtype(stored["run"]),
    )
    check("no run is listed twice", runs[runs.duplicated(list(KEY), keep=False)], len(runs))
    by_number = ["dataset", "subject", "session", "task", "other_entities", "run"]
    twice = runs[runs.duplicated(by_number, keep=False) & runs["run"].notna()]
    check("a run number that occurs twice is flagged", twice[~twice["dup_numeric_run"]])
    check("a flagged run does not qualify", runs[runs["dup_numeric_run"] & runs["qualifies"]])

    digits = runs["run_label"].fillna("").str.fullmatch(r"[0-9]+")
    expect = runs["run_label"].where(digits).astype("Float64").astype("Int64")
    same = (runs["run"] == expect).fillna(False) | (runs["run"].isna() & expect.isna())
    check("the run number is the number of its label", runs[~same], len(runs))

    group = ["dataset", "subject", "session", "task", "other_entities"]
    numbered = runs[runs["run"].notna()]
    falls = numbered.groupby(group, dropna=False)["run"].diff().lt(0)
    check("runs are in the order of their numbers, not of their labels", numbered[falls])

    for flag, files in PER_FLAG.items():
        have = pd.Series(True, index=runs.index)
        for f in files:
            have &= (
                runs[f"{f}_key"].notna()
                & runs[f"{f}_size"].gt(0).fillna(False)
                & runs[f"{f}_etag"].fillna("").ne("")
            )
        check(f"{flag} says that key, size above 0 and ETag are there", runs[runs[flag] != have])
    for f in FILES:
        partial = runs[f"{f}_key"].notna() != runs[f"{f}_size"].notna()
        partial |= runs[f"{f}_key"].notna() != runs[f"{f}_etag"].notna()
        check(f"{f}: key, size and ETag are there together or not at all", runs[partial])
    expect_q = runs[list(QUALIFY)].all(axis=1) & ~runs["dup_numeric_run"]
    check("qualifies means (a) to (g) and a unique run number", runs[runs["qualifies"] != expect_q])

    for f in RUN_FILES:
        have = runs[runs[f"{f}_key"].notna()]
        key = have[f"{f}_key"]
        name = key.str.rsplit("/", n=1).str[-1]
        folder_ok = pd.Series(
            [f"/sub-{s}/" in k for s, k in zip(have["subject"], key, strict=True)],
            index=have.index,
        )
        entity_ok = pd.Series(
            [
                n.startswith(f"sub-{s}_")
                and (f"_ses-{e}_" in n if isinstance(e, str) else "_ses-" not in n)
                and f"_task-{t}_" in n
                for n, s, e, t in zip(
                    name, have["subject"], have["session"], have["task"], strict=True
                )
            ],
            index=have.index,
        )
        check(
            f"{f}: the key names the subject, session and task of its row",
            have[~(folder_ok & entity_ok)],
        )
        check(f"{f}: the key has the ending of its kind", have[~key.str.endswith(ENDINGS[f])])
        found = [run_of(k) for k in key]
        number_ok = pd.Series(
            [
                (not has and pd.isna(r))
                or (has and n is not None and not pd.isna(r) and n == int(r))
                or (has and n is None and pd.isna(r))
                for (has, n), r in zip(found, have["run"], strict=True)
            ],
            index=have.index,
        )
        check(f"{f}: the key has the run number of its row, as a number", have[~number_ok])
    for f in ("fsnative_xfm", *FS_FILES):
        have = runs[runs[f"{f}_key"].notna()]
        ok = [f"/sub-{s}/" in k for s, k in zip(have["subject"], have[f"{f}_key"], strict=True)]
        check(f"{f}: the key names the subject of its row", have[~pd.Series(ok, index=have.index)])
    raw = runs[runs["raw_key"].notna()]
    starts = [
        k.startswith(f"{d}/sub-{s}/")
        for k, d, s in zip(raw["raw_key"], raw["dataset"], raw["subject"], strict=True)
    ]
    check(
        "raw: the key lies in the raw dataset of its row", raw[~pd.Series(starts, index=raw.index)]
    )
    as_text = raw["run_label"].fillna("") == raw["raw_run_label"].fillna("")
    check("an exact match has equal labels", raw[(raw["raw_match"] == "exact") & ~as_text])
    check(
        "a numeric match has labels that differ as text",
        raw[(raw["raw_match"] == "numeric") & as_text],
    )
    check(
        "a raw run was matched exactly or by number",
        raw[~raw["raw_match"].isin(["exact", "numeric"])],
    )


def rules_on_a_made_up_listing() -> None:
    """The rules, tried on names that the real listings may not have."""
    p, fs = "fmriprep/dsX-fmriprep/", "fmriprep/dsX-fmriprep/sourcedata/freesurfer/"

    def deriv(stem: str, size: int = 9) -> list:
        func = [
            "space-T1w_desc-preproc_bold.nii.gz",
            "desc-confounds_timeseries.tsv",
            "from-scanner_to-T1w_mode-image_xfm.txt",
            "space-T1w_desc-brain_mask.nii.gz",
        ]
        sub = stem.split("_")[0]
        return [(f"{p}{sub}/func/{stem}_{name}", size, "e") for name in func]

    listing = {
        "deriv": [
            *deriv("sub-01_task-a_run-1"),
            *deriv("sub-01_task-a_run-2"),
            *deriv("sub-01_task-a_run-10"),
            *deriv("sub-01_task-b_run-3"),
            *deriv("sub-01_task-c_run-1", size=0),
            *deriv("sub-02_task-a_run-1"),
            *deriv("sub-02_task-a_run-01"),
            (f"{p}sub-01/anat/sub-01_from-T1w_to-fsnative_mode-image_xfm.txt", 9, "e"),
            (f"{p}sub-02/ses-x/anat/sub-02_ses-x_from-T1w_to-fsnative_mode-image_xfm.txt", 9, "e"),
        ],
        "fs": [
            (f"{fs}{sub}/{rel}", 9, "e")
            for sub in ("sub-01", "sub-02")
            for rel in ("mri/T1.mgz", "surf/lh.white", "surf/rh.white", "surf/lh.pial.T1")
        ]
        + [(f"{fs}sub-01/surf/rh.pial.T1", 9, "e"), (f"{fs}sub-01/surf/rh.pial.T2", 9, "e")],
        "raw": [
            ("dsX/sub-01/func/sub-01_task-a_run-01_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-a_run-01_bold.json", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-a_run-10_echo-2_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-a_run-10_echo-1_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-a_run-10_echo-1_part-phase_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-b_run-03_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-b_run-003_bold.nii.gz", 9, "e"),
            ("dsX/sub-01/func/sub-01_task-c_run-1_bold.nii.gz", 9, "e"),
            ("dsX/sub-02/func/sub-02_task-a_run-1_bold.nii.gz", 9, "e"),
        ],
        "raw_top": [("dsX/task-a_bold.json", 9, "e"), ("dsX/task-b_run-4_bold.json", 9, "e")],
        "raw_subjects": ["sub-01", "sub-02"],
    }
    meta = {
        "accession": "dsX",
        "prefix": p,
        "version": "21.0.2",
        "fs_root": "sourcedata/freesurfer/",
    }
    t = to_frame(build_rows(meta, listing)).set_index(["subject", "task", "run_label"])

    def cell(subject: str, task: str, label: str, column: str):
        value = t.loc[(subject, task, label), column]
        return None if pd.isna(value) else value

    def ends(value, ending: str) -> bool:
        return isinstance(value, str) and value.endswith(ending)

    one, ten, three = ("01", "a", "1"), ("01", "a", "10"), ("01", "b", "3")
    made_up = {
        "run-1 finds run-01 and qualifies": lambda: (
            cell(*one, "qualifies") and cell(*one, "raw_match") == "numeric"
        ),
        "run-1 takes run-01, not run-10": lambda: ends(
            cell(*one, "raw_key"), "_run-01_bold.nii.gz"
        ),
        "its sidecar is found": lambda: ends(cell(*one, "raw_json_key"), "_run-01_bold.json"),
        "run-2 has no raw run and does not qualify": lambda: (
            not cell("01", "a", "2", "has_raw") and not cell("01", "a", "2", "qualifies")
        ),
        "run-10 takes the first echo of run-10": lambda: ends(
            cell(*ten, "raw_key"), "_run-10_echo-1_bold.nii.gz"
        ),
        "run-10 counts two echoes, not the phase": lambda: cell(*ten, "raw_n_echoes") == 2,
        "without its own sidecar the one higher up is named": lambda: (
            cell(*ten, "raw_json_inherited_key") == "dsX/task-a_bold.json"
            and cell(*ten, "raw_json_key") is None
        ),
        "a sidecar for another run number does not apply": lambda: (
            cell(*three, "raw_json_inherited_key") is None
        ),
        "two raw labels of one number and none exact: no raw run": lambda: (
            cell(*three, "raw_match") == "ambiguous" and not cell(*three, "has_raw")
        ),
        "files of size 0 are absent": lambda: (
            not cell("01", "c", "1", "has_preproc") and not cell("01", "c", "1", "qualifies")
        ),
        "run-1 beside run-01 qualifies neither": lambda: (
            not cell("02", "a", "1", "qualifies")
            and not cell("02", "a", "01", "qualifies")
            and cell("02", "a", "1", "dup_numeric_run")
        ),
        "a pial surface under its FreeSurfer 7 name is found": lambda: (
            cell(*one, "fs_lh_pial_variant") == "pial.T1" and cell(*one, "has_fs")
        ),
        "the T2 pial is taken before the T1 pial": lambda: (
            cell(*one, "fs_rh_pial_variant") == "pial.T2"
        ),
        "a missing surface is missing": lambda: not cell("02", "a", "1", "has_fs"),
        "the fsnative transform is found in a session folder": lambda: (
            cell("02", "a", "1", "fsnative_xfm_where") == "session"
        ),
        "run numbers order the table: 1, 2, 10": lambda: (
            list(t.loc[("01", "a")].index) == ["1", "2", "10"]
        ),
    }
    for name, rule in made_up.items():
        try:
            ok = bool(rule())
        except Exception as exc:  # a rule that cannot be evaluated has failed
            ok = False
            name = f"{name} ({type(exc).__name__})"
        check(f"made-up listing: {name}", not ok)


def against_listings(out: Path, runs: pd.DataFrame) -> None:
    datasets = pd.read_csv(out / "datasets.csv", dtype={"dataset": str})
    check(
        "no dataset is in datasets.csv twice", datasets[datasets["dataset"].duplicated(keep=False)]
    )
    done = sorted(p.name[: -len(".json")] for p in (out / "listings").glob("*.json"))
    check("datasets.csv has the datasets that were listed", sorted(datasets["dataset"]) != done)
    check("the table has no dataset that was not listed", sorted(set(runs["dataset"]) - set(done)))
    wrong, absent, count, built = [], [], [], []
    by_name = {
        "has_preproc": ENDINGS["preproc"],
        "has_confounds": ENDINGS["confounds"],
        "has_xfm": ENDINGS["xfm"],
        "has_mask": ENDINGS["mask"],
    }
    for accession in done:
        if not listing_paths(out, accession)[0].exists():
            absent.append(accession)
            continue
        meta, listing = read_listing(out, accession)
        known = {
            key: (size, etag)
            for source in ("deriv", "fs", "raw", "raw_top")
            for key, size, etag in listing[source]
        }
        rows = runs[runs["dataset"] == accession]
        for f in FILES:
            have = rows[rows[f"{f}_key"].notna()]
            for key, size, etag in zip(
                have[f"{f}_key"], have[f"{f}_size"], have[f"{f}_etag"], strict=True
            ):
                if known.get(key) != (int(size), etag):
                    wrong.append((f, key))
        stems, unread = set(), 0
        for key, _, _ in listing["deriv"]:
            folder, _, name = key.rpartition("/")
            if not folder.endswith("/func") or "_task-" not in name:
                continue
            m = STEM.match(name)
            unread += m is None
            if m:
                stems.add(m.group(1))
        if unread or stems != set(rows["run_stem"]) or len(stems) != len(rows):
            count.append((accession, len(stems), len(rows), unread))
        shown = set()
        for key, size, _ in listing["deriv"]:
            folder, _, name = key.rpartition("/")
            m = STEM.match(name) if folder.endswith("/func") and size > 0 else None
            rest = re.sub(r"^_echo-[^_]+", "", name[m.end() - 1 :]) if m else ""
            for flag, endings in by_name.items():
                if rest in endings:
                    shown.add((folder, m.group(1), flag))
        for row in rows.itertuples(index=False):
            session = f"ses-{row.session}/" if isinstance(row.session, str) else ""
            folder = f"{meta['prefix']}sub-{row.subject}/{session}func"
            for flag in by_name:
                if ((folder, row.run_stem, flag) in shown) != getattr(row, flag):
                    built.append((flag, folder, row.run_stem))
    check("every listing that is marked done is there", absent)
    check("every key of the table is in its listing with that size and ETag", wrong)
    check("one row for each run name that a regular expression finds in func/", count)
    check("(a), (b), (c), (f) are what the listing shows under the name built from the row", built)


def against_the_buckets(runs: pd.DataFrame, n: int, seed: int) -> None:
    """HEAD requests for the files of n qualifying runs drawn with a seed."""
    from concurrent.futures import ThreadPoolExecutor

    import list_s3 as s3  # only here: the other checks send no request

    q = runs[runs["qualifies"]]
    sample = q.sample(min(n, len(q)), random_state=seed)
    asked = []
    for row in sample.to_dict("records"):
        for f in HEAD_FILES:
            if isinstance(row[f"{f}_key"], str):
                base = s3.RAW if f.startswith("raw") else s3.DERIV
                asked.append((f, base, row[f"{f}_key"], int(row[f"{f}_size"]), row[f"{f}_etag"]))
    with ThreadPoolExecutor(s3.MAX_REQUESTS) as pool:
        answers = list(pool.map(lambda a: s3.head(a[1], a[2]), asked))
    bad = [
        (f, answer)
        for (f, _, _, size, etag), answer in zip(asked, answers, strict=True)
        if answer != (200, size, etag)
    ]
    name = f"{len(asked)} files of {len(sample)} qualifying runs (seed {seed})"
    check(f"the buckets answer HEAD with 200 and the size and ETag of the table: {name}", bad)
    print(f"     {s3.requests_made()} requests")


def md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def worked_example(runs: pd.DataFrame, data: Path, fs: Path) -> None:
    rows = runs[(runs["dataset"] == "ds000102") & (runs["subject"] == "01")]
    check("worked example: two runs, numbered 1 and 2", list(rows["run"]) != [1, 2])
    check("worked example: both runs qualify", rows[~rows["qualifies"]])
    local = {
        "preproc": "deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz",
        "confounds": "deriv/run-{run}_desc-confounds_timeseries.tsv",
        "xfm": "deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt",
        "mask": "deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz",
        "fsnative_xfm": "deriv/anat_from-T1w_to-fsnative_mode-image_xfm.txt",
        "raw": "raw/run-{run}_bold.nii.gz",
    }
    paths = {f: (data, name) for f, name in local.items()}
    paths |= {"fs_t1": (fs, "sub-01/mri/T1.mgz")}
    paths |= {
        f"fs_{h}_{s}": (fs, f"sub-01/surf/{h}.{s}") for h in ("lh", "rh") for s in ("white", "pial")
    }
    bad, n = [], 0
    for row in rows.to_dict("records"):
        for f, (root, name) in paths.items():
            path = root / name.format(run=row["run"])
            if not path.exists() or pd.isna(row[f"{f}_key"]):
                bad.append((f, "not on disk or not in the table"))
                continue
            n += 1
            etag = row[f"{f}_etag"]
            if path.stat().st_size != row[f"{f}_size"]:
                bad.append((f, "size"))
            elif "-" not in etag and md5(path) != etag:
                bad.append((f, "MD5"))
    check(f"worked example: {n} files on disk have the size and MD5 of the table", bad)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help") or args[0].startswith("--"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out = Path(args[0])
    example = None
    if "--example" in args:
        at = args.index("--example")
        if len(args) < at + 3:
            print(__doc__.strip(), file=sys.stderr)
            sys.exit(2)
        example = (Path(args[at + 1]), Path(args[at + 2]))
    head = None
    if "--head" in args:
        at = args.index("--head")
        if len(args) < at + 3:
            print(__doc__.strip(), file=sys.stderr)
            sys.exit(2)
        head = (int(args[at + 1]), int(args[at + 2]))
    stored = pd.read_parquet(out / "runs.parquet")
    runs = read_runs(out / "runs.parquet")
    n_datasets, n_qualifying = runs["dataset"].nunique(), int(runs["qualifies"].sum())
    print(f"{len(runs)} runs of {n_datasets} datasets, {n_qualifying} qualifying")
    table_rules(stored, runs)
    rules_on_a_made_up_listing()
    if "--no-listings" not in args:
        against_listings(out, runs)
    if example:
        worked_example(runs, *example)
    if head:
        against_the_buckets(runs, *head)
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("every check holds")


if __name__ == "__main__":
    main()
