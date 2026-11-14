"""Choose the runs that stage 1 renders: six hard cases by head motion and two quiet comparisons.

usage: select_runs.py LISTING MOTION OUT [--no-headers]
       select_runs.py --self-check

Step 2 of stage 1 (README.md of the prototype, "Next steps"), its second half. LISTING is the
directory that list_runs.py wrote (runs.parquet), MOTION the one that rank_motion.py wrote
(motion.parquet), OUT a directory outside the repository: what is written there names datasets.
--self-check applies the rule to two made-up tables whose answers are known; it reads no file
and sends no request.

---- the rule ----
Whose rule this is. A default of the session that wrote this script on 2026-11-14 (UTC), not a
decision of Tomas Knapen. He decided that the hard cases are the runs with the most head motion
(decision 2 of the README). How many, by which measure and how many per dataset is open
question 4 of the README, which he has not answered. The rule was written, as this text and as
the code between the same two marks further down, before any confounds table was read and
before any image of these runs existed. rank_motion.py records a digest of the two marked parts
at every call and this script records it with the selection, so a change of the rule after the
tables were read shows.

What the ranking is. fMRIPrep's own estimate of head motion. The rule selects for motion and
for nothing else that can go wrong in preprocessing: not for distortion, not for a failed
coregistration, not for dropout, not for spikes that the motion estimate does not follow.

1. Value. The ranking value of a run is the mean framewise displacement of its confounds table,
   over the volumes that have a value (the first volume has none). Recorded beside it and not
   used by the rule: the maximum, the share of those volumes above 0.5 mm, the median, the
   number of volumes and standardised DVARS. Neither the listing nor the confounds table gives
   a TR or a voxel size; both are read from the headers of the two series, for the runs that
   the walks below come to (rule 5).
2. Ranked runs. A run is ranked when it qualifies (list_table.py), its confounds table was read
   and is the file of the listing (size and ETag), the table has a framewise displacement
   column and at least 50 volumes (rows), the mean is finite and above 0, and the raw run is
   one series (one echo). Every other run is not ranked, and the reason is recorded.
3. Order. Mean framewise displacement, highest first. Runs with equal means go in the order of
   the table of list_runs.py: dataset, subject, session, task, other entities, run index as a
   number, run label. No random number is used anywhere.
4. Size limit. The two series of a run (the T1w-space series and the raw series) are together
   at most 1,500,000,000 bytes, so that they can be fetched on the machine this was made on.
   The limit biases the selection against long and multiband runs. So every run that a walk
   passes over is written down with its rank and size, the number of the 50 highest-ranked
   runs that are over the limit is reported, and so is the selection that the same walk makes
   without the limit.
5. Volume check. The first bytes of both series of a run are read (two ranged requests). A run
   whose T1w-space series and raw series do not both have as many volumes as its confounds
   table has rows is passed over, and so is one whose header cannot be read: the two streams
   are compared volume by volume, and the raw bucket holds the latest state of a dataset, which
   need not be the one that was preprocessed. With --no-headers nothing is read and the check
   is not made; the table says which.
6. Hard cases: six. Walk down the order of rule 3 and take a run unless it is over the size
   limit, its subject already has a hard case (one run per subject), its dataset already has
   two, or it fails the volume check. Stop at six.
7. Quiet comparisons: two. For the first hard case, then for the second: among the ranked runs
   of the same dataset whose subject has neither a hard case nor a quiet comparison, the run
   with the lowest mean framewise displacement (equal means in table order) that is within the
   size limit and passes the volume check. When the first two hard cases share a dataset, both
   comparisons come from it, from two subjects. When a dataset has no such run, the dataset of
   the next hard case (the third, then the fourth, ...) takes its place, and that is recorded.
8. Ids. The hard cases are H01 to H06 in the order of rule 3. The quiet comparisons are Q01 and
   Q02 in the order of the hard cases whose dataset they come from. The ids name no dataset,
   subject or task; the table that maps them stays in OUT.
9. Sessions. What gets rendered is the whole session of a selected run, since the one
   registration of a session is that of its anchor run, which may be another run. So every
   qualifying run of the session is recorded with the selected run, ranked or not, in table
   order, as <id>-01, <id>-02, ... A dataset without ses- labels has one session per subject.
   The size limit and the volume check apply to the selected run only.
---- end of the rule ----

Written to OUT (all but summary.txt name datasets):

  selection.parquet   one row per qualifying run of the eight sessions; SELECTION_COLUMNS below
                      has the columns and read_selection() reads it with their types
  ids.csv             the eight ids with dataset, subject, session, task and run
  passed_over.csv     every run that a walk came to and did not take, with rank, size and why
  ranked.parquet      every qualifying run with its rank or the reason why it has none
  headers.json        what was read from the headers of the series, by key and ETag; a second
                      call reads no header twice
  summary.txt         counts, the distribution, the eight runs by id; it names no dataset
  invocations.jsonl   time, digest of the rule and requests of each call

Requests: none for the ranking itself. The volume check sends two ranged requests (8 kB each at
most) per run that a walk comes to and would otherwise take, through fetch() of rank_motion.py:
anonymous, read-only, at most 4 at a time.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from list_table import COLUMNS, KEY, read_runs, sort_key
from rank_motion import (
    DERIV,
    FD_OVER_MM,
    RAW,
    STATUS_OK,
    VALUE_COLUMNS,
    fetch,
    read_motion,
    requests_made,
    rule_digest,
)

# ---- the rule ----
MIN_VOLUMES = 50  # rule 2
LIMIT_BYTES = 1_500_000_000  # rule 4: T1w-space series plus raw series of one run
N_HARD = 6  # rule 6
PER_SUBJECT = 1  # rule 6; the code below knows no other value
PER_DATASET = 2  # rule 6
N_QUIET = 2  # rule 7
TOP = 50  # rule 4: how many of the highest-ranked runs are counted against the limit

PASS_SIZE = "over the size limit"
PASS_SUBJECT = "subject has a hard case"
PASS_DATASET = "dataset has two hard cases"
PASS_VOLUMES = "volume check"


def why_not_ranked(row: dict) -> str | None:
    """Rule 2: the reason why a run is not ranked; None when it is."""
    if not row["qualifies"]:
        return "does not qualify"
    if pd.isna(row["status"]):
        return "confounds table not read"
    if row["status"] != STATUS_OK:
        return f"confounds table: {row['status']}"
    if pd.isna(row["raw_n_echoes"]) or row["raw_n_echoes"] != 1:
        return "several raw echoes"
    if pd.isna(row["n_volumes"]) or row["n_volumes"] < MIN_VOLUMES:
        return f"fewer than {MIN_VOLUMES} volumes"
    mean = row["mean_fd"]
    if pd.isna(mean) or not math.isfinite(mean) or mean <= 0:
        return "no positive mean framewise displacement"
    return None


def rank_runs(table: pd.DataFrame) -> pd.DataFrame:
    """Rules 2 and 3. The table must be in the order of list_runs.py. Adds order (the position
    in the table), not_ranked (the reason, missing for a ranked run) and rank (1 is the highest
    mean framewise displacement; missing for a run that is not ranked)."""
    table = table.reset_index(drop=True)
    table["order"] = np.arange(len(table))
    table["mean_fd"] = table["mean_fd"].astype("float64")
    table["not_ranked"] = pd.array(
        [why_not_ranked(row) for row in table.to_dict("records")], dtype="string"
    )
    ranked = table[table["not_ranked"].isna()].sort_values(
        ["mean_fd", "order"], ascending=[False, True], kind="mergesort"
    )
    table["rank"] = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index).reindex(
        table.index
    )
    table["rank"] = table["rank"].astype("Int64")
    return table


def walk_hard(table: pd.DataFrame, limit: float = LIMIT_BYTES, volumes_ok=None):
    """Rule 6. Returns (the positions of the hard cases, in order; the runs passed over).

    volumes_ok(position) is the volume check of rule 5: "" when the run passes, else what is
    wrong. It is asked only for a run that would otherwise be taken. None: no check.
    """
    taken: list[int] = []
    passed: list[dict] = []
    subjects: set[tuple] = set()
    per_dataset: Counter = Counter()
    ranked = table[table["rank"].notna()].sort_values("rank")
    for at, row in zip(ranked.index, ranked.to_dict("records"), strict=True):
        if len(taken) == N_HARD:
            break
        subject = (row["dataset"], row["subject"])
        reasons, detail = [], ""
        if row["series_bytes"] > limit:
            reasons.append(PASS_SIZE)
        if subject in subjects:
            reasons.append(PASS_SUBJECT)
        if per_dataset[row["dataset"]] >= PER_DATASET:
            reasons.append(PASS_DATASET)
        if not reasons and volumes_ok is not None:
            detail = volumes_ok(at)
            if detail:
                reasons.append(PASS_VOLUMES)
        if reasons:
            passed.append(
                {"at": at, "walk": "hard", "reasons": "; ".join(reasons), "detail": detail}
            )
            continue
        taken.append(at)
        subjects.add(subject)
        per_dataset[row["dataset"]] += 1
    return taken, passed


def walk_quiet(table: pd.DataFrame, hard: list[int], limit: float = LIMIT_BYTES, volumes_ok=None):
    """Rule 7. Returns ([(position of the quiet run, number of its hard case from 1)], the runs
    passed over, notes on hard cases whose dataset gave no comparison)."""
    taken: list[tuple[int, int]] = []
    passed: list[dict] = []
    notes: list[str] = []
    used = {(table.at[at, "dataset"], table.at[at, "subject"]) for at in hard}
    for number, at_hard in enumerate(hard, start=1):
        if len(taken) == N_QUIET:
            break
        dataset = table.at[at_hard, "dataset"]
        pool = table[(table["dataset"] == dataset) & table["rank"].notna()].sort_values(
            ["mean_fd", "order"], kind="mergesort"
        )
        found = None
        for at, row in zip(pool.index, pool.to_dict("records"), strict=True):
            if (dataset, row["subject"]) in used:
                continue  # a subject with a hard case or a quiet comparison is no candidate
            walk = f"quiet for H{number:02d}"
            if row["series_bytes"] > limit:
                passed.append({"at": at, "walk": walk, "reasons": PASS_SIZE, "detail": ""})
                continue
            detail = volumes_ok(at) if volumes_ok is not None else ""
            if detail:
                passed.append({"at": at, "walk": walk, "reasons": PASS_VOLUMES, "detail": detail})
                continue
            found = at
            break
        if found is None:
            notes.append(f"the dataset of H{number:02d} has no run for a quiet comparison")
            continue
        taken.append((found, number))
        used.add((dataset, table.at[found, "subject"]))
    return taken, passed, notes


def session_mask(table: pd.DataFrame, at: int) -> pd.Series:
    """Rule 9: which runs of the table belong to the session of the run at a position."""
    row = table.loc[at]
    same = (table["dataset"] == row["dataset"]) & (table["subject"] == row["subject"])
    if pd.isna(row["session"]):
        same &= table["session"].isna()
    else:
        same &= (table["session"] == row["session"]).fillna(False)
    return same.astype(bool)


def session_runs(table: pd.DataFrame, at: int) -> pd.DataFrame:
    """Rule 9: the qualifying runs of the session of the run at a position, in table order."""
    return table[session_mask(table, at) & table["qualifies"]].sort_values("order")


# ---- end of the rule ----

HEAD_BYTES = 8192  # of a gzip series; the 348 bytes of the header come out of far fewer
HEADER_COLUMNS = {"matrix": "string", "voxel_mm": "string", "n_volumes": "Int64", "tr_s": "float64"}

# column -> pandas dtype of selection.parquet, in order; the columns of runs.parquet follow them
SELECTION_COLUMNS: dict[str, str] = {
    "id": "string",  # H01 to H06, Q01, Q02: the selected run whose session the row belongs to
    "kind": "string",  # "hard" or "quiet"
    "compares_with": "string",  # for a quiet run: the hard case whose dataset it comes from
    "run_id": "string",  # <id>-NN: the run's place among the qualifying runs of its session
    "selected": "bool",  # the run the rule chose; the other rows are the rest of its session
    "session_runs": "Int64",  # qualifying runs of the session
    "session_runs_not_qualifying": "Int64",  # runs of the session in the derivative that do not
    "session_bytes": "Int64",  # the two series of all qualifying runs of the session
    "subject_sessions": "Int64",  # sessions of the subject with a qualifying run, this one included
    "subject_runs": "Int64",  # qualifying runs of the subject in all its sessions
    "series_bytes": "Int64",  # the two series of this run (all raw echoes)
    "rank": "Int64",  # by mean framewise displacement among the ranked runs; 1 is the highest
    "not_ranked": "string",  # why the run has no rank; missing when it has one
    "status": "string",  # of the confounds table, from rank_motion.py
    **VALUE_COLUMNS,  # what rank_motion.py measured: n_volumes, mean_fd, max_fd, ...
    "volumes_checked": "bool",  # rule 5 was applied to this run (selected runs only)
    **{f"{s}_{c}": t for s in ("preproc", "raw") for c, t in HEADER_COLUMNS.items()},
}


def read_selection(path: str | Path) -> pd.DataFrame:
    """selection.parquet with the column types of SELECTION_COLUMNS and of list_table.COLUMNS."""
    frame = pd.read_parquet(path)
    types = {**SELECTION_COLUMNS, **COLUMNS}
    if list(frame.columns) != list(types):
        raise ValueError(f"{Path(path).name}: not the table of select_runs.py")
    return frame.astype(types)


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_header(head: bytes) -> dict:
    """Matrix, voxel size, volumes and TR from the 348 bytes of a NIfTI-1 header."""
    if len(head) < 348:
        return {"error": "fewer than 348 bytes"}
    for order in "<>":
        if struct.unpack(order + "i", head[:4])[0] == 348:
            break
    else:
        return {"error": "not a NIfTI-1 header"}
    dim = struct.unpack(order + "8h", head[40:56])
    pix = struct.unpack(order + "8f", head[76:108])
    units = head[123] & 0x38  # the time unit of pixdim[4]: 8 s, 16 ms, 24 us, 0 not said
    return {
        "matrix": "x".join(str(n) for n in dim[1:4]),
        "voxel_mm": " x ".join(f"{round(p, 3):g}" for p in pix[1:4]),
        "n_volumes": int(dim[4]) if dim[0] >= 4 else 1,
        "tr_s": round(float(pix[4]) * {16: 1e-3, 24: 1e-6}.get(units, 1.0), 6),
        "tr_unit": {8: "s", 16: "ms", 24: "us"}.get(units, "not said, read as s"),
    }


def read_header(base: str, key: str) -> dict:
    """The header of a series in a bucket, from a ranged request for its first bytes."""
    plain = key.endswith(".nii")
    last = 347 if plain else HEAD_BYTES - 1
    try:
        status, body = fetch(base, key, headers={"Range": f"bytes=0-{last}"})
    except RuntimeError as exc:
        return {"error": f"request failed ({exc})"}
    if status not in (200, 206):
        return {"error": f"HTTP {status}"}
    try:
        head = body[:348] if plain else zlib.decompressobj(31).decompress(body, 348)
    except zlib.error as exc:
        return {"error": f"not gzip ({exc})"}
    return parse_header(head)


class Headers:
    """The headers read so far, kept in OUT/headers.json by bucket, key and ETag."""

    def __init__(self, path: Path, rows: dict[int, dict]):
        self.path, self.rows = path, rows
        self.store: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def of(self, at: int, series: str) -> dict:
        """The header of the "preproc" or the "raw" series of the run at a position."""
        row = self.rows[at]
        base = DERIV if series == "preproc" else RAW
        name = f"{series}|{row[f'{series}_key']}|{row[f'{series}_etag']}"
        if name not in self.store:
            self.store[name] = read_header(base, row[f"{series}_key"])
            tmp = self.path.with_name(self.path.name + ".part")
            tmp.write_text(json.dumps(self.store, indent=1))
            tmp.replace(self.path)
        return self.store[name]

    def volumes_ok(self, at: int) -> str:
        """Rule 5 for the run at a position: "" when it passes, else what is wrong."""
        heads = {series: self.of(at, series) for series in ("preproc", "raw")}
        errors = [f"{series}: {h['error']}" for series, h in heads.items() if "error" in h]
        wrong = ""
        if errors:
            wrong = "header not read (" + "; ".join(errors) + ")"
        else:
            counts = (
                int(self.rows[at]["n_volumes"]),
                heads["preproc"]["n_volumes"],
                heads["raw"]["n_volumes"],
            )
            if len(set(counts)) != 1:
                wrong = (
                    f"volumes differ: confounds table {counts[0]}, T1w-space series"
                    f" {counts[1]}, raw series {counts[2]}"
                )
        if wrong:  # said at once: a walk that passes over many runs sends two requests for each
            print(f"  volume check, rank {self.rows[at]['rank']}: {wrong}", file=sys.stderr)
        return wrong


def join(runs: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
    """The runs of the listing, in their order, with what rank_motion.py measured for each."""

    def keys(frame: pd.DataFrame) -> list[tuple]:
        return list(map(tuple, frame[list(KEY)].astype("string").fillna("").to_numpy()))

    records = runs.astype(object).where(runs.notna(), None).to_dict("records")
    if sorted(range(len(records)), key=lambda i: sort_key(records[i])) != list(range(len(records))):
        raise SystemExit("runs.parquet is not in the order of list_table.sort_key")
    motion = motion.assign(_key=keys(motion)).drop_duplicates("_key", keep="last")
    wanted = motion.set_index("_key")[["status", *VALUE_COLUMNS]]
    table = runs.assign(_key=keys(runs)).join(wanted, on="_key").drop(columns="_key")
    table["series_bytes"] = table["preproc_size"] + table["raw_size_all"]
    return table


def distribution(values: np.ndarray) -> str:
    q = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])
    return (
        f"min {q[0]:.3f}, quartiles {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f}, 90th percentile"
        f" {q[4]:.3f}, 95th {q[5]:.3f}, 99th {q[6]:.3f}, max {q[7]:.3f} mm"
    )


def describe(row: dict, head: dict[str, dict] | None) -> str:
    """One selected run in words; no dataset, subject or task."""
    text = (
        f"rank {row['rank']}: mean FD {row['mean_fd']:.3f} mm, max {row['max_fd']:.2f} mm,"
        f" {100 * row['share_fd_over']:.1f}% of volumes above {FD_OVER_MM:g} mm,"
        f" {row['n_volumes']} volumes"
    )
    if head and "error" not in head["preproc"] and "error" not in head["raw"]:
        unit = (
            ""
            if head["raw"]["tr_unit"] == "s"
            else f" (unit in the header: {head['raw']['tr_unit']})"
        )
        text += (
            f", TR {head['raw']['tr_s']:g} s{unit}, raw voxels"
            f" {head['raw']['voxel_mm']} mm in a matrix of {head['raw']['matrix']}, T1w-space"
            f" voxels {head['preproc']['voxel_mm']} mm in {head['preproc']['matrix']}"
        )
    text += (
        f"; series {row['series_bytes'] / 1e6:.0f} MB (T1w-space {row['preproc_size'] / 1e6:.0f},"
        f" raw {row['raw_size_all'] / 1e6:.0f})"
    )
    return text


def first_digest(motion_dir: Path) -> str | None:
    """The digest of the rule that rank_motion.py recorded at its first call in MOTION."""
    path = motion_dir / "invocations.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[0]).get("rule_digest") if lines else None


def select(listing: Path, motion_dir: Path, out: Path, headers: bool) -> None:
    started = now()
    out.mkdir(parents=True, exist_ok=True)
    table = rank_runs(
        join(read_runs(listing / "runs.parquet"), read_motion(motion_dir / "motion.parquet"))
    )
    rows = table.to_dict("index")
    store = Headers(out / "headers.json", rows)
    check = store.volumes_ok if headers else None

    hard, passed = walk_hard(table, LIMIT_BYTES, check)
    quiet, passed_quiet, notes = walk_quiet(table, hard, LIMIT_BYTES, check)
    passed += passed_quiet
    unlimited, _ = walk_hard(table, math.inf, None)

    chosen = [(f"H{n:02d}", "hard", None, at) for n, at in enumerate(hard, start=1)]
    chosen += [
        (f"Q{n:02d}", "quiet", f"H{number:02d}", at) for n, (at, number) in enumerate(quiet, 1)
    ]

    parts = []
    for ident, kind, compares, at in chosen:
        session = session_runs(table, at).copy()
        others = session_mask(table, at) & ~table["qualifies"]
        session.insert(0, "id", ident)
        session["kind"], session["compares_with"] = kind, compares
        session["run_id"] = [f"{ident}-{n:02d}" for n in range(1, len(session) + 1)]
        session["selected"] = session.index == at
        session["session_runs"] = len(session)
        session["session_runs_not_qualifying"] = int(others.sum())
        session["session_bytes"] = int(session["series_bytes"].sum())
        person = table[
            (table["dataset"] == rows[at]["dataset"])
            & (table["subject"] == rows[at]["subject"])
            & table["qualifies"]
        ]
        session["subject_sessions"] = person["session"].nunique(dropna=False)
        session["subject_runs"] = len(person)
        session["volumes_checked"] = session["selected"] & headers
        for series in ("preproc", "raw"):
            head = store.of(at, series) if headers else {}
            for column in HEADER_COLUMNS:
                session[f"{series}_{column}"] = None
                if "error" not in head and head:
                    session.loc[at, f"{series}_{column}"] = head[column]
        parts.append(session)
    types = {**SELECTION_COLUMNS, **COLUMNS}
    selection = (
        pd.concat(parts, ignore_index=True)[list(types)].astype(types)
        if parts
        else pd.DataFrame(columns=list(types)).astype(types)
    )
    selection.to_parquet(out / "selection.parquet", index=False)

    named = ["dataset", "subject", "session", "task", "other_entities", "run_label", "run_stem"]
    ids = selection[selection["selected"]][["id", "kind", "compares_with", "rank", *named]]
    ids.to_csv(out / "ids.csv", index=False)

    over = pd.DataFrame(passed, columns=["at", "walk", "reasons", "detail"])
    facts = table.loc[
        [p["at"] for p in passed], ["rank", "mean_fd", "n_volumes", "series_bytes", *named]
    ]
    over = pd.concat([over.drop(columns="at"), facts.reset_index(drop=True)], axis=1)
    over.to_csv(out / "passed_over.csv", index=False)

    keep = [*KEY, "run", "qualifies", "order", "rank", "not_ranked", "status", "series_bytes"]
    table[[*keep, *VALUE_COLUMNS]].to_parquet(out / "ranked.parquet", index=False)

    # ---- the summary, which names no dataset -------------------------------------------------
    ranked = table[table["rank"].notna()].sort_values("rank")
    qualifying = table[table["qualifies"]]
    digest, first = rule_digest(Path(__file__)), first_digest(motion_dir)
    lines = [
        "Selection of runs for stage 1: six hard cases by head motion, two quiet comparisons.",
        "This text names no dataset, subject or task; ids.csv and selection.parquet do.",
        f"Made {started} by select_runs.py from the tables of list_runs.py and rank_motion.py.",
        f"Digest of the rule: {digest}",
        "Digest that rank_motion.py recorded at its first call, before it read a table: "
        + (f"{first} ({'the same' if first == digest else 'NOT THE SAME'})" if first else "none"),
        "The ranking value is fMRIPrep's own estimate of head motion (mean framewise"
        " displacement); it selects for motion and for nothing else.",
        "",
        f"Qualifying runs: {len(qualifying)}; ranked: {len(ranked)}; not ranked:"
        f" {len(qualifying) - len(ranked)}",
    ]
    reasons = Counter(qualifying["not_ranked"].dropna())
    lines += [f"  {reason}: {n}" for reason, n in sorted(reasons.items())]
    unranked = qualifying[qualifying["rank"].isna() & qualifying["mean_fd"].notna()]
    if hard and len(unranked):
        above = int((unranked["mean_fd"] > table.at[hard[-1], "mean_fd"]).sum())
        lines.append(
            f"  of the runs not ranked, {above} have a mean framewise displacement above that of"
            f" the last hard case; their highest is {unranked['mean_fd'].max():.3f} mm"
        )
    if len(ranked):
        lines += [
            f"Ranked runs come from {ranked['dataset'].nunique()} datasets and"
            f" {ranked.groupby(['dataset', 'subject']).ngroups} subjects.",
            "Mean framewise displacement of the ranked runs: "
            + distribution(ranked["mean_fd"].to_numpy()),
        ]
    big = ranked[ranked["series_bytes"] > LIMIT_BYTES]
    top = ranked.head(TOP)
    top_over = top[top["series_bytes"] > LIMIT_BYTES]
    lines += [
        "",
        f"The size limit of {LIMIT_BYTES:,} bytes for the two series of a run:",
        f"  ranked runs over it: {(ranked['series_bytes'] > LIMIT_BYTES).sum()} of {len(ranked)}"
        + (f"; the highest-ranked of them is at rank {int(big['rank'].min())}" if len(big) else ""),
        f"  of the {len(top)} highest-ranked runs: {len(top_over)} over it"
        + "".join(
            f"\n    rank {r['rank']}: {r['series_bytes'] / 1e9:.2f} GB, {r['n_volumes']} volumes,"
            f" mean FD {r['mean_fd']:.3f} mm"
            for r in top_over.to_dict("records")
        ),
        f"  the {len(top)} highest-ranked runs come from {top['dataset'].nunique()} datasets and"
        f" {top.groupby(['dataset', 'subject']).ngroups} subjects",
        "  hard cases that the same walk takes without the limit and without the volume check:"
        f" ranks {[int(table.at[at, 'rank']) for at in unlimited]}",
        f"  hard cases taken: ranks {[int(table.at[at, 'rank']) for at in hard]}",
        "",
        f"Volume check (rule 5): {'made' if headers else 'NOT made (--no-headers)'}",
        "",
        f"Hard cases: {len(hard)}, from {table.loc[hard, 'dataset'].nunique()} datasets and"
        f" {table.loc[hard].groupby(['dataset', 'subject']).ngroups} subjects",
    ]
    for ident, _, compares, at in chosen:
        head = {s: store.of(at, s) for s in ("preproc", "raw")} if headers else None
        mine = selection[selection["id"] == ident]
        sidecars = int(mine["raw_json_key"].notna().sum())
        if ident == "Q01":
            lines += ["", f"Quiet comparisons: {len(quiet)}"]
        lines += [
            f"  {ident}"
            + (f" (from the dataset of {compares})" if compares else "")
            + f": {describe(rows[at], head)}",
            f"       its session: {len(mine)} qualifying runs,"
            f" {int(mine['session_bytes'].iloc[0]) / 1e9:.2f} GB for their series,"
            f" {int(mine['rank'].notna().sum())} of them ranked, {sidecars} with a JSON sidecar"
            " beside the raw series",
            f"       its subject: {int(mine['subject_sessions'].iloc[0])} session(s) with a"
            f" qualifying run, {int(mine['subject_runs'].iloc[0])} qualifying runs in all",
        ]
    lines += [f"  note: {note}" for note in notes]
    lines += ["", f"Runs that a walk came to and did not take: {len(over)}"]
    lines += [
        f"  {r['walk']}, rank {r['rank']}: {r['reasons']}"
        + (f" ({r['detail']})" if r["detail"] else "")
        + f"; {r['series_bytes'] / 1e9:.2f} GB, {r['n_volumes']} volumes,"
        f" mean FD {r['mean_fd']:.3f} mm"
        for r in over.to_dict("records")
    ]
    lines += ["", f"Requests sent by this call: {requests_made()} (ranged, for headers)"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    with open(out / "invocations.jsonl", "a") as f:
        record = {
            "started": started,
            "ended": now(),
            "rule_digest": digest,
            "rule_digest_at_first_ranking_call": first,
            "headers": headers,
            "requests": requests_made(),
            "selected": len(chosen),
        }
        f.write(json.dumps(record) + "\n")
    print("\n".join(lines))


# ---- the self-check: the rule on made-up tables ----------------------------------------------


def _made_up(rows: list[tuple]) -> pd.DataFrame:
    """A table in the columns the rule reads, from (dataset, subject, name, mean FD, volumes,
    bytes, echoes, status); name is the task, so the rows are told apart by it."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "subject",
            "task",
            "mean_fd",
            "n_volumes",
            "series_bytes",
            "raw_n_echoes",
            "status",
        ],
    )
    frame["session"] = pd.array([None] * len(frame), dtype="string")
    frame["qualifies"] = True
    return frame.sort_values(["dataset", "subject", "task"], kind="mergesort")


def self_check() -> int:
    """Apply the rule to two made-up tables and compare with the answers worked out by hand."""
    ok, small, big = STATUS_OK, 10**8, 2 * 10**9
    first = _made_up(
        [
            ("A", "s1", "a_top_but_big", 3.0, 200, big, 1, ok),
            ("A", "s1", "b_h01", 2.9, 200, small, 1, ok),
            ("A", "s1", "c_same_subject", 2.8, 200, small, 1, ok),
            ("A", "s2", "h02", 2.7, 200, small, 1, ok),
            ("A", "s3", "a_third_of_dataset", 2.6, 200, small, 1, ok),
            ("A", "s3", "b_quiet_but_not_quietest", 0.07, 200, small, 1, ok),
            ("A", "s4", "quietest_but_big", 0.05, 200, big, 1, ok),
            ("A", "s5", "a_q01", 0.06, 200, small, 1, ok),
            ("A", "s6", "q02_tied_with_q01", 0.06, 200, small, 1, ok),
            ("A", "s2", "quiet_run_of_hard_subject", 0.01, 200, small, 1, ok),
            ("B", "s1", "short", 2.5, MIN_VOLUMES - 1, small, 1, ok),
            ("B", "s1", "echoes", 2.4, 200, small, 4, ok),
            ("B", "s1", "no_fd", 2.3, 200, small, 1, "no framewise displacement column"),
            ("B", "s1", "zero", 0.0, 200, small, 1, ok),
            ("B", "s3", "badvol", 2.25, 200, small, 1, ok),
            ("B", "s2", "h03", 2.2, MIN_VOLUMES, small, 1, ok),
            ("C", "s1", "h04_tied_with_h03", 2.2, 200, small, 1, ok),
            ("C", "s2", "h05", 2.0, 200, LIMIT_BYTES, 1, ok),
            ("D", "s1", "h06", 1.9, 200, small, 1, ok),
            ("D", "s2", "seventh", 1.8, 200, small, 1, ok),
        ]
    )
    table = rank_runs(first)
    name = table["task"]

    def volumes(at: int) -> str:
        return "made-up failure" if name[at] == "badvol" else ""

    hard, passed = walk_hard(table, LIMIT_BYTES, volumes)
    quiet, passed_quiet, notes = walk_quiet(table, hard, LIMIT_BYTES, volumes)
    unlimited, _ = walk_hard(table, math.inf, None)
    got = {
        "hard": [name[at] for at in hard],
        "passed": [(name[p["at"]], p["reasons"]) for p in passed],
        "quiet": [(name[at], number) for at, number in quiet],
        "passed_quiet": [(name[p["at"]], p["walk"], p["reasons"]) for p in passed_quiet],
        "notes": notes,
        "not_ranked": sorted(name[table["rank"].isna()]),
        "unlimited": [name[at] for at in unlimited],
        "session": list(session_runs(table, hard[0])["task"]),
    }
    want = {
        "hard": ["b_h01", "h02", "h03", "h04_tied_with_h03", "h05", "h06"],
        "passed": [
            ("a_top_but_big", PASS_SIZE),
            ("c_same_subject", PASS_SUBJECT),
            ("a_third_of_dataset", PASS_DATASET),
            ("badvol", PASS_VOLUMES),
        ],
        "quiet": [("a_q01", 1), ("q02_tied_with_q01", 2)],
        "passed_quiet": [
            ("quietest_but_big", "quiet for H01", PASS_SIZE),
            ("quietest_but_big", "quiet for H02", PASS_SIZE),
        ],
        "notes": [],
        "not_ranked": ["echoes", "no_fd", "short", "zero"],
        "unlimited": ["a_top_but_big", "h02", "badvol", "h03", "h04_tied_with_h03", "h05"],
        "session": ["a_top_but_big", "b_h01", "c_same_subject"],
    }
    second = rank_runs(
        _made_up(
            [
                ("X", "s1", "h01_alone_in_its_dataset", 3.0, 200, small, 1, ok),
                ("Y", "s1", "h02", 2.0, 200, small, 1, ok),
                ("Y", "s2", "q01", 0.1, 200, small, 1, ok),
                ("Y", "s3", "h03_the_last_ranked_run", 0.2, 200, small, 1, ok),
            ]
        )
    )
    hard2, _ = walk_hard(second)
    quiet2, _, notes2 = walk_quiet(second, hard2)
    # X has one subject, so H01 gets no comparison; Y gives one for H02 and has none left for H03
    got["second"] = (
        [second["task"][at] for at in hard2],
        [(second["task"][at], number) for at, number in quiet2],
        len(notes2),
    )
    want["second"] = (
        ["h01_alone_in_its_dataset", "h02", "h03_the_last_ranked_run"],
        [("q01", 2)],
        2,
    )
    failed = [key for key in want if got[key] != want[key]]
    for key in failed:
        print(f"self-check FAILED at {key}:\n  got  {got[key]}\n  want {want[key]}")
    print(f"self-check: {len(want) - len(failed)} of {len(want)} answers are as worked out by hand")
    return 1 if failed else 0


def main() -> None:
    args = sys.argv[1:]
    if args == ["--self-check"]:
        sys.exit(self_check())
    headers = "--no-headers" not in args
    paths = [a for a in args if a != "--no-headers"]
    if len(paths) != 3 or any(a.startswith("-") for a in paths):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    select(Path(paths[0]), Path(paths[1]), Path(paths[2]), headers)


if __name__ == "__main__":
    main()
