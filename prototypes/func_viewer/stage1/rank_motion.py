"""Read the confounds table of every qualifying run and write down what it says about head motion.

usage: rank_motion.py LISTING OUT [--max-runs N] [--table-only]

Step 2 of stage 1 (README.md of the prototype, "Next steps"), its first half. LISTING is the
directory that list_runs.py wrote (runs.parquet); OUT is a directory outside the repository:
what is written there names datasets. select_runs.py then applies the selection rule to the
table that this script writes. The rule is in select_runs.py and was fixed before this script
read its first table: every call writes the digest of the rule (rule_digest) to
OUT/invocations.jsonl before it sends a request.

What is read. The desc-confounds_timeseries.tsv of every run that qualifies, whole, from the
OpenNeuroDerivatives bucket: anonymous, read-only, at most 4 requests at a time, streamed into
memory and never written to disk. A table is used only when it has the size of the listing
and, where the ETag of the listing is an MD5, that MD5. If the tables of the qualifying runs
are together more than 10,000,000,000 bytes by the listing, not all are read: the runs of the
first k subjects of every dataset, in the text order of their labels, with k as large as fits
(a dataset with fewer subjects gives all it has), and the summary says what was left out.

What is measured, per table. The number of volumes (rows). Of framewise displacement, over the
volumes that have a value (fMRIPrep writes none for the first): mean, median, maximum, and the
number and share above 0.5 mm. Of standardised DVARS: mean and maximum. How many
non-steady-state columns the table has and how many volumes they mark. And the largest
difference between the framewise displacement of the table and the one computed here from its
six motion parameters (Power: translations plus rotations on a sphere of 50 mm), which says
whether the column is what it is taken for. fMRIPrep has named these columns differently over
its versions; the names in COLUMN_NAMES are all accepted, and which one a table has is recorded
and counted. A table without a framewise displacement column is recorded with that reason.

So the value that select_runs.py ranks by is fMRIPrep's own estimate of head motion, and
nothing else that can go wrong in preprocessing enters it.

Written to OUT:

  rows.jsonl          one line per table read, appended as it is read; this is what makes the
                      script resumable: a run whose table (key and ETag) has a line is not read
                      again, unless the line says that the request failed
  motion.parquet      one row per qualifying run, in the order of runs.parquet; MOTION_COLUMNS
                      below has the columns, read_motion() reads it with their types
  summary.txt         counts and distributions; it names no dataset
  invocations.jsonl   per call: a line when it starts (time, digest of the rule) and a line when
                      it ends (requests, bytes, seconds)

--max-runs N reads at most N more tables and then writes the table of what is read so far.
--table-only sends no request and writes motion.parquet and summary.txt from rows.jsonl.

A helper of select_runs.py as well: fetch() is the one place where the two scripts send a
request, and rule_digest() is how both name the rule.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
sys.path.append(str(Path(__file__).resolve().parents[1] / "survey"))
from list_table import KEY, read_runs
from survey import DERIV, RAW

__all__ = [
    "DERIV",
    "FD_OVER_MM",
    "MOTION_COLUMNS",
    "RAW",
    "STATUS_OK",
    "VALUE_COLUMNS",
    "fetch",
    "measure",
    "read_motion",
    "requests_made",
    "rule_digest",
]

MAX_REQUESTS = 4
BUDGET_BYTES = 10_000_000_000  # of confounds tables, by the listing; above it not all are read
FD_OVER_MM = 0.5
RADIUS_MM = 50.0  # of the sphere on which Power's framewise displacement puts the rotations
STATUS_OK = "ok"
FAILED = "request failed"  # the one status that a later call tries again

# What fMRIPrep has called the columns: the name since 1.2 first, the older one after it.
COLUMN_NAMES = {
    "fd": ("framewise_displacement", "FramewiseDisplacement"),
    "dvars": ("std_dvars", "stdDVARS"),
    "nss": ("non_steady_state_outlier", "NonSteadyStateOutlier"),  # prefixes; a number follows
    "motion": (
        ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"),
        ("X", "Y", "Z", "RotX", "RotY", "RotZ"),
    ),
}

# column -> pandas dtype: what is measured on a table. select_runs.py copies these columns.
VALUE_COLUMNS: dict[str, str] = {
    "n_volumes": "Int64",  # rows of the table
    "n_columns": "Int64",
    "fd_column": "string",  # the name of the framewise displacement column, as the table has it
    "n_fd": "Int64",  # volumes with a framewise displacement
    "first_fd_missing": "boolean",  # the first volume has none, as fMRIPrep writes it
    "mean_fd": "float64",  # mm; the ranking value
    "median_fd": "float64",
    "max_fd": "float64",
    "n_fd_over": "Int64",  # volumes above FD_OVER_MM
    "share_fd_over": "float64",  # n_fd_over / n_fd
    "dvars_column": "string",
    "mean_std_dvars": "float64",
    "max_std_dvars": "float64",
    "nss_column": "string",  # the prefix of the non-steady-state columns, as the table has it
    "n_nss_columns": "Int64",
    "n_nss_volumes": "Int64",  # volumes that such a column marks
    "fd_recomputed_diff": "float64",  # largest |FD of the table - FD from its motion parameters|
}
MOTION_COLUMNS: dict[str, str] = {
    **dict.fromkeys(KEY, "string"),
    "run": "Int64",
    "confounds_key": "string",
    "confounds_size": "Int64",
    "confounds_etag": "string",
    "status": "string",  # STATUS_OK, or why the table could not be used
    "detail": "string",
    "bytes_read": "Int64",
    "md5_checked": "boolean",  # the ETag of the listing is an MD5 and was compared
    "read_at": "string",
    **VALUE_COLUMNS,
}

RULE_MARKS = ("---- the rule ----", "---- end of the rule ----")
RETRY_STATUS = (500, 502, 503, 504)
ATTEMPTS = 7

_slots = threading.BoundedSemaphore(MAX_REQUESTS)
_lock = threading.Lock()
_requests = 0
_bytes = 0
_client = httpx.Client(timeout=120.0, headers={"User-Agent": "listing-survey/0.3"})


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def requests_made() -> int:
    """How many requests this process has sent so far, retries included."""
    with _lock:
        return _requests


def bytes_read() -> int:
    """How many bytes of bodies this process has received so far."""
    with _lock:
        return _bytes


def fetch(base: str, key: str, headers: dict | None = None) -> tuple[int, bytes]:
    """(status, body) of one object of a bucket: an unsigned GET, streamed into memory.

    Every request of rank_motion.py and select_runs.py goes through here: it holds a semaphore
    of MAX_REQUESTS and counts requests and bytes. Retries on 5xx (S3 answers SlowDown with
    503) and on transport errors, a body that ends early among them.
    """
    global _requests, _bytes
    url = f"{base}/{quote(key, safe='/')}"
    last: object = None
    for attempt in range(ATTEMPTS):
        with _slots:
            with _lock:
                _requests += 1
            try:
                with _client.stream("GET", url, headers=headers) as r:
                    if r.status_code not in RETRY_STATUS:
                        body = b"".join(r.iter_bytes())
                        with _lock:
                            _bytes += len(body)
                        return r.status_code, body
                    last = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(1.5 * 2**attempt, 30.0))  # outside the semaphore: a wait holds no slot
    raise RuntimeError(f"giving up after {ATTEMPTS} attempts ({last})")


def rule_digest(script: Path) -> str:
    """16 hex characters over the two marked parts of select_runs.py: the rule as text in its
    docstring and the rule as code. Trailing white space of a line does not count."""
    kept: list[str] = []
    inside, parts = False, 0
    for line in Path(script).read_text().splitlines():
        mark = line.strip().removeprefix("#").strip()
        if mark == RULE_MARKS[0] and not inside:
            inside, parts = True, parts + 1
        elif mark == RULE_MARKS[1] and inside:
            inside = False
        elif inside:
            kept.append(line.rstrip())
    if parts != 2 or inside:
        raise ValueError(f"{Path(script).name}: the rule is not marked twice")
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()[:16]


def _number(value) -> float | None:
    """A float for a JSON line; None for a value that is not finite."""
    value = float(value)
    return value if np.isfinite(value) else None


def measure(body: bytes) -> tuple[str, str, dict]:
    """(status, detail, values) of a confounds table given as the bytes of the file."""
    values: dict = {}
    first, _, rest = body.partition(b"\n")
    names = first.decode("utf-8", "replace").rstrip("\r").split("\t")
    if len(names) < 2 or not rest.strip():
        return "not a table", f"{len(names)} column(s) in the first line", values
    have = set(names)
    fd = next((n for n in COLUMN_NAMES["fd"] if n in have), None)
    dvars = next((n for n in COLUMN_NAMES["dvars"] if n in have), None)
    motion = next((m for m in COLUMN_NAMES["motion"] if have.issuperset(m)), None)
    prefix = next((p for p in COLUMN_NAMES["nss"] if any(n.startswith(p) for n in names)), None)
    nss = [n for n in names if prefix and n.startswith(prefix)]
    wanted = list(dict.fromkeys([c for c in (fd, dvars) if c] + list(motion or ()) + nss))
    wanted = wanted or names[:1]  # some column, so that the rows are counted
    try:
        frame = pd.read_csv(io.BytesIO(body), sep="\t", usecols=wanted, na_values=["n/a"])
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return "not a table", f"{type(exc).__name__}: {str(exc)[:200]}", values
    frame = frame.apply(pd.to_numeric, errors="coerce")
    values.update(
        n_volumes=len(frame),
        n_columns=len(names),
        fd_column=fd,
        dvars_column=dvars,
        nss_column=prefix,
        n_nss_columns=len(nss),
        n_nss_volumes=int((frame[nss] > 0).any(axis=1).sum()) if nss else 0,
    )
    if dvars:
        d = frame[dvars].to_numpy("float64")
        d = d[np.isfinite(d)]
        if len(d):
            values.update(mean_std_dvars=_number(d.mean()), max_std_dvars=_number(d.max()))
    if fd is None:
        return "no framewise displacement column", "", values
    series = frame[fd].to_numpy("float64")
    finite = series[np.isfinite(series)]
    values.update(n_fd=len(finite), first_fd_missing=bool(len(series) and np.isnan(series[0])))
    if not len(finite):
        return "no framewise displacement value", "", values
    over = int((finite > FD_OVER_MM).sum())
    values.update(
        mean_fd=_number(finite.mean()),
        median_fd=_number(np.median(finite)),
        max_fd=_number(finite.max()),
        n_fd_over=over,
        share_fd_over=over / len(finite),
    )
    if motion and len(series) > 1:
        steps = np.abs(np.diff(frame[list(motion)].to_numpy("float64"), axis=0))
        again = steps[:, :3].sum(axis=1) + RADIUS_MM * steps[:, 3:].sum(axis=1)
        both = np.isfinite(again) & np.isfinite(series[1:])
        if both.any():
            values["fd_recomputed_diff"] = _number(np.abs(again - series[1:])[both].max())
    return STATUS_OK, "", values


def read_one(run: dict) -> dict:
    """Fetch and measure the confounds table of one run; never raises."""
    line = {k: run[k] for k in (*KEY, "run", "confounds_key", "confounds_size", "confounds_etag")}
    line.update(status=FAILED, detail="", bytes_read=None, md5_checked=None)
    try:
        code, body = fetch(DERIV, run["confounds_key"])
        line["bytes_read"] = len(body)
        etag = run["confounds_etag"]
        is_md5 = len(etag) == 32 and "-" not in etag
        line["md5_checked"] = is_md5
        if code != 200:
            line["status"] = f"HTTP {code}"
        elif len(body) != run["confounds_size"]:
            line["status"] = "size differs from the listing"
            line["detail"] = f"{len(body)} bytes read"
        elif is_md5 and hashlib.md5(body).hexdigest() != etag:
            line["status"] = "ETag differs from the listing"
        else:
            line["status"], line["detail"], values = measure(body)
            line.update(values)
    except RuntimeError as exc:
        line["detail"] = str(exc)
    except Exception as exc:  # a table that breaks the reader is a result, not a crash
        line["status"], line["detail"] = "not a table", f"{type(exc).__name__}: {str(exc)[:200]}"
    line["read_at"] = now()
    return line


def within_budget(
    runs: pd.DataFrame, budget: int = BUDGET_BYTES
) -> tuple[pd.DataFrame, int | None]:
    """The runs whose tables are read: all of them when their tables fit the budget (then the
    second value is None), else those of the first k subjects of every dataset in the text
    order of their labels, with k as large as fits."""
    if int(runs["confounds_size"].sum()) <= budget:
        return runs, None
    subjects = runs[["dataset", "subject"]].drop_duplicates().sort_values(["dataset", "subject"])
    subjects["place"] = subjects.groupby("dataset").cumcount() + 1
    place = runs.merge(subjects, on=["dataset", "subject"], how="left")["place"].to_numpy()
    by_place = runs["confounds_size"].astype("int64").groupby(place).sum().sort_index().cumsum()
    fitting = by_place[by_place <= budget]
    k = int(fitting.index.max()) if len(fitting) else 0
    return runs[place <= k], k


def read_lines(out: Path) -> dict[tuple, dict]:
    """The lines of rows.jsonl, the last one per table (key and ETag of the listing)."""
    lines: dict[tuple, dict] = {}
    path = out / "rows.jsonl"
    if path.exists():
        with open(path) as f:
            for text in f:
                if text.strip():
                    line = json.loads(text)
                    lines[(line["confounds_key"], line["confounds_etag"])] = line
    return lines


def read_motion(path: str | Path) -> pd.DataFrame:
    """motion.parquet with the column types of MOTION_COLUMNS; refuses a table of other columns."""
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(MOTION_COLUMNS):
        raise ValueError(f"{Path(path).name}: not the table of rank_motion.py")
    return frame.astype(MOTION_COLUMNS)


def quantiles(values, digits: int = 3, unit: str = "") -> str:
    q = np.percentile(np.asarray(values, dtype="float64"), [0, 25, 50, 75, 95, 99, 100])
    f = f"{{:.{digits}f}}".format
    return (
        f"min {f(q[0])}, quartiles {f(q[1])} / {f(q[2])} / {f(q[3])}, 95th percentile {f(q[4])},"
        f" 99th {f(q[5])}, max {f(q[6])}{unit}"
    )


def write_table(out: Path, qualifying: pd.DataFrame, wanted: pd.DataFrame, k: int | None) -> None:
    """motion.parquet and summary.txt from rows.jsonl."""
    lines = read_lines(out)
    rows = []
    for run in wanted.to_dict("records"):
        line = lines.get((run["confounds_key"], run["confounds_etag"]))
        if line is not None:
            rows.append({c: line.get(c) for c in MOTION_COLUMNS})
    motion = pd.DataFrame(rows, columns=list(MOTION_COLUMNS)).astype(MOTION_COLUMNS)
    motion.to_parquet(out / "motion.parquet", index=False)

    calls = []
    if (out / "invocations.jsonl").exists():
        calls = [json.loads(t) for t in (out / "invocations.jsonl").read_text().splitlines() if t]
    ends = [c for c in calls if c.get("event") == "end"]
    ok = motion[motion["status"] == STATUS_OK]
    text = [
        "Head motion by the confounds tables of the qualifying runs (fMRIPrep 23.0.2 or older,",
        "OpenNeuroDerivatives bucket). This text names no dataset; motion.parquet does.",
        f"Table written {now()}.",
        "The values are fMRIPrep's own estimate of head motion and say nothing else about a run.",
        "",
        f"Qualifying runs in the listing: {len(qualifying)}; their confounds tables:"
        f" {int(qualifying['confounds_size'].sum()):,} bytes",
    ]
    if k is None:
        text.append(f"  within the budget of {BUDGET_BYTES:,} bytes: every table is to be read")
    else:
        left = len(qualifying) - len(wanted)
        subjects = ["dataset", "subject"]
        fewer = qualifying.groupby(subjects).ngroups - wanted.groupby(subjects).ngroups
        text.append(
            f"  over the budget of {BUDGET_BYTES:,} bytes: the first {k} subjects of every"
            f" dataset in label order are read, {len(wanted)} runs with"
            f" {int(wanted['confounds_size'].sum()):,} bytes; left out: {left} runs of"
            f" {fewer} subjects"
        )
    text += [
        f"Tables with a line in rows.jsonl: {len(motion)} of {len(wanted)}",
        f"  usable (read, the file of the listing, with framewise displacement): {len(ok)}",
    ]
    text += [
        f"  not usable, {status}: {n}"
        for status, n in sorted(Counter(motion["status"]).items())
        if status != STATUS_OK
    ]
    checked = motion["md5_checked"].fillna(False).astype(bool)
    text.append(
        f"  compared with the MD5 of the listing: {int(checked.sum())}; by size only:"
        f" {int((~checked).sum())}"
    )
    read = motion[motion["n_volumes"].notna()]
    text += ["", f"Column names, over the {len(read)} tables that were parsed:"]
    for column, what in (
        ("fd_column", "framewise displacement"),
        ("dvars_column", "standardised DVARS"),
        ("nss_column", "non-steady-state columns (prefix)"),
    ):
        counts = Counter(read[column].fillna("(none)"))
        text.append(f"  {what}: " + ", ".join(f"{name} {n}" for name, n in sorted(counts.items())))
    if len(read):
        marked = read.loc[read["n_nss_columns"] > 0, "n_nss_volumes"]
        text += [
            f"  tables with at least one non-steady-state column: {len(marked)};"
            " volumes marked per such table: " + (quantiles(marked, 0) if len(marked) else "none"),
            f"  columns per table: {quantiles(read['n_columns'], 0)}",
        ]
    if len(ok):
        late = ok["n_volumes"] - ok["n_fd"] - ok["first_fd_missing"].astype("Int64")
        short, no_first = int((ok["n_volumes"] < 50).sum()), int(ok["first_fd_missing"].sum())
        diff = ok["fd_recomputed_diff"].dropna()
        text += [
            "",
            f"Over the {len(ok)} usable tables:",
            f"  volumes: {quantiles(ok['n_volumes'], 0)}; with fewer than 50: {short}",
            f"  the first volume has no framewise displacement in {no_first};"
            f" a later volume has none in {int((late > 0).sum())} tables",
            f"  mean framewise displacement: {quantiles(ok['mean_fd'], 3, ' mm')}",
            f"  maximum framewise displacement: {quantiles(ok['max_fd'], 2, ' mm')}",
            f"  share of volumes above {FD_OVER_MM:g} mm: {quantiles(ok['share_fd_over'], 3)}",
            f"  framewise displacement computed again from the six motion parameters (sphere of"
            f" {RADIUS_MM:g} mm): compared in {len(diff)} tables; largest difference per table: "
            + (quantiles(diff, 6, " mm") if len(diff) else "none")
            + f"; above 0.001 mm in {int((diff > 1e-3).sum())}",
        ]
    text += [
        "",
        f"Calls of the script that sent requests: {sum(c['requests'] > 0 for c in ends)};"
        f" requests {sum(c['requests'] for c in ends)}, bytes received"
        f" {sum(c['bytes'] for c in ends):,}, seconds {sum(c['seconds'] for c in ends):.0f},"
        f" at most {MAX_REQUESTS} requests at a time",
    ]
    if calls:
        text.append(
            f"Digest of the selection rule at the first call ({calls[0]['started']}):"
            f" {calls[0].get('rule_digest')}; at the last: {calls[-1].get('rule_digest')}"
        )
    (out / "summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def main() -> None:
    args = sys.argv[1:]
    table_only = "--table-only" in args
    max_runs = None
    if "--max-runs" in args:
        at = args.index("--max-runs")
        if at + 1 >= len(args) or not args[at + 1].isdigit():
            args = []
        else:
            max_runs = int(args[at + 1])
            del args[at : at + 2]
    paths = [a for a in args if a != "--table-only"]
    if len(paths) != 2 or any(a.startswith("-") for a in paths):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    listing, out = Path(paths[0]), Path(paths[1])
    out.mkdir(parents=True, exist_ok=True)

    runs = read_runs(listing / "runs.parquet")
    qualifying = runs[runs["qualifies"]]
    wanted, k = within_budget(qualifying)
    here = Path(__file__).resolve().parent / "select_runs.py"
    start = {
        "event": "start",
        "started": now(),
        "rule_digest": rule_digest(here),
        "select_runs_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "table_only": table_only,
        "max_runs": max_runs,
    }
    with open(out / "invocations.jsonl", "a") as f:
        f.write(json.dumps(start) + "\n")  # before any request: the rule as it is now

    done = {key for key, line in read_lines(out).items() if line["status"] != FAILED}
    pending = [
        run
        for run in wanted.astype(object).where(wanted.notna(), None).to_dict("records")
        if (run["confounds_key"], run["confounds_etag"]) not in done
    ]
    todo = [] if table_only else pending[:max_runs]
    t0 = time.monotonic()
    print(
        f"{len(qualifying)} qualifying runs, {len(wanted)} to read, {len(wanted) - len(pending)}"
        f" read before, {len(todo)} in this call",
        file=sys.stderr,
    )
    if todo:
        with (
            open(out / "rows.jsonl", "a") as f,
            ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool,
        ):
            futures = [pool.submit(read_one, run) for run in todo]
            for n, future in enumerate(as_completed(futures), start=1):
                f.write(json.dumps(future.result()) + "\n")
                f.flush()
                if n % 500 == 0 or n == len(todo):
                    print(
                        f"  {n} of {len(todo)} tables, {bytes_read() / 1e6:.0f} MB,"
                        f" {requests_made()} requests, {time.monotonic() - t0:.0f} s",
                        file=sys.stderr,
                    )
    end = {
        "event": "end",
        "started": start["started"],
        "ended": now(),
        "rule_digest": start["rule_digest"],
        "tables": len(todo),
        "requests": requests_made(),
        "bytes": bytes_read(),
        "seconds": round(time.monotonic() - t0, 1),
    }
    with open(out / "invocations.jsonl", "a") as f:
        f.write(json.dumps(end) + "\n")
    write_table(out, qualifying, wanted, k)


if __name__ == "__main__":
    main()
