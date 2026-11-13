"""One row per dataset prefix: fMRIPrep version, what func/ holds, and the layout; then the
counts that the report states.

usage: layout_table.py OUT [--offline]
       (after raw_and_versions.py OUT and layout_all.py OUT; writes
        OUT/func_layout_all_datasets.csv, which holds accessions and stays out of the repository)

The feasibility study made this table by hand from all_layouts.json and all_versions.json and
checked three datasets whose first listing page looked incomplete against a full listing of
their first subject. This script does the same by rule. A dataset is listed again in full
(first subject only) when its flags fit neither layout as a whole: a preprocessed series in
BOLD space without one of the T1w-space series, the MNI series, the scanner-to-T1w transform
and the confounds, or files of both layouts. --offline skips that and marks the rows.

Layout: "full" when a preprocessed BOLD series in BOLD space is there, else "minimal".
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from layout_all import FLAGS, note_key
from survey import DERIV, list_prefix

FULL = ("preproc_boldspace", "preproc_T1w", "preproc_MNI", "scanner_to_T1w", "confounds")
MINIMAL = ("hmc_boldref", "coreg_xfm")


def version_key(version):
    """(major, minor) of a version such as 23.2.0 or 22.1.0+0.gce344b39.dirty."""
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def needs_full_listing(row):
    full = [row[k] for k in FULL]
    minimal = [row[k] for k in MINIMAL]
    return (any(full) and not all(full)) or (any(full) and any(minimal))


def relist(row):
    """The flags again, from every key of the first subject."""
    _, subjects = list_prefix(DERIV, row["prefix"] + "sub-", "/")
    keys, _ = list_prefix(DERIV, sorted(subjects)[0])
    seen = {**dict.fromkeys(FLAGS, False), "func_keys": 0}
    for key, *_ in keys:
        note_key(seen, key)
    changed = [k for k in FLAGS if seen[k] != row[k]]
    row.update({k: seen[k] for k in FLAGS})
    if changed:
        return "full listing of the first subject changed: " + ", ".join(changed)
    return "full listing of the first subject: flags as on the first page"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    offline = "--offline" in sys.argv[2:]
    versions = {
        v["prefix"]: v["version"] for v in json.loads((out_dir / "all_versions.json").read_text())
    }
    rows = json.loads((out_dir / "all_layouts.json").read_text())
    for row in rows:
        row["version"] = versions[row["prefix"]]
        row["accession"] = row["prefix"].split("/")[1].split("-")[0]
        row["note"] = ""
        if needs_full_listing(row):
            row["note"] = "not listed in full (--offline)" if offline else relist(row)
        row["layout"] = "full" if row["preproc_boldspace"] else "minimal"
    fields = ["prefix", *FLAGS, "func_keys", "pages", "version", "accession", "layout", "note"]
    with open(out_dir / "func_layout_all_datasets.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    old = [r for r in rows if version_key(r["version"]) < (23, 2)]
    new = [r for r in rows if version_key(r["version"]) >= (23, 2)]
    by_version = sorted(Counter(r["version"] for r in rows).items(), key=lambda kv: kv[0])
    print(n, "dataset prefixes")
    print("fMRIPrep versions:", ", ".join(f"{v} ({c})" for v, c in by_version))
    print(f"made with 23.0.x or older: {len(old)}; with 23.2.0 or newer: {len(new)}")
    for layout in ("full", "minimal"):
        print(
            f"layout {layout}: {sum(r['layout'] == layout for r in rows)}"
            f" ({sum(r['layout'] == layout for r in old)} of the older,"
            f" {sum(r['layout'] == layout for r in new)} of the newer)"
        )
    for flag in FLAGS:
        print(f"  {flag}: {sum(r[flag] for r in rows)} of {n}")
    to_t1w = sum(r["coreg_xfm"] or r["scanner_to_T1w"] for r in rows)
    none = sum(not (r["preproc_boldspace"] or r["preproc_T1w"] or r["preproc_MNI"]) for r in rows)
    bold_only = sum(
        r["preproc_boldspace"] and not (r["preproc_T1w"] or r["preproc_MNI"]) for r in rows
    )
    print(f"a BOLD-to-T1w transform (either name): {to_t1w} of {n}")
    print(f"no preprocessed BOLD series in any space: {none} of {n}; BOLD space only: {bold_only}")
    for r in rows:
        if r["note"]:
            print(f"  index {rows.index(r)} (fMRIPrep {r['version']}): {r['note']}")


if __name__ == "__main__":
    main()
