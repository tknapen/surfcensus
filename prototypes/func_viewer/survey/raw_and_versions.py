"""Second pass: raw-bucket listing of the same first subject for the 30 sampled datasets,
and dataset_description.json of all dataset prefixes (fMRIPrep version only).

usage: raw_and_versions.py OUT    (after survey.py OUT)

Listing and small JSON only; anonymous; at most 4 requests at a time (the feasibility study
used 8 for the descriptions). Writes OUT/raw_listings.json and OUT/all_versions.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import DERIV, MAX_REQUESTS, RAW, get, list_prefix


def raw_subject(ds: dict) -> dict:
    acc = ds["prefix"].split("/")[1].split("-")[0]
    sub = ds["first_subject"].rstrip("/").split("/")[-1]
    keys, _ = list_prefix(RAW, f"{acc}/{sub}/")
    top_keys, top_prefixes = list_prefix(RAW, f"{acc}/", "/")
    return {
        "accession": acc,
        "subject": sub,
        "keys": [(k, s) for k, s, _, _ in keys],
        "top_keys": [(k, s) for k, s, _, _ in top_keys],
        "top_prefixes": top_prefixes,
    }


def version_of(prefix: str) -> dict:
    r = get(f"{DERIV}/{prefix}dataset_description.json")
    out = {"prefix": prefix, "status": r.status_code, "version": None}
    if r.status_code == 200:
        try:
            desc = json.loads(r.content)
        except ValueError:
            return out
        for g in desc.get("GeneratedBy") or []:
            if str(g.get("Name", "")).lower() == "fmriprep":
                out["version"] = g.get("Version")
        if out["version"] is None:
            out["version"] = (desc.get("PipelineDescription") or {}).get("Version")
    return out


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    d = json.loads((out_dir / "sample_listings.json").read_text())
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        raws = list(pool.map(raw_subject, d["datasets"]))
    (out_dir / "raw_listings.json").write_text(json.dumps(raws))
    prefixes = json.loads((out_dir / "fmriprep_dataset_prefixes.json").read_text())
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        vers = list(pool.map(version_of, prefixes))
    (out_dir / "all_versions.json").write_text(json.dumps(vers, indent=0))
    print(len(raws), "raw listings;", len(vers), "descriptions", file=sys.stderr)


if __name__ == "__main__":
    main()
