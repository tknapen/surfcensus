"""Survey of functional outputs in s3://openneuro-derivatives/fmriprep/ (listing only).

usage: survey.py OUT

Lists the dataset prefixes, takes an even sample of 30 in key order, reads the
dataset_description.json of each and lists its first subject in full. Anonymous, read-only, at
most 4 requests at a time (the feasibility study ran it with 6). Writes to OUT:
fmriprep_dataset_prefixes.json and sample_listings.json. Both hold accessions and stay out of
the repository.
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

DERIV = "https://openneuro-derivatives.s3.amazonaws.com"
RAW = "https://s3.amazonaws.com/openneuro.org"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
N_SAMPLE = 30
MAX_REQUESTS = 4

client = httpx.Client(timeout=60.0, headers={"User-Agent": "listing-survey/0.1"})


def get(url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    last = None
    for attempt in range(6):
        try:
            r = client.get(url, params=params, headers=headers)
            if r.status_code in (500, 502, 503, 504):
                last = r
                time.sleep(1.5 * 2**attempt)
                continue
            return r
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(1.5 * 2**attempt)
    raise RuntimeError(f"giving up on {url} {params}: {last}")


def list_prefix(base: str, prefix: str, delimiter: str | None = None):
    """Return (keys [(key, size, etag, last_modified)], common_prefixes)."""
    keys, prefixes = [], []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = get(base, params)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall(f"{NS}Contents"):
            keys.append(
                (
                    c.findtext(f"{NS}Key"),
                    int(c.findtext(f"{NS}Size")),
                    c.findtext(f"{NS}ETag"),
                    c.findtext(f"{NS}LastModified"),
                )
            )
        for c in root.findall(f"{NS}CommonPrefixes"):
            prefixes.append(c.findtext(f"{NS}Prefix"))
        if root.findtext(f"{NS}IsTruncated") == "true":
            token = root.findtext(f"{NS}NextContinuationToken")
        else:
            break
    return keys, prefixes


def dataset_prefixes() -> list[str]:
    """The dataset prefixes under fmriprep/, in key order."""
    _, ds_prefixes = list_prefix(DERIV, "fmriprep/", "/")
    return sorted(ds_prefixes)


def sample_indices(n: int) -> list[int]:
    """Even sample in key order: index floor(i * n / 30), i = 0..29."""
    return [(i * n) // N_SAMPLE for i in range(N_SAMPLE)]


def fmriprep_version(description: dict | None) -> str | None:
    """The fMRIPrep version that a dataset_description.json names."""
    desc = description or {}
    for g in desc.get("GeneratedBy") or []:
        if str(g.get("Name", "")).lower() == "fmriprep":
            return g.get("Version")
    return (desc.get("PipelineDescription") or {}).get("Version")


def survey_dataset(ds_prefix: str) -> dict:
    out: dict = {"prefix": ds_prefix}
    r = get(f"{DERIV}/{ds_prefix}dataset_description.json")
    out["description_status"] = r.status_code
    if r.status_code == 200:
        try:
            out["description"] = json.loads(r.content)
        except ValueError:
            out["description"] = None
            out["description_raw_len"] = len(r.content)
    top_keys, top_prefixes = list_prefix(DERIV, ds_prefix, "/")
    out["top_keys"] = [(k, s) for k, s, _, _ in top_keys]
    out["top_prefixes"] = top_prefixes
    subs = sorted(p for p in top_prefixes if p[len(ds_prefix) :].startswith("sub-"))
    out["n_subject_dirs"] = len(subs)
    if subs:
        first = subs[0]
        out["first_subject"] = first
        keys, _ = list_prefix(DERIV, first)
        out["subject_keys"] = [(k, s) for k, s, _, _ in keys]
    return out


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_prefixes = dataset_prefixes()
    (out_dir / "fmriprep_dataset_prefixes.json").write_text(json.dumps(ds_prefixes, indent=0))
    n = len(ds_prefixes)
    idx = sample_indices(n)
    sample = [ds_prefixes[i] for i in idx]
    print(n, "dataset prefixes; sample indices", idx, file=sys.stderr)
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        results = list(pool.map(survey_dataset, sample))
    (out_dir / "sample_listings.json").write_text(
        json.dumps({"n_prefixes": n, "indices": idx, "datasets": results})
    )
    for res in results:
        print(
            res["prefix"],
            res["description_status"],
            res.get("n_subject_dirs"),
            len(res.get("subject_keys", [])),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
