"""Which func layout does each of the dataset prefixes have? One to three listing pages per
dataset, at most 4 requests at a time (the feasibility study used 8).

usage: layout_all.py OUT    (after survey.py OUT; writes OUT/all_layouts.json)

A page holds 1,000 keys in key order from the first subject on, and reading stops once 20 keys
of a func/ folder were seen, so a flag can be false because the page ended before the file.
layout_table.py finds those datasets and lists their first subject in full.
"""

import json
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import DERIV, MAX_REQUESTS, NS, get

FLAGS = (
    "preproc_boldspace",
    "preproc_T1w",
    "preproc_MNI",
    "hmc_boldref",
    "coreg_xfm",
    "scanner_to_T1w",
    "confounds",
)


def note_key(seen: dict, key: str) -> None:
    """Set the flags that one key of a listing shows; count it if it lies in a func/ folder."""
    if "/func/" not in key:
        return
    seen["func_keys"] += 1
    n = key.rsplit("/", 1)[-1]
    if n.endswith("_desc-preproc_bold.nii.gz"):
        if "_space-T1w_" in n:
            seen["preproc_T1w"] = True
        elif "_space-MNI" in n:
            seen["preproc_MNI"] = True
        elif "_space-" not in n:
            seen["preproc_boldspace"] = True
    if n.endswith("_desc-hmc_boldref.nii.gz"):
        seen["hmc_boldref"] = True
    if "from-boldref_to-T1w" in n:
        seen["coreg_xfm"] = True
    if "from-scanner_to-T1w" in n:
        seen["scanner_to_T1w"] = True
    if n.endswith("_desc-confounds_timeseries.tsv"):
        seen["confounds"] = True


def layout(prefix):
    token = None
    seen = {**dict.fromkeys(FLAGS, False), "func_keys": 0, "pages": 0}
    for _ in range(3):
        params = {"list-type": "2", "prefix": prefix + "sub-", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = get(DERIV, params)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        seen["pages"] += 1
        for c in root.findall(f"{NS}Contents"):
            note_key(seen, c.findtext(f"{NS}Key"))
        if seen["func_keys"] >= 20 or root.findtext(f"{NS}IsTruncated") != "true":
            break
        token = root.findtext(f"{NS}NextContinuationToken")
    return {"prefix": prefix, **seen}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    prefixes = json.loads((out_dir / "fmriprep_dataset_prefixes.json").read_text())
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        res = list(pool.map(layout, prefixes))
    (out_dir / "all_layouts.json").write_text(json.dumps(res, indent=0))


if __name__ == "__main__":
    main()
