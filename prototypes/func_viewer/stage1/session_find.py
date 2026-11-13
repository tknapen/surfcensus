"""Find subjects with several sessions of several runs among the older derivatives. Listing only.

usage: session_find.py SURVEY_DIR OUT [--min-sessions 2] [--min-runs 2]

SURVEY_DIR holds fmriprep_dataset_prefixes.json and all_versions.json, as survey/survey.py and
survey/raw_and_versions.py write them. For every derivative that names fMRIPrep 23.0.2 or older
the first subject folder (in key order) is listed in full: anonymous, read-only, at most 4
requests at a time. A run counts when the listing has its T1w-space preprocessed series and its
from-scanner_to-T1w transform. Writes OUT/session_candidates.json, which names datasets and stays
out of the repository, and prints one line per subject that has at least --min-sessions sessions
with at least --min-runs such runs each: the index of its dataset among the prefixes in key
order, the runs per session, and the size of the largest T1w-space series. No name is printed.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
sys.path.append(str(Path(__file__).resolve().parent.parent / "survey"))  # the listing functions
from session import parse_name, run_key, session_of
from survey import DERIV, MAX_REQUESTS, list_prefix

OLDER_THAN = (23, 2, 0)
SERIES = "_space-T1w_desc-preproc_bold.nii.gz"
XFM = "_from-scanner_to-T1w_mode-image_xfm.txt"


def version_tuple(text):
    """(major, minor, patch) of a version such as 22.1.0+0.gce344b39.dirty; None if unreadable."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(g) for g in m.groups()) if m else None


def first_subject(prefix):
    """The keys and sizes of the first subject folder of a derivative, and the folder count."""
    _, top = list_prefix(DERIV, prefix + "sub-", "/")
    subs = sorted(top)
    if not subs:
        return {"prefix": prefix, "n_subjects": 0, "subject": None, "keys": []}
    keys, _ = list_prefix(DERIV, subs[0])
    return {
        "prefix": prefix,
        "n_subjects": len(subs),
        "subject": subs[0],
        "keys": [(k, s) for k, s, _, _ in keys],
    }


def sessions_of(keys):
    """session label ('' without ses- folders) -> {run key: size of its T1w-space series}."""
    series, xfm = {}, set()
    for key, size in keys:
        name = key.rsplit("/", 1)[-1]
        if "/func/" not in key:
            continue
        if name.endswith(SERIES):
            series[name[: -len(SERIES)]] = size
        elif name.endswith(XFM):
            xfm.add(name[: -len(XFM)])
    out = {}
    for stem, size in series.items():
        if stem in xfm:
            ent = parse_name(stem)
            out.setdefault(session_of(ent), {})[run_key(ent)] = size
    return out


def main():
    ap = argparse.ArgumentParser(usage=__doc__.strip().split("\n\n")[1])
    ap.add_argument("survey_dir")
    ap.add_argument("out")
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--min-runs", type=int, default=2)
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    args = ap.parse_args()
    survey_dir, out_dir = Path(args.survey_dir), Path(args.out)
    prefixes = json.loads((survey_dir / "fmriprep_dataset_prefixes.json").read_text())
    versions = {
        v["prefix"]: version_tuple(v["version"])
        for v in json.loads((survey_dir / "all_versions.json").read_text())
    }
    older = [p for p in prefixes if versions.get(p) is not None and versions[p] < OLDER_THAN]
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as pool:
        listed = list(pool.map(first_subject, older))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_candidates.json").write_text(json.dumps(listed))
    n_multi = 0
    for item in listed:
        ses = sessions_of(item["keys"])
        good = {s: r for s, r in ses.items() if len(r) >= args.min_runs}
        n_multi += len(ses) > 1
        if len(good) < args.min_sessions:
            continue
        largest = max(size for r in good.values() for size in r.values())
        print(
            f"index {prefixes.index(item['prefix'])}: {len(ses)} sessions,"
            f" runs per session {[len(r) for _, r in sorted(ses.items())]},"
            f" largest T1w-space series {largest / 1e6:.0f} MB"
        )
    print(
        f"{len(older)} older derivatives of {len(prefixes)} prefixes listed;"
        f" the first subject has more than one session in {n_multi}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
