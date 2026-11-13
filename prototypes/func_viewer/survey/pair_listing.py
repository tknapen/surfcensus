"""List one subject in both buckets: the fMRIPrep derivative and the raw dataset.

usage: pair_listing.py OUT ACCESSION [SUBJECT]     (SUBJECT with its sub- prefix)

Writes OUT/pair_<ACCESSION>_<SUBJECT>.json with the keys and sizes under "deriv" and "raw".
Listing only, anonymous, one request at a time. The feasibility study made one such listing by
hand, for a dataset whose first derivative subject is absent from the raw bucket; analyze.py
reads it with --pair. Without SUBJECT the script takes the subject that the study took: the
first subject folder of the derivative, in key order, that the raw dataset also has (two more
listings, of the top level of each).
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import DERIV, RAW, list_prefix


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir, accession = Path(sys.argv[1]), sys.argv[2]
    if len(sys.argv) > 3:
        subject = sys.argv[3]
    else:
        _, in_deriv = list_prefix(DERIV, f"fmriprep/{accession}-fmriprep/sub-", "/")
        _, in_raw = list_prefix(RAW, f"{accession}/sub-", "/")
        raw_subjects = {p.rstrip("/").split("/")[-1] for p in in_raw}
        both = [
            s for s in sorted(p.rstrip("/").split("/")[-1] for p in in_deriv) if s in raw_subjects
        ]
        if not both:
            sys.exit(f"{accession}: no subject folder is in both buckets")
        subject = both[0]
    deriv, _ = list_prefix(DERIV, f"fmriprep/{accession}-fmriprep/{subject}/")
    raw, _ = list_prefix(RAW, f"{accession}/{subject}/")
    out = {"deriv": [(k, s) for k, s, _, _ in deriv], "raw": [(k, s) for k, s, _, _ in raw]}
    path = out_dir / f"pair_{accession}_{subject}.json"
    path.write_text(json.dumps(out))
    print(len(deriv), "derivative keys,", len(raw), "raw keys ->", path.name)


if __name__ == "__main__":
    main()
