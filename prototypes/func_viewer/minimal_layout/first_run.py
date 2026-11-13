"""Print the arguments of fetch_min.py for the first functional run of the first subject of a
dataset in the survey's sample. Reads a file that survey/survey.py wrote; sends no request.

usage: first_run.py SURVEY_OUT INDEX

SURVEY_OUT is the directory that survey.py filled. INDEX is the position of the dataset in the
key-ordered list of prefixes under fmriprep/, and must be one of the sample indices that
survey.py prints. "First" means first in key order: the first subject folder of the dataset,
and in the listing of that subject the first key in a func/ folder that has a task label. One
line goes to standard output:

    ACCESSION SUBJECT --task TASK [--session SES] [--run RUN]

Standard error gets the number of prefixes, the fMRIPrep version and the number of subject
folders of the dataset, to compare with what README.md says of the dataset that the feasibility
study used ("The second subject"). The printed line names a dataset and stays out of the
repository.

Not part of the feasibility study, which wrote the labels by hand; added on 2026-11-13 so that
the next session need not search the listing for them.
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "survey"))  # also under -I
from survey import fmriprep_version


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
        sys.exit(2)
    out_dir, index = Path(sys.argv[1]), int(sys.argv[2])
    d = json.loads((out_dir / "sample_listings.json").read_text())
    if index not in d["indices"]:
        sys.exit(f"index {index} is not in the sample of survey.py: {d['indices']}")
    ds = d["datasets"][d["indices"].index(index)]
    accession = ds["prefix"].split("/")[1].split("-")[0]
    subject = ds["first_subject"].rstrip("/").split("/")[-1]
    keys = sorted(k for k, _ in ds["subject_keys"] if "/func/" in k and "_task-" in k)
    if not keys:
        sys.exit(f"index {index}: the first subject has no key with a task label under func/")
    name = keys[0].rsplit("/", 1)[-1]
    args = [accession, subject]
    for option, entity in (("--task", "task"), ("--session", "ses"), ("--run", "run")):
        m = re.search(rf"_{entity}-([A-Za-z0-9]+)", name)
        if m:
            args += [option, m.group(1)]
    print(
        f"index {index} of {d['n_prefixes']} prefixes: fMRIPrep"
        f" {fmriprep_version(ds.get('description'))}, {ds.get('n_subject_dirs')} subject folders",
        file=sys.stderr,
    )
    print(" ".join(args))


if __name__ == "__main__":
    main()
