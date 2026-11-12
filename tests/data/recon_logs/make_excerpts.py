"""Cut the excerpts in this directory from six real recon-all logs.

The logs are those of two subjects in the OpenNeuroDerivatives bucket, fetched anonymously on
2026-11-12 from ``https://openneuro-derivatives.s3.amazonaws.com/fmriprep/``:

* ``ds000102-fmriprep/sourcedata/freesurfer/sub-01/scripts/`` (FreeSurfer 6.0.1): ``fs6_*``;
* ``ds003020-fmriprep/sourcedata/freesurfer/sub-UTS01/scripts/`` (FreeSurfer 7.3.2): ``fs7_*``;

``recon-all.log``, ``recon-all-lh.log`` and ``recon-all-rh.log`` of each. To cut them again,
download the six files into ``SOURCE/ds000102/`` and ``SOURCE/ds003020/`` under their own
names and run, from the repository root::

    uv run python -I tests/data/recon_logs/make_excerpts.py SOURCE

An excerpt keeps, of each invocation, the lines recon-all writes at its start (from the
``New invocation`` separator to the ``uname`` line), and every line that holds truncation
counts, hole counts or the closing verdict, with a line or two around them. Everything else
is replaced by one line ``[... N lines cut ...]``.

What is changed in the lines that are kept, so that no excerpt names a person or a machine:

* the two directory prefixes of the logs, which hold a user name and an allocation number,
  become ``/data/`` and ``/data/work/``;
* the dataset accession becomes ``ds999999`` (a path in an excerpt then reads
  ``/data/ds999999-fmriprep/sourcedata/freesurfer/sub-01``);
* the host name in the ``uname`` line becomes ``HOST``.

Subject labels, dates, the kernel version and ``/opt/freesurfer`` are kept. The script stops
if anything of the original prefixes, the host or an accession is left, and if an excerpt
does not parse to the same :class:`ReconLog` as the log it was cut from.

``plain_recon-all.log`` is not cut from a log. It is written by hand after the lines that
``scripts/recon-all`` of FreeSurfer v6.0.0 writes for one ``recon-all -all`` run: one
invocation, and the thickness lines of both hemispheres in ``recon-all.log``, left first (the
script loops over ``lh rh``). No log of such a run was looked at.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from surfcensus.inventory.provenance import decode_text, parse_recon_log

HERE = Path(__file__).parent
SUBJECTS = {"ds000102": "fs6", "ds003020": "fs7"}
LOGS = ("recon-all.log", "recon-all-lh.log", "recon-all-rh.log")

SEPARATOR = re.compile(r"#?New#? invocation of recon-all")
SUBJID = re.compile(r"subjid\s+\S+\s*")
UNAME = re.compile(r"(Linux) \S+ ")
KEPT = (  # pattern, lines kept above, lines kept below
    (re.compile(r"thickness calculation complete"), 1, 1),
    (re.compile(r"orig\.nofix lhholes"), 1, 1),
    (re.compile(r"finished without error|exited with ERRORS"), 1, 0),
)
PREFIXES = (
    (re.compile(r"/scratch1/\w+/\w+/openneuro_derivatives/derivatives/fmriprep/"), "/data/"),
    (re.compile(r"/scratch1/\w+/\w+/work_dir/fmriprep/"), "/data/work/"),
)
ACCESSION = re.compile(r"ds\d{6}")
LEFT_OVER = re.compile(r"scratch|openneuro|work_dir|tacc|frontera|utexas|\w@\w|/home|/Users", re.I)


def kept_lines(lines: list[str]) -> list[int]:
    keep: set[int] = set()
    for index, line in enumerate(lines):
        if SUBJID.fullmatch(line):
            above = range(max(0, index - 12), index)
            first = next((i for i in above if SEPARATOR.match(lines[i])), max(0, index - 6))
            below = range(index, min(len(lines), index + 9))
            last = next((i for i in below if UNAME.match(lines[i])), index)
            keep.update(range(first, last + 1))
        for pattern, before, after in KEPT:
            if pattern.search(line):
                keep.update(range(max(0, index - before), min(len(lines), index + after + 1)))
    return sorted(keep)


def anonymous(line: str) -> str:
    for pattern, replacement in PREFIXES:
        line = pattern.sub(replacement, line)
    line = ACCESSION.sub("ds999999", line)
    return UNAME.sub(r"\1 HOST ", line, count=1) if UNAME.match(line) else line


def excerpt(text: str) -> str:
    lines = text.split("\n")
    if lines[-1] == "":  # what follows the final line end is not a line
        lines.pop()
    out: list[str] = []
    previous = -1
    for index in kept_lines(lines):
        if index - previous > 1:
            out.append(f"[... {index - previous - 1} lines cut ...]")
        out.append(anonymous(lines[index]))
        previous = index
    if len(lines) - previous > 1:
        out.append(f"[... {len(lines) - previous - 1} lines cut ...]")
    return "\n".join(out) + "\n"


def main(source: Path) -> None:
    for accession, prefix in SUBJECTS.items():
        for name in LOGS:
            text = decode_text((source / accession / name).read_bytes())
            cut = excerpt(text)
            residue = [line for line in cut.split("\n") if LEFT_OVER.search(line)]
            if residue or accession in cut or "\r" in cut:
                raise SystemExit(f"{accession}/{name}: not anonymous: {residue[:3]}")
            if parse_recon_log(cut) != parse_recon_log(text):
                raise SystemExit(f"{accession}/{name}: the excerpt parses differently")
            target = HERE / f"{prefix}_{name}"
            target.write_text(cut, encoding="utf-8", newline="\n")
            print(f"{target.name}: {cut.count(chr(10))} of {text.count(chr(10))} lines")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
