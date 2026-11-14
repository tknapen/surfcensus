"""The local files of one session that pack_run.py packs: which runs, in which order, with what.

A module for pack_run.py and check_player.py; it has no command line.

DATA holds two folders, as pipeline/fetch.py writes them for the worked example and as
session_fetch.py names them for another subject:

  deriv/<stem>_space-T1w_desc-preproc_bold.nii.gz     the preprocessed series (the display grid)
  deriv/<stem>_space-T1w_desc-brain_mask.nii.gz       its brain mask
  deriv/<stem>_from-scanner_to-T1w_mode-image_xfm.txt the run's own stored transform
  deriv/<stem>_desc-confounds_timeseries.tsv          framewise displacement, standardised DVARS
  deriv/<stem>_space-T1w_desc-preproc_bold.json       RepetitionTime (else the NIfTI header)
  deriv/*from-T1w_to-fsnative_mode-image_xfm.txt      for the surfaces (one file)
  deriv/<stem>_boldref.nii.gz                         BOLD-space reference image: which affine
                                                      of the raw header (session.raw_affine)
  deriv/<stem>_desc-summary_bold.html                 fMRIPrep's summary report of the run:
                                                      distortion and slice-timing correction
  fetch_log.json                                      of fetch_run.py: the runs not fetched
  raw/<stem>_bold.nii.gz                              the raw series
  raw/<stem>_bold.json                                its sidecar, when the dataset has one

<stem> is a BIDS stem (sub-.._ses-.._task-.._run-..) or what is left of one (run-1 in the worked
example). A raw run and a derivative run are the same run when session.run_key of their names is
equal, so run-01 and run-1 match.
"""

import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from session import (
    RULE_TIME,
    acquisition_seconds,
    name_order,
    parse_name,
    pick_anchor,
    run_key,
    session_of,
    sidecar_time,
)

PREPROC = "_space-T1w_desc-preproc_bold.nii.gz"
MASK = "_space-T1w_desc-brain_mask.nii.gz"
XFM = "_from-scanner_to-T1w_mode-image_xfm.txt"
FSNATIVE = "*from-T1w_to-fsnative_mode-image_xfm.txt"
NON_STEADY = "non_steady_state_outlier"
SUMMARY = "_desc-summary_bold.html"

# the states that the manifest records for a correction (preprocessing_of)
APPLIED, NOT_APPLIED, UNKNOWN = "applied", "not applied", "unknown"
SOURCE_REPORT = "fMRIPrep's summary report of the run"
SOURCE_LISTING = "the listing only: fMRIPrep's distortion-correction figure of the run is listed"
SOURCE_NONE = "not recorded: no summary report of the run under deriv/"
MOTION_SOURCE = (
    "fMRIPrep applies head-motion correction to every preprocessed series; its summary report"
    " does not state it per run"
)


class InputError(ValueError):
    """The files under DATA cannot be packed as one session; the message says why."""


@dataclass
class Run:
    """One run of the session with the paths of its files."""

    stem: str  # the derivative's stem; names a dataset, so it never goes into a pack
    preproc: Path
    mask: Path
    xfm: Path
    confounds: Path
    raw: Path
    sidecar: Path | None
    acquisition_time: str | None
    id: str = ""  # R1, R2, ... in acquisition order
    anchor: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Session:
    """The runs of one session in acquisition order, the anchor first by either rule."""

    label: str  # '' for a dataset without ses- folders
    runs: list
    rule: str  # session.RULE_TIME or session.RULE_NAME
    why: str  # what the rule saw; holds an AcquisitionTime under the time rule, so not for a pack
    left_out: list  # (stem, reason) of runs under DATA that lack a file
    fsnative: Path | None


def _stems(folder, suffix):
    return {p.name[: -len(suffix)]: p for p in sorted(Path(folder).glob("*" + suffix))}


def find_session(data, session=None):
    """The session under DATA: its runs with every file, ordered, with the anchor marked.

    session: the label without 'ses-' when DATA holds more than one. Raises InputError when DATA
    holds no run with every file, more than one subject, or several sessions and none is named.
    """
    data = Path(data)
    deriv, raw = data / "deriv", data / "raw"
    if not deriv.is_dir() or not raw.is_dir():
        raise InputError("DATA needs the folders deriv/ and raw/")
    preproc = _stems(deriv, PREPROC)
    if not preproc:
        raise InputError("deriv/ holds no *" + PREPROC)
    subjects = {parse_name(stem).get("sub", "") for stem in preproc}
    if len(subjects) > 1:
        raise InputError(f"deriv/ holds runs of {len(subjects)} subjects; a pack is one session")
    labels = sorted({session_of(parse_name(stem)) for stem in preproc})
    if session is None:
        if len(labels) > 1:
            raise InputError(f"deriv/ holds {len(labels)} sessions; name one with --session")
        session = labels[0]
    elif session not in labels:
        raise InputError("deriv/ holds no run of the session that --session names")
    raw_by_key = {}
    for stem, path in _stems(raw, "_bold.nii.gz").items():
        ent = parse_name(stem)
        if "echo" not in ent:  # multi-echo runs are not handled
            raw_by_key[session_of(ent), run_key(ent)] = (stem, path)
    runs, left_out = [], []
    for stem, path in preproc.items():
        ent = parse_name(stem)
        if session_of(ent) != session:
            continue
        confounds = sorted(deriv.glob(stem + "_desc-confounds_*.tsv"))
        files = {
            "brain mask": deriv / (stem + MASK),
            "scanner-to-T1w transform": deriv / (stem + XFM),
            "confounds table": confounds[0] if confounds else deriv / (stem + "_desc-confounds"),
        }
        missing = [what for what, p in files.items() if not p.is_file()]
        if (session, run_key(ent)) not in raw_by_key:
            missing.append("raw series")
        if missing:
            left_out.append((stem, "no " + ", no ".join(missing)))
            continue
        raw_stem, raw_path = raw_by_key[session, run_key(ent)]
        sidecar = raw / (raw_stem + "_bold.json")
        runs.append(
            Run(
                stem=stem,
                preproc=path,
                mask=files["brain mask"],
                xfm=files["scanner-to-T1w transform"],
                confounds=files["confounds table"],
                raw=raw_path,
                sidecar=sidecar if sidecar.is_file() else None,
                acquisition_time=sidecar_time(sidecar) if sidecar.is_file() else None,
            )
        )
    if not runs:
        raise InputError(
            "no run of the session has every file: " + "; ".join(r for _, r in left_out)
        )
    anchor = pick_anchor([(run.stem, run.acquisition_time) for run in runs])
    if anchor.rule == RULE_TIME:

        def order(run):
            return (acquisition_seconds(run.acquisition_time)[1], name_order(parse_name(run.stem)))
    else:

        def order(run):
            return (0.0, name_order(parse_name(run.stem)))

    runs.sort(key=lambda run: (order(run), run.stem))
    # the anchor is the first run by either rule; put it first also among equal times
    runs.sort(key=lambda run: run.stem != anchor.name)
    for n, run in enumerate(runs, start=1):
        run.id, run.anchor = f"R{n}", run.stem == anchor.name
    fsnative = sorted(deriv.glob(FSNATIVE))
    return Session(
        session, runs, anchor.rule, anchor.why, left_out, fsnative[0] if fsnative else None
    )


def repetition_time(run, header_zooms):
    """TR in seconds: RepetitionTime of the derivative's sidecar, else the NIfTI header."""
    sidecar = run.preproc.with_name(run.preproc.name[: -len(".nii.gz")] + ".json")
    try:
        value = json.loads(sidecar.read_text()).get("RepetitionTime")
    except (OSError, ValueError, AttributeError):
        value = None
    if isinstance(value, int | float) and value > 0:
        return float(value)
    return float(header_zooms[3]) if len(header_zooms) > 3 else 0.0


def read_confounds(path):
    """(framewise displacement, standardised DVARS, indices of non-steady-state volumes).

    The two series are lists with None where the table has no number (the first volume). The
    non-steady-state volumes are those with a 1 in a non_steady_state_outlier column; an empty
    list when the table has no such column.
    """
    table = pd.read_csv(path, sep="\t")

    def column(name):
        if name not in table:
            return [None] * len(table)
        values = pd.to_numeric(table[name], errors="coerce")
        return [None if pd.isna(v) else round(float(v), 5) for v in values]

    flags = [c for c in table.columns if c.startswith(NON_STEADY)]
    non_steady = sorted({int(i) for c in flags for i in table.index[table[c] == 1]})
    return column("framewise_displacement"), column("std_dvars"), non_steady


def _report_line(text, label):
    """The text after '<label>:' in an <li> of fMRIPrep's summary report, tags removed."""
    m = re.search(r"<li>\s*" + re.escape(label) + r"\s*:\s*(.*?)</li>", text, re.S | re.I)
    if not m:
        return None
    value = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
    return " ".join(value.split())[:160] or None


def report_facts(text):
    """What fMRIPrep's per-run summary report (figures/<stem>_desc-summary_bold.html) says about
    two corrections of the preprocessed series. Returns a dict:

      sdc            'applied', 'not applied' or 'unknown' (the line 'Susceptibility distortion
                     correction: ...'; 'None' is not applied, any other value applied)
      sdc_method     the report's words for the method when applied, e.g. 'FLB ("fieldmap-less",
                     SyN-based)' or 'FMB (fieldmap-based) - phase-difference map'; else None
      slice_timing   'applied', 'not applied' or 'unknown' (the line 'Slice timing correction:
                     ...'; 'Not applied' is not applied, a value that starts with 'Applied' is
                     applied, anything else unknown)
    """
    sdc_text = _report_line(text, "Susceptibility distortion correction")
    if sdc_text is None:
        sdc, method = UNKNOWN, None
    elif sdc_text.lower() == "none":
        sdc, method = NOT_APPLIED, None
    else:
        sdc, method = APPLIED, sdc_text
    stc_text = _report_line(text, "Slice timing correction")
    if stc_text is None:
        stc = UNKNOWN
    elif stc_text.lower().startswith("not applied"):
        stc = NOT_APPLIED
    elif stc_text.lower().startswith("applied"):
        stc = APPLIED
    else:
        stc = UNKNOWN
    return {"sdc": sdc, "sdc_method": method, "slice_timing": stc}


def read_fetch_log(data):
    """The '_session' entry of DATA/fetch_log.json that fetch_run.py writes, or {} (the worked
    example's fetch_log.json of pipeline/fetch.py has none)."""
    try:
        log = json.loads((Path(data) / "fetch_log.json").read_text())
    except (OSError, ValueError):
        return {}
    entry = log.get("_session") if isinstance(log, dict) else None
    return entry if isinstance(entry, dict) else {}


def preprocessing_of(run, session_log):
    """What the pipeline did to the preprocessed series of a run, as the manifest records it:
    head-motion correction (always, by fMRIPrep's design), and whether it corrected for
    susceptibility distortion (and by which method) and for slice timing (see report_facts for
    the values). The raw stream has none of these; the page states them so that a difference
    between the two streams can be read as the effect of these steps.

    The source is the run's summary report, deriv/<stem>_desc-summary_bold.html, which
    fetch_run.py fetches with the run. Without it, a run whose distortion-correction figure
    (figures/<stem>_desc-sdc_bold.svg) the listing has counts as applied with an unknown method;
    otherwise both are 'unknown'. 'sdc_figure_listed' is that listing fact as fetch_run.py
    recorded it (None when not recorded), kept beside the report's statement as a cross-check.
    """
    listed = (session_log.get("sdc_figure_listed") or {}).get(run.stem)
    listed = bool(listed) if listed is not None else None
    report = run.preproc.parent / (run.stem + SUMMARY)
    if report.is_file():
        facts = report_facts(report.read_text(errors="replace"))
        source = SOURCE_REPORT
    elif listed:
        facts = {"sdc": APPLIED, "sdc_method": None, "slice_timing": UNKNOWN}
        source = SOURCE_LISTING
    else:
        facts = {"sdc": UNKNOWN, "sdc_method": None, "slice_timing": UNKNOWN}
        source = SOURCE_NONE
    return {
        "motion_correction": APPLIED,
        "motion_correction_source": MOTION_SOURCE,
        **facts,
        "source": source,
        "sdc_figure_listed": listed,
    }
