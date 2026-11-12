"""Provenance from text: build stamp, recon-all logs, ``dataset_description.json``.

Pure functions on text: no file access, no network. Bytes become text through
:func:`decode_text`. No function here raises on any text.

What the rules rest on:

* the logs and build stamps of two subjects of the OpenNeuroDerivatives bucket, read on
  2026-11-12: ``ds000102`` sub-01 (FreeSurfer 6.0.1 under fMRIPrep 21.0.2) and ``ds003020``
  sub-UTS01 (FreeSurfer 7.3.2), three logs each;
* ``scripts/recon-all`` of FreeSurfer v6.0.0 and v7.4.1, read the same day: the lines the
  script writes at the start and at the end of an invocation, and the spellings of the flags.

No log of a plain ``recon-all -all`` run was looked at. Its expected layout (one invocation,
both truncation lines in ``recon-all.log``, left hemisphere first) follows from the script
and is an assumption.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from surfcensus.records import Provenance
from surfcensus.sources.base import HEMIS, FileRef, FileRole, Hemi
from surfcensus.sources.patterns import ROLE_CANDIDATES

__all__ = [
    "EDIT_FLAGS",
    "BuildStamp",
    "Provenance",
    "ReconLog",
    "decode_text",
    "parse_build_stamp",
    "parse_dataset_description",
    "parse_recon_log",
    "subject_provenance",
]

EDIT_FLAGS: tuple[str, ...] = (
    "-autorecon2-cp",
    "-autorecon2-wm",
    "-autorecon2-pial",
    "-autorecon-pial",
)
"""recon-all flags of a rerun after manual edits. ``-autorecon2-pial`` is not a flag of
v6.0.0 or v7.4.1; it is kept for logs of older versions (from memory, not checked)."""

# The spellings that set and clear each switch in the argument parser of recon-all, the same
# in v6.0.0 and v7.4.1 unless noted. The parser compares whole arguments, case-sensitively,
# and lists every spelling it takes: "-t2pial" is one, "-T2PIAL" is not.
_T2PIAL_ON = frozenset({"-T2pial", "-t2pial", "-T2pial-only", "-autorecon3-T2pial"})
_T2PIAL_OFF = frozenset({"-noT2pial", "-no-T2pial", "-no_T2pial", "-no-t2pial", "-no_t2pial"})
_FLAIRPIAL_ON = frozenset({"-FLAIRpial", "-flairpial"})
_FLAIRPIAL_OFF = frozenset(
    {"-noFLAIRpial", "-no-FLAIRpial", "-no_FLAIRpial", "-no-flairpial", "-no_flairpial"}
)
# "-highres" is v7.4.1 only. Neither version has a flag that clears "-hires"; "-nohires" is
# here so that the three switches follow one rule, and cannot occur in a log of these versions.
# "-conf2hires" (v7.4.1) is another stream and is left to ``flags``.
_HIRES_ON = frozenset({"-hires", "-highres"})
_HIRES_OFF = frozenset({"-nohires"})

_LOG_ROLES: tuple[FileRole, ...] = (
    FileRole.RECON_LOG,
    FileRole.RECON_LOG_LH,
    FileRole.RECON_LOG_RH,
)
_HEMI_LOG: dict[Hemi, FileRole] = {"lh": FileRole.RECON_LOG_LH, "rh": FileRole.RECON_LOG_RH}

# A bare carriage return ends a line too: the logs hold progress output such as
# "\rstep 000: RMS=0.158 ...\rstep 005: ..." on what "\n" alone would make one line.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_SUBJID = re.compile(r"subjid\s+\S+\s*")
_FLAG = re.compile(r"-[A-Za-z][A-Za-z0-9_-]*")
# the two flags of recon-all that do not start with a letter (the 3T atlas)
_DIGIT_FLAGS = frozenset({"-3T", "-3t"})
_ACCESSION = re.compile(r"ds[0-9]{6}")
_STAMP_LINE = re.compile(r"build-stamp\.txt:\s*(\S.*)")
# Numbers are bounded in length: int() refuses very long digit strings, and nothing here
# may raise. A longer number does not match at all.
_TRUNCATIONS = re.compile(r"thickness calculation complete, ([0-9]{1,9}):([0-9]{1,9}) truncations")
_HOLES = re.compile(
    r"orig\.nofix lhholes\s*=\s*([0-9]{1,9}),\s*rhholes\s*=\s*([0-9]{1,9})(?![0-9])"
)
_VERSION = re.compile(r"(?:^|-)v?([0-9]{1,4}\.[0-9]{1,4}(?:\.[0-9]{1,4})?)(?=-|$)")
_PARTICIPANT = re.compile(r"sub-[A-Za-z0-9]+")
_BOM = chr(0xFEFF)
_FINISHED = "finished without error"
_ERRORS = "exited with ERRORS"


def decode_text(data: bytes) -> str:
    """The text of a Tier A file: UTF-8, undecodable bytes replaced, a byte-order mark dropped."""
    return data.decode("utf-8", errors="replace").removeprefix(_BOM)


@dataclass(frozen=True)
class BuildStamp:
    raw: str
    """First non-blank line of ``scripts/build-stamp.txt``, stripped; empty if there is none."""
    version: str | None
    """For example ``6.0.1``; None for a development build."""
    major: int | None


@dataclass(frozen=True)
class ReconLog:
    invocations: tuple[tuple[str, ...], ...]
    """Flag names of each recon-all invocation, in file order; within an invocation in the
    order of the command line, repeats included."""
    build_stamp: str | None
    """From the last ``build-stamp.txt: ...`` line."""
    truncations: tuple[tuple[int, int], ...]
    """Every ``thickness calculation complete, N:M truncations`` line, in file order."""
    holes: tuple[int, int] | None
    """The last ``orig.nofix lhholes = N, rhholes = M`` line."""
    finished: bool | None
    """Whether the last invocation ended without error; None when the text is partial or its
    last invocation has no closing line."""


def parse_build_stamp(text: str) -> BuildStamp:
    """The FreeSurfer version in a build stamp.

    ``freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a`` gives 6.0.1 and
    ``freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275`` gives 7.3.2: the first
    dash-delimited field that is a two- or three-part number, with or without a ``v``. A
    stamp without one, such as ``freesurfer-x86_64-redhat-linux-gnu-dev4-20090216`` of the
    fsaverage next to the subjects of ds000102, is a development build: version None.
    """
    raw = next((line.strip() for line in _LINE_BREAK.split(text) if line.strip()), "")
    match = _VERSION.search(raw)
    if match is None:
        return BuildStamp(raw=raw, version=None, major=None)
    version = match.group(1)
    return BuildStamp(raw=raw, version=version, major=int(version.split(".")[0]))


def _is_flag(token: str) -> bool:
    if _FLAG.fullmatch(token) is None and token not in _DIGIT_FLAGS:
        return False
    # A flag-shaped piece of a path with spaces could carry an accession; no flag of
    # recon-all holds "ds" and six digits.
    return _ACCESSION.search(token) is None


def _flag_names(arguments: str) -> tuple[str, ...]:
    tokens = arguments.split()
    if tokens and tokens[0].endswith("recon-all"):
        tokens = tokens[1:]
    return tuple(token for token in tokens if _is_flag(token))


def parse_recon_log(text: str, *, partial: bool = False) -> ReconLog:
    """What a recon-all log says about how the subject was processed.

    recon-all appends to its log, so one file holds every invocation: four in
    ``recon-all.log`` of ds000102 sub-01 and five in that of ds003020 sub-UTS01, two in each
    of their per-hemisphere logs.

    An invocation is found at each line ``subjid NAME``, which the script writes after the
    command line. The arguments are the nearest non-blank line above it: alone on that line in
    FreeSurfer 6, after the path of the script in FreeSurfer 7. Of the arguments only flag
    names are kept: tokens of the form ``-name`` made of letters, digits, ``_`` and ``-``
    after a first letter, plus ``-3T`` and ``-3t``. A path, a number or a subject name is not
    of that form, and a token that holds ``ds`` and six digits is dropped whatever its form.
    What can still pass is a value that itself looks like a flag, such as the piece
    ``-study`` of a path ``/data/my -study/x`` with a space in it.

    The invocations that recon-all starts itself under its ``-make`` option write no
    ``subjid`` line (read in the script, not seen in a log) and are not counted.

    ``finished`` is the verdict of the last invocation: True for ``finished without error``,
    False for ``exited with ERRORS``, whichever comes last after its ``subjid`` line, and None
    when neither does, as in a run that was killed or is still going.

    ``partial``: the text is the head of a log. ``finished`` is then None, and a last line
    without a line end is ignored, because it may be cut.
    """
    lines = _LINE_BREAK.split(text)
    if partial:
        lines.pop()  # what follows the last line break: nothing, or a line that may be cut

    invocations: list[tuple[str, ...]] = []
    build_stamp: str | None = None
    truncations: list[tuple[int, int]] = []
    holes: tuple[int, int] | None = None
    finished: bool | None = None
    arguments = ""
    for line in lines:
        if _SUBJID.fullmatch(line):
            invocations.append(_flag_names(arguments))
            finished = None  # the verdict of an earlier invocation says nothing about this one
        if line.strip():
            arguments = line
        stamp = _STAMP_LINE.match(line)
        if stamp is not None:
            build_stamp = stamp.group(1).strip()
        for match in _TRUNCATIONS.finditer(line):
            truncations.append((int(match.group(1)), int(match.group(2))))
        for match in _HOLES.finditer(line):
            holes = (int(match.group(1)), int(match.group(2)))
        good, bad = line.rfind(_FINISHED), line.rfind(_ERRORS)
        if good >= 0 or bad >= 0:
            finished = good > bad
    return ReconLog(
        invocations=tuple(invocations),
        build_stamp=build_stamp,
        truncations=tuple(truncations),
        holes=holes,
        finished=None if partial else finished,
    )


def _decides(invocation: Iterable[str], on: frozenset[str], off: frozenset[str]) -> bool | None:
    """What one command line leaves a switch at: its last setting wins; None if it has none."""
    decision: bool | None = None
    for flag in invocation:
        if flag in on:
            decision = True
        elif flag in off:
            decision = False
    return decision


def _switched_on(
    invocations: Iterable[tuple[str, ...]], on: frozenset[str], off: frozenset[str]
) -> bool:
    return any(_decides(invocation, on, off) is True for invocation in invocations)


def _pial_variant(files: Mapping[FileRole, FileRef]) -> str | None:
    """``pial``, ``pial.T1``, ``pial.T2`` or ``pial.FLAIR``: the left pial file that is read."""
    ref = files.get(FileRole.LH_PIAL)
    if ref is None:
        return None
    name = ref.variant if ref.variant is not None else PurePosixPath(ref.location).name
    names = [PurePosixPath(candidate).name for candidate in ROLE_CANDIDATES[FileRole.LH_PIAL]]
    return name.removeprefix("lh.") if name in names else None


def subject_provenance(
    texts: Mapping[FileRole, str],
    files: Mapping[FileRole, FileRef],
    subject_label: str,
    *,
    partial: bool = False,
) -> Provenance:
    """What the build stamp, the logs and the file names say about one subject.

    ``texts``: the decoded content of ``build_stamp``, ``recon_log``, ``recon_log_lh`` and
    ``recon_log_rh``, each present only if it could be read; other roles are ignored.
    ``files``: the subject's listing. ``partial``: at least one log is a head (see
    :func:`parse_recon_log`); a caller that read a whole log through a byte range passes False.

    * Version: from the build stamp file; if that is absent or holds no version, from the
      last stamp line of ``recon_log``. ``build_stamp`` is the stamp the version was read
      from, or the first stamp there is when none holds a version.
    * ``flags``: the sorted union over every invocation found in the three logs;
      ``n_invocations`` their number (8 for ds000102 sub-01: 4 + 2 + 2), None when no log was
      read. Of heads, both say what the heads hold.
    * ``t2pial``, ``flairpial``, ``hires``: within one command line the last spelling that
      sets or clears the switch decides, and the value is True if any command line decides
      True. A pial variant ``pial.T2`` or ``pial.FLAIR`` makes its switch True whatever the
      logs say.
    * ``expert``: any ``-expert``; ``xopts``: any flag that starts with ``-xopts-``;
      ``edited``: any of :data:`EDIT_FLAGS`.
    * These six are False only when the logs can show it: every log the subject has in
      ``files`` was read, in full, and at least one invocation was found. fMRIPrep passes
      ``-noT2pial -noFLAIRpial`` to FreeSurfer 7, which then reads False. Otherwise what was
      not seen is None: a head of ds000102 sub-01 of 64 kB holds two of the four invocations of
      ``recon-all.log``, and a later one could carry the flag.
    * ``finished``: from ``recon_log`` alone.
    * ``holes``: from ``recon_log``, else from the first per-hemisphere log that has the line
      (it is in ``recon_log`` in both subjects looked at).
    * ``truncations`` of a hemisphere: the last such line of its own log (where the
      per-hemisphere runs of fMRIPrep write it); else, when ``recon_log`` has exactly two,
      the first for ``lh`` and the second for ``rh`` (assumed for a plain run); else None.
    * ``participant``: the leading ``sub-<label>`` of the directory name, else the whole name;
      ``longitudinal``: the name contains ``.long.``.
    """
    logs = {
        role: parse_recon_log(texts[role], partial=partial) for role in _LOG_ROLES if role in texts
    }
    main = logs.get(FileRole.RECON_LOG)

    candidates: list[BuildStamp] = []
    if FileRole.BUILD_STAMP in texts:
        candidates.append(parse_build_stamp(texts[FileRole.BUILD_STAMP]))
    if main is not None and main.build_stamp is not None:
        candidates.append(parse_build_stamp(main.build_stamp))
    candidates = [candidate for candidate in candidates if candidate.raw]
    stamp = next(
        (candidate for candidate in candidates if candidate.version is not None),
        candidates[0] if candidates else None,
    )

    invocations = [invocation for log in logs.values() for invocation in log.invocations]
    flags = {flag for invocation in invocations for flag in invocation}
    complete = (
        not partial
        and bool(invocations)
        and all(role in texts for role in _LOG_ROLES if role in files)
    )
    pial_variant = _pial_variant(files)

    def established(seen: bool) -> bool | None:
        if seen:
            return True
        return False if complete else None

    truncations: dict[Hemi, tuple[int, int] | None] = {}
    for index, hemi in enumerate(HEMIS):
        own = logs.get(_HEMI_LOG[hemi])
        if own is not None and own.truncations:
            truncations[hemi] = own.truncations[-1]
        elif main is not None and len(main.truncations) == 2:
            truncations[hemi] = main.truncations[index]
        else:
            truncations[hemi] = None

    participant = _PARTICIPANT.match(subject_label)
    return Provenance(
        build_stamp=stamp.raw if stamp is not None else None,
        fs_version=stamp.version if stamp is not None else None,
        fs_major=stamp.major if stamp is not None else None,
        flags=tuple(sorted(flags)),
        n_invocations=len(invocations) if logs else None,
        finished=main.finished if main is not None else None,
        t2pial=established(
            pial_variant == "pial.T2" or _switched_on(invocations, _T2PIAL_ON, _T2PIAL_OFF)
        ),
        flairpial=established(
            pial_variant == "pial.FLAIR" or _switched_on(invocations, _FLAIRPIAL_ON, _FLAIRPIAL_OFF)
        ),
        hires=established(_switched_on(invocations, _HIRES_ON, _HIRES_OFF)),
        expert=established("-expert" in flags),
        xopts=established(any(flag.startswith("-xopts-") for flag in flags)),
        edited=established(any(flag in flags for flag in EDIT_FLAGS)),
        pial_variant=pial_variant,
        holes=next((log.holes for log in logs.values() if log.holes is not None), None),
        truncations=truncations,
        participant=(participant.group(0) if participant else subject_label) or None,
        longitudinal=".long." in subject_label,
    )


def _version(entry: object) -> str | None:
    version = entry.get("Version") if isinstance(entry, dict) else None
    return version.strip() if isinstance(version, str) and version.strip() else None


def parse_dataset_description(text: str) -> str | None:
    """The fMRIPrep version in a ``dataset_description.json``, or None.

    ``GeneratedBy[*].Version`` of the entry whose ``Name`` is ``fMRIPrep`` (ignoring case),
    as in ``ds000102-fmriprep`` (fMRIPrep 21.0.2); else ``PipelineDescription.Version``, the
    form of older fMRIPrep versions (from memory, no such file was looked at). Text that
    is not JSON, or JSON of another shape, gives None: this never raises.
    """
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    generated = data.get("GeneratedBy")
    for entry in generated if isinstance(generated, list) else []:
        name = entry.get("Name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.lower() == "fmriprep" and _version(entry):
            return _version(entry)
    return _version(data.get("PipelineDescription"))
