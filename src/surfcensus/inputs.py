"""The input set: the files of a review named one by one, and the patterns that name them
(open question O31, ``docs/dev/input_plan.md``; steps I0 and I1).

An input set lists, per subject and session, the FreeSurfer subject directory and, per
functional run, its raw series, its preprocessed series and the companions of the
preprocessed series (mask, transform to the anatomy, reference image, confounds table,
report, sidecars). Paths are absolute. The set says nothing that a name would otherwise say:
the space of the preprocessed series, which image was fixed when the transform was estimated,
and how the anatomy relates to FreeSurfer's frame are statements of the user
(:class:`Statements`), never guesses.

Patterns. One pattern per data type, an absolute path with the glob wildcards ``*`` and ``?``
(within one path component) and the named fields ``{sub}``, ``{ses}`` and ``{run}``, each of
which matches one or more characters of one path component. A field named twice in one
pattern takes one value. ``{{`` and ``}}`` are literal braces; ``[`` and ``]`` are literal.
A file matched by a pattern in two ways (``{sub}_{run}`` on ``a_b_c``) is an error.

Pairing a raw series with its preprocessed series (the plan, "Wildcards per data type"):

1. ``fields``: both patterns carry ``{run}``; files with equal subject, session and run are
   one run. Where both names of a pair also parse as BIDS and name different runs, the
   resolver stops: the two rules would disagree.
2. ``bids``: neither carries ``{run}`` and every name parses as BIDS; paired by subject,
   session and ``func.bids.run_key``.
3. ``order``: only when asked; within each subject and session the raw and the preprocessed
   files are paired in the order of their paths, and the counts must be equal.

Otherwise :func:`resolve` raises :class:`InputError`, whose lines list what it found. A run
with a raw series and no preprocessed one, or the reverse, is kept with the other side None,
so that the table shows it and the planner gives it its reason.

Companions are keyed as the pairing is: under ``fields`` a pattern with ``{run}`` names the
file of one run and one without names the file of every run of its session (a transform per
session, say); under ``bids`` a name that parses gives the run by its run key, or the session
when its key is the session's alone; under ``order`` a pattern with one file per session is
of the session and one with a file per run is paired in path order. A file of a role that no
run takes is reported, and two files of one role for one run stop the resolver.

Nothing here opens a file: patterns are resolved from directory listings.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from surfcensus._io import atomic_path
from surfcensus.func import bids

INPUTS_FILE = "inputs.yaml"
"""The input set of a run, in its run directory."""

SCHEMA_VERSION = 1

FIELDS: tuple[str, ...] = ("sub", "ses", "run")
"""The named fields a pattern may carry."""

RUN_ROLES: tuple[str, ...] = (
    "raw",
    "raw_sidecar",
    "pre",
    "pre_sidecar",
    "mask",
    "xfm",
    "reference",
    "confounds",
    "report",
)
"""The files of one functional run, by their names in an input set."""

COMPANION_ROLES: tuple[str, ...] = tuple(r for r in RUN_ROLES if r not in ("raw", "pre"))

SESSION_ROLES: tuple[str, ...] = ("fsnative_xfm",)
"""The files of a session: the transform between the anatomy and FreeSurfer's frame."""

FREESURFER = "freesurfer"
"""The role of the FreeSurfer subject directory of a session."""

PATTERN_ROLES: tuple[str, ...] = (*RUN_ROLES, *SESSION_ROLES, FREESURFER)

OPTIONS: dict[str, str] = {
    "raw": "--raw-files",
    "raw_sidecar": "--raw-sidecar-files",
    "pre": "--pre-files",
    "pre_sidecar": "--pre-sidecar-files",
    "mask": "--mask-files",
    "xfm": "--xfm-files",
    "reference": "--reference-files",
    "confounds": "--confounds-files",
    "report": "--report-files",
    "fsnative_xfm": "--fsnative-xfm-files",
    FREESURFER: "--fs-subject",
}
"""The command-line option of each pattern (step I3 adds them to ``review`` and
``epi-session``)."""

PreSpace = Literal["anat", "native"]
"""Where the preprocessed series lies: in the anatomy's world, or in the run's own frame."""

XfmFixed = Literal["anat", "epi"]
"""Which image was the fixed one when the transform to the anatomy was estimated. An ITK
text transform of ANTs (``antsRegistration -f FIXED -m MOVING``) maps points of the fixed
image to points of the moving one; the planner needs the map from the anatomy to the EPI
reference, so ``anat`` is read as it is and ``epi`` is inverted."""

FsFrame = Literal["same", "transform"]
"""The anatomy and FreeSurfer's scanner RAS: the same frame, or related by the session's
``fsnative_xfm``."""

PairRule = Literal["fields", "bids", "order"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunInputs(_Model):
    """One functional run: what its files are, and by what key they were paired."""

    name: str
    """The run as the pages show it: the ``{run}`` value, the BIDS name without the subject,
    or (``order``) the name of the raw file without its extension."""
    raw: str | None = None
    raw_sidecar: str | None = None
    pre: str | None = None
    pre_sidecar: str | None = None
    mask: str | None = None
    xfm: str | None = None
    reference: str | None = None
    confounds: str | None = None
    report: str | None = None


class SessionInputs(_Model):
    label: str | None = None
    """Without ``ses-``; None for a dataset without sessions."""
    freesurfer: str | None = None
    fsnative_xfm: str | None = None
    runs: tuple[RunInputs, ...] = ()


class SubjectInputs(_Model):
    label: str
    """Without ``sub-``: ASCII letters and digits."""
    sessions: tuple[SessionInputs, ...] = ()


class Statements(_Model):
    """What the user states because no name says it. None: not stated."""

    pre_space: PreSpace | None = None
    xfm_fixed: XfmFixed | None = None
    fs_frame: FsFrame | None = None


class InputSet(_Model):
    schema_version: int = SCHEMA_VERSION
    patterns: dict[str, str]
    """Role to the pattern that named its files, so that a later call resolves them again."""
    pair: PairRule
    statements: Statements = Statements()
    subjects: tuple[SubjectInputs, ...] = ()
    unused: tuple[tuple[str, str], ...] = ()
    """``(role, path)`` of files that a pattern named and no run or session took."""


class InputError(ValueError):
    """A pattern that cannot be used, or files that cannot be paired. ``lines`` lists what
    was found, for the user to read."""

    def __init__(self, message: str, lines: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.lines = tuple(lines)


# --- patterns ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[*?]|[{}]")
_LABEL = re.compile(r"[A-Za-z0-9]+")
_RUN_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


@dataclass(frozen=True)
class Pattern:
    """A pattern turned into a glob for the listing and two expressions for the fields."""

    text: str
    glob: str
    greedy: re.Pattern[str]
    lazy: re.Pattern[str]
    fields: frozenset[str]


def compile_pattern(text: str) -> Pattern:
    """``InputError`` for a pattern that is not absolute, names an unknown field, holds a
    lone brace, or puts a field or a wildcard across a ``/``."""
    if not isinstance(text, str) or not text.strip():
        raise InputError("a pattern is a path")
    expanded = os.path.expanduser(text)
    if not os.path.isabs(expanded):
        raise InputError(f"{text}: a pattern is an absolute path")
    globs: list[str] = []
    greedy: list[str] = []
    lazy: list[str] = []
    fields: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(expanded):
        literal = expanded[pos : m.start()]
        globs.append(glob.escape(literal))
        greedy.append(re.escape(literal))
        lazy.append(re.escape(literal))
        token = m.group(0)
        if token in ("{{", "}}"):
            globs.append(glob.escape(token[0]))
            greedy.append(re.escape(token[0]))
            lazy.append(re.escape(token[0]))
        elif token in ("*", "?"):
            globs.append(token)
            greedy.append("[^/]*" if token == "*" else "[^/]")
            lazy.append("[^/]*?" if token == "*" else "[^/]")
        elif token in ("{", "}"):
            raise InputError(f"{text}: a lone {token!r}; write {token * 2} for a literal brace")
        else:
            name = m.group(1)
            if name not in FIELDS:
                raise InputError(
                    f"{text}: unknown field {{{name}}}; the fields are "
                    + ", ".join(f"{{{f}}}" for f in FIELDS)
                )
            globs.append("*")
            if name in fields:
                greedy.append(f"(?P={name})")
                lazy.append(f"(?P={name})")
            else:
                fields.append(name)
                greedy.append(f"(?P<{name}>[^/]+)")
                lazy.append(f"(?P<{name}>[^/]+?)")
        pos = m.end()
    tail = expanded[pos:]
    globs.append(glob.escape(tail))
    greedy.append(re.escape(tail))
    lazy.append(re.escape(tail))
    return Pattern(
        text=text,
        glob="".join(globs),
        greedy=re.compile("".join(greedy)),
        lazy=re.compile("".join(lazy)),
        fields=frozenset(fields),
    )


def match_files(pattern: Pattern, *, directories: bool = False) -> list[tuple[str, dict[str, str]]]:
    """The files (or, with ``directories``, the directories) a pattern names, in path order,
    each with its field values. ``InputError`` for one that the pattern matches in two ways."""
    out = []
    for path in sorted(glob.glob(pattern.glob)):
        if not (os.path.isdir(path) if directories else os.path.isfile(path)):
            continue
        posix = Path(path).as_posix()
        a = pattern.greedy.fullmatch(posix)
        b = pattern.lazy.fullmatch(posix)
        if a is None or b is None:
            continue
        if a.groupdict() != b.groupdict():
            raise InputError(
                f"{pattern.text}: {posix} is matched in two ways "
                f"({_fields_text(a.groupdict())} and {_fields_text(b.groupdict())}); "
                "put a literal between the fields"
            )
        out.append((posix, dict(a.groupdict())))
    return out


def _fields_text(fields: Mapping[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))


# --- keys ---------------------------------------------------------------------------------


def _label(value: str, prefix: str, what: str, source: str) -> str:
    label = value[len(prefix) :] if value.startswith(prefix) else value
    if not _LABEL.fullmatch(label):
        raise InputError(f"{source}: the {what} label {value!r} is not ASCII letters and digits")
    return label


@dataclass(frozen=True)
class _File:
    path: str
    sub: str | None
    ses: str | None
    run: str | None
    """The run within its session: the ``{run}`` value or the BIDS run key without ``ses``."""
    name: str | None


def _bids(path: str) -> bids.RunEntities | None:
    try:
        return bids.parse_entities(path)
    except ValueError:
        return None


def _run_of(entities: bids.RunEntities) -> str:
    key = bids.run_key(bids.RunEntities(**{**_asdict(entities), "ses": None}))
    return key


def _asdict(entities: bids.RunEntities) -> dict[str, str | None]:
    return {k: getattr(entities, k) for k in bids.ENTITIES}


def _keyed(
    role: str, path: str, fields: Mapping[str, str], rule: PairRule, subject: str | None
) -> _File:
    """Subject, session and run of one file. The fields of the pattern come first; a name
    that parses as BIDS gives what the fields do not; ``subject`` is the last resort."""
    entities = _bids(path)
    if "sub" in fields:
        sub = _label(fields["sub"], "sub-", "subject", path)
    elif entities is not None:
        sub = entities.sub
    elif subject is not None:
        sub = subject
    else:
        raise InputError(
            f"{OPTIONS[role]}: {path} names no subject; put {{sub}} into the pattern, "
            "or name the subject"
        )
    if "ses" in fields:
        ses: str | None = _label(fields["ses"], "ses-", "session", path)
    else:
        ses = entities.ses if entities is not None else None
    run = name = None
    if rule == "fields" and "run" in fields:
        run = name = fields["run"]
        if not _RUN_VALUE.fullmatch(run):
            raise InputError(f"{path}: the run {run!r} holds a character a run name may not hold")
    elif rule == "bids" and role not in (*SESSION_ROLES, FREESURFER):
        if entities is not None:
            run = _run_of(entities) or None
            name = bids.run_name(entities)
        elif role in ("raw", "pre"):
            raise InputError(f"{OPTIONS[role]}: {path} is no BIDS name")
    return _File(path, sub, ses, run, name)


# --- resolving ------------------------------------------------------------------------------


def _rule(
    patterns: Mapping[str, Pattern],
    files: Mapping[str, list[tuple[str, dict[str, str]]]],
    pair: str,
) -> PairRule:
    raw, pre = patterns["raw"], patterns["pre"]
    if pair == "order":
        return "order"
    if pair not in ("auto", "fields", "bids"):
        raise InputError(f"unknown pairing {pair!r}; it is auto, fields, bids or order")
    with_run = ("run" in raw.fields, "run" in pre.fields)
    if pair == "fields" or (pair == "auto" and any(with_run)):
        if not all(with_run):
            which = OPTIONS["pre"] if with_run[0] else OPTIONS["raw"]
            raise InputError(
                f"{which} has no {{run}}; put {{run}} into both patterns, or into neither"
            )
        return "fields"
    names = [p for role in ("raw", "pre") for p, _ in files[role]]
    bad = [p for p in names if _bids(p) is None]
    if bad:
        raise InputError(
            "the raw and the preprocessed files cannot be paired: these names are no BIDS "
            "names, and the patterns carry no {run} field. Put {sub}, {ses} and {run} into "
            "both patterns, or pair them in path order (--pair order) and read the table.",
            [f"no BIDS name: {p}" for p in bad[:20]]
            + ([f"... and {len(bad) - 20} more"] if len(bad) > 20 else []),
        )
    return "bids"


def resolve(
    patterns: Mapping[str, str],
    *,
    pair: str = "auto",
    subject: str | None = None,
    statements: Statements | None = None,
) -> InputSet:
    """The input set that the patterns name (module docstring). ``raw`` and ``pre`` are
    required; ``subject`` is the label of the one subject when the names give none."""
    unknown = sorted(set(patterns) - set(PATTERN_ROLES))
    if unknown:
        raise InputError(f"unknown roles: {', '.join(unknown)}")
    for role in ("raw", "pre"):
        if not patterns.get(role):
            raise InputError(f"{OPTIONS[role]} is needed")
    if subject is not None:
        subject = _label(subject, "sub-", "subject", "the subject given")
    compiled = {role: compile_pattern(text) for role, text in patterns.items() if text}
    files = {role: match_files(p, directories=role == FREESURFER) for role, p in compiled.items()}
    for role in ("raw", "pre"):
        if not files[role]:
            raise InputError(f"{OPTIONS[role]} {patterns[role]} names no file")
    rule = _rule(compiled, files, pair)
    keyed = {
        role: [_keyed(role, path, fields, rule, subject) for path, fields in found]
        for role, found in files.items()
    }
    runs = _pair(keyed["raw"], keyed["pre"], rule)
    taken: dict[tuple[str | None, str | None, str], dict[str, str]] = {k: {} for k in runs}
    session_files: dict[tuple[str | None, str | None], dict[str, str]] = {}
    unused: list[tuple[str, str]] = []
    sessions = sorted({(k[0], k[1]) for k in runs}, key=_session_order)
    for role in (*COMPANION_ROLES, *SESSION_ROLES, FREESURFER):
        if role not in keyed:
            continue
        session_level = role in SESSION_ROLES or role == FREESURFER
        _assign(
            role,
            keyed[role],
            compiled[role],
            rule,
            runs,
            taken,
            session_files,
            sessions,
            session_level,
            unused,
        )
    return InputSet(
        patterns={role: text for role, text in patterns.items() if text},
        pair=rule,
        statements=statements or Statements(),
        subjects=_subjects(runs, taken, session_files),
        unused=tuple(unused),
    )


_RunKey = tuple[str | None, str | None, str]


def _session_order(key: tuple[str | None, str | None]) -> tuple[str, tuple[int, object]]:
    sub, ses = key
    return (sub or "", (0, "") if ses is None else (1, _natural(ses)))


def _natural(text: str) -> tuple[tuple[int, int | str], ...]:
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", text) if p)


def _pair(raw: list[_File], pre: list[_File], rule: PairRule) -> dict[_RunKey, dict[str, _File]]:
    runs: dict[_RunKey, dict[str, _File]] = {}
    if rule == "order":
        groups: dict[tuple[str | None, str | None], dict[str, list[_File]]] = {}
        for role, found in (("raw", raw), ("pre", pre)):
            for f in found:
                groups.setdefault((f.sub, f.ses), {"raw": [], "pre": []})[role].append(f)
        for (sub, ses), g in sorted(groups.items(), key=lambda kv: _session_order(kv[0])):
            if len(g["raw"]) != len(g["pre"]):
                raise InputError(
                    f"sub-{sub}{'' if ses is None else ' ses-' + ses}: {len(g['raw'])} raw and "
                    f"{len(g['pre'])} preprocessed files, which cannot be paired in order",
                    [f"raw: {f.path}" for f in g["raw"]]
                    + [f"preprocessed: {f.path}" for f in g["pre"]],
                )
            for i, (r, p) in enumerate(zip(g["raw"], g["pre"], strict=True)):
                name = Path(r.path).name.split(".", 1)[0]
                shown = _File(name, sub, ses, None, name)
                runs[(sub, ses, f"{i:04d}")] = {"raw": r, "pre": p, "name": shown}
        return runs
    for role, found in (("raw", raw), ("pre", pre)):
        for f in found:
            key = (f.sub, f.ses, f.run or "")
            entry = runs.setdefault(key, {})
            if role in entry:
                raise InputError(
                    f"{OPTIONS[role]}: two files of one run "
                    f"(sub-{f.sub}{'' if f.ses is None else ' ses-' + f.ses} {f.name})",
                    [entry[role].path, f.path],
                )
            entry[role] = f
    if rule == "fields":
        for entry in runs.values():
            if "raw" in entry and "pre" in entry:
                a, b = _bids(entry["raw"].path), _bids(entry["pre"].path)
                if a is not None and b is not None and bids.run_key(a) != bids.run_key(b):
                    raise InputError(
                        "the fields pair two files whose BIDS names name different runs",
                        [entry["raw"].path, entry["pre"].path],
                    )
    return runs


def _assign(
    role: str,
    found: list[_File],
    pattern: Pattern,
    rule: PairRule,
    runs: Mapping[_RunKey, dict[str, _File]],
    taken: dict[_RunKey, dict[str, str]],
    session_files: dict[tuple[str | None, str | None], dict[str, str]],
    sessions: list[tuple[str | None, str | None]],
    session_level: bool,
    unused: list[tuple[str, str]],
) -> None:
    def to_session(f: _File, matches: list[tuple[str | None, str | None]]) -> None:
        if not matches:
            unused.append((role, f.path))
        for key in matches:
            have = session_files.setdefault(key, {})
            if role in have and have[role] != f.path:
                raise InputError(
                    f"{OPTIONS[role]}: two files for one session", [have[role], f.path]
                )
            have[role] = f.path

    def sessions_of(f: _File) -> list[tuple[str | None, str | None]]:
        # a file that names no session is of every session of its subject
        return [s for s in sessions if s[0] == f.sub and (f.ses is None or s[1] == f.ses)]

    def to_run(f: _File, key: _RunKey) -> None:
        if key not in runs:
            unused.append((role, f.path))
            return
        have = taken[key]
        if role in have:
            entry = runs[key]
            shown = (entry.get("name") or entry.get("raw") or entry["pre"]).name
            hint = "" if "run" in pattern.fields else "; put {run} into the pattern"
            raise InputError(
                f"{OPTIONS[role]}: two files for one run ({shown}){hint}", [have[role], f.path]
            )
        have[role] = f.path

    if session_level:
        for f in found:
            to_session(f, sessions_of(f))
        return
    if rule == "fields":
        for f in found:
            if "run" in pattern.fields:
                to_run(f, (f.sub, f.ses, f.run or ""))
            else:
                for s in sessions_of(f):
                    for key in runs:
                        if key[:2] == s:
                            to_run(f, key)
                if not sessions_of(f):
                    unused.append((role, f.path))
        return
    if rule == "bids":
        for f in found:
            key = (f.sub, f.ses, f.run or "")
            if key in runs and f.run:
                to_run(f, key)
            elif not f.run:
                matches = sessions_of(f)
                for s in matches:
                    for k in runs:
                        if k[:2] == s:
                            to_run(f, k)
                if not matches:
                    unused.append((role, f.path))
            else:
                unused.append((role, f.path))
        return
    # order: one file per session, or one per run in path order
    by_session: dict[tuple[str | None, str | None], list[_File]] = {}
    for f in found:
        by_session.setdefault((f.sub, f.ses), []).append(f)
    for s, fs in by_session.items():
        keys = sorted(k for k in runs if k[:2] == s)
        if len(fs) == 1:
            for k in keys:
                to_run(fs[0], k)
        elif len(fs) == len(keys):
            for f, k in zip(fs, keys, strict=True):
                to_run(f, k)
        else:
            raise InputError(
                f"{OPTIONS[role]}: {len(fs)} files for {len(keys)} runs of one session, "
                "which cannot be paired in order",
                [f.path for f in fs],
            )


def _subjects(
    runs: Mapping[_RunKey, dict[str, _File]],
    taken: Mapping[_RunKey, dict[str, str]],
    session_files: Mapping[tuple[str | None, str | None], dict[str, str]],
) -> tuple[SubjectInputs, ...]:
    by_session: dict[tuple[str | None, str | None], list[RunInputs]] = {}
    for key in sorted(runs, key=lambda k: (*_session_order(k[:2]), _natural(k[2]))):
        entry = runs[key]
        first = entry.get("name") or entry.get("raw") or entry["pre"]
        by_session.setdefault(key[:2], []).append(
            RunInputs(
                name=first.name or "",
                raw=entry["raw"].path if "raw" in entry else None,
                pre=entry["pre"].path if "pre" in entry else None,
                **taken[key],
            )
        )
    subjects: dict[str, list[SessionInputs]] = {}
    for (sub, ses), session_runs in by_session.items():
        extra = session_files.get((sub, ses), {})
        subjects.setdefault(sub or "", []).append(
            SessionInputs(
                label=ses,
                freesurfer=extra.get(FREESURFER),
                fsnative_xfm=extra.get("fsnative_xfm"),
                runs=tuple(session_runs),
            )
        )
    return tuple(SubjectInputs(label=sub, sessions=tuple(s)) for sub, s in subjects.items())


# --- what the set still lacks, and the table ------------------------------------------------


def missing_statements(inputs: InputSet) -> list[str]:
    """The statements a run of this set needs and the user has not made, as the options that
    make them; empty when none is missing."""
    runs = [r for s in inputs.subjects for se in s.sessions for r in se.runs]
    out = []
    if inputs.statements.pre_space is None and any(r.pre for r in runs):
        out.append("--pre-space anat|native (where the preprocessed series lie)")
    if inputs.statements.xfm_fixed is None and any(r.xfm for r in runs):
        out.append(
            "--xfm-fixed anat|epi (which image was fixed when the transforms were estimated)"
        )
    sessions = [se for s in inputs.subjects for se in s.sessions]
    if inputs.statements.fs_frame is None and any(se.freesurfer for se in sessions):
        out.append(
            "--fs-frame same, or --fsnative-xfm-files "
            "(how the anatomy relates to FreeSurfer's frame)"
        )
    return out


def table(inputs: InputSet) -> list[str]:
    """The resolved set as lines for a terminal: per run its files, then what no run took."""
    lines = [f"Paired by {_RULE_WORDS[inputs.pair]}."]
    width = max((len(r) for r in RUN_ROLES), default=0)
    for subject in inputs.subjects:
        for session in subject.sessions:
            head = f"sub-{subject.label}"
            if session.label is not None:
                head += f" ses-{session.label}"
            lines.append(head + f": {len(session.runs)} run{'s' if len(session.runs) != 1 else ''}")
            if session.freesurfer:
                lines.append(f"  {'freesurfer':<{width}}  {session.freesurfer}")
            if session.fsnative_xfm:
                lines.append(f"  {'fsnative_xfm':<{width}}  {session.fsnative_xfm}")
            for run in session.runs:
                lines.append(f"  {run.name}")
                for role in RUN_ROLES:
                    value = getattr(run, role)
                    if value is not None:
                        lines.append(f"    {role:<{width}}  {value}")
                    elif role in ("raw", "pre"):
                        why = "no raw series" if role == "raw" else "no preprocessed series"
                        lines.append(f"    {role:<{width}}  none: {why}")
    for role, path in inputs.unused:
        lines.append(f"Taken by no run ({OPTIONS[role]}): {path}")
    for need in missing_statements(inputs):
        lines.append(f"Not stated: {need}")
    return lines


_RULE_WORDS = {
    "fields": "the {sub}, {ses} and {run} fields of the patterns",
    "bids": "BIDS names (subject, session and run key)",
    "order": "path order within each session (asked for; read the pairs below)",
}


# --- inputs.yaml ---------------------------------------------------------------------------


def write_inputs(run_dir: Path | str, inputs: InputSet) -> Path:
    path = Path(run_dir) / INPUTS_FILE
    text = yaml.safe_dump(inputs.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return path


def read_inputs(run_dir: Path | str) -> InputSet | None:
    """The input set of a run directory, or None when it has none. ``ValueError`` for a file
    that is not one of this schema."""
    path = Path(run_dir) / INPUTS_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{INPUTS_FILE} is not an input set of schema {SCHEMA_VERSION}")
    return InputSet.model_validate(data)
