"""The calibration set of a run: ``calibration.yaml`` in the run directory.

Subjects with consensus labels and marks that every rater rates first
(docs/rating_protocol.md), each with a short written justification and, for the help screen
of the rating page, example slices. ``surfcensus calibration freeze`` writes the file from
the final labels and marks of a run (``app.finalize.freeze_calibration``), with empty
justifications and no examples; those are then written into the file by hand. The file is
read like ``protocol.yaml``: when the rater app starts, and the SHA-256 of its bytes is
stored with every calibration submission.

Subjects are named by ``subject_uid``, so the file holds nothing private.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Callable, Collection
from pathlib import Path

import yaml
from pydantic import ConfigDict, Field, ValidationError, model_validator

from surfcensus.config import ConfigError, validation_detail
from surfcensus.protocol import CriterionId, MarkClass, Protocol
from surfcensus.records import SUBJECT_UID_PATTERN
from surfcensus.render.model import PanelName, RenderInfo, Vec3, _Frozen

CALIBRATION_FILE = "calibration.yaml"


class CalibrationMark(_Frozen):
    """A consensus mark: its class and its place in tkr RAS of the subject."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    mark_class: MarkClass = Field(alias="class")
    tkr: Vec3


class CalibrationExample(_Frozen):
    """A slice of a calibration subject that shows a scale level of a criterion, for the
    ``?`` screen of the rating page."""

    criterion: CriterionId
    value: int
    panel: PanelName
    slice: int = Field(ge=0)
    """Position in the panel's slice list of the subject's render."""
    caption: str = ""


class CalibrationSubject(_Frozen):
    subject_uid: str = Field(pattern=f"^{SUBJECT_UID_PATTERN}$")
    values: dict[CriterionId, int] = Field(default_factory=dict)
    """The consensus value per criterion; may be empty for an unratable subject."""
    unratable: str | None = None
    """One of the protocol's unratable reasons, when that is the consensus."""
    marks: tuple[CalibrationMark, ...] = ()
    justification: str = ""
    """The short text shown to a rater after calibration. Written by hand."""
    examples: tuple[CalibrationExample, ...] = ()


class CalibrationSet(_Frozen):
    """Contents of ``calibration.yaml``."""

    version: str = Field(min_length=1)
    subjects: tuple[CalibrationSubject, ...]

    @model_validator(mode="after")
    def _each_subject_once(self) -> CalibrationSet:
        seen: set[str] = set()
        for subject in self.subjects:
            if subject.subject_uid in seen:
                raise ValueError(f"subjects: {subject.subject_uid} is listed more than once")
            seen.add(subject.subject_uid)
        return self

    def subject(self, subject_uid: str) -> CalibrationSubject | None:
        for subject in self.subjects:
            if subject.subject_uid == subject_uid:
                return subject
        return None


def dump_calibration(calibration: CalibrationSet) -> str:
    """The text of ``calibration.yaml``: fields in the order of the models, so equal sets
    give equal bytes and a subject reads from its uid to its examples."""
    data = calibration.model_dump(mode="json", by_alias=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)


def parse_calibration(data: bytes | str) -> CalibrationSet:
    """The calibration set in the text of a ``calibration.yaml``.

    Raises ``ConfigError("invalid", ...)`` that names each entry that cannot be used. What
    needs the protocol and the run is checked by :func:`validate_calibration`.
    """
    try:
        content = yaml.safe_load(data)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError("invalid", f"{CALIBRATION_FILE} is not YAML: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError("invalid", f"{CALIBRATION_FILE} is not a mapping of settings")
    try:
        return CalibrationSet.model_validate(content)
    except ValidationError as exc:
        raise ConfigError("invalid", f"{CALIBRATION_FILE}: {validation_detail(exc)}") from exc


def load_calibration(run_dir: Path | str) -> tuple[CalibrationSet, str] | None:
    """The calibration set of a run and the SHA-256 of the file's bytes; None when the run
    has no ``calibration.yaml``.

    Raises ``ConfigError("invalid", ...)`` for a file that cannot be read or used.
    """
    path = Path(run_dir) / CALIBRATION_FILE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ConfigError(
            "invalid", f"{CALIBRATION_FILE} cannot be read: {exc.strerror or exc}"
        ) from exc
    return parse_calibration(data), hashlib.sha256(data).hexdigest()


def write_calibration(run_dir: Path | str, calibration: CalibrationSet) -> Path:
    """Write ``calibration.yaml`` and return its path.

    A file that is there is never replaced: it may hold justifications and examples written
    by hand, and submissions refer to its bytes. ``ConfigError("invalid", ...)`` then, and
    nothing is written. The file appears whole or not at all.
    """
    final = Path(run_dir) / CALIBRATION_FILE
    exists = ConfigError(
        "invalid",
        f"{CALIBRATION_FILE} exists in {run_dir} and is not overwritten; remove it to freeze "
        "the calibration set again",
    )
    if final.exists():
        raise exists
    text = dump_calibration(calibration)
    tmp = final.with_name(f".{final.name}.{secrets.token_hex(8)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            os.link(tmp, final)  # atomic, and fails when the name was taken meanwhile
        except FileExistsError:
            raise exists from None
        except OSError:  # a file system without hard links: exclusive still, not atomic
            try:
                with final.open("x", encoding="utf-8") as stream:
                    stream.write(text)
            except FileExistsError:
                raise exists from None
            except OSError:
                final.unlink(missing_ok=True)  # half written, and made here
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return final


def validate_calibration(
    calibration: CalibrationSet,
    protocol: Protocol,
    *,
    subjects: Collection[str],
    render: Callable[[str], RenderInfo | None],
) -> None:
    """Check a calibration set against the protocol and the run it is used in.

    ``subjects`` are the ``subject_uid``s of the catalog. ``render(subject_uid)`` gives the
    current render of a subject, or None when it has none; it is called for the subjects
    that have examples only.

    Raises ``ConfigError("invalid", ...)`` for the first entry that cannot be used: a subject
    the catalog does not have; consensus values that are not exactly the protocol's criteria
    with values on its scale (an unratable subject may have none); an unratable reason the
    protocol lacks; an example of a criterion or a value the protocol lacks, or of a slice
    that the subject's render does not have.
    """
    known = set(subjects)
    scale = set(protocol.values())
    criteria = protocol.criterion_ids()

    def refuse(subject: CalibrationSubject, problem: str) -> ConfigError:
        return ConfigError(
            "invalid", f"{CALIBRATION_FILE}: subject {subject.subject_uid}: {problem}"
        )

    for subject in calibration.subjects:
        if subject.subject_uid not in known:
            raise refuse(subject, "is not a subject of this run")
        if subject.unratable is not None and subject.unratable not in protocol.unratable_reasons:
            raise refuse(
                subject, f"{subject.unratable!r} is not an unratable reason of the protocol"
            )
        rated = sorted(subject.values)
        if (subject.unratable is None or rated) and rated != sorted(criteria):
            raise refuse(
                subject,
                f"values are needed for the criteria {', '.join(criteria)}, "
                f"got {', '.join(rated) or 'none'}",
            )
        for criterion, value in subject.values.items():
            if value not in scale:
                raise refuse(subject, f"the value {value} of {criterion} is not on the scale")
        for example in subject.examples:
            if example.criterion not in criteria:
                raise refuse(subject, f"an example of criterion {example.criterion}")
            if example.value not in scale:
                raise refuse(
                    subject, f"an example of the value {example.value}, which is not on the scale"
                )
        if not subject.examples:
            continue
        info = render(subject.subject_uid)
        if info is None:
            raise refuse(subject, "has examples and no render")
        for example in subject.examples:
            n_slices = len(info.panels[example.panel].slices)
            if example.slice >= n_slices:
                raise refuse(
                    subject,
                    f"an example on slice {example.slice} of the {example.panel} panel, "
                    f"which has {n_slices} slices",
                )
