"""The rating protocol of a run: ``protocol.yaml`` in the run directory.

The scale, the criteria, the keys of the rating bar and the unratable reasons are data, so
the wording can change without a code change (docs/rating_protocol.md). The shipped file is
``data/protocol.yaml``; :func:`install_protocol` copies it into a run that has none, and the
copy is the user's to edit. No program rewrites it: whether marks are required in a local
run is a setting of the run (:func:`marks_required`).

The file is read when the rater app starts (:func:`load_protocol`), and the SHA-256 of its
bytes is stored with every submission beside the version it names.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from surfcensus._io import atomic_path
from surfcensus.config import AppConfig, ConfigError, validation_detail
from surfcensus.render.model import _Frozen

PROTOCOL_FILE = "protocol.yaml"

CriterionId = Literal["A", "B", "C"]
CRITERION_IDS: tuple[CriterionId, ...] = ("A", "B", "C")
"""A: superior sagittal sinus and falx. B: tentorium and transverse sinus. C: white surface."""

MarkClass = Literal["A", "B", "C", "X"]
MARK_CLASSES: tuple[MarkClass, ...] = ("A", "B", "C", "X")
"""The class of a click mark: the criterion it belongs to, or X for anything else."""
OTHER_MARK_CLASS: MarkClass = "X"

RENDER_REASONS: tuple[str, ...] = ("render failure", "wrong orientation", "missing surface")
"""The reasons of ``render.api.RenderError``. A protocol lists them among its unratable
reasons, so that a subject whose render failed can get a final label (``unratable``)."""

FIXED_KEYS: tuple[str, ...] = (
    " ",
    "ArrowLeft",
    "ArrowRight",
    "[",
    "]",
    "m",
    "z",
    "Delete",
    "h",
    "s",
    "u",
    "o",
    "n",
    "Enter",
    "k",
    "Backspace",
    "?",
    "t",
)
"""The keys of the keyboard map of docs/rater_app.md that no protocol can move, as
``KeyboardEvent.key`` gives them. The keys of a protocol must differ from all of them.
``k`` skips the subject on screen, ``t`` opens the tour of the rating screen."""


class ScaleLevel(_Frozen):
    value: int
    label: str = Field(min_length=1)
    help: str = ""


class Criterion(_Frozen):
    id: CriterionId
    label: str = Field(min_length=1)
    question: str = ""
    keys: tuple[str, ...]
    """One key per scale level, in the order of the scale."""
    mark_key: str
    """Makes the criterion's class the active mark class."""


class Protocol(_Frozen):
    """Contents of ``protocol.yaml``. Sent to the rating page as it is."""

    version: str = Field(min_length=1)
    scale: tuple[ScaleLevel, ...]
    criteria: tuple[Criterion, ...]
    other_mark_key: str
    """Makes X the active mark class."""
    require_marks_for_positive: bool
    unratable_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Protocol:
        values = [level.value for level in self.scale]
        if len(values) < 2 or values != list(range(len(values))):
            raise ValueError(
                f"scale: the values must be 0 to n - 1 in order, for at least two levels, "
                f"got {values}"
            )
        if not self.criteria:
            raise ValueError("criteria: at least one criterion is needed")
        ids = [criterion.id for criterion in self.criteria]
        for criterion_id in ids:
            if ids.count(criterion_id) > 1:
                raise ValueError(f"criteria: the id {criterion_id} is used more than once")
        for criterion in self.criteria:
            if len(criterion.keys) != len(self.scale):
                raise ValueError(
                    f"criteria.{criterion.id}.keys: {len(criterion.keys)} keys for "
                    f"{len(self.scale)} scale levels"
                )
        self._check_keys()
        if len(set(self.unratable_reasons)) != len(self.unratable_reasons):
            raise ValueError("unratable_reasons: a reason is listed more than once")
        for reason in self.unratable_reasons:
            if not reason.strip():
                raise ValueError("unratable_reasons: an empty reason")
        for reason in RENDER_REASONS:
            if reason not in self.unratable_reasons:
                raise ValueError(
                    f"unratable_reasons: {reason!r} is missing; the renderer gives it to a "
                    "subject it cannot render"
                )
        return self

    def _check_keys(self) -> None:
        fixed = {key.lower() for key in FIXED_KEYS}
        seen: dict[str, str] = {}
        entries = [
            (f"criteria.{criterion.id}.keys", key)
            for criterion in self.criteria
            for key in criterion.keys
        ]
        entries += [
            (f"criteria.{criterion.id}.mark_key", criterion.mark_key) for criterion in self.criteria
        ]
        entries.append(("other_mark_key", self.other_mark_key))
        for where, key in entries:
            if len(key) != 1 or key.isspace() or not key.isprintable():
                raise ValueError(f"{where}: a key is one visible character, got {key!r}")
            folded = key.lower()
            if folded in fixed:
                raise ValueError(f"{where}: the key {key!r} has a fixed meaning in the rater app")
            if folded in seen:
                raise ValueError(f"{where}: the key {key!r} is already used by {seen[folded]}")
            seen[folded] = where

    def values(self) -> tuple[int, ...]:
        """The values of the scale, ascending from 0."""
        return tuple(level.value for level in self.scale)

    def positive_values(self) -> frozenset[int]:
        """Every scale value above the lowest: the values that say there is an error, and
        that need a mark where marks are required."""
        return frozenset(self.values()[1:])

    def criterion_ids(self) -> tuple[CriterionId, ...]:
        """In the order of the file."""
        return tuple(criterion.id for criterion in self.criteria)

    def label(self, value: int) -> str:
        """The label of a scale value (``major``); KeyError for a value off the scale."""
        for level in self.scale:
            if level.value == value:
                return level.label
        raise KeyError(f"no scale level {value}")

    def unratable_index(self, reason: str) -> int:
        """The position of an unratable reason in the protocol, which is the ``value`` of a
        final label of criterion ``unratable``; KeyError for a reason the protocol lacks."""
        if reason not in self.unratable_reasons:
            raise KeyError(f"no unratable reason {reason!r}")
        return self.unratable_reasons.index(reason)


def render_reason(reason: str | None) -> str:
    """The unratable reason for a render that failed with ``reason``: the reason itself when
    it is one of :data:`RENDER_REASONS`, else ``render failure`` (a render can also fail with
    ``source changed``, which is no property of the surfaces)."""
    return reason if reason in RENDER_REASONS else "render failure"


def parse_protocol(data: bytes | str) -> Protocol:
    """The protocol in the text of a ``protocol.yaml``.

    Raises ``ConfigError("invalid", ...)`` that names each entry that cannot be used.
    """
    try:
        content = yaml.safe_load(data)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ConfigError("invalid", f"{PROTOCOL_FILE} is not YAML: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError("invalid", f"{PROTOCOL_FILE} is not a mapping of settings")
    try:
        return Protocol.model_validate(content)
    except ValidationError as exc:
        raise ConfigError("invalid", f"{PROTOCOL_FILE}: {validation_detail(exc)}") from exc


def shipped_protocol_bytes() -> bytes:
    """The bytes of ``data/protocol.yaml``, the protocol a new run starts with."""
    return (resources.files("surfcensus") / "data" / PROTOCOL_FILE).read_bytes()


def install_protocol(run_dir: Path | str) -> Path:
    """Copy the shipped protocol into a run directory that has none; return its path. A
    ``protocol.yaml`` that is there is left as it is, whatever it holds."""
    path = Path(run_dir) / PROTOCOL_FILE
    if not path.exists():
        with atomic_path(path) as tmp:
            tmp.write_bytes(shipped_protocol_bytes())
    return path


def load_protocol(run_dir: Path | str) -> tuple[Protocol, str]:
    """The protocol of a run and the SHA-256 of the file's bytes.

    Raises ``ConfigError``: ``missing`` when the run has no ``protocol.yaml``
    (:func:`install_protocol` gives it one), ``invalid`` for a file that cannot be read or
    used.
    """
    path = Path(run_dir) / PROTOCOL_FILE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ConfigError("missing", f"{run_dir} has no {PROTOCOL_FILE}") from None
    except OSError as exc:
        raise ConfigError(
            "invalid", f"{PROTOCOL_FILE} cannot be read: {exc.strerror or exc}"
        ) from exc
    return parse_protocol(data), hashlib.sha256(data).hexdigest()


def marks_required(protocol: Protocol, app: AppConfig) -> bool:
    """Whether a positive value on a criterion needs a mark of its class: the setting of the
    run (``app.require_marks``) when it has one, else what the protocol says. A local run is
    created with the setting false, a census run without it."""
    return protocol.require_marks_for_positive if app.require_marks is None else app.require_marks
