"""Names of strata and of references, shared by everything that builds or applies a reference.

A stratum is a set of subjects that share a per-vertex reference (docs/metrics.md). Its name
says how it is defined:

* ``dataset:<pseudonym>``: the subjects of one dataset;
* ``field:<T>|fs:<major>``: one field-strength bin and one FreeSurfer major version;
* ``fs:<major>``: one FreeSurfer major version (a shipped reference only, for a subject whose
  field strength is not known);
* ``all``: every subject.

Standard library only, so that any module can import it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

FIELD_BINS: tuple[float, ...] = (1.5, 3.0, 7.0)
"""Field strengths in tesla that subjects are binned to: a scanner sold as 3 T reports
2.89 T, and a stratum per reported value would hold a handful of subjects each."""

ALL_STRATUM = "all"
SET_PREFIX = "set-"
"""A reference built within a run from its own subjects is ``set-<12 hex>``."""
SHIPPED_PREFIX = "shipped-"
RATED_PREFIX = "rated-"


def field_bin(field_strength: float | None) -> str | None:
    """The bin of a field strength as it appears in a stratum name: ``1.5``, ``3`` or ``7``.

    The nearest of :data:`FIELD_BINS`, the lower one when two are equally near. None for an
    unknown field strength, and for one that is not a positive finite number.
    """
    if field_strength is None or not math.isfinite(field_strength) or field_strength <= 0:
        return None
    nearest = min(FIELD_BINS, key=lambda tesla: (abs(tesla - field_strength), tesla))
    return f"{nearest:g}"


def dataset_stratum(pseudonym: str) -> str:
    """``dataset:<pseudonym>``."""
    return f"dataset:{pseudonym}"


def version_stratum(fs_major: int | None) -> str | None:
    """``fs:<major>``; None when the FreeSurfer version is not known."""
    return None if fs_major is None else f"fs:{fs_major}"


def field_version_stratum(field_strength: float | None, fs_major: int | None) -> str | None:
    """``field:<T>|fs:<major>``; None when field strength or FreeSurfer version is not known."""
    tesla, version = field_bin(field_strength), version_stratum(fs_major)
    if tesla is None or version is None:
        return None
    return f"field:{tesla}|{version}"


def members_sha256(map_digests: Mapping[str, str]) -> str:
    """Digest of the members of one stratum: ``subject_uid`` -> SHA-256 of its fsaverage map
    file (``tables/fsavg/<subject_uid>.npz``).

    The SHA-256 over the lines ``subject_uid<TAB>digest``, sorted by ``subject_uid`` and
    joined with a line feed. It changes when a member's map does, although the member's
    ``subject_uid`` stays.
    """
    lines = [f"{uid}\t{digest}" for uid, digest in sorted(map_digests.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def set_reference_id(reference_stage_hash: str, members: Mapping[str, str]) -> str:
    """The id of a reference built within a run: ``set-`` and 12 hex characters.

    ``members`` maps each stratum name to its :func:`members_sha256`. The id is the SHA-256
    of the canonical JSON of ``{"config": reference_stage_hash, "members": members}``, so a
    changed setting, a changed set of members and a changed map of one member each give
    another id, and no two exports carry different numbers under one ``reference_id``.
    """
    payload = json.dumps(
        {"config": reference_stage_hash, "members": dict(members)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return SET_PREFIX + hashlib.sha256(payload.encode()).hexdigest()[:12]
