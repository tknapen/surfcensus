"""Keyed identifiers and the scrubbing of text (decision D10, docs/architecture.md).

Nothing else in the package computes a dataset pseudonym, a ``subject_uid``, a
``render_key`` or a cache name, nor, for the EPI interface in a run directory, a
``participant_uid``, ``session_uid``, ``run_uid`` or ``pack_key``. All are HMAC-SHA256
values: an unkeyed hash of an accession, or of the size and ETag of a public object, could
be inverted by enumeration.

Two secrets exist. The :class:`HashKey` of a run (the per-run salt of a local run, the
dataset key of a census run) is hashed into pseudonyms and subject identifiers. The run
secret, 32 random bytes kept in ``catalog.sqlite``, is hashed into render keys and cache
names, so that neither can be recomputed from a bucket listing.

The key file, pseudonym collisions and the reveal path of a census run follow in milestone 5.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from surfcensus._io import canonical_json

if TYPE_CHECKING:
    from surfcensus.store import Catalog

KeyKind = Literal["local-salt", "dataset-key"]
_SECRET_BYTES = 32


@dataclass(frozen=True, repr=False)
class HashKey:
    """The secret that pseudonyms and subject identifiers are hashed with.

    Its ``repr`` shows the kind and the fingerprint only, so a key cannot reach a log line
    or a traceback by accident.
    """

    secret: bytes
    kind: KeyKind

    def __post_init__(self) -> None:
        if len(self.secret) != _SECRET_BYTES:
            raise ValueError(f"a key is {_SECRET_BYTES} bytes, got {len(self.secret)}")

    @property
    def fingerprint(self) -> str:
        """First 12 hex characters of the SHA-256 of the secret; safe to store and print."""
        return hashlib.sha256(self.secret).hexdigest()[:12]

    def __repr__(self) -> str:
        return f"HashKey(kind={self.kind!r}, fingerprint={self.fingerprint!r})"


def new_local_salt() -> HashKey:
    """A fresh per-run salt for a local run (docs/architecture.md, "Run directory")."""
    return HashKey(secrets.token_bytes(_SECRET_BYTES), "local-salt")


def local_salt_from_hex(text: str) -> HashKey:
    """The salt as ``config.yaml`` stores it: 64 hex characters. Raises ValueError otherwise."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise ValueError("a local salt is 64 hex characters")
    return HashKey(bytes.fromhex(text), "local-salt")


def _hmac_hex(secret: bytes, message: str) -> str:
    return hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()


def dataset_pseudonym(key: HashKey, source: str, identity: str, *, length: int = 8) -> str:
    """``D-`` and the first ``length`` hex characters of HMAC-SHA256(key, "source:identity").

    ``identity`` is the accession of a remote dataset and the stored root of a local one.
    """
    if not 1 <= length <= 64:
        raise ValueError(f"a pseudonym has 1 to 64 hex characters, got {length}")
    return "D-" + _hmac_hex(key.secret, f"{source}:{identity}")[:length]


def subject_uid(key: HashKey, source: str, identity: str, pipeline: str, subject_label: str) -> str:
    """16 hex characters that name a subject in every path, table and URL.

    Keyed like the pseudonym, so a media path does not reveal the dataset to a rater. The
    four parts are joined by a separator that none of them contains in practice (U+001F),
    behind a prefix that keeps this hash apart from the others of this module.
    """
    message = "subject\x1f" + "\x1f".join((source, identity, pipeline, subject_label))
    return _hmac_hex(key.secret, message)[:16]


def render_key(run_secret: bytes, tile_hash: str, digests: Mapping[str, str]) -> str:
    """16 hex characters that say which tiles a rating was made on.

    ``tile_hash`` is ``RenderConfig.tile_hash()`` and ``digests`` maps each role the render
    read to the SHA-256 of that file (``inputs`` of ``render.json``). The key therefore
    changes when a tile could, and not when a file was only touched, moved or re-encoded
    into a video. It is keyed with the run secret because the digests of public objects
    could otherwise be matched against it.
    """
    payload = canonical_json({"tiles": tile_hash, "inputs": dict(digests)})
    return _hmac_hex(run_secret, "render\x1f" + payload)[:16]


def _func_message(prefix: str, *parts: str) -> str:
    return prefix + "\x1f" + "\x1f".join(parts)


def participant_uid(
    key: HashKey, source: str, identity: str, connector: str, participant: str
) -> str:
    """16 hex characters that name the functional stream of one participant (decision D38:
    every session of a subject, one after the other): the unit of the EPI page and of its
    routes (``/epi/<participant_uid>``) and the directory ``epi/<participant_uid>/``.

    Keyed like :func:`subject_uid`. ``identity`` is the identity of the derivative
    (``FuncSourceRecord.deriv_identity``), ``connector`` its pipeline (``fmriprep``),
    ``participant`` the BIDS label (``sub-01``). A participant is keyed the same whether or not
    it has a FreeSurfer subject, so a participant without one has an identifier as well.
    """
    return _hmac_hex(
        key.secret, _func_message("func-participant", source, identity, connector, participant)
    )[:16]


def session_uid(
    key: HashKey,
    source: str,
    identity: str,
    connector: str,
    participant: str,
    session: str | None,
) -> str:
    """16 hex characters that name one functional session: the unit of the one registration
    (decision D36) and of the pack phase, and the directory
    ``epi/<participant_uid>/<session_uid>/``. ``session`` is the ``ses-`` label without
    ``ses-``, None in a dataset without sessions (hashed as the empty string)."""
    return _hmac_hex(
        key.secret,
        _func_message("func-session", source, identity, connector, participant, session or ""),
    )[:16]


def run_uid(
    key: HashKey,
    source: str,
    identity: str,
    connector: str,
    participant: str,
    session: str | None,
    run_key: str,
) -> str:
    """16 hex characters that name one functional run. ``run_key`` is ``func.bids.run_key``:
    the entities of the run without the subject and the echo, the run index as a number, so
    ``run-01`` and ``run-1`` give one identifier and the echoes of a multi-echo acquisition
    are one run, as in step 1."""
    return _hmac_hex(
        key.secret,
        _func_message("func-run", source, identity, connector, participant, session or "", run_key),
    )[:16]


def pack_key(run_secret: bytes, pack_hash: str, digests: Mapping[str, str]) -> str:
    """16 hex characters that say which pack a page showed or a judgement was made on, the
    counterpart of :func:`render_key`.

    ``pack_hash`` is ``EpiConfig.pack_hash()`` and ``digests`` maps ``<base>:<role>`` of every
    input of the pack (``PackInfo.inputs``: the run's files, the anchor's transform, the
    fsnative transform, the FreeSurfer files of the contours) to its SHA-256. Keyed with the
    run secret because the digests are those of public files (decision D20).
    """
    payload = canonical_json({"pack": pack_hash, "inputs": dict(digests)})
    return _hmac_hex(run_secret, "pack\x1f" + payload)[:16]


def cache_name(run_secret: bytes, bucket: str, key: str, etag: str) -> str:
    """32 hex characters under which the cache stores one version of one remote object."""
    return _hmac_hex(run_secret, "cache\x1f" + "\x1f".join((bucket, key, etag)))[:32]


_ROOT = "<root>"
# where an absolute path or a URL starts: a scheme and two slashes, a Windows drive, a UNC
# prefix, a root that scrub() has already replaced, or a slash that does not continue a word
# (so that the relative "surf/lh.pial" and "and/or" are left alone)
_PATH_START = (
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\"
    rf"|{re.escape(_ROOT)}|(?<![A-Za-z0-9_.~\\/-])/)"
)
_PATH = re.compile(
    # in quotes a path runs to the closing quote and may hold spaces
    rf"(?P<quote>['\"]){_PATH_START}(?P<quoted>[^'\"]*)(?P=quote)"
    rf"|{_PATH_START}(?P<bare>[^\s'\"]*)"
)
_SEPARATORS = re.compile(r"[\\/]+")
_ACCESSION = re.compile(r"(?<![A-Za-z0-9])ds\d{6}(?!\d)")
_SUBJECT = re.compile(r"(?<![A-Za-z0-9])sub-[A-Za-z0-9]+")


def _last_component(match: re.Match[str]) -> str:
    quote = match.group("quote") or ""
    rest = match.group("quoted") if quote else match.group("bare")
    parts = [part for part in _SEPARATORS.split(rest) if part]
    if not parts:  # "/" alone, a bare drive or scheme, the root marker itself
        return match.group(0)
    return f"{quote}{parts[-1]}{quote}"


def scrub(
    text: str, *, pseudonyms: Mapping[str, str] | None = None, roots: Collection[str] = ()
) -> str:
    """Text of an exception or a log line with what could name a dataset taken out.

    1. Every string of ``roots`` (dataset roots and identities) becomes ``<root>``. A root
       that is itself an accession (``ds000102``) is left to step 3, which says more and
       reveals as little.
    2. Every absolute path or URL, POSIX or Windows, also inside quotes, is replaced by its
       last component.
    3. Every ``ds`` accession becomes its pseudonym from ``pseudonyms``, or ``D-?`` when it
       has none.
    4. Every ``sub-<label>`` token becomes ``sub-?`` when ``pseudonyms`` is given, that is,
       in a census run. A local run keeps subject labels.

    White space is collapsed to single spaces. Pure; never raises for a string.
    """
    for root in sorted({r for r in roots if r}, key=lambda r: (-len(r), r)):
        if not _ACCESSION.fullmatch(root):
            text = text.replace(root, _ROOT)
    text = _PATH.sub(_last_component, text)
    known = pseudonyms or {}
    text = _ACCESSION.sub(lambda match: known.get(match.group(0), "D-?"), text)
    if pseudonyms is not None:
        text = _SUBJECT.sub("sub-?", text)
    return " ".join(text.split())


def scrub_with(catalog: Catalog, text: str) -> str:
    """:func:`scrub` with what ``catalog`` holds when this is called: the roots and
    identities of its datasets and, in a census run, their pseudonyms. A local run keeps
    subject labels: they are the user's own."""
    pseudonyms, roots = catalog.scrub_terms()
    census = catalog.meta("mode") == "census"
    return scrub(text, pseudonyms=pseudonyms if census else None, roots=roots)
