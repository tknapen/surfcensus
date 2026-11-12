"""Phase 2 for a run: list the subjects of each source and record them in the catalog.

For every dataset of every source the subjects are listed, which opens no file. A subject
whose listing is what the catalog holds is skipped. For the others the build stamp and the
recon-all logs are read through the source and parsed (``inventory/provenance.py``), and the
subject is stored with its files. Subjects that the listing no longer contains are marked
absent; their rows, renders and ratings stay.

The inventory has no rows in the ``work`` ledger. Its skip test is ``subjects.inventory_hash``:
the input signature over all of a subject's files, with the identity of the inventory stage.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from surfcensus import privacy
from surfcensus.config import ConfigError, InventoryConfig, RunConfig, stage_hashes
from surfcensus.inventory.provenance import decode_text, subject_provenance
from surfcensus.log import get_logger
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport, Provenance
from surfcensus.sources.base import (
    DatasetRecord,
    FileRole,
    Source,
    SourceError,
    SubjectRecord,
    input_signature,
)
from surfcensus.store import Catalog

PHASE = "inventory"

PROVENANCE_ROLES: tuple[FileRole, ...] = (
    FileRole.BUILD_STAMP,
    FileRole.RECON_LOG,
    FileRole.RECON_LOG_LH,
    FileRole.RECON_LOG_RH,
)
"""The Tier A files that the inventory reads."""

STATS_ROLES: tuple[FileRole, ...] = (FileRole.LH_STATS, FileRole.RH_STATS, FileRole.ASEG_STATS)
"""The Tier A files that the inventory fetches from a remote source without reading them, so
that the regional metrics need no network (docs/phases.md, phase 2)."""


@dataclass(frozen=True)
class _TierA:
    texts: Mapping[FileRole, str]
    """Decoded content of the roles of ``PROVENANCE_ROLES`` that could be read."""
    partial: bool
    """At least one log is a head."""
    unread: tuple[tuple[FileRole, SourceError], ...]
    """Roles that the subject has and that could not be read, with the reason."""


def _read_tier_a(source: Source, record: SubjectRecord, config: InventoryConfig) -> _TierA:
    texts: dict[FileRole, str] = {}
    unread: list[tuple[FileRole, SourceError]] = []
    partial = False
    for role in PROVENANCE_ROLES:
        ref = record.files.get(role)
        if ref is None:
            continue
        # a file that fits in the head is read whole, and then nothing about it is partial
        head = config.logs == "head" and ref.size > config.head_bytes
        try:
            data = source.read_bytes(
                record, role, byte_range=(0, config.head_bytes - 1) if head else None
            )
        except SourceError as exc:
            unread.append((role, exc))
            continue
        texts[role] = decode_text(data)
        if head and role is not FileRole.BUILD_STAMP:  # of the stamp only the first line is used
            partial = True
    return _TierA(texts, partial, tuple(unread))


def inventory_subject(source: Source, record: SubjectRecord, config: InventoryConfig) -> Provenance:
    """Read the build stamp and the logs of one subject through ``source.read_bytes`` and
    parse them.

    The files are read whole, or, with ``config.logs == "head"``, only their first
    ``config.head_bytes`` bytes; what the provenance then leaves open is None (see
    ``provenance.subject_provenance``). A Tier A file that the subject does not have, or
    that the source cannot read (any ``SourceError``), gives None fields, not an error.
    """
    read = _read_tier_a(source, record, config)
    return subject_provenance(read.texts, record.files, record.subject_label, partial=read.partial)


@dataclass
class _Tally:
    done: int = 0
    skipped: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    notes: dict[str, list[str]] = field(default_factory=dict)
    """Pseudonym -> sentences about that dataset."""

    def note(self, pseudonym: str, sentence: str) -> None:
        self.notes.setdefault(pseudonym, []).append(f"{pseudonym}: {sentence}")


def _pseudonym(catalog: Catalog, key: HashKey, dataset: DatasetRecord) -> str:
    """The dataset's pseudonym: the stored one when the catalog has the dataset, which may be
    longer than the default after a collision at discovery, else a new one.

    Raises ``ConfigError("key", ...)`` when the stored pseudonym is not what ``key`` gives.
    """
    rows = catalog.query(
        "SELECT pseudonym FROM datasets WHERE source = ? AND identity = ? LIMIT 1",
        (dataset.source, dataset.identity),
    )
    if not rows:
        return privacy.dataset_pseudonym(key, dataset.source, dataset.identity)
    stored = rows[0]["pseudonym"]
    computed = privacy.dataset_pseudonym(
        key, dataset.source, dataset.identity, length=len(stored) - len("D-")
    )
    if computed != stored:
        raise ConfigError(
            "key",
            f"a dataset is {stored} in the catalog and {computed} with the key given: the run "
            "was made with another key",
        )
    return stored


def _fetch_stats(source: Source, record: SubjectRecord) -> SourceError | None:
    """Have a remote source fetch the stats files of ``record``; its cache keeps Tier A
    files after the lease. Nothing is fetched from a local source."""
    roles = [role for role in STATS_ROLES if role in record.files]
    if source.kind == "local" or not roles:
        return None
    try:
        with source.lease(record, roles):
            pass
    except SourceError as exc:
        return exc
    return None


def _inventory_dataset(
    catalog: Catalog,
    source: Source,
    dataset: DatasetRecord,
    pseudonym: str,
    config: RunConfig,
    key: HashKey,
    *,
    stage: str,
    force: bool,
    tally: _Tally,
) -> None:
    log = get_logger()
    dataset_id = catalog.upsert_dataset(dataset, pseudonym, None)
    stored = {
        row["subject_label"]: (row["subject_uid"], row["inventory_hash"])
        for row in catalog.query(
            "SELECT subject_label, subject_uid, inventory_hash FROM subjects WHERE dataset_id = ?",
            (dataset_id,),
        )
    }
    labels: list[str] = []
    uids: list[str] = []
    done = skipped = 0
    reread: dict[str, int] = {}  # reason -> subjects with a Tier A file that was not read
    stopped: SourceError | None = None
    try:
        for record in source.subjects(dataset):
            uid = privacy.subject_uid(
                key, dataset.source, dataset.identity, dataset.pipeline, record.subject_label
            )
            labels.append(record.subject_label)
            uids.append(uid)
            extra = [stage]
            if record.problem is not None:
                # the directory could not be listed: a failed unit until a listing succeeds
                tally.failures.append((uid, record.problem.partition(": ")[0]))
                extra.append(f"problem\t{record.problem}")
            signature = input_signature(record.files, None, extra)
            if not force and stored.get(record.subject_label) == (uid, signature):
                skipped += 1
                continue

            read = _read_tier_a(source, record, config.inventory)
            problems = [(role.value, exc) for role, exc in read.unread]
            unfetched = _fetch_stats(source, record)
            if unfetched is not None:
                problems.append(("stats", unfetched))
            if problems:
                for what, exc in problems:
                    log.warning(
                        "%s: %s was not read: %s", uid, what, privacy.scrub_with(catalog, str(exc))
                    )
                reason = problems[0][1].reason
                reread[reason] = reread.get(reason, 0) + 1
                # a hash that no listing gives, so that the next inventory reads the subject
                # again; the listing alone cannot tell that a file has become readable
                signature = input_signature(
                    record.files, None, [*extra, *(f"unread\t{what}" for what, _ in problems)]
                )
            provenance = subject_provenance(
                read.texts, record.files, record.subject_label, partial=read.partial
            )
            catalog.upsert_subject(uid, dataset_id, record, provenance, signature)
            done += 1
    except SourceError as exc:  # from the listing; reading a subject raises none
        stopped = exc

    tally.done += done
    tally.skipped += skipped
    if reread:
        counts = ", ".join(f"{reason}: {n}" for reason, n in sorted(reread.items()))
        tally.note(
            pseudonym,
            f"{sum(reread.values())} subject(s) with a build stamp, log or stats file that "
            f"could not be read ({counts}); what it would have said is left empty, and the "
            "next inventory reads these subjects again",
        )
    if stopped is not None:
        # what was stored stays, and nobody is marked absent on the word of half a listing
        detail = privacy.scrub_with(catalog, str(stopped))
        log.warning("%s: the subjects could not be listed: %s", pseudonym, detail)
        tally.failures.append((f"dataset:{pseudonym}", stopped.reason))
        tally.note(pseudonym, f"the subjects could not be listed ({detail})")
        return

    catalog.upsert_dataset(dataset, pseudonym, len(labels))
    gone = catalog.mark_absent(dataset_id, labels)
    if gone:
        tally.note(
            pseudonym,
            f"{len(gone)} subject(s) are no longer listed and were marked absent; their "
            "renders and ratings are kept",
        )
    field_strength = source.config().field_strength
    if field_strength is not None:
        with catalog.write() as conn:
            conn.executemany(
                "UPDATE subjects SET field_strength = ? "
                "WHERE subject_uid = ? AND field_strength IS NOT ?",
                [(field_strength, uid, field_strength) for uid in uids],
            )
    log.info(
        "%s: %d subjects listed, %d inventoried, %d unchanged, %d marked absent",
        pseudonym,
        len(labels),
        done,
        skipped,
        len(gone),
    )


def run_inventory(
    catalog: Catalog,
    sources: Sequence[Source],
    config: RunConfig,
    key: HashKey,
    *,
    datasets: Collection[str] | None = None,
    force: bool = False,
) -> PhaseReport:
    """Upsert the datasets and subjects of ``sources``; subjects whose listing is unchanged
    are skipped without reading a file. The one phase function that takes the private
    catalog.

    ``key`` is the key of the run (``config.run_key``): the salt of a local run, the dataset
    key of a census run. ``datasets`` are pseudonyms; None means every dataset of every
    source. ``force`` reads every listed subject again.

    Per dataset: the dataset row is written with its pseudonym; each listed subject gets its
    ``subject_uid`` and ``inventory_hash``, and is read, parsed and stored unless that hash
    is the stored one; subjects of the dataset that are not listed get ``present = 0``, and
    listed ones that were absent ``present = 1``; a ``field_strength`` of the source is
    written to the listed subjects. For a remote source the stats files are fetched in the
    same step, for its cache to keep.

    What the report counts:

    * ``done``: subjects read and stored; ``skipped``: subjects left as they were.
    * ``failed``: the listed subjects whose directory could not be listed (``problem`` of
      the record; the unit is the ``subject_uid``, the reason the text before its colon),
      whether stored now or before, and the datasets whose subjects could not be listed or
      that no source has (unit ``dataset:<pseudonym>``, the ``SourceError`` reason or
      ``not found``). Sorted by unit.
    * A Tier A file that could not be read is not a failure: the subject is stored with what
      is known, a note counts such subjects per dataset, and the next inventory reads them
      again.

    Notes are ordered by pseudonym and name no subject. Raises ``ConfigError``: ``key`` when
    ``key`` is not the key the catalog was made with, ``invalid`` when two sources yield the
    same dataset or two subjects get one ``subject_uid``. A ``SourceError`` from listing the
    datasets of a source is raised as it is.
    """
    start = time.perf_counter()
    if catalog.meta("key_fingerprint") != key.fingerprint:
        raise ConfigError(
            "key",
            f"the key given has fingerprint {key.fingerprint}, the catalog was made with "
            f"{catalog.meta('key_fingerprint')}",
        )
    wanted = None if datasets is None else set(datasets)
    accessions: set[str] | None = None
    if wanted is not None:
        # a remote source is asked for these datasets only, by accession; what it yields is
        # then filtered by pseudonym like everything else
        accessions = {
            row["accession"]
            for row in catalog.query("SELECT pseudonym, accession FROM datasets")
            if row["pseudonym"] in wanted and row["accession"] is not None
        }
    stage = stage_hashes(config)[PHASE]
    tally = _Tally()
    seen: dict[tuple[str, str, str], str] = {}
    for source in sources:
        for dataset in source.datasets(accessions):
            pseudonym = _pseudonym(catalog, key, dataset)
            if wanted is not None and pseudonym not in wanted:
                continue
            identity = (dataset.source, dataset.identity, dataset.pipeline)
            if identity in seen:
                raise ConfigError(
                    "invalid", f"two sources of this run hold the same dataset, {pseudonym}"
                )
            seen[identity] = pseudonym
            _inventory_dataset(
                catalog,
                source,
                dataset,
                pseudonym,
                config,
                key,
                stage=stage,
                force=force,
                tally=tally,
            )
    for pseudonym in sorted((wanted or set()) - set(seen.values())):
        tally.failures.append((f"dataset:{pseudonym}", "not found"))
        tally.note(pseudonym, "no source of this run has this dataset")

    failures = tuple(sorted(tally.failures))
    return PhaseReport(
        phase=PHASE,
        done=tally.done,
        skipped=tally.skipped,
        failed=len(failures),
        failures=failures,
        notes=tuple(note for pseudonym in sorted(tally.notes) for note in tally.notes[pseudonym]),
        seconds=time.perf_counter() - start,
    )
