"""Phase 2 for a run: list the subjects of each source and record them in the catalog.

Stub of milestone 2: the names and signatures are final, the bodies are written by work
package 2.E.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from surfcensus.config import InventoryConfig, RunConfig
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport, Provenance
from surfcensus.sources.base import Source, SubjectRecord
from surfcensus.store import Catalog

_PACKAGE = "work package 2.E (inventory/subjects.py)"


def inventory_subject(source: Source, record: SubjectRecord, config: InventoryConfig) -> Provenance:
    """Read the build stamp and the logs of one subject through ``source.read_bytes`` and
    parse them. A missing or unreadable Tier A file gives None fields, not an error."""
    raise NotImplementedError(_PACKAGE)


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
    catalog. ``datasets`` are pseudonyms."""
    raise NotImplementedError(_PACKAGE)
