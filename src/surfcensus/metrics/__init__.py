"""Quantitative metrics: regional thickness, fsaverage maps, clusters, scores. See docs/metrics.md.

Nothing is imported here; use the modules (``metrics.regional``, ``metrics.fsaverage``, ...).
This file holds the one error class they share.
"""

from __future__ import annotations


class MetricError(Exception):
    """A metric cannot be computed for a subject or a hemisphere.

    ``reason`` is one of ``missing file``, ``unreadable``, ``vertex count mismatch``,
    ``degenerate sphere``, ``low coverage``; ``detail`` says what and where, without a path.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[type[MetricError], tuple[str, str]]:
        # pickle would call __init__ with the one argument of Exception.args; workers of a
        # process pool send their exceptions to the parent pickled
        return type(self), (self.reason, self.detail)
