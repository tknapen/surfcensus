"""The rater web app: rating screen, click marks, queue, review. See docs/rater_app.md.

No module of the package is imported here; use the modules. ``app.schemas`` (the shapes of
the API) needs pydantic only. Every other module needs the ``app`` extra (fastapi, uvicorn,
sqlmodel, jinja2). Code outside this package reads ``app.sqlite`` with the standard library,
by the names below, so that ``surfcensus export`` works on a core install.
"""

import json

APP_DB_FILE = "app.sqlite"
"""The rater app's database in the run directory, separate from the catalog."""

APP_TABLES: tuple[str, ...] = (
    "raters",
    "assignments",
    "submissions",
    "ratings",
    "marks",
    "events",
    "final_labels",
    "final_marks",
    "app_meta",
)
"""The tables of ``app.sqlite``; their columns are those of the models in ``app.models``."""

RATINGS_FLAT_VIEW = "ratings_flat"
"""A view over the ratings of current submissions (``app.db``)."""

LABEL_REASONS_KEY = "unratable_reasons"
"""The ``app_meta`` row that says what the ``unratable`` final labels mean: a JSON array of
reasons, and the ``value`` of such a label is a position in it. It is the list of
``protocol.yaml`` as it was when final labels were last written (``app.finalize``), followed
by the reasons that the protocol has lost since and that a label still names. A database
without the row has labels that were written with the protocol the run has now."""


def label_reasons(value: object) -> tuple[str, ...]:
    """The reasons that the row :data:`LABEL_REASONS_KEY` holds; ``ValueError`` for a value
    that is not a JSON array of texts."""
    try:
        reasons = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        reasons = None
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError(f"app_meta.{LABEL_REASONS_KEY} is no list of unratable reasons")
    return tuple(reasons)
