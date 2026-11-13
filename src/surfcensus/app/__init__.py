"""The rater web app: rating screen, click marks, queue, review. See docs/rater_app.md.

Nothing is imported here; use the modules. ``app.schemas`` (the shapes of the API) needs
pydantic only. Every other module needs the ``app`` extra (fastapi, uvicorn, sqlmodel,
jinja2). Code outside this package reads ``app.sqlite`` with the standard library, by the
names below, so that ``surfcensus export`` works on a core install.
"""

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
