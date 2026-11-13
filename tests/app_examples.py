"""Recorded bodies of the rater app's API, for the tests of the app and for a mock server.

``tests/data/render_small.json`` is the ``render`` object a rating page receives
(``app.schemas.RenderOut``) for ``sub-001`` of the ``small_run`` fixture: panels of 64 px
with 7, 6 and 7 slices, a vertex map with fsaverage6 counterparts. To record it again after
a change of the renderer or of the fixture::

    SURFCENSUS_RECORD=1 uv run pytest tests/test_app_schemas.py -k recorded

``tests/data/api_examples.json`` has one example per shape, written by hand from
docs/rater_app.md. ``load_api_examples`` puts the recorded render and the shipped protocol
where the file names them, so every example validates against its shape as it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from surfcensus.app import schemas
from surfcensus.protocol import parse_protocol, shipped_protocol_bytes

DATA = Path(__file__).parent / "data"
RENDER_SMALL = DATA / "render_small.json"
API_EXAMPLES = DATA / "api_examples.json"
RENDER_PLACEHOLDER = "@render_small.json"
PROTOCOL_PLACEHOLDER = "@protocol.yaml"
RECORDED_LABEL = "sub-001"
"""The subject of ``small_run`` whose render is recorded."""

EXAMPLE_SHAPES: dict[str, type[BaseModel]] = {
    "session": schemas.SessionOut,
    "next_assignment": schemas.NextOut,
    "next_revision": schemas.NextOut,
    "next_rendering": schemas.NextOut,
    "next_done": schemas.NextOut,
    "rating_in": schemas.RatingIn,
    "rating_in_unratable": schemas.RatingIn,
    "rating_out": schemas.RatingOut,
    "events_in": schemas.EventsIn,
    "events_out": schemas.EventsOut,
    "review": schemas.ReviewOut,
    "subject": schemas.SubjectOut,
    "stats": schemas.StatsOut,
    "errors_422": schemas.ErrorsOut,
    "errors_409": schemas.ErrorsOut,
    "health": schemas.HealthOut,
}
"""Key of ``api_examples.json`` -> the shape its value has."""

EXAMPLE_ROUTES: dict[str, str] = {
    "session": "GET /api/session",
    "next_assignment": "GET /api/next",
    "next_revision": "GET /api/back",
    "next_rendering": "GET /api/next",
    "next_done": "GET /api/next",
    "rating_in": "POST /api/rating",
    "rating_in_unratable": "POST /api/rating",
    "rating_out": "POST /api/rating",
    "events_in": "POST /api/events",
    "events_out": "POST /api/events",
    "review": "GET /api/review",
    "subject": "GET /api/review/{subject_uid}",
    "stats": "GET /api/stats",
    "errors_422": "POST /api/rating",
    "errors_409": "POST /api/rating",
    "health": "GET /healthz",
}
"""Key of ``api_examples.json`` -> the route that receives or answers it."""


def load_render_small() -> dict[str, Any]:
    """The recorded ``render`` object, as JSON data."""
    return json.loads(RENDER_SMALL.read_text(encoding="utf-8"))


def _filled(value: Any, render: dict[str, Any], protocol: dict[str, Any]) -> Any:
    if value == RENDER_PLACEHOLDER:
        return render
    if value == PROTOCOL_PLACEHOLDER:
        return protocol
    if isinstance(value, dict):
        return {key: _filled(item, render, protocol) for key, item in value.items()}
    if isinstance(value, list):
        return [_filled(item, render, protocol) for item in value]
    return value


def load_api_examples() -> dict[str, Any]:
    """Key -> example body as JSON data, with the placeholders filled in."""
    raw = json.loads(API_EXAMPLES.read_text(encoding="utf-8"))
    protocol = parse_protocol(shipped_protocol_bytes()).model_dump(mode="json")
    render = load_render_small()
    return {key: _filled(value, render, protocol) for key, value in raw.items() if key != "_about"}
