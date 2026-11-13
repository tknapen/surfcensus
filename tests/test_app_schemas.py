"""The shapes of the rater app's API (``surfcensus/app/schemas.py``): the examples of
docs/rater_app.md validate, the limits hold, and nothing a rating page receives has a place
for what a rater must not see."""

from __future__ import annotations

import json
import math
import os
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from app_examples import (  # pyrefly: ignore[missing-import]
    API_EXAMPLES,
    EXAMPLE_ROUTES,
    EXAMPLE_SHAPES,
    RECORDED_LABEL,
    RENDER_PLACEHOLDER,
    RENDER_SMALL,
    load_api_examples,
    load_render_small,
)
from pydantic import BaseModel, ValidationError

from surfcensus import schemas as table_schemas
from surfcensus.app import schemas
from surfcensus.app.schemas import (
    MAX_EVENT_PAYLOAD_BYTES,
    MAX_EVENTS_PER_CALL,
    MAX_MARKS_PER_RATING,
    MAX_NOTE_CHARS,
    MAX_REVIEW_ROWS,
    MEDIA_FILE,
    RENDER_PRIVATE_FIELDS,
    ROUTES,
    EventIn,
    EventsIn,
    MarkIn,
    MarkVertices,
    NextOut,
    PreviousMark,
    RatingIn,
    RatingOut,
    RenderOut,
    ResolvedMark,
    ReviewQuery,
    Route,
    SavedMark,
    SavedOut,
    SessionOut,
    SkipIn,
    StatsOut,
    media_base,
    render_out,
    tile_name,
)
from surfcensus.protocol import parse_protocol, shipped_protocol_bytes
from surfcensus.render.model import RenderInfo, TileFiles, read_render
from surfcensus.store import PublicCatalog, media_dir

MARK = {"client_id": "m1", "class": "A", "panel": "sagittal", "slice": 3, "px": 30.5, "py": 28.25}
PROGRESS = {
    "done_today": 0,
    "done_total": 0,
    "remaining": 0,
    "skipped": 0,
    "rendered": 0,
    "n_subjects": 0,
}


def rating(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "assignment_id": 12,
        "protocol_version": "0.1-draft",
        "values": {"A": 0, "B": 0, "C": 0},
        "ms_on_subject": 100,
    }
    return body | changes


# --- the examples ---------------------------------------------------------------------------


def test_every_shape_of_the_contract_exists() -> None:
    for name in (
        "MarkIn",
        "ResolvedMark",
        "SessionOut",
        "AssignmentOut",
        "NextOut",
        "RatingIn",
        "RatingOut",
        "EventsIn",
        "ReviewOut",
        "SubjectOut",
        "StatsOut",
        "ErrorsOut",
    ):
        shape = getattr(schemas, name)
        assert issubclass(shape, BaseModel)
        assert shape.model_config.get("frozen") and shape.model_config.get("extra") == "forbid"


def test_the_examples_cover_every_route_shape() -> None:
    assert set(EXAMPLE_SHAPES) == set(load_api_examples()) == set(EXAMPLE_ROUTES)
    assert {shape.__name__ for shape in EXAMPLE_SHAPES.values()} == {
        "SessionOut",
        "NextOut",
        "RatingIn",
        "RatingOut",
        "SkipIn",
        "EventsIn",
        "EventsOut",
        "ReviewQuery",
        "ReviewOut",
        "SubjectOut",
        "StatsOut",
        "ErrorsOut",
        "HealthOut",
    }


def test_the_routes_are_those_of_the_docs_with_their_shapes() -> None:
    # docs/dev/contract.md, section 7.5: the table of routes
    assert {(route.method, route.path): route for route in ROUTES} == {
        ("GET", "/"): Route("GET", "/"),
        ("GET", "/review"): Route("GET", "/review", unblinded=True),
        ("GET", "/static/{path}"): Route("GET", "/static/{path}"),
        ("GET", "/media/{subject_uid}/{path}"): Route("GET", "/media/{subject_uid}/{path}"),
        ("GET", "/healthz"): Route("GET", "/healthz", schemas.HealthOut),
        ("GET", "/api/session"): Route("GET", "/api/session", SessionOut),
        ("GET", "/api/next"): Route("GET", "/api/next", NextOut),
        ("GET", "/api/back"): Route("GET", "/api/back", NextOut),
        ("POST", "/api/rating"): Route("POST", "/api/rating", RatingOut, RatingIn),
        # a skip is answered like a request for the next subject
        ("POST", "/api/skip"): Route("POST", "/api/skip", NextOut, SkipIn),
        ("POST", "/api/events"): Route("POST", "/api/events", schemas.EventsOut, EventsIn),
        ("GET", "/api/review"): Route(
            "GET", "/api/review", schemas.ReviewOut, ReviewQuery, unblinded=True
        ),
        ("GET", "/api/review/{subject_uid}"): Route(
            "GET", "/api/review/{subject_uid}", schemas.SubjectOut, unblinded=True
        ),
        ("GET", "/api/stats"): Route("GET", "/api/stats", StatsOut, unblinded=True),
        ("GET", "/api/export"): Route("GET", "/api/export", unblinded=True),
    }
    assert len(ROUTES) == 15  # no route twice
    # what is unblinded is the review page and everything under its API, and nothing a
    # rating page calls
    unblinded = {route.path for route in ROUTES if route.unblinded}
    assert unblinded == {
        "/review",
        "/api/review",
        "/api/review/{subject_uid}",
        "/api/stats",
        "/api/export",
    }
    for route in ROUTES:
        if route.request is not None and route.method == "POST":
            assert route.response is not None


def test_every_example_belongs_to_a_route_and_has_one_of_its_shapes() -> None:
    routes = {f"{route.method} {route.path}": route for route in ROUTES}
    for key, where in EXAMPLE_ROUTES.items():
        route = routes[where]
        shape = EXAMPLE_SHAPES[key]
        assert shape in (route.response, route.request, schemas.ErrorsOut), key
    # and every route that answers or takes JSON has an example of each of its shapes
    for where, route in routes.items():
        for shape in (route.response, route.request):
            if shape is not None:
                assert any(
                    EXAMPLE_ROUTES[key] == where and EXAMPLE_SHAPES[key] is shape
                    for key in EXAMPLE_SHAPES
                ), f"{where}: {shape.__name__}"


def test_the_query_of_the_review_table_has_its_defaults_and_limits() -> None:
    plain = ReviewQuery()
    assert (plain.sort, plain.order, plain.filter, plain.offset, plain.limit) == (
        None,
        "asc",
        "",
        0,
        200,
    )
    # as a server reads it from a URL: every value is text
    asked = ReviewQuery.model_validate(
        {"sort": "composite", "order": "desc", "offset": "20", "limit": "10"}
    )
    assert (asked.sort, asked.order, asked.offset, asked.limit) == ("composite", "desc", 20, 10)
    assert ReviewQuery(limit=MAX_REVIEW_ROWS).limit == MAX_REVIEW_ROWS == 1000
    for field, value in (
        ("order", "up"),
        ("offset", -1),
        ("limit", 0),
        ("limit", MAX_REVIEW_ROWS + 1),
        ("page", 2),
    ):
        with pytest.raises(ValidationError, match=field):
            ReviewQuery.model_validate({field: value})


@pytest.mark.parametrize("key", sorted(EXAMPLE_SHAPES))
def test_an_example_validates_and_comes_back_as_it_was(key: str) -> None:
    example = load_api_examples()[key]
    shape = EXAMPLE_SHAPES[key]
    model = shape.model_validate(example)
    dumped = json.loads(model.model_dump_json(by_alias=True))
    # what the example says is in the dump unchanged; the dump may add defaults
    assert _contains(dumped, example), key
    assert shape.model_validate(dumped) == model


def _contains(whole: Any, part: Any) -> bool:
    if isinstance(part, dict):
        return isinstance(whole, dict) and all(
            key in whole and _contains(whole[key], value) for key, value in part.items()
        )
    if isinstance(part, list):
        return (
            isinstance(whole, list)
            and len(whole) == len(part)
            and all(_contains(a, b) for a, b in zip(whole, part, strict=True))
        )
    if isinstance(part, float) or isinstance(whole, float):
        return whole == pytest.approx(part)
    return whole == part


def test_the_examples_file_names_the_render_and_the_protocol_by_placeholder() -> None:
    raw = json.loads(API_EXAMPLES.read_text())
    assert raw["next_assignment"]["assignment"]["render"] == RENDER_PLACEHOLDER
    examples = load_api_examples()
    assert examples["next_assignment"]["assignment"]["render"] == load_render_small()
    shipped = parse_protocol(shipped_protocol_bytes())
    assert SessionOut.model_validate(examples["session"]).protocol == shipped


def test_the_examples_fit_the_recorded_render() -> None:
    # the marks and slices of the examples lie on tiles of the render they come with
    render = RenderOut.model_validate(load_render_small())
    examples = load_api_examples()
    marks = [
        *examples["rating_in"]["marks"],
        *examples["next_revision"]["assignment"]["previous"]["marks"],
        *examples["subject"]["marks"],
    ]
    assert len(marks) == 3
    for mark in marks:
        panel = render.panels[mark["panel"]]
        assert 0 <= mark["slice"] < len(panel.slices)
        assert 0 <= mark["px"] <= panel.width and 0 <= mark["py"] <= panel.height
        if mark.get("mode"):
            frame = render.modes[mark["mode"]].frames[mark["frame"]]
            assert frame[list(render.panels).index(mark["panel"])] == mark["slice"]
    prefetch = examples["next_assignment"]["prefetch"][0]
    assert prefetch["n_slices"] == {name: len(p.slices) for name, p in render.panels.items()}


def test_the_bodies_written_out_in_the_contract_validate() -> None:
    # section 7.5 of docs/dev/contract.md, as far as it writes a body out in full
    posted = {
        "assignment_id": 12,
        "protocol_version": "0.1-draft",
        "values": {"A": 2, "B": 0, "C": 0},
        "unratable": None,
        "other": False,
        "note": "",
        "marks": [
            {
                "client_id": "m1",
                "class": "A",
                "panel": "sagittal",
                "slice": 31,
                "px": 101.5,
                "py": 88.25,
                "frame": 57,
                "mode": "sync",
                "ts": "2026-11-12T14:03:20.004Z",
            }
        ],
        "client_ts": "2026-11-12T14:03:22.120Z",
        "ms_on_subject": 14200,
    }
    assert RatingIn.model_validate(posted).marks[0].mark_class == "A"
    saved = {
        "assignment_id": 12,
        "submission_id": 30,
        "revision": 1,
        "marks": [
            {
                "client_id": "m1",
                "mark_id": 8,
                "hemi": "lh",
                "pial_vertex": 101234,
                "pial_dist_mm": 0.8,
                "white_vertex": 101234,
                "white_dist_mm": 2.9,
                "fsavg6_vertex": 12950,
                "vertex_surface": "pial",
                "box_dist_mm": 0.0,
            }
        ],
    }
    assert SavedOut.model_validate(saved).marks[0].fsavg6_vertex == 12950
    events = {
        "assignment_id": 12,
        "events": [{"ts": "2026-11-12T14:03:12Z", "type": "pause", "payload": {"frame": 57}}],
    }
    assert EventsIn.model_validate(events).events[0].payload == {"frame": 57}
    stats = {
        "raters": [
            {
                "rater": "local",
                "n_completed": 40,
                "median_s": 14.2,
                "by_day": [{"day": "2026-11-12", "n": 40}],
            }
        ],
        "agreement": None,
        "drift": None,
    }
    assert StatsOut.model_validate(stats).raters[0].by_day[0].n == 40
    for state in (
        {"status": "rendering", "retry_after_s": 2, "progress": PROGRESS},
        {"status": "done", "progress": PROGRESS},
    ):
        assert NextOut.model_validate(state).assignment is None


def test_a_stray_key_is_refused_and_a_shape_cannot_be_changed() -> None:
    examples = load_api_examples()
    for key, shape in EXAMPLE_SHAPES.items():
        with pytest.raises(ValidationError, match="stray"):
            shape.model_validate(examples[key] | {"stray": 1})
    model = RatingIn.model_validate(rating())
    with pytest.raises(ValidationError, match="frozen"):
        model.note = "changed"  # pyrefly: ignore[read-only]


# --- the render a page receives -------------------------------------------------------------


def _small_run_render(small_run: Path) -> RenderInfo:
    with PublicCatalog.open(small_run) as public:
        subjects = public.subjects()
    uid = dict(zip(subjects["subject_label"], subjects["subject_uid"], strict=True))[RECORDED_LABEL]
    info = read_render(media_dir(small_run, uid))
    assert info is not None
    return info


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def test_a_page_receives_the_render_without_the_digests_of_its_sources(small_run: Path) -> None:
    assert RENDER_PRIVATE_FIELDS == {"inputs", "template"}
    assert set(RenderOut.model_fields) == set(RenderInfo.model_fields) - RENDER_PRIVATE_FIELDS
    info = _small_run_render(small_run)
    assert info.inputs and info.template is not None  # the run has both to leave out
    out = render_out(info)
    dumped = out.model_dump(mode="json")
    whole = info.model_dump(mode="json")
    assert dumped == {key: whole[key] for key in whole if key not in RENDER_PRIVATE_FIELDS}
    text = out.model_dump_json()
    assert not {"inputs", "sha256", "size", "template_id", "sphere_sha256"} & _keys(dumped)
    for file in info.inputs.values():
        assert file.sha256 not in text
    for digest in info.template.sphere_sha256.values():
        assert digest not in text
    # what a page needs to know about counterparts stays
    assert out.vertex_map is not None and out.vertex_map.fsaverage == info.template.mesh


def _assert_same(recorded: Any, fresh: Any, where: str = "render") -> None:
    if isinstance(recorded, dict):
        assert isinstance(fresh, dict) and set(recorded) == set(fresh), where
        for key in recorded:
            _assert_same(recorded[key], fresh[key], f"{where}.{key}")
    elif isinstance(recorded, list):
        assert isinstance(fresh, list) and len(recorded) == len(fresh), where
        for position, (a, b) in enumerate(zip(recorded, fresh, strict=True)):
            _assert_same(a, b, f"{where}[{position}]")
    elif isinstance(recorded, float) and not isinstance(recorded, bool):
        assert isinstance(fresh, int | float), where
        assert math.isclose(recorded, fresh, rel_tol=1e-9, abs_tol=1e-9), where
    else:
        assert recorded == fresh, where


def test_the_recorded_render_is_what_the_small_run_renders(small_run: Path) -> None:
    fresh = render_out(_small_run_render(small_run)).model_dump(mode="json")
    if os.environ.get("SURFCENSUS_RECORD"):
        RENDER_SMALL.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    recorded = load_render_small()
    render = RenderOut.model_validate(recorded)
    assert {name: len(panel.slices) for name, panel in render.panels.items()} == {
        "sagittal": 7,
        "coronal": 6,
        "axial": 7,
    }
    assert {panel.width for panel in render.panels.values()} == {64}
    assert render.vertex_map is not None and render.vertex_map.fsaverage == "fsaverage6"
    assert not RENDER_PRIVATE_FIELDS & set(recorded)
    # the libraries and the package version are those of the machine that recorded it
    for volatile in ("versions", "surfcensus_version"):
        recorded.pop(volatile)
        fresh.pop(volatile)
    _assert_same(recorded, fresh)


# --- marks ----------------------------------------------------------------------------------


def test_a_mark_is_posted_under_class_and_read_under_mark_class() -> None:
    mark = MarkIn.model_validate(MARK)
    assert mark.mark_class == "A" and mark.frame is None and mark.mode is None
    assert MarkIn(**{**MARK, "class": "X"}).mark_class == "X"
    by_name = {key: value for key, value in MARK.items() if key != "class"}
    assert MarkIn.model_validate(by_name | {"mark_class": "A"}) == mark
    assert json.loads(mark.model_dump_json(by_alias=True))["class"] == "A"
    with pytest.raises(ValidationError, match="class"):
        MarkIn.model_validate(MARK | {"class": "D"})
    with pytest.raises(ValidationError, match="panel"):
        MarkIn.model_validate(MARK | {"panel": "oblique"})
    with pytest.raises(ValidationError, match="mode"):
        MarkIn.model_validate(MARK | {"mode": "loop"})


@pytest.mark.parametrize("field", ["px", "py"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_mark_needs_finite_pixel_coordinates(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        MarkIn.model_validate(MARK | {field: value})


def test_a_mark_needs_a_client_id() -> None:
    with pytest.raises(ValidationError, match="client_id"):
        MarkIn.model_validate(MARK | {"client_id": ""})
    with pytest.raises(ValidationError, match="client_id"):
        MarkIn.model_validate(MARK | {"client_id": "m" * 65})


def test_the_mark_shapes_share_their_vertex_fields() -> None:
    vertices = (
        "hemi",
        "pial_vertex",
        "pial_dist_mm",
        "white_vertex",
        "white_dist_mm",
        "fsavg6_vertex",
        "vertex_surface",
        "box_dist_mm",
    )
    assert tuple(MarkVertices.model_fields) == vertices
    assert set(ResolvedMark.model_fields) == {*vertices, "vox", "tkr", "ras"}
    assert set(SavedMark.model_fields) == {*vertices, "client_id", "mark_id"}
    assert set(PreviousMark.model_fields) == {*vertices, *MarkIn.model_fields, "mark_id"}
    # a mark without a vertex in range is a point: only the distance to the box is required
    bare = ResolvedMark(vox=(1, 2, 3), tkr=(1, 2, 3), ras=(1, 2, 3), box_dist_mm=4.5)
    assert bare.hemi is None and bare.pial_vertex is None and bare.fsavg6_vertex is None
    with pytest.raises(ValidationError, match="box_dist_mm"):
        ResolvedMark.model_validate({"vox": (1, 2, 3), "tkr": (1, 2, 3), "ras": (1, 2, 3)})
    previous = PreviousMark.model_validate(MARK | {"mark_id": 8, "box_dist_mm": 0.0})
    assert previous.mark_class == "A" and previous.mark_id == 8


# --- the queue ------------------------------------------------------------------------------


def test_the_status_of_next_says_what_comes_with_it() -> None:
    assignment = load_api_examples()["next_assignment"]["assignment"]
    with pytest.raises(ValidationError, match="only with it"):
        NextOut.model_validate({"status": "done", "assignment": assignment, "progress": PROGRESS})
    with pytest.raises(ValidationError, match="only with it"):
        NextOut.model_validate({"status": "assignment", "progress": PROGRESS})
    with pytest.raises(ValidationError, match="retry_after_s"):
        NextOut.model_validate({"status": "rendering", "progress": PROGRESS})
    with pytest.raises(ValidationError, match="retry_after_s"):
        NextOut.model_validate({"status": "done", "retry_after_s": 2, "progress": PROGRESS})


def test_a_page_is_told_calibration_or_main_and_never_repeat() -> None:
    example = load_api_examples()["next_assignment"]
    for kind in ("calibration", "main"):
        example["assignment"]["kind"] = kind
        assert NextOut.model_validate(example).assignment.kind == kind
    for kind in ("repeat", "adjudication"):
        assert kind in schemas.ASSIGNMENT_KINDS
        example["assignment"]["kind"] = kind
        with pytest.raises(ValidationError, match="kind"):
            NextOut.model_validate(example)


# --- a rating -------------------------------------------------------------------------------


def test_a_rating_has_its_limits() -> None:
    assert RatingIn.model_validate(rating()).marks == ()
    unratable = RatingIn.model_validate(rating(values={}, unratable="render failure"))
    assert unratable.values == {} and unratable.unratable == "render failure"
    with pytest.raises(ValidationError, match="ms_on_subject"):
        RatingIn.model_validate(rating(ms_on_subject=-1))
    with pytest.raises(ValidationError, match="same client_id"):
        RatingIn.model_validate(rating(marks=[MARK, MARK]))
    many = [MARK | {"client_id": f"m{n}"} for n in range(MAX_MARKS_PER_RATING + 1)]
    assert len(RatingIn.model_validate(rating(marks=many[:-1])).marks) == MAX_MARKS_PER_RATING
    with pytest.raises(ValidationError, match="marks"):
        RatingIn.model_validate(rating(marks=many))
    assert RatingIn.model_validate(rating(note="n" * MAX_NOTE_CHARS)).note
    with pytest.raises(ValidationError, match="note"):
        RatingIn.model_validate(rating(note="n" * (MAX_NOTE_CHARS + 1)))
    for missing in ("assignment_id", "protocol_version", "ms_on_subject"):
        body = rating()
        del body[missing]
        with pytest.raises(ValidationError, match=missing):
            RatingIn.model_validate(body)


def test_the_values_of_a_rating_are_left_to_the_rules_of_the_server() -> None:
    # a criterion the protocol lacks is the server's to refuse, with the field it names
    assert RatingIn.model_validate(rating(values={"D": 7})).values == {"D": 7}


# --- events ---------------------------------------------------------------------------------


def test_a_batch_of_events_has_its_limits() -> None:
    assert (MAX_EVENTS_PER_CALL, MAX_EVENT_PAYLOAD_BYTES) == (500, 16_000)
    event = {"ts": "2026-11-12T14:03:12Z", "type": "pause", "payload": {"frame": 5}}
    full = EventsIn.model_validate({"assignment_id": 1, "events": [event] * MAX_EVENTS_PER_CALL})
    assert len(full.events) == MAX_EVENTS_PER_CALL
    with pytest.raises(ValidationError, match="events"):
        EventsIn.model_validate({"assignment_id": 1, "events": [event] * (MAX_EVENTS_PER_CALL + 1)})
    # {"k":"..."} is 8 bytes around the text
    fits = {"k": "x" * (MAX_EVENT_PAYLOAD_BYTES - 8)}
    assert EventIn.model_validate(event | {"payload": fits}).payload == fits
    with pytest.raises(ValidationError, match="at most 16000"):
        EventIn.model_validate(event | {"payload": {"k": "x" * (MAX_EVENT_PAYLOAD_BYTES - 7)}})
    # a type the app does not know is a valid event
    assert EventIn.model_validate(event | {"type": "something_new"}).type == "something_new"
    assert EventIn.model_validate({"ts": "t", "type": "load"}).payload == {}
    with pytest.raises(ValidationError, match="type"):
        EventIn.model_validate(event | {"type": ""})
    assert "frames" in schemas.EVENT_TYPES and len(set(schemas.EVENT_TYPES)) == 16
    assert {"tour_open", "tour_close"} <= set(schemas.EVENT_TYPES)


def test_a_page_cannot_send_the_event_that_the_server_writes_for_a_skip() -> None:
    # how often a rater skips is read from these events: only the server writes them
    assert schemas.SKIP_EVENT == "skip" and schemas.SERVER_EVENT_TYPES == ("skip",)
    assert not set(schemas.SERVER_EVENT_TYPES) & set(schemas.EVENT_TYPES)
    event = {"ts": "2026-11-12T14:03:12Z", "type": "skip", "payload": {"n": 9}}
    with pytest.raises(ValidationError, match="stored by the server"):
        EventIn.model_validate(event)
    with pytest.raises(ValidationError, match="stored by the server"):
        EventsIn.model_validate({"assignment_id": 1, "events": [event]})
    assert EventIn.model_validate(event | {"type": "skipped"}).type == "skipped"


# --- a skip ---------------------------------------------------------------------------------


def test_a_skip_names_its_assignment_and_its_time_and_nothing_of_a_rating() -> None:
    skip = SkipIn.model_validate(load_api_examples()["skip_in"])
    assert (skip.assignment_id, skip.client_ts, skip.ms_on_subject) == (
        12,
        "2026-11-12T14:03:22.120Z",
        9400,
    )
    # what was entered stays in the page: the shape has no place for it
    assert set(SkipIn.model_fields) == {"assignment_id", "client_ts", "ms_on_subject"}
    body = {"assignment_id": 12, "client_ts": "2026-11-12T14:03:22.120Z", "ms_on_subject": 0}
    assert SkipIn.model_validate(body).ms_on_subject == 0
    for stray in ("values", "marks", "note", "unratable"):
        with pytest.raises(ValidationError, match=stray):
            SkipIn.model_validate(body | {stray: {}})
    # the time names the skip, so that a request sent twice is one skip: it is needed
    for change in ({"client_ts": ""}, {"client_ts": None}, {"client_ts": "x" * 65}):
        with pytest.raises(ValidationError, match="client_ts"):
            SkipIn.model_validate(body | change)
    with pytest.raises(ValidationError, match="client_ts"):
        SkipIn.model_validate({key: value for key, value in body.items() if key != "client_ts"})
    with pytest.raises(ValidationError, match="ms_on_subject"):
        SkipIn.model_validate(body | {"ms_on_subject": -1})


def test_the_progress_line_counts_the_skipped_subjects_that_wait() -> None:
    progress = schemas.ProgressOut.model_validate(PROGRESS | {"remaining": 5, "skipped": 2})
    assert (progress.remaining, progress.skipped) == (5, 2)
    without = {key: value for key, value in PROGRESS.items() if key != "skipped"}
    with pytest.raises(ValidationError, match="skipped"):
        schemas.ProgressOut.model_validate(without)
    after = NextOut.model_validate(load_api_examples()["next_after_skip"])
    assert after.assignment is not None and after.assignment.previous is None
    assert after.progress.skipped == 2


# --- media ----------------------------------------------------------------------------------


def test_media_base_and_tile_names() -> None:
    assert media_base("3f9a1c07d2e4b8a1") == "/media/3f9a1c07d2e4b8a1/"
    for base in ("https://media.example.org/run", "https://media.example.org/run/"):
        assert media_base("3f9a1c07d2e4b8a1", base) == (
            "https://media.example.org/run/3f9a1c07d2e4b8a1/"
        )
    tiles = TileFiles()
    assert tile_name(tiles.gray, "sagittal", 31) == tiles.gray_path("sagittal", 31)
    assert tile_name(tiles.contours, "axial", 5) == "tiles/axial_005_c.webp"


@pytest.mark.parametrize(
    "name",
    [
        "poster.webp",
        "anim.mp4",
        "tiles/sagittal_000.webp",
        "tiles/coronal_053_c.webp",
        "tiles/axial_999.webp",
        "tiles/axial_1000_c.webp",
    ],
)
def test_the_files_of_a_render_that_are_served(name: str) -> None:
    assert MEDIA_FILE.fullmatch(name)


@pytest.mark.parametrize(
    "name",
    [
        "render.json",
        "vertex_map.npz",
        "tiles/",
        "tiles/sagittal_00.webp",
        "tiles/sagittal_0001.webp",  # the renderer writes no leading zero beyond three digits
        "tiles/sagittal_\u0660\u0660\u0660.webp",  # Arabic-Indic digits: digits, and not ASCII
        "tiles/oblique_000.webp",
        "tiles/sagittal_000.webp.tmp",
        "tiles/../render.json",
        "../poster.webp",
        "/poster.webp",
        "poster.webp/",
        "poster.webp\n",
        "POSTER.WEBP",
        "anim.mp4.part",
        "",
    ],
)
def test_the_names_that_are_not_served(name: str) -> None:
    assert MEDIA_FILE.fullmatch(name) is None


# --- blindness ------------------------------------------------------------------------------


def _models_under(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if typing.get_origin(annotation) in (typing.Union, types.UnionType, tuple, dict, list):
        return [model for arg in typing.get_args(annotation) for model in _models_under(arg)]
    return []


def _field_names(shape: type[BaseModel], seen: set[type[BaseModel]] | None = None) -> set[str]:
    """The names of the fields of a shape and of every shape it holds."""
    seen = set() if seen is None else seen
    if shape in seen:
        return set()
    seen.add(shape)
    names = set()
    for name, field in shape.model_fields.items():
        names.add(field.alias or name)
        for model in _models_under(field.annotation):
            names |= _field_names(model, seen)
    return names


def test_no_shape_of_a_rating_route_has_a_place_for_what_a_rater_must_not_see() -> None:
    private = {
        "subject_label",
        "participant",
        "dataset",
        "pseudonym",
        "fs_version",
        "inputs",
        "sha256",
        "template",
        "scores",
        "ratings",
        "clusters",
        "final",
        "rater_id",
        "detail",
        "reason",
    }
    score_columns = set(table_schemas.SCORES.names) - {"subject_uid"}
    assert {"composite", "primary_rank", "flags", "dataset", "template_id"} <= score_columns
    for shape in (SessionOut, NextOut, RatingOut):
        names = _field_names(shape)
        assert "subject_uid" in names or shape is SessionOut
        assert not names & private, shape.__name__
        assert not names & score_columns, shape.__name__
    # the walk sees the nested shapes: the render and the marks of a revision
    assert {"pix2vox", "vox2pix", "frames", "client_id", "class", "pial_vertex"} <= _field_names(
        NextOut
    )
    # and it finds the unblinded shapes, which do have such places
    assert {"scores", "ratings", "clusters"} <= _field_names(schemas.SubjectOut)
