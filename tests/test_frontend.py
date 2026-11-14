"""The frontend of the rater app against ``app/schemas.py``, without a browser.

Two kinds of test. The first reads the sources (``app/static/``, ``app/templates/``): every
route the JavaScript calls is one of ``schemas.ROUTES``, every field it reads from an answer
or writes into a body exists in the shape of that name, nothing is fetched from another
host, and the pages render with the three variables the server gives them. The second runs
the modules that need no page (coordinates, marks, keys, rating rules, the event log) under
Node and compares them with the Python they mirror; it is skipped where there is no
``node``.

How the sources are read. Names in the JavaScript follow the shapes: a variable called
``assignment`` holds an ``AssignmentOut``, ``render`` a ``RenderOut``, and so on (``NAMES``
below). An access chain that starts at such a name (``render.panels[name].slices``,
``this.session.protocol.criteria``) is followed through the fields of the models, and a
name that is no field fails the test. An object literal after a comment ``/* shape: Name
*/`` is a body of that shape: its keys must be fields, and every required field a key. The
page itself is driven in a browser by ``tests/test_frontend_browser.py``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import numpy as np
import pytest
from pydantic import BaseModel

from surfcensus import protocol as protocol_module
from surfcensus import schemas as table_schemas
from surfcensus.app import schemas
from surfcensus.protocol import Criterion, Protocol, ScaleLevel
from surfcensus.render import geometry
from surfcensus.render.model import (
    PANEL_ORDER,
    BoxInfo,
    PanelInfo,
    RenderConfig,
)

APP_DIR = Path(schemas.__file__).parent
STATIC_DIR = APP_DIR / "static"
JS_DIR = STATIC_DIR / "js"
TEMPLATES_DIR = APP_DIR / "templates"

JS_FILES = (
    "api.js",
    "coords.js",
    "events.js",
    "help.js",
    "keys.js",
    "marks.js",
    "rate.js",
    "rating.js",
    "review.js",
    "tour.js",
    "viewer.js",
)
"""The modules of docs/dev/contract.md, section 7.9."""

USED_ROUTES = {
    ("GET", "/"),
    ("GET", "/review"),
    ("GET", "/api/session"),
    ("GET", "/api/next"),
    ("GET", "/api/back"),
    ("POST", "/api/rating"),
    ("POST", "/api/skip"),
    ("POST", "/api/events"),
    ("GET", "/api/review"),
    ("GET", "/api/review/{subject_uid}"),
    ("GET", "/api/export"),
}
"""What ``api.js`` names. The pages also load ``/static/...`` and ``/media/...``; they call
neither ``/api/stats`` nor ``/healthz``."""

REVIEW_COLUMNS = frozenset(re.findall(r"``(\w+)``", schemas.ReviewOut.__doc__ or "")) | frozenset(
    table_schemas.SCORES.names
)
"""The keys a row of ``GET /api/review`` can have: those its shape names, and the columns of
the scores table."""

# A JavaScript name -> what it holds: a shape, several shapes (a field of any), or the keys
# of a row that is a plain mapping.
NAMES: dict[str, type[BaseModel] | tuple[type[BaseModel], ...] | frozenset[str]] = {
    "session": schemas.SessionOut,
    "protocol": Protocol,
    "criterion": Criterion,
    "level": ScaleLevel,
    "example": schemas.ExampleOut,
    "next": schemas.NextOut,
    "answer": schemas.NextOut,
    "assignment": schemas.AssignmentOut,
    "previous": schemas.PreviousOut,
    "ahead": schemas.PrefetchOut,
    "progress": schemas.ProgressOut,
    "render": schemas.RenderOut,
    "panel": PanelInfo,
    "config": RenderConfig,
    "stored": schemas.RatingOut,
    "review": schemas.ReviewOut,
    "subject": schemas.SubjectOut,
    "cluster": schemas.ClusterOut,
    "problem": schemas.ErrorOut,
    "mark": (schemas.PreviousMark, schemas.ReviewMark),
    "row": REVIEW_COLUMNS,
}

SHAPES: dict[str, type[BaseModel]] = {
    "RatingIn": schemas.RatingIn,
    "SkipIn": schemas.SkipIn,
    "MarkIn": schemas.MarkIn,
    "EventsIn": schemas.EventsIn,
    "EventIn": schemas.EventIn,
    "ReviewQuery": schemas.ReviewQuery,
}

COLUMN_LISTS: dict[str, type[BaseModel]] = {
    "RATING_COLUMNS": schemas.ReviewRating,
    "MARK_COLUMNS": schemas.ReviewMark,
    "CLUSTER_COLUMNS": schemas.ClusterOut,
}
"""Arrays of field names in ``review.js`` -> the shape whose fields they name."""


# --- reading JavaScript -----------------------------------------------------------------------

_REGEX_AFTER = set("(,=:[!&|?{};")


def blank_js(source: str) -> str:
    """``source`` with comments, the text of strings and template literals, and regular
    expressions replaced by spaces, so that what is left is code. Positions are kept. The
    code inside ``${...}`` of a template literal stays."""
    out = list(source)
    n = len(source)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    def template_text(start: int) -> int:
        """Blank template text from ``start``; return where code goes on."""
        j = start
        while j < n:
            if source[j] == "\\":
                j += 2
            elif source[j] == "`":
                blank(start, j)
                return j + 1
            elif source.startswith("${", j):
                blank(start, j)
                templates.append(0)
                return j + 2
            else:
                j += 1
        raise ValueError("a template literal without an end")

    templates: list[int] = []
    last = ""
    i = 0
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            blank(i, end)
            i = end
        elif c in "'\"":
            j = i + 1
            while source[j] != c:
                j += 2 if source[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
            last = c
        elif c == "`":
            i = template_text(i + 1)
            last = "`"
        elif c == "/" and (last in _REGEX_AFTER or last == ""):
            j = i + 1
            in_class = False
            while source[j] != "/" or in_class:
                if source[j] == "\\":
                    j += 1
                elif source[j] == "[":
                    in_class = True
                elif source[j] == "]":
                    in_class = False
                j += 1
            blank(i + 1, j)
            i = j + 1
            last = "/"
        else:
            if templates and c == "{":
                templates[-1] += 1
            elif templates and c == "}":
                if templates[-1] == 0:
                    templates.pop()
                    i = template_text(i + 1)
                    last = "`"
                    continue
                templates[-1] -= 1
            if not c.isspace():
                last = c
            i += 1
    return "".join(out)


def wire_fields(model: type[BaseModel]) -> dict[str, Any]:
    """Field name as it is in JSON (the alias where there is one) -> annotation."""
    return {field.alias or name: field.annotation for name, field in model.model_fields.items()}


class Each:
    """A list or a mapping of models: an index gives the model."""

    def __init__(self, model: type[BaseModel]) -> None:
        self.model = model


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def follow(annotation: Any) -> type[BaseModel] | Each | None:
    """What a field holds, as far as the check goes on: a model, models, or nothing known."""
    if _is_model(annotation):
        return annotation
    origin = get_origin(annotation)
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if origin in (Union, types.UnionType):
        return follow(args[0]) if len(args) == 1 else None
    if origin in (tuple, list, dict) and args:
        element = args[-2] if origin is tuple and args[-1] is Ellipsis else args[-1]
        return Each(element) if _is_model(element) else None
    return None


_CHAIN = re.compile(
    r"(?<![\w$.])(?:this\.)?("
    + "|".join(sorted(NAMES))
    + r")\b((?:\s*\??\.\s*[A-Za-z_$][\w$]*|\[[^\[\]]*\])*)"
)
_STEP = re.compile(r"\s*\??\.\s*([A-Za-z_$][\w$]*)|(\[[^\[\]]*\])")


def field_problems(code: str, where: str, seen: dict[str, set[str]] | None = None) -> list[str]:
    """Every access in blanked code that names something no shape has."""
    problems = []
    code = code.replace("...", "   ")  # a spread is no access: `{ ...previous.values }`
    for match in _CHAIN.finditer(code):
        start: Any = NAMES[match.group(1)]
        chain = match.group(1)
        current: Any = start
        for step in _STEP.finditer(match.group(2)):
            name, index = step.group(1), step.group(2)
            if index is not None:
                chain += index
                current = current.model if isinstance(current, Each) else None
            elif isinstance(current, Each) or current is None:
                break  # a method of a list, or of a value the shapes say nothing about
            else:
                chain += f".{name}"
                if isinstance(current, frozenset):
                    fields: dict[str, Any] = dict.fromkeys(current)
                    owner = "a row of the review"
                elif isinstance(current, tuple):
                    fields = {key: None for model in current for key in wire_fields(model)}
                    owner = " or ".join(model.__name__ for model in current)
                else:
                    fields = wire_fields(current)
                    owner = current.__name__
                if seen is not None:
                    seen.setdefault(owner, set()).add(name)
                if name not in fields:
                    line = code.count("\n", 0, match.start()) + 1
                    problems.append(f"{where}:{line}: {chain}: {owner} has no field {name!r}")
                    break
                current = follow(fields[name])
            if current is None:
                break
    return problems


_SHAPE = re.compile(r"/\*\s*shape:\s*(\w+)\s*\*/\s*\{")


def literal_keys(code: str, open_brace: int) -> list[str]:
    """The keys of the object literal that starts at ``open_brace`` of blanked code."""
    depth = 0
    entries = [""]
    for position in range(open_brace, len(code)):
        c = code[position]
        if c in "{[(":
            depth += 1
            if depth == 1:
                continue
        elif c in "}])":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and c == ",":
            entries.append("")
        else:
            entries[-1] += c
    keys = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        key = re.match(r"([A-Za-z_$][\w$]*)\s*(:|$)", entry)
        if key is None:
            raise ValueError(f"not a plain key: {entry[:40]!r}")
        keys.append(key.group(1))
    return keys


def shape_problems(source: str, where: str, seen: set[str] | None = None) -> list[str]:
    """Every object literal marked ``/* shape: Name */`` whose keys are not that shape's."""
    code = blank_js(source)
    problems = []
    for match in _SHAPE.finditer(source):
        name = match.group(1)
        line = source.count("\n", 0, match.start()) + 1
        if name not in SHAPES:
            problems.append(f"{where}:{line}: no shape {name!r} is known to this test")
            continue
        if seen is not None:
            seen.add(name)
        model = SHAPES[name]
        fields = wire_fields(model)
        required = {
            field.alias or key for key, field in model.model_fields.items() if field.is_required()
        }
        keys = literal_keys(code, match.end() - 1)
        problems += [
            f"{where}:{line}: {name} has no field {key!r}" for key in keys if key not in fields
        ]
        problems += [
            f"{where}:{line}: {name} needs {key!r}" for key in sorted(required - set(keys))
        ]
    return problems


def js_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(JS_DIR.glob("*.js"))}


# --- the files -------------------------------------------------------------------------------


def test_the_frontend_files_are_those_of_the_contract() -> None:
    assert set(js_sources()) == set(JS_FILES)
    assert (STATIC_DIR / "app.css").is_file()
    assert {path.name for path in TEMPLATES_DIR.iterdir()} == {
        "base.html",
        "epi.html",
        "rate.html",
        "review.html",
    }
    # nothing to build and nothing else to ship: scripts, one stylesheet, four pages (the
    # EPI page's modules are in js/epi/ and tested in test_epi_frontend.py)
    others = [
        str(path.relative_to(STATIC_DIR))
        for path in STATIC_DIR.rglob("*")
        if path.is_file() and path.suffix not in {".js", ".css"}
    ]
    assert others == []


def test_modules_import_each_other_by_relative_path_and_nothing_else() -> None:
    for name, source in js_sources().items():
        imports = re.findall(r"""^\s*import\s[^;]*?from\s+["']([^"']+)["']""", source, flags=re.M)
        imports += re.findall(r"""^\s*import\s+["']([^"']+)["']""", source, flags=re.M)
        for target in imports:
            assert target.startswith("./"), f"{name} imports {target}"
            assert (JS_DIR / target[2:]).is_file(), f"{name} imports {target}, which does not exist"
        assert "import(" not in blank_js(source), f"{name} imports at run time"
        assert "require(" not in blank_js(source)


def test_every_name_a_module_imports_is_exported_by_the_module_it_names() -> None:
    sources = js_sources()
    exported = {
        name: set(
            re.findall(r"^export\s+(?:async\s+)?(?:function|const|class|let)\s+(\w+)", text, re.M)
        )
        for name, text in sources.items()
    }
    checked = 0
    for name, source in sources.items():
        for names, target in re.findall(r"import\s*\{([^}]*)\}\s*from\s*\"\./([\w.]+)\"", source):
            for wanted in filter(None, (part.strip() for part in names.split(","))):
                assert wanted in exported[target], f"{name} imports {wanted} from {target}"
                checked += 1
        for alias, target in re.findall(
            r"import\s*\*\s*as\s+(\w+)\s+from\s*\"\./([\w.]+)\"", source
        ):
            for wanted in set(re.findall(rf"\b{alias}\.(\w+)", blank_js(source))):
                assert wanted in exported[target], f"{name} uses {alias}.{wanted}"
                checked += 1
    assert checked > 40


# --- routes ----------------------------------------------------------------------------------


def js_routes() -> dict[str, tuple[str, str]]:
    source = js_sources()["api.js"]
    table = source[source.index("export const ROUTES = {") :]
    table = table[: table.index("};")]
    return {
        name: (method, path)
        for name, method, path in re.findall(r'(\w+):\s*\["(GET|POST)",\s*"([^"]+)"\]', table)
    }


def test_every_route_the_javascript_calls_is_a_route_of_the_schemas() -> None:
    known = {(route.method, route.path) for route in schemas.ROUTES}
    routes = js_routes()
    assert routes, "api.js has no route table"
    assert set(routes.values()) <= known
    assert set(routes.values()) == USED_ROUTES
    # a body goes to a route that takes one, and to no other
    takes_body = {(route.method, route.path) for route in schemas.ROUTES if route.method == "POST"}
    assert {routes["rating"], routes["skip"], routes["events"]} == takes_body


def test_requests_are_made_in_api_js_only_and_by_the_names_of_its_table() -> None:
    routes = js_routes()
    for name, source in js_sources().items():
        code = blank_js(source)
        for call in (
            r"(?<![\w.$])fetch\(",
            r"\bsendBeacon\(",
            "XMLHttpRequest",
            "WebSocket",
            "EventSource",
        ):
            found = len(re.findall(call, code))
            if name == "api.js" and call.endswith("\\("):
                assert found == 1, f"api.js calls {call} in {found} places"
            else:
                assert found == 0, f"{name} uses {call}"
        for used in re.findall(r'\b(?:request|routePath)\(\s*"(\w+)"', source):
            assert used in routes, f"{name} asks for the route {used!r}, which api.js does not have"
        # a path of the server appears as text in the route table and nowhere else
        paths = re.findall(r"""["'`](/(?:api|media|static|review|healthz)\b[^"'`]*)["'`]""", source)
        if name == "api.js":
            assert set(paths) <= {path for _, path in routes.values()}
        else:
            assert paths == [], f"{name} writes out the path {paths}"


_URL = re.compile(r"""(?:https?:)?//[\w.-]+\.[a-z]{2,}[^\s"'`)]*""", flags=re.I)
_NAMESPACES = {"http://www.w3.org/2000/svg"}


def frontend_texts() -> Iterator[tuple[str, str]]:
    for path in sorted([*STATIC_DIR.rglob("*"), *TEMPLATES_DIR.rglob("*")]):
        if path.is_file():
            yield str(path.relative_to(APP_DIR)), path.read_text(encoding="utf-8")


def test_nothing_is_fetched_from_another_host() -> None:
    for name, text in frontend_texts():
        urls = set(_URL.findall(text)) - _NAMESPACES
        assert urls == set(), f"{name} names another host: {sorted(urls)}"
        assert "@import" not in text, f"{name} imports a stylesheet"
        assert "@font-face" not in text, f"{name} loads a font"
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    for target in re.findall(r"url\(\s*[\"']?([^\"')]*)", css):
        assert target.startswith("data:image/svg+xml,"), f"app.css fetches {target[:40]}"


def test_the_url_check_finds_a_cdn_and_a_protocol_relative_url() -> None:
    assert _URL.findall('<script src="https://cdn.example.org/x.js">')
    assert _URL.findall("@import url(//fonts.example.com/css)")
    assert not (
        set(_URL.findall('createElementNS("http://www.w3.org/2000/svg", "g")')) - _NAMESPACES
    )
    assert not _URL.findall("const a = b // c.length is odd")


# --- fields ----------------------------------------------------------------------------------


def test_every_field_the_javascript_reads_exists_in_its_shape() -> None:
    seen: dict[str, set[str]] = {}
    problems = []
    for name, source in js_sources().items():
        problems += field_problems(blank_js(source), name, seen)
    assert problems == []
    # the check saw the code: these are read somewhere, by these names
    expected = {
        "SessionOut": {
            "protocol",
            "settings",
            "require_marks",
            "examples",
            "protocol_sha256",
            "rater",
        },
        "NextOut": {"status", "assignment", "prefetch", "progress", "retry_after_s"},
        "AssignmentOut": {"assignment_id", "previous"},
        "PreviousOut": {"values", "unratable", "other", "note", "marks"},
        "PrefetchOut": {"tiles", "n_slices", "media_base", "render_key"},
        "ProgressOut": {
            "done_today",
            "done_total",
            "remaining",
            "skipped",
            "rendered",
            "n_subjects",
        },
        "RenderOut": {"panels", "modes", "tiles", "config", "volume"},
        "PanelInfo": {
            "slices",
            "slice_mm",
            "pix2vox",
            "vox2pix",
            "slice_voxel_axis",
            "h_voxel_axis",
            "v_voxel_axis",
            "world_axis",
            "width",
            "height",
            "edges",
        },
        "RenderConfig": {"fps", "crosshair_colour", "crosshair_alpha"},
        "Protocol": {"version", "scale", "criteria", "other_mark_key", "unratable_reasons"},
        "Criterion": {"id", "label", "question", "keys", "mark_key"},
        "RatingOut": {"next"},
        "ReviewOut": {"total", "columns", "rows"},
        "SubjectOut": {"render", "marks", "clusters", "ratings", "scores", "subject_uid"},
        "ErrorOut": {"field", "message"},
    }
    for owner, fields in expected.items():
        assert fields <= seen.get(owner, set()), f"{owner}: {fields - seen.get(owner, set())}"
    assert {"client_id", "class", "panel", "slice", "px", "py"} <= seen[
        "PreviousMark or ReviewMark"
    ]
    # the viewer is given an assignment on the rating page and a subject on the review page,
    # under the name `subject`: what it reads of it is in both shapes
    shown = {"render", "media_base", "render_key"}
    assert shown <= seen["SubjectOut"]
    assert shown <= set(wire_fields(schemas.AssignmentOut))


@pytest.mark.parametrize(
    ("snippet", "said"),
    [
        ("show(assignment.subject_label);", "AssignmentOut has no field 'subject_label'"),
        ("const a = this.session.protocol.criteria[0].colour;", "Criterion has no field 'colour'"),
        ("x = render.panels[name].slice_px;", "PanelInfo has no field 'slice_px'"),
        ("if (next.assignment?.render.inputs) {}", "RenderOut has no field 'inputs'"),
        ("render.template", "RenderOut has no field 'template'"),
        ("for (const m of assignment.previous.marks) m; mark.vertex", "no field 'vertex'"),
        ("row.accession", "a row of the review has no field 'accession'"),
        ("const all = { ...previous.values, ...assignment.scores };", "no field 'scores'"),
        ("`${progress.done} of ${progress.n_subjects}`", "ProgressOut has no field 'done'"),
    ],
)
def test_the_field_check_finds_a_field_that_no_shape_has(snippet: str, said: str) -> None:
    problems = field_problems(blank_js(snippet), "snippet")
    assert len(problems) == 1 and said in problems[0]


def test_the_field_check_leaves_strings_comments_and_other_names_alone() -> None:
    code = blank_js(
        """
        // assignment.nothing is a comment
        const text = "render.nothing" + 'next.nothing' + `progress.nothing ${next.status}`;
        const pattern = /assignment.nothing/g;
        viewer.render.anything; stored.next.progress.remaining; render.modes[mode].frames.length;
        session.examples.map((example) => example.tiles[layer].replace("a", "b"));
        """
    )
    assert field_problems(code, "snippet") == []


def test_every_body_the_javascript_builds_has_the_fields_of_its_shape() -> None:
    seen: set[str] = set()
    problems = []
    for name, source in js_sources().items():
        problems += shape_problems(source, name, seen)
    assert problems == []
    assert seen == set(SHAPES)


def test_the_body_check_finds_a_stray_and_a_missing_key() -> None:
    stray = (
        "const body = /* shape: RatingIn */ "
        "{ assignment_id: 1, protocol_version, ms_on_subject, score: 2 };"
    )
    assert shape_problems(stray, "snippet") == ["snippet:1: RatingIn has no field 'score'"]
    missing = "post(/* shape: MarkIn */ { client_id, class: c, panel, slice: f(a, b), px });"
    assert shape_problems(missing, "snippet") == ["snippet:1: MarkIn needs 'py'"]
    assert shape_problems("/* shape: Nothing */ {}", "snippet") != []


def test_the_tables_of_the_review_screen_name_fields_of_their_shapes() -> None:
    source = js_sources()["review.js"]
    for constant, model in COLUMN_LISTS.items():
        found = re.search(rf"const {constant} = \[([^\]]*)\]", source)
        assert found, constant
        columns = re.findall(r'"(\w+)"', found.group(1))
        assert columns and set(columns) <= set(wire_fields(model)), constant


def test_the_events_the_page_logs_are_the_event_types_of_the_schemas() -> None:
    sources = js_sources()
    logged = set(re.findall(r'events\.add\(\s*"(\w+)"', sources["rate.js"]))
    logged |= set(re.findall(r'this\.onState\(\s*"(\w+)"', sources["viewer.js"]))
    logged |= set(re.findall(r'this\.add\(\s*"(\w+)"', sources["events.js"]))
    assert logged == set(schemas.EVENT_TYPES)
    # the event of a skip is the server's to write: no page sends one
    assert not logged & set(schemas.SERVER_EVENT_TYPES)
    assert {"tour_open", "tour_close"} <= logged


# --- pages -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    jinja2 = pytest.importorskip("jinja2")
    from surfcensus.app.server import PAGES, STATIC_URL
    from surfcensus.app.server import TEMPLATES_DIR as served

    assert served == TEMPLATES_DIR
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        undefined=jinja2.StrictUndefined,
        autoescape=True,
    )
    # the three variables the server gives a page, and nothing else
    return {
        route: environment.get_template(name).render(
            static=STATIC_URL, version="1.2.3", mode="local"
        )
        for route, name in PAGES.items()
    }


def test_the_pages_render_with_static_version_and_mode_alone(pages: dict[str, str]) -> None:
    assert set(pages) == {"/", "/review"}
    for route, script in (("/", "rate.js"), ("/review", "review.js")):
        text = pages[route]
        assert f'<script type="module" src="/static/js/{script}?v=1.2.3"></script>' in text
        assert '<link rel="stylesheet" href="/static/app.css?v=1.2.3">' in text
        assert 'data-mode="local"' in text
        # the browser asks for no favicon, which the server does not have
        assert '<link rel="icon" href="data:,">' in text
        assert len(re.findall(r"<script\b", text)) == 1 and "<style" not in text
        assert not re.search(r"\son\w+\s*=", text), "an inline event handler"


def test_the_pages_load_and_link_to_the_server_only(pages: dict[str, str]) -> None:
    routes = {path for _, path in js_routes().values()}
    for text in pages.values():
        for target in re.findall(r"""(?:src|href)="([^"]*)\"""", text):
            if target.startswith("/static/") or target == "data:,":
                continue
            assert target in routes, target


def test_the_pages_hold_nothing_about_a_subject(pages: dict[str, str]) -> None:
    for text in pages.values():
        assert not re.search(r"\b[0-9a-f]{16}\b", text)
        assert "/media/" not in text and "sub-" not in text


@pytest.mark.parametrize(("route", "script"), [("/", "rate.js"), ("/review", "review.js")])
def test_every_element_a_page_script_looks_up_is_in_its_page(
    pages: dict[str, str], route: str, script: str
) -> None:
    wanted = set(re.findall(r'byId\("([\w-]+)"\)', js_sources()[script]))
    assert wanted
    present = set(re.findall(r'\bid="([\w-]+)"', pages[route]))
    assert wanted <= present, wanted - present


def test_the_keys_named_on_the_rating_page_are_keys_of_the_keyboard_map(
    pages: dict[str, str],
) -> None:
    named = set(re.findall(r"<kbd>([^<]+)</kbd>", pages["/"]))
    aliases = {"Esc": "Escape", "Ctrl": None, "Z": None}
    fixed = set(protocol_module.FIXED_KEYS) | {"Escape"}
    for key in named:
        key = aliases.get(key, key)
        assert key is None or key in fixed, key


# --- the modules under Node --------------------------------------------------------------------

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def js_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The modules where Node reads them as ES modules whatever its version."""
    root = tmp_path_factory.mktemp("js")
    shutil.copytree(JS_DIR, root / "js")
    (root / "package.json").write_text('{"type": "module"}\n', encoding="utf-8")
    return root


def run_js(js_copy: Path, script: str, data: Any = None) -> Any:
    """Run a module script beside a copy of the modules; ``input`` is ``data``, and what it
    passes to ``output`` comes back."""
    assert NODE is not None
    (js_copy / "input.json").write_text(json.dumps(data), encoding="utf-8")
    main = (
        'import { readFileSync } from "node:fs";\n'
        'const text = readFileSync(new URL("./input.json", import.meta.url), "utf8");\n'
        "const input = JSON.parse(text);\n"
        "const output = (value) => console.log(JSON.stringify(value));\n" + script
    )
    (js_copy / "main.mjs").write_text(main, encoding="utf-8")
    done = subprocess.run(
        [NODE, "main.mjs"], cwd=js_copy, capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
@pytest.mark.parametrize("name", JS_FILES)
def test_every_module_parses(js_copy: Path, name: str) -> None:
    assert NODE is not None
    done = subprocess.run(
        [NODE, "--check", f"js/{name}"], cwd=js_copy, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def affine(rows: list[list[float]]) -> np.ndarray:
    return np.array([*rows, [0, 0, 0, 1]], dtype=float)


GEOMETRIES = {
    "conformed": (affine([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128]]), (256, 256, 256), {}),
    "hires_flipped": (
        affine([[-0.7, 0, 0, 112], [0, 0, 0.7, -112], [0, -0.7, 0, 112]]),
        (320, 320, 320),
        {"radiological": False, "sagittal_anterior_left": True, "panel_px": 333},
    ),
    "ras_anisotropic": (
        affine([[1, 0, 0, -50], [0, 2, 0, -120], [0, 0, 3, -210]]),
        (100, 120, 140),
        {"panel_px": 96, "slice_step_mm": 2.0},
    ),
}


def geometry_case(name: str) -> tuple[dict[str, PanelInfo], dict[str, Any], np.ndarray]:
    """Panels, the part of a render object that the page arithmetic reads, and the affine."""
    vox2ras_tkr, shape, settings = GEOMETRIES[name]
    config = RenderConfig(**settings)
    centre = (0.3, -95.2, -10.6) if name != "ras_anisotropic" else (0.3, -40.1, -20.3)
    box = BoxInfo(
        centre=centre,
        low=(
            centre[0] + config.box_x_mm[0],
            centre[1] + config.box_y_mm[0],
            centre[2] + config.box_z_mm[0],
        ),
        high=(
            centre[0] + config.box_x_mm[1],
            centre[1] + config.box_y_mm[1],
            centre[2] + config.box_z_mm[1],
        ),
    )
    panels = geometry.build_panels(box, geometry.axis_map(vox2ras_tkr, shape), config)
    modes = geometry.frame_tables(panels)
    render = {
        "panels": {key: panel.model_dump(mode="json") for key, panel in panels.items()},
        "modes": {key: mode.model_dump(mode="json") for key, mode in modes.items()},
        "volume": {"vox2ras_tkr": vox2ras_tkr.tolist()},
    }
    return dict(panels), render, vox2ras_tkr


@needs_node
@pytest.mark.parametrize("case", sorted(GEOMETRIES))
def test_a_coordinate_half_way_between_two_slices_goes_to_the_first(
    js_copy: Path, case: str
) -> None:
    """``nearestSlice`` gives the first of two tiles that are equally near, as numpy's
    ``argmin`` does in ``geometry.nearest_slice``. The review screen places a mark and a
    centroid on a tile with it, so a tie that went to the other tile would show them on
    another slice than the renderer names."""
    panels, render, _ = geometry_case(case)
    wanted = []
    for name in PANEL_ORDER:
        slices = panels[name].slices
        assert len(slices) >= 3
        for position in (0, len(slices) // 2, len(slices) - 2):
            wanted.append((name, position, (slices[position] + slices[position + 1]) / 2))
    out = run_js(
        js_copy,
        """
        import { nearestSlice } from "./js/coords.js";
        output(
          input.wanted.map(([name, _first, at]) => nearestSlice(input.render.panels[name], at)),
        );
        """,
        {"render": render, "wanted": wanted},
    )
    for (name, position, at), got in zip(wanted, out, strict=True):
        slices = np.array(panels[name].slices, dtype=float)
        # a tie in exact arithmetic: both distances are the same float
        assert abs(slices[position] - at) == abs(slices[position + 1] - at)
        assert got == position == int(np.argmin(np.abs(slices - at)))
        assert got == geometry.nearest_slice(panels[name], at)


@needs_node
@pytest.mark.parametrize("case", sorted(GEOMETRIES))
def test_coords_js_is_the_arithmetic_of_render_geometry(js_copy: Path, case: str) -> None:
    panels, render, vox2ras_tkr = geometry_case(case)
    rng = np.random.default_rng(7)
    points = []
    for name in PANEL_ORDER:
        panel = panels[name]
        for _ in range(40):
            points.append(
                {
                    "panel": name,
                    "slice": int(rng.integers(len(panel.slices))),
                    "px": float(rng.uniform(0, panel.width)),
                    "py": float(rng.uniform(0, panel.height)),
                    "shown": float(rng.uniform(100, 900)),
                }
            )
    out = run_js(
        js_copy,
        """
        import * as coords from "./js/coords.js";
        const render = input.render;
        output(input.points.map((p) => {
          const panel = render.panels[p.panel];
          // a click on a tile shown `shown` px wide, at the place of (px, py)
          const [px, py] = coords.clickToPixel(
            panel, p.px * p.shown / panel.width, p.py * p.shown / panel.height, p.shown, p.shown);
          const vox = coords.pixelToVoxel(panel, p.slice, px, py);
          const tkr = coords.voxelToTkr(render, vox);
          return {
            px, py, vox, tkr,
            back: coords.voxelToPixel(panel, vox),
            voxAgain: coords.tkrToVoxel(render, tkr),
            nearest: coords.nearestSlice(panel, vox[panel.slice_voxel_axis] + 0.3),
            distance: coords.sliceDistanceMm(panel, p.slice, tkr),
          };
        }));
        """,
        {"render": render, "points": points},
    )
    for point, got in zip(points, out, strict=True):
        panel = panels[point["panel"]]
        assert got["px"] == pytest.approx(point["px"], abs=1e-9)
        assert got["py"] == pytest.approx(point["py"], abs=1e-9)
        vox = geometry.pixel_to_voxel(panel, point["slice"], point["px"], point["py"])
        np.testing.assert_allclose(got["vox"], vox, atol=1e-9)
        np.testing.assert_allclose(got["tkr"], geometry.apply_affine(vox2ras_tkr, vox), atol=1e-9)
        px, py, along = geometry.voxel_to_pixel(panel, vox)
        np.testing.assert_allclose(got["back"], [float(px), float(py), float(along)], atol=1e-9)
        np.testing.assert_allclose(got["voxAgain"], vox, atol=1e-8)
        assert got["nearest"] == geometry.nearest_slice(panel, float(along) + 0.3)
        assert got["distance"] == pytest.approx(0.0, abs=1e-9)


@needs_node
@pytest.mark.parametrize("case", sorted(GEOMETRIES))
def test_the_crosshairs_are_those_of_the_composite_strip(js_copy: Path, case: str) -> None:
    panels, render, _ = geometry_case(case)
    out = run_js(
        js_copy,
        """
        import { crosshairPixels, crosshairCell } from "./js/coords.js";
        const all = {};
        for (const mode of ["sync", "seq"]) {
          all[mode] = input.modes[mode].frames.map((frame) => {
            const cross = crosshairPixels(input, frame);
            const cells = {};
            for (const name of Object.keys(cross)) cells[name] = cross[name].map(crosshairCell);
            return { cross, cells };
          });
        }
        output(all);
        """,
        render,
    )
    for mode, frames in geometry.frame_tables(panels).items():
        assert len(out[mode]) == len(frames.frames)
        for frame, got in zip(frames.frames, out[mode], strict=True):
            expected = geometry.crosshair_pixels(panels, frame)
            for name in PANEL_ORDER:
                np.testing.assert_allclose(got["cross"][name], expected[name], atol=1e-9)
                # the pixel column and row that frames.composite_strip colours
                cells = [int(np.floor(value + 1e-6)) for value in expected[name]]
                assert got["cells"][name] == cells


@needs_node
def test_tile_names_are_those_of_the_schemas_and_of_the_media_route(js_copy: Path) -> None:
    patterns = schemas.TileFiles()
    wanted = [
        (pattern, panel, index)
        for pattern in (patterns.gray, patterns.contours)
        for panel in PANEL_ORDER
        for index in (0, 7, 60, 999, 1000, 12345)
    ]
    out = run_js(
        js_copy,
        """
        import { tileName, tileUrl } from "./js/coords.js";
        output(input.map(([pattern, panel, index]) =>
          [tileName(pattern, panel, index),
           tileUrl("/media/0123456789abcdef/", pattern, panel, index, "5d0c")]));
        """,
        wanted,
    )
    for (pattern, panel, index), (name, url) in zip(wanted, out, strict=True):
        assert name == schemas.tile_name(pattern, panel, index)
        assert schemas.MEDIA_FILE.fullmatch(name)
        assert url == f"{schemas.media_base('0123456789abcdef')}{name}?v=5d0c"


@needs_node
def test_a_subject_has_every_tile_once_and_the_first_frame_in_front(js_copy: Path) -> None:
    _, render, _ = geometry_case("conformed")
    out = run_js(
        js_copy,
        """
        import { subjectTiles, sliceCounts, nearestFrame, wrapFrame } from "./js/viewer.js";
        const render = { ...input, tiles: { gray: "tiles/{panel}_{index:03d}.webp",
                                            contours: "tiles/{panel}_{index:03d}_c.webp" } };
        const counts = sliceCounts(render);
        const first = render.modes.sync.frames[5];
        output({
          counts,
          tiles: subjectTiles(render.tiles, counts, "/media/a/", "k", first)
            .map((tile) => tile.url),
          plain: subjectTiles(render.tiles, counts, "/media/a/", "k").length,
          nearest: render.modes.seq.frames
            .map((frame) => nearestFrame(render.modes.sync.frames, frame)),
          wrapped: [-1, 120, 131, -11].map((frame) => wrapFrame(frame, 120)),
        });
        """,
        render,
    )
    assert out["counts"] == {"sagittal": 61, "coronal": 54, "axial": 61}
    assert len(out["tiles"]) == len(set(out["tiles"])) == out["plain"] == 2 * (61 + 54 + 61)
    assert out["tiles"][:6] == [
        "/media/a/tiles/sagittal_005.webp?v=k",
        "/media/a/tiles/sagittal_005_c.webp?v=k",
        "/media/a/tiles/coronal_004.webp?v=k",
        "/media/a/tiles/coronal_004_c.webp?v=k",
        "/media/a/tiles/axial_005.webp?v=k",
        "/media/a/tiles/axial_005_c.webp?v=k",
    ]
    assert out["wrapped"] == [119, 0, 11, 109]
    # a change of mode lands on the frame with the smallest sum of slice differences
    sync = np.array(render["modes"]["sync"]["frames"])
    for frame, nearest in zip(render["modes"]["seq"]["frames"], out["nearest"], strict=True):
        assert nearest == int(np.argmin(np.abs(sync - np.array(frame)).sum(axis=1)))


@needs_node
def test_a_click_is_taken_at_the_middle_of_its_screen_pixel(js_copy: Path) -> None:
    """A browser names the screen pixel under the pointer by its corner, in whole pixels;
    the middle is the place without a bias. A coordinate with a fraction is the pointer's
    own. Through ``clickToPixel``, a click on screen pixel 100 of a 320 px tile shown 454 px
    wide is tile pixel 100.5 * 320 / 454."""
    out = run_js(
        js_copy,
        """
        import { pointerAt } from "./js/viewer.js";
        import { clickToPixel } from "./js/coords.js";
        const panel = { width: 320, height: 320 };
        output({
          at: [0, 100, 453, 100.25, 0.5, -3].map(pointerAt),
          px: clickToPixel(panel, pointerAt(111) - 11, pointerAt(40.75) - 11, 454, 454),
        });
        """,
        {},
    )
    assert out["at"] == [0.5, 100.5, 453.5, 100.25, 0.5, -2.5]
    assert out["px"] == pytest.approx([100.5 * 320 / 454, 29.75 * 320 / 454])


@needs_node
def test_a_click_beyond_the_shown_tile_is_clamped_to_the_tile(js_copy: Path) -> None:
    """The offset of a click on the last screen pixel can exceed the shown size: the middle
    of a whole screen pixel less a box edge with a fraction. The server refuses a mark
    outside ``0 <= px <= width`` and ``0 <= py <= height``, so ``clickToPixel`` keeps it on
    the tile, along each axis on its own."""
    out = run_js(
        js_copy,
        """
        import { pointerAt } from "./js/viewer.js";
        import { clickToPixel } from "./js/coords.js";
        const panel = { width: 320, height: 300 };
        output({
          edge: pointerAt(464) - 10.25,
          px: [
            clickToPixel(panel, pointerAt(464) - 10.25, pointerAt(435) - 10.25, 454, 425),
            clickToPixel(panel, -0.3, -2, 454, 425),
            clickToPixel(panel, 500, -1, 454, 425),
            clickToPixel(panel, -1, 500, 454, 425),
            clickToPixel(panel, 454, 425, 454, 425),
            clickToPixel(panel, 0, 0, 454, 425),
          ],
        });
        """,
        {},
    )
    # the last screen pixel of a box that starts at 10.25: an offset of 454.25 in 454
    assert out["edge"] == 454.25
    assert out["px"] == [[320, 300], [0, 0], [320, 0], [0, 300], [320, 300], [0, 0]]


@needs_node
@pytest.mark.parametrize("case", ["conformed", "hires_flipped"])
def test_a_mark_shows_in_every_panel_within_the_distance_and_fades(
    js_copy: Path, case: str
) -> None:
    panels, render, vox2ras_tkr = geometry_case(case)
    side = panels["sagittal"].width
    marks = [
        # inside the box, so every panel has slices near them
        {"panel": "sagittal", "slice": 20, "px": 0.45 * side, "py": 0.52 * side},
        {"panel": "coronal", "slice": 3, "px": 0.6 * side, "py": 0.4 * side},
        {"panel": "axial", "slice": 40, "px": 0.38 * side, "py": 0.55 * side},
        # in the margin of a tile, beyond the slices of the coronal panel
        {
            "panel": "sagittal",
            "slice": 5,
            "px": 0.99 * side if case == "conformed" else 0.01 * side,
            "py": 0.5 * side,
        },
    ]
    out = run_js(
        js_copy,
        """
        import { markInPanel, markOpacity, MIN_OPACITY } from "./js/marks.js";
        const render = input.render;
        const shown = input.marks.map((mark) => {
          const panels = {};
          for (const name of Object.keys(render.panels)) {
            panels[name] = render.panels[name].slices
              .map((_, slice) => markInPanel(render, mark, name, slice, 3));
          }
          return panels;
        });
        output({ shown, MIN_OPACITY, opacity: [0, 1.5, 3, 4].map((d) => markOpacity(d, 3)),
                 zero: markOpacity(0, 0) });
        """,
        {"render": render, "marks": marks},
    )
    low = out["MIN_OPACITY"]
    assert 0 < low < 1
    assert out["opacity"] == pytest.approx([1.0, (1 + low) / 2, low, low]) and out["zero"] == 1
    for mark, shown in zip(marks, out["shown"], strict=True):
        vox = geometry.pixel_to_voxel(panels[mark["panel"]], mark["slice"], mark["px"], mark["py"])
        tkr = geometry.apply_affine(vox2ras_tkr, vox)
        for name in PANEL_ORDER:
            panel = panels[name]
            px, py, _ = geometry.voxel_to_pixel(panel, vox)
            n_shown = 0
            for position, got in enumerate(shown[name]):
                distance = abs(float(tkr[panel.world_axis]) - panel.slice_mm[position])
                if distance > 3.0:
                    assert got is None, (name, position)
                    continue
                n_shown += 1
                assert got["px"] == pytest.approx(float(px)) and got["py"] == pytest.approx(
                    float(py)
                )
                assert got["distance"] == pytest.approx(distance, abs=1e-9)
                assert got["opacity"] == pytest.approx(1 - (1 - low) * distance / 3.0)
            # one slice every millimetre: a point inside the sweep is near six or seven
            if mark is marks[3] and name == "coronal":
                assert n_shown == 0
            else:
                assert n_shown >= 4, (mark, name)
        # on its own tile the mark is where it was put, at full strength
        own = shown[mark["panel"]][mark["slice"]]
        assert (own["px"], own["py"]) == pytest.approx((mark["px"], mark["py"]))
        assert own["opacity"] == pytest.approx(1.0)


@needs_node
def test_the_marks_of_a_subject_add_undo_delete_and_post(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { MarkStore } from "./js/marks.js";
        const store = new MarkStore();
        const click = (panel, slice) =>
          ({ panel, slice, px: 10.5, py: 20.25, frame: 3, mode: "sync", target: null });
        store.reset(input.previous);
        const first = store.add("A", click("sagittal", 3), "2026-11-13T10:00:00.000Z");
        const second = store.add("X", click("axial", 1), "2026-11-13T10:00:01.000Z");
        const steps = { ids: store.marks.map((mark) => mark.client_id), selected: store.selected,
                        countA: store.count("A"), posted: store.posted() };
        store.select("m8");
        steps.removed = store.remove("m8").client_id;
        steps.selectedAfter = store.selected;
        steps.undone = [store.undo().client_id, store.undo().client_id, store.undo()];
        steps.left = store.marks.length;
        store.reset(steps.posted);
        steps.next = store.add("B", click("coronal", 0), null).client_id;
        output(steps);
        """,
        {
            "previous": [
                {
                    "client_id": "m8",
                    "mark_id": 8,
                    "class": "A",
                    "panel": "sagittal",
                    "slice": 3,
                    "px": 30.5,
                    "py": 28.25,
                    "frame": 3,
                    "mode": "sync",
                    "ts": "2026-11-12T14:03:20.004Z",
                    "hemi": "lh",
                    "pial_vertex": 211,
                    "box_dist_mm": 0.0,
                }
            ]
        },
    )
    assert out["ids"] == ["m8", "n1", "n2"] and out["selected"] == "n2" and out["countA"] == 2
    # what is posted is a MarkIn each: the resolved fields of an earlier answer are left out
    posted = [schemas.MarkIn.model_validate(mark) for mark in out["posted"]]
    assert [mark.mark_class for mark in posted] == ["A", "A", "X"]
    assert posted[1].px == 10.5 and posted[1].frame == 3 and posted[1].mode == "sync"
    assert out["removed"] == "m8" and out["selectedAfter"] is None
    # undo takes the marks placed on this page, last first, and no others
    assert out["undone"] == ["n2", "n1", None] and out["left"] == 0
    # a draft that is loaded again goes on counting where it was
    assert out["next"] == "n3"


@needs_node
def test_the_keyboard_map_has_every_fixed_key_and_the_keys_of_the_protocol(
    js_copy: Path, shipped_protocol: Protocol
) -> None:
    out = run_js(
        js_copy,
        """
        import { FIXED_ACTIONS, VIEWER_KEYS, buildKeyMap } from "./js/keys.js";
        output({ fixed: Object.keys(FIXED_ACTIONS), viewer: VIEWER_KEYS,
                 map: [...buildKeyMap(input).entries()],
                 viewerMap: [...buildKeyMap(null, VIEWER_KEYS).keys()] });
        """,
        shipped_protocol.model_dump(mode="json"),
    )
    assert set(out["fixed"]) == set(protocol_module.FIXED_KEYS) | {"Escape"}
    assert set(out["viewer"]) <= set(out["fixed"]) and set(out["viewerMap"]) == set(out["viewer"])
    actions = dict(out["map"])
    assert len(actions) == len(out["map"])
    for criterion in shipped_protocol.criteria:
        for key, level in zip(criterion.keys, shipped_protocol.scale, strict=True):
            assert actions[key] == {
                "type": "value",
                "criterion": criterion.id,
                "value": level.value,
            }
        assert actions[criterion.mark_key] == {"type": "class", "class": criterion.id}
    assert actions[shipped_protocol.other_mark_key] == {"type": "class", "class": "X"}
    assert actions[" "] == {"type": "toggle_play"} and actions["Enter"] == {"type": "submit"}
    assert actions["s"] == {"type": "hide_contours", "hold": True}
    assert actions["h"] == {"type": "hide_marks", "hold": True}
    assert actions["k"] == {"type": "skip"} and actions["t"] == {"type": "tour"}
    assert actions["Backspace"] == {"type": "back"}


KEY_CASES: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = [
    # (event, context, action)
    ({"key": " "}, {}, {"type": "toggle_play"}),
    ({"key": "ArrowRight"}, {}, {"type": "step", "by": 1}),
    ({"key": "ArrowLeft", "shiftKey": True}, {}, {"type": "step", "by": -10}),
    ({"key": "]"}, {}, {"type": "speed", "by": 1}),
    ({"key": "M"}, {}, {"type": "toggle_mode"}),
    ({"key": "3"}, {}, {"type": "value", "criterion": "A", "value": 2}),
    ({"key": "B", "shiftKey": True}, {}, {"type": "class", "class": "B"}),
    ({"key": "?", "shiftKey": True}, {}, {"type": "help"}),
    ({"key": "z", "ctrlKey": True}, {}, {"type": "undo_mark"}),
    ({"key": "z", "metaKey": True}, {}, {"type": "undo_mark"}),
    ({"key": "Z", "ctrlKey": True, "shiftKey": True}, {}, None),
    # a shortcut of the browser is the browser's
    ({"key": "s", "ctrlKey": True}, {}, None),
    ({"key": "r", "metaKey": True}, {}, None),
    ({"key": "ArrowLeft", "altKey": True}, {}, None),
    ({"key": "F5"}, {}, None),
    ({"key": "q"}, {}, None),
    # a text field takes everything but Escape
    ({"key": "s"}, {"text": True}, None),
    ({"key": "3"}, {"text": True}, None),
    ({"key": "Enter"}, {"text": True}, None),
    ({"key": "Backspace"}, {"text": True}, None),
    ({"key": " "}, {"text": True}, None),
    ({"key": "z", "ctrlKey": True}, {"text": True}, None),
    ({"key": "Escape"}, {"text": True}, {"type": "escape"}),
    # the reasons take the digits; nothing else of the map gets through
    ({"key": "3"}, {"dialog": "unratable"}, {"type": "reason", "index": 2}),
    ({"key": "ArrowDown"}, {"dialog": "unratable"}, {"type": "reason_move", "by": 1}),
    ({"key": "Enter"}, {"dialog": "unratable"}, {"type": "reason_choose"}),
    ({"key": "u"}, {"dialog": "unratable"}, {"type": "reason_cancel"}),
    ({"key": "Escape"}, {"dialog": "unratable"}, {"type": "reason_cancel"}),
    ({"key": " "}, {"dialog": "unratable"}, None),
    ({"key": "Backspace"}, {"dialog": "unratable"}, None),
    ({"key": "z", "ctrlKey": True}, {"dialog": "unratable"}, None),
    ({"key": "?"}, {"dialog": "help"}, {"type": "help"}),
    ({"key": "Escape"}, {"dialog": "help"}, {"type": "help"}),
    ({"key": "Enter"}, {"dialog": "help"}, None),
    ({"key": "1"}, {"dialog": "help"}, None),
    ({"key": "k"}, {"dialog": "help"}, None),
    # skip and the tour, on the rating screen
    ({"key": "k"}, {}, {"type": "skip"}),
    ({"key": "K", "shiftKey": True}, {}, {"type": "skip"}),
    ({"key": "t"}, {}, {"type": "tour"}),
    ({"key": "k", "ctrlKey": True}, {}, None),
    ({"key": "k"}, {"text": True}, None),
    ({"key": "t"}, {"text": True}, None),
    ({"key": "k"}, {"dialog": "unratable"}, None),
    ({"key": "t"}, {"dialog": "unratable"}, None),
    # the help screen leads to the tour
    ({"key": "t"}, {"dialog": "help"}, {"type": "tour"}),
    # the tour takes its own keys and lets nothing through to the rating screen
    ({"key": "Enter"}, {"dialog": "tour"}, {"type": "tour_step", "by": 1}),
    ({"key": "ArrowRight"}, {"dialog": "tour"}, {"type": "tour_step", "by": 1}),
    ({"key": "ArrowRight", "shiftKey": True}, {"dialog": "tour"}, {"type": "tour_step", "by": 1}),
    ({"key": "ArrowLeft"}, {"dialog": "tour"}, {"type": "tour_step", "by": -1}),
    ({"key": "Escape"}, {"dialog": "tour"}, {"type": "tour_close"}),
    ({"key": "t"}, {"dialog": "tour"}, {"type": "tour_close"}),
    ({"key": "Escape"}, {"dialog": "tour", "text": True}, {"type": "tour_close"}),
    *[
        ({"key": key}, {"dialog": "tour"}, None)
        for key in (" ", "1", "5", "9", "a", "x", "k", "u", "o", "n", "m", "z", "s", "h", "?")
    ],
    ({"key": "Backspace"}, {"dialog": "tour"}, None),
    ({"key": "Delete"}, {"dialog": "tour"}, None),
    ({"key": "z", "ctrlKey": True}, {"dialog": "tour"}, None),
    ({"key": "Enter", "ctrlKey": True}, {"dialog": "tour"}, None),
]


@needs_node
def test_what_a_key_does_depends_on_where_the_page_is(
    js_copy: Path, shipped_protocol: Protocol
) -> None:
    out = run_js(
        js_copy,
        """
        import { buildKeyMap, keyAction } from "./js/keys.js";
        const map = buildKeyMap(input.protocol);
        output(input.cases.map(([event, context]) => keyAction(
          { ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...event },
          map, { text: false, dialog: null, ...context })));
        """,
        {
            "protocol": shipped_protocol.model_dump(mode="json"),
            "cases": [[event, context] for event, context, _ in KEY_CASES],
        },
    )
    for (event, context, wanted), got in zip(KEY_CASES, out, strict=True):
        assert got == wanted, (event, context)


RATING_CASES: list[tuple[dict[str, Any], dict[str, int], bool, list[str]]] = [
    # (rating, mark counts, marks required, fragments of the blockers)
    ({"values": {}}, {}, False, ["no value yet for A, B, C"]),
    ({"values": {"A": 0, "C": 1}}, {}, False, ["no value yet for B"]),
    ({"values": {"A": 0, "B": 0, "C": 0}}, {}, True, []),
    ({"values": {"A": 2, "B": 1, "C": 0}}, {}, False, []),
    (
        {"values": {"A": 2, "B": 1, "C": 0}},
        {"B": 1, "X": 4},
        True,
        ["criterion A is rated major but has no mark of class A"],
    ),
    ({"values": {"A": 2, "B": 1, "C": 0}}, {"A": 1, "B": 2}, True, []),
    ({"values": {}, "unratable": "missing surface"}, {}, True, []),
    ({"values": {"A": 2}, "unratable": "missing surface"}, {}, True, []),
    (
        {"values": {"A": 0, "B": 0, "C": 0}, "other": True, "note": "  "},
        {},
        False,
        ["needs a note"],
    ),
    ({"values": {}, "unratable": "render failure", "other": True}, {}, False, ["needs a note"]),
    ({"values": {"A": 0, "B": 0, "C": 0}, "other": True, "note": "skull"}, {}, False, []),
]


@needs_node
def test_enter_is_held_back_by_the_rules_of_a_submission(
    js_copy: Path, shipped_protocol: Protocol
) -> None:
    cases = [
        [{"unratable": None, "other": False, "note": "", **rating}, counts, required]
        for rating, counts, required, _ in RATING_CASES
    ]
    out = run_js(
        js_copy,
        """
        import { blockers, emptyRating, isPositive, ratingFromPrevious } from "./js/rating.js";
        output({
          blockers: input.cases.map(([rating, counts, required]) =>
            blockers(input.protocol, rating, counts, required)),
          empty: emptyRating(),
          positive: [0, 1, 2].map((value) => isPositive(input.protocol, value)),
          previous: ratingFromPrevious(
            { values: { A: 1 }, unratable: null, other: true, note: "x", marks: [] }),
        });
        """,
        {"protocol": shipped_protocol.model_dump(mode="json"), "cases": cases},
    )
    for (rating, counts, required, fragments), got in zip(
        RATING_CASES, out["blockers"], strict=True
    ):
        assert len(got) == len(fragments), (rating, counts, required, got)
        for fragment, sentence in zip(fragments, got, strict=True):
            assert fragment in sentence
    assert out["empty"] == {"values": {}, "unratable": None, "other": False, "note": ""}
    assert out["positive"] == [value in shipped_protocol.positive_values() for value in (0, 1, 2)]
    assert out["previous"] == {"values": {"A": 1}, "unratable": None, "other": True, "note": "x"}


def test_the_doc_of_the_app_does_not_call_the_submit_button_disabled() -> None:
    # docs/rater_app.md said in one place that `Enter` stays disabled while a mark is
    # missing, and in another that the button is never disabled and a refused `Enter` is
    # answered in a box under it, which is what the page does (the tests around this one)
    doc = Path(__file__).parent.parent / "docs" / "rater_app.md"
    text = " ".join(doc.read_text(encoding="utf-8").split())
    assert "stays disabled" not in text
    assert "The submit button is not disabled while a rating is incomplete" in text
    assert "`Enter` is refused, and the box under the submit button names" in text


LACKING_CASES: list[tuple[dict[str, Any], dict[str, int], bool, dict[str, str]]] = [
    # (rating, mark counts, marks required, criterion -> what its row lacks)
    ({"values": {}}, {}, False, {"A": "value", "B": "value", "C": "value"}),
    ({"values": {"A": 0, "C": 1}}, {}, False, {"B": "value"}),
    ({"values": {"A": 0, "C": 1}}, {}, True, {"B": "value", "C": "mark"}),
    ({"values": {"A": 2, "B": 1, "C": 0}}, {"B": 1, "X": 4}, True, {"A": "mark"}),
    ({"values": {"A": 2, "B": 1, "C": 0}}, {"A": 1, "B": 2}, True, {}),
    ({"values": {"A": 2, "B": 1, "C": 0}}, {}, False, {}),
    # an unratable subject lacks nothing in its rows
    ({"values": {"A": 2}, "unratable": "missing surface"}, {}, True, {}),
]


@needs_node
def test_a_refused_enter_names_the_rows_that_hold_it_back(
    js_copy: Path, shipped_protocol: Protocol
) -> None:
    cases = [
        [{"unratable": None, "other": False, "note": "", **rating}, counts, required]
        for rating, counts, required, _ in LACKING_CASES
    ]
    out = run_js(
        js_copy,
        """
        import { blockers, rowsLacking } from "./js/rating.js";
        output(input.cases.map(([rating, counts, required]) => ({
          lacking: rowsLacking(input.protocol, rating, counts, required),
          blockers: blockers(input.protocol, rating, counts, required),
        })));
        """,
        {"protocol": shipped_protocol.model_dump(mode="json"), "cases": cases},
    )
    for (rating, counts, required, lacking), got in zip(LACKING_CASES, out, strict=True):
        assert got["lacking"] == lacking, (rating, counts, required)
        # a row is named exactly when a rule about the rows holds the rating back
        assert bool(got["lacking"]) == bool(got["blockers"]), (rating, got)


@needs_node
def test_the_body_of_a_skip_is_a_skip_in_and_holds_nothing_of_the_rating(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { skipBody } from "./js/rating.js";
        const assignment = { assignment_id: 12, subject_uid: "3f9a1c07d2e4b8a1", previous: null };
        output([
          skipBody(assignment, 9400.4, "2026-11-12T14:03:22.120Z"),
          skipBody(assignment, -3, "2026-11-12T14:03:23.000Z"),
        ]);
        """,
    )
    assert out[0] == {
        "assignment_id": 12,
        "client_ts": "2026-11-12T14:03:22.120Z",
        "ms_on_subject": 9400,
    }
    first, second = (schemas.SkipIn.model_validate(body) for body in out)
    assert first.ms_on_subject == 9400 and second.ms_on_subject == 0


@needs_node
def test_the_body_of_a_rating_is_a_rating_in(js_copy: Path, shipped_protocol: Protocol) -> None:
    mark = {
        "client_id": "n1",
        "class": "A",
        "panel": "sagittal",
        "slice": 3,
        "px": 30.5,
        "py": 28.25,
        "frame": 3,
        "mode": "sync",
        "ts": "2026-11-12T14:03:20.004Z",
    }
    out = run_js(
        js_copy,
        """
        import { ratingBody } from "./js/rating.js";
        const assignment = { assignment_id: 12 };
        const rated =
          { values: { A: 2, B: 0, C: 0 }, unratable: null, other: true, note: "  dura  " };
        const unratable =
          { values: { A: 2 }, unratable: "missing surface", other: false, note: "" };
        const at = "2026-11-12T14:03:22.120Z";
        output([
          ratingBody(assignment, input.protocol, rated, [input.mark], 14200.6, at),
          ratingBody(assignment, input.protocol, unratable, [input.mark], -5, at),
        ]);
        """,
        {"protocol": shipped_protocol.model_dump(mode="json"), "mark": mark},
    )
    rated, unratable = (schemas.RatingIn.model_validate(body) for body in out)
    assert rated.assignment_id == 12 and rated.protocol_version == shipped_protocol.version
    assert rated.values == {"A": 2, "B": 0, "C": 0} and rated.unratable is None
    assert rated.other and rated.note == "dura" and rated.ms_on_subject == 14201
    assert [m.model_dump(by_alias=True) for m in rated.marks] == [mark]
    # an unratable subject goes without values and marks, and time is never negative
    assert unratable.unratable == "missing surface" and unratable.values == {}
    assert unratable.marks == () and unratable.ms_on_subject == 0


@needs_node
def test_frame_viewing_is_run_length_encoded_per_mode(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { FrameLog, MAX_VISITS_PER_EVENT } from "./js/events.js";
        const log = new FrameLog();
        const done = [];
        const take = (payloads) => done.push(...payloads);
        take(log.show("sync", 0, 1000));
        take(log.show("sync", 1, 1083));
        take(log.show("sync", 2, 1166));
        take(log.show("sync", 2, 1300));        // the same frame again: one visit
        take(log.drain(2000));                  // a flush: the frame stays on screen
        take(log.show("seq", 40, 2500));        // the mode changes: the stretch ends
        take(log.stop(2600));                   // the page is hidden
        take(log.show("seq", 40, 9000));        // and shown again: the time between is not viewing
        take(log.show("seq", 41, 9100));
        take(log.stop(9150));
        const long = new FrameLog();
        const full = [];
        for (let frame = 0; frame <= MAX_VISITS_PER_EVENT + 5; frame += 1) {
          full.push(...long.show("sync", frame % 120, frame * 10));
        }
        output({ done, full: full.map((payload) => payload.visits.length), MAX_VISITS_PER_EVENT });
        """,
    )
    assert out["done"] == [
        {"mode": "sync", "visits": [[0, 83], [1, 83], [2, 834]]},
        {"mode": "sync", "visits": [[2, 500]]},
        {"mode": "seq", "visits": [[40, 100]]},
        {"mode": "seq", "visits": [[40, 100], [41, 50]]},
    ]
    assert out["full"] == [out["MAX_VISITS_PER_EVENT"]]
    # the largest payload a full event can have stays under what the server stores
    visits = [[345, 3_600_000]] * out["MAX_VISITS_PER_EVENT"]
    schemas.EventIn(ts="t", type="frames", payload={"mode": "sync", "visits": visits})


@needs_node
def test_events_are_posted_in_batches_and_kept_while_the_server_is_away(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { EventLog, MAX_EVENTS_PER_CALL } from "./js/events.js";
        let now = 0;
        let fail = null;
        const sent = [];
        const beacons = [];
        const log = new EventLog({
          send: async (body) => {
            if (fail) throw fail;
            sent.push(body);
          },
          beacon: (body) => beacons.push(body),
          now: () => now,
          wall: () => `t${now}`,
        });
        log.add("play", {});                     // no subject yet: dropped
        log.begin(12);
        log.add("load", { ms: 5 });
        log.frame("sync", 0); now = 100;
        log.frame("sync", 1); now = 150;
        await log.flush();
        const first = sent.length;
        for (let n = 0; n < MAX_EVENTS_PER_CALL + 3; n += 1) log.add("mark_add", { n });
        log.begin(13);                           // the next subject: the frames of 12 are closed
        log.add("load", {});
        fail = Object.assign(new Error("away"), { name: "NetworkError" });
        await log.flush();
        const whileAway = sent.length;
        fail = Object.assign(new Error("refused"), { status: 422 });
        log.begin(14);
        log.add("zoom", { zoom: 2 });
        const queuedBefore = log.queue.length;
        fail = null;
        await log.flush();
        log.add("help", { open: true });
        log.frame("seq", 7); now = 400;
        log.leave();
        output({ sent, first, whileAway, queuedBefore, beacons, left: log.queue.length });
        """,
    )
    batches = [schemas.EventsIn.model_validate(body) for body in out["sent"]]
    beacons = [schemas.EventsIn.model_validate(body) for body in out["beacons"]]
    assert out["first"] == 1
    assert [event.type for event in batches[0].events] == ["load", "frames"]
    assert batches[0].assignment_id == 12
    assert batches[0].events[1].payload == {"mode": "sync", "visits": [[0, 100], [1, 50]]}
    # nothing arrived while the server did not answer, and nothing was lost
    assert out["whileAway"] == 1 and out["queuedBefore"] == schemas.MAX_EVENTS_PER_CALL + 6
    later = batches[1:]
    assert [(batch.assignment_id, len(batch.events)) for batch in later] == [
        (12, schemas.MAX_EVENTS_PER_CALL),
        (12, 4),
        (13, 1),
        (14, 1),
    ]
    assert [event.payload for event in later[0].events[:2]] == [{"n": 0}, {"n": 1}]
    assert later[1].events[-1].type == "frames"
    # leaving the page hands the rest to the browser
    assert [(batch.assignment_id, [e.type for e in batch.events]) for batch in beacons] == [
        (14, ["help", "frames"])
    ]
    assert out["left"] == 0


@needs_node
def test_a_refused_batch_of_events_is_dropped_and_the_next_one_sent(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { EventLog } from "./js/events.js";
        const sent = [];
        const log = new EventLog({
          send: async (body) => {
            if (body.assignment_id === 1) throw Object.assign(new Error("gone"), { status: 404 });
            sent.push(body);
          },
          beacon: () => {}, now: () => 0, wall: () => "t",
        });
        log.begin(1); log.add("play", {});
        log.begin(2); log.add("pause", {});
        await log.flush();
        output({ sent, left: log.queue.length });
        """,
    )
    assert [body["assignment_id"] for body in out["sent"]] == [2] and out["left"] == 0


API_STUB = """
const calls = [];
let answers = [];
globalThis.fetch = async (path, options) => {
  calls.push([options.method, path, options.body ?? null]);
  const answer = answers.shift();
  if (answer === "down") throw new TypeError("fetch failed");
  const text = async () => JSON.stringify(answer.body);
  return { ok: answer.status < 400, status: answer.status, text };
};
const store = new Map();
globalThis.sessionStorage = {
  getItem: (key) => (store.has(key) ? store.get(key) : null),
  setItem: (key, value) => store.set(key, String(value)),
  removeItem: (key) => store.delete(key),
};
const sleeps = [];
const sleep = async (ms) => { sleeps.push(ms); };
"""


@needs_node
def test_a_rating_is_sent_again_until_the_server_answers_and_kept_meanwhile(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        API_STUB
        + """
        import * as api from "./js/api.js";
        const body = { assignment_id: 12, protocol_version: "v", ms_on_subject: 1 };
        const busy = { errors: [{ field: "request", message: "busy" }] };
        answers = ["down", { status: 503, body: busy }, { status: 200, body: { saved: 1 } }];
        const kept = [];
        const retries = [];
        const stored = await api.submitRating(body, {
          sleep: async (ms) => { sleeps.push(ms); kept.push(api.unsentRating()); },
          onRetry: (attempt, failure) =>
            retries.push([attempt, failure.name, failure.status ?? null]),
        });
        const afterStored = api.unsentRating();
        const noMark = { errors: [{ field: "marks", message: "no mark of class A" }] };
        answers = [{ status: 422, body: noMark }];
        let refused = null;
        try { await api.submitRating(body, { sleep }); } catch (failure) {
          const { name, status, errors, message } = failure;
          refused = { name, status, errors, message };
        }
        const afterRefused = api.unsentRating();
        output({ stored, calls, kept, retries, sleeps, afterStored, refused, afterRefused });
        """,
    )
    assert out["stored"] == {"saved": 1}
    assert [call[:2] for call in out["calls"]] == [["POST", "/api/rating"]] * 4
    body = {"assignment_id": 12, "protocol_version": "v", "ms_on_subject": 1}
    assert json.loads(out["calls"][0][2]) == body
    # while the server is away the rating is where a reloaded page finds it
    assert out["kept"] == [body, body] and out["afterStored"] is None
    assert out["retries"] == [[1, "NetworkError", None], [2, "ApiError", 503]]
    assert out["sleeps"] == [1000, 2000]
    # a rating the server refuses is not sent again
    assert out["refused"] == {
        "name": "ApiError",
        "status": 422,
        "errors": [{"field": "marks", "message": "no mark of class A"}],
        "message": "no mark of class A",
    }
    assert out["afterRefused"] is None


@needs_node
def test_a_skip_is_sent_again_with_the_same_body_until_the_server_answers(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        API_STUB
        + """
        import * as api from "./js/api.js";
        const body = { assignment_id: 12, client_ts: "2026-11-12T14:03:22.120Z", ms_on_subject: 7 };
        const busy = { errors: [{ field: "request", message: "busy" }] };
        const following = { status: "done", progress: {} };
        answers = ["down", { status: 503, body: busy }, { status: 200, body: following }];
        const retries = [];
        const next = await api.skipAssignment(body, {
          sleep,
          onRetry: (attempt, failure) =>
            retries.push([attempt, failure.name, failure.status ?? null]),
        });
        const stale = { errors: [{ field: "assignment_id", message: "not open" }] };
        answers = [{ status: 409, body: stale }];
        let refused = null;
        try { await api.skipAssignment(body, { sleep }); } catch (failure) {
          refused = { name: failure.name, status: failure.status, message: failure.message };
        }
        output({ next, calls, retries, sleeps, refused, unsent: api.unsentRating() });
        """,
    )
    assert out["next"] == {"status": "done", "progress": {}}
    assert [call[:2] for call in out["calls"]] == [["POST", "/api/skip"]] * 4
    # the same body every time: its client_ts is how the server knows the skip again
    bodies = [json.loads(call[2]) for call in out["calls"]]
    assert bodies[0] == bodies[1] == bodies[2]
    assert schemas.SkipIn.model_validate(bodies[0]).client_ts == "2026-11-12T14:03:22.120Z"
    assert out["retries"] == [[1, "NetworkError", None], [2, "ApiError", 503]]
    assert out["sleeps"] == [1000, 2000]
    # a skip the server refuses is not sent again
    assert out["refused"] == {"name": "ApiError", "status": 409, "message": "not open"}
    # and a skip is no rating: nothing is kept for a reloaded page to send
    assert out["unsent"] is None


@needs_node
def test_next_is_asked_again_while_nothing_is_rendered(js_copy: Path) -> None:
    rendering = {"status": "rendering", "retry_after_s": 2, "progress": {"rendered": 0}}
    out = run_js(
        js_copy,
        API_STUB
        + """
        import * as api from "./js/api.js";
        answers = [{ status: 200, body: input }, "down", { status: 200, body: { status: "done" } }];
        const waits = [];
        const onRendering = (answer) => waits.push(answer.status);
        const next = await api.settle(input, { sleep, onRendering });
        answers = [{ status: 500, body: null }];
        let failed = null;
        try { await api.getSession(); } catch (failure) {
          failed = [failure.name, failure.status, failure.message];
        }
        answers = [{ status: 200, body: { total: 0 } }, { status: 200, body: {} }];
        const filter = "final_A:2,fs_version:7.3.2";
        await api.getReview({ sort: null, order: "asc", filter, offset: 0, limit: 200 });
        await api.getSubject("0123456789abcdef");
        output({ next, waits, sleeps, calls, failed });
        """,
        rendering,
    )
    assert out["next"] == {"status": "done"}
    # asked again after retry_after_s each time, also when the server did not answer
    assert out["sleeps"] == [2000, 2000, 2000] and out["waits"] == ["rendering"] * 3
    assert [call[:2] for call in out["calls"][:3]] == [["GET", "/api/next"]] * 3
    assert out["failed"] == ["ApiError", 500, "the server answered 500"]
    assert out["calls"][4][1] == (
        "/api/review?order=asc&filter=final_A%3A2%2Cfs_version%3A7.3.2&offset=0&limit=200"
    )
    assert out["calls"][5][:2] == ["GET", "/api/review/0123456789abcdef"]
    query = schemas.ReviewQuery(
        order="asc", filter="final_A:2,fs_version:7.3.2", offset=0, limit=200
    )
    assert query.sort is None


@needs_node
def test_the_review_page_helpers(js_copy: Path) -> None:
    _, render, _ = geometry_case("conformed")
    out = run_js(
        js_copy,
        """
        import { cellText, finalLevel, frameShowing } from "./js/review.js";
        const frames = input.modes.sync.frames;
        output({
          text: [null, 3, 0.123456, true, "x", 2.0].map(cellText),
          levels: input.rows.map(finalLevel),
          own: frameShowing(frames, [30, 10, 30], ["coronal", 50]),
          free: frameShowing(frames, frames[17]),
          none: frameShowing(frames, [3, 3, 3], ["coronal", 999]),
        });
        """,
        {
            **render,
            "rows": [
                {"final_A": 0, "final_B": 2, "unratable_reason": None},
                {"final_A": None, "final_B": None, "unratable_reason": None},
                {"final_A": None, "final_B": None, "unratable_reason": "missing surface"},
                {"final_A": 1, "unratable_reason": None},
            ],
        },
    )
    assert out["text"] == ["", "3", "0.1235", "yes", "x", "2"]
    assert out["levels"] == [2, "not rated", "unratable", 1]
    frames = render["modes"]["sync"]["frames"]
    assert frames[out["own"]][1] == 50
    assert out["free"] == 17 and frames[out["none"]] == [3, 3, 3]


# --- the tour ---------------------------------------------------------------------------------

TOUR_STEPS = ("views", "rating", "marks", "moving", "help")


def tour_of(js_copy: Path, protocol: Protocol) -> list[dict[str, Any]]:
    return run_js(
        js_copy,
        """
        import { buildKeyMap } from "./js/keys.js";
        import { tourSteps } from "./js/tour.js";
        output(tourSteps(input, buildKeyMap(input)));
        """,
        protocol.model_dump(mode="json"),
    )


def keys_named(step: dict[str, Any]) -> list[str]:
    return [
        key
        for paragraph in step["text"]
        for part in paragraph
        if isinstance(part, dict)
        for key in part["keys"]
    ]


def words_of(step: dict[str, Any]) -> str:
    return " ".join(
        part if isinstance(part, str) else " ".join(part["keys"])
        for paragraph in step["text"]
        for part in paragraph
    )


@needs_node
def test_the_tour_has_five_short_steps_each_about_a_part_of_the_rating_page(
    js_copy: Path, shipped_protocol: Protocol, pages: dict[str, str]
) -> None:
    steps = tour_of(js_copy, shipped_protocol)
    assert tuple(step["id"] for step in steps) == TOUR_STEPS
    assert len(steps) <= 5
    present = set(re.findall(r'\bid="([\w-]+)"', pages["/"]))
    for step in steps:
        assert step["title"] and step["targets"] and 1 <= len(step["text"]) <= 3
        for selector in step["targets"]:
            # every part it points at is of the rating page or built into it by a module
            named = re.findall(r"#([\w-]+)", selector)
            assert named and set(named) <= present, selector
        text = words_of(step)
        assert len(text) <= 330, (step["id"], len(text))
        for sentence in re.split(r"(?<=[.:;]) ", text):
            assert len(sentence.split()) <= 22, sentence
    # what each step says, in the words a rater needs
    by_id = {step["id"]: step for step in steps}
    assert keys_named(by_id["views"]) == ["Space", "←", "→", "s"]
    assert "none, minor or major" in words_of(by_id["rating"])
    assert keys_named(by_id["rating"]) == [str(n) for n in range(1, 10)]
    assert keys_named(by_id["marks"]) == ["a", "b", "c", "x"]
    assert "minor or major" in words_of(by_id["marks"])
    moving = words_of(by_id["moving"])
    assert keys_named(by_id["moving"]) == ["Enter", "u", "k", "Backspace"]
    assert "every row has a value" in moving and "next subject" in moving
    assert "nothing is stored" in moving and "comes back later" in moving
    assert keys_named(by_id["help"]) == ["?", "t"]
    assert "this tour" in words_of(by_id["help"])


@needs_node
def test_the_tour_names_the_keys_of_the_protocol_of_the_session(
    js_copy: Path, shipped_protocol: Protocol
) -> None:
    content = shipped_protocol.model_dump(mode="json")
    for criterion, keys, mark_key in zip(
        content["criteria"], (["q", "w", "e"], ["r", "y", "i"], ["f", "g", "j"]), "abd", strict=True
    ):
        criterion["keys"] = keys
        criterion["mark_key"] = mark_key
    content["other_mark_key"] = "p"
    content["scale"][1]["label"] = "slight"
    content["scale"][2]["label"] = "gross"
    other = Protocol.model_validate(content)
    by_id = {step["id"]: step for step in tour_of(js_copy, other)}
    assert keys_named(by_id["rating"]) == ["q", "w", "e", "r", "y", "i", "f", "g", "j"]
    assert keys_named(by_id["marks"]) == ["a", "b", "d", "p"]
    assert "none, slight or gross" in words_of(by_id["rating"])
    assert "slight or gross" in words_of(by_id["marks"])
    everything = " ".join(words_of(step) for step in by_id.values())
    for key in "123456789cx":
        assert not re.search(rf"(?<![\w]){key}(?![\w])", everything), key
    # the fixed keys are those of the keyboard map
    assert keys_named(by_id["moving"]) == ["Enter", "u", "k", "Backspace"]


def test_the_tour_writes_no_key_into_its_text() -> None:
    # every key it shows comes from the key map: the source has no <kbd> text of a rating key
    code = js_sources()["tour.js"]
    steps = code[code.index("export function tourSteps") : code.index("function html")]
    assert "keysFor(map" in steps
    assert not re.search(r'keys:\s*\[\s*"', steps), "a key written out in the steps"
    for literal in re.findall(r'"([^"]*)"', blank_free(steps)):
        assert not re.fullmatch(r"[a-z0-9?]|Enter|Backspace|Space|Esc", literal), literal


def blank_free(code: str) -> str:
    """``code`` without its comments, with the strings as they are."""
    return re.sub(r"^\s*//.*$", "", code, flags=re.M)


@needs_node
def test_the_card_of_the_tour_lies_beside_its_part_and_inside_the_window(js_copy: Path) -> None:
    card = {"width": 380, "height": 200}
    wide = {"width": 1400, "height": 900}
    narrow = {"width": 400, "height": 700}
    cases = {
        # under the part when there is room
        "under": [{"left": 10, "top": 10, "right": 1390, "bottom": 470}, card, wide],
        # over it when the part is at the bottom
        "over": [{"left": 1100, "top": 770, "right": 1380, "bottom": 800}, card, wide],
        # no room either way: at the far end of the window from the part
        "top_part": [{"left": 0, "top": 0, "right": 400, "bottom": 600}, card, narrow],
        "bottom_part": [{"left": 0, "top": 150, "right": 400, "bottom": 700}, card, narrow],
        # a part at the right edge: the card stays inside the window
        "right": [{"left": 1300, "top": 100, "right": 1395, "bottom": 130}, card, wide],
        "left": [{"left": 2, "top": 100, "right": 40, "bottom": 130}, card, wide],
        # a window smaller than the card
        "tiny": [
            {"left": 0, "top": 0, "right": 300, "bottom": 50},
            card,
            {"width": 300, "height": 150},
        ],
        # no part on the page: in the middle
        "none": [None, card, wide],
    }
    out = run_js(
        js_copy,
        """
        import { placeCard, unionBox } from "./js/tour.js";
        const placed = {};
        for (const [name, [spot, card, view]] of Object.entries(input)) {
          placed[name] = placeCard(spot, card, view);
        }
        const box = (left, top, width, height) =>
          ({ left, top, right: left + width, bottom: top + height, width, height });
        output({
          placed,
          union: unionBox([box(10, 20, 30, 40), box(100, 5, 50, 10), box(0, 0, 0, 0)]),
          nothing: unionBox([box(5, 5, 0, 0)]),
          empty: unionBox([]),
        });
        """,
        cases,
    )
    placed = out["placed"]
    assert placed["under"] == {"left": 510, "top": 480}
    assert placed["over"]["top"] == 770 - 10 - 200
    assert placed["top_part"]["top"] == 700 - 8 - 200  # the part is at the top: card at the bottom
    assert placed["bottom_part"]["top"] == 8
    assert placed["right"]["left"] == 1400 - 8 - 380 and placed["left"]["left"] == 8
    assert placed["none"] == {"left": 510, "top": 350}
    for name, (_, size, view) in cases.items():
        spot = placed[name]
        assert spot["left"] >= 8 and spot["top"] >= 8, name
        if name != "tiny":
            assert spot["left"] + size["width"] <= view["width"] - 8, name
            assert spot["top"] + size["height"] <= view["height"] - 8, name
    assert placed["tiny"] == {"left": 8, "top": 8}
    assert out["union"] == {
        "left": 10,
        "top": 5,
        "right": 150,
        "bottom": 60,
        "width": 140,
        "height": 55,
    }
    assert out["nothing"] is None and out["empty"] is None


@needs_node
def test_a_page_that_scrolls_is_scrolled_so_that_the_part_and_the_card_fit(js_copy: Path) -> None:
    # (spot top, bottom), card height, window height, (scrolled to, can scroll to) -> scroll by
    cases = [
        # on screen with room under it: nothing
        ((8, 132), 250, 800, (0, 300), 0),
        # on screen with room over it only: nothing
        ((500, 590), 250, 800, (0, 300), 0),
        # under the window: to the top of the window, the card under it
        ((900, 970), 250, 800, (0, 2000), 900 - 12),
        # the page cannot scroll that far, but far enough for the card over it
        ((900, 970), 250, 800, (0, 300), 970 - (800 - 12)),
        # above the window (the rater scrolled down): back up
        ((-300, -210), 250, 800, (400, 900), -312),
        # on screen, but the card fits neither under nor over it: to the top
        ((180, 250), 240, 400, (0, 500), 168),
        # the same at the end of the page: to the bottom, the card over it
        ((180, 250), 240, 400, (220, 220), 250 - 388),
        # nowhere to scroll: nothing
        ((180, 250), 240, 400, (0, 0), 0),
    ]
    out = run_js(
        js_copy,
        """
        import { scrollFor } from "./js/tour.js";
        output(input.map(([[top, bottom], card, view, [at, max]]) =>
          scrollFor({ top, bottom }, card, view, { at, max })));
        """,
        [list(case[:4]) for case in cases],
    )
    assert out == [case[4] for case in cases]
    # after the scroll the part is in the window with room for the card, in every case but
    # the last, where the page cannot move
    for ((top, bottom), card, view, (at, most), by), got in zip(cases[:-1], out, strict=False):
        assert 0 <= at + got <= most
        top, bottom = top - by, bottom - by
        assert 0 <= top and bottom <= view
        assert bottom + 10 + card <= view - 8 or top - 10 - card >= 8, (top, bottom)


@needs_node
def test_the_tour_is_remembered_per_rater_and_opens_again_without_storage(js_copy: Path) -> None:
    out = run_js(
        js_copy,
        """
        import { markTourSeen, tourSeen } from "./js/tour.js";
        const result = {};
        // no storage at all (Node has none): never seen, and marking it does not throw
        result.none = [tourSeen("local"), (markTourSeen("local"), tourSeen("local"))];
        const store = new Map();
        globalThis.localStorage = {
          getItem: (key) => (store.has(key) ? store.get(key) : null),
          setItem: (key, value) => store.set(key, String(value)),
        };
        result.before = tourSeen("local");
        markTourSeen("local");
        result.after = [tourSeen("local"), tourSeen("second rater")];
        result.keys = [...store.keys()];
        // a storage that refuses every access, as a browser with storage blocked does
        const refuse = () => { throw new Error("denied"); };
        globalThis.localStorage = { getItem: refuse, setItem: refuse };
        markTourSeen("local");
        result.blocked = tourSeen("local");
        Object.defineProperty(globalThis, "localStorage", { get: refuse, configurable: true });
        markTourSeen("local");
        result.absent = tourSeen("local");
        output(result);
        """,
    )
    assert out["none"] == [False, False]
    assert out["before"] is False and out["after"] == [True, False]
    assert out["keys"] == ["surfcensus.tour.seen:local"]
    assert out["blocked"] is False and out["absent"] is False


@needs_node
def test_a_key_is_written_as_the_screen_shows_it(js_copy: Path, shipped_protocol: Protocol) -> None:
    out = run_js(
        js_copy,
        """
        import { buildKeyMap, keyLabel, keysFor } from "./js/keys.js";
        const map = buildKeyMap(input);
        output({
          labels: [" ", "ArrowLeft", "ArrowRight", "Escape", "Enter", "k"].map(keyLabel),
          play: keysFor(map, (action) => action.type === "toggle_play"),
          values: keysFor(map, (action) => action.type === "value" && action.criterion === "B"),
          none: keysFor(map, (action) => action.type === "nothing"),
        });
        """,
        shipped_protocol.model_dump(mode="json"),
    )
    assert out == {
        "labels": ["Space", "←", "→", "Esc", "Enter", "k"],
        "play": ["Space"],
        "values": ["4", "5", "6"],
        "none": [],
    }
