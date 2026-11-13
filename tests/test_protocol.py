"""The rating protocol (``surfcensus/protocol.py``): the shipped file, each rule of its
validation, and how a run reads it."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import surfcensus
from surfcensus.config import AppConfig, ConfigError, new_census_config, new_local_config
from surfcensus.privacy import HashKey
from surfcensus.protocol import (
    CRITERION_IDS,
    FIXED_KEYS,
    MARK_CLASSES,
    OTHER_MARK_CLASS,
    PROTOCOL_FILE,
    RENDER_REASONS,
    Protocol,
    install_protocol,
    load_protocol,
    marks_required,
    parse_protocol,
    render_reason,
    shipped_protocol_bytes,
)

PACKAGE_DIR = Path(surfcensus.__file__).parent


@pytest.fixture
def content() -> dict[str, Any]:
    """The shipped protocol as YAML data, for a test to change one entry of."""
    return yaml.safe_load(shipped_protocol_bytes())


def refused(content: dict[str, Any]) -> str:
    with pytest.raises(ConfigError) as caught:
        parse_protocol(yaml.safe_dump(content))
    assert caught.value.reason == "invalid"
    assert caught.value.detail.startswith(f"{PROTOCOL_FILE}: ")
    return caught.value.detail


# --- the shipped file -----------------------------------------------------------------------


def test_the_shipped_protocol_validates() -> None:
    protocol = parse_protocol(shipped_protocol_bytes())
    assert protocol.version == "0.1-draft"
    assert [(level.value, level.label) for level in protocol.scale] == [
        (0, "none"),
        (1, "minor"),
        (2, "major"),
    ]
    assert protocol.criterion_ids() == CRITERION_IDS == ("A", "B", "C")
    assert [criterion.keys for criterion in protocol.criteria] == [
        ("1", "2", "3"),
        ("4", "5", "6"),
        ("7", "8", "9"),
    ]
    assert [criterion.mark_key for criterion in protocol.criteria] == ["a", "b", "c"]
    assert protocol.other_mark_key == "x"
    assert protocol.require_marks_for_positive is True
    assert protocol.unratable_reasons == (
        "render failure",
        "wrong orientation",
        "missing surface",
        "image too poor to judge",
    )


def test_what_a_protocol_answers() -> None:
    protocol = parse_protocol(shipped_protocol_bytes())
    assert protocol.values() == (0, 1, 2)
    assert protocol.positive_values() == frozenset({1, 2})
    assert protocol.label(2) == "major"
    with pytest.raises(KeyError):
        protocol.label(3)
    assert protocol.unratable_index("missing surface") == 2
    with pytest.raises(KeyError):
        protocol.unratable_index("no reason at all")
    with pytest.raises(ValidationError, match="frozen"):
        protocol.version = "2"  # pyrefly: ignore[read-only]


def test_the_classes_of_marks_are_the_criteria_and_x() -> None:
    assert MARK_CLASSES == (*CRITERION_IDS, OTHER_MARK_CLASS) == ("A", "B", "C", "X")


def test_the_keyboard_map_of_the_docs_is_fixed() -> None:
    # docs/rater_app.md, "Keyboard map": every key there that is not a criterion or mark key
    assert set(FIXED_KEYS) == {
        " ",
        "ArrowLeft",
        "ArrowRight",
        "[",
        "]",
        "m",
        "z",
        "Delete",
        "h",
        "s",
        "u",
        "o",
        "n",
        "Enter",
        "k",
        "Backspace",
        "?",
        "t",
    }
    assert len(set(FIXED_KEYS)) == len(FIXED_KEYS)


# --- the reasons of the renderer ------------------------------------------------------------


def _render_error_reasons() -> set[str]:
    """The first argument of every ``RenderError(...)`` in the render package."""
    reasons = set()
    for path in sorted((PACKAGE_DIR / "render").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RenderError"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                reasons.add(node.args[0].value)
    return reasons


def test_the_render_reasons_are_those_the_renderer_raises() -> None:
    assert _render_error_reasons() == set(RENDER_REASONS)
    protocol = parse_protocol(shipped_protocol_bytes())
    assert set(RENDER_REASONS) <= set(protocol.unratable_reasons)


def test_a_render_that_failed_otherwise_counts_as_a_render_failure() -> None:
    for reason in RENDER_REASONS:
        assert render_reason(reason) == reason
    assert render_reason("source changed") == "render failure"
    assert render_reason(None) == "render failure"


# --- each rule, violated once ---------------------------------------------------------------


def test_scale_values_run_from_zero_in_order(content: dict[str, Any]) -> None:
    for values in ([1, 2, 3], [0, 2, 1], [0, 1, 3], [0, 0, 1]):
        changed = copy.deepcopy(content)
        for level, value in zip(changed["scale"], values, strict=True):
            level["value"] = value
        assert "scale: the values must be 0 to n - 1 in order" in refused(changed)
    content["scale"] = content["scale"][:1]
    assert "at least two levels" in refused(content)


def test_a_criterion_has_one_key_per_scale_level(content: dict[str, Any]) -> None:
    content["criteria"][1]["keys"] = ["4", "5"]
    assert "criteria.B.keys: 2 keys for 3 scale levels" in refused(content)


def test_a_fourth_level_needs_a_fourth_key(content: dict[str, Any]) -> None:
    content["scale"].append({"value": 3, "label": "severe", "help": ""})
    assert "criteria.A.keys: 3 keys for 4 scale levels" in refused(content)
    for criterion, key in zip(content["criteria"], "qwe", strict=True):
        criterion["keys"].append(key)
    protocol = parse_protocol(yaml.safe_dump(content))
    assert protocol.positive_values() == frozenset({1, 2, 3})


@pytest.mark.parametrize(
    ("where", "change", "message"),
    [
        ("criteria.B.keys", lambda c: c["criteria"][1]["keys"].__setitem__(0, "1"), "already used"),
        ("criteria.C.mark_key", lambda c: c["criteria"][2].__setitem__("mark_key", "a"), "already"),
        ("other_mark_key", lambda c: c.__setitem__("other_mark_key", "b"), "already used by"),
        ("other_mark_key", lambda c: c.__setitem__("other_mark_key", "3"), "criteria.A.keys"),
        # the same letter in the other case is the same key
        ("other_mark_key", lambda c: c.__setitem__("other_mark_key", "A"), "already used"),
    ],
)
def test_keys_are_unique_across_the_file(
    content: dict[str, Any], where: str, change: Any, message: str
) -> None:
    change(content)
    detail = refused(content)
    assert where in detail and message in detail


@pytest.mark.parametrize(
    "key", ["m", "z", "h", "s", "u", "o", "n", "?", "[", "]", "M", "S", "k", "K", "t", "T"]
)
def test_a_key_of_the_fixed_keyboard_map_is_refused(content: dict[str, Any], key: str) -> None:
    content["criteria"][0]["mark_key"] = key
    detail = refused(content)
    assert "criteria.A.mark_key" in detail and "fixed meaning" in detail


@pytest.mark.parametrize("key", ["", " ", "ab", "Enter", "\t", "\x07"])
def test_a_key_is_one_visible_character(content: dict[str, Any], key: str) -> None:
    content["other_mark_key"] = key
    assert "other_mark_key: a key is one visible character" in refused(content)


def test_criterion_ids_are_unique_and_known(content: dict[str, Any]) -> None:
    changed = copy.deepcopy(content)
    changed["criteria"][2]["id"] = "A"
    assert "criteria: the id A is used more than once" in refused(changed)
    content["criteria"][2]["id"] = "D"
    assert "criteria.2.id" in refused(content)


def test_a_protocol_needs_a_criterion(content: dict[str, Any]) -> None:
    content["criteria"] = []
    assert "criteria: at least one criterion is needed" in refused(content)


@pytest.mark.parametrize("reason", RENDER_REASONS)
def test_the_reasons_of_the_renderer_must_be_listed(content: dict[str, Any], reason: str) -> None:
    content["unratable_reasons"].remove(reason)
    detail = refused(content)
    assert "unratable_reasons" in detail and repr(reason) in detail


def test_unratable_reasons_are_unique_and_not_empty(content: dict[str, Any]) -> None:
    changed = copy.deepcopy(content)
    changed["unratable_reasons"].append("render failure")
    assert "unratable_reasons: a reason is listed more than once" in refused(changed)
    content["unratable_reasons"].append("  ")
    assert "unratable_reasons: an empty reason" in refused(content)


@pytest.mark.parametrize(
    ("change", "names"),
    [
        (lambda c: c.pop("version"), "version"),
        (lambda c: c.__setitem__("version", ""), "version"),
        (lambda c: c.__setitem__("version", 0.1), "version"),
        (lambda c: c.pop("require_marks_for_positive"), "require_marks_for_positive"),
        (lambda c: c.__setitem__("colour", "red"), "colour: unknown key"),
        (lambda c: c["scale"][0].__setitem__("label", ""), "scale.0.label"),
        (lambda c: c["criteria"][0].__setitem__("keys", [1, 2, 3]), "criteria.0.keys.0"),
        (lambda c: c["criteria"][0].pop("mark_key"), "criteria.0.mark_key"),
    ],
)
def test_a_field_that_is_missing_or_of_another_type_is_named(
    content: dict[str, Any], change: Any, names: str
) -> None:
    change(content)
    assert names in refused(content)


def test_text_that_is_no_protocol_is_refused() -> None:
    for text in ("- a\n- b\n", "just text", "", "version: [unclosed"):
        with pytest.raises(ConfigError, match=PROTOCOL_FILE) as caught:
            parse_protocol(text)
        assert caught.value.reason == "invalid"
    with pytest.raises(ConfigError, match="not YAML"):
        parse_protocol(b"\xff\xfe\x00")


# --- in a run directory ---------------------------------------------------------------------


def test_install_copies_the_shipped_file_once_and_never_replaces_one(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=f"missing: .* has no {PROTOCOL_FILE}") as caught:
        load_protocol(tmp_path)
    assert caught.value.reason == "missing"
    path = install_protocol(tmp_path)
    assert path == tmp_path / PROTOCOL_FILE
    assert path.read_bytes() == shipped_protocol_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == [PROTOCOL_FILE]  # no temporary file
    protocol, digest = load_protocol(tmp_path)
    assert protocol == parse_protocol(shipped_protocol_bytes())
    assert digest == hashlib.sha256(shipped_protocol_bytes()).hexdigest()

    edited = shipped_protocol_bytes().replace(b'"0.1-draft"', b'"0.2-pilot"')
    path.write_bytes(edited)
    before = path.stat().st_mtime_ns
    assert install_protocol(tmp_path) == path
    assert path.read_bytes() == edited and path.stat().st_mtime_ns == before
    protocol, digest = load_protocol(tmp_path)
    assert protocol.version == "0.2-pilot"
    assert digest == hashlib.sha256(edited).hexdigest()


def test_the_digest_is_of_the_bytes_and_not_of_the_content(tmp_path: Path) -> None:
    install_protocol(tmp_path)
    _, digest = load_protocol(tmp_path)
    path = tmp_path / PROTOCOL_FILE
    path.write_bytes(path.read_bytes() + b"# a comment\n")
    protocol, commented = load_protocol(tmp_path)
    assert commented != digest and protocol == parse_protocol(shipped_protocol_bytes())


def test_a_damaged_file_in_a_run_is_invalid_and_named(tmp_path: Path) -> None:
    (tmp_path / PROTOCOL_FILE).write_text("version: 1\n")
    with pytest.raises(ConfigError, match=rf"invalid: {PROTOCOL_FILE}: version") as caught:
        load_protocol(tmp_path)
    assert str(tmp_path) not in caught.value.detail


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_file_that_may_not_be_read_is_invalid_and_not_missing(tmp_path: Path) -> None:
    path = install_protocol(tmp_path)
    path.chmod(0)
    try:
        with pytest.raises(ConfigError, match="cannot be read") as caught:
            load_protocol(tmp_path)
        assert caught.value.reason == "invalid" and str(tmp_path) not in caught.value.detail
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_a_directory_under_the_name_is_invalid(tmp_path: Path) -> None:
    (tmp_path / PROTOCOL_FILE).mkdir()
    with pytest.raises(ConfigError, match="cannot be read") as caught:
        load_protocol(tmp_path)
    assert caught.value.reason == "invalid"


# --- whether marks are required -------------------------------------------------------------


def test_the_setting_of_the_run_decides_and_else_the_protocol(content: dict[str, Any]) -> None:
    requires = parse_protocol(shipped_protocol_bytes())
    content["require_marks_for_positive"] = False
    lenient = parse_protocol(yaml.safe_dump(content))
    undecided = AppConfig(queue_seed=1)
    assert undecided.require_marks is None
    assert marks_required(requires, undecided) is True
    assert marks_required(lenient, undecided) is False
    for protocol in (requires, lenient):
        assert marks_required(protocol, AppConfig(queue_seed=1, require_marks=False)) is False
        assert marks_required(protocol, AppConfig(queue_seed=1, require_marks=True)) is True


def test_a_new_local_run_does_not_require_marks_and_a_census_run_follows_the_protocol(
    tmp_path: Path,
) -> None:
    shipped = parse_protocol(shipped_protocol_bytes())
    local = new_local_config(tmp_path)
    assert local.app.require_marks is False and marks_required(shipped, local.app) is False
    # also when the app section is given in part
    partial = new_local_config(tmp_path, app={"blind": True})
    assert partial.app.require_marks is False and partial.app.blind is True
    asked = new_local_config(tmp_path, app={"require_marks": True})
    assert marks_required(shipped, asked.app) is True
    census = new_census_config(HashKey(bytes(range(32)), "dataset-key"))
    assert census.app.require_marks is None and marks_required(shipped, census.app) is True
    partial_census = new_census_config(
        HashKey(bytes(range(32)), "dataset-key"), app={"blind": True}
    )
    assert partial_census.app.raters_per_subject == 2 and partial_census.app.repeat_rate == 0.05


def test_the_protocol_goes_to_a_page_as_it_is() -> None:
    protocol = parse_protocol(shipped_protocol_bytes())
    assert Protocol.model_validate(protocol.model_dump(mode="json")) == protocol
    assert set(protocol.model_dump()) == {
        "version",
        "scale",
        "criteria",
        "other_mark_key",
        "require_marks_for_positive",
        "unratable_reasons",
    }
