"""The calibration set of a run (``surfcensus/calibration.py``): the models, the file, and
the check against the protocol and the run."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from surfcensus.calibration import (
    CALIBRATION_FILE,
    CalibrationExample,
    CalibrationMark,
    CalibrationSet,
    CalibrationSubject,
    dump_calibration,
    load_calibration,
    parse_calibration,
    validate_calibration,
    write_calibration,
)
from surfcensus.config import ConfigError
from surfcensus.protocol import Protocol, parse_protocol, shipped_protocol_bytes
from surfcensus.render.model import RenderInfo
from surfcensus.store import PublicCatalog, media_dir

UID_A = "3f9a1c07d2e4b8a1"
UID_B = "9b2e41c07d2e4b8a"


def subject(uid: str = UID_A, **fields: Any) -> CalibrationSubject:
    values = {"subject_uid": uid, "values": {"A": 2, "B": 0, "C": 0}} | fields
    return CalibrationSubject.model_validate(values)


def calibration(*subjects: CalibrationSubject, version: str = "1") -> CalibrationSet:
    return CalibrationSet(version=version, subjects=subjects or (subject(),))


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    return parse_protocol(shipped_protocol_bytes())


FULL = calibration(
    subject(
        UID_A,
        marks=[{"class": "A", "tkr": (-5.0, -66.0, 11.5)}, {"class": "X", "tkr": (1.0, 2.0, 3.0)}],
        justification="The pial surface encloses the sinus over twelve slices; no é lost.",
        examples=[
            {"criterion": "A", "value": 2, "panel": "sagittal", "slice": 3, "caption": "enclosed"}
        ],
    ),
    subject(UID_B, values={}, unratable="image too poor to judge"),
    version="pilot-1",
)


# --- the models -----------------------------------------------------------------------------


def test_a_mark_is_written_under_class_and_read_under_mark_class() -> None:
    mark = CalibrationMark.model_validate({"class": "B", "tkr": [1, 2, 3]})
    assert mark.mark_class == "B" and mark.tkr == (1.0, 2.0, 3.0)
    assert CalibrationMark(mark_class="B", tkr=(1, 2, 3)) == mark
    assert mark.model_dump(by_alias=True) == {"class": "B", "tkr": (1.0, 2.0, 3.0)}
    with pytest.raises(ValidationError, match="class"):
        CalibrationMark.model_validate({"class": "D", "tkr": [1, 2, 3]})
    with pytest.raises(ValidationError, match="tkr"):
        CalibrationMark.model_validate({"class": "A", "tkr": [1, 2]})


def test_a_subject_is_named_by_its_uid() -> None:
    assert subject().justification == "" and subject().examples == () and subject().marks == ()
    for name in ("sub-01", "3F9A1C07D2E4B8A1", UID_A[:-1], UID_A + "0", UID_A + "\n", ""):
        with pytest.raises(ValidationError, match="subject_uid"):
            subject(name)
    with pytest.raises(ValidationError, match="values"):
        subject(values={"D": 1})
    with pytest.raises(ValidationError, match="stray"):
        subject(stray=1)


def test_an_example_names_a_criterion_a_panel_and_a_slice() -> None:
    example = CalibrationExample(criterion="B", value=1, panel="coronal", slice=0)
    assert example.caption == ""
    with pytest.raises(ValidationError, match="slice"):
        CalibrationExample(criterion="B", value=1, panel="coronal", slice=-1)
    with pytest.raises(ValidationError, match="panel"):
        CalibrationExample.model_validate(
            {"criterion": "B", "value": 1, "panel": "oblique", "slice": 0}
        )
    with pytest.raises(ValidationError, match="criterion"):
        CalibrationExample.model_validate(
            {"criterion": "X", "value": 1, "panel": "axial", "slice": 0}
        )


def test_a_set_lists_a_subject_once_and_finds_it() -> None:
    with pytest.raises(ValidationError, match=f"{UID_A} is listed more than once"):
        calibration(subject(), subject())
    with pytest.raises(ValidationError, match="version"):
        calibration(version="")
    assert FULL.subject(UID_B) == FULL.subjects[1]
    assert FULL.subject("0" * 16) is None
    assert CalibrationSet(version="1", subjects=()).subjects == ()


# --- the file -------------------------------------------------------------------------------


def test_the_file_round_trips_and_reads_from_uid_to_examples() -> None:
    text = dump_calibration(FULL)
    assert parse_calibration(text) == FULL
    assert dump_calibration(parse_calibration(text)) == text
    data = yaml.safe_load(text)
    assert list(data) == ["version", "subjects"]
    assert list(data["subjects"][0]) == [
        "subject_uid",
        "values",
        "unratable",
        "marks",
        "justification",
        "examples",
    ]
    assert data["subjects"][0]["marks"][0] == {"class": "A", "tkr": [-5.0, -66.0, 11.5]}
    assert "é" in text  # written as it is, not escaped
    assert "tkr: [-5.0, -66.0, 11.5]" in text  # a position on one line


def test_write_then_load_gives_the_set_and_the_digest_of_the_bytes(tmp_path: Path) -> None:
    assert load_calibration(tmp_path) is None
    path = write_calibration(tmp_path, FULL)
    assert path == tmp_path / CALIBRATION_FILE
    assert sorted(p.name for p in tmp_path.iterdir()) == [CALIBRATION_FILE]  # no temporary file
    loaded = load_calibration(tmp_path)
    assert loaded is not None
    assert loaded[0] == FULL
    assert loaded[1] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_text(encoding="utf-8") == dump_calibration(FULL)


def test_an_existing_file_is_never_replaced(tmp_path: Path) -> None:
    path = write_calibration(tmp_path, FULL)
    by_hand = path.read_bytes() + b"# justification written by hand\n"
    path.write_bytes(by_hand)
    with pytest.raises(ConfigError, match=r"exists .* and is not overwritten") as caught:
        write_calibration(tmp_path, calibration())
    assert caught.value.reason == "invalid"
    assert path.read_bytes() == by_hand
    assert sorted(p.name for p in tmp_path.iterdir()) == [CALIBRATION_FILE]


def test_a_name_taken_while_writing_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # another process writes the file between the look and the link
    real_link = os.link

    def racing_link(src: Any, dst: Any) -> None:
        Path(dst).write_text("version: other\nsubjects: []\n")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(ConfigError, match="is not overwritten"):
        write_calibration(tmp_path, FULL)
    assert (tmp_path / CALIBRATION_FILE).read_text() == "version: other\nsubjects: []\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [CALIBRATION_FILE]


def test_writing_works_where_there_are_no_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_link(src: Any, dst: Any) -> None:
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "link", no_link)
    path = write_calibration(tmp_path, FULL)
    assert parse_calibration(path.read_bytes()) == FULL
    with pytest.raises(ConfigError, match="is not overwritten"):
        write_calibration(tmp_path, FULL)
    assert sorted(p.name for p in tmp_path.iterdir()) == [CALIBRATION_FILE]


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("- a\n", "not a mapping"),
        ("", "not a mapping"),
        ("version: [unclosed", "not YAML"),
        ("version: '1'\n", "subjects"),
        ("version: '1'\nsubjects: []\nextra: 1\n", "extra: unknown key"),
        (f"version: '1'\nsubjects:\n- subject_uid: {UID_A}\n  values: {{A: high}}\n", "values.A"),
        ("version: '1'\nsubjects:\n- subject_uid: sub-01\n", "subjects.0.subject_uid"),
    ],
)
def test_a_file_that_is_no_calibration_set_is_invalid_and_says_where(
    tmp_path: Path, text: str, says: str
) -> None:
    (tmp_path / CALIBRATION_FILE).write_text(text)
    with pytest.raises(ConfigError, match=CALIBRATION_FILE) as caught:
        load_calibration(tmp_path)
    assert caught.value.reason == "invalid" and says in caught.value.detail
    assert str(tmp_path) not in caught.value.detail


def test_a_directory_under_the_name_is_invalid(tmp_path: Path) -> None:
    (tmp_path / CALIBRATION_FILE).mkdir()
    with pytest.raises(ConfigError, match="cannot be read"):
        load_calibration(tmp_path)


# --- against the protocol and the run -------------------------------------------------------


def no_render(uid: str) -> RenderInfo | None:
    raise AssertionError(f"the render of {uid} was asked for")


def check(
    cal: CalibrationSet,
    protocol: Protocol,
    subjects: tuple[str, ...] = (UID_A, UID_B),
    render: Any = no_render,
) -> None:
    validate_calibration(cal, protocol, subjects=subjects, render=render)


def refused(cal: CalibrationSet, protocol: Protocol, **how: Any) -> str:
    with pytest.raises(ConfigError) as caught:
        check(cal, protocol, **how)
    assert caught.value.reason == "invalid"
    assert caught.value.detail.startswith(f"{CALIBRATION_FILE}: subject ")
    return caught.value.detail


def test_a_set_without_examples_needs_no_render(protocol: Protocol) -> None:
    check(
        calibration(subject(UID_A), subject(UID_B, values={}, unratable="render failure")), protocol
    )
    check(CalibrationSet(version="1", subjects=()), protocol, subjects=())


def test_every_subject_is_in_the_catalog(protocol: Protocol) -> None:
    detail = refused(calibration(subject(UID_A), subject(UID_B)), protocol, subjects=(UID_A,))
    assert detail == f"{CALIBRATION_FILE}: subject {UID_B}: is not a subject of this run"


def test_values_are_the_criteria_of_the_protocol_on_its_scale(protocol: Protocol) -> None:
    assert "values are needed for the criteria A, B, C, got A, B" in refused(
        calibration(subject(values={"A": 1, "B": 0})), protocol
    )
    assert "got none" in refused(calibration(subject(values={})), protocol)
    assert "the value 3 of B is not on the scale" in refused(
        calibration(subject(values={"A": 0, "B": 3, "C": 0})), protocol
    )
    assert "the value -1 of A" in refused(
        calibration(subject(values={"A": -1, "B": 0, "C": 0})), protocol
    )


def test_an_unratable_subject_names_a_reason_of_the_protocol(protocol: Protocol) -> None:
    check(calibration(subject(values={}, unratable="wrong orientation")), protocol)
    # values beside the reason are checked like any
    check(
        calibration(subject(values={"A": 0, "B": 0, "C": 0}, unratable="wrong orientation")),
        protocol,
    )
    assert "values are needed" in refused(
        calibration(subject(values={"A": 0}, unratable="wrong orientation")), protocol
    )
    assert "'too dark' is not an unratable reason of the protocol" in refused(
        calibration(subject(values={}, unratable="too dark")), protocol
    )


def test_a_protocol_with_fewer_criteria_has_fewer_values() -> None:
    content = yaml.safe_load(shipped_protocol_bytes())
    content["criteria"] = content["criteria"][:2]
    two = parse_protocol(yaml.safe_dump(content))
    check(calibration(subject(values={"A": 1, "B": 0})), two)
    assert "values are needed for the criteria A, B, got A, B, C" in refused(calibration(), two)
    example = {"criterion": "C", "value": 1, "panel": "axial", "slice": 0}
    with_example = calibration(subject(values={"A": 1, "B": 0}, examples=[example]))
    assert "an example of criterion C" in refused(with_example, two)


def test_examples_are_checked_against_the_render_of_their_subject(
    small_run: Path, protocol: Protocol
) -> None:
    with PublicCatalog.open(small_run) as public:
        uids = public.subject_uids()
    uid = uids[0]
    info = RenderInfo.model_validate_json(
        (media_dir(small_run, uid) / "render.json").read_text(encoding="utf-8")
    )
    asked: list[str] = []

    def render(subject_uid: str) -> RenderInfo | None:
        asked.append(subject_uid)
        return info if subject_uid == uid else None

    def with_example(target: str = uid, **fields: Any) -> CalibrationSet:
        example = {"criterion": "A", "value": 2, "panel": "coronal", "slice": 0} | fields
        return calibration(subject(target, examples=[example]), subject(uids[1]))

    n_coronal = len(info.panels["coronal"].slices)
    assert n_coronal == 6
    check(with_example(slice=n_coronal - 1), protocol, subjects=tuple(uids), render=render)
    assert asked == [uid]  # the subject without examples was not asked for
    detail = refused(with_example(slice=n_coronal), protocol, subjects=tuple(uids), render=render)
    assert "an example on slice 6 of the coronal panel, which has 6 slices" in detail
    assert "an example of the value 5, which is not on the scale" in refused(
        with_example(value=5), protocol, subjects=tuple(uids), render=render
    )
    other = calibration(
        subject(uids[1], examples=[{"criterion": "A", "value": 2, "panel": "axial", "slice": 0}])
    )
    assert "has examples and no render" in refused(
        other, protocol, subjects=tuple(uids), render=render
    )
