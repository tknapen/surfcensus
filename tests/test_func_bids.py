"""BIDS names of functional runs (``surfcensus.func.bids``): entities, run keys, the name
order, sidecar times and the listing of a raw dataset, on names written here and on the
synthetic folder of ``tests/synth_func.py`` in both generations."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    GENERATIONS,
    SyntheticFuncDataset,
    write_func_dataset,
)

from surfcensus.func import bids
from surfcensus.records import RunEntities
from surfcensus.sources.base import FuncRole


@pytest.fixture(scope="module", params=GENERATIONS)
def dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"), generation=request.param)


# --- names --------------------------------------------------------------------------------


def test_the_entities_of_a_name() -> None:
    e = bids.parse_entities("sub-01_ses-pre_task-rest_acq-mb4_dir-AP_run-01_echo-2_bold.nii.gz")
    assert e == RunEntities("01", "pre", "rest", "mb4", None, None, "AP", "01", "2")
    # a path is allowed; the last part is read
    assert bids.parse_entities("/data/sub-01/func/sub-01_task-rest_bold.nii.gz").task == "rest"
    # unknown entities are ignored, and the first of an entity named twice is kept
    e = bids.parse_entities("sub-01_task-rest_space-T1w_desc-preproc_run-2_task-other_bold")
    assert (e.task, e.run) == ("rest", "2")
    assert bids.parse_entities("sub-A1").sub == "A1"
    # the extension starts at the first dot
    assert bids.parse_entities("sub-01_task-re.st_bold").task == "re"


@pytest.mark.parametrize(
    "name",
    [
        "task-rest_bold.json",  # no sub
        "",
        "bold.nii.gz",
        "ses-1_sub-01_task-rest_bold",  # sub not first
        "sub-_task-rest_bold",  # empty label
        "sub-01_task-rest_run-a_bold",  # run is a number
        "sub-01_task-rest_run-\u0661_bold",  # an Arabic-Indic one is not an ASCII digit
        "sub-01_task-rest_echo-x_bold",
        "sub-01_task-rést_bold",  # labels are ASCII letters and digits
        "sub-01_task-rest_run-1234567890123456789_bold",  # at most 18 digits
    ],
)
def test_a_name_that_does_not_parse_is_no_run(name: str) -> None:
    with pytest.raises(ValueError):
        bids.parse_entities(name)


def test_run_numbers_are_compared_as_numbers() -> None:
    assert bids.run_index("01") == bids.run_index("1") == 1
    assert bids.run_index("0010") == 10
    for wrong in (None, "", "a", "1a", "\u0661", "-1", "1" * 19):
        assert bids.run_index(wrong) is None
    a = bids.parse_entities("sub-01_ses-1_task-rest_run-01_bold")
    b = bids.parse_entities("sub-01_ses-1_task-rest_run-1_desc-preproc_bold")
    assert bids.run_key(a) == bids.run_key(b) == "ses-1_task-rest_run-1"
    # 2 before 10, which a comparison of labels gets the other way round
    names = ["sub-01_task-rest_run-10", "sub-01_task-rest_run-2", "sub-01_task-rest_run-02"]
    keys = sorted(names, key=lambda n: bids.name_order(bids.parse_entities(n)))
    assert [bids.parse_entities(n).run for n in keys][-1] == "10"
    assert bids.name_order(bids.parse_entities(names[1])) == bids.name_order(
        bids.parse_entities(names[2])
    )


def test_the_run_key_has_no_subject_and_no_echo() -> None:
    one = bids.parse_entities("sub-01_task-me_acq-x_ce-y_rec-z_dir-PA_run-3_echo-1")
    two = bids.parse_entities("sub-02_task-me_acq-x_ce-y_rec-z_dir-PA_run-03_echo-2")
    assert bids.run_key(one) == bids.run_key(two) == "task-me_acq-x_ce-y_rec-z_dir-PA_run-3"
    assert bids.run_key(bids.parse_entities("sub-01_task-rest")) == "task-rest"
    # a label made by hand that is no number is kept as written
    assert bids.run_key(RunEntities("01", task="rest", run="x")) == "task-rest_run-x"


def test_the_name_order() -> None:
    def order(name: str) -> tuple[str | int, ...]:
        return bids.name_order(bids.parse_entities(name))

    assert order("sub-01_task-a_run-9") < order("sub-01_task-b_run-1")  # task first
    assert order("sub-01_task-a") < order("sub-01_task-a_run-1")  # absent run first
    assert order("sub-01_task-a_run-1") < order("sub-01_task-a_acq-x_run-1")  # absent acq first
    assert order("sub-01_task-a_run-1_echo-2") > order("sub-01_task-a_run-1_echo-1")
    assert order("sub-01_task-a_run-1") == (*("a", "", "", "", ""), 1, -1)


def test_the_name_of_a_run() -> None:
    e = bids.parse_entities("sub-01_ses-1_task-rest_run-01_bold.nii.gz")
    assert bids.run_name(e) == "ses-1_task-rest_run-01"
    assert bids.run_name(bids.parse_entities("sub-01_task-rest_echo-1")) == "task-rest_echo-1"


# --- acquisition times --------------------------------------------------------------------


def test_acquisition_times() -> None:
    assert bids.acquisition_seconds("17:30:2.685000") == ("time", 17 * 3600 + 30 * 60 + 2.685)
    assert bids.acquisition_seconds(" 9:5:3.5 ") == ("time", 9 * 3600 + 5 * 60 + 3.5)
    # without leading zeros, 9:5 is before 9:12, which text order gets wrong
    earlier = bids.acquisition_seconds("9:5:3.5")
    later = bids.acquisition_seconds("9:12:40.0")
    assert earlier is not None and later is not None and earlier[1] < later[1]
    stamp = bids.acquisition_seconds("2020-01-01T10:00:00")
    assert stamp == ("datetime", 1577872800.0)
    assert bids.acquisition_seconds("2020-01-01T11:00:00+01:00") == stamp
    for wrong in (
        None,
        3600,
        "",
        "noon",
        "24:00:00",
        "10:60:00",
        "10:00:61",
        "2020-01-01",  # a date alone orders nothing
        "10:00",
        "\u0661:00:00",
        "1" * 100,
        "2020-13-01T10:00:00",
    ):
        assert bids.acquisition_seconds(wrong) is None, wrong


def test_the_time_of_a_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "sub-01_task-rest_bold.json"
    path.write_text(json.dumps({"AcquisitionTime": "9:5:3.5"}))
    assert bids.sidecar_time(path) == "9:5:3.5"
    for text in (
        "{",
        "[1, 2]",
        json.dumps({"AcquisitionTime": 3}),
        json.dumps({"RepetitionTime": 2}),
        "[" * 100_000 + "]" * 100_000,  # deeper than the parser goes
        "/annex/objects/MD5E-s10--0123456789abcdef0123456789abcdef.json\n",
    ):
        path.write_text(text)
        assert bids.sidecar_time(path) is None
    path.write_bytes(b"\xff\xfe\x00")
    assert bids.sidecar_time(path) is None
    assert bids.sidecar_time(tmp_path / "absent.json") is None
    assert bids.sidecar_time(tmp_path) is None  # a folder


# --- the listing --------------------------------------------------------------------------


def test_every_raw_series_of_the_fixture_is_listed(dataset: SyntheticFuncDataset) -> None:
    root = dataset.raw_root
    assert bids.is_raw_dataset(root)
    assert not bids.is_raw_dataset(dataset.deriv_root)  # derivative names are no raw series
    assert not bids.is_raw_dataset(dataset.root / "absent")
    for participant in ("sub-01", "sub-02"):
        listed = bids.list_raw_runs(root, participant)
        expected = [r for r in dataset.runs if r.design.participant == participant and r.raw_stem]
        by_key: dict[str, list] = {}
        for run in listed:
            by_key.setdefault(run.key, []).append(run)
        assert sorted(by_key) == sorted(r.key for r in expected)
        for run in expected:
            found = by_key[run.key]
            if run.design.reason == "multi-echo":
                assert [f.entities.echo for f in found] == ["1", "2"]
                paths = [run.files["raw_bold_echo1"], run.files["raw_bold_echo2"]]
            else:
                assert len(found) == 1 and found[0].stem == run.raw_stem
                paths = [run.files["raw_bold"]]
            for item, path in zip(found, paths, strict=True):
                ref = item.files[FuncRole.RAW_BOLD]
                st = path.stat()
                assert (ref.base, ref.location) == ("raw", path.relative_to(root).as_posix())
                assert (ref.size, ref.etag_or_mtime) == (st.st_size, str(st.st_mtime_ns))
                assert item.order == bids.name_order(item.entities)
                assert item.entities.ses == run.design.session
        # the run without a raw series is not there, and nothing of the derivative is
        assert all(r.files[FuncRole.RAW_BOLD].location.startswith(participant) for r in listed)
        assert [r.order for r in listed if r.entities.ses is None] == sorted(
            r.order for r in listed if r.entities.ses is None
        )


def test_sidecars_beside_and_inherited(dataset: SyntheticFuncDataset) -> None:
    listed = {r.stem: r for r in bids.list_raw_runs(dataset.raw_root, "sub-02")}
    # beside its series
    beside = listed["sub-02_task-nback_run-1"].files[FuncRole.RAW_SIDECAR]
    assert beside.location == "sub-02/func/sub-02_task-nback_run-1_bold.json"
    # the raw name run-02 has no sidecar of its own and inherits the top-level one
    inherited = listed["sub-02_task-nback_run-02"].files[FuncRole.RAW_SIDECAR]
    assert inherited.location == "task-nback_bold.json"
    assert bids.sidecar_time(dataset.raw_root / inherited.location) is None
    # a run of another task inherits nothing
    assert FuncRole.RAW_SIDECAR not in listed["sub-02_task-rest_run-2"].files
    times = {
        r.stem: bids.sidecar_time(dataset.raw_root / r.files[FuncRole.RAW_SIDECAR].location)
        for r in bids.list_raw_runs(dataset.raw_root, "sub-01")
    }
    assert times["sub-01_ses-1_task-rest_run-2"] == "9:5:3.5"


def test_the_nearest_sidecar_with_the_most_entities_wins(tmp_path: Path) -> None:
    func = tmp_path / "sub-01" / "ses-1" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_ses-1_task-rest_run-02_bold.nii.gz").write_bytes(b"x")
    for place, name in (
        (tmp_path, "task-rest_bold.json"),
        (tmp_path / "sub-01", "sub-01_task-rest_bold.json"),
        (tmp_path / "sub-01" / "ses-1", "sub-01_ses-1_bold.json"),
        (tmp_path / "sub-01" / "ses-1", "sub-01_ses-1_task-rest_run-2_bold.json"),
        (tmp_path / "sub-01" / "ses-1", "sub-01_ses-1_task-rest_run-3_bold.json"),
    ):
        (place / name).write_text("{}")
    (run,) = bids.list_raw_runs(tmp_path, "sub-01")
    # run-2 applies to run-02, run-3 does not; of the session level the one with more entities
    location = run.files[FuncRole.RAW_SIDECAR].location
    assert location == "sub-01/ses-1/sub-01_ses-1_task-rest_run-2_bold.json"
    (tmp_path / "sub-01" / "ses-1" / "sub-01_ses-1_task-rest_run-2_bold.json").unlink()
    (run,) = bids.list_raw_runs(tmp_path, "sub-01")
    assert run.files[FuncRole.RAW_SIDECAR].location == "sub-01/ses-1/sub-01_ses-1_bold.json"


def test_names_that_are_no_raw_run_are_passed_over(tmp_path: Path) -> None:
    func = tmp_path / "sub-01" / "func"
    func.mkdir(parents=True)
    names = [
        "sub-01_task-rest_run-1_bold.nii.gz",
        "sub-01_task-rest_run-1_bold.nii",  # the .nii.gz of the same stem is listed only
        "sub-01_task-rest_run-2_bold.nii",
        "sub-01_task-rest_run-3_part-mag_bold.nii.gz",
        "sub-01_task-rest_run-3_part-phase_bold.nii.gz",
        "sub-01_ses-2_task-rest_run-1_bold.nii.gz",  # names a session it is not in
        "sub-02_task-rest_run-1_bold.nii.gz",  # another participant
        "sub-01_run-1_bold.nii.gz",  # no task
        "sub-01_task-rest_run-x_bold.nii.gz",
        "sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz",
        "sub-01_task-rest_run-4_sbref.nii.gz",
        "sub-01_task-rest_run-4_bold.json",
        "sub-01_task-rest_run-5_bold.nii.gz.tmp",
        "_bold.nii.gz",
    ]
    for name in names:
        (func / name).write_bytes(b"x")
    (func / "sub-01_task-rest_run-6_bold.nii.gz").mkdir()  # a folder is no file
    os.symlink(tmp_path / "absent", func / "sub-01_task-rest_run-7_bold.nii.gz")  # dangling
    with contextlib.suppress(OSError):  # a file system that refuses such a name has no such file
        os.close(os.open(os.fsencode(func) + b"/sub-01_task-r\xe9st_run-8_bold.nii.gz", os.O_CREAT))
    listed = bids.list_raw_runs(tmp_path, "sub-01")
    assert [r.files[FuncRole.RAW_BOLD].location.rsplit("/", 1)[1] for r in listed] == [
        "sub-01_task-rest_run-1_bold.nii.gz",
        "sub-01_task-rest_run-2_bold.nii",
        "sub-01_task-rest_run-3_part-mag_bold.nii.gz",
    ]
    assert bids.list_raw_runs(tmp_path, "sub-02") == []  # no folder
    assert bids.list_raw_runs(tmp_path, "01") == []  # not a participant
    assert bids.list_raw_runs(tmp_path / "absent", "sub-01") == []
    assert bids.is_raw_dataset(tmp_path)


def test_a_folder_that_cannot_be_listed_is_named_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    func = tmp_path / "sub-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_task-rest_bold.nii.gz").write_bytes(b"x")
    real = os.scandir

    def scandir(path: object) -> object:
        if Path(str(path)) == func:
            raise PermissionError(13, "Permission denied", str(path))
        return real(path)  # pyrefly: ignore[no-matching-overload]

    monkeypatch.setattr(bids.os, "scandir", scandir)
    with pytest.raises(OSError) as raised:
        bids.list_raw_runs(tmp_path, "sub-01")
    assert str(raised.value) == "[Errno 13] sub-01/func: Permission denied"
    assert str(tmp_path) not in str(raised.value) and raised.value.filename is None


def test_the_sessions_of_runs(dataset: SyntheticFuncDataset) -> None:
    assert bids.session_labels(bids.list_raw_runs(dataset.raw_root, "sub-01")) == ["1", "2"]
    assert bids.session_labels(bids.list_raw_runs(dataset.raw_root, "sub-02")) == [None]
    runs = bids.list_raw_runs(dataset.raw_root, "sub-01") + bids.list_raw_runs(
        dataset.raw_root, "sub-02"
    )
    assert bids.session_labels(runs) == [None, "1", "2"]
    assert bids.session_labels([]) == []
