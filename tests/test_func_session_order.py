"""The order of a participant's sessions in its EPI stream (decision D42): by the acquisition
dates of the raw BIDS dataset (``sub-X_sessions.tsv``, else ``sub-X_ses-Y_scans.tsv``) when
every session has one, else by label; the rule recorded, the dates never.

Tomas, on a real subject whose sessions ``ses-test`` and ``ses-retest`` played retest first:
"order sessions by acquisition date instead of label".

The run-level tests use ``synth_func.write_review_folder``: ``sub-01`` has the sessions ``1``
and ``2`` (label order 1, 2), so dates that put ``2`` first show the rule against labels.
"""

from __future__ import annotations

import builtins
import datetime
import io
import json
import os
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from synth_annex import make_repository, unfetch  # pyrefly: ignore[missing-import]
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_review_folder,
    write_session_dates,
)

from surfcensus.config import RunConfig, new_local_config, run_key
from surfcensus.func import bids, inventory
from surfcensus.func.inventory import (
    ACQUISITION_ORDER,
    SESSION_ORDER,
    order_sessions,
    order_words,
)
from surfcensus.inventory.subjects import run_inventory
from surfcensus.privacy import HashKey
from surfcensus.records import PhaseReport
from surfcensus.sources.local import LocalSource
from surfcensus.store import Catalog

D = datetime.datetime
TEST_FIRST = {"test": "1900-01-02T09:00:00", "retest": "1900-02-03T09:00:00"}
"""Shifted years, as anonymised public datasets write them: test before retest."""
DATE_WORDS = re.compile(r"190[01]-|T(09|10):[0-9]{2}|01-0[12]")
"""What a date of the tests would look like anywhere it leaked (no hex digest holds a ``-`` or
a ``:``)."""


# --- reading a date ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1900-01-02T09:00:00", D(1900, 1, 2, 9, 0, 0)),
        ("2021-03-04T05:06:07Z", D(2021, 3, 4, 5, 6, 7)),
        ("2021-03-04T05:06:07.250000", D(2021, 3, 4, 5, 6, 7, 250000)),
        ("2021-03-04T05:06:07.1234567891Z", D(2021, 3, 4, 5, 6, 7, 123456)),
        ("2021-03-04T05:06:07+02:00", D(2021, 3, 4, 3, 6, 7)),
        ("2021-03-04T23:30:00-01:00", D(2021, 3, 5, 0, 30, 0)),
        ("  2021-03-04T05:06:07  ", D(2021, 3, 4, 5, 6, 7)),
        ("n/a", None),
        ("", None),
        ("2021-03-04", None),
        ("05:06:07", None),
        ("2021-13-04T05:06:07", None),
        ("2021-02-30T05:06:07", None),
        ("yesterday", None),
        ("2021-03-04 05:06:07", None),
        (None, None),
        (20210304, None),
    ],
)
def test_an_acq_time_is_read_as_bids_writes_it(text: object, expected: D | None) -> None:
    assert bids.parse_acq_time(text) == expected


# --- the rule ---------------------------------------------------------------------------------


def test_dates_order_test_before_retest_where_labels_put_retest_first() -> None:
    labels = ["test", "retest"]
    assert order_sessions(labels).labels == ("retest", "test")  # the label order of before
    dates = {lab: bids.parse_acq_time(v) for lab, v in TEST_FIRST.items()}
    order = order_sessions(labels, dates)
    assert order.rule == ACQUISITION_ORDER == "acquisition time"
    assert order.labels == ("test", "retest") and order.undated == ()
    assert order_words(order) == "in the order of their acquisition times"


def test_one_session_without_a_date_puts_all_in_label_order() -> None:
    dates = {"test": D(1900, 1, 2), "retest": None}
    order = order_sessions(["test", "retest"], dates)
    assert order.rule == SESSION_ORDER == "session label"
    assert order.labels == ("retest", "test") and order.undated == ("retest",)
    assert "no acquisition date for ses-retest" in order_words(order)
    none = order_sessions(["test", "retest"], {})
    assert none.undated == ("retest", "test")
    assert order_words(none) == (
        "in the order of their labels (no acquisition date in the raw dataset)"
    )


def test_a_tie_goes_by_the_label_order_and_no_session_is_unaffected() -> None:
    same = D(2020, 5, 5, 12)
    order = order_sessions(["10", "2", "b"], {"10": same, "2": same, "b": D(2020, 1, 1)})
    assert order.rule == ACQUISITION_ORDER and order.labels == ("b", "2", "10")
    alone = order_sessions([None], {None: same})
    assert alone.rule == SESSION_ORDER and alone.labels == (None,) and alone.undated == ()


# --- where a date comes from ------------------------------------------------------------------


def _raw(tmp_path: Path, labels: tuple[str, ...] = ("test", "retest")) -> Path:
    raw = tmp_path / "raw"
    for label in labels:
        (raw / "sub-01" / f"ses-{label}" / "func").mkdir(parents=True)
    return raw


def test_sessions_tsv_wins_over_scans_tsv_and_scans_tsv_gives_its_earliest(
    tmp_path: Path,
) -> None:
    raw = _raw(tmp_path)
    write_session_dates(
        raw,
        "sub-01",
        sessions={"test": "1900-01-02T09:00:00"},  # no row for retest
        scans={
            "test": ["1901-01-01T00:00:00"],
            "retest": ["1900-02-03T10:00:00Z", "1900-02-03T09:30:00.5", "n/a"],
        },
    )
    dates = bids.session_dates(raw, "sub-01", ["test", "retest", None])
    assert dates == {
        "test": D(1900, 1, 2, 9),
        "retest": D(1900, 2, 3, 9, 30, 0, 500000),
        None: None,
    }


@pytest.mark.parametrize("value", ["n/a", "garbage", "1900-01-02"])
def test_a_value_that_is_no_date_falls_back_to_scans_tsv_or_gives_none(
    tmp_path: Path, value: str
) -> None:
    raw = _raw(tmp_path)
    write_session_dates(raw, "sub-01", sessions={"test": value, "retest": value})
    assert bids.session_dates(raw, "sub-01", ["test", "retest"]) == {"test": None, "retest": None}
    write_session_dates(raw, "sub-01", scans={"retest": ["1900-02-03T09:00:00"]})
    dates = bids.session_dates(raw, "sub-01", ["test", "retest"])
    assert dates == {"test": None, "retest": D(1900, 2, 3, 9)}


@pytest.mark.parametrize("kind", ["link", "pointer"])
def test_a_table_that_is_a_git_annex_placeholder_gives_no_date(tmp_path: Path, kind: str) -> None:
    raw = _raw(tmp_path)
    written = write_session_dates(raw, "sub-01", sessions=TEST_FIRST)
    assert bids.session_dates(raw, "sub-01", ["test"]) == {"test": D(1900, 1, 2, 9)}
    make_repository(raw)
    unfetch(raw, written, kind)  # pyrefly: ignore[bad-argument-type]
    assert bids.session_dates(raw, "sub-01", ["test", "retest"]) == {"test": None, "retest": None}
    # its listing facts are in the hash all the same, so fetching it plans again
    assert len(bids.session_date_lines(raw, "sub-01", ["test", "retest"])) == 1


def test_an_unreadable_table_gives_no_date_and_no_error(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    (path,) = write_session_dates(raw, "sub-01", sessions=TEST_FIRST)
    path.write_bytes(b"session_id\tacq_time\n\xff\xfe\tbad\n")
    assert bids.session_dates(raw, "sub-01", ["test"]) == {"test": None}
    path.unlink()
    path.mkdir()  # a directory under the name
    assert bids.session_dates(raw, "sub-01", ["test"]) == {"test": None}


# --- in a run ---------------------------------------------------------------------------------


@dataclass
class Bench:
    ds: SyntheticFuncDataset
    config: RunConfig
    key: HashKey
    catalog: Catalog
    run_dir: Path

    def func(self) -> PhaseReport:
        return inventory.run_func_inventory(
            self.catalog, [LocalSource(self.ds.raw_root)], self.config, self.key
        )

    def order(self, participant: str = "sub-01") -> tuple[str, list[str | None]]:
        rule = self.catalog.query(
            "SELECT session_order FROM func_participants WHERE participant = ?", (participant,)
        )[0][0]
        rows = self.catalog.query(
            "SELECT s.session_label, s.stream_order FROM func_sessions s JOIN func_participants "
            "p USING (participant_uid) WHERE p.participant = ? AND s.present = 1 "
            "ORDER BY s.stream_order",
            (participant,),
        )
        orders = [r[1] for r in rows]
        assert orders == list(range(len(rows))), orders  # one place per session
        return str(rule), [r[0] for r in rows]


@pytest.fixture
def bench(tmp_path: Path) -> Iterator[Bench]:
    ds = write_review_folder(tmp_path / "data")
    config = new_local_config(ds.raw_root)
    key = run_key(config)
    run_dir = tmp_path / "run"
    with Catalog.create(run_dir / "catalog.sqlite", mode="local", key=key) as catalog:
        run_inventory(catalog, [LocalSource(ds.raw_root)], config, key)
        yield Bench(ds, config, key, catalog, run_dir)


def test_without_dates_the_order_is_that_of_the_labels(bench: Bench) -> None:
    bench.func()
    assert bench.order() == (SESSION_ORDER, ["1", "2"])
    assert inventory.func_order_lines(bench.catalog) == [
        "sub-01: 2 sessions in the order of their labels (no acquisition date in the raw dataset)."
    ]


def test_dates_that_put_session_2_first_order_the_stream_by_date(bench: Bench) -> None:
    write_session_dates(
        bench.ds.raw_root,
        "sub-01",
        sessions={"1": "1900-03-01T10:00:00Z", "2": "1900-01-01T10:00:00Z"},
    )
    bench.func()
    assert bench.order() == (ACQUISITION_ORDER, ["2", "1"])
    assert inventory.func_order_lines(bench.catalog) == [
        "sub-01: 2 sessions in the order of their acquisition times."
    ]


def test_scans_tsv_alone_orders_the_stream(bench: Bench) -> None:
    write_session_dates(
        bench.ds.raw_root,
        "sub-01",
        scans={"1": ["1900-03-01T10:00:00", "1900-03-01T11:00:00"], "2": ["1900-01-01T10:00:00"]},
    )
    bench.func()
    assert bench.order() == (ACQUISITION_ORDER, ["2", "1"])


def test_a_session_without_a_date_falls_back_to_labels_and_is_named(bench: Bench) -> None:
    write_session_dates(bench.ds.raw_root, "sub-01", sessions={"2": "1900-01-01T10:00:00"})
    bench.func()
    assert bench.order() == (SESSION_ORDER, ["1", "2"])
    assert inventory.func_order_lines(bench.catalog) == [
        "sub-01: 2 sessions in the order of their labels, not of their acquisition times (no "
        "acquisition date for ses-1 in the raw dataset)."
    ]


def test_a_second_inventory_opens_no_file_and_a_new_table_plans_every_session_again(
    bench: Bench,
) -> None:
    bench.func()
    seen: list[str] = []
    real_open, real_io_open = builtins.open, io.open

    def b_open(file: object, *args: object, **kwargs: object) -> object:
        seen.append(os.fsdecode(file))  # pyrefly: ignore[bad-argument-type]
        return real_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    def i_open(file: object, *args: object, **kwargs: object) -> object:
        seen.append(os.fsdecode(file))  # pyrefly: ignore[bad-argument-type]
        return real_io_open(file, *args, **kwargs)  # pyrefly: ignore[no-matching-overload]

    def watched() -> PhaseReport:
        seen.clear()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builtins, "open", b_open)
            patch.setattr(io, "open", i_open)
            return bench.func()

    report = watched()
    raw = str(bench.ds.raw_root)
    assert (report.done, report.skipped) == (0, 3)
    assert [p for p in seen if p.startswith(f"{raw}/sub-")] == []
    # a sessions.tsv that appears plans both sessions of sub-01, and only those
    write_session_dates(
        bench.ds.raw_root, "sub-01", sessions={"1": "1900-03-01T10:00:00", "2": "1900-01-01T1:00"}
    )
    report = watched()
    assert (report.done, report.skipped) == (2, 1)
    assert bench.order() == (SESSION_ORDER, ["1", "2"])  # "1900-01-01T1:00" is no date
    # a changed table (another size or time) plans them again, and now orders by date
    time.sleep(0.01)
    write_session_dates(
        bench.ds.raw_root,
        "sub-01",
        sessions={"1": "1900-03-01T10:00:00", "2": "1900-01-01T10:00:00"},
    )
    assert (watched().done, bench.order()) == (2, (ACQUISITION_ORDER, ["2", "1"]))
    assert (watched().done, watched().skipped) == (0, 3)


def test_a_session_added_later_that_sorts_first_moves_the_stored_ones(bench: Bench) -> None:
    """The gap before D42: a session added later whose label sorts before a stored one left
    the stored one with its old ``stream_order``, so two sessions shared place 0."""
    import shutil

    hold = bench.run_dir.parent / "hold"
    for base, name in ((bench.ds.raw_root, "raw"), (bench.ds.deriv_root, "deriv")):
        shutil.move(base / "sub-01" / "ses-1", hold / name)
    bench.func()
    assert bench.order() == (SESSION_ORDER, ["2"])
    for base, name in ((bench.ds.raw_root, "raw"), (bench.ds.deriv_root, "deriv")):
        shutil.move(hold / name, base / "sub-01" / "ses-1")
    report = bench.func()
    assert (report.done, report.skipped) == (2, 1)
    assert bench.order() == (SESSION_ORDER, ["1", "2"])


def test_no_date_is_stored_in_the_catalog(bench: Bench) -> None:
    write_session_dates(
        bench.ds.raw_root,
        "sub-01",
        sessions={"1": "1901-03-01T10:00:00", "2": "1900-01-01T10:00:00"},
    )
    bench.func()
    with sqlite3.connect(bench.run_dir / "catalog.sqlite") as db:
        lines = "\n".join(db.iterdump())
    assert "1901-" not in lines and "1900-" not in lines


# --- end to end: review, the pack phase, the page's API, epi-session --------------------------


def _strings_under(root: Path) -> str:
    return "\n".join(
        p.read_text(errors="replace") for p in sorted(root.rglob("*")) if p.suffix == ".json"
    )


def test_review_orders_by_date_and_a_table_added_after_the_packs_reorders_without_packing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("fastapi")
    import httpx

    from surfcensus.func.api import epi_subject
    from surfcensus.review import open_review, review
    from surfcensus.store import EPI_DIR

    ds = write_review_folder(tmp_path / "data")
    run_dir = tmp_path / "run"
    review(ds.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    first = capsys.readouterr()
    assert "sub-01: 2 sessions in the order of their labels (no acquisition date" in first.err

    def stream_of(participant: str = "sub-01") -> dict[str, object]:
        for path in (run_dir / EPI_DIR).glob("*/stream.json"):
            data = json.loads(path.read_text())
            if data["label"] == participant:
                return data
        raise AssertionError("no stream")

    def labels_of(data: dict[str, object]) -> list[str]:
        root = next((run_dir / EPI_DIR).glob(f"*/{data['sessions'][0]}")).parent  # type: ignore[index]
        return [
            json.loads((root / sid / "session.json").read_text())["label"]
            for sid in data["sessions"]  # type: ignore[union-attr]
        ]

    before = stream_of()
    assert before["order"] == SESSION_ORDER and labels_of(before) == [
        "sub-01 ses-1",
        "sub-01 ses-2",
    ]
    packs = {p: p.stat().st_mtime_ns for p in (run_dir / EPI_DIR).rglob("pack.json")}

    write_session_dates(
        ds.raw_root, "sub-01", sessions={"1": "1900-03-01T10:00:00Z", "2": "1900-01-01T10:00:00Z"}
    )
    run, app = open_review(ds.raw_root, run_dir=run_dir, jobs=1, port=0)
    try:
        assert app is not None and app.wait_rendered(180)
        said = capsys.readouterr()
        assert "sub-01: 2 sessions in the order of their acquisition times." in said.err
        after = stream_of()
        assert after["order"] == ACQUISITION_ORDER
        assert labels_of(after) == ["sub-01 ses-2", "sub-01 ses-1"]
        # nothing packed again
        assert {p: p.stat().st_mtime_ns for p in (run_dir / EPI_DIR).rglob("pack.json")} == packs
        with httpx.Client(base_url=app.url, trust_env=False) as client:
            index = client.get("/api/epi").json()["participants"]
            puid = next(p["participant_uid"] for p in index if p["label"] == "sub-01")
            answer = client.get(f"/api/epi/{puid}")
            assert answer.status_code == 200
            body = answer.json()
            assert body["order"] == ACQUISITION_ORDER
            shown = json.dumps(body) + json.dumps(index)
    finally:
        if app is not None:
            app.stop()
        run.close()
    # no date anywhere: the display cache, what the page received, the public views, the lines
    assert not DATE_WORDS.search(_strings_under(run_dir / EPI_DIR))
    assert not DATE_WORDS.search(shown)
    assert not DATE_WORDS.search(first.err + first.out + said.err + said.out)
    with sqlite3.connect(run_dir / "catalog.sqlite") as db:
        views = [n for (n,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'view'")]
        for view in views:
            assert not DATE_WORDS.search(repr(db.execute(f"SELECT * FROM {view}").fetchall()))
    # epi-session gives the same order and names the rule
    out = tmp_path / "out"
    lines: list[str] = []
    stream = epi_subject(ds.raw_root, "sub-01", out, progress=lines.append)
    assert stream.order == ACQUISITION_ORDER
    assert [
        json.loads((out / sid / "session.json").read_text())["label"] for sid in stream.sessions
    ] == ["sub-01 ses-2", "sub-01 ses-1"]
    assert "sub-01: 2 sessions in the order of their acquisition times." in lines
    assert not DATE_WORDS.search("\n".join(lines) + _strings_under(out))
    # with --session naming a subset, the rule over those
    one = epi_subject(ds.raw_root, "sub-01", tmp_path / "one", sessions=["1"])
    assert one.order == ACQUISITION_ORDER and len(one.sessions) == 1


def test_epi_session_without_dates_says_the_labels(tmp_path: Path) -> None:
    from surfcensus.func.api import epi_subject

    ds = write_review_folder(tmp_path / "data")
    write_session_dates(ds.raw_root, "sub-01", sessions={"2": "1900-01-01T10:00:00"})
    lines: list[str] = []
    stream = epi_subject(ds.raw_root, "sub-01", tmp_path / "out", progress=lines.append)
    assert stream.order == SESSION_ORDER
    assert (
        "sub-01: 2 sessions in the order of their labels, not of their acquisition times (no "
        "acquisition date for ses-1 in the raw dataset)."
    ) in lines


def test_review_without_the_app_writes_the_new_order_and_packs_nothing(tmp_path: Path) -> None:
    """``review --no-serve``: a session that is pending with every pack current (its place
    moved) is written again by the pack phase, which ``pack_counts`` alone does not see."""
    pytest.importorskip("fastapi")
    from surfcensus.review import review
    from surfcensus.store import EPI_DIR

    ds = write_review_folder(tmp_path / "data")
    run_dir = tmp_path / "run"
    review(ds.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    packs = {p: p.stat().st_mtime_ns for p in (run_dir / EPI_DIR).rglob("pack.json")}
    write_session_dates(
        ds.raw_root, "sub-01", scans={"1": ["1900-03-01T10:00:00"], "2": ["1900-01-01T10:00:00"]}
    )
    review(ds.raw_root, run_dir=run_dir, jobs=1, serve=False).close()
    orders = {
        data["label"]: data["order"]
        for data in (json.loads(p.read_text()) for p in (run_dir / EPI_DIR).glob("*/stream.json"))
    }
    assert orders == {"sub-01": ACQUISITION_ORDER, "sub-02": SESSION_ORDER}
    assert {p: p.stat().st_mtime_ns for p in (run_dir / EPI_DIR).rglob("pack.json")} == packs
