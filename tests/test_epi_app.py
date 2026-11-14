"""The pack-only server of the EPI page and the shapes it answers with (work package E1.D):
its routes, what it refuses, the files of the packs it serves and the ones it never serves,
and the shapes against the models of ``func.model`` field by field.

The packs are those of ``tests/epi_packs.py`` (two sessions of one subject, two grids, a
brighter run, a flagged run, a session without contours, a run listed and not packed)."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from epi_packs import PackedStream, make_stream, stream_order  # pyrefly: ignore[missing-import]
from fastapi.testclient import TestClient

from surfcensus.app import epi_schemas as shapes
from surfcensus.app.epi import create_epi_app, serve_packs, start_pack_server
from surfcensus.config import ConfigError
from surfcensus.func import PACK_JSON, SESSION_JSON, STREAM_JSON, TRACES_JSON
from surfcensus.func.model import (
    ChunkFile,
    ContourFile,
    GridInfo,
    PackFile,
    PackInfo,
    RegistrationInfo,
    SessionInfo,
    SessionRunInfo,
    StreamFiles,
    StreamInfo,
    TracesInfo,
)
from surfcensus.func.packread import read_pack

BASE = "http://127.0.0.1"


@pytest.fixture(scope="module")
def packed(tmp_path_factory: pytest.TempPathFactory) -> PackedStream:
    return make_stream(tmp_path_factory.mktemp("epiapp"), n_volumes=12)


@pytest.fixture(scope="module")
def client(packed: PackedStream) -> TestClient:
    return TestClient(create_epi_app(packed.out), base_url=BASE)


@pytest.fixture(scope="module")
def body(client: TestClient) -> dict[str, Any]:
    answer = client.get("/api/epi/stream")
    assert answer.status_code == 200
    return answer.json()


def listed_files(body: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """``(session, file, sha256, version)`` of every file the answer lists."""
    out = []
    for session in body["sessions"]:
        sid = session["session_uid"]
        for grid in session["grids"]:
            for entry in (grid["contours"] or {}).values():
                out.append((sid, entry["file"], entry["sha256"], grid["token"]))
        for run in session["runs"]:
            if run["pack"] is None:
                continue
            for stream in run["pack"]["streams"].values():
                for still in ("ref", "mean", "sd"):
                    e = stream[still]
                    out.append((sid, e["file"], e["sha256"], run["pack"]["token"]))
                for e in stream["chunks"]:
                    out.append((sid, e["file"], e["sha256"], run["pack"]["token"]))
    return out


# --- the shapes against the models ------------------------------------------------------------

# Shape -> (its model, the fields of the model that the page is not given, and the fields of
# the shape that the model does not have). A field added to a model fails here until it is
# decided whether a page gets it.
PAIRS: list[tuple[type, type, set[str], set[str]]] = [
    (
        shapes.EpiStreamOut,
        StreamInfo,
        {"format", "version", "surfcensus_version"},
        {"reading", "memory_mib", "rules", "contour_unit"},
    ),
    (
        shapes.EpiSessionOut,
        SessionInfo,
        {"format", "version", "surfcensus_version", "session", "config_hash", "rules"},
        {"session_uid", "media_url"},
    ),
    (shapes.GridOut, GridInfo, set(), {"token"}),
    (shapes.ContourOut, ContourFile, set(), set()),
    (shapes.RegistrationOut, RegistrationInfo, set(), set()),
    (shapes.EpiRunOut, SessionRunInfo, {"run"}, {"run_uid", "pack"}),
    (
        shapes.EpiPackOut,
        PackInfo,
        {
            "format",
            "version",
            "surfcensus_version",
            "run",
            "session",
            "config_hash",
            "settings",
            "inputs",
            "grid_key",
            "anchor_run",
            "n_volumes",
            "tr_s",
            "window",
            "sd_window",
            "content_digest",
        },
        {"token"},
    ),
    (shapes.StreamOut, StreamFiles, set(), set()),
    (shapes.FileOut, PackFile, set(), set()),
    (shapes.ChunkOut, ChunkFile, set(), set()),
    (shapes.TracesOut, TracesInfo, set(), set()),
]


@pytest.mark.parametrize(("shape", "model", "left_out", "added"), PAIRS)
def test_the_fields_of_a_shape_are_those_of_its_model_as_decided(
    shape: type, model: type, left_out: set[str], added: set[str]
) -> None:
    mine = set(shape.model_fields)
    theirs = set(model.model_fields)
    assert theirs - mine == left_out
    assert mine - theirs == added


def test_the_pack_files_are_given_as_pack_json_lists_them_and_nothing_of_its_inputs() -> None:
    # EpiPackOut.traces holds the content of traces.json, not the file entry of pack.json
    assert shapes.EpiPackOut.model_fields["traces"].annotation is shapes.TracesOut
    for name in ("inputs", "settings", "content_digest", "config_hash"):
        assert name not in shapes.EpiPackOut.model_fields


# --- the answer -------------------------------------------------------------------------------


def test_the_stream_is_the_sessions_of_stream_json_and_their_packed_runs_in_order(
    body: dict[str, Any], packed: PackedStream
) -> None:
    shapes.EpiStreamOut.model_validate(body)
    assert [s["session_uid"] for s in body["sessions"]] == list(packed.sessions)
    assert body["label"] == "sub-01" and body["memory_mib"] == 512
    order = []
    total = 0
    for session in body["sessions"]:
        first = 0
        assert session["media_url"] == f"/epi-media/{session['session_uid']}/"
        for run in session["runs"]:
            if run["pack"] is None:
                assert run["reason"] is not None
                continue
            assert run["first"] == first
            order.append((session["session_uid"], run["run_uid"], total, run["n_volumes"]))
            first += run["n_volumes"]
            total += run["n_volumes"]
    assert order == stream_order(packed)
    reasons = [r["reason"] for s in body["sessions"] for r in s["runs"] if r["reason"]]
    assert reasons == ["no raw series"]
    a, b = body["sessions"]
    assert len(a["grids"]) == 2 and all(g["contours"] for g in a["grids"])
    assert len(b["grids"]) == 1 and b["grids"][0]["contours"] is None
    flagged = [r["name"] for r in a["runs"] if r["frame"] and r["frame"]["flagged"]]
    assert flagged == ["ses-1_task-rest_run-3"]
    windows = [r["window"][1] for r in a["runs"]]
    assert max(windows) > 1.5 * min(windows)  # the brighter run


def test_the_answer_holds_no_path_no_input_digest_and_no_affine(
    body: dict[str, Any], packed: PackedStream
) -> None:
    text = json.dumps(body)
    assert str(packed.out) not in text and str(packed.dataset.root) not in text
    assert "/tmp" not in text and ".nii" not in text
    for word in ("inputs", "affine", "content_digest", "config_hash", "settings", "anchor_run"):
        assert f'"{word}"' not in text
    for run in packed.runs:
        info = read_pack(packed.out / run.session, run.run)
        assert info is not None
        for item in info.inputs:
            assert item.sha256 not in text
        assert info.content_digest not in text
        assert info.content_digest[:16] not in text


def test_a_pack_that_is_not_whole_is_not_in_the_stream_and_its_files_are_not_served(
    packed: PackedStream, tmp_path: Path
) -> None:
    out = tmp_path / "packs"
    shutil.copytree(packed.out, out)
    run = packed.runs[1]
    info = read_pack(out / run.session, run.run)
    assert info is not None
    (out / run.session / info.streams["raw"].chunks[0].file).unlink()
    client = TestClient(create_epi_app(out), base_url=BASE)
    body = client.get("/api/epi/stream").json()
    listed = {r["run_uid"]: r for s in body["sessions"] for r in s["runs"]}
    assert listed[run.run]["pack"] is None
    other = info.streams["pre"].chunks[0].file
    assert client.get(f"/epi-media/{run.session}/{other}").status_code == 404


# --- the files --------------------------------------------------------------------------------


def test_every_listed_file_is_served_as_it_is_with_its_cache_rule(
    client: TestClient, body: dict[str, Any], packed: PackedStream
) -> None:
    files = listed_files(body)
    assert len(files) > 100
    for sid, file, sha, version in files:
        answer = client.get(f"/epi-media/{sid}/{file}", params={"v": version})
        assert answer.status_code == 200, file
        assert hashlib.sha256(answer.content).hexdigest() == sha
        assert answer.content == (packed.out / sid / file).read_bytes()
        assert "immutable" in answer.headers["cache-control"]
    sid, file, _, version = files[0]
    for asked in ({}, {"v": "0" * 16}, {"v": version[:-1]}):
        answer = client.get(f"/epi-media/{sid}/{file}", params=asked)
        assert answer.status_code == 200 and answer.headers["cache-control"] == "no-cache"


def test_the_version_is_a_digest_of_the_served_digests_only(body: dict[str, Any]) -> None:
    for session in body["sessions"]:
        for run in session["runs"]:
            if run["pack"] is None:
                continue
            digests = []
            for stream in ("raw", "pre"):
                s = run["pack"]["streams"][stream]
                digests += [s[x]["sha256"] for x in ("ref", "mean", "sd")]
                digests += [c["sha256"] for c in s["chunks"]]
            want = hashlib.sha256("\n".join(digests).encode()).hexdigest()[:16]
            assert run["pack"]["token"] == want


def test_the_json_files_of_a_pack_are_never_served(
    client: TestClient, packed: PackedStream
) -> None:
    sid = packed.sessions[0]
    run = packed.runs[0].run
    for path in (
        f"/epi-media/{sid}/{SESSION_JSON}",
        f"/epi-media/{sid}/{run}/{PACK_JSON}",
        f"/epi-media/{sid}/{run}/{TRACES_JSON}",
        f"/epi-media/{sid}/../{STREAM_JSON}",
        f"/epi-media/{sid}/%2e%2e/{STREAM_JSON}",
        f"/epi-media/{sid}/{run}/raw/../../{SESSION_JSON}",
        f"/{STREAM_JSON}",
        "/static/../../func/model.py",
    ):
        answer = client.get(path)
        assert answer.status_code == 404, path
        assert b'"format"' not in answer.content


def test_a_name_that_is_not_listed_or_not_of_the_pattern_is_404(
    client: TestClient, packed: PackedStream, body: dict[str, Any]
) -> None:
    a, b = packed.sessions
    run_a = packed.runs[0].run
    run_b = next(r.run for r in packed.runs if r.session == b)
    missing = packed.not_packed[0]
    grid = body["sessions"][0]["grids"][0]["grid_key"]
    for path in (
        f"/epi-media/{a}/{run_a}/raw/c9999.bin",  # the pattern, not listed
        f"/epi-media/{a}/{run_a}/raw/c0000.bin.tmp",
        f"/epi-media/{a}/{run_a}/raw/c\u0660\u0660\u0660\u0660.bin",  # Unicode digits
        f"/epi-media/{a}/{run_b}/raw/ref.bin",  # a run of the other session
        f"/epi-media/{b}/{missing}/raw/ref.bin",  # a run that is not packed
        f"/epi-media/{b}/contours/{grid}/axial.bin",  # a session without contours
        f"/epi-media/{'0' * 16}/{run_a}/raw/ref.bin",  # a session not in the stream
        f"/epi-media/{a.upper()}/{run_a}/raw/ref.bin",
        f"/epi-media/{a}/{run_a}/raw/",
        f"/epi-media/{a}/{run_a}/raw",
        f"/epi-media/{a}",
        "/epi-media/",
        "/static/js/epi/missing.js",
        "/api/epi/stream/",
        "/api/session",
        "/review",
    ):
        assert client.get(path).status_code == 404, path


def test_a_link_that_leads_out_of_the_session_is_404(packed: PackedStream, tmp_path: Path) -> None:
    out = tmp_path / "packs"
    shutil.copytree(packed.out, out)
    run = packed.runs[0]
    info = read_pack(out / run.session, run.run)
    assert info is not None
    name = info.streams["raw"].chunks[0].file
    outside = tmp_path / "elsewhere.bin"
    target = out / run.session / name
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    client = TestClient(create_epi_app(out), base_url=BASE)
    assert client.get(f"/epi-media/{run.session}/{name}").status_code == 404
    # a link within the session leads to a file that has to have a pack file's name too
    other = info.streams["raw"].chunks[1].file
    moved = out / run.session / run.run / "raw" / "kept.bin"
    (out / run.session / other).rename(moved)
    (out / run.session / other).symlink_to(moved)
    assert client.get(f"/epi-media/{run.session}/{other}").status_code == 404
    third = info.streams["raw"].chunks[2].file
    renamed = out / run.session / run.run / "raw" / "c0100.bin"
    (out / run.session / third).rename(renamed)
    (out / run.session / third).symlink_to(renamed)
    assert client.get(f"/epi-media/{run.session}/{third}").status_code == 200


# --- the page and the guard -------------------------------------------------------------------


def test_the_page_is_served_at_both_routes_with_the_security_headers(client: TestClient) -> None:
    for route in ("/", "/epi"):
        answer = client.get(route)
        assert answer.status_code == 200
        text = answer.text
        assert 'data-api="/api/epi/stream"' in text
        assert '<script type="module" src="/static/js/epi/epi.js?v=' in text
        assert "raw data" in text and "preprocessing" in text
        csp = answer.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert answer.headers["x-frame-options"] == "DENY"
    for name in ("epi.js", "logic.js", "store.js", "worker.js", "views.js", "traces.js"):
        assert client.get(f"/static/js/epi/{name}").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/healthz").json()["ok"] is True


def test_another_host_another_origin_and_other_methods_are_refused(
    client: TestClient, packed: PackedStream
) -> None:
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403
    assert (
        client.get("/api/epi/stream", headers={"Origin": "http://evil.example"}).status_code == 403
    )
    answer = client.get("/api/epi/stream", headers={"Sec-Fetch-Site": "cross-site"})
    assert answer.status_code == 403
    assert client.post("/api/epi/stream", json={}).status_code in (403, 405)
    assert client.post("/api/epi/stream", json={}, headers={"Origin": BASE}).status_code == 405
    answer = client.get("/epi", follow_redirects=False)
    assert answer.status_code == 200 and "location" not in answer.headers


def test_a_directory_without_stream_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        create_epi_app(tmp_path)
    assert caught.value.reason == "missing"


def test_a_server_binds_to_localhost_only_and_stops(packed: PackedStream) -> None:
    with pytest.raises(ConfigError):
        start_pack_server(packed.out, host="0.0.0.0", port=0)
    handle = start_pack_server(packed.out, port=0)
    try:
        answer = httpx.get(f"{handle.url}healthz")
        assert answer.status_code == 200
        assert httpx.get(f"{handle.url}api/epi/stream").json()["label"] == "sub-01"
    finally:
        handle.stop()
    assert not handle.running


def test_serve_packs_serves_until_it_is_told_to_stop(
    packed: PackedStream, capsys: pytest.CaptureFixture[str]
) -> None:
    stop = threading.Event()
    worker = threading.Thread(
        target=serve_packs,
        args=(packed.out,),
        kwargs={"port": 0, "open_browser": False, "stop": stop},
    )
    worker.start()
    said = ""
    try:
        for _ in range(200):
            said += capsys.readouterr().err
            if "EPI page: " in said or not worker.is_alive():
                break
            stop.wait(0.05)
        url = said.split("EPI page: ", 1)[1].split()[0]
        assert url.startswith("http://127.0.0.1:")
        assert httpx.get(f"{url}healthz").status_code == 200
    finally:
        stop.set()
        worker.join(10)
    assert not worker.is_alive()
