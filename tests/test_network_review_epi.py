"""``review`` end to end on the worked example (network): ``ds000102`` sub-01 laid out as a BIDS
dataset root with its fMRIPrep derivative below ``derivatives/`` and its FreeSurfer subject in
``sourcedata/freesurfer/`` (``test_network_func_api.bids_root``, links into the pytest cache).

What is checked (step 2 of the EPI interface, work package E2.Z): one subject rendered and two
runs packed by ``review(..., serve=False)``; every decoded volume of both streams of both runs
equal to those of ``func.api.epi_subject`` (the command ``epi-session``) on the same folder,
with the same settings; and a second call that renders and packs nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from synth_annex import snapshot_links  # pyrefly: ignore[missing-import]
from test_func_pack import _fetch_worked_example  # pyrefly: ignore[missing-import]
from test_network_func_api import _fetch_extra, bids_root  # pyrefly: ignore[missing-import]

from surfcensus.func import PACK_JSON, SESSION_JSON, STREAM_JSON, api
from surfcensus.func.model import PackInfo, SessionInfo, StreamInfo, read_info
from surfcensus.func.packread import stream_bytes
from surfcensus.review import pack_counts, render_counts, review
from surfcensus.store import EPI_DIR

pytestmark = pytest.mark.network


def test_review_of_the_worked_example_renders_the_subject_and_packs_both_runs(
    request: pytest.FixtureRequest, real_subject: Path, tmp_path: Path
) -> None:
    data = _fetch_worked_example(request.config.cache.mkdir("ds000102-sub-01-func"))
    extra = _fetch_extra(request.config.cache.mkdir("ds000102-sub-01-func-bids"))
    ds = bids_root(tmp_path / "bids", data, extra, real_subject)
    before = snapshot_links(tmp_path / "bids")
    run_dir = tmp_path / "run"

    run = review(ds, run_dir=run_dir, serve=False, jobs=2)
    try:
        assert render_counts(run) == (1, 0, 0)
        assert pack_counts(run) == (2, 0, 0)
        runs = run.public().func_runs()
        names = dict(zip(runs["run_uid"], runs["name"], strict=True))
    finally:
        run.close()
    assert snapshot_links(tmp_path / "bids") == before

    out = tmp_path / "out"
    stream = api.epi_subject(ds, "sub-01", out)
    (sid,) = stream.sessions
    info = read_info(out / sid / SESSION_JSON, SessionInfo)
    assert info is not None
    their_names = {r.run: r.name for r in info.runs}
    theirs = {}
    for pack_json in (out / sid).glob(f"*/{PACK_JSON}"):
        pack = read_info(pack_json, PackInfo)
        assert pack is not None
        theirs[their_names[pack.run]] = (out / sid, pack)
    ours = {}
    (participant,) = [p for p in (run_dir / EPI_DIR).iterdir() if p.is_dir()]
    assert read_info(participant / STREAM_JSON, StreamInfo) is not None
    for pack_json in participant.glob(f"*/*/{PACK_JSON}"):
        pack = read_info(pack_json, PackInfo)
        assert pack is not None and pack.run == pack_json.parent.name
        ours[names[pack.run]] = (pack_json.parent.parent, pack)
    assert sorted(ours) == sorted(theirs) == ["task-flanker_run-1", "task-flanker_run-2"]
    for name in ours:
        (our_dir, our_pack), (their_dir, their_pack) = ours[name], theirs[name]
        assert our_pack.content_digest == their_pack.content_digest
        for stream_name in ("raw", "pre"):
            mine = list(stream_bytes(our_dir, our_pack, stream_name))
            other = list(stream_bytes(their_dir, their_pack, stream_name))
            assert len(mine) == len(other) == 146
            assert [t for t in range(146) if mine[t] != other[t]] == [], (name, stream_name)

    run = review(ds, run_dir=run_dir, serve=False, jobs=2)
    try:
        report = run.pack()
        assert (report.done, report.skipped) == (0, 2)
    finally:
        run.close()
