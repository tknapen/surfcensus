"""``review openneuro:ds000102 --subjects sub-01 --epi`` on the real buckets (network; decision
D50): the functional files of ``ds000102`` sub-01 are fetched anonymously and read-only from
``s3://openneuro.org`` and ``s3://openneuro-derivatives`` into the run's cache, both runs are
packed, and every decoded volume of both streams of both runs equals that of a local review of
the copies the network tests keep in the pytest cache (``test_network_review_epi``). A second
call fetches nothing but listings and packs nothing. Only that subject is fetched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_func_pack import _fetch_worked_example  # pyrefly: ignore[missing-import]
from test_network_func_api import _fetch_extra, bids_root  # pyrefly: ignore[missing-import]

from surfcensus import remote
from surfcensus.func import PACK_JSON
from surfcensus.func.model import PackInfo, read_info
from surfcensus.func.packread import stream_bytes
from surfcensus.review import pack_counts, review
from surfcensus.store import EPI_DIR

pytestmark = pytest.mark.network


def _packs(run_dir: Path, names: dict[str, str]) -> dict[str, tuple[Path, PackInfo]]:
    found = {}
    for pack_json in (run_dir / EPI_DIR).glob(f"*/*/*/{PACK_JSON}"):
        pack = read_info(pack_json, PackInfo)
        assert pack is not None
        found[names[pack.run]] = (pack_json.parent.parent, pack)
    return found


def test_a_remote_review_with_epi_packs_the_streams_of_a_local_review(
    request: pytest.FixtureRequest,
    real_subject: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    data = _fetch_worked_example(request.config.cache.mkdir("ds000102-sub-01-func"))
    extra = _fetch_extra(request.config.cache.mkdir("ds000102-sub-01-func-bids"))
    ds = bids_root(tmp_path / "bids", data, extra, real_subject)
    local_dir = tmp_path / "local"
    with review(ds, run_dir=local_dir, serve=False, jobs=2) as run:
        runs = run.public().func_runs()
        local_names = dict(zip(runs["run_uid"], runs["name"], strict=True))

    remote_dir = tmp_path / "ds000102-surfcensus-run"
    with review(
        "openneuro:ds000102", subjects=["sub-01"], epi=True, run_dir=remote_dir, serve=False, jobs=2
    ) as run:
        assert pack_counts(run) == (2, 0, 0)
        runs = run.public().func_runs()
        remote_names = dict(zip(runs["run_uid"], runs["name"], strict=True))

    ours, theirs = _packs(remote_dir, remote_names), _packs(local_dir, local_names)
    assert sorted(ours) == sorted(theirs) == ["task-flanker_run-1", "task-flanker_run-2"]
    for name in ours:
        (our_dir, our_pack), (their_dir, their_pack) = ours[name], theirs[name]
        # the inputs agree but for the validation report (an empty file of the bucket), which
        # the remote tree holds and the cached copies of the local fixture do not; the report
        # is listed among a pack's inputs, so the content digests differ by it alone
        mine_in = [i for i in our_pack.inputs if i.role != "validation_report"]
        assert mine_in == list(their_pack.inputs), name
        for stream in ("raw", "pre"):
            mine = list(stream_bytes(our_dir, our_pack, stream))
            other = list(stream_bytes(their_dir, their_pack, stream))
            assert len(mine) == len(other) == 146
            assert [t for t in range(146) if mine[t] != other[t]] == [], (name, stream)

    start = remote.stats()
    with review(
        "openneuro:ds000102", subjects=["sub-01"], run_dir=remote_dir, serve=False, jobs=2
    ) as run:
        report = run.pack()
        assert (report.done, report.skipped) == (0, 2)
    # listings and the derivative's description alone: nothing of a series or a surface
    assert remote.stats().bytes - start.bytes < 2_000_000
