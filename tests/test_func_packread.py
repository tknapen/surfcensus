"""Reading a pack back (``func.packread``): the JSON of this version only, whether a pack is
whole, the decoding of volumes, stills and contours with their digests checked, and what a
stored byte stands for under a run's own window and under the common scale of a stream.

The packs are made by ``func.pack`` on the synthetic fixture (``tests/test_func_pack.py``
builds the inputs); the expected values are computed here.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest
from synth_func import SyntheticFuncDataset, write_func_dataset  # pyrefly: ignore[missing-import]
from test_func_pack import (  # pyrefly: ignore[missing-import]
    _geometry_ready,
    fake_contours,
    plan_of,
    run_input,
    ses1,
    session_id,
)

from surfcensus.config import EpiConfig
from surfcensus.func import PACK_JSON, PLANES, SESSION_JSON, STILLS, STREAMS
from surfcensus.func.model import PackInfo
from surfcensus.func.pack import pack_session, quantise
from surfcensus.func.packread import (
    byte_values,
    common_scale,
    decode_contours,
    decode_still,
    decode_volume,
    pack_on_disk,
    read_pack,
    read_session,
    stream_bytes,
)


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _geometry_ready()
    ds: SyntheticFuncDataset = write_func_dataset(tmp_path_factory.mktemp("func"))
    runs, anchor = ses1(ds)
    plan = plan_of(ds, runs, anchor, "sub-01")
    contours = {key: fake_contours(grid) for key, grid in plan.grids.items()}
    out = tmp_path_factory.mktemp("out")
    sid = session_id("sub-01", "1")
    inputs = [run_input(r, anchor) for r in runs[:2]]
    plan = plan_of(ds, runs[:2], anchor, "sub-01")
    pack_session(out, sid, plan, inputs, contours, EpiConfig())
    return out / sid


@pytest.fixture
def copy(session_dir: Path, tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / session_dir.name
    shutil.copytree(session_dir, target)
    return target


def _first(session_dir: Path) -> PackInfo:
    session = read_session(session_dir)
    assert session is not None
    info = read_pack(session_dir, session.runs[0].run)
    assert info is not None
    return info


def test_read_session_and_read_pack_take_this_version_only(copy: Path) -> None:
    info = _first(copy)
    assert read_pack(copy, "0" * 16) is None
    path = copy / info.run / PACK_JSON
    data = json.loads(path.read_text())
    data["version"] = 1
    path.write_text(json.dumps(data))
    assert read_pack(copy, info.run) is None
    path.write_text("{")
    assert read_pack(copy, info.run) is None
    (copy / SESSION_JSON).write_text("[]")
    assert read_session(copy) is None


def test_pack_on_disk_needs_every_listed_file(copy: Path) -> None:
    info = _first(copy)
    assert pack_on_disk(copy, info.run)
    for relative in (info.traces.file, info.streams["raw"].sd.file):
        target = copy / relative
        moved = target.with_name("elsewhere")
        target.rename(moved)
        assert not pack_on_disk(copy, info.run)
        moved.rename(target)
        assert pack_on_disk(copy, info.run)
    # a name that is there counts, whatever it holds (two listings, no file is opened)
    (copy / info.streams["pre"].chunks[0].file).write_bytes(b"")
    assert pack_on_disk(copy, info.run)


def test_a_damaged_file_is_refused_not_decoded(copy: Path) -> None:
    info = _first(copy)
    assert decode_volume(copy, info, "raw", 0).dtype == np.uint8
    chunk = copy / info.streams["raw"].chunks[0].file
    blob = bytearray(chunk.read_bytes())
    blob[20] ^= 0xFF
    chunk.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="SHA-256"):
        decode_volume(copy, info, "raw", 0)
    with pytest.raises(ValueError, match="SHA-256"):
        list(stream_bytes(copy, info, "raw"))
    assert decode_volume(copy, info, "pre", 0).shape == (17, 18, 14)
    with pytest.raises(IndexError):
        decode_volume(copy, info, "pre", info.n_volumes)


def test_stream_bytes_and_decode_volume_agree(session_dir: Path) -> None:
    info = _first(session_dir)
    for stream in STREAMS:
        volumes = list(stream_bytes(session_dir, info, stream))
        assert len(volumes) == info.n_volumes
        for t in (0, 1, info.n_volumes - 1):
            v = decode_volume(session_dir, info, stream, t)
            assert v.tobytes(order="F") == volumes[t]


def test_the_stills_decode_to_volumes_of_the_grid(session_dir: Path) -> None:
    info = _first(session_dir)
    for stream in STREAMS:
        for still in STILLS:
            v = decode_still(session_dir, info, stream, still)
            assert v.shape == (17, 18, 14) and v.max() > 0
    with pytest.raises(ValueError):
        decode_still(session_dir, info, "raw", "median")


def test_contours_are_checked_against_session_json(copy: Path) -> None:
    session = read_session(copy)
    assert session is not None
    grid = session.grids[0]
    assert grid.contours is not None
    for plane in PLANES:
        assert decode_contours(copy, grid.grid_key, plane).startswith(
            b"contours of " + plane.encode()
        )
    path = copy / grid.contours["axial"].file
    path.write_bytes(zlib.compress(b"other"))
    with pytest.raises(ValueError, match="SHA-256"):
        decode_contours(copy, grid.grid_key, "axial")


def test_decoding_needs_the_grid_of_session_json(copy: Path) -> None:
    info = _first(copy)
    (copy / SESSION_JSON).unlink()
    with pytest.raises(ValueError, match="shape"):
        decode_volume(copy, info, "pre", 0)
    assert len(next(stream_bytes(copy, info, "pre"))) == 17 * 18 * 14


def test_what_a_stored_byte_stands_for() -> None:
    window = (0.0, 1035.2579345703125)
    values = byte_values(window)
    assert values[0] == 0.0 and values[255] == pytest.approx(window[1])
    # every value in the middle third of a step quantises back to its byte
    rng = np.random.default_rng(5)
    b = rng.integers(0, 256, 2000)
    step = window[1] / 255
    signal = values[b] + rng.uniform(-0.45, 0.45, b.size) * step
    got = quantise(signal.astype(np.float32).reshape(-1, 1, 1), *window).ravel()
    assert np.array_equal(got, b)


def test_the_common_scale_of_a_stream() -> None:
    assert np.array_equal(common_scale((0.0, 800.0), 800.0), np.arange(256, dtype=np.uint8))
    half = common_scale((0.0, 400.0), 800.0)
    assert half[0] == 0 and half[255] == 128 and half[1] == 1  # floor(0.5 + 0.5)
    assert np.all(np.diff(half.astype(int)) >= 0)
    with pytest.raises(ValueError):
        common_scale((0.0, 900.0), 800.0)
    # the shown value approximates the signal under the one window (0, top)
    shown = byte_values((0.0, 800.0))[half]
    stands = byte_values((0.0, 400.0))
    assert np.abs(shown - stands).max() <= 800.0 / 255 / 2 + 1e-9


def test_the_common_scale_rounds_as_its_rule_is_written() -> None:
    """Page review 9: the rule is floor(b * (hi / H) + 0.5). At an exact half the order of the
    operations decides: with hi = 7, H = 10 and b = 45, 45 * (7 / 10) is just below 31.5 in
    float64 and gives 31, where 45 * 7 / 10 = 31.5 exactly would give 32."""
    from surfcensus.func.model import RULES

    table = common_scale((0.0, 7.0), 10.0)
    assert int(table[45]) == 31
    assert math.floor(45 * (7.0 / 10.0) + 0.5) == 31 and math.floor(45 * 7.0 / 10.0 + 0.5) == 32
    assert "floor(b * (hi / H) + 0.5)" in RULES["common_scale"]
    assert int(common_scale((0.0, 10.0), 10.0)[45]) == 45  # the largest keeps its values
