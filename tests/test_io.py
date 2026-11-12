"""Tests of surfcensus/_io.py and surfcensus/log.py: atomic writes, byte-stable archives,
digests, and the package logger."""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from surfcensus import _io, log
from surfcensus.render import encode, vertexmap

# SHA-256 and size of the archive that the vertex-map writer of milestone 1 (commit 6827032)
# wrote for `fixed_map()`, recorded before the writer moved to _io.write_npz
MILESTONE_1_SHA256 = "cd9bbd3c9a6bc6f0b84adcb564238d7c5b461a23589e3a81aed69bca4e42067e"
MILESTONE_1_SIZE = 2574
DTYPES = {"index": "<i4", "pial": "<f4", "white": "<f4", "fsavg6": "<i4", "visible": "|b1"}


def fixed_map() -> dict[str, np.ndarray]:
    """A vertex map with wide dtypes, as ``build_vertex_map`` could hand it to the writer."""
    vmap: dict[str, np.ndarray] = {}
    for hemi, k in (("lh", 5), ("rh", 3)):
        base = np.arange(k * 3, dtype=np.float64).reshape(k, 3)
        vmap[f"{hemi}_index"] = np.arange(k, dtype=np.int64) * 7
        vmap[f"{hemi}_pial"] = base * 0.5 - 3.25
        vmap[f"{hemi}_white"] = base * 0.25 + 1.125
        vmap[f"{hemi}_fsavg6"] = np.full(k, -1, dtype=np.int64)
        vmap[f"{hemi}_visible"] = np.arange(k) % 2 == 0
    return vmap


def milestone_1_bytes(vmap: dict[str, np.ndarray]) -> bytes:
    """The archive as ``render/vertexmap.write_vertex_map`` built it in milestone 1."""
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for key in vertexmap.KEYS:
            buffer = io.BytesIO()
            dtype = DTYPES[key.split("_")[1]]
            np.lib.format.write_array(buffer, np.ascontiguousarray(vmap[key], dtype=dtype))
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, buffer.getvalue())
    return target.getvalue()


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- write_npz, read_npz ------------------------------------------------------------------


def test_the_vertex_map_has_the_bytes_of_the_milestone_1_writer(tmp_path: Path) -> None:
    vmap = fixed_map()
    vertexmap.write_vertex_map(tmp_path / "vertex_map.npz", vmap)
    written = (tmp_path / "vertex_map.npz").read_bytes()
    assert written == milestone_1_bytes(vmap)
    assert len(written) == MILESTONE_1_SIZE
    assert hashlib.sha256(written).hexdigest() == MILESTONE_1_SHA256
    assert leftovers(tmp_path) == ["vertex_map.npz"]


def test_write_npz_gives_those_bytes_for_arrays_of_the_stored_dtypes(tmp_path: Path) -> None:
    vmap = fixed_map()
    cast = {key: vmap[key].astype(DTYPES[key.split("_")[1]]) for key in vertexmap.KEYS}
    _io.write_npz(tmp_path / "a.npz", cast)
    assert (tmp_path / "a.npz").read_bytes() == milestone_1_bytes(vmap)


def test_write_npz_is_byte_identical_between_calls_and_keeps_the_mapping_order(
    tmp_path: Path,
) -> None:
    arrays = {"b": np.arange(4, dtype=np.int16), "a": np.float32([[1.5, -2.0]]), "c": np.bool_([])}
    _io.write_npz(tmp_path / "one.npz", arrays)
    _io.write_npz(tmp_path / "two.npz", dict(arrays))
    assert (tmp_path / "one.npz").read_bytes() == (tmp_path / "two.npz").read_bytes()
    with zipfile.ZipFile(tmp_path / "one.npz") as archive:
        assert archive.namelist() == ["b.npy", "a.npy", "c.npy"]
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert entry.compress_type == zipfile.ZIP_STORED
    _io.write_npz(tmp_path / "other.npz", {"a": arrays["a"], "b": arrays["b"], "c": arrays["c"]})
    assert (tmp_path / "other.npz").read_bytes() != (tmp_path / "one.npz").read_bytes()


def test_read_npz_returns_what_was_written_in_order(tmp_path: Path) -> None:
    arrays = {
        "thickness": np.linspace(0, 5, 7).astype(np.float16),
        "mask": np.array([True, False, True]),
        "index": np.arange(6, dtype=np.int32).reshape(2, 3),
        "scalar": np.array(2.5),
    }
    _io.write_npz(tmp_path / "x.npz", arrays)
    back = _io.read_npz(tmp_path / "x.npz")
    assert list(back) == list(arrays)
    for key, array in arrays.items():
        assert back[key].dtype == np.asarray(array).dtype
        np.testing.assert_array_equal(back[key], array)
    # numpy's own reader agrees
    with np.load(tmp_path / "x.npz") as archive:
        np.testing.assert_array_equal(archive["index"], arrays["index"])


def test_write_npz_stores_little_endian_whatever_the_array_has(tmp_path: Path) -> None:
    values = np.array([1, 258, -3], dtype=">i4")
    _io.write_npz(tmp_path / "big.npz", {"v": values, "f": values.astype(">f8")})
    _io.write_npz(tmp_path / "little.npz", {"v": values.astype("<i4"), "f": values.astype("<f8")})
    assert (tmp_path / "big.npz").read_bytes() == (tmp_path / "little.npz").read_bytes()
    back = _io.read_npz(tmp_path / "big.npz")
    assert back["v"].dtype == np.dtype("<i4") and back["f"].dtype == np.dtype("<f8")
    np.testing.assert_array_equal(back["v"], [1, 258, -3])


def test_write_npz_takes_a_view_that_is_not_contiguous(tmp_path: Path) -> None:
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    _io.write_npz(tmp_path / "view.npz", {"columns": grid[:, ::2]})
    np.testing.assert_array_equal(_io.read_npz(tmp_path / "view.npz")["columns"], grid[:, ::2])


def test_write_npz_refuses_objects_and_leaves_nothing_behind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"[Oo]bject"):
        _io.write_npz(tmp_path / "bad.npz", {"ok": np.arange(3), "o": np.array([{}, None])})
    assert leftovers(tmp_path) == []


def test_read_npz_refuses_a_pickled_entry(tmp_path: Path) -> None:
    np.savez(tmp_path / "pickled.npz", o=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle"):
        _io.read_npz(tmp_path / "pickled.npz")


# --- atomic_path ----------------------------------------------------------------------------


def test_atomic_path_writes_beside_the_target_and_renames(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "file.txt"
    with _io.atomic_path(target) as tmp:
        assert tmp.parent == target.parent and tmp != target
        assert tmp.name.startswith(".file.txt.") and tmp.name.endswith(".tmp")
        tmp.write_text("new")
        assert not target.exists()
    assert target.read_text() == "new"
    assert leftovers(target.parent) == ["file.txt"]


def test_atomic_path_leaves_nothing_behind_on_an_exception(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("old")
    with pytest.raises(RuntimeError, match="stop"), _io.atomic_path(target) as tmp:
        tmp.write_text("half")
        raise RuntimeError("stop")
    assert target.read_text() == "old"
    assert leftovers(tmp_path) == ["file.txt"]
    # likewise when nothing was written, and for an interruption
    with pytest.raises(KeyboardInterrupt), _io.atomic_path(tmp_path / "never.txt"):
        raise KeyboardInterrupt
    assert leftovers(tmp_path) == ["file.txt"]


def test_atomic_path_takes_a_string_and_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"old content, longer")
    with _io.atomic_path(str(target)) as tmp:
        tmp.write_bytes(b"new")
    assert target.read_bytes() == b"new"


def test_the_renderer_uses_the_same_atomic_path() -> None:
    # render/api.py removes the temporaries of an interrupted render by this name pattern
    assert encode.atomic_path is _io.atomic_path


# --- sha256_file, canonical_json ------------------------------------------------------------


def test_sha256_file_is_the_digest_of_the_content(tmp_path: Path) -> None:
    content = bytes(range(256)) * 5000  # larger than one read block
    (tmp_path / "blob").write_bytes(content)
    assert _io.sha256_file(tmp_path / "blob") == hashlib.sha256(content).hexdigest()
    (tmp_path / "empty").touch()
    assert _io.sha256_file(str(tmp_path / "empty")) == hashlib.sha256(b"").hexdigest()
    with pytest.raises(FileNotFoundError):
        _io.sha256_file(tmp_path / "absent")


def test_canonical_json_does_not_depend_on_key_order() -> None:
    one = {"b": [1, 2.5, "x"], "a": {"d": None, "c": True}}
    two = {"a": {"c": True, "d": None}, "b": [1, 2.5, "x"]}
    assert _io.canonical_json(one) == _io.canonical_json(two)
    assert _io.canonical_json(one) == '{"a":{"c":true,"d":null},"b":[1,2.5,"x"]}'
    assert _io.canonical_json({"t": (1, 2)}) == _io.canonical_json({"t": [1, 2]})
    with pytest.raises(TypeError):
        _io.canonical_json({"path": Path("x")})


# --- log --------------------------------------------------------------------------------------


@pytest.fixture
def package_logger() -> logging.Logger:
    """The package logger, put back as it was after the test."""
    logger = log.get_logger()
    handlers, level = list(logger.handlers), logger.level
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_get_logger_is_the_package_logger() -> None:
    assert log.get_logger() is logging.getLogger("surfcensus")
    assert log.get_logger().name == log.LOGGER_NAME


def test_library_messages_reach_caplog(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="surfcensus"):
        log.get_logger().info("listed %d subjects", 3)
    assert caplog.messages == ["listed 3 subjects"]


def test_cli_logging_adds_one_stderr_handler_however_often_it_is_called(
    package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    before = len(package_logger.handlers)
    log.configure_cli_logging()
    log.configure_cli_logging()
    assert len(package_logger.handlers) == before + 1
    package_logger.info("not shown")
    package_logger.warning("two directories resolve to %s", "sub-01")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "WARNING: two directories resolve to sub-01\n"

    log.configure_cli_logging(verbose=True)
    assert len(package_logger.handlers) == before + 1
    package_logger.info("shown")
    assert capsys.readouterr().err == "INFO: shown\n"


def test_cli_logging_writes_to_the_stderr_of_the_moment(
    package_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a test runner replaces sys.stderr per invocation; a handler that kept the stream it was
    # made with would write to a closed one
    log.configure_cli_logging()
    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    package_logger.warning("one")
    first.close()
    monkeypatch.setattr(sys, "stderr", second)
    package_logger.warning("two")
    assert second.getvalue() == "WARNING: two\n"
