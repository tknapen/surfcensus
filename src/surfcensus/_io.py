"""File helpers shared by the package: atomic writes, byte-stable archives, digests.

Standard library and numpy only, so that any module can import it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def atomic_path(path: Path | str) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; rename it into place on success, else remove it."""
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.{secrets.token_hex(8)}.tmp")
    try:
        yield tmp
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


def write_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an ``.npz`` that ``numpy.load`` reads, byte-identical for identical input.

    ``numpy.savez`` stamps every zip entry with the current time, so the archive is written
    here: entries in the order of the mapping, stored uncompressed, dated 1980-01-01, each
    array in little-endian byte order with the dtype it has. Atomic. An array of Python
    objects is refused (ValueError), since it could only be stored pickled.
    """
    with atomic_path(path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            data = np.ascontiguousarray(array)
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, data.astype(data.dtype.newbyteorder("<"), copy=False), allow_pickle=False
            )
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3  # Unix on every platform, with fixed permissions
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, buffer.getvalue())


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Every array of an ``.npz``, in the order of the archive. Pickled entries are refused.

    A file that cannot be opened is the ``OSError`` of the system (``FileNotFoundError`` for
    one that is not there). One that is opened and is not such an archive is a
    ``ValueError`` that names the file and not its path, whatever numpy and ``zipfile``
    raise for it: a damaged archive gives ``BadZipFile``, ``EOFError`` or ``KeyError``, a
    damaged array header an error of Python's tokenizer, a damaged zip header
    ``NotImplementedError``.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except OSError:
        raise
    except Exception as exc:
        name = Path(path).name
        raise ValueError(f"{name} cannot be read as an archive of arrays") from exc


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of the content of ``path``, read in blocks."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json(obj: object) -> str:
    """JSON with sorted keys and no white space: the form that is hashed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
