"""FreeSurfer annotations (``label/?h.aparc.annot``) and label files (``label/?h.cortex.label``).

Both readers raise ``OSError`` for a file that is absent or may not be opened and
``ValueError`` for one that is damaged. A ``ValueError`` names the file, never its path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from jaxtyping import Bool, Int

NON_CORTEX_NAMES: tuple[str, ...] = ("unknown", "corpuscallosum")
"""Parcels of the Desikan annotation that are not cortex."""

_INT = np.dtype(">i4")
"""Every number of an annotation file is a big-endian int32."""


@dataclass(frozen=True)
class Annotation:
    labels: Int[np.ndarray, " n"]
    """int32 index into ``names`` per vertex; -1 for a vertex without a label."""
    names: tuple[str, ...]
    """Parcel names, decoded as UTF-8, in the order of the file's colour table."""

    def mask(self, *names: str) -> Bool[np.ndarray, " n"]:
        """Vertices labelled with one of ``names``. Raises KeyError naming every name that
        the file lacks."""
        missing = [name for name in dict.fromkeys(names) if name not in self.names]
        if missing:
            raise KeyError("the annotation has no parcel " + ", ".join(map(repr, missing)))
        return np.isin(self.labels, self._indices(names))

    def cortex(self) -> Bool[np.ndarray, " n"]:
        """Vertices with a label that is not one of :data:`NON_CORTEX_NAMES`."""
        return (self.labels >= 0) & ~np.isin(self.labels, self._indices(NON_CORTEX_NAMES))

    def _indices(self, names: tuple[str, ...]) -> Int[np.ndarray, " k"]:
        wanted = set(names)
        return np.array(
            [index for index, name in enumerate(self.names) if name in wanted], dtype=np.int32
        )


def read_annotation(path: Path | str) -> Annotation:
    """Read an annotation with nibabel. Raises ValueError when it has no colour table.

    A vertex gets the index of the colour table entry whose colour its annotation value
    packs. It gets -1 when the value is 0 (no label), and also when the table holds no such
    colour: for that case nibabel's own mapping (``orig_ids=False``, nibabel 5.5.0) takes
    the entry with the next higher value, or raises IndexError when there is none. Of two
    entries with one colour the first in the table is taken.

    The layout is checked before nibabel reads the file, because nibabel checks no length:
    it takes what is left of a file that ends inside the colour of the last table entry for
    that colour. ValueError therefore also for a file that ends early, for vertex records
    that are not numbered 0 to n - 1 in that order (nibabel takes the values by position),
    for an unknown table version, and for a colour table whose entries are not numbered 0 to
    m - 1 in that order: nibabel returns the names in file order without their numbers, so a
    label could not be named.

    The table also names the file it was made from (in the real files that were looked at,
    a path on the machine that built the atlas). It is skipped and not returned.
    """
    path = Path(path)
    _check_layout(path)
    values, table, raw_names = fs.read_annot(str(path), orig_ids=True)
    names = tuple(bytes(name).decode("utf-8", errors="replace") for name in raw_names)
    return Annotation(labels=_table_index(values, table[:, 4]), names=names)


def read_label(path: Path | str) -> Int[np.ndarray, " k"]:
    """The vertex indices of a ``.label`` file, int64, sorted ascending (the file's own
    order is arbitrary: ``fsaverage/label/lh.cortex.label`` is not sorted).

    The result is a set of vertices: a vertex that the file lists twice is returned once.
    A label file is text: a comment line, a line with the number of rows, then one row per
    vertex starting with its index. Raises ValueError when the second line is not a count,
    when the number of rows differs from it or the last row has fewer fields than the first
    (a file that was cut short), when a row does not start with an integer, and when an
    index is negative. FreeSurfer writes -1 for the points of a label that is not assigned
    to a surface (from memory; no such file was looked at), and as an array index -1 would
    silently name the last vertex.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as handle:
        handle.readline()
        declared = handle.readline().strip()
        rows = [line for line in handle if line.strip()]
    try:
        count = int(declared)
    except ValueError:
        raise ValueError(f"{path.name}: not a label file (no row count in line 2)") from None
    if count != len(rows):
        raise ValueError(f"{path.name}: declares {count} rows and holds {len(rows)}")
    if not rows:
        return np.empty(0, dtype=np.int64)
    if len(rows[-1].split()) != len(rows[0].split()):
        # the row count cannot show a file that was cut inside its last row
        raise ValueError(f"{path.name}: the last row is incomplete")
    try:
        # comments=None: a row that starts with "#" is a row, and fails as one
        indices = np.loadtxt(rows, dtype=np.int64, usecols=[0], ndmin=1, comments=None)
    except ValueError as error:
        raise ValueError(f"{path.name}: not a label file ({error})") from None
    if indices.min() < 0:
        raise ValueError(f"{path.name}: holds a negative vertex index ({indices.min()})")
    return np.unique(indices)


def _check_layout(path: Path) -> None:
    """Raise ValueError unless the file is a whole annotation that nibabel reads as meant.

    The layout, as ``nibabel.freesurfer.read_annot`` reads it, every number a big-endian
    int32: the number of vertices, that many pairs (vertex number, annotation value), a flag
    that says a colour table follows, then the table.
    """
    content = path.read_bytes()
    if len(content) < _INT.itemsize or (n_vertices := _int_at(content, 0)) < 0:
        raise ValueError(f"{path.name}: not an annotation file (no vertex count)")
    records_end = _INT.itemsize * (1 + 2 * n_vertices)
    if len(content) < records_end:
        raise ValueError(
            f"{path.name}: declares {n_vertices} vertices and ends before the last of them"
        )
    numbers = np.frombuffer(content, _INT, 2 * n_vertices, offset=_INT.itemsize)[0::2]
    if not np.array_equal(numbers, np.arange(n_vertices)):
        raise ValueError(f"{path.name}: the vertex records are not in vertex order")
    if len(content) < records_end + _INT.itemsize or _int_at(content, records_end) == 0:
        raise ValueError(f"{path.name}: the annotation has no colour table")
    try:
        _check_table(content, records_end + _INT.itemsize)
    except ValueError as error:
        raise ValueError(f"{path.name}: the colour table cannot be read ({error})") from None


def _check_table(content: bytes, offset: int) -> None:
    """Walk the colour table that starts at ``offset``; ValueError says what is wrong.

    The two layouts of ``nibabel.freesurfer.io`` (5.5.0). The present one, in the real files
    that were looked at (FreeSurfer 6.0.1 subject, fsaverage): -2, the number of indices,
    the name of the original table, the number of entries, and per entry its index, its
    name and four colour values. The older one: a positive number of entries, the name of
    the original table, and per entry its name and four colour values. A name is its length
    followed by that many bytes.
    """

    def integer() -> int:
        nonlocal offset
        if offset + _INT.itemsize > len(content):
            raise ValueError("the file ends inside it")
        offset += _INT.itemsize
        return _int_at(content, offset - _INT.itemsize)

    def skip(length: int) -> None:
        nonlocal offset
        if length < 0:
            raise ValueError(f"it holds a length of {length}")
        if offset + length > len(content):
            raise ValueError("the file ends inside it")
        offset += length

    colour = 4 * _INT.itemsize
    n_entries = integer()
    if n_entries > 0:
        skip(integer())
        for _ in range(n_entries):
            skip(integer())
            skip(colour)
        return
    if n_entries != -2:
        raise ValueError(f"table version {-n_entries} is not known")
    n_indices = integer()
    skip(integer())
    n_entries = integer()
    if n_entries != n_indices:
        raise ValueError(
            f"{n_entries} entries for {n_indices} indices; a table with gaps is not supported"
        )
    for position in range(n_entries):
        if (index := integer()) != position:
            raise ValueError(f"entry {position} has the index {index}")
        skip(integer())
        skip(colour)


def _int_at(content: bytes, offset: int) -> int:
    return int.from_bytes(content[offset : offset + _INT.itemsize], "big", signed=True)


def _table_index(
    values: Int[np.ndarray, " n"], codes: Int[np.ndarray, " m"]
) -> Int[np.ndarray, " n"]:
    """For each annotation value the index of the first entry of ``codes`` equal to it; -1
    for the value 0 and for a value that ``codes`` does not hold."""
    if len(codes) == 0:
        return np.full(len(values), -1, dtype=np.int32)
    values, codes = values.astype(np.int64), codes.astype(np.int64)
    # a stable sort and a left-side search give the first of equal codes
    order = np.argsort(codes, kind="stable")
    ranked = codes[order]
    position = np.minimum(np.searchsorted(ranked, values), len(codes) - 1)
    known = (ranked[position] == values) & (values != 0)
    return np.where(known, order[position], -1).astype(np.int32)
