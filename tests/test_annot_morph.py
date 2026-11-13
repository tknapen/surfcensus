"""fsio.annot and fsio.morph: annotations, label files and per-vertex data.

Files are written with nibabel where it has a writer and by hand where it has none (label
files) or where the case is a damaged file. Two tests (six cases) read the real subjects from the
pytest cache and skip when they are absent.
"""

from __future__ import annotations

import struct
import warnings
from collections.abc import Sequence
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest
from synth_template import (  # pyrefly: ignore[missing-import]
    DESIKAN_NAMES,
    write_synthetic_template,
)

from surfcensus.fsio.annot import NON_CORTEX_NAMES, Annotation, read_annotation, read_label
from surfcensus.fsio.morph import read_morph

NAMES = ("unknown", "cuneus", "corpuscallosum", "lingual", "insula")
COLOURS = np.array(
    [[25, 5, 25, 0], [220, 180, 140, 0], [120, 70, 50, 0], [225, 140, 140, 0], [255, 192, 32, 0]]
)
LABELS = np.array([1, 3, -1, 0, 4, 2, 1, 1, -1, 3], dtype=np.int32)


def _write_annot(
    path: Path,
    labels: np.ndarray = LABELS,
    names: Sequence[str | bytes] = NAMES,
    colours: np.ndarray = COLOURS,
) -> Path:
    fs.write_annot(str(path), np.asarray(labels, dtype=np.int32), colours, list(names))
    return path


def _packed(colours: np.ndarray) -> np.ndarray:
    """The annotation value of each colour: R + G * 2^8 + B * 2^16."""
    return colours[:, 0] + colours[:, 1] * 2**8 + colours[:, 2] * 2**16


def _records_end(n_vertices: int) -> int:
    """Offset of the colour table flag: after the count and one (vertex, value) pair per
    vertex, each a big-endian int32."""
    return 4 + 8 * n_vertices


def _set_int(content: bytes, offset: int, value: int) -> bytes:
    return content[:offset] + struct.pack(">i", value) + content[offset + 4 :]


def _write_label(path: Path, vertices: Sequence[int], *, count: int | None = None) -> Path:
    rows = [
        f"{vertex}  {1.5 * i:.3f}  -2.000  3.250 0.0000000000" for i, vertex in enumerate(vertices)
    ]
    declared = len(rows) if count is None else count
    path.write_text(
        "\n".join(["#!ascii label  , from subject x vox2ras=TkReg", str(declared), *rows]) + "\n"
    )
    return path


# --- read_annotation ----------------------------------------------------------------------


def test_an_annotation_reads_as_int32_indices_and_decoded_names(tmp_path: Path) -> None:
    annotation = read_annotation(_write_annot(tmp_path / "lh.aparc.annot"))
    assert isinstance(annotation, Annotation)
    np.testing.assert_array_equal(annotation.labels, LABELS)
    # nibabel hands out big-endian int32 and names as bytes
    assert annotation.labels.dtype == np.int32 and annotation.labels.dtype.isnative
    assert annotation.names == NAMES
    assert all(type(name) is str for name in annotation.names)


def test_a_vertex_without_a_label_is_minus_one(tmp_path: Path) -> None:
    labels = np.array([-1, -1, 2, -1], dtype=np.int32)
    annotation = read_annotation(_write_annot(tmp_path / "a.annot", labels))
    np.testing.assert_array_equal(annotation.labels, [-1, -1, 2, -1])
    # in the file such a vertex has the annotation value 0
    stored = np.fromfile(tmp_path / "a.annot", ">i4", 9)[1:].reshape(4, 2)
    np.testing.assert_array_equal(stored[:, 1] == 0, labels == -1)


def test_a_path_may_be_a_string(tmp_path: Path) -> None:
    path = _write_annot(tmp_path / "a.annot")
    np.testing.assert_array_equal(read_annotation(str(path)).labels, LABELS)


def test_names_are_decoded_as_utf8_and_a_bad_byte_does_not_stop_the_reading(
    tmp_path: Path,
) -> None:
    names = ["unknown", "gyrus_é", b"caf\xe9", "lingual", "insula"]
    annotation = read_annotation(_write_annot(tmp_path / "a.annot", names=names))
    assert annotation.names == ("unknown", "gyrus_é", "caf�", "lingual", "insula")
    np.testing.assert_array_equal(annotation.mask("gyrus_é"), LABELS == 1)


def test_random_annotations_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261113)
    for case in range(20):
        n_names = int(rng.integers(1, 40))
        # distinct colours, none black: an annotation value of 0 means "no label"
        packed = rng.choice(np.arange(1, 2**24), size=n_names, replace=False)
        colours = np.column_stack(
            [packed % 256, packed // 256 % 256, packed // 65536, np.zeros(n_names, dtype=int)]
        )
        labels = rng.integers(-1, n_names, size=int(rng.integers(0, 300))).astype(np.int32)
        names = [f"parcel{i}" for i in range(n_names)]
        path = _write_annot(tmp_path / f"{case}.annot", labels, names, colours)
        annotation = read_annotation(path)
        np.testing.assert_array_equal(annotation.labels, labels)
        assert annotation.names == tuple(names)


def test_an_annotation_without_a_colour_table_is_a_value_error(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    end = _records_end(len(LABELS))
    assert struct.unpack(">i", content[end : end + 4]) == (1,)

    flagged = tmp_path / "lh.flagged.annot"
    flagged.write_bytes(content[:end] + struct.pack(">i", 0))
    with pytest.raises(
        ValueError, match=r"^lh\.flagged\.annot: the annotation has no colour table$"
    ):
        read_annotation(flagged)

    # the same with the table still in the file: the flag decides
    flag_only = tmp_path / "flag_only.annot"
    flag_only.write_bytes(_set_int(content, end, 0))
    with pytest.raises(ValueError, match="no colour table"):
        read_annotation(flag_only)

    ends = tmp_path / "ends.annot"
    ends.write_bytes(content[:end])
    with pytest.raises(ValueError, match="no colour table"):
        read_annotation(ends)


@pytest.mark.parametrize(
    ("cut", "message"),
    [
        (0, "no vertex count"),
        (3, "no vertex count"),
        (4, "declares 10 vertices and ends before the last of them"),
        (_records_end(len(LABELS)) - 3, "declares 10 vertices and ends before the last of them"),
        (_records_end(len(LABELS)) + 2, "no colour table"),
        (_records_end(len(LABELS)) + 4, r"the colour table cannot be read \(the file ends"),
        (_records_end(len(LABELS)) + 30, r"the colour table cannot be read \(the file ends"),
        # inside the colour of the last entry: nibabel 5.5.0 reads such a file without an error
        (-10, r"^cut\.annot: the colour table cannot be read \(the file ends inside it\)$"),
        (-1, r"the colour table cannot be read \(the file ends inside it\)"),
    ],
)
def test_an_annotation_that_ends_early_is_a_value_error(
    tmp_path: Path, cut: int, message: str
) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    damaged = tmp_path / "cut.annot"
    damaged.write_bytes(content[:cut])
    with pytest.raises(ValueError, match=message):
        read_annotation(damaged)


def test_a_file_that_is_no_annotation_is_a_value_error(tmp_path: Path) -> None:
    text = tmp_path / "notes.annot"
    text.write_text("this is not an annotation file\n" * 4)
    with pytest.raises(ValueError, match=r"^notes\.annot: declares \d+ vertices and ends"):
        read_annotation(text)

    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    negative = tmp_path / "negative.annot"
    negative.write_bytes(_set_int(content, 0, -3))
    with pytest.raises(ValueError, match="no vertex count"):
        read_annotation(negative)

    # the number after the flag is -2 for the table format that FreeSurfer 6 and 7 write
    version = _records_end(len(LABELS)) + 4
    assert struct.unpack(">i", content[version : version + 4]) == (-2,)
    other = tmp_path / "version.annot"
    other.write_bytes(_set_int(content, version, -3))
    with pytest.raises(
        ValueError, match=r"the colour table cannot be read \(table version 3 is not known\)"
    ):
        read_annotation(other)
    other.write_bytes(_set_int(content, version, 0))
    with pytest.raises(ValueError, match=r"table version 0 is not known"):
        read_annotation(other)


@pytest.mark.parametrize("layout", ["present", "older"])
def test_an_annotation_cut_at_any_byte_is_a_value_error(tmp_path: Path, layout: str) -> None:
    if layout == "present":
        content = _write_annot(tmp_path / "ok.annot").read_bytes()
    else:
        content = _older_layout()
    path = tmp_path / "lh.aparc.annot"
    path.write_bytes(content)
    np.testing.assert_array_equal(read_annotation(path).labels, LABELS)
    for cut in range(len(content)):
        path.write_bytes(content[:cut])
        with pytest.raises(ValueError, match=r"^lh\.aparc\.annot: "):
            read_annotation(path)
    # bytes after the table are not read
    path.write_bytes(content + b"\x00\x01\x02")
    np.testing.assert_array_equal(read_annotation(path).labels, LABELS)


def test_an_absent_annotation_is_an_os_error_not_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_annotation(tmp_path / "lh.aparc.annot")
    with pytest.raises(IsADirectoryError):
        read_annotation(tmp_path)


def test_a_value_error_names_the_file_and_not_its_path(tmp_path: Path) -> None:
    folder = tmp_path / "ds999999" / "sub-CANARY7" / "label"
    folder.mkdir(parents=True)
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    damaged = {
        "no table": content[: _records_end(len(LABELS))],
        "cut records": content[:20],
        "cut table": content[:-10],
        "gap": _set_int(content, _records_end(len(LABELS)) + 8, len(NAMES) + 1),
    }
    for case, data in damaged.items():
        (folder / "lh.aparc.annot").write_bytes(data)
        with pytest.raises(ValueError) as info:
            read_annotation(folder / "lh.aparc.annot")
        text = str(info.value)
        assert text.startswith("lh.aparc.annot: "), case
        assert "CANARY7" not in text and "ds999999" not in text and str(tmp_path) not in text, case


def test_a_value_that_the_colour_table_lacks_is_no_label(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    n = len(LABELS)
    records = np.frombuffer(content[4 : _records_end(n)], dtype=">i4").reshape(n, 2).copy()
    codes = np.sort(_packed(COLOURS))
    between = int(codes[1] + 1)
    assert codes[1] < between < codes[2]
    # above every entry, between two entries, below every entry
    records[0, 1], records[1, 1], records[4, 1] = int(codes[-1] + 5), between, 7
    foreign = tmp_path / "foreign.annot"
    foreign.write_bytes(content[:4] + records.tobytes() + content[_records_end(n) :])

    expected = LABELS.copy()
    expected[[0, 1, 4]] = -1
    np.testing.assert_array_equal(read_annotation(foreign).labels, expected)


def test_a_black_entry_cannot_be_told_from_no_label(tmp_path: Path) -> None:
    """A vertex of a black parcel is stored with the value 0, like a vertex without a label.
    It reads as -1, as with nibabel, and never as the black entry."""
    colours = COLOURS.copy()
    colours[0] = 0
    labels = np.array([0, 1, -1, 0, 4], dtype=np.int32)
    path = _write_annot(tmp_path / "a.annot", labels, colours=colours)
    annotation = read_annotation(path)
    np.testing.assert_array_equal(annotation.labels, [-1, 1, -1, -1, 4])
    np.testing.assert_array_equal(annotation.labels, fs.read_annot(str(path))[0])
    assert annotation.names == NAMES and not annotation.mask("unknown").any()


def test_of_two_entries_with_one_colour_the_first_is_taken(tmp_path: Path) -> None:
    colours = COLOURS.copy()
    colours[3] = colours[1]
    labels = np.array([1, 3, 4, 0, -1], dtype=np.int32)
    annotation = read_annotation(_write_annot(tmp_path / "a.annot", labels, colours=colours))
    np.testing.assert_array_equal(annotation.labels, [1, 1, 4, 0, -1])
    assert annotation.names == NAMES

    # also when the entry that comes second in the table sorts first
    colours = COLOURS.copy()
    colours[0] = colours[4]
    annotation = read_annotation(_write_annot(tmp_path / "b.annot", labels, colours=colours))
    np.testing.assert_array_equal(annotation.labels, [1, 3, 0, 0, -1])


def test_vertex_records_out_of_order_are_refused(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    n = len(LABELS)
    records = np.frombuffer(content[4 : _records_end(n)], dtype=">i4").reshape(n, 2)
    np.testing.assert_array_equal(records[:, 0], np.arange(n))
    swapped = records.copy()
    swapped[[2, 5]] = swapped[[5, 2]]
    shuffled = tmp_path / "shuffled.annot"
    shuffled.write_bytes(content[:4] + swapped.tobytes() + content[_records_end(n) :])
    with pytest.raises(
        ValueError, match=r"^shuffled\.annot: the vertex records are not in vertex order$"
    ):
        read_annotation(shuffled)


def test_a_colour_table_with_gaps_is_refused(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    # after the flag and the version: the highest index plus one, which nibabel sizes the table by
    size = _records_end(len(LABELS)) + 8
    assert struct.unpack(">i", content[size : size + 4]) == (len(NAMES),)
    gapped = tmp_path / "gapped.annot"
    gapped.write_bytes(_set_int(content, size, len(NAMES) + 2))
    with pytest.raises(
        ValueError, match="5 entries for 7 indices; a table with gaps is not supported"
    ):
        read_annotation(gapped)


def _first_entry(content: bytes) -> int:
    """Offset of the first entry of the colour table in a file that nibabel wrote: after the
    flag, the version, the number of indices, the name of the original table and the number
    of entries."""
    name_length = _records_end(len(LABELS)) + 12
    (length,) = struct.unpack(">i", content[name_length : name_length + 4])
    return name_length + 4 + length + 4


def test_colour_table_entries_out_of_index_order_are_refused(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    first = _first_entry(content)
    index, name_length = struct.unpack(">ii", content[first : first + 8])
    assert (index, name_length) == (0, len("unknown") + 1)
    assert content[first + 8 : first + 8 + name_length] == b"unknown\x00"
    second = first + 8 + name_length + 16
    assert struct.unpack(">i", content[second : second + 4]) == (1,)

    # nibabel puts the colour at the index and the name at the position in the file
    path = tmp_path / "permuted.annot"
    path.write_bytes(_set_int(_set_int(content, first, 1), second, 0))
    with pytest.raises(
        ValueError, match=r"the colour table cannot be read \(entry 0 has the index 1\)"
    ):
        read_annotation(path)
    # the same index twice: nibabel would leave one row of the table without a colour
    path.write_bytes(_set_int(content, second, 0))
    with pytest.raises(ValueError, match=r"entry 1 has the index 0"):
        read_annotation(path)
    path.write_bytes(_set_int(content, second, len(NAMES)))
    with pytest.raises(ValueError, match=r"entry 1 has the index 5"):
        read_annotation(path)


def test_a_negative_length_in_the_colour_table_is_refused(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    path = tmp_path / "a.annot"
    path.write_bytes(_set_int(content, _first_entry(content) + 4, -8))
    with pytest.raises(
        ValueError, match=r"the colour table cannot be read \(it holds a length of -8\)"
    ):
        read_annotation(path)


def _older_layout() -> bytes:
    """An annotation with the older colour table, built by hand from the reading rule of
    ``nibabel.freesurfer.io`` (5.5.0): a positive entry count after the flag, the name of
    the original table, then name and colour per entry."""

    def text(value: bytes) -> bytes:
        return struct.pack(">i", len(value) + 1) + value + b"\x00"

    values = np.where(LABELS >= 0, _packed(COLOURS)[LABELS], 0)
    records = np.column_stack([np.arange(len(LABELS)), values]).astype(">i4")
    table = b"".join(
        text(name.encode()) + struct.pack(">4i", *colour)
        for name, colour in zip(NAMES, COLOURS.tolist(), strict=True)
    )
    return (
        struct.pack(">i", len(LABELS))
        + records.tobytes()
        + struct.pack(">ii", 1, len(NAMES))
        + text(b"colortable.txt")
        + table
    )


def test_the_older_colour_table_layout_is_read_as_nibabel_reads_it(tmp_path: Path) -> None:
    """No real file of this layout was looked at; the real annotations have -2 where this
    one has its entry count."""
    path = tmp_path / "old.annot"
    path.write_bytes(_older_layout())
    annotation = read_annotation(path)
    np.testing.assert_array_equal(annotation.labels, LABELS)
    assert annotation.labels.dtype == np.int32 and annotation.labels.dtype.isnative
    assert annotation.names == NAMES


def test_a_colour_table_without_entries_names_nothing(tmp_path: Path) -> None:
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    size = _records_end(len(LABELS)) + 8
    first = _first_entry(content)
    path = tmp_path / "a.annot"
    path.write_bytes(_set_int(_set_int(content[:first], size, 0), first - 4, 0))
    annotation = read_annotation(path)
    assert annotation.names == ()
    np.testing.assert_array_equal(annotation.labels, np.full(len(LABELS), -1))
    assert annotation.labels.dtype == np.int32
    assert not annotation.cortex().any()


def test_an_annotation_of_no_vertices(tmp_path: Path) -> None:
    # nibabel does not write one: the table of a written file after a count of 0
    content = _write_annot(tmp_path / "ok.annot").read_bytes()
    path = tmp_path / "a.annot"
    path.write_bytes(struct.pack(">i", 0) + content[_records_end(len(LABELS)) :])
    annotation = read_annotation(path)
    assert annotation.labels.shape == (0,) and annotation.labels.dtype == np.int32
    assert annotation.names == NAMES
    assert annotation.cortex().shape == (0,)


# --- Annotation.mask and Annotation.cortex ------------------------------------------------


def test_mask_is_the_union_of_the_named_parcels() -> None:
    annotation = Annotation(labels=LABELS, names=NAMES)
    cuneus = annotation.mask("cuneus")
    assert cuneus.dtype == np.bool_ and cuneus.shape == LABELS.shape
    np.testing.assert_array_equal(cuneus, LABELS == 1)
    np.testing.assert_array_equal(
        annotation.mask("lingual", "cuneus"), (LABELS == 1) | (LABELS == 3)
    )
    np.testing.assert_array_equal(annotation.mask("cuneus", "cuneus"), LABELS == 1)
    # index 0 is a parcel like any other, and is not the "no label" of -1
    np.testing.assert_array_equal(annotation.mask("unknown"), LABELS == 0)
    assert not annotation.mask().any() and annotation.mask().shape == LABELS.shape


def test_mask_of_a_parcel_without_vertices_is_empty_not_an_error() -> None:
    annotation = Annotation(labels=np.array([1, 1, -1], dtype=np.int32), names=NAMES)
    assert not annotation.mask("lingual").any()
    np.testing.assert_array_equal(annotation.mask("lingual", "cuneus"), [True, True, False])


def test_mask_with_an_unknown_name_is_a_key_error_that_names_each_of_them() -> None:
    annotation = Annotation(labels=LABELS, names=NAMES)
    with pytest.raises(KeyError) as info:
        annotation.mask("cuneus", "precuneus", "lingual", "fusiform", "precuneus")
    assert info.value.args == ("the annotation has no parcel 'precuneus', 'fusiform'",)
    with pytest.raises(KeyError, match="'Cuneus'"):
        annotation.mask("Cuneus")  # names are compared as they are
    with pytest.raises(KeyError, match="'cune'"):
        annotation.mask("cune")


def test_mask_covers_every_entry_that_carries_the_name() -> None:
    annotation = Annotation(labels=np.array([0, 1, 2, -1], dtype=np.int32), names=("a", "b", "a"))
    np.testing.assert_array_equal(annotation.mask("a"), [True, False, True, False])


def test_cortex_is_every_labelled_vertex_outside_unknown_and_corpus_callosum() -> None:
    assert NON_CORTEX_NAMES == ("unknown", "corpuscallosum")
    annotation = Annotation(labels=LABELS, names=NAMES)
    cortex = annotation.cortex()
    assert cortex.dtype == np.bool_
    # labels 1, 3, -1, 0 (unknown), 4, 2 (corpuscallosum), 1, 1, -1, 3
    np.testing.assert_array_equal(
        cortex, [True, True, False, False, True, False, True, True, False, True]
    )
    np.testing.assert_array_equal(
        cortex, (LABELS >= 0) & ~annotation.mask("unknown") & ~annotation.mask("corpuscallosum")
    )


def test_cortex_of_an_annotation_without_the_two_names_is_every_labelled_vertex() -> None:
    labels = np.array([0, -1, 1, 2, -1], dtype=np.int32)
    annotation = Annotation(labels=labels, names=("cuneus", "lingual", "Unknown"))
    np.testing.assert_array_equal(annotation.cortex(), labels >= 0)


# --- read_label ---------------------------------------------------------------------------


def test_label_indices_come_back_sorted_as_int64(tmp_path: Path) -> None:
    indices = read_label(_write_label(tmp_path / "lh.cortex.label", [88, 3, 4000, 17, 0, 2**31]))
    assert indices.dtype == np.int64 and indices.ndim == 1
    np.testing.assert_array_equal(indices, [0, 3, 17, 88, 4000, 2**31])
    np.testing.assert_array_equal(read_label(str(tmp_path / "lh.cortex.label")), indices)


def test_a_label_with_one_row_or_none(tmp_path: Path) -> None:
    one = read_label(_write_label(tmp_path / "one.label", [7]))
    assert one.dtype == np.int64 and one.tolist() == [7]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numpy warns about input without data
        none = read_label(_write_label(tmp_path / "none.label", []))
    assert none.dtype == np.int64 and none.shape == (0,)


def test_a_vertex_listed_twice_is_returned_once(tmp_path: Path) -> None:
    indices = read_label(_write_label(tmp_path / "twice.label", [5, 2, 5, 9, 2]))
    np.testing.assert_array_equal(indices, [2, 5, 9])


def test_a_label_as_freesurfer_writes_it(tmp_path: Path) -> None:
    """The first lines of the real ``fsaverage/label/lh.cortex.label`` (149,955 rows, read
    2026-11-13), with a count of 3: two spaces before each coordinate, one before the last
    field."""
    path = tmp_path / "lh.cortex.label"
    path.write_text(
        "#label , from subject ??\n3\n"
        "88  -42.261  -81.724  -13.242 0.000000\n"
        "445  -28.781  -85.827  -16.289 0.000000\n"
        "446  -39.862  -74.518  -14.432 0.000000\n"
    )
    np.testing.assert_array_equal(read_label(path), [88, 445, 446])


def test_blank_lines_and_a_first_line_that_is_not_utf8_do_not_matter(tmp_path: Path) -> None:
    path = tmp_path / "lh.cortex.label"
    path.write_bytes(b"#label , from subject caf\xe9\n2\n9 0 0 0 0\n\n4 0 0 0 0\n\n   \n")
    np.testing.assert_array_equal(read_label(path), [4, 9])
    # without a line end after the last row
    path.write_bytes(b"#label\n2\n9 0 0 0 0\n4 0 0 0 0")
    np.testing.assert_array_equal(read_label(path), [4, 9])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("#!ascii label\n3\n7 0 0 0 0\n9 0 0 0 0\n", "declares 3 rows and holds 2"),
        ("#!ascii label\n1\n7 0 0 0 0\n9 0 0 0 0\n", "declares 1 rows and holds 2"),
        ("#!ascii label\n0\n7 0 0 0 0\n", "declares 0 rows and holds 1"),
        ("#!ascii label\n2\n7 0.5 0.5 0.5 0\n9 0.5 0.", "the last row is incomplete"),
        ("#!ascii label\n2\n7 0.5 0.5 0.5 0\n9", "the last row is incomplete"),
        ("#!ascii label\n2\n7 0 0 0 0\n9.5 0 0 0 0\n", "not a label file"),
        ("#!ascii label\n2\nseven 0 0 0 0\n9 0 0 0 0\n", "not a label file"),
        ("#!ascii label\n2\n7 0 0 0 0\n# 9 0 0 0\n", "not a label file"),
        ("#!ascii label\n7 0 0 0 0\n9 0 0 0 0\n", r"not a label file \(no row count in line 2\)"),
        ("#!ascii label\n", r"not a label file \(no row count in line 2\)"),
        ("", r"not a label file \(no row count in line 2\)"),
        ("#!ascii label\n2\n7 0 0 0 0\n-1 0 0 0 0\n", r"holds a negative vertex index \(-1\)"),
    ],
)
def test_a_damaged_label_is_a_value_error_that_names_the_file(
    tmp_path: Path, content: str, message: str
) -> None:
    folder = tmp_path / "sub-CANARY7"
    folder.mkdir()
    path = folder / "lh.cortex.label"
    path.write_text(content)
    with pytest.raises(ValueError, match=message) as info:
        read_label(path)
    assert str(info.value).startswith("lh.cortex.label: ")
    assert "CANARY7" not in str(info.value)


def test_an_absent_label_is_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_label(tmp_path / "lh.cortex.label")


# --- the synthetic template ---------------------------------------------------------------


def test_the_files_of_a_synthetic_template_read_as_its_masks_say(tmp_path: Path) -> None:
    template = write_synthetic_template(tmp_path, order=3)
    for hemi in ("lh", "rh"):
        masks = template.masks[hemi]
        annotation = read_annotation(template.path / "label" / f"{hemi}.aparc.annot")
        assert annotation.names == DESIKAN_NAMES
        assert annotation.labels.shape == (template.n_vertices,)
        np.testing.assert_array_equal(annotation.labels == -1, masks["medial_wall"])
        np.testing.assert_array_equal(annotation.cortex(), masks["cortex"])
        parcels = [name for name in masks if name in DESIKAN_NAMES]
        assert len(parcels) > 3
        for name in parcels:
            np.testing.assert_array_equal(annotation.mask(name), masks[name])
        for name in set(DESIKAN_NAMES) - set(parcels):
            assert not annotation.mask(name).any()

        label_file = template.path / "label" / f"{hemi}.cortex.label"
        listed = fs.read_label(str(label_file))
        assert (np.diff(listed) < 0).all()  # the fixture writes it in descending order
        indices = read_label(label_file)
        np.testing.assert_array_equal(indices, np.flatnonzero(masks["cortex"]))


# --- read_morph ---------------------------------------------------------------------------


def test_morph_data_come_back_as_native_float32(tmp_path: Path) -> None:
    values = np.array([0.0, 2.5, 5.0, 4.9, 1.234567, 3.0], dtype=np.float32)
    path = tmp_path / "lh.thickness"
    fs.write_morph_data(str(path), values)
    # on disk: three bytes ff, three big-endian int32 (the first is the count), the values
    content = path.read_bytes()
    assert content[:3] == b"\xff\xff\xff" and struct.unpack(">i", content[3:7]) == (6,)
    assert content[15:] == values.astype(">f4").tobytes()
    assert fs.read_morph_data(str(path)).dtype == np.dtype(">f4")

    data = read_morph(path)
    assert data.dtype == np.float32 and data.dtype.isnative
    assert data.shape == (6,) and data.flags.c_contiguous
    np.testing.assert_array_equal(data, values)
    np.testing.assert_array_equal(read_morph(str(path)), values)


def test_morph_data_of_no_vertices(tmp_path: Path) -> None:
    path = tmp_path / "lh.thickness"
    fs.write_morph_data(str(path), np.empty(0, dtype=np.float32))
    data = read_morph(path)
    assert data.shape == (0,) and data.dtype == np.float32


def test_special_values_pass_through(tmp_path: Path) -> None:
    values = np.array([np.nan, np.inf, -1.0, 0.0], dtype=np.float32)
    path = tmp_path / "lh.thickness"
    fs.write_morph_data(str(path), values)
    np.testing.assert_array_equal(read_morph(path), values)


@pytest.mark.parametrize(
    ("cut", "message"),
    [
        (0, r"not a morphometry file \(no header\)"),
        (2, r"not a morphometry file \(no header\)"),
        (3, r"not a morphometry file \(the header ends early\)"),
        (6, r"not a morphometry file \(the header ends early\)"),
        (7, "declares 10 values and holds 0"),
        (15, "declares 10 values and holds 0"),
        (15 + 4 * 8, "declares 10 values and holds 8"),
        (-1, "declares 10 values and holds 9"),
    ],
)
def test_morph_data_that_end_early_are_a_value_error(
    tmp_path: Path, cut: int, message: str
) -> None:
    whole = tmp_path / "whole"
    fs.write_morph_data(str(whole), np.linspace(1, 4, 10, dtype=np.float32))
    folder = tmp_path / "sub-CANARY7"
    folder.mkdir()
    damaged = folder / "rh.thickness"
    damaged.write_bytes(whole.read_bytes()[:cut])
    with pytest.raises(ValueError, match=message) as info:
        read_morph(damaged)
    assert str(info.value).startswith("rh.thickness: ")
    assert "CANARY7" not in str(info.value)


def test_morph_data_cut_at_any_byte_are_a_value_error(tmp_path: Path) -> None:
    whole = tmp_path / "whole"
    fs.write_morph_data(str(whole), np.linspace(1, 4, 10, dtype=np.float32))
    content = whole.read_bytes()
    assert len(content) == 15 + 4 * 10
    path = tmp_path / "lh.thickness"
    for cut in range(len(content)):
        path.write_bytes(content[:cut])
        with pytest.raises(ValueError, match=r"^lh\.thickness: "):
            read_morph(path)


def test_a_header_that_declares_more_or_fewer_values_than_the_file_holds(tmp_path: Path) -> None:
    whole = tmp_path / "whole"
    fs.write_morph_data(str(whole), np.linspace(1, 4, 10, dtype=np.float32))
    content = whole.read_bytes()
    path = tmp_path / "lh.thickness"

    path.write_bytes(content[:3] + struct.pack(">i", 2**31 - 1) + content[7:])
    with pytest.raises(ValueError, match="declares 2147483647 values and holds 10"):
        read_morph(path)
    # numpy reads to the end of the file for a negative count, so nibabel returns 10 values
    path.write_bytes(content[:3] + struct.pack(">i", -5) + content[7:])
    with pytest.raises(ValueError, match=r"not a morphometry file \(declares -5 values\)"):
        read_morph(path)
    # fewer than the file holds: the declared number is read, as nibabel does
    path.write_bytes(content[:3] + struct.pack(">i", 4) + content[7:])
    np.testing.assert_array_equal(read_morph(path), np.linspace(1, 4, 10, dtype=np.float32)[:4])


def test_a_text_file_is_no_morph_data(tmp_path: Path) -> None:
    path = tmp_path / "lh.thickness"
    path.write_text("an error page, not thickness\n" * 3)
    assert fs.read_morph_data(str(path)).size > 0  # nibabel reads numbers out of it
    with pytest.raises(ValueError, match=r"^lh\.thickness: declares \d+ values and holds \d+$"):
        read_morph(path)


def test_absent_morph_data_are_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_morph(tmp_path / "lh.thickness")
    with pytest.raises(IsADirectoryError):
        read_morph(tmp_path)


def test_the_older_layout_is_read_as_nibabel_reads_it(tmp_path: Path) -> None:
    """Built by hand from the reading rule of ``nibabel.freesurfer.read_morph_data`` (5.5.0):
    three bytes of vertex count, three of face count, then int16 hundredths. No real file
    of this layout was looked at."""
    hundredths = np.array([250, 100, 0, 500, 317], dtype=">i2")
    path = tmp_path / "lh.curv"
    path.write_bytes((5).to_bytes(3, "big") + (8).to_bytes(3, "big") + hundredths.tobytes())
    data = read_morph(path)
    assert data.dtype == np.float32 and data.dtype.isnative
    np.testing.assert_array_equal(data, np.array([2.5, 1.0, 0.0, 5.0, 3.17], dtype=np.float32))

    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="declares 5 values and holds 4"):
        read_morph(path)


# --- the real subjects, when the pytest cache has them ------------------------------------

REAL_THICKNESS = {
    # subject, hemisphere: vertices, at exactly 5.0, at 4.9 or more, zeros (read 2026-11-13)
    ("ds000102-sub-01", "lh"): (148_332, 16, 28, 5_919),
    ("ds000102-sub-01", "rh"): (146_947, 41, 61, 5_875),
    ("ds003020-sub-UTS01", "lh"): (131_402, 4, 12, 8_086),
    ("ds003020-sub-UTS01", "rh"): (132_122, 3, 10, 7_911),
}


@pytest.mark.parametrize(("subject", "hemi"), sorted(REAL_THICKNESS))
def test_the_thickness_of_a_real_subject(cached_subjects: Path, subject: str, hemi: str) -> None:
    path = cached_subjects / subject / "surf" / f"{hemi}.thickness"
    if not path.is_file():
        pytest.skip(f"{subject} is not in the pytest cache (the network tests fetch it)")
    n, at_max, near_max, zeros = REAL_THICKNESS[subject, hemi]
    data = read_morph(path)
    assert data.dtype == np.float32 and data.dtype.isnative and data.shape == (n,)
    assert path.stat().st_size == 15 + 4 * n
    assert np.isfinite(data).all() and data.min() == 0.0
    assert data.max() == 5.0  # clamped at the maximum, in FreeSurfer 6.0.1 and in 7.3.2
    assert (int((data == 5.0).sum()), int((data >= 4.9).sum()), int((data == 0).sum())) == (
        at_max,
        near_max,
        zeros,
    )
    np.testing.assert_array_equal(data, fs.read_morph_data(str(path)))


REAL_ANNOT = {
    # hemisphere: vertices, without a label, in the four occipital parcels (read 2026-11-13)
    "lh": (148_332, 8_027, 16_429),
    "rh": (146_947, 8_178, 16_313),
}


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_the_annotation_of_a_real_subject(cached_subjects: Path, hemi: str) -> None:
    subject = cached_subjects / "ds000102-sub-01"
    path = subject / "label" / f"{hemi}.aparc.annot"
    if not path.is_file():
        pytest.skip("ds000102 sub-01 is not in the pytest cache (the network tests fetch it)")
    n, unlabelled, occipital = REAL_ANNOT[hemi]
    annotation = read_annotation(path)
    assert annotation.labels.dtype == np.int32 and annotation.labels.dtype.isnative
    assert annotation.labels.shape == (n,)
    assert annotation.names == DESIKAN_NAMES  # 36 names, "corpuscallosum" at index 4
    assert int((annotation.labels == -1).sum()) == unlabelled
    assert annotation.labels.max() == 35
    # no vertex of this subject carries one of the two names that are not cortex
    assert not annotation.mask(*NON_CORTEX_NAMES).any()
    assert int(annotation.cortex().sum()) == n - unlabelled
    assert (
        int(annotation.mask("cuneus", "lateraloccipital", "lingual", "pericalcarine").sum())
        == occipital
    )
    # every value of this file is in its colour table, so nibabel's mapping gives the same
    np.testing.assert_array_equal(annotation.labels, fs.read_annot(str(path))[0])
    # one label per thickness value
    thickness = subject / "surf" / f"{hemi}.thickness"
    if thickness.is_file():
        assert read_morph(thickness).shape == annotation.labels.shape
