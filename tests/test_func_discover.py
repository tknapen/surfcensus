"""``func.discover``: the functional source of a dataset of a run, found by rule from the
dataset record the surface source stored, and the derivative and the raw dataset as places
that are only read (docs/epi_interface.md, "Which folders are found by rule" and "Step 2: the
plan"; decision D35 extended to the functional side).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    SyntheticFuncDataset,
    write_func_dataset,
    write_review_folder,
)

from surfcensus.config import ConfigError, EpiConfig, new_local_config
from surfcensus.func import api, discover
from surfcensus.sources.base import DatasetRecord, SourceError
from surfcensus.sources.local import (
    LocalSource,
    detect_layout,
    functional_places,
    refuse_in_sources,
    refuse_run_in_sources,
)


def _record(path: Path) -> DatasetRecord:
    return next(iter(LocalSource(path).datasets()))


@pytest.fixture
def folder(tmp_path: Path) -> SyntheticFuncDataset:
    return write_review_folder(tmp_path)


# --- the layouts ------------------------------------------------------------------------------


def test_a_bids_root_finds_the_derivative_that_holds_the_subjects(
    folder: SyntheticFuncDataset,
) -> None:
    deriv, raw, fs = folder.deriv_root, folder.raw_root, folder.fs_subjects_dir
    # the dataset root, the derivative and the subjects directory are one source
    for start in (raw, raw / "derivatives", deriv, fs):
        found = discover.func_source_of(_record(start))
        assert found is not None, start
        assert Path(found.deriv_root).resolve() == deriv.resolve(), start
        assert found.raw_root is not None and Path(found.raw_root).resolve() == raw.resolve()
        assert (found.deriv_found, found.raw_found) == ("surface layout", "dataset root"), start
        assert found.deriv_identity == str(deriv.resolve())
        assert (found.connector, found.match.pipeline_version) == ("fmriprep", "21.0.2")


def test_a_stored_root_gives_the_same_source_without_a_search(
    folder: SyntheticFuncDataset,
) -> None:
    # what a run stores is the folder the surface source found; it is read again with the
    # rules above only (nested=False), never searched below derivatives/ (decision D35)
    stored = _record(folder.raw_root)
    again, _ = detect_layout(stored.root, nested=False, quiet=True)
    assert discover.func_source_of(again) == discover.func_source_of(stored)
    # a second derivative beside it changes nothing: the one that holds the subjects is taken
    other = folder.raw_root / "derivatives" / "aaa-fmriprep"
    shutil.copytree(folder.deriv_root, other, ignore=shutil.ignore_patterns("sourcedata"))
    found = discover.func_source_of(again)
    assert found is not None and Path(found.deriv_root) == folder.deriv_root


def test_the_folder_itself_is_the_derivative(tmp_path: Path) -> None:
    ds = write_func_dataset(tmp_path, raw_inside=False)
    # FreeSurfer subjects in freesurfer/ inside the derivative: a plain BIDS derivatives
    # layout whose root is itself an fMRIPrep output
    shutil.move(str(ds.fs_subjects_dir), str(ds.deriv_root / "freesurfer"))
    (ds.deriv_root / "sourcedata").rmdir()
    record = _record(ds.deriv_root)
    found = discover.func_source_of(record)
    assert found is not None
    assert (Path(found.deriv_root), found.deriv_found) == (ds.deriv_root, "folder itself")
    assert (found.raw_root, found.raw_found) == (None, None)
    given = discover.func_source_of(record, ds.raw_root)
    assert given is not None
    assert (given.raw_root, given.raw_found) == (str(ds.raw_root), "given")


def test_a_derivative_under_another_name_in_derivatives(tmp_path: Path) -> None:
    ds = write_func_dataset(tmp_path)
    derivatives = ds.raw_root / "derivatives"
    shutil.move(str(ds.fs_subjects_dir), str(derivatives / "freesurfer"))
    (ds.deriv_root / "sourcedata").rmdir()
    renamed = derivatives / "fmriprep-21.0.2"
    shutil.move(str(ds.deriv_root), str(renamed))
    record = _record(ds.raw_root)
    assert record.layout == "bids_derivatives"
    found = discover.func_source_of(record)
    assert found is not None
    assert (Path(found.deriv_root), found.deriv_found) == (renamed, "derivatives folder")
    assert (Path(str(found.raw_root)), found.raw_found) == (ds.raw_root, "dataset root")


def test_the_raw_dataset_in_sourcedata(tmp_path: Path) -> None:
    ds = write_func_dataset(tmp_path, raw_inside=False)
    shutil.move(str(ds.raw_root), str(ds.deriv_root / "sourcedata" / "raw"))
    found = discover.func_source_of(_record(ds.deriv_root))
    assert found is not None
    assert found.raw_root == str(ds.deriv_root / "sourcedata" / "raw")
    assert found.raw_found == "sourcedata/raw"


def test_no_functional_part_and_a_raw_that_is_not(
    tmp_path: Path, folder: SyntheticFuncDataset
) -> None:
    plain = tmp_path / "plain"
    shutil.copytree(folder.fs_subjects_dir, plain)
    assert discover.func_source_of(_record(plain)) is None
    record = _record(folder.raw_root)
    with pytest.raises(SourceError) as err:
        discover.func_source_of(record, folder.deriv_root / "sub-01")
    assert err.value.reason == "layout" and "--raw" in err.value.detail
    with pytest.raises(SourceError) as err:
        discover.func_source_of(record, tmp_path / "nowhere")
    assert err.value.reason == "not found"


def test_step_1_uses_the_same_rules() -> None:
    # one implementation: func.api imports the rules from here
    assert api.find_folders is discover.find_folders
    assert api.FuncFolders is discover.FuncFolders
    assert api.refuse_out is discover.refuse_out


# --- the places that are only read ----------------------------------------------------------


def test_a_run_directory_in_a_participant_folder_is_refused_and_beside_them_not(
    tmp_path: Path, folder: SyntheticFuncDataset
) -> None:
    # the participant folders of the derivative and of the raw dataset are read; the roots
    # are not refused themselves, so that a run directory inside a dataset (asked for with
    # --run-dir, or made there by an earlier version) works as decision D35 holds (E2.Z)
    fs = folder.fs_subjects_dir
    for inside in (
        folder.raw_root / "sub-01" / "surfcensus-run",
        folder.raw_root / "sub-02",
        folder.deriv_root / "sub-01" / "run",
        folder.deriv_root / "sub-02" / "ses-1" / "run",
    ):
        with pytest.raises(ConfigError) as err:
            refuse_in_sources(inside, [fs], what="run")
        assert "only read" in err.value.detail, inside
    for allowed in (
        tmp_path / "surfcensus-run",  # beside the dataset
        folder.raw_root / "surfcensus-run",  # inside the dataset root
        folder.deriv_root / "surfcensus-run",  # inside the derivative, beside sourcedata/
        folder.raw_root / "derivatives" / "surfcensus-run",
    ):
        refuse_in_sources(allowed, [fs], what="run")
    whats = {what.split(" of ")[-1] for what, _ in functional_places(fs)}
    assert whats == {"the derivative", "the raw dataset"}
    names = {place.name for _, place in functional_places(fs)}
    assert names == {"sub-01", "sub-02"}


def test_epi_session_still_refuses_an_out_anywhere_in_the_folders(
    folder: SyntheticFuncDataset,
) -> None:
    folders = discover.find_folders(folder.raw_root)
    for inside in (
        folder.raw_root / "out",
        folder.deriv_root / "out",
        folder.fs_subjects_dir / "out",
    ):
        with pytest.raises(ConfigError):
            discover.refuse_out(inside, folders)


def test_a_participant_link_of_the_raw_dataset_is_a_place(
    tmp_path: Path, folder: SyntheticFuncDataset
) -> None:
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "inside").mkdir(parents=True)
    (folder.raw_root / "sub-09").symlink_to(elsewhere)
    other = tmp_path / "other"
    (other / "inside").mkdir(parents=True)
    (folder.raw_root / "extra").symlink_to(other)  # not a participant folder
    with pytest.raises(ConfigError):
        refuse_in_sources(elsewhere / "inside", [folder.fs_subjects_dir], what="output")
    refuse_in_sources(other / "inside", [folder.fs_subjects_dir], what="output")
    source = discover.func_source_of(_record(folder.raw_root))
    assert source is not None
    # named by its target, as a linked subject is (review of step 2, A-F1)
    assert (
        "the directory that sub-09 in the raw dataset links to,",
        elsewhere.resolve(),
    ) in discover.read_places(source)
    for written in (elsewhere / "inside", folder.raw_root / "sub-09" / "inside"):
        with pytest.raises(ConfigError) as err:
            refuse_in_sources(written, [folder.fs_subjects_dir], what="output")
        assert str(elsewhere.resolve()) in err.value.detail


def test_the_commands_that_read_config_yaml_refuse_the_raw_dataset_given(
    tmp_path: Path,
) -> None:
    ds = write_func_dataset(tmp_path / "data", raw_inside=False)
    config = new_local_config(ds.deriv_root)
    # without --raw the raw dataset beside the derivative is not one of its places
    refuse_run_in_sources(ds.raw_root / "sub-01" / "run", config)
    with pytest.raises(ConfigError):
        refuse_run_in_sources(ds.deriv_root / "sub-01" / "run", config)
    refuse_run_in_sources(ds.deriv_root / "run", config)  # the root itself is not refused
    named = config.model_copy(update={"epi": EpiConfig(raw=str(ds.raw_root))})
    with pytest.raises(ConfigError) as err:
        refuse_run_in_sources(ds.raw_root / "sub-01" / "run", named)
    assert "the raw dataset" in err.value.detail
    refuse_run_in_sources(ds.raw_root / "run", named)
    places = discover.stored_read_places(named)
    assert ("sub-01 of the raw dataset", ds.raw_root / "sub-01") in places
    assert ("sub-01 of the derivative", ds.deriv_root / "sub-01") in places


def test_a_folder_without_functional_data_adds_no_place(tmp_path: Path) -> None:
    # an fMRIPrep output without sub-*/func is no derivative of a connector: the rule of a
    # run directory beside the subjects inside it holds as before
    ds = write_func_dataset(tmp_path / "data", raw_inside=False)
    for func in ds.deriv_root.glob("sub-*/**/func"):
        shutil.rmtree(func)
    assert functional_places(ds.fs_subjects_dir) == []
    refuse_in_sources(ds.deriv_root / "sourcedata" / "run", [ds.fs_subjects_dir], what="run")
