"""What the modules of step 1 of the EPI interface share (docs/epi_interface.md, Build order):
the names that the packages E1.A to E1.Z import from each other, with their parameters, the
fields of the records and of the JSON files of a pack, the file roles, the settings, and the
layout of a pack's files.

The names were fixed before the functions were written (work package E1.0), so that the
packages can be written side by side. A package that renames or reorders something fails
here; a change that is meant goes here as well, in the same change.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import pickle
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
from test_imports import PACKAGE_DIR, private_names  # pyrefly: ignore[missing-import]

from surfcensus import func, records
from surfcensus.config import EPI_PACK_FIELDS, EPI_SESSION_FIELDS, STAGE_SCHEMAS, EpiConfig
from surfcensus.func import model
from surfcensus.func.connectors import CONNECTORS, connector
from surfcensus.func.connectors.base import Connector, ConnectorError
from surfcensus.func.itk import ItkError
from surfcensus.func.pack import PackError
from surfcensus.records import IDENTITY4, Grid, to_matrix4
from surfcensus.sources.base import (
    FUNC_RAW_ROLES,
    FUNC_READ_ROLES,
    FUNC_SESSION_ROLES,
    PACK_REQUIRED,
    FuncRole,
)

INTERFACE: dict[str, dict[str, tuple[str, ...] | None]] = {
    # module -> name -> parameter names of a function, or None for a class or a constant
    "surfcensus.func": {
        "PACK_FORMAT": None,
        "PACK_VERSION": None,
        "SESSION_JSON": None,
        "PACK_JSON": None,
        "TRACES_JSON": None,
        "CONTOURS_DIR": None,
        "STREAMS": None,
        "STILLS": None,
        "PLANES": None,
        "SURFACES": None,
        "CONTOUR_UNIT": None,
        "ID_PATTERN": None,
        "GRID_KEY_PATTERN": None,
        "chunk_name": ("index",),
        "stream_file": ("run", "stream", "name"),
        "contour_file": ("grid_key", "plane"),
        "is_pack_path": ("path",),
        "func_id": ("key", "parts"),
    },
    "surfcensus.func.bids": {
        "ENTITIES": None,
        "ORDER_ENTITIES": None,
        "RAW_SUFFIXES": None,
        "SIDECAR_SUFFIX": None,
        "parse_entities": ("name",),
        "run_index": ("label",),
        "run_key": ("entities",),
        "name_order": ("entities",),
        "run_name": ("entities",),
        "acquisition_seconds": ("value",),
        "sidecar_time": ("path",),
        "is_raw_dataset": ("folder",),
        "list_raw_runs": ("raw_root", "participant"),
        "session_labels": ("runs",),
    },
    "surfcensus.func.itk": {
        "LPS": None,
        "TRANSFORM_KINDS": None,
        "ItkError": None,
        "read_itk": ("text",),
        "read_itk_file": ("path",),
    },
    "surfcensus.func.connectors": {
        "CONNECTORS": None,
        "connector": ("name",),
        "detect": ("folder",),
    },
    "surfcensus.func.connectors.base": {
        "CONNECTOR_REASONS": None,
        "ConnectorError": None,
        "Connector": None,
    },
    "surfcensus.func.connectors.fmriprep": {
        "NAME": None,
        "SCANNER_T1W": None,
        "BOLDREF_T1W": None,
        "FIRST_BOLDREF_VERSION": None,
        "ANAT_SPACE": None,
        "ANAT_SPACES": None,
        "FUNC_SUFFIXES": None,
        "NATIVE_SUFFIXES": None,
        "FIGURE_SUFFIXES": None,
        "FSNATIVE_SUFFIX": None,
        "REPORT_LABELS": None,
        "MOTION_SOURCE": None,
        "CONFOUND_COLUMNS": None,
        "NON_STEADY_PREFIX": None,
        "generation_of": ("version",),
        "read_summary_report": ("text", "sdc_figure_listed"),
        "read_confounds": ("path",),
        "FmriprepConnector": None,
        "make": (),
    },
    "surfcensus.func.session": {
        "RULE_TIME": None,
        "RULE_NAME": None,
        "AFFINE_HEADER": None,
        "AFFINE_QFORM": None,
        "AFFINE_REFERENCE": None,
        "AFFINE_UNCHECKED": None,
        "AFFINE_TOLERANCE": None,
        "FRAME_LIMIT_MM": None,
        "pick_anchor": ("runs",),
        "acquisition_order": ("runs", "anchor"),
        "raw_affine_choice": ("raw_bold", "reference"),
        "frame_displacement": ("run_matrix", "anchor_matrix", "mask", "mask_affine", "limit_mm"),
        "check_frame": ("transforms", "anchor", "masks", "limit_mm"),
        "plan_session": (
            "participant",
            "session",
            "deriv_runs",
            "raw_runs",
            "connector",
            "deriv_root",
            "raw_root",
            "run_ids",
            "fs_subject",
            "frame_limit_mm",
        ),
    },
    "surfcensus.func.geometry": {
        "GRID_TOLERANCE": None,
        "SPLINE_ORDER": None,
        "compose": ("matrices",),
        "inverse": ("matrix",),
        "grid_of": ("series",),
        "same_grid": ("a", "b"),
        "voxel_mm": ("grid",),
        "oblique_deg": ("grid",),
        "edge_letters": ("grid",),
        "voxel_map": ("raw_affine", "anchor_matrix", "grid", "display_to_anat", "portable"),
        "resample_voxels": ("volume", "vox", "shape", "order"),
        "raw_stream": (
            "raw_bold",
            "raw_affine",
            "grid",
            "anchor_matrix",
            "display_to_anat",
            "volumes",
            "portable",
        ),
    },
    "surfcensus.func.contours": {
        "surfaces_in_display": ("fs_subject_dir", "anat_to_fsnative", "grid", "display_to_anat"),
        "plane_contours": ("surfaces", "grid"),
        "read_contours": ("data",),
        "chain": ("segments",),
    },
    "surfcensus.func.pack": {
        "ZLIB_LEVEL": None,
        "MAX_CHUNK": None,
        "WINDOW_STEP": None,
        "PERCENTILE": None,
        "PackError": None,
        "quantise": ("volume", "lo", "hi"),
        "volume_bytes": ("volume",),
        "chunk_volumes": ("n_voxels", "chunk_bytes"),
        "series_volumes": ("series", "volumes"),
        "session_windows": ("runs",),
        "pack_run": ("session_dir", "run", "config", "window", "sd_window"),
        "write_contours": ("session_dir", "grid_key", "planes"),
        "pack_session": ("out_dir", "session_id", "plan", "runs", "contours", "config"),
        "content_digest": ("session_dir", "info"),
    },
    "surfcensus.func.packread": {
        "read_session": ("session_dir",),
        "read_pack": ("session_dir", "run"),
        "pack_on_disk": ("session_dir", "run"),
        "decode_volume": ("session_dir", "info", "stream", "t"),
        "decode_still": ("session_dir", "info", "stream", "still"),
        "decode_contours": ("session_dir", "grid_key", "plane"),
    },
    "surfcensus.func.model": {
        "READING": None,
        "RULES": None,
        "PackFile": None,
        "ChunkFile": None,
        "StreamFiles": None,
        "PackInput": None,
        "PackSettings": None,
        "StatementInfo": None,
        "CorrectionsInfo": None,
        "corrections_info": ("corrections",),
        "FrameInfo": None,
        "ContourFile": None,
        "GridInfo": None,
        "RegistrationInfo": None,
        "SessionRunInfo": None,
        "SessionInfo": None,
        "TracesInfo": None,
        "PackInfo": None,
        "info_text": ("info",),
        "read_info": ("path", "model"),
    },
    "surfcensus.func.api": {
        "FuncFolders": None,
        "find_folders": ("path", "raw"),
        "epi_session": ("path", "subject", "out", "session", "raw", "config", "key"),
    },
    "surfcensus.records": {
        "Matrix4": None,
        "IDENTITY4": None,
        "to_matrix4": ("rows",),
        "PACK_REASONS": None,
        "DISPLAY_REASONS": None,
    },
}

RECORDS: dict[str, tuple[str, ...]] = {
    "RunEntities": ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo"),
    "FuncFileRef": ("role", "base", "location", "size", "etag_or_mtime", "variant"),
    "RawRun": ("entities", "key", "order", "stem", "files"),
    "DerivRun": ("entities", "key", "order", "stem", "spaces", "files", "echoes"),
    "ConnectorMatch": ("connector", "pipeline_version", "generation"),
    "Display": ("layout", "space", "reason"),
    "RigidPath": ("matrix", "dof", "shear_scale", "rigid", "source", "reference"),
    "Motion": ("fd", "dvars", "non_steady", "source"),
    "Statement": ("state", "words", "source", "note"),
    "Corrections": (
        "motion",
        "sdc",
        "sdc_method",
        "slice_timing",
        "registration",
        "dof",
        "sdc_figure_listed",
    ),
    "RawAffine": ("matrix", "choice", "reference_image"),
    "AnchorCandidate": ("run", "order", "acquisition_time"),
    "Anchor": ("run", "rule", "why"),
    "FrameDisplacement": (
        "mean_mm",
        "max_mm",
        "rotation_deg",
        "translation_mm",
        "translation_ras",
        "shear_scale",
        "n_voxels",
        "flagged",
    ),
    "FrameCheck": (
        "anchor",
        "limit_mm",
        "runs",
        "flagged",
        "shared_frame_plausible",
        "anchor_is_outlier",
    ),
    "Grid": ("shape", "affine"),
    "PlaneContours": ("plane", "data", "slices", "lines", "points"),
    "SessionRun": ("run", "name", "deriv", "raw", "display", "reason", "missing"),
    "SessionPlan": (
        "participant",
        "session",
        "fs_subject",
        "runs",
        "anchor",
        "frame",
        "grids",
        "grid_of",
        "files",
    ),
    "RunPackInput": (
        "run",
        "name",
        "files",
        "grid",
        "display_to_anat",
        "anchor_run",
        "anchor_matrix",
        "raw_affine",
        "motion",
        "corrections",
        "rigid",
        "tr_s",
    ),
}

MODELS: dict[str, tuple[str, ...]] = {
    # func/model.py: what session.json, pack.json and traces.json hold
    "SessionInfo": (
        "format",
        "version",
        "surfcensus_version",
        "session",
        "label",
        "config_hash",
        "fd_threshold_mm",
        "registration",
        "window",
        "sd_window",
        "grids",
        "runs",
        "contours_from",
        "rules",
    ),
    "RegistrationInfo": (
        "anchor",
        "rule",
        "limit_mm",
        "shared_frame_plausible",
        "anchor_is_outlier",
        "portable_voxel_map",
        "reading",
    ),
    "GridInfo": (
        "grid_key",
        "shape",
        "voxel_mm",
        "oblique_deg",
        "edge_letters",
        "contours",
        "slice_axis",
        "slice_mm",
    ),
    "SessionRunInfo": (
        "run",
        "name",
        "acquisition_order",
        "anchor",
        "grid_key",
        "frame",
        "reason",
        "n_volumes",
        "tr_s",
        "first",
    ),
    "FrameInfo": (
        "mean_mm",
        "max_mm",
        "rotation_deg",
        "translation_mm",
        "translation_ras",
        "shear_scale",
        "flagged",
    ),
    "PackInfo": (
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
        "chunk_volumes",
        "stills_volumes",
        "raw_affine",
        "reference_image",
        "rigid",
        "corrections",
        "streams",
        "traces",
        "content_digest",
    ),
    "PackSettings": (
        "portable_voxel_map",
        "chunk_bytes",
        "window_step",
        "percentile",
        "zlib_level",
        "contour_unit",
    ),
    "PackInput": ("role", "base", "size", "sha256"),
    "PackFile": ("file", "bytes", "sha256"),
    "ChunkFile": ("file", "bytes", "sha256", "first", "count"),
    "ContourFile": ("file", "bytes", "sha256", "slices", "lines", "points"),
    "StreamFiles": ("ref", "mean", "sd", "chunks"),
    "TracesInfo": ("fd", "dvars", "non_steady"),
    "StatementInfo": ("state", "words", "source", "note"),
    "CorrectionsInfo": RECORDS["Corrections"],
}

STEP_1_MODULES = (
    "func/__init__.py",
    "func/bids.py",
    "func/itk.py",
    "func/session.py",
    "func/geometry.py",
    "func/contours.py",
    "func/model.py",
    "func/pack.py",
    "func/packread.py",
    "func/api.py",
    "func/connectors/__init__.py",
    "func/connectors/base.py",
    "func/connectors/fmriprep.py",
)


@pytest.mark.parametrize("module", sorted(INTERFACE))
def test_the_shared_names_exist_with_their_parameters(module: str) -> None:
    loaded = import_module(module)
    for name, parameters in INTERFACE[module].items():
        assert hasattr(loaded, name), f"{module}.{name}"
        if parameters is not None:
            found = tuple(inspect.signature(getattr(loaded, name)).parameters)
            assert found == parameters, f"{module}.{name}"


def test_the_records_keep_their_fields() -> None:
    for name, fields in RECORDS.items():
        cls = getattr(records, name)
        assert tuple(f.name for f in dataclasses.fields(cls)) == fields, name
        assert cls.__dataclass_params__.frozen, name  # pyrefly: ignore[missing-attribute]


def test_the_json_files_of_a_pack_keep_their_fields() -> None:
    for name, fields in MODELS.items():
        assert tuple(getattr(model, name).model_fields) == fields, name


def test_the_file_roles_of_a_functional_run() -> None:
    assert [role.value for role in FuncRole] == [
        "raw_bold",
        "raw_sidecar",
        "pre_bold",
        "pre_sidecar",
        "pre_mask",
        "xfm_to_anat",
        "reference",
        "confounds",
        "summary_report",
        "validation_report",
        "sdc_figure",
        "xfm_anat_fsnative",
    ]
    assert FUNC_RAW_ROLES == (FuncRole.RAW_BOLD, FuncRole.RAW_SIDECAR)
    assert FUNC_SESSION_ROLES == (FuncRole.XFM_ANAT_FSNATIVE,)
    assert set(PACK_REQUIRED) <= set(FUNC_READ_ROLES)
    assert FuncRole.SDC_FIGURE not in FUNC_READ_ROLES
    assert FuncRole.REFERENCE not in PACK_REQUIRED  # without it the raw affine is unchecked


def test_the_reasons_and_the_choices() -> None:
    assert records.PACK_REASONS == (
        "no raw series",
        "no preprocessed series",
        "standard space only",
        "native space only",
        "no transform to the anatomy",
        "no brain mask",
        "volume counts differ",
        "runs on different grids",
        "multi-echo",
        "source changed",
        "pack failure",
    )
    assert set(records.DISPLAY_REASONS) <= set(records.PACK_REASONS)
    from surfcensus.func import session

    assert (
        session.AFFINE_HEADER,
        session.AFFINE_QFORM,
        session.AFFINE_REFERENCE,
        session.AFFINE_UNCHECKED,
    ) == ("header", "qform", "reference", "header, unchecked")
    assert (session.RULE_TIME, session.RULE_NAME) == ("acquisition_time", "name_order")


def test_a_matrix_is_four_rows_of_python_floats() -> None:
    m = to_matrix4([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
    assert m[0] == (1.0, 0.0, 0.0, 2.0) and all(type(x) is float for row in m for x in row)
    assert to_matrix4(IDENTITY4) == IDENTITY4 and hash(m) == hash(to_matrix4(m))
    for wrong in (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        [[float("nan"), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    ):
        with pytest.raises(ValueError):
            to_matrix4(wrong)


def test_a_grid_key_follows_from_shape_and_affine() -> None:
    grid = Grid(
        (17, 18, 14), to_matrix4([[8, 0, 0, -64], [0, 8, 0, -86], [0, 0, 8, -36], [0, 0, 0, 1]])
    )
    assert grid.key == "e2babfc70bbc"  # pinned: the name of a contour folder on every machine
    assert Grid((17, 18, 15), grid.affine).key != grid.key
    assert func.contour_file(grid.key, "axial") == f"contours/{grid.key}/axial.bin"


def test_the_settings_of_the_epi_interface() -> None:
    config = EpiConfig()
    assert (
        config.enabled,
        config.runs,
        config.portable_voxel_map,
        config.frame_limit_mm,
        config.fd_threshold_mm,
        config.chunk_bytes,
        config.max_gb,
    ) == (True, (), True, 10.0, 0.5, 4 * 1024 * 1024, None)
    assert EPI_PACK_FIELDS == ("portable_voxel_map", "chunk_bytes")
    assert EPI_SESSION_FIELDS == ("frame_limit_mm", "fd_threshold_mm")
    assert STAGE_SCHEMAS["pack"] == 1 and STAGE_SCHEMAS["epi_session"] == 1
    # each hash follows the fields of its own and nothing else
    for name, value in (
        ("chunk_bytes", 1024),
        ("portable_voxel_map", False),
        ("frame_limit_mm", 5.0),
        ("fd_threshold_mm", 0.2),
        ("runs", ("*task-rest*",)),
        ("max_gb", 2.0),
        ("enabled", False),
    ):
        changed = config.model_copy(update={name: value})
        assert (changed.pack_hash() != config.pack_hash()) == (name in EPI_PACK_FIELDS), name
        assert (changed.session_hash() != config.session_hash()) == (name in EPI_SESSION_FIELDS), (
            name
        )
    assert config.pack_hash() != config.session_hash()
    with pytest.raises(ValueError):
        EpiConfig(frame_limit_mm=0)
    with pytest.raises(ValueError):
        EpiConfig(unknown=1)  # pyrefly: ignore[unexpected-keyword]


def test_the_files_of_a_pack_and_what_may_be_served() -> None:
    run = "0123456789abcdef"
    assert func.chunk_name(0) == "c0000.bin" and func.chunk_name(12345) == "c12345.bin"
    assert func.stream_file(run, "raw", "mean") == f"{run}/raw/mean.bin"
    assert func.stream_file(run, "pre", func.chunk_name(7)) == f"{run}/pre/c0007.bin"
    served = (
        "contours/0123456789ab/sagittal.bin",
        f"{run}/raw/ref.bin",
        f"{run}/pre/sd.bin",
        f"{run}/raw/c0000.bin",
        f"{run}/raw/c10000.bin",
    )
    refused = (
        "session.json",
        f"{run}/pack.json",
        f"{run}/traces.json",
        f"{run}/raw/c000.bin",
        f"{run}/raw/c01234.bin",
        f"{run}/raw/c\u0660\u0660\u0660\u0660.bin",
        f"{run}/both/mean.bin",
        f"{run.upper()}/raw/mean.bin",
        "contours/0123456789ab/oblique.bin",
        "contours/0123456789AB/axial.bin",
        f"../{run}/raw/mean.bin",
        f"{run}/raw/../raw/mean.bin",
        f"/{run}/raw/mean.bin",
    )
    assert all(func.is_pack_path(p) for p in served)
    assert not any(func.is_pack_path(p) for p in refused)
    for wrong in (("raw", "c000.bin"), ("both", "mean")):
        with pytest.raises(ValueError):
            func.stream_file(run, wrong[0], wrong[1])
    with pytest.raises(ValueError):
        func.chunk_name(-1)


def test_an_id_is_keyed() -> None:
    a = func.func_id(b"k" * 32, "sub-01", "1", "task-rest_run-1")
    assert a == func.func_id(b"k" * 32, "sub-01", "1", "task-rest_run-1")
    assert a != func.func_id(b"j" * 32, "sub-01", "1", "task-rest_run-1")
    assert a != func.func_id(b"k" * 32, "sub-01", "1task-rest_run-1")
    import re

    assert re.fullmatch(func.ID_PATTERN, a) and re.fullmatch(records.SUBJECT_UID_PATTERN, a)
    with pytest.raises(ValueError):
        func.func_id(b"", "x")


def test_the_connectors_and_the_errors() -> None:
    assert CONNECTORS == ("fmriprep",)
    made = connector("fmriprep")
    assert isinstance(made, Connector) and made.name == "fmriprep"
    with pytest.raises(KeyError):
        connector("xcp_d")
    for error in (ConnectorError("invalid", "x.tsv: no column"), PackError("no brain mask", "m")):
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error) and (copy.reason, copy.detail) == (
            error.reason,
            error.detail,
        )
    assert issubclass(ItkError, ValueError)


def test_the_json_files_are_written_one_way(tmp_path: Path) -> None:
    traces = model.TracesInfo(fd=(None, 0.25), dvars=(None, 1.5), non_steady=(0,))
    text = model.info_text(traces)
    expected = json.dumps(traces.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
    assert text == expected and text.startswith('{\n "dvars": [\n  null,\n  1.5\n ],')
    (tmp_path / "traces.json").write_text(text)
    assert model.read_info(tmp_path / "traces.json", model.TracesInfo) == traces
    assert model.read_info(tmp_path / "absent.json", model.TracesInfo) is None
    (tmp_path / "bad.json").write_text("{")
    assert model.read_info(tmp_path / "bad.json", model.TracesInfo) is None
    statement = records.Statement("applied", "Applied", FuncRole.SUMMARY_REPORT)
    unknown = records.Statement("unknown", None, None, "no summary report")
    corrections = records.Corrections(
        motion=records.Statement("applied", None, None, "by design"),
        sdc=unknown,
        sdc_method=None,
        slice_timing=statement,
        registration=unknown,
        dof=None,
        sdc_figure_listed=None,
    )
    info = model.corrections_info(corrections)
    assert info.slice_timing.source == "summary_report" and info.sdc.note == "no summary report"


@pytest.mark.parametrize("path", STEP_1_MODULES)
def test_the_modules_of_step_1_name_nothing_private(path: str) -> None:
    # decision D20: the functional code of step 1 is downstream of any inventory; the
    # inventory of step 2 (func/inventory.py, func/discover.py) will be private-side like
    # inventory/subjects.py and is not in this list
    source = (PACKAGE_DIR / path).read_text()
    assert private_names(source, signatures_too=True) == []


def test_the_light_modules_load_no_numpy() -> None:
    code = (
        "import sys\n"
        "import surfcensus.func, surfcensus.func.model, surfcensus.func.connectors\n"
        "import surfcensus.func.connectors.base\n"
        "print(sorted(m for m in ('numpy', 'nibabel', 'scipy', 'pandas') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
