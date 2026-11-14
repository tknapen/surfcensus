"""What the packages of step 2 of the EPI interface share (docs/epi_interface.md, "Step 2: the
plan"): the keyed identifiers with pinned values, the records the catalog, the functional
inventory, the pack phase and the app hand each other, the settings and their place in
``config.yaml``, the DDL of migration 2 with the columns of every table and view, the method
signatures on ``Catalog`` and ``PublicCatalog``, the modules ``func.discover``,
``func.inventory`` and ``func.batch``, the stubs on ``Run``, and the fixture of one folder with
both interfaces.

The names were fixed before the bodies were written (work package E2.0), so that E2.A to E2.D
can be written side by side. A package that renames or reorders something fails here; a change
that is meant goes here as well, in the same change. A stub that is written keeps its name and
parameters.
"""

from __future__ import annotations

import dataclasses
import inspect
import sqlite3
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
import yaml
from synth_func import func_counts, write_review_folder  # pyrefly: ignore[missing-import]
from test_imports import PACKAGE_DIR, private_names  # pyrefly: ignore[missing-import]

from surfcensus import config as cfg
from surfcensus import privacy, records, schemas, store
from surfcensus.config import EpiConfig, epi_settings, new_census_config, new_local_config
from surfcensus.records import FuncFileRef, func_signature
from surfcensus.sources.base import FileRef, FileRole, FuncRole

SALT = privacy.local_salt_from_hex("ab" * 32)
IDENTITY = "/study/ds/derivatives/fmriprep"

INTERFACE: dict[str, dict[str, tuple[str, ...] | None]] = {
    # module -> name -> parameter names of a function, or None for a class or a constant
    "surfcensus.privacy": {
        "participant_uid": ("key", "source", "identity", "connector", "participant"),
        "session_uid": ("key", "source", "identity", "connector", "participant", "session"),
        "run_uid": (
            "key",
            "source",
            "identity",
            "connector",
            "participant",
            "session",
            "run_key",
        ),
        "pack_key": ("run_secret", "pack_hash", "digests"),
    },
    "surfcensus.records": {
        "EPI_SESSION_PHASE": None,
        "FUNC_INVENTORY_PHASE": None,
        "PACK_PHASE": None,
        "session_unit": ("session_uid",),
        "func_signature": ("refs", "freesurfer", "extra"),
        "RawFound": None,
        "DerivFound": None,
    },
    "surfcensus.config": {
        "epi_settings": ("config",),
        "EPI_INVENTORY_FIELDS": None,
    },
    "surfcensus.schemas": {"PACK_STATES": None},
    "surfcensus.store": {
        "EPI_DIR": None,
        "_MIGRATION_2": None,
        "epi_participant_dir": ("run_dir", "participant_uid"),
        "epi_session_dir": ("run_dir", "participant_uid", "session_uid"),
    },
    "surfcensus.func.discover": {
        "func_source_of": ("record", "raw"),
        "read_places": ("source",),
        "stored_read_places": ("config",),
    },
    "surfcensus.func.inventory": {
        "run_func_inventory": ("catalog", "sources", "config", "key", "force"),
        "session_work": ("facts", "config"),
        "func_fetch_lines": ("catalog",),
    },
    "surfcensus.func.batch": {
        "PackPlan": None,
        "session_order": ("ctx",),
        "pack_plan": ("ctx", "sessions", "force", "retry_failed"),
        "pack_pending": ("ctx", "sessions", "force", "retry_failed"),
        "pack_status": ("ctx",),
        "pack_failures": ("ctx",),
        "pack_estimate": ("ctx",),
        "run_pack": ("ctx", "jobs", "sessions", "force", "retry_failed", "stop", "waiting"),
        "pack_session_unit": ("session_uid", "work", "epi_dir", "config", "keep"),
        "write_stream": ("ctx", "participant_uid"),
    },
}

METHODS: dict[str, dict[str, tuple[str, ...]]] = {
    # class -> method -> parameter names without self
    "Catalog": {
        "upsert_func_source": ("dataset_id", "record"),
        "func_source": ("dataset_id",),
        "upsert_func_participant": (
            "func_id",
            "participant_uid",
            "participant",
            "n_sessions",
            "session_order",
        ),
        "upsert_func_session": ("row",),
        "func_inventory_hash": ("session_uid",),
        "mark_func_absent": ("func_id", "present_participants", "present_sessions"),
        "relocate_func": ("func_id", "deriv_root", "raw_root", "raw_found"),
        "func_lease_facts": ("session_uid",),
        "func_participant_uids": (),
    },
    "PublicCatalog": {
        "has_func": (),
        "func_sources": (),
        "func_participants": (),
        "func_sessions": ("participant_uid",),
        "func_runs": ("session_uid",),
        "packs": (),
        "pack": ("run_uid",),
        "pack_signature": ("run_uid",),
        "epi_session_signature": ("session_uid",),
        "record_packs": ("rows",),
        "forget_packs": ("run_uids",),
        "set_func_digests": ("owner_kind", "owner", "digests"),
    },
    "Run": {
        "func_inventory": ("force",),
        "pack": ("jobs", "sessions", "force", "retry_failed", "stop", "waiting"),
    },
}

RECORDS: dict[str, tuple[str, ...]] = {
    "FuncSourceRecord": (
        "connector",
        "match",
        "deriv_root",
        "deriv_identity",
        "raw_root",
        "deriv_found",
        "raw_found",
    ),
    "FuncLeaseFacts": (
        "source",
        "participant_uid",
        "session_uid",
        "participant",
        "session",
        "run_ids",
        "fs_subjects_dir",
        "fs_subject",
        "subject_uid",
        "inventory_hash",
    ),
    "ContourJob": ("fs_dir", "fs_subject", "anat_to_fsnative"),
    "SessionWork": (
        "session_uid",
        "participant_uid",
        "plan",
        "inputs",
        "input_failures",
        "contours",
        "contour_note",
    ),
    "SessionResult": ("session_uid", "info", "packs", "kept", "failures", "seconds"),
    "PackRow": (
        "run_uid",
        "session_uid",
        "config_hash",
        "input_signature",
        "status",
        "pack_key",
        "reason",
        "detail",
        "path",
        "n_volumes",
        "bytes",
        "content_digest",
        "anchor_run_uid",
        "grid_key",
        "packed_at",
    ),
    "FuncRunRow": (
        "run_uid",
        "session_uid",
        "run_key",
        "name",
        "entities",
        "session_order",
        "acquisition_order",
        "anchor",
        "selected",
        "packable",
        "reason",
        "detail",
        "missing_roles",
        "n_volumes",
        "tr_s",
        "echoes",
        "display",
        "spaces",
        "grid",
        "raw_affine",
        "reference_image",
        "rigid",
        "dof",
        "corrections",
        "frame",
        "files",
    ),
    "FuncSessionRow": (
        "session_uid",
        "participant_uid",
        "session",
        "stream_order",
        "subject_uid",
        "fs_subject",
        "anchor",
        "anchor_note",
        "frame",
        "n_grids",
        "problem",
        "inventory_hash",
        "files",
        "runs",
    ),
    "PackPlan": (
        "pack_hash",
        "session_hash",
        "pending",
        "keep",
        "signatures",
        "skipped",
        "failed",
        "stale",
    ),
}

TABLES: dict[str, tuple[str, ...]] = {
    "func_sources": (
        "func_id",
        "dataset_id",
        "connector",
        "pipeline_version",
        "generation",
        "deriv_identity",
        "deriv_root",
        "deriv_found",
        "raw_root",
        "raw_found",
        "present",
        "listed_at",
    ),
    "func_participants": (
        "participant_uid",
        "func_id",
        "participant",
        "session_order",
        "n_sessions",
        "present",
        "listed_at",
    ),
    "func_sessions": (
        "session_uid",
        "participant_uid",
        "session_label",
        "stream_order",
        "subject_uid",
        "fs_subject",
        "anchor_run_uid",
        "anchor_rule",
        "anchor_why",
        "anchor_note",
        "frame_limit_mm",
        "shared_frame_plausible",
        "anchor_is_outlier",
        "n_runs",
        "n_packable",
        "n_grids",
        "problem",
        "present",
        "inventory_hash",
        "inventoried_at",
    ),
    "func_runs": (
        "run_uid",
        "session_uid",
        "run_key",
        "name",
        "entities",
        "session_order",
        "acquisition_order",
        "anchor",
        "selected",
        "packable",
        "reason",
        "detail",
        "missing_roles",
        "n_volumes",
        "tr_s",
        "echoes",
        "display_layout",
        "display_space",
        "spaces",
        "grid_key",
        "grid_shape",
        "voxel_mm",
        "oblique_deg",
        "raw_affine",
        "reference_image",
        "rigid",
        "dof",
        "corrections",
        "motion_state",
        "sdc",
        "sdc_method",
        "slice_timing",
        "frame_mean_mm",
        "frame_max_mm",
        "frame_rotation_deg",
        "frame_translation_mm",
        "flagged",
        "present",
        "inventoried_at",
    ),
    "func_files": (
        "owner_kind",
        "owner",
        "role",
        "base",
        "location",
        "variant",
        "size",
        "etag_or_mtime",
        "sha256",
    ),
    "packs": (
        "run_uid",
        "session_uid",
        "config_hash",
        "input_signature",
        "pack_key",
        "status",
        "reason",
        "detail",
        "path",
        "n_volumes",
        "bytes",
        "content_digest",
        "anchor_run_uid",
        "grid_key",
        "packed_at",
    ),
}

VIEWS: dict[str, tuple[str, ...]] = {
    "public_func_sources": (
        "dataset",
        "connector",
        "pipeline_version",
        "generation",
        "deriv_found",
        "raw_found",
        "present",
    ),
    "public_func_participants": (
        "participant_uid",
        "dataset",
        "participant",
        "session_order",
        "n_sessions",
        "present",
    ),
    "public_func_sessions": (
        "session_uid",
        "participant_uid",
        "session_label",
        "stream_order",
        "subject_uid",
        "fs_subject",
        "anchor_run_uid",
        "anchor_rule",
        "anchor_note",
        "frame_limit_mm",
        "shared_frame_plausible",
        "anchor_is_outlier",
        "n_runs",
        "n_packable",
        "n_grids",
        "problem",
        "present",
    ),
    "public_func_runs": (
        "run_uid",
        "session_uid",
        "name",
        "session_order",
        "acquisition_order",
        "anchor",
        "selected",
        "packable",
        "reason",
        "detail",
        "missing_roles",
        "n_volumes",
        "tr_s",
        "echoes",
        "display_layout",
        "grid_key",
        "grid_shape",
        "voxel_mm",
        "oblique_deg",
        "raw_affine",
        "reference_image",
        "rigid",
        "dof",
        "corrections",
        "motion_state",
        "sdc",
        "sdc_method",
        "slice_timing",
        "frame_mean_mm",
        "frame_max_mm",
        "frame_rotation_deg",
        "frame_translation_mm",
        "flagged",
        "present",
    ),
    "public_packs": (
        "run_uid",
        "session_uid",
        "config_hash",
        "pack_key",
        "status",
        "reason",
        "detail",
        "path",
        "n_volumes",
        "bytes",
        "anchor_run_uid",
        "grid_key",
        "packed_at",
    ),
}

PRIVATE_COLUMNS = {
    "deriv_root",
    "raw_root",
    "deriv_identity",
    "inventory_hash",
    "anchor_why",
    "entities",
    "location",
    "size",
    "etag_or_mtime",
    "sha256",
    "input_signature",
    "content_digest",
    "func_id",
    "dataset_id",
}
"""Columns that no public view may have (decision D20)."""


@pytest.mark.parametrize("module", sorted(INTERFACE))
def test_the_shared_names_exist_with_their_parameters(module: str) -> None:
    loaded = import_module(module)
    for name, parameters in INTERFACE[module].items():
        assert hasattr(loaded, name), f"{module}.{name}"
        if parameters is not None:
            found = tuple(inspect.signature(getattr(loaded, name)).parameters)
            assert found == parameters, f"{module}.{name}"


def test_the_methods_of_the_catalog_and_the_run() -> None:
    from surfcensus.run import Run

    classes = {"Catalog": store.Catalog, "PublicCatalog": store.PublicCatalog, "Run": Run}
    for cls_name, methods in METHODS.items():
        for name, parameters in methods.items():
            method = getattr(classes[cls_name], name)
            found = tuple(inspect.signature(method).parameters)[1:]
            assert found == parameters, f"{cls_name}.{name}"


def test_the_records_keep_their_fields() -> None:
    from surfcensus.func import batch

    for name, fields in RECORDS.items():
        cls = getattr(records, name, None) or getattr(batch, name)
        assert tuple(f.name for f in dataclasses.fields(cls)) == fields, name
        assert cls.__dataclass_params__.frozen, name  # pyrefly: ignore[missing-attribute]


def test_the_run_context_gains_a_functional_lease_and_a_pack_key() -> None:
    fields = [f.name for f in dataclasses.fields(records.RunContext)]
    assert fields[-2:] == ["func_lease", "pack_key"]
    by_name = {f.name: f for f in dataclasses.fields(records.RunContext)}
    assert by_name["func_lease"].default is None and by_name["pack_key"].default is None


def test_the_identifiers_are_keyed_and_pinned() -> None:
    p = privacy.participant_uid(SALT, "local", IDENTITY, "fmriprep", "sub-01")
    s1 = privacy.session_uid(SALT, "local", IDENTITY, "fmriprep", "sub-01", "1")
    s0 = privacy.session_uid(SALT, "local", IDENTITY, "fmriprep", "sub-02", None)
    r = privacy.run_uid(SALT, "local", IDENTITY, "fmriprep", "sub-01", "1", "ses-1_task-rest_run-1")
    digests = {"raw:raw_bold": "a" * 64, "deriv:pre_bold": "b" * 64}
    k = privacy.pack_key(b"\x07" * 32, "0123456789ab", digests)
    # pinned: these name directories and routes of every run made from now on
    assert (p, s1, s0, r, k) == (
        "56db11d1036b0100",
        "c5cabff260efb501",
        "243a931c9bf7f31a",
        "ba5d4d964eef70fa",
        "89d6ac5dfc286e5b",
    )
    # every identifier differs from a subject_uid of the same parts, and from the others
    assert p != privacy.subject_uid(SALT, "local", IDENTITY, "fmriprep", "sub-01")
    assert len({p, s1, s0, r}) == 4
    other = privacy.new_local_salt()
    assert privacy.participant_uid(other, "local", IDENTITY, "fmriprep", "sub-01") != p
    # a session without a label is the empty label; the run index enters as run_key gives it
    assert s0 == privacy.session_uid(SALT, "local", IDENTITY, "fmriprep", "sub-02", "")
    assert r != privacy.run_uid(
        SALT, "local", IDENTITY, "fmriprep", "sub-01", "1", "ses-1_task-rest_run-2"
    )
    # a pack key follows the settings and the digests, not their order
    assert k == privacy.pack_key(b"\x07" * 32, "0123456789ab", dict(reversed(digests.items())))
    assert k != privacy.pack_key(b"\x08" * 32, "0123456789ab", digests)
    assert k != privacy.pack_key(b"\x07" * 32, "0123456789ac", digests)


def test_a_functional_signature_follows_the_listing() -> None:
    ref = FuncFileRef(FuncRole.RAW_BOLD, "raw", "sub-01/func/x_bold.nii.gz", 10, "123")
    t1 = FileRef(FileRole.T1, "mri/T1.mgz", 5, "9")
    assert func_signature([ref]) == "38532efb9542ab9e"
    assert func_signature([ref], {FileRole.T1: t1}, ["anchor\tx"]) == "35974fec97ba5825"
    base = func_signature([ref])
    for changed in (
        dataclasses.replace(ref, size=11),
        dataclasses.replace(ref, etag_or_mtime="124"),
        dataclasses.replace(ref, location="sub-01/func/y_bold.nii.gz"),
        dataclasses.replace(ref, base="deriv"),
        dataclasses.replace(ref, role=FuncRole.PRE_BOLD),
        dataclasses.replace(ref, variant="desc-coreg_boldref"),
    ):
        assert func_signature([changed]) != base
    other = FuncFileRef(FuncRole.CONFOUNDS, "deriv", "c.tsv", 1, "2")
    assert func_signature([ref, other]) != func_signature([other, ref])  # the order is the caller's
    assert func_signature([ref], extra=["a"]) != base


def test_the_phases_and_the_states() -> None:
    assert records.EPI_SESSION_PHASE == "epi_session"
    assert records.FUNC_INVENTORY_PHASE == "func_inventory"
    assert records.PACK_PHASE == "pack"
    assert records.session_unit("0123456789abcdef") == "session:0123456789abcdef"
    assert schemas.PACK_STATES == (
        "packed",
        "pending",
        "failed",
        "cannot be packed",
        "not selected",
    )
    assert cfg.STAGE_SCHEMAS["func_inventory"] == 2  # raised for the session order (decision D42)


def test_the_settings_of_the_epi_interface_in_a_run(tmp_path: Path) -> None:
    local = new_local_config(tmp_path)
    assert local.epi is None and epi_settings(local) == EpiConfig()
    census = new_census_config(privacy.HashKey(b"\x01" * 32, "dataset-key"))
    assert epi_settings(census).enabled is False
    assert cfg.EPI_INVENTORY_FIELDS == ("runs", "frame_limit_mm")
    assert EpiConfig().inventory_hash() == "5ae1f2914378"  # pinned: a line of every session hash
    for name, value in (
        ("runs", ("*task-rest*",)),
        ("frame_limit_mm", 5.0),
        ("fd_threshold_mm", 0.2),
        ("chunk_bytes", 1024),
        ("raw", "/elsewhere"),
        ("enabled", False),
    ):
        changed = EpiConfig().model_copy(update={name: value})
        moved = changed.inventory_hash() != EpiConfig().inventory_hash()
        assert moved == (name in cfg.EPI_INVENTORY_FIELDS), name
        # the raw root is in no hash: it says where files are read, the listing is compared
        if name == "raw":
            assert changed.pack_hash() == EpiConfig().pack_hash()
            assert changed.session_hash() == EpiConfig().session_hash()
    with pytest.raises(cfg.ConfigError):
        cfg.with_changes(census, epi=EpiConfig())  # a census run has no EPI interface
    assert cfg.with_changes(census, epi=EpiConfig(enabled=False)).epi is not None


def test_a_config_without_epi_options_writes_the_bytes_of_before(tmp_path: Path) -> None:
    local = new_local_config(tmp_path)
    text = cfg.dump_config(local)
    assert "epi" not in yaml.safe_load(text)
    assert set(yaml.safe_load(text)) == set(type(local).model_fields) - {"epi"}
    cfg.write_config(tmp_path, local)
    assert cfg.load_config(tmp_path) == local
    given = cfg.with_changes(local, epi=EpiConfig(raw=str(tmp_path / "raw"), runs=("*rest*",)))
    data = yaml.safe_load(cfg.dump_config(given))
    assert data["epi"]["raw"] == str(tmp_path / "raw") and data["epi"]["runs"] == ["*rest*"]
    cfg.write_config(tmp_path, given)
    assert cfg.load_config(tmp_path) == given


def _catalog_with_both_migrations() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for migration in (store._MIGRATION_1, store._MIGRATION_2):
        for statement in store._statements(migration):
            conn.execute(statement)
    return conn


def test_migration_2_is_additive_and_has_the_columns_of_the_plan() -> None:
    statements = list(store._statements(store._MIGRATION_2))
    assert statements and all(s.startswith(("CREATE TABLE ", "CREATE VIEW ")) for s in statements)
    assert store.MIGRATIONS[0] == store._MIGRATION_1
    conn = _catalog_with_both_migrations()
    for table, columns in TABLES.items():
        found = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
        assert found == columns, table
    for view, columns in VIEWS.items():
        found = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({view})"))
        assert found == columns, view
        assert not set(found) & PRIVATE_COLUMNS, view


def test_the_public_views_hide_labels_outside_a_local_run() -> None:
    conn = _catalog_with_both_migrations()
    conn.execute("INSERT INTO meta VALUES ('mode', 'census')")
    conn.execute(
        "INSERT INTO datasets (dataset_id, pseudonym, source, identity, root, fs_root, layout, "
        "pipeline, discovered_at) VALUES (1, 'D-1', 'local', 'i', 'r', '.', 'fmriprep', "
        "'fmriprep', 't')"
    )
    conn.execute(
        "INSERT INTO func_sources (func_id, dataset_id, connector, generation, deriv_identity, "
        "deriv_root, deriv_found, listed_at) VALUES (1, 1, 'fmriprep', 'scanner-T1w', 'i', 'r', "
        "'surface layout', 't')"
    )
    conn.execute(
        "INSERT INTO func_participants VALUES ('0123456789abcdef', 1, 'sub-01', 'session label', "
        "1, 1, 't')"
    )
    rows = conn.execute("SELECT participant FROM public_func_participants").fetchall()
    assert rows == [(None,)]
    conn.execute("UPDATE meta SET value = 'local' WHERE key = 'mode'")
    rows = conn.execute("SELECT participant FROM public_func_participants").fetchall()
    assert rows == [("sub-01",)]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO func_files VALUES ('run', 'x', 'raw_bold', 'elsewhere', 'a', NULL, 1, "
            "'2', NULL)"
        )


def test_the_display_cache_lies_under_epi(tmp_path: Path) -> None:
    p, s = "0123456789abcdef", "fedcba9876543210"
    assert store.EPI_DIR == "epi"
    assert store.epi_participant_dir(tmp_path, p) == tmp_path / "epi" / p
    assert store.epi_session_dir(tmp_path, p, s) == tmp_path / "epi" / p / s
    for wrong in ("../x", "0123456789ABCDEF", "stream"):
        with pytest.raises(ValueError):
            store.epi_participant_dir(tmp_path, wrong)
        with pytest.raises(ValueError):
            store.epi_session_dir(tmp_path, p, wrong)


def test_the_pack_phase_names_nothing_private() -> None:
    # decision D20: func/batch.py is downstream of the inventory; it passes input signatures
    # back to the catalog and names nothing else of the private side
    source = (PACKAGE_DIR / "func" / "batch.py").read_text()
    assert private_names(source, signatures_too=False) == []


def test_the_new_modules_load_no_numpy() -> None:
    code = (
        "import sys\n"
        "import surfcensus.func.discover, surfcensus.func.inventory, surfcensus.func.batch\n"
        "print(sorted(m for m in ('numpy', 'nibabel', 'scipy') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_fixture_of_one_folder_with_both_interfaces(tmp_path: Path) -> None:
    folder = write_review_folder(tmp_path)
    counts = func_counts(folder)
    assert folder.raw_root == tmp_path / "ds"
    assert folder.deriv_root == tmp_path / "ds" / "derivatives" / "fmriprep"
    assert counts.participants == ("sub-01", "sub-02")
    assert counts.sessions == (("sub-01", "1"), ("sub-01", "2"), ("sub-02", None))
    assert (counts.runs, counts.packable) == (19, 11)
    assert sum(counts.reasons.values()) == counts.runs - counts.packable
    assert counts.fs_subjects == ("sub-01", "sub-02")
    for name in counts.fs_subjects:
        assert (folder.fs_subjects_dir / name / "mri" / "T1.mgz").is_file()

    alone = write_review_folder(tmp_path / "one", without_fs=("sub-02",))
    assert func_counts(alone).fs_subjects == ("sub-01",)
    assert not (alone.fs_subjects_dir / "sub-02").exists()
    assert func_counts(alone).packable == 11  # the functional data are all there
    with pytest.raises(ValueError):
        write_review_folder(tmp_path / "bad", without_fs=("sub-09",))
