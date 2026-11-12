"""Tests of surfcensus/config.py: the models, ``config.yaml``, keys and stage identities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from surfcensus import config as cfg
from surfcensus.config import (
    CONFIG_SCHEMA_VERSION,
    FSAVERAGE_STAGE_FIELDS,
    REFERENCE_STAGE_FIELDS,
    AppConfig,
    CacheConfig,
    ConfigError,
    IdentityConfig,
    InventoryConfig,
    MetricsConfig,
    RegionConfig,
    RemoteConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
    dump_config,
    load_config,
    new_census_config,
    new_local_config,
    run_key,
    stage_hash,
    stage_hashes,
    with_changes,
    write_config,
)
from surfcensus.privacy import HashKey, new_local_salt
from surfcensus.render.model import NON_TILE_FIELDS, TILE_FIELDS, RenderConfig
from surfcensus.sources.base import FileRole

DATASET_KEY = HashKey(bytes(range(32)), "dataset-key")


def local_config(tmp_path: Path, **overrides: object) -> RunConfig:
    return new_local_config(tmp_path / "data", **overrides)


def changed(config: RunConfig, section: str, **update: object) -> RunConfig:
    part = getattr(config, section).model_copy(update=update)
    return config.model_copy(update={section: part})


# --- new configurations ---------------------------------------------------------------------


def test_a_new_local_config_has_a_salt_a_seed_one_local_source_and_no_video(
    tmp_path: Path,
) -> None:
    config = new_local_config(tmp_path / "data", subjects=["sub-0*", "sub-11"])
    assert config.mode == "local" and config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.sources == (
        SourceConfig(
            kind="local", root=str((tmp_path / "data").resolve()), subjects=("sub-0*", "sub-11")
        ),
    )
    assert config.identity.kind == "local-salt"
    assert re.fullmatch(r"[0-9a-f]{64}", config.identity.local_salt or "")
    assert re.fullmatch(r"[0-9a-f]{12}", config.identity.key_fingerprint)
    assert config.render_video is False
    assert config.app.raters_per_subject == 1 and config.app.repeat_rate == 0.0
    assert 0 <= config.app.queue_seed < 2**31
    # the defaults of the other sections
    assert config.render == RenderConfig() and config.metrics == MetricsConfig()
    assert config.inventory == InventoryConfig() and config.scoring == ScoringConfig()
    assert config.remote == RemoteConfig() and config.cache == CacheConfig()


def test_every_new_local_config_draws_its_own_salt(tmp_path: Path) -> None:
    one, two = local_config(tmp_path), local_config(tmp_path)
    assert one.identity.local_salt != two.identity.local_salt
    assert one.identity.key_fingerprint != two.identity.key_fingerprint


def test_the_root_of_a_new_local_config_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    assert new_local_config("data").sources[0].root == str((tmp_path / "data").resolve())


def test_overrides_are_run_config_fields_as_models_or_mappings(tmp_path: Path) -> None:
    config = local_config(
        tmp_path,
        render=RenderConfig(panel_px=64),
        metrics={"min_reference_n": 3, "min_vertex_n": 2},
        render_video=True,
        app={"blind": True},
    )
    assert config.render.panel_px == 64
    assert config.metrics.min_reference_n == 3 and config.metrics.min_vertex_n == 2
    assert config.render_video is True
    assert config.app.blind is True and config.app.queue_seed >= 0
    assert local_config(tmp_path, app=AppConfig(queue_seed=7)).app.queue_seed == 7
    with pytest.raises(ConfigError, match=r"invalid: rendr: unknown key") as caught:
        local_config(tmp_path, rendr=RenderConfig())
    assert caught.value.reason == "invalid"
    with pytest.raises(ConfigError, match=r"metrics\.knn"):
        local_config(tmp_path, metrics={"knn": 0})


def test_a_new_census_config_stores_the_fingerprint_of_the_key_and_nothing_else() -> None:
    config = new_census_config(DATASET_KEY)
    assert config.mode == "census" and config.sources == ()
    assert config.identity == IdentityConfig(
        kind="dataset-key", key_fingerprint=DATASET_KEY.fingerprint
    )
    assert config.render_video is True
    assert config.app.raters_per_subject == 2 and config.app.repeat_rate == 0.05
    text = dump_config(config)
    assert DATASET_KEY.secret.hex() not in text and "local_salt: null" in text
    with pytest.raises(ConfigError, match=r"^key: ") as caught:
        new_census_config(new_local_salt())
    assert caught.value.reason == "key"


# --- model rules ----------------------------------------------------------------------------


def test_models_are_frozen_and_refuse_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        MetricsConfig().knn = 4  # pyrefly: ignore[read-only]
    with pytest.raises(ValidationError, match="Extra inputs"):
        MetricsConfig(knnn=4)  # pyrefly: ignore[unexpected-keyword]
    with pytest.raises(ValidationError, match="queue_seed"):
        AppConfig()  # pyrefly: ignore[missing-argument]


def test_the_defaults_are_those_of_the_specification() -> None:
    regions = RegionConfig()
    assert regions.occipital == ("cuneus", "lateraloccipital", "lingual", "pericalcarine")
    assert regions.posterior == ("fusiform", "precuneus") and regions.cut_mm == 45.0
    assert regions.medial == ("cuneus", "pericalcarine", "precuneus")
    assert regions.inferior == ("fusiform", "lingual")
    assert set(regions.medial) | set(regions.inferior) <= set(regions.occipital) | set(
        regions.posterior
    )
    metrics = MetricsConfig()
    assert metrics.mesh == "fsaverage6" and metrics.fsaverage_dir is None
    assert (metrics.min_reference_n, metrics.cluster_z, metrics.thickness_max_mm) == (30, 3.0, 5.0)
    assert ScoringConfig().components == (
        "mass_medial",
        "mass_inferior",
        "ceiling_frac",
        "occ_ctrl_ratio",
    )
    assert InventoryConfig() == InventoryConfig(logs="full", head_bytes=65536)
    assert RemoteConfig().max_concurrency == 8
    # the snap radius of a mark fits inside the vertex map of the default render
    assert AppConfig(queue_seed=0).mark_max_dist_mm <= RenderConfig().vertex_map_margin_mm


def test_a_source_config_has_the_fields_of_its_kind_only() -> None:
    SourceConfig(kind="local", root="/data", identity="/data", subjects=("a*",), field_strength=7)
    SourceConfig(kind="openneuro-derivatives", bucket="openneuro-derivatives", prefix="fmriprep/")
    with pytest.raises(ValidationError, match="a local source has no bucket"):
        SourceConfig(kind="local", root="/data", bucket="b")
    with pytest.raises(ValidationError, match="needs a bucket"):
        SourceConfig(kind="openneuro-derivatives")
    with pytest.raises(ValidationError, match="has no root, subjects"):
        SourceConfig(kind="openneuro-derivatives", bucket="b", root="/data", subjects=("a",))
    with pytest.raises(ValidationError, match="field_strength"):
        SourceConfig(kind="local", root="/data", field_strength=0)


def test_the_identity_holds_a_salt_only_for_a_local_run() -> None:
    salt = new_local_salt()
    IdentityConfig(kind="local-salt", local_salt=salt.secret.hex(), key_fingerprint="0" * 12)
    with pytest.raises(ValidationError, match="needs local_salt"):
        IdentityConfig(kind="local-salt", key_fingerprint="0" * 12)
    with pytest.raises(ValidationError, match="64 hex"):
        IdentityConfig(kind="local-salt", local_salt="abc", key_fingerprint="0" * 12)
    with pytest.raises(ValidationError, match="only for kind"):
        IdentityConfig(kind="dataset-key", local_salt=salt.secret.hex(), key_fingerprint="0" * 12)
    with pytest.raises(ValidationError, match="key_fingerprint"):
        IdentityConfig(kind="dataset-key", key_fingerprint="XYZ")


def test_mode_identity_and_sources_must_agree(tmp_path: Path) -> None:
    local, census = local_config(tmp_path), new_census_config(DATASET_KEY)
    remote = SourceConfig(kind="openneuro-derivatives", bucket="b", prefix="fmriprep/")
    with pytest.raises(ValidationError, match="does not use identity kind"):
        RunConfig(mode="census", sources=(), identity=local.identity, app=local.app)
    with pytest.raises(ValidationError, match="does not use identity kind"):
        RunConfig(mode="local", sources=(), identity=census.identity, app=local.app)
    with pytest.raises(ValidationError, match="a local run has no openneuro-derivatives source"):
        RunConfig(mode="local", sources=(remote,), identity=local.identity, app=local.app)
    with pytest.raises(ValidationError, match="stored by its root"):
        RunConfig(
            mode="local",
            sources=(SourceConfig(kind="local"),),
            identity=local.identity,
            app=local.app,
        )
    RunConfig(mode="census", sources=(remote,), identity=census.identity, app=census.app)


def test_a_census_config_with_a_local_source_holds_no_path() -> None:
    census = new_census_config(DATASET_KEY)
    by_pseudonym = SourceConfig(kind="local", dataset="D-3f9a1c07", field_strength=7.0)
    config = new_census_config(DATASET_KEY, sources=(by_pseudonym,))
    text = dump_config(config)
    assert "D-3f9a1c07" in text
    assert "/" not in text and "\\" not in text
    for with_path in (
        SourceConfig(kind="local", root="/data/7T", dataset="D-3f9a1c07"),
        SourceConfig(kind="local", identity="/data/7T", dataset="D-3f9a1c07"),
        SourceConfig(kind="local", root="/data/7T"),
        SourceConfig(kind="local"),
    ):
        with pytest.raises(ValidationError, match="by its pseudonym only"):
            RunConfig(mode="census", sources=(with_path,), identity=census.identity, app=census.app)
        with pytest.raises(ConfigError, match="by its pseudonym only"):
            new_census_config(DATASET_KEY, sources=(with_path,))


# --- config.yaml ----------------------------------------------------------------------------


def test_defaults_round_trip_through_yaml_byte_identically(tmp_path: Path) -> None:
    for name, config in (
        ("local", new_local_config(tmp_path / "data", subjects=["sub-*"])),
        ("census", new_census_config(DATASET_KEY)),
    ):
        run_dir = tmp_path / name
        write_config(run_dir, config)
        first = (run_dir / "config.yaml").read_bytes()
        loaded = load_config(run_dir)
        assert loaded == config
        write_config(run_dir, loaded)
        assert (run_dir / "config.yaml").read_bytes() == first
        assert sorted(p.name for p in run_dir.iterdir()) == ["config.yaml"]
        assert first.decode() == dump_config(config)


def test_a_changed_config_round_trips_with_every_section(tmp_path: Path) -> None:
    salt = HashKey(bytes([0x12, 0x34] * 16), "local-salt")
    config = RunConfig(
        mode="local",
        sources=(
            SourceConfig(
                kind="local",
                root="/data/Ünïcode study/derivatives",
                identity="/old/place",
                subjects=("sub-0?", "*ses-2"),
                field_strength=2.89,
            ),
        ),
        # a salt and a fingerprint of digits only must come back as strings
        identity=IdentityConfig(
            kind="local-salt", local_salt="1234" * 16, key_fingerprint=salt.fingerprint
        ),
        inventory=InventoryConfig(logs="head", head_bytes=1000),
        render=RenderConfig(panel_px=64, pial_colour="#00ff00", box_x_mm=(-10.5, 20.25)),
        render_video=True,
        metrics=MetricsConfig(
            mesh="fsaverage4", fsaverage_dir="/opt/fs/subjects/fsaverage", cluster_z=2.5
        ),
        scoring=ScoringConfig(mass="sum_z", standardize="robust_z", population="stratum"),
        app=AppConfig(queue_seed=123456789012, blind=True, require_marks=False, rater_name="tk"),
        cache=CacheConfig(retain_roles=(FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)),
    )
    write_config(tmp_path, config)
    loaded = load_config(tmp_path)
    assert loaded == config
    assert loaded.cache.retain_roles == (FileRole.LH_SPHERE_REG, FileRole.RH_SPHERE_REG)
    assert loaded.identity.local_salt == "1234" * 16
    all_digits = config.model_copy(
        update={"identity": config.identity.model_copy(update={"key_fingerprint": "123456789012"})}
    )
    write_config(tmp_path, all_digits)
    assert load_config(tmp_path).identity.key_fingerprint == "123456789012"


def test_config_yaml_is_plain_sorted_yaml(tmp_path: Path) -> None:
    config = local_config(tmp_path)
    write_config(tmp_path, config)
    data = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert list(data) == sorted(data)
    assert data["mode"] == "local" and data["schema_version"] == CONFIG_SCHEMA_VERSION
    assert data["render"]["box_y_mm"] == [-8.0, 45.0]
    assert data["sources"][0]["root"] == str((tmp_path / "data").resolve())
    assert "!!python" not in (tmp_path / "config.yaml").read_text()


def test_a_run_directory_without_config_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"^missing: .* has no config.yaml") as caught:
        load_config(tmp_path)
    assert caught.value.reason == "missing"
    with pytest.raises(ConfigError, match=r"^missing: "):
        load_config(tmp_path / "absent")


def test_an_unknown_key_is_an_error_that_names_it(tmp_path: Path) -> None:
    write_config(tmp_path, local_config(tmp_path))
    path = tmp_path / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["metrics"]["cluster_zz"] = 3.5
    data["colour"] = "red"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as caught:
        load_config(tmp_path)
    assert caught.value.reason == "invalid"
    assert "metrics.cluster_zz: unknown key" in caught.value.detail
    assert "colour: unknown key" in caught.value.detail
    assert str(caught.value) == f"invalid: {caught.value.detail}"


def test_a_wrong_value_names_its_key(tmp_path: Path) -> None:
    write_config(tmp_path, local_config(tmp_path))
    path = tmp_path / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["render"]["panel_px"] = "wide"
    data["scoring"]["standardize"] = "zscore"
    del data["app"]["queue_seed"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as caught:
        load_config(tmp_path)
    assert caught.value.reason == "invalid"
    for key in ("render.panel_px", "scoring.standardize", "app.queue_seed"):
        assert key in caught.value.detail


@pytest.mark.parametrize(
    "content", ["", "just a sentence\n", "- a\n- list\n", "mode: [unclosed\n", "\x00\x01"]
)
def test_a_file_that_is_not_a_mapping_is_invalid(tmp_path: Path, content: str) -> None:
    (tmp_path / "config.yaml").write_text(content)
    with pytest.raises(ConfigError, match=r"^invalid: config.yaml"):
        load_config(tmp_path)


def test_a_newer_schema_is_refused_before_its_keys_are_judged(tmp_path: Path) -> None:
    write_config(tmp_path, local_config(tmp_path))
    path = tmp_path / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["schema_version"] = CONFIG_SCHEMA_VERSION + 1
    data["a_section_of_the_future"] = {"x": 1}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=r"^newer schema: config.yaml has schema 2") as caught:
        load_config(tmp_path)
    assert caught.value.reason == "newer schema"


# --- run_key --------------------------------------------------------------------------------


def test_run_key_of_a_local_run_is_its_salt(tmp_path: Path) -> None:
    config = local_config(tmp_path)
    key = run_key(config)
    assert key.kind == "local-salt" and key.secret.hex() == config.identity.local_salt
    assert run_key(config, DATASET_KEY) == key  # a dataset key is not used by a local run
    tampered = config.model_copy(
        update={"identity": config.identity.model_copy(update={"local_salt": "00" * 32})}
    )
    with pytest.raises(ConfigError, match=r"^key: the salt") as caught:
        run_key(tampered)
    assert caught.value.reason == "key"


def test_run_key_of_a_census_run_needs_the_key_it_was_created_with() -> None:
    config = new_census_config(DATASET_KEY)
    assert run_key(config, DATASET_KEY) == DATASET_KEY
    with pytest.raises(ConfigError, match=r"^key: this command needs the dataset key"):
        run_key(config)
    with pytest.raises(ConfigError, match=r"^key: .*fingerprint") as caught:
        run_key(config, HashKey(bytes(range(1, 33)), "dataset-key"))
    assert DATASET_KEY.secret.hex() not in str(caught.value)
    with pytest.raises(ConfigError, match=r"^key: .*not a local salt"):
        run_key(config, HashKey(DATASET_KEY.secret, "local-salt"))


# --- stage identities -----------------------------------------------------------------------


def test_stage_hash_is_12_hex_and_stable_against_key_order() -> None:
    one = stage_hash("fsaverage", 1, {"knn": 8, "mesh": "fsaverage6", "regions": {"a": 1, "b": 2}})
    two = stage_hash("fsaverage", 1, {"regions": {"b": 2, "a": 1}, "mesh": "fsaverage6", "knn": 8})
    assert one == two and re.fullmatch(r"[0-9a-f]{12}", one)
    assert stage_hash("x", 1, {"t": (1, 2)}) == stage_hash("x", 1, {"t": [1, 2]})
    # a model hashes like its JSON form
    assert stage_hash("inventory", 1, InventoryConfig()) == stage_hash(
        "inventory", 1, {"head_bytes": 65536, "logs": "full"}
    )


def test_stage_hash_changes_with_name_schema_payload_and_upstream() -> None:
    base = stage_hash("reference", 1, {"cluster_z": 3.0}, ("abc",))
    others = [
        stage_hash("score", 1, {"cluster_z": 3.0}, ("abc",)),
        stage_hash("reference", 2, {"cluster_z": 3.0}, ("abc",)),
        stage_hash("reference", 1, {"cluster_z": 3.5}, ("abc",)),
        stage_hash("reference", 1, {"cluster_z": 3.0}, ("abd",)),
        stage_hash("reference", 1, {"cluster_z": 3.0}),
        stage_hash("reference", 1, {"cluster_z": 3.0}, ("abc", "def")),
        stage_hash("reference", 1, {"cluster_z": 3.0}, ("def", "abc")),
    ]
    assert len({base, *others}) == len(others) + 1


def test_every_metrics_field_belongs_to_one_stage_or_is_named_as_unhashed() -> None:
    hashed = set(FSAVERAGE_STAGE_FIELDS) | set(REFERENCE_STAGE_FIELDS)
    assert not set(FSAVERAGE_STAGE_FIELDS) & set(REFERENCE_STAGE_FIELDS)
    assert len(hashed) == len(FSAVERAGE_STAGE_FIELDS) + len(REFERENCE_STAGE_FIELDS)
    # a new field of MetricsConfig must be put into one of the tuples, or here
    assert set(MetricsConfig.model_fields) - hashed == {"fsaverage_dir"}


def test_stage_hashes_names_the_stages_the_config_decides(tmp_path: Path) -> None:
    config = local_config(tmp_path)
    hashes = stage_hashes(config, template_id="0123456789ab")
    assert set(hashes) == {
        "inventory",
        "regional",
        "fsaverage",
        "reference",
        "score",
        "render",
        "native",
    }
    assert all(re.fullmatch(r"[0-9a-f]{12}", value) for value in hashes.values())
    assert len(set(hashes.values())) == len(hashes)
    assert hashes["render"] == config.render.config_hash()
    assert hashes == stage_hashes(config, template_id="0123456789ab")
    # the same settings in another run directory, with another salt, are the same stages
    assert hashes == stage_hashes(new_local_config(tmp_path / "other"), template_id="0123456789ab")
    assert set(cfg.STAGE_SCHEMAS) >= set(hashes) - {"render"}


CHANGES = [
    # (section, update, the stages whose identity changes)
    ("inventory", {"logs": "head"}, {"inventory"}),
    ("inventory", {"head_bytes": 1000}, {"inventory"}),
    ("render", {"panel_px": 64}, {"render"}),
    ("render", {"fps": 10}, {"render"}),
    ("metrics", {"knn": 4}, {"fsaverage", "reference", "score", "native"}),
    ("metrics", {"mesh": "fsaverage5"}, {"fsaverage", "reference", "score", "native"}),
    ("metrics", {"min_coverage": 0.8}, {"fsaverage", "reference", "score", "native"}),
    (
        "metrics",
        {"regions": RegionConfig(cut_mm=50.0)},
        {"fsaverage", "reference", "score", "native"},
    ),
    ("metrics", {"cluster_z": 2.5}, {"reference", "score", "native"}),
    ("metrics", {"min_reference_n": 20}, {"reference", "score", "native"}),
    ("metrics", {"clusters_kept": 3}, {"reference", "score", "native"}),
    ("metrics", {"fsaverage_dir": "/opt/freesurfer/subjects/fsaverage"}, set()),
    ("scoring", {"standardize": "robust_z"}, {"score"}),
    ("scoring", {"population": "stratum"}, {"score"}),
    ("scoring", {"mass": "sum_z"}, {"score"}),
    ("remote", {"max_concurrency": 2}, set()),
    ("cache", {"batch_subjects": 4}, set()),
    ("cache", {"retain_roles": (FileRole.LH_SPHERE_REG,)}, set()),
    ("app", {"blind": True}, set()),
    ("app", {"queue_seed": 99}, set()),
    ("app", {"mark_max_dist_mm": 3.0}, set()),
    ("select", {"render_scope": "selected"}, set()),
    ("intensity", {"depths": (0.5, 1.0)}, set()),
]


@pytest.mark.parametrize(("section", "update", "expected"), CHANGES)
def test_a_changed_field_changes_the_hash_of_its_stage_and_what_follows_only(
    tmp_path: Path, section: str, update: dict[str, object], expected: set[str]
) -> None:
    config = local_config(tmp_path)
    before = stage_hashes(config, template_id="0123456789ab")
    after = stage_hashes(changed(config, section, **update), template_id="0123456789ab")
    assert {stage for stage in before if before[stage] != after[stage]} == expected


def test_fields_outside_the_sections_change_no_stage(tmp_path: Path) -> None:
    config = local_config(tmp_path)
    before = stage_hashes(config)
    other = config.model_copy(
        update={
            "render_video": True,
            "sources": (SourceConfig(kind="local", root="/elsewhere", subjects=("sub-1*",)),),
        }
    )
    assert stage_hashes(other) == before


def test_the_template_and_the_absence_of_a_reference_are_part_of_the_identities(
    tmp_path: Path,
) -> None:
    config = local_config(tmp_path)
    with_template = stage_hashes(config, template_id="0123456789ab")
    other_template = stage_hashes(config, template_id="ba9876543210")
    without = stage_hashes(config)
    for stage in ("fsaverage", "reference", "score", "native"):
        assert len({with_template[stage], other_template[stage], without[stage]}) == 3
    for stage in ("inventory", "regional", "render"):
        assert with_template[stage] == other_template[stage] == without[stage]
    # no reference applies (decision D16): the score hangs off nothing, whatever the template
    fallback = stage_hashes(config, template_id="0123456789ab", with_reference=False)
    assert set(fallback) == {"inventory", "regional", "fsaverage", "score", "render"}
    assert fallback["score"] not in (with_template["score"], without["score"])
    assert fallback["score"] == stage_hashes(config, with_reference=False)["score"]
    assert fallback["fsaverage"] == with_template["fsaverage"]


# --- RenderConfig.tile_hash -------------------------------------------------------------------


def test_the_tile_hash_field_list_is_pinned() -> None:
    # a new field of RenderConfig must be put on one side or the other
    assert set(TILE_FIELDS) | set(NON_TILE_FIELDS) == set(RenderConfig.model_fields)
    assert not set(TILE_FIELDS) & set(NON_TILE_FIELDS)
    assert len(TILE_FIELDS) == len(set(TILE_FIELDS)) == 14
    assert set(NON_TILE_FIELDS) == {
        "crosshair_colour",
        "crosshair_alpha",
        "fps",
        "crf",
        "vertex_map_margin_mm",
    }


TILE_CHANGES = {
    "box_x_mm": (-20.0, 30.0),
    "box_y_mm": (-8.0, 40.0),
    "box_z_mm": (-35.0, 20.0),
    "margin_mm": 12.0,
    "slice_step_mm": 2.0,
    "panel_px": 64,
    "radiological": False,
    "sagittal_anterior_left": True,
    "pial_colour": "#00FF00",
    "white_colour": "#0000FF",
    "line_width_px": 2.0,
    "window_percentiles": (2.0, 99.0),
    "brainmask_dilation": 1,
    "webp_quality": 90,
    "crosshair_colour": "#FFFFFF",
    "crosshair_alpha": 0.9,
    "fps": 24,
    "crf": 30,
    "vertex_map_margin_mm": 8.0,
}


def test_the_tile_hash_changes_with_the_tile_fields_and_with_no_other() -> None:
    assert set(TILE_CHANGES) == set(RenderConfig.model_fields)
    base = RenderConfig()
    assert re.fullmatch(r"[0-9a-f]{12}", base.tile_hash())
    assert base.tile_hash() == RenderConfig().tile_hash()
    assert base.tile_hash() != base.config_hash()
    seen = {base.tile_hash()}
    for name, value in TILE_CHANGES.items():
        other = RenderConfig(**{name: value})
        assert other != base, name
        assert other.config_hash() != base.config_hash(), name
        if name in TILE_FIELDS:
            assert other.tile_hash() not in seen, name
            seen.add(other.tile_hash())
        else:
            assert other.tile_hash() == base.tile_hash(), name
    # two spellings of one colour are one tile
    assert RenderConfig(pial_colour="#ffd400").tile_hash() == base.tile_hash()


# --- with_changes -----------------------------------------------------------------------------


def test_with_changes_replaces_fields_and_validates_the_whole_again(tmp_path: Path) -> None:
    config = local_config(tmp_path)
    assert with_changes(config) == config
    video = with_changes(config, render_video=True, cache={"batch_subjects": 4})
    assert (video.render_video, video.cache.batch_subjects) == (True, 4)
    assert video.model_copy(update={"render_video": False, "cache": config.cache}) == config
    assert config.render_video is False  # the original is untouched

    sources = [s.model_dump() | {"field_strength": 7.0} for s in config.sources]
    assert with_changes(config, sources=sources).sources[0].field_strength == 7.0
    by_model = with_changes(config, render=RenderConfig(panel_px=64))
    assert by_model.render.panel_px == 64 and by_model.identity == config.identity

    for changes, problem in (
        ({"colour": "blue"}, "colour: unknown key"),
        ({"cache": {"batch_subjects": 0}}, "cache.batch_subjects"),
        ({"sources": [s | {"field_strength": -3.0} for s in sources]}, "field_strength"),
        ({"mode": "census"}, "census run"),
    ):
        with pytest.raises(ConfigError) as raised:
            with_changes(config, **changes)
        assert raised.value.reason == "invalid" and problem in raised.value.detail
