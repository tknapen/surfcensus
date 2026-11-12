"""The configuration of a run: ``config.yaml`` in the run directory, and stage identities.

Every model is frozen and refuses unknown keys. ``config.yaml`` holds the salt of a local
run and the root of its source; it never holds the dataset key, the run secret or an
accession, and in a census run it holds no path either.

A value is hashed into the identity of the stage it changes (:func:`stage_hashes`), so a
changed option redoes that stage and what follows it, and nothing else.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from surfcensus._io import atomic_path, canonical_json
from surfcensus.privacy import HashKey, local_salt_from_hex, new_local_salt
from surfcensus.render.model import RenderConfig, _Frozen
from surfcensus.sources.base import FileRole, SourceKind

CONFIG_SCHEMA_VERSION = 1
CONFIG_FILE = "config.yaml"


class ConfigError(Exception):
    """A run directory or its configuration cannot be used as asked.

    ``reason`` is one of ``missing``, ``invalid``, ``other root``, ``newer schema``, ``key``.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class SourceConfig(_Frozen):
    """A source as a run stores it; ``run.from_config`` builds the source from it."""

    kind: SourceKind
    root: str | None = None
    """Local source in a local run: absolute path. None in a census run, where the root is
    in the catalog only."""
    identity: str | None = None
    """Local source in a local run: set at first inventory, then fixed (see
    ``DatasetRecord.identity``)."""
    dataset: str | None = None
    """Local source in a census run: its pseudonym, the link to the catalog row."""
    subjects: tuple[str, ...] = ()
    """Local only: globs on directory names; empty means all."""
    field_strength: float | None = Field(default=None, gt=0)
    """Local only: tesla, copied to the subjects of the source."""
    bucket: str | None = None
    """Remote only; likewise ``prefix``."""
    prefix: str | None = None

    @model_validator(mode="after")
    def _fields_of_its_kind(self) -> SourceConfig:
        if self.kind == "local":
            given = {"bucket": self.bucket, "prefix": self.prefix}
        else:
            given = {
                "root": self.root,
                "identity": self.identity,
                "dataset": self.dataset,
                "subjects": self.subjects or None,
                "field_strength": self.field_strength,
            }
            if self.bucket is None:
                raise ValueError(f"a {self.kind} source needs a bucket")
        extra = sorted(name for name, value in given.items() if value is not None)
        if extra:
            raise ValueError(f"a {self.kind} source has no {', '.join(extra)}")
        return self


class IdentityConfig(_Frozen):
    """Which secret pseudonyms and subject identifiers are hashed with."""

    kind: Literal["local-salt", "dataset-key"]
    local_salt: str | None = None
    """64 hex characters; only when ``kind`` is ``local-salt``. The dataset key of a census
    run is never stored here."""
    key_fingerprint: str = Field(pattern=r"^[0-9a-f]{12}$")
    """Of the salt or of the dataset key (decision D10)."""

    @model_validator(mode="after")
    def _salt_only_for_a_local_run(self) -> IdentityConfig:
        if self.kind == "dataset-key":
            if self.local_salt is not None:
                raise ValueError("local_salt is only for kind 'local-salt'")
        elif self.local_salt is None:
            raise ValueError("kind 'local-salt' needs local_salt")
        else:
            local_salt_from_hex(self.local_salt)
        return self


class InventoryConfig(_Frozen):
    logs: Literal["full", "head"] = "full"
    """Read the recon-all logs whole, or only their first ``head_bytes``."""
    head_bytes: int = Field(default=65536, ge=1)


class RegionConfig(_Frozen):
    """Desikan parcel names that define the regions on the template. Hashed into its id."""

    occipital: tuple[str, ...] = ("cuneus", "lateraloccipital", "lingual", "pericalcarine")
    posterior: tuple[str, ...] = ("fusiform", "precuneus")
    """Parcels of which only the part within ``cut_mm`` of the occipital pole is searched."""
    cut_mm: float = Field(default=45.0, gt=0)
    medial: tuple[str, ...] = ("cuneus", "pericalcarine", "precuneus")
    inferior: tuple[str, ...] = ("fusiform", "lingual")
    control: tuple[str, ...] = ("caudalmiddlefrontal", "rostralmiddlefrontal", "superiorfrontal")


MeshName = Literal["fsaverage4", "fsaverage5", "fsaverage6", "fsaverage"]
"""Icosahedral order 4, 5, 6 and 7."""


class MetricsConfig(_Frozen):
    mesh: MeshName = "fsaverage6"
    fsaverage_dir: str | None = None
    """An explicit template directory. Not hashed: the template's content is."""
    regions: RegionConfig = RegionConfig()
    knn: int = Field(default=8, ge=1)
    min_valid_weight: float = Field(default=0.5, ge=0, le=1)
    thickness_max_mm: float = Field(default=5.0, gt=0)
    ceiling_margin_mm: float = Field(default=0.1, ge=0)
    min_coverage: float = Field(default=0.9, ge=0, le=1)
    """Fraction of template cortex vertices that must get a value."""
    max_fallback_frac: float = Field(default=0.01, ge=0, le=1)
    """More nearest-neighbour fallbacks than this fraction of the targets: no map."""
    min_reference_n: int = Field(default=30, ge=1)
    min_vertex_n: int = Field(default=20, ge=1)
    mad_smooth_iterations: int = Field(default=2, ge=0)
    mad_floor_mm: float = Field(default=0.05, ge=0)
    cluster_z: float = 3.0
    cluster_min_vertices: int = Field(default=1, ge=1)
    clusters_kept: int = Field(default=5, ge=1)


FSAVERAGE_STAGE_FIELDS: tuple[str, ...] = (
    "mesh",
    "regions",
    "knn",
    "min_valid_weight",
    "thickness_max_mm",
    "ceiling_margin_mm",
    "min_coverage",
    "max_fallback_frac",
)
"""Fields of :class:`MetricsConfig` that change a subject's fsaverage map and summary."""

REFERENCE_STAGE_FIELDS: tuple[str, ...] = (
    "min_reference_n",
    "min_vertex_n",
    "mad_smooth_iterations",
    "mad_floor_mm",
    "cluster_z",
    "cluster_min_vertices",
    "clusters_kept",
)
"""Fields of :class:`MetricsConfig` that change the reference and the clusters."""


class ScoringConfig(_Frozen):
    components: tuple[str, ...] = ("mass_medial", "mass_inferior", "ceiling_frac", "occ_ctrl_ratio")
    mass: Literal["excess", "sum_z"] = "excess"
    standardize: Literal["normal_score", "robust_z"] = "normal_score"
    population: Literal["run", "stratum"] = "run"


class IntensityConfig(_Frozen):
    depths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    """Fractions of the way from the white to the pial surface."""
    percentiles: tuple[float, float] = (1.0, 99.0)


class SelectConfig(_Frozen):
    min_per_dataset: int = Field(default=5, ge=0)
    top_decile_factor: float = Field(default=3.0, gt=0)
    extreme_quantile: float = Field(default=0.99, ge=0, le=1)
    render_scope: Literal["all", "selected"] = "all"


class AppConfig(_Frozen):
    """Rater app settings. Not hashed into any stage."""

    rater_name: str = "local"
    raters_per_subject: int = Field(default=1, ge=1)
    repeat_rate: float = Field(default=0.0, ge=0, le=1)
    repeat_gap: int = Field(default=200, ge=0)
    mark_max_dist_mm: float = Field(default=5.0, ge=0)
    """Snap radius of a mark; must not exceed ``render.vertex_map_margin_mm``."""
    mark_near_mm: float = Field(default=2.0, ge=0)
    mark_show_mm: float = Field(default=3.0, ge=0)
    require_marks: bool | None = None
    """None: ``protocol.yaml`` decides."""
    blind: bool = False
    queue_seed: int
    """Drawn once when the run is created."""
    media_base_url: str | None = None


class RemoteConfig(_Frozen):
    """Not hashed into any stage."""

    max_concurrency: int = Field(default=8, ge=1)
    max_retries: int = Field(default=6, ge=0)
    timeout_s: float = Field(default=120.0, gt=0)


class CacheConfig(_Frozen):
    """Not hashed into any stage."""

    max_bytes: int = Field(default=4_000_000_000, ge=1)
    batch_subjects: int = Field(default=32, ge=1)
    retain_roles: tuple[FileRole, ...] = ()
    """Roles kept after their lease, beside Tier A and the template."""


class RunConfig(_Frozen):
    """Contents of ``config.yaml``."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    mode: Literal["local", "census"]
    sources: tuple[SourceConfig, ...]
    identity: IdentityConfig
    inventory: InventoryConfig = InventoryConfig()
    render: RenderConfig = RenderConfig()
    render_video: bool = True
    """Whether renders of the run include ``anim.mp4``. A new local run sets False."""
    metrics: MetricsConfig = MetricsConfig()
    scoring: ScoringConfig = ScoringConfig()
    intensity: IntensityConfig = IntensityConfig()
    select: SelectConfig = SelectConfig()
    app: AppConfig
    remote: RemoteConfig = RemoteConfig()
    cache: CacheConfig = CacheConfig()

    @model_validator(mode="after")
    def _mode_fits_identity_and_sources(self) -> RunConfig:
        local = self.mode == "local"
        if (self.identity.kind == "local-salt") != local:
            raise ValueError(f"a {self.mode} run does not use identity kind {self.identity.kind!r}")
        for source in self.sources:
            if source.kind != "local":
                if local:
                    raise ValueError(f"a local run has no {source.kind} source")
            elif local and (source.root is None or source.dataset is not None):
                raise ValueError("a local source in a local run is stored by its root")
            elif not local and (
                source.root is not None or source.identity is not None or source.dataset is None
            ):
                # no path in the config.yaml of a census run: the root is in the catalog
                raise ValueError("a local source in a census run is stored by its pseudonym only")
        return self


def _queue_seed() -> int:
    return secrets.randbelow(2**31)


def _build(defaults: dict[str, object], overrides: Mapping[str, object]) -> RunConfig:
    values = defaults | dict(overrides)
    app = values["app"]
    if isinstance(app, Mapping):  # a partial app section keeps the seed drawn here
        values["app"] = {"queue_seed": _queue_seed()} | dict(app)
    try:
        return RunConfig.model_validate(values)
    except ValidationError as exc:
        raise ConfigError("invalid", _validation_detail(exc)) from exc


def new_local_config(
    root: Path | str, *, subjects: Sequence[str] = (), **overrides: object
) -> RunConfig:
    """The configuration of a new local run on ``root``: a fresh salt and queue seed, no video.

    ``overrides`` are fields of :class:`RunConfig`, as models or as mappings, for example
    ``render=RenderConfig(panel_px=64)``.
    """
    salt = new_local_salt()
    source = SourceConfig(kind="local", root=str(Path(root).resolve()), subjects=tuple(subjects))
    defaults: dict[str, object] = {
        "mode": "local",
        "sources": (source,),
        "identity": IdentityConfig(
            kind="local-salt", local_salt=salt.secret.hex(), key_fingerprint=salt.fingerprint
        ),
        "render_video": False,
        "app": AppConfig(queue_seed=_queue_seed()),
    }
    return _build(defaults, overrides)


def new_census_config(key: HashKey, **overrides: object) -> RunConfig:
    """The configuration of a new census run. It stores the fingerprint of ``key`` only."""
    if key.kind != "dataset-key":
        raise ConfigError("key", "a census run needs the dataset key, not a local salt")
    defaults: dict[str, object] = {
        "mode": "census",
        "sources": (),
        "identity": IdentityConfig(kind="dataset-key", key_fingerprint=key.fingerprint),
        "app": AppConfig(raters_per_subject=2, repeat_rate=0.05, queue_seed=_queue_seed()),
    }
    return _build(defaults, overrides)


def _validation_detail(exc: ValidationError) -> str:
    problems = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        what = "unknown key" if error["type"] == "extra_forbidden" else error["msg"]
        problems.append(f"{where}: {what}" if where else what)
    return "; ".join(problems)


def load_config(run_dir: Path | str) -> RunConfig:
    """Read ``config.yaml`` of a run directory.

    Raises :class:`ConfigError`: ``missing`` when there is none, ``newer schema`` when it
    was written by a later version of the package, ``invalid`` for anything else that
    cannot be used, naming each offending key.
    """
    path = Path(run_dir) / CONFIG_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError("missing", f"{run_dir} has no {CONFIG_FILE}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError("invalid", f"{CONFIG_FILE} cannot be read: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError("invalid", f"{CONFIG_FILE} is not YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("invalid", f"{CONFIG_FILE} is not a mapping of settings")
    version = data.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool):
        if version > CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                "newer schema",
                f"{CONFIG_FILE} has schema {version}; this version of surfcensus reads up to "
                f"{CONFIG_SCHEMA_VERSION}",
            )
    try:
        return RunConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("invalid", f"{CONFIG_FILE}: {_validation_detail(exc)}") from exc


def dump_config(config: RunConfig) -> str:
    """The text of ``config.yaml``: keys sorted, so equal configs give equal bytes."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)


def write_config(run_dir: Path | str, config: RunConfig) -> None:
    """Write ``config.yaml`` atomically."""
    with atomic_path(Path(run_dir) / CONFIG_FILE) as tmp:
        tmp.write_text(dump_config(config), encoding="utf-8")


def run_key(config: RunConfig, dataset_key: HashKey | None = None) -> HashKey:
    """The key that pseudonyms and subject identifiers of this run are hashed with.

    A local run: its stored salt. A census run: ``dataset_key``, which must be given and
    must have the fingerprint the run was created with. Raises ``ConfigError("key", ...)``.
    """
    identity = config.identity
    if identity.kind == "local-salt":
        assert identity.local_salt is not None  # IdentityConfig checks it
        key = local_salt_from_hex(identity.local_salt)
        problem = f"the salt in {CONFIG_FILE} does not have the stored fingerprint"
    elif dataset_key is None:
        raise ConfigError("key", "this command needs the dataset key of the census run")
    elif dataset_key.kind != "dataset-key":
        raise ConfigError("key", "a census run is keyed with the dataset key, not a local salt")
    else:
        key = dataset_key
        problem = (
            f"the dataset key has fingerprint {dataset_key.fingerprint}, the run was created "
            f"with {identity.key_fingerprint}"
        )
    if key.fingerprint != identity.key_fingerprint:
        raise ConfigError("key", problem)
    return key


def stage_hash(
    name: str,
    schema: int,
    payload: BaseModel | Mapping[str, object],
    upstream: Sequence[str] = (),
) -> str:
    """Identity of a stage: 12 hex characters of its name, schema number, settings and the
    identities of what it is computed from. Independent of the key order of ``payload``."""
    config = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    text = canonical_json(
        {"stage": name, "schema": schema, "config": config, "upstream": list(upstream)}
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


STAGE_SCHEMAS: dict[str, int] = {
    "inventory": 1,
    "enrich": 1,
    "regional": 1,
    "fsaverage": 1,
    "reference": 1,
    "score": 1,
    "select": 1,
    "native": 1,
    "intensity": 1,
}
"""Schema number of each stage: raised when the stage's output changes for equal settings."""


def stage_hashes(
    config: RunConfig, *, template_id: str | None = None, with_reference: bool = True
) -> dict[str, str]:
    """Identities of the stages that follow from the configuration and the template alone.

    Keys: ``inventory``, ``regional``, ``fsaverage``, ``score``, ``render``, and with a
    reference also ``reference`` and ``native``. ``template_id`` is None for a run without
    a template; ``with_reference`` is False when no reference applies (decision D16), and
    the score then hangs off nothing. ``enrich``, ``select`` and ``intensity`` depend on
    values that only their phase knows and are computed there with :func:`stage_hash`.
    """
    metrics = config.metrics.model_dump(mode="json")
    hashes = {
        "inventory": stage_hash("inventory", STAGE_SCHEMAS["inventory"], config.inventory),
        "regional": stage_hash("regional", STAGE_SCHEMAS["regional"], {}),
        "fsaverage": stage_hash(
            "fsaverage",
            STAGE_SCHEMAS["fsaverage"],
            {name: metrics[name] for name in FSAVERAGE_STAGE_FIELDS},
            (template_id or "none",),
        ),
        "render": config.render.config_hash(),
    }
    if with_reference:
        hashes["reference"] = stage_hash(
            "reference",
            STAGE_SCHEMAS["reference"],
            {name: metrics[name] for name in REFERENCE_STAGE_FIELDS},
            (hashes["fsaverage"],),
        )
        hashes["native"] = stage_hash("native", STAGE_SCHEMAS["native"], {}, (hashes["reference"],))
    hashes["score"] = stage_hash(
        "score", STAGE_SCHEMAS["score"], config.scoring, (hashes.get("reference", "none"),)
    )
    return hashes
