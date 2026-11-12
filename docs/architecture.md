# Architecture

## Two uses, one pipeline

The package serves two audiences with the same code path.

1. A researcher with their own FreeSurfer subjects (a `SUBJECTS_DIR`, an fMRIPrep output folder, or any folder of FreeSurfer subject directories) wants to score, render and rate them. This must work in one command with no cloud access:

   ```
   pip install "surfcensus[app]"
   surfcensus review /path/to/SUBJECTS_DIR
   ```

   That command inventories the subjects, computes metrics, renders the occipital animations, and opens the rater app in the browser. Ratings are saved in the run directory and exported with `surfcensus export`.

2. The paper team runs the census over OpenNeuro. The census is the same pipeline with a remote source, plus discovery and enrichment phases that local users never need.

So the pipeline is organized around a `Source` abstraction. Everything downstream of the source (inventory, metrics, scoring, rendering, rating, export) is identical for local and remote subjects.

```
Source  ->  inventory  ->  metrics  ->  scoring  ->  select  ->  render  ->  rate  ->  export
(local | openneuro-derivatives | openneuro-embedded)
```

## Sources

- `local`: a path. Auto-detects three layouts: a FreeSurfer `SUBJECTS_DIR` (children containing `surf/` and `mri/`); an fMRIPrep output (`sourcedata/freesurfer/` or, for older versions, `freesurfer/` next to the fMRIPrep folder); a BIDS derivatives root containing `freesurfer/`. Skips `fsaverage*` and other template subjects. Accepts `--subjects` globs. Reads in place, never copies or modifies source files.
- `openneuro-derivatives`: the `s3://openneuro-derivatives/fmriprep/` prefixes (see `data_sources.md`).
- `openneuro-embedded`: `derivatives/` folders inside raw OpenNeuro datasets.

Each source yields `SubjectRecord`s with a resolver that turns a file role (`t1`, `lh_pial`, ...) into a local path, downloading into the cache when needed. Downstream code only ever sees local paths.

## Scoring with a normative reference

Outlier detection runs in fsaverage space: each subject's thickness is resampled to fsaverage6 through its `sphere.reg` and z-scored per vertex against a reference distribution (see `metrics.md`). For the census, the reference is the census itself, stratified. A local user with 12 subjects cannot build a stable per-vertex reference from their own data. The package therefore ships a normative reference derived from the census as package data (`surfcensus/data/reference/`). It holds per-vertex median and MAD maps of globally normalized thickness on fsaverage6 (about 0.7 MB per stratum), stratified by field strength and FreeSurfer major version, plus the regional parcel reference, all computed only from subjects that raters passed. Local scoring reports z-maps against the shipped reference and, when the local set has at least 30 subjects, against the local set. The reference is versioned; scores record which reference version they used. The fsaverage6 template is taken from a local FreeSurfer install, from an `fsaverage/` directory next to the subjects, or from a copy bundled in the wheel, in that order (decision D14).

Until the census has produced ratings, the shipped reference is empty. Local sets of at least 30 subjects are scored within the set; smaller sets are ranked on raw ceiling fraction and occipital-to-control thickness ratio only (decision D16).

## Package layout

```
surfcensus/
  pyproject.toml
  justfile
  README.md
  CLAUDE.md
  docs/
  src/surfcensus/
    __init__.py
    cli.py                 # typer app
    config.py              # pydantic settings; YAML config per run, sensible defaults for local use
    privacy.py             # dataset key, keyed pseudonyms, reveal mapping
    store.py               # SQLite catalog + parquet tables; schema migrations
    sources/
      base.py              # Source protocol, SubjectRecord, file roles
      local.py             # SUBJECTS_DIR / fMRIPrep / BIDS-derivatives layouts
      openneuro_derivatives.py
      openneuro_embedded.py
      patterns.py          # path patterns that identify a FreeSurfer subject dir
    remote/
      s3.py                # anonymous S3 listing, ranged GET, retry, rate limit
      openneuro.py         # OpenNeuro GraphQL client
      cache.py             # content-addressed cache keyed by (bucket, key, etag)
    inventory/
      subjects.py          # file presence, sizes, etags or mtimes
      provenance.py        # FreeSurfer version, recon-all flags, fMRIPrep version
    enrich/                # census only
      participants.py      # age, sex from participants.tsv
      acquisition.py       # field strength, manufacturer from T1w sidecars
    fsio/
      surfaces.py          # read white/pial in tkr RAS, normals
      volumes.py           # read T1.mgz, vox2ras-tkr, intensity windows
      stats.py             # parse ?h.aparc.stats, aseg.stats
      annot.py             # read ?h.aparc.annot
    templates.py           # locate or fetch fsaverage spheres, cortex labels, search-region label
    fsio/registration.py   # sphere.reg mapping: barycentric resampling, vertex correspondences
    metrics/
      regional.py          # parcel thickness from stats files
      fsaverage.py         # resample thickness to fsaverage6, per-vertex reference, z-maps
      clusters.py          # outlier clusters on the fsaverage mesh, centroid back-mapping
      ceiling.py           # ceiling fraction on native surfaces
      intensity.py
      scoring.py           # composite score
    reference/
      build.py             # build a normative reference from a rated census run
      load.py
    data/reference/        # shipped reference tables (parquet + JSON metadata)
    select/
      sampling.py
    render/
      model.py             # RenderConfig and the contents of render.json
      geometry.py          # pole, box, panels, pixel <-> voxel <-> tkr RAS
      slicing.py           # mesh-plane intersection
      frames.py            # grayscale tiles, contour overlays, composite strips
      encode.py            # WebP and MP4 writers
      vertexmap.py         # vertex_map.npz
      api.py               # three_plane()
      freeview.py          # optional backend, wraps the original gist approach
    app/
      server.py            # FastAPI
      models.py            # SQLModel tables
      queue.py
      marks.py             # click -> voxel -> tkr/scanner RAS -> native vertex -> fsaverage6 vertex
      static/              # vanilla JS + CSS, no build step, ships in the wheel
      templates/
    export/
      tables.py
  tests/
  workflow/
    Snakefile              # census orchestration over the CLI
  notebooks/               # marimo analysis notebooks
```

## Dependencies

Core: `nibabel`, `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `imageio`, `imageio-ffmpeg`, `pillow`, `httpx`, `typer`, `pydantic`, `pydantic-settings`, `pyyaml`, `jaxtyping`, `tqdm`.

Extras:

- `app`: `fastapi`, `uvicorn`, `sqlmodel`, `jinja2`.
- `freeview`: no Python deps; requires a FreeSurfer install and `xvfb-run` at runtime.
- `analysis`: `marimo`, `statsmodels`, `scikit-learn`, `krippendorff`.
Dependency group `dev` (uv): `pytest`, `ruff`, `pyrefly`, `beartype`, `hypothesis`.

S3 access uses plain HTTPS against the public bucket endpoints (ListObjectsV2 with `list-type=2`, GET with `Range`), so `boto3` is not needed. `boto3` with `botocore.UNSIGNED` is an acceptable alternative if listing at scale turns out to need it.

## Packaging

The real `pyproject.toml` is in the repository root and is the source of truth: hatchling build backend, static version (switch to `hatch-vcs` once the repo is under git), extras `app`, `analysis` and `all`, a `dev` dependency group for uv, and the `surfcensus` console script. Package data (`data/`, and later `app/static/` and `app/templates/`) lives inside `src/surfcensus/` so it ships in the wheel.

`surfcensus` was free on PyPI on 2026-11-01 (HTTP 404 from `pypi.org/pypi/surfcensus/json`), as were `occiqc`, `occipitalqc`, `pialwatch`, `sinuscheck` and `fscensus`. Treat the name as a placeholder until the paper team agrees.

## CLI

Local users:

```
surfcensus review PATH [--run-dir DIR] [--subjects GLOB] [--rater NAME] [--no-browser]
    # inventory + metrics + scoring + render + serve, in one go
surfcensus export --run-dir DIR [--format csv|parquet]
surfcensus render-subject SUBJECT_DIR --out DIR
    # one subject, no run directory; the CLI form of sc.render.three_plane
```

`review` defaults the run directory to `PATH/../surfcensus-run` if writable, else `./surfcensus-run`. It renders every subject (local sets are small enough), orders the rating queue by composite score descending unless `--blind` is given, and starts the app on `127.0.0.1` with a single local rater.

Census (and power users), one subcommand per phase:

```
surfcensus key init   [--path FILE]          # create the dataset key (census only)
surfcensus discover   --run-dir DIR [--source openneuro-derivatives|openneuro-embedded]
surfcensus inventory  --run-dir DIR [--dataset ds000102 ...]
surfcensus enrich     --run-dir DIR
surfcensus metrics    --run-dir DIR
surfcensus select     --run-dir DIR --n N --seed S
surfcensus render     --run-dir DIR [--backend mpl|freeview] [--jobs J]
surfcensus serve      --run-dir DIR --host 0.0.0.0 --port 8000
surfcensus export     --run-dir DIR [--reveal]   # --reveal writes the pseudonym mapping, needs the key
surfcensus reference build --run-dir DIR --out reference/
```

Every phase command is idempotent and resumable. It skips work whose outputs exist with a matching config hash, and accepts `--dataset` filters so Snakemake can parallelize per dataset. Remote concurrency is bounded (`remote.max_concurrency`, default 8) with exponential backoff on 5xx and S3 `SlowDown`.

## Python API

The CLI is a thin layer over a public API, so the package can be used from notebooks or other pipelines:

```python
import surfcensus as sc

src = sc.sources.local("/data/SUBJECTS_DIR")
run = sc.Run("/data/sc-run", source=src)
run.inventory()
run.metrics()
run.render(jobs=8)
scores = run.scores()  # pandas DataFrame
sc.render.three_plane(subject_dir, out="sub-01.mp4")  # single subject, no run dir
```

## Run directory and data model

```
run/
  config.yaml
  catalog.sqlite         # sources, datasets, subjects, files, provenance, selection, renders
  tables/
    regional.parquet     # subject x hemisphere x parcel
    vertex_summary.parquet
    scores.parquet       # one row per subject
    clusters.parquet     # outlier clusters per subject: hemi, area, peak z, centroid vertex, native tkr RAS
    fsavg/<subject_uid>.npz   # fsaverage6 thickness (and z) maps, float16
    reference/           # per-vertex reference maps per stratum for this run
  templates/             # cached fsaverage sphere, cortex label, search-region label
  cache/                 # remote downloads only, content-addressed, prunable
  media/<subject_uid>/
    anim.mp4
    tiles/               # per-panel WebP slice tiles in two layers, see rendering.md
    poster.webp
    render.json          # geometry, per-panel pixel-to-voxel affines, frame tables, config hash
    vertex_map.npz       # native vertices in the box and their fsaverage6 counterparts
  app.sqlite             # rater app state, separate from the catalog
```

Datasets appear everywhere outside `catalog.sqlite` by keyed pseudonym (`D-3f9a1c07`), never by accession; see decision D10 in `decisions.md`. `privacy.py` owns key loading, pseudonym computation and the reveal path, and nothing else in the package computes or prints accessions for display. `config.yaml` records the key fingerprint.

`subject_uid` is a keyed hash (same key) of `(source, accession or root path, pipeline, subject_label)`, so media paths do not reveal the dataset to raters. Local runs do not need the dataset key: `surfcensus review` generates a random per-run salt, stored in `config.yaml`, since nothing about local identities needs protecting beyond keeping raters blind.

Because subject and dataset identifiers depend on the key, the key is part of a census run's identity. Losing it means the run cannot be extended consistently, so `surfcensus key init` prints a reminder to back it up.

Core tables in `catalog.sqlite`:

- `datasets(dataset_id, pseudonym, source, accession, root, pipeline, pipeline_version, derivative_ref, snapshot_tag, n_subjects_listed, discovered_at)`; for a local source, one row with `root` set to the path.
- `subjects(subject_uid, dataset_id, subject_label, fs_dir, complete, fs_version, recon_flags, t2pial, flairpial, age, sex, field_strength, manufacturer)`
- `files(subject_uid, role, location, size, etag_or_mtime)` with roles `t1`, `brainmask`, `lh_white`, `rh_white`, `lh_pial`, `rh_pial`, `lh_thickness`, `rh_thickness`, `lh_sphere_reg`, `rh_sphere_reg`, `lh_annot`, `rh_annot`, `lh_stats`, `rh_stats`, `aseg_stats`, `talairach`, `recon_log`, `build_stamp`
- `selection(subject_uid, stratum, reason, seed)`
- `renders(subject_uid, config_hash, status, media_path, n_frames, rendered_at)`

## Testing

- Unit tests on synthetic meshes (a sphere, a folded slab) for slicing, normals, pole finding and ROI construction.
- Layout detection tests for the local source on fake directory trees (SUBJECTS_DIR, fMRIPrep old and new layouts, BIDS derivatives).
- Golden-image tests for frame composition with a tolerance, using a tiny synthetic volume.
- One network fixture: a single real subject from a small OpenNeuroDerivatives dataset, fetched once into a pytest cache, marked `network`. The same subject, copied locally, doubles as the end-to-end test of `surfcensus review --no-browser`.
- Property tests (hypothesis) for mesh-plane intersection: every segment endpoint lies on the plane and on a mesh edge.
- Resampling tests: resampling fsaverage onto itself is the identity; a smooth field on a synthetic sphere survives resampling within tolerance; agreement with `mri_surf2surf` on the network fixture within a stated tolerance (run once by a developer with FreeSurfer, stored as a reference array).
- Click round-trip tests: random voxels to panel pixels and back for every panel and slice in `render.json`; marks resolve to the expected vertex on a synthetic mesh.
