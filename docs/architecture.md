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

- `local`: a path. Auto-detects three layouts: a FreeSurfer `SUBJECTS_DIR` (children containing `surf/` and `mri/`); an fMRIPrep output (`sourcedata/freesurfer/` or, for older versions, `freesurfer/` next to the fMRIPrep folder); a BIDS derivatives root containing `freesurfer/`. Skips `fsaverage*` and other template subjects. Accepts `--subjects` globs. Reads in place, never copies or modifies source files. A path that is one subject directory is read as its parent directory restricted to that subject. When both `sourcedata/freesurfer/` and `freesurfer/` hold subjects, the first is used and a warning names the other. A directory with `mri/` and `surf/` that lacks files or cannot be examined is listed with what it has or with its problem, not dropped. Two names for one directory are listed once, under the first in name order. A directory whose name is not valid UTF-8 is skipped with a warning, because a subject label is stored and hashed as text. A source whose root path is not valid UTF-8 is refused for the same reason (`SourceError("invalid")`). A run directory inside the subjects directory is refused.
- `openneuro-derivatives`: the `s3://openneuro-derivatives/fmriprep/` prefixes (see `data_sources.md`).
- `openneuro-embedded`: `derivatives/` folders inside raw OpenNeuro datasets.

Each source implements the `Source` protocol of `sources/base.py`. It yields `DatasetRecord`s and, per dataset, `SubjectRecord`s in label order, from a listing alone: one `FileRef` (location relative to the subject directory, size, ETag or modification time) per file role (`t1`, `lh_pial`, ...) that the subject has. Two calls hand out content. `read_bytes(subject, role)` returns a small text file (build stamp, logs), whole or as a byte range. `lease(subject, roles)` is a context manager that gives local paths for the requested roles while it is open: a local source yields the files where they are, a remote one downloads into the cache and removes them on exit. Both first compare the file with its `FileRef` and raise `SourceError("changed")` when size or modification time differ, so nothing is computed from a file other than the one that was inventoried. `config()` returns the `SourceConfig` that a run stores, from which `run.from_config` builds an equal source. Downstream code only ever sees local paths.

## Scoring with a normative reference

Outlier detection runs in fsaverage space: each subject's thickness is resampled to fsaverage6 through its `sphere.reg` and z-scored per vertex against a reference distribution (see `metrics.md`). For the census, the reference is the census itself, stratified. A local user with 12 subjects cannot build a stable per-vertex reference from their own data. The package therefore ships a normative reference derived from the census as package data (`surfcensus/data/reference/`). It holds per-vertex median, MAD and subject-count maps of globally normalized thickness on fsaverage6 (0.98 MB per stratum, as `.npz` beside a JSON index), stratified by field strength and FreeSurfer major version, plus the regional parcel reference, all computed only from subjects that raters passed. Local scoring reports z-maps against the shipped reference and, when the local set has at least 30 subjects, against the local set. The reference is versioned; scores record which reference they used (`scores.reference_id`). The fsaverage6 template is built at run time from an `fsaverage` directory: the one given with `--fsaverage`, else that of a local FreeSurfer install, else the one next to the subjects. Nothing fsaverage-derived ships in the wheel, so without such a directory there is no per-location scoring (decisions D14 and D22). A run keeps the template it first used.

Until the census has produced ratings, the shipped reference is empty. Local sets of at least 30 subjects are scored within the set; smaller sets, and sets without a template, are ranked on raw ceiling fraction and occipital-to-control thickness ratio only (decision D16).

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
    config.py              # RunConfig, the config.yaml of a run; stage identities
    privacy.py             # keyed pseudonyms, subject_uid, render keys, scrubbing; dataset key, reveal mapping
    store.py               # SQLite catalog (private Catalog, PublicCatalog) + parquet tables; schema migrations
    records.py             # plain records shared between modules (provenance, ledger rows, RunContext)
    schemas.py             # Arrow schemas and keys of the parquet tables
    run.py                 # Run: a run directory, its sources and the phases
    batches.py             # leased, batched per-subject work
    _io.py                 # atomic writes, byte-stable .npz, digests
    log.py                 # the package logger
    sources/
      base.py              # Source protocol, DatasetRecord, SubjectRecord, FileRef, file roles and tiers
      local.py             # SUBJECTS_DIR / fMRIPrep / BIDS-derivatives layouts
      openneuro_derivatives.py
      openneuro_embedded.py
      patterns.py          # path patterns that identify a FreeSurfer subject dir
    remote/
      s3.py                # anonymous S3 listing, ranged GET, retry, rate limit
      openneuro.py         # OpenNeuro GraphQL client
      cache.py             # content-addressed cache keyed by (bucket, key, etag)
    inventory/
      subjects.py          # phase 2 for a run: listing, skip test, subjects and files into the catalog
      provenance.py        # FreeSurfer version, recon-all flags, fMRIPrep version
    enrich/                # census only
      participants.py      # age, sex from participants.tsv
      acquisition.py       # field strength, manufacturer from T1w sidecars
    fsio/
      surfaces.py          # read white/pial in tkr RAS, normals
      volumes.py           # read T1.mgz, vox2ras-tkr, intensity windows
      stats.py             # parse ?h.aparc.stats, aseg.stats
      annot.py             # read ?h.aparc.annot and .label files
      morph.py             # read ?h.thickness
    templates.py           # the fsaverage template: find a directory, derive the target mesh,
                           # cortex, regions and vertex areas, cache it by content
    fsio/registration.py   # sphere.reg mapping: barycentric resampling, vertex correspondences
    metrics/
      __init__.py          # MetricError
      regional.py          # parcel thickness from stats files
      fsaverage.py         # resample thickness to fsaverage6, per-vertex reference, z-maps, strata
      clusters.py          # outlier clusters on the fsaverage mesh, centroids on the native surface
      ceiling.py           # ceiling fraction on native surfaces
      intensity.py         # milestone 5
      scoring.py           # subject metrics, standardisation, composite score, ranks
      pipeline.py          # phase 4 for a run: the stages, their skip tests, their tables
    reference/
      build.py             # build a normative reference from a rated census run (milestone 6)
      load.py              # a reference in memory and on disk
      strata.py            # names of strata and of references, field-strength bins
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
      api.py               # three_plane(), render_subject()
      batch.py             # phase 6 for a run: pending subjects, the renders table, the check sample
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

Core: `nibabel`, `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `imageio-ffmpeg`, `pillow`, `httpx`, `typer`, `pydantic`, `pydantic-settings`, `pyyaml`, `jaxtyping`, `tqdm`.

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
surfcensus render-subject SUBJECT_DIR --out DIR [--fsaverage DIR | --no-template]
    # one subject, no run directory; the CLI form of sc.render.three_plane
```

`review` defaults the run directory to `PATH/../surfcensus-run` if writable, else `./surfcensus-run`. It renders every subject (local sets are small enough), orders the rating queue by composite score descending unless `--blind` is given, and starts the app on `127.0.0.1` with a single local rater.

Census (and power users), one subcommand per phase:

```
surfcensus key init   [--path FILE]          # create the dataset key (census only)
surfcensus discover   --run-dir DIR [--source openneuro-derivatives|openneuro-embedded]
surfcensus inventory  --run-dir DIR [--source PATH] [--subjects GLOB]... [--field-strength T]
                      [--dataset ID]... [--force] [--relocate [PSEUDONYM]]
surfcensus enrich     --run-dir DIR
surfcensus metrics    --run-dir DIR [--stage regional|fsaverage|reference|score|native|merge]...
                      [--dataset ID]... [--jobs J] [--fsaverage DIR] [--retemplate]
                      [--force] [--retry-failed]
surfcensus select     --run-dir DIR --n N --seed S
surfcensus render     --run-dir DIR [--jobs J] [--dataset ID]... [--all | --selected]
                      [--video / --no-video] [--fsaverage DIR] [--force] [--retry-failed]
surfcensus render     --run-dir DIR --check-sample [--fraction 0.02] [--seed S]
surfcensus status     --run-dir DIR [--failures]
surfcensus serve      --run-dir DIR --host 0.0.0.0 --port 8000
surfcensus export     --run-dir DIR [--reveal]   # --reveal writes the pseudonym mapping, needs the key
surfcensus reference build --run-dir DIR --out reference/
```

Every phase command is idempotent and resumable. It skips work whose outputs exist with a matching config hash, and accepts `--dataset` filters so Snakemake can parallelize per dataset. Remote concurrency is bounded (`remote.max_concurrency`, default 8) with exponential backoff on 5xx and S3 `SlowDown`.

`inventory`, `render` and `status` (milestone 2) and `metrics` (milestone 3) are built, for local sources:

- `inventory --source PATH` creates a local run in `DIR` when it holds none, and says which layout it found, how many subjects are complete and have Tier A and Tier B files, their FreeSurfer versions, and which subjects lack what. On a run that exists, `--source` only says which source is meant: it must be the stored root, or come with `--relocate` when the data moved. `--relocate` replaces the stored root and, when the data now lies in another layout of the same pipeline (an fMRIPrep output moved from `sourcedata/freesurfer/` to the older `freesurfer/` beside `fmriprep/`), where the subjects are under it. Nothing else changes, so every `subject_uid`, render and rating stays; with several local sources it takes the pseudonym of the dataset that moved. Data that an inventory finds in the layout of another pipeline (a subjects directory that became an fMRIPrep output) is a new set of subjects with new identifiers: the earlier ones are marked absent, with a note, and keep their renders and ratings (decision D21).
- `render` renders what is pending and prints counts and one line per failure reason. When nothing is pending it reads no source, so it also works while the data is not mounted. A run with a pinned template still reads that template, from `templates/` in the run directory; when that directory was deleted and no fsaverage directory is found to build it from again, `render` stops with `changed: the template this run was scored with is no longer available`, also when nothing is pending (decision D22; `status` works). `--all` renders every complete subject, `--selected` only the selection (which is every complete subject while nothing is selected). `--check-sample` renders nothing: it draws a seeded sample of the done renders for the by-eye check of phase 6 and writes `logs/render-check-<seed>.txt` and `.html`. A `vertex_map.npz` that is there and cannot be read (a truncated copy) is not noticed by the skip test, which looks at names; the `native` stage at the end of `render` notices it when it next reads the vertex maps, leaves that subject's centroids without a native position and says so in a note with the `subject_uid`, at every call until the file is removed and the render made again.
- `metrics` computes what is pending of phase 4 and prints the counts, one line per failure reason, the notes of the report and one line that says what the subjects were scored against. Without `--stage` it runs the stages `regional`, `fsaverage`, `reference` and `score`; with `--dataset` and no `--stage`, only the two that work per subject, since a stage on the whole run is not restricted by a dataset. `native` (cluster centroids on the native surface, from the vertex maps of the renders) also runs at the end of every `render`, and not at the end of `metrics`. So when `metrics` finds clusters on a run that is already rendered, for the first time or because the reference was built again, those clusters have no native position until the next `render` (one with nothing to render will do) or `metrics --stage native`. `Run.clusters()` joins by reference, so a position of an earlier reference is never attached to a cluster of the present one; until then `clusters_native.parquet` still holds the rows of the earlier reference. `merge` joins the per-dataset parts of each table. A present subject that no longer has any stats file, or any thickness file, is no unit of `regional`, or of `fsaverage`, any more: the stage removes its rows, its map and its ledger row, so the scores do not keep numbers from files that are gone, and the subject is scored like one that never had them (`no_stats`, `no_thickness`). An fsaverage map of the run that cannot be read when the `reference` stage needs it stops the command with exit code 2; the damaged maps are removed, and the next `metrics` makes them again from the sources. `--fsaverage DIR` gives the template for this call; the directory is not written to `config.yaml`, because the run keeps the template itself, by content, from its first use. A directory that builds another template than the one the run is pinned to stops the command (`changed: <old> -> <new>; pass --retemplate`); `--retemplate` takes the new one, and every map is made again. `render` takes `--fsaverage` as well.
- `status` prints one line per phase from the catalog: subjects inventoried and unreadable; for each metric stage that has run, its units done and failed (the rows of the ledger: a `native` line says that the stage ran, not that its table follows the present clusters); renders done, failed and pending, where pending is what `render` would do now. `--failures` lists every failed unit with its reason and detail; for the metrics these are the failed rows of the ledger; for the render these are the subjects counted as failed on the line above, so an incomplete subject is listed before `render` has recorded it, and a recorded failure of a subject that left the source or is pending again is not. It reads the private catalog, so its output is for the operator's terminal, not for a log.
- `--dataset` takes a pseudonym or an accession. An accession given on the command line is resolved in the catalog and stored nowhere.
- Options that are stored in `config.yaml` (`--subjects`, `--field-strength`, `--video`) hold until they are given again: a command run a second time without them changes nothing. A subject glob that matches nothing is refused before it is stored.
- Exit codes: 0 when no unit is in the failed state when the command ends; 1 when at least one is, also when it failed in an earlier call and was not tried again; 2 for a usage or configuration error, including a source that cannot be found, printed as one line `reason: detail`. What lies in the machine and not in a subject is such an error as well and is recorded against no subject: a run directory that cannot be made or written, a catalog that another process keeps locked, media on a full or read-only disk, a video encoder that is absent or cannot encode, a template directory that was given and cannot be used, a template that differs from the one the run is pinned to, an fsaverage map in `tables/fsavg/` that cannot be read. Once the cause is gone, the same command does the work (decision D29). `status` reports and always ends with 0.
- Two commands that create a run in the same directory at the same time: one creates it, the other either opens it or is refused with exit code 2 and changes nothing. A command that opens a run while another is still creating it gets the same run. A command that is killed while it creates a run leaves no run, or one that the next command finishes: `config.yaml` alone, or with a catalog in which no run was made yet.
- `--backend mpl|freeview` is not offered until the Freeview backend exists.

## Python API

The CLI is a thin layer over a public API, so the package can be used from notebooks or other pipelines:

```python
import surfcensus as sc

src = sc.sources.local("/data/SUBJECTS_DIR")
run = sc.Run("/data/sc-run", source=src)
run.inventory()
run.metrics()
run.render(jobs=8)
scores = run.scores()  # pandas DataFrame, one row per subject, by rank
clusters = run.clusters()  # outlier clusters, with native positions where a render gives them
z = run.zmap(scores["subject_uid"][0])  # {"lh": ..., "rh": ...} on the template mesh
sc.render.three_plane(subject_dir, out="sub-01.mp4")  # single subject, no run dir
```

`Run(run_dir, source)` creates a run in a new or empty directory; `Run(run_dir)` opens one, without a key also for a census run (only `inventory` and, later, `discover` need the key). Each phase method returns a `PhaseReport` with the counts of units done, skipped and failed. `run.template()` returns the fsaverage template of the run, or None for a run that never had one; it stores the template in the run directory and pins the run to it on first use, and raises when a pinned template can neither be read nor built again. `run.zmap(subject_uid)` computes a subject's z-map from its stored thickness map and the reference; z-maps are not stored.

`Run` holds the private catalog and only delegates. A phase function receives a `RunContext` (`records.py`), never the `Run`: the run directory, the configuration, a `PublicCatalog`, a `lease(subject_uid, roles)` function that looks the subject up and leases through its source, a `render_key` function with the run secret bound, and a `scrub` function for the text of exceptions. So code downstream of the inventory names subjects by `subject_uid` and datasets by pseudonym, and cannot reach an accession, a root or an input signature even by attribute access. `run.public()` and `PublicCatalog.open(run_dir)` give the same public side to the app and to exports.

## Run directory and data model

```
run/
  config.yaml            # RunConfig. A local run: its salt and the root of its source. Never the
                         # dataset key, the run secret or an accession; in a census run no path either
  catalog.sqlite         # datasets, subjects, files, provenance, the work ledger, selection, renders
  tables/
    regional.parquet     # subject x hemisphere x parcel
    vertex_summary.parquet    # subject x hemisphere: resampling facts, ceiling counts, flags
    scores.parquet       # one row per subject and reference; one of them primary per subject
    clusters.parquet     # outlier clusters per subject, reference, hemisphere and region:
                         # area, mass, excess mass, peak z, peak and centroid vertex
    clusters_native.parquet   # the clusters whose centroid has a native position in a render
    _parts/<table>/<pseudonym>.parquet   # the rows of one dataset; merged into the table above
    fsavg/<subject_uid>.npz   # fsaverage6 thickness maps of both hemispheres, float16, 164 kB
    reference/<reference_id>/ # a reference built in this run: index.json (written last, marks it
                              # complete), stratum_<nnn>.npz (median, MAD, count), members.parquet
  templates/<template_id>/    # the template of the run: template.json, template.npz (5.9 MB);
                              # written again when it does not read back
  cache/                 # remote downloads only, content-addressed, prunable
  logs/                  # render-check-<seed>.txt and .html; in a census run also the phase logs
  media/<subject_uid>/
    anim.mp4
    tiles/               # per-panel WebP slice tiles in two layers, see rendering.md
    poster.webp
    render.json          # geometry, per-panel pixel-to-voxel affines, frame tables, config hash
    vertex_map.npz       # native vertices the tiles show, plus a margin, and their fsaverage6 counterparts
  app.sqlite             # rater app state, separate from the catalog
```

Datasets appear everywhere outside `catalog.sqlite` by keyed pseudonym (`D-3f9a1c07`), never by accession; see decisions D10 and D20 in `decisions.md`. `privacy.py` owns every keyed identifier (the pseudonym, `subject_uid`, `render_key`, the names of cached files), the scrubbing of text that comes from exceptions and, from milestone 5, key loading and the reveal path. Nothing else in the package computes one of these or prints an accession for display. `config.yaml` records the key fingerprint.

`subject_uid` is 16 hex characters of a keyed hash (same key) of `(source, identity, pipeline, subject_label)`, so media paths do not reveal the dataset to raters. `identity` is the accession of a remote dataset. For a local one it is the resolved root path when the run was created; it is stored and never recomputed, so data that moved (`--relocate`) keeps every `subject_uid` and with it every render and rating. Local runs do not need the dataset key: a new local run draws a random per-run salt, stored in `config.yaml`, since nothing about local identities needs protecting beyond keeping raters blind.

Because subject and dataset identifiers depend on the key, the key is part of a census run's identity. Losing it means the run cannot be extended consistently, so `surfcensus key init` prints a reminder to back it up.

A second secret, the run secret, is 32 random bytes drawn when the catalog is created and kept in it (`meta.run_secret`), in both modes. `render_key` and the names of cached files are keyed with it. Both are derived from files whose sizes and ETags are public, so an unkeyed hash of them could be recomputed from the bucket listing by anyone who holds an export or a media URL. `render_key` says which tiles a rating was made on: it is built from `RenderConfig.tile_hash()` and the SHA-256 digests of the files the render read, so a file that was only touched or moved keeps its key.

`catalog.sqlite` is SQLite through the standard library, with `PRAGMA user_version` migrations. A migration is one transaction, and each statement of a migration script ends at the end of a line. A local run uses the default rollback journal, because run directories on network file systems are common and SQLite's write-ahead log does not work there, with a busy timeout of 30 s on every connection. A census run uses the write-ahead log, with 60 s for writers and 5 s for read-only connections, and warns when its directory is on a network file system (detected on Linux only, from `/proc/mounts`). Every write is one short `BEGIN IMMEDIATE` transaction, so several processes can write to one catalog.

A new catalog is built under a temporary name in the run directory and gets the name `catalog.sqlite` through a hard link when it holds the schema, the identity of the run and the run secret. So `catalog.sqlite` holds a run or does not exist, whenever the process ends, and no other process opens one that is half made. On a file system without hard links the name is taken with an empty file and the run is made in that file in one transaction. A catalog in which no run was made yet (an empty file, or the tables without a row) is never migrated or removed: opening the run makes the run in it, under SQLite's write lock, with the salt of `config.yaml` or, for a census run, the key. A database that holds anything else is refused as not made by surfcensus and left as it is.

Tables in `catalog.sqlite` (the DDL is migration 1 in `store.py`):

- `meta(key, value)`: run id, mode, key kind and fingerprint, creation time, package version, the id of the template the run is pinned to (`template_id`, from its first use); and the run secret.
- `datasets(dataset_id, pseudonym, source, identity, accession, root, fs_root, template_location, layout, pipeline, pipeline_version, derivative_ref, snapshot_tag, n_subjects_listed, ..., discovered_at)`, unique on `(source, identity, pipeline)`; for a local source, one row with `root` set to the path.
- `subjects(subject_uid, dataset_id, subject_label, participant, fs_dir, present, complete, tier_a, tier_b, missing_roles, problem, longitudinal, fs_build_stamp, fs_version, fs_major, recon_flags, n_invocations, recon_finished, t2pial, flairpial, hires, expert, xopts, edited, pial_variant, holes_lh, holes_rh, trunc_lh_1, trunc_lh_2, trunc_rh_1, trunc_rh_2, age, sex, field_strength, manufacturer, inventory_hash, inventoried_at)`. `complete` means every file a render needs is present; `tier_a` and `tier_b` that the stats files, and the thickness and `sphere.reg` files, are. `present = 0` marks a subject that the source no longer lists; its row, render and ratings stay.
- `files(subject_uid, role, location, variant, size, etag_or_mtime, last_modified, version_id, sha256)` with roles `t1`, `brainmask`, `lh_white`, `rh_white`, `lh_pial`, `rh_pial`, `lh_thickness`, `rh_thickness`, `lh_sphere_reg`, `rh_sphere_reg`, `lh_annot`, `rh_annot`, `lh_stats`, `rh_stats`, `aseg_stats`, `talairach`, `recon_log`, `build_stamp`, `recon_log_lh`, `recon_log_rh`. `location` is relative to the subject directory, so the table holds no accession. `sha256` is set when a file was read in full: by the inventory for the build stamp and the recon-all logs (not for a log of which only the head was read), by the metrics for the `aparc.stats`, thickness, annotation and `sphere.reg` files that a stage read (the `sphere.reg` only in a run with a template; neither annotation nor `sphere.reg` of a hemisphere without a thickness file; nothing from a stage that failed for the subject), by a render for the T1, the surfaces, the brainmask and, with a template, the `sphere.reg` files. `aseg.stats` and `talairach.xfm` are read by nothing yet and have none.
- `dataset_files(...)`: dataset-level files that a stored value was read from (`dataset_description.json`, `participants.tsv`, a T1w sidecar). Empty until milestone 5.
- `templates(template_id, mesh, origin, dataset_id, build_stamp, files_json, created_at)`: every template the run has used, with the size and SHA-256 of the fsaverage files it was built from.
- `work(phase, unit, config_hash, input_signature, status, reason, detail, output, surfcensus_version, finished_at)`: the ledger of every phase but inventory and render; a unit is a `subject_uid`, `dataset:<pseudonym>` or `run`. The metric stages are the phases `regional` and `fsaverage` (one unit per subject) and `reference`, `score` and `native` (the unit `run`).
- `renders(subject_uid, config_hash, input_signature, tile_hash, render_key, video, status, reason, detail, media_path, n_frames, template_id, rendered_at)`
- `selection(subject_uid, reason, stratum, inclusion_prob, seed)`

What is private stays in the catalog and is reached only through `store.Catalog`, which `run.py`, `cli.py` and the inventory use: `dataset_id` (assigned in the order of the bucket listing), identity, accession, root, `fs_root`, template location, derivative ref and snapshot tag; `fs_dir`; file locations, sizes, ETags and digests; every input signature; the run secret; and in a census run the subject labels and every failure detail (a local run shows the details of its own renders). Three views (`public_datasets`, `public_subjects`, `public_files`) leave these out, and no frame that `store.PublicCatalog` returns holds any of them. It hands out an input signature only as a value to pass back for the skip test. Everything `PublicCatalog` returns is ordered by pseudonym or `subject_uid`, never by `dataset_id`.

An input signature is 16 hex characters of the SHA-256 over role, location, size and ETag or modification time of the files a stage reads, and for a role with several candidate names the name that was read (`variant`), which is what changes when a `?h.pial` link is pointed at another variant. It comes from the listing, so deciding to skip a unit opens no source file. The skip rule is the same in every phase: a unit is skipped when its row is `done`, its config hash and input signature are those of the present configuration and listing, and its output exists. For a render the output is the whole media directory: `render.json` and a file under every name it lists, so a directory that lost files (a partial copy, a restore) is rendered again. A `failed` row that matches is not tried again unless `--retry-failed`; a changed configuration or input tries it again; `--force` ignores the row. Outputs are written first and the row last, so a crash leaves an output without a row, which the next run redoes, never a row without its output. A subject that is no longer a unit of a metric stage, because the files that made it one are gone, loses its outputs first and its row last, in the same order. The inventory has no ledger rows; its skip test is `subjects.inventory_hash`, the signature over all listed files of the subject. The stages on the whole run have signatures of their own: that of `reference` covers the subjects that have a map, with the signatures of their files, their dataset, field strength and FreeSurfer version; that of `score` the content of the three tables it reads, what it takes from the catalog and who is scored against which reference; that of `native` the content of `clusters.parquet` and the render keys. So one more subject redoes the stages on the run and nothing per subject.

## Testing

- Unit tests on synthetic meshes (ellipsoids, and a folded closed surface that a slice crosses in several loops) for slicing, contour drawing, normals, pole finding and ROI construction.
- Layout detection tests for the local source on fake directory trees (SUBJECTS_DIR, fMRIPrep old and new layouts, BIDS derivatives).
- Golden-image tests for frame composition with a tolerance, using a tiny synthetic volume.
- Network fixtures, marked `network`: `ds000102` sub-01 (FreeSurfer 6.0.1; Tier A, B and C, 43 MB) and `ds003020` sub-UTS01 (FreeSurfer 7.3.2; Tier A, B and C, 37 MB) and the template files of the `fsaverage` next to the subjects of `ds000102` (40 MB), each fetched once into the pytest cache. Linked into an fMRIPrep layout, they are the end-to-end test of inventory, metrics and render through `Run`, and later of `surfcensus review --no-browser`.
- Property tests (hypothesis) for mesh-plane intersection: every segment endpoint lies on the plane and on a mesh edge.
- Resampling tests: resampling fsaverage onto itself is the identity; a smooth field on a synthetic sphere survives resampling within tolerance; agreement with `mri_surf2surf` on the network fixture within a stated tolerance (run once by a developer with FreeSurfer, stored as a reference array; not made yet, and the test is skipped until it is).
- Metric fixtures: a synthetic fsaverage directory of the real size (163,842 vertices, coarsened to fsaverage6 by the code under test) with Desikan parcels laid out by rule, and synthetic subjects whose `sphere.reg` is a known warp and whose thickness is a known function of the template position, with a 5 mm patch planted in some. A cohort of 32 crosses the 30-subject rule; one of 5 does not. `small_run` is a run of three such subjects that went through inventory, metrics and render. `planted_run` is a run of three with values that the cohorts lack planted in single hemispheres (vertices at the ceiling in the control region, near the ceiling, at 0 and above the maximum, a sphere with folded faces and one wound inward), for the tests of what the stages copy into their tables.
- Click round-trip tests: random voxels to panel pixels and back for every panel and slice in `render.json`; marks resolve to the expected vertex on a synthetic mesh.
