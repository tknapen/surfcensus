# Architecture

## Two uses, one pipeline

The package serves two audiences with the same code path.

1. A researcher with their own FreeSurfer subjects (a `SUBJECTS_DIR`, an fMRIPrep output folder, a BIDS dataset with FreeSurfer subjects below `derivatives/`, or any folder of FreeSurfer subject directories) wants to score, render and rate them. This must work in one command with no cloud access:

   ```
   pip install surfcensus
   surfcensus review /path/to/SUBJECTS_DIR
   ```

   That command inventories the subjects, computes metrics, renders the occipital animations, and opens the rater app in the browser. Ratings are saved in the run directory and exported with `surfcensus export`.

   The install is that one line: the packages of the rater app are dependencies of the package (decision O20), so no extra is needed, and `uvx surfcensus review /path/to/SUBJECTS_DIR` does both steps without an environment of the user's. `pip install "surfcensus[app]"`, the line these docs gave while the app was an extra, still works: the extra exists and is empty. The package is not on PyPI (O1). Until it is, the two lines name the repository: `pip install git+https://github.com/tknapen/surfcensus`, and `uvx --from git+https://github.com/tknapen/surfcensus surfcensus review /path/to/SUBJECTS_DIR`.

   This quick start is a main target of the package (decision D33): one line installs it, and one command, pointed at a folder that holds the FreeSurfer outputs of an OpenNeuro dataset or of any other study, shows the rating screen without a further step. The browser opens as soon as the first subject of the queue is rendered, and the rest is rendered behind the running app. What of this is built is said under D33.

2. The paper team runs the census over OpenNeuro. The census is the same pipeline with a remote source, plus discovery and enrichment phases that local users never need.

So the pipeline is organized around a `Source` abstraction. Everything downstream of the source (inventory, metrics, scoring, rendering, rating, export) is identical for local and remote subjects.

```
Source  ->  inventory  ->  metrics  ->  scoring  ->  select  ->  render  ->  rate  ->  export
(local | openneuro-derivatives | openneuro-embedded)
```

## Sources

- `local`: a path. Auto-detects three layouts: a FreeSurfer `SUBJECTS_DIR` (children containing `surf/` and `mri/`); an fMRIPrep output (`sourcedata/freesurfer/` or, for older versions, `freesurfer/` next to the fMRIPrep folder); a BIDS derivatives root containing `freesurfer/`. Skips `fsaverage*` and other template subjects. Accepts `--subjects` globs. Reads in place, never copies or modifies source files. A path that is one subject directory is read as its parent directory restricted to that subject. When both `sourcedata/freesurfer/` and `freesurfer/` hold subjects, the first is used and a warning names the other. A directory with `mri/` and `surf/` that lacks files or cannot be examined is listed with what it has or with its problem, not dropped. Two names for one directory are listed once, under the first in name order. A directory whose name is not valid UTF-8 is skipped with a warning, because a subject label is stored and hashed as text. A source whose root path is not valid UTF-8 is refused for the same reason (`SourceError("invalid")`). A run directory inside the subjects directory is refused, and so is an export directory there (`export --out`, `export_run`): nothing the package writes goes into a source. A subject directory that is a link (a study that takes its subjects from a shared store, the layout of the network tests) leads to a directory that is read as much as the subjects directory is, and so does a linked `fsaverage` next to the subjects: a run directory or an export directory that is, or lies inside, what such a link leads to is refused as well, whether it is named through the link or at its target (`sources.local.source_places`, `refuse_in_sources`). The check is made before `config.yaml` of a new run is written, so a refused run leaves nothing. A run that was moved into its source afterwards is not written to: `review`, `inventory`, `metrics` and `render` refuse it when they build the source, and `serve`, `finalize`, `calibration freeze` and `export` (also without `--out`) refuse it from the roots in `config.yaml`; `status`, which only reads, works. Of the fsaverage directories outside the subjects directory, a call knows the one it is given (`--fsaverage DIR`) and that of a FreeSurfer install (`$FREESURFER_HOME/subjects/fsaverage`), and refuses a run directory or an output in either. The directory a pinned template was once built from is not stored (the run keeps the template by content, decision D22), so a later call that does not name it cannot refuse an output there. Not followed: a file that is a link inside a plain subject directory. The renderer refuses an output beside such a file's target when it reads the file (`rendering.md`); the check of a run directory and of an export does not look at single files.

  The root of a BIDS dataset, as OpenNeuro hands one out, is a path as well, and so is its `derivatives/` folder: the FreeSurfer subjects are found below `derivatives/` without the user naming the subfolder (decision D33). A path counts as such a root when it has a `derivatives/` folder, or is that folder (by its name, or the name of what it links to); a `dataset_description.json` is not needed. This is a way to find a folder, not a fourth layout: the source is the one that the folder found gives when it is named itself, with that folder's root, identity, layout and pipeline. So `review DATASET`, `review DATASET/derivatives` and `review DATASET/derivatives/fmriprep` are one source with the same `subject_uid`s, and a run stores the folder that was found, not the dataset root.

  - Where it looks, first match first: `derivatives/` itself, read by the rules above (which finds `derivatives/freesurfer/`, and reads it as the older fMRIPrep layout when `derivatives/fmriprep/` is beside it); then the folders of `derivatives/` that hold subject directories themselves (`derivatives/freesurfer-7.3.2/`), in name order; then those with `sourcedata/freesurfer/` (`derivatives/fmriprep/sourcedata/freesurfer/`), in name order; then those with `freesurfer/` (`derivatives/fmriprep/freesurfer/`, `derivatives/smriprep/freesurfer/`), in name order. A pipeline folder is matched by what it holds, not by its name, so versioned and otherwise named folders (`fmriprep-21.0.2`) are found. The deepest place that is reached is `derivatives/<folder>/sourcedata/freesurfer/`, three folders below `derivatives/`, as in phase 1 of `phases.md`.
  - Not matched: anything deeper or under another name inside a pipeline folder (`derivatives/<folder>/ses-01/...`, `derivatives/<folder>/freesurfer-7.3.2/`); a folder of `derivatives/` that is itself a subject or is named like a template; subjects under a top-level folder other than `derivatives/` (`sourcedata/freesurfer/` and `freesurfer/` at the top are the layouts above). Such a folder is given as the path itself.
  - Several places with subjects: the first in that order is used, and one warning names the others with the folder to give for each (`... using derivatives/freesurfer, not derivatives/fmriprep/sourcedata/freesurfer (to use it, give /data/ds/derivatives/fmriprep as the path)`). A path that the rules above match is read as before, also when its `derivatives/` holds subjects; the warning then names those. One subjects directory that is reached from several folders counts once.
  - A folder below `derivatives/` that cannot be examined is passed over with a warning when another holds subjects, and is named by the error when none does.
  - Nothing found: `layout`, in one or two lines, with exit code 2. A path with `dataset_description.json` and `sub-*` folders and no `derivatives/` is told that it looks like a BIDS dataset without derivatives and where subjects were looked for. With a `derivatives/` folder the message names the folders that were searched and the folders for FreeSurfer output that hold no subject (`derivatives/freesurfer (empty)`).
  - Identity and existing runs (decision D21). The dataset root gets no pipeline and no identity of its own, so no `subject_uid` of a path that the rules above match changes, and a run made before dataset roots were read opens as it was. A source that is rebuilt from the root a run stores is read by the rules above only, never by this search: a run whose subjects are gone from its folder stops with `layout`, and does not go on with the subjects of another folder under the same identity. `--relocate` takes a dataset root as the new place and stores the folder found there. The subjects directory named itself (`DATASET/derivatives/fmriprep/sourcedata/freesurfer`) is a plain subjects directory with a root, and in an fMRIPrep output a pipeline, of its own, as it always was: on a run of the dataset it is refused as another root. A run that was made of the subjects directory follows to the dataset root with `--relocate` when both are read as one pipeline (`derivatives/freesurfer/` without an `fmriprep/` folder beside it: the stored root and where the subjects lie under it change, every `subject_uid` stays); when they are not, `--relocate` is refused, since the subjects would get other identifiers.
  - The template next to the subjects (`fsaverage` in the same folder) is found as in every layout.

  A DataLad dataset, or any git-annex repository, that was cloned and not fetched holds files by name without their content (decision D33). The package fetches nothing, runs no git, git-annex or DataLad command, and writes nothing into the dataset; it says what is missing and how to fetch it.

  - A file whose content is not there is, in a locked branch (the default), a link into `.git/annex/objects/` whose target does not exist, and, in an unlocked or adjusted branch, a pointer file, a small text file whose first line is `/annex/objects/<key>` (git-annex 10.20261026, `Annex/Link.hs`; checked on a clone made with DataLad 1.8.0 on 2026-11-14). The key carries the size (`MD5E-s613--...mgz`). A link is told from the listing alone (`readlink`). A pointer file is a regular file: of a local source, the inventory reads the first 1,024 bytes of every file of at most 32,768 bytes (the largest pointer git-annex takes) of a subject it reads anyway, and the build stamp and the logs it reads whole; a subject that the skip test passes over is not opened. Such a file is no file of the subject: it is not in `files`, so the subject is incomplete when a render file is among them and lacks Tier A or B accordingly, and provenance that a log without content would give is unknown, not false.
  - The subject's `problem` names them, with the size from the keys (none in a census run): `not fetched: 5 files are git-annex links whose content is not here: mri/T1.mgz, surf/lh.white, surf/rh.white, surf/lh.pial.T1, surf/rh.pial.T1; 23.2 MB`. That is no failure of the inventory (`unreadable: ...` is), and `status` does not count it as one. The render row is `missing surface`, the unratable reason of the protocol, with the sentence in its detail; `status --failures` and the `problem` column of the export carry it.
  - `review` and `inventory` print one command for all such subjects of the run, after the lines that name them: `Fetch them (23.2 MB) with: datalad -C /data/ds get derivatives/freesurfer/sub-01/mri/T1.mgz ...` for one subject, and for several `Fetch the files of these 26 subjects (1.2 GB) with: cd /data/ds && datalad get derivatives/freesurfer/sub-*/mri/T1.mgz ...`, one shell pattern per file name. The command runs in the DataLad dataset that holds the folder named (or the innermost below it), with paths relative to it; `datalad get` installs a subdataset on the way and passes over files whose content is there. Without DataLad it is `git -C <repository> annex get <paths>`. FreeSurfer 7's `lh.pial -> lh.pial.T1`, a link in git, is fetched as `lh.pial.T1`.
  - Once the files are fetched their size and modification time are those of the content, so the next call inventories, scores and renders the subject without `--force`.
  - A subdataset that is not installed is an empty folder that the `.gitmodules` of the repository above registers. No subject can be listed in it, so the search below `derivatives/` ends with only that: `layout: no FreeSurfer subject found in /data/ds: derivatives is a subdataset that is not installed (an empty folder); install it, without the file content, with: datalad -C /data/ds get -n derivatives`; a path inside it gets the same sentence after `not found`. Without `.datalad/` the command is `git -C <repository> submodule update --init <folder>`.
  - A `dataset_description.json` that is an annex link without content marks a dataset as one with content does (DataLad annexed it in the clone that was tried).
  - An `fsaverage` next to the subjects whose missing template files are all such links is named in the template lookup as `not fetched (git-annex): <files>; fetch them with: datalad -C <dataset> get <paths>`, not as incomplete; the fetch line of the subjects does not cover it. When no subject of the run can be rendered and a fetch line was printed, `review` ends with `No subject of this run can be rendered, so there is nothing to rate. Fetch the files as said above, then run the same command again.`
- `openneuro-derivatives`: the `s3://openneuro-derivatives/fmriprep/` prefixes (see `data_sources.md`).
- `openneuro-embedded`: `derivatives/` folders inside raw OpenNeuro datasets.

Each source implements the `Source` protocol of `sources/base.py`. It yields `DatasetRecord`s and, per dataset, `SubjectRecord`s in label order, from a listing alone: one `FileRef` (location relative to the subject directory, size, ETag or modification time) per file role (`t1`, `lh_pial`, ...) that the subject has. Two calls hand out content. `read_bytes(subject, role)` returns a small text file (build stamp, logs), whole or as a byte range. `lease(subject, roles)` is a context manager that gives local paths for the requested roles while it is open: a local source yields the files where they are, a remote one downloads into the cache and removes them on exit. Both first compare the file with its `FileRef` and raise `SourceError("changed")` when size or modification time differ, so nothing is computed from a file other than the one that was inventoried. `config()` returns the `SourceConfig` that a run stores, from which `run.from_config` builds an equal source. Downstream code only ever sees local paths.

## Scoring with a normative reference

Outlier detection runs in fsaverage space: each subject's thickness is resampled to fsaverage6 through its `sphere.reg` and z-scored per vertex against a reference distribution (see `metrics.md`). For the census, the reference is the census itself, stratified. A local user with 12 subjects cannot build a stable per-vertex reference from their own data. The package therefore ships a normative reference derived from the census as package data (`surfcensus/data/reference/`). It holds per-vertex median, MAD and subject-count maps of globally normalized thickness on fsaverage6 (0.98 MB per stratum, as `.npz` beside a JSON index), stratified by field strength and FreeSurfer major version, plus the regional parcel reference, all computed only from subjects that raters passed. Local scoring reports z-maps against the shipped reference and, when the local set has at least 30 subjects, against the local set. The reference is versioned; scores record which reference they used (`scores.reference_id`). The fsaverage6 template is built at run time from an `fsaverage` directory: the one given with `--fsaverage`, else that of a local FreeSurfer install, else the one next to the subjects, looked for when the template is first needed, so a directory put there after the inventory is found. Nothing fsaverage-derived ships in the wheel, so without such a directory there is no per-location scoring (decisions D14 and D22). A run keeps the template it first used.

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
    protocol.py            # the rating protocol of a run (protocol.yaml): scale, criteria, keys, unratable reasons
    calibration.py         # the calibration set of a run (calibration.yaml)
    review.py              # `surfcensus review`: inventory, metrics, selection, render, app (milestone 4)
    stopping.py            # SIGTERM and Ctrl-C of the commands: a stop flag without a lock, SIG_IGN at the end
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
    func/                  # the EPI movie interface, steps 1 and 2 (docs/epi_interface.md)
      bids.py              # BIDS entities, run keys, raw series, sidecar inheritance
      itk.py               # ITK text transforms as RAS matrices
      connectors/          # what is specific to a pipeline; fmriprep.py is the one built
      session.py           # anchor, raw affine, frame check, grids of a session
      geometry.py          # display grid, voxel map, the raw stream's one resampling
      contours.py          # white and pial contours in display voxels
      model.py             # session.json, pack.json, traces.json, stream.json
      pack.py              # the display cache: format epi-pack version 2
      packread.py          # a decoder of packs, for tests and checks
      api.py               # epi_subject, epi_session (surfcensus epi-session)
      discover.py          # step 2: the derivative and the raw dataset of a run, the places that are read
      inventory.py         # step 2: phase 2f (the functional inventory), the lease of a session, fetch lines
      batch.py             # step 2: the pack phase of a run (epi/), its skip rule and its rows
    app/
      __init__.py          # the name of app.sqlite and of its tables, for code that reads it without the app's packages
      schemas.py           # the shapes of the API (pydantic only); RenderOut, the render a page receives
      server.py            # FastAPI
      models.py            # SQLModel tables
      db.py                # opening app.sqlite, its schema version, storing a submission
      queue.py
      events.py            # interaction events
      finalize.py          # final labels and marks; unratable labels for failed renders; freezing a calibration set
      marks.py             # click -> voxel -> tkr/scanner RAS -> native vertex -> fsaverage6 vertex
      epi.py               # the EPI pages: a server for packs alone (epi-session --serve), and Packs, which the app of a run serves
      epi_schemas.py       # the shapes the EPI page receives
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

Core: `nibabel`, `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `imageio-ffmpeg`, `pillow`, `httpx`, `typer`, `pydantic`, `pydantic-settings`, `pyyaml`, `jaxtyping`, `tqdm`, and the packages of the rater app: `fastapi`, `uvicorn`, `sqlmodel`, `jinja2`.

The app's packages were an extra until the quick start became a main target (decisions D33 and O20). They are imported by the modules under `app/` alone, and only by the commands that serve or read the app's database through them (`review` with the app, `serve`, `finalize`, `calibration freeze`): `surfcensus --help`, `render-subject`, `inventory`, `metrics`, `render`, `status`, `export` and `review --no-serve` load none of them, which a test checks command by command. Importing the four takes about 0.55 s (measured 2026-11-14, Python 3.13, one Linux machine), against 0.15 s for `surfcensus --help` as a whole. `uvicorn` is required without its `standard` extra: the server names `h11` and `asyncio` (decision D34), so `httptools`, `uvloop`, `watchfiles` and `websockets` would be installed and never used (four packages, 5.7 MB of wheels on Linux and Python 3.13).

What a plain install pulls in (the wheel of 2026-11-14 with `pip` on Linux x86_64 and Python 3.13, nothing pinned): 50 packages with the package itself, 227 MB of wheels, of which `imageio-ffmpeg` is 68 MB, `pyarrow` 55 MB and `scipy` 35 MB, and about 0.8 GB on disk. The app's four packages bring five more (`starlette`, `sqlalchemy`, `click`, `markupsafe`, `opentelemetry-api`), 5.4 MB together, of which `sqlalchemy` is 4.7 MB; without them the install was 41 packages and 222 MB. Python 3.11 gets one package more (`importlib-resources`).

An installation that lacks one of the app's packages all the same (a package removed, or the package installed with `--no-deps` into an environment that has the core packages and not those of the app) is told so by the commands of the app, in one line with exit code 2 that names the package and `pip install surfcensus`, which installs what is missing (tried 2026-11-14 with pip 26.2.2: it keeps the installed `surfcensus` and installs its missing dependencies); the other commands work in it. An installation without the core packages (`--no-deps` alone) ends with a traceback at the first import, as any Python package does.

Extras:

- `app`: empty. It is kept as a name so that `pip install "surfcensus[app]"` still installs.
- `freeview`: no Python deps; requires a FreeSurfer install and `xvfb-run` at runtime.
- `analysis`: `marimo`, `statsmodels`, `scikit-learn`, `krippendorff`.
- `all`: `app` and `analysis`.

Dependency group `dev` (uv): `pytest`, `ruff`, `pyrefly`, `beartype`, `hypothesis`, and `uvicorn[standard]`, so that the default tests run the server in an environment that has `httptools` and `uvloop`, which it must not pick up; the install test runs it in one that has neither. Dependency group `browser`: `playwright`, for the browser test of the rater app, which drives a browser that is installed already (the executable named by `SURFCENSUS_BROWSER`, else Chrome) and is not part of the default tests or of CI.

S3 access uses plain HTTPS against the public bucket endpoints (ListObjectsV2 with `list-type=2`, GET with `Range`), so `boto3` is not needed. `boto3` with `botocore.UNSIGNED` is an acceptable alternative if listing at scale turns out to need it.

## Packaging

The real `pyproject.toml` is in the repository root and is the source of truth: hatchling build backend, static version (switch to `hatch-vcs` once the repo is under git), extras `app` (empty), `analysis` and `all`, the dependency groups `dev` and `browser` for uv, and the `surfcensus` console script. Package data (`data/`, `app/static/` and `app/templates/`) lives inside `src/surfcensus/` so it ships in the wheel. Beside the modules the wheel holds the scripts, the style sheet and the templates of the app, `data/protocol.yaml` and `data/reference/README.md`, and nothing else: nothing fsaverage-derived (decision D22), nothing from `tests/` or `prototypes/`. The install test compares the wheel's file list with that.

`surfcensus` was free on PyPI on 2026-11-01 (HTTP 404 from `pypi.org/pypi/surfcensus/json`), as were `occiqc`, `occipitalqc`, `pialwatch`, `sinuscheck` and `fscensus`. Treat the name as a placeholder until the paper team agrees.

## CLI

Local users:

```
surfcensus review PATH [--run-dir DIR] [--subjects GLOB]... [--rater NAME] [--blind / --no-blind]
                       [--require-marks / --no-require-marks] [--field-strength T] [--jobs J]
                       [--fsaverage DIR] [--video / --no-video] [--port N] [--no-browser]
                       [--no-serve] [--serve-after N] [--relocate] [--force]
                       [--raw DIR] [--epi / --no-epi] [--epi-runs GLOB]...
    # inventory + functional inventory + metrics + scoring + render + serve, in one go;
    # the EPI packs are made behind the app (docs/epi_interface.md, Step 2 as built)
surfcensus serve    --run-dir DIR [--host H] [--port N] [--mode local|team] [--rater NAME] [--no-browser]
surfcensus finalize --run-dir DIR
surfcensus calibration freeze --run-dir DIR [--subject UID]... [--version V]
surfcensus export --run-dir DIR [--out DIR] [--format csv|parquet] [--release]
surfcensus render-subject SUBJECT_DIR --out DIR [--fsaverage DIR | --no-template]
    # one subject, no run directory; the CLI form of sc.render.three_plane
surfcensus epi-session PATH --subject LABEL --out DIR [--session LABEL]... [--raw DIR]
                       [--epi-runs GLOB]... [--serve] [--port N] [--no-browser] [--force]
    # the EPI movies of one subject, no run directory (docs/epi_interface.md, Step 1 as built)
```

`review` defaults the run directory to `PATH/../surfcensus-run` when a run is there already or that place can be written to, else `./surfcensus-run`; for one subject directory as `PATH`, to `surfcensus-run` beside the subjects directory, since a run may not lie inside it; that is the directory that lists the subject, also when the subject directory is a link, so the run never lands beside what the link leads to. Subjects that lie below the `derivatives/` folder of a BIDS dataset: the run goes beside the dataset, `DATASET/../surfcensus-run`, never into it, whether `PATH` is the dataset or a folder inside it, because a dataset (which may be a DataLad dataset) gets no files unless a run directory in it is asked for with `--run-dir`. The dataset is the outermost directory with a `dataset_description.json` that has the subjects below its `derivatives/`, or `PATH` itself when the subjects were found below its `derivatives/`. A run of the same folder that an earlier version made inside the dataset, beside the folder that was named, is still found and used. The same holds when the place beside `PATH` lies inside a DataLad dataset (a directory with `.git` and `.datalad`; `sources.local.datalad_dataset_holding`), such as a clone of an fMRIPrep output that holds no `derivatives/` folder: the run goes beside the outermost such dataset, which would otherwise get untracked files. A git-annex repository without `.datalad` is not told apart. An fMRIPrep output that holds its subjects as `sourcedata/freesurfer/` and has a `dataset_description.json` (a BIDS-derivative dataset, as fMRIPrep writes it and OpenNeuroDerivatives publishes it, also when it was downloaded without DataLad) counts as such a dataset when the place beside `PATH` lies inside it: `review ds/sourcedata/freesurfer` puts the run beside `ds`. A plain git repository (no `.datalad`) is not told apart, so a run of a folder inside one lands in its work tree. When the place beside the dataset cannot be written to, the run goes to `./surfcensus-run`; when the current directory lies inside the dataset, `review` stops with one line that asks for `--run-dir`. A run of the same root comes first, wherever of these places it is: beside the dataset, where an earlier version put it inside the dataset, or in `<name>-surfcensus-run` beside the dataset (`<name>` the dataset, or the folder named when there is none), which is where a second dataset of one folder is told to put its run. Several datasets of one parent folder share the default `surfcensus-run` (open question O23): when it holds the run of another root and no run of this one is found, `review` stops before it opens a run, with exit code 2 and one line, `other root: <dir> holds the run of <root>, ...`, that says what to give. When that root still holds its subjects in another directory it is another dataset, and the line names `--run-dir <parent>/<name>-surfcensus-run`; when it holds the same subjects named another way it names that root as the path to go on with that run, and `--relocate` only within one pipeline; when it holds no subjects now it names `--relocate`. It orders the rating queue by composite score descending unless `--blind` is given, renders in that order, and starts the app on `127.0.0.1` with a single local rater as soon as the first subject of the queue is rendered; the others are rendered while the app runs, and the app says so when a rater is faster than the renderer (decision D33). `--serve-after N` waits for N renders, and `--serve-after 0` for all of them.

`review`, `serve`, `finalize`, `calibration freeze` and `export` are built (milestone 4), for local runs:

- `review` does the steps of the phase commands in one process and says on stderr, one block per step, which run directory it uses and whether it is new, how many subjects it found, where under the path (`Found 48 subjects in derivatives/fmriprep/sourcedata/freesurfer (fMRIPrep layout, FreeSurfer 7.3.2).`; left out when they lie in the path itself), in which layout and FreeSurfer version and which cannot be rendered for lack of which file, what the subjects were scored against, how many are rendered, and the address of the app with the time from the start of the command. Every present subject is selected (reason `all`, probability 1, the seed of the queue); a subject that lacks a file a render needs is in the selection, gets its `missing surface` render row and an unratable final label, and is never offered. `serve` and `finalize` on a local run that the phase commands made, which has no selection rows, treat every present subject as selected in the same way.
- The first `--serve-after` subjects of the queue (1 unless given) are rendered before the app starts, one in the process itself. Then a second `Run` in a thread renders the rest in queue order, a batch of `--jobs` subjects at a time (default: the processors, at most 8), so that each is offered soon after it is made; the app answers `rendering` when nothing is ready. A failure of the run or the machine ends that renderer with one line, and the app goes on with what is rendered.
- Ctrl-C or SIGTERM stops the server, then the renderer, which finishes the batch it is on and records it; the rest stays pending for the next call. The command then prints how many subjects the rater has rated and how many are left, and the export command, and ends with exit code 0. From the first SIGTERM on, the process ignores SIGTERM (`SIG_IGN`), so a burst of them and every later one change nothing, while the command stops and after its closing line up to the end of the process; from the moment the app and the renderer have stopped, just before the closing line, it ignores SIGINT as well (`serve_review(stopped=...)`, `serve(stopped=...)`). Ignoring is what protects the end of the process: CPython takes away every handler written in Python while the interpreter finalizes, and leaves an ignored signal ignored. A second Ctrl-C while the app stops ends the command with 130 and without the closing line. The handler takes no lock: it sets a flag (`stopping.StopFlag`) that the wait polls, because a second signal that arrives while a handler holds the lock of a `threading.Event` waits for it for good. `serve` behaves the same. The functions `review.review` and `app.server.serve` handle SIGTERM with such a flag for the length of the call and put back the handler they found when they return; `review.serve_review` and `serve(stop=...)` take a stop request (a `StopFlag`, or a `threading.Event`) from a caller that handles the signal itself. While `review` still prepares (inventory, metrics, the first renders), SIGTERM ends it as Ctrl-C does, by `KeyboardInterrupt`, which the batches are written for (running work is waited for, leases are released, the batch it interrupts is not recorded), with the line `terminated; what was recorded stays, and the next call goes on` and exit code 143; from that SIGTERM on both signals are ignored, so nothing interrupts the unwinding a second time. When Ctrl-C began the end, a SIGTERM raises nothing and the code is 130; a second Ctrl-C there is the user's way out as it always was. Worker processes take SIGTERM with its default, also when they are started after the parent began to ignore it (`batches._start_worker`).
- Options that are stored (`--subjects`, `--field-strength`, `--blind`, `--require-marks`, `--video`) hold until they are given again; the defaults (not blind, marks not required, no video) apply only when the run is created. `--rater` is not stored. `--blind` cannot change once the run has a submission, unless `--force`; the change is then written into the events of the app. One subject directory as `PATH` names its own subject set: on a run of the same subjects directory it replaces the stored one.
- The same command again does the same steps with everything unchanged skipped: on two real subjects it is back at the app in 0.8 s (`rater_app.md`, what was measured). `--relocate` is for data that moved, as for `inventory`, and is refused while the stored root still holds its subjects in another subjects directory.
- `--no-serve` renders every subject and stops; it needs no package of the app. `review` ends with exit code 1 only when no subject of the run can be rendered and no functional run is packed or can be packed, with 2 for what stops a phase command, and with 0 otherwise, also when some subjects failed: those are named and are unratable.
- `review` is for a local run with one source. A census run is refused; `serve` is the command for it (team mode, milestone 5).
- Functional data (step 2 of the EPI interface, `epi_interface.md`, "Step 2 as built"). When the folder also holds an fMRIPrep derivative with functional runs and their raw BIDS dataset (found by rule from the surface source, or named with `--raw DIR`), `review` lists them after the surface inventory (phase 2f: listings, sidecars, reports, transforms, masks and headers, never a volume) and prints one block: where, how many runs in how many sessions of how many participants, how many can be packed and why the others cannot, the flagged runs and outlier anchors as statements about the raw data, and the fetch lines of functional files without content. The thread behind the app renders the remaining surfaces first and then packs the functional sessions into `epi/`, a batch of `--jobs` sessions at a time; the app offers the EPI index at `/epi` (address printed as `EPI movies:`), unblinded like `/review`. When no subject can be rendered and a run can be packed, the app opens at the first packed session and `--serve-after` counts sessions. The closing line says how many packs are made. `--raw`, `--no-epi` (and `--epi`) and `--epi-runs` are stored in the section `epi` of `config.yaml`, which is written only once it differs from the defaults, so a folder without functional data, or a call without these options, writes `config.yaml` as before. `review` ends with 1 only when nothing can be rendered and nothing packed.
- `serve` starts the app on a run that is rendered and renders nothing; `finalize`, `calibration freeze` and `export` are described in `rater_app.md` and `phases.md`. An installation that lacks a package of the app (a stripped or broken one: the packages are dependencies) gets one line that names the package and `pip install surfcensus`, with exit code 2, before any work is done.

Census (and power users), one subcommand per phase:

```
surfcensus key init   [--path FILE]          # create the dataset key (census only)
surfcensus discover   --run-dir DIR [--source openneuro-derivatives|openneuro-embedded]
surfcensus inventory  --run-dir DIR [--source PATH] [--subjects GLOB]... [--field-strength T]
                      [--dataset ID]... [--force] [--relocate [PSEUDONYM]]
                      [--raw DIR] [--epi / --no-epi] [--epi-runs GLOB]...
surfcensus pack       --run-dir DIR [--jobs J] [--epi-runs GLOB]... [--force] [--retry-failed]
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

`inventory`, `render` and `status` (milestone 2) and `metrics` (milestone 3) are built, for local sources, and `pack` (step 2 of the EPI interface), for local runs:

- `inventory` runs the functional inventory (phase 2f) after the surface one when no `--dataset` is given, prints the functional block of `review` and a `func_inventory:` line, and ends with 1 when either has a failed unit. `pack` packs the pending functional sessions into `epi/`, with the exit codes and the signal handling of `render`; with nothing pending it builds no source, so it works with the data not mounted; a census run is refused (`epi_interface.md`, "Step 2 as built").

- `inventory --source PATH` creates a local run in `DIR` when it holds none, and says which layout it found and where the subjects lie under the path (`D-3f9a1c07: FreeSurfer subjects found in derivatives/fmriprep/sourcedata/freesurfer (fMRIPrep layout, fMRIPrep 21.0.2)`; without `--source`, under the stored root; a census run, and a run with several sources, name layout and pipeline as the catalog has them), how many subjects are complete and have Tier A and Tier B files, their FreeSurfer versions, and which subjects lack what. On a run that exists, `--source` only says which source is meant: it must be the stored root, or come with `--relocate` when the data moved. Another root without `--relocate` is refused (`other root`) with what to give: `--relocate` when the stored root holds no subjects now or holds the same subjects in the same pipeline under another name, the stored root as the path when it holds the same subjects in another layout, and a run directory of its own when it holds subjects of another directory, which is another dataset. `--relocate` is refused while the stored root still holds subjects in another subjects directory than the new path (`invalid: --relocate: the data of this run are still where they were ...`): it would bind the run, its renders and ratings to other data under the same identifiers. A copy that the run should no longer read is moved or renamed first. `--relocate` replaces the stored root and, when the data now lies in another layout of the same pipeline (an fMRIPrep output moved from `sourcedata/freesurfer/` to the older `freesurfer/` beside `fmriprep/`), where the subjects are under it. Nothing else changes, so every `subject_uid`, render and rating stays; with several local sources it takes the pseudonym of the dataset that moved. Data that an inventory finds in the layout of another pipeline (a subjects directory that became an fMRIPrep output) is a new set of subjects with new identifiers: the earlier ones are marked absent, with a note, and keep their renders and ratings (decision D21).
- `render` renders what is pending and prints counts and one line per failure reason. When nothing is pending it reads no source, so it also works while the data is not mounted. A run with a pinned template still reads that template, from `templates/` in the run directory; when that directory was deleted and no fsaverage directory is found to build it from again, `render` stops with `changed: the template this run was scored with is no longer available`, also when nothing is pending (decision D22; `status` works). `--all` renders every complete subject, `--selected` only the selection (which is every complete subject while nothing is selected). `--check-sample` renders nothing: it draws a seeded sample of the done renders for the by-eye check of phase 6 and writes `logs/render-check-<seed>.txt` and `.html`. A `vertex_map.npz` that is there and cannot be read (a truncated copy, a changed byte) or does not hold the arrays of a vertex map is not noticed by the skip test, which looks at names; the `native` stage at the end of `render` notices it when it next reads the vertex maps, leaves that subject's centroids without a native position and says so in a note with the `subject_uid`, at every call until the file is removed and the render made again.
- `metrics` computes what is pending of phase 4 and prints the counts, one line per failure reason, the notes of the report and one line that says what the subjects were scored against. Without `--stage` it runs the stages `regional`, `fsaverage`, `reference` and `score`; with `--dataset` and no `--stage`, only the two that work per subject, since a stage on the whole run is not restricted by a dataset. `native` (cluster centroids on the native surface, from the vertex maps of the renders) also runs at the end of every `render`, and not at the end of `metrics`. So when `metrics` finds clusters on a run that is already rendered, for the first time or because the reference was built again, those clusters have no native position until the next `render` (one with nothing to render will do) or `metrics --stage native`. `Run.clusters()` joins by reference, so a position of an earlier reference is never attached to a cluster of the present one; until then `clusters_native.parquet` still holds the rows of the earlier reference. `merge` joins the per-dataset parts of each table. A present subject that no longer has any stats file, or any thickness file, is no unit of `regional`, or of `fsaverage`, any more: the stage removes its rows, its map and its ledger row, so the scores do not keep numbers from files that are gone, and the subject is scored like one that never had them (`no_stats`, `no_thickness`). An fsaverage map of the run that cannot be read when the `reference` stage needs it, or that holds maps of another vertex count than the template's mesh, stops the command with exit code 2; the damaged maps are removed, and the next `metrics` makes them again from the sources. A reference built in the run is read back before the `reference` stage is skipped: one that does not read (a truncated `stratum_<nnn>.npz`, a damaged `index.json`, no `members.parquet`) is built again by `metrics`, with the bytes it had. `metrics --stage score` alone on such a run stops with exit code 2 and says to run `surfcensus metrics`; `Run.zmap()` raises the same `ConfigError`. `--fsaverage DIR` gives the template for this call; the directory is not written to `config.yaml`, because the run keeps the template itself, by content, from its first use. A directory that builds another template than the one the run is pinned to stops the command (`changed: <old> -> <new>; pass --retemplate`); `--retemplate` takes the new one, and every map is made again. `render` takes `--fsaverage` as well.
- `status` prints one line per phase from the catalog: subjects inventoried and unreadable (a subject with files without content is inventoried); for each metric stage that has run, its units done and failed (the rows of the ledger: a `native` line says that the stage ran, not that its table follows the present clusters); renders done, failed and pending, where pending is what `render` would do now. `--failures` lists every failed unit with its reason and detail; for the metrics these are the failed rows of the ledger; for the render these are the subjects counted as failed on the line above, so an incomplete subject is listed before `render` has recorded it, and a recorded failure of a subject that left the source or is pending again is not. It reads the private catalog, so its output is for the operator's terminal, not for a log.
- `--dataset` takes a pseudonym or an accession. An accession given on the command line is resolved in the catalog and stored nowhere.
- Options that are stored in `config.yaml` (`--subjects`, `--field-strength`, `--video`) hold until they are given again: a command run a second time without them changes nothing. A subject glob that matches nothing is refused before it is stored.
- Exit codes: 0 when no unit is in the failed state when the command ends; 1 when at least one is, also when it failed in an earlier call and was not tried again; 2 for a usage or configuration error, including a source that cannot be found, printed as one line `reason: detail`. What lies in the machine and not in a subject is such an error as well and is recorded against no subject: a run directory that cannot be made or written, a catalog that another process keeps locked, media on a full or read-only disk, a video encoder that is absent or cannot encode, a template directory that was given and cannot be used, a template that differs from the one the run is pinned to, an fsaverage map in `tables/fsavg/` that cannot be read or is on another mesh, a reference in `tables/reference/` that cannot be read when the scores are computed from it, a `tables/clusters.parquet` that is no parquet file or a table of another schema (the line names `surfcensus metrics --stage reference --force`, which writes it again). Once the cause is gone, the same command does the work (decision D29). 130 when Ctrl-C ended the command and 143 when a SIGTERM did, each after one line (`interrupted; ...`, `terminated; ...`): SIGTERM ends a phase command as Ctrl-C does, through `KeyboardInterrupt`, and is ignored from then on, as SIGINT is (`review` above). `render-subject` takes both the same way. `status` reports and always ends with 0.
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
scores = run.scores()  # pandas DataFrame, one row per subject, by primary_rank
clusters = run.clusters()  # outlier clusters, with native positions where a render gives them
z = run.zmap(scores["subject_uid"][0])  # {"lh": ..., "rh": ...} on the template mesh
sc.render.three_plane(subject_dir, out="sub-01.mp4")  # single subject, no run dir

from surfcensus.review import review
from surfcensus.export.tables import export_run

run = review("/data/SUBJECTS_DIR", serve=False)  # every step, every render, no app
export_run(run.run_dir)  # the tables of phase 8
```

`review(path, ...)` is the command as a function and returns the run; with the app it returns after Ctrl-C. `review.open_review(...)` returns the run and the running app with the renderer behind it (`ReviewApp`: `url`, `rendering`, `wait_rendered()`, `stop()`), for a caller that wants to do something while it runs, and `review.serve_review(run, app, stop=...)` is the rest of `review`: it names the address, waits for Ctrl-C or the `stop` event, stops the app and prints the closing line. `Run.render(stop=...)` takes a function that is asked before each batch and ends the call between two batches; `native=False` leaves the closing `native` stage to a later call.

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
    scores.parquet       # one row per subject and reference; one of them primary per subject;
                         # rank within the reference, primary_rank over the primary rows
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
  epi/<participant_uid>/      # the EPI display cache of a participant (step 2): only .json and .bin files
    stream.json          # the participant's sessions in stream order, packed or not yet
    <session_uid>/       # session.json, contours/<grid_key>/<plane>.bin, <run_uid>/pack.json raw/ pre/ traces.json
  media/<subject_uid>/
    anim.mp4
    tiles/               # per-panel WebP slice tiles in two layers, see rendering.md
    poster.webp
    render.json          # geometry, per-panel pixel-to-voxel affines, frame tables, config hash
    vertex_map.npz       # native vertices the tiles show, plus a margin, and their fsaverage6 counterparts
  app.sqlite             # rater app state, separate from the catalog
  protocol.yaml          # the rating protocol of the run; the shipped one is installed when the app first starts
  calibration.yaml       # the calibration set, once `surfcensus calibration freeze` has written it
  export/                # surfcensus export: ten tables as CSV or parquet and manifest.json (analysis profile)
  release/               # surfcensus export --release, a census run only
```

Datasets appear everywhere outside `catalog.sqlite` by keyed pseudonym (`D-3f9a1c07`), never by accession; see decisions D10 and D20 in `decisions.md`. `privacy.py` owns every keyed identifier (the pseudonym, `subject_uid`, `render_key`, the names of cached files), the scrubbing of text that comes from exceptions and, from milestone 5, key loading and the reveal path. Nothing else in the package computes one of these or prints an accession for display. `config.yaml` records the key fingerprint.

`subject_uid` is 16 hex characters of a keyed hash (same key) of `(source, identity, pipeline, subject_label)`, so media paths do not reveal the dataset to raters. `identity` is the accession of a remote dataset. For a local one it is the resolved root path when the run was created; it is stored and never recomputed, so data that moved (`--relocate`) keeps every `subject_uid` and with it every render and rating. Local runs do not need the dataset key: a new local run draws a random per-run salt, stored in `config.yaml`, since nothing about local identities needs protecting beyond keeping raters blind.

Because subject and dataset identifiers depend on the key, the key is part of a census run's identity. Losing it means the run cannot be extended consistently, so `surfcensus key init` prints a reminder to back it up.

A second secret, the run secret, is 32 random bytes drawn when the catalog is created and kept in it (`meta.run_secret`), in both modes. `render_key` and the names of cached files are keyed with it. Both are derived from files whose sizes and ETags are public, so an unkeyed hash of them could be recomputed from the bucket listing by anyone who holds an export or a media URL. `render_key` says which tiles a rating was made on: it is built from `RenderConfig.tile_hash()` and the SHA-256 digests of the files the render read, so a file that was only touched or moved keeps its key.

`catalog.sqlite` is SQLite through the standard library, with `PRAGMA user_version` migrations. A migration is one transaction, and each statement of a migration script ends at the end of a line. A local run uses the default rollback journal, because run directories on network file systems are common and SQLite's write-ahead log does not work there, with a busy timeout of 30 s on every connection. A census run uses the write-ahead log, with 60 s for writers and 5 s for read-only connections, and warns when its directory is on a network file system (detected on Linux only, from `/proc/mounts`). Every write is one short `BEGIN IMMEDIATE` transaction, so several processes can write to one catalog.

A new catalog is built under a temporary name in the run directory and gets the name `catalog.sqlite` through a hard link when it holds the schema, the identity of the run and the run secret. So `catalog.sqlite` holds a run or does not exist, whenever the process ends, and no other process opens one that is half made. On a file system without hard links the name is taken with an empty file and the run is made in that file in one transaction. A catalog in which no run was made yet (an empty file, or the tables without a row) is never migrated or removed: opening the run makes the run in it, under SQLite's write lock, with the salt of `config.yaml` or, for a census run, the key. A database that holds anything else is refused as not made by surfcensus and left as it is.

Tables in `catalog.sqlite` (the DDL is migration 1 in `store.py`; migration 2 adds the tables of the EPI interface, below):

- `meta(key, value)`: run id, mode, key kind and fingerprint, creation time, package version, the id of the template the run is pinned to (`template_id`, from its first use); and the run secret.
- `datasets(dataset_id, pseudonym, source, identity, accession, root, fs_root, template_location, layout, pipeline, pipeline_version, derivative_ref, snapshot_tag, n_subjects_listed, ..., discovered_at)`, unique on `(source, identity, pipeline)`; for a local source, one row with `root` set to the path.
- `subjects(subject_uid, dataset_id, subject_label, participant, fs_dir, present, complete, tier_a, tier_b, missing_roles, problem, longitudinal, fs_build_stamp, fs_version, fs_major, recon_flags, n_invocations, recon_finished, t2pial, flairpial, hires, expert, xopts, edited, pial_variant, holes_lh, holes_rh, trunc_lh_1, trunc_lh_2, trunc_rh_1, trunc_rh_2, age, sex, field_strength, manufacturer, inventory_hash, inventoried_at)`. `complete` means every file a render needs is present; `tier_a` and `tier_b` that the stats files, and the thickness and `sphere.reg` files, are. A file without content (git-annex, see Sources) counts as absent for these and for `missing_roles`. `problem` is `unreadable: <reason>` for a directory that could not be listed (a failed unit of the inventory), or `not fetched: ...` for a subject with files without content, which names them and is no failure; no column was added for it. `present = 0` marks a subject that the source no longer lists; its row, render and ratings stay.
- `files(subject_uid, role, location, variant, size, etag_or_mtime, last_modified, version_id, sha256)` with roles `t1`, `brainmask`, `lh_white`, `rh_white`, `lh_pial`, `rh_pial`, `lh_thickness`, `rh_thickness`, `lh_sphere_reg`, `rh_sphere_reg`, `lh_annot`, `rh_annot`, `lh_stats`, `rh_stats`, `aseg_stats`, `talairach`, `recon_log`, `build_stamp`, `recon_log_lh`, `recon_log_rh`. `location` is relative to the subject directory, so the table holds no accession. `sha256` is set when a file was read in full: by the inventory for the build stamp and the recon-all logs (not for a log of which only the head was read), by the metrics for the `aparc.stats`, thickness, annotation and `sphere.reg` files that a stage read (the `sphere.reg` only in a run with a template; neither annotation nor `sphere.reg` of a hemisphere without a thickness file; nothing from a stage that failed for the subject), by a render for the T1, the surfaces, the brainmask and, with a template, the `sphere.reg` files. `aseg.stats` and `talairach.xfm` are read by nothing yet and have none.
- `dataset_files(...)`: dataset-level files that a stored value was read from (`dataset_description.json`, `participants.tsv`, a T1w sidecar). Empty until milestone 5.
- `templates(template_id, mesh, origin, dataset_id, build_stamp, files_json, created_at)`: every template the run has used, with the size and SHA-256 of the fsaverage files it was built from.
- `work(phase, unit, config_hash, input_signature, status, reason, detail, output, surfcensus_version, finished_at)`: the ledger of every phase but inventory and render; a unit is a `subject_uid`, `dataset:<pseudonym>` or `run`. The metric stages are the phases `regional` and `fsaverage` (one unit per subject) and `reference`, `score` and `native` (the unit `run`).
- `renders(subject_uid, config_hash, input_signature, tile_hash, render_key, video, status, reason, detail, media_path, n_frames, template_id, rendered_at)`
- `selection(subject_uid, reason, stratum, inclusion_prob, seed)`. `PublicCatalog.selected_uids()` gives its subjects, or None while no selection is made: the table is empty, or it is that of a local run and holds only the rows of a frozen calibration set (reason `calibration`). `render` then takes every complete present subject, and the rater app and `finalize` of a local run every present one.

Migration 2 (`store._MIGRATION_2`, `SCHEMA_VERSION` 2; step 2 of the EPI interface, decision D40) is additive: `func_sources` (the derivative and the raw dataset of a dataset, with their roots and the derivative's identity, private), `func_participants`, `func_sessions` (the anchor, the frame check, the inventory hash), `func_runs` (one row per run with its reason, grid, corrections and frame numbers), `func_files` (keyed by owner kind and owner) and `packs` (the counterpart of `renders`), with five public views that leave out the roots, identities, file facts, input signatures and content digests; the session unit of the pack phase is a row of `work` (phase `epi_session`). A writer migrates every run it opens; a read-only opener (`serve`, `export`, `finalize`) and the two commands that open the catalog for writing without migrating it (`status`, which reads, and `calibration freeze`, which writes selection rows; `migrate=False`, review of step 2) read a catalog of schema 1 as a run without functional data. A version of the package from before step 2 refuses a migrated catalog (`newer schema`).

What is private stays in the catalog and is reached only through `store.Catalog`, which `run.py`, `cli.py` and the inventory use: `dataset_id` (assigned in the order of the bucket listing), identity, accession, root, `fs_root`, template location, derivative ref and snapshot tag; `fs_dir`; file locations, sizes, ETags and digests; every input signature; the run secret; and in a census run the subject labels and every failure detail (a local run shows the details of its own renders). Three views (`public_datasets`, `public_subjects`, `public_files`) leave these out, and no frame that `store.PublicCatalog` returns holds any of them. It hands out an input signature only as a value to pass back for the skip test. Everything `PublicCatalog` returns is ordered by pseudonym or `subject_uid`, never by `dataset_id`.

An input signature is 16 hex characters of the SHA-256 over role, location, size and ETag or modification time of the files a stage reads, and for a role with several candidate names the name that was read (`variant`), which is what changes when a `?h.pial` link is pointed at another variant. It comes from the listing, so deciding to skip a unit opens no source file. The skip rule is the same in every phase: a unit is skipped when its row is `done`, its config hash and input signature are those of the present configuration and listing, and its output exists. For a render the output is the whole media directory: `render.json` and a file under every name it lists, so a directory that lost files (a partial copy, a restore) is rendered again. A `failed` row that matches is not tried again unless `--retry-failed`; a changed configuration or input tries it again; `--force` ignores the row. Outputs are written first and the row last, so a crash leaves an output without a row, which the next run redoes, never a row without its output. A subject that is no longer a unit of a metric stage, because the files that made it one are gone, loses its outputs first and its row last, in the same order. The inventory has no ledger rows; its skip test is `subjects.inventory_hash`, the signature over all listed files of the subject. The stages on the whole run have signatures of their own: that of `reference` covers the subjects that have a map, with the signatures of their files, their dataset, field strength and FreeSurfer version; that of `score` the content of the three tables it reads, what it takes from the catalog and who is scored against which reference; that of `native` the content of `clusters.parquet` and the render keys. So one more subject redoes the stages on the run and nothing per subject.

## Testing

- Unit tests on synthetic meshes (ellipsoids, and a folded closed surface that a slice crosses in several loops) for slicing, contour drawing, normals, pole finding and ROI construction.
- Layout detection tests for the local source on fake directory trees (SUBJECTS_DIR, fMRIPrep old and new layouts, BIDS derivatives, and BIDS dataset roots with each of these below `derivatives/`: the order of the places, what is not reached, the messages when nothing is found, and a pin of the `subject_uid`s of the earlier layouts on a fixed salt).
- Golden-image tests for frame composition with a tolerance, using a tiny synthetic volume.
- Network fixtures, marked `network`: `ds000102` sub-01 (FreeSurfer 6.0.1; Tier A, B and C, 43 MB) and `ds003020` sub-UTS01 (FreeSurfer 7.3.2; Tier A, B and C, 37 MB) and the template files of the `fsaverage` next to the subjects of `ds000102` (40 MB), each fetched once into the pytest cache. Linked into an fMRIPrep layout, they are the end-to-end test of inventory, metrics and render through `Run`, and later of `surfcensus review --no-browser`.
- Property tests (hypothesis) for mesh-plane intersection: every segment endpoint lies on the plane and on a mesh edge.
- Resampling tests: resampling fsaverage onto itself is the identity; a smooth field on a synthetic sphere survives resampling within tolerance; agreement with `mri_surf2surf` on the network fixture within a stated tolerance (run once by a developer with FreeSurfer, stored as a reference array; not made yet, and the test is skipped until it is).
- Metric fixtures: a synthetic fsaverage directory of the real size (163,842 vertices, coarsened to fsaverage6 by the code under test) with Desikan parcels laid out by rule, and synthetic subjects whose `sphere.reg` is a known warp and whose thickness is a known function of the template position, with a 5 mm patch planted in some. A cohort of 32 crosses the 30-subject rule; one of 5 does not. `small_run` is a run of three such subjects that went through inventory, metrics and render. `planted_run` is a run of three with values that the cohorts lack planted in single hemispheres (vertices at the ceiling in the control region, near the ceiling, at 0 and above the maximum, a sphere with folded faces and one wound inward), for the tests of what the stages copy into their tables.
- Click round-trip tests: random voxels to panel pixels and back for every panel and slice in `render.json`; marks resolve to the expected vertex on a synthetic mesh.
- Rater app: the queue, the database, final labels, the export and the server each have tests on copies of `small_run` (three synthetic subjects, rendered); the server is tested through Starlette's test client and over a socket. `tests/test_review.py` and `tests/test_cli_review.py` run `review` on synthetic directories, with the app in a thread and, for Ctrl-C and SIGTERM, as a process.
- Frontend: `tests/test_frontend.py` reads the JavaScript and the templates (every route and field against `app/schemas.py`, no other host) and runs the modules without a page under Node, where there is one. `tests/test_frontend_browser.py` (marked `browser`) drives the pages in a browser against the mock server of `tests/frontend_mock.py`; `SURFCENSUS_FRONTEND_MEDIA` points it at other renders. `tests/test_browser.py` (marked `browser`) drives the real pages on the real server, on a run that `review` made of two synthetic 1 mm subjects, at two window widths: what a rater does with keys and mouse, the stored mark against the surface file, and a first-time rater who learns from the screen alone how to get to the next subject. Browser tests need a browser that is installed (`SURFCENSUS_BROWSER`, else Chrome) and are not part of the default run or of CI.
- Install: `tests/test_install.py` (marked `install`, `just test-install`) is the quick start as a user gets it. It builds the wheel with `uv build`, makes an environment outside the repository on the Python of the test run, installs the wheel there with whatever the package index gives for its dependencies (no lock file, no extra), and runs the installed command from another directory with `PYTHONPATH` unset: `surfcensus --help`, then `surfcensus review` on two synthetic 1 mm subjects up to the line with the address, the rating page with its scripts, `/api/session`, `/api/next`, one tile, a posted rating, SIGTERM with exit code 0 and the closing line, and `surfcensus export`. It asserts that the package and the served files are those of the environment and not of the source tree, that the subjects directory is unchanged, and what the wheel holds (Packaging). It needs `uv` and the package index, so it is not part of the default run; CI runs it in a job of its own, on Linux and the oldest supported Python. Measured 2026-11-14 on one 4-core Linux machine: 17 s with the wheels in uv's cache and 20 s with an empty cache on a fast connection, of which the environment takes 1 to 3 s and `review` 6 s to the address.
