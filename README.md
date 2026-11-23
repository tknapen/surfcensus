# surfcensus

Look at your FreeSurfer surfaces where they most often go wrong.

Near the occipital pole, FreeSurfer's pial and white surfaces often absorb the superior sagittal sinus, the transverse sinus or the cerebellar tentorium, and these errors are rarely edited. `surfcensus` scores reconstructions with per-vertex cortical-thickness outlier metrics in fsaverage space, renders sagittal, coronal and axial views that sweep through a box around the occipital pole with white and pial contours, and serves those views in a keyboard-driven rating app in which raters click on errors to mark where they are.

## Status

Milestones 1 to 4 of 6 are built: rendering one subject; inventorying, scoring and rendering a directory of subjects into a run directory; and the rating app for one researcher's own data, with `surfcensus review` and `surfcensus export`, which since the quick-start follow-up installs in one line and finds the subjects of a downloaded or DataLad-cloned dataset by itself. `surfcensus review openneuro:dsXXXXXX` does the same for a dataset of the public OpenNeuroDerivatives bucket that is not on disk. The census over OpenNeuro, several raters at once and the normative reference are planned and do not exist yet. Nobody has rated a brain with the app so far: it was tested by scripts that drive a browser, on synthetic subjects and on two real ones. The scores have been run on real subjects but not yet compared with anyone's judgment of the surfaces, so they say which subjects stand out on the numbers, not which have an error. The package is not on PyPI. The specification is in `docs/`; the build order and what has been checked are in `CLAUDE.md`.

## Getting started

The app has two interfaces. The surface interface shows FreeSurfer's white and pial surfaces over the T1 around the occipital pole, and you rate them. The EPI movies show the raw and the preprocessed EPI series of every functional run, one stream over all runs and sessions of a subject, with the surfaces as contours, for quality control of the raw data and of the preprocessing. `surfcensus review` prepares both and opens them in your browser.

### 1. Get the code

You need Python 3.11 or later, `git` and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` installs it). No FreeSurfer installation or licence is needed.

```bash
git clone https://github.com/tknapen/surfcensus.git
```

```bash
cd surfcensus
```

```bash
uv sync
```

`uv run surfcensus --help` lists the commands. Every command below is run from the clone with `uv run` in front.

Without a clone, `uvx --from git+https://github.com/tknapen/surfcensus surfcensus ...` runs the same commands in an environment that uv makes for the purpose. `pip install git+https://github.com/tknapen/surfcensus` installs the package into an environment of yours; leave `uv run` off then. The package is not on PyPI yet.

### 2. An OpenNeuro dataset, without downloading it

```bash
uv run surfcensus review openneuro:ds000102 --subjects sub-01 --epi
```

This reviews `sub-01` of `ds000102` straight from the public buckets: the FreeSurfer subject and fMRIPrep files from the OpenNeuroDerivatives bucket, the raw series from `s3://openneuro.org`. Access is anonymous and read-only (GET requests only, at most 8 at a time, no credential). Before fetching anything, the command says how much it will fetch, estimated from the listed sizes. Then it fetches, scores and renders, and opens the rating screen at `http://127.0.0.1:8765/`. The EPI movies are at `http://127.0.0.1:8765/epi` (printed as `EPI movies:`) and are packed behind the rating screen.

- Measured on 2026-11-15 for this command on one 4-core Linux machine: the rating screen 20 to 25 s after the command, after 210 MB of downloads, and both EPI movies ready after 52 to 62 s.
- Leave out `--epi` for the surfaces alone: nothing is then asked of the raw bucket, and the download is much smaller (about 110 MB before the rating screen for two subjects).
- `--subjects` takes shell patterns (`'sub-0[1-5]'`; `'*'` for all) and is kept for later calls. A new review that would fetch more than 2 GB before the rating screen stops and asks for `--subjects`.
- The EPI movies need an fMRIPrep derivative with T1w-space series. In OpenNeuroDerivatives those are the derivatives made with fMRIPrep below 23.2; for other datasets the surfaces are reviewed and the functional runs are listed with the reason they cannot be shown.
- The work goes into `./ds000102-surfcensus-run`, or `--run-dir DIR`. The files of each batch of subjects are removed once they are scored and rendered. Kept: the renders, the scores, the small log and stats files, and with `--epi` the functional files and the surfaces of the contours.
- Ctrl-C stops it. The same command again goes on and fetches nothing twice; it needs the network to list the dataset. Without a network, `uv run surfcensus serve --run-dir ./ds000102-surfcensus-run` opens the app on what is there.

### 3. An OpenNeuro subject, downloaded

```bash
uv run python scripts/fetch_example.py data
```

This fetches `ds000102` sub-01 (about 160 MB, the same public buckets, anonymous) into `data/ds000102`, laid out as a BIDS dataset: the raw functional series and their sidecars, and below `derivatives/fmriprep/` what surfcensus reads of the fMRIPrep derivative, the FreeSurfer subject in `sourcedata/freesurfer/` included. `--dataset dsNNNNNN --subject LABEL` fetches one subject of another dataset that OpenNeuroDerivatives has an fMRIPrep derivative of. A file that is already there is not fetched again. Then:

```bash
uv run surfcensus review data/ds000102
```

This is the same review as in step 2, read from disk, with the run directory at `data/surfcensus-run`. The `fsaverage` template is not part of the download, so the subject is ranked without per-location scores (see `metrics` below); `--fsaverage DIR` names one.

### 4. A BIDS dataset with fMRIPrep derivatives of your own

```bash
uv run surfcensus review /path/to/mystudy
```

`review` finds everything below the dataset root by itself. For both interfaces the dataset looks like this:

```
mystudy/
  dataset_description.json
  sub-01/ses-1/func/sub-01_ses-1_task-rest_run-1_bold.nii.gz   (and its .json)
  derivatives/fmriprep/
    dataset_description.json
    sub-01/anat/sub-01_from-T1w_to-fsnative_mode-image_xfm.txt
    sub-01/ses-1/func/sub-01_ses-1_task-rest_run-1_space-T1w_desc-preproc_bold.nii.gz
                      ..._space-T1w_desc-brain_mask.nii.gz
                      ..._desc-confounds_timeseries.tsv
                      ..._from-scanner_to-T1w_mode-image_xfm.txt
    sub-01/figures/..._desc-summary_bold.html
    sourcedata/freesurfer/sub-01/   (mri/, surf/, label/, stats/, scripts/)
    sourcedata/freesurfer/fsaverage/
```

That is the output of fMRIPrep run with FreeSurfer and with `T1w` among its `--output-spaces`. It has been tried on real output of fMRIPrep versions from 20.2 to 23.0; the file names of 23.2 and later are recognised, but no real derivative of those versions has been tried.

- The raw dataset is found as the dataset root above `derivatives/`, or inside the derivative at `sourcedata/raw/`; when it lies elsewhere, name it with `--raw /path/to/raw_bids`.
- The surfaces alone need only the FreeSurfer subjects. `review` takes a FreeSurfer `SUBJECTS_DIR`, one subject directory, an fMRIPrep output folder (`sourcedata/freesurfer/`, or the older `freesurfer/` beside `fmriprep/`), or a dataset root like the one above, and the folders below `derivatives/` may have other names (`derivatives/fmriprep-21.0.2/...`). The first line it prints says where it found the subjects. With fewer than 30 subjects, or without an `fsaverage` (next to the subjects, in `$FREESURFER_HOME`, or `--fsaverage DIR`), subjects are ranked on two raw numbers only, and the command says so.
- A DataLad clone whose files are not fetched: `review` fetches nothing and changes nothing in it. It names the files that have no content and prints one `datalad ... get` command that fetches them; run that, then the same `review` again.
- The work goes into `surfcensus-run` beside the dataset, never inside it, or `--run-dir DIR`. Nothing is written into your data.
- The EPI movies of one subject without a run directory: `uv run surfcensus epi-session /path/to/mystudy --subject 01 --out epi-sub-01 --serve`.

### 5. On a server

The app listens on `127.0.0.1` of the machine it runs on, and loads nothing from the internet. On a server, start it without a browser:

```bash
uv run surfcensus review /path/to/mystudy --no-browser --port 8790
```

and forward the port from your own machine:

```bash
ssh -N -L 8790:127.0.0.1:8790 you@yourserver
```

Then open `http://127.0.0.1:8790/` on your machine. When the port is taken on the server, `review` takes the next free one and prints it on the line `Rating app: http://127.0.0.1:NNNN/`; forward that port instead. When it is taken on your own machine, `ssh` says `bind: Address already in use`; use another local number (`-L 9000:127.0.0.1:8790`, then `http://127.0.0.1:9000/`).

### 6. Afterwards

- Ctrl-C stops the app. The same command again continues where you were; `uv run surfcensus serve --run-dir DIR` opens a run that is ready without preparing anything.
- `uv run surfcensus status --run-dir DIR` says what is rendered, packed, failed or waiting; `--failures` names the failures and their reasons.
- `uv run surfcensus export --run-dir DIR` writes the ratings, marks, scores and, with functional data, `func_sessions` and `func_runs` as tables into `DIR/export/` (below).
- Two people can rate one run one after the other (`--rater NAME`); `uv run surfcensus finalize --run-dir DIR` then writes the labels they agree on.

What the install costs, measured on 2026-11-14 on one 4-core Linux machine with a fast connection: 50 packages, 227 MB of downloads and about 0.8 GB on disk.

## What works today

### Look at your subjects and rate them

```
surfcensus review /path/to/SUBJECTS_DIR
```

`PATH` is a FreeSurfer `SUBJECTS_DIR`, an fMRIPrep output folder, a BIDS derivatives folder with `freesurfer/`, one subject directory, or the root of a BIDS dataset whose FreeSurfer subjects lie below `derivatives/` (`derivatives/freesurfer/`, `derivatives/fmriprep/sourcedata/freesurfer/` and the like are found without naming them). The command lists the subjects, scores them, renders the first one and opens the rating app in your browser at `http://127.0.0.1:8765/`; the other subjects are rendered while you rate. It keeps its work in a run directory, `surfcensus-run` next to `PATH` (next to the dataset when the subjects lie in one, never inside it; or `--run-dir DIR`), and says on the terminal what it found, which subjects cannot be rendered and why, and how long it took to get to the app (about 11 s for two real subjects on a 4-core machine, under 1 s when you come back). Ctrl-C stops it; the same command again continues where you were.

- The rating screen shows three views that sweep through the back of the brain. A first visit starts with a short tour. You give each of three questions (sagittal sinus and falx, tentorium and transverse sinus, white surface) a value with the number keys, click on an error to mark where it is, and press Enter for the next subject. `k` skips a subject, which then comes back after the others; `Backspace` goes back; `?` shows the definitions and every key.
- Subjects come worst-scoring first. `--blind` offers them in a random order and keeps the review screen closed until all are rated. `--rater NAME` says who is rating; two people can rate the same run one after the other, and `surfcensus finalize --run-dir DIR` then writes the labels they agree on and lists the subjects they do not.
- The review screen (`/review`) lists every subject with its scores and labels and shows a rated subject with its marks.
- `--no-browser` does not open a browser, `--no-serve` prepares everything and stops, `--serve-after 0` renders every subject before the app opens, `--fsaverage DIR` names the template (see `metrics` below).

```
surfcensus export --run-dir /path/to/surfcensus-run
```

writes `export/` in the run directory: `subjects`, `scores`, `selection`, `ratings`, `marks`, `final_marks`, `clusters`, `datasets`, `behaviour` and `skips` as CSV (`--format parquet` for parquet), and a `manifest.json` with the checksum of each. CSV has no types: read identifiers such as `subject_uid` as text (`pd.read_csv(path, dtype={"subject_uid": str})`), or use parquet.

The same from Python:

```python
from surfcensus.review import review
from surfcensus.export.tables import export_run

run = review("/path/to/SUBJECTS_DIR", serve=False)  # every step, every render, no app
export_run(run.run_dir)
```

The app listens on your own machine only and loads nothing from the internet.

### A directory of subjects, step by step

`review` runs these commands for you. To run them one at a time, point `inventory` at the same kinds of folder and give it a run directory outside your subjects directory to write to:

```
surfcensus inventory --run-dir runs/mystudy --source /path/to/SUBJECTS_DIR
surfcensus metrics --run-dir runs/mystudy --jobs 4
surfcensus render --run-dir runs/mystudy --jobs 4
surfcensus status --run-dir runs/mystudy
```

- `inventory` lists the subjects, without opening their images, and reads each subject's build stamp and recon-all logs. It prints which layout it found, how many subjects can be rendered, their FreeSurfer versions, and which subjects lack which files. `--subjects 'sub-0*'` restricts the run to matching directory names.
- `metrics` scores the subjects. It resamples each subject's cortical thickness to fsaverage6 through its `sphere.reg`, compares it per location with the other subjects of the run, finds clusters of unusually thick cortex around the occipital pole, counts vertices at FreeSurfer's 5 mm thickness maximum, and reads regional thickness from the stats files. The tables are in `tables/` of the run directory; `scores.parquet` has one row per subject with a rank. Two things decide what it can do:
  - It needs an `fsaverage` directory. It finds the one of a FreeSurfer installation (`$FREESURFER_HOME`) or one next to your subjects, as fMRIPrep writes it; otherwise pass `--fsaverage /path/to/fsaverage`. The run then keeps that template.
  - Comparing per location needs at least 30 subjects with a `sphere.reg`. With fewer, or without an fsaverage directory, subjects are ranked on two raw numbers only: the share of occipital vertices at the thickness maximum, and occipital thickness relative to frontal thickness. The command says which of the two it did.
- `render` writes `media/<id>/` for every subject that has a T1 and white and pial surfaces of both hemispheres: the tiles, `poster.webp`, `render.json` and `vertex_map.npz` described below, and `anim.mp4` with `--video`. Subjects get an identifier that does not show their name, because the rating app keeps raters blind.
- `status` says how many subjects are inventoried, scored and rendered, failed or pending; `status --failures` lists each failure with its reason.

All four can be run again at any time. A second call does only what is new: a subject whose files changed is read, scored and rendered again, the others are skipped, and a subject that failed is tried again when its files change or with `--retry-failed`. If the data moves, `inventory --source /new/place --relocate` keeps the run. The exit code is 0 when nothing is in the failed state, 1 when a subject is, and 2 for a wrong call or for what lies in the machine (a full or read-only disk, a video encoder that does not work, an fsaverage directory that cannot be used): that is held against no subject, and the same command does the work once it is put right.

The same from Python:

```python
import surfcensus as sc

run = sc.Run("runs/mystudy", sc.sources.local("/path/to/SUBJECTS_DIR"))
run.inventory()
run.metrics(jobs=4)
run.render(jobs=4)
run.public().subjects()  # a pandas DataFrame: one row per subject
run.scores()  # one row per subject, by primary_rank
run.clusters()  # the clusters of thick cortex, with their position where rendered
```

Tested on synthetic subjects with planted thick patches and on real subjects from OpenNeuroDerivatives (FreeSurfer 6.0.1 and 7.3.2): two with all their files, and 34 with the files the scores need. Rendering took 9 s for the two without video in two processes; scoring the 34 took 12 s in four processes. The 34 are not rated, and what the scores of real subjects are worth is not known yet (`docs/metrics.md` has what was measured).

### One subject

```
surfcensus render-subject /path/to/SUBJECTS_DIR/sub-01 --out media/sub-01
```

This reads `mri/T1.mgz`, `surf/?h.white` and `surf/?h.pial` (and `mri/brainmask.mgz` if present) and writes to `media/sub-01`:

- `tiles/`: for every slice of the three panels, a grayscale WebP of the T1 and a transparent WebP of the surface contours;
- `anim.mp4` and `poster.webp`: the three panels side by side, sweeping through a box around the occipital pole;
- `render.json` and `vertex_map.npz`: the geometry that turns a click on a tile into a voxel, a tkr RAS coordinate and a surface vertex.

An `--out` that ends in `.mp4` writes the video alone; `--no-video` leaves the video out. An `--out` inside the subject directory is refused. When an `fsaverage` directory is found (as for `metrics`, or with `--fsaverage DIR`) and the subject has `surf/?h.sphere.reg`, `vertex_map.npz` also says which fsaverage6 vertex each surface vertex corresponds to; `--no-template` leaves that out.

The same from Python:

```python
import surfcensus as sc

info = sc.render.three_plane("/path/to/SUBJECTS_DIR/sub-01", "media/sub-01")
```

The renderer's frames have not yet been compared by eye with Freeview. Details are in `docs/rendering.md`.

### The EPI movies of one subject

```
surfcensus epi-session /path/to/bids_dataset --subject 01 --out epi-sub-01 --serve
```

This packs the raw and the preprocessed EPI series of every functional run of the subject, all sessions in one stream (in the order of their acquisition dates when the raw dataset's `sessions.tsv` or `scans.tsv` gives one for every session, else in the order of their labels, which is what most public datasets get, since their dates are removed or shifted), into `epi-sub-01` and shows them on a page on `127.0.0.1`: the same slices of the anatomy with the FreeSurfer white and pial surfaces as contours. It serves two uses at once: quality control of the raw data (motion, artefacts in single volumes, distortion and dropout as acquired, movement between runs) and quality control of the preprocessing (what fMRIPrep corrected, what remains, how the preprocessed series fits the surfaces). The raw stream is the raw BIDS series resampled once (cubic interpolation) onto the grid of the preprocessed series by one rigid placement per session, the stored transform of the session's first run, and nothing else; which runs are packed does not change that placement. The preprocessed stream is the series as fMRIPrep wrote it.

The path is a BIDS dataset with an fMRIPrep derivative below `derivatives/`, the derivative itself, or its FreeSurfer subjects directory; `--raw DIR` names the raw dataset when it lies elsewhere. `--session LABEL` packs only that session, `--epi-runs GLOB` only the runs whose name matches. The packs are a display cache: nothing is written into the data, no NIfTI file is written anywhere, and an `--out` inside one of the folders that are read is refused. Runs that cannot be shown (no raw series, multi-echo, only a standard-space or native-space series, files of a DataLad clone that are not fetched) are listed with their reason. Only fMRIPrep derivatives with `space-T1w` output are packed. The same command again keeps the packs that are current and makes only what changed (`--force` makes all again), so a second look at the two runs below takes about 3 s where the first took about 31 s.

Without data of your own, `python scripts/fetch_example.py OUT` (from a clone; `uv run python ...`) fetches a small real example, `ds000102` sub-01 with its fMRIPrep derivative and FreeSurfer subject, about 160 MB, anonymously from the public buckets into `OUT/ds000102`, and prints the two commands to run on it. `--dataset dsNNNNNN --subject LABEL` fetches one subject of another dataset that OpenNeuroDerivatives has an fMRIPrep derivative of, by listing the two public buckets; the EPI movies need a derivative with T1w-space series (fMRIPrep below 23.2 there).

This is step 1 of the EPI interface (`docs/epi_interface.md`): one subject, no run directory, view only; nothing is rated. On the two runs of `ds000102` sub-01 (146 volumes each, 3 mm) the page was up 34 s after the command on one machine. Two independent reviews recomputed the packed values from the files and drove the page on real packs; no person has looked at a pack yet.

### Both interfaces from one command

```
surfcensus review /path/to/bids_dataset
```

When the folder holds an fMRIPrep derivative with functional runs below `derivatives/` and the raw BIDS series beside it, `review` also lists the functional runs after the FreeSurfer subjects, says how many can be shown and why the others cannot, and, once the rating screen is up and the surfaces are rendered, packs the raw and preprocessed EPI series of every session behind the app. The app then offers the EPI movies at `/epi` (the address is printed as `EPI movies:`, also by `surfcensus serve`): one stream per participant over all runs and sessions, the same as `epi-session` shows. When no subject can be rated, the browser opens at the EPI movies. `--raw DIR` names a raw dataset that lies elsewhere, `--epi-runs GLOB` packs only the runs whose name matches, and `--no-epi` leaves the functional side out; all three are kept for later calls. `surfcensus pack --run-dir DIR` makes the packs of a run on its own; a run whose pack failed is tried again with `--retry-failed`. The packs are a display cache in the run directory; nothing is written into the data, and an export or a run directory inside a participant folder of the raw dataset or the derivative is refused. A folder without functional data is reviewed exactly as before. While you rate, the link "functional runs of this subject" under the rating bar opens the page of the subject on screen: its surfaces, and every functional session in the order of the stream with its anchor run, its one registration and which runs moved more than 10 mm from it (a statement about the raw data), and every run with its state, its volumes, how far it lies from the session's registration and what the pipeline did to it (head-motion, distortion and slice-timing correction, the registration), each with the file it was read from; what every run of a session has is said once for the session. Each packed run has a link that opens the EPI movie at that run, and the EPI movie then links back to the subject's page. `/subjects` lists every subject, and the review screen has two columns that lead to the same pages and a link to the EPI movies. In a blind run, or with a second rater, these pages open once you have nothing left to rate. `surfcensus status` then also says how many functional runs can be shown, how many packs are made, failed or waiting, and how many runs are flagged, and `--failures` names the runs that cannot be shown and why. `surfcensus export` also writes `func_sessions` and `func_runs`, one row per session and per run, without motion traces. These pages and tables are step 3 of the EPI interface (`docs/epi_interface.md`, "Step 3 as built"). All of it was tested on synthetic folders and on `ds000102` sub-01; no person has looked at the result yet.

## Planned

- The rating app for a team: several raters at once with their own links, a calibration set that every rater rates first, adjudication of disagreements, and a dashboard.
- `pip install surfcensus` from PyPI.
- A normative reference to score small sets against, built from rated census subjects.
- The OpenNeuro census: the same pipeline over FreeSurfer outputs in public OpenNeuro derivatives, to measure how common these errors are and how often quantitative checks miss them. See `docs/phases.md`.

## Development

From a clone: `just setup`, `just test` and `just lint`; the `justfile` shows the `uv` commands behind them, and `uv run surfcensus ...` runs the command of the clone. `just test-install` builds the wheel, installs it into a fresh environment and rates a synthetic subject through `surfcensus review` there; it needs the package index. The code is licensed BSD-3-Clause.
