# surfcensus

A visual quality-control tool for MRI and fMRI sessions, with two interfaces in one browser app:

- **Surfaces.** FreeSurfer's white and pial surfaces over the T1, in sagittal, coronal and axial views that sweep through a box around the occipital pole, where the pial surface most often absorbs the superior sagittal sinus, the transverse sinus or the tentorium. Subjects are scored with per-vertex cortical-thickness outlier metrics in fsaverage space, and a rater gives each a value on three criteria and clicks on errors to mark where they are.
- **EPI movies.** The raw and the preprocessed EPI series of every functional run, played as one continuous stream over all runs and sessions of a subject, on the same slices of the anatomy, with the white and pial surfaces as contours. The raw stream is the raw BIDS series placed once per session into the anatomy and nothing else; the preprocessed stream is the series as fMRIPrep wrote it. The difference between them, and how each lies on the surfaces, is what the movies show: quality control of the raw data and of the preprocessing at once.

## Status

Both interfaces are built and run on local data and on datasets of the public OpenNeuro buckets. They have been tested by scripts that drive a browser, on synthetic data and on a few real subjects of `ds000102` and `ds003020`; nobody has yet rated a brain with the app or judged the EPI movies by eye. The thickness scores have been run on real subjects but not compared with anyone's judgment, so they say which subjects stand out on the numbers, not which have an error. Several raters at once, the census over OpenNeuro and a normative reference are planned. The package is not on PyPI. The specification is in `docs/`; what has been built and checked is in `CLAUDE.md`.

## Getting started

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

`uv run surfcensus --help` lists the commands. Every command below is run from the clone with `uv run` in front. Without a clone, `uvx --from git+https://github.com/tknapen/surfcensus surfcensus ...` runs the same commands, and `pip install git+https://github.com/tknapen/surfcensus` installs the package into an environment of yours (leave `uv run` off then). The install is about 50 packages and 230 MB of downloads.

### 2. An OpenNeuro dataset, without downloading it

```bash
uv run surfcensus review openneuro:ds000102 --subjects sub-01 --epi
```

This reviews `sub-01` of `ds000102` straight from the public buckets: the FreeSurfer subject and the fMRIPrep files from the OpenNeuroDerivatives bucket, the raw series from `s3://openneuro.org`. Access is anonymous and read-only. The command first says how much it will fetch, then fetches, scores and renders, and opens the app at `http://127.0.0.1:8765/`. That start page leads to both interfaces: the rating screen of the surfaces (`/rate`) and the EPI movies (`/epi`), which are packed behind it and appear there as they are made. For this subject, on one 4-core machine, the rating screen was up 20 to 25 s after the command, after 210 MB of downloads, and both EPI movies 30 to 40 s after that.

- Without `--epi` only the surfaces are reviewed, and nothing is asked of the raw bucket.
- `--subjects` takes shell patterns (`'sub-0[1-5]'`, or `'*'` for all) and is kept for later calls. A new review that would fetch more than 2 GB before the rating screen stops and asks for `--subjects`.
- The EPI movies need an fMRIPrep derivative with T1w-space series; in OpenNeuroDerivatives those are the derivatives made with fMRIPrep below 23.2. For other datasets the surfaces are reviewed and each functional run is listed with the reason it cannot be shown.
- The work goes into `./ds000102-surfcensus-run` (or `--run-dir DIR`). Files are removed once they are scored and rendered; the renders, the scores and, with `--epi`, the functional files stay.
- Ctrl-C stops it, and the same command again goes on without fetching anything twice. Without a network, `uv run surfcensus serve --run-dir ./ds000102-surfcensus-run` opens what is there.

### 3. An OpenNeuro subject, downloaded

```bash
uv run python scripts/fetch_example.py data
```

```bash
uv run surfcensus review data/ds000102
```

The first command fetches `ds000102` sub-01 (about 160 MB, anonymously) into `data/ds000102`, laid out as a BIDS dataset with its fMRIPrep derivative and FreeSurfer subject below `derivatives/fmriprep/`; `--dataset dsNNNNNN --subject LABEL` fetches a subject of another dataset. The second is the review of step 2, read from disk, with its run directory at `data/surfcensus-run`. The download holds no `fsaverage` template, so subjects are ranked without per-location scores unless `--fsaverage DIR` names one.

### 4. A BIDS dataset with fMRIPrep derivatives of your own

```bash
uv run surfcensus review /path/to/mystudy
```

`review` finds the files below the dataset root by itself. For both interfaces the dataset looks like this:

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

That is the output of fMRIPrep run with FreeSurfer and with `T1w` among its `--output-spaces`. It has been tried on real output of fMRIPrep 20.2 to 23.0; the file names of 23.2 and later are recognised, but no real derivative of those versions has been tried.

- A raw dataset that does not lie above `derivatives/` (or at `sourcedata/raw/` inside the derivative) is named with `--raw /path/to/raw_bids`.
- The surfaces alone need only FreeSurfer subjects: `review` also takes a `SUBJECTS_DIR`, one subject directory or an fMRIPrep output folder, and finds pipeline folders below `derivatives/` under other names (`fmriprep-21.0.2/`). The first line it prints says where it found the subjects.
- Per-location scores need an `fsaverage` (next to the subjects, in `$FREESURFER_HOME`, or `--fsaverage DIR`) and at least 30 subjects. Otherwise subjects are ranked on two raw numbers, and the command says so.
- A DataLad clone whose files are not fetched is not changed: `review` names the files without content and prints one `datalad ... get` command for them. Run it, then the same `review` again.
- The work goes into `surfcensus-run` beside the dataset (or `--run-dir DIR`). Nothing is written into your data, and no image volume is written anywhere.

### 5. On a server

The app listens on `127.0.0.1` of the machine it runs on and loads nothing from the internet. On a server, start it without a browser and forward its port:

```bash
uv run surfcensus review /path/to/mystudy --no-browser --port 8790
```

```bash
ssh -N -L 8790:127.0.0.1:8790 you@yourserver
```

Then open `http://127.0.0.1:8790/` on your own machine. When the port is taken on the server, `review` takes the next free one and prints it on the line `Rating app: http://127.0.0.1:NNNN/`; forward that one instead. When it is taken on your machine, `ssh` says `bind: Address already in use`; pick another local number (`-L 9000:127.0.0.1:8790`, then `http://127.0.0.1:9000/`).

## Using the interfaces

The start page (`http://127.0.0.1:8765/`, or the port `review` prints) shows both interfaces side by side: how many subjects are rated and how many are left, how many functional runs are packed, and a link into each. Every page links back to it ("start"). In a blind run, or with a second rater, the EPI movies and the review screen open once the rater has nothing left to rate; until then the start page says so and shows no numbers of the functional data.

### Rating surfaces

- Three views sweep through the back of the brain; Space pauses, the arrow keys step, `s` hides the contours while held. A first visit starts with a short tour.
- Each subject gets a value on three criteria (sagittal sinus and falx, tentorium and transverse sinus, white surface) with the number keys. A click on an error marks where it is. Enter goes to the next subject, `k` skips one (it comes back after the others), `Backspace` goes back, and `?` shows the definitions and every key.
- Subjects come worst-scoring first; `--blind` shuffles them. `--rater NAME` says who is rating; two people can rate one run one after the other, and `surfcensus finalize --run-dir DIR` then writes the labels they agree on and lists those they do not.
- `/review` lists every subject with its scores and labels and shows a rated subject with its marks. In a blind run, or with a second rater, it opens once you have nothing left to rate.

### Watching EPI movies

- `/epi` lists the participants. A participant's page plays every run of every session in one stream, in acquisition order, raw or preprocessed, with the surfaces as contours; `f` flips from one to the other at the same volume and slices.
- Space plays and pauses, the arrow keys step one volume, `[` and `]` change the rate (up to 300 volumes per second; above 60 the page says that not every volume is painted), `n` switches the intensity normalization per run on and off, `x` plays the volumes in a seeded random order, `s` toggles the surfaces, and `?` lists every key.
- Above the views, a flagged run (one that lies more than 10 mm from the session's registration) and a raw affine that could not be checked are said as statements about the raw data. Under them, the corrections the pipeline applied to the run on screen are named with the file they were read from.
- "functional runs of this subject" on the rating screen and `/subjects` lead to a page per subject: its sessions, their registration, and every run with its state, volumes, displacement and corrections, each linked into the EPI movie at that run.

### Afterwards

- Ctrl-C stops the app; the same command again continues where you were. `surfcensus serve --run-dir DIR` opens a run that is ready without preparing anything.
- `surfcensus status --run-dir DIR` says what is rendered, packed, failed or waiting; `--failures` names each failure and its reason.
- `surfcensus export --run-dir DIR` writes `DIR/export/`: `subjects`, `scores`, `selection`, `ratings`, `marks`, `final_marks`, `clusters`, `datasets`, `behaviour` and `skips`, and with functional data `func_sessions` and `func_runs`, as CSV (`--format parquet` for parquet), with a `manifest.json` of checksums. Read identifiers such as `subject_uid` as text from CSV.

## Commands

`review` does everything in one go. The steps it runs are also commands of their own, each of which can be run again at any time and does only what is new:

```
surfcensus inventory --run-dir DIR --source PATH   # list the subjects and functional runs
surfcensus metrics   --run-dir DIR --jobs 4        # thickness maps, clusters, scores
surfcensus render    --run-dir DIR --jobs 4        # the tiles of the rating screen
surfcensus pack      --run-dir DIR --jobs 4        # the EPI movies
surfcensus status    --run-dir DIR [--failures]
```

Two commands work without a run directory:

```
surfcensus render-subject /path/to/SUBJECTS_DIR/sub-01 --out media/sub-01   # one subject's tiles, video and poster
surfcensus epi-session /path/to/mystudy --subject 01 --out epi-sub-01 --serve  # one subject's EPI movies
```

Useful options of `review`: `--no-browser`, `--port N`, `--run-dir DIR`, `--subjects GLOB`, `--blind`, `--rater NAME`, `--jobs N`, `--fsaverage DIR`, `--raw DIR`, `--no-epi`, `--epi-runs GLOB` (only the runs whose name matches), `--no-serve` (prepare everything and stop), `--serve-after 0` (render every subject before the app opens), `--relocate` (the data moved). Options that describe the run are kept for later calls. Exit codes: 0 when nothing failed, 1 when a subject or run did, 2 for a wrong call or a problem of the machine (a full disk, a network that cannot be reached), which is held against no subject; the same command does the work once it is put right.

The same from Python:

```python
from surfcensus.review import review
from surfcensus.export.tables import export_run

run = review("/path/to/mystudy", serve=False)  # every step, no app
run.scores()  # one row per subject, worst first
export_run(run.run_dir)
```

## What has been checked

- Renders: on synthetic subjects and on real subjects of FreeSurfer 6.0.1 and 7.3.2; the placement of the contours agrees with Freeview's source and with VTK, but no frame has been compared by eye with Freeview.
- Scores: on 34 real, unrated subjects of `ds000102` and `ds003020`; every table was recomputed by independent code (`docs/metrics.md`).
- Marks: every stored click equals a computation from the surface files alone.
- EPI movies: on `ds000102` sub-01 and one other real subject; independent reviews recomputed the raw and preprocessed streams and the contours from the files (no differing voxel) and drove the page on real packs.
- Remote reviews: a review from the buckets gives the same renders, maps and packs as a review of the same files on disk.

## Planned

- The rating app for a team: several raters at once with their own links, a calibration set, adjudication and a dashboard.
- `pip install surfcensus` from PyPI.
- A normative reference to score small sets against, built from rated census subjects.
- The OpenNeuro census: how common these surface errors are in public derivatives, and how often quantitative checks miss them (`docs/phases.md`).

## Development

From a clone: `just setup`, `just test` and `just lint`; the `justfile` shows the `uv` commands behind them. `just test-install` builds the wheel, installs it into a fresh environment and rates a synthetic subject there. The code is licensed BSD-3-Clause.
