# surfcensus

Look at your FreeSurfer surfaces where they most often go wrong.

Near the occipital pole, FreeSurfer's pial and white surfaces often absorb the superior sagittal sinus, the transverse sinus or the cerebellar tentorium, and these errors are rarely edited. `surfcensus` scores reconstructions with per-vertex cortical-thickness outlier metrics in fsaverage space, renders sagittal, coronal and axial views that sweep through a box around the occipital pole with white and pial contours, and serves those views in a keyboard-driven rating app in which raters click on errors to mark where they are.

## Status

Milestones 1 to 4 of 6 are built: rendering one subject; inventorying, scoring and rendering a directory of subjects into a run directory; and the rating app for one researcher's own data, with `surfcensus review` and `surfcensus export`. The census over OpenNeuro, several raters at once and the normative reference are planned and do not exist yet. Nobody has rated a brain with the app so far: it was tested by scripts that drive a browser, on synthetic subjects and on two real ones. The scores have been run on real subjects but not yet compared with anyone's judgment of the surfaces, so they say which subjects stand out on the numbers, not which have an error. The package is not on PyPI. The specification is in `docs/`; the build order and what has been checked are in `CLAUDE.md`.

## Quick start

One line, with [uv](https://docs.astral.sh/uv/), and nothing to install first:

```
uvx --from git+https://github.com/tknapen/surfcensus surfcensus review /path/to/SUBJECTS_DIR
```

Or install the package into an environment of yours (Python 3.11 or later) and run the command:

```
pip install git+https://github.com/tknapen/surfcensus
surfcensus review /path/to/SUBJECTS_DIR
```

The package is not on PyPI yet, so both lines name the repository and need `git`. Once it is, they are `uvx surfcensus review PATH` and `pip install surfcensus`.

That is the whole install: the rating app is part of the package and needs no extra. (`pip install "surfcensus[app]"`, the line these docs gave earlier, still works and installs the same.) No FreeSurfer installation or licence is needed. Your data is only read: nothing is written into a subject directory or into the directory that holds your subjects.

What to expect, measured on 2026-11-14 on one 4-core Linux machine with a fast connection: `pip` on Python 3.13 installs 50 packages, 227 MB of downloads and about 0.8 GB on disk, in 28 s. With the packages already in uv's cache, the `uvx` form had the rating app of two synthetic subjects running 9 s after the command was given, and 2 s the second time.

## What works today

### Look at your subjects and rate them

```
surfcensus review /path/to/SUBJECTS_DIR
```

`PATH` is a FreeSurfer `SUBJECTS_DIR`, an fMRIPrep output folder, a BIDS derivatives folder with `freesurfer/`, one subject directory, or the root of a BIDS dataset whose FreeSurfer subjects lie below `derivatives/` (`derivatives/freesurfer/`, `derivatives/fmriprep/sourcedata/freesurfer/` and the like are found without naming them). The command lists the subjects, scores them, renders the first one and opens the rating app in your browser at `http://127.0.0.1:8765/`; the other subjects are rendered while you rate. It keeps its work in a run directory, `surfcensus-run` next to `PATH` (next to the BIDS dataset when the subjects lie in one, never inside it; or `--run-dir DIR`), and says on the terminal what it found, which subjects cannot be rendered and why, and how long it took to get to the app (about 11 s for two real subjects on a 4-core machine, under 1 s when you come back). Ctrl-C stops it; the same command again continues where you were.

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

## Planned

- The rating app for a team: several raters at once with their own links, a calibration set that every rater rates first, adjudication of disagreements, and a dashboard.
- `pip install surfcensus` from PyPI, and `review` on a downloaded OpenNeuro dataset without naming its subfolder.
- A normative reference to score small sets against, built from rated census subjects.
- The OpenNeuro census: the same pipeline over FreeSurfer outputs in public OpenNeuro derivatives, to measure how common these errors are and how often quantitative checks miss them. See `docs/phases.md`.

## Development

From a clone: `just setup`, `just test` and `just lint`; the `justfile` shows the `uv` commands behind them, and `uv run surfcensus ...` runs the command of the clone. `just test-install` builds the wheel, installs it into a fresh environment and rates a synthetic subject through `surfcensus review` there; it needs the package index. The code is licensed BSD-3-Clause.
