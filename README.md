# surfcensus

Look at your FreeSurfer surfaces where they most often go wrong.

Near the occipital pole, FreeSurfer's pial and white surfaces often absorb the superior sagittal sinus, the transverse sinus or the cerebellar tentorium, and these errors are rarely edited. `surfcensus` is being built to score reconstructions with per-vertex cortical-thickness outlier metrics in fsaverage space, to render animated sagittal, coronal and axial views around the occipital pole with white and pial contours, and to serve those views in a keyboard-driven rating app in which raters click on errors to mark where they are.

## Status

Milestones 1 to 3 of 6 are built: rendering one subject; inventorying and rendering a directory of subjects into a run directory; and scoring them. Everything else is planned and does not exist yet, including `surfcensus review`, `surfcensus export` and the rating app. The scores have been run on real subjects but not yet compared with anyone's judgment of the surfaces, so they say which subjects stand out on the numbers, not which have an error. The package is not on PyPI. The specification is in `docs/`; the build order and what has been checked are in `CLAUDE.md`.

## What works today

Install from a clone with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

No FreeSurfer installation or licence is needed. Your data is only read: nothing is written into a subject directory or into the directory that holds your subjects.

### A directory of subjects

Point `inventory` at a FreeSurfer `SUBJECTS_DIR`, an fMRIPrep output folder, a BIDS derivatives folder with `freesurfer/`, or one subject directory, and give it a run directory outside your subjects directory to write to:

```
uv run surfcensus inventory --run-dir runs/mystudy --source /path/to/SUBJECTS_DIR
uv run surfcensus metrics --run-dir runs/mystudy --jobs 4
uv run surfcensus render --run-dir runs/mystudy --jobs 4
uv run surfcensus status --run-dir runs/mystudy
```

- `inventory` lists the subjects, without opening their images, and reads each subject's build stamp and recon-all logs. It prints which layout it found, how many subjects can be rendered, their FreeSurfer versions, and which subjects lack which files. `--subjects 'sub-0*'` restricts the run to matching directory names.
- `metrics` scores the subjects. It resamples each subject's cortical thickness to fsaverage6 through its `sphere.reg`, compares it per location with the other subjects of the run, finds clusters of unusually thick cortex around the occipital pole, counts vertices at FreeSurfer's 5 mm thickness maximum, and reads regional thickness from the stats files. The tables are in `tables/` of the run directory; `scores.parquet` has one row per subject with a rank. Two things decide what it can do:
  - It needs an `fsaverage` directory. It finds the one of a FreeSurfer installation (`$FREESURFER_HOME`) or one next to your subjects, as fMRIPrep writes it; otherwise pass `--fsaverage /path/to/fsaverage`. The run then keeps that template.
  - Comparing per location needs at least 30 subjects with a `sphere.reg`. With fewer, or without an fsaverage directory, subjects are ranked on two raw numbers only: the share of occipital vertices at the thickness maximum, and occipital thickness relative to frontal thickness. The command says which of the two it did.
- `render` writes `media/<id>/` for every subject that has a T1 and white and pial surfaces of both hemispheres: the tiles, `poster.webp`, `render.json` and `vertex_map.npz` described below, and `anim.mp4` with `--video`. Subjects get an identifier that does not show their name, because the rating app will keep raters blind.
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
run.scores()  # one row per subject, by rank
run.clusters()  # the clusters of thick cortex, with their position where rendered
```

Tested on synthetic subjects with planted thick patches and on real subjects from OpenNeuroDerivatives (FreeSurfer 6.0.1 and 7.3.2): two with all their files, and 34 with the files the scores need. Rendering took 9 s for the two without video in two processes; scoring the 34 took 12 s in four processes. The 34 are not rated, and what the scores of real subjects are worth is not known yet (`docs/metrics.md` has what was measured).

### One subject

```
uv run surfcensus render-subject /path/to/SUBJECTS_DIR/sub-01 --out media/sub-01
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

- The rating app: `surfcensus review /path/to/SUBJECTS_DIR` will inventory, score and render the subjects and open the rating app in the browser, worst-scoring subjects first. `surfcensus export` will write the ratings.
- A normative reference to score small sets against, built from rated census subjects.
- The OpenNeuro census: the same pipeline over FreeSurfer outputs in public OpenNeuro derivatives, to measure how common these errors are and how often quantitative checks miss them. See `docs/phases.md`.

## Development

`just setup`, `just test` and `just lint`; the `justfile` shows the `uv` commands behind them. The code is licensed BSD-3-Clause.
