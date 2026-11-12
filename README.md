# surfcensus

Look at your FreeSurfer surfaces where they most often go wrong.

Near the occipital pole, FreeSurfer's pial and white surfaces often absorb the superior sagittal sinus, the transverse sinus or the cerebellar tentorium, and these errors are rarely edited. `surfcensus` is being built to score reconstructions with per-vertex cortical-thickness outlier metrics in fsaverage space, to render animated sagittal, coronal and axial views around the occipital pole with white and pial contours, and to serve those views in a keyboard-driven rating app in which raters click on errors to mark where they are.

## Status

Milestone 1 of 6 is built: rendering one subject. Everything else is planned and does not exist yet, including `surfcensus review`, `surfcensus export`, the metrics and the rating app. The package is not on PyPI. The specification is in `docs/`; the build order and what has been checked are in `CLAUDE.md`.

## What works today

Install from a clone with [uv](https://docs.astral.sh/uv/), then render one FreeSurfer subject directory:

```
uv sync
uv run surfcensus render-subject /path/to/SUBJECTS_DIR/sub-01 --out media/sub-01
```

This reads `mri/T1.mgz`, `surf/?h.white` and `surf/?h.pial` (and `mri/brainmask.mgz` if present) and writes to `media/sub-01`:

- `tiles/`: for every slice of the three panels, a grayscale WebP of the T1 and a transparent WebP of the surface contours;
- `anim.mp4` and `poster.webp`: the three panels side by side, sweeping through a box around the occipital pole;
- `render.json` and `vertex_map.npz`: the geometry that turns a click on a tile into a voxel, a tkr RAS coordinate and a surface vertex.

An `--out` that ends in `.mp4` writes the video alone; `--no-video` leaves the video out. No FreeSurfer installation or licence is needed, and the subject directory is only read.

The same from Python:

```python
import surfcensus as sc

info = sc.render.three_plane("/path/to/SUBJECTS_DIR/sub-01", "media/sub-01")
```

The renderer is tested on synthetic subjects and on one real subject (`ds000102` sub-01, FreeSurfer 6.0.1). Its frames have not yet been compared by eye with Freeview. Details are in `docs/rendering.md`.

## Planned

- Your own data: `surfcensus review /path/to/SUBJECTS_DIR` will inventory, score and render the subjects of a FreeSurfer `SUBJECTS_DIR`, an fMRIPrep output folder or a BIDS derivatives folder with `freesurfer/`, and open the rating app in the browser. `surfcensus export` will write the ratings.
- The OpenNeuro census: the same pipeline over FreeSurfer outputs in public OpenNeuro derivatives, to measure how common these errors are and how often quantitative checks miss them. See `docs/phases.md`.

## Development

`just setup`, `just test` and `just lint`; the `justfile` shows the `uv` commands behind them. The code is licensed BSD-3-Clause.
