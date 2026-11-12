# surfcensus

Project memory for Claude Code. Read this first; the detailed specs live in `docs/` and are imported below.

## What this project is

`surfcensus` is a pip-installable Python package that scores FreeSurfer reconstructions quantitatively, renders animated three-plane views around the occipital pole, and serves those renderings to human raters through a web app built for thousands of brains.

It has two uses with one pipeline (see `docs/architecture.md`):

1. Any researcher points it at their own FreeSurfer subjects (`surfcensus review /path/to/SUBJECTS_DIR`) and rates them in the browser, offline, with no FreeSurfer install. This is a first-class use, not an afterthought.
2. The paper team runs a census over FreeSurfer outputs found in public OpenNeuro derivatives.

It is the surface-reconstruction arm of a paper on visualizing fMRI preprocessing quality. The paper's thesis is that people must look at their data. This package supplies the evidence for that thesis on the anatomical side: a census of how often FreeSurfer pial and white surfaces in shared public derivatives absorb the superior sagittal sinus, the transverse sinus, or the cerebellar tentorium near the occipital pole, and how often quantitative checks miss it.

The scientific claims the package must support are:

1. Prevalence: the fraction of public FreeSurfer reconstructions with visible non-brain inclusion in occipital cortex, by dataset, field strength, FreeSurfer version, and T2/FLAIR-pial use.
2. Detectability: how well cortical-thickness outlier scores, computed per vertex in fsaverage space, predict rater judgments, both per subject and per location (raters click-mark errors). In particular, the rate of rater-identified failures among subjects that look normal on the numbers: the blind spot of quantitative QC.

Everything in the package serves one of these two numbers.

## Status

Milestone 1 (`fsio` and `render`) is built and tested on synthetic subjects and on `ds000102` sub-01; `surfcensus render-subject SUBJECT_DIR --out DIR` works. Its exit check, the by-eye comparison with Freeview, is still to be done by Tomas. Milestone 2 (local source, config, catalog, inventory, `Run`) is built: `surfcensus inventory`, `surfcensus render` and `surfcensus status` work on a local directory and are resumable, and `sc.Run(...)` and `sc.sources.local(...)` are the Python API behind them. It is tested on synthetic directory trees and, end to end, on two real subjects linked into an fMRIPrep layout (`ds000102` sub-01, FreeSurfer 6.0.1; `ds003020` sub-UTS01, FreeSurfer 7.3.2). A census run can be created and opened from Python, with local sources only; no command creates one yet. There are no metrics, no scores and no rater app yet, so `surfcensus review` does not exist. Milestones 3 to 6 are not started. The docs below are the specification. When the docs and the code disagree, the docs win until a decision in `docs/decisions.md` changes them; update the docs in the same change as the code.

## Build order

Build the local path first: it is the smallest thing that is useful on its own, and everything in the census reuses it. Each milestone ends with tests passing and a short note in `docs/decisions.md` for any choice made along the way.

1. `fsio` and `render`: read a FreeSurfer subject with nibabel, find the occipital pole, slice meshes, write per-panel WebP tiles in two layers (decision D11), a composite MP4 and poster, `render.json` and `vertex_map.npz`. Test on synthetic data, then on one real subject (fetch `ds000102` sub-01 Tier B and C files from the OpenNeuroDerivatives bucket into a pytest cache). Compare a few frames by eye with Freeview output from the gist.
2. `sources/local` and `inventory`: layout detection, file roles, run directory, catalog.
3. `templates`, `fsio/registration`, `metrics/fsaverage`, `metrics/clusters`, `metrics/regional`, `metrics/ceiling`, `metrics/scoring`: resampling to fsaverage6, within-set z-maps (shipped reference is empty at first), and the composite score that `review` sorts by (decisions D16, D18).
4. `app`: rating screen, keyboard map, click marks with coordinate round-trip, local mode, export. Then `surfcensus review` end to end.
5. Census: `privacy`, `remote`, `sources/openneuro_derivatives`, discovery, inventory, enrichment, selection, team mode of the app, Snakemake workflow.
6. Embedded-derivatives source (only if decision O9 says so), reference building, analysis notebooks.

Phase 0 of the census (pilot, `docs/phases.md`) can start as soon as milestone 4 works.

## Verified facts the design rests on (checked 2026-11-01)

- The OpenNeuroDerivatives S3 bucket (`s3://openneuro-derivatives`, HTTPS `https://openneuro-derivatives.s3.amazonaws.com`) is listable anonymously. Top-level prefixes: `fitlins/`, `fmriprep/`, `mriqc/`, `xcpd/`. Under `fmriprep/` there were 308 dataset prefixes of the form `dsXXXXXX-fmriprep/`.
- In `fmriprep/ds000102-fmriprep/`, each subject has a FreeSurfer subjects directory at `sourcedata/freesurfer/sub-XX/`. For `sub-01` this holds 309 files, including `mri/T1.mgz` (about 5.6 MB), `surf/{lh,rh}.{white,pial}` (about 5.3 MB each), `surf/{lh,rh}.thickness` (about 0.6 MB), `mri/transforms/talairach.xfm`, `scripts/recon-all.log` (about 220 kB), `scripts/build-stamp.txt`, and `stats/{lh,rh}.aparc.stats`.
- Subjects also have `surf/{lh,rh}.sphere.reg` (about 5.3 MB each), needed for resampling to fsaverage. The same `sourcedata/freesurfer/` folder contains `fsaverage/` and `fsaverage5/`; `fsaverage/surf/lh.sphere.reg` is about 5.9 MB.
- nilearn's `fetch_surf_fsaverage` provides fsaverage3 to fsaverage7 meshes (fsaverage6: 40,962 vertices; fsaverage: 163,842), including sphere meshes.
- Tomas's Freeview gist (`https://gist.github.com/tknapen/85d9a23a09c95d15f4b23ba1965fa54d`) renders sagittal slices with white (red) and pial (white) surfaces over `T1.mgz` through `freeview -cmd`, then joins frames with ffmpeg. It uses fixed voxel slices (x = 90 to 239, y = z = 127).
- `mris_thickness` documents a default maximum thickness of 5 mm (`-max`). How vertices above the maximum are stored is not documented on that page and must be checked in source or empirically (see `docs/metrics.md`).

Checked 2026-11-12 on `ds000102` sub-01 (one subject, FreeSurfer 6.0.1, fMRIPrep 21.0.2; what was checked for FreeSurfer 7 is in the next list):

- Over-maximum thickness is clamped, not zeroed. `lh.thickness` has a maximum of exactly 5.0, with 16 vertices at that value (41 in `rh`); at those vertices the white-to-pial vertex distance has a median of 7.3 mm. Zeros are the medial wall: 5,914 of the 5,919 zero-valued `lh` vertices lie outside `label/lh.cortex.label`, 5 lie inside it.
- In `fsaverage/surf/`, `lh.sphere.reg` and `lh.sphere` hold identical coordinates, at radius 100 (99.992 to 100.008). The subject's `lh.sphere.reg` also has radius 100.
- Icosahedral nesting holds for fsaverage5: `fsaverage5/surf/lh.sphere.reg` equals the first 10,242 vertices of fsaverage `lh.sphere.reg` exactly. fsaverage6 by truncation to 40,962 vertices is inferred from this, not checked directly (the folder has no `fsaverage6/`).
- `mri/T1.mgz` is 256 x 256 x 256, 1 mm, uint8, with vox2ras-tkr rows `[-1 0 0 128]`, `[0 0 1 -128]`, `[0 -1 0 128]`.
- `surf/lh.pial` is a regular key; there is no `lh.pial.T1` (FreeSurfer 6 layout).
- `scripts/recon-all.log` (220,842 bytes) has no `thickness calculation complete` line. The line is in `scripts/recon-all-lh.log` (212,143 bytes; `39:225 truncations`) and in `scripts/recon-all-rh.log` (206,247 bytes; `93:434 truncations`), once each.
- The `fsaverage/` and `fsaverage5/` directories next to the subjects each hold `mri/`, `surf/`, `label/` and `scripts/`, and `fsaverage5/mri/T1.mgz` exists (bucket listing). A template directory therefore passes for a subject directory but for its name.
- Lossy WebP at quality 80 desaturates thin coloured contours: on 15 sagittal tiles (292 x 320 px, `lh` surfaces only) the mean absolute error on contour pixels was 47, 16 and 70 (R, G, B, of 255). Tile sizes: 14.7 kB with contours baked in; 8.7 kB grayscale only; 16.3 kB for the contours as a separate lossless RGBA layer.
- Both buckets are reachable anonymously from the Claude Code cloud environment; `gist.githubusercontent.com` is not.

Checked 2026-11-12 on `ds003020` sub-UTS01 (one subject, FreeSurfer 7.3.2, fMRIPrep 23.0.1) and on the logs of both subjects:

- The listing of the subject's prefix (337 keys) has `surf/lh.pial.T1` and `surf/rh.pial.T1` and no `surf/?h.pial` key: the link that recon-all makes is absent on S3.
- Over-maximum thickness is clamped in FreeSurfer 7.3.2 as well: the maximum of `lh.thickness` and of `rh.thickness` is exactly 5.0. Zeros were not counted for this subject.
- `scripts/recon-all.log` holds 4 recon-all invocations in the 6.0.1 subject and 5 in the 7.3.2 subject; each per-hemisphere log holds 2. An invocation's header ends with a line `subjid NAME`; the arguments are on the last non-blank line above it, alone in 6.0.1 and after the script path in 7.3.2. Invocations are separated by `New invocation of recon-all` in 6.0.1 and `#New# invocation of recon-all` in 7.3.2. The first 64 kB of either `recon-all.log` hold 2 invocations.
- The truncation lines are in the per-hemisphere logs only, in both subjects: `lh` 39:225 and `rh` 93:434 (6.0.1), `lh` 22:230 and `rh` 23:325 (7.3.2). The line `orig.nofix lhholes = N, rhholes = M` is in `recon-all.log`: 41 and 26, and 4 and 15. `stats/aseg.stats` has the same numbers in both subjects.
- Both `recon-all.log` files end with a line `recon-all -s NAME finished without error at DATE`. The 7.3.2 logs pass `-noT2pial -noFLAIRpial`. No log holds a T2, FLAIR, hires, expert or edit flag.
- Build stamps: `freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a` and `freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275`. The `fsaverage` next to `ds000102` has `freesurfer-x86_64-redhat-linux-gnu-dev4-20090216`, which holds no version number.
- Sizes from the two listings: Tier A (three stats files, build stamp, three logs) 0.66 and 0.98 MB; Tier B (thickness, `sphere.reg`, `aparc.annot`) 14.2 and 12.7 MB, with `?h.aparc.annot` at 1.05 to 1.19 MB; Tier C (T1, brainmask, white, pial) 28.5 and 23.2 MB.
- The renderer runs on the 7.3.2 subject: `mri/T1.mgz` is 256 x 256 x 256 at 1 mm and passes the orientation check, 156 of 176 tiles have contours, 8.1 s and 6.3 MB without video. One poster frame was looked at; the contours lie on the tissue boundaries. Nothing was compared with Freeview.

Read in source on 2026-11-12 (FreeSurfer and Freeview on GitHub, nibabel 5.5.0 as installed); none of it checked against a running Freeview, and against FreeSurfer 7 data only where the list above says so:

- Freeview's default 2D views are radiological in coronal and axial (subject's left on the right), anterior up in axial, and in sagittal show posterior at the left and anterior at the right (`RenderView2D.cpp`: sagittal camera at +x, view-up +z).
- Freeview treats an integer voxel index as the voxel centre and cuts the surface with the plane through that centre; no half-voxel offset between image and contour in the code path.
- `nibabel.freesurfer.read_geometry` returns coordinates as stored (tkr RAS) and does not apply the footer's `c_ras`. `MGHHeader.get_vox2ras_tkr()` ignores the direction cosines.
- From FreeSurfer 7.0.0, recon-all writes `?h.pial.T1` and makes `?h.pial` a relative link to it (or to `?h.pial.T2` / `?h.pial.FLAIR`).
- `MRISmeasureCorticalThickness` clamps each of its two distance measurements at the maximum and stores their mean, in v6.0.0, v7.0.0 and v7.4.1; recon-all in FreeSurfer 7 calls it through `mris_place_surface --thickness` with a maximum of 5.
- Lossy WebP is always 4:2:0; Pillow's lossless WebP zeroes RGB under alpha 0 unless `exact=True`.
- `scripts/recon-all` at v6.0.0 and v7.4.1: `-hemi lh` sets the log name to `recon-all-lh.log`. T2pial is set by `-T2pial`, `-t2pial`, `-T2pial-only` and `-autorecon3-T2pial` and cleared by `-noT2pial`, `-no-T2pial`, `-no_T2pial`, `-no-t2pial` and `-no_t2pial`; FLAIRpial likewise. The argument parser compares whole arguments, case-sensitively. There is no flag that clears `-hires`. `-3T` and `-3t` are the only flags that start with a digit. `-autorecon2-pial` is in neither version.

Measured 2026-11-12 on a full render of `ds000102` sub-01: 9 s on one core; 10.9 MB of media (tiles 4.1 MB, `anim.mp4` 4.3 MB, `vertex_map.npz` 2.5 MB since it follows the tiles and not the box, 1.8 MB before); 155 of 176 tiles contain contours.

Measured 2026-11-12 on a local run over the two real subjects (one machine, files on a local disk): `surfcensus inventory` under 0.1 s for the phase, with the logs read whole; `surfcensus render --jobs 2` 9.0 s without video; `surfcensus render --video` 17.5 s in one process; a second call of either does nothing in under 0.1 s; `catalog.sqlite` 64 kB; the source files unchanged in size and modification time. The media of `ds000102` sub-01 rendered through a run with video are byte-identical, file for file, to the render that `render-subject` made of it at the end of milestone 1 (same machine).

Anything else in these docs marked "(to verify)" has not been checked. Do not promote it to fact without checking.

## Hard rules

- Read-only, anonymous access to all public buckets. Never write to, tag, or request credentials for OpenNeuro resources.
- Local sources are read in place and never modified. Outputs go to the run directory only.
- Core package has no dependency on a FreeSurfer installation or license. Reading `.mgz` and surface files goes through nibabel. Freeview is an optional rendering backend only.
- Every derived number traces back to an exact source: dataset accession, snapshot or derivative commit, subject, file key, file size or ETag. Store these in the inventory.
- Renders and metrics are deterministic for a given input and config. Any randomness (rater queue order, sampling) takes an explicit seed that is recorded.
- Raters never see dataset IDs, metric scores, or other raters' ratings while rating.
- Datasets are identified by keyed pseudonym (HMAC with a secret key, `D-xxxxxxxx`) in every output: app, gallery, figures, tables, exports, logs. Accessions live only in `catalog.sqlite`. Never use an unkeyed hash of an accession; the accession space is small enough to invert one by enumeration. The same holds for bucket metadata: no unkeyed hash of an object's key, size or ETag leaves `catalog.sqlite` either. Never commit or copy the key into the repo or a run directory. See decisions D10 and D20.
- Network-touching tests are marked `@pytest.mark.network` and excluded from the default test run.

## Lab conventions this project follows

- Environment and packaging with `uv`; `pyproject.toml` with hatchling build backend; src layout.
- Lint and format with `ruff`; type checking with `pyrefly`.
- Array shapes annotated with `jaxtyping`, runtime-checked with `beartype` in tests and in debug mode.
- `justfile` for developer commands; Snakemake (in `workflow/`) orchestrates the full census run by calling the package CLI; marimo notebooks (in `notebooks/`) for analysis.
- DataLad is acceptable for fetching specific files from OpenNeuro datasets where S3 keys are not directly usable.

## Commands

Built:

```
just setup            # uv sync with dev extras
just test             # pytest, no network
just test-network     # pytest -m network
just lint             # ruff check + ruff format --check + pyrefly
surfcensus render-subject SUBJECT_DIR --out DIR   # one subject, no run directory
surfcensus inventory --run-dir DIR [--source PATH]   # phase 2; --source creates a local run
surfcensus render --run-dir DIR [--jobs J]           # phase 6
surfcensus status --run-dir DIR [--failures]         # where a run stands
```

Planned:

```
surfcensus review PATH   # local data: inventory, metrics, render, open rater app
surfcensus discover   # phase 1
surfcensus enrich     # phase 3
surfcensus metrics    # phase 4
surfcensus select     # phase 5
surfcensus serve      # phase 7, rater web app
surfcensus export     # phase 8, tables for analysis
```

## Detailed specs

@docs/architecture.md
@docs/data_sources.md
@docs/phases.md
@docs/metrics.md
@docs/rendering.md
@docs/rater_app.md
@docs/rating_protocol.md
@docs/analysis_plan.md
@docs/decisions.md
@docs/prior_art.md
