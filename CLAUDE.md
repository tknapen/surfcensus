# surfcensus

Project memory for Claude Code. Read this first; the detailed specs live in `docs/` and are imported below.

## Governing purpose (Tomas, 2026-11-14)

> "Yes, this is the governing purpose, to present people with a full quality control visualization tool that shows, perhaps even on one page, could be different pages, a surface segmentation quality control interface and a raw and pre-processed EPI movie interface. All of those exist for people to be able to judge visually the quality of acquisition and processing on a full session or multiple sessions of fMRI and MRI data."

So the project is one quality-control visualization tool with two interfaces, and everything else serves it:

- The surface interface: FreeSurfer white and pial surfaces over the T1 around the occipital pole, scored, rendered and rated. Built (milestones 1 to 4 and the quick start).
- The EPI movie interface: the raw and the preprocessed EPI series of every run of a session as moving pictures on the same slices, with the surfaces as contours (its purpose statement is in the section "The functional viewer" below; decision D36). Built as a prototype outside the package (`prototypes/func_viewer/`); to be brought into the package and its app.
- The unit a person looks at is a session, or several sessions, of one subject's MRI and fMRI data: the anatomy and its surfaces, and every functional run. What is judged by eye is the quality of the acquisition and of the processing.
- Whether the two interfaces share one page or have pages of their own is open (his words leave both); the default of the build is one app with a page for each, linked per subject and session (decision D37).

The census of public FreeSurfer derivatives, described below, is one use of the surface interface and the source of the paper's numbers. Where the census and the tool pull apart, the tool comes first.

## What this project is

`surfcensus` is a pip-installable Python package that scores FreeSurfer reconstructions quantitatively, renders animated three-plane views around the occipital pole, and serves those renderings to human raters through a web app built for thousands of brains.

It has two uses with one pipeline (see `docs/architecture.md`):

1. Any researcher points it at their own FreeSurfer subjects (`surfcensus review /path/to/SUBJECTS_DIR`) and rates them in the browser, offline, with no FreeSurfer install. This is a first-class use, not an afterthought.
2. The paper team runs a census over FreeSurfer outputs found in public OpenNeuro derivatives.

It is the surface-reconstruction arm of a paper on visualizing fMRI preprocessing quality. The paper's thesis is that people must look at their data. This package supplies the evidence for that thesis on the anatomical side: a census of how often FreeSurfer pial and white surfaces in shared public derivatives absorb the superior sagittal sinus, the transverse sinus, or the cerebellar tentorium near the occipital pole, and how often quantitative checks miss it.

The scientific claims that the census must support are:

1. Prevalence: the fraction of public FreeSurfer reconstructions with visible non-brain inclusion in occipital cortex, by dataset, field strength, FreeSurfer version, and T2/FLAIR-pial use.
2. Detectability: how well cortical-thickness outlier scores, computed per vertex in fsaverage space, predict rater judgments, both per subject and per location (raters click-mark errors). In particular, the rate of rater-identified failures among subjects that look normal on the numbers: the blind spot of quantitative QC.

Everything in the census serves one of these two numbers. The package as a whole serves the governing purpose above.

## The functional viewer: purpose and priority (Tomas, 2026-11-14)

Beside the package, `prototypes/func_viewer/` holds a viewer for the functional side of the same paper: the raw and the preprocessed EPI series of a run as moving pictures on the same slices, with the FreeSurfer contours. Its purpose is Tomas's statement of 2026-11-14, which governs every change there:

> "Well, let's be clear, the reason that we're showing both the raw and the fully pre-processed EPI time series as this video is exactly because we want to visualize the effects of pre-processing on the sort of movies, the time courses of the EPI volumes. And so it would be quite insane to perform all manner of processing on the raw data, because the whole idea is to show the difference between that and the pre-processed, and to also show how the pre-processed data now conform to, for example, the FreeSurfer surfaces, etc., etc., right? So it's meant to serve as a quality control type tool for both the raw data and the pre-processing pipeline at the same time. And so the raw and pre-processed visualizations should serve those purposes explicitly. I want you to record this and make it a full priority."

What follows from it (the reading of the building session; decision D36 in `docs/decisions.md`):

- The raw stream is the raw BIDS series with nothing done to it but one rigid placement per session into the anatomy (his decision of 2026-11-13: "Raw bids means single per session registration to the anatomy to ensure comparability across sessions, processing stages, and subjects"). No motion, slice-timing or distortion correction, and no other processing, is ever applied to it, also not in order to explain a difference.
- The preprocessed stream is the series as the pipeline wrote it. The page says which steps the pipeline applied to the run, so that a difference between the streams can be read as the effect of those steps.
- A difference between the streams, and the raw stream's misfit to the contours, is what the viewer exists to show. It is never treated as a confound to remove, to mark as a fault or to select against.
- Both uses are served explicitly and named in the page and the docs: quality control of the raw data (motion, artefacts in single volumes, distortion and dropout as acquired, movement between runs) and quality control of the preprocessing (what was corrected, what remains, how the preprocessed series conforms to the white and pial surfaces).
- The movie of a session is one continuous stream over all its runs, in acquisition order (his words of 2026-11-14, decision D38: "no skips between runs other than the ones imposed by the image content itself"): no stop or blank at a run boundary, the same slices in the anatomy, so that the only jumps are those of the data. The sessions of a subject follow each other in the same stream. Intensity normalization can be switched on (a window per run) and off (one window for the stream). The tool visualizes the files that exist: it never writes a NIfTI file or any other image volume and builds no common grid across runs or sessions; the packs in the run directory are a display cache, which he confirmed ("the display cache is perfect, just no new nifti files").
- "A full priority" is read as: work on the viewer comes before milestone 5 of the census, until Tomas says otherwise.

The hand-over, with what is built and measured, is `prototypes/func_viewer/README.md`. The rules of that folder hold: no media or data file and no dataset accession but `ds000102` in the repository, results by neutral id, read-only anonymous access.

## Status

Milestone 1 (`fsio` and `render`) is built and tested on synthetic subjects and on `ds000102` sub-01; `surfcensus render-subject SUBJECT_DIR --out DIR` works. Its exit check, the by-eye comparison with Freeview, is still to be done by Tomas. Milestone 2 (local source, config, catalog, inventory, `Run`) is built: `surfcensus inventory`, `surfcensus render` and `surfcensus status` work on a local directory and are resumable, and `sc.Run(...)` and `sc.sources.local(...)` are the Python API behind them. It is tested on synthetic directory trees and, end to end, on two real subjects linked into an fMRIPrep layout (`ds000102` sub-01, FreeSurfer 6.0.1; `ds003020` sub-UTS01, FreeSurfer 7.3.2). A census run can be created and opened from Python, with local sources only; no command creates one yet. Milestone 3 (template, registration, fsaverage maps, reference, clusters, ceiling and regional metrics, scoring, render schema 3) is built: `surfcensus metrics` computes phase 4 for a local run and is resumable, `sc.Run.metrics()`, `.scores()`, `.clusters()`, `.zmap()` and `.template()` are the Python API behind it, and renders carry fsaverage6 counterparts and the fsaverage pole when the run has a template. The template is built at run time from an `fsaverage` directory (a FreeSurfer install, the one next to the subjects, or `--fsaverage DIR`); nothing fsaverage-derived ships. Without a template, and for fewer than 30 subjects with a map, subjects are ranked as decision D16 says. It is tested on synthetic cohorts with planted thick patches and, end to end, on the two real subjects with a real `fsaverage` next to them and on Tier A and B of 34 real, unrated subjects (all of `ds000102` and `ds003020`). Not built or not done: the shipped reference (empty), the intensity metric (milestone 5), the comparison with `mri_surf2surf` (needs FreeSurfer), and any look at a scored subject by eye. Milestone 4 (rater app, `review`, `export`) is built for local runs: `surfcensus review PATH` inventories, scores, selects and renders the subjects of a directory and opens the rating app at the first render, with the others rendered behind it (decision D33); `surfcensus serve`, `finalize`, `calibration freeze` and `export` work, and `surfcensus.review.review(...)` is the Python API. The app has the rating screen (three panels from tiles, both playback modes, marks by click, the keyboard map, a skip key, a tour for first-time raters, a visible refusal of an incomplete rating), a review screen, a local queue by `primary_rank` or blind, final labels for one rater and, through `finalize`, for several, and an export of ten tables. It is tested on synthetic runs through the server and in a browser, and was driven end to end in Chromium on the two real subjects, where every stored mark equals a computation from the surface files (decisions D26, D28, D34). No human has rated anything with it: the time a rating takes and how raters use it are unknown until phase 0. The pieces of D33 that follow milestone 4 are built (decision D35): the app's packages are core dependencies, so the wheel installs everything `review` needs, and `tests/test_install.py` (`just test-install`, a CI job of its own) installs it into a fresh environment and rates a synthetic subject there; `review PATH` takes the root of a BIDS dataset and finds the FreeSurfer subjects below `derivatives/` itself, with the run directory beside the dataset; a DataLad or git-annex clone whose files are not fetched gets its subjects listed as incomplete with the files, their size and one command that fetches them, and a subdataset that is not installed gets the command that installs it, while the package fetches nothing. That was checked on synthetic trees, on the two real subjects linked into a made-up dataset root and run from an environment made from the wheel, and on local clones made with real git-annex and DataLad; no real OpenNeuro dataset root and no DataLad dataset from the network were used, and the package is not on PyPI (O1), so the one install line names the repository. The review of the quick start (2026-11-14) changed how the commands stop at SIGTERM and Ctrl-C (`stopping.py`: no lock in a handler, SIGTERM ignored from the first one on, `KeyboardInterrupt` during the preparation), what `review` does with several datasets in one folder (a run of this root is looked for first, `--relocate` is refused while the stored root still holds its subjects; the name of the default run directory is still O23), and three messages; a rare end by SIGINT after the closing line (5 of about 400 stress runs) turned out to be CPython 3.13 killing the process when a caught `KeyboardInterrupt` had passed through code run from a string, as scipy and dataclasses run while they are imported; `stopping.py` now clears that record before the command exits (`docs/decisions.md`, after D35). Not built: team mode (tokens, several raters at once, the calibration queue, adjudication, the dashboard, the error map), which is milestone 5, and `review openneuro:dsXXXXXX` (O20), which comes with the remote source of milestone 5. Milestones 5 and 6 are not started. The docs below are the specification. When the docs and the code disagree, the docs win until a decision in `docs/decisions.md` changes them; update the docs in the same change as the code.

## Build order

Build the local path first: it is the smallest thing that is useful on its own, and everything in the census reuses it. Each milestone ends with tests passing and a short note in `docs/decisions.md` for any choice made along the way.

1. `fsio` and `render`: read a FreeSurfer subject with nibabel, find the occipital pole, slice meshes, write per-panel WebP tiles in two layers (decision D11), a composite MP4 and poster, `render.json` and `vertex_map.npz`. Test on synthetic data, then on one real subject (fetch `ds000102` sub-01 Tier B and C files from the OpenNeuroDerivatives bucket into a pytest cache). Compare a few frames by eye with Freeview output from the gist.
2. `sources/local` and `inventory`: layout detection, file roles, run directory, catalog.
3. `templates`, `fsio/registration`, `metrics/fsaverage`, `metrics/clusters`, `metrics/regional`, `metrics/ceiling`, `metrics/scoring`: resampling to fsaverage6, within-set z-maps (shipped reference is empty at first), and the composite score that `review` sorts by (decisions D16, D18).
4. `app`: rating screen, keyboard map, click marks with coordinate round-trip, local mode, export. Then `surfcensus review` end to end.
5. Census: `privacy`, `remote`, `sources/openneuro_derivatives`, discovery, inventory, enrichment, selection, team mode of the app, Snakemake workflow.
6. Embedded-derivatives source (only if decision O9 says so), reference building, analysis notebooks.

Phase 0 of the census (pilot, `docs/phases.md`) can start as soon as milestone 4 works.

Before milestone 5 (decisions D36 and D37): the EPI movie interface is brought into the package in the three steps of `docs/epi_interface.md` (one session without a run directory, view only; then the run directory and `review`; then the subject pages, status and export). Step 1 is built (2026-11-14, decision D39): `surfcensus epi-session PATH --subject LABEL --out DIR [--serve]` packs the functional sessions of one subject of a local folder (fMRIPrep derivatives with `space-T1w` series, the raw BIDS series, the FreeSurfer subject) into a display cache and plays them in the browser as one continuous stream over all runs and sessions (D38), raw beside preprocessed with the white and pial contours, with a normalization toggle. On the worked example (`ds000102` sub-01) its decoded streams equal the prototype's pack and the page's address is printed 34 s after the command; two independent reviews recomputed the raw and preprocessed streams, the contours and the frame check without the package's code (0 differing voxels on the worked example and on variants staged from it) and drove the page on real packs (contiguous at 60 volumes per second through every boundary, canvas values exact), and their findings are fixed: one anchor per session whatever is packed, packs kept on a second call, nothing on the page moving at a boundary, the pictures first. The real multi-run sessions of stage 1 were no longer on disk, so real flagged runs, real distortion correction and a real second session were not played; no person has looked at the page. Steps 2 and 3 are not done. The defaults of the spec wait for Tomas's answers (its section "Decisions for Tomas").

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
- Icosahedral nesting holds for the vertices of fsaverage5: `fsaverage5/surf/lh.sphere.reg` equals the first 10,242 vertices of fsaverage `lh.sphere.reg` exactly. The faces of a coarser mesh are not a subset of the finer faces; they follow by coarsening (see 2026-11-13 below). The folder has no `fsaverage6/`, so the fsaverage6 mesh is derived, not read: vertices by truncation to 40,962, faces by coarsening once.
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

Checked 2026-11-13 on the left hemisphere of the `fsaverage` next to `ds000102` (five files, read with nibabel 5.5.0; the right hemisphere was not read):

- `label/lh.cortex.label` is text: a comment line, a line with the count, then one line per vertex (index, three coordinates, a value). It lists 149,955 vertices, not in ascending order; 37,487 of them are below 40,962.
- `label/lh.aparc.annot` has 36 names; 13,887 vertices have no label and 2 are labelled `unknown`. The vertices with a label are exactly those of the cortex label.
- `surf/lh.white.avg.area.mgh` is an MGH volume of shape (163842, 1, 1), big-endian float32. Its values sum to 82,219.9 mm2, and to 76,468 mm2 over the cortex label.
- Every vertex with index 40,962 or more has exactly two neighbours with a lower index, and 81,920 faces are made of three such vertices. Replacing each of those faces by the three shared neighbours gives 81,920 faces over the first 40,962 vertices that are all wound outward and form a closed surface with 12 vertices of degree 5 and 40,950 of degree 6. So the fsaverage6 vertices are the first 40,962 of fsaverage, and its faces follow from the fsaverage faces by this coarsening, not by truncating the face list. No `fsaverage6` directory was compared.
- The cortex vertex with the smallest y on `surf/lh.white` is 12950 (y = -102.5 mm). On `lh.sphere.reg` the occipital parcels lie around -y and the medial wall towards +x.

Checked 2026-11-13 on the `fsaverage` next to the subjects of `ds000102`, on the two real subjects and on the copy of `fsaverage` next to `ds003020`:

- Coarsening the 327,680 faces of fsaverage `lh.sphere.reg` once gives 81,920 faces on the first 40,962 vertices (every vertex of degree 6 but 12 of degree 5), and coarsening those once more gives the faces of `fsaverage5/surf/lh.sphere.reg` exactly.
- On fsaverage6, `lh` and `rh`: `cortex.label` covers 37,487 and 37,482 vertices (the label lists 149,955 `lh` rows, unsorted); the cortex vertex with the smallest white y is 12950 and 5558; the search region of `metrics.md` has 4,856 and 4,715 vertices, the control region 5,789 and 5,804. The `lh` annotation has 36 names; 2 vertices inside `lh.cortex.label` are labelled `unknown`.
- `surf/?h.white.avg.area.mgh` is an MGH volume of shape (163842, 1, 1), float32, and sums to 82,220 (`lh`) and 82,168 mm2 (`rh`); the area of the fsaverage6 white mesh is 65,096 and 64,692 mm2, 0.77 of it over the `lh` cortex and 0.59 in lateraloccipital.
- `fsaverage/surf/lh.sphere.reg` and `rh.sphere.reg` are byte-identical (5,898,546 bytes). No face of the `sphere.reg` of sub-01 or of fsaverage is wound inward, in either hemisphere.
- The copy of `fsaverage` next to `ds003020` (FreeSurfer 7.3.2) has byte-identical `?h.sphere.reg`, `?h.cortex.label`, `?h.aparc.annot` and `?h.white.avg.area.mgh`. Its `?h.white` differs in the creation line (2018 against 2013) and in the position of vertex 40969 (by 0.07 and 0.11 mm), which is not on fsaverage6. Both carry the build stamp `freesurfer-x86_64-redhat-linux-gnu-dev4-20090216`.
- `stats/?h.aparc.stats` has 34 data rows of 10 fields in both subjects. The FreeSurfer 6.0.1 files have 10 `# Measure` lines, one without the comma after its name (`# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3`); the 7.3.2 files, written with `-noglobal`, have 3 (`NumVert`, `WhiteSurfArea`, `MeanThickness`). On `ds000102` sub-01 `lh`: `MeanThickness` 2.39047, `NumVert` 140,304. `stats/aseg.stats` has `lhSurfaceHoles` and `rhSurfaceHoles` in both: 41 and 26, and 4 and 15, the numbers of the logs.
- The occipital-to-control thickness ratio (area-weighted `ThickAvg`, hemispheres pooled) is 0.811 for `ds000102` sub-01 and 0.828 for `ds003020` sub-UTS01.
- `surf/?h.thickness` is stored as big-endian float32, 15 + 4n bytes. sub-01 `lh` has 148,332 values: 16 at exactly 5.0, 28 at 4.9 or more, 5,919 zeros (`rh`: 146,947, 41, 61 and 5,875). In the `lh` annotation of sub-01, 8,027 vertices have no label and none is labelled `unknown` or `corpuscallosum`; the four occipital parcels hold 16,429 vertices.
- Mapping sub-01 onto fsaverage6 with 8 nearest neighbours: 3 (`lh`) and 2 (`rh`) of 40,962 targets lie in no tested face and take the nearest vertex, none with 16 neighbours; an exhaustive search over all faces agrees on 2,000 random targets per hemisphere. 37,420 of the 37,487 `lh` cortex vertices get a value; the `lh` map has mean 2.425 mm and global median 2.461 mm. The `lh` native search region has 21,484 vertices, none at 4.9 mm or more and none at 0. One map takes 0.4 s.
- A render of sub-01 with the template: every vertex of the vertex map has an fsaverage6 counterpart, and the 352 tiles and the poster are byte-identical to a render without it. The fsaverage pole lies 1.2 mm (`lh`) and 6.3 mm (`rh`) from the geometric pole; in `ds003020` sub-UTS01, whose `sphere.reg` files map as well, 8.6 and 2.0 mm.

Measured 2026-11-13 on Tier A and B of 34 unrated subjects (all 26 of `ds000102`, FreeSurfer 6.0.1, and all 8 of `ds003020`, 7.3.2; 583 MB fetched anonymously), as two local fMRIPrep trees in one run with the `fsaverage` next to them; details in `docs/metrics.md`:

- Every hemisphere got a map: 0 to 7 fallbacks in 67 hemispheres and 26 in one, at least 99.5% of the template cortex covered. Each of the 103 fallback targets lies in a face that a search over more neighbours finds (22 are left with 16, none with 256); at the 66 in the template cortex the nearest-vertex thickness is off by a median of 0.33 mm and at most 2.03 mm; 3 lie in the search region, none beyond z 3. The maximum of every one of the 68 thickness files is exactly 5.0. Holes in the log and in `aseg.stats` agree in all 34 subjects, and every `?h.aparc.stats` has 34 rows.
- The mean of the fsaverage6 map over a parcel, weighted by vertex area, against `ThickAvg` of the stats file (2,312 parcels): correlation 0.983, median difference 0.009 mm, median absolute difference 0.024 mm; the parcel means differ by more than 0.07 mm only in insula (0.14) and isthmuscingulate (0.20).
- With all 34 in one reference: 0.0045 of the search-region vertices and 0.0026 of the control-region vertices have z above 3. The search regions hold 5 vertices at 4.9 mm or more and 16 at exactly 0, in 12 subjects. The regional ratio is 0.752 to 0.868.
- One process, local disk: inventory 0.7 s for the 34, the maps 0.84 s per subject, reference, clusters and scores 0.5 s for the run; 12 s in all with four worker processes, with every table, map and reference file byte-identical to the run with one. 164 kB per subject for the maps.

Read in source on 2026-11-12 (FreeSurfer and Freeview on GitHub, nibabel 5.5.0 as installed); none of it checked against a running Freeview, and against FreeSurfer 7 data only where the list above says so:

- Freeview's default 2D views are radiological in coronal and axial (subject's left on the right), anterior up in axial, and in sagittal show posterior at the left and anterior at the right (`RenderView2D.cpp`: sagittal camera at +x, view-up +z).
- Freeview treats an integer voxel index as the voxel centre and cuts the surface with the plane through that centre; no half-voxel offset between image and contour in the code path.
- `nibabel.freesurfer.read_geometry` returns coordinates as stored (tkr RAS) and does not apply the footer's `c_ras`. `MGHHeader.get_vox2ras_tkr()` ignores the direction cosines.
- From FreeSurfer 7.0.0, recon-all writes `?h.pial.T1` and makes `?h.pial` a relative link to it (or to `?h.pial.T2` / `?h.pial.FLAIR`).
- `MRISmeasureCorticalThickness` clamps each of its two distance measurements at the maximum and stores their mean, in v6.0.0, v7.0.0 and v7.4.1; recon-all in FreeSurfer 7 calls it through `mris_place_surface --thickness` with a maximum of 5.
- Lossy WebP is always 4:2:0; Pillow's lossless WebP zeroes RGB under alpha 0 unless `exact=True`.
- `mris_anatomical_stats` at v6.0.0 and v7.4.1 (read 2026-11-13) prints a parcel row as `%-40s%5d  %5.0f  %5.0f  %5.3f %5.3f ...`: vertex count, area and volume are integers in the file. A parcel without vertices, `unknown` and `corpuscallosum` get no row. The measure line without its comma is in v6.0.0 and corrected in v7.4.1.
- `mri_segstats` at v5.3.0, v6.0.0 and v7.4.1 (read 2026-11-13) writes the hole counts only with `--euler`, as `1 - euler / 2`.
- `scripts/recon-all` at v6.0.0 and v7.4.1: `-hemi lh` sets the log name to `recon-all-lh.log`. T2pial is set by `-T2pial`, `-t2pial`, `-T2pial-only` and `-autorecon3-T2pial` and cleared by `-noT2pial`, `-no-T2pial`, `-no_T2pial`, `-no-t2pial` and `-no_t2pial`; FLAIRpial likewise. The argument parser compares whole arguments, case-sensitively. There is no flag that clears `-hires`. `-3T` and `-3t` are the only flags that start with a digit. `-autorecon2-pial` is in neither version.

Measured 2026-11-12 on a full render of `ds000102` sub-01: 9 s on one core; 10.9 MB of media (tiles 4.1 MB, `anim.mp4` 4.3 MB, `vertex_map.npz` 2.5 MB since it follows the tiles and not the box, 1.8 MB before); 155 of 176 tiles contain contours.

Measured 2026-11-12 on a local run over the two real subjects (one machine, files on a local disk): `surfcensus inventory` under 0.1 s for the phase, with the logs read whole; `surfcensus render --jobs 2` 9.0 s without video; `surfcensus render --video` 17.5 s in one process; a second call of either does nothing in under 0.1 s; `catalog.sqlite` 64 kB; the source files unchanged in size and modification time. The media of `ds000102` sub-01 rendered through a run with video are byte-identical, file for file, to the render that `render-subject` made of it at the end of milestone 1 (same machine).

Measured 2026-11-13 with the rater app of milestone 4 (one machine, 4 processors, local disk, Chromium 153 headless on the same machine; details in `docs/rater_app.md`). No human rated anything: every rating was made by a script that drives the browser.

- A local fMRIPrep tree of `ds000102` sub-01 and `ds003020` sub-UTS01 with the real `fsaverage` next to them: `surfcensus review PATH --no-browser` reaches the running app in 11.0, 11.2 and 11.8 s (three new run directories; the first render is about 7 s of it), and in 0.8 s when the run is finished. One real subject through `review(..., serve=False)`: 9.7 s, then 0.1 s.
- The browser loads 352 tiles per subject in 352 requests, 4.06 MB (sub-01) and 3.77 MB (sub-UTS01), plus one answer of about 10 kB with the geometry; the page itself is 12 requests and 138 kB. First subject with every tile loaded 3.2 s after opening the page; next subject 0.13 s after `Enter` (it is fetched ahead). No request to another host, no cookie.
- Nine marks placed by clicks on drawn pial and white contours in all three panels (1400 by 900 px, 2x zoom, and 900 by 700 px at a device pixel ratio of 2) equal a computation from the T1 header, the surface files and the `sphere.reg` files with nibabel and numpy alone: tkr and scanner RAS within 1e-9 mm, hemisphere, pial and white vertex with distances, fsaverage6 vertex. The stored points lie 0.00 to 0.18 mm from the clicked surface. A browser reports a click by the corner of its screen pixel; taking the middle removed a shift of half a screen pixel up and to the left (0.05 to 0.65 tile pixels on four marks before the change).
- Package 4.A, same day, `ds000102` sub-01 at 320 px in both display orientations: for a click at the centre of every opaque pial contour pixel (231,079 and 231,036 pixels) the resolved hemisphere, vertices, distances and fsaverage6 vertex equal those from the full surfaces; 0.305% of those clicks resolve to a vertex that is not `visible`.
- `ms_on_subject` of three submissions and one skip equals the time the subject was on screen in that visit within 12 ms; the time the tour was open is not in it.
- `review` on the 26 subjects of `ds000102` with Tier A and B only: 35 s, every subject `missing surface`, exit code 1. On a synthetic set of 32 the queue is the partial-score block, then `primary_rank`; blind, the seeded shuffle.
- When the milestone was closed, the same tree through the commands with two raters one after the other (`review --rater`, then `serve --rater`): `finalize` gave one subject by consensus and one open, the six exported marks equal a computation from the surface files, and the tiles, poster and video of `ds000102` sub-01 are byte-identical to the render of milestone 2 (all 356 files to that of milestone 3). The 34 subjects with Tier A and B as one run gave the numbers of `docs/metrics.md` again and 34 unratable labels (`docs/decisions.md`, after D34).

Measured 2026-11-14 for the quick-start follow-up (one shared 4-core Linux machine, Python 3.13; details in `docs/decisions.md`, D35):

- An environment made with `uv venv` from the wheel of the tree (packages from uv's cache), commands run from outside the repository with `PYTHONPATH` unset. `surfcensus review ROOT --no-browser` on a made-up BIDS dataset root whose `derivatives/fmriprep/sourcedata/freesurfer/` links to `ds000102` sub-01, `ds003020` sub-UTS01 and the real `fsaverage`: the run directory beside the dataset, `Found 2 subjects in derivatives/fmriprep/sourcedata/freesurfer (fMRIPrep layout, FreeSurfer 6.0.1, 7.3.2).`, the app 16.9 s after the start by the command's clock in a new run directory (the first imports of a fresh uv environment compile the packages) and 1.0 s on the finished run. The dataset got no file; the cached source files kept their sizes and modification times. `surfcensus --help` took 0.2 s there and loaded none of `fastapi`, `sqlmodel`, `uvicorn`, `matplotlib` and `nibabel`.
- With git-annex 10.20261026 and DataLad 1.8.0 on local clones of a DataLad copy of that root (`derivatives/` a subdataset, the real files copied in): a fresh clone ends with `layout` and `datalad -C <clone> get -n derivatives`, exit code 2; after `get -n`, both subjects are named with 19 files each (43.4 and 36.8 MB from the keys) and one fetch line, exit code 1; with the six Tier C files of sub-01 dropped, the app offers sub-UTS01 alone, the printed `datalad -C ... get` fetched the six files, and the next `review` rendered both. No clone changed through `review`, `status` and `export`.
- pip 26.2.2: with `jinja2` removed from an installed environment, `serve` and `review` print one line naming it and end with 2, and `pip install surfcensus` keeps the installed package and installs `jinja2` again.

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
just test-browser     # pytest -m browser: the pages in a real browser (SURFCENSUS_BROWSER, else Chrome); not in CI
just test-install     # pytest -m install: the wheel in a fresh environment, review up to a rating; needs uv and the package index
just lint             # ruff check + ruff format --check + pyrefly
surfcensus render-subject SUBJECT_DIR --out DIR   # one subject, no run directory
surfcensus epi-session PATH --subject LABEL --out DIR [--session LABEL]... [--raw DIR] [--serve] [--force]
                                                     # the EPI movies of one subject: packs, and the page (step 1, no run directory)
surfcensus inventory --run-dir DIR [--source PATH]   # phase 2; --source creates a local run
surfcensus metrics --run-dir DIR [--jobs J] [--fsaverage DIR]   # phase 4
surfcensus render --run-dir DIR [--jobs J]           # phase 6
surfcensus status --run-dir DIR [--failures]         # where a run stands
surfcensus review PATH [--run-dir DIR] [--rater NAME] [--blind] [--no-browser] [--no-serve] [--serve-after N]
                                                     # local data (also a BIDS dataset root): inventory, metrics, render, rater app
surfcensus serve --run-dir DIR [--rater NAME]        # the rater app on a rendered run (local mode)
surfcensus finalize --run-dir DIR                    # final labels from the ratings of several local raters
surfcensus calibration freeze --run-dir DIR          # calibration.yaml from the final labels
surfcensus export --run-dir DIR [--format csv|parquet]   # phase 8, tables for analysis
```

Planned:

```
surfcensus discover   # phase 1
surfcensus enrich     # phase 3
surfcensus select     # phase 5
surfcensus serve --mode team   # phase 7 for a census: tokens, several raters, adjudication
surfcensus export --release | --reveal   # what a census publishes; the pseudonym mapping
```

## Detailed specs

@docs/architecture.md
@docs/data_sources.md
@docs/phases.md
@docs/metrics.md
@docs/rendering.md
@docs/rater_app.md
@docs/epi_interface.md
@docs/rating_protocol.md
@docs/analysis_plan.md
@docs/decisions.md
@docs/prior_art.md
