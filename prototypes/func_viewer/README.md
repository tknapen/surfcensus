# Functional viewer prototype

Hand-over of a feasibility study made on 2026-11-13 and of stage 1 of the plan that followed it, built on 2026-11-13 and 2026-11-14. It is written for a reader who has none of the files or the context of the sessions that made them.

## What this folder is

`surfcensus` scores, renders and rates FreeSurfer surfaces. The paper it belongs to argues that people must look at their data, and Tomas Knapen asked whether the same could be shown for functional data: the EPI time series before preprocessing and after fMRIPrep, as moving pictures over the anatomy. A feasibility study answered with a prototype on one subject, a survey of what the public derivatives hold, a search for prior art and three critiques. `REPORT.md` is the report of that study as Tomas received it. This file records what he decided, what the scripts do, what was measured, what stage 1 built and found, and what comes next.

The folder is not part of the package. Nothing in `src/` imports it, it is in neither the wheel nor the sdist, pytest does not collect it, and pyrefly does not check it (the last three by configuration, see "Rules that bind this work"; that nothing imports it is a fact of the code). The scripts import four things from the package and change nothing in it: `surfcensus.render.slicing.MeshSlicer`, `surfcensus.render.frames.ContourRenderer` and `blend`, and `surfcensus.render.encode.write_mp4` (the playback measurements also use `ffmpeg_exe`, `write_gray_webp` and `write_rgba_webp` of the same module).

No image, video or table of the study is in the repository, and no dataset is named but `ds000102`, whose `sub-01` is the worked example here as it is in the package's tests. The reasons are under "Rules that bind this work".

## Decisions

Tomas Knapen, 2026-11-13, in answer to the four questions at the end of `REPORT.md`, with one remark added. His words are quoted; the rest is how they are read here.

1. "viewer". The functional part is a viewer for the opinion paper. It is not a functional census. A census would be a separate, later decision.
2. "i don't - your option sounds like a good strategy". He has no functional subjects to name whose preprocessing is known to have gone wrong. Hard cases are therefore picked as the highest-motion runs, from the confounds files of the derivatives that have them.
3. "limit - i'm not running anyone's fmriprep." With his opening line, "the fmri part, which i would want to render pre-preprocessing (ras bids) and fmriprep preprocessed", this means two streams: the series before preprocessing and the series as fMRIPrep preprocessed it. They are rendered only for derivatives that hold a preprocessed series. No stream is rebuilt from stored transforms, and nobody's fMRIPrep is run. "ras bids" is read here as raw BIDS, the series as the raw dataset shares it. That reading was his open question 1; decision 6 answers it.
4. "yeah please do so that another session can pick it up." The prototype is committed here, under `prototypes/func_viewer/`, outside the package and the wheel.
5. "i also think that the speed of the video can be much much faster - even 60fps is fine as long as the interface allows scrubbing". The 12 volumes per second of the study are no longer the target. The requirement is fast playback, up to 60 volumes per second, together with scrubbing that lands on the exact volume.
6. Tomas Knapen, 2026-11-13, in answer to open question 1: "Raw bids means single per session registration to the anatomy to ensure comparability across sessions, processing stages, and subjects. Please continue with that too". How it is read here (the reading is that of the session that built stage 1, not his words): the stream before preprocessing is the raw BIDS series, resampled once into the anatomy (the display grid in fMRIPrep's T1w space), through ONE rigid registration per session that is the same for every run and every volume of the session. So what moves within a session, within a run and between runs stays visible, while sessions, the two streams and subjects are seen in one anatomical frame. No motion, slice-timing or distortion correction. The registration is the stored `from-scanner_to-T1w` transform of the session's anchor run; nothing is registered anew. The anchor rule was fixed before any image was looked at: the run acquired first, by `AcquisitionTime` of the raw JSON sidecars when every run of the session has one, else the first in BIDS name order (task by name, run index as a number); which of the two applied is recorded. A dataset without `ses-` folders has one session per subject. That the runs of a session share the scanner's frame of reference is checked, not assumed (stage 1 below, "One registration per session").

What decision 3 leaves, by the survey of 2026-11-13 (anonymous listing of the 308 dataset prefixes under `fmriprep/` in the OpenNeuroDerivatives bucket; listed again the same day after the port, with the same counts):

- 72 derivatives name fMRIPrep 23.0.2 or older in their `dataset_description.json` (versions 20.2.3 to 23.0.2). 236 name 23.2.0 or newer (up to 25.2.3).
- 71 of the 72 show a preprocessed series in T1w space (`space-T1w_desc-preproc_bold.nii.gz`). The same 71 show a confounds table and a `from-scanner_to-T1w` transform. The one other has a preprocessed series in BOLD space only, with no transform and no confounds. None of the 236 shows a preprocessed series in any space.
- "Shows" means: seen among the first one to three listing pages of the dataset's subject folders (1,000 keys each, in key order from the first subject on; `survey/layout_all.py` stops after the page on which it has counted 20 keys of a `func/` folder, and after the third page at the latest), or, for three datasets whose flags fitted neither layout, in a full listing of the first subject. It is a statement about the first subjects of a dataset, not about every subject. In a separate check of 12 other datasets, one of the older group had a first subject without the T1w-space series, transform and confounds and a second subject with all of them.

So the two streams can be shown for at most 71 of the 308 derivatives as they stood on 2026-11-13, all of them processed with fMRIPrep 23.0.2 or older, and which subjects and runs of those qualify has to be listed before anything is chosen (stage 1 below). The count is 71 and not 72 because the criterion used here is the T1w-space series. The 72nd derivative does hold a preprocessed series, in BOLD space, but it shows no transform to T1w space, so that series could not be put under the surfaces, and no confounds table, so its runs could not be ranked by motion (decision 2).

## What the prototype does

For one subject (`ds000102` `sub-01`: two runs of 146 volumes, TR 2 s, voxels of 3 x 3 x 4 mm, fMRIPrep 21.0.2) it draws three orthogonal planes of every volume with the FreeSurfer white (red) and pial (yellow) surfaces as contours, and writes one video frame per volume. Three videos: the raw stream (A), the preprocessed stream (B), and both side by side over the traces of framewise displacement and standardised DVARS from fMRIPrep's confounds file, with a cursor, a red frame on volumes with framewise displacement above 0.5 mm and a coloured bar on the first six volumes of a run (C).

The two streams:

- Preprocessed: `space-T1w_desc-preproc_bold.nii.gz` of the derivative, as it is. Its grid is the display grid: 49 x 62 x 36 voxels of 3 x 3 x 4 mm, axis-aligned in the world space of fMRIPrep's T1w image.
- Raw: every volume of the raw BIDS series is resampled once onto that same grid. The point map is the run's `from-scanner_to-T1w_mode-image_xfm.txt`, read as an ITK affine in LPS that takes a point of the T1w image (fixed) to the point of the scanner image (moving), and converted to RAS. It is one rigid transform per run, the same for all its volumes. The interpolation is a cubic B-spline (`scipy.ndimage.affine_transform`, order 3); a point outside the raw field of view is 0. No motion correction, no slice-timing correction and no distortion correction is applied.

So the raw stream as shown is the raw series after one rigid resampling into the anatomy. It is not the raw series on its own grid. In the feasibility study the transform was each run's own; stage 1 uses one per session (decision 6). That choice puts both streams on the same slices, pixel for pixel, at the price of one interpolation that the raw data never had, and part of the measured difference between the streams may be that interpolation (the two were not separated). The alternative is to show the raw series on the grid it was acquired on (64 x 64 x 40 voxels of 3 x 3 x 4 mm in this subject, axis-aligned) and to move the surfaces into scanner space with the same transform, which costs no resampling of the data because surfaces are points. The streams would then lie on different slices, and a series acquired obliquely would need oblique cuts, which `pipeline/epiview.py` does not do. Decision 6 settled the choice: resampled into the anatomy, with one registration per session.

Surfaces: FreeSurfer's tkr RAS is taken to scanner RAS through `mri/T1.mgz` (vox2ras times the inverse of vox2ras-tkr) and then to T1w space with the point map of `from-T1w_to-fsnative_mode-image_xfm.txt`. Moving points from fsnative to T1w needs the file that is named the other way round, for the reason given in `pipeline/geom.py`.

Display: fixed slices (sagittal 12 mm off the centre of the brain mask, coronal and axial near it), 2 screen pixels per mm, nearest neighbour, so a voxel is a block. One intensity window per run for both streams: 0 to the 99.5th percentile of the preprocessed series inside the brain mask. The videos are H.264 from the package's writer at its default CRF 23.

## What was measured

Everything in this section is from 2026-11-13. Unless another subject is named, it is the worked example, one subject with little motion. The pipeline was run again after the port and gave the same bytes for all 13 videos and 9 check images. Of its eight text and JSON results, six are identical; `tile_sizes.txt` differs in the one time it reports, which is measured anew at each run, and `check_videos.txt` in the key-frame counts that the port appends to each line.

### Geometry

- The temporal mean of the raw stream correlates with the mean of fMRIPrep's T1w-space series at r = 0.9993 inside the brain mask, in each run (0.9998 over all voxels). An extra shift of 0.25 mm along one axis lowers it to between 0.995 and 0.999, 1 mm to between 0.952 and 0.981, 3 mm to between 0.74 and 0.82. With the transform inverted no voxel of the grid is non-zero. With no transform at all, r against fMRIPrep's reference image is 0.08 in the mask.
- Against `space-T1w_boldref` the same resampling gives only 0.905 and 0.918, because that reference image is not the mean of the series (the two correlate at 0.91 and 0.92). A test of the geometry should use the series mean.
- Contrast across the white surface, sampled 1 mm inside and outside along the normal on the T1w image: 42.3% of the mean at the stored position, 33 to 36% with the surfaces shifted by 1 mm. On the EPI reference image, 2 mm either side: -8.0% (gray brighter than white) at the stored position, -3.2 to -4.8% with a shift of 3 mm, about 0 with 6 mm.
- The fsnative transform of this subject is nearly the identity, so its direction cannot be shown here: the two readings differ by at most 0.30 mm. On the second subject (below) FreeSurfer's `lh.white` taken through the reading used here equals fMRIPrep's own `hemi-L_white.surf.gii` to 0.0000 mm, and the other reading is off by 0.19 mm.

### What the frames show and do not show

- The two streams correlate at 0.9988 in the median volume (lowest 0.985, run 1) and differ on screen by 1.8 gray levels of 255 on average. The RMS difference per volume is 1.4% of the brain median in the median volume and follows the displacement from the reference at r = 0.98.
- The subject barely moved: the largest translation is 0.69 mm, and framewise displacement exceeds 0.5 mm in 3 of 292 volumes (0.63, 0.55 and 0.77 mm, all in run 1). So this subject cannot say whether motion is visible in a moving picture.
- An agent read five side-by-side frames as images, among them those at 0.63 and 0.77 mm, and could not tell the two halves apart. It also read a difference image and a temporal-SD map, where the motion shows as a rim at the edge of the brain. No person has looked at any of these.
- Static content that the frames do carry: 3.9% of the voxels wholly inside the T1w brain mask have a mean below 40% of the brain median (93% of them more than 15 mm below the centre), and the EPI outline lies a median of 3 mm, one voxel, beyond the T1w mask anteriorly. fMRIPrep's report for these runs says that no distortion correction was applied.

### Videos, compression and cost

| Video, 292 volumes, CRF 23 | Size at 12 per s | Size at 24 per s | Time, one core |
|---|---|---|---|
| A, raw, 960 x 434 px | 1.44 MB | 0.92 MB | 17.6 s |
| B, preprocessed, 960 x 434 px | 1.49 MB | 0.95 MB | 7.1 s |
| C, side by side with traces, 1920 x 612 px | 2.93 MB | 1.89 MB | 22.1 s |

Times are whole calls after the port (the study measured 17.8, 7.3 and 24.0 s), on a 4-core machine on which another job ran. Every call pins itself to the first CPU it may use, so these times hold only while nothing else uses that CPU: in a review of this hand-over the two shell scripts were started at the same time, and video C then took 50.2 s, with the same bytes. About 12 to 15 s of video C is the resampling of the raw volumes. Peak memory was 0.42 GB for Python and 0.15 GB for ffmpeg.

- H.264 changes what is being inspected. Decoded frames against the frames they were made from (run 1, raw stream, no contours, head pixels): the frame-to-frame change is 2.94 gray levels in the source. At 12 per second, CRF 23 keeps 1.38 of it (r = 0.66 with the source change, 3.08 MB) and CRF 12 keeps 2.59 (r = 0.94, 12.0 MB).
- The download for this subject is 133 MB in 29 files (two raw runs 56 MB, two T1w-space runs 59 MB, the T1w 17 MB), 5.5 s at 4 requests at a time, plus 27 MB for the five FreeSurfer files.
- Sizes elsewhere, from listings of the first subject of 30 evenly sampled datasets (6 of them with a preprocessed series): raw series plus T1w-space series, which is what decision 3 needs, are 54 to 507 MB per run as dataset medians (median 108 MB) and 0.11 to 19.2 GB per first subject (median 0.27 GB). Six datasets are too few for a total. First subjects have 1 to 79 runs (median 4). A confounds table is 0.3 to 3.4 MB per run.
- Length of an experiment in volumes: 292 in the worked example (counted by the pipeline), and 900, 1,704 and 1,776 for the first subjects of three other derivatives, which `survey/headers.py` read from the headers of their T1w-space series. The first subject of a fourth derivative has 1,400 volumes in the first 8 of its 40 runs; its total was not read.

### Fast playback and scrubbing

Measured for decision 5 with `pipeline/run_playback.sh`, on one core. Video C of the worked example at 60 volumes per second lasts 4.87 s for its 292 volumes.

| Side-by-side video at 60 per s | Key frames | Size | Whole call, s (CPU s of ffmpeg) | r with source change |
|---|---|---|---|---|
| CRF 23, as the package writes it | 2 of 292 | 0.91 MB | 22.5 s (4.3 s) | |
| CRF 23, every frame a key frame | 292 | 13.47 MB | 22.3 s (4.9 s) | |
| the same two without contours | 2, 292 | 0.94, 9.32 MB | | 0.36, 0.61 |
| CRF 12 without contours | 2, 292 | 6.72, 19.92 MB | | 0.86, 0.89 |

The last column is the comparison of the paragraph above: how much of the frame-to-frame change of the source the decoded video has.

| One image per volume and panel | Tiles | Size |
|---|---|---|
| WebP quality 80, the package's tile writer | 1,752 | 7.85 MB (4.48 kB each), 1.93 to 1.99 MB per run and stream |
| WebP lossless | 1,752 | 6.1 MB (3.46 kB each; from every fifth volume) |
| contour layer, lossless, once per panel | 3 | 66 kB |
| for comparison: all 36 slices of a volume as one mosaic, native voxels | 1 per volume | 74 kB lossless, 22 kB at quality 80: 10.8 or 3.2 MB per run and stream |

The 1,752 tiles are three panels (372 x 288, 294 x 288 and 294 x 372 px) of 292 volumes in two streams. Writing them at quality 80 took 12.1 s.

What these numbers mean, without a choice being made here:

- A video with the key frames x264 sets is the smallest by far. To show an arbitrary frame, a decoder has to start at the key frame before it: on average 109.5 frames earlier, at most 249. At 60 per second and CRF 23 it also keeps less of the frame-to-frame change than at 12 per second (r = 0.36 against 0.66). The same frames at the same CRF take 2.93 MB at 12 per second, 1.89 MB at 24 and 0.91 MB at 60, so the encoder spends fewer bits on a frame at a higher rate; why was not looked up.
- With every frame a key frame, any frame can be decoded alone. The file is about 15 times larger at CRF 23 (13.47 against 0.91 MB, with contours). The comparison with the source was made on the files without contours only (9.32 against 0.94 MB): there the decoded frames of the all-key-frame file change more from one to the next than the source does (3.97 against 2.94 gray levels, head pixels, run 1 of the raw stream). That is read here as coding noise that differs between frames; the reading was not tested.
- One image per volume and panel is how the surface rater app plays its slice tiles from a frame table. Any volume is one image, at any speed the page chooses, and the lossless tiles hold the pixels exactly. For these block-like tiles lossless is smaller than quality 80. The cost is in files and requests: 6 images per volume for two streams, so 1,752 for this subject and 10,656 for an experiment of 1,776 volumes, and at 60 volumes per second a page has to show 360 images per second.
- A mosaic per volume holds every slice, so it also allows scrubbing through slices, with one image per volume and stream.

Not measured: anything in a browser. Whether a video element lands on the exact frame when it is sought, how fast it scrubs either kind of file, and whether a page can fetch and draw 360 tiles per second are all open. No person has played any of these files, so nothing is claimed about how 60 volumes per second looks.

### The second subject

One run of one subject of a derivative made with fMRIPrep 23.2.0 (125 volumes, 3 x 3 x 3.5 mm) was fetched to try the newer layout, which holds reference images and transforms and no preprocessed series. Its `fmriprep.toml` says `level = "minimal"`. After decision 3 that layout is out of scope, but three results carry over:

- the direction of the fsnative transform, given above;
- the gray/white contrast of the EPI reference across the white surface is -10.7% with the stored coregistration, -7.5% with the headers alone and -2.8% with the transform inverted, and falls to between -4.1 and -4.9% with a shift of 3 mm;
- in two stills, the row with the stored coregistration and the row with the headers alone looked alike to the agent that made them. The same stills show the eyes, since EPI is not defaced.

The dataset is not named here. It was the dataset at index 123 of the 308 prefixes in key order on 2026-11-13, and of it the first subject and that subject's first functional run, "first" meaning first in key order (the first subject folder; in its listing, the first key under `func/` with a task label, which gives the task, the session and the run). `survey/survey.py` writes the prefixes and the listing and prints the indices of its sample, which includes 123, and `minimal_layout/first_run.py` reads the labels from that listing and prints them as the arguments of `fetch_min.py`. The index holds only while the bucket has the prefixes it had on 2026-11-13. `first_run.py` therefore also prints what to compare: on 2026-11-13 the listing had 308 prefixes, and the dataset at index 123 named fMRIPrep 23.2.0 and had 48 subject folders. If these differ, the index may point at another dataset.

### What the critiques found

Three reviews of the proposal, by agents with separate briefs. The points that bear on the next step:

- A moving picture has not been shown to add anything to stills. Everything seen in the frames of the worked example is static and would be in a mean image. The catches by eye that the prior-art search found in the literature (flipped images, dropout, distortion, misalignment) were visible in stills, and the search found no study that compares movies with stills. The search could not open several sites and lists what it took from summaries only.
- Coregistration may not be judgeable by eye at 3 mm voxels. Besides the two stills above there is one blind trial (`minimal_layout/blind_shift.py`): of four shuffled rows, an agent took the row shifted by 3 mm in z for the correct one. That is one trial by a non-expert and settles nothing, but it argues for testing detection of known shifts before any claim that eyes see what a number misses.
- Agreement between raters on functional QC is moderate in the literature the search read: pairwise kappa 0.44 to 0.60 among four raters in Williams et al. 2023 (doi:10.3389/fnins.2023.1070413).
- EPI is not defaced, so a whole-head functional picture shows more of a participant than anything the surface census renders. Run structure also identifies datasets: in the 30 sampled datasets, the pair (runs of the first subject, number of subject folders) is unique for 28. The header of every frame of the prototype prints run count, volume count and time.
- In a raw stream, uncorrected distortion and the bright CSF band of EPI are visible by construction, so a viewer has to be told that the raw stream is the reference for what changed and not the thing being judged.
- The side-by-side layout halves the pixels of each panel and spends half the screen on a comparison that showed nothing in this subject. One stream at full size with a key that flips to the other, and the difference and temporal-SD maps beside it, was proposed in its place.
- If ratings ever follow, the traces and the red frames must not be on the rating screen: they are metric scores, which raters do not see (hard rule of the project).

### What was not checked

- No person has watched a video or looked at a frame.
- Any subject with large motion, spikes, distortion correction, multiband or multi-echo data, sessions, or runs on different grids. Of the fMRIPrep versions that decision 3 leaves (20.2.3 to 23.0.2), only 21.0.2 was opened.
- Which subjects and runs of the 71 derivatives have every file, and whether their raw runs are still in the raw bucket. In the sample of 30, the first derivative subject of one dataset was absent from the raw bucket, one subject had 102 raw runs against 40 in the derivative, and run numbers differed in zero padding in one dataset.
- How much of the difference between the streams is interpolation (cubic B-spline here, fMRIPrep's own in the preprocessed series).
- Whether the resampling gives the same bytes on another machine. The package's hard rule on deterministic renders would require that before any of this enters it.
- Scaling to 2 mm data and to long runs, which is multiplication so far. All timings were taken on a machine that was doing other work.
- OpenNeuro's policy on defacing functional images; its documentation could not be reached from the session.
- Prior art that was not opened: FSLeyes movie mode, Q'kay, dashQC and BrainSwipes rest on search summaries. A claim of novelty needs these checked first, and CONN's QA movie and AFNI's `@epi_review` seen running: the first is known here from its paper and the second from its source.
- macOS and Windows: `--one-core` uses `os.sched_setaffinity` and the memory figures read `/proc`.

### Where the report says more than the material shows

`REPORT.md` is kept as it was given. Four of its statements are firmer than what lies behind them, and two of its citations cover less than the sentence they stand behind:

- "written at output level `minimal`" is said of all 236 newer derivatives. The setting was read in one of them; for the others it is inferred from the layout.
- "This holds in every frame": five frames were read, by an agent. The rest is the numbers above. Three volumes exceed 0.5 mm, of which the report names two.
- "A whole experiment takes 24 to 148 s to watch once" (at 12 per second) is the range of four subjects. The fifth has 40 runs, of which 8 were read, and would take longer.
- "run counts and TR come close to fingerprinting a dataset": what was measured is run count together with the number of subjects, unique for 28 of 30. TR was not read in the survey.
- "CONN and AFNI already play EPI series locally" carries one citation, Morfini et al. 2023 (doi:10.3389/fnins.2023.1092125), which is the paper on quality control in CONN. The AFNI half does not rest on that paper. It rests on the prior-art search, which read the script in AFNI's source repository that writes `@epi_review`; the script was not run.
- Taylor et al. 2023 (doi:10.3389/fnins.2023.1205928), cited beside Lepping et al. 2023 for the catches by eye, is the editorial of the research topic "Demonstrating quality control (QC) procedures in fMRI" (PubMed lists it as an editorial, checked 2026-11-13). It reports what the teams of the topic found and is not a study of its own.

Also, its 1.7 to 5.7 TB for 5,000 subjects is the cost of the cheapest route for every dataset, which decision 3 has set aside, and the "about 600 lines" count the core of the pipeline; the scripts here were about 3,300 lines after the port and are about 3,500 after the changes that followed the reviews.

## How to run it

All commands are given at the repository root. The data go into a directory of your choice outside the repository, and results go into directories of their own, apart from the downloads. `-I` keeps Python from importing anything that lies next to downloaded data; the scripts find their sibling modules themselves.

The scripts need only the core dependencies of the package (numpy, scipy, nibabel, pandas, matplotlib, pillow, httpx, imageio-ffmpeg). In a clone that already has an environment nothing has to be installed, because `uv run` uses the environment as it is. In a clone without one, `just setup` makes it; the recipe is `uv sync --all-extras --group dev`. `uv sync --group dev` is not the same command and should not be used in a working clone: in a dry run on 2026-11-13 it listed 40 packages to uninstall, among them fastapi, jinja2, uvicorn, playwright and marimo (the `app` and `analysis` extras and the `browser` group), which the tests of the rater app need. `just setup` itself listed three in the same environment: playwright and greenlet, which are the `browser` group and which `just test-browser` installs again, and one package that the lock file does not name.

```
just setup                          # only in a clone that has no environment yet
D=/some/place/outside/the/repository

# the worked example: 29 files, 133 MB, 5.5 s here; then 5 FreeSurfer files, 27 MB, 1.5 s
uv run python -I prototypes/func_viewer/pipeline/fetch.py $D/data
uv run python -I prototypes/func_viewer/pipeline/fetch_freesurfer.py $D/fs

# the study's pipeline: geometry checks, check images, 13 videos, analysis. 5 min here
uv run bash prototypes/func_viewer/pipeline/run_all.sh $D/data $D/fs/sub-01 $D/out $D/work

# the playback measurements for decision 5: 6 videos and the tiles. 3 min 40 s here.
# Start it when run_all.sh has ended, not beside it (see below)
uv run bash prototypes/func_viewer/pipeline/run_playback.sh $D/data $D/fs/sub-01 $D/playback
```

`$D/out` then holds the seven example videos, frames `example_*_tNNNN.png` (volumes 0, 59, 60, 95, 145, 146 and 200; 59, 95 and 145 are the three over 0.5 mm), `checks/` with the surfaces on the T1w and on the EPI, the difference image and the temporal-SD map, and `timings.jsonl` with one line for each of the seven videos. `$D/work` holds `geometry.txt`, `analysis.txt`, `fidelity.txt`, `tile_sizes.txt`, `check_videos.txt`, the six videos that were made only to be measured and a `timings.jsonl` of its own for the six calls of `make_videos.py` that wrote there. `$D/playback` has a third `timings.jsonl`. The times in the tables above were read from these three files in the run after the port.

Both shell scripts check their inputs before they make anything, print one line per step on the terminal and name a step that fails. Both pin every call to one CPU with a Linux call (`--one-core`), so they need Linux as they stand. The CPU is the first one the process may use, so the two scripts share it when they are started at the same time. In a review on 2026-11-13 that was done: they took 8 min 19 s and 7 min 01 s, each video call about twice its time, and wrote the same bytes. `taskset -c N` in front of a command gives it another CPU. Run one after the other after the changes listed under "The scripts", they took 5 min 16 s and 3 min 52 s while other work ran on the machine, and wrote the same bytes as after the port: 13 videos, 9 check images and 22 frames from `run_all.sh`, 6 videos from `run_playback.sh`, and every text result but for the times in it.

Every script can also be run alone. Its first lines say how, and a script that is given too few arguments prints those lines and exits with status 2 without reading a file or sending a request.

The survey, listing only, at most 4 requests at a time, each step a few seconds here:

```
P=prototypes/func_viewer/survey
uv run python -I $P/survey.py $D/survey              # the prefixes; 30 first subjects in full
uv run python -I $P/raw_and_versions.py $D/survey    # the same subjects in the raw bucket; all versions
uv run python -I $P/layout_all.py $D/survey          # what func/ holds, every dataset
uv run python -I $P/layout_table.py $D/survey        # the counts quoted under "Decisions"
uv run python -I $P/analyze.py $D/survey             # sizes per run and subject, scale-up
uv run python -I $P/spot.py $D/survey                # 12 other datasets, seeded; writes spot.txt
uv run python -I $P/headers.py 0 30 120 154          # volumes and TR: three first subjects, 8 runs of a fourth
```

What these write and print names datasets. It stays in `$D`.

Run on 2026-11-13 after the changes listed under "The scripts", these seven commands wrote the same nine listing and table files and the same text of `layout_table.py`, `spot.py` and `headers.py` as the listing made after the port. The study's own analysis output has four more lines than `analyze.py` prints here, a supplement on one subject and three means that leave one dataset out. No number in this file or in `REPORT.md` is taken from those lines (their numbers were searched for in both). They come from two options of `analyze.py`:

```
uv run python -I $P/pair_listing.py $D/survey ACCESSION_A     # prints the name of the file it wrote
uv run python -I $P/analyze.py $D/survey --pair $D/survey/pair_ACCESSION_A_sub-LABEL.json --without ACCESSION_B
```

`ACCESSION_A` is the one dataset of the sample whose first derivative subject is absent from the raw bucket: the row with `raw_subject_present` False in `func_survey_30_datasets.csv`, which the `analyze.py` call above wrote. `pair_listing.py` then lists, in both buckets, the first subject in key order that both have. `ACCESSION_B` is the dataset with by far the largest first subject: the row with the largest `option_a_bytes` in `func_survey_subject_totals.csv`. Run this way on 2026-11-13, the output was the study's but for the wording of the supplement line.

The second subject (97 MB, 4.7 s to fetch; `check_min.py` takes 20 s). It needs `$D/survey` from `survey.py` above:

```
M=prototypes/func_viewer/minimal_layout
uv run python -I $M/first_run.py $D/survey 123       # prints what follows $D/min in the next line; no request
uv run python -I $M/fetch_min.py $D/min ACCESSION sub-LABEL --task TASK --session SES --run RUN
uv run python -I $M/check_min.py $D/min
uv run python -I $M/still_min.py $D/min $D/min_out
uv run python -I $M/blind_shift.py $D/min $D/min_out 7 --key-dir $D/min_key
```

`first_run.py` also prints, on standard error, the number of prefixes, the fMRIPrep version and the number of subject folders, to compare with "The second subject" above before anything is fetched. The stills go to a folder apart from the download, and the key of the blind trial to a third, so that the folder with the picture can be shown to a viewer without the answer in it.

### The scripts

| Path | What it does |
|---|---|
| `pipeline/geom.py` | ITK text transforms as RAS point maps, resampling, surfaces in T1w space |
| `pipeline/epiview.py` | three-plane panels of a volume on an axis-aligned grid, contours, edge letters |
| `pipeline/fetch.py`, `fetch_freesurfer.py` | download the worked example; the second is new |
| `pipeline/verify_geometry.py`, `overlays.py` | direction of the transforms in numbers and in pictures |
| `pipeline/make_videos.py` | the three videos; `--fps` is volumes per second; `--keyint` is new |
| `pipeline/gop_encode.py` | new: the package's ffmpeg command with a key-frame interval |
| `pipeline/check_videos.py` | decodes every video; frame count, error against PNG frames, key frames |
| `pipeline/analyze.py` | what differs between the streams; dropout; difference and temporal-SD pictures |
| `pipeline/fidelity.py` | how much frame-to-frame change survives H.264 |
| `pipeline/tile_sizes.py` | mosaics per volume; `--panels` (new) for one image per volume and panel |
| `pipeline/run_all.sh`, `run_playback.sh` | the study's sequence; the playback measurements (new) |
| `survey/survey.py`, `raw_and_versions.py`, `layout_all.py`, `analyze.py` | the survey of the report |
| `survey/layout_table.py` | new: the table of all datasets that the study made by hand, by rule |
| `survey/pair_listing.py` | new: one subject in both buckets (the study did this by hand once); without a subject, the first that both have |
| `survey/headers.py`, `spot.py` | experiment lengths; the independent check of 12 datasets, written to `spot.txt` |
| `minimal_layout/first_run.py` | new: the arguments of `fetch_min.py` for a dataset of the sample, from the survey's listing |
| `minimal_layout/fetch_min.py`, `check_min.py`, `still_min.py`, `blind_shift.py` | the second subject: fetch, checks of the stored fits, stills, the blind trial |
| `stage1/list_runs.py`, `list_table.py`, `list_s3.py` | step 1: every functional run of the older derivatives and what it has; the table's reader and columns; counted, limited requests |
| `stage1/check_listing.py` | self-check of the listing table; with `--example` the worked example's files, with `--head N SEED` HEAD requests for a seeded sample |
| `stage1/session.py` | the anchor of a session, the raw stream through the anchor's transform, the frame check and its flag rule |
| `stage1/check_session.py` | self-check of the session logic on the worked example |
| `stage1/session_measure.py` | per run: displacement from the anchor, and the two temporal means against fMRIPrep's |
| `stage1/session_find.py`, `session_frames.py`, `session_fetch.py` | first subjects with several sessions; their stored transforms; the small files of one subject |
| `stage1/rank_motion.py` | step 2: reads every confounds table of the listing and records what it says about motion |
| `stage1/select_runs.py` | the selection rule (marked, with a digest) and the selection table; `--self-check` |
| `stage1/pack_run.py`, `pack_inputs.py`, `pack_contours.py` | steps 3 and 4: the pack of one session (streams, stills, traces, contours, frame check) |
| `stage1/pack_digest.py` | one SHA-256 over a pack, every file against the manifest; compares two packs |
| `stage1/pack_synth.py` | a synthetic long run (1,200 volumes on a 2 mm grid) from the worked example's anatomy |
| `stage1/serve.py`, `stage1/player/` | the page and a server for one pack on 127.0.0.1 |
| `stage1/check_player.py` | drives the page in Chromium: exactness, frame rate, scrubbing, markers, memory, screenshots |
| `stage1/fetch_run.py` | fetches what one selected session needs, checked against the listing, resumable |
| `stage1/run_line.py` | one line of measurements of a packed session, by neutral ids |
| `stage1/run_example.sh`, `run_stage1.sh` | the worked example through every self-check to a pack; the selection to its packs |

Changes made in the port, none of which changes a result: directories are arguments, at most 4 requests at a time everywhere (the survey had used 6 and 8), two dataset names in `survey/analyze.py` became the options `--without` and `--pair`, and the code passes the repository's ruff settings.

Changes made on 2026-11-13 after three reviews of the hand-over, which also change no result:

- A script run with too few arguments prints its usage and exits with status 2. Twelve scripts had ended in a traceback, `headers.py` without an index had listed the bucket and printed nothing, and `spot.py`, which took no argument, had run its whole check and printed dataset names.
- `spot.py` takes the output directory and writes `spot.txt` there. The lines are those it used to print.
- `pair_listing.py` picks the subject itself when none is given, and `first_run.py` is new, so that neither command needs a search through the listings by hand.
- `still_min.py` and `blind_shift.py` make their output directory, and `blind_shift.py` can write its key elsewhere (`--key-dir`).
- The two shell scripts check their inputs before they make anything, print one line per step and name a step that fails. Before, `run_all.sh` with a wrong data directory ended with status 1 and nothing on the terminal.
- `.gitignore` in this folder keeps downloads and results out of the index if a directory is ever put here.

Left out of the study's scratch scripts, because the scripts above supersede them or they produced no number that is used: the architecture study's first movie scripts (`movie.py`, `rawmovie.py`, `facts.py`, which used linear interpolation and gave a temporal SNR that was withdrawn) and its small listers (`sample.py`, `minimal.py`, `versions.py`, `get.py`, `s3ls.py`); `look_at_headers.py` of the prototype, a header dump; `kinds.py` and `explore.py` of the survey, which tabulate file names; `ls.py` of the proposal; and `kwic.py`, a text search in the literature that was read.

## Next steps

The stages of the report, adapted to the decisions. Each stage is kept apart from the surface path, so that the surface census is untouched if the next stage is never built.

### Stage 1: in this folder, nothing in the package

Steps 1 to 4 are built and were run on 2026-11-13 and 2026-11-14 ("Stage 1, as built" above); step 5 is proposed there and not run; step 6 is optional and not built. The steps as they were planned:

1. List what qualifies. A script that lists, anonymously and at most 4 requests at a time, every subject and run of the derivatives made with fMRIPrep 23.0.2 or older, and records for each run whether it has the T1w-space series, the confounds table, the scanner-to-T1w transform, the fsnative transform, the FreeSurfer surfaces under `sourcedata/freesurfer/` and its raw run in the raw bucket (run numbers compared as numbers). `survey/survey.py` has the listing functions. The list names datasets and stays outside the repository.
2. Rank by motion. Read the confounds table of every qualifying run (whole files; 0.3 to 3.4 MB each in the six sampled datasets; how many runs there are is not known yet and should be counted from the listing first). Fix the selection rule before looking at any image, for example the highest mean framewise displacement, at most one run per subject and two per dataset, and write down the rule and the seed if one is used.
3. Render the selected runs in both streams, in a form that plays at up to 60 volumes per second and scrubs to the exact volume. The measurements above are the basis for choosing between a video with every frame a key frame, tiles and mosaics; the surface app's player is built on its own render model, so a functional player is a separate small page (plain HTML and JavaScript, no build step, as the project does elsewhere).
4. Add what the frames of the worked example lacked: scrubbing through slices, the difference between the streams and the temporal SD as views of their own, contours as a layer that can be switched off, and markers that last longer than one volume.
5. State the failure criterion before anyone watches: if the moving picture shows nothing that the mean, the temporal-SD map and the difference stills of the same runs do not show, the paper says so. A way to make that testable is to have each viewer first write down what the stills show for a run and then what the moving picture adds, run by run; the exact procedure is for Tomas to fix (open question 2).
6. Optional, from the critiques: a shift-detection test, in which viewers judge runs with known added shifts of 0 to 5 mm, before coregistration is offered as something eyes can check.

### Stage 2: after milestone 4

A single-subject command on a local fMRIPrep folder, in a subpackage that imports the package's core and that nothing imports. It writes its own page model and player. Entry conditions from the critiques: the resampling is shown to be deterministic under the locked environment, and the text scrubbing and the source scan of decision D20 cover the new code and BIDS names (a task label can name a dataset).

### Stage 3: after milestone 5

Run-level inventory and a view-only tab. The functional part gets its own catalog file, run directory layout and source protocol, built with the same machinery, so that the surface catalog is never migrated. Leasing has to be per run: one subject can exceed the whole cache (9.0 GB of T1w-space series and 22.4 GB of raw series for one first subject in the sample).

### Stage 4: only after a small rated pilot

Ratings and a blind-spot number, measured against the strongest numbers available for each criterion and with a rating screen that shows no metric. By decision 1 this is not planned; it is named so that nothing built earlier rules it out.

## Rules that bind this work

- Read-only, anonymous access to the public buckets, at most 4 requests at a time here, no credentials. Local data are never modified.
- No media file is committed (mp4, png, webp, nii, npy): EPI is not defaced, and the repository is not the place for images of participants. Outputs go to a directory outside the repository. No script chooses a directory by itself, so one inside the repository is possible; `.gitignore` in this folder covers that case for the file types the scripts download and write, and for folders named `sub-*`.
- No list of dataset accessions is committed. Scripts that regenerate a listing are fine. `ds000102` `sub-01` is the one named example.
- The results of stage 1 are named by neutral ids (H01 to H06, Q01, Q02 for the selection, W01 for the worked example's pack, R1, R2 for the runs of a pack). The tables that map them to datasets and files (`selection.parquet`, `ids.csv`, the `map.tsv` of `pack_run.py --map`, `fetch_log.json`) stay outside the repository, and `pack_run.py` refuses to write a map inside it. The ids are not pseudonyms: H01 is the first rank of a ranking that anyone can repeat from the public buckets with these scripts, so they keep dataset names off the screen and no more.
- The positions that this file and the docstring of `survey/headers.py` give (index 123 for the second subject; 0, 30, 120 and 154 for the experiment lengths) are not accessions, but they are no pseudonyms either: anyone who repeats the public listing while the bucket is unchanged turns them back into the five datasets. They are kept because the next session cannot fetch the same data without them. They must not be carried into anything that is shown or published. Stage 1 adds no position. It names its second subject by a rule instead ("the first line that `stage1/session_find.py` prints") and its selected runs by neutral ids; both can be turned back into datasets in the same way by whoever repeats the listing.
- No absolute path of a machine and no e-mail address in a committed file.
- The package is not changed for the prototype. What the prototype lacks in the package it carries itself (`pipeline/gop_encode.py`).
- Datasets appear by keyed pseudonym in everything the project shows or exports (decision D10 in `docs/decisions.md`). How far that carries to functional data is open: run counts, volume counts, TR, task names and the slab that was scanned are public and come close to identifying a dataset, and a whole-head EPI shows the eyes. Until that is settled, nothing functional is published from here, and file names with task labels are treated as identifying.
- Configuration that keeps the folder apart, all in `pyproject.toml`: pytest collects `tests/` only; the wheel is built from `src/surfcensus`; the sdist excludes `/prototypes`; pyrefly excludes `prototypes` (its table also names the `basic` preset, which pyrefly otherwise applies only while it has no configuration). `ruff` does check the folder, and it passes.
- The writing rules of the project apply to anything written here: claims in proportion to the evidence, every number with what it was measured on and when.

## Open questions for Tomas

1. Does "ras bids" mean the raw BIDS series, and should it be shown on its own grid, untouched, or resampled once into the anatomy with fMRIPrep's stored rigid transform, as the prototype does?
2. The failure criterion of stage 1: is the wording above right, who is the second viewer, and should viewers see the stills before the moving picture?
3. Which form of playback, given the measurements: a video with every frame a key frame, tiles, or mosaics? And does the frame-to-frame noise have to survive compression, which decides between CRF 23 and something near lossless?
4. How many hard cases, and by which rule: mean or peak framewise displacement, and how many runs per dataset? Should quiet runs be rendered beside them for comparison?
5. The report also proposed a dataset with fieldmaps, a multiband 2 mm dataset and your own 7T data. Are those still wanted, now that the hard cases are chosen by motion?
6. May the datasets of the examples be named in the opinion paper, and is anything functional shown outside the lab as a whole head or cropped to the brain?
