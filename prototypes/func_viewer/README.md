# Functional viewer prototype

Hand-over of a feasibility study made on 2026-11-13 and of stage 1 of the plan that followed it, built on 2026-11-13 and 2026-11-14. It is written for a reader who has none of the files or the context of the sessions that made them.

## What this folder is

`surfcensus` scores, renders and rates FreeSurfer surfaces. The paper it belongs to argues that people must look at their data, and Tomas Knapen asked whether the same could be shown for functional data: the EPI time series before preprocessing and after fMRIPrep, as moving pictures over the anatomy. A feasibility study answered with a prototype on one subject, a survey of what the public derivatives hold, a search for prior art and three critiques. `REPORT.md` is the report of that study as Tomas received it. This file records what he decided, what the scripts do, what was measured, what stage 1 built and found, and what comes next.

The folder is not part of the package. Nothing in `src/` imports it, it is in neither the wheel nor the sdist, pytest does not collect it, and pyrefly does not check it (the last three by configuration, see "Rules that bind this work"; that nothing imports it is a fact of the code). The scripts import four things from the package and change nothing in it: `surfcensus.render.slicing.MeshSlicer`, `surfcensus.render.frames.ContourRenderer` and `blend`, and `surfcensus.render.encode.write_mp4` (the playback measurements also use `ffmpeg_exe`, `write_gray_webp` and `write_rgba_webp` of the same module).

No image, video or table of the study is in the repository, and no dataset is named but `ds000102`, whose `sub-01` is the worked example here as it is in the package's tests. The reasons are under "Rules that bind this work".

## Decisions

### Decision 7, which governs decisions 1 to 6: what the two streams are for

Tomas Knapen, 2026-11-14, after the review of stage 1 (D36 in `docs/decisions.md`):

"Well, let's be clear, the reason that we're showing both the raw and the fully pre-processed EPI time series as this video is exactly because we want to visualize the effects of pre-processing on the sort of movies, the time courses of the EPI volumes. And so it would be quite insane to perform all manner of processing on the raw data, because the whole idea is to show the difference between that and the pre-processed, and to also show how the pre-processed data now conform to, for example, the FreeSurfer surfaces, etc., etc., right? So it's meant to serve as a quality control type tool for both the raw data and the pre-processing pipeline at the same time. And so the raw and pre-processed visualizations should serve those purposes explicitly. I want you to record this and make it a full priority."

What follows from it. The first two points restate his words; the rest is the reading of the session that applied it, and is his to correct:

- The raw stream is the raw BIDS series with nothing done to it but the one rigid placement per session into the anatomy (decision 6). No motion, slice-timing or distortion correction and no other processing is applied to it, now or later, also not to explain a difference between the streams.
- The difference between the streams, and the raw stream's misfit to the FreeSurfer contours, are what the viewer exists to show: the effect of the preprocessing (head-motion correction, distortion correction where it was applied, fMRIPrep's per-run registration) and the state of the raw data. They are not a confound of the comparison.
- So the pack records, and the page states near the views as information, what the pipeline did to each run: head-motion correction; distortion correction and its method, or none; slice-timing correction or none (from fMRIPrep's summary report of the run). A viewer can then read a difference between the streams as the effect of those steps. Where distortion correction was applied, the raw stream shows the distortion as acquired against the contours, and the preprocessed stream shows how well the correction brought the series onto the surfaces: that is the viewer doing its job.
- The two uses are named on the page and here, view by view. Quality control of the raw data: head motion, artefacts confined to single volumes, distortion and dropout as acquired, and movement between runs under the one registration (a flagged run is a statement about the raw data, not a fault of the pack). Quality control of the preprocessing: what was corrected, what remains after correction, and how the preprocessed series conforms to the white and pial contours.
- The failure criterion of stage 1 (step 5) is about these two uses, and the selection of cases should cover the effects the viewer is meant to show, not motion alone ("What remains of stage 1" and "Next steps").

### Decision 8: the governing purpose of the project

Tomas Knapen, 2026-11-14, recorded as D37 in `docs/decisions.md` (decision 7 above is D36 there):

"Yes, this is the governing purpose, to present people with a full quality control visualization tool that shows, perhaps even on one page, could be different pages, a surface segmentation quality control interface and a raw and pre-processed EPI movie interface. All of those exist for people to be able to judge visually the quality of acquisition and processing on a full session or multiple sessions of fMRI and MRI data."

For this folder it means what "Next steps" says: the EPI movie interface moves into the package and its app, beside the surface interface, and is specified in `docs/`; this folder is the record of stage 1 and the reference for the pack format and the page.

### Decisions 1 to 6

Tomas Knapen, 2026-11-13, in answer to the four questions at the end of `REPORT.md`, with one remark added (decisions 1 to 5), and in answer to the first of the questions that this file then asked (decision 6). His words are quoted; the rest is how they are read here.

1. "viewer". The functional part is a viewer for the opinion paper. It is not a functional census. A census would be a separate, later decision.
2. "i don't - your option sounds like a good strategy". He has no functional subjects to name whose preprocessing is known to have gone wrong. Hard cases are therefore picked as the highest-motion runs, from the confounds files of the derivatives that have them.
3. "limit - i'm not running anyone's fmriprep." With his opening line, "the fmri part, which i would want to render pre-preprocessing (ras bids) and fmriprep preprocessed", this means two streams: the series before preprocessing and the series as fMRIPrep preprocessed it. They are rendered only for derivatives that hold a preprocessed series. No stream is rebuilt from stored transforms, and nobody's fMRIPrep is run. "ras bids" is read here as raw BIDS, the series as the raw dataset shares it. That reading was open question 1 of the earlier list; decision 6 answers it.
4. "yeah please do so that another session can pick it up." The prototype is committed here, under `prototypes/func_viewer/`, outside the package and the wheel.
5. "i also think that the speed of the video can be much much faster - even 60fps is fine as long as the interface allows scrubbing". The 12 volumes per second of the study are no longer the target. The requirement is fast playback, up to 60 volumes per second, together with scrubbing that lands on the exact volume.
6. Tomas Knapen, 2026-11-13, in answer to open question 1 of the earlier list ("ras bids"): "Raw bids means single per session registration to the anatomy to ensure comparability across sessions, processing stages, and subjects. Please continue with that too". How it is read here (the reading is that of the session that built stage 1, not his words): the stream before preprocessing is the raw BIDS series, resampled once into the anatomy (the display grid in fMRIPrep's T1w space), through ONE rigid registration per session that is the same for every run and every volume of the session. So what moves within a session, within a run and between runs stays visible, while sessions, the two streams and subjects are seen in one anatomical frame. No motion, slice-timing or distortion correction. The registration is the stored `from-scanner_to-T1w` transform of the session's anchor run; nothing is registered anew. The anchor rule was fixed before any image was looked at: the run acquired first, by `AcquisitionTime` of the raw JSON sidecars when every run of the session has one, else the first in BIDS name order (task by name, run index as a number); which of the two applied is recorded. A dataset without `ses-` folders has one session per subject. That the runs of a session share the scanner's frame of reference is checked, not assumed (stage 1 below, "One registration per session").

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
- The download for this subject is 133 MB in 29 files (two raw runs 56 MB, two T1w-space runs 59 MB, the T1w 17 MB), 5.5 s at 4 requests at a time, plus 27 MB for the five FreeSurfer files. Since 2026-11-14 `pipeline/fetch.py` also fetches fMRIPrep's summary report of each run (two files of 3.4 and 3.5 kB), 31 files in all; they say that this subject's runs had neither distortion nor slice-timing correction.
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

## Stage 1, as built

Stage 1 (see "Next steps") was built on 2026-11-13 and 2026-11-14 in `stage1/`, in five packages: the listing of what qualifies, one registration per session, the ranking by motion and the selection, the packed run and the player page, and the integration that fetches, packs and looks at the selected sessions. Everything it downloads or writes lies outside the repository. What is not done yet is at the end of this section ("What remains of stage 1").

What the stage is for (decision 7): the page shows the raw and the preprocessed series of each run so that a viewer can judge, at the same time, the raw data and what the pipeline did to it. In short: every functional run of the 72 older derivatives was listed (9,373 qualify); the confounds table of every qualifying run was read and the runs were ranked by mean framewise displacement; a rule fixed before any table was read chose six high-motion runs and two quiet ones; the sessions of those eight runs were fetched and packed with one registration per session; and a page plays a pack at up to 60 volumes per second with scrubbing through volumes and slices, both streams, their difference, stills, contours and traces. An agent has looked at screenshots of every pack. No person has.

### Defaults of this build

Three questions that the packages raised have no answer from Tomas yet. The build takes these defaults. They are defaults of the build, not his decisions, and they are his to change:

- A run that the frame check flags (more than 10 mm somewhere in its brain mask between its own stored transform and the anchor's) is shown through the session's one registration, as a statement about the raw data (the head, or the subject, moved that much between runs, or a stored registration failed), with a warning on the page that cannot be missed: the run button says "flagged" and a red box above the views says that the run may not share the anchor's frame and that its raw stream is not in register with the anatomy.
- The anchor is the first run by the rule as written (acquisition time when every run has one, else BIDS name order), also where the anchor is the odd one of its session.
- The selection is the one that `select_runs.py` made (H01 to H06, Q01, Q02, below), and only the session of each selected run is fetched and rendered, not the other sessions of its subject.

### Step 1: what qualifies (`stage1/list_runs.py`)

Made on 2026-11-13. The survey above looked at the first subjects of each derivative. This step lists every subject, session and functional run of the derivatives made with fMRIPrep 23.0.2 or older, and records for each run whether it has what the two streams need. It reads listings only, and of each derivative its `dataset_description.json`.

A run is the set of files in a `func/` folder of a derivative that share `sub`, `ses`, `task`, `acq`, `ce`, `rec`, `dir` and `run`. A run qualifies when it has all of:

- (a) the T1w-space preprocessed series, (b) the confounds table, (c) its own `from-scanner_to-T1w` transform, (f) its brain mask in T1w space;
- (d) the subject's `from-T1w_to-fsnative` transform, in `sub-X/anat/` or in `sub-X/ses-Y/anat/`;
- (e) `mri/T1.mgz` and the white and pial surfaces of both hemispheres under `sourcedata/freesurfer/` of the derivative, the pial surface under any of its names (`pial`, `pial.T2`, `pial.FLAIR`, `pial.T1`);
- (g) the raw run in the raw bucket, with the same entities and the same run number. Run numbers are compared as numbers: the derivatives write `run-1` where many raw datasets have `run-01`.

A session is a subject and its `ses-` label; a dataset without such labels has one session per subject. This is the unit of the one registration per session that Tomas asked for on 2026-11-13.

```
D=/some/place/outside/the/repository
uv run python -I prototypes/func_viewer/stage1/list_runs.py $D/listing        # 7,150 requests, under 3 min here
uv run python -I prototypes/func_viewer/stage1/check_listing.py $D/listing    # no request; exit status 1 if a rule is broken
```

`$D/listing` then holds `runs.parquet` (one row per run; `stage1/list_table.py` documents the columns and has the reader, `read_runs`), `datasets.csv`, `summary.txt` and, under `listings/`, every key that was listed with its size, ETag and date. The table and the listings name datasets and stay out of the repository; `summary.txt` names none. For every file that a later step reads, the table has the key, the size and the ETag, so that a changed file can be told. The listing is resumable per dataset: a dataset that is listed is not listed again in the same directory, and `--table-only` rebuilds the table from the stored listings without a request. `check_listing.py` can also compare the worked example with the files on disk (`--example`) and ask the buckets about a seeded sample of runs with HEAD requests (`--head N SEED`).

What the listing of 2026-11-13 (23:41 to 23:44 UTC) found. Anonymous, at most 4 requests at a time; a listing made six minutes earlier gave the same table byte for byte.

- 308 dataset prefixes; 72 name fMRIPrep 23.0.2 or older. All 72 were listed in full: 2,190 subject folders, 2,492 sessions, 9,462 functional runs.
- 9,373 runs qualify, in 71 datasets, 2,167 subjects and 2,470 sessions. The dataset counts are those of the survey (72 and 71).
- The 89 runs that do not qualify all have a preprocessed series in BOLD space and none in T1w space. 60 are every run of the 72nd derivative (10 subjects). The other 29 belong to 12 subjects of two derivatives whose other subjects qualify; these 12 have no FreeSurfer folder either. So a dataset can hold subjects of both kinds, and the first subject does not speak for the rest.
- Every run of the derivatives has its raw run in the raw bucket. For 3,102 of the 9,462 that holds only when run numbers are compared as numbers. The raw datasets hold 191 further runs of the same subjects that the derivatives lack, in 3 datasets.
- Runs per session: 1 in 780 sessions, 2 in 249, 3 in 370, 4 in 223, 5 to 9 in 766, 10 to 17 in 104. 1,690 of the 2,470 qualifying sessions have more than one qualifying run, so the question whether the runs of a session share a frame of reference arises for most sessions.
- Sessions per subject: 1 for 2,075 subjects, 2 for 61, 3 for 14, 4 for 17, 5 to 11 for 22. 114 of the 2,167 qualifying subjects have more than one qualifying session, in 10 datasets.
- The fsnative transform lies at subject level for 2,008 subjects and in a session folder for 169. All FreeSurfer folders are at `sourcedata/freesurfer/`. The left pial surface is named `pial` for 1,677 subjects, `pial.T1` for 440 and `pial.T2` for 59.
- A JSON sidecar lies beside the raw series for 4,311 of the 9,373 qualifying runs; 5,036 more have one higher up only. In 949 of the 2,470 qualifying sessions every run has its own, so the acquisition time can decide the anchor run in at most those; whether the sidecars hold it was not read.
- Sizes: the T1w-space series and the raw series of a qualifying run are together 74 MB in the median (quartiles 46 and 154 MB, largest 1.83 GB), and 1.79 TB for all qualifying runs. The confounds tables that step 2 reads are 8.86 GB in 9,373 files (median 0.43 MB, largest 19 MB).
- Four qualifying runs are multi-echo (four raw echoes each); the table records the first echo and the number of echoes. Three T1w-space series have an ETag that is not an MD5 (multipart upload).
- Cost: 7,150 requests and 168 s for the whole listing (156 s the first time), on a machine that was doing other work.

Checked: `check_listing.py` holds on the full table (82 checks); the two runs of the worked example qualify and its 22 files on disk have the sizes and MD5 sums of the table; 302 HEAD requests for the files of 40 qualifying runs (seed 20261113) all answered with the size and ETag of the table; thirteen copies of the table, each broken in one way, all fail the check.

Not checked: nothing but listings was read, so the table does not say that a file is readable or how many volumes a series has; raw and derivative runs are matched by name only, and the raw bucket may hold a later snapshot than the one that was preprocessed; the version is the one `dataset_description.json` names; the rules for cases that the real listings do not have (two raw labels for one run number, a file of size 0, a phase series) were tried on a made-up listing only; the 236 newer derivatives were not listed.

### One registration per session (`stage1/session.py`)

The decision and its reading are decision 6 above.

#### What the code does

`stage1/session.py` holds the logic as functions with a small command line.

- `pick_anchor` takes the runs of one session with their `AcquisitionTime` and returns the anchor, the rule that chose it and one line that says what the rule saw.
- `session_stream` gives the raw stream of any run on a display grid in T1w space: T1w point to scanner point by the anchor's transform, then to a voxel of this run by this run's own header affine. Which affine of the header is meant is decided by `raw_affine` (added in the integration, see "The selected sessions" below): the one that agrees with fMRIPrep's BOLD-space reference image of the run when that image lies in `deriv/`, else nibabel's `img.affine`. The anchor is chosen among the runs that have a stored transform. The interpolation is that of the feasibility study (cubic B-spline, 0 outside the field of view).
- `frame_displacement` and `check_frame` measure, for every run, how far the run is drawn from where its own stored transform puts it when it is shown through the anchor's: mean and maximum over the run's brain mask in mm, and the rotation and translation that the difference decomposes into.
- `frame_flag` is the rule: a run is flagged when that displacement exceeds 10 mm somewhere in its brain mask. A flagged run must not be shown as if it shared the anchor's frame.

By default the voxel map is computed as `pipeline/geom.py` computes it, so a run through its own transform has the bytes of the study's per-run stream. That expression uses numpy's inverse and matrix product, whose last bits can depend on the linear-algebra library. `portable=True` computes the map in Python floats in a fixed order. Which of the two stage 1 uses is open (open question 12); the packs were made with the default.

#### What was measured

Everything here is from 2026-11-13, on one machine. No picture was made or looked at: these are numbers.

The worked example (`ds000102` `sub-01`, two runs in one session):

- The raw runs have no JSON sidecar, so there is no `AcquisitionTime` and the anchor is run 1 by name order.
- The anchor run's stream is identical, bit for bit, to the per-run stream of the feasibility study in all 146 volumes, and so are the three-plane strips cut from them.
- Run 2's own transform differs from the anchor's by 0.83 mm on average and 1.09 mm at most over its brain mask (39,801 voxels): a rotation of 0.20 degrees and a translation of 0.82 mm, mostly posterior and superior.
- The temporal mean of run 2's raw stream correlates with the mean of its preprocessed series at r = 0.9790 inside the brain mask through the anchor's transform, against 0.9993 through its own. Per volume the median is 0.9784 against 0.9991.
- On screen (one window, in the brain mask) the raw and the preprocessed stream of run 2 differ by 6.1 gray levels of 255 on average through the anchor's transform, against 1.8 through its own.

What that means for a viewer: under one registration per session, run 2 sits 0.8 mm off the surfaces compared with where its own registration puts it, and the difference between its two streams is about three times as large as before, most of it this offset. In this subject that is a quarter of a voxel.

A second subject, to have more than one example: the first subject of an older derivative with two sessions of five runs each (fMRIPrep 22.0.1, 182 volumes per run, voxels of 3.125 x 3.125 x 4 mm). It is the first line that `stage1/session_find.py` prints; the dataset is not named here.

- It has no `AcquisitionTime` either. The name-order fallback takes, in each session, run 1 of the task whose name sorts first. Whether that run was acquired first is not known.
- Within a session the other runs lie 0.4 to 2.7 mm (mean) and 0.7 to 5.3 mm (maximum) from the anchor, with rotations of 0.2 to 2.1 degrees. The correlation of the temporal mean with the preprocessed mean is 0.981 to 0.994 through a run's own transform and 0.84 to 0.98 through the anchor's. No run is flagged.
- Between the two sessions the anchors differ by 9.3 mm on average and 20 mm at most, with a rotation of 12 degrees, and the correlation falls from 0.98 to 0.52. Taken for one session, the second would be flagged.

The stored transforms of many sessions (`stage1/session_frames.py`: text files and brain masks only, no series), for the first subject of the 70 older derivatives whose first subject has a transform, 103 sessions and 555 runs:

- `AcquisitionTime` is readable for 222 of the 555 runs, and the time rule applies in 30 of the 103 sessions. In the 26 of those with two or more runs, name order would have picked another run in 9. The sidecars write the time without leading zeros (17:30:2.685000).
- Within a session (452 pairs of a run and its anchor) the largest displacement in the brain mask is 2.7 mm in the median pair, 4.7 mm at the 75th percentile, 9.8 mm at the 95th and 33.9 mm at most. In 39 of the 77 sessions with two or more runs, at least one run is more than 5 mm from the anchor somewhere in the brain.
- Between sessions (33 pairs of anchors of two sessions of one subject, 10 subjects), which are known not to share a frame, the median is 12.6 mm, the smallest 1.9 mm and the largest 44.5 mm.
- The two overlap. At the limit of 10 mm, 21 of the 452 within-session pairs are flagged, in 12 of the 77 sessions, and 9 of the 33 between-session pairs are not (6 of the 9 in one subject).
- In one of the 77 sessions all five other runs are more than 10 mm from the anchor and within 3.3 mm of each other. There the first run, or its stored registration, is the odd one, and the whole session would be drawn off the anatomy. `check_frame` reports this case and changes nothing.

So the limit of 10 mm is about the 95th percentile of what the runs of one session show, and three quarters of the pairs known to be in different frames lie beyond it. It is a limit of plausibility. The stored transforms cannot tell a head that moved between two runs from a subject who was taken out and put back, or from a registration that failed. The raw headers do not help: pairs with the same raw affine as the anchor exceed the limit as well (17 of 420).

#### How to run it

```
S=prototypes/func_viewer/stage1
uv run python -I $S/check_session.py $D/data                    # self-check on the worked example, 35 s here
uv run python -I $S/session_measure.py $D/data $D/session_out   # the numbers above for the worked example
uv run python -I $S/session.py anchor NAME_OR_SIDECAR ...       # the anchor of each session and the rule
uv run python -I $S/session.py frame MASK ANCHOR_XFM RUN_XFM ...   # displacement per run; status 1 when flagged
uv run python -I $S/session.py digest RAW_BOLD GRID_IMAGE ANCHOR_XFM   # SHA-256 of a run's stream

uv run python -I $S/session_find.py $D/survey $D/sessions       # listing only; prints candidates without names
uv run python -I $S/session_frames.py $D/sessions               # transforms and masks of the first subjects; 1 to 2 min
uv run python -I $S/session_fetch.py $D/second ACCESSION sub-LABEL --means
uv run python -I $S/session_measure.py $D/second $D/second_out
```

`check_session.py` exits with status 1 when the anchor run is not bit-identical to the per-run stream, when the stream of run 2 through the anchor's transform does not have the same SHA-256 in two calls and in a new process, or when the displacement between the two runs leaves the values above by more than 0.002. What `session_find.py`, `session_frames.py`, `session_fetch.py` and `session_measure.py` write names datasets, subjects and tasks and stays in `$D`; what `session_measure.py` prints calls sessions and runs S1, R1 and so on.

| Path | What it does |
|---|---|
| `stage1/session.py` | anchor of a session, raw stream through the anchor's transform, frame check and flag rule |
| `stage1/check_session.py` | self-check on the worked example |
| `stage1/session_measure.py` | per run: displacement from the anchor, and the two pictures against fMRIPrep's |
| `stage1/session_find.py` | listing: first subjects of older derivatives with several sessions of several runs |
| `stage1/session_frames.py` | the stored transforms of those first subjects: within-session and between-session displacements |
| `stage1/session_fetch.py` | the small files of one subject, and on request the temporal means of its series |

#### What was not checked

- No picture was made and nobody looked at one. What a viewer sees of an offset of 1 or 3 mm is not known.
- Whether the runs of any session really share a scanner frame. There is no ground truth: the rule rests on 77 sessions of first subjects, one subject per dataset, and on 10 subjects with several sessions. Why 12 sessions hold a run beyond 10 mm was not looked into.
- A subject without `ses- ` folders who was scanned on several visits counts as one session. The stored transforms cannot show that.
- The time rule on a rendered stream: neither measured subject has sidecars. A session that passes midnight is ordered wrongly, since the sidecars hold a time of day.
- Multi-echo runs, a run without a stored transform, and a session whose runs lie on different T1w-space grids.
- Whether the derivatives applied distortion correction was not looked at when this was written; the review of stage 1 found that it can be read per run, and the packs now record it ("The selected sessions" below).
- The same bytes on another machine, with either voxel map. With the portable map the anchor run differs from the study's stream in one voxel of each of 5 of 146 volumes, by at most 1e-6.

### Step 2: ranking by motion and the selection (`stage1/rank_motion.py`, `stage1/select_runs.py`)

Made on 2026-11-14 (UTC), on the listing of step 1. `rank_motion.py` reads the confounds table of every qualifying run and writes down what it says about head motion; `select_runs.py` applies a selection rule to that table and names the runs that stage 1 renders.

What the ranking is: fMRIPrep's own estimate of head motion. It selects for motion and for nothing else that can go wrong in preprocessing: not for distortion, a failed coregistration, dropout, or spikes that the motion estimate does not follow.

The rule is a default of the session that wrote it, not a decision of Tomas Knapen: he decided that hard cases are the runs with the most motion (decision 2), and open question 3 below (how many, which measure, how many per dataset; question 4 of the earlier list) is unanswered. It was written into `select_runs.py`, as text and as code between two marks, before any confounds table was read and before any image of these runs existed. `rank_motion.py` writes a digest of the marked rule before it sends a request, and `select_runs.py` reports whether the digest is still the same. The evidence that the rule was fixed before any table was read is that digest log, which stays in scratch with the ranking's output: it timestamps the digest at 00:04:40 UTC, ten seconds before the first table was read (00:04:50 in `invocations.jsonl` of the ranking), and the digest of the marked parts of the present file is still that of the first ranking call (`3cf8718cbe00cd18`; recomputed by the review of stage 1 with its own code over both marked parts). The git history is not that evidence: the files were first committed at 00:24, after the ranking, and `select_runs.py` changed after the ranking outside the marked parts. The full text is the docstring of `select_runs.py`; in short:

- The value of a run is the mean framewise displacement of its confounds table over the volumes that have one (the first has none). The maximum, the share of volumes above 0.5 mm, the median, the number of volumes and standardised DVARS are recorded beside it.
- A run is ranked when it qualifies, its table is the file of the listing (size and MD5), the table has at least 50 volumes, the mean is above 0 and the raw run is one series (multi-echo runs are left out). Equal means go in the order of the table of step 1. No random number is used.
- Hard cases: the six highest-ranked runs, at most one per subject and two per dataset.
- Quiet comparisons: two, each the ranked run with the lowest mean of another subject, in the dataset of the first and of the second hard case.
- A size limit of 1.5 GB for the two series of a run, so that they can be fetched. Every run passed over is listed with rank and size, and the summary says how many of the 50 highest-ranked runs are over the limit and what the same walk selects without it.
- A volume check: a run whose raw series and T1w-space series do not both have as many volumes as the confounds table has rows is passed over (the first bytes of both series are read). TR and voxel size come from the same headers; neither the listing nor the confounds table has them.
- Selected runs are called H01 to H06 and Q01, Q02. The whole session of a selected run is recorded, as `<id>-01`, `<id>-02`, ..., because the one registration of a session is that of its anchor run, which may be another run.

```
D=/some/place/outside/the/repository
P=prototypes/func_viewer/stage1
uv run python -I $P/select_runs.py --self-check                              # the rule on made-up tables; no file, no request
uv run python -I $P/rank_motion.py $D/listing $D/motion                      # 9,373 requests, 8.9 GB, 8 min 18 s here
uv run python -I $P/select_runs.py $D/listing $D/motion $D/motion/selection  # 16 ranged requests, 7 s here
```

`$D/motion` then holds `motion.parquet` (one row per qualifying run), `rows.jsonl` (one line per table as it was read, which makes the reading resumable), `summary.txt` and `invocations.jsonl`. `$D/motion/selection` holds `selection.parquet` (one row per qualifying run of the eight sessions, with every key, size and ETag that the rendering needs; `select_runs.py` documents the columns and has the reader, `read_selection`), `ids.csv` (the table that maps the ids to datasets), `passed_over.csv`, `ranked.parquet`, `headers.json` and `summary.txt`. All but the two `summary.txt` name datasets and stay out of the repository. The tables are read whole, anonymously, at most 4 requests at a time, streamed into memory and not written to disk. If the tables were together more than 10 GB by the listing, the script would read those of the first k subjects of each dataset in label order and say what it left out; they are 8.86 GB, so every table was read.

What was found on 2026-11-14, 00:04 to 00:17 UTC, on a machine that was doing other work:

- All 9,373 tables were read and used: each has the size and the MD5 of the listing, and each has a `framewise_displacement` and a `std_dvars` column. The older column names that the script also accepts occur in none. 2,000 tables have non-steady-state columns (1 to 12 volumes marked). The first volume has no framewise displacement in every table, and no later volume lacks one.
- Framewise displacement computed again from the six motion parameters of the same table (translations plus rotations on a sphere of 50 mm) equals the column in all 9,373 tables, to 1.5e-14 mm.
- 9,186 runs are ranked, from 69 datasets and 2,151 subjects. 187 are not: 183 have fewer than 50 volumes and 4 are multi-echo. The highest mean among those is 0.533 mm, so neither exclusion removed a candidate.
- Mean framewise displacement of the ranked runs: quartiles 0.084, 0.117 and 0.177 mm, 90th percentile 0.282, 95th 0.397, 99th 0.789, largest 1.914 mm. The largest single step in a run has quartiles 0.29, 0.51 and 1.04 mm over all tables, 99th percentile 7.5 mm, largest 41.6 mm.
- The six hard cases are ranks 1 to 6: means 1.914, 1.540, 1.501, 1.415, 1.341 and 1.290 mm, with 26 to 82% of their volumes above 0.5 mm. They are six subjects in six datasets, so neither cap passed over a run. The 50 highest-ranked runs come from 12 datasets and 35 subjects.
- The size limit removed nothing: 57 ranked runs are over it, the highest-ranked of them at rank 812, none among the top 50, and the walk without the limit takes the same six.
- The two quiet comparisons have means of 0.070 and 0.067 mm (ranks 7,971 and 8,165). The second has another voxel size than the hard case of its dataset, so that pair is from one dataset but not one protocol.
- All eight runs passed the volume check. Between them they have 144 to 453 volumes at a TR of 1 or 2 s, voxels from 1.7 x 1.7 x 3.5 mm to 3.1 x 3.1 x 3.4 mm, and two series of 54 MB to 1.26 GB per run.
- The eight sessions hold 19 qualifying runs and 6.55 GB of series. Four are sessions of one run. The other four have 2 to 5 runs, and in none of them does a run have a JSON sidecar beside its raw series, so their anchor run is the first in BIDS name order. One session is 5.0 GB in four runs; of it, the integration fetched and packed the anchor and the selected run only (below). Two of the selected subjects have three sessions each; only the session of the selected run is recorded.
- Cost: 9,373 requests and 8.86 GB in 498 s for the tables, 16 ranged requests of 8 kB for the headers.

Checked: the self-check of the rule (two made-up tables, answers worked out by hand) passes, and twelve copies of the script, each with one part of the rule changed, fail it. A separate script that derives the selection without the functions of `select_runs.py` gives the same eight runs and the same sessions. 51 tables fetched again and parsed with another reader agree with the table in rows, values and MD5. A second call of `rank_motion.py` sends no request and writes the same bytes. On the worked example the reader gives the three volumes above 0.5 mm that are reported above.

Not checked: no image of a selected run exists, so whether these runs show motion on screen is not known, and the motion estimate itself can be wrong where realignment failed. The volume check compares counts, not content, and only for the eight selected runs. The older column names, the passes for size, subject and dataset, the failing branch of the volume check and the sampling for more than 10 GB were tried on made-up input only, because the real data has none of these cases. Whether the runs of a session share a scanner frame is the check of the next step.

The ids are not pseudonyms. H01 is the first rank of a ranking that anyone can repeat from the public buckets with these scripts, so the ids keep dataset names off the screen and no more than that.

### Steps 3 and 4: the packed run and the player page (`stage1/pack_run.py`, `stage1/player/`)

`stage1/pack_run.py` packs the runs of one session for a page that plays in the browser. Both streams of every run are on the display grid of the preprocessed series: the preprocessed series as stored (what the pipeline made of the run: quality control of the preprocessing), and the raw series resampled once through the session's one registration (that of the anchor run; `session.py`), with nothing else done to it (quality control of the raw data; decision 7). The raw stream's resampling sets a display voxel to 0 when its point lies outside the outermost voxel centres of the raw series along any axis, so brain within half a voxel of the edge of the raw slab is black in the raw stream; the docstring of `session.resample_voxels` said until 2026-11-14 that the edge value reached half a voxel further, which the code never did (finding F4 of the review). The code was not changed, since the packs and the bit identity with the feasibility study rest on it. Each run also gets its temporal mean and SD stills and its framewise displacement and DVARS. Runs are in acquisition order, the anchor is marked, and each other run's measured displacement from where its own registration would put it is recorded. White and pial contours of both hemispheres are cut at every slice of every plane. The pack names nothing: a neutral id and runs R1, R2 and so on. `--map` writes the id-to-file table outside the pack and the repository.

Voxels are stored as 8-bit values under one window per run: 0 to the 99.5th percentile of the preprocessed series in the brain mask, over every tenth volume. They are kept as zlib-compressed residuals against a reference volume, in chunks of about 4 MiB decoded. The page reverses this exactly. On the worked example a volume of 109,368 voxels takes 56 kB, against 86 kB for zlib of the values and 74 kB for a lossless WebP mosaic. A WebP mosaic of the residuals would be 49 kB, but it would put an image decoder between the file and the values. Two packs of the worked example are byte-identical (`pack_digest.py`).

`stage1/serve.py PACK` serves the pack and the page on 127.0.0.1 only. The page (`stage1/player/`, plain modules, no build step, nothing from another host) offers:
- play at 1 to 60 volumes per second, painting every volume and skipping none;
- the time slider, the arrow keys and clicks in the traces land on the exact volume;
- slices in all three planes, scrubbed with sliders, the arrow keys, the wheel or a click, with the crosshair of the other two planes;
- the raw stream, the preprocessed stream, both side by side, and raw minus preprocessed in a diverging map whose scale is written under it;
- the temporal mean and SD of each stream and their differences as stills;
- contours that s switches off;
- under the run buttons, one line on what the pipeline did to the run (head-motion correction; distortion correction and its method, or none; slice-timing correction or none), and one on how many qualifying runs of the session are not in the pack (both from 2026-11-14, below);
- traces with a cursor, bands on volumes above the FD threshold and on non-steady-state volumes, and a marker on the frame that stays at least 500 ms.

Keys follow the rater app where the action is the same, and ? shows them all. A run that does not fit in memory decoded is held in pieces around the cursor, up to a limit (512 MiB by default).

The pack format ('func-viewer-pack', version 1) is documented in the docstring of `pack_run.py`: a `manifest.json` with the grid (shape, voxel size, obliqueness; not the affine), the rules as text, the registration (anchor, rule, limit, whether a shared frame is plausible) and per run its frame check, volumes, TR, window, traces and the files with size and SHA-256; one zlib file per stream for the reference volume, the mean and the SD; chunks of residuals; and one contour file per plane. Every chunk's volumes are 8-bit values, voxel (i, j, k) at i + nx (j + ny k).

The page draws a flagged run with a red box above the views and "flagged" on its run button, and a box with its own text when the anchor is the odd one of its session; the line under the run buttons gives each run's displacement from where its own registration puts it. Both are statements about the raw data under one registration (the head moved that much between runs, or a stored registration failed), and the page says so.

Fields added on 2026-11-14 after the review, within version 1 of the format (a manifest without them opens; the page then says that the facts are not recorded). Per run, `preprocessing`: `motion_correction` (always applied by fMRIPrep, which its report does not state per run), `sdc` and `slice_timing` (applied, not applied or unknown), `sdc_method` (the report's words), `source`, and `sdc_figure_listed` (whether the listing has the run's `figures/<stem>_desc-sdc_bold.svg`, a cross-check); and `raw_header`: whether fMRIPrep's BOLD-space reference image of the run was there and which affine of the raw header the raw stream took (`header`, `qform`, `reference`, or `header, unchecked` without a reference image). At the top, `runs_not_fetched`: the qualifying runs of the session that `fetch_run.py` did not fetch, from `fetch_log.json`, or null where that was not recorded (the worked example). The older `runs_left_out` counts only runs under DATA that lack a file. `pack_run.py` prints the same facts, and a WARNING line for a run without a usable reference image or where the report and the listing disagree.

`stage1/check_player.py` drives the page in Chromium and measures. On 2026-11-14 (one machine, 4 CPUs shared with other work, Chromium 153 headless), by the package that built the page, on a pack of the worked example that it gave the id H01 (before the selection used that id; `run_example.sh` gives it W01) and on a synthetic long run:

- The worked example (2 runs of 146 volumes, 34.4 MB packed, 31 to 36 s to pack):
  - first frame 0.5 s after opening, everything loaded 1.1 s, 34.5 MB in 49 requests;
  - 12, 30 and 60 volumes per second asked gave 12.00, 30.00 and 60.00 painted in one stream, and 60.01 side by side and 60.02 for the difference, with no volume repeated or skipped. Before the contour layer was drawn only when it changes, side by side painted 35.8 per second on the same shared machine;
  - 240 random volume, plane, slice and stream combinations: every canvas pixel equals the 8-bit value computed in Python from the NIfTI files with the prototype's own reader and resampler (`pipeline/geom.py`), and so does every voxel centre in 12 screenshots. This shows that the pack and the page show the values that the prototype's resampling gives, not that the resampling puts the raw stream in the right place (below, "The independent geometry check");
  - jumps land on the asked volume in 2 to 6 ms.
- The synthetic long run (1,200 volumes on a 2 mm grid, 742 MB packed, 1.48 GB decoded, 6 min to pack):
  - 60 per second in every view, decoded chunks never above the limit (peak 534 of 537 MB);
  - a jump to a volume not in memory takes 141 ms (median), 317 ms at most;
  - the renderer process held 1.3 to 1.6 GB.

No person has used the page. What an agent saw in screenshots is not what a viewer sees.

### The selected sessions, fetched and packed (`stage1/fetch_run.py`, `stage1/run_stage1.sh`)

`stage1/fetch_run.py` fetches what one selected session needs into a directory that is given: for each run the T1w-space series, its brain mask, the scanner-to-T1w transform, the confounds table, the BOLD-space reference image and the derivative's sidecar, the raw series and the raw sidecar when the run has its own; for the subject the fsnative transform and the five FreeSurfer files (a pial variant such as `lh.pial.T1` is stored as `lh.pial`). It sends at most 4 requests at a time, anonymously, compares every file with the listing (size, and the MD5 against the ETag where the ETag is one), continues a partial file with a range request, and skips a file that is in place and checked. It chooses the anchor as `session.pick_anchor` does, after fetching the raw sidecars, and fetches at most four runs of a session by default: the anchor, the selected run, then the others in BIDS name order. `stage1/run_stage1.sh` takes each id through fetch, `session_measure.py`, `pack_run.py`, optionally `check_player.py`, the deletion of the series and one line of measurements (`stage1/run_line.py`), as described under "How to run it".

Run on 2026-11-14, 08:56 to 09:38 UTC, on one machine whose 4 CPUs were shared with other work, so the times are noisy. All eight sessions of the selection were fetched and packed, with these limits: at most 8 GB transferred, at most 4 requests at a time, at most 5 GB in the scratch folder at any time.

What was left out. H03 has five qualifying runs; by the default of at most four, its fourth run in name order was not fetched (the anchor, the selected run and the two runs after the anchor were). H05 has four runs of about 1.25 GB each; for lack of disk room only its anchor and the selected run were fetched (`--runs H05=1,3`), 2.53 GB. Every other session was packed whole.

Per session, by neutral id. "Preprocessing" is what fMRIPrep's summary report of each run (`figures/<stem>_desc-summary_bold.html`) says the pipeline did besides head-motion correction, which it always applies: distortion correction and its method, and slice-timing correction; the review of stage 1 read it for all 19 qualifying runs of the eight sessions, and the same reading is now done by `pack_run.py` (below, "After the review"). "Frame" is the displacement of a run, shown through the anchor's transform, from where its own stored transform puts it, over its brain mask (mean and maximum, mm); "r" is the correlation, inside the brain mask, of the temporal mean of the raw stream with the temporal mean of fMRIPrep's preprocessed series, through the run's own transform and through the anchor's. Run ids are those of the pack (R1, R2, ... in acquisition order, R1 the anchor); grids, voxel sizes, volume counts and TR are given as ranges below, since together they come close to identifying a dataset.

| Id | Runs in session, packed | Preprocessing besides motion correction | Frame of the other runs, mean / max mm | Flagged | r, own transform | r, anchor's transform | Pack | Time to fetch, to pack |
|---|---|---|---|---|---|---|---|---|
| H01 | 1, 1 | distortion correction, fieldmap-less (SyN) | none | no | 0.950 | 0.950 | 21.1 MB | 2 s, 15 s |
| H02 | 1, 1 | distortion correction, fieldmap-less (SyN) | none | no | 0.882 | 0.882 | 17.7 MB | 4 s, 17 s |
| H03 | 5, 4 | none | R2 1.48 / 2.54; R3 1.99 / 3.84; R4 4.37 / 10.06 | R4, the selected run | 0.85 to 0.95 | 0.84 to 0.94 | 81.4 MB | 4 s, 59 s |
| H04 | 2, 2 | none | R2 0.54 / 0.94 | no | 0.977, 0.984 | 0.981, 0.984 | 40.7 MB | 3 s, 33 s |
| H05 | 4, 2 | distortion correction, fieldmap-based (phase-difference map) | R2 3.39 / 6.58 | no | 0.916, 0.863 | 0.916, 0.749 | 423.0 MB | 279 s, 287 s |
| H06 | 4, 4 | none | R2 1.34 / 2.61; R3 6.30 / 12.72; R4 2.23 / 3.82 | R3 | 0.933 to 0.947 | R2 0.863, R3 0.505, R4 0.766 | 171.0 MB | 11 s, 147 s |
| Q01 | 1, 1 | distortion correction, fieldmap-less (SyN) | none | no | 0.854 | 0.854 | 13.8 MB | 3 s, 18 s |
| Q02 | 1, 1 | distortion correction, fieldmap-less (SyN) | none | no | 0.892 | 0.892 | 11.2 MB | 4 s, 17 s |

So in five of the eight sessions, both quiet comparisons among them, fMRIPrep corrected the preprocessed series for distortion: four by its fieldmap-less SyN method (`FLB ("fieldmap-less", SyN-based)` in the report) and one with a phase-difference fieldmap (`FMB (fieldmap-based) - phase-difference map`). In H03, H04 and H06 it applied none; H03's dataset has phase-difference fieldmaps all the same. Slice-timing correction was applied in none of the 19 runs. Every run of a session has the same entries. The listing agrees in all 19 runs: the distortion-correction figure (`figures/<stem>_desc-sdc_bold.svg`) is listed for exactly the runs whose report names a correction. The eight packs in scratch were made before these facts were written into the manifest; they predate the fields, and their page says that the facts are not recorded.

- Every pack has the grid of its preprocessed series: 46 to 87 voxels by 54 to 95 by 37 to 60, voxels of 1.7 to 3.1 mm in plane and 2.4 to 4 mm between slices, 127 to 453 volumes per run, TR 1 or 2 s. The eight packs take 780 MB together. Grids, voxel sizes, volume counts and TR of the selected sessions are given in this file as ranges over the eight only, never per id, since together they come close to identifying a dataset (see "Whether anything the page shows identifies a dataset" below for what a pack itself carries).
- The anchor was the first run in name order in all eight sessions. The four single-run sessions have a raw sidecar of their own, but none of these sidecars holds an `AcquisitionTime`; the four multi-run sessions have none of their own.
- Two sessions are flagged: the selected run of H03 (10.06 mm at most, just over the limit) and the third run of H06 (12.72 mm). Both are shown with the warning of the default above. In H06 the flagged run's temporal mean correlates at 0.94 with fMRIPrep's through its own transform and at 0.51 through the anchor's.
- In the four multi-run sessions, the runs that are not the anchor lie 0.5 to 6.3 mm (mean) from where their own registration puts them. The correlation of the temporal means drops with it: for the five runs between 1.3 and 3.4 mm, from 0.86 to 0.95 through their own transform to 0.75 to 0.89 through the anchor's, and not at all for the run of H04 at 0.5 mm (0.977 through its own transform, 0.981 through the anchor's). That last run was not looked into.
- r through a run's own transform is 0.85 to 0.98 in these high-motion and quiet runs, against 0.9993 in the worked example. That is a measure of how much the two streams differ, which is what the viewer shows (decision 7), not a defect. Why they differ this much, from the review of stage 1 (its own code: nibabel, numpy and scipy only). What was shown: it is not a misplacement of the raw stream by the prototype's chain (a search over shifts of up to 3 mm raises r by at most 0.012); not slice timing, which no run had corrected; not the masks, which give 0.9993 on the worked example. In the quiet run Q01, motion does not explain it: the mean over its 143 volumes with framewise displacement below 0.2 mm gives the same r, 0.8538. The runs without distortion correction and with low or moderate motion are the highest (the worked example 0.9993, H04 0.98), and every session with distortion correction is lower (H01 0.950, H02 0.88, H05 0.92 and 0.86, Q01 0.854, Q02 0.89). In Q01, local shifts of the raw mean along the anterior-posterior axis raise r in lateral and inferior blocks (+5 mm: 0.59 to 0.64; +2 mm: 0.79 to 0.82), the non-rigid pattern along the phase-encoding direction that distortion correction removes from the preprocessed series. Motion accounts for part of it in the hard cases: H01 with its low-motion volumes only goes from 0.950 to 0.970, H06 from 0.933 to 0.935. H03 (no distortion correction, high motion) is at 0.85 for its selected run. What is a guess and was not shown: that the distortion correction accounts for most of the gap in Q01, Q02, H02 and H05 (showing it would take the derivative's fieldmap coefficients, which the prototype does not read), and that H06's 0.93 comes from motion together with its 1.7 mm voxels and dropout or edge effects in the inferior slab. Read as the viewer means it: in the sessions with distortion correction the raw stream shows the distortion as acquired against the contours, and the preprocessed stream how well the correction brought the series onto them.
- Bytes: 5.20 GB in 166 requests for the eight sessions, H04 and H06 included twice (below), plus about 1 MB in about 30 small requests (sidecars of a dry run, and NIfTI headers read with range requests to find the error below). 326 s of fetching in all. The scratch folder held at most about 4 GB, during H05 (the two runs' series, 2.5 GB, with everything else that was kept).

An error that was found and fixed. In the first pass, the raw stream of H06 lay about 22 mm from the anatomy in every run, the anchor included: the correlation of its temporal mean with fMRIPrep's was 0.27, and `check_player.py` passed all its checks on it, because it computes its expected values with the pack's own reader and resampler (`pipeline/geom.py`) and the same header affine, so it shares any such error (finding F3 of the review; `check_geometry.py` below is the independent check). The cause: the four raw series of H06 store a qform and an sform (code 2) that differ by 21 to 25 mm. nibabel's `img.affine` takes the sform; fMRIPrep's reference image of each run (`<stem>_boldref.nii.gz`, BOLD space) has the qform, and the stored transform refers to that image. In H04 one raw series has the same kind of header (4.4 mm apart) and there fMRIPrep's reference image has the sform. So no rule on the header alone fits both, and `session.raw_affine` now takes the affine of the run's BOLD-space reference image as the arbiter: the header's `img.affine` when the reference image agrees with it to 1e-4, else the qform when that agrees, else the reference's own affine; without a reference image, `img.affine` as before. `fetch_run.py` fetches the reference images, and `check_player.py` applies the same rule in its own code. On the worked example the headers agree and nothing changes. H04 and H06 were fetched and packed again after the fix; the numbers above are those of the second pass. Of the 19 qualifying raw series of the eight sessions (16 of them packed), five have such a header (H04-02, 4.45 mm apart; the four runs of H06, 20.7 to 24.6 mm); this file said "15" before the review. fMRIPrep's validation report of each run (`figures/<stem>_desc-validation_bold.html`) says what it did with such a header: "sform has been copied from qform" for the runs of H06, "qform has been copied from sform" for H04-02.

Which packs were made with that rule (finding F2 of the review). Only H04 and H06 were fetched again after the fix, with their reference images, so only their packs were made with the reference image as the arbiter. The packs of H01, H02, H03, H05, Q01 and Q02 date from before the fix: their data folders hold no `*_boldref.nii.gz`, and `session.raw_affine` fell back to the raw header's `img.affine` for them, without saying so. The review read the headers of all 19 raw series (ranged requests) and the 19 reference images: in those six sessions `img.affine` equals the reference image's affine to 1e-4 in every run. So those six packs place the raw stream as the rule would have, but because their headers happen to agree, not because the rule was applied. From 2026-11-14 on, `fetch_run.py` fetches the reference image of every run with the rest (it did so already when the listing has it, which it has for all 19 runs here) and names on standard error a run whose reference image the listing lacks; `pack_run.py` records in the manifest per run whether a reference image was there and which affine was taken, and prints a WARNING when none was usable. A pack made from now on therefore never falls back silently. Whether the pipeline scripts of the feasibility study (`pipeline/geom.py`, `minimal_layout/`) meet such headers was not checked; they read `img.affine`.

#### `check_player.py` on two packs of the selection

Run inside `run_stage1.sh`, before the series were deleted, on H05 (the largest pack) and on H06 (four runs, one flagged), on 2026-11-14; the CPUs were shared with other work. All checks hold on both. On H05 (2 runs, the largest pack, 423 MB packed and more than the page's default limit of 512 MiB decoded):

- first frame 0.46 s after opening, everything wanted loaded 5.0 s, 166 requests and 317 MB (the page holds at most the limit, so not every chunk is loaded at once); the renderer process held 1.2 to 1.4 GB;
- 240 random volume, plane, slice and stream combinations: every canvas pixel equals the 8-bit value computed in Python from the NIfTI files with the prototype's own resampling (0 differ), and every voxel centre in 12 screenshots (0 of 66,025);
- the stills of both runs: 0 differ;
- 12, 30 and 60 volumes per second asked gave 12.02, 30.00 and 60.00 painted in one stream, 12.00, 30.00 and 60.01 side by side, and 60.00 for the difference, with no volume repeated or skipped; decoded chunks peaked at 536 MB of the 537 MB limit;
- 40 jumps of the time slider showed the asked volume; the slice sliders, the wheel, a click in a panel and the keys moved as asked;
- the page is not wider than the window at 1400 or 400 px.

On H06 (4 runs, 171 MB packed): the same checks hold; exactness 0 of 240 combinations and 0 of 64,364 screenshot samples differ; 60.00, 60.00 and 60.03 volumes per second at 60 in one stream, side by side and difference; first frame 0.47 s, everything loaded 3.2 s; renderer 0.93 GB. The markers part played at 60 per second on a run with 77% of its volumes above 0.5 mm, so the marker stayed on for the whole 8 s; in H05, 61% of the volumes of the run are above it. In runs like these nearly every volume carries the marker.

#### What an agent saw in the packs

An agent (not a person) took screenshots of every pack in the page at 1400 by 900 px: per run, the volume with the largest framewise displacement side by side and as the difference, the temporal-mean difference, the temporal SD side by side, and the raw stream at that volume and the two before and after it. What follows is that agent's reading of those still images and of those five-frame sequences. Nobody watched a pack play, and no reading was checked by a second look.

- H01. Over the last volumes (framewise displacement up to 25.9 mm at the last volume) the raw brain slides against the contours and the sagittal slice shows horizontal bands of alternating brightness, which read as motion between the slices of one volume. The SD stills show a bright rim at the brain's edge in the raw stream and a weaker one in the preprocessed stream; they do not show the bands or when the movement happened.
- H02. At the volume of largest displacement (13.2 mm) the sagittal raw slice shows darker bands low in the head and a bright patch at the occipital pole that changes from volume to volume; the contours stay roughly on the brain. The SD stills show a rim at the frontal and occipital edge. The difference at that volume is a red and blue speckle with a blue rim.
- H03. The flagged run R4 is shown with the red warning box and "flagged" on its run button. Its raw stream sits visibly off the contours in every frame, and its temporal-mean difference shows a strong blue and yellow edge pattern, so the still already shows the offset. Frame to frame, the agent saw little change at the five volumes around the largest displacement (8.8 mm).
- H04. Around the volume of largest displacement (5.6 mm) single volumes show bands of alternating brightness across the sagittal and coronal slices, on one or two volumes and not the next. The SD stills show a speckled map without the bands.
- H05. The second run (3.4 mm mean from where its own registration puts it) sits off the contours in the raw stream, frontal pial contours lying above the brain; the temporal-mean difference shows it as a blue rim at the front and top and a bright rim at the back and bottom. Around the volume of largest displacement (12.2 mm) a single volume shows bands in the sagittal slice and the next a shift against the contours.
- H06. The flagged run R3 sits off the contours with the warning shown. In R4, the volume of largest displacement (13.3 mm) shows strong bands in sagittal and coronal and a brain displaced downward against the contours, while the volumes before and after it look like the rest of the run. The SD still marks the run as noisy but does not single out that volume.
- Q01. The five frames around its largest displacement (0.23 mm) look the same to the agent; the SD stills show vessels and CSF and no rim at the edge.
- Q02. Raw and preprocessed look alike at its one volume above 0.5 mm. The slab ends above the bottom of the brain, so the lower temporal lobes are outside the field of view in both streams.

Read together, and only as the agent's impression that the failure criterion has to test: where a run is off the anatomy as a whole (a flagged run, a run far from the anchor), the temporal-mean difference already shows it. What the stills did not show in these screenshots is when motion happens, the direction in which the brain moves against the contours in a given volume, and slice-wise bands confined to one or two volumes.

### What was not checked in stage 1

Beyond what each step lists above:

- No person has looked at a pack, played one or used the keys. Every statement here about what the pictures show is an agent's reading of screenshots and of short sequences of frames, made to describe them, not a judgement of a trained viewer.
- Whether the runs of any session really share a scanner frame. The flag rule is a limit of plausibility derived from 77 sessions of first subjects; there is no ground truth.
- The `AcquisitionTime` branch of the anchor rule on a rendered session: none of the eight selected sessions takes it (the single-run sessions have sidecars without a usable time, the multi-run sessions none of their own).
- A session whose anchor is the odd one, multi-echo runs, a raw series stored as `.nii`, and a session whose runs lie on different T1w-space grids: none occurred in the selection.
- Determinism of the packs of the selection (only the worked example was packed twice), and of any pack on another machine or with `--portable`.
- Whether distortion correction, where fMRIPrep applied it, brought a series onto the surfaces well: the packs record that it was applied and by which method, and the page says so, but no viewer has judged a correction. The reviewer's evidence below describes how much the streams differ in those sessions, not whether a correction is good.
- Any browser but headless Chromium 153 on Linux, a real display, a high-DPI screen, a network between server and page.
- Whether anything the page shows identifies a dataset. The page shows neutral ids, run ids, volume counts, the TR and whole-head EPI; the README's own measurement says that run structure comes close to identifying a dataset, and EPI is not defaced. More than that, a pack identifies its run outright: its manifest carries every run's framewise displacement and standardised DVARS to five decimals, and they equal the public confounds table exactly (the review compared them; for one run of H06 the first values are equal in both to all five decimals). Anyone who has the confounds tables of the public derivatives can match a pack to its run without repeating the ranking. The packs are not committed and must not be shared as if they were anonymous. That is a further reason why nothing functional is published from here until question 5 and the identification question are settled.

### What remains of stage 1

Steps 1 to 4 are built. Step 5, the failure criterion, is not, and step 6 is optional.

The failure criterion and its procedure. What follows is a proposal of the session that built stage 1, rewritten on 2026-11-14 around decision 7, for Tomas to fix; nothing of it has been run.

- The claim it tests: for the two uses of the viewer, the moving picture of the raw and the preprocessed stream lets a viewer judge something about a run that the stills of the same run (temporal mean and temporal SD of each stream, the difference of the means) do not.
- The judgements, written as questions on a form with fixed fields, per run. About the raw data: (R1) Does the head move within the run, when, and in which direction against the contours? (R2) Are there artefacts confined to single volumes or slices (bands, spikes, dropout in one volume)? (R3) Where is the distortion and dropout as acquired, against the contours? (R4) For a run that is not the anchor, how far did the head move since the anchor run, and is a flagged run a head that moved, or a registration that failed? About the preprocessing: (P1) Did head-motion correction remove the movement seen in the raw stream, and what remains (residual motion, bands that it cannot remove)? (P2) Where distortion correction was applied (the page says so), does the preprocessed stream lie on the white and pial contours where the raw stream did not, and where does it still not? (P3) Where none was applied, how far is the preprocessed stream off the contours, and where? (P4) Anything in the preprocessed stream that is not in the raw one (an artefact of the pipeline)?
- Who looks: two viewers who know fMRI, Tomas and one other, each alone. Neither has seen the packs before. The first viewer does not tell the second what was seen.
- In which order: the sessions of the selection and the worked example, in an order drawn with a recorded seed, not labelled. Per run, first the stills only (the page's mean and SD views of both streams and their difference, with the traces switched off by `t` so that the metric is not seen first), answering R1 to P4 as far as the stills allow, each answer with where it is (plane, region; a volume where the answer names one); then the moving picture (series view, raw, preprocessed and difference, at any rate and with scrubbing), answering the same questions again; the traces are switched on last. The page's line on what the pipeline did to the run is visible throughout, since P2 and P3 depend on it.
- What is written down for each question: the answer from the stills; the answer after the moving picture; whether the second adds or changes anything; for such a change, whether it could be seen in the stills once one knows where to look (yes, no, unsure); and the time spent on stills and on the moving picture.
- What is counted: a judgement "gains from motion" when the moving picture adds or changes an answer, the viewer marks it "no" for the stills, and the other viewer, shown the run and the place, agrees. Counted separately for the raw-data questions (R) and the preprocessing questions (P).
- What outcome makes the paper say that the moving picture adds nothing: no judgement gains from motion in either viewer's forms for either use, or only what the traces already give (that a volume has large framewise displacement) without anything about it that the traces do not say. The opposite outcome makes the paper say, for each use, which judgements the moving picture made possible on these runs, each with its run and volume. With this number of sessions it is a demonstration, not an estimate, and the paper says so.
- What the agent's look already suggests, for the viewers to confirm or reject, and not to be read as a result: in high-motion runs the stills show a bright rim of temporal SD at the brain edge, while the moving picture also shows when the motion happens, in which direction the brain moves against the contours, and slice-wise bands in single volumes (R1, R2). Nothing was looked at yet for P2 and P3 in the sessions with distortion correction.
- For P2 and P3 to be tested, the cases have to include runs with and without distortion correction and with each method; the present selection has five with (four fieldmap-less, one fieldmap-based) and three without, chosen by motion only ("Next steps").

The optional shift-detection test (step 6). Before coregistration is offered as something eyes can check: take a quiet run (Q01 or Q02), add known rigid shifts of 0, 1, 2, 3 and 5 mm along each axis to the session's registration (a copy of the anchor's transform, never the stored file), pack each, and have the viewers judge in a seeded blind order whether the raw stream sits on the surfaces. The pack format holds this without change. `minimal_layout/blind_shift.py` did one such trial with stills and one agent.

## How to run it

All commands are given at the repository root. The data go into a directory of your choice outside the repository, and results go into directories of their own, apart from the downloads. `-I` keeps Python from importing anything that lies next to downloaded data; the scripts find their sibling modules themselves.

The scripts need only the core dependencies of the package (numpy, scipy, nibabel, pandas, matplotlib, pillow, httpx, imageio-ffmpeg). In a clone that already has an environment nothing has to be installed, because `uv run` uses the environment as it is. In a clone without one, `just setup` makes it; the recipe is `uv sync --all-extras --group dev`. `uv sync --group dev` is not the same command and should not be used in a working clone: in a dry run on 2026-11-13 it listed 40 packages to uninstall, among them fastapi, jinja2, uvicorn, playwright and marimo (the `app` and `analysis` extras and the `browser` group), which the tests of the rater app need. `just setup` itself listed three in the same environment: playwright and greenlet, which are the `browser` group and which `just test-browser` installs again, and one package that the lock file does not name.

```
just setup                          # only in a clone that has no environment yet
D=/some/place/outside/the/repository

# the worked example: 31 files, 133 MB, 5.5 s here; then 5 FreeSurfer files, 27 MB, 1.5 s
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

### Stage 1

Start with the worked example; it needs no listing and no request beyond the two fetches above:

```
S=prototypes/func_viewer/stage1
uv run python -I prototypes/func_viewer/pipeline/fetch.py $D/data
uv run python -I prototypes/func_viewer/pipeline/fetch_freesurfer.py $D/fs
uv run bash $S/run_example.sh $D/data $D/fs/sub-01 $D/example     # 4 min 43 s here
uv run python -I $S/serve.py $D/example/pack                       # open the address it prints
```

`run_example.sh` runs every self-check of stage 1 that needs no listing, each with one line on the terminal: `check_session.py` (the anchor run's stream equals the feasibility study's per-run stream bit for bit, the session stream is deterministic, the between-run displacement is the reported one), `select_runs.py --self-check` (the selection rule on made-up tables), `pack_run.py` twice and `pack_digest.py` (the two packs are the same bytes), and `check_player.py` (exactness, frame rate, scrubbing, markers, screenshots). The pack of the worked example carries the id W01. `check_player.py` needs the Python package `playwright` (the repository's dependency group `browser`) and an installed Chrome, or a browser named by `SURFCENSUS_BROWSER`; `--no-browser` as fourth argument leaves it out. `$D/example` then holds the pack, the map of its run ids to file names (`map.tsv`, which names files, so it stays outside the repository), and the output of every check. Run so on 2026-11-14 (09:38 to 09:43 UTC, on the 4 shared CPUs, with the files of the worked example fetched on 2026-11-13), every step said ok: the six lines of `check_session.py`; the two packs of 36 files and 34,385,125 bytes had one SHA-256; and `check_player.py` found 0 of 240 combinations and 0 of 29,072 screenshot samples off the values computed from the NIfTI files, painted 60.01, 60.00 and 60.00 volumes per second at 60 in one stream, side by side and difference, with none repeated or skipped, and showed the first frame 0.47 s after opening.

The selection, from the listing to the packs. Each command is resumable; the first two send most of the requests:

```
uv run python -I $S/list_runs.py $D/listing                          # 7,150 requests, 3 min here
uv run python -I $S/check_listing.py $D/listing                      # no request
uv run python -I $S/rank_motion.py $D/listing $D/motion              # 9,373 requests, 8.9 GB, 8 min here
uv run python -I $S/select_runs.py $D/listing $D/motion $D/motion/selection
uv run bash $S/run_stage1.sh $D/listing $D/motion/selection $D/work $D/packs \
    --runs H05=1,3 --check H05 --check H06                           # 30 min here, half of it H05
uv run python -I $S/serve.py $D/packs/H01
```

`run_stage1.sh` takes, for each id of the selection (H01 to H06, Q01, Q02 unless ids are named), these steps: `fetch_run.py` fetches the session (at most 4 requests at a time, sizes and MD5 against the listing, resumable); `session_measure.py` measures the frame check and the correlation of the two temporal means; `pack_run.py` packs the session under the id; `check_player.py` runs where `--check` names the id; the downloaded series are deleted; `run_line.py` appends one line of measurements, by neutral ids only, to `$D/work/measurements.jsonl`. By default a session gives at most four runs: the anchor, the selected run, then the others in BIDS name order; `--runs ID=N,N` names the runs of one session by their number in it. `--runs H05=1,3` is how the selection was packed on 2026-11-14, for lack of disk room (below). An id whose pack and line exist is skipped, so the script can be started again. `$D/work` names datasets and files; `$D/packs` names nothing.

`fetch_run.py LISTING SELECTION ID OUT --plan` prints which runs would be fetched and how many bytes, and fetches only the raw sidecars, which the anchor rule needs.

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
| `stage1/check_player.py` | drives the page in Chromium: the page shows the values of the prototype's own resampling; frame rate, scrubbing, markers, memory, screenshots |
| `stage1/check_geometry.py` | the independent check: the raw stream of a few volumes recomputed with its own transform reader and resampling, compared value by value with the pack; where the raw stream lies on the anatomy (correlations, the header's other affine, shifts) |
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

Rewritten on 2026-11-14 after decision 8 (Tomas Knapen's statement of the governing purpose, D37 in `docs/decisions.md`). The earlier plan kept the functional part apart from the package: a subpackage that nothing imports (stage 2, after milestone 4) and a catalog and run directory of its own (stage 3, after milestone 5). That plan is superseded. The EPI movie interface becomes part of the surfcensus package and of its app, beside the surface interface, with the session as the unit a person looks at. The next step is to bring the pack and the page into the package behind `surfcensus review`, for a local folder that holds raw BIDS data, its fMRIPrep derivative and the FreeSurfer subjects. That work is specified in the package's documents (`docs/`), not here. This folder stays as the place where stage 1 was built and measured, and as the reference for the pack format (`stage1/pack_run.py`), the page (`stage1/player/`) and the checks (`stage1/check_geometry.py`, `stage1/check_player.py`).

### Stage 1, as it was planned

Steps 1 to 4 are built and were run on 2026-11-13 and 2026-11-14 ("Stage 1, as built" above); step 5 is proposed there and not run; step 6 is optional and not built. The steps as they were planned:

1. List what qualifies. A script that lists, anonymously and at most 4 requests at a time, every subject and run of the derivatives made with fMRIPrep 23.0.2 or older, and records for each run whether it has the T1w-space series, the confounds table, the scanner-to-T1w transform, the fsnative transform, the FreeSurfer surfaces under `sourcedata/freesurfer/` and its raw run in the raw bucket (run numbers compared as numbers). `survey/survey.py` has the listing functions. The list names datasets and stays outside the repository.
2. Rank by motion. Read the confounds table of every qualifying run (whole files; 0.3 to 3.4 MB each in the six sampled datasets; how many runs there are is not known yet and should be counted from the listing first). Fix the selection rule before looking at any image, for example the highest mean framewise displacement, at most one run per subject and two per dataset, and write down the rule and the seed if one is used.
3. Render the selected runs in both streams, in a form that plays at up to 60 volumes per second and scrubs to the exact volume. The measurements above are the basis for choosing between a video with every frame a key frame, tiles and mosaics; the surface app's player is built on its own render model, so a functional player is a separate small page (plain HTML and JavaScript, no build step, as the project does elsewhere).
4. Add what the frames of the worked example lacked: scrubbing through slices, the difference between the streams and the temporal SD as views of their own, contours as a layer that can be switched off, and markers that last longer than one volume.
5. State the failure criterion before anyone watches: if the moving picture shows nothing that the mean, the temporal-SD map and the difference stills of the same runs do not show, for judging the raw data or for judging the preprocessing (decision 7), the paper says so. A way to make that testable is to have each viewer first write down what the stills show for a run and then what the moving picture adds, run by run; the exact procedure is for Tomas to fix (open question 1; a proposal is under "What remains of stage 1").
6. Optional, from the critiques: a shift-detection test, in which viewers judge runs with known added shifts of 0 to 5 mm, before coregistration is offered as something eyes can check.

A proposal for Tomas that follows from decision 7 (2026-11-14; a proposal of the session that applied it, not decided): the cases should cover the effects the viewer is meant to show, not motion alone: runs with high and with low motion (head-motion correction), runs with distortion correction by each method fMRIPrep used (fieldmap-less SyN, phase-difference fieldmap, and others such as fieldmaps from reversed phase encoding where a derivative has them) and runs without it, and runs whose preprocessed series fits the surfaces well and badly. The listing of step 1 has the per-run summary reports (`figures/<stem>_desc-summary_bold.html`), so the distortion-correction method of every qualifying run can be read before anything is chosen, as motion was; the rule would be written down before any image is looked at, as the motion rule was.

### Ratings, only after a small rated pilot

Ratings and a blind-spot number, measured against the strongest numbers available for each criterion and with a rating screen that shows no metric. By decision 1 this was not planned for the functional part; whether the EPI interface records judgements at all is open in the package's documents (O27 in `docs/decisions.md`).

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

Question 1 of the earlier list ("ras bids") is answered by decision 6. The others stand, renumbered, and the packages of stage 1 added more. Where the build took a default, the default is named; each is his to change.

From the feasibility study:

1. The failure criterion of stage 1: is the proposal under "What remains of stage 1", rewritten on 2026-11-14 around decision 7 (questions about the raw data and about the preprocessing, stills before the moving picture, what is written down, what outcome makes the paper say that the moving picture adds nothing for either use), right, and who is the second viewer?
2. Which form of playback. Stage 1 chose a volume store of 8-bit lossless chunks played by a page (numbers under "Steps 3 and 4"); a video with every frame a key frame and image tiles were not built. Is that acceptable, and does the frame-to-frame noise have to survive exactly, as it does now?
3. How many hard cases, and by which rule. Default taken: six runs by mean framewise displacement, at most one per subject and two per dataset, and two quiet runs from the datasets of the first two hard cases. On this listing the caps and the size limit changed nothing. Mean framewise displacement favours runs that move throughout (61 to 82% of volumes above 0.5 mm in four of the six); should some hard cases be chosen by the largest single step, to have runs that are quiet but for one or two events?
4. The report also proposed a dataset with fieldmaps, a multiband 2 mm dataset and your own 7T data. Are those still wanted, now that the hard cases are chosen by motion?
5. May the datasets of the examples be named in the opinion paper, and is anything functional shown outside the lab as a whole head or cropped to the brain?

From stage 1:

6. Under one registration per session a run that is not the anchor sits off the surfaces by the head's movement between runs (0.8 mm in the worked example, 0.5 to 6.3 mm on average in the multi-run sessions that were packed), and the difference between its two streams is then mostly that offset. Is that what the viewer should show?
7. A flagged run (more than 10 mm from the anchor somewhere in its brain mask). Default taken: shown through the session's registration with a visible warning. The alternatives are its own transform with a note, or leaving it out. Is 10 mm the limit?
8. When the first run of a session is the odd one (1 of 77 surveyed sessions), does it stay the anchor? Default taken: yes, the rule as written.
9. The anchor rule falls back to BIDS name order in every multi-run session of the selection: no run there has a raw sidecar of its own. Should acquisition order be looked for elsewhere (inherited sidecars, `_scans.tsv`, which 308 of the 2,470 qualifying sessions have), or is name order enough? Default taken: name order.
10. A dataset without `ses-` folders counts as one session per subject, although its runs may come from several visits. Is that still the rule there?
11. Two selected subjects (those of H01 and Q01) have three sessions of one run each. Default taken: only the session of the selected run is rendered. Should the other sessions be rendered too, since decision 6 speaks of comparability across sessions?
12. Should the raw stream use the BLAS-free voxel map (`--portable`), at the price of the last bits of the anchor run (5 single voxels in 146 volumes of the worked example, by at most 1e-6)? Default: the numpy map, which keeps the feasibility study's bytes.
13. Multi-echo runs (4 qualifying runs) are left out of the ranking, since it is undecided which raw echo stands for the series before preprocessing. Leave them out for good?
14. The ids H01 to Q02 hide nothing from someone who repeats the ranking from the public buckets with these scripts. Is that acceptable for what the page shows, or should ids be assigned so that they cannot be re-derived?
15. Should the traces and markers be off when a pack is opened, so that the moving picture is seen before the metric? Is 512 MiB of decoded chunks the right default memory limit, given that the browser then held 1.3 to 1.6 GB on the long synthetic run? Should trace bands be widened for long runs, where one volume is about 2 px wide?
From the review of stage 1 and decision 7 (2026-11-14):

16. Should the cases be chosen to cover the effects the viewer is meant to show (head-motion correction at high and low motion, distortion correction by each method and none, good and bad fits to the surfaces) instead of by motion alone, by a rule written down before any image is looked at ("Next steps")? The present eight have distortion correction in five (four fieldmap-less, one fieldmap-based) and none in three, by chance of the motion ranking.
17. `check_geometry.py` fails a pack by limits that the session that wrote it chose (more than 0.1% of mask voxels off by more than 1 gray level; a gain in correlation of more than 0.05 for the header's other affine or for the pack's stills, of more than 0.03 for a shift of up to 3 mm). Are these acceptable as the test that the raw stream is in place?
