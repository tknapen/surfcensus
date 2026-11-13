This is the feasibility report as given to Tomas Knapen on 2026-11-13, unchanged below this line. README.md in this folder records the decisions taken on it and what came out when its numbers were checked again. The `results.json` that the report mentions is not in the repository, because it names datasets and paths of the machine it was written on.

# fMRI preprocessing viewer: feasibility report

Written 2026-11-13 from a study on OpenNeuroDerivatives listings, one prototype subject (`ds000102` sub-01), a prior-art search and three independent critiques. This is the text of the report as given in the chat; the per-part results it summarises are in `results.json` next to this file.

The feasibility study is back, and I just sent you its prototype: two videos, one frame and one summary map from a real subject. My answer is that the viewer can be added and is cheap to build, but the study changed three things I believed this morning, and the case for it now rests on a test that has not been run.

## What was measured

**The pipeline works on EPI.** The package's mesh slicer, contour renderer and video writer ran unchanged on `ds000102` sub-01 (two runs, 292 volumes). The raw series under the one stored rigid transform matches the mean of fMRIPrep's T1w-space series at r = 0.9993 inside the brain mask, and the contours sit on the EPI's tissue boundaries. The side-by-side video took about 23 s on one core and is 2.9 MB.

**"The preprocessed result we can just get from OpenNeuro" exists for 71 of the 308 datasets.** The 72 derivatives made with fMRIPrep 23.0.x or older hold preprocessed series. The 236 made with 23.2.0 or newer were written at output level `minimal`: reference images and transforms, with no preprocessed BOLD and no confounds table. The raw-plus-rigid view can be built for essentially all of them (307 of 308 have the transform). A "preprocessed" stream for the 236 would be our own resampling of fMRIPrep's stored fits, without distortion or slice-timing correction, and so not fMRIPrep's output.

**In this subject, raw and preprocessed cannot be told apart by eye.** This holds in every frame, including the two with framewise displacement of 0.63 and 0.77 mm. The subject barely moved (largest translation 0.69 mm), so the comparison is untested, not refuted. What the frames do show is static: the fit of the contours and the signal dropout. Motion appears only in the difference image and the temporal-SD map, as a rim at the brain edge.

**Cost.** A collection of 5,000 subjects with every run is roughly 1.7 to 5.7 TB, against about 200 GB for the surface census. A whole experiment takes 24 to 148 s to watch once.

## What the critiques add

Three independent reviewers attacked the proposal. Their strongest points:

- **Video over stills is unproven.** The published catches by eye (flipped images, dropout, distortion, misalignment) were all visible in still images, and no study compares movies with stills ([Lepping et al. 2023](https://doi.org/10.3389/fnins.2023.1076824); [Taylor et al. 2023](https://doi.org/10.3389/fnins.2023.1205928)). CONN and AFNI already play EPI series locally ([Morfini et al. 2023](https://doi.org/10.3389/fnins.2023.1092125)). What would be new is scale, anatomy contours, blinded multi-rater rating, and a measured blind-spot rate.
- **Coregistration may not be judgeable by eye at 3 mm voxels.** On a second subject, a correct and a headers-only registration looked alike; only a contrast number separated them. If numbers beat eyes there, the functional thesis inverts.
- **Rater agreement on functional QC is moderate**, with kappa 0.44 to 0.60 in [Williams et al. 2023](https://doi.org/10.3389/fnins.2023.1070413).
- **Identifiability.** EPI is not defaced (eyeballs are plainly visible), and run counts and TR come close to fingerprinting a dataset, so the pseudonym courtesy would not carry over as it stands.

## Recommendation

Build it as a viewer for the opinion paper, in stages, and treat a functional census as a separate decision.

1. **Now, outside the package:** render hard cases (high motion, a fieldmap dataset, multiband 2 mm, your 7T data), add slice scrubbing and the difference and temporal-SD maps, and have you and a collaborator watch them. State the failure criterion in advance: if the video shows nothing that a mean, SD and difference still sheet does not, the paper should say so.
2. **After milestone 4:** a single-subject command on a local fMRIPrep folder, isolated from the surface path.
3. **After milestone 5:** run-level inventory and a view-only tab, with its own catalog file so the surface census is never migrated.
4. **Only after a small rated pilot:** ratings and a blind-spot number.

## Decisions for you

1. **Viewer or census?** I recommend the viewer; a census only if stages 1 and 4 justify it.
2. **Hard cases.** Can you name two or three subjects whose functional preprocessing you know went wrong? Otherwise I pick the highest-motion runs from the confounds files of the older datasets.
3. **The 236 newer datasets.** Either restrict the two-stream comparison to the 71 with a real preprocessed series, or show a rebuilt stream labelled as ours. I recommend the first for anything that claims to show fMRIPrep's result. For a user's own local fMRIPrep output the series is normally on disk, so this limit mostly concerns the OpenNeuro-wide route.
4. **The prototype scripts** (about 600 lines) live in this session's scratch space and will be lost when it ends. Shall I commit them under `prototypes/func_viewer/`, outside the package and the wheel?

Nobody has played the two videos yet; the agents could only read decoded frames. Whether 12 volumes per second is watchable, and whether you see anything in the motion, is the first piece of evidence only you can supply.
