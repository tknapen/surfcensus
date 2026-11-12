# Prior art

What exists, what this package takes from it, and what it adds. Checked 2026-11-01 unless marked.

## fsqc (Deep-MI, formerly qatools-python)

Repository `github.com/Deep-MI/fsqc`, on PyPI as `fsqc`. Computes per-subject FreeSurfer QC metrics (WM and GM SNR, white/gray contrast, holes and defects, topology-fixing time, Talairach rotation, MRIQC-style image metrics), flags outlier structures by sample-based rules (1.5 × IQR, ±2 SD, minimum 10 subjects) or normative bounds, and optionally writes screenshots with aseg and surface overlays, surface renderings, skull-strip overlays and HTML summary pages. Its README does not describe a rating interface or anything specific to sinus, dura or occipital pial errors.

Relation: complementary. surfcensus can optionally import fsqc metrics as covariates. It does not duplicate generic QC.

## Qoala-T

Klapwijk, van de Kamp, van der Meulen, Peters, Wierenga. Preprint bioRxiv `10.1101/278358`; journal version in NeuroImage 2019 (to verify volume and pages). Repository `github.com/Qoala-T/QC`. A supervised random-forest model that predicts manual include/exclude ratings from FreeSurfer-derived measures (Desikan thickness and area, subcortical volumes, surface holes), giving a 0 to 100 score and recommending manual checks between 30 and 70. Surface holes were the most important predictor.

Relation: evidence that summary measures predict global quality ratings. surfcensus asks a narrower question: whether they catch a specific, local, common failure. Qoala-T's scores could be computed as a baseline comparator in the detectability analysis.

## Mindcontrol

Keshavan et al., "Mindcontrol: a web application for brain segmentation quality control" (bioRxiv `10.1101/090431`; journal version in NeuroImage 2018, to verify). A Meteor and MongoDB web dashboard linking group statistics to individual volumes in the Papaya.js viewer, with Pass, Fail, Edited and Needs-Edits labels, notes, task assignment and voxel editing; images hosted externally.

Relation: the closest prior rater app. surfcensus differs in being pip-installable with no JavaScript build or database server, in serving pre-rendered targeted animations instead of a free 3D viewer (faster per subject, standardized across raters), and in being built for blinded multi-rater reliability measurement.

## braindr

Keshavan, Yeatman, Rokem, "Combining citizen science and deep learning to amplify expertise in neuroimaging", Frontiers in Neuroinformatics 2019, doi `10.3389/fninf.2019.00029`. Crowdsourced swipe-style pass/fail ratings of 2D slices, then a deep network trained to amplify expert labels.

Relation: shows that fast, image-level ratings scale to large datasets and can train models. The released surfcensus ratings could serve the same purpose for this failure mode.

## fMRIPrep reports

fMRIPrep's per-subject HTML reports include surface reconstruction overlays on a fixed set of slices. They are designed for checking one subject at a time and use a mosaic of slices across the brain, which may not include the slices where occipital capture is visible.

Relation: the paper's broader argument. These reports exist and are rarely inspected; surfcensus measures what that costs for one failure mode.

## Mailing-list evidence for the failure mode

The FreeSurfer mailing list has recurring reports of dura inclusion in the pial surface near the occipital lobe and cerebellum across many subjects in a dataset (e.g. threads from 2019 and 2025 at `mail.nmr.mgh.harvard.edu`). These are anecdotes, without prevalence estimates, which is the gap the census fills.

## To check before writing the paper

- VisualQC (Raamana) and its FreeSurfer modules (not yet checked).
- ENIGMA cortical QC protocols (not yet checked).
- Any published prevalence estimates of dura or sinus inclusion in FreeSurfer surfaces (a literature search, not yet done).
