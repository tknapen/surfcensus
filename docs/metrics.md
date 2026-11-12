# Quantitative metrics

Cortical thickness is the quantitative anchor: a number per subject and per location that can always be returned to when choosing which brains to look at and when testing what the raters saw. Outlier detection is done primarily in fsaverage space, where every subject's thickness is compared with the distribution at the same anatomical location across many subjects.

## The failure mode and what it should do to thickness

When the pial surface absorbs the superior sagittal sinus, the transverse sinus or the tentorium, it moves outward past the true gray/CSF boundary on the cortical faces adjacent to those structures. The white surface is usually unaffected; sometimes it is also pulled when bright non-brain tissue gets labelled as white matter. The expected signature is local:

- inflated thickness in contiguous patches, not across a whole parcel;
- on the medial occipital and medial parietal surface next to the falx and superior sagittal sinus;
- on the inferior occipital and posterior inferior temporal surface over the tentorium and transverse sinus;
- often near or at the thickness maximum, because capture can add several millimetres.

This is a hypothesis about the signature, derived from where the structures sit. The pilot tests it against rated and click-marked examples before scores are trusted.

## Why fsaverage space

Normal thickness varies strongly across cortex, and occipital cortex in particular varies over short distances (thin calcarine cortex next to thicker surrounding cortex). A patch of 3.5 mm cortex is unremarkable in one place and extreme in another. Parcel means dilute local patches, and per-subject native ROIs cannot say what is normal at a given location. Resampling every subject to a common mesh gives a per-vertex normative distribution, so each subject's thickness becomes a z-map. It also places rater click marks (see `rater_app.md`) on the same vertices, so detection can be evaluated spatially, not only per subject.

## Resampling to fsaverage

Inputs per subject: `surf/?h.thickness` and `surf/?h.sphere.reg` (about 0.6 and 5.3 MB per hemisphere; both present in OpenNeuroDerivatives subjects, verified on `ds000102` sub-01).

Target mesh: fsaverage6 by default (40,962 vertices per hemisphere), with fsaverage (163,842) as an option. Source of the target sphere, in order of preference:

1. `$FREESURFER_HOME/subjects/fsaverage/surf/?h.sphere.reg` if FreeSurfer is installed;
2. the `fsaverage/` directory shipped next to the subjects in the same fMRIPrep derivative (present in `ds000102-fmriprep/sourcedata/freesurfer/`, with `surf/lh.sphere.reg` at 5.9 MB), which also matches the FreeSurfer version that produced the subject's registration;
3. nilearn's `fetch_surf_fsaverage`, which returns sphere meshes for fsaverage3 to fsaverage7 as GIFTI. Whether its coordinates match FreeSurfer's `sphere.reg` convention (radius 100, same orientation) is to verify.

fsaverage templates are fetched and cached on first use, not bundled in the wheel. Lower-resolution icosahedral meshes are believed to be the first N vertices of fsaverage (nesting), so fsaverage6 can be derived from the fsaverage sphere by truncation (verified for fsaverage5 on 2026-11-12, inferred for fsaverage6; see `CLAUDE.md`).

Algorithm: normalize both spheres to unit radius. For each target vertex, query a KD-tree of subject sphere vertices for the k nearest, test the faces in their one-rings for containment, and interpolate thickness barycentrically. Fall back to nearest neighbour if no containing face is found. This is pure numpy and scipy. FreeSurfer's `mri_surf2surf` uses a different default mapping (to verify which), so values will differ slightly; the pilot compares the two on a few subjects and reports the difference.

Medial wall and non-cortex: masked using fsaverage's `label/?h.cortex.label`.

Storage: per subject, two float16 arrays (one per hemisphere) of fsaverage6 thickness, about 160 kB per subject, written to `tables/fsavg/<subject_uid>.npz`. 5,000 subjects take under 1 GB.

Caveat: FreeSurfer's spherical registration is driven by folding patterns (curvature and sulcal depth), so a misplaced pial surface should affect registration little. A misplaced white surface could. Subjects rated major on the white-surface criterion are analysed separately.

## Per-vertex scoring

Global normalization: subtract each subject's median cortical thickness over the whole hemisphere outside the occipital search region. This removes offsets from age, scanner and contrast that shift thickness everywhere.

Reference distribution at each vertex: median and MAD of normalized thickness across reference subjects in a stratum. Strata, when they hold at least 30 subjects: dataset; else field strength × FreeSurfer major version; else all subjects. The MAD map is lightly smoothed on the mesh to stabilize it.

Subject z-map: z_v = (t_v − median_v) / (1.4826 × MAD_v).

Circularity: if capture is common at some vertices, it shifts that vertex's median up and inflates its MAD, which hides exactly the errors we look for in exactly the place they occur. The first pass uses all subjects; after rating, the reference is rebuilt from rater-passed subjects only, and all scores are recomputed. The paper reports detection with both references, since the difference measures how much contamination of the reference costs.

Occipital search region: defined once on fsaverage (the occipital Desikan parcels plus posterior precuneus and posterior fusiform, cut at a fixed distance anterior to the occipital pole) and shipped as a label with the package.

## Subject-level metrics derived from the z-map

- Largest outlier cluster: connected components on the fsaverage mesh of vertices with z above a threshold (default 3) within the search region. Report area (mm², using the subject's own resampled vertex areas, or fsaverage white-surface vertex areas, to decide in pilot), peak z, and centroid vertex.
- Cluster mass: sum of z over the largest cluster, which combines extent and severity.
- Medial and inferior split: the same cluster metrics restricted to the medial and inferior parts of the search region, defined on fsaverage by location. This replaces per-subject direction-of-normal ROIs, which are no longer needed for scoring.
- Control: the same metrics in a frontal control region, to separate subjects with occipital-specific outliers from subjects with noisy surfaces everywhere.

Each cluster centroid maps back to the subject's native surface (via the barycentric mapping) and to tkr RAS on the native pial surface. The renderer can then centre an optional view on it, and the review screen can show it.

## The 5 mm ceiling

`mris_thickness` documents a default maximum thickness of 5 mm. How vertices exceeding it are stored (clamped to 5 mm, or set to 0) is not stated on that page (to verify in source and empirically). Clamping produces a pile-up at 5.0 that is itself a sign of capture. Zeroing would make captured cortex look thin and would also corrupt interpolation on fsaverage. In that case zero-valued vertices inside the cortex label must be treated as missing and counted separately. On `ds000102` sub-01 (FreeSurfer 6.0.1) values are clamped at exactly 5.0 and in-cortex zeros are negligible (5 vertices; see `CLAUDE.md`). Phase 0 still inspects thickness histograms of known-bad subjects and of a FreeSurfer 7 subject before this is treated as settled. Either way, the ceiling fraction (fraction of native search-region vertices at or above 4.9 mm, or at exactly 0) is reported as its own metric.

## Regional metrics (Tier A, every subject, no surfaces needed)

From `?h.aparc.stats`: `ThickAvg` and `ThickStd` for the occipital parcels and a frontal control set. These are cheap and available even when surfaces are missing, so they serve as a fallback score and as a baseline in the detectability analysis. They show how much the fsaverage z-maps add over what anyone can compute from stats files.

From `aseg.stats` or the log: surface holes or Euler number, as a general quality covariate. Qoala-T found surface holes to be the most important single predictor of manual include/exclude ratings (see `prior_art.md`).

## Intensity metric (Tier C, rendered subjects)

Sample `T1.mgz` along each native vertex's line from white to pial at a few fractional depths. Dura and sinus differ from gray matter on T1, but the sign depends on sequence, flow and contrast agent. So the test is an outlier test against the subject's own control-region profile: the fraction of search-region vertices whose outer-depth intensity falls outside the control region's 1st to 99th percentile. Resampled to fsaverage like thickness, it can join the z-map as a second channel.

## Composite score

The maximum over a small set of z-scored subject-level metrics: largest-cluster mass (medial and inferior), ceiling fraction, regional occipital-to-control ratio. The maximum suits a failure that can appear in one place only. The composite is a ranking device for selection and review, not a classifier. Its predictive value is estimated against ratings in `analysis_plan.md` and reported, never assumed.

## Local users

The shipped normative reference (see `architecture.md`) is a set of per-vertex median and MAD maps on fsaverage6 per stratum, about 0.7 MB per stratum. Local subjects are resampled and z-scored against it. When a local set has at least 30 subjects, a within-set z-map is also computed.

## What not to do

- No fixed universal thresholds in the paper's main analysis. Thresholds are for cluster definition, selection and display, and the analysis reports sensitivity to them.
- Do not train a classifier on the census ratings and then report its accuracy on the same ratings. Anything fitted is evaluated by leave-one-dataset-out cross-validation.
