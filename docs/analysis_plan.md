# Analysis plan

Written before rating starts, so the main analyses are fixed in advance. Exploratory analyses are labelled as such.

## Outcome definitions

Per subject, the final label on criteria A and B (consensus or adjudicated). Primary binary outcome: major (2) on A or B. Secondary: any inclusion (1 or 2) on A or B.

## Prevalence

From the prevalence sample (phase 5), weighted to the subject population of each source:

- prevalence of the primary outcome per source (OpenNeuroDerivatives and embedded derivatives, reported separately);
- by dataset, as a dot plot with Wilson intervals, ordered by prevalence, datasets labelled by pseudonym (decision D10);
- by field strength, FreeSurfer major version, and T2/FLAIR-pial use, with a mixed-effects logistic model (dataset as random intercept) to separate these factors from dataset idiosyncrasy. Field strength and dataset are confounded; the model makes this explicit rather than resolving it.

Scope of the claim: prevalence in public OpenNeuro FreeSurfer derivatives. Datasets that shared FreeSurfer outputs are a selected subset, so the result does not estimate prevalence for FreeSurfer use in general.

## Detectability

From the detectability sample, with sampling weights:

- ROC and precision-recall curves of the composite score and of each component metric against the primary outcome;
- leave-one-dataset-out cross-validation for anything fitted;
- the blind-spot rate: among subjects whose composite score is below the 90th percentile (looks normal), the fraction rated major. This is the number that tests the paper's thesis. If it is near zero, the honest conclusion is that thickness screening suffices for this failure mode, and the paper should say so;
- the reverse: among high-scoring subjects, the fraction rated none, and what raters noted about them (thick-looking for anatomical reasons, other errors).

## Spatial detectability

From click marks resolved to fsaverage6 vertices:

- Census error maps: per-vertex frequency of final marks by class (A, B, C), smoothed with a small geodesic kernel, shown on the inflated and flattened fsaverage occipital surface. This shows where FreeSurfer fails, which no per-subject rating can.
- Marks reach the analysis as stored by the server: the tile pixel of the click, the voxel, tkr and scanner RAS, one hemisphere per mark, the nearest pial and white vertex of that hemisphere with their distances (null beyond the snap radius of 5 mm), and the fsaverage6 counterpart of the pial vertex, or of the white vertex for class C (`marks.vertex_surface`). Positives are defined in native space, from the mark and the vertex map, and carried to fsaverage through the counterparts.
- Vertex-level ROC: within rated subjects, the evaluable vertices are the search-region vertices that the subject's tiles show (`visible` in `vertex_map.npz`, decision D15), since raters cannot mark what no tile shows. Each evaluable vertex is positive if it lies within a tolerance (default 5 mm geodesic on fsaverage, with sensitivity analysis at 3 and 10 mm) of a final mark. The distance is taken from the vertex that the mark resolved to, whether or not that vertex is evaluable itself: a mark on a contour at the end of a sweep or at a tile edge can resolve to a vertex just outside the shown region (0.32 % of the pixels of the drawn pial contour on `ds000102` sub-01, at most 1.07 mm outside; `rendering.md`). Such a mark is kept and makes the evaluable vertices within the tolerance positive; its own vertex is not counted. The number of final marks whose vertex is not evaluable is reported. Score: the subject's z-map value at that vertex. Pooled across subjects, with a subject-level bootstrap for intervals.
- Cluster localization: for subjects rated major, the fraction whose largest outlier cluster contains or lies within tolerance of a final mark. This separates "the metric flagged the subject" from "the metric flagged the right place".
- Reference contamination: all of the above with the first-pass reference (all subjects) and the second-pass reference (rater-passed subjects only).

## Rater behaviour (exploratory)

Time per subject, and whether frames viewed or scrubbing predicts agreement with the consensus label. Spatial agreement between raters (see `rating_protocol.md`).

Skips. A rater can defer a subject (`rater_app.md`); it comes back, and no queue ends with a subject unrated. Reported: the number of skips per rater, and per subject the number of raters who skipped it and how often, from the `skips` table of the export, with the share of all assignments that were skipped at least once. A subject that many raters defer is a hard one, which is itself a finding. Three things the main analyses take from this:

- Time on subject. `ms_on_subject` of a rating is the time of the visit that ended with the submission. For a subject that was skipped first, the earlier visits are in `behaviour.ms_skipped_visits`; analyses of looking time use the sum and say so.
- Order. The assignment logic randomizes order per rater; a skipped subject is rated later than its place in that order, by the rater's own choice. Analyses of order effects and of drift use the position at which a subject was rated, and are repeated without the assignments that were skipped.
- First impression. A rating after a skip is a second look. The sensitivity analysis repeats the agreement and detectability estimates without ratings that followed a skip (`behaviour.n_skips` above 0).

A queue that is stopped before it is finished leaves skipped subjects without a rating from that rater; `surfcensus finalize` names such raters, and a subject that lacks a rater's rating is labelled from the others only, which the export shows (`final_method`, `n_ratings`).

## Effect on downstream measures (exploratory, optional)

For subjects rated major, the change in occipital parcel thickness compared with matched rated-none subjects in the same dataset. Optionally, for a handful of 7T subjects with retinotopy, the effect of manual correction on surface-projected pRF maps. This connects the census to the functional-preprocessing argument of the paper.

## Reporting

All analyses in marimo notebooks in `notebooks/`, reading only from `surfcensus export` outputs. Sampling weights come from the `selection` table of the export (one row per subject and reason, with its inclusion probability). The per-vertex tables that the spatial analyses read (`vertices.parquet`, `error_map.parquet`) are written in milestone 6. Figures are regenerated from the notebooks; no hand-edited numbers.
