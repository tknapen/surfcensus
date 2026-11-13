# Phases of the search

The census proceeds through nine phases. Each phase has a defined input, output, and a check that must pass before the next phase runs at full scale. Local users run phases 2, 4, 6 and 7 implicitly through `surfcensus review`, which also selects every subject; `surfcensus inventory`, `surfcensus metrics` and `surfcensus render` run the phases one at a time.

Cheap phases run on everything; expensive phases run on what the cheap phases select. The guiding cost asymmetry is that downloading and computing are cheap compared with rater time, so the pipeline spends compute freely to make every rated minute informative.

## Phase 0: pilot

Goal: fix the file layout assumptions, the renderer and the rating scale on a small sample before anything runs at scale.

- Pick 5 OpenNeuroDerivatives datasets that differ in field strength and population, about 10 subjects each, plus 10 subjects from Tomas's own 7T data where failures are known.
- Run every later phase by hand on these 60 subjects.
- Tomas and one other experienced rater rate all 60 independently, then discuss disagreements and fix the scale wording in `rating_protocol.md`.
- Inspect the thickness histograms near the 5 mm ceiling to settle how over-maximum vertices are stored (see `metrics.md`).
- Compare the package's fsaverage resampling with `mri_surf2surf` on a few subjects.
- Measure time per subject for rating with and without marking.

Exit check: renderer output agrees with Freeview on the same slices (visual comparison, a handful of subjects); the two raters reach acceptable agreement on the revised scale; the calibration set (about 20 subjects with consensus labels covering every scale level) is frozen.

## Phase 1: discovery

Input: nothing. Output: `datasets` table.

- List `fmriprep/` prefixes in `s3://openneuro-derivatives`.
- Enumerate raw OpenNeuro datasets (GraphQL or bucket listing) and list their `derivatives/` prefixes to depth 3.
- Match FreeSurfer subject directories by pattern: a directory containing `surf/lh.pial` (or `lh.pial.T1`/`.T2`) and `mri/T1.mgz`, excluding `fsaverage*`.
- Record source, accession, pipeline, version where visible, and the derivative ref.
- Assign each dataset its keyed pseudonym (decision D10) and check for collisions. Requires the dataset key; discovery refuses to run without one.

Exit check: count of datasets with at least one FreeSurfer subject, per source. This is the first reportable number.

## Phase 2: inventory

Input: `datasets`. Output: `subjects`, `files`.

- One prefix-scoped listing per subject directory. Record presence, size and ETag (for a local source, the modification time) for every file role.
- Fetch Tier A files (stats, build stamp, the three recon-all logs, whole; see `data_sources.md`) and parse provenance.
- Mark a subject `complete` if every file a render cannot do without is present (T1, white and pial of both hemispheres), and record separately whether it has the stats files (`tier_a`) and the thickness and `sphere.reg` files (`tier_b`). A directory with `mri/` and `surf/` that lacks render files is recorded as incomplete, with the roles it lacks.
- A subject whose listing is what the catalog already holds is skipped without reading a file. A subject that the listing no longer contains is marked absent; its render and ratings stay.

Exit check: number of complete subjects by source, FreeSurfer version, and T2/FLAIR-pial use. A second reportable number. Also decides whether the remaining phases fit in time and storage (Tier B 13 to 14 MB per subject, Tier B plus C 36 to 43 MB).

Built (milestone 2) for local sources: `surfcensus inventory --run-dir DIR --source PATH`.

## Phase 3: enrichment

Input: `subjects`. Output: age, sex, field strength, manufacturer filled where available.

- Fetch `participants.tsv` and one T1w sidecar per raw dataset (sidecars are often inherited at dataset level).
- Map heterogeneous column names to canonical fields with a reviewed mapping table in `src/surfcensus/enrich/columns.yaml`.

Exit check: fraction of subjects with field strength and age known. If field strength is mostly missing, use dataset-level stratification instead (see `metrics.md`).

## Phase 4: metrics

Input: complete subjects. Output: `regional.parquet`, `vertex_summary.parquet`, `scores.parquet`.

- 4a, all subjects, Tier A: regional thickness from `?h.aparc.stats`, surface holes or Euler number from `aseg.stats` or the log.
- 4b, all subjects, Tier B: resample thickness to fsaverage6 via `sphere.reg`, store per-subject maps (see `metrics.md`). Tier B files are deleted afterwards unless retained by config or needed for rendering.
- 4c, census-wide: build per-vertex reference maps (median, MAD) per stratum from all subjects (first pass), compute each subject's z-map, outlier clusters, ceiling fraction and control-region metrics.
- 4d, rendered subjects only, Tier C: T1 intensity sampling between white and pial. Runs inside phase 6 while T1 and surfaces are on disk.
- Scoring: composite score and rank.

Exit check: score distributions per dataset look sane (no dataset whose whole distribution is shifted by a parsing bug); the pilot subjects' scores agree with the pilot ratings at least roughly.

Built (milestone 3) for local sources: `surfcensus metrics --run-dir DIR`, with the stages `regional` (4a), `fsaverage` (4b), `reference` (4c: the reference of the set and the clusters) and `score`, and `native`, which gives cluster centroids their native position from the vertex maps of the renders and also runs at the end of `render`. `regional` takes every present subject with at least one `?h.aparc.stats` file and `fsaverage` every one with a thickness file, complete or not; each stores the SHA-256 of the files it read in the catalog, and each removes the rows, the map and the ledger row of a subject that had such files at an earlier call and has none now. The maps need an fsaverage template (`metrics.md`); without one, and for fewer than 30 subjects, the scores are the ranking of decision D16. 4d and the shipped reference are not built. Run on 34 real subjects on 2026-11-13; what was measured is in `metrics.md`.

## Phase 5: selection

Input: `scores`. Output: `selection` table with seeds.

Rendering everything is affordable; rating everything may not be. Selection determines the rating sample, not the rendering set.

- The prevalence sample: stratified random sample within datasets, proportional to dataset size with a minimum per dataset, blind to scores. This sample estimates prevalence.
- The detectability sample: stratified by composite-score decile, oversampling the top decile and including every subject beyond a fixed extreme threshold. This sample estimates how scores relate to ratings.
- The two samples overlap; each selected subject records why it was selected, so the analysis can weight correctly.
- If rater capacity allows, rate everything and drop the distinction.

Exit check: expected rater hours (n selected × raters per subject × seconds per subject) fit the rater budget.

## Phase 6: render

Input: selected (or all) subjects. Output: `media/<subject_uid>/`.

- Streaming batches: fetch Tier C (and Tier B if already deleted) for a batch, render, write `vertex_map.npz`, compute 4d intensity metrics, delete source files, next batch. Disk use stays at batch size × 39 MB.
- Render parameters and the config hash are written to `render.json`.
- Failed renders (missing or unreadable files, degenerate meshes, a file that changed since the inventory) are recorded with a reason, not retried silently: a failed subject is tried again when its files or the render settings change, or with `--retry-failed`. Every present subject that lacks a render file gets a `missing surface` row, so that it appears in review and export with its reason.
- A census run renders every complete subject by default (`select.render_scope`); a local run renders its selection, which is every complete subject until one is made. The rater app and `finalize` read the selection of a local run the same way (every present subject until one is made), so a run of the phase commands can be rated without `review`.

Exit check: a random 2% of renders is opened and inspected by eye. Blank panels, missing contours, wrong orientation or pole misplacement stop the phase. `surfcensus render --check-sample` draws the sample from the done renders with a seed that it names, and writes the list and a page of posters to `logs/`.

Built (milestone 2) for local sources: `surfcensus render --run-dir DIR`. The 4d intensity metrics follow in milestone 5.

## Phase 7: rating

Input: media, selection. Output: `app.sqlite`.

- Raters complete the calibration set first. The app then serves assignments (see `rater_app.md` and `rating_protocol.md`).
- Each subject in the sample is rated by at least two raters; 5% are re-served to the same rater later for intra-rater reliability.
- Raters click-mark every error they rate minor or major; the server resolves marks to native and fsaverage6 vertices on submission.
- Disagreements beyond one scale step go to adjudication, where the final mark set is also fixed.

- A rater can skip a subject; it comes back after the others and stays with that rater, so no queue ends with a subject unrated and no subject changes hands by being skipped. Every skip is recorded.
- A selected subject whose render failed gets an unratable final label with the reason of the failure and is not offered.

Exit check: inter-rater agreement on the main criteria above the threshold fixed in the pilot; all sampled subjects have final labels.

Built (milestone 4) for local runs: `surfcensus review PATH` (phases 2, 4, 6 and 7 in one command), `surfcensus serve`, and `surfcensus finalize`, which writes the final labels of two or more local raters by the one-step rule and lists the subjects that need a decision. The team queue (calibration, k raters per subject, repeats), tokens, adjudication and the dashboard are milestone 5.

## Phase 8: export and analysis

Input: everything. Output: analysis tables, figures, a public gallery, the normative reference, a released ratings dataset.

- `surfcensus export` joins ratings, scores and provenance into one table per analysis unit: `subjects`, `scores`, `selection`, `ratings`, `marks`, `final_marks`, `clusters`, `datasets`, `behaviour` and `skips`, as CSV or parquet, with a `manifest.json` that holds the run id, the stage hashes, the template and reference ids, the protocol and calibration versions and hashes, how the rating queue is ordered (blind or by score, its seed, the repeat settings), and the row count and SHA-256 of every file. It works on a run without ratings. `selection` holds the rows of the catalog: it is empty for a local run that `review` never selected in, although every present subject of such a run counts as selected. An export into a directory that holds an earlier one replaces it. An output directory in a source is refused, and so is one among the run's own files (`rater_app.md`, Commands). CSV has no types, so identifiers are read as text. Built (milestone 4).
- Two profiles. `analysis`: every subject, pseudonymous, for the team's notebooks; the only profile of a local run. `release` (a census run): selected subjects only in every table; no age, sex, manufacturer, pipeline version or notes; field strength binned to 1.5, 3 and 7 T; the subject count of a dataset binned (1 to 9, 10 to 29, 30 to 99, 100 or more); the sampling probability to two significant digits. Raters are `R-01`, `R-02`, ... in every export of a census run.
- `surfcensus reference build` rebuilds the per-vertex fsaverage reference from rater-passed subjects, rescoring every subject against it (second pass), and writes it into the package data for the next release.
- Census error maps: per-vertex frequency of final marks on fsaverage, by class.
- Analyses as in `analysis_plan.md`.
- Release: ratings, scores and provenance under CC0, with datasets identified by pseudonym (decision D10) and subjects by `subject_uid`. The pseudonym-to-accession mapping is withheld; whether and when to reveal it is decision O2.
- Public gallery: pseudonyms only, with coarse per-dataset descriptors (field strength, binned subject count).
