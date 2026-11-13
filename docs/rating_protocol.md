# Rating protocol

Draft. Phase 0 revises the wording after two experienced raters rate the pilot set independently and compare. The app reads the current version from `protocol.yaml`; every rating stores the protocol version.

## Criteria

Each subject gets a value on three criteria plus optional flags. Raters judge the surfaces, not the image: a noisy T1 with correct surfaces is "none" on every criterion.

A. Superior sagittal sinus and falx. Does the pial surface of either hemisphere include the superior sagittal sinus, the falx or dura along the medial occipital or medial parietal surface?

B. Tentorium and transverse sinus. Does the pial surface include the tentorium, the transverse or straight sinus, or cerebellar tissue along the inferior occipital or posterior inferior temporal surface?

C. White surface. Is the white surface misplaced in the box: cutting into white matter, missing a gyral white-matter core, or including non-brain tissue?

## Scale, the same for each criterion

- 0, none: no inclusion visible in any frame.
- 1, minor: inclusion visible but confined. As a working definition: present in fewer than about 5 consecutive slices in any one plane, and not extending the cortex outward by more than about 2 mm. Would change local thickness, unlikely to change a parcel mean or a pRF map much.
- 2, major: inclusion over a larger extent, or sinus or tentorium clearly enclosed within the pial surface. Would visibly corrupt surface-based analyses of early visual cortex. A careful analyst would edit or exclude.

The numeric anchors in the minor definition are placeholders until the pilot settles them.

## Flags

- Unratable, with a reason: render failure, wrong orientation, missing surface, image too poor to judge.
- Other problem: anything else in the box worth noting (topological defect, skull inclusion away from sinuses, gross misregistration), with a note.

## Procedure

1. Play at least one full loop in synchronized mode before rating.
2. Pause and scrub through any frame where a contour touches or crosses a dark or bright band outside the cortex.
3. Rate A and B from the sagittal and coronal panels primarily; use the axial panel to confirm extent.
4. For every criterion rated minor or major, mark the error: click on the centre of each affected stretch and on both of its ends, in whichever panel shows it most clearly. Separate stretches get separate marks. Click on the misplaced contour where it crosses into the non-brain structure, not on the middle of the sinus, so the mark maps to the vertices that are wrong. Five marks per stretch at most; marks are for location, not tracing.
5. Check that each mark looks right in a second panel as the slices sweep past it.
6. When uncertain between two levels, choose the lower one and add a note. Raters are not asked for confidence; disagreement between raters measures uncertainty.
7. A subject can be left for later with the skip key (`k`). Nothing is stored for it, and it comes back after the subjects not yet shown; a rater's queue is finished only when every subject has a rating or is marked unratable. Skipping is for a subject one wants to see again with a fresh eye or after the others, not a way to avoid a hard one: the subject stays with the rater who skipped it, in the census as well, and every skip is recorded with its time.

## Skips

The rule is the same for a local rater, in the calibration stage and in the census: a skipped subject is deferred, never dropped, never handed to another rater, and never counted as a rating. Without that rule, which subjects a rater rates would depend on which ones the rater finds hard, and the prevalence estimate would lose exactly those. What is recorded: per assignment the number of skips and the time of the last one, and per skip an event with the client and server time and the time the subject was on screen in that visit. The analysis reports skips per rater and per subject (`analysis_plan.md`). A subject that comes back after a skip has been seen before, so its rating is not a first impression; the number of skips before a rating is in the export for that reason.

A first-time rater is shown a tour of five steps (the views, the rating rows, marks, moving on, help). It shows nothing of the subject, and its time is not counted as time on the subject. Whether raters read it is in the `tour_open` and `tour_close` events.

## Calibration set

About 20 subjects from the pilot with consensus labels, covering every level of A and B at least three times, from at least two field strengths. Frozen after phase 0. Each has a short written justification shown to raters after calibration. Reference images for the `?` help screen come from this set.

The set lives in `calibration.yaml` in the run directory: per subject the consensus value of each criterion, the consensus marks (class and tkr RAS position), the justification and the reference examples (criterion, value, panel, slice, caption). `surfcensus calibration freeze` writes it from the final labels and final marks of the pilot run, with empty justifications and no examples, which are then written into the file by hand; the file is never overwritten. Its SHA-256 is stored with every calibration submission. In milestone 4 the file feeds the reference examples of the help screen only. The calibration queue and the consensus feedback are team mode (milestone 5); there, a calibration rating cannot be revised once its consensus was shown to the rater.

## Spatial agreement

Marks from two raters on the same subject and class are compared by the distance from each mark to the other rater's nearest mark, on the native pial surface (geodesic, or Euclidean in tkr RAS as an approximation). Reported as the median distance and the fraction of marks with a counterpart within 10 mm. Calibration-set subjects have consensus marks, so a rater's spatial accuracy is also measurable during calibration.

## Agreement targets

How the pilot's two raters get final labels on milestone 4: each rates every subject on the same run directory, one after the other, blind (`surfcensus review PATH --run-dir DIR --rater NAME --blind`); `surfcensus finalize` then gives a subject rated by both the lower value where they are within one scale step on every criterion (method `consensus`, with the union of their marks, without the marks of a criterion whose consensus is none), and lists the subjects where they are two steps apart, or where one said unratable, with what each said. Those are decided together and rated again; the adjudication screen is milestone 5.

Decided in phase 0 from the pilot agreement, before the main rating starts, and reported whatever the outcome. A plausible target is weighted kappa of at least 0.6 on A and B; if the pilot falls short, revise the scale wording rather than lower the target.
