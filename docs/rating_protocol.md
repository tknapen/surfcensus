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

## Calibration set

About 20 subjects from the pilot with consensus labels, covering every level of A and B at least three times, from at least two field strengths. Frozen after phase 0. Each has a short written justification shown to raters after calibration. Reference images for the `?` help screen come from this set.

## Spatial agreement

Marks from two raters on the same subject and class are compared by the distance from each mark to the other rater's nearest mark, on the native pial surface (geodesic, or Euclidean in tkr RAS as an approximation). Reported as the median distance and the fraction of marks with a counterpart within 10 mm. Calibration-set subjects have consensus marks, so a rater's spatial accuracy is also measurable during calibration.

## Agreement targets

Decided in phase 0 from the pilot agreement, before the main rating starts, and reported whatever the outcome. A plausible target is weighted kappa of at least 0.6 on A and B; if the pilot falls short, revise the scale wording rather than lower the target.
