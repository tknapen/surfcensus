# Rendering

## What a rater sees

Three panels side by side around the occipital pole: sagittal, coronal, axial. Each shows `T1.mgz` in grayscale with the white and pial surface contours from both hemispheres. All three slice positions sweep through a box around the occipital pole, forward and then back (ping-pong), in a loop. Each panel carries thin crosshair lines marking the current positions of the other two planes, so the rater can see where the three views intersect.

Two playback modes, selected in the app:

- synchronized: all three planes advance together, phase-locked across their ranges;
- one at a time: sagittal sweeps while the other two hold at the box centre, then coronal, then axial.

The rater can pause and scrub frame by frame. Scrubbing is the main way experienced raters work, so the renderer produces individual frames as well as a video.

## Relation to the original gist

The gist (`tknapen/85d9a23a09c95d15f4b23ba1965fa54d`) uses `freeview -cmd` with `-viewport sagittal`, `-slice x 127 127` and `-ss` per slice, then ffmpeg. It is kept as the `freeview` backend for publication figures and for comparison. The default backend is pure Python (nibabel and matplotlib). It needs no X server, FreeSurfer install or license, produces identical output on any machine, and parallelizes across processes.

## Geometry

Coordinates: FreeSurfer surfaces are in tkr RAS (surface RAS). `T1.mgz` is a conformed 256³ 1 mm volume; nibabel gives its voxel-to-tkr-RAS affine via `MGHHeader.get_vox2ras_tkr()`. Slices are taken along voxel axes, so no resampling is needed; the slice plane in tkr RAS follows from the affine.

Occipital pole: per hemisphere, the pial vertex with the most posterior y. The box centre in x is the midline, estimated as the midpoint between the medial-most pial vertices of the two hemispheres within the occipital ROI. In y and z, the centre is the mean of the two hemispheric poles.

Default box, all in mm relative to the centre, configurable:

- x: −30 to +30 (covers both medial occipital surfaces, the falx and the superior sagittal sinus);
- y: −8 (just behind the pole, to show the sinus confluence and skull boundary) to +45 anterior;
- z: −35 to +25 (covers the tentorium and transverse sinus below, the parieto-occipital region above).

Slice step 1 mm, so about 60 frames per sweep and 120 frames per ping-pong loop. Panel field of view: the box plus a 10 mm margin, so the rater sees just outside the brain.

The box is a reasonable default for adult brains. Pilot renders on infant, elderly or atypical brains decide whether the box needs to scale with brain size (for example by the distance between the occipital and frontal poles).

## Contours

Each surface is intersected with the slice plane: for each triangle, signed distances of its three vertices to the plane; triangles with sign changes contribute one segment between the two edge crossings. Vectorized in numpy per slice, this costs milliseconds for a 150k-vertex mesh. Precompute per-vertex signed distances once per axis and reuse across slices by offset. `trimesh.intersections.mesh_multiplane` is an alternative if it proves faster (optional dependency, to benchmark).

Segments are drawn with matplotlib `LineCollection`, antialiased, linewidth about 1 px at the output resolution.

Default colours: pial yellow (#FFD400), white surface red (#FF3B30), on grayscale. The gist used white pial and red white; white pial is hard to see against bright tissue, hence the change. Colours are configurable and the defaults must be checked for visibility on both 3T and 7T T1 contrasts in the pilot.

## Intensity window

Per subject, window from the 1st to the 99.5th percentile of `T1.mgz` within a dilated brainmask. A fixed window (FreeSurfer normalizes WM to 110) is an option, but the percentile window is more forgiving of residual bias field at 7T. The window is recorded in `render.json`.

## Output per subject

```
media/<subject_uid>/
  frames/sync_000.webp ... sync_119.webp      # synchronized mode, one ping-pong loop
  frames/seq_000.webp ...                     # one-at-a-time mode
  anim.mp4                                    # synchronized mode, H.264, for passive viewing and the gallery
  poster.webp                                 # middle frame, for thumbnails
  render.json                                 # see below
  vertex_map.npz                              # native vertices in the box -> fsaverage6, for click marks
```

`render.json` holds everything the app needs to turn a click into anatomy: the vox2ras-tkr and vox2ras affines of `T1.mgz`, the box, the intensity window, and for every frame of every mode and every panel the slice axis, voxel slice index, and a 3 × 3 affine from panel pixel coordinates to the two in-plane voxel indices (crop, scale and display flips included). It also stores the inverse, so the client can draw existing marks. A unit test round-trips random voxels through pixel space and back for every panel.

`vertex_map.npz` holds, per hemisphere, the indices and tkr RAS coordinates of native pial and white vertices inside the box (plus a 10 mm margin) and each vertex's fsaverage6 counterpart from the sphere mapping in `metrics.md`. This makes it cheap to resolve click marks to vertices after the Tier C files have been deleted. Size is well under a megabyte per subject (estimate).

Frame size: three panels of 320 × 320 px each, 960 × 320 px per frame by default. WebP at quality 80 should be on the order of tens of kB per frame, so a few MB per subject (estimate, to measure in pilot). For 5,000 subjects that is on the order of 10 to 20 GB of media, served statically.

Video: H.264 via imageio-ffmpeg, `-preset veryfast`, CRF 23, `yuv420p`, 12 fps. 120 frames gives a 10 s loop.

## Optional views

- Patch-centred view: a fourth small panel centred on the native location of the largest fsaverage outlier cluster from `metrics.md`, when one exists. Off by default during rating, so the rating stays blind to the metric.
- Whole-brain sagittal sweep, as in the gist, for subjects flagged "other" by raters.

## Performance target

About 120 + 180 frames per subject at 960 × 320 px with matplotlib Agg: target under 20 s per subject per core, excluding download. On a 32-core machine that is about 1 hour per 5,000 subjects of compute, so download throughput will dominate. Measure in pilot.
