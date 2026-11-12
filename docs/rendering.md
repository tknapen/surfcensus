# Rendering

## What a rater sees

Three panels side by side around the occipital pole: sagittal, coronal, axial. Each shows `T1.mgz` in grayscale with the white and pial surface contours from both hemispheres. All three slice positions sweep through a box around the occipital pole, forward and then back (ping-pong), in a loop. Each panel carries thin crosshair lines marking the current positions of the other two planes, so the rater can see where the three views intersect.

Two playback modes, selected in the app:

- synchronized: all three planes advance together, phase-locked across their ranges;
- one at a time: sagittal sweeps while the other two hold at the box centre, then coronal, then axial.

The rater can pause and scrub frame by frame. Scrubbing is the main way experienced raters work, so the renderer produces individual slice tiles as well as a video. The rater can also hide the contours to see the tissue under the line (decision D11).

## Relation to the original gist

The gist (`tknapen/85d9a23a09c95d15f4b23ba1965fa54d`) uses `freeview -cmd` with `-viewport sagittal`, `-slice x 127 127` and `-ss` per slice, then ffmpeg. It is kept as the `freeview` backend for publication figures and for comparison. The default backend is pure Python (nibabel, numpy and matplotlib). It needs no X server, FreeSurfer install or license, produces identical output for a given input, config and locked environment (decision D18), and parallelizes across processes.

## Geometry

Coordinates: FreeSurfer surfaces are in tkr RAS (surface RAS): x to the subject's right, y anterior, z superior, in mm. `T1.mgz` is normally a conformed 256³ 1 mm volume; nibabel gives its voxel-to-tkr-RAS affine via `MGHHeader.get_vox2ras_tkr()`. Nothing in the renderer assumes 1 mm or 256³: all geometry is in mm and is converted through the affine, so `-hires` volumes work (decision D18).

Voxel convention: an integer voxel index is the centre of that voxel, so voxel `i` covers `[i - 0.5, i + 0.5)` along its axis. This is the convention of the affine. Whether Freeview draws the image with the same half-voxel placement relative to the contours is part of the by-eye comparison.

Axis alignment: slices are taken along voxel axes, so no resampling is needed. This requires each voxel axis to be parallel to one tkr axis, which holds for every FreeSurfer `mri/` volume. A volume whose voxel-to-tkr matrix is not a scaled signed permutation (tolerance 1e-4 relative) is refused, and the render is recorded as failed with reason `wrong orientation`.

Occipital pole: per hemisphere, the pial vertex with the most posterior y (decision D12). In y and z, the box centre is the mean of the two hemispheric poles. In x, the centre is the midline, estimated as the midpoint between the medial-most pial vertices of the two hemispheres (largest x on the left, smallest x on the right) among vertices whose y and z fall inside the box ranges below. If either hemisphere has no vertex in that range, the centre x is the mean x of the two poles.

Default box, all in mm relative to the centre, configurable:

- x: −30 to +30 (covers both medial occipital surfaces, the falx and the superior sagittal sinus);
- y: −8 (just behind the pole, to show the sinus confluence and skull boundary) to +45 anterior;
- z: −35 to +25 (covers the tentorium and transverse sinus below, the parieto-occipital region above).

Slices: along each tkr axis, positions from the low to the high end of the box in steps of 1 mm (configurable), each snapped to the nearest voxel centre; duplicates and positions outside the volume are dropped. With the default box and 1 mm voxels that gives 61 sagittal, 54 coronal and 61 axial slices, ordered from low to high tkr coordinate (left to right, posterior to anterior, inferior to superior). The contour plane of a slice is the plane through its voxel centres.

Panel field of view: every panel is square. Its side is the largest box extent plus a 10 mm margin on both sides (80 mm by default), centred on the midpoint of the box along each in-plane axis, so the rater sees just outside the brain. At the default 320 px per panel this is 4 px per mm.

Display orientation (decision D13), recorded per panel in `render.json` as the anatomical letter at each edge:

- sagittal: horizontal is y, vertical is z, superior up; anterior at the left by default (`sagittal_anterior_left`);
- coronal: horizontal is x, vertical is z, superior up; radiological by default (`radiological`), so the subject's right is at the left of the panel;
- axial: horizontal is x, vertical is y, anterior up; radiological by default.

Pixel convention: panel pixel coordinates `(px, py)` are continuous, with the origin at the top-left corner of the tile, `px` to the right and `py` down, in units of tile pixels. The pixel in column `c` and row `r` covers `[c, c + 1) × [r, r + 1)` and is sampled at its centre `(c + 0.5, r + 0.5)`. A browser click at offset `(ox, oy)` on a tile displayed at width `w` maps to `px = ox × width / w`.

The box is a reasonable default for adult brains. Pilot renders on infant, elderly or atypical brains decide whether the box needs to scale with brain size (for example by the distance between the occipital and frontal poles).

## Contours

Each surface is intersected with the slice plane: for each triangle, signed distances of its three vertices to the plane; triangles with sign changes contribute one segment between the two edge crossings. A vertex exactly on the plane counts as on the positive side, so no triangle yields a degenerate case. Vectorized in numpy per slice, this costs milliseconds for a 150k-vertex mesh. `trimesh.intersections.mesh_multiplane` is an alternative if it proves faster (optional dependency, to benchmark).

Segments are drawn with matplotlib `LineCollection` on the Agg canvas, antialiased, linewidth about 1.25 px at the output resolution, onto a transparent RGBA tile. The white surface is drawn first and the pial surface over it.

Default colours: pial yellow (#FFD400), white surface red (#FF3B30), on grayscale. The gist used white pial and red white; white pial is hard to see against bright tissue, hence the change. Colours are configurable and the defaults must be checked for visibility on both 3T and 7T T1 contrasts in the pilot.

## Grayscale tiles and intensity window

The grayscale tile samples the slice by nearest neighbour at each pixel centre, as Freeview does (decision D18); pixels outside the volume are black.

Per subject, the window runs from the 1st to the 99.5th percentile of `T1.mgz` within a dilated brainmask (three iterations of binary dilation of `brainmask.mgz > 0`). Without a brainmask, the percentiles are taken over the non-zero voxels of `T1.mgz`. A fixed window (FreeSurfer normalizes WM to 110) is an option, but the percentile window is more forgiving of residual bias field at 7T. The window is recorded in `render.json`.

## Output per subject

```
media/<subject_uid>/
  tiles/sagittal_000.webp ...          # grayscale slice tiles, lossy WebP, quality 80
  tiles/sagittal_000_c.webp ...        # contour tiles, lossless RGBA WebP, transparent background
  tiles/coronal_000.webp ...  tiles/axial_000.webp ...
  anim.mp4                             # synchronized mode, composite strip, H.264, for passive viewing and the gallery
  poster.webp                          # middle synchronized frame as a composite strip, for thumbnails
  render.json                          # see below
  vertex_map.npz                       # native vertices in the box -> fsaverage6, for click marks
```

Tile numbers index the panel's slice list in `render.json`. The browser stacks the two layers of each panel, lays the three panels out side by side, and draws crosshairs, orientation letters and marks itself (decision D11). The composite strip used for `anim.mp4` and `poster.webp` is the same composition done in Python: sagittal, coronal, axial from left to right, contours over grayscale, crosshairs and orientation letters on top.

`render.json` holds everything the app needs to turn a click into anatomy and to play the animation. Its schema is the pydantic model `RenderInfo` in `src/surfcensus/render/model.py`:

- `schema_version`, `surfcensus_version`, the full render config and its hash;
- `volume`: shape, voxel size, and the vox2ras-tkr and vox2ras affines of `T1.mgz`;
- `inputs`: for each file role used, its size and SHA-256 (no paths, so the file can be served to raters), and which pial file variant was read;
- `poles`, `box` (centre, low and high corners in tkr RAS) and the intensity `window`;
- `panels`: for each of the three panels, the tkr axis normal to the slices, the voxel axes along the slice direction and along the panel's horizontal and vertical, the voxel index and tkr coordinate of every slice, the tile size, the edge letters, and a 3 × 3 affine `pix2vox` from homogeneous panel pixel coordinates to the two in-plane voxel coordinates (crop, scale and display flips included) with its inverse `vox2pix`, so the client can draw existing marks;
- `modes`: for the synchronized and the one-at-a-time mode, the frame table of one full ping-pong loop, each frame being the slice number shown in the sagittal, coronal and axial panel;
- `tiles`, `video`, `poster` and `vertex_map` file names, and the versions of numpy, matplotlib, pillow and nibabel that produced the render.

It contains no timestamps, so identical inputs give an identical file. A unit test round-trips random voxels through pixel space and back for every panel and slice.

Frame tables. Synchronized: with N the largest slice count of the three panels, forward frame f (0 to N − 1) shows slice `round(f × (n − 1) / (N − 1))` of a panel with n slices; the loop is the forward frames followed by the same frames in reverse without the two end frames, 2N − 2 frames in all (120 by default). One at a time: for each panel in turn, the same ping-pong over its own slices while the other two panels hold the slice nearest the box centre (346 frames by default). Neither mode needs extra images.

`vertex_map.npz` holds, per hemisphere, the indices and tkr RAS coordinates (float32) of the native vertices whose pial or white position lies inside the box plus a 10 mm margin, and each vertex's fsaverage6 counterpart from the sphere mapping in `metrics.md`. Until that mapping exists (milestone 3) the fsaverage6 entries are −1. This makes it cheap to resolve click marks to vertices after the Tier C files have been deleted. Size is well under a megabyte per subject (estimate).

Sizes: three panels of 320 × 320 px. Measured on 15 sagittal tiles of `ds000102` sub-01 with one hemisphere's surfaces: 8.7 kB per grayscale tile and 16.3 kB per contour tile, so about 176 tile pairs come to roughly 4 MB per subject and on the order of 20 GB for 5,000 subjects, served statically. To measure properly in the pilot.

Video: H.264 via imageio-ffmpeg, `-preset veryfast`, CRF 23, `yuv420p`, 12 fps, single-threaded and with bit-exact flags so the file is reproducible. 120 frames gives a 10 s loop.

## Optional views

- Patch-centred view: a fourth small panel centred on the native location of the largest fsaverage outlier cluster from `metrics.md`, when one exists. Off by default during rating, so the rating stays blind to the metric.
- Whole-brain sagittal sweep, as in the gist, for subjects flagged "other" by raters.

## Performance target

About 176 tile pairs and 120 composite video frames per subject: target under 20 s per subject per core, excluding download. On a 32-core machine that is about 1 hour per 5,000 subjects of compute, so download throughput will dominate. Measure in pilot.
