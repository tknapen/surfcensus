# Rendering

## What a rater sees

Three panels side by side around the occipital pole: sagittal, coronal, axial. Each shows `T1.mgz` in grayscale with the white and pial surface contours from both hemispheres. All three slice positions sweep through a box around the occipital pole, forward and then back (ping-pong), in a loop. Each panel carries thin crosshair lines marking the current positions of the other two planes, so the rater can see where the three views intersect.

Two playback modes, selected in the app:

- synchronized: all three planes advance together, phase-locked across their ranges;
- one at a time: sagittal sweeps while the other two hold at the box centre, then coronal, then axial.

The rater can pause and scrub frame by frame. Scrubbing is the main way experienced raters work, so the renderer produces individual slice tiles as well as a video. The rater can also hide the contours to see the tissue under the line (decision D11).

## Relation to the original gist

The gist (`tknapen/85d9a23a09c95d15f4b23ba1965fa54d`) uses `freeview -cmd` with `-viewport sagittal`, `-slice x 127 127` and `-ss` per slice, then ffmpeg. It is kept as the `freeview` backend for publication figures and for comparison. That backend is not built yet: milestone 1 has the default backend only, and `--backend freeview` is an option of the census `render` command (`architecture.md`). The default backend is pure Python (nibabel, numpy and matplotlib). It needs no X server, FreeSurfer install or license, produces identical output for a given input, config and locked environment (decision D18), and parallelizes across processes.

## Geometry

Coordinates: FreeSurfer surfaces are in tkr RAS (surface RAS): x to the subject's right, y anterior, z superior, in mm. `T1.mgz` is normally a conformed 256³ 1 mm volume; nibabel gives its voxel-to-tkr-RAS affine via `MGHHeader.get_vox2ras_tkr()`. Nothing in the renderer assumes 1 mm or 256³: all geometry is in mm and is converted through the affine, so `-hires` volumes work (decision D18).

Voxel convention: an integer voxel index is the centre of that voxel, so voxel `i` covers `[i - 0.5, i + 0.5)` along its axis. This is the convention of the affine. Freeview's source (dev branch, read 2026-11-12) uses the same convention and has no half-voxel offset between image and surface contour: image samples sit at voxel centres, `-slice` places the plane through the centre of the named voxel, and the surface is cut by that same plane. The reading has not been checked against a Freeview frame; that is the by-eye comparison. One difference is known: Freeview places surfaces through the volume geometry stored in the surface file's footer, while this renderer uses the `T1.mgz` vox2ras-tkr directly. The two agree when the footer geometry matches the displayed volume, which is the normal case for recon-all output.

Axis alignment: slices are taken along voxel axes, so no resampling is needed. This requires each voxel axis to be parallel to one tkr axis, which holds for every FreeSurfer `mri/` volume. nibabel builds vox2ras-tkr from the voxel sizes and dimensions alone, with FreeSurfer's fixed LIA axes, whatever the direction cosines of the file. So tkr RAS is anatomical RAS only for a volume in conformed (LIA) orientation, and the test that can fail on a real file is the comparison of the two affines: a volume whose scanner affine (vox2ras) and vox2ras-tkr differ in their linear part (tolerance 1e-4 relative) is refused, and the render fails with reason `wrong orientation`. A vox2ras-tkr that is not a finite scaled signed permutation is refused with the same reason.

Occipital pole: per hemisphere, the pial vertex with the most posterior y (decision D12). In y and z, the box centre is the mean of the two hemispheric poles. In x, the centre is the midline, estimated as the midpoint between the medial-most pial vertices of the two hemispheres (largest x on the left, smallest x on the right) among vertices whose y and z fall inside the box ranges below. If either hemisphere has no vertex in that range, the centre x is the mean x of the two poles.

Default box, all in mm relative to the centre, configurable:

- x: −30 to +30 (covers both medial occipital surfaces, the falx and the superior sagittal sinus);
- y: −8 (just behind the pole, to show the sinus confluence and skull boundary) to +45 anterior;
- z: −35 to +25 (covers the tentorium and transverse sinus below, the parieto-occipital region above).

Slices: along each tkr axis, positions from the low to the high end of the box in steps of 1 mm (configurable), each snapped to the nearest voxel centre (a position exactly half-way goes to the higher voxel index); duplicates and positions outside the volume are dropped. With voxels smaller than the step some voxel planes get no tile: a 0.8 mm volume at the default 1 mm step skips every fifth plane. With the default box and 1 mm voxels that gives 61 sagittal, 54 coronal and 61 axial slices, ordered from low to high tkr coordinate (left to right, posterior to anterior, inferior to superior). The contour plane of a slice is the plane through its voxel centres.

Panel field of view: every panel is square. Its side is the largest box extent plus a 10 mm margin on both sides (80 mm by default), centred on the midpoint of the box along each in-plane axis, so the rater sees just outside the brain. At the default 320 px per panel this is 4 px per mm.

Display orientation (decision D13), recorded per panel in `render.json` as the anatomical letter at each edge:

- sagittal: horizontal is y, vertical is z, superior up; posterior at the left by default, as in Freeview, which views the head from the subject's right (`sagittal_anterior_left` flips it);
- coronal: horizontal is x, vertical is z, superior up; radiological by default (`radiological`), so the subject's right is at the left of the panel;
- axial: horizontal is x, vertical is y, anterior up; radiological by default.

Pixel convention: panel pixel coordinates `(px, py)` are continuous, with the origin at the top-left corner of the tile, `px` to the right and `py` down, in units of tile pixels. The pixel in column `c` and row `r` covers `[c, c + 1) × [r, r + 1)` and is sampled at its centre `(c + 0.5, r + 0.5)`. A browser click at offset `(ox, oy)` on a tile displayed at width `w` maps to `px = ox × width / w`.

The box is a reasonable default for adult brains. Pilot renders on infant, elderly or atypical brains decide whether the box needs to scale with brain size (for example by the distance between the occipital and frontal poles).

## Contours

Each surface is intersected with the slice plane: for each triangle, signed distances of its three vertices to the plane; triangles with sign changes contribute one segment between the two edge crossings. A vertex exactly on the plane counts as on the positive side, so no triangle yields a degenerate case. A per-axis index of the faces (sorted by their lowest coordinate) limits each cut to the faces near the plane; one cut of a 164k-vertex mesh takes about 1 ms.

Segments that share endpoints are joined into polylines and drawn as stroked paths on the matplotlib Agg canvas, antialiased, with round joins, linewidth 1.25 px at the output resolution, onto a transparent RGBA tile. Joining matters: separately drawn segments blend twice where they meet, which thickens the line unevenly (by 8% to 28%, measured). Zero-length segments are dropped. The white surface is drawn first and the pial surface over it. The line width is configurable up to an eighth of the panel size (40 px at the default panel); a wider line is refused when the config is made. A contour tile that its lines cover completely cannot be stored with an alpha channel, and the bound keeps lines far from that. It is a practical limit, not a proof that no tile can be covered.

The tile does not depend on the user's matplotlib settings. The renderer draws on its own Agg canvas without pyplot, and it resets `path.sketch` on its artists. That setting, from a `matplotlibrc` or left behind by `plt.xkcd()` earlier in the process, is copied by every artist when it is made and draws lines with a wobble. The review of milestone 1 changed each of matplotlib's rcParams in turn and found it to be the only one that changes a tile; after `plt.xkcd()`, 157 of the 356 files of a render of `ds000102` sub-01 differed and line pixels lay up to 16 px (4 mm) from the true contour. With the reset, renders of sub-01 and of the 160³ synthetic subject after `plt.xkcd()` and under `path.sketch: 3, 30, 5` are byte-identical to the plain render (356 of 356 files each).

Default colours: pial yellow (#FFD400), white surface red (#FF3B30), on grayscale. The gist used white pial and red white; white pial is hard to see against bright tissue, hence the change. Colours are configurable as `#RRGGBB`; anything else is refused when the config is made. They are stored in upper case, so `#ffd400` and `#FFD400` are one config with one hash. The defaults must be checked for visibility on both 3T and 7T T1 contrasts in the pilot.

## Grayscale tiles and intensity window

The grayscale tile samples the slice by nearest neighbour at each pixel centre, as Freeview does (decision D18); pixels outside the volume are black. The window maps linearly to 0 to 255, clipped and rounded.

Per subject, the window runs from the 1st to the 99.5th percentile of `T1.mgz` within a dilated brainmask (three iterations of binary dilation of `brainmask.mgz > 0`). Without a brainmask, the percentiles are taken over the non-zero voxels of `T1.mgz`. A fixed window (FreeSurfer normalizes WM to 110) is an option, but the percentile window is more forgiving of residual bias field at 7T. The window is recorded in `render.json`.

## Output per subject

```
media/<subject_uid>/
  tiles/sagittal_000.webp ...          # grayscale slice tiles, lossy WebP, quality 80
  tiles/sagittal_000_c.webp ...        # contour tiles, lossless RGBA WebP, transparent background
  tiles/coronal_000.webp ...  tiles/axial_000.webp ...
  anim.mp4                             # synchronized mode, composite strip, H.264, for passive viewing and the gallery
  poster.webp                          # middle forward synchronized frame as a composite strip, for thumbnails
  render.json                          # see below
  vertex_map.npz                       # native vertices the tiles show, and a margin -> fsaverage6, for click marks
```

Tile numbers index the panel's slice list in `render.json`. The browser stacks the two layers of each panel, lays the three panels out side by side, and draws crosshairs, orientation letters and marks itself (decision D11). The composite strip used for `anim.mp4` and `poster.webp` is the same composition done in Python: sagittal, coronal, axial from left to right, contours over grayscale, crosshairs and orientation letters on top. The letters use Pillow's built-in bitmap font, so no system font is involved.

`render.json` holds everything the app needs to turn a click into anatomy and to play the animation. Its schema is the pydantic model `RenderInfo` in `src/surfcensus/render/model.py`:

- `schema_version` (2 since the vertex map follows the tiles and not the box; a render of another version is rendered again), `surfcensus_version`, the full render config and its hash;
- `volume`: shape, voxel size, and the vox2ras-tkr and vox2ras affines of `T1.mgz`;
- `inputs`: for each file role used, its size and SHA-256 and the name FreeSurfer gives a file of that role, which tells which pial file variant was read. No path is recorded and no name from elsewhere: a file handed over under another name, or a link to one, is recorded under the role's first name (`T1.mgz`, `lh.pial`). The sizes and digests are those of the source files, which for a census are public bucket files, so `render.json` itself stays in the run directory and is not served or exported as a file (decision D20); the app of milestone 4 passes its geometry on without `inputs`;
- `poles`, `box` (centre, low and high corners in tkr RAS) and the intensity `window`;
- `panels`: for each of the three panels, the tkr axis normal to the slices, the voxel axes along the slice direction and along the panel's horizontal and vertical, the voxel index and tkr coordinate of every slice, the tile size, the edge letters, and a 3 × 3 affine `pix2vox` from homogeneous panel pixel coordinates to the two in-plane voxel coordinates (crop, scale and display flips included) with its inverse `vox2pix`, so the client can draw existing marks. Both are a scale and a shift per axis. `vox2pix` is written out entry by entry (`1 / scale` and `-shift / scale`) and holds no negative zero, so its digits do not depend on the linear-algebra library numpy is linked with;
- `modes`: for the synchronized and the one-at-a-time mode, the frame table of one full ping-pong loop, each frame being the slice number shown in the sagittal, coronal and axial panel;
- `tiles`, `video`, `poster` and `vertex_map` file names, for the vertex map also the number of vertices per hemisphere and how many of them are visible (see below), and the versions of numpy, matplotlib, pillow and nibabel that produced the render.

It contains no timestamps, so identical inputs give an identical file. A unit test round-trips random voxels through pixel space and back for every panel and slice.

Frame tables. Synchronized: with N the largest slice count of the three panels, forward frame f (0 to N − 1) shows slice `round(f × (n − 1) / (N − 1))` of a panel with n slices (an exact half rounds to the even slice); the loop is the forward frames followed by the same frames in reverse without the two end frames, 2N − 2 frames in all (120 by default; a single slice gives one frame). One at a time: for each panel in turn, the same ping-pong over its own slices while the other two panels hold the slice nearest the box centre (346 frames by default). Neither mode needs extra images.

`vertex_map.npz` stands in for the surfaces when click marks are resolved to vertices, after the Tier C files have been deleted. It therefore follows what the tiles show, not the view box: with the default config the tiles show 10 mm more than the box in x and z and 13.5 mm more in y (see Panel field of view).

The clickable region is the union of three boxes in tkr RAS, one per panel. Along the panel's slice axis its box runs from half a slice step (`slice_step_mm / 2`) before the first slice plane to half a step past the last; along the two in-plane axes it spans the panel's field of view. Every point that a click on a tile can name lies in the region, on a slice plane. The half step makes each slice stand for one step of the sweep, so that a vertex just outside the last plane, whose contour that tile still shows, counts as shown.

The file holds, per hemisphere `lh` and `rh`, every native vertex whose pial or white position lies in one of the three boxes grown by `vertex_map_margin_mm` on every side (boundary included):

- `{hemi}_index`: native vertex indices, ascending (int32);
- `{hemi}_pial`, `{hemi}_white`: their positions in tkr RAS (float32);
- `{hemi}_fsavg6`: the fsaverage6 counterpart from the sphere mapping in `metrics.md` (int32); −1 until that mapping exists (milestone 3);
- `{hemi}_visible`: true where the pial position lies in the clickable region itself, without the margin (bool). These are the vertices that the sweep of at least one panel passes, and the ones on which decision D15 evaluates spatial detectability.

`visible` says where a vertex lies, not which vertex a mark resolves to. A point of a drawn contour lies in the clickable region, on a face of the mesh, and the vertex nearest to it can lie outside the region: past the half slice step at an end of a sweep, or past a tile edge. Measured on a default render of `ds000102` sub-01: of the 373,536 tile pixels that show pial contour, 1,181 (0.32 %) resolve to a pial vertex that is not visible. These are 515 of the 50,514 vertices that the drawn pial contour resolves to, and they lie a median of 0.11 mm and at most 1.07 mm outside the region. On the synthetic subjects 0.34 % (1 mm) and 0.37 % (0.8 mm) of the points on the pial contours do. A mark off the contour can resolve farther outside, up to the snap radius. So the vertex of a mark must not be taken for a visible one; `analysis_plan.md` says how the vertex-level analysis treats it.

The margin defaults to 5 mm, the mark snap radius of `rater_app.md`. For any snap radius up to the margin, a mark has the same nearest pial vertex and the same nearest white vertex in the file as on the full surfaces, and the file has no vertex in reach exactly when the surface has none. A larger snap radius loses that guarantee, so the app must refuse one.

The archive is written uncompressed with fixed entry dates, so its bytes do not depend on the zlib build or the clock. On `ds000102` sub-01 it holds 74,654 vertices (37,356 left, 37,298 right), of which 55,534 are visible, and takes 2.5 MB.

Until the review of milestone 1 the file held the vertices inside the box plus a 10 mm margin (56,031 on sub-01, 1.8 MB). That ends 3.5 mm short of the tile edge along y, where the anterior strip of the sagittal and axial tiles shows cortex, and it leaves nothing for the snap radius at the other edges. Measured on a default render of sub-01, for the two ends and the middle of every pial contour segment that falls inside a tile (590,427 points, all with a pial vertex within 5 mm): through the old file 14,385 resolved to another vertex than on the full surface and 9 to none. Through the present file none does, and neither does any click on a grid of every second pixel of every tile (2.1 million clicks with a pial vertex within 5 mm); the same holds for the white surface. Tiles, poster and video are byte-identical under the two rules.

Sizes: three panels of 320 × 320 px. Measured on a full render of `ds000102` sub-01 (176 tile pairs): grayscale tiles 10.5 kB on average (1.9 MB in all), contour tiles 12.5 kB (2.2 MB), `anim.mp4` 4.3 MB, `vertex_map.npz` 2.5 MB, `poster.webp` 67 kB, `render.json` 26 kB; 10.9 MB per subject, of which the app needs the tiles (4.1 MB) and the vertex map. For 5,000 subjects that is about 55 GB with videos and about 33 GB without. One subject; to measure across datasets in the pilot.

Video: H.264 through the ffmpeg binary of imageio-ffmpeg, `-preset veryfast`, CRF 23, `yuv420p` (BT.709, limited range, tagged), 12 fps, single-threaded and with bit-exact flags, so the file is reproducible for a given ffmpeg binary. The binary is the one imageio-ffmpeg finds: the path in `IMAGEIO_FFMPEG_EXE` if set, else the binary in its wheel, else an installed one. 120 frames gives a 10 s loop. The 4:2:0 chroma subsampling desaturates the thin contours in the video (mean absolute error about 47 of 255 on line pixels); the tiles are exact, and raters work from the tiles.

## Failures and re-rendering

A subject that cannot be rendered raises `RenderError` with one of the unratable reasons of the rating protocol: `missing surface` (a required file is absent), `wrong orientation` (see Geometry), or `render failure` (unreadable file, degenerate mesh, white and pial surfaces of different vertex counts, brainmask on another grid, box outside the volume, and no ffmpeg binary when the video is asked for). Unreadable includes a file that may not be opened and a file in a directory that may not be searched: such a file is never reported as absent. For an unreadable file the message names the file and the error, not the path, also for a damaged file where nibabel's own message holds the path (it does for an empty file).

A config that cannot be drawn raises `ValueError`: a colour that is not `#RRGGBB`, a line wider than an eighth of the panel, a box whose low end lies above its high end along an axis, a box without extent along every axis together with a margin of 0, an odd panel size together with the video. The config is validated when it is made and again when a render starts, because pydantic's `model_copy(update=...)` does not validate.

Every refusal happens before anything is written. The ffmpeg binary is looked for at that point too, when the video is asked for: it is used only after `render.json` has been removed and every tile written, so a render that found it missing then would leave a complete render incomplete. The check is that the file exists and may be executed; nothing is started, so a binary that exists and fails is still found only when the video is encoded, as a `RuntimeError`. That holds for a video target as well (`three_plane(subject, "x.mp4")`): no directory is created for a refused subject, and directories that the call created are removed again if the render fails later.

The render config has two identities. `config_hash()` covers the schema number and every setting, and decides whether a directory holds the render that is asked for. `tile_hash()` covers only the settings that change a tile's pixels or geometry: the box, the margin, the slice step, the panel size, the two orientation switches, the contour colours and line width, the window percentiles, the brainmask dilation and the WebP quality. It has no schema number and no video, crosshair or vertex-map setting in it, so it stays the same when a render is only encoded again or its vertex map rewritten. It is not written to `render.json`. `privacy.render_key` combines it with the SHA-256 digests of the inputs under a per-run secret. A run stores that key with each render (`renders.render_key`) and, from milestone 4, with each rating, so that a rating stays with the tiles it was made on. A file that was only touched or moved keeps its key; changed content or a changed tile setting gives another.

`render.json` is removed when rendering starts and written last, so its presence marks a complete render. Rendering into a directory that already holds a complete render of the same inputs (size and SHA-256), config and video setting does nothing. Otherwise the directory is rendered again and what the earlier render left is removed, by name: tiles named `<panel>_<number>.webp` or `<panel>_<number>_c.webp` that the new render does not list, with the number as the renderer writes it (ASCII digits, three of them, or more without a leading zero), and the temporary files (`.<name>.<16 hex digits>.tmp`, for the names a render writes) of a render that was killed. Only files and links are removed: a directory under such a name stays. Every other file in the directory is left alone, whatever its extension. All files are written to a temporary name and renamed into place.

`anim.mp4` is removed by a render without video only if the `render.json` found in the directory lists a video. It is then the video of the earlier render, and it is removed first, before that `render.json`. An `anim.mp4` that no `render.json` lists is left alone: the directory may be one the user keeps other things in, and the file their own. The same holds after a render with video that was interrupted: it leaves an `anim.mp4`, the earlier one or its own, and no `render.json` to say whose the file is, so a later render without video leaves it in place, unlisted. A render with video writes `anim.mp4` and so replaces any file of that name.

Nothing is rendered into a source directory. `three_plane` and `render_subject` raise `OutputInSourceError`, a `ValueError`, before anything is written, when the output directory or the video path is, or lies inside, a directory that the inputs are read from. For an input at its FreeSurfer place (`<subject>/mri/T1.mgz`) that is the subject directory; for an input handed over from elsewhere it is the directory that holds the file; for a link, also the directory that holds its target. Directories are compared as the file system identifies them, so a link to the subject directory and a relative path into it are refused too. A directory that holds the subject directory is not inside it and is accepted.

Two renders into one directory run one after the other. A render holds an exclusive `flock` on the media directory from before it removes `render.json` until it has written the new one. A second render waits, then looks again: if the directory now holds the complete render it was asked for, it writes nothing, otherwise it renders. The lock is on the directory itself, so it adds no file to the media directory, and the kernel releases it when the process ends, however it ends. Where there is no `flock` (Windows has no `fcntl`) or the file system refuses it, there is no lock: the render goes ahead, and it leaves temporary files in place, since one may belong to a render that is still running. Checked on Linux on a local file system with two processes started 0 to 1.6 s apart on the 160³ synthetic subject, with the same config and with different pial colours: in 8 of 8 runs both exited with status 0 and the directory was byte-identical to a render of the config in `render.json` alone. Before the lock, the review of milestone 1 found the second process failing in 3 of 3 such runs, and with different configs a directory that held the tiles of one config under the `render.json` of the other. Not checked on macOS or on NFS.

`surfcensus render-subject` exits with status 1 and one line on stderr for a `RenderError` and for an output location that cannot be written. It refuses an empty `--out` (an unset shell variable), which as a path would be the current directory, and an `--out` in the subject directory, both as usage errors with status 2. It turns SIGTERM into an exit with status 143, so that a render stopped by a scheduler removes its temporary files as one stopped by Ctrl-C does.

## FreeSurfer 7

The renderer was run on one FreeSurfer 7.3.2 subject on 2026-11-12 (`ds003020` sub-UTS01 from OpenNeuroDerivatives, pial surfaces as `?h.pial.T1` only). `mri/T1.mgz` is 256³ at 1 mm and passes the orientation check; the default box gives 61, 54 and 61 slices; 156 of 176 tiles have contours; the render took 8.1 s without video and 6.3 MB. `render.json` records the pial inputs as `lh.pial.T1` and `rh.pial.T1`. On the poster frame, looked at once, the white and pial contours lie on the tissue boundaries. No frame was compared with Freeview, and no other FreeSurfer 7 subject was rendered.

## Rendering a run

`surfcensus render --run-dir DIR` and `Run.render()` render the subjects of a run into `media/<subject_uid>/` and record every outcome in the `renders` table of the catalog (`render/batch.py`).

Which subjects: every complete present subject in a census run (`select.render_scope`, or `--all`); in a local run the selection, which is every complete present subject until a selection exists. Without an order given they go by pseudonym, then `subject_uid`.

What is pending is decided from the catalog before any source file is opened; a call that finds nothing pending needs no source at all, so it works while the data is not mounted. A subject is skipped when its row says `done` for the present render config, the same input signature (the listing of the files a render reads) and the same video setting, and the render is whole on disk: a `render.json` of that config and video setting, and a file under every name it lists (tiles, poster, vertex map, video). A media directory that lost files, through a partial copy or a restore, is pending again, in `render` and in `status`, and is rendered again from the source. The names are looked up in two directory listings per subject, so the test opens no source file and no tile; measured on one default render (355 listed files, local disk): 0.8 ms. A row that says `failed` for them is not tried again unless `--retry-failed`. `--force` renders over what is there. The rest is rendered batch by batch (`cache.batch_subjects`, 32 by default): the files of a batch are leased from the source, rendered in `--jobs` worker processes by `render_subject`, recorded in one transaction, and released. The files and rows are the same for one worker process and for several.

Failures are recorded, not raised, and the batch goes on:

- a `RenderError` with its reason (`missing surface`, `wrong orientation`, `render failure`);
- `source changed` when a file differs in size or modification time from what the inventory listed; the next inventory picks the change up;
- `render failure` with the type of the exception for anything else, a worker process that died included. Media that cannot be written are the exception, see below.

The detail of a failure is stored after `privacy.scrub`, so it holds file names and no path. Every present subject that lacks a file a render cannot do without gets a `missing surface` row that names the roles, without its files being touched, so that it appears in review and export with its reason.

What is wrong with the run or the machine and not with a subject stops the phase with exit code 2 and is recorded against no subject: a video without an ffmpeg binary or with one that cannot encode (two 16 px frames are encoded into a temporary directory before the first batch), a video with an odd panel size, a source that cannot be found, a run directory inside a subject directory, and media that cannot be written. The last means an `OSError` of a render that says the disk is full, over quota or read-only, or that a path under `media/` may not be written, and a failure of ffmpeg with the text of the first three. It ends the call at the batch in which it occurs. Nothing of that batch is recorded; its renders that did succeed are complete on disk and are recorded by the next call without being rendered again. The reason for stopping: a failed row is not tried again without `--retry-failed`, so a full disk recorded as a failure of each subject would outlast the full disk. Any other error of writing the output is still a `render failure` of its subject.

After a render the SHA-256 digests in `render.json` are copied to the catalog and the render key is stored with the row. The row is written after the files. A crash in between leaves a complete render without a row; the next call finds it complete, renders nothing and records it.

A new local run renders without `anim.mp4` (`render_video` is false; `--video` turns it on and is remembered): the rater app plays the tiles.

`surfcensus render --check-sample` supports the exit check of phase 6. It draws `ceil(fraction × n)` of the `n` done renders (2% by default) with a seed that it prints, and writes `logs/render-check-<seed>.txt` (the seed in the first line, then one `subject_uid` and poster path per line) and a page `logs/render-check-<seed>.html` that shows the posters.

## Optional views

Neither is built yet; the patch-centred view needs the outlier clusters of milestone 3.

- Patch-centred view: a fourth small panel centred on the native location of the largest fsaverage outlier cluster from `metrics.md`, when one exists. Off by default during rating, so the rating stays blind to the metric.
- Whole-brain sagittal sweep, as in the gist, for subjects flagged "other" by raters.

## Performance target

About 176 tile pairs and 120 composite video frames per subject: target under 20 s per subject per core, excluding download. Measured on `ds000102` sub-01: 9 s on one core, of which WebP encoding takes about 3.4 s, the video 2.5 s and contour drawing 2 s. Through a run, the two real subjects of the network tests took 9.0 s together without video in two worker processes and 17.5 s with video in one (2026-11-12, one machine). On a 32-core machine that is about 25 minutes per 5,000 subjects of compute, so download throughput will dominate.
