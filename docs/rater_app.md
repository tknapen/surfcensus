# Rater web app

The app lets raters work through thousands of brains quickly, keeps ratings blind and reliable, and serves both a single researcher rating their own 15 subjects on a laptop and a team of raters working through a census.

## Deployment modes

Local (default for `surfcensus review`): FastAPI and uvicorn on `127.0.0.1`, one rater, SQLite in the run directory, media served from the run directory. Opens the browser automatically. No login.

Team (`surfcensus serve --host 0.0.0.0`): same app, several raters, run on a lab server or a small cloud VM behind HTTPS (a reverse proxy such as Caddy; documented, not bundled). Media can be served from the app or from any static host or bucket via `media.base_url`; the app only stores ratings and events. Raters authenticate with per-rater links that carry a token (`/r/<token>`), issued by `surfcensus raters add NAME`. No passwords; this is a lab tool, not a public service.

Offline export (optional, later): `surfcensus static-export` writes a self-contained static site with a manifest. Ratings are kept in the browser and downloaded as a JSON file at the end. For a collaborator who cannot reach the server. Browser storage is fragile, so the app autosaves a downloadable backup every 50 ratings.

The frontend is plain HTML, CSS and JavaScript with no build step, shipped inside the wheel, so `pip install` is all a user needs.

## Rating screen

Layout, top to bottom:

1. The viewer: the three panels (sagittal, coronal, axial) side by side at the largest size the window allows, with a frame slider underneath. Each panel is composed in the browser from a grayscale tile and a contour tile (decision D11), with crosshairs and orientation letters drawn client-side.
2. The rating bar: one row per criterion with its scale buttons, each labelled with its keyboard key.
3. A free-text note field, collapsed by default (key `n`).
4. A progress line: done today, done in total, remaining in the assignment.

Nothing about the subject is shown: no dataset, subject label, metric, or prior rating. Media URLs use `subject_uid` only.

Playback uses the WebP tile stack and the frame tables in `render.json`, preloaded for the current subject and prefetched for the next two, so moving between subjects is instant. Video is used only in the gallery.

## Marking errors by clicking

Raters click on errors in any panel to mark where they are. A rating says that a subject has a problem; marks say where it is, which makes a spatial ground truth possible (see `metrics.md` and `analysis_plan.md`).

Interaction:

- A mark always belongs to a class: A (sinus and falx), B (tentorium and transverse sinus), C (white surface), or X (other). The active class is shown in the cursor and set with `a`, `b`, `c`, `x`. Rating a criterion as minor or major also makes that class active, so the usual flow is: press `3` (A major), click on the error.
- Clicking pauses playback, then places a marker at the clicked point on the current frame.
- A mark is a 3D point, so it appears in all three panels whenever their current slices pass within 3 mm of it, drawn with opacity falling off with distance. A click in the sagittal panel thus shows up in the coronal and axial panels as they sweep past, which lets the rater confirm a mark from a second view.
- Clicking an existing marker selects it; `Delete` removes it; `Ctrl/Cmd+Z` undoes the last mark. `h` hides all marks temporarily.
- Extent is marked with several points, not a brush. The protocol asks for the centre and both ends of an affected stretch.
- Marks are drawn client-side as an SVG layer over the frame image, so they cost nothing to render and stay sharp under zoom.

Click to coordinates: `render.json` stores, for every panel, the slice axis, the voxel slice index of each tile, and the 2D affine from panel pixels to voxel indices (crop, scale, flips). The client converts a click to voxel coordinates. The server converts voxels to tkr RAS (via the stored vox2ras-tkr) and scanner RAS (via vox2ras), and stores all of them.

Click to vertices: at render time the renderer writes `vertex_map.npz`: the pial and white coordinates (tkr RAS, float32) of the native vertices in the region the tiles show, grown by a margin (`vertex_map_margin_mm`, 5 mm by default; the exact rule is in `rendering.md`), with, for each, the corresponding fsaverage6 vertex from the sphere mapping and a flag `visible` that is true where the pial position lies in the shown region itself. The server maps each mark to the nearest pial vertex within 5 mm (the snap radius, configurable up to the margin of the render: within that limit the file gives the same vertex as the full surface would, beyond it the server refuses) using a KD-tree, records the distance and the hemisphere, and looks up the fsaverage6 vertex. A mark with no vertex within range is kept as a volumetric point with a null vertex. The misplaced surface is by definition near the structure it absorbed, so nearest-pial-vertex is the natural target for classes A and B; for class C the nearest white vertex is also recorded.

Requirement: in census mode, a minor or major value on A, B or C cannot be submitted without at least one mark of that class (`Enter` stays disabled, with a hint). In local mode marks are optional unless `--require-marks` is set. Marking adds time only for positive subjects; phase 0 measures how much.

## Keyboard map

Every action has a key, so a rater never has to use the mouse.

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | previous / next frame (paused) |
| Shift + ← / → | jump 10 frames |
| [ / ] | slower / faster playback |
| m | toggle synchronized vs one-at-a-time mode |
| z | toggle zoom (2×, centred on the box) |
| 1 2 3 | superior sagittal sinus / falx criterion: none, minor, major |
| 4 5 6 | tentorium / transverse sinus criterion: none, minor, major |
| 7 8 9 | white surface criterion: none, minor, major |
| click | place a mark of the active class at the clicked point (pauses playback) |
| a b c x | set active mark class: A sinus/falx, B tentorium, C white surface, X other |
| Delete | remove the selected mark |
| Ctrl/Cmd + Z | undo last mark |
| h | hide marks while held |
| s | hide surface contours while held |
| u | unratable (image quality or render failure), with a reason picker |
| o | other problem, opens note |
| n | note |
| Enter | submit and next (only enabled when every criterion has a value or `u` was set) |
| Backspace | go back to the previous subject and revise |
| ? | show the scale definitions with reference examples |

Scale definitions and criteria are in `rating_protocol.md`. The app reads them from a YAML file in the run directory, so the protocol can change without a code change, and the protocol version is stored with every rating.

## Assignment logic

- Calibration first: a new rater rates the frozen calibration set (about 20 subjects) and then sees the consensus labels with the reference examples. Calibration ratings are stored but excluded from the analysis.
- Then the queue: the server assigns the next subject from the selection table. Each subject needs k raters (default 2). The next subject for a rater is chosen at random (seeded per rater) among subjects with the fewest completed ratings that the rater has not seen, so coverage grows evenly and order effects are randomized.
- Repeats: 5% of a rater's assignments are subjects they rated at least 200 assignments earlier, for intra-rater reliability. Repeats look identical to new subjects.
- Local mode differs: one rater, no repeats unless enabled, queue ordered by composite score descending unless `--blind` is set. A researcher checking their own data usually wants the worst cases first. A study wanting unbiased ratings should use `--blind`.

## Adjudication and review screens

Separate screens, available to raters with the `adjudicator` role, and only after their own assignment is done:

- Adjudication: subjects where raters differ by two scale steps on any criterion. Shows the same viewer, each rater's ratings and notes, and asks for a final label.
- Review: unblinded browsing of all subjects, sortable by any metric and filterable by rating, dataset pseudonym and FreeSurfer version (the app never shows accessions, see decision D10), with a scatter of composite score against final rating. Clicking a point opens that subject with all raters' marks and the fsaverage outlier-cluster centroids overlaid in different colours, so agreement and disagreement between eye and metric are visible in place. This is also the screen a local user ends up using most after rating.
- Adjudication also shows each rater's marks; the adjudicator can keep, move or delete marks to produce the final mark set.
- Census error map: an fsaverage flatmap or inflated view (static image regenerated on demand) showing the per-vertex frequency of final marks, by class, next to the mean z-map. Team lead and adjudicators only.

## Progress and agreement dashboard

For the team lead: completed ratings by rater and per day; median seconds per subject per rater; inter-rater agreement per criterion (weighted Cohen's kappa for pairs, Krippendorff's alpha for all raters), recomputed on demand; intra-rater agreement from repeats; drift in a rater's rating distribution over time.

## Event logging

Every rating stores rater, subject, criterion values, protocol version, app version, client timestamp, server timestamp, and time on subject. Interaction events are logged per subject: frames viewed, play and pause, mode switches, zoom, back-navigation. This answers a question the paper can use: how long careful looking actually takes, and whether raters who scrub more catch more.

## Database tables (`app.sqlite`)

- `raters(rater_id, name, token_hash, role, created_at)`
- `assignments(assignment_id, rater_id, subject_uid, kind[calibration|main|repeat|adjudication], assigned_at, completed_at)`
- `ratings(rating_id, assignment_id, criterion, value, note, protocol_version, app_version, client_ts, server_ts, ms_on_subject)`
- `events(assignment_id, ts, type, payload_json)`
- `marks(mark_id, assignment_id, class[A|B|C|X], panel, frame, px, py, vox_i, vox_j, vox_k, tkr_x, tkr_y, tkr_z, ras_x, ras_y, ras_z, hemi, pial_vertex, pial_dist_mm, white_vertex, white_dist_mm, fsavg6_vertex, created_ts, deleted_ts)`
- `final_labels(subject_uid, criterion, value, method[consensus|adjudicated|single])`
- `final_marks(subject_uid, class, tkr_x, tkr_y, tkr_z, hemi, pial_vertex, fsavg6_vertex, source_mark_ids)`

## API

- `GET /api/next` → next assignment for the authenticated rater: `subject_uid`, frame URLs, protocol version.
- `POST /api/rating` → criterion values, marks (panel, frame, px, py, class), note, timing; returns the next assignment. The server validates marks against `render.json` and resolves coordinates and vertices.
- `POST /api/events` → batched interaction events.
- `GET /api/back` → previous assignment for revision.
- `GET /api/review?sort=&filter=` → adjudicator or local mode only.
- `GET /api/stats` → dashboard numbers.
- `GET /api/export` → CSV of ratings, team lead only.

## Throughput budget

At 15 s per subject (before marking time on positive subjects) and 2 raters per subject, 5,000 subjects take about 21 rater-hours per rater. Phase 0 measures actual time per subject; phase 5 sizes the sample from it.
