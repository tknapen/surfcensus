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
2. The rating bar: one row per criterion with its scale buttons, each labelled with its keyboard key, and under them the controls of the subject: unratable, other problem, note, back, skip, help, submit.
3. A free-text note field, collapsed by default (key `n`).
4. A progress line: done today, done in total, remaining in the assignment, and how many subjects were skipped and wait to be offered again.

Nothing about the subject is shown: no dataset, subject label, metric, or prior rating. Media URLs use `subject_uid` only.

Playback uses the WebP tile stack and the frame tables in `render.json`, preloaded for the current subject and prefetched for the next two, so moving between subjects is instant. Video is used only in the gallery.

## From subject to subject: submit, skip, back

`Enter` stores the rating and shows the next subject. It needs a value for every criterion, or `u` (unratable). A rater who cannot or will not rate the subject on screen now presses `k` (the button `k skip`, next to the back button):

- A skip stores no rating, mark or note. What the rater had entered stays in the browser as the draft of that assignment (session storage, one draft per assignment, as a reload of the page keeps it) and is there again when the subject comes back.
- A skipped subject is deferred, never dropped. It is offered again after every subject the rater has not been shown yet, the one skipped longest ago first. When only skipped subjects are left, skip goes round them. The queue is done only when every subject has a submission: a rating, or unratable.
- A skip gives the subject to nobody else and does not count as a completed rating, so in a census a rater cannot avoid a subject by skipping it, which would bias which subjects are rated by whom. The same rule holds within the calibration stage.
- While a render of the selection is still awaited, the rater waits for it before a skipped subject comes back, since that subject has not been shown either.
- `Backspace` goes back to the subject the rater left last, whether it was rated or skipped. A skipped subject comes back as it was left, without a rating; the skip stays recorded. On a subject that is shown again for revision, `k` leaves the earlier rating as it is and goes on; nothing is sent.
- Every skip is recorded (see Event logging), because how often and on which subjects raters defer is data for the paper.

When `Enter` or the submit button is refused, the reason appears in a box directly under the submit button, in a colour that nothing else on the screen has: which criteria have no value (or no mark, or that the other problem needs a note), and that `k` skips the subject. The rows that lack something are framed and labelled until they have it. The submit button is not disabled while a rating is incomplete (it carries `aria-disabled`), so that pressing it is answered as well.

## The tour of the screen

A rater who sees the rating screen for the first time gets a tour of five steps. Each is a short text beside the part of the screen it is about, with that part lit and the rest dimmed: the three panels (the sweep, pause, stepping, hiding the contours); the rating bar (every row needs a value, the keys are on the buttons); marks (click on an error, the mark goes to the active row); moving on (`Enter`, `k`, `Backspace`); help (`?`, and `t` for the tour).

- It opens by itself once per rater and browser (a flag in local storage under the rater's name; where the browser allows no storage, once per load of the page), and again with `t` or from the help screen.
- `Enter` or `→` shows the next step, `←` the one before, `Esc`, `t` or the close button ends it; `Enter` on the last step ends it too.
- While it is open no other key reaches the rating screen, no click places a mark, and playback is paused; what was playing plays again when it closes.
- The keys it names are read from the protocol and the keyboard map of the session, so a `protocol.yaml` with other criterion keys gets a tour with those keys.
- It shows and sends nothing about the subject. Time spent in it does not count as time on the subject (`ms_on_subject`), nor as viewing time of the frame under it; `tour_open` and `tour_close` events record that it was opened and how far it was read.
- It is a dialog for assistive technology (`role="dialog"`, `aria-modal`, focus kept on its buttons and returned afterwards).

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
| Enter | submit and next (needs a value for every criterion, or `u`; otherwise the page says what is missing) |
| k | skip: nothing is stored, and the subject comes back after the others |
| Backspace | go back to the subject left last, rated (to revise it) or skipped |
| ? | show the scale definitions with reference examples, every key, and a button for the tour |
| t | the tour of the rating screen |

Scale definitions and criteria are in `rating_protocol.md`. The app reads them from a YAML file in the run directory, so the protocol can change without a code change, and the protocol version is stored with every rating.

## Assignment logic

- Calibration first: a new rater rates the frozen calibration set (about 20 subjects) and then sees the consensus labels with the reference examples. Calibration ratings are stored but excluded from the analysis.
- Then the queue: the server assigns the next subject from the selection table. Each subject needs k raters (default 2). The next subject for a rater is chosen at random (seeded per rater) among subjects with the fewest completed ratings that the rater has not seen, so coverage grows evenly and order effects are randomized.
- Repeats: 5% of a rater's assignments are subjects they rated at least 200 assignments earlier, for intra-rater reliability. Repeats look identical to new subjects.
- Local mode differs: one rater, no repeats unless enabled, queue ordered by composite score descending unless `--blind` is set. A researcher checking their own data usually wants the worst cases first. A study wanting unbiased ratings should use `--blind`.
- A skipped subject keeps its assignment, which stays open, and waits: it comes back after the subjects not yet shown, in the order of skipping (see "From subject to subject"). In every mode a skip is neither a completed rating nor a reason to give the subject to another rater. A repeat that is skipped waits like any other subject, and no second repeat of its subject is drawn meanwhile.

## Adjudication and review screens

Separate screens, available to raters with the `adjudicator` role, and only after their own assignment is done:

- Adjudication: subjects where raters differ by two scale steps on any criterion. Shows the same viewer, each rater's ratings and notes, and asks for a final label.
- Review: unblinded browsing of all subjects, sortable by any metric and filterable by rating, dataset pseudonym and FreeSurfer version (the app never shows accessions, see decision D10), with a scatter of composite score against final rating. Clicking a point opens that subject with all raters' marks and the fsaverage outlier-cluster centroids overlaid in different colours, so agreement and disagreement between eye and metric are visible in place. This is also the screen a local user ends up using most after rating.
- Adjudication also shows each rater's marks; the adjudicator can keep, move or delete marks to produce the final mark set.
- Census error map: an fsaverage flatmap or inflated view (static image regenerated on demand) showing the per-vertex frequency of final marks, by class, next to the mean z-map. Team lead and adjudicators only.

## Progress and agreement dashboard

For the team lead: completed ratings by rater and per day; median seconds per subject per rater; inter-rater agreement per criterion (weighted Cohen's kappa for pairs, Krippendorff's alpha for all raters), recomputed on demand; intra-rater agreement from repeats; drift in a rater's rating distribution over time.

## Event logging

Every rating stores rater, subject, criterion values, protocol version, app version, client timestamp, server timestamp, and time on subject. Interaction events are logged per subject: frames viewed, play and pause, mode switches, zoom, back-navigation, the tour. This answers a question the paper can use: how long careful looking actually takes, and whether raters who scrub more catch more.

Skips are recorded twice. The assignment carries the time of its latest skip and the number of its skips (`skipped_at`, `skip_count`). And each skip is an event of type `skip`, written by the server when it takes the skip and never accepted from a page: its `ts` is the client's time of the skip, and its payload holds the server's time, the number of the skip for that assignment, and the time the subject was on a visible page in the visit that ended with the skip. A request that arrives twice is recorded once.

Time on subject is per visit. `ms_on_subject` of a submission is the time of the visit that ended with the submission; the time of earlier visits to a subject that was skipped is in its skip events. Time while the page is hidden or the tour is open is not counted.

## Database tables (`app.sqlite`)

The tables are the SQLModel classes of `app/models.py`. Timestamps are UTC ISO 8601 text with milliseconds. A column with a fixed vocabulary (written `[a|b]` below) carries a CHECK constraint, so the database refuses a row with a value the app does not know. Code outside `app/` reads the tables with the standard library, by these names.

- `raters(rater_id, name, token_hash, role[rater|adjudicator|lead], active, created_at)`; `name` is unique.
- `assignments(assignment_id, rater_id, subject_uid, kind[calibration|main|repeat|adjudication], seq, render_key, assigned_at, completed_at, consensus_served_at, skipped_at, skip_count)`, unique on `(rater_id, seq)`. `seq` numbers a rater's assignments from 1. `render_key` is `renders.render_key` of the render that was shown (`rendering.md`), so a rating stays with the tiles it was made on. `consensus_served_at` is for calibration: once the rater has seen the consensus, the rating is final. `skipped_at` is the server's time of the latest skip (null for an assignment that was never skipped) and `skip_count` the number of skips (0 by default). The two columns were added without a new schema version, because no `app.sqlite` existed outside tests when they were.
- `submissions(submission_id, assignment_id, revision, current, unratable_reason, other, note, protocol_version, protocol_sha256, calibration_sha256, app_version, client_ts, server_ts, ms_on_subject)`, unique on `(assignment_id, revision)`: one row per press of `Enter`. A revision (`Backspace`, then `Enter`) is a new row with the next `revision`; `current` moves to it, and the marks of the superseded submission get `deleted_ts`. Nothing is overwritten.
- `ratings(rating_id, submission_id, assignment_id, criterion[A|B|C], value)`, unique on `(submission_id, criterion)`. An unratable submission has no rating rows. The view `ratings_flat` gives, for current submissions, the columns this table was first described with: `rating_id, assignment_id, criterion, value, note, protocol_version, app_version, client_ts, server_ts, ms_on_subject`.
- `marks(mark_id, submission_id, assignment_id, class[A|B|C|X], panel[sagittal|coronal|axial], slice, frame, mode[sync|seq], px, py, vox_i, vox_j, vox_k, tkr_x, tkr_y, tkr_z, ras_x, ras_y, ras_z, hemi[lh|rh], pial_vertex, pial_dist_mm, white_vertex, white_dist_mm, fsavg6_vertex, vertex_surface[pial|white], box_dist_mm, created_ts, deleted_ts)`. `slice` is the position in the panel's slice list of the tile that was clicked, which is what the geometry needs; `frame` and `mode` are the playback context of the click and may be null. `vox_*` are continuous voxel coordinates. `vertex_surface` says whose vertex `fsavg6_vertex` is the counterpart of, and `box_dist_mm` is the distance from the mark to the view box (0 inside).
- `events(event_id, assignment_id, ts, type, payload_json)`; `assignment_id` is null for an event of the run, such as a changed setting. An event of type `skip` is written by the server (see Event logging).
- `final_labels(subject_uid, criterion[A|B|C|unratable], value, method[consensus|adjudicated|single])`, key `(subject_uid, criterion)`. In the row of criterion `unratable`, `value` is the position of the reason in the protocol's list of unratable reasons.
- `final_marks(final_mark_id, subject_uid, class[A|B|C|X], tkr_x, tkr_y, tkr_z, hemi, pial_vertex, fsavg6_vertex, source_mark_ids)`; `source_mark_ids` is a JSON array.
- `app_meta(key, value)`: `schema_version`, `run_id` and `mode`, which tie the database to its run.

The tables and the view are written and tested (first step of milestone 4, decision D32). Nothing fills them yet: the functions that open the database and store a submission follow in the milestone.

## API

- `GET /api/next` → next assignment for the authenticated rater: `subject_uid`, frame URLs, protocol version.
- `POST /api/rating` → criterion values, marks (class, panel, slice, px, py, and the frame and playback mode as context), note, timing; returns the next assignment. The server validates marks against `render.json` and resolves coordinates and vertices.
- `POST /api/skip` → the assignment id, the client's time and the time on the subject; nothing of the rating. Returns the next assignment, in the shape of `GET /api/next`. Sent twice with the same client time it is one skip. 409, as for a stale rating, when the assignment is not open for the rater: it has a submission, or its render is no longer the subject's.
- `POST /api/events` → batched interaction events. An event of type `skip` is refused: the server writes those itself.
- `GET /api/back` → the assignment the rater left last, rated (for revision, with its rating) or skipped (as it is, without one).
- `GET /api/review?sort=&filter=` → adjudicator or local mode only.
- `GET /api/stats` → dashboard numbers.
- `GET /api/export` → CSV of ratings, team lead only.

The request and response bodies are the pydantic models of `app/schemas.py`, and its `ROUTES` lists every route with the shapes it takes and answers with; the frontend and the server are both written against that module. A page receives the geometry of a render (`RenderOut`) without the sizes and digests of the source files and without the template's digests: `render.json` itself is never served. The progress of an answer (`ProgressOut`) says how many skipped subjects wait. The shapes are written and tested; no route answers yet.

Of skip, the refused `Enter` and the tour, what is built: the rule of the queue and the write of a skip (`app/queue.py`, `app/db.py`), tested on a database, and the page, tested in a browser against a stand-in for the server (`tests/frontend_mock.py`) at windows 1400 and 400 px wide and at two that are too low for the page (400 by 440 and 1366 by 560 px), on synthetic subjects. The route `POST /api/skip` is written with the server of the milestone.

## Throughput budget

At 15 s per subject (before marking time on positive subjects) and 2 raters per subject, 5,000 subjects take about 21 rater-hours per rater. Phase 0 measures actual time per subject; phase 5 sizes the sample from it.
