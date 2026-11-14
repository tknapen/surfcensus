# Rater web app

The app lets raters work through thousands of brains quickly, keeps ratings blind and reliable, and serves both a single researcher rating their own 15 subjects on a laptop and a team of raters working through a census.

## Deployment modes

Local (default for `surfcensus review`): FastAPI and uvicorn on `127.0.0.1`, one rater, SQLite in the run directory, media served from the run directory. Opens the browser automatically. No login.

Local mode is built (milestone 4). Team mode is milestone 5: `surfcensus serve --mode team` stops with a line that says so.

Team (`surfcensus serve --host 0.0.0.0`): same app, several raters, run on a lab server or a small cloud VM behind HTTPS (a reverse proxy such as Caddy; documented, not bundled). Media can be served from the app or from any static host or bucket via `media.base_url`; the app only stores ratings and events. Raters authenticate with per-rater links that carry a token (`/r/<token>`), issued by `surfcensus raters add NAME`. No passwords; this is a lab tool, not a public service.

Offline export (optional, later): `surfcensus static-export` writes a self-contained static site with a manifest. Ratings are kept in the browser and downloaded as a JSON file at the end. For a collaborator who cannot reach the server. Browser storage is fragile, so the app autosaves a downloadable backup every 50 ratings.

The frontend is plain HTML, CSS and JavaScript with no build step, shipped inside the wheel, and the packages the server needs (`fastapi`, `uvicorn`, `sqlmodel`, `jinja2`) are dependencies of the package, so `pip install surfcensus` is all a user needs: no extra, no second step (decision O20; `architecture.md`, Dependencies).

## Rating screen

Layout, top to bottom:

1. The viewer: the three panels (sagittal, coronal, axial) side by side at the largest size the window allows, with a frame slider underneath. Each panel is composed in the browser from a grayscale tile and a contour tile (decision D11), with crosshairs and orientation letters drawn client-side.
2. The rating bar: one row per criterion with its scale buttons, each labelled with its keyboard key, and under them the controls of the subject: unratable, other problem, note, back, skip, help, submit.
3. A free-text note field, collapsed by default (key `n`).
4. A progress line: done today, done in total, remaining in the assignment, and how many subjects were skipped and wait to be offered again.

Nothing about the subject is shown: no dataset, subject label, metric, or prior rating. Media URLs use `subject_uid` only.

What is entered for the open subject (values, note, marks) is kept in the browser's session storage until it is submitted, one draft per assignment, so a reload does not lose it. The time the subject was looked at so far is kept the same way when the page is left, so a reload does not start the visit anew (Event logging). A rating that was posted and not answered is kept there too and posted again, also by the reloaded page; the server stores a rating that arrives twice (same assignment, same client time, same content) once. While the note field has the focus, every key but `Esc` goes into the note, `Enter` included. While the unratable reasons are open, the digits choose a reason and no other key of the map acts. The playback mode goes back to synchronized with every new subject; speed and zoom stay.

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

- A mark always belongs to a class: A (sinus and falx), B (tentorium and transverse sinus), C (white surface), or X (other). The active class is shown in the cursor and set with `a`, `b`, `c`, `x`. Rating a criterion as minor or major also makes that class active, so the usual flow is: press `3` (A major), click on the error. No class is active when a subject opens: a click then pauses playback and places nothing, and the page names the keys that choose a class. A default class would turn every stray click into a mark. The active class is reset with every subject.
- Clicking pauses playback, then places a marker at the clicked point on the current frame.
- A mark is a 3D point, so it appears in all three panels whenever their current slices pass within 3 mm of it, drawn with opacity falling off with distance. A click in the sagittal panel thus shows up in the coronal and axial panels as they sweep past, which lets the rater confirm a mark from a second view.
- Clicking an existing marker selects it; `Delete` removes it; `Ctrl/Cmd+Z` undoes the last mark. `h` hides all marks temporarily.
- Extent is marked with several points, not a brush. The protocol asks for the centre and both ends of an affected stretch.
- Marks are drawn client-side as an SVG layer over the frame image, so they cost nothing to render and stay sharp under zoom.

Click to coordinates: `render.json` stores, for every panel, the slice axis, the voxel slice index of each tile, and the 2D affine from panel pixels to voxel indices (crop, scale, flips). The page sends the panel, the slice and the tile pixel of a click, and draws marks with the same arithmetic. The server is the authority: it converts the pixel to voxel coordinates, the voxel to tkr RAS (via the stored vox2ras-tkr) and scanner RAS (via vox2ras), and stores all of them (`app/marks.py`).

A browser names the screen pixel under the pointer by its upper left corner, in whole CSS pixels. The page takes the middle of that pixel as the place of the click (`pointerAt` in `viewer.js`), so a mark is at most half a screen pixel from the pointer in either direction and has no bias; with the corner, every mark lay half a screen pixel up and to the left on average (0.09 mm at a panel 454 px wide, more in a smaller window). A coordinate that the browser gives with a fraction is taken as it is.

A mark is checked against the render it was made on: the slice exists in its panel, the pixel lies on the tile (0 to the tile size, edges included), and, when the frame and the mode of the click are both given, that frame shows that slice. A frame or a mode given alone is stored as it is. The nearest vertex and its distance are computed in float64 from the positions of the vertex map; equally near vertices go by rule (`lh` first, then the lowest native index), and the snap radius includes its boundary. One mark has one hemisphere: that of its nearest pial vertex, or for class C of its nearest white vertex.

Click to vertices: at render time the renderer writes `vertex_map.npz`: the pial and white coordinates (tkr RAS, float32) of the native vertices in the region the tiles show, grown by a margin (`vertex_map_margin_mm`, 5 mm by default; the exact rule is in `rendering.md`), with, for each, the corresponding fsaverage6 vertex from the sphere mapping and a flag `visible` that is true where the pial position lies in the shown region itself. The server maps each mark to the nearest pial vertex within 5 mm (the snap radius, configurable up to the margin of the render: within that limit the file gives the same vertex as the full surface would, beyond it the server refuses) using a KD-tree, records the distance and the hemisphere, and looks up the fsaverage6 vertex. A mark with no vertex within range is kept as a volumetric point with a null vertex. The misplaced surface is by definition near the structure it absorbed, so nearest-pial-vertex is the natural target for classes A and B; for class C the nearest white vertex is also recorded.

Requirement: in census mode, a minor or major value on A, B or C cannot be submitted without at least one mark of that class (`Enter` is refused, and the box under the submit button names the criterion whose mark is missing; the server answers such a rating with 422). In local mode marks are optional unless `--require-marks` is set. Marking adds time only for positive subjects; phase 0 measures how much.

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
| Esc | leave the note field; close the help screen, the unratable reasons or the tour; drop the selection of a mark |

Scale definitions and criteria are in `rating_protocol.md`. The app reads them from a YAML file in the run directory, so the protocol can change without a code change, and the protocol version is stored with every rating.

## Assignment logic

- Calibration first: a new rater rates the frozen calibration set (about 20 subjects) and then sees the consensus labels with the reference examples. Calibration ratings are stored but excluded from the analysis.
- Then the queue: the server assigns the next subject from the selection table. Each subject needs k raters (default 2). The next subject for a rater is chosen at random (seeded per rater) among subjects with the fewest completed ratings that the rater has not seen, so coverage grows evenly and order effects are randomized.
- Repeats: 5% of a rater's assignments are subjects they rated at least 200 assignments earlier, for intra-rater reliability. Repeats look identical to new subjects.
- Local mode differs: one rater, no repeats unless enabled, queue ordered by composite score descending unless `--blind` is set. A researcher checking their own data usually wants the worst cases first. A study wanting unbiased ratings should use `--blind`.
- An assignment belongs to the render it shows (`assignments.render_key`). A subject is done for a rater when the rater has a completed `main` assignment of the subject's present render: a subject whose surfaces or tile settings changed comes back, one that was only encoded again does not. An open assignment whose render changed, failed or is being made again is left open and is not returned again; the subject is offered anew under its present render. A repeat shows the render that was rated and is drawn before the next new subject, so it can be the last assignment of a rater.
- A render is ready for the queue when it was made with the render config of the running version. A row of another config, done or failed, counts as out of date, because `surfcensus render` makes it again; only a failure under the present config makes a subject unratable. A selected subject whose render failed, or that lacks a file a render needs, is never offered and carries an unratable final label with its reason. A render file that a DataLad or git-annex clone holds without its content is such a lacking file: the reason is `missing surface`, and the detail of the render row names the files that are not fetched.
- Which subjects are selected: the rows of the selection table. `review` writes one for every present subject. A local run made with `inventory`, `metrics` and `render` has none, and its selection is then every present subject, for the app and for `finalize` as for `render` (`rendering.md`), so `surfcensus serve` on such a run offers every rendered subject. The rows of a frozen calibration set are no selection of a local run: `calibration freeze` on such a run leaves its queue and its renders as they are. A census run without a selection has an empty queue.
- Local raters: one per server process, named with `--rater NAME` (default `app.rater_name`, `local`). The rater is made when the app starts, not at the first rating. So `serve` or `review` without `--rater` on a run that other raters have rated makes one more rater, `local`, who has every subject left: the unblinded screens are closed to that rater, and `finalize` lists the rater from then on. The app says so in a warning that names the raters the run has; the row is not removed again. Each local rater is offered every subject. Two pilot raters work one after the other on the same run directory (`surfcensus review PATH --run-dir DIR --rater NAME --blind`), and `surfcensus finalize` then writes the final labels.
- A skipped subject keeps its assignment, which stays open, and waits: it comes back after the subjects not yet shown, in the order of skipping (see "From subject to subject"). In every mode a skip is neither a completed rating nor a reason to give the subject to another rater. A repeat that is skipped waits like any other subject, and no second repeat of its subject is drawn meanwhile.

## Adjudication and review screens

Separate screens, available to raters with the `adjudicator` role, and only after their own assignment is done:

- Adjudication: subjects where raters differ by two scale steps on any criterion. Shows the same viewer, each rater's ratings and notes, and asks for a final label.
- Unblinded screens are open in local mode unless the run is blind or has more than one rater; then they open for a rater only once nothing is left to rate, skipped subjects included. A closed route answers 403; a browser that asks for the page `/review` gets the reason as a page with a way back to the rating screen. The adjudication screen, the dashboard and the error map are team mode (milestone 5); the review screen is built.
- Review: unblinded browsing of all subjects, sortable by any metric and filterable by rating, dataset pseudonym and FreeSurfer version (the app never shows accessions, see decision D10), with a scatter of composite score against final rating. Clicking a point opens that subject with all raters' marks and the fsaverage outlier-cluster centroids overlaid in different colours, so agreement and disagreement between eye and metric are visible in place. This is also the screen a local user ends up using most after rating.
- Adjudication also shows each rater's marks; the adjudicator can keep, move or delete marks to produce the final mark set.
- Census error map: an fsaverage flatmap or inflated view (static image regenerated on demand) showing the per-vertex frequency of final marks, by class, next to the mean z-map. Team lead and adjudicators only.

## Progress and agreement dashboard

For the team lead: completed ratings by rater and per day; median seconds per subject per rater; inter-rater agreement per criterion (weighted Cohen's kappa for pairs, Krippendorff's alpha for all raters), recomputed on demand; intra-rater agreement from repeats; drift in a rater's rating distribution over time.

## Event logging

Every rating stores rater, subject, criterion values, protocol version, app version, client timestamp, server timestamp, and time on subject. Interaction events are logged per subject: frames viewed, play and pause, mode switches, zoom, back-navigation, the tour. This answers a question the paper can use: how long careful looking actually takes, and whether raters who scrub more catch more.

Skips are recorded twice. The assignment carries the time of its latest skip and the number of its skips (`skipped_at`, `skip_count`). And each skip is an event of type `skip`, written by the server when it takes the skip and never accepted from a page: its `ts` is the client's time of the skip, and its payload holds the server's time, the number of the skip for that assignment, and the time the subject was on a visible page in the visit that ended with the skip. A request that arrives twice is recorded once.

In the export, `skips` has one row per skip (assignment, number of the skip, subject, rater, client and server time, the time of that visit, and whether the assignment was rated in the end), so skips can be counted per rater and per subject, also for a subject that was skipped and never rated; `behaviour` gives, per completed assignment, the number of skips before the rating and the time of those visits together (`n_skips`, `ms_skipped_visits`), and the time of the visits that ended with a submission that a revision replaced (`ms_revised_visits`). A `back` from the screen that says nothing is left belongs to no assignment and is not logged.

Time on subject is per visit. `ms_on_subject` of a submission is the time of the visit that ended with the submission; the time of earlier visits to a subject that was skipped is in its skip events. Time while the page is hidden or the tour is open is not counted. What ends a visit, and where its time goes:

- A submission or a skip ends it, and carries its time. A revision (`Backspace`, then `Enter`) is a visit of its own: its submission has the time of that visit, and the submission it replaces keeps the time of the visit that made the first rating. The export has the three kinds of visit in three columns of `behaviour` (`ms_on_subject` for the current submission, `ms_skipped_visits`, `ms_revised_visits`), and their sum covers the visits that the counts of frames, plays and pauses in the same row are taken over.
- A reload of the page does not end it. When the page is left, it keeps the time of the open visit in session storage, per assignment, and the page that shows the subject again goes on from it. This holds for a subject without a rating; a subject shown for revision is not shown again by a reload, and the time of that look is then in no table. Without session storage the time before a reload is not counted.
- The clock starts when the subject is opened, not when its tiles are loaded, so the time includes the loading. The first frame shows as soon as its tiles are there; the others follow. For a subject that was fetched ahead that is next to nothing; for the first subject of a page load it was 2.8 and 3.0 s in the two sessions of the review on the two real subjects (the `load` event has the time; it is not in the export). An analysis of looking time should know that the first rating after every page load is longer by that much.

How the queue was ordered is in the export as well, since an order by score is an order effect: `ratings.seq` is the place of the assignment among those of its rater, `ratings.blind` says whether the local queue was the seeded shuffle (true) or the order of the scores (false) when the submission was stored, from the record the server makes of the setting at its start, and the manifest has the setting, the seed and the repeat settings as they are when the export is made (`queue`).

## Database tables (`app.sqlite`)

The tables are the SQLModel classes of `app/models.py`. Timestamps that the server writes are UTC ISO 8601 text with milliseconds. A time that a page sends (`submissions.client_ts`, `marks.created_ts`, `events.ts`) is stored as it came and is not checked: the shipped page sends such text, another client can send any text of up to 64 characters, so an analysis must not take these three columns for times, or for times that order with the server's (a mark can carry a `created_ts` later than its `deleted_ts`). A column with a fixed vocabulary (written `[a|b]` below) carries a CHECK constraint, so the database refuses a row with a value the app does not know. Code outside `app/` reads the tables with the standard library, by these names.

- `raters(rater_id, name, token_hash, role[rater|adjudicator|lead], active, created_at)`; `name` is unique.
- `assignments(assignment_id, rater_id, subject_uid, kind[calibration|main|repeat|adjudication], seq, render_key, assigned_at, completed_at, consensus_served_at, skipped_at, skip_count)`, unique on `(rater_id, seq)`. `seq` numbers a rater's assignments from 1. `render_key` is `renders.render_key` of the render that was shown (`rendering.md`), so a rating stays with the tiles it was made on. `consensus_served_at` is for calibration: once the rater has seen the consensus, the rating is final. `skipped_at` is the server's time of the latest skip (null for an assignment that was never skipped) and `skip_count` the number of skips (0 by default). The two columns were added without a new schema version, because no `app.sqlite` existed outside tests when they were.
- `submissions(submission_id, assignment_id, revision, current, unratable_reason, other, note, protocol_version, protocol_sha256, calibration_sha256, app_version, client_ts, server_ts, ms_on_subject)`, unique on `(assignment_id, revision)`: one row per press of `Enter`. A revision (`Backspace`, then `Enter`) is a new row with the next `revision`; `current` moves to it, and the marks of the superseded submission get `deleted_ts`. Nothing is overwritten. A unique index (`ux_submissions_current`, on `assignment_id` where `current = 1`) lets an assignment have at most one current submission.
- `ratings(rating_id, submission_id, assignment_id, criterion[A|B|C], value)`, unique on `(submission_id, criterion)`. An unratable submission has no rating rows. The view `ratings_flat` gives, for current submissions, the columns this table was first described with: `rating_id, assignment_id, criterion, value, note, protocol_version, app_version, client_ts, server_ts, ms_on_subject`.
- `marks(mark_id, submission_id, assignment_id, class[A|B|C|X], panel[sagittal|coronal|axial], slice, frame, mode[sync|seq], px, py, vox_i, vox_j, vox_k, tkr_x, tkr_y, tkr_z, ras_x, ras_y, ras_z, hemi[lh|rh], pial_vertex, pial_dist_mm, white_vertex, white_dist_mm, fsavg6_vertex, vertex_surface[pial|white], box_dist_mm, created_ts, deleted_ts)`. `slice` is the position in the panel's slice list of the tile that was clicked, which is what the geometry needs; `frame` and `mode` are the playback context of the click and may be null. `vox_*` are continuous voxel coordinates. `vertex_surface` says whose vertex `fsavg6_vertex` is the counterpart of, and `box_dist_mm` is the distance from the mark to the view box (0 inside).
- `events(event_id, assignment_id, ts, type, payload_json)`; `assignment_id` is null for an event of the run, such as a changed setting. An event of type `skip` is written by the server (see Event logging). `payload_json` is JSON: a payload with a NaN or an infinity, which Python's parser takes from a request body and JSON has no text for, is refused with the batch it came in.
- `final_labels(subject_uid, criterion[A|B|C|unratable], value, method[consensus|adjudicated|single])`, key `(subject_uid, criterion)`. In the row of criterion `unratable`, `value` is the position of the reason in a list that the database keeps with the labels (`app_meta.unratable_reasons`, below), so the label means what it was written as whatever `protocol.yaml` holds when it is read. With one rater each submission rewrites the subject's rows (method `single`). With several, the rows come from `surfcensus finalize`; a later submission removes the `single` rows and the stale `consensus` rows of its subject until `finalize` runs again. A skipped subject has no label from the rater who skipped it.
- `final_marks(final_mark_id, subject_uid, class[A|B|C|X], tkr_x, tkr_y, tkr_z, hemi, pial_vertex, fsavg6_vertex, source_mark_ids)`; `source_mark_ids` is a JSON array. A mark of class A, B or C is a final mark only when the final label of that criterion is not none, from one rating as in a consensus: a rater can rate a criterion minor, mark it and then rate it none, and that mark stays in `marks` and is no final mark. Marks of class X always are.
- `app_meta(key, value)`: `schema_version`, `run_id` and `mode`, which tie the database to its run, and `unratable_reasons`, a JSON array of reasons: what the values of the `unratable` final labels are positions in. The row is written with the first final label. It is the list of `protocol.yaml` as it was when final labels were last written (when the server started, at a submission, by `finalize`), followed by any reason that the protocol has lost since and that a label still names. Whatever writes final labels first moves the list and the labels to the present protocol, by reason; whatever reads them (the review screen, `export`, `calibration freeze`) names a label through this list and not through the protocol. So a `protocol.yaml` whose reasons were put in another order, or that lost one, does not rename a stored label, also before a server has seen the edit. A database without the row is read with the protocol of the run.

`app/db.py` opens the database (`open_app_db`: a new file gets the tables, the view and the `app_meta` rows in one transaction; a file of a newer schema, of another run or of the other mode is refused and left as it was) and stores a submission (`store_submission`). The server fills the tables: one request is one transaction, so a refused rating or skip stores nothing. Every transaction of the app on this database starts with `BEGIN IMMEDIATE`, also one that only reads, so that two requests that read and then write wait for each other; a team database uses the write-ahead log, a local one the rollback journal, with busy timeouts of 60 s and 30 s.

## API

- `GET /api/next` → next assignment for the authenticated rater: `subject_uid`, frame URLs, protocol version.
- `POST /api/rating` → criterion values, marks (class, panel, slice, px, py, and the frame and playback mode as context), note, timing; returns the next assignment. The server validates marks against `render.json` and resolves coordinates and vertices. A rating that arrives twice is one submission when it carries the client's time of the press (`client_ts`), which the page always sends, and the same content: the values, the unratable reason, the other flag, the note, the time on the subject and, of each mark, its class, panel, slice, pixel, frame, mode and its own time when the body gives one. A body that differs in any of these under the same client time is a revision. The field is optional in the shape: a client that leaves it out and sends a body again stores a revision each time, since nothing tells a second send from a deliberate identical revision.
- `POST /api/skip` → the assignment id, the client's time and the time on the subject; nothing of the rating. Returns the next assignment, in the shape of `GET /api/next`. Sent twice with the same client time it is one skip. 409, as for a stale rating, when the assignment is not open for the rater: it has a submission, or its render is no longer the subject's.
- `POST /api/events` → batched interaction events. An event of type `skip` is refused: the server writes those itself. So is a payload that holds a number JSON has not (NaN, infinity; 422, and nothing of the batch is stored).
- `GET /api/back` → the assignment the rater left last, rated (for revision, with its rating) or skipped (as it is, without one).
- `GET /api/review?sort=&filter=` → adjudicator or local mode only.
- `GET /api/stats` → dashboard numbers.
- `GET /api/export` → CSV of ratings, team lead only.

The request and response bodies are the pydantic models of `app/schemas.py`, and its `ROUTES` lists every route with the shapes it takes and answers with; the frontend and the server are both written against that module. A page receives the geometry of a render (`RenderOut`) without the sizes and digests of the source files and without the template's digests: `render.json` itself is never served. The progress of an answer (`ProgressOut`) says how many skipped subjects wait. The server (`app/server.py`) answers every route of `ROUTES` and no other, in local mode. It is tested through Starlette's test client and over a socket on synthetic runs, and was driven in Chromium on renders of the two real subjects (`ds000102` sub-01, `ds003020` sub-UTS01): tour, rate, mark, submit, reload, skip, back, revise, review.

Of skip, the refused `Enter` and the tour, what is built: the rule of the queue and the write of a skip (`app/queue.py`, `app/db.py`), tested on a database, and the page, tested in a browser against a stand-in for the server (`tests/frontend_mock.py`) at windows 1400 and 400 px wide and at two that are too low for the page (400 by 440 and 1366 by 560 px), on synthetic subjects. The route `POST /api/skip` is served: 404 for no such assignment, 403 for another rater's, 409 for one that is not open, and a request sent twice is one skip. `tests/test_browser.py` drives the real pages on the real server, on a run that `review` made of two synthetic subjects.

What a local server holds against. It binds to `127.0.0.1` or `localhost` only. A request whose `Host` header names another host gets 403 (DNS rebinding). A request for `/api` or a POST that the browser says comes from another origin (`Origin`, `Sec-Fetch-Site`) gets 403; a POST must be `application/json` (415) of at most 16 MB (413). Every answer forbids framing and embedding by another origin and loads nothing from elsewhere (`Content-Security-Policy: default-src 'self'`). `/static` serves the files of the package and `/media` the tiles, the poster and the video of a ready render under the names a render writes; any other name, a path that climbs and a link that leads elsewhere are 404, and no answer is a redirect. A tile asked for with `?v=<render_key>` of the present render is cached as immutable, any other request is `no-cache`. An error of a rating route says nothing an exception said.

## Commands

- `surfcensus review PATH` prepares the run and opens the app (`architecture.md`). The app opens when the first subject of the queue is rendered (`--serve-after N` waits for N renders, 0 for all); the others are rendered behind it in queue order, a batch of `--jobs` subjects at a time, and the rating screen says how many renders are done and takes the next subject by itself when it is there. Ctrl-C or SIGTERM stops the server, then the renderer, which finishes the batch it is on; the command then says how many subjects are rated and how many are left. The same command again continues.
- `surfcensus serve --run-dir DIR` opens the app on a run that is rendered and renders nothing. It refuses a local run whose directory lies in a directory its source reads (a run that was moved there), before `app.sqlite` is opened, as `finalize` and `calibration freeze` do. It refuses a run whose renders are out of date and says so when selected subjects are not rendered yet, and when the run has no subject to rate. The run need not come from `review`: one made with `inventory`, `metrics` and `render` is served with every present subject in its queue (Assignment logic).
- `surfcensus finalize --run-dir DIR` writes the final labels and marks from the ratings of all raters, prints the counts, names each rater who still has subjects to rate (skipped ones included) and lists the open subjects with what each rater said.
- `surfcensus calibration freeze --run-dir DIR [--subject UID]...` writes `calibration.yaml` from the final labels and marks; the file is never overwritten.
- `surfcensus export --run-dir DIR [--out DIR] [--format csv|parquet]` writes the tables for analysis (`phases.md`, phase 8). An output directory that is, or lies inside, a directory that a local source reads is refused with exit code 2 before anything is written, as a run directory there is: a source is only read. That is the subjects directory (a subject directory, a link to one), the directory that a linked subject or a linked `fsaverage` in it leads to, named through the link or at its target, and the `fsaverage` of a FreeSurfer install (`architecture.md`, Sources). The default `export/` counts as well, so a run that was moved into its source gets no export. Inside the run directory an `--out` is taken only in `export/` or `release/` or below them: an export has tables named `clusters` and `scores`, as the run has, and one written into `tables/` would replace the run's own.

## What was measured, and what was not

No human has rated anything with this app. Every rating in the checks below was made by a script that drives a browser, so the time a rating takes, how raters use the tour, the skip key and the marks, and whether the colours and sizes suit a rater's eye are not known. Phase 0 measures them.

Measured 2026-11-13 on one machine (4 processors, local disk, Chromium 153 headless, server and browser on the same machine) on a local fMRIPrep tree of the two real subjects (`ds000102` sub-01, FreeSurfer 6.0.1; `ds003020` sub-UTS01, 7.3.2) with the real `fsaverage` next to them:

- From `surfcensus review PATH --no-browser` to the line that names the app: 11.0, 11.2 and 11.8 s in three new run directories, of which the first render takes about 7 s, the metrics 1.4 s and starting Python about 1.5 s. The second subject was rendered behind the app. The same command on the finished run: 0.8 s. One real subject alone, from Python: 9.7 s, and 0.1 s the second time.
- From opening the page to the first subject with every tile loaded: 3.2 s (twice), while the tiles of the next subject were being fetched as well.
- What the browser loads for one subject: 352 tiles (176 grayscale, 176 contour) in 352 requests, 4.06 MB for sub-01 and 3.77 MB for sub-UTS01 (4.27 MB with response headers for sub-01), and one answer of about 10 kB that carries the geometry. The page itself is 12 requests and 138 kB, fetched once per page load. No request went to another host, and the page sets no cookie.
- From `Enter` to the next subject with every tile loaded: 0.13 s, with two requests in between (the events and the rating), because the next two subjects are fetched ahead. From `k` to the next subject: 0.10 s.
- A full session as a first-time rater (tour, a refused `Enter`, a skip, two ratings with four marks of classes A, B and C on pial and white contours of all three panels, back, a revision, the review page) made 1,106 requests, those answered from the browser's cache counted, all answered 200.
- Each of the nine marks stored in these sessions (at a window of 1400 by 900 px, at 2x zoom, and at 900 by 700 px with a device pixel ratio of 2) was compared with a computation from the files alone (nibabel and numpy, no code of the package): tkr and scanner RAS from the T1 header within 1e-9 mm, the hemisphere, the nearest pial and white vertex with their distances by a search over every vertex, and the fsaverage6 vertex as the nearest template vertex to the mark's vertex on the registered sphere. All agree. The stored points lie 0.00 to 0.18 mm from the surface whose drawn contour was clicked. A mark at the right edge of the radiological coronal panel is at tkr x = -44 mm, in the left hemisphere.
- `ms_on_subject` of the three submissions and of the skip equal the time the subject was on screen in that visit, as the event times give it, within 12 ms; the 5.8 s that the tour was open are not in the 1.5 s of the skipped visit.
- Blindness: everything the rating screen received in that session (the page, and the answers of `/api/session`, `/api/next`, `/api/skip`, `/api/rating` and `/api/back`) was searched for the two subject labels, the dataset pseudonym, the two accessions, the names of the score columns, the FreeSurfer versions, the template id, paths of the run and of the data, and the key `inputs`. None occurs. What a queue that is not blind does reveal is its order: the first subjects are the worst-scoring ones. `--blind` removes that (below).

On the 26 subjects of `ds000102` with Tier A and B files only (no T1, no surfaces), `review` took 35 s, said which files every subject lacks, scored them, and ended with exit code 1 and `No subject of this run can be rendered, so there is nothing to rate.`; `export` then lists each with `render_reason` `missing surface`. On a synthetic set of 32 subjects the app offered the two subjects that could not be fully scored first, then the others in the order of `primary_rank` (the three with a planted patch first); with `--blind` it offered them in the seeded shuffle, and the position in the queue had a rank correlation of 0.07 with the score.

From an installation that is new (measured 2026-11-14, same machine, the wheel installed with its dependencies from the package index, the command run from another directory; two synthetic 1 mm subjects of 160 voxels a side without an fsaverage, which are lighter than real ones: `review` is at the app in about 3.5 s for them, where the two real subjects above took 11 s. So these times say what the install and the start cost, not what real subjects take):

- `pip install` of the wheel in a `python -m venv` environment: 28 s on Python 3.13 (pip 26.2.2) and 29 s on 3.11 (pip 24.0) on a fast connection, 50 and 51 packages. Then `surfcensus review PATH --no-browser` named the address 4.6 and 4.7 s after the command was given; the command's own clock, which starts after its imports, said 3.4 and 3.6 s.
- `uv venv` and `uv pip install` with the wheels in uv's cache: under 1 s. The first `review` then took 8.0 s (3.13) and 8.3 s (3.11) to the address, of which the command counted 4.7 and 4.8 s: uv does not compile the packages at install, so the first import of each does. The same command again on the finished run: 1.9 and 2.0 s to the address (0.8 s by the command's clock), so starting Python and importing takes about 1.1 s of any call.
- `uvx --from WHEEL surfcensus review PATH --no-browser`, which makes the environment itself: 8.9 s to the address with the wheels in uv's cache, 1.7 s the second time. `uvx --from git+file://COPY`, a git repository made of a copy of the working tree, which is the form of the line that names the repository: 8.0 s.
- In each of these the page and `/api/next` answered, and SIGTERM ended the command with exit code 0 and its closing line, also through `uvx`.

Not measured: an install over a slow connection or with an empty pip cache on a home line (227 MB of wheels), macOS and Windows, a network between server and browser, a slow disk, more than 32 subjects in a queue, any browser but Chromium, a screen reader, a touch screen.

## Throughput budget

At 15 s per subject (before marking time on positive subjects) and 2 raters per subject, 5,000 subjects take about 21 rater-hours per rater. The 15 s are an assumption: no rating was timed (see above). Phase 0 measures actual time per subject; phase 5 sizes the sample from it. Loading is not what limits a rater: the next subject is on screen 0.13 s after `Enter`.
