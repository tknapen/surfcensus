# EPI movie interface

Specification of the second interface of the tool: the raw and the preprocessed EPI series of every functional run of a session, as moving pictures on the same slices with the FreeSurfer white and pial surfaces as contours, served by the same app and prepared by the same command as the surface interface. Written on 2026-11-14 from the prototype of stage 1 (`prototypes/func_viewer/`), its independent review, the code of the package and the sources of fMRIPrep on PyPI. Step 1 of its build order is built (2026-11-14; section "Step 1 as built"); steps 2 and 3 are not, and the shared names of step 2 are laid out (section "Step 2: the plan (E2.0)"). Where this document and `prototypes/func_viewer/README.md` disagree about the package, this document holds; the README remains the record of what stage 1 built and measured.

## Why, in his words

Two decisions of Tomas Knapen govern everything below.

D36: "the reason that we're showing both the raw and the fully pre-processed EPI time series as this video is exactly because we want to visualize the effects of pre-processing on the sort of movies, the time courses of the EPI volumes. And so it would be quite insane to perform all manner of processing on the raw data, because the whole idea is to show the difference between that and the pre-processed, and to also show how the pre-processed data now conform to, for example, the FreeSurfer surfaces, etc., etc., right? So it's meant to serve as a quality control type tool for both the raw data and the pre-processing pipeline at the same time." With his decision of 2026-11-13: "Raw bids means single per session registration to the anatomy to ensure comparability across sessions, processing stages, and subjects."

D37: "to present people with a full quality control visualization tool that shows, perhaps even on one page, could be different pages, a surface segmentation quality control interface and a raw and pre-processed EPI movie interface. All of those exist for people to be able to judge visually the quality of acquisition and processing on a full session or multiple sessions of fMRI and MRI data."

O29 adds, for later: preprocessed series of other pipelines, found through "a sort of connector interface ... as long as they conform to the T1 weighted suffix or something like that".

What follows for this design, and is not to be weakened by any choice below:

- The raw stream is the raw BIDS series placed once per session into the anatomy by one rigid transform, that of the session's anchor run, and nothing else: no motion, slice-timing or distortion correction, no filtering, no other processing, also not to explain a difference.
- The preprocessed stream is the series as the pipeline wrote it, voxel for voxel on its own grid.
- The difference between the streams, and the raw stream's misfit to the contours, is the signal. It is never removed, never marked as a fault of the tool, and never a reason to leave a run out.
- Both uses are named wherever a view is described: quality control of the raw data (motion, artefacts in single volumes, distortion and dropout as acquired, movement between runs) and quality control of the preprocessing (what was corrected, what remains, how the preprocessed series fits the white and pial contours).
- The interface is a viewer (D37, O27). A judgement protocol is proposed below, marked as such, and is not built without his word.
- Nothing of a public dataset's functional data is shown outside the local machine (O28).

## One continuous stream (Tomas, 2026-11-14)

> "For the functional interface it's important that the epi videos span all acquisitions in one continuous video stream - no skips between runs other than the ones imposed by the image content itself."

This was said after the rest of this specification was written, and it overrides the page and the pack wherever they treat a run as a film of its own (decision D38). What follows from it, in the reading of the building session:

- One timeline per session. The movie of a session is all its packed runs, one after the other in acquisition order, as one stream of volumes: playing does not stop, pause, blank or wait for a key at the end of a run, the time slider spans the session, and a loop is a loop over the session. Choosing a run is a jump on that timeline, not another film.
- Nothing but the images changes at a run boundary. The slices stay at the same place in the anatomy: when the next run lies on another grid, each plane goes to the slice of the new grid that is nearest, in millimetres, to the one that was shown. The intensity scale is handled as the next paragraph says. The view, the zoom, the layers and the playback rate stay as they are. What the page says about the run (its name, whether it is the anchor, its displacement under the session's registration, the corrections the pipeline applied) changes with the run, beside the pictures and without interrupting them, and the boundary is a mark on the timeline and in the traces.
- What does change is the content: the head lies elsewhere in the next run, the field of view or the voxel size differs, the contrast differs. Those jumps are the data, and under the session's one registration they are exactly what the raw stream is there to show (D36).
- Loading follows the stream: the chunks of the next run are fetched before the last volume of the present one is shown, so that the boundary is not a wait. Where the machine cannot keep up, the clock waits and no volume is skipped, as within a run.
- Several sessions of a subject are one stream as well (his answer of the same day, below): after the last run of a session the first run of the next follows, in the order of the sessions, with the same rules as at a run boundary.

His answer to the questions this raised, the same day:

> "Yes also sessions as a continuous stream. It would be great to be able to toggle the intensity normalization on and off. And the voxel grid for each session is a single rigid body step so I realize that the voxel grid would move between sessions, which is fine if we don't want to actually create new intermediary files, just visualize the existing files"

What follows from it:

- Sessions. The stream of a subject is every session in order and every run within it. Each session keeps its one rigid registration (D36), so the placement of the raw data, and with it the voxel grid on screen, changes at a session boundary; that is accepted, and nothing is resampled to a grid common to the sessions.
- No new image files. The tool visualizes the files that exist. It writes no resampled series, no volume and no derivative into or beside the data, and it does not build a common grid across runs or sessions for the sake of the display. What it does keep is its own display cache in the run directory (the packs: 8-bit values on the grid of each preprocessed series, with the raw stream placed once by the session's registration), which exists so that a browser can play 60 volumes a second and which can be deleted and made again from the data at any time. He confirmed this reading the same day: "Yeah the display cache is perfect, just no new nifti files". So the rule is: the tool never writes a NIfTI file, or any other image volume, anywhere; the display cache in the run directory is what it writes.
- Intensity normalization can be switched on and off on the page. On: every run is shown under its own window (0 to the 99.5th percentile of its preprocessed series in the brain mask), so that runs of different gain can be compared as pictures. Off: one linear window for the whole stream, 0 to the largest upper limit of its runs, so that a run or a session that is brighter or darker than another is seen as it is. The pack stores each run under its own window, which keeps the most of each run in 8 bits, and records that window; the page makes the other scale from it with a table, so the toggle needs no second copy of the data and no new file. With normalization on, a value on screen is the stored value; with it off, it is the stored value through that table, and the page says which is on. The default is off, since D38 asks that the display add nothing at a boundary; the key is `n` (to be checked against the page's other keys).

Where the sections below speak of a run selector, of a window per run or of playback of a run, read them with this section.

## Step 1 as built (2026-11-14)

`surfcensus epi-session PATH --subject LABEL [--session LABEL]... [--raw DIR] --out DIR [--epi-runs GLOB]... [--serve] [--port N] [--no-browser]` packs the functional sessions of one subject of a local folder and, with `--serve`, plays them in the browser. No run directory, no catalog, view only. The choices are decision D39; where this section and the sections below disagree, this section holds, because it describes what exists.

- Modules: `func/bids.py`, `func/itk.py` and `func/connectors/fmriprep.py` (the connector), `func/session.py`, `func/geometry.py` and `func/contours.py` (anchor, raw affine, frame check, grids, voxel map, contours), `func/pack.py` and `func/packread.py` (format `epi-pack` version 2), `func/api.py` (`find_folders`, `epi_subject`, `epi_session`), `app/epi.py` and `app/epi_schemas.py` (a server for packs alone and the shapes a page receives), `app/static/js/epi/` and `app/templates/epi.html` (the page).
- One stream (D38). Without `--session` every session of the subject is packed, each into `<out>/<session_id>/`, and `<out>/stream.json` is written last: the sessions in stream order (no session first, then `ses-` labels with numbers compared as numbers), each session's packed runs in acquisition order. The page plays that as one timeline and loops over the whole of it.
- Runs on different grids are packed, each on its own grid with its own contours. `runs on different grids` is a reason no run gets any more. Each grid lists its slice planes as a unit normal and distances in millimetres from the origin of the anatomy's world (`GridInfo.slice_normal`, `slice_mm`). The affine is not written out, but it can be worked back from the normals, the distances and the shape (the review recovered it to 4.8e-7), so what is served does hold the geometry of the preprocessed file; the server is local. The page remembers the point of the anatomy where the slices were last chosen and shows the nearest slices on every grid, so a change of grid and back returns to the same slices.
- Intensity. A pack stores each run under its own window. The page's normalization toggle (key `n`, default off) shows either the stored values (on) or one linear window for the whole stream, 0 to the largest upper limit of its runs (off), through a table of 256 values per run (`RULES["common_scale"]`). The difference view is raw minus preprocessed of the gray levels on screen; with normalization off it can differ by one gray level from the scaled difference of the stored values.
- Identity. A pack is identified by `content_digest`, a SHA-256 over its decoded values, stills, traces and contour bytes and over its defining JSON, not by its compressed bytes. `pack.json` lists every input by role, base, size and SHA-256 (no name, no path): the run's files, the anchor's transform, the fsnative transform and the FreeSurfer files of the contours. The cache token a page gets is a digest of digests the page receives anyway, never of `content_digest`, which covers the input digests (D20).
- Ids are 16 hex characters under a key drawn per output directory; no key is written. A second call into the same `--out` reuses the ids recorded in `stream.json` (sessions by label, runs by name); an `--out` that holds another participant's stream is refused.
- Reasons added to those below: `native space only`, `no confounds table`, `not fetched` (git-annex links without content and pointer files, found from the listing and the heads of files of at most 32 kB; the files are named in `runs[].detail`; the fetch command is step 2).
- An `--out` that is, or lies inside, the raw dataset, the derivative, the FreeSurfer subjects directory or what a top-level link in them leads to is refused before anything is written. Only `.json` and `.bin` files are written, and nothing into the three folders.
- The server (`app/epi.py`): the page at `/` and `/epi`, `GET /api/epi/stream`, `GET /epi-media/{session_id}/{path}`, `/static/...`, `/healthz`; localhost only, with the `Host`, origin, framing and content-security rules of the rater app's server. A pack file is served only when its name matches the pattern, a `pack.json` or `session.json` lists it, and it resolves inside its session directory; no JSON file of a pack is served. The decoded-memory limit (512 MiB) is a setting of the server, not of the browser.
- Corrections to the sections below, found while building: fMRIPrep's summary report says `Skipped (too few volumes)` in every version read from 20.2.3 on, not only in 25.2.3; the tolerance of the raw-affine arbiter is an absolute 1e-4 on the affine's entries; "acquisition order, the anchor first" is acquisition order (the anchor is the first run acquired by rule, and no run is moved for it); multi-echo runs are recognised from the raw listing (several raw series with one run key), because fMRIPrep's default output carries no `echo` entity.

Measured (one shared 4-core Linux machine, Python 3.13):

- The worked example (`ds000102` sub-01, 2 runs of 146 volumes on 109,368 voxels) through the command: the address 33.7 s after the start, 14.5 and 14.7 s per run. Every decoded volume of both streams equals the pack of the stage 1 prototype (window, SD window and stills included); the frame numbers are those of stage 1 (0.829 and 1.087 mm). One run costs about 26 MB above the imports, against 0.42 GB for the prototype's read of the whole series, and 0.51 to 0.53 bytes per voxel, volume and stream.
- Portable against numpy voxel map on the anchor run of the worked example: no 8-bit value of the 292 decoded volumes differs. One run; the question of "Determinism" stays open for others.
- The page, in headless Chromium 153 on synthetic packs of 8 runs of 24 volumes on 4,284 voxels: 12.02, 29.98 and 59.67 volumes painted per second at 12, 30 and 60 asked, the painted sequence contiguous through every run boundary, grid change, session boundary and the end of the stream, no stall; first picture 0.36 s after the page opens.

Not checked when step 1 was put together (the review below filled part of it): a real pack in the page; a long run; a real session with two grids; an oblique grid on screen; the 23.2 and later generation, native-space series and multi-echo data on real files; any browser but Chromium; another machine or zlib build. No person has looked at any of it.

### After the independent review (2026-11-14)

Two reviewers checked step 1, one the data path and one the page, the server and the command, and two rounds of fixes followed. The real multi-run sessions of stage 1 no longer had their series on disk, so both worked on the worked example (W) and on variants staged from its real files: three runs with a moved sform, a false 15 mm transform and acquisition times (X), a second, oblique grid (Y), one run of 584 volumes on 495,504 voxels (Lg), and one made-up subject of two sessions and six runs whose runs are links to the two real ones, one with its transform shifted by 12 mm (M).

Confirmed by recomputation that shares no code with the package:

- The raw stream: all 146 volumes of every run of W and X against the reviewer's own ITK reading, arbiter, cubic `map_coordinates` and quantisation, 0 voxels differing; a shift search of the raw mean against the preprocessed mean peaks at zero shift in every case (r 0.994 to 0.9997), also for the run with the moved sform, which the arbiter draws through its qform. The code path from file to byte is one read, one resampling, the window and the quantisation; nothing else touches the raw values.
- The preprocessed stream equals the file in every volume. The contours lie within 0.011 voxel of the reviewer's own cut in all three planes, also on the oblique grid; the slice planes of a grid agree with its affine to 5e-7 mm.
- The frame check agrees to 1e-4 mm with the reviewer's own on W, X and the real transforms and masks of four stage 1 sessions, both flagged runs among them.
- Two processes give the same content digest and, on one machine, the same bytes. Portable against numpy voxel map on four more runs (a non-anchor run of W, three of X): no 8-bit value differs.
- The sources are untouched, and only `.json` and `.bin` files are written.
- The page on real packs (W, and M with a 48 MiB limit as well): the painted sequence is contiguous at 12, 30 and 60 volumes per second through every run boundary, the session boundary and the end of the stream, no repeat, skip, blank frame or stall; first picture 0.8 s after the page opens; 720 canvas comparisons against the reviewer's own decoding with normalization on and off, in every view and still, 0 differing; 40 random jumps exact. The server refused every foreign host and origin, climbing path, JSON file and link out of a session in about 8,000 probes.
- Cost: W 31 s; the long run 215 s, 0.37 s per volume of both streams, peak memory 315 MB.

Changed after the review:

- One anchor per session, whatever is packed (D36). The rule (acquisition time of every run's raw sidecar, else name order) is decided over every run of the session, and the anchor is the first in that order whose stored transform to the anatomy reads, before `--epi-runs` and before any reason a run is not packed. Before, the anchor was chosen among the packable runs, so a filter, or a first run without its confounds table, moved the raw data of every other run. A run left out by `--epi-runs` is listed as `not selected`, keeps its place in the session's order (`runs[].session_order`) and can be the anchor without being packed (`registration.anchor_packed`); when the first run by the rule has no readable transform the next is the anchor and `registration.note` says so. On W, `--epi-runs '*run-2*'` now gives run 2 a pack whose content digest and every decoded value equal those of the full session.
- A second call into the same `--out` keeps every run whose pack is current (whole on disk, same version, settings, anchor, grid, and inputs by role, size and SHA-256, the session's inputs included); `--force` packs again. On W: 3.1 s against 30.7 s. The earlier `stream.json` and `session.json` stay until the new ones are renamed over them, so an output that is being served stays servable and an interrupted call leaves one that can be served. Packs and contour folders that the new `session.json` no longer lists, those of runs not selected included, are removed after it is written.
- An `--out` inside a directory that holds the target of a linked input file is refused (the per-file links of a DataLad clone). A series that is a link without its target is said to be one. Uncompressed `.nii` files of a derivative are read like `.nii.gz`.
- The command names the raw dataset by the path given, says per session which runs are flagged and when the anchor is the outlier or is not packed, as statements about the raw data, and after a signal says which stage was stopped.
- The page. Nothing on it moves at a run, grid or session boundary: the text that depends on the run sits in a strip of fixed height under the timeline, with the whole text in an overlay (before, the panels jumped 17 to 57 px at boundaries). The views come directly under one bar: at 1400 by 900 the single views take 0.36 of the window (0.19 to 0.22 before) and the page does not overflow; side by side stays at 0.22, limited by the width. The timeline carries the runs as labelled segments, which are the jump targets, and ticks on the slider; session boundaries are stronger than run boundaries and the cursor has a colour of its own. In the difference view the contours are green (white surface) and magenta (pial) over a dark outline, since the difference colours are those of the series views' contours. The orientation letters sit in margins above and below the canvas and cover no voxel. A slice step moves the remembered place along that plane's normal only. The flagged-run notice is amber and worded as a statement about the raw data. Each correction carries its own source. The page says that the raw series is resampled once, by cubic interpolation, onto the grid of the preprocessed series, and that this smooths single-volume detail slightly and enters the difference. A page whose packs were made again says so and offers to reload.
- At a change of grid the canvas inside its fixed box can change size, because the field of view differs; the box, the scale in pixels per millimetre and the centre do not.

Still open after the review: no real session with four or more runs, real flagged run, real distortion correction or real second session was played in the page, and no real session with two grids exists here; in the side-by-side view at 1400 by 900 about 200 px below the controls stay unused, and the corrections line is cut there (its whole text is in the overlay); the command is named `epi-session` and packs every session of a subject; a multi-echo or otherwise unpackable first run is the anchor of its session when its transform reads, which follows the rule and may not be what is wanted. No person has looked at any of it.

## Step 2: the plan (E2.0)

Work package E2.0 laid out on 2026-11-14 what the packages of step 2 share, before they are written side by side, as E1.0 did for step 1 (D39) and D30 and D32 did for milestones 3 and 4. The names, fields, columns and identifier values below are pinned in `tests/test_func_step2_interfaces.py`. Where this section and the older sections below disagree, this section holds until step 2 is built; what it changes in them is listed at its end. The decision entry is D40.

### The units under D38

Tomas's answer of 2026-11-14 ("Yes also sessions as a continuous stream") makes the page's unit a participant's stream over all its sessions, not a session. Three units follow, each with its own keyed identifier (`privacy.py`, under the run's `HashKey`: the salt of a local run):

- The participant: `participant_uid = privacy.participant_uid(key, source, deriv_identity, connector, participant)`. It is the unit of the EPI page, of its routes (`/epi/<participant_uid>`) and of the directory `epi/<participant_uid>/`, which holds the participant's `stream.json`. A participant is keyed by its BIDS label within the derivative, whether or not it has a FreeSurfer subject, so a participant without one has an identifier and a page. It is not the `subject_uid`: one participant can have several FreeSurfer subjects (`sub-X_ses-Y`, sessionwise processing) and one FreeSurfer subject has no functional data at all.
- The session: `session_uid = privacy.session_uid(key, source, deriv_identity, connector, participant, session)`, the label without `ses-`, None hashed as the empty string. It is the unit of the one registration (D36), of `session.json` and of the pack phase.
- The run: `run_uid = privacy.run_uid(key, source, deriv_identity, connector, participant, session, run_key)`, with `run_key` that of `func.bids.run_key` (the entities without subject and echo, the run index as a number), so `run-01` and `run-1` are one run and the echoes of a multi-echo acquisition one run, as in step 1. The specification listed `echo` among the parts; step 1 keys a run without it, and that holds.
- `pack_key = privacy.pack_key(run_secret, EpiConfig.pack_hash(), digests)`, the counterpart of `render_key`, over `<base>:<role>` to SHA-256 of every input of the pack (`PackInfo.inputs`), keyed with the run secret because the digests are those of public files (D20).

Each of the four HMACs has its own prefix, so none equals a `subject_uid` of the same parts. `deriv_identity` is the resolved derivative root when the functional source is first stored and is kept across `--relocate` (D21); the raw root enters no identifier. Values on a fixed salt are pinned in the test.

The ids of step 1's `--out` directories are another family: `func.func_id` under a random key per call (D39), never written. They and the ids of a run directory never meet, and nothing carries ids from one to the other. A run directory's `epi/` is written by the run path only. Nothing refuses an `epi-session --out` that names a place inside it; so the pack phase removes only directories that the catalog names (stale runs, sessions and participants of the run), never what a listing of `epi/` finds, and a foreign directory there is left alone.

### The catalog: migration 2

Migration 2 is the string `store._MIGRATION_2`, written in full by E2.0 and applied by E2.A (append it to `MIGRATIONS`, raise `SCHEMA_VERSION` to 2). Only `CREATE TABLE` and `CREATE VIEW`. Tables (private columns starred; no public view has them):

- `func_sources(func_id, dataset_id, connector, pipeline_version, generation, deriv_identity*, deriv_root*, deriv_found, raw_root*, raw_found, present, listed_at)`, unique on `(dataset_id, connector, deriv_identity)`. `deriv_found` is how the derivative was found (`surface layout`, `folder itself`, `derivatives folder`), `raw_found` how the raw dataset was (`given`, `dataset root`, `sourcedata/raw`, or null when none was). The specification's `raw_identity` is dropped: nothing is keyed by it.
- `func_participants(participant_uid, func_id*, participant, session_order, n_sessions, present, listed_at)`. `session_order` is the rule of the stream (`session label`, step 1's `StreamInfo.order`). This table is new: a participant is the unit of the page under D38.
- `func_sessions(session_uid, participant_uid, session_label, stream_order, subject_uid, fs_subject, anchor_run_uid, anchor_rule, anchor_why*, anchor_note, frame_limit_mm, shared_frame_plausible, anchor_is_outlier, n_runs, n_packable, n_grids, problem, present, inventory_hash*, inventoried_at)`. `stream_order` is the place of the session in its participant's stream (step 1's `session_key`: no session first, then labels with numbers compared as numbers). `anchor_why` holds an acquisition time and stays private, as step 1 keeps it out of a pack. `problem` is `not fetched: ...` (placeholders) or `unreadable: ...` (a plan that could not be made).
- `func_runs(run_uid, session_uid, run_key, name, entities*, session_order, acquisition_order, anchor, selected, packable, reason, detail, missing_roles, n_volumes, tr_s, echoes, display_layout, display_space, spaces, grid_key, grid_shape, voxel_mm, oblique_deg, raw_affine, reference_image, rigid, dof, corrections, motion_state, sdc, sdc_method, slice_timing, frame_mean_mm, frame_max_mm, frame_rotation_deg, frame_translation_mm, flagged, present, inventoried_at)`, unique on `(session_uid, run_key)`. What step 1 records per run is here: `session_order` (the place among all runs by the anchor's rule, the same whatever `--epi-runs` selects) and `acquisition_order` (the place in `session.json`), `selected` (false for `not selected`), the anchor chosen among all runs (`anchor`, true for one run of a session, packed or not), the reason with its detail, the grid of a packable run (runs on different grids are packed, each on its own), the corrections as statements (`corrections` is the JSON of `func.model.CorrectionsInfo`, with the three states beside it for queries). `entities` (task labels) is private.
- `func_files(owner_kind, owner, role, base, location*, variant, size*, etag_or_mtime*, sha256*)`, keyed by `(owner_kind, owner, role)`: `owner_kind` is `run` (owner a `run_uid`) or `session` (a `session_uid`, for `xfm_anat_fsnative`). One key column with its kind, not two nullable ones.
- `packs(run_uid, session_uid, config_hash, input_signature*, pack_key, status, reason, detail, path, n_volumes, bytes, content_digest*, anchor_run_uid, grid_key, packed_at)`, the counterpart of `renders`, one row per packable run that was packed or failed. A run that cannot be packed has no row; its reason is `func_runs.reason`. `content_digest` is private: it covers the defining JSON of `pack.json`, which holds the SHA-256 of the source files.
- The session unit is a row of the existing `work` ledger: phase `epi_session`, unit `session:<session_uid>` (`records.session_unit`).

Views `public_func_sources`, `public_func_participants`, `public_func_sessions`, `public_func_runs`, `public_packs` leave the starred columns out, and give labels, names, notes and details in a local run only, by the rule of `public_subjects`.

Opening. Writers migrate as today, so the first phase command of the new version brings a catalog to schema 2 whether or not the run has functional data. A read-only opener (`PublicCatalog.open`, `Catalog(readonly=True)`, used by `serve`, `export`, `finalize`, `calibration freeze`) accepts a catalog of schema 1 and reads it as one without functional data (every functional reader answers empty; a functional writer raises), instead of asking for a phase command first (E2.A changes the refusal in `_Db._prepare` and the test of `tests/test_store.py` that pins it). An older version of the package refuses a migrated catalog with `ConfigError("newer schema")` in every command, `status` and `serve` included: it can no longer open the run at all. That is the price the specification named, and it now falls on every run that the new version opens for writing, not only on runs with functional data (For Tomas, below).

Methods, each a stub that raises `NotImplementedError("work package E2.A: store.py")`:

- `Catalog`: `upsert_func_source(dataset_id, record) -> int`, `func_source(dataset_id)`, `upsert_func_participant(func_id, participant_uid, participant, n_sessions)`, `upsert_func_session(row)` (a session with its files and every run with its files, in one transaction), `func_inventory_hash(session_uid)`, `mark_func_absent(func_id, present_participants, present_sessions) -> list[str]`, `relocate_func(func_id, deriv_root, raw_root, raw_found)`, `func_lease_facts(session_uid) -> FuncLeaseFacts`, `func_participant_uids() -> dict[str, str]`.
- `PublicCatalog`: `has_func()`, `func_sources()`, `func_participants()`, `func_sessions(participant_uid=None)`, `func_runs(session_uid=None)`, `packs()`, `pack(run_uid) -> PackRow | None`, `pack_signature(run_uid)`, `epi_session_signature(session_uid)`, `record_packs(rows)`, `forget_packs(run_uids)`, `set_func_digests(owner_kind, owner, digests)`.
- Complete already: `store.EPI_DIR` (`epi`), `epi_participant_dir(run_dir, participant_uid)`, `epi_session_dir(run_dir, participant_uid, session_uid)`.

### The skip rules

Phase 2f has no ledger rows. Its skip test is `func_sessions.inventory_hash`: `records.func_signature` (16 hex characters of a SHA-256 over one line per listed file: base, role, location, size, modification time and variant) over every listed file of every run of the session and of the session itself, the FreeSurfer files of the contours (`t1`, `lh_white`, `rh_white`, `lh_pial`, `rh_pial` of the session's FreeSurfer subject), and the extra lines `EpiConfig.inventory_hash()` (the stage hash over `runs` and `frame_limit_mm`), the FreeSurfer subject's name and the run ids. A session whose hash is the stored one is skipped without opening a file.

The pack phase decides from the catalog and the run directory, never from a source file (D21). This replaces step 1's `pack.current_pack` for a run directory: `current_pack` reads and hashes the inputs (some 3 s on the worked example), which a skip decision may not do. In a run the decision is `func.batch.pack_plan` (E2.C), with two signatures that the catalog computes from the listing:

- `PublicCatalog.pack_signature(run_uid)`: the run's own files in `FuncRole` order; for a run that is not the anchor, the anchor's `xfm_to_anat` under the role `anchor_xfm_to_anat`; the session's files; the FreeSurfer files of the contours with their variants; and the lines `anchor`, `grid` and `fs` (the anchor's `run_uid`, the grid key, the FreeSurfer `subject_uid`). The arbiter (`reference`) is a role of the run and so in it.
- `PublicCatalog.epi_session_signature(session_uid)`: the session's `inventory_hash` and the pack signature of every packable run in acquisition order.

A session is skipped when its `epi_session` row is `done` for `EpiConfig.session_hash()` and its session signature, its `session.json` is there, and every packable run has a `packs` row `done` for `EpiConfig.pack_hash()` and its pack signature whose pack is whole on disk (`func.packread.pack_on_disk`, which lists the pack's directories and reads `pack.json` of the run directory, not a source). Otherwise the session is pending. Within a pending session the runs whose rows are current are kept: E2.C adds a keyword `keep` to `func.pack.pack_session`, with which the runs it names are taken from their `pack.json` as they are (whole on disk, else packed) and the others are packed without `current_pack`; without `keep` the behaviour of step 1 holds. A `failed` row that matches is not tried again without `--retry-failed`; `--force` ignores every row. A new anchor changes the pack signature of every run of its session, so the whole session is packed again. After a pack, the SHA-256 digests in `PackInfo.inputs` are copied into `func_files.sha256` (`set_func_digests`) and, for the FreeSurfer files, into `files.sha256` (`set_file_digests`), as a render's are. Outputs first, rows last: a run that is no longer packable or selected, and a session or participant that left, lose their directories first and their `packs` and `work` rows after.

### The layout on disk

```
run/epi/<participant_uid>/
  stream.json                     the participant's sessions in stream order (func.model.StreamInfo)
  <session_uid>/                  exactly step 1's session directory (func.pack)
    session.json
    contours/<grid_key>/<plane>.bin
    <run_uid>/pack.json raw/... pre/... traces.json
```

The directory of a participant has the layout of step 1's `--out`, so the server reads it with `app.epi.Packs` as it is. `stream.json` is written by the pack phase (`func.batch.write_stream`) from the catalog after each batch that touched the participant: the present sessions that have a packable run, in stream order. In a run directory it lists sessions packed or not yet; `Packs` shows those whose `session.json` is there, so the page grows as packs are made behind the app. That is a change of meaning against step 1, where `stream.json` is written last and marks a directory whose every session is in place; the model is unchanged. The catalog is the record; `stream.json` and the packs are the display cache, made again from the data at any time, and the tool writes no NIfTI file or other image volume anywhere (D38).

### The phases

- Phase 2f (`Run.func_inventory`, E2.Z, over `func.inventory.run_func_inventory`, E2.B), called after `Run.inventory` by `inventory` and `review`, a separate call so that the surface report keeps its numbers. Per local dataset: the functional source by rule (`func.discover.func_source_of`, from the dataset record the surface source found; a stored root is never searched below `derivatives/`, D35); per participant and session, the identifiers, the hash, and when it changed the plan (`func.session.plan_session`: anchor among all runs, frame check, grids, `not selected`, `not fetched`) and the statements of every run (raw affine, TR, corrections, rigidity, from headers, sidecars, reports and transforms). It reads listings, sidecars, reports, transforms, masks and the headers of the series, never a volume. A dataset without a functional part gives no row; a census run and `--no-epi` give an empty report.
- The pack phase (`Run.pack`, E2.Z, over `func.batch.run_pack`, E2.C): sessions in the order of `func.batch.session_order` (participants with a FreeSurfer subject in the order of the rating queue, `store.queue_order`; then the others by label; within a participant its sessions in stream order), a batch of `--jobs` sessions at a time through `batches.run_batches`, each session in a worker (`func.batch.pack_session_unit`). The unit of work is the session, not the run: `session.json`, the contours and the registration are per session, and `pack_session` is the tested unit. The price: a subject with one long session uses one core.
- The lease of a session is `RunContext.func_lease(session_uid)` (bound by `Run`, E2.Z), which yields a `records.SessionWork` made by `func.inventory.session_work` (E2.B) on the private side: the participant's derivative and raw dataset listed again, the listing compared with the stored `inventory_hash` (`SourceError("changed")` otherwise), the plan made again with the stored `run_uid`s, the inputs of every packable run and the contour job. The worker cuts the contours and packs. So the downstream code receives the plan, with file locations relative to the roots inside it and local paths valid while the lease is open; nothing of it is served, logged or written into a pack, and the EPI phases never run in a census run. `RunContext.pack_key` binds the run secret.
- `review` (E2.Z) keeps D33: inventory, the functional inventory, metrics, the first surface render, the app; behind the app the remaining surfaces are rendered first, then the packs, in the same background thread (`Run.pack(stop=...)` after the renderer), with Ctrl-C and SIGTERM stopping it between batches. When no surface subject can be rendered and some run can be packed, the app opens when the first session is packed, and `--serve-after N` counts sessions. `--raw DIR`, `--no-epi` and `--epi-runs GLOB` are stored and hold until given again.
- `surfcensus pack --run-dir DIR [--jobs J] [--epi-runs GLOB]... [--force] [--retry-failed]` (E2.Z): exit codes of the phase commands (0; 1 when a pack row is failed; 2 for usage, configuration and the machine; 130 and 143). Runs that cannot be packed are not failures. When nothing is pending it reads no source. `review` ends with 1 only when neither a surface subject can be rendered nor a run packed.
- What lies in the machine stops the phase with `ConfigError` and exit code 2, recorded against no run (D29); the rule of `render.batch._is_unwritable` applies under `epi/`.
- `status` and `export` print and write the functional tables in step 3; the catalog holds what they need from step 2 on (`func.batch.pack_status`, `pack_failures`).

### The configuration

`RunConfig.epi: EpiConfig | None = None`, and `EpiConfig.raw: str | None` (the root given with `--raw`, absolute, part of no hash). While it is None the defaults hold (`config.epi_settings`: on in a local run, off in a census run), and `dump_config` leaves the key out, so a run without an EPI option writes the bytes it wrote before and an earlier version reads its `config.yaml`. Once `--raw`, `--no-epi` or `--epi-runs` is given the key is written and an earlier version refuses it (`epi: unknown key`). A `config.yaml` of an earlier version loads with `epi` None. A census run refuses an `epi` section that is enabled or names a raw root. `CONFIG_SCHEMA_VERSION` stays 1. `STAGE_SCHEMAS` gains `func_inventory`. The derivative found by rule is in the catalog, not in `config.yaml`; the commands that refuse a run directory from `config.yaml` alone (the app, `finalize`, `calibration freeze`, `export`) find it again by rule from the stored root (`func.discover.stored_read_places`), so a run directory or an export inside the derivative or the raw dataset is refused there as well.

### Discovery: one implementation

`func.api.find_folders` (with `FuncFolders`, `refuse_out` and the places that are read) moves to `func/discover.py` unchanged and `func.api` imports it from there; `func_source_of(record, raw)` starts from the surface dataset record and calls it. The planning of `func.api.epi_subject` (the loop over sessions, placeholders, dangling links, the inputs of the packable runs) moves to `func/inventory.py` and `epi_subject` calls it. So `epi-session` and a run find the same folders and make the same plans. What the run path does not reuse of `func.api`: the ids (`func.func_id` under a random key), the writing of `stream.json` (the run writes it from the catalog), `current_pack` (the run decides from the catalog), and the printing.

### The app in a run directory

Step 2's exit check is `review` end to end with both interfaces, so the EPI page is reachable from a run in step 2; the subject pages stay in step 3. The main app (`app/server.py`) gains, in local mode, each unblinded:

| Method | Route | Answers with |
|---|---|---|
| GET | `/epi` | page: the participants of the run with their sessions, runs and pack states, each a link |
| GET | `/api/epi` | `EpiIndexOut` |
| GET | `/epi/{participant_uid}` | the EPI page of step 1, on the participant's stream |
| GET | `/api/epi/{participant_uid}` | `EpiStreamOut` of step 1, built by `app.epi.Packs` on `epi/<participant_uid>/` |
| GET | `/epi-media/{participant_uid}/{session_uid}/{path}` | a pack file, by the patterns and rules of step 1 |

They reuse `app/epi.py` and the page as they are: `Packs` gains the prefix of its media URLs as a parameter (default that of step 1, so the pack-only server is unchanged), and the page takes the address of its stream from the template, as it does. `EpiIndexOut` lists `EpiParticipantOut(participant_uid, label, subject_uids, n_sessions, n_runs, n_packed, n_pending, n_failed, n_flagged, url)` from `PublicCatalog.func_participants`, `func_runs` and `packs`. The blindness table of "Blindness" holds as route rules: in a local run that is not blind and has one rater the routes are open; in a blind run or one with several raters they are closed until the rater has nothing left to rate (`closed_to`, as `/review`: 403, a page with a link back for a browser); in a census run they are not mounted (404). The rating screen gets one link, to `/epi`, shown under the same rule as its review link and only when the run has a functional source (`SessionOut` gains `epi_url`, null otherwise); it carries no subject. The link per subject ("functional runs of this subject") comes with the subject pages in step 3. `serve` offers the same routes on a run with packs.

### The work packages

One owner per file. E2.0 and E2.Z are the integrator; E2.A, E2.B, E2.C and E2.D are written side by side against the names of E2.0.

| Package | Files it owns | Depends on | Exit check |
|---|---|---|---|
| E2.0 (done) | `privacy.py`, `records.py`, `schemas.py`, `config.py` (the step 2 names, complete with tests); `store.py` (the DDL of migration 2, the stubs, `EPI_DIR` and its two path functions); the stubs of `func/discover.py`, `func/inventory.py`, `func/batch.py` and of `Run.func_inventory`, `Run.pack`; `tests/synth_func.py` (`write_review_folder`, `func_counts`); `tests/test_func_step2_interfaces.py`; this section | step 1 | every earlier test passes; the step 2 names are pinned |
| E2.A catalog | `store.py` (migration 2 applied, `SCHEMA_VERSION = 2`, the read-only rule for schema 1, every method of the list), `tests/test_store.py` and a new `tests/test_store_func.py` | E2.0 | migration from a schema-1 catalog of the fixtures; read-only opening of schema 1 (empty functional answers, writers refuse); an older schema refusing a newer catalog; the public views against the private columns; the two signatures change with each listed fact and not otherwise; the skip of an unchanged session |
| E2.B discovery and inventory | `func/discover.py`, `func/inventory.py`, `func/api.py` (moving the rules and the planning out, behaviour unchanged), `sources/local.py` (the derivative and the raw dataset among the places that are read: `refuse_in_sources`, `refuse_run_in_sources`), new tests | E2.0; E2.A for its catalog tests | the layouts of "Which folders are found by rule" on synthetic trees; `func_source_of` from a stored root; the counts of `func_counts` on `write_review_folder`; placeholders and the fetch line; a second inventory opens no file; nothing written into the three folders; every step 1 test |
| E2.C pack phase | `func/batch.py`, `func/pack.py` (the keyword `keep` of `pack_session`), `batches.py` (the lease of `run_batches` generic in what it yields, typing only), new tests | E2.0; E2.A and E2.B for end-to-end tests (a stand-in `RunContext` before) | the skip rule; a changed anchor repacking the session; a removed or unselected run losing its pack and its row; a pack missing a file pending; two processes and `--jobs 1` and `2` giving one `content_digest`; a full disk stopping with 2; `stream.json` growing behind |
| E2.D the app in a run | `app/server.py` (the five routes, the blindness rules, `epi_url`), `app/schemas.py` (`ROUTES`, `SessionOut.epi_url`), `app/epi.py` (the media prefix of `Packs`, the index), `app/epi_schemas.py` (`EpiIndexOut`, `EpiParticipantOut`), `app/templates/epi_index.html` (new), `app/templates/rate.html` or the module that draws the review link, the step 1 page files only where the prefix needs it, new tests | E2.0; E2.A for the index | routes and their 403 and 404 cases in every row of the blindness table; the shapes against `pack.json` and `session.json`; the pack-only server of step 1 unchanged; no answer of a rating route holds a run name, a trace or a participant |
| E2.Z integrator | `run.py` (`func_inventory`, `pack`, the lease, the functional source in `context()`, `--relocate`), `review.py`, `cli.py` (`pack`, `--raw`, `--no-epi`, `--epi-runs`, the lines printed), `docs/`, `CLAUDE.md` status | all | `review` on `write_review_folder` with both interfaces, the EPI page reached from the rating screen in Chromium; the worked example end to end (network); the time to the first rating screen on the two real subjects unchanged (D33); review |

Shared files that more than one package needs and none edits but its owner: `records.py`, `privacy.py`, `config.py`, `schemas.py` (E2.0's; a change goes through the integrator and the pinning test); `tests/synth_func.py` (E2.0's; a package that needs more of it says so, and the integrator adds it).

### What this changes in the sections below

- "Where it is recorded": the table list is that of migration 2 above (`func_participants` added, `raw_identity` dropped, the file table keyed by kind and owner).
- "Identifiers": `participant_uid` added; `run_uid` is over `run_key`, without the echo.
- "Where packs live": `epi/<participant_uid>/<session_uid>/`, not `epi/<session_uid>/`, with `stream.json` per participant.
- "The pack as a stage": the unit of work is the session; runs have rows in `packs`; the decision is `func.batch.pack_plan`, not `current_pack`.
- "Routes and shapes": `/epi/{participant_uid}`, `/api/epi/{participant_uid}` and `/epi-media/{participant_uid}/{session_uid}/{path}` in place of the per-session routes, `EpiStreamOut` in place of `EpiSessionOut` (as step 1 built it), and `/epi` with `/api/epi` added; the subject routes stay in step 3.
- "Phases and order": the skip test of 2f is `func_sessions.inventory_hash`, not `func_runs.inventory_hash`; runs are taken in acquisition order (step 1: the anchor is not moved).
- "Configuration": `epi.raw` is added; the section is written only when an option was given.

## Scope

In scope: local folders that hold a raw BIDS dataset, an fMRIPrep derivative of it and the FreeSurfer subjects of that derivative; one app with a page per interface and per subject; the phases that list functional runs and pack them; their tables, tests and build order; the connector interface of O29 with one implementation, fMRIPrep.

Not in scope, and not to be designed in by stealth: a functional census, functional metrics or scores, any processing of the raw stream, a second connector, remote functional sources (only kept possible, see "The rules"), team mode for the EPI pages.

## What carries over from stage 1

Kept as stage 1 built and measured it, because it is sound and measured:

- The display grid is the grid of the preprocessed series as written; the contours are cut in its voxel space with the package's `MeshSlicer` and stored as geometry (`pack_contours.py`); their placement agreed with an independent recomputation to 0.011 voxel at most on two subjects (review of stage 1, 1d).
- The raw stream: cubic B-spline (`scipy.ndimage.affine_transform`, order 3, edge values repeated), 0 wherever the point lies outside the outermost raw voxel centres along any axis (`session.resample_voxels`; finding F4 of the review fixed the docstring, not the code). An independent recomputation found 0 differing voxels on five runs of four sessions (review 1a).
- The raw affine: the one that agrees, to 1e-4, with fMRIPrep's BOLD-space reference image of the run (`session.raw_affine_choice`): the header's `img.affine`, else the qform, else the reference image's own affine; without a usable reference image, `img.affine`, recorded as unchecked and warned about. Without this rule one session lay about 22 mm off the anatomy (README, "An error that was found and fixed").
- The anchor rule and the frame check with its flag at 10 mm (`session.pick_anchor`, `check_frame`, `frame_flag`), recorded as statements about the raw data.
- The 8-bit window and quantisation, the residual chunks against a reference volume, the stills over steady-state volumes, the contour file layout (`pack_run.py`, format `func-viewer-pack` version 1). Two packs of the worked example were byte-identical on one machine; 0 of 240 sampled canvas states and 0 of up to 83,772 screenshot voxels differed from the values computed from the NIfTI files (`check_player.py`, 2026-11-14).
- The page: plain modules, workers that fetch, check the SHA-256, inflate and add the reference, a decoded-memory limit of 512 MiB, play at up to 60 volumes per second painting every volume and skipping none (60.00 to 60.03 painted at 60 asked on three packs), the views raw, preprocessed, side by side and difference, the stills, the traces with markers of at least 500 ms.
- The record of the corrections fMRIPrep applied, read from its per-run summary report (`pack_inputs.report_facts`).

Changed, because a script is not a package:

- `pack_run.py` holds the whole preprocessed series of a run in memory (`get_unscaled()`, about 1.5 GB for the synthetic long run; report of S1.D). The package reads one volume at a time through the array proxy, as `check_player.py` was changed to do, and must show that the decoded values equal those of the prototype on the worked example.
- The pack names nothing and is traceable to nothing. The package traces every pack to its exact source files in the catalog (role, location, size, modification time, SHA-256), as a render is (D20, D21), and keeps that link out of everything served.
- The manifest is one file that the page reads whole, with traces, the frame check and the rules. The package splits it: `pack.json` per run with the inputs, never served (as `render.json` is not, D20), and a shape of `app/schemas.py` that the page receives.
- Runs are found by file names under `deriv/` and `raw/` folders that a fetch script wrote. The package finds them in a BIDS dataset and its derivative as they lie, through a connector.
- Ids are neutral strings (H01, R1). The package uses keyed identifiers, like `subject_uid`.
- Compressed bytes are the pack's identity in stage 1. zlib's output depends on the zlib build (the reason D19 writes `vertex_map.npz` uncompressed), so the package identifies a pack by the digest of its decoded values and lets compressed bytes differ between builds (see "Determinism").

Stays in `prototypes/func_viewer/` as the record of stage 1: the survey, the listing of 72 derivatives, the motion ranking and selection, the fetch scripts, `check_listing.py`, `session_measure.py`, `session_frames.py`, the pipeline of the feasibility study, and the README. `check_geometry.py` and `check_player.py` stay there and are the models for tests of the package (see "Testing"); `pack_synth.py` stays and the package gets a small synthetic run of its own.

## What a user does

### The one command

```
surfcensus review PATH [--raw DIR] [--no-epi] [--epi-runs GLOB]... (and every option of today)
```

`PATH` is what it is today (`architecture.md`, Sources): a FreeSurfer subjects directory, an fMRIPrep output, a BIDS derivatives folder, or the root of a BIDS dataset with FreeSurfer subjects below `derivatives/`. The surface interface is prepared exactly as today. In addition, when the folder also yields functional runs (below), `review` lists them, packs them behind the running app, and the app gains the subject pages and the EPI page.

- Only surfaces there (a subjects directory, a derivative without functional runs, a BIDS root without raw `func/` series): nothing changes. No functional table gets a row, no line about functional data is printed but one, at info level only.
- `--no-epi` turns the functional part off for this call and is stored (as `--video` is), so that a run whose user does not want packs never makes them.
- `--raw DIR` names the raw BIDS dataset when it does not lie where the rules look (an fMRIPrep output cloned on its own). It is stored in `config.yaml` of a local run as a second root, like the root of the source, and refused when it lies inside the run directory or the run directory inside it, by the same refusal as a source (`refuse_in_sources`).
- `--epi-runs GLOB` restricts the runs packed by their BIDS name (`*task-rest*`); stored, like `--subjects`.
- Nothing is fetched, and nothing is written into any of the three folders.

### Which folders are found by rule

A functional run needs three things: its raw series, its preprocessed series with the files of the connector, and the FreeSurfer subject that the derivative used. The rules start from what the surface source found and look in fixed places, matched by what they hold, as the BIDS root rule does (D35).

1. The derivative. The fMRIPrep output that holds the FreeSurfer subjects: for the fMRIPrep layout, the folder above `sourcedata/freesurfer/` or beside `freesurfer/` (`fmriprep/`); for a BIDS root, the folder of `derivatives/` that the surface search reached. A folder is an fMRIPrep derivative when its `dataset_description.json` names fMRIPrep in `GeneratedBy` (or `PipelineDescription`, older versions; to verify on a real file of 20.2.x) and it holds `sub-*/[ses-*/]func/`. A derivatives folder whose FreeSurfer subjects lie in `derivatives/freesurfer/` beside `derivatives/fmriprep/` (fMRIPrep run with `--fs-subjects-dir`, or the older layout) takes `derivatives/fmriprep/`.
2. The raw dataset. In order: `--raw DIR`; the BIDS dataset root whose `derivatives/` holds the derivative (the case of a dataset root given as `PATH`, or found as the outermost directory with `dataset_description.json` above the derivative, the rule of D35 for the run directory); `sourcedata/raw/` inside the derivative when it holds `sub-*/` with `func/` (the layout of an OpenNeuroDerivatives clone; to verify that such clones have it). A raw dataset is a directory with `sub-*/[ses-*/]func/*_bold.nii[.gz]`.
3. The FreeSurfer subject: the subject directory the surface source lists for the participant (below, "The unit").

When the derivative is found and the raw dataset is not, the runs are listed with the reason `no raw series` and the message names `--raw DIR`. When neither a derivative nor raw functional data exist, the folder has no functional part, silently.

### Derivatives that do not allow both streams

Listed, never packed, with a reason per run:

- `no preprocessed series`: fMRIPrep 23.2 and later at level `minimal` or `resampling`, or without an output space the contours can be brought into (see "The display geometry"). The prototype's survey found no preprocessed series in any of the 236 newer derivatives of OpenNeuroDerivatives (README, decision 3), so this is the common case for newer public data.
- `standard space only`: a preprocessed series in a template space (`space-MNI152NLin2009cAsym`) and in no space with a rigid path to the anatomy.
- `no transform to the anatomy`, `no brain mask`, `no raw series`, `multi-echo` (left out, as stage 1 left them out of the ranking; README question 13).

Whether such a run should get a raw-only pack is a decision for Tomas (below). Without one there is no difference to show, which is what D36 asks for; the default is no pack.

### Files without content

The rule of D35 holds for functional files: a git-annex link without content, or a pointer file of at most 32 kB, is not a file. A run whose packing files lack content is listed incomplete with `not fetched: ...` in its problem, and `review` prints one fetch command for the runs of the named subjects, built as the surface command is (`datalad -C <dataset> get ...`, else `git -C <repo> annex get ...`), after the surface command. It covers every file the pack reads (raw series, preprocessed series, mask, transform, reference image, confounds table, summary report), because a pack cannot be made from part of them. A raw dataset and a derivative are often two DataLad datasets; the command names each. The package fetches nothing and starts no git, git-annex or DataLad process. Pointer files are probed only among the small files the connector reads anyway (sidecars, reports, transforms, confounds tables up to 32 kB); a series is a link or a large file and is judged from the listing.

### What it prints

After the surface lines of today, one block, then one line per later step:

```
Functional: 14 runs in 3 sessions of 2 subjects in derivatives/fmriprep (fMRIPrep 21.0.2),
  raw series in . (BIDS); 12 can be packed (T1w space), 2 cannot: sub-02 task-rest run-2
  (no raw series).
EPI packs: 0 of 12 made; they are made while the app runs (about 9 min on 4 processes).
...
App: http://127.0.0.1:8765/ (11.2 s). Subjects: http://127.0.0.1:8765/subjects
All EPI packs are made: 12 of 12 can be viewed.
```

The estimate is the measured rate of "The pipeline" times the volumes and voxels of the listing, rounded, and says it is an estimate. Subjects are named by label, runs by their BIDS name without the subject (local run only; a census run is not offered the EPI interface).

## The unit and the navigation

### Participant, FreeSurfer subject, session, run

- A participant is the BIDS `sub-` label. The surface source already stores it (`subjects.participant`, the leading `sub-<label>` of the FreeSurfer directory name).
- A session is a participant and a `ses-` label; a dataset without `ses-` folders has one session per participant, as stage 1 decided (README, decision 6 and question 10). It is the unit of the one registration (D36) and the unit a person looks at (D37).
- A run is the set of files of one `func/` acquisition, keyed by `sub`, `ses`, `task`, `acq`, `ce`, `rec`, `dir`, `run` (run index compared as a number, so `run-01` and `run-1` match, as stage 1 found necessary in 3,102 of 9,462 runs) and `echo`.
- The FreeSurfer subject of a session. fMRIPrep writes one FreeSurfer subject per participant (`sub-X` in its subjects directory) by default; from version 25 `--subject-anatomical-reference sessionwise` processes each session on its own (fMRIPrep 25.2.3, `cli/parser.py`, lines 248 to 257). Which FreeSurfer name that gives is not known here (to verify). The rule: a session takes the FreeSurfer subject named `sub-X_ses-Y` when the source lists one, else `sub-X`; the anatomical transform to fsnative is taken from the same level of the derivative (`sub-X/ses-Y/anat/` before `sub-X/anat/`; stage 1 found it at subject level for 2,008 subjects and in a session folder for 169). A session without a FreeSurfer subject is packed without contours and says so.
- One FreeSurfer subject has one surface rating and any number of sessions; a longitudinal FreeSurfer directory (`.long.`) ties to nothing until it is decided which session it stands for.

### Pages

The default of D37: one app, a page per interface, and a page per subject; the two interfaces keep their own screens because each fills a window.

| Page | Route | What it shows |
|---|---|---|
| Rating screen | `/` | the surface interface, unchanged; in a local run that is not blind and has one rater, one link in the progress line, "functional runs of this subject", when the subject has sessions |
| Review screen | `/review` | unchanged, with a column for sessions and packed runs and a link per row to the subject page |
| Subjects | `/subjects` | every subject of the run: label, surface state (rendered, rated, unratable with reason, not rendered), sessions, runs listed, packed, failed, flagged |
| Subject | `/subject/{subject_uid}` | the surface state with links to the review screen for this subject and back to the rating screen; per session: its label, the anchor and the rule that chose it, whether a shared frame is plausible, and every functional run in acquisition order with volumes, TR, the corrections applied, its displacement from the anchor, its flag, its pack state or why it cannot be packed, and a link to the EPI page at that run |
| EPI | `/epi/{session_uid}` | the EPI interface: every packed run of the session under its one registration, selected by the run buttons (`?run=<run_uid>`), with a link to the subject page and to the previous and next session of the subject |

A participant whose functional runs have no FreeSurfer subject (no surfaces at all) still has a subject page, keyed by a session-only identifier, and the subjects page lists it without a surface state.

From a surface rating to the runs and back: the link on the rating screen opens the subject page in the same tab. The rating screen keeps the draft and the time of the open visit in session storage when the page is left (`rater_app.md`, Event logging), so going back continues the visit without the time away. The subject page's "back to rating" link returns to `/`, which shows the open assignment. Opening a chosen subject for rating out of the queue's order is not offered (a decision for Tomas, below): it would put an order effect into `ratings.seq` that the analysis plan does not foresee.

A combined page, if Tomas asks for one after seeing both: the rating screen's three panels and the EPI views each need most of a window at 1400 px (the EPI views at 2 screen pixels per mm of a 3 mm grid are about 300 px per panel), so a combined page would show one at a time with a switch, or the EPI views of one run in a fourth panel below the surface panels at reduced size, with one keyboard map in which the EPI keys move to unused letters (the rating keys 1 to 9 conflict with the player's views 1 to 4). It needs a new template and a merge of two key maps, and no change of the data model.

### Blindness

The surface rating screen stays blind (D6): it receives nothing new but the link, and the link carries the `subject_uid` that its media URLs carry already.

| Run | Subjects, subject and EPI pages |
|---|---|
| local, not blind, one rater | open, with subject labels and BIDS run names |
| local, blind, or more than one rater | unblinded routes: closed, as `/review` is, until the rater has nothing left to rate; a closed route answers 403, a browser asking for the page gets the reason as a page with a link back (`rater_app.md`) |
| census | not offered: the routes answer 404 and the phases do not run (O28, and no functional census) |

The EPI page shows framewise displacement and DVARS. They are motion estimates of the functional data, not scores of the surfaces, so they do not unblind a surface rating; they are a metric, which is why a judgement screen would hide them (below).

## The connector

Everything that is specific to fMRIPrep sits behind one interface, so that a second pipeline is a second implementation and not a rewrite (O29). Only the fMRIPrep connector is built.

A connector is a module under `func/connectors/` with a class that implements this protocol (`func/connectors/base.py`, standard library and numpy only):

- `name` and `detect(folder) -> ConnectorMatch | None`: whether a folder is a derivative of this pipeline, with the pipeline version it states and the generation of its layout (for fMRIPrep: `scanner-T1w`, below 23.2; `boldref-T1w`, 23.2 and later).
- `list_runs(folder, participant) -> list[DerivRun]`, from the listing alone: for each run its BIDS entities and the files of every role it has (location relative to the derivative root), so that the input signature opens no file (D21).
- `display(run) -> Display`: which preprocessed series is the display grid and in which space (`anat`, the anatomy's world; or `native`, the run's own reference frame), or why there is none (`no preprocessed series`, `standard space only`, `multi-echo`).
- `raw_to_anat(run, files) -> RigidPath`: the 4x4 point map from the anatomy's world (the space of the anatomical image the FreeSurfer transform starts from) to the scanner world of the run's reference image, as RAS millimetres, read from the stored transform, with its degrees of freedom when stated, and the reference image that arbitrates the raw affine (or None).
- `anat_to_fsnative(session, files) -> 4x4`: the point map that takes FreeSurfer's scanner RAS to the anatomy's world, for the contours.
- `mask(run) -> role`: the brain mask on the display grid.
- `motion(run, files) -> Motion | None`: framewise displacement and DVARS per volume and the non-steady-state volumes, if the pipeline estimated them; None is allowed.
- `corrections(run, files) -> Corrections`: head-motion correction, susceptibility distortion correction and its method, slice-timing correction, each `applied`, `not applied` or `unknown`, the registration as stated, and the source of each statement (a file role and the words read).
- `repetition_time(run, files)`.

A connector reads text and headers only; it never resamples. The rest of the package (session, geometry, pack, page) sees only these answers.

## The fMRIPrep connector

What each generation writes for a run, from the sources of fMRIPrep on PyPI (sdists of 20.2.3, 21.0.2, 23.0.2, 23.2.0 and 25.2.3, and the wheel of sMRIPrep 0.19.2, downloaded 2026-11-14) and from the key names in the listing of 72 OpenNeuroDerivatives derivatives made with 20.2.3 to 23.0.2 (stage 1, 2026-11-13). `<stem>` is the run's BIDS stem; `space-T1w` is the anatomy's world.

| Role | 20.2.x to 23.0.2 | 23.2 and later | Source |
|---|---|---|---|
| preprocessed series, anatomy | `<stem>_space-T1w_desc-preproc_bold.nii.gz`, with `T1w` among the output spaces | the same, level `full` and `T1w` (or `anat`) among the output spaces | 21.0.2 `workflows/bold/outputs.py` 268 to 271; 25.2.3 `workflows/bold/base.py` 372 to 427; listing: 9,373 of 9,462 runs |
| preprocessed series, native | `<stem>_desc-preproc_bold.nii.gz`, with `func` (or `run`, `bold`, `boldref`, `sbref`) among the output spaces | the same, level `full` | 21.0.2 outputs.py 215 to 236; 25.2.3 base.py 312 to 313; listing: 9,462 of 9,462 |
| brain mask | `<stem>_space-T1w_desc-brain_mask.nii.gz` (and `<stem>_desc-brain_mask.nii.gz` native) | the same, and `<stem>_desc-brain_mask.nii.gz` in boldref space at every level | 21.0.2 outputs.py 232, 279; 25.2.3 outputs.py 505 to 515 |
| reference image (raw-affine arbiter) | `<stem>_boldref.nii.gz` (BOLD space; written with the native outputs) | `<stem>_desc-hmc_boldref.nii.gz` and `<stem>_desc-coreg_boldref.nii.gz` at every level | 21.0.2 outputs.py 227; 25.2.3 `workflows/bold/fit.py` 401 to 406, 566 to 571 |
| transform raw to anatomy | `<stem>_from-scanner_to-T1w_mode-image_xfm.txt` (and the inverse `from-T1w_to-scanner`) | `<stem>_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt` | 21.0.2 outputs.py 193 to 205; 23.0.2 outputs.py 269 to 294; 25.2.3 fit.py 679 to 687 and outputs.py 557 to 570; listing: 9,373 |
| head-motion transforms | `<stem>_from-scanner_to-boldref_mode-image_xfm.txt` (in 1,275 listed runs, version not attributed) | `<stem>_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt` | 23.0.2 outputs.py 309 to 318; 25.2.3 outputs.py 611 to 623 |
| confounds | `<stem>_desc-confounds_timeseries.tsv` | the same, level `full` only | 21.0.2 outputs.py 188; 25.2.3 base.py 372 and 775 |
| summary report | `figures/<stem>_desc-summary_bold.html` | the same | 25.2.3 outputs.py 218; template in `interfaces/reports.py` of every version |
| validation report | `figures/<stem>_desc-validation_bold.html` | the same | 25.2.3 outputs.py 231 |
| distortion figure | `figures/<stem>_desc-sdc_bold.svg` when SDC ran | the same | 25.2.3 outputs.py 366; review of stage 1, F1 |
| anatomy to fsnative | `sub-X[/ses-Y]/anat/sub-X_from-T1w_to-fsnative_mode-image_xfm.txt` and the inverse | the same names, written by sMRIPrep with `from` the anatomical image type | sMRIPrep 0.19.2 `workflows/outputs.py` 666 to 702; listing: 2,178 of 2,190 subjects |

What that means for the generations:

- 20.2.x to 23.0.2: the layout of the prototype, measured on 19 runs of 8 sessions and the worked example. The arbiter `<stem>_boldref.nii.gz` exists only when a native output space was asked for; OpenNeuroDerivatives asked for it in every listed run, other users may not have. Without it the raw affine is unchecked and the page and `status` say so.
- 23.2 and later: the transform starts from `boldref`, the frame of fMRIPrep's reference image of the run, in place of `scanner`. Read as: the same scanner world, since the head-motion reference is computed from the raw volumes on their grid (25.2.3 fit.py 387 to 406). That the hmc and the coreg reference share grid and affine with each other, and with the raw series after fMRIPrep's header validation, is to verify on a real derivative before this generation is packed; until then the connector lists such runs and packs them only after that check (a work package below). At level `minimal` (no series, no confounds) and `resampling` (no anatomy-space series, no confounds; base.py 372 to 377) nothing can be packed.
- The ITK convention: a `mode-image` text transform maps a point of the reference (fixed, here T1w) to the moving space (here scanner or boldref), read by the review of stage 1 in the source of nitransforms 24.1.1 (`io/itk.py`, `ITKLinearTransform.to_ras`; `linear.py`, the `Affine` docstring); the stored values are LPS and are turned into RAS (`pipeline/geom.read_itk_ras`). The fsnative transform is used in the direction `from-T1w_to-fsnative` for points from fsnative to T1w, which on one FreeSurfer 7 subject made FreeSurfer's `lh.white` equal fMRIPrep's `hemi-L_white.surf.gii` to 0.0000 mm (README, Geometry).
- Rigidity. The registration has 6 degrees of freedom by default and 9 or 12 when asked (`--bold2t1w-dof`, `--bold2anat-dof`; 25.2.3 `cli/parser.py` 418 to 423). The summary report's `Registration:` line states it. The connector records it and the shear and scale of the stored matrix (`frame_displacement` has `shear_scale`); a transform that is not rigid to 1e-3 is used as stored (it is fMRIPrep's placement) and the page says "affine registration, not rigid".
- Corrections, from the summary report's lines `Slice timing correction:` (`Applied`, `Not applied`, `n/a` in 23.2.0 and 25.2.3, `Skipped (too few volumes)` in 25.2.3; `interfaces/reports.py` of 23.2.0, lines 235 to 241, and of 25.2.3, lines 238 to 244), `Susceptibility distortion correction:` (`None` or the method) and `Registration:`. `Skipped` reads as not applied and `n/a` as unknown, each with the report's words kept. Head-motion correction is always applied by fMRIPrep and its report does not say it per run (`pack_inputs.MOTION_SOURCE`). The distortion figure being listed is kept beside the report as a cross-check, as stage 1 does.
- Motion: `framewise_displacement`, `std_dvars` and `non_steady_state_outlier*` of the confounds table, as stage 1 reads them; in 9,373 tables these columns were present and framewise displacement recomputed from the motion parameters agreed to 1.5e-14 mm (README, step 2). Older column names (fMRIPrep before 20.2) are not in the listing and not supported until seen.

## The display geometry

The rule, generalised from the prototype. Let `G` be the affine of the display grid (voxel to world), `S_r` the point map of run `r` from the anatomy's world to its scanner world (`raw_to_anat`), `S_a` that of the session's anchor run, and `R_r` the raw affine of run `r` (the arbitrated one). The preprocessed stream is the stored series on `G`. The raw stream of run `r` samples the raw series at voxel `R_r^-1 . S_a . P_r . G . v` for every display voxel `v`, where `P_r` takes the display world to the anatomy's world. The contours take FreeSurfer's tkr RAS through `T1.mgz` (vox2ras times the inverse of vox2ras-tkr) to scanner RAS, through `anat_to_fsnative` to the anatomy's world, and through `(P_r . G)^-1` to display voxels. No other transform is ever applied to the raw data.

Three layouts, in the order in which they are built:

1. Anatomy space (`space-T1w` series, fMRIPrep below 23.2): `G` is the series' affine, `P_r` the identity. The prototype's case: one grid for the whole session, one set of contours. Built first.
2. Native space (`desc-preproc_bold` without a space, in the run's reference frame): `G` is the series' affine, in run `r`'s scanner world, and `P_r = S_r^-1`: the display voxel goes to the anatomy by the run's own registration, which is how the pipeline placed the preprocessed series, then to the raw data through the anchor's. Each run has its own grid, so its own contours; the raw stream of the anchor run is then the raw series on its own grid but for the header arbitration and the resampling, and of other runs, the raw series displaced by the head's movement between runs. The grid may be oblique and not in RAS order; the page shows voxel axes with the orientation letters computed from `G` and says how oblique it is (`oblique_deg`, as the prototype records). Built after the first, on the decision below.
3. Standard space only: contours would need the nonlinear warp `from-T1w_to-MNI..._mode-image_xfm.h5` applied to the surfaces, a reader of ITK composite HDF5 files that the package does not have, and a series that is no longer the anatomy's. Not packed; listed as `standard space only`.

When a derivative has both 1 and 2, the first is used (the prototype's choice, measured).

Grid checks before packing, each a reason of its own: every run of a session on the same grid in layout 1 (`grid_of`; else `runs on different grids`, and each grid is packed as its own group with its own contours, a choice that stage 1 did not meet and that is listed as not verified); a brain mask on the grid; volume counts of the raw series, the preprocessed series and the confounds table equal (stage 1's volume check).

### The raw affine, the anchor, the frame check

As statements about the raw data (D36), recorded per run and shown:

- The raw affine by the arbiter rule above, with which affine was taken (`header`, `qform`, `reference`, `header, unchecked`). An unchecked run is packed and carries a warning on the subject and EPI pages, since five of 19 raw series in stage 1 had a qform and sform 4 to 25 mm apart.
- The anchor: the first run acquired, by `AcquisitionTime` of the raw sidecars when every run of the session has a readable one (the sidecars write times without leading zeros), else the first in BIDS name order (task by name, run index as a number); the rule is recorded. In stage 1 the time rule applied in 30 of 103 surveyed sessions and in none of the 8 packed ones. Sidecars inherited from a higher level count as the run's (BIDS inheritance) when the listing has them; stage 1 read only sidecars beside the series (to verify that inheritance gives usable times anywhere).
- The frame check per run: mean and maximum displacement over its brain mask between its own stored transform and the anchor's, with rotation and translation; flagged above 10 mm somewhere in the mask; `anchor_is_outlier` when every other run of at least two is flagged and they agree with each other. Shown as in stage 1: a red box above the views and "flagged" on the run button, with the text that the run may not share the anchor's frame and is shown through the session's registration all the same.

### The corrections, as information

The EPI page states, near the views and for the run on screen, what the pipeline did to the preprocessed series: head-motion correction (always, by fMRIPrep's design), distortion correction and its method or none, slice-timing correction or none, the registration line, and the file each statement was read from. Where distortion correction was applied, the page adds that the raw stream shows the distortion as acquired against the contours and the preprocessed stream how well the correction brought the series onto them. Nothing is inferred beyond what the report says.

## The data model

### File roles of a functional run

A new enum `FuncRole` in `sources/base.py` beside `FileRole`: `raw_bold`, `raw_sidecar` (beside the series or inherited), `pre_bold`, `pre_sidecar`, `pre_mask`, `xfm_to_anat`, `reference` (the arbiter), `confounds`, `summary_report`, `validation_report`, `sdc_figure` (listed, not read) and, per session, `xfm_anat_fsnative`. FreeSurfer files stay roles of the FreeSurfer subject (`t1`, `lh_white`, ...), read through the surface source; a pack's inputs name them under those roles.

### Where it is recorded: the catalog, by migration 2

Two places were weighed.

New tables in `catalog.sqlite` (migration 2, `SCHEMA_VERSION = 2`): everything private already lives there and is reached only through `store.Catalog`, with public views for the rest (D20); the ledger `work` and the skip rule of D21 apply unchanged; one transaction can record a subject and its runs; `relocate` moves roots in one place. An older version of the package refuses a catalog of a newer schema (`ConfigError("newer schema")`, `store.py` 612 to 617), so it cannot misread a newer run; it cannot open it either, which is the price.

A database of its own (`epi.sqlite`): an older version never notices it, so the run stays usable for the surfaces by an older version. But a second private store needs its own public side, its own secret handling and its own refusals, two databases have no common transaction, a run secret must be shared between them, and an older version that ignores the functional tables can still remove absent subjects or move roots without the functional side following.

Recommended: migration 2 in the catalog. It is additive (only `CREATE TABLE` and `CREATE VIEW`), so one rule is added to keep D35's concern small: a read-only opener (`PublicCatalog.open`, used by `serve`, `export`, `finalize`) accepts a catalog of schema 1 and reads it as one without functional data, instead of asking for a phase command first. Writers migrate as today.

Tables of migration 2 (private columns marked; public views leave them out):

- `func_sources(func_id, dataset_id, connector, connector_version, generation, deriv_root*, deriv_identity*, raw_root*, raw_identity*, found_how, listed_at)`: the derivative and the raw dataset of a dataset row; for a local source the roots are paths, in a census they would be accessions and prefixes.
- `func_sessions(session_uid, func_id, participant, session_label, subject_uid, anchor_run_uid, anchor_rule, shared_frame_plausible, anchor_is_outlier, n_runs, present, problem, inventory_hash, inventoried_at)`. `subject_uid` is the FreeSurfer subject or null.
- `func_runs(run_uid, session_uid, entities*, acquisition_order, n_volumes, tr_s, echo_count, display_layout, grid_shape, voxel_mm, oblique_deg, dof, sdc, sdc_method, slice_timing, registration, corrections_source, raw_affine_choice, frame_mean_mm, frame_max_mm, flagged, packable, reason, missing_roles, problem, present, inventory_hash, inventoried_at)`. `entities` (task names, run numbers) is private in a census run, where task labels come close to identifying a dataset (README, rules); a local run shows it, as it shows subject labels.
- `func_files(run_uid, role, base[raw|deriv], location*, variant, size*, etag_or_mtime*, sha256*)`, and session-level roles with a `session_uid` key in the same table (one of the two keys set).
- `packs(run_uid, config_hash, input_signature*, pack_key, status, reason, detail*, path, n_volumes, bytes, content_digest, packed_at)`, the counterpart of `renders`.
- The session-level stage (registration, frame check, contours) is a unit `session:<session_uid>` of phase `epi_session` in the existing `work` ledger.

### Identifiers

Keyed like `subject_uid` (16 hex characters of an HMAC under the run's key or local salt; `privacy.py` owns them):

- `session_uid` over `(source, deriv_identity, connector, participant, session_label)`;
- `run_uid` over `(source, deriv_identity, connector, participant, session_label, task, acq, ce, rec, dir, run index as a number, echo)`;
- `pack_key`, the counterpart of `render_key`: an HMAC under the run secret over the pack's tile settings (window rule, quantisation, chunking, contour unit, voxel-map kind) and the SHA-256 of every input file, the anchor's transform and the FreeSurfer files included. A judgement, if one is ever stored, belongs to a `pack_key`.

The identity is that of the derivative, fixed when the functional source is first stored and kept across `--relocate`, as a local identity is (D21). The raw root does not enter the identifiers, so naming the raw data with `--raw` later keeps them.

### The pack as a stage

- Unit: one run (phase `pack`, table `packs`), plus the session unit `epi_session` that writes `session.json` (registration, frame check) and the contours of each grid. A new run in a session repacks nothing else unless it becomes the anchor.
- Config hash: the pack settings (`config.epi`, below) and `STAGE_SCHEMAS["pack"]`.
- Input signature, from the listing (D21): role, location, size, modification time and variant of every file the pack reads, and those of the anchor's transform and of the arbiter. A changed anchor changes the signature of every run of the session.
- Skip rule as for a render: `done`, same hashes, and the pack whole on disk (`pack.json` and every file it lists, by two directory listings).
- Outputs first, row last. A run that is no longer packable (files gone, now multi-echo, `--epi-runs` excludes it) loses its pack directory and its row, outputs first, as a metric stage does (D31). A pack directory is never edited in place: a new pack is written under a temporary name and renamed over the old one, under the `flock` of the session directory, as renders are.
- Failures recorded, not raised, with the reasons `no raw series`, `no preprocessed series`, `standard space only`, `no transform to the anatomy`, `no brain mask`, `volume counts differ`, `runs on different grids`, `multi-echo`, `source changed`, `pack failure` (type of the exception for anything else). What lies in the machine (a full disk, a read-only run directory) stops the phase with exit code 2 and is recorded against no run (D29).

### Where packs live

```
run/
  epi/<session_uid>/
    session.json                       registration, anchor, frame check, grids; served as a shape
    contours/<grid_key>/<plane>.bin    as stage 1 (pack_contours.py), one set per grid
    <run_uid>/
      pack.json                        inputs (sizes, SHA-256), settings, content digest; never served
      raw/ref.bin raw/mean.bin raw/sd.bin raw/c0000.bin ...
      pre/ref.bin pre/mean.bin pre/sd.bin pre/c0000.bin ...
      traces.json                      framewise displacement, DVARS, non-steady-state volumes
```

`grid_key` is 12 hex characters over the grid's shape and affine; the affine itself is not served (its numbers are those of a public file, as stage 1 decided), only shape, voxel size and obliqueness. The file formats of chunks, stills and contours are those of version 1; the container is version 2 (`epi-pack`), since the manifest is split and runs are named by `run_uid`.

### Size, and long runs

Measured: 0.54 bytes per voxel, volume and stream on the worked example (34,385,125 bytes for 2 runs of 146 volumes on 109,368 voxels, both streams, contours and stills included) and 0.50 on the synthetic long run (741,983,705 bytes, 1,200 volumes on 617,760 voxels). So about half a byte per voxel per volume per stream. A subject of 20 runs of 300 volumes on a 3 mm grid of 110,000 voxels takes about 0.66 GB; at 2 mm (390,000 voxels, the grid of H05) and 450 volumes, about 3.5 GB. These are multiplications, not measurements.

No run is cut, subsampled in time or reduced in resolution: the page skips no volume by design, and a pack that held fewer volumes would show something the data do not. What the package does instead: it prints the estimate before packing; `--epi-runs` restricts what is packed; `--no-epi` turns it off; and a `epi.max_gb` setting (default none) stops the phase before it starts when the estimate exceeds it, with the estimate in the message. The page holds at most its memory limit of decoded chunks (512 MiB default, as measured; renderer process 1.3 to 1.6 GB on the long run), so a long run is viewable with a wait of 141 ms median (317 ms at most) for a jump out of memory.

### Determinism

What stage 1 showed: two packs of the worked example byte-identical on one machine (three times); the session stream of a non-anchor run with the same SHA-256 in two calls and a new process (`check_session.py`). Not known: the same on another machine, with another zlib, with another numpy.

What the package does:

- The voxel map is computed in Python floats in a fixed order (`portable=True` of `session.voxel_map`), so that it does not depend on the linear-algebra library, as D19 did for `vox2pix`. On the worked example this changes single float32 values in 5 of 146 volumes of the anchor run by at most 9.5e-7 against the prototype's numpy map (2026-11-13); whether any 8-bit value changes is to verify. This is a decision for Tomas (README question 12); proposed default: portable.
- The identity of a pack is `content_digest`, the SHA-256 over the decoded 8-bit volumes, references, stills, traces and contour bytes in a fixed order, which does not depend on the zlib build. The compressed files carry their own SHA-256 in `pack.json` and in the shape the page receives, which the page checks, so a pack is consistent whatever zlib wrote it.
- `scipy.ndimage.affine_transform` and `np.percentile` are numpy and C code without BLAS; whether their results differ between machines is to verify, by the test of "Testing".

## The pipeline

### Phases and order

- Phase 2f, functional inventory, runs inside `inventory` when the run has a functional source (`config.func`): lists sessions and runs through the connector, reads the small files (sidecars, summary and validation reports, the transforms; headers of the series for volume counts, grids and TR, which reads 352 bytes of each gzip stream), decides the anchor and the frame check (which needs the transforms and the masks, a few MB per run), writes the tables. Skip test: `func_runs.inventory_hash` over the run's listing, as for subjects.
- Phase 6f, packing (`surfcensus pack`, and inside `review`): the session unit first, then the runs, in a stable order: subjects in the order of the rating queue (`store.queue_order`), then participants without surfaces by label, sessions by label, runs in acquisition order with the anchor first.
- `review` keeps D33: inventory, metrics and the first surface render come first and the app opens at the first surface render, as now; the renderer behind the app renders the remaining surfaces, then packs. If no subject of the run can be rendered and some run can be packed, the app opens when the first session's first run is packed, and `--serve-after N` counts packs. Interleaving surfaces and packs, or packing first, is a decision for Tomas; the default puts the rating queue first because a rater waits on it.
- The packer is a second `Run.pack(stop=...)` call in the same background thread, in batches of `--jobs` runs; Ctrl-C and SIGTERM stop it between two batches as they stop the renderer.

### Cost

Measured in stage 1 on one machine with 4 shared CPUs (times noisy): the worked example's runs of 146 volumes on 109,368 voxels took 13.7 to 13.9 s each to pack, of which the raw stream about 10 s, and the contours 3.6 to 5.8 s per session; the eight packed sessions took 15 to 287 s each; the long synthetic run took 354 s (raw 240 s) for 1,200 volumes on 617,760 voxels. That is about 0.09 s per volume on a 3 mm grid and 0.3 s per volume on a 2 mm grid, mostly the cubic resampling. A subject of 20 runs of 300 volumes at 3 mm is therefore about 9 minutes on one core and a few minutes with `--jobs 4`; at 2 mm and 450 volumes, about 45 minutes on one core. The per-volume read of the package (instead of the whole series) has not been timed.

Memory per worker: one raw volume, one preprocessed volume, the float64 sums and squares of both streams (32 bytes per voxel per stream) and the chunk being written: about 30 MB on a 2 mm grid. The worked example's whole-series reads took 0.42 GB in the feasibility study.

### Commands, exit codes, status, export

- `surfcensus inventory` lists functional runs when the run has a functional source; `--raw DIR` and `--no-epi` as for `review`.
- `surfcensus pack --run-dir DIR [--jobs J] [--epi-runs GLOB]... [--force] [--retry-failed]` packs what is pending and prints counts and one line per failure reason. When nothing is pending it reads no source.
- `review` ends with 1 only when neither a surface subject can be rendered nor a run packed; with 0 otherwise, with failures named.
- `status` adds three lines: functional sessions and runs listed (packable, not packable by reason); packs done, failed, pending; flagged runs and unchecked raw affines. `--failures` lists failed packs and runs that cannot be packed, with their reasons.
- `export` (analysis profile, local run): `func_sessions` and `func_runs`, one row per session and run, with the columns of their tables that are public, the pack state and reason, and the corrections; the manifest adds the pack config hash and the connector and its version. Traces are not exported: they are the confounds table's, which the user has. `export --release` and any census export leave the functional tables out (O28).

### Configuration

A section `epi` in `RunConfig`: `enabled` (true when a functional source exists, false with `--no-epi`), `runs` (the globs of `--epi-runs`), `portable_voxel_map`, `frame_limit_mm` (10), `fd_threshold_mm` (0.5, for the page's bands), `chunk_bytes` (4 MiB decoded), `max_gb` (none). The pack's config hash covers those that change a pack's bytes (`portable_voxel_map`, `chunk_bytes`); `frame_limit_mm` and `fd_threshold_mm` change only what is shown and are part of the session unit's hash and of the shape served.

## The app

### Routes and shapes

Added to `app/schemas.ROUTES` (unblinded routes marked as `/review` is):

| Method | Route | Answers with | Unblinded |
|---|---|---|---|
| GET | `/subjects` | page | yes |
| GET | `/subject/{subject_uid}` | page | yes |
| GET | `/epi/{session_uid}` | page | yes |
| GET | `/api/subjects` | `SubjectsOut` | yes |
| GET | `/api/subjects/{subject_uid}` | `SubjectPageOut` | yes |
| GET | `/api/epi/{session_uid}` | `EpiSessionOut` | yes |
| GET | `/epi-media/{session_uid}/{path}` | a pack file | yes |

Shapes (pydantic, in the style of `app/schemas.py`; field lists, not final names):

- `SubjectsOut`: `subjects: list[SubjectRowOut]`, each with `subject_uid`, `label`, `surface` (`rendered`, `rated`, `unratable`, `not rendered`, `none`), `unratable_reason`, `n_sessions`, `n_runs`, `n_packed`, `n_failed`, `n_flagged`.
- `SubjectPageOut`: the row, `review_url`, and `sessions: list[SessionRowOut]`, each with `session_uid`, `label`, `anchor_run_uid`, `anchor_rule`, `shared_frame_plausible`, `anchor_is_outlier`, `runs: list[RunRowOut]` (`run_uid`, `name` (BIDS name without subject), `n_volumes`, `tr_s`, `corrections: CorrectionsOut`, `frame: FrameOut`, `raw_affine`, `state` (`packed`, `pending`, `failed`, `cannot be packed`), `reason`).
- `EpiSessionOut`: `session_uid`, `subject_uid`, `label`, `prev_session`, `next_session`, `grids: list[GridOut]` (`grid_key`, `shape`, `voxel_mm`, `oblique_deg`, `edge_letters`, `contours: dict[plane, PackFileOut]`), `registration` (anchor, rule, limit, plausible, outlier, the statement of D36 as text), `fd_threshold_mm`, `runs: list[EpiRunOut]` with `run_uid`, `name`, `grid_key`, `anchor`, `frame`, `corrections`, `raw_affine`, `n_volumes`, `tr_s`, `window`, `sd_window`, `chunk_volumes`, `stills_volumes`, `pack_key`, `traces` (`fd`, `dvars`, `non_steady`), and `streams: {raw, pre}` with `ref`, `mean`, `sd`, `chunks: list[ChunkOut]` (`file`, `bytes`, `sha256`, `first`, `count`). No input size or digest, no affine, no path.

`render.json` and `pack.json` are never served; a test compares the fields of `EpiSessionOut` with those of `pack.json` and `session.json`, as one does for `RenderOut`, so a field added to a pack fails until someone decides whether a page gets it.

### Serving packs

- Names are matched by pattern and nothing else: `contours/<12 hex>/(sagittal|coronal|axial).bin`, `<run_uid>/(raw|pre)/(ref|mean|sd|c\d{4,}).bin`; a path that climbs, a link that leads out of `epi/<session_uid>/`, any other name and a pack that is not `done` for the present config are 404, as for `/media`.
- Whole files. A chunk is at most about 4 MiB decoded and smaller compressed, and the page fetches each whole, so range requests are not needed; Starlette 1.10.0, installed here, has range handling in `FileResponse` (its source mentions it; not tried).
- Caching as for tiles: immutable when asked with `?v=<pack_key>` of the present pack, `no-cache` otherwise.
- The CSP stays `default-src 'self'`: the workers are same-origin modules and need no `blob:`; `DecompressionStream` and `crypto.subtle` need no policy. `crypto.subtle` exists only in a secure context; 127.0.0.1 is one, a team server over plain HTTP would not be, and the worker then skips the digest check (it does so already when `crypto.subtle` is missing), which the page must then say.

### The page in the package

| Prototype file | Goes to | What changes |
|---|---|---|
| `player/logic.js` | `app/static/js/epi/logic.js` | unchanged, with its tests moved to `tests/` and run by `test_frontend.py` under Node |
| `player/store.js`, `worker.js` | `app/static/js/epi/store.js`, `worker.js` | chunk URLs from `/epi-media/...?v=`; memory limit from the session shape; otherwise unchanged |
| `player/views.js`, `traces.js` | `app/static/js/epi/` | colours from the app's tokens (`--crosshair`, contour colours of the render config: pial `#FFD400`, white `#FF3B30`), edge letters from `GridOut` |
| `player/main.js` | `app/static/js/epi/epi.js` | reads `/api/epi/{session_uid}` instead of `manifest.json`; run buttons by run name; links to the subject page and to the neighbouring sessions; the help sheet through the app's `help.js` pattern; the `window.__player` hooks kept for the browser tests |
| `player/index.html` | `app/templates/epi.html` (extends `base.html`) | the header of the app; the two uses named under the title; the corrections line and the warnings kept |
| `player/player.css` | `app/static/app.css`, a section under `.epi` | the app's tokens (`--bg`, `--surface`, `--line`, `--text`, `--warn`, `--warn-line`) in place of its own |
| `serve.py` | removed | the app's server |

Keys keep the player's map, which already follows the rater app where the action is the same: Space, the arrows (volumes; up and down move the active plane's slice), Shift for steps of 10, `[` `]` rate, `s` contours (held: off while held), `?` help, `Esc` closes. EPI-only keys: `1` to `4` views, `f` flip raw and preprocessed, `q` `w` `e` series, mean, SD, `p` active plane, `r` `R` next and previous run, `c` crosshair, `t` traces. `t` is the tour on the rating screen; the EPI page has no tour in this build, so `t` stays traces, and an EPI tour would take another key.

What the page says, always visible: under the title, "Raw: the raw BIDS series, placed once per session by the anchor run's registration, nothing else done to it. Preprocessed: the series as the pipeline wrote it. Their difference, and how each lies on the white and pial contours, is what this page shows: quality control of the raw data and of the preprocessing." Under the run buttons, the corrections of the run on screen and their source. Above the views, the warnings of a flagged run, an outlier anchor, an unchecked raw affine, a non-rigid registration. In the views' legend, the scale of the difference in gray levels and in units of signal, as now.

### Memory in the browser

The decoded-chunk limit stays 512 MiB by default and is settable per browser (stored in local storage, as the tour flag is), not by URL parameter. Measured: on the long synthetic run the page held 534 of 537 MB of decoded chunks at its peak and the renderer process 1.3 to 1.6 GB; on the largest real pack (423 MB packed) 1.2 to 1.4 GB. A machine with less memory than that is not measured.

## Judgements: a proposal, not to be built without his word

D37 leaves the EPI interface a viewer (O27). The smallest protocol that fits his words ("the quality of acquisition and processing"):

- Per run, two judgements: raw data (acquisition) and preprocessing (processing). Each: none, minor, major, as the surface scale, with a note. Per session, one more: whether the runs share a frame plausibly ("movement between runs"), none, minor, major.
- Optional marks in time: a click on the trace or `m` on the current volume marks the volume (and, on a view, the voxel) with a class (`motion`, `artefact`, `distortion`, `fit`, `other`).
- Stored in `app.sqlite` beside the surface ratings: an assignment kind `epi` keyed by `run_uid` and `pack_key` instead of `subject_uid` and `render_key`, submissions and ratings as now with criteria `R` (raw) and `P` (preprocessing), marks in a table of their own (`epi_marks`: volume, plane, slice, voxel, class), the protocol in `protocol.yaml` under a section of its own. A migration of `app.sqlite` (schema 2).
- Blindness for a judgement screen: no traces, no markers, no frame check numbers until submitted (they are metrics, and the review of the feasibility study said so); the corrections line stays, because the preprocessing judgement depends on it; the flag stays, because it says the run is shown off its own registration.
- To decide in a pilot: the wording of the two criteria and of minor and major; whether the session judgement is needed; whether marks in time are used; the time a judgement takes; the agreement of two viewers; whether stills first and the moving picture after (the failure criterion of stage 1, README, "What remains of stage 1") is the procedure.

## The rules

- Sources are read in place and never modified; the raw dataset and the derivative are sources, refused as places for a run directory or an export like the FreeSurfer source (`source_places` and `refuse_in_sources` gain the two roots and what their links lead to).
- Nothing is fetched; placeholders get the fetch command of D35.
- Outputs go to the run directory only (`epi/`, the catalog, the export).
- An EPI series is not defaced and a whole-head volume shows the face region (README, critiques). The local user looks at their own data; nothing functional is published by this code: no gallery, no static export, no release table, no demo page with functional data of a public dataset, until O28 is decided. A pack's traces equal the public confounds table to five decimals (review F6), so a pack identifies its run even without names: `pack.json`, `traces.json` and the chunks are not exported or released, and `export` writes no trace.
- In a census run the EPI phases do not run and the routes are 404.
- Labels and BIDS names are shown only in a local run, on unblinded pages.
- The remote case (`review openneuro:...`, milestone 5) only as far as not ruling it out: the connector works from a listing (locations, sizes, ETags) and reads small files through `read_bytes`; the pack reads its files through a lease (`lease(run_uid, roles)`), which a remote source fills by download into the cache and empties after the batch, as for renders; `func_sources` holds identities and roots that can be accessions and prefixes. Nothing in the geometry, the pack or the page needs a local path outside a lease.

## Testing

- Synthetic data (default tests). A generator `tests/synth_func.py` writes a BIDS dataset with raw `func/` series, an fMRIPrep-like derivative of either generation (series in `space-T1w` and native, masks, reference images, ITK text transforms with a known rotation and translation and a fixed centre, confounds tables with known framewise displacement, summary and validation reports in the template of `interfaces/reports.py`), and the FreeSurfer subject of `tests/synth.py`, on grids of a few thousand voxels and tens of volumes, in under a second. Cases: two sessions; runs moved between them by known amounts, one beyond 10 mm; an anchor that is the outlier; sidecars with and without `AcquisitionTime` (also without leading zeros); a qform and sform that differ, with the reference image taking either; a run without a reference image; volume counts that differ; runs on two grids; multi-echo; a native-only derivative; a standard-space-only derivative; level `minimal`; placeholders.
- Exactness against an independent recomputation, as `check_geometry.py` does: the raw stream of chosen volumes computed in the test with its own ITK reader and `map_coordinates`, and the preprocessed stream from nibabel, compared voxel by voxel with the decoded pack. On the synthetic data the expected placement is also known in closed form.
- The geometry: contours in display voxels against the closed-form transform of the synthetic surfaces (within 1/64 voxel), and with each transform inverted in turn, which must fail.
- Determinism: two packs of one run give one `content_digest` in two processes and with `--jobs 1` and `--jobs 2`; the portable voxel map gives the same digest under two BLAS libraries where the environment has both (skipped otherwise).
- The frame check and the anchor rule: the stage 1 cases of `session.py` with the numbers of the worked example as a network test, and the synthetic cases above.
- The catalog: migration 2 from a schema-1 catalog of the fixtures; a read-only opening of schema 1; an older schema refusing a newer catalog; the skip rule, a changed anchor repacking the session, a removed run losing its pack, a pack directory missing a file being pending.
- The app: routes, 403 and 404 cases, patterns of `/epi-media`, the shapes against `pack.json`, blindness of the rating routes (the rating screen's answers hold no run name, trace or frame number).
- Browser tests (marker `browser`, not in CI), from `check_player.py`: exactness of canvas pixels and screenshot voxel centres against the decoded values, painted rate at 12, 30 and 60 per second with none repeated or skipped, jumps to the exact volume, the memory limit, the warnings on a flagged run and an unchecked affine, the pages at 1400 and 400 px, the links between the rating screen, the subject page and the EPI page.
- Network fixture (marker `network`): the worked example, `ds000102` sub-01, whose functional files are 133 MB in 31 files (two raw runs 56 MB, two T1w-space series 59 MB, the T1w 17 MB; README, 2026-11-13) and whose FreeSurfer files the surface tests already cache. A cut-down fixture is sound for geometry, the frame check and exactness if every file of a run is cut the same way: the first 20 volumes of both series and the first 20 rows of the confounds table, with the transforms, masks, reference images and reports whole. The cut files are no longer the public files' bytes, so the fixture is made in the cache from the downloaded files by a script of the test and its digests are pinned. The window and the stills then differ from those of the full run, which the test computes independently; the claim "the anchor run's stream equals the feasibility study's bit for bit" can only be tested on the whole run, in a second, slow network test. The two runs have neither distortion nor slice-timing correction, so the connector's reading of a report that says applied is tested on the synthetic reports and on the 38 real report files that the review of stage 1 fetched, of which a few lines would be kept as committed excerpts with names replaced, as for the recon-all logs (D29).
- What cannot be tested here: what a person sees; any browser but Chromium; the 23.2 and later generation on a real derivative with `space-T1w` output (none was found among OpenNeuroDerivatives); a native-space-only derivative; multi-echo; sessionwise FreeSurfer subjects; another machine.

## Build order

Rules as in the contract's section 14: one owner per file, an integrator first and last, packages of a wave depend only on step 0 and earlier waves. Three steps, each useful on its own and each followed by an independent review.

### Step 1: one session, view only

The first useful thing: `surfcensus epi-session PATH --subject LABEL [--session LABEL] [--raw DIR] --out DIR [--serve]`, the counterpart of `render-subject`: no run directory, no catalog, one session of a local folder in the anatomy-space layout packed into `DIR` and, with `--serve`, viewed in the app's EPI page on 127.0.0.1.

| Package | Files it owns | Depends on | Exit check |
|---|---|---|---|
| E1.0 integrator | `sources/base.py` (`FuncRole`), `records.py` (records of runs, sessions, displays), `func/__init__.py`, `func/connectors/base.py` (the protocol), `config.py` (`EpiConfig`), `tests/synth_func.py` and its test | nothing | the fixture holds what it says (closed-form transforms, counts); names that two packages need exist |
| E1.A connector | `func/bids.py` (entities, run key, name order, sidecar times), `func/itk.py`, `func/connectors/fmriprep.py` (both generations listed, the report parser, the confounds reader) | E1.0 | listing of every synthetic case; report parsing on the excerpts; the ITK reader against nitransforms' convention on a transform with a centre |
| E1.B geometry and session | `func/session.py` (anchor, frame check, raw affine), `func/geometry.py` (display grid, voxel map, portable map, resampling), `func/contours.py` | E1.0 | independent recomputation on synthetic runs; the frame numbers of stage 1 on the worked example (network) |
| E1.C pack | `func/pack.py` (format 2, per-volume reads, content digest), `func/packread.py` (a Python decoder for tests) | E1.0 | two packs one digest; decoded values equal the prototype's pack of the worked example (network, with the numpy map) |
| E1.D page | `app/static/js/epi/*`, `app/templates/epi.html`, the `.epi` section of `app.css`, the routes `/epi`, `/api/epi`, `/epi-media` and their shapes in `app/server.py` and `app/schemas.py`, a pack-only mode of the server | E1.0, the shapes of E1.C | the browser checks of `check_player.py` on a synthetic pack |
| E1.Z integrator | `cli.py` (`epi-session`), `README.md` | all | the worked example from the command to the page; review |

Review after step 1: the geometry against an independent recomputation on the worked example and on one real session with several runs and a qform and sform that differ (from the files stage 1 fetched, kept outside the repository); determinism in two processes; the page against the stage 1 page side by side; nothing written into the three folders.

### Step 2: the run directory and `review`

| Package | Files it owns | Depends on | Exit check |
|---|---|---|---|
| E2.0 integrator | `privacy.py` (`session_uid`, `run_uid`, `pack_key`), `schemas.py`, `config.py` (stage schemas) | step 1 | identifiers pinned on a fixed salt |
| E2.A catalog | `store.py` (migration 2, public views, methods, the read-only rule for schema 1) | E2.0 | migration and refusal tests |
| E2.B discovery and inventory | `func/discover.py` (the rules of "Which folders"), `func/inventory.py` (phase 2f), additions to `sources/local.py` (`source_places` with the two roots, placeholders of functional files) | E2.0, E2.A | layouts on synthetic trees; placeholders; nothing written |
| E2.C pack phase | `func/batch.py` (pending, ledger, removal, leases, the session unit) | E2.0, E2.A | skip rule, anchor change, removal, machine errors stop with 2 |
| E2.Z integrator | `run.py` (`Run.pack`, the functional source), `review.py` (order behind the app, lines printed, `--serve-after` with packs), `cli.py` (`pack`, options), `docs/` | all | `review` on a synthetic BIDS root with both interfaces; the worked example end to end (network); review |

Review after step 2: the skip rule and the privacy boundary (no path, accession, task name or digest in anything served or logged in a census-like run made from Python), what an older version does with a migrated catalog, the time to the first rating screen unchanged on the two real subjects (D33).

### Step 3: navigation, status, export

| Package | Files it owns | Depends on | Exit check |
|---|---|---|---|
| E3.A pages | `app/templates/subjects.html`, `subject.html`, `app/static/js/subjects.js`, the routes and shapes of the subject pages, the link on the rating screen, the column on the review screen | step 2 | blindness table as tests; browser links both ways |
| E3.B status and export | additions to `cli.py` status lines (through the integrator), `export/tables.py` (`func_sessions`, `func_runs`, refusal in release) | step 2 | tables and manifest; no trace exported |
| E3.Z integrator | `docs/`, `CLAUDE.md` status | all | the whole path in Chromium on the worked example; review |

Review after step 3: the blindness of the rating screen with the link, what the pages say about the two uses and the corrections, the export against the catalog.

After step 3, on Tomas's decisions: the native-space layout (a step of its own, like step 1 for that layout), the 23.2 and later generation once its frames are verified on a real derivative, and the judgement protocol if he asks for it.

## Decisions for Tomas

Each with the default proposed here and what changes if he decides otherwise.

1. Layouts. Default: only preprocessed series in the anatomy's space (fMRIPrep `space-T1w`) are packed in the first build; native-space series follow as a later step; standard-space-only runs are listed and not shown. If native space is wanted first, step 1 gains per-run grids and oblique views.
2. Derivatives without a preprocessed series (fMRIPrep 23.2 and later at level `minimal` or `resampling`, the common case of newer public derivatives). Default: listed with the reason, no pack. A raw-only pack would show the raw stream on the anatomy through the stored transform, with no difference to show; it is a page variant and a reason less.
3. The voxel map. Default: portable (Python floats), so packs do not depend on the linear-algebra library, at the price of last bits against the prototype's packs. Otherwise numpy's map, as stage 1.
4. A flagged run. Default: shown through the session's registration with the warning, limit 10 mm, as stage 1. The alternatives (its own transform with a note; left out) change the frame and the pack's input signature.
5. The anchor rule. Default: unchanged (raw sidecar `AcquisitionTime` of every run, else BIDS name order). Adding `_scans.tsv` `acq_time` (308 of 2,470 qualifying sessions have one, README question 9) as a second source would change anchors, and so packs, where it applies.
6. FreeSurfer subject of a session. Default: `sub-X_ses-Y` when the source lists it, else `sub-X`. If sessionwise processing names subjects otherwise, the rule follows the name once seen.
7. Order behind the app. Default: surface renders first, then packs. Packing first, or alternating, would bring the EPI pages up sooner and the rating queue later.
8. Choosing a subject to rate from its subject page. Default: not offered; the queue order stays as recorded. If offered, the assignment records that it was chosen, and the analysis plan has to say what that does to order effects.
9. Traces on the EPI page. Default: visible, as stage 1 (a viewer, no judgement). If he wants the moving picture seen before the metric, they start hidden (`t` shows them).
10. Viewing events. Default: the EPI page logs nothing. Logging frames and plays, as the rating screen does, would let the paper say how long careful looking at EPI takes; it needs a place in `app.sqlite`.
11. Size. Default: no limit, an estimate printed, `--epi-runs` and `--no-epi`, an optional `epi.max_gb`. A default limit would refuse large subjects silently in a quick start.
12. Where functional data are recorded. Default: migration 2 of `catalog.sqlite`, with read-only openers accepting schema 1. A database of its own keeps old versions able to open the run, at the cost of a second private store.
13. Multi-echo runs. Default: listed as `multi-echo`, not packed (README question 13).
14. A combined page. Default: not built, until he has seen both.
15. Judgements (O27). Default: none. The proposal above is two judgements per run and one per session, none, minor, major, with notes and optional marks in time.

## What could not be verified

- fmriprep.org does not resolve from this machine; every statement about fMRIPrep's outputs is from its source code on PyPI (versions named) and from listings of 72 derivatives made with 20.2.3 to 23.0.2, not from its documentation.
- The 23.2 and later generation: that `boldref` in `from-boldref_to-T1w` is the same world as the raw series after header validation; that `desc-hmc_boldref` and `desc-coreg_boldref` share grid and affine; which of them is the right arbiter; what a real derivative of that generation with `space-T1w` output holds. No such derivative was opened; the one of stage 1 was at level `minimal`.
- Whether a `<stem>_boldref.nii.gz` exists in derivatives made without a native output space, and so how often the raw affine is unchecked outside OpenNeuroDerivatives.
- The `dataset_description.json` field that names fMRIPrep in 20.2.x (`GeneratedBy` or `PipelineDescription`).
- Whether OpenNeuroDerivatives clones hold the raw dataset at `sourcedata/raw/`.
- The FreeSurfer subject name under `--subject-anatomical-reference sessionwise`.
- Whether inherited sidecars (BIDS inheritance) carry `AcquisitionTime` usable for the anchor.
- Whether any 8-bit value changes between the numpy and the portable voxel map; whether `affine_transform` and `percentile` give the same results on another machine; whether zlib-ng or another zlib changes compressed bytes (expected, not tried).
- Range handling of Starlette's `FileResponse` (not needed; not tried).
- Runs of one session on different grids, multi-echo runs, `.nii` raw series, native-space-only derivatives, an anchor that is the outlier on a rendered session, the time rule of the anchor on a rendered session: none occurred in stage 1.
- The per-volume read of the package, in time and memory; packing on a slow disk; more than 20 runs per subject.
- Everything about what a person sees: no person has looked at a pack.
