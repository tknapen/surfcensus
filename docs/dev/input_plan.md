# Pointing the tool at files: wildcards and a setup page (plan)

A plan, written on 2026-11-16 at Tomas's request (open question O31 in `docs/decisions.md`). Nothing of it is built. His words:

> "It would be cool to be able to point the interface to files also in a different way: by selecting files in a GUI, or by specifying wildcards for each of the raw and preprocessed data types. Please make a plan to integrate those input strategies."

Today the tool is pointed at a place and finds the files by rule: a folder (`review PATH`, the layouts of `architecture.md`) or a dataset of the bucket (`review openneuro:dsXXXXXX`). Both rest on names: BIDS names for the raw series, fMRIPrep's names for the preprocessed series and its companions, FreeSurfer's names for the anatomy. The request adds two ways of saying which files are meant, for data that do not lie where the rules look or are not named as the rules expect.

## What the code requires today

Three facts of the present code decide the shape of the plan.

- The functional planner works from two roots and a connector. `func.inventory.list_participant(folders, connector, participant)` lists the derivative's runs through the connector (`Connector.list_runs(folder, participant, match)`) and the raw series through `func.bids.list_raw_runs(raw_root, participant)`, and `func.session.plan_session` pairs them by `run_key`, which is built from BIDS entities. Every file is a `FuncFileRef` with a location relative to one of the two roots. A file named outside BIDS, or lying outside the two roots, has no place in these records.
- A run is packed only when it has five roles (`sources.base.PACK_REQUIRED`): the raw series, the preprocessed series, the brain mask on its grid, the stored transform to the anatomy, and the confounds table. The fMRIPrep connector reads the transform as an ITK text file and the corrections from fMRIPrep's summary report. The connector protocol already allows a pipeline without motion estimates (`motion` may return None) and without a report (statements become `unknown`); the planner does not yet.
- The surface side is already file-based underneath: `render.api.render_subject(files, ...)` takes a mapping of roles to paths, and the `Source` protocol hands out one `FileRef` per role. Only the local source's listing assumes a FreeSurfer subject directory.

So the two input strategies are not two features. Both need one thing that does not exist: a description of the inputs at the level of files, which the planner and the surface inventory can read in place of a folder listing. The wildcards and the setup page are two ways of producing that description.

## The design in one paragraph

An input set is an explicit list, per subject, of the files of each role: the FreeSurfer subject, and per functional run its raw series, its preprocessed series and their companions. Four things can produce one: the folder rules (as today), a bucket dataset (as today), wildcards per data type (new), and choices made on a setup page in the browser (new). The input set is stored in the run directory, shown to the user as a table before anything is computed, and is what a second call of the same command reads. Downstream of it nothing changes: the same session plan, packs, pages, skip rules and identifiers.

## The input set

A pydantic model in a new module `inputs.py`, written to `inputs.yaml` in the run directory of a local run (which already stores the root path of its source in `config.yaml`, so no new kind of value leaves the catalog). Per subject:

- `label`, as the pages show it;
- `freesurfer`: a subject directory, or a mapping of `FileRole` to file for an anatomy that is not laid out as FreeSurfer writes it;
- `sessions`, each with a label and its runs. A run has a `name`, its `raw` series and optional `raw_sidecar`, its `pre` series and the optional companions `mask`, `xfm`, `reference`, `confounds`, `report`, `pre_sidecar`. A session has the optional `fsnative_xfm`;
- two statements that names can no longer make: the space of the preprocessed series (`anat` or `native`), and how the anatomy relates to FreeSurfer's frame (a transform file, or the statement that they are the same frame).

Paths are absolute. Globs and patterns are kept beside the resolved list, so that the table can say which pattern gave which file and a later call can resolve them again and notice new runs.

## Wildcards per data type

One pattern per data type, on `review` and on `epi-session`. `--raw DIR` and `--epi-runs GLOB` exist with other meanings (the raw dataset's root; a filter on run names), so the new options carry other names:

```
surfcensus review --raw-files '/data/raw/sub-01/func/*_bold.nii.gz' \
                  --pre-files '/data/prep/sub-01/func/*_space-T1w_desc-preproc_bold.nii.gz' \
                  --fs-subject /data/fs/sub-01
```

with `--mask-files`, `--xfm-files`, `--confounds-files`, `--reference-files` and `--report-files` for companions that do not lie beside the preprocessed series under fMRIPrep's names. When a companion pattern is not given, the companion is looked for beside the preprocessed file under the names the fMRIPrep connector knows, so that for fMRIPrep output two patterns are enough.

Pairing a raw series with its preprocessed series is the step that can go wrong silently, and a wrong pair is the worst error this tool can make: the difference view would then show two different runs as the effect of preprocessing. The proposed rule, in order:

1. Both names parse as BIDS: paired by `run_key`, as today.
2. The patterns carry named fields, `{sub}`, `{ses}` and `{run}`, which match one path component or part of one: `'/data/raw/{sub}/{run}.nii.gz'` and `'/data/prep/{sub}/{run}_preproc.nii.gz'`. Files with equal field values are one run. The fields also give the subject and session labels when the names are not BIDS.
3. Otherwise the command stops and shows both lists. Pairing by sorted order is offered only on an explicit option (`--pair order`), and the table of pairs is then printed for the user to read.

Two checks that exist already guard a pair after it is made: the volume counts of the two series must agree (`volume counts differ`), and the repetition times are compared. They catch many wrong pairs, not all: two runs of one task have the same length.

A dry run prints the resolved input set and what each run would be (packed, or the reason it cannot be) without writing anything: `surfcensus inputs ...` with the same options, and `--dry-run` on `review`. The same table is what the setup page shows.

## What the viewer does when a companion is missing

This is the scientific part of the plan, and each point is a decision for Tomas. The constraint is D36: the raw stream gets one rigid placement per session, read from what the pipeline stored, and nothing else.

- No transform to the anatomy. The tool never estimates a registration: an estimate of ours would be processing applied to the raw data, and its errors would be read as properties of the data. Proposed: such a run is not packed (`no transform to the anatomy`, as today) unless the user states that the raw and the preprocessed series share a world frame (`--same-frame`). The raw series is then placed by its own header alone, the page says so in the place where it now names the session's registration, and the frame check does not apply. Whether surfaces can be drawn then depends on whether a transform to the anatomy exists at all.
- Transform formats. Only ITK text transforms are read (`func/itk.py`). FSL `.mat`, FreeSurfer `.lta` and AFNI `.1D` each need a reader and a convention check against a real file of that pipeline; an FSL matrix also needs the geometry of both images. Proposed: ITK in the first build, others one at a time when he names a pipeline, each verified on real output before it is offered.
- No brain mask. The mask sets the intensity window and the voxels of the frame check. Proposed: without one, the window is taken over the non-zero voxels of the preprocessed series, as the surface renderer does without `brainmask.mgz`, and the page says that no mask was given.
- No confounds table. Proposed: the run is packed without traces, the stills are taken over all volumes (the non-steady-state volumes are unknown), and the page says that the pipeline's motion estimates were not given. The planner's requirement becomes a property of the connector.
- No report. Every statement about the corrections is `unknown`, which the page already shows. The user can state them (`--corrections 'motion=applied,sdc=none,slice-timing=unknown'`), and the page then names the user as the source, not a file.
- The space of the preprocessed series. With fMRIPrep names it is read from `space-T1w`. Otherwise it must be stated; a guess is not made. A cheap check that processes nothing is available: whether the world bounding box of the preprocessed grid overlaps that of the anatomy. It can only warn.
- The anatomy and FreeSurfer's frame. The contours need the transform between the pipeline's anatomical space and FreeSurfer's scanner RAS (fMRIPrep's `from-T1w_to-fsnative`). Without it: no contours, said on the page, unless the user states that the two frames are the same.

These relaxations change what a pack can lack, so the page has to say for every run which of them apply. The wording belongs to the same list as the corrections line.

## The generic connector

Open question O29 anticipated this: preprocessed series of other pipelines, found "as long as they conform to the T1 weighted suffix or something like that". The wildcard input is that connector. Two ways to build it were weighed.

- A link tree, as D50 did for the bucket: lay the named files out as a BIDS dataset and an fMRIPrep derivative of symbolic links in the run directory and run the unchanged planner on it. It needs no change to the planner. It does require inventing BIDS names for files that have none and recording the result as an fMRIPrep derivative with a pipeline version, which is false for any other pipeline, and it cannot express a run without a confounds table or a mask.
- A second connector (`func/connectors/explicit.py`) that lists runs from the input set, with the planner's listing step taking an input set in place of two roots. It touches `list_participant`, `FuncFileRef` (a location that need not lie under a root) and `PACK_REQUIRED`, and it records what is true: connector `explicit`, statements as the user or the files gave them.

Proposed: the second. The first stays the right mechanism where the files really are fMRIPrep's, which is why D50 used it.

## The setup page

A page of the local app that is served before a run exists. `surfcensus review` without a path (or `surfcensus open`) starts the server on 127.0.0.1 and opens the page, which offers three ways to say what to look at: a folder, a dataset of the bucket, or files. For files the user navigates the directories of the machine the server runs on, picks files or types a pattern per data type, and sees the resolved table update: the pairs, the companions found, what is missing, what each run would be. A button starts the run, and the page moves to the rating screen or the EPI page.

Two alternatives were weighed and are not proposed.

- The browser's own file dialog. A browser hands a page the content of chosen files, never their path on disk. The server would have to receive the series as uploads and write gigabytes of image volumes into the run directory, against the rule that sources are read in place and that the tool writes no image volume. The File System Access API changes nothing about paths and exists in Chromium only.
- A native dialog of the operating system (tkinter). It needs a display, so it does nothing over SSH, and the Python that `uvx` provides may lack it: `import tkinter` fails in the uv-managed Python 3.13 of the building environment (tried 2026-11-16). It could be an optional convenience later.

The directory browser in the page works wherever the app works, including through a forwarded port, where it browses the machine that holds the data, which is the machine one wants.

It is also a new surface, and the plan treats it as one. Today no route of the app reveals anything about the file system outside the run directory. The setup routes list directory names, sizes and kinds, never content, and are proposed with three limits beyond the existing `Host` and origin guard: a random token made at each launch, carried in the page's address and required by every setup route; routes that exist only in local mode and only until the run is started; and no setup routes at all in `serve`. A rater who is to stay blind is handed a prepared run, as now.

Routes, in the style of `app/schemas.ROUTES`: `GET /setup` (the page), `GET /api/setup/list?dir=` (one directory), `POST /api/setup/preview` (patterns or picks in, the input set and its plan out), `POST /api/setup/start`.

## Identity, skip rules and the hard rules

- Identity. A local source's identity is its resolved root when the run is created, stored and never recomputed (D21). An input set has no root, so its identity is a name drawn when the run is created and stored in `config.yaml`; `subject_uid`, `session_uid` and `run_uid` are then keyed by that name and the labels, as now. Patterns that are edited and still resolve to the same labels keep every identifier, render and pack.
- Skip rule. The input signature of a unit is built from role, location, size and modification time of its files (D21); with absolute locations it works unchanged, and deciding to skip opens no file.
- Sources are read in place. The directories that hold named files are places that are read: a run directory or an export in one of them is refused, by the rule the renderer already applies to a file handed over from elsewhere (`rendering.md`).
- Nothing is fetched, no image volume is written, and the census is untouched: an input set is a source of a local run only.
- Files without content (git-annex) are reported per file as today; the fetch line names the files.

## Build order

Rules as before: shared names first, one owner per file, an independent review after each step, and each step useful on its own.

| Step | What | Exit check |
|---|---|---|
| I0 | `inputs.py` (the model, `inputs.yaml`), the options' names, the decisions below recorded | names pinned in a test |
| I1 | The resolver: patterns with fields to an input set, pairing, companions beside the preprocessed file, the dry run and its table | synthetic trees with BIDS and non-BIDS names; every ambiguous case stops |
| I2 | The planner on an input set and the `explicit` connector; requirements per connector; the page's statements for what a pack lacks | on the worked example (`ds000102` sub-01), an input set made from two globs gives decoded streams equal to those of `review PATH` |
| I3 | `epi-session` and `review` with the options and with `--inputs inputs.yaml`; identity, signatures, the places that are read | the same command twice does nothing the second time; sources unchanged |
| I4 | The setup page: the token, the three routes, the page | browser tests at two window sizes; the routes refused without the token, in `serve`, and after the start |
| I5 | Surface files by role (`--t1-files`, `--white-files`, `--pial-files`) through a source that lists from the input set | a subject given by files renders the bytes of the same subject given as a directory |

I1 and I2 on `epi-session` are the first useful slice, as step 1 of the EPI interface was: one subject, no run directory, files named by two patterns. I2 is the largest step and the one that carries risk, because it changes the planner that three reviews have checked; its exit check is equality with the present path on real data.

## Decisions for Tomas

Each with the proposed default.

1. Which data prompt this: fMRIPrep output that lies somewhere unusual, or the output of another pipeline, or data without BIDS names. The first needs I1 and I3 only; the others need I2. Default: plan for all three, build in the order above.
2. Pairing. Default: BIDS entities, else named fields in the patterns, else stop; sorted order only on request.
3. A run without a stored transform. Default: not packed, unless the user states that raw and preprocessed share a frame; the tool never estimates a registration.
4. A run without a mask, a confounds table or a report. Default: packed, with what is missing said on the page.
5. Transform formats beyond ITK text. Default: none until a pipeline is named.
6. The setup page as a directory browser served by the local app, with a token. Default: yes; no uploads, no native dialog.
7. Whether a bare `surfcensus review` opens the setup page, or a command of its own does. Default: both `review` without a path and `surfcensus open`.
8. Surface files by role (I5). Default: last, since a FreeSurfer subject directory covers the usual case.

## What is not known

- Which transform conventions and file names the pipelines he has in mind use; nothing but fMRIPrep output was read for this plan.
- Whether pairing by named fields is convenient on real directory trees; it was not tried on any.
- How the `--same-frame` case behaves for sessions with several runs, where no anchor registration exists and movement between runs is then visible only if the preprocessed grids share a frame.
- Whether a directory listing route is acceptable on shared machines, where localhost is reachable by other users of the machine. The token is the mitigation; it was not reviewed.
- Windows: path patterns, drive letters and the absence of symbolic links were not considered beyond choosing a design that needs no links.
