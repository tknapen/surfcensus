# surfcensus: interface contract for milestones 2 to 5

> Working interface contract for milestones 2 to 5, produced by a design pass on 2026-11-12 (two independent proposals, a merge, three reviews). It is a build aid, not part of the specification that `CLAUDE.md` imports. When a milestone lands, the code and the docs in `docs/` are authoritative; this file is not kept in sync afterwards. Paths that mention a scratch directory refer to experiments that are not in the repository.

Written 2026-11-12. Read against commit `2b5c71c` plus the uncommitted milestone 1 hardening that was in the working tree when this was written (other agents were still changing it). Two things in that working tree differ from what the draft assumed, and the contract follows the working tree: `RENDER_SCHEMA_VERSION` is already 2 (the vertex map is selected by the clickable region of the tiles, carries `{h}_visible`, and reaches `vertex_map_margin_mm` = 5 mm beyond it), so milestone 3 bumps the schema to 3. When this was first written `docs/rendering.md` and D15 still described the vertex map as "the box plus 10 mm"; both follow the code since the milestone 1 review (row G14).

Nothing here is implemented, and the repository was not touched. This file replaces a draft (`contract_draft.md`, not in the repository), which three reviewers read; what was done with each of their 54 problems is in section 11.

Contents

1. How to read this, and what changed since the draft
2. Differences between the two proposals, resolved
3. What was checked, and what is assumed
4. Conventions shared by all milestones
5. Milestone 2: sources, config, catalog, inventory, `Run`
6. Milestone 3: template, registration, metrics, scoring, render schema 3
7. Milestone 4: rater app, `review`, `export`
8. Milestone 5: census (interfaces and data flow)
9. Tests
10. Doc changes that go with the code
11. Reviewer problems and what was done with each
12. Where the docs are silent or disagree, and the choice made
13. Decisions for Tomas, each with the default that work proceeds on
14. Order of implementation and work packages

---

## 1. How to read this, and what changed since the draft

Sections 5 to 8 are the contract, one per milestone. Sections 11 to 14 are the end matter that was asked for: the reviewers' problems, the doc gaps, the decisions that are Tomas's, and the build order with work packages. Nothing in section 13 blocks milestones 2 to 4: each item has a default, and each default is a config value or a run-time computation.

The changes since the draft that move an interface, in one list:

- Privacy. Nothing that leaves `catalog.sqlite` is an unkeyed hash of public bucket metadata any more. `render_key` is keyed with a per-run secret and built from content digests; input signatures, `dataset_id` and failure details stay private; parts, work units and Snakemake targets are named by pseudonym; the cache directory names are keyed. `render.json` is no longer served as a file, and the copy the API sends has no `inputs`.
- The privacy boundary is real for attribute access as well as for imports: phase code receives a `RunContext` (a `PublicCatalog`, a lease function, the config), not a `Run`. `PublicCatalog.open(run_dir)` is the only way `app/`, `export/`, `select/` and `reference/` reach the catalog. `--reveal` is implemented in `cli.py` and `privacy.py` and writes outside the run directory.
- Scoring. Subject-level metrics are standardised over the whole run, not within the dataset, by rank-based normal scores with minimum-rank ties (the robust-z cascade stays as an option). `scores.parquet` is keyed by subject and reference. Cluster mass in the composite is the excess over the threshold. A stratum's reference is built from every subject that matches its definition.
- Marks. One hemisphere per mark, chosen by the surface the mark's class is about; the white vertex is searched in that hemisphere. `in_box` became `box_dist_mm`. Vertex-level positives are defined in native space from the stored point and the vertex map.
- Cluster centroids reach native space through `vertex_map.npz`, in a cheap stage of their own, so no source file is needed after rendering and `clusters.parquet` is never rewritten.
- Render: schema 3; `RenderConfig.tile_hash()`; `render --all`, `render --check-sample`; the worker returns extras, which is where the intensity metric (now specified, section 8.9) plugs in.
- App: options are tri-state and sticky; every unblinded route is closed to a rater who still has subjects to rate; repeats are reported as `main`; a `ServerHandle`; `calibration.yaml`; `finalize` exists from milestone 4, so the two pilot raters can work before team mode exists.
- Build order: shared names (`records.py`, `schemas.py`, the pydantic shapes) move into the first step of each milestone, and the circular import between `config.py` and `sources/` is removed.

---
## 2. Differences between the two proposals, resolved

"L" is the local-researcher proposal, "C" the census proposal. Where both agreed, the agreement was checked against the docs, the code in the working tree and, for format facts, the files (section 3). One agreed point was changed: both wrote `three_plane(..., **kwargs)`, which the working tree has since replaced by explicit `video` and `force` keywords.

| # | Topic | L | C | Taken | Why |
|---|---|---|---|---|---|
| 1 | File access in `Source` | `resolve()` + `release()` | `lease()` context manager + `read_bytes()` | C | A context manager cannot forget to release; it is a no-op for local files, and it is the one hook that bounds census disk use. |
| 2 | Dataset identity in `subject_uid` | constant `local-1` | `identity`: accession, or the root path at first inventory, stored and never recomputed | C | Follows `architecture.md` (root path is hashed) and still lets data move; allows a second local source in a census run (pilot 7T data). |
| 3 | Privacy (D10) | by convention, plus a grep test | private columns, public SQL views, `PublicCatalog`, canary test | C | The hard rule is enforced by what downstream code can reach. The AST test is narrowed (it cannot ban `subject_label`, which local review and export show). |
| 4 | Secret | raw `bytes` | `HashKey` with a redacted `repr` | C | A key cannot end up in a log line by accident. |
| 5 | Skip rules | `metrics` table + `meta['scores_hash']` | generic `work` ledger keyed by phase and unit | C | One rule for every phase, including enrichment and selection. |
| 6 | Tables under parallel writers | `upsert_rows` on one file | per-dataset parts, merged | both | Parts per dataset (needed under Snakemake); a part is updated by replacing the rows of the subjects redone (L's upsert). Parts are named by pseudonym. |
| 7 | z-maps | computed on demand | stored per reference | L | A z-map is one subtraction and one division from stored arrays; storing it means rewriting every subject when the set changes. `fsavg/<uid>.npz` stays immutable. |
| 8 | Template files | sphere, white, cortex label, annot (8 files) | the same plus pial (10) | L | The pole vertex is 12950 on white and pial alike; pial adds two files and nothing else. |
| 9 | Template lookup | `--fsaverage DIR` first, then the run's own cached template, then D14 order | D14 order, pin, `--retemplate` | both | L's order (a run stays scorable offline after first use); C's refusal when an explicit directory differs from the pin unless `--retemplate`. After review the pin is looked at first, so a pinned run survives an explicit directory that has gone (6.3). |
| 10 | Medial / inferior split | smoothed normal direction | by parcel | C | Measured today, see T1. |
| 11 | Region cut | 45 mm on white y | 50 mm on pial y | L | Matches the render box; needs no pial file. T2. |
| 12 | Invocations in `recon-all.log` | anchored on each `subjid NAME` line | split on `New invocation of recon-all` | L | Checked today: the FreeSurfer 7.3.2 log has five invocations and spells its separator `#New# invocation of recon-all`, not as 6.0.1 does (corrected when milestone 2 was built). L's rule finds 4 and 5 on the two real logs. |
| 13 | T2 / FLAIR flags | whole-token, case-sensitive | `-noX` cancels `-X`; `pial_variant` also sets the flag | both | Case-sensitive whole tokens (recon-all's `switch` is); C's `-noX` rule, applied per invocation; variant as second evidence. |
| 14 | Flags stored | whitelist of 11 | every flag-shaped token | C, narrowed | Tokens matching `^-[A-Za-z][A-Za-z0-9_-]*$`: full provenance, and no path, number or subject name can match. Checked on the three real logs. |
| 15 | Containing-face rule | best of the incident faces, `det > 0` required, k = 4 | first containing face, `abs(det)` test, k = 8 | both | First containing face in a fixed order (simpler, deterministic), `det > 0` required (a flipped face never matches), k = 8 (3 and 2 fallbacks on sub-01 against 14 and 9). |
| 16 | Missing values in resampling | valid = thickness > 0; target is NaN below half the weight | valid = thickness > 0 and annotation cortex; any valid corner counts | both | C's mask (2,113 non-cortex `lh` vertices of sub-01 have non-zero thickness), L's half-weight rule (no value extrapolated from a 5% corner). 37,420 of 37,487 `lh` template cortex vertices get a value. |
| 17 | Ceiling fraction | at or above 4.9 mm | at or above 4.9 mm, or 0 inside cortex | C | That is the definition in `metrics.md`. |
| 18 | `gap_frac` (white-to-pial distance above 5 mm, computed at render time) | absent | in `vertex_summary`, filled in phase 6 | milestone 5 | A Tier C metric. It is computed with the intensity metric inside phase 6 and stored in `intensity.parquet` (8.9). |
| 19 | Cluster order | area | mass | L | `metrics.md` wording; T6. |
| 20 | Scores against a coarser stratum as well | no | yes, `scores_by_reference.parquet` | not now | T4; `clusters.parquet` and `scores.parquet` are keyed by `reference_id`, so it is more rows, not a new schema. |
| 21 | Vertices per reference | half of `min_reference_n` | `min_reference_n` | L | With exactly 30 subjects, one missing value would otherwise blank the vertex. Own config field, 20 after review (6.7). |
| 22 | Ratings schema | reserved criterion rows (`unratable`, `note`) | `submissions` table, revisions kept | C | Unratable, notes and revisions have a place; `ratings` keeps one row per criterion. |
| 23 | Re-rating after a re-render | `assignments.render_key` | absent | L, keyed | Local users edit surfaces and review again (T19). The key is an HMAC over the tile settings and the content of the tile inputs (5.1), so a template change, a re-encode or a touched file does not invalidate a rating, and the key cannot be recomputed from the bucket listing. |
| 24 | `sphere.reg` with the wrong vertex count at render | that hemisphere keeps -1 | `RenderError` | L | A bad registration must not stop anyone from rating the surfaces. |
| 25 | Template appears after rendering | render again | rewrite `vertex_map.npz` and `render.json` only | C | 0.2 s against 9 s per subject, and the likely local sequence is: read the message, pass `--fsaverage`, run again. |
| 26 | `review` before serving | serve after 20 renders | render all | C by default | Docs. `--serve-after N` and the `rendering` state of the API are specified so it can be switched on (T20). |
| 27 | Video in local `review` | off | on | L | `rater_app.md`: video is used only in the gallery. `--video` is stored, so it holds on later runs. |
| 28 | `require_marks` for local runs | override when loading | rewrite a line of `protocol.yaml` | L | No program edits a user-editable YAML file. |
| 29 | Export location | `export/` | `exports/<timestamp>/` | L, plus C's `manifest.json` | Notebooks need a stable path; the manifest says what the export was made from. A census run has a second profile, `release/` (7.8). |
| 30 | SQLite | default journal | WAL, `STRICT` tables | split | Local runs use the default journal, because run directories on network file systems are common and WAL does not work there. Census runs use WAL and require a local disk. No `STRICT` (needs SQLite 3.37, which a system Python may not have). |
| 31 | Incomplete subjects at render | no row | `failed` row, reason `missing surface` | C | They appear in review and export with a reason. |
| 32 | Subjects that vanish from the source | `present = 0`, ratings kept | not covered | L | |
| 33 | Test fixtures | new modules beside `tests/synth.py` | all in `tests/synth.py` | L | Module agents never edit a shared file. |
| 34 | Messages and helpers | `_io.say()`, `_io.py` | `log.py`, `atomic.py` | `_io.py` + `log.py` | `logging` so tests capture with `caplog` and the census can redact. |
| 35 | CLI of `inventory` | `inventory [PATH]` | `inventory --run-dir DIR [--source PATH]` | C | The form in `architecture.md`. `status` (C) is kept. |
| 36 | Exit codes | 0 / 1 | 0 / 1 / 2 | C | Snakemake has to tell "some units failed" from "stop". |

---

## 3. What was checked, and what is assumed

Checked by the two proposals on 2026-11-12 (nibabel 5.5.0; `ds000102` sub-01, FreeSurfer 6.0.1, fMRIPrep 21.0.2; `ds003020` sub-UTS01, FreeSurfer 7.3.2; the `fsaverage` next to sub-01). Items marked * were run again for this merge, on the files in the scratch directory.

File formats

- `read_morph_data(lh.thickness)`: `>f4`, 148,332 values, maximum exactly 5.0 (16 vertices), 28 at 4.9 or more, 5,919 zeros.
- `read_annot(lh.aparc.annot)`*: labels `>i4`, -1 for unlabelled vertices (8,027 in the subject, 13,887 in fsaverage), 36 names as bytes, starting `unknown`, `bankssts`; `corpuscallosum` at index 4. In fsaverage 2 vertices carry `unknown`.
- `read_label(lh.cortex.label)`*: int64 indices. The fsaverage file has 149,955, not sorted, 37,487 of them below 40,962; no fsaverage vertex with a parcel lies outside the label and none inside it lacks one. The subject's file has 140,351.
- `read_geometry(sphere.reg)`: float64 coordinates, `>i4` faces, radius 100. No flipped face in sub-01*.
- `stats/?h.aparc.stats`*: `# Measure` lines, a `# ColHeaders` line, 34 rows of 10 fields. FreeSurfer 6.0.1 has one malformed line (`# Measure Cortex, CortexVol Total cortical gray matter volume, 484608.441133, mm^3`); the rule of 6.1 reads it as `CortexVol`. `stats/aseg.stats`* has `lhSurfaceHoles`, `rhSurfaceHoles`, `SurfaceHoles`, `eTIV` in both versions (41 and 26 holes in sub-01).
- Stats headers* carry `hostname`, `user`, `SUBJECTS_DIR`, `cmdline`, `subjectname`; log argument lines carry paths with the accession.
- Build stamps*: `freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a`, `freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275`; the template's own is `freesurfer-x86_64-redhat-linux-gnu-dev4-20090216`. The regex of 5.7 gives 6.0.1, 7.3.2 and None.
- nibabel writes geometry, morph data and annotations; a `.label` file is plain text and has no writer.

recon-all logs under fMRIPrep*

- `recon-all.log` holds 4 (FreeSurfer 6.0.1) and 5 (7.3.2) invocations, each ending its header with a line `subjid NAME`. The last non-blank line before it holds the arguments: alone in FreeSurfer 6, after the script path in FreeSurfer 7. The 6.0.1 log separates invocations with `New invocation of recon-all`; the 7.3.2 log with `#New# invocation of recon-all` (corrected when milestone 2 was built: the first reading, that the 7.3.2 log has no separator, had searched for the 6.0.1 spelling).
- The first 64 kB of the 6.0.1 log (220,842 bytes) contain 2 of the 4 invocations.
- `thickness calculation complete, 39:225 truncations.` is in `recon-all-lh.log` and not in `recon-all.log`.
- The 7.3.2 log passes `-noT2pial -noFLAIRpial`.
- `orig.nofix lhholes =   41, rhholes = 26` is in `recon-all.log` in both versions.
- Flag-shaped tokens from all invocations of the three logs: 24 to 28 names, none with a `/` or an accession.

Template and registration*

- fsaverage6 faces cannot be had by truncating the fsaverage face list. They follow from coarsening: every vertex with index 40,962 or more has exactly two neighbours below it. One step gives 81,920 faces, 12 vertices of degree 5 and 40,950 of degree 6, all wound outward, in 0.2 s. A second step reproduces `fsaverage5/surf/lh.sphere.reg` faces exactly, in file order. The same function applied to `tests/synth.py: icosphere(k)` returns the faces of `icosphere(k - 1)` exactly for k = 1 to 5.
- The cortex mask by truncation is not FreeSurfer's own lower-resolution label (fsaverage5: 9,372 by truncation, 9,354 in its own file).
- Occipital pole: vertex 12950 on fsaverage `lh`, on white and pial alike.
- Region counts on fsaverage6 `lh` with the defaults of this contract: occipital parcels 3,534; search 4,856; medial 1,851; inferior 1,405; lateral 1,600; control 5,789.
- Mapping sub-01 to fsaverage6 with the rule of 6.4: 3 (`lh`) and 2 (`rh`) of 40,962 targets fall back to nearest neighbour at k = 8; 0.2 s per hemisphere. 180 fsaverage6 vertices are no native vertex's nearest.
- With the missing-value rule of 6.4, 37,420 of 37,487 `lh` template cortex vertices get a value; their mean is 2.43 mm.

Environment*: SQLite 3.53, fastapi 0.162, sqlmodel 0.0.49, starlette 1.10, httpx 0.28. `fastapi.testclient.TestClient` works and warns that httpx with it is deprecated. A SQLModel column named `class` works through `sa_column=Column("class", String)`. The Python `playwright` package and `snakemake` are not installed; `/usr/bin/google-chrome` exists.

Bucket (C only, not re-run): 308 `fmriprep/` prefixes; one listing per subject returns every key with size and ETag; no multipart ETag seen, so the ETag is the MD5; `If-Match` with a wrong ETag gives 412; GET returns `x-amz-version-id`; `git ls-remote` on the GitHub mirror gives the derivative commit; the raw bucket needs path-style addressing; FreeSurfer 7.3.2 derivatives hold `?h.pial.T1` and no `?h.pial` key (11 of 11); FreeSurfer 7.3.2 thickness is clamped at 5.0 (one subject).

Assumed, not checked

- Where `-T2pial` appears in an fMRIPrep log when a T2 exists. No such dataset was inspected.
- The layout of a plain `recon-all -all` log (expected: one invocation, two truncation lines, `lh` first).
- Which of the two truncation counts is white-to-pial.
- Names of template subjects other than `fsaverage*` in a FreeSurfer install (`cvs_avg35*`, `V1_average`, `?h.EC_average`), and the flag `-autorecon2-pial`, from memory.
- FreeSurfer 5.3 files of any kind.
- That fsaverage6 is the first 40,962 vertices of fsaverage (strongly supported by the coarsening result; no real `fsaverage6` directory was compared).
- The OpenNeuro GraphQL endpoint (unreachable from this environment), and the old fMRIPrep layout on S3 (not seen in 15 datasets).
- The `mri_surf2surf` flags in section 9.3.

The scripts behind the items marked * above were scratch experiments (`merge/`) and are not in the repository.

Checked again for this revision (2026-11-12; the script, `final_checks/chk.py`, is a scratch experiment and not in the repository)

- The working tree: `render/model.py` has `RENDER_SCHEMA_VERSION = 2` and `vertex_map_margin_mm = 5.0`; `render/vertexmap.py` selects vertices by `geometry.clickable_region` and writes `{h}_visible`; `RenderConfig.config_hash()` hashes the schema number and every field, including `fps`, `crf`, the crosshair settings and `vertex_map_margin_mm`; `sources/__init__.py` holds a module docstring and no code; `pyproject.toml` has `sqlmodel` only in the `app` extra, `scikit-learn` and `krippendorff` only in `analysis`, and `imageio` as a core dependency that no module imports (only `imageio_ffmpeg` is).
- The draft's `tile_signature` can be computed from the public bucket listing alone: the formula over the six tile roles of `ds000102` sub-01, from a saved listing of the subject's prefix (`sub01_listing.xml`, not in the repository) with no file downloaded, gives `f6501bd0d69d68d1`, the value the reviewer reported.
- The draft's scale cascade: 26 zeros and the values 10, 50, 200, 1000 have MAD 0 and IQR 0, so the SD (184.7) is used and the z values are 0.05, 0.27, 1.08, 5.41. Normal scores with minimum-rank ties give -2.13 for the zeros and 1.19, 1.38, 1.64, 2.13.
- D16 ranking on 12 subjects with no ceiling vertex: average-rank ties give 7 distinct composites of 12, minimum-rank ties 12 of 12.
- The expected maximum of 2 standard normal scores is 0.57, of 4 it is 1.03.
- `surf/{lh,rh}.white.avg.area.mgh` is in the listing of the `fsaverage` next to `ds000102` sub-01. The scratch copy of the FreeSurfer 7.3.2 subject has `recon-all.log` and no `recon-all-lh.log`.

Taken from the reviewers' experiments and not run again (scripts in `sci_review/` and in `../review_build/`): the per-parcel ratio of fsaverage6 white-mesh area to `white.avg.area` (0.59 to 1.01); the null rate of z > 3 at small n (0.0021 at n = 30, 0.0030 at n = 15 with the smoothed MAD); the sensitivity of the nearest fsaverage6 vertex to a 0.5 mm shift of a click (7.4% move more than 5 mm); the share of nearest white vertices in the other hemisphere near the midline (14.0% within 6 mm of x = 0); a globally flipped winding giving 40,962 fallbacks; `TestClient` sending `Host: testserver`; the import cycle between `config.py` and `sources/__init__.py` in a skeleton; `ImageFileError` text carrying the full path of an empty `.mgz`; `icosphere(7)` taking 0.25 s.

---

## 4. Conventions shared by all milestones

These follow what milestone 1 does.

- Data that is written to disk or sent over HTTP is a frozen pydantic model (`ConfigDict(frozen=True, extra="forbid")`, the `_Frozen` base of `render/model.py`). Data that holds arrays or crosses a process boundary is a frozen dataclass. Arrays are annotated with jaxtyping.
- `import surfcensus` imports the standard library only. `surfcensus.__getattr__` gains `Run`, `sources`, `metrics` beside `render`. `cli.py` imports inside each command.
- Package `__init__` files import nothing from the package at module level, except `render/__init__.py`, which imports `render.model`. A module that needs a type of a module that imports it, only for an annotation, imports it under `TYPE_CHECKING` (`records.py` for `PublicCatalog` and `RunConfig`; `privacy.py` for `store.Catalog`; `sources/base.py` for `SourceConfig`). A test imports every module of the package in a fresh interpreter, one process per module, so an import cycle fails whatever the collection order.
- `Hemi = Literal["lh", "rh"]` and `HEMIS: tuple[Hemi, Hemi] = ("lh", "rh")` live in `sources/base.py`; `render/vertexmap.HEMIS` becomes an alias. Always in that order.
- New module `surfcensus/_io.py` (standard library and numpy only):

  ```python
  def atomic_path(path: Path | str) -> Iterator[Path]            # moved from render/encode.py, which re-exports it
  def write_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None
      # the byte-stable writer of render/vertexmap.py: entries in mapping order, stored, dated 1980-01-01,
      # little-endian, atomic. vertexmap.write_vertex_map becomes a call to it.
  def read_npz(path: Path | str) -> dict[str, np.ndarray]        # allow_pickle=False
  def sha256_file(path: Path) -> str
  def canonical_json(obj: object) -> str                         # json.dumps(sort_keys=True, separators=(",", ":"))
  ```

- New module `surfcensus/log.py`: `get_logger() -> logging.Logger` (name `surfcensus`) and `configure_cli_logging(verbose: bool = False) -> None` (one stderr handler). Library code logs; `cli.py` prints results with `typer.echo`. In a census run a `RedactingFilter` is added (section 8.1).
- Errors a user can cause: one exception class per area, each with a short `reason` and a `detail`, like `RenderError`.

  | Class | Module | Reasons |
  |---|---|---|
  | `RenderError` | `render/api.py` (unchanged) | `missing surface`, `wrong orientation`, `render failure` |
  | `SourceError` | `sources/base.py` | `layout`, `not found`, `changed`, `unreadable`, `unreachable`, `invalid` (a root or subject glob that is not valid UTF-8) |
  | `ConfigError` | `config.py` | `missing`, `invalid`, `other root`, `newer schema`, `key` |
  | `TemplateNotFound`, `TemplateError` | `templates.py` | `TemplateNotFound.looked_in`; `TemplateError` reasons `not icosahedral`, `missing parcel`, `changed` |
  | `MetricError` | `metrics/__init__.py` (which holds nothing else) | `missing file`, `unreadable`, `vertex count mismatch`, `degenerate sphere`, `low coverage` |
  | `MarkError` | `app/marks.py` | carries `field` and `message` |

- CLI exit codes: 0 when no unit is in the failed state when the command ends; 1 when at least one is (one summary line per reason, details in the catalog); 2 for a usage or configuration error. A failed unit that is not retried counts as failed, so a second run of the same command reports `done = 0`, the same failed count and the same exit code. Error lines are `reason: detail` on one line.
- Nothing derived contains a timestamp, an absolute path or a host name, except `catalog.sqlite`, `app.sqlite`, `logs/`, `cache/` and the `config.yaml` of a local run (which names the source root). Tables are written sorted by their key.
- Text that comes from an exception passes through `privacy.scrub` (5.1) in the parent process before it is stored or printed.
- Workers are a `ProcessPoolExecutor` with the `spawn` start method on every platform; `jobs=1` runs in-process. Workers receive local paths and a `subject_uid`, never a catalog. The parent process is the only writer of the catalog in its process.
- Stage identity:

  ```python
  def stage_hash(name: str, schema: int, payload: BaseModel | Mapping[str, object], upstream: Sequence[str] = ()) -> str
      # first 12 hex of sha256(canonical_json({"stage": name, "schema": schema, "config": payload, "upstream": list(upstream)}))
  ```

  `RenderConfig.config_hash()` keeps its present form.

  | Stage | `name`, schema | Payload | Upstream | Output file checked by the skip rule |
  |---|---|---|---|---|
  | inventory | `inventory`, 1 | `InventoryConfig` | | none (`subjects.inventory_hash`) |
  | enrich (phase 3) | `enrich`, 1 | `{"columns_sha256": ...}` of `enrich/columns.yaml` | | none |
  | regional (4a) | `regional`, 1 | `{}` | | the dataset's `regional` part |
  | fsaverage (4b) | `fsaverage`, 1 | `mesh`, `regions`, `knn`, `min_valid_weight`, `thickness_max_mm`, `ceiling_margin_mm`, `min_coverage`, `max_fallback_frac` | `template_id` or `"none"` | `tables/fsavg/<uid>.npz` when the row's `output` is set, else the dataset's `vertex_summary` part |
  | reference (4c) | `reference`, 1 | `min_reference_n`, `min_vertex_n`, `mad_smooth_iterations`, `mad_floor_mm`, `cluster_z`, `cluster_min_vertices`, `clusters_kept` | fsaverage hash | `tables/reference/<id>/index.json` and `tables/clusters.parquet` |
  | score | `score`, 1 | `ScoringConfig` | reference hash or `"none"` | `tables/scores.parquet` |
  | select (phase 5) | `select`, 1 | `SelectConfig`, the two sample sizes, the seed | score hash | none (the `selection` rows) |
  | render | `RenderConfig.config_hash()` | | | `media/<uid>/render.json` |
  | native | `native`, 1 | `{}` | reference hash | `tables/clusters_native.parquet` |
  | intensity (4d, milestone 5) | `intensity`, 1 | `IntensityConfig` | `RenderConfig.config_hash()`, `template_id` or `"none"` | the dataset's `intensity` part |

- Input signature: first 16 hex of the SHA-256 of `"\n".join(f"{role}\t{location}\t{size}\t{etag_or_mtime}")` over the requested roles that the subject has, in `FileRole` definition order, followed by one line per `extra` string. A role whose `FileRef` has a `variant` adds `\t{variant}` to its line: a local `lh.pial` that is a link keeps its location when it is pointed at another variant, the pial variants of one subject have one size, and a copy can keep the modification time, so without it a retargeted link was skipped and `pial_variant` and `t2pial` went stale (review finding S1). It comes from the inventory, so deciding to skip costs no download and no hashing of source files. For a remote subject it is a function of the public bucket listing, so it is private: it is stored in `catalog.sqlite` (`work.input_signature`, `renders.input_signature`, `subjects.inventory_hash`) and nowhere else, no `PublicCatalog` method returns one in a frame, and no API response, table, export or log line carries one. What identifies a render outside the catalog is the keyed `render_key` (5.1).
- The skip rule, the same in every phase: a unit is skipped when its row is `done`, `config_hash` and `input_signature` are equal, and the output file named in the table above exists (for a render: `render.json` and every file it lists, `render.api.render_on_disk`). A `failed` row with equal hashes is not retried unless `--retry-failed`; a changed config or input retries it. `--force` ignores the row.
- A commit writes files first and the ledger rows last, in one transaction. A crash in between leaves outputs without a row, which the next run redoes; it never leaves a row without its output.
- Run-level stages (`reference`, `score`, `native`, `select`) always use every present subject and ignore `--dataset`. When `--dataset` is given without `--stage`, only the per-subject stages run.
- Per-dataset output is always ordered by pseudonym, and per-subject output by `subject_uid`. Catalog row order (`dataset_id`, discovery order) follows the bucket listing and is never the order of anything written outside the catalog.

---

## 5. Milestone 2: sources, config, catalog, inventory, `Run`

Result: `surfcensus inventory`, `surfcensus render` and `surfcensus status` work on a local directory and are resumable; `sc.Run(...)` and `sc.sources.local(...)` exist. No metrics yet.

New files: `_io.py`, `log.py`, `records.py`, `schemas.py`, `privacy.py` (the part below), `config.py`, `store.py`, `run.py`, `batches.py`, `render/batch.py`, `sources/local.py`, `inventory/__init__.py`, `inventory/subjects.py`, `inventory/provenance.py`. Changed: `sources/base.py`, `sources/patterns.py`, `sources/__init__.py`, `render/model.py` (`tile_hash`), `render/encode.py` and `render/vertexmap.py` (helpers moved to `_io.py`), `render/api.py` (one refusal, 5.10), `cli.py`, `__init__.py`.

`records.py` holds the plain records that more than one module needs, so that no module has to import another one's module to name a type: `Provenance` (5.7), `PhaseReport`, `WorkRow`, `RenderRow`, `RenderResult` and `RunContext` (5.9). It imports the standard library and `sources/base.py` only; `PublicCatalog`, `RunConfig`, `RenderInfo` and `pandas.DataFrame` appear in it under `TYPE_CHECKING`. `schemas.py` holds the Arrow schemas and keys of every table of section 6.10 and 8.9 from the start; they are fully specified here, and `store.merge_parts` needs the keys.

### 5.1 `privacy.py` (milestone 2 part)

Nothing else in the package computes a pseudonym, a `subject_uid`, a `render_key` or a cache name.

```python
@dataclass(frozen=True, repr=False)
class HashKey:
    secret: bytes                                   # 32 bytes; repr shows kind and fingerprint only
    kind: Literal["local-salt", "dataset-key"]
    @property
    def fingerprint(self) -> str                    # sha256(secret).hexdigest()[:12]

def new_local_salt() -> HashKey                     # secrets.token_bytes(32)
def local_salt_from_hex(text: str) -> HashKey       # as stored in config.yaml (64 hex)
def dataset_pseudonym(key: HashKey, source: str, identity: str, *, length: int = 8) -> str
    # "D-" + HMAC-SHA256(key.secret, f"{source}:{identity}".encode()).hexdigest()[:length]        (D10)
def subject_uid(key: HashKey, source: str, identity: str, pipeline: str, subject_label: str) -> str
    # HMAC-SHA256(key.secret, ("subject\x1f" + "\x1f".join((source, identity, pipeline, subject_label))).encode())
    #   .hexdigest()[:16]
def render_key(run_secret: bytes, tile_hash: str, digests: Mapping[str, str]) -> str
    # first 16 hex of HMAC-SHA256(run_secret, ("render\x1f" + canonical_json({"tiles": tile_hash, "inputs": digests})).encode())
    # digests: role -> SHA-256 of the file, for the TIER_C roles the render read (from render.json `inputs`)
def cache_name(run_secret: bytes, bucket: str, key: str, etag: str) -> str
    # first 32 hex of HMAC-SHA256(run_secret, ("cache\x1f" + "\x1f".join((bucket, key, etag))).encode())
def scrub(text: str, *, pseudonyms: Mapping[str, str] | None = None, roots: Collection[str] = ()) -> str
    # 1. every string of `roots` (dataset roots and identities) is replaced by "<root>";
    # 2. every absolute path (POSIX or Windows, also inside quotes) is replaced by its last component;
    # 3. every token matching ds\d{6} is replaced by pseudonyms[token], or by "D-?" when it has none;
    # 4. every token matching sub-[A-Za-z0-9]+ is replaced by "sub-?" when `pseudonyms` is not None (a census run).
    # Collapses white space to single spaces. Pure; never raises.
```

- `identity` is the accession for a remote dataset. For a local one it is the resolved absolute root path at the time the dataset was first inventoried; it is stored (`datasets.identity`, and `SourceConfig.identity` in a local run) and never recomputed, so `--relocate` keeps every `subject_uid` and with it every rating.
- 16 hex characters. A collision between two different subjects stops the inventory with an error.
- A local run hashes pseudonyms and uids with its per-run salt. A census run hashes them with the dataset key, including a local source added to it. Only `discover` and `inventory` need the key; every other command opens a census run without it.
- The run secret is 32 random bytes made by `Catalog.create` and kept in `catalog.sqlite` (`meta.run_secret`), in both modes. It is not the dataset key and protects nothing that the catalog does not already hold. It exists so that `render_key` and cache names, which are derived from files whose size and ETag are public, cannot be recomputed from the bucket listing by someone who holds an export, a log or a media URL. Because `render_key` is built from content digests, a `touch`, a copy without preserved mtimes or `--relocate` leaves it unchanged.

What is private, and how that is enforced:

1. Private values exist only in `catalog.sqlite`: `meta.run_secret`; `datasets.dataset_id` (assigned in discovery order, which is the order of the bucket listing), `.identity`, `.accession`, `.root`, `.fs_root`, `.template_location`, `.derivative_ref`, `.snapshot_tag`; the whole `dataset_files` table; `subjects.fs_dir`, `.inventory_hash`; `files.location`, `.size`, `.etag_or_mtime`, `.last_modified`, `.version_id`, `.sha256` (the sizes and digests of the files a render read are also in the `render.json` on disk, which is never served or exported as a file: 7.5, T27, decision D20); every `input_signature`; every failure `detail`; and, in a census run, `subjects.subject_label` and `.participant` (`sub-UTS01` names its dataset).
2. `files.location` is relative to the subject directory, so the files table holds no accession.
3. Everything downstream of inventory (`metrics/`, `render/`, `app/`, `export/`, `select/`, `reference/`, `calibration.py`) reaches the catalog only as a `PublicCatalog` (5.5), opened with `PublicCatalog.open(run_dir)` or handed over inside a `RunContext` (5.9). Phase functions receive a `RunContext`, never a `Run`; `Run` has no public attribute for the private catalog (it is `Run._catalog`). A test walks those packages with `ast` and fails on: an import of `surfcensus.store.Catalog`, of `surfcensus.run`, or of `privacy.reveal_table` / `privacy.write_reveal`; an attribute access named `_catalog`; the string literals `accession` and `dataset_id`; and, in `app/` and `export/`, the string literal `input_signature`. `review.py` is exempt from the `surfcensus.run` rule (it is a user of the public API) and from nothing else.
4. Parsers drop what names a dataset: stats headers other than `# Measure` and `# ColHeaders` are not read; recon-all arguments are reduced to flag names (5.7); `render.json` holds role-derived file names only.
5. Failure text is scrubbed (section 4), and in a census run `PublicCatalog` returns no `detail` at all; `surfcensus status --failures`, which runs in `cli.py` with the private catalog, is where an operator reads details.
6. The canary test (section 9) runs every phase in census mode on a fake bucket dataset `ds999999` with subject `sub-CANARY7`, plus a local source at a path containing `CANARYROOT`, with one subject whose `T1.mgz` is empty (a failure path) and one without `lh.pial`. It then searches every byte written outside `catalog.sqlite` and `cache/`, the captured log, the CLI output and every API response for `ds999999`, `CANARY7`, `CANARYROOT`, every input signature and every ETag in the catalog, and checks that the three test datasets cannot be put in discovery order from the order of lines, rows or file names of anything outside the catalog.

### 5.2 `sources/base.py` (additions)

```python
Hemi = Literal["lh", "rh"]
HEMIS: tuple[Hemi, Hemi] = ("lh", "rh")

class FileRole(StrEnum):                  # the 18 existing members, then
    RECON_LOG_LH = "recon_log_lh"         # scripts/recon-all-lh.log
    RECON_LOG_RH = "recon_log_rh"         # scripts/recon-all-rh.log

TIER_A: tuple[FileRole, ...]   # lh_stats, rh_stats, aseg_stats, build_stamp, recon_log, recon_log_lh, recon_log_rh
TIER_B: tuple[FileRole, ...]   # lh/rh_thickness, lh/rh_sphere_reg, lh/rh_annot
TIER_C: tuple[FileRole, ...]   # RENDER_ROLES + (BRAINMASK,)
TIER_A_REQUIRED: tuple[FileRole, ...]   # lh_stats, rh_stats
TIER_B_REQUIRED: tuple[FileRole, ...]   # lh/rh_thickness, lh/rh_sphere_reg (the annotation is used when present)

SourceKind = Literal["local", "openneuro-derivatives", "openneuro-embedded"]
Layout = Literal["subjects_dir", "fmriprep", "fmriprep_legacy", "bids_derivatives"]

@dataclass(frozen=True)
class FileRef:
    role: FileRole
    location: str                  # POSIX path relative to the subject directory, e.g. "surf/lh.pial.T1"
    size: int
    etag_or_mtime: str             # remote: ETag without quotes; local: st_mtime_ns in decimal
    variant: str | None = None     # file name read when a role has several candidates or is a link in its
                                   # own directory, e.g. "lh.pial.T1"; always one of the role's candidate names
    last_modified: str | None = None   # remote: LastModified of the listing, ISO 8601

@dataclass(frozen=True)
class DatasetRecord:
    source: SourceKind
    identity: str                  # hashed into pseudonym and subject_uid (5.1)
    accession: str | None          # None for local
    root: str                      # local: absolute path; remote: "s3://bucket/prefix/"
    fs_root: str                   # subjects directory relative to root: "sourcedata/freesurfer", "freesurfer", "."
    layout: Layout
    pipeline: str                  # "fmriprep" | "freesurfer"
    pipeline_version: str | None = None
    template_location: str | None = None   # an fsaverage directory next to the subjects, relative to root
    derivative_ref: str | None = None
    snapshot_tag: str | None = None

@dataclass(frozen=True)
class SubjectRecord:
    dataset: DatasetRecord
    subject_label: str             # the directory name
    fs_dir: str                    # POSIX, relative to dataset.root
    files: Mapping[FileRole, FileRef]
    problem: str | None = None     # e.g. "unreadable: Permission denied"; files is then empty

class SourceError(Exception):
    def __init__(self, reason: str, detail: str) -> None     # str(exc) == f"{reason}: {detail}"; .reason

class Source(Protocol):
    kind: SourceKind
    def datasets(self, only: Collection[str] | None = None) -> Iterator[DatasetRecord]: ...
    def subjects(self, dataset: DatasetRecord) -> Iterator[SubjectRecord]: ...
    def read_bytes(self, subject: SubjectRecord, role: FileRole, *,
                   byte_range: tuple[int, int] | None = None) -> bytes: ...
    def lease(self, subject: SubjectRecord,
              roles: Collection[FileRole]) -> AbstractContextManager[Mapping[FileRole, Path]]: ...
    def template_dir(self, dataset: DatasetRecord) -> Path | None: ...
    def config(self) -> SourceConfig: ...        # SourceConfig is imported under TYPE_CHECKING
```

- `datasets()`: local yields exactly one; remote one per dataset prefix, `only` restricting by accession.
- `subjects()` yields in ascending `subject_label` order. It lists and stats; it opens nothing. One file per role, the first existing candidate of `ROLE_CANDIDATES`.
- `read_bytes()` is for Tier A files. `byte_range` is inclusive `(first, last)`.
- `lease()` is the resolver of `architecture.md`: local paths for the requested roles that the subject has; absent roles are absent from the mapping; it never raises for a missing role. Local: the files in place, nothing copied, nothing done on exit. Remote: downloaded into the cache, pinned while the context is open, deleted on exit unless the cache retains that role.
- `read_bytes()` and `lease()` first compare the file's size and `etag_or_mtime` with the `FileRef`; a difference raises `SourceError("changed", ...)`. The unit then fails with reason `source changed`, and the next inventory picks the change up.
- `template_dir()`: a local directory holding the `fsaverage` next to the subjects, or None. Remote: its template files (the eight required ones and the two optional area files, 6.3) are fetched once and retained.
- `config()`: the `SourceConfig` from which `run.from_config` (5.9) builds an equal source. `Run` stores it when a run is created, so a source never has to be described twice.
- A `SubjectRecord` rebuilt from the catalog equals the one the source yielded, so every phase after inventory starts from the catalog and never lists the source again.

`sources/patterns.py` gains the two `ROLE_CANDIDATES` entries and:

```python
TEMPLATE_FILES: tuple[str, ...] = (          # per hemisphere h, relative to an fsaverage directory; all required
    "surf/{h}.sphere.reg", "surf/{h}.white", "label/{h}.cortex.label", "label/{h}.aparc.annot",
)
TEMPLATE_OPTIONAL_FILES: tuple[str, ...] = ("surf/{h}.white.avg.area.mgh",)     # used when present (6.3)
TEMPLATE_NAME_PATTERNS: tuple[str, ...] = ("fsaverage*", "cvs_avg35*", "V1_average", "?h.EC_average")
def is_template_name(name: str) -> bool                      # fnmatchcase against the patterns
def is_subject_candidate(path: Path | str) -> bool
    # a directory (links followed) that has subdirectories mri/ and surf/, and whose own name and the name of
    # what it links to are not template names. Raises OSError when the directory cannot be examined.
```

`is_freesurfer_subject_dir` is unchanged. A candidate that fails it is inventoried as incomplete, so a crashed recon-all shows up in the count.

### 5.3 `sources/local.py`

```python
class LocalSource:
    kind: SourceKind = "local"
    def __init__(self, path: Path | str, *, subjects: Sequence[str] = (), identity: str | None = None,
                 field_strength: float | None = None) -> None
    dataset: DatasetRecord                      # detected in __init__; SourceError("layout", ...) if nothing matches
    # field_strength: positive and finite, else ValueError
    # implements Source; config() returns SourceConfig(kind="local", root, identity, subjects, field_strength)

def local(path: Path | str, *, subjects: Sequence[str] = (),
          field_strength: float | None = None) -> LocalSource                   # sc.sources.local(...)
def detect_layout(path: Path) -> tuple[DatasetRecord, tuple[str, ...]]          # record, implied subject globs
def find_adjacent_template(subjects_dir: Path) -> Path | None
```

`sources/__init__.py` has no top-level import: a module `__getattr__` returns `local` and `LocalSource` from `sources.local` on first use, as `render/__init__.py` does for the renderer. `from_config` is in `run.py` (5.9), its only caller. Both follow from an import cycle found in review: `config.py` imports `SourceKind` from `sources/base.py`, which runs `sources/__init__.py` first, which must therefore not import `config.py`.

Detection, first match wins; the CLI says which matched.

| # | Condition | `layout` | `root` | `fs_root` |
|---|---|---|---|---|
| 1 | `path` is itself a subject candidate | `subjects_dir` | `path.parent` | `.`; the implied glob is the directory name, escaped |
| 2 | `path/sourcedata/freesurfer` has a subject candidate | `fmriprep` | `path` | `sourcedata/freesurfer` |
| 3 | `path/freesurfer` has one | `fmriprep_legacy` if `path/fmriprep` is a directory, else `bids_derivatives` | `path` | `freesurfer` |
| 4 | `path.name == "fmriprep"` and `path/../freesurfer` has one | `fmriprep_legacy` | `path.parent` | `freesurfer` |
| 5 | `path` has a child that is a subject candidate | `subjects_dir` | `path` | `.` |
| 6 | otherwise | `SourceError("layout", ...)` listing the five places looked in | | |

- If rules 2 and 3 both match, rule 2 wins and a warning names the other directory.
- `pipeline` is `fmriprep` for the two fMRIPrep layouts, else `freesurfer`. `pipeline_version` comes from `dataset_description.json` (`root/` for `fmriprep`, `root/fmriprep/` for `fmriprep_legacy`): `GeneratedBy[*].Version` where `Name` equals `fMRIPrep` ignoring case, else `PipelineDescription.Version`. Missing or unparsable gives None, never an error.
- `root` in the record is the path as resolved (`Path.resolve()`). `identity` is the `identity` argument when one is given (a stored source being rebuilt, also after `--relocate`), else the resolved root. So a `SubjectRecord` built after the data moved still hashes to the same `subject_uid`.
- Subjects are the children of `root/fs_root` that are subject candidates, sorted by name. A child that cannot be examined (OSError) is yielded with `problem = "unreadable: <strerror>"` and no files. A child whose name is not valid UTF-8 is skipped with a warning: a label is hashed and stored as text (as built, review of milestone 2). Two names that resolve to the same directory: the first in sorted order is kept and a warning names the other.
- Globs (`subjects`) are `fnmatch.fnmatchcase` patterns on the directory name; several are a union. They define the subject set of the run and are stored in `config.yaml`. A glob that matches nothing is `SourceError("not found", ...)` naming the glob and up to ten names that exist.
- `variant` of a `FileRef`: `patterns.variant_name` (a link inside its own directory names its target if that is a candidate name of the role), which `_input_file` in `render/api.py` calls as well, so the two cannot differ (as built, review of milestone 2).
- Adjacent template: `root/fs_root/fsaverage` (a directory or a link to one) that has every file of `patterns.TEMPLATE_FILES` for both hemispheres. A dangling link gives None. One that lacks files is not used and is reported as `incomplete: <files>` in the template message (6.3).
- Read-only by construction: the module has no call that opens for writing, creates or removes. A run directory inside `root/fs_root` is refused (`ConfigError("invalid", ...)`), because the next inventory would walk it. Tests compare `snapshot_tree` before and after every phase.
- Longitudinal (`sub-01.long.base`) and session-level (`sub-01_ses-02`) directories are ordinary subjects with those labels.

### 5.4 `config.py`

All models are `_Frozen`. An unknown key in `config.yaml` is an error that names the key. `config.py` imports `SourceKind` and `FileRole` from `sources/base.py` and nothing else from `sources/`.

```python
CONFIG_SCHEMA_VERSION = 1

class SourceConfig(_Frozen):
    kind: SourceKind
    root: str | None = None                  # local source in a local run: absolute path. None in a census run,
                                             # where the root is in datasets.root only
    identity: str | None = None              # local source in a local run: set at first inventory, then fixed
    dataset: str | None = None               # local source in a census run: its pseudonym, the link to datasets.root
    subjects: tuple[str, ...] = ()           # local only: globs; () = all
    field_strength: float | None = None      # local only: --field-strength T, copied to subjects.field_strength
    bucket: str | None = None                # remote only
    prefix: str | None = None                # remote only

class IdentityConfig(_Frozen):
    kind: Literal["local-salt", "dataset-key"]
    local_salt: str | None = None            # 64 hex; only when kind == "local-salt"
    key_fingerprint: str                     # 12 hex; for both kinds

class InventoryConfig(_Frozen):
    logs: Literal["full", "head"] = "full"
    head_bytes: int = 65536

class RegionConfig(_Frozen):                 # hashed into the template id
    occipital: tuple[str, ...] = ("cuneus", "lateraloccipital", "lingual", "pericalcarine")
    posterior: tuple[str, ...] = ("fusiform", "precuneus")
    cut_mm: float = 45.0
    medial: tuple[str, ...] = ("cuneus", "pericalcarine", "precuneus")
    inferior: tuple[str, ...] = ("fusiform", "lingual")
    control: tuple[str, ...] = ("caudalmiddlefrontal", "rostralmiddlefrontal", "superiorfrontal")

MeshName = Literal["fsaverage4", "fsaverage5", "fsaverage6", "fsaverage"]     # icosahedral order 4, 5, 6, 7

class MetricsConfig(_Frozen):
    mesh: MeshName = "fsaverage6"            # the small meshes exist for tests and for O5
    fsaverage_dir: str | None = None         # --fsaverage DIR; not hashed (the template's content is)
    regions: RegionConfig = RegionConfig()
    knn: int = 8
    min_valid_weight: float = 0.5
    thickness_max_mm: float = 5.0
    ceiling_margin_mm: float = 0.1
    min_coverage: float = 0.9                # fraction of template cortex vertices that must get a value
    max_fallback_frac: float = 0.01          # more nearest-neighbour fallbacks than this: no map (6.6)
    min_reference_n: int = 30
    min_vertex_n: int = 20
    mad_smooth_iterations: int = 2
    mad_floor_mm: float = 0.05
    cluster_z: float = 3.0
    cluster_min_vertices: int = 1            # 1 = no minimum extent (T25)
    clusters_kept: int = 5

class ScoringConfig(_Frozen):
    components: tuple[str, ...] = ("mass_medial", "mass_inferior", "ceiling_frac", "occ_ctrl_ratio")
    mass: Literal["excess", "sum_z"] = "excess"                      # T25
    standardize: Literal["normal_score", "robust_z"] = "normal_score"   # T5
    population: Literal["run", "stratum"] = "run"                    # T24

class IntensityConfig(_Frozen):              # milestone 5 (8.9); present from the start so the schema does not change
    depths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)               # fractions of the way from white to pial
    percentiles: tuple[float, float] = (1.0, 99.0)

class SelectConfig(_Frozen):                 # milestone 5
    min_per_dataset: int = 5
    top_decile_factor: float = 3.0
    extreme_quantile: float = 0.99
    render_scope: Literal["all", "selected"] = "all"                 # T28

class AppConfig(_Frozen):                    # not hashed
    rater_name: str = "local"
    raters_per_subject: int = 1              # census: 2
    repeat_rate: float = 0.0                 # census: 0.05; honoured in local mode too when above 0
    repeat_gap: int = 200
    mark_max_dist_mm: float = 5.0            # must not exceed render.vertex_map_margin_mm (checked at server start)
    mark_near_mm: float = 2.0                # neighbourhood of a mark for vertex-level analysis (7.3)
    mark_show_mm: float = 3.0
    require_marks: bool | None = None        # None: protocol.yaml decides
    blind: bool = False
    queue_seed: int                          # drawn once when the run is created
    media_base_url: str | None = None

class RemoteConfig(_Frozen):                 # not hashed
    max_concurrency: int = 8
    max_retries: int = 6
    timeout_s: float = 120.0

class CacheConfig(_Frozen):                  # not hashed
    max_bytes: int = 4_000_000_000
    batch_subjects: int = 32
    retain_roles: tuple[FileRole, ...] = ()  # roles kept after their lease, beside Tier A and the template (8.2)

class RunConfig(_Frozen):
    schema_version: int = CONFIG_SCHEMA_VERSION
    mode: Literal["local", "census"]
    sources: tuple[SourceConfig, ...]
    identity: IdentityConfig
    inventory: InventoryConfig = InventoryConfig()
    render: RenderConfig = RenderConfig()
    render_video: bool = True                # new_local_config sets False
    metrics: MetricsConfig = MetricsConfig()
    scoring: ScoringConfig = ScoringConfig()
    intensity: IntensityConfig = IntensityConfig()
    select: SelectConfig = SelectConfig()
    app: AppConfig
    remote: RemoteConfig = RemoteConfig()
    cache: CacheConfig = CacheConfig()

class ConfigError(Exception):                # reason, detail
def new_local_config(root: Path, *, subjects: Sequence[str] = (), **overrides: object) -> RunConfig
def new_census_config(key: HashKey, **overrides: object) -> RunConfig
def load_config(run_dir: Path) -> RunConfig          # ConfigError("missing" | "invalid" | "newer schema")
def write_config(run_dir: Path, config: RunConfig, *, exclusive: bool = False) -> None
    # atomic. exclusive (creating a run): an existing config.yaml is left alone and ConfigError("invalid") raised,
    # so of two processes creating a run in one directory only one does
def run_key(config: RunConfig, dataset_key: HashKey | None = None) -> HashKey
def stage_hash(...) -> str                           # section 4
```

- `config.yaml` is `yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)`, written atomically. It holds the salt of a local run and never the dataset key, the run secret or an accession: `--dataset` filters are command-line arguments and are not stored. In a census run it holds no path either: a local source is stored as `{kind: local, dataset: <pseudonym>}` and its root is read from the catalog when the run is opened.
- The salt is created by `new_local_config` and never regenerated. `run_key` returns the salt for a local run; for a census run it requires `dataset_key` and checks its fingerprint (`ConfigError("key", ...)`). It is called by `discover` and `inventory` only.
- Stored options are sticky. Every command-line option that maps to a stored value (`--blind`, `--require-marks`, `--video`, `--subjects`, `--fsaverage`, `--field-strength`; `--jobs`, `--port` and `--rater` are per invocation and not stored) is tri-state: when it is not given, the stored value is used, and the default applies only when the run is created. A given option that differs from the stored value updates `config.yaml`; stages whose hash changed run again, the others are skipped. So running a command a second time without flags changes nothing: `config.yaml` is byte-identical and nothing is redone.
- Two changes are guarded. Changing `app.blind` on a run that has a completed assignment is refused (exit 2) unless `--force`; with it, the change is printed on stderr and recorded as an event of type `config_change` in `app.sqlite`. Changing `--subjects` is allowed; subjects that no longer match get `present = 0` at the next inventory and keep their ratings.
- Values that must not be in the run directory (the key file path) come from `pydantic-settings` in milestone 5: `class RuntimeSettings(BaseSettings)`, prefix `SURFCENSUS_`, field `key_file: Path = ~/.config/surfcensus/dataset.key`.

### 5.5 `store.py`

Standard-library `sqlite3`, `PRAGMA user_version` migrations (D18), `PRAGMA foreign_keys = ON`. Journal: the default rollback journal in a local run (network file systems), WAL with `busy_timeout = 60 s` in a census run, where `Catalog.create` warns when the run directory is on a network file system. Every write is one short `BEGIN IMMEDIATE` transaction. No `STRICT` tables. A file that another connection keeps locked beyond the busy timeout is `ConfigError("invalid", "<name> is in use by another process; ...")` on opening and at the start of a write, never "not a SQLite database". A failed write rolls back only when SQLite has not ended the transaction itself (a full disk), so the cause is what is raised.

```sql
-- migration 1 (milestone 2). Later milestones add none: templates stays empty until milestone 3,
-- dataset_files until milestone 5.
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
-- keys: run_id (uuid4 hex), mode, key_kind, key_fingerprint, created_at, surfcensus_version, template_id,
-- and run_secret (64 hex; private, 5.1)

CREATE TABLE datasets (
  dataset_id        INTEGER PRIMARY KEY, -- private: assigned in discovery order, which is the bucket's order
  pseudonym         TEXT NOT NULL,
  source            TEXT NOT NULL CHECK (source IN ('local','openneuro-derivatives','openneuro-embedded')),
  identity          TEXT NOT NULL,     -- private
  accession         TEXT,              -- private
  root              TEXT NOT NULL,     -- private
  fs_root           TEXT NOT NULL,     -- private
  template_location TEXT,              -- private
  layout            TEXT NOT NULL,
  pipeline          TEXT NOT NULL,
  pipeline_version  TEXT,
  derivative_ref    TEXT,              -- private
  derivative_ref_source TEXT,
  snapshot_tag      TEXT,              -- private
  n_subjects_listed INTEGER,
  fs_subject_found  INTEGER,           -- census: 1 if a probed subject passes the pattern of phases.md (8.3)
  template_sphere_match INTEGER,       -- 1 if the adjacent fsaverage sphere equals the pinned template's
  discovered_at     TEXT NOT NULL,     -- UTC, ISO 8601
  UNIQUE (source, identity, pipeline)
);
CREATE INDEX datasets_pseudonym ON datasets (pseudonym);

CREATE TABLE subjects (
  subject_uid    TEXT PRIMARY KEY,
  dataset_id     INTEGER NOT NULL REFERENCES datasets(dataset_id),
  subject_label  TEXT NOT NULL,        -- private in a census run
  participant    TEXT,                 -- private in a census run; "sub-01" of "sub-01_ses-2"
  fs_dir         TEXT NOT NULL,        -- private; relative to datasets.root
  present        INTEGER NOT NULL DEFAULT 1,   -- 0: no longer in the source's listing
  complete       INTEGER NOT NULL,     -- every RENDER_ROLES role present (phases.md)
  tier_a         INTEGER NOT NULL,     -- TIER_A_REQUIRED present
  tier_b         INTEGER NOT NULL,     -- TIER_B_REQUIRED present
  missing_roles  TEXT NOT NULL DEFAULT '',     -- comma-separated, among TIER_A_REQUIRED, TIER_B, RENDER_ROLES
  problem        TEXT,
  longitudinal   INTEGER NOT NULL DEFAULT 0,   -- directory name contains ".long."
  fs_build_stamp TEXT,
  fs_version     TEXT,                 -- "6.0.1"; NULL for a development build
  fs_major       INTEGER,
  recon_flags    TEXT,                 -- JSON array of flag names, sorted, no values
  n_invocations  INTEGER,              -- over the three logs
  recon_finished INTEGER,              -- NULL if not read or read as a head range
  t2pial INTEGER, flairpial INTEGER, hires INTEGER, expert INTEGER, xopts INTEGER, edited INTEGER,
  pial_variant   TEXT,                 -- "pial", "pial.T1", "pial.T2", "pial.FLAIR"
  holes_lh INTEGER, holes_rh INTEGER,  -- from the log line
  trunc_lh_1 INTEGER, trunc_lh_2 INTEGER, trunc_rh_1 INTEGER, trunc_rh_2 INTEGER,   -- in file order
  age REAL, sex TEXT, field_strength REAL, manufacturer TEXT,                       -- phase 3
  inventory_hash TEXT NOT NULL,
  inventoried_at TEXT NOT NULL,
  UNIQUE (dataset_id, subject_label)
) WITHOUT ROWID;

CREATE TABLE files (
  subject_uid   TEXT NOT NULL REFERENCES subjects(subject_uid) ON DELETE CASCADE,
  role          TEXT NOT NULL,
  location      TEXT NOT NULL,         -- relative to the subject directory; this and the next five are private
  variant       TEXT,
  size          INTEGER NOT NULL,
  etag_or_mtime TEXT NOT NULL,
  last_modified TEXT,
  version_id    TEXT,                  -- remote: x-amz-version-id seen when the file was fetched
  sha256        TEXT,                  -- set when the file was read in full
  PRIMARY KEY (subject_uid, role)
) WITHOUT ROWID;

CREATE TABLE dataset_files (            -- private; dataset-level files that a stored value was read from
  dataset_id  INTEGER NOT NULL REFERENCES datasets(dataset_id),
  role        TEXT NOT NULL,           -- 'dataset_description' | 'participants' | 't1w_sidecar'
  bucket      TEXT,                    -- NULL for a local source
  key         TEXT NOT NULL,           -- object key, or path relative to datasets.root
  size        INTEGER NOT NULL,
  etag        TEXT NOT NULL,           -- or st_mtime_ns for a local file
  version_id  TEXT,
  sha256      TEXT,
  fetched_at  TEXT NOT NULL,
  PRIMARY KEY (dataset_id, role, key)
) WITHOUT ROWID;

CREATE TABLE templates (
  template_id  TEXT PRIMARY KEY,       -- 12 hex (6.3)
  mesh         TEXT NOT NULL,
  origin       TEXT NOT NULL,          -- 'explicit' | 'freesurfer_home' | 'adjacent' | 'bundled'
  dataset_id   INTEGER REFERENCES datasets(dataset_id),   -- for 'adjacent'
  build_stamp  TEXT,
  files_json   TEXT NOT NULL,          -- {"surf/lh.sphere.reg": {"size": ..., "sha256": ...}, ...}
  created_at   TEXT NOT NULL
);

CREATE TABLE work (
  phase           TEXT NOT NULL,       -- 'regional' | 'fsaverage' | 'reference' | 'score' | 'native' | 'enrich' |
                                       -- 'select' | 'intensity'
  unit            TEXT NOT NULL,       -- subject_uid, 'dataset:<pseudonym>', or 'run'
  config_hash     TEXT NOT NULL,
  input_signature TEXT NOT NULL,       -- private
  status          TEXT NOT NULL CHECK (status IN ('done','failed')),
  reason          TEXT,
  detail          TEXT,                -- private; scrubbed text of the failure
  output          TEXT,                -- path relative to the run directory, when there is one file
  surfcensus_version TEXT NOT NULL,
  finished_at     TEXT NOT NULL,
  PRIMARY KEY (phase, unit)
) WITHOUT ROWID;

CREATE TABLE renders (
  subject_uid     TEXT PRIMARY KEY REFERENCES subjects(subject_uid),
  config_hash     TEXT NOT NULL,       -- RenderConfig.config_hash()
  input_signature TEXT NOT NULL,       -- private; render roles, brainmask, sphere_reg roles, template_id
  tile_hash       TEXT,                -- RenderConfig.tile_hash(); NULL for a failed render
  render_key      TEXT,                -- privacy.render_key(...), 16 hex; NULL for a failed render
  video           INTEGER NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('done','failed')),
  reason          TEXT,                -- a RenderError reason, or 'source changed'
  detail          TEXT,                -- scrubbed; not returned by PublicCatalog in a census run
  media_path      TEXT,                -- 'media/<subject_uid>'
  n_frames        INTEGER,
  template_id     TEXT,
  rendered_at     TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE selection (
  subject_uid    TEXT NOT NULL REFERENCES subjects(subject_uid),
  reason         TEXT NOT NULL,        -- 'all' | 'prevalence' | 'detectability' | 'extreme' | 'calibration'
  stratum        TEXT NOT NULL,
  inclusion_prob REAL NOT NULL,
  seed           INTEGER NOT NULL,
  PRIMARY KEY (subject_uid, reason)
) WITHOUT ROWID;

CREATE VIEW public_datasets AS
  SELECT pseudonym, source, layout, pipeline, pipeline_version, n_subjects_listed, fs_subject_found FROM datasets;

CREATE VIEW public_subjects AS
  SELECT s.subject_uid, d.pseudonym AS dataset,
         CASE WHEN (SELECT value FROM meta WHERE key = 'mode') = 'local' THEN s.subject_label END AS subject_label,
         CASE WHEN (SELECT value FROM meta WHERE key = 'mode') = 'local' THEN s.participant END AS participant,
         s.present, s.complete, s.tier_a, s.tier_b, s.missing_roles, s.problem, s.longitudinal,
         s.fs_version, s.fs_major, s.recon_flags, s.n_invocations, s.recon_finished,
         s.t2pial, s.flairpial, s.hires, s.expert, s.xopts, s.edited, s.pial_variant,
         s.holes_lh, s.holes_rh, s.trunc_lh_1, s.trunc_lh_2, s.trunc_rh_1, s.trunc_rh_2,
         s.age, s.sex, s.field_strength, s.manufacturer
  FROM subjects s JOIN datasets d USING (dataset_id);

CREATE VIEW public_files AS
  SELECT subject_uid, role, variant FROM files;
```

`pseudonym` is not `UNIQUE`: two rows of one accession (possible only for the embedded source) share it. `privacy.assign_pseudonyms` guarantees that different `(source, identity)` pairs get different pseudonyms.

```python
SCHEMA_VERSION = 1

class Catalog:
    """The private catalog. Imported only by remote/, inventory/, enrich/, privacy.py, run.py, cli.py and (milestone 5)
    calibration_transfer.py."""
    def __init__(self, path: Path, *, readonly: bool = False) -> None       # applies pending migrations; a file with a
                                                                            # higher user_version: ConfigError("newer schema")
    @classmethod
    def create(cls, path: Path, *, mode: Literal["local", "census"], key: HashKey,
               exist_ok: bool = False) -> Catalog                           # also draws meta.run_secret; built under a
                                                                            # temporary name, named when whole. exist_ok:
                                                                            # opens a catalog of the same run that is there,
                                                                            # finishes an unfinished one (UnfinishedCatalog)
    def close(self) -> None                                                 # also a context manager
    def write(self) -> AbstractContextManager[sqlite3.Connection]           # BEGIN IMMEDIATE; commit or roll back
    def meta(self, key: str) -> str | None
    def set_meta(self, key: str, value: str) -> None
    def run_secret(self) -> bytes
    def upsert_dataset(self, record: DatasetRecord, pseudonym: str, n_subjects_listed: int | None) -> int
    def upsert_subject(self, uid: str, dataset_id: int, record: SubjectRecord, provenance: Provenance,
                       inventory_hash: str, *, digests: Mapping[FileRole, str] | None = None) -> None
        # replaces the files rows in the same transaction; digests: SHA-256 of the files this inventory read in
        # full. Another file keeps its digest while location, variant, size and ETag or mtime are unchanged
    def record_dataset_file(self, dataset_id: int, role: str, bucket: str | None, key: str, *, size: int, etag: str,
                            version_id: str | None, sha256: str | None) -> None
    def mark_absent(self, dataset_id: int, present_labels: Collection[str]) -> list[str]
    def relocate(self, dataset_id: int, record: DatasetRecord) -> None      # root, fs_root, template_location; identity stays;
                                                                             # subjects.fs_dir follows a changed fs_root
    def dataset_ids(self, only: Collection[str] | None = None) -> list[int] # accessions or pseudonyms
    def pseudonyms(self, only: Collection[str] | None = None) -> list[str]  # accessions or pseudonyms -> sorted pseudonyms;
                                                                            # an unknown value: ConfigError("invalid")
    def dataset_record(self, dataset_id: int) -> DatasetRecord
    def subject_record(self, uid: str) -> SubjectRecord
    def inventory_hash(self, uid: str) -> str | None
    def upsert_template(self, template_id: str, mesh: str, origin: str, dataset_id: int | None,
                        build_stamp: str | None, files: Mapping[str, Mapping[str, object]]) -> None
    def failures(self, phase: str | None = None) -> pd.DataFrame            # unit, phase, reason, detail; for `status --failures`
    def public(self) -> PublicCatalog

class PublicCatalog:
    """What metrics/, render/, app/, export/, select/, reference/ and calibration.py may hold. It lives in
    store.py and reads the same file, but it returns nothing that item 1 of 5.1 lists as private."""
    @classmethod
    def open(cls, run_dir: Path, *, readonly: bool = True) -> PublicCatalog
        # needs no key. readonly: a `file:...?mode=ro` connection with a 5 s busy timeout that applies no migration
        # and raises ConfigError("invalid", "catalog schema N is older than this version; run any phase command
        # once") when user_version < SCHEMA_VERSION. A newer user_version: ConfigError("newer schema").
    def close(self) -> None
    def mode(self) -> Literal["local", "census"]
    def run_id(self) -> str
    def meta(self, key: str) -> str | None             # only run_id, mode, key_fingerprint, surfcensus_version, template_id
    def datasets(self) -> pd.DataFrame                 # public_datasets, ordered by pseudonym
    def subjects(self) -> pd.DataFrame                 # public_subjects, ordered by subject_uid
    def subject_uids(self, *, datasets: Collection[str] | None = None, present: bool | None = True,
                     complete: bool | None = None, tier_a: bool | None = None,
                     tier_b: bool | None = None) -> list[str]     # datasets: pseudonyms; ordered by (pseudonym, subject_uid)
    def dataset_of(self, uid: str) -> str              # pseudonym
    def roles(self, uid: str) -> dict[FileRole, str | None]      # role -> variant, for the roles the subject has
    def input_signature(self, uid: str, roles: Collection[FileRole], extra: Sequence[str] = ()) -> str
        # computed inside store.py; the caller may pass it back to work_done / record_* and to nothing else
    def set_file_digests(self, uid: str, digests: Mapping[FileRole, str]) -> None    # files.sha256, write-only
    def work(self, phase: str) -> pd.DataFrame         # unit, status, reason, output, finished_at; no signature, no detail
    def work_done(self, phase: str, unit: str, config_hash: str, input_signature: str) -> bool | None
        # True: done and matching. False: failed and matching. None: no row, or a row that does not match.
    def record_work(self, rows: Sequence[WorkRow]) -> None           # one transaction
    def renders(self) -> pd.DataFrame                  # subject_uid, config_hash, tile_hash, render_key, video, status,
                                                       # reason, media_path, n_frames, template_id; and `detail` in a
                                                       # local run only
    def render(self, uid: str) -> RenderRow | None
    def record_renders(self, rows: Sequence[RenderRow]) -> None      # one transaction
    def selection(self) -> pd.DataFrame
    def replace_selection(self, rows: pd.DataFrame, reasons: Collection[str]) -> None   # replaces rows of those reasons
```

`WorkRow` and `RenderRow` are frozen dataclasses in `records.py` with the columns of their tables. A local run sets `busy_timeout = 30 s` on every connection (the app reads while `--serve-after` renders); a census run 60 s.

Table helpers, same module. Arrow schemas and keys come from `surfcensus/schemas.py` (`SCHEMAS: dict[str, pa.Schema]`, `KEYS: dict[str, tuple[str, ...]]`), written in the first step of this milestone.

```python
def write_table(path: Path, frame: pd.DataFrame, schema: pa.Schema, *, sort_by: Sequence[str]) -> None
    # casts to the schema (a missing or extra column is an error), sorts, drops the pandas index and metadata,
    # writes zstd parquet atomically: the bytes depend on the rows and the locked pyarrow only
def read_table(path: Path, schema: pa.Schema | None = None) -> pd.DataFrame
    # an absent file with a schema gives an empty frame of that schema
def upsert_part(run_dir: Path, table: str, dataset: str, frame: pd.DataFrame, *, replace_uids: Collection[str]) -> Path
    # tables/_parts/<table>/<pseudonym>.parquet: drops the existing rows of replace_uids, appends frame, writes
def merge_parts(run_dir: Path, table: str, *, present: Collection[str]) -> Path
    # concatenates the parts, drops rows whose subject_uid is not in `present`, sorts by KEYS[table],
    # writes tables/<table>.parquet
```

### 5.6 Run directory

```
run/
  config.yaml                 no accession, no key, no run secret; a local run holds its salt and its source root
  protocol.yaml               copied from package data on first review or serve (milestone 4)
  calibration.yaml            milestone 4; absent until a calibration set is frozen
  catalog.sqlite              (+ -wal, -shm in a census run)
  app.sqlite                  milestone 4
  tables/
    regional.parquet  vertex_summary.parquet  clusters.parquet  clusters_native.parquet  scores.parquet
    intensity.parquet                             milestone 5
    _parts/<table>/<pseudonym>.parquet
    fsavg/<subject_uid>.npz                       thickness maps; never rewritten for unchanged inputs
    reference/<reference_id>/index.json, <stratum>.npz, members.parquet
  templates/<template_id>/template.json, template.npz     derived arrays, computed locally
  cache/                      remote downloads only; private like the catalog; absent in a local run (8.2)
  media/<subject_uid>/        as in rendering.md
  logs/                       census only: <phase>-<utc timestamp>-<pid>.log, <phase>-summary.json,
                              render-check-<seed>.txt; pseudonyms and uids only
  export/                     written by `surfcensus export`; `release/` by `surfcensus export --release`
```

### 5.7 `inventory/provenance.py`

Pure functions on text: no file access, no network. Text is decoded as UTF-8 with `errors="replace"`.

```python
@dataclass(frozen=True)
class BuildStamp:
    raw: str                       # first line, stripped
    version: str | None            # "6.0.1"
    major: int | None
def parse_build_stamp(text: str) -> BuildStamp
    # version: first match of (?:^|-)v?(\d+\.\d+(?:\.\d+)?)(?=-|$) in raw; none (a development build) gives None

@dataclass(frozen=True)
class ReconLog:
    invocations: tuple[tuple[str, ...], ...]      # flag names of each invocation, in file order
    build_stamp: str | None                       # from the last "build-stamp.txt: ..." line
    truncations: tuple[tuple[int, int], ...]      # every "thickness calculation complete, N:M truncations", in file order
    holes: tuple[int, int] | None                 # last "orig.nofix lhholes = N, rhholes = M"
    finished: bool | None                         # None when partial
def parse_recon_log(text: str, *, partial: bool = False) -> ReconLog

EDIT_FLAGS: tuple[str, ...] = ("-autorecon2-cp", "-autorecon2-wm", "-autorecon2-pial", "-autorecon-pial")

@dataclass(frozen=True)
class Provenance:                                  # defined in records.py, re-exported here; what upsert_subject stores
    build_stamp: str | None
    fs_version: str | None
    fs_major: int | None
    flags: tuple[str, ...]                         # sorted union over all invocations of all logs
    n_invocations: int | None
    finished: bool | None
    t2pial: bool | None                            # None: no log read and the pial variant does not say
    flairpial: bool | None
    hires: bool | None
    expert: bool | None
    xopts: bool | None
    edited: bool | None
    pial_variant: str | None
    holes: tuple[int, int] | None
    truncations: Mapping[Hemi, tuple[int, int] | None]
    participant: str | None
    longitudinal: bool
def subject_provenance(texts: Mapping[FileRole, str], files: Mapping[FileRole, FileRef], subject_label: str,
                       *, partial: bool = False) -> Provenance
def parse_dataset_description(text: str) -> str | None        # the pipeline version (5.3)
```

Rules, each fixed by a test on excerpts of the two real logs with paths replaced:

- An invocation is found at each line matching `^subjid\s+\S+\s*$`. Its arguments are the whitespace tokens of the nearest non-blank line above, minus a first token that ends in `recon-all`. (FreeSurfer 6 prints the arguments on their own line, FreeSurfer 7 after the script path; both verified.)
- Flag names are the tokens matching `^-[A-Za-z][A-Za-z0-9_-]*$`. Everything else (paths, numbers, subject names) is dropped. This is what keeps accessions out of `recon_flags`.
- For X in `T2pial`, `FLAIRpial`, `hires`: within one invocation the last of `-X` and `-noX` decides; the flag is True if any invocation of any log decides True. Whole tokens, case-sensitive. A `pial_variant` of `pial.T2` or `pial.FLAIR` sets `t2pial` or `flairpial` True whatever the logs say.
- `expert`: any `-expert`. `xopts`: any flag starting with `-xopts-`. `edited`: any of `EDIT_FLAGS`.
- `finished`, from `recon_log` only: the closing line of the last invocation (after its last `subjid` line): True for `finished without error`, False for `exited with ERRORS`; None when that invocation has neither, or `partial`. (As built, decision D29 and `data_sources.md`; the draft took the last such line of the whole file, which calls a log finished whose last invocation was killed.)
- Truncations per hemisphere: the last match in that hemisphere's own log; else, when `recon_log` has exactly two matches, those in the order `lh`, `rh` (assumed for a plain recon-all run); else None. The two numbers are stored in file order as `_1`, `_2`.
- `fs_version`: from the `build_stamp` file, else from `ReconLog.build_stamp` of `recon_log`.
- `pial_variant`: `FileRef.variant` of `lh_pial` without the `lh.` prefix.
- `participant`: the leading `sub-<label>` of the directory name when it has that form (`^(sub-[A-Za-z0-9]+)`), else the whole name. `longitudinal`: the name contains `.long.`.

### 5.8 `inventory/subjects.py`

```python
def inventory_subject(source: Source, record: SubjectRecord, config: InventoryConfig) -> Provenance
    # reads build_stamp and the logs through source.read_bytes (full, or the first head_bytes with partial=True
    # when config.logs == "head"); a missing or unreadable Tier A file gives None fields, not an error
def run_inventory(catalog: Catalog, sources: Sequence[Source], config: RunConfig, key: HashKey, *,
                  datasets: Collection[str] | None = None, force: bool = False) -> PhaseReport
```

Per dataset of each source: upsert the dataset with its pseudonym; list the subjects; for each, compute `subject_uid` and `inventory_hash` (input signature over all of its `FileRef`s, with the inventory stage hash as `extra`). A subject whose `inventory_hash` is unchanged is skipped without reading its logs, unless `force`. Otherwise read, parse, upsert. The build stamp and the logs that were read in full are stored with their SHA-256 (`files.sha256`) in the transaction of the upsert, not by a later `set_file_digests`: a subject stored without them would be skipped from then on. A log of which only the head was read gets none. For a remote source the three stats files are fetched in the same step and retained in the cache (`phases.md`: phase 2 fetches Tier A), so stage 4a reads no network; for a local source nothing is fetched. `SourceConfig.field_strength`, when set, is written to `subjects.field_strength` of that source's subjects. Subjects of the dataset that the listing no longer contains get `present = 0`; their ratings and media stay. That includes the subjects of a row of the same `(source, identity)` whose pipeline no source yields any more (data found in the layout of another pipeline at the same root): once the new listing has succeeded they are marked absent, the row's `n_subjects_listed` is set to 0, and a note says so. The skip test also compares the stored `fs_dir`, which the hash does not cover, so a subject that moved into another layout of the same pipeline is stored again. Running it twice changes nothing. Inventory is not in the `work` ledger; `inventory_hash` is its skip test. `run_inventory` is the one phase function that takes the private catalog; `Run.inventory` passes its own members.

### 5.9 `records.py`, `run.py`, `batches.py`, `render/batch.py`

```python
# records.py
@dataclass(frozen=True)
class PhaseReport:
    phase: str
    done: int
    skipped: int
    failed: int                                 # units in the failed state when the command ends, retried or not
    failures: tuple[tuple[str, str], ...]       # (unit, reason); a unit is a subject_uid, "dataset:<pseudonym>" or "run"
    notes: tuple[str, ...] = ()                 # sentences for the user, e.g. the D16 message
    seconds: float = 0.0

@dataclass(frozen=True)
class RenderResult:                             # what a render worker returns
    info: RenderInfo
    extras: Mapping[str, pd.DataFrame] = field(default_factory=dict)   # table name -> rows for that subject;
                                                                       # empty until milestone 5 adds "intensity"

@dataclass(frozen=True)
class RunContext:                               # what a phase function may touch
    run_dir: Path
    config: RunConfig
    public: PublicCatalog
    lease: Callable[[str, Collection[FileRole]], AbstractContextManager[Mapping[FileRole, Path]]]
    render_key: Callable[[str, Mapping[str, str]], str]    # (tile_hash, digests) -> key; closes over the run secret
    scrub: Callable[[str], str]                            # privacy.scrub with this run's pseudonyms and roots

# run.py
METRIC_STAGES: tuple[str, ...] = ("regional", "fsaverage", "reference", "score")

class Run:
    def __init__(self, run_dir: Path | str, source: Source | Sequence[Source] | None = None, *,
                 config: RunConfig | None = None, key: HashKey | None = None, relocate: bool = False,
                 transport: httpx.BaseTransport | None = None) -> None
    run_dir: Path
    config: RunConfig
    def inventory(self, *, datasets: Collection[str] | None = None, force: bool = False) -> PhaseReport
    def metrics(self, *, datasets: Collection[str] | None = None, stages: Collection[str] = METRIC_STAGES,
                jobs: int = 1, force: bool = False, retry_failed: bool = False) -> PhaseReport
    def render(self, *, jobs: int = 1, datasets: Collection[str] | None = None,
               subjects: Sequence[str] | None = None, all_subjects: bool | None = None, video: bool | None = None,
               force: bool = False, retry_failed: bool = False) -> PhaseReport
    def scores(self) -> pd.DataFrame            # primary rows of tables/scores.parquet, sorted by rank; local: with subject_label
    def clusters(self) -> pd.DataFrame          # clusters.parquet left-joined with clusters_native.parquet
    def public(self) -> PublicCatalog
    def context(self) -> RunContext
    def media_dir(self, subject_uid: str) -> Path
    def close(self) -> None                     # also a context manager

def from_config(config: SourceConfig, run_config: RunConfig, run_dir: Path, *, root: str | None = None,
                transport: httpx.BaseTransport | None = None) -> Source
    # `local` in milestone 2; the remote kinds (milestone 5) take bucket and prefix from the SourceConfig and their
    # client and cache from run_config.remote, run_config.cache and run_dir/cache. `root` overrides
    # SourceConfig.root (a local source in a census run, whose root is in the catalog).
```

- `Run(run_dir, source=...)` on an empty or absent directory creates it: `config.yaml` (a local run gets a new salt; each source contributes `source.config()`), written with `write_config(..., exclusive=True)`, then the catalog with its run secret. A place where neither can be written is `ConfigError("invalid", "no run can be made in ...")`. Creating a census run requires `key`. `Run(run_dir)` opens an existing run and rebuilds its sources with `from_config`; a census run opens without a key, and `Run.inventory` then raises `ConfigError("key", ...)`.
- A `source` is matched to a stored `SourceConfig` by kind and, for a local source, by root. A local source whose root matches none raises `ConfigError("other root", "this run directory belongs to <old>; pass --run-dir for a new run, or --relocate if the data moved")`. With `relocate=True` and exactly one local source stored, the stored root is replaced (`Catalog.relocate`, `config.yaml`); `identity` stays, so no `subject_uid` changes. With several local sources `relocate` needs the dataset's pseudonym (`--relocate D-xxxxxxxx`).
- `--dataset` values (accessions or pseudonyms) are resolved by `Run` through `Catalog.pseudonyms` and reach the phase functions as pseudonyms.
- `Run.context()` builds the `RunContext`. Its `lease` looks the subject up in the private catalog and leases through its source, so phase code never holds a `SubjectRecord`; its `render_key` is `privacy.render_key` with the run secret bound.
- `run.py` only delegates, importing inside each method: `inventory` to `inventory.subjects.run_inventory`, `metrics` to `metrics.pipeline.run_metrics`, `render` to `render.batch.run_render`. Until milestone 3, `metrics`, `scores` and `clusters` raise `NotImplementedError("metrics arrive in milestone 3")`. `Run.template()` and `Run.zmap()` are added in milestone 3 by the same owner, because their signatures name types of `templates.py`. Later milestones add methods and do not edit these signatures.

```python
# batches.py
def run_batches(
    units: Sequence[str],                                                   # pending subject_uids, in order
    lease: Callable[[str], AbstractContextManager[Mapping[FileRole, Path]]],
    work: Callable[[str, Mapping[FileRole, Path]], R],                      # a picklable top-level function; runs in a worker
    commit: Callable[[Sequence[tuple[str, R | BaseException]]], None],      # parent; called once per batch
    *, batch_subjects: int, jobs: int, progress: str | None = None,        # tqdm description, stderr
) -> None
```

Per batch: lease every subject of the batch, run `work` in the pool, call `commit`, release the leases. An exception from `lease` or `work` for one subject is handed to `commit` for that subject and the batch goes on. `KeyboardInterrupt` stops after releasing the leases; at most one batch is lost.

```python
# render/batch.py
def run_render(ctx: RunContext, *, template: Template | None = None, jobs: int = 1,
               datasets: Collection[str] | None = None, subjects: Sequence[str] | None = None,
               all_subjects: bool | None = None, video: bool | None = None,
               force: bool = False, retry_failed: bool = False) -> PhaseReport
def render_check_sample(ctx: RunContext, *, fraction: float = 0.02, seed: int) -> Path
def render_pending(ctx: RunContext, *, datasets=None, subjects=None, all_subjects=None, video=None,
                   force: bool = False, retry_failed: bool = False) -> int
    # how many subjects run_render would render with the same arguments; opens no source file, records nothing.
    # Run.render builds its sources only when this is not 0 (as built, review of milestone 2)
def render_status(ctx: RunContext) -> tuple[int, int, int]      # (done, failed, pending) of a plain call
def render_failures(ctx: RunContext) -> dict[str, tuple[str, str | None]]
    # subject_uid -> (reason, detail) of the subjects render_status counts as failed. detail is None where the
    # renders row has it; an incomplete subject without its row yet comes with the detail the row will have
    # (as built, review of milestone 2: `status --failures` lists these, so its list and its count agree)
```

- Units: `subjects` in the order given; else every complete present subject when `all_subjects` is true; else the subjects in `selection`, in queue order when scores exist; else (no selection rows) every complete present subject. `all_subjects=None` means: `config.select.render_scope == "all"` in a census run, false in a local run. Without an explicit order, units go by `(pseudonym, subject_uid)`. `video=None` means `config.render_video`.
- Every present subject that is not `complete` gets a `failed` row with reason `missing surface` and a detail naming the roles, whether or not it is selected, without touching the source. So incomplete subjects appear in review and export with their reason.
- Skip test from the `renders` row (section 4), plus `video` equal and `media/<uid>/render.json` present, before any lease.
- Lease `TIER_C`, and from milestone 3 the two `sphere_reg` roles when the run has a template. `work` calls `render_subject(files, media_dir, config.render, template=..., video=...)` and returns a `RenderResult`. A `RenderError` becomes a `failed` row with its reason; a `SourceError("changed")` becomes `failed` with `source changed`; any other exception becomes `render failure` with the exception type. In each case the detail is the exception text after `ctx.scrub`. When the video is on, `run_render` looks the ffmpeg binary up itself (`encode.ffmpeg_exe`) before the first batch and stops with `ConfigError("missing", ...)`, exit 2, if there is none: the milestone 1 review noted that this failure lies in the environment and must not be recorded as a property of a subject. As built, the same rule covers two more cases (decision D29). The encoder is tried, not only looked up: two 16 px frames are encoded into a temporary directory, so a binary that cannot encode is `ConfigError("missing", ...)` as well. And an outcome that says the media cannot be written (an `OSError` with `ENOSPC`, `EDQUOT` or `EROFS`, or with `EACCES` or `EPERM` on a path under `media/`; a `RuntimeError` of `write_mp4` with the text of the first three) raises `ConfigError("invalid", "the media cannot be written: ...")` from `commit`, before anything of that batch is recorded.
- After a successful render the parent copies the SHA-256 digests of `render.json` into the catalog (`PublicCatalog.set_file_digests`), computes `render_key = ctx.render_key(config.render.tile_hash(), digests of the TIER_C roles read)` and stores it with `tile_hash` in the `renders` row. Rows of `RenderResult.extras` go to the per-dataset part of the table they name.
- Output is byte-identical for `jobs=1` and `jobs=2`.
- From milestone 3, `run_render` ends with the `native` stage of 6.9 when `clusters.parquet` exists.
- `render_check_sample` supports the phase 6 exit check: it draws `ceil(fraction x n)` of the done renders with `numpy.random.default_rng(seed)` from the sorted uids, and writes `logs/render-check-<seed>.txt` (one `subject_uid` and poster path per line) and `logs/render-check-<seed>.html` (the posters on one page, relative links). The seed is in the file name and the first line.

`render/model.py` gains one method in this milestone:

```python
class RenderConfig(_Frozen):
    def tile_hash(self) -> str
        # first 12 hex of the SHA-256 of the canonical JSON of the fields that change a tile's pixels or geometry:
        # box_x_mm, box_y_mm, box_z_mm, margin_mm, slice_step_mm, panel_px, radiological, sagittal_anterior_left,
        # pial_colour, white_colour, line_width_px, window_percentiles, brainmask_dilation, webp_quality.
        # No schema number, no video, crosshair or vertex-map field.
```

A test pins the field list against `RenderConfig.model_fields`: a new field must be put on one side or the other.

### 5.10 CLI of this milestone

```
surfcensus inventory --run-dir DIR [--source PATH] [--subjects GLOB]... [--field-strength T] [--dataset ID]...
                     [--force] [--relocate [PSEUDONYM]]
surfcensus render    --run-dir DIR [--jobs J] [--dataset ID]... [--all | --selected] [--video / --no-video]
                     [--force] [--retry-failed]
surfcensus render    --run-dir DIR --check-sample [--fraction 0.02] [--seed S]
surfcensus status    --run-dir DIR [--failures]
```

- `--source PATH` creates a local run when `DIR` has none; on an existing run it must match the stored root, or be given with `--relocate`.
- An `OSError` or `sqlite3.OperationalError` that reaches a phase command (a read-only or full run directory) is printed as `invalid: the run directory cannot be used: ...` with exit code 2, like a `ConfigError`. An invalid `--field-strength` is `invalid: ...` with exit code 2 on a new run as on a stored one.
- `inventory` prints: the layout found, subject counts (found, complete, with Tier A, with Tier B), FreeSurfer versions with counts, and each incomplete or unreadable subject with what it lacks.
- `render` shows a progress bar and one closing line with counts and the failed subjects with their reasons. `--check-sample` renders nothing; without `--seed` it draws one and prints it.
- `status` prints one line per phase with done, failed and pending, from the catalog; pseudonyms only. `--failures` lists every failed unit with its reason and detail; it reads the private catalog and its output is for the operator's terminal, not for a log. Its render lines are the subjects that the render line counts as failed (`render_failures`), not every failed row of `renders` (as built, review of milestone 2).
- `--backend mpl|freeview` of `architecture.md` is not offered until the Freeview backend exists.
- `render-subject`, `three_plane` and `render_subject` refuse an output that resolves to a place inside the subject directory: `OutputInSourceError`, a `ValueError`, before anything is written (exit code 2), since a source directory is never written to. The milestone 1 review had left this to this milestone; it was built when the review's fixes were verified (`rendering.md`, "Failures and re-rendering"), so nothing remains to do here.

---

## 6. Milestone 3: template, registration, metrics, scoring, render schema 3

Result: `run.metrics()`, `run.scores()`, `run.clusters()`, `run.zmap()`, `run.template()` and `surfcensus metrics` work; renders carry fsaverage6 vertices and the D12 pole distance; cluster centroids have native coordinates.

New files: `fsio/stats.py`, `fsio/annot.py`, `fsio/morph.py`, `fsio/registration.py`, `templates.py`, `metrics/{__init__,regional,ceiling,fsaverage,clusters,scoring,pipeline}.py`, `reference/{__init__,load}.py`, `data/reference/README.md`. Changed: `render/model.py`, `render/api.py`, `render/vertexmap.py`, `render/batch.py` (passes the template; ends with the `native` stage), `run.py` (`template`, `zmap`; the three methods stop raising), `cli.py` (`metrics`; `render-subject --fsaverage / --no-template`). `schemas.py` exists since milestone 2.

### 6.1 `fsio/stats.py`

```python
@dataclass(frozen=True)
class StatsFile:
    measures: dict[str, tuple[float, str]]     # "MeanThickness" -> (2.39047, "mm"); "lhSurfaceHoles" -> (41.0, "unitless")
    columns: tuple[str, ...]                   # from the "# ColHeaders" line
    rows: tuple[tuple[str, ...], ...]          # data rows as strings

def parse_stats(text: str) -> StatsFile
def read_stats(path: Path | str) -> StatsFile                  # UTF-8, errors="replace"
def aparc_frame(stats: StatsFile) -> pd.DataFrame
    # StructName (str), NumVert, SurfArea, GrayVol (int64), ThickAvg, ThickStd (float64);
    # ValueError when a column is missing
def surface_holes(aseg: StatsFile) -> tuple[int, int] | None   # (lh, rh); None when the measures are absent
def euler_number(holes: int) -> int                            # 2 - 2 * holes, the pre-fix surface
```

- A measure line is `# Measure <a>, <b>..., <value>, <unit>`, split on commas: the unit is the last field, the value the one before it, the key the first whitespace-separated token of the second field. A line that does not end in a number and a unit is skipped. This reads the malformed FreeSurfer 6 line as `CortexVol`.
- Only `# Measure`, `# ColHeaders` and data rows are read. `SUBJECTS_DIR`, `hostname`, `user`, `cmdline`, `subjectname` are never parsed or kept.
- Data rows are split on whitespace; a row whose field count differs from the header raises `ValueError` with the line number.
- Holes come from `aseg.stats`; when it has none, the caller uses the log's (`subjects.holes_*`). The Euler number is not stored: it is `2 - 2 * holes` (the relation `mris_euler_number` prints, from memory).

### 6.2 `fsio/annot.py`, `fsio/morph.py`

```python
@dataclass(frozen=True)
class Annotation:
    labels: Int[np.ndarray, "n"]            # int32, index into names; -1 = no label
    names: tuple[str, ...]                  # decoded as UTF-8
    def mask(self, *names: str) -> Bool[np.ndarray, "n"]       # KeyError naming every name the file lacks
    def cortex(self) -> Bool[np.ndarray, "n"]                  # labels >= 0 and name not in ("unknown", "corpuscallosum")

def read_annotation(path: Path | str) -> Annotation            # nibabel read_annot; ValueError when there is no colour table
def read_label(path: Path | str) -> Int[np.ndarray, "k"]       # int64 vertex indices of a .label file, sorted ascending
def read_morph(path: Path | str) -> Float[np.ndarray, "n"]     # ?h.thickness: float32, native byte order
```

### 6.3 `templates.py`

The template is everything the metrics need on the target mesh, derived at run time from a FreeSurfer `fsaverage` directory. Nothing fsaverage-derived ships until T15 is answered.

```python
TEMPLATE_SCHEMA = 1
from surfcensus.sources.patterns import TEMPLATE_FILES, TEMPLATE_OPTIONAL_FILES   # 5.2
RegionName = Literal["search", "medial", "inferior", "lateral", "control"]
REGION_NAMES: tuple[RegionName, ...]
Origin = Literal["explicit", "freesurfer_home", "adjacent", "bundled"]
MESH_ORDER: dict[MeshName, int] = {"fsaverage4": 4, "fsaverage5": 5, "fsaverage6": 6, "fsaverage": 7}

@dataclass(frozen=True)
class TemplateHemi:
    sphere: Float[np.ndarray, "v 3"]         # unit vectors, float64; v = 40,962 for fsaverage6
    faces: Int[np.ndarray, "f 3"]            # int32, 81,920, wound outward
    white: Float[np.ndarray, "v 3"]          # float32, the template's surface RAS
    cortex: Bool[np.ndarray, "v"]
    parcel: Int[np.ndarray, "v"]             # int16, index into Template.parcel_names; -1 none
    area: Float[np.ndarray, "v"]             # float32, mm2; see below
    regions: Mapping[RegionName, Bool[np.ndarray, "v"]]
    pole_vertex: int

@dataclass(frozen=True)
class Template:
    mesh: MeshName
    order: int                               # MESH_ORDER[mesh]
    template_id: str                         # 12 hex
    origin: Origin
    files: Mapping[str, tuple[int, str]]     # relative path -> (size, sha256) of the source files that were read
    area_source: Literal["white.avg.area", "white mesh"]
    build_stamp: str | None                  # scripts/build-stamp.txt of the directory, if any
    parcel_names: tuple[str, ...]
    regions: RegionConfig
    hemis: Mapping[Hemi, TemplateHemi]
    def adjacency(self, hemi: Hemi) -> scipy.sparse.csr_matrix     # symmetric 0/1, no diagonal; cached

class TemplateError(Exception):              # reason, detail
class TemplateNotFound(Exception):
    looked_in: tuple[tuple[str, str, str], ...]    # (origin, where, "not set" | "absent" | "incomplete: <files>" | "not shipped")

def coarsen_faces(faces: Int[np.ndarray, "f 3"], n_coarse: int) -> Int[np.ndarray, "g 3"]
def template_candidates(*, adjacent: Path | None, env: Mapping[str, str] = os.environ) -> list[tuple[Origin, Path | None]]
def build_template(directory: Path, origin: Origin, *, mesh: MeshName = "fsaverage6",
                   regions: RegionConfig = RegionConfig()) -> Template
def write_template(directory: Path, template: Template) -> None       # template.json, template.npz
def read_template(directory: Path) -> Template
def find_template(*, config: MetricsConfig, adjacent: Path | None, cache_dir: Path | None = None,
                  pinned: str | None = None, retemplate: bool = False,
                  env: Mapping[str, str] = os.environ) -> Template    # raises TemplateNotFound, TemplateError
```

`build_template`:

- A directory is usable when all eight required files exist (`TEMPLATE_FILES` for both hemispheres). `sphere.reg` must have `10 * 4**k + 2` vertices with `k >= order`, else `TemplateError("not icosahedral", ...)`. `order` is `MESH_ORDER[mesh]`. Coordinates are normalized to unit length; radii must agree within 1%.
- `coarsen_faces` is applied `k - order` times, and the first `10 * 4**order + 2` entries of every per-vertex array are kept. For one step with `n_coarse` coarse vertices: every vertex with index `>= n_coarse` must have exactly two neighbours below `n_coarse` (its parents), and the faces made of three such vertices must number `2 * n_coarse - 4`; otherwise `TemplateError("not icosahedral", ...)`. For such a face `(m0, m1, m2)`, in input order, the coarse face is `(a, b, c)` with `a` the parent shared by `m2` and `m0`, `b` by `m0` and `m1`, `c` by `m1` and `m2`. Verified: this reproduces the real fsaverage5 faces and the faces of `icosphere(k - 1)` exactly.
- `cortex`: indices of `cortex.label` below the vertex count. `parcel`: the first labels of `aparc.annot`. A name of `RegionConfig` that the annotation lacks is `TemplateError("missing parcel", ...)`.
- `pole_vertex`: the cortex vertex with the smallest white y (lowest index on a tie). On fsaverage `lh`: 12950.
- Regions, all intersected with `cortex`: `search` = `occipital` parcels, plus `posterior` parcels where `white_y <= white_y[pole_vertex] + cut_mm`; `medial` and `inferior` = `search` restricted to those parcel lists; `lateral` = the rest of `search`; `control` = the control parcels.
- `area`. The fsaverage white surface is an average of coordinates and is smaller than real cortex by a factor that differs between regions: against FreeSurfer's own average of subject areas (`surf/{h}.white.avg.area.mgh`) the fsaverage6 white-mesh area is 0.77 over the cortex, 0.59 in lateraloccipital, 0.68 in cuneus, 0.72 in lingual and 1.01 in insula (reviewer measurement). So when the optional file `surf/{h}.white.avg.area.mgh` exists (it does in a FreeSurfer install and in the copy next to fMRIPrep subjects), `area[v]` is the sum of its full-resolution values over the fsaverage vertices whose nearest vertex of the coarse mesh, on the sphere, is `v`, and `area_source = "white.avg.area"`. Otherwise `area[v]` is a third of the areas of the incident coarse faces at `white`, and `area_source = "white mesh"`. The file is read with nibabel as an MGH volume and flattened; its length must equal the vertex count of the full sphere.
- `template_id`: first 12 hex of the SHA-256 of `canonical_json({"schema": TEMPLATE_SCHEMA, "mesh": mesh, "order": order, "regions": regions, "files": {path: sha256}})`, over the files that were read (the optional one included when present). It depends on content only, so the same fsaverage found in two places is the same template.

Cache format, `templates/<template_id>/`: `template.npz` with, per hemisphere `h`, `{h}_sphere` `<f8`, `{h}_faces` `<i4`, `{h}_white` `<f4`, `{h}_cortex` `u1`, `{h}_parcel` `<i2`, `{h}_area` `<f4`, `{h}_region_<name>` `u1`; and `template.json` with `schema`, `template_id`, `mesh`, `order`, `origin`, `area_source`, `build_stamp`, `parcel_names`, `regions`, `files`, `pole_vertex`. No path.

`find_template`, in order:

1. `pinned` with `cache_dir/<pinned>/` present: `read_template`. A run keeps the template it was first scored with, whatever the environment looks like later. If `config.fsaverage_dir` is also set: when it is usable and builds another id, `TemplateError("changed", "<pinned> -> <new>; pass --retemplate")` unless `retemplate` (then the new one is built and becomes the pin); when it is not usable, the pinned template is used and one warning says so.
2. `config.fsaverage_dir` (`--fsaverage DIR`), when nothing is pinned or `retemplate`. A directory that is not usable raises `TemplateNotFound`; there is no fall-through to the candidates below.
3. D14 order: `$FREESURFER_HOME/subjects/fsaverage`, `adjacent`, the bundled copy `surfcensus/data/fsaverage` (absent for now). The first usable one is built.
4. `TemplateNotFound` with everything looked at.

`Run.template(*, retemplate: bool = False) -> Template | None` (added to `run.py` in this milestone) calls it with `adjacent` from the first dataset, in pseudonym order, that has a `template_location`; writes the cache, the `templates` row and `meta.template_id` on first use. When `meta.template_id` is set and no template with that id can be read from the cache or rebuilt from the candidates, it raises `TemplateError("changed", "the template this run was scored with is no longer available; pass --fsaverage DIR or --retemplate")`: a run that has scores never falls back to D16 silently because a volume was unmounted or `templates/` was deleted. Only a run that never had a template gets None, on `TemplateNotFound`, after logging once, at warning level:

```
No fsaverage template found. Looked in:
  $FREESURFER_HOME/subjects/fsaverage   (FREESURFER_HOME is not set)
  /data/SUBJECTS_DIR/fsaverage          (absent)
  bundled copy                          (not shipped with this version)
Thickness is not compared per location, and click marks will not map to fsaverage. Subjects are ranked on
the raw ceiling fraction and the occipital-to-control thickness ratio (decision D16).
To enable: pass --fsaverage /path/to/fsaverage (any FreeSurfer install has one in subjects/).
```

The path lines appear in a local run only.

### 6.4 `fsio/registration.py`

```python
@dataclass(frozen=True)
class SphereMap:
    n_source: int
    n_target: int
    corners: Int[np.ndarray, "t 3"]            # source vertices combined for each target vertex
    weights: Float[np.ndarray, "t 3"]          # barycentric, non-negative, rows sum to 1; a fallback row is (1, 0, 0)
    fallback: Bool[np.ndarray, "t"]            # True where no containing face was found (nearest neighbour)
    nearest_source: Int[np.ndarray, "t"]       # source vertex nearest each target vertex
    nearest_target: Int[np.ndarray, "s"]       # target vertex nearest each source vertex
    flipped: bool                              # the source faces were wound inward and were reversed

def unit_sphere(vertices: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]
    # ValueError("degenerate sphere: ...") for a non-finite or zero radius, or radii more than 1% from their median
def build_sphere_map(source: Surface, target: Float[np.ndarray, "t 3"], *, k: int = 8) -> SphereMap
def resample(values: Float[np.ndarray, "s"], smap: SphereMap, *, valid: Bool[np.ndarray, "s"] | None = None,
             min_valid_weight: float = 0.5) -> Float[np.ndarray, "t"]
```

- Both spheres are put on the unit sphere. A `scipy.spatial.cKDTree` over the source vertices gives, for each target point `p`, its `k` nearest source vertices, nearest first (`nearest_source` is the first). For `j = 0 .. k-1`, for the points still unresolved, the faces incident to the `j`-th nearest vertex are tested in ascending face index. With face corners `a, b, c`: `det = a . (b x c)`, `c0 = p . (b x c)`, `c1 = p . (c x a)`, `c2 = p . (a x b)`, `s = c0 + c1 + c2`. The face contains `p` when `det > 0`, `s > 0` and every `ci >= -1e-9 * s`. The first containing face wins. Weights are `max(ci, 0)` renormalized. A folded or degenerate face (`det <= 0`) never matches, and nothing is inverted, so no input raises. Before the search, if more than half of the source faces have `det < 0` (a sphere written by another tool, or mirrored), every face is reversed once and `flipped` is set; without this such a sphere matches no face at all and every target silently becomes a nearest-neighbour value (40,962 of 40,962 on the real subject with reversed faces). The order makes the result independent of thread count and platform.
- Points unresolved after `k` passes get `corners = (nearest,) * 3`, `weights = (1, 0, 0)`, `fallback = True`. Measured on sub-01: 3 and 2 of 40,962.
- `nearest_target` is one KD-tree query of the source vertices against the target sphere (lowest index on a tie).
- `resample` with `valid`: weights of invalid corners are set to 0. A target whose remaining weight is below `min_valid_weight` is NaN; otherwise the remaining weights are renormalized.
- One `SphereMap` per hemisphere per subject serves three uses. It is built when needed and not stored (0.2 s). Cluster centroids do not use it: they go to native space through the counterparts stored in `vertex_map.npz` (6.7), which needs no source file.

  | Use | How | Used by |
  |---|---|---|
  | thickness to fsaverage6 | `resample(thickness, smap, valid=...)` | `metrics/fsaverage.py` |
  | a template vertex (the D12 pole) to the native surface | `native = smap.nearest_source[vertex]`, position `pial[native]` | `render/api.py` |
  | native vertex to fsaverage6 vertex | `smap.nearest_target[index]` | `render/vertexmap.py`, `metrics/ceiling.py` |

### 6.5 `metrics/regional.py`, `metrics/ceiling.py`

```python
def regional_rows(subject_uid: str, hemi: Hemi, stats: StatsFile) -> pd.DataFrame      # rows of regional.parquet
def region_mean_thickness(rows: pd.DataFrame, parcels: Sequence[str]) -> float
    # SurfArea-weighted mean of ThickAvg over the parcels present, both hemispheres pooled; NaN if none
def occ_ctrl_ratio(rows: pd.DataFrame, regions: RegionConfig) -> tuple[float, float, float]
    # (occipital mm, control mm, ratio); occipital = regions.occipital, control = regions.control. sub-01: 0.811

@dataclass(frozen=True)
class CeilingCounts:
    n_region: int
    n_exact: int            # thickness >= max_mm (both searches truncated)
    n_near: int             # thickness >= max_mm - margin_mm; includes n_exact
    n_zero: int             # thickness == 0 inside the cortex mask
    @property
    def fraction(self) -> float          # (n_near + n_zero) / n_region; NaN when n_region == 0

def ceiling_counts(thickness: Float[np.ndarray, "n"], region: Bool[np.ndarray, "n"], *,
                   max_mm: float = 5.0, margin_mm: float = 0.1) -> CeilingCounts
def native_region(name: RegionName, *, annot: Annotation | None, smap: SphereMap | None,
                  template_hemi: TemplateHemi | None,
                  regions: RegionConfig) -> tuple[Bool[np.ndarray, "n"] | None, Literal["template", "annot", "none"]]
```

- With a template and a sphere map, a native vertex is in a region when its nearest template vertex is in that region and, when an annotation is at hand, the annotation says cortex (kind `template`; 21,484 `lh` vertices of sub-01, none of them at 4.9 mm or more). Without a template, `search` is the subject's own `occipital` parcels (16,429 `lh` vertices of sub-01) and `control` its `control` parcels (kind `annot`). With neither, `(None, "none")` and the counts are null.
- What the fraction cannot see, to be written into the docstring and `metrics.md`: the stored value is the mean of two clamped distances, so a vertex reads 5.0 only when both searches were cut short. With one cut short and the other measuring `d` it reads `(5 + d) / 2`, which is below 4.9 whenever `d < 4.8`. `n_near` is therefore a lower bound on truncated vertices; no threshold on the value recovers the rest. The complement is the pair of truncation counts from the logs (whole hemisphere, `subjects.trunc_*`): on sub-01 `lh` the log says 39 and 225 while 16 vertices are at exactly 5.0 and 28 at 4.9 or more.
- A hemisphere whose maximum thickness exceeds `max_mm + 0.01` gets the flag `thickness_above_max` (recon-all ran with another maximum); its counts are still reported.

### 6.6 `metrics/fsaverage.py`

```python
from surfcensus.reference.load import ReferenceMaps      # the dataclass lives there (6.8)

@dataclass(frozen=True)
class HemiMap:
    thickness: Float[np.ndarray, "v"]         # float32 mm; NaN = missing or outside the template cortex
    n_fallback: int
    n_missing_cortex: int                     # template cortex vertices without a value
    global_median: float
    flipped: bool                             # the sphere's winding was reversed (6.4)

def resample_hemi(thickness: Float[np.ndarray, "n"], sphere_reg: Surface, template_hemi: TemplateHemi, *,
                  annot: Annotation | None, config: MetricsConfig) -> tuple[HemiMap, SphereMap]
def global_median(thickness_fs: Float[np.ndarray, "v"], template_hemi: TemplateHemi) -> float
def write_maps(path: Path, maps: Mapping[Hemi, Float[np.ndarray, "v"]]) -> None
def read_maps(path: Path) -> dict[Hemi, Float[np.ndarray, "v"]]      # float32
def build_reference(stack: Float[np.ndarray, "s v"], template_hemi: TemplateHemi, adjacency: scipy.sparse.csr_matrix, *,
                    min_vertex_n: int, smooth_iterations: int, mad_floor_mm: float) -> ReferenceMaps
def zmap(thickness_fs: Float[np.ndarray, "v"], global_median: float, ref: ReferenceMaps) -> Float[np.ndarray, "v"]
def assign_strata(subjects: pd.DataFrame, *, min_n: int) -> pd.Series           # subject_uid -> stratum name or None
def stratum_members(subjects: pd.DataFrame, stratum: str) -> list[str]          # sorted subject_uids
```

- `resample_hemi`: `MetricError("vertex count mismatch", ...)` unless thickness, sphere and (if given) annotation agree in length; `MetricError("degenerate sphere", ...)` from `unit_sphere`, and also when more than `config.max_fallback_frac` of the targets fell back to the nearest neighbour (a folded or otherwise unusable registration; 3 of 40,962 on the real subject). `valid` = thickness finite and `> 0`, and annotation cortex when an annotation is given. Target vertices outside `template_hemi.cortex` are NaN. If fewer than `min_coverage` of the template cortex vertices get a value: `MetricError("low coverage", ...)`. In each of these cases the hemisphere has no map and a flag.
- `global_median`: the median over finite values in cortex and outside `search`, per hemisphere. NaN when fewer than 2.5% of the mesh's vertices qualify (1,024 on fsaverage6).
- `write_maps` writes `tables/fsavg/<subject_uid>.npz` with `lh_thickness`, `rh_thickness` as `<f2`, NaN for missing, through `_io.write_npz`: 164 kB per subject. A subject with one usable hemisphere stores all-NaN for the other. float16 changes thickness by at most 0.002 mm.
- Everything downstream (global median, reference, z, clusters) is computed from the stored float16 maps, never from the values in memory, so results can be reproduced from the run directory alone.
- `build_reference`: `stack` holds thickness minus each subject's global median. Per vertex: median and MAD over the finite values; a vertex with fewer than `min_vertex_n` values is NaN in both. The MAD map is then replaced `smooth_iterations` times by the mean over the vertex and its neighbours (NaN ignored, cortex only), and floored at `mad_floor_mm`.
- `zmap` = `(thickness_fs - global_median - median) / (1.4826 * mad)`; NaN wherever a term is NaN.
- Strata. A stratum is defined by a predicate on subjects that have a map: `dataset:<pseudonym>` (all subjects of that dataset), `field:<T>|fs:<major>` (all subjects with that field-strength bin and FreeSurfer major version), `all` (all subjects). Field strength is binned to the nearest of 1.5, 3, 7. `stratum_members` returns every subject that matches the predicate, and a stratum's reference is built from all of them, whether or not they are assigned to it. A stratum exists when it has at least `min_n` members. `assign_strata` gives each subject the first existing stratum in the order dataset, field and version, `all`; a subject with unknown field strength or version skips the middle level; a subject for which none exists has no stratum (D16). So in a census of three datasets of 40, 40 and 10 subjects at 3 T, the 10 are scored against a field-and-version reference built from all 90.
- The set reference includes the subject being scored (no leave-one-out), as in the census first pass.
- Memory: one hemisphere of one stratum is `n x 40,962` float32, 0.8 GB at `n = 5,000`. `pipeline` calls `build_reference` per stratum and hemisphere on a stack read from the float16 files, in blocks of vertices above 8,000 subjects.

### 6.7 `metrics/clusters.py`

```python
@dataclass(frozen=True)
class Cluster:
    vertices: Int[np.ndarray, "c"]            # template vertices, ascending
    area_mm2: float                           # sum of TemplateHemi.area over the members
    mass: float                               # sum of z over the members (the wording of metrics.md)
    excess_mass: float                        # sum of (z - threshold) over the members
    peak_z: float
    peak_vertex: int
    centroid_vertex: int

def find_clusters(z: Float[np.ndarray, "v"], mask: Bool[np.ndarray, "v"], template_hemi: TemplateHemi,
                  adjacency: scipy.sparse.csr_matrix, *, threshold: float, min_vertices: int = 1) -> list[Cluster]
def native_centroids(rows: pd.DataFrame, vmap: Mapping[str, np.ndarray],
                     adjacency: Mapping[Hemi, scipy.sparse.csr_matrix]) -> pd.DataFrame      # rows of clusters_native
```

- A cluster is a connected component (`scipy.sparse.csgraph.connected_components` on the sub-graph) of the vertices in `mask` with finite `z > threshold`, with at least `min_vertices` members. Positive side only.
- Noise makes clusters in every subject. With a reference of n subjects the null rate of z > 3 is above the Gaussian 0.00135: 0.0021 at n = 30 and 0.0030 at n = 15 with the smoothed MAD (reviewer simulation), which is 10 to 15 vertices per hemisphere of the 4,856-vertex search region. Two consequences are built in. `excess_mass` goes to 0 continuously as a cluster shrinks to the threshold, while `mass` jumps from 0 to at least 3; the composite uses `excess_mass` by default (`ScoringConfig.mass`, T25). And `min_vertex_n` is 20, not 15, with the rate at small n stated in `metrics.md`.
- `area_mm2` uses real average vertex areas where the template has them (6.3), so clusters in different parcels are ranked on one scale. `mass` and `excess_mass` are sums over vertices and not area-weighted; vertex areas on fsaverage6 range from 0.42 to 6.74 mm2, and `n_vertices` is stored so the analysis can check what that does.
- Order: area descending, then `excess_mass` descending, then lowest first vertex. Rank 1 is "the largest cluster" (T6).
- `peak_vertex`: highest z, lowest index on a tie. `centroid_vertex`: the member nearest, on the unit sphere, to the normalized z-weighted mean of the members' sphere positions (lowest index on a tie).
- Clusters are found separately in each of the five regions, per hemisphere, so a cluster that crosses the medial-inferior border is one cluster in `search` and one in each part, clipped.
- `native_centroids` maps centroids to the subject's native surface from the vertex map of its render alone, so it needs no source file and can run long after Tier C was deleted. For a centroid `v` on hemisphere `h`: the candidates are the entries of `{h}_fsavg6` equal to `v`; if there are none (180 of 40,962 fsaverage6 vertices are no native vertex's nearest on the real subject), the entries equal to a mesh neighbour of `v`. `native_vertex` is the candidate whose pial position is nearest the mean pial position of the candidates (lowest index on a tie), and `tkr_*` its pial position. No candidate (the centroid lies outside what the tiles show, which is the normal case for the control region, or the map has no counterparts): no row. This is a nearest-vertex mapping; `metrics.md` says "via the barycentric mapping", and an integer vertex is what the renderer and the review screen need (section 12).

### 6.8 `metrics/scoring.py`, `reference/load.py`

```python
def subject_metrics(vertex_summary: pd.DataFrame, clusters: pd.DataFrame | None, regional: pd.DataFrame,
                    subjects: pd.DataFrame, regions: RegionConfig, *, mass: Literal["excess", "sum_z"]) -> pd.DataFrame
    # one row per subject and reference_id ("none" when there are no clusters). Per-hemisphere metrics
    # (largest-cluster mass and area, peak z, ceiling fractions) take the larger of the two hemispheres; the
    # regional ratio pools both. Cluster metrics are 0 for a subject that has a z-map and no cluster, null for one
    # without a z-map. `mass_*` hold excess_mass or mass according to `mass`.
def normal_scores(values: pd.Series) -> pd.Series
    # scipy.stats.norm.ppf((r - 0.5) / n), r = the minimum rank of the value among the n non-NaN values
    # (1 = smallest; tied values all get the lowest rank of the tie). NaN stays NaN. n = 1 gives 0.
def robust_z(values: pd.Series) -> pd.Series
    # (x - median) / scale. scale is the first usable of 1.4826 * MAD, IQR / 1.349, SD (ddof 1), 1.
    # Usable: finite and > 1e-6 * max(|x - median|). NaN stays NaN.
def quantile_scores(values: pd.Series, grid: Sequence[float]) -> pd.Series
    # for a shipped reference: grid holds the 0th to 100th percentiles of the component in the reference
    # population; the value's percentile is interpolated in it, clipped to [0.5, 99.5], and passed to norm.ppf
def composite(metrics: pd.DataFrame, strata: pd.Series, config: ScoringConfig, *,
              norms: Mapping[str, Sequence[float]] | None = None) -> pd.DataFrame
def fallback_rank(metrics: pd.DataFrame) -> pd.DataFrame
def add_rank(scores: pd.DataFrame) -> pd.DataFrame
```

- Every component is oriented so that larger is more suspicious.
- `composite` standardises each component (`config.standardize`: `normal_score` by default, `robust_z` as the option; `quantile_scores` when `norms` are given) over a population (`config.population`): `run`, the default, is every subject of the run that has the component, for that reference; `stratum` is the subject's stratum (subjects without one: all subjects). The default is the run because ceiling fraction and the occipital-to-control ratio are raw numbers that are comparable between datasets, and standardising them inside each dataset would remove the between-dataset differences the census is about: the typical subject of a dataset in which most reconstructions are bad would score 0 (T24). The cluster masses already carry the stratification through the reference.
- Why normal scores by default: cluster mass and ceiling fraction are zero for most subjects. With a robust z the scale is then either tiny (ceiling fraction is quantised in steps of one vertex, 4.7e-5 on the real subject, so 1% scores z = 144) or, when more than half the subjects are at zero, falls through to the SD, which the outliers themselves inflate (masses 10, 50, 200, 1000 among 26 zeros score 0.05, 0.27, 1.08, 5.41). Which of the two happens depends on whether just under or just over half the subjects are non-zero. Normal scores are bounded, need no scale, and give each component equal say (T5).
- `composite` returns, per row: `std_<component>` for each component; `composite` = the maximum over the components that are not NaN; `composite_component` = the name that attained it (first in config order on a tie); `n_components` = how many were not NaN; `score_kind = "composite_z"`. A subject with fewer components than configured (no `sphere.reg`, low coverage, a degenerate sphere) has a maximum over fewer scores, which is lower on average (0.57 against 1.03 for 2 against 4 independent scores), so it gets the flag `partial_score`, and the queue shows such subjects first (7.4) instead of letting them sink.
- `fallback_rank` (D16): per component (`ceiling_frac`, `occ_ctrl_ratio`), the percentile `(r - 1) / (n - 1)` with `r` the minimum rank among the `n` subjects with a value, 0 when `n = 1`; `composite` = the larger of the two; `score_kind = "fallback_rank"`, `reference_id = "none"`. Minimum-rank ties matter here: with no ceiling vertex anywhere (the common case) every ceiling percentile is 0 and the order follows the ratio.
- `add_rank`, within one `reference_id`: rank 1 = highest composite; ties by the mean of the non-NaN `std_*` scores descending, then `subject_uid`; subjects without a composite after all others.

```python
# reference/load.py
@dataclass(frozen=True)
class ReferenceMaps:
    median: Float[np.ndarray, "v"]            # float32; NaN where there is no reference
    mad: Float[np.ndarray, "v"]               # float32; smoothed and floored
    n: Int[np.ndarray, "v"]                   # int32; subjects with a value

@dataclass(frozen=True)
class Reference:
    reference_id: str                         # "shipped-<version>", "set-<hash12>", later "rated-<hash12>"
    kind: Literal["shipped", "set", "rated"]
    mesh: str
    template_id: str | None                   # None for a shipped reference
    sphere_sha256: Mapping[Hemi, str]         # of the source ?h.sphere.reg; must equal the run template's, else it does not apply
    config_hash: str
    strata: Mapping[str, Mapping[Hemi, ReferenceMaps]]
    members: Mapping[str, tuple[str, ...]]    # stratum -> sorted subject_uids (6.6); empty for a shipped reference
    metric_norms: Mapping[str, Mapping[str, tuple[float, ...]]]    # stratum -> component -> 101 percentiles; shipped and rated only
    parcel_norms: Mapping[str, Mapping[str, tuple[float, float]]]  # stratum -> parcel -> (median, MAD) of ThickAvg minus the
                                                                   # subject's mean thickness; shipped and rated only

def shipped_reference() -> Reference | None   # None while data/reference/ has no index.json (now)
def applicable_stratum(ref: Reference, field_strength: float | None, fs_major: int | None) -> str | None
    # the first that the reference has of: "field:<T>|fs:<major>", "fs:<major>", "all"
def write_reference(directory: Path, ref: Reference) -> None
def read_reference(directory: Path) -> Reference
```

- On disk, `tables/reference/<reference_id>/` (and `surfcensus/data/reference/` for the shipped one): `index.json` (`reference_id`, `kind`, `mesh`, `template_id`, `sphere_sha256`, `config_hash`, `metric_norms`, `parcel_norms`, and `strata`: name -> `{"file": "stratum_000.npz", "n": ..., "members_sha256": ...}`); one `stratum_<nnn>.npz` per stratum with `{lh,rh}_median` `<f4`, `{lh,rh}_mad` `<f4`, `{lh,rh}_n` `<i4`; `members.parquet` (`stratum`, `subject_uid`). `architecture.md` says "parquet + JSON"; per-vertex maps are arrays, so they are `.npz` (section 12).
- `members_sha256` is the SHA-256 over the lines `subject_uid\t<SHA-256 of tables/fsavg/<uid>.npz>` of the members, sorted. The id of a set reference is `"set-"` plus the first 12 hex of the SHA-256 of `canonical_json({"config": <reference stage hash>, "members": {stratum: members_sha256}})`. So when a member's surfaces change and its map with them, the reference gets a new id although its uid is the same, and no two exports carry different numbers under one `reference_id`.
- A local run has field strength only when the user states it (`--field-strength`, 5.4); without it `applicable_stratum` falls to `fs:<major>` or `all`. The stratum used is recorded in `scores.stratum`.
- `reference/build.py` (second pass, milestone 6) reuses `build_reference` and fills `metric_norms` and `parcel_norms`.

### 6.9 `metrics/pipeline.py`

```python
def run_metrics(ctx: RunContext, *, template: Template | None, datasets: Collection[str] | None = None,
                stages: Collection[str] = METRIC_STAGES, jobs: int = 1,
                force: bool = False, retry_failed: bool = False) -> PhaseReport
def update_native(ctx: RunContext, *, template: Template | None, force: bool = False) -> PhaseReport   # stage `native`
def merge_tables(ctx: RunContext) -> PhaseReport                                                       # stage `merge`
```

| Stage | Unit | Subjects | Leases | Writes |
|---|---|---|---|---|
| `regional` (4a) | subject | `tier_a` | `lh_stats`, `rh_stats`, `aseg_stats` | `regional` part |
| `fsaverage` (4b) | subject | `tier_b`, or having thickness and annotation | `TIER_B` | `fsavg/<uid>.npz`, `vertex_summary` part |
| `reference` (4c) | `run` | those with a map | none | `reference/<id>/`, `clusters.parquet` |
| `score` | `run` | all present | none | `scores.parquet` |
| `native` | `run` | those with clusters and a done render that has counterparts | none (reads `media/<uid>/vertex_map.npz`) | `clusters_native.parquet` |
| `merge` | `run` | | none | every `tables/<table>.parquet` that has parts |

- `fsaverage`, per hemisphere: read thickness, `sphere.reg` and the annotation when present. With a template: `resample_hemi`, then the ceiling counts on the `template` regions. Without: no map. In both cases, when the hemisphere has an annotation, the ceiling counts on the `annot` regions are computed as well and stored beside the others, so subjects with and without a map can be compared on one definition. A hemisphere that raises `MetricError` gets no map and a flag (`low_coverage`, `degenerate_sphere`, `no_sphere_reg`, ...); the subject fails only when a file that is present cannot be read (`unreadable`).
- `reference` and `score` first merge the parts they need, dropping rows of subjects that are no longer present. Which path a run takes:
  1. A template exists and the shipped reference has a stratum for the subject: z-maps against it.
  2. A template exists and `assign_strata` gives the subject a stratum: z-maps within the set.
  3. Otherwise no z-map. If no subject has one, `fallback_rank`, and the report carries the note `12 subjects: too few for per-location scoring (30 needed). Ranked on raw ceiling fraction and occipital-to-control thickness.` or the template message of 6.3.
  When both 1 and 2 apply, `clusters.parquet` and `scores.parquet` hold rows for both references (their keys include `reference_id`), and the primary one is the shipped reference in a local run and the set in a census run. The second-pass reference of D9 (milestone 6) adds a third set of rows in the same way, so the paper can report detection with both.
- The `reference` unit's input signature is the SHA-256 over the sorted `(subject_uid, fsaverage input signature, dataset, field_strength, fs_major)` of the subjects with a map; the `score` unit's adds the regional signatures and the reference ids. Adding one subject therefore redoes the set-level stages and nothing per subject. Like every input signature it stays in the ledger.
- `native` reads `clusters.parquet` and, for each subject with clusters, the vertex map of its current render; its input signature covers the SHA-256 of `clusters.parquet` and the sorted `(subject_uid, render_key, template_id)` of the done renders. It takes milliseconds per subject. `Run.render()` ends with it, and `metrics --stage native` exists for the case that the reference is rebuilt after rendering. `clusters.parquet` itself is written by the `reference` stage only and never rewritten by a later phase.
- Run-level stages ignore `datasets` (section 4). `Run.metrics()` with no stage filter is the whole of phase 4 for a local run. Snakemake calls the stages separately.

`Run.zmap(subject_uid, reference_id=None) -> dict[Hemi, np.ndarray]` (added to `run.py` in this milestone) reads the subject's map and the reference (the primary one by default) and returns `zmap(...)` per hemisphere. z-maps are not stored.

### 6.10 Tables (`schemas.py`)

String columns are Arrow `string`. Floats are `float32` unless noted. Key columns come first, and the key is the sort order.

`regional.parquet`, key (`subject_uid`, `hemi`, `parcel`): `subject_uid`, `hemi`, `parcel` string; `num_vert`, `surf_area_mm2`, `gray_vol_mm3` int32; `thick_avg_mm`, `thick_std_mm` float32. All parcels of the file (34 per hemisphere).

`vertex_summary.parquet`, key (`subject_uid`, `hemi`):

| Column | Type | |
|---|---|---|
| `subject_uid`, `hemi` | string | |
| `n_native` | int32 | vertices in the thickness file |
| `thickness_max_mm` | float32 | observed maximum |
| `template_id` | string, nullable | null when no template or no map |
| `n_fallback`, `n_missing_cortex` | int32, nullable | |
| `global_median_mm` | float32, nullable | from the stored float16 map |
| `region_kind` | string | `template`, `annot` or `none`: the kind behind `ceiling_frac` |
| `search_n`, `search_n_exact`, `search_n_near`, `search_n_zero` | int32, nullable | ceiling counts, native search region of `region_kind` |
| `control_n`, `control_n_exact`, `control_n_near`, `control_n_zero` | int32, nullable | |
| `ceiling_frac`, `control_ceiling_frac` | float32, nullable | |
| `annot_search_n`, `annot_search_n_near`, `annot_search_n_zero` | int32, nullable | the same counts on the subject's own occipital parcels, whenever there is an annotation |
| `ceiling_frac_annot` | float32, nullable | |
| `flags` | string | comma-separated, sorted: `no_template`, `no_sphere_reg`, `no_thickness`, `no_annot`, `low_coverage`, `degenerate_sphere`, `sphere_flipped`, `thickness_above_max` |

`clusters.parquet`, key (`subject_uid`, `reference_id`, `hemi`, `region`, `rank`), the `clusters_kept` largest per group:

| Column | Type | |
|---|---|---|
| `subject_uid`, `reference_id`, `hemi`, `region` | string | |
| `rank` | int16 | 1 = largest area |
| `n_vertices` | int32 | |
| `area_mm2`, `mass`, `excess_mass`, `peak_z` | float32 | |
| `peak_vertex`, `centroid_vertex` | int32 | template vertices |

`clusters_native.parquet`, same key, only the clusters that have a native position: `native_vertex` int32; `tkr_x`, `tkr_y`, `tkr_z` float32 (native pial position of the centroid); `template_id` string.

`scores.parquet`, key (`subject_uid`, `reference_id`): one row per present subject and reference; `reference_id` is `"none"` for `fallback_rank` rows and for subjects without a z-map.

| Column | Type | |
|---|---|---|
| `subject_uid`, `reference_id` | string | |
| `primary` | bool | exactly one row per subject is primary; `Run.scores()`, the queue and `select` read these |
| `dataset` | string | the pseudonym |
| `score_kind` | string | `composite_z` or `fallback_rank` |
| `stratum`, `template_id` | string, nullable | |
| `n_reference` | int32, nullable | members of the stratum's reference |
| `composite` | float32, nullable | |
| `composite_component` | string, nullable | |
| `n_components` | int8 | components that were not NaN |
| `rank` | int32 | 1 = looks worst, within the `reference_id` |
| `mass_kind` | string | `excess` or `sum_z` |
| `mass_search`, `mass_medial`, `mass_inferior`, `mass_lateral`, `mass_control` | float32, nullable | of the largest cluster, larger hemisphere |
| `area_search_mm2`, `peak_z_search` | float32, nullable | |
| `ceiling_frac`, `control_ceiling_frac`, `ceiling_exact_frac`, `zero_frac`, `ceiling_frac_annot` | float32, nullable | larger hemisphere |
| `ceiling_region` | string | `template`, `annot` or `none` |
| `occ_thick_mm`, `ctrl_thick_mm`, `occ_ctrl_ratio` | float32, nullable | from `aparc.stats` |
| `std_mass_medial`, `std_mass_inferior`, `std_ceiling_frac`, `std_occ_ctrl_ratio` | float32, nullable | the standardised components |
| `holes_lh`, `holes_rh` | int32, nullable | `aseg.stats`, else the log |
| `trunc_frac` | float32, nullable | (sum of the four truncation counts) / (2 x native vertices) |
| `flags` | string | union of the hemispheres' flags, plus `no_stats`, `partial_score` |
| `score_config_hash` | string | |

CLI:

```
surfcensus metrics --run-dir DIR [--stage regional|fsaverage|reference|score|native|merge]... [--dataset ID]...
                   [--jobs J] [--fsaverage DIR] [--retemplate] [--force] [--retry-failed]
```

One closing line with counts, the reference used, and the notes of the report. Without `--stage` the four stages of `METRIC_STAGES` run (only the two per-subject ones when `--dataset` is given).

### 6.11 Render changes: `vertex_map.npz`, the fsaverage pole, schema 3

The working tree already uses schema 2 (the vertex map selected by the clickable region, `{h}_visible`). This milestone makes it 3.

```python
RENDER_SCHEMA_VERSION = 3

class TemplateInfo(_Frozen):
    mesh: str
    template_id: str
    sphere_sha256: dict[Literal["lh", "rh"], str]

class FsaveragePole(_Frozen):                 # decision D12
    template_vertex: int
    native_vertex: int                        # smap.nearest_source[template_vertex]
    tkr: Vec3                                 # its native pial position
    distance_mm: float                        # to the geometric pole of the same hemisphere

class RenderInfo(_Frozen):                    # two new fields
    template: TemplateInfo | None = None
    fsaverage_poles: dict[Literal["lh", "rh"], FsaveragePole] | None = None

def render_subject(files, out_dir, config=None, *, template: Template | None = None,
                   video: bool = True, force: bool = False) -> RenderInfo
def three_plane(subject_dir, out, config=None, *, template: Template | Path | Literal["auto"] | None = "auto",
                video: bool = True, force: bool = False) -> RenderInfo
# render/vertexmap.py
def fill_fsaverage(vmap: Mapping[str, np.ndarray],
                   nearest_target: Mapping[Hemi, Int[np.ndarray, "n"]]) -> dict[str, np.ndarray]
```

- With a template, for each hemisphere that has a `sphere_reg` file whose vertex count equals the pial surface's: `{h}_fsavg6 = nearest_target[index]`, `inputs` gains `{h}_sphere_reg`, and `fsaverage_poles[h]` is filled. `vertex_map.fsaverage` becomes the mesh name and `template` is set when at least one hemisphere was filled. The key stays `{h}_fsavg6` whatever the mesh.
- A hemisphere without `sphere_reg`, or with one of another vertex count or a degenerate sphere (6.6), keeps -1 and gets no pole entry. That is not a render failure.
- Without a template nothing changes from the working tree except the schema number.
- The tiles do not depend on the template. A directory whose tiles are current (schema, config hash, `video`, the inputs of the `TIER_C` roles, every listed file present) but whose `template` or sphere inputs differ is brought up to date by rewriting `vertex_map.npz` and `render.json` only, under the directory lock of milestone 1, with `render.json` removed first and written last. 0.2 s instead of 9 s.
- Schema bump: `config_hash` includes the schema number, so every schema 2 render is out of date and is rendered again by the next `render` or `review`. The only such renders are test and trial outputs; no rating exists. For later bumps: `tile_hash` and `render_key` (5.9, 5.1) do not include the schema number, so a rating stays attached to its tiles across a bump, and `serve` refuses to start on renders that are out of date instead of serving an old `render.json` to a new page (7.4).
- `three_plane(template=...)`: a `Template` is used as given; a `Path` is an fsaverage directory, built with the default metrics config (`TemplateNotFound` or `TemplateError` raised to the caller, since it was asked for); `None` renders without counterparts; `"auto"` calls `find_template` with `adjacent = subject_dir/../fsaverage` and no cache, and on `TemplateNotFound` or `TemplateError` renders without counterparts. `render-subject` gains `--fsaverage DIR` and `--no-template` for the two explicit forms, and with `"auto"` prints one line when none was used: `no fsaverage template found: click marks will not map to fsaverage`. The template that was used is recorded in `render.json`, so a render says what it was made from even when it was found through the environment.
- `Run.render()` passes `Run.template()`. The `renders` row stores `template_id`; its `input_signature` covers the two `sphere_reg` roles and the template id, while `tile_hash` and `render_key` do not.

---

## 7. Milestone 4: rater app, `review`, `export`

Result: `surfcensus review PATH` end to end in local mode, `surfcensus serve --run-dir DIR` on an existing run, `surfcensus finalize`, `surfcensus calibration freeze`, `surfcensus export`. Team mode (tokens, several raters at once, adjudication, dashboard) is milestone 5: its tables and the queue interface exist from the start so that no migration is needed; its code does not.

New files: `protocol.py`, `calibration.py`, `app/{__init__,schemas,models,db,marks,queue,events,finalize,server}.py`, `app/static/`, `app/templates/`, `export/{__init__,tables}.py`, `review.py`. Changed: `cli.py` (`review`, `serve`, `finalize`, `calibration`, `export`), `pyproject.toml` (a `browser` dependency group with `playwright`; the `browser` marker; `addopts = "-m 'not network and not browser'"`), `justfile` (`test-browser`).

The app opens the catalog with `PublicCatalog.open(run_dir)`: a read-only connection, because a renderer may be recording rows. It cannot reach an accession, a subject path or an input signature, and a team server never holds the dataset key. Only modules under `app/` import `sqlmodel`; `export/`, `reference/`, `calibration.py` and `review.py` read `app.sqlite` with the standard-library `sqlite3` (through `pandas.read_sql_query`) by the table and column names of 7.2, so `surfcensus export` works on a core install.

### 7.1 `protocol.py`

```python
class ScaleLevel(_Frozen):
    value: int
    label: str
    help: str
class Criterion(_Frozen):
    id: Literal["A", "B", "C"]
    label: str
    question: str
    keys: tuple[str, ...]
    mark_key: str
class Protocol(_Frozen):
    version: str
    scale: tuple[ScaleLevel, ...]
    criteria: tuple[Criterion, ...]
    other_mark_key: str
    require_marks_for_positive: bool
    unratable_reasons: tuple[str, ...]
    def positive_values(self) -> frozenset[int]          # every scale value above the lowest

def install_protocol(run_dir: Path) -> Path                    # copies data/protocol.yaml when the run has none
def load_protocol(run_dir: Path) -> tuple[Protocol, str]       # (protocol, sha256 of the file's bytes)
def marks_required(protocol: Protocol, app: AppConfig) -> bool # app.require_marks if not None, else the protocol's flag
```

Validation beyond the field types, each failure naming the entry: scale values are `0 .. n-1`; each criterion has as many keys as scale levels; keys and mark keys are unique across the file and collide with no fixed key of the keyboard map of `rater_app.md`; criterion ids are unique; the three `RenderError` reasons are in `unratable_reasons`. The file is read at server start. A local run is created with `app.require_marks = False` (`--require-marks` sets and stores true), so the shipped `true` never surprises a local user, and the YAML file is never rewritten by a program. The model classes are written in the first step of the milestone, because `app/schemas.py` embeds `Protocol`.

### 7.2 `app/models.py`, `app/db.py` (SQLModel, `app.sqlite`)

The tables of `rater_app.md`; additions are marked `+`. Timestamps are UTC ISO 8601 text.

```
raters        Rater         rater_id PK, name UNIQUE, token_hash NULL, role ('rater'|'adjudicator'|'lead'),
                            active + (default 1), created_at
assignments   Assignment    assignment_id PK, rater_id FK, subject_uid, kind ('calibration'|'main'|'repeat'|'adjudication'),
                            seq + (per rater, from 1), render_key +, assigned_at, completed_at NULL,
                            consensus_served_at + NULL, UNIQUE (rater_id, seq)
submissions + Submission    submission_id PK, assignment_id FK, revision (from 1), current (0|1), unratable_reason NULL,
                            other (0|1), note, protocol_version, protocol_sha256, calibration_sha256 NULL, app_version,
                            client_ts, server_ts, ms_on_subject, UNIQUE (assignment_id, revision)
ratings       Rating        rating_id PK, submission_id FK +, assignment_id FK, criterion ('A'|'B'|'C'), value,
                            UNIQUE (submission_id, criterion)
marks         Mark          mark_id PK, submission_id FK +, assignment_id FK, class ('A'|'B'|'C'|'X'), panel, slice +,
                            frame NULL, mode + NULL ('sync'|'seq'), px, py, vox_i, vox_j, vox_k, tkr_x, tkr_y, tkr_z,
                            ras_x, ras_y, ras_z, hemi NULL, pial_vertex NULL, pial_dist_mm NULL, white_vertex NULL,
                            white_dist_mm NULL, fsavg6_vertex NULL, vertex_surface + NULL ('pial'|'white'),
                            box_dist_mm +, created_ts, deleted_ts NULL
events        Event         event_id PK +, assignment_id FK NULL (NULL: an event of the run, e.g. config_change), ts, type,
                            payload_json
final_labels  FinalLabel    subject_uid, criterion ('A'|'B'|'C'|'unratable'), value NULL, method ('consensus'|'adjudicated'|'single'),
                            PRIMARY KEY (subject_uid, criterion)
final_marks   FinalMark     final_mark_id PK +, subject_uid, class, tkr_x, tkr_y, tkr_z, hemi NULL, pial_vertex NULL,
                            fsavg6_vertex NULL, source_mark_ids (JSON array)
app_meta +    AppMeta       key PK, value            -- schema_version, run_id, mode
```

```python
APP_SCHEMA_VERSION = 1
MIGRATIONS: tuple[str, ...] = ()      # SQL script k takes schema k + 1 to k + 2; applied in order, each in one transaction
def open_app_db(run_dir: Path, *, run_id: str, mode: Literal["local", "team"]) -> Engine
```

- `open_app_db` creates the tables with `SQLModel.metadata.create_all` only for a new file. An existing file of an older schema is brought up by `MIGRATIONS` (`create_all` does not alter tables, so a run with ratings is never left to it); a newer schema, another `run_id`, or a `mode` other than `app_meta.mode` is refused (`ConfigError("invalid", ...)`, exit 2). WAL in team mode.
- The `class` column: the Python field is `mark_class`, declared with `sa_column=Column("class", String, nullable=False)` (checked with sqlmodel 0.0.49).
- `submissions` is the one structural addition. One submission per press of `Enter`. A revision (`Backspace`, then `Enter`) is a new submission with `revision + 1`; `current` moves to it, and the marks of the superseded submission get `deleted_ts`. Nothing is overwritten. A view `ratings_flat` reproduces the `ratings` columns of `rater_app.md` for current submissions.
- `assignments.render_key` is `renders.render_key` of the render that was shown (5.1, T19): keyed, built from the tile settings and the content of the tile inputs. It does not change when a file is touched or the data moves, when the render schema is bumped, when the video or the template changes, or when the vertex map is rewritten; it changes when a surface or the T1 changes or a tile setting does.
- `marks.slice` is the position in the panel's slice list, which is what the geometry needs. `frame` and `mode` are the playback context of the click, kept for the behaviour analysis. `vox_*` are continuous voxel coordinates.
- `final_labels` with criterion `unratable`: `value` is the index of the reason in the protocol. A selected subject whose render failed gets such a row (method `single`, the render's reason) from `finalize.sync_unratable`, which runs at server start and in `finalize`; so every selected subject can end with a final label, and the unratable reasons of the protocol that come from the renderer are recorded as labels.
- Who writes final labels. With one rater in the database (the normal local case), each submission rewrites the subject's `final_labels` (method `single`) and `final_marks`, so review and export read the same two tables in every mode. As soon as a second rater exists, the rows of method `single` that come from ratings are deleted and nothing is written per submission; final labels then come from `finalize` (7.7) and, in milestone 5, adjudication.

### 7.3 `app/marks.py`

Pure functions on a `RenderInfo` and a vertex map; no database, no web framework. The two models live in `app/schemas.py`, which the first step of the milestone writes.

```python
class MarkIn(_Frozen):                        # as posted by the client; populate_by_name
    client_id: str
    mark_class: Literal["A", "B", "C", "X"] = Field(alias="class")
    panel: PanelName
    slice: int
    px: float
    py: float
    frame: int | None = None
    mode: Literal["sync", "seq"] | None = None
    ts: str | None = None                     # client time of the click

class ResolvedMark(_Frozen):
    vox: Vec3
    tkr: Vec3
    ras: Vec3
    hemi: Literal["lh", "rh"] | None          # the one hemisphere both vertices belong to
    pial_vertex: int | None
    pial_dist_mm: float | None
    white_vertex: int | None
    white_dist_mm: float | None
    fsavg6_vertex: int | None
    vertex_surface: Literal["pial", "white"] | None    # whose vertex fsavg6_vertex is the counterpart of
    box_dist_mm: float                        # distance from the mark to the view box; 0 inside

class MarkError(ValueError):
    def __init__(self, field: str, message: str) -> None      # field in "panel", "slice", "px", "py", "frame"

class VertexIndex:
    """KD-trees over the pial and the white positions of one vertex_map.npz, per hemisphere."""
    def __init__(self, vmap: Mapping[str, np.ndarray]) -> None
    def nearest(self, surface: Literal["pial", "white"], tkr: Vec3, max_dist_mm: float, *,
                hemi: Hemi | None = None) -> tuple[Hemi, int, float, int] | None
        # (hemi, native vertex, distance, fsaverage6 vertex or -1). hemi=None searches both; the nearer wins, lh on
        # an exact tie.
    def within(self, surface: Literal["pial", "white"], tkr: Vec3, radius_mm: float, *,
               hemi: Hemi | None = None) -> list[tuple[Hemi, int, float, int]]     # ascending distance

def validate_mark(info: RenderInfo, mark: MarkIn) -> None
def resolve_mark(info: RenderInfo, index: VertexIndex | None, mark: MarkIn, *, max_dist_mm: float = 5.0) -> ResolvedMark
def marked_vertices(index: VertexIndex, points: Sequence[Vec3], tolerance_mm: float, *,
                    surface: Literal["pial", "white"] = "pial") -> dict[Hemi, Int[np.ndarray, "k"]]
    # native vertex indices within tolerance_mm of any of the points; for error maps and vertex-level positives
```

- Validation against `render.json`: `0 <= slice < len(panel.slices)`; `0 <= px <= width`, `0 <= py <= height`; if `frame` and `mode` are both given, `modes[mode].frames[frame]` shows `slice` in that panel.
- Conversion: `vox = geometry.pixel_to_voxel(panel, slice, px, py)`; `tkr = geometry.apply_affine(vox2ras_tkr, vox)`; `ras = geometry.apply_affine(vox2ras, vox)`. The server computes all of it; a voxel sent by a client is not trusted.
- One hemisphere per mark. The mark's own surface is the white surface for class C and the pial surface for A, B and X. The nearest vertex of that surface within `max_dist_mm`, over both hemispheres, sets `hemi`; the other surface is then searched in that hemisphere only. If the mark's own surface has no vertex in range, the other surface is searched over both hemispheres and sets `hemi`. So `pial_vertex` and `white_vertex` always index the same hemisphere's surfaces. (Searching the two surfaces independently, as the draft did, puts the white vertex in the other hemisphere for 14% of clicks on medial pial vertices within 6 mm of the midline, which is where class A errors are.)
- `fsavg6_vertex` is the counterpart of the vertex on the mark's own surface, else of the other one; `vertex_surface` says which; null when the counterpart is -1 or no vertex is in range. A mark with no vertex in range is kept as a volumetric point.
- The snap radius and the vertex map. The vertex map holds every vertex within `render.vertex_map_margin_mm` of what the tiles show, so any click on a tile resolves to the same vertex as on the full surface as long as `max_dist_mm` does not exceed that margin. `create_app` checks `app.mark_max_dist_mm <= render.vertex_map_margin_mm` and refuses to start otherwise. `box_dist_mm` is descriptive: it lets the analysis restrict marks to the swept box, or not (T30).
- A single nearest vertex is not a stable summary of a click. Where two banks of a sulcus lie within a millimetre, the nearest vertex flips to the other bank for a sub-millimetre change of the click, and the other bank is far away on the surface: shifting clicks by 0.5 mm moved the resolved fsaverage6 vertex by more than 5 mm on the template for 7.4% of them, and by more than 10 mm for 2.6% (reviewer measurement on the real subject). So the stored vertex is for display and for a first look. Vertex-level positives and error maps are computed from the stored `tkr` point and the retained vertex map by `marked_vertices`: a native vertex is positive when its position is within the tolerance of a final mark in native space, and is then carried to fsaverage6 through its counterpart. `app.mark_near_mm` (2 mm) is the radius used for the error-map display. Nothing extra is stored per mark, because point and vertex map are both kept.
- `index=None` (no vertex map): coordinates only.
- The server keeps `VertexIndex` objects in an LRU cache of 16 subjects, keyed by `(subject_uid, render_key, template_id)`.

### 7.4 `app/queue.py`

```python
@dataclass(frozen=True)
class QueuePlan:
    mode: Literal["local", "team"]
    order: tuple[str, ...]                    # selected subjects in queue order (see below)
    render_keys: Mapping[str, str]            # subjects whose render is ready -> render_key
    failed: Mapping[str, str]                 # selected subjects whose render failed -> reason
    pending: int                              # selected subjects with no render row yet
    outdated: int                             # selected subjects whose render was made with another config hash
    calibration: tuple[str, ...]
    raters_per_subject: int
    repeat_rate: float
    repeat_gap: int
    seed: int

@dataclass(frozen=True)
class Next:
    status: Literal["assignment", "rendering", "done"]
    assignment: Assignment | None = None
    prefetch: tuple[tuple[str, str], ...] = ()     # the next two ready (subject_uid, render_key)

def build_plan(public: PublicCatalog, scores: pd.DataFrame | None, app: AppConfig,
               mode: Literal["local", "team"], *, render_config_hash: str,
               calibration: CalibrationSet | None = None) -> QueuePlan
def next_assignment(session: Session, rater: Rater, plan: QueuePlan) -> Next
def previous_assignment(session: Session, rater: Rater) -> Assignment | None     # the rater's most recent completed one
def has_work_left(session: Session, rater: Rater, plan: QueuePlan) -> bool      # an open assignment, or next would give one
```

- A render is ready when its row is `done` and its `config_hash` equals `render_config_hash`, the hash of the running version's config (which includes the schema number). A `done` row with another hash counts as `outdated`.
- Selected subjects are the rows of `selection` (a local run: every present subject, reason `all`). Local order without `blind`: first the subjects flagged `partial_score` (they could not be fully scored and would otherwise sink, 6.8), by `subject_uid`; then by `scores.rank` of the primary rows ascending (worst first); then subjects without a score, by `subject_uid`. With `blind`: `random.Random(seed).shuffle` of the sorted uids. Team: sorted uids (the order is not used).
- `next_assignment` is idempotent: an open assignment is returned again, so a reload never skips a subject.
- Local: walk `order`. A subject is done for a rater when the rater has a completed `main` assignment whose `render_key` equals the subject's current one. The first subject that is not done and is ready gets an assignment. Nothing ready and `pending + outdated > 0`: `rendering`. Nothing left: `done`. Subjects whose render failed are never offered; they have an unratable final label (7.2) and appear in review and export with their reason. Several local raters each walk the same order on their own.
- Repeats, in both modes, when `repeat_rate > 0`: with that probability the next assignment is a subject this rater completed at least `repeat_gap` assignments ago and has not yet repeated. It is stored with kind `repeat`.
- Team (milestone 5, same function): calibration subjects first, in a per-rater seeded order. Then a repeat as above, or a subject the rater has not seen, drawn uniformly among those with the fewest assignments (completed or open) that still need raters. Every draw uses `random.Random(f"{seed}:{rater_id}:{seq}")`, so the queue is reproducible from the database.
- `build_plan` is called per request in local mode and cached for 30 s in team mode.

### 7.5 `app/server.py`, `app/schemas.py`, `app/events.py`

```python
def create_app(run_dir: Path, *, mode: Literal["local", "team"] = "local", rater: str | None = None,
               allowed_hosts: Collection[str] | None = None) -> FastAPI

@dataclass
class ServerHandle:
    url: str
    port: int
    def stop(self, timeout: float = 5.0) -> None

def start_server(run_dir: Path, *, host: str = "127.0.0.1", port: int = 8765,
                 mode: Literal["local", "team"] = "local", rater: str | None = None) -> ServerHandle
    # uvicorn.Server in a thread, stopped through should_exit; returns once the port accepts connections
def serve(run_dir: Path, *, host: str = "127.0.0.1", port: int = 8765, mode: Literal["local", "team"] = "local",
          rater: str | None = None, open_browser: bool = True) -> None
    # start_server, print the URL, open the browser, wait for Ctrl-C or SIGTERM in the main thread, stop
```

`app/schemas.py` holds the request and response shapes below as pydantic models (`MarkIn`, `ResolvedMark`, `SessionOut`, `AssignmentOut`, `NextOut`, `RatingIn`, `RatingOut`, `EventsIn`, `ReviewOut`, `SubjectOut`, `StatsOut`, `ErrorsOut`); the frontend and the server are both written against it. It imports `protocol.py` for the `Protocol` model and `render/model.py`, and nothing from the rest of `app/`.

Local mode: one rater per server process, named by `rater` (default `app.rater_name`), role `lead`, created on first use. `serve` refuses a host other than `127.0.0.1` or `localhost`. Requests whose `Host` header names another host than those in `allowed_hosts` (default in local mode: `127.0.0.1` and `localhost`; the port is ignored) get 403 (DNS rebinding). If the port is taken, `start_server` tries the next 20; `port=0` asks the system for a free one; the handle carries the port it got. `serve` exits 2 before starting when `plan.outdated > 0` and no renderer is attached: `N renders are out of date; run surfcensus render --run-dir DIR`. Team mode: 7.6 and 8.7.

Unblinded routes are `/review`, `/api/review*`, `/api/stats`, `/api/export` and, in milestone 5, `/api/errormap*`, `/api/adjudication*` and `/dashboard`. One server-side rule covers all of them, for every role: 403 while `has_work_left` is true for the requesting rater. In team mode the rule always applies. In local mode it applies when `app.blind` is true or the database has more than one rater; otherwise the review page stays open at any time, as `rater_app.md` describes for a researcher looking at their own data in score order.

| Route | Who | Returns |
|---|---|---|
| `GET /` | rater | the rating page |
| `GET /review` | unblinded (rule above); team: adjudicator and lead | the review page |
| `GET /static/{path}` | | frontend files from the wheel |
| `GET /media/{subject_uid}/{path}` | rater | a file of that subject's render. `subject_uid` must match `[0-9a-f]{16}` and have a ready render; `path` must match `poster\.webp`, `anim\.mp4` or `tiles/(sagittal\|coronal\|axial)_\d{3,}(_c)?\.webp`; anything else is 404. `render.json` and `vertex_map.npz` are never served as files. `Cache-Control: private, max-age=31536000, immutable`; the client appends `?v=<render_key>`. Not mounted when `media_base_url` is set. |
| `GET /healthz` | | `{"ok": true, "version": "0.0.1"}` |
| `GET /api/session` | rater | below |
| `GET /api/next` | rater | below |
| `GET /api/back` | rater | the rater's most recent completed assignment, in the envelope of `next` with `previous` filled; 404 when there is none; 409 when it is a calibration assignment whose consensus has been served |
| `POST /api/rating` | rater | below |
| `POST /api/events` | rater | `{"stored": n}` |
| `GET /api/review?sort=&order=&filter=&offset=&limit=` | unblinded | `{"total": n, "columns": [...], "rows": [...]}`: per subject the primary `scores.parquet` columns, `fs_version`, render status and reason, `final` labels, `n_ratings`, `n_marks`, and in a local run `subject_label`. `filter` is a comma-separated list of `column:value`. |
| `GET /api/review/{subject_uid}` | unblinded | `{"media_base", "render_key", "render", "scores", "ratings": [...], "marks": [...], "clusters": [{"reference_id", "hemi", "region", "rank", "tkr": [x, y, z], "area_mm2", "peak_z"}]}`; clusters are those with a row in `clusters_native`; the page draws marks and centroids through `vox2pix` |
| `GET /api/stats` | unblinded; team: lead | `StatsOut`, below |
| `GET /api/export` | unblinded; team: lead | the `ratings` CSV of 7.8, produced by `export.tables.ratings_frame` |

Errors are `{"errors": [{"field": "...", "message": "..."}]}` with status 422 (invalid submission; nothing is stored), 404 (unknown assignment or subject), 403 (not this rater's, not allowed in this mode, or an unblinded route while the rater has work left), 409 (the protocol version or the subject's render changed since the page loaded, or a locked calibration assignment; the client fetches `next` again), 401 (team mode, no session).

`GET /api/session`

```json
{"mode": "local", "rater": {"id": 1, "name": "local", "role": "lead"}, "app_version": "0.0.1",
 "protocol": {"version": "0.1-draft", "scale": [], "criteria": [], "other_mark_key": "x",
              "require_marks_for_positive": true, "unratable_reasons": []},
 "protocol_sha256": "9c1f...", "require_marks": false,
 "settings": {"mark_show_mm": 3.0, "blind": false},
 "examples": []}
```

`examples` feeds the `?` help screen: a list of `{"criterion": "A", "value": 2, "panel": "sagittal", "slice": 31, "media_base": "...", "render_key": "...", "caption": "..."}` taken from `calibration.yaml` (7.7). It is empty when the run has no calibration set, and empty for a rater who has not finished calibration, so that nobody sees a calibration subject's consensus before rating it.

`GET /api/next`

```json
{"status": "assignment",
 "assignment": {"assignment_id": 12, "kind": "main", "subject_uid": "3f9a1c07d2e4b8a1",
                "media_base": "/media/3f9a1c07d2e4b8a1/", "render_key": "5d0c1f2a9b7e4c31",
                "render": {}, "has_vertex_map": true, "previous": null},
 "prefetch": [{"subject_uid": "9b2e41c07d2e4b8a", "media_base": "/media/9b2e41c07d2e4b8a/", "render_key": "0a1b2c3d4e5f6a7b"}],
 "progress": {"done_today": 3, "done_total": 40, "remaining": 110, "rendered": 120, "n_subjects": 150}}
```

- `render` is the subject's `RenderInfo` without `inputs` (the file sizes and digests there match the public bucket listing, and the page has no use for them); tile URLs are `media_base` plus its `tiles` patterns, with `?v=<render_key>`. Nothing else about the subject is in the object.
- `kind` is `calibration` or `main`, nothing else. A repeat is reported as `main` with `previous: null` and counts as an ordinary subject in `progress`, so the payload of a repeat differs from that of a new subject only in ids (`rater_app.md`: repeats look identical to new subjects).
- `previous` is null for a new subject and otherwise `{"values": {"A": 0, "B": 2, "C": 0}, "unratable": null, "other": false, "note": "", "marks": [<MarkIn fields plus mark_id and the resolved fields>]}`.
- The other two states are `{"status": "rendering", "retry_after_s": 2, "progress": {...}}` and `{"status": "done", "progress": {...}}`.

`POST /api/rating`

```json
{"assignment_id": 12, "protocol_version": "0.1-draft",
 "values": {"A": 2, "B": 0, "C": 0}, "unratable": null, "other": false, "note": "",
 "marks": [{"client_id": "m1", "class": "A", "panel": "sagittal", "slice": 31, "px": 101.5, "py": 88.25,
            "frame": 57, "mode": "sync", "ts": "2026-11-12T14:03:20.004Z"}],
 "client_ts": "2026-11-12T14:03:22.120Z", "ms_on_subject": 14200}
```

Rules, in this order, all-or-nothing in one transaction:

1. The assignment belongs to the rater (403). A completed assignment may be submitted again only if it is the rater's most recent one (the `Backspace` revision; `GET /api/back` never shows another) and, for a calibration assignment, only while its consensus has not been served. `protocol_version` equals the loaded protocol's and the assignment's `render_key` is still current (409).
2. Either `unratable` is one of the protocol's reasons (`values` may then be empty and marks are ignored), or `values` has exactly the protocol's criterion ids with values on the scale.
3. Every mark passes `validate_mark`.
4. If marks are required (`marks_required`), each criterion with a positive value has at least one mark of its class. Message: `criterion A is rated major but has no mark of class A`.
5. `other: true` needs a non-empty note.
6. Marks are resolved and stored. With a single rater `final_labels` and `final_marks` are rewritten (7.2).

Response: `{"saved": {"assignment_id": 12, "submission_id": 30, "revision": 1, "marks": [{"client_id": "m1", "mark_id": 8, "hemi": "lh", "pial_vertex": 101234, "pial_dist_mm": 0.8, "white_vertex": 101234, "white_dist_mm": 2.9, "fsavg6_vertex": 12950, "vertex_surface": "pial", "box_dist_mm": 0.0}]}, "next": <as GET /api/next>}`.

`POST /api/events`: `{"assignment_id": 12, "events": [{"ts": "...", "type": "pause", "payload": {"frame": 57}}]}`. At most 500 events per call, each payload at most 16 kB (422 otherwise). Types: `load`, `play`, `pause`, `frames`, `mode`, `speed`, `zoom`, `contours`, `marks_hidden`, `mark_add`, `mark_delete`, `mark_undo`, `help`, `back`. Frame viewing is run-length encoded, not logged per frame: `{"type": "frames", "payload": {"mode": "sync", "visits": [[frame, ms], ...]}}`. Unknown types are stored as given. The client flushes every 5 s, on submit, and with `navigator.sendBeacon` on unload.

`GET /api/stats` (`StatsOut`): `{"raters": [{"rater": "local", "n_completed": 40, "median_s": 14.2, "by_day": [{"day": "2026-11-12", "n": 40}]}], "agreement": null, "drift": null}`. The milestone 4 pages do not use it; milestone 5 fills `agreement` and `drift` and adds the dashboard page (8.7).

### 7.6 Local mode and team mode

| | Local (`review`; `serve` on a local run) | Team (`serve --mode team`, milestone 5) |
|---|---|---|
| Bind | `127.0.0.1`, free port from 8765 | `--host`, `--port`, behind HTTPS |
| Login | none; the rater named by `--rater`, one per server process | `/r/<token>`, cookie |
| Queue | by score, or seeded shuffle with `--blind` | calibration, fewest ratings first, repeats |
| Raters per subject | each local rater rates every subject | `raters_per_subject` |
| Marks | optional unless `--require-marks` | per protocol (required) |
| Final labels | one rater: at each submission, method `single`. Several: `surfcensus finalize` | `finalize` and adjudication |
| Unblinded screens | open, unless `--blind` or several raters: then only once the rater has nothing left | adjudicator and lead, only once they have nothing left; pseudonyms only |
| Rater names in export | as given | `R-01`, `R-02`, ... in a census run |
| Media | from the run directory | run directory or `media_base_url` |

`serve --mode` defaults to `local` for a local run and `team` for a census run.

### 7.7 `review.py`, `app/finalize.py`, `calibration.py`; `surfcensus review`, `serve`, `finalize`, `calibration freeze`

```python
def review(path: Path, *, run_dir: Path | None = None, subjects: Sequence[str] | None = None,
           rater: str | None = None, blind: bool | None = None, require_marks: bool | None = None,
           field_strength: float | None = None, jobs: int | None = None, fsaverage: Path | None = None,
           video: bool | None = None, port: int = 8765, open_browser: bool = True, serve: bool = True,
           serve_after: int | None = None, relocate: bool = False, force: bool = False) -> Run
```

```
surfcensus review PATH [--run-dir DIR] [--subjects GLOB]... [--rater NAME] [--blind / --no-blind]
                       [--require-marks / --no-require-marks] [--field-strength T] [--jobs J] [--fsaverage DIR]
                       [--video / --no-video] [--port N] [--no-browser] [--no-serve] [--serve-after N] [--relocate]
                       [--force]
surfcensus serve    --run-dir DIR [--host H] [--port N] [--mode local|team] [--rater NAME] [--no-browser]
surfcensus finalize --run-dir DIR
surfcensus calibration freeze --run-dir DIR [--subject UID]...
```

Every option of `review` that maps to a stored value is tri-state (5.4): `None`, or the flag not given, keeps what the run has; the defaults (`blind` false, marks not required, no video) apply only when the run is created. `--rater` is not stored: it names the rater of this server process, and without it the rater is `app.rater_name`. `--force` is needed to change `blind` once a rating exists.

What `review` does, and what it prints (stderr, one block per step):

1. Run directory: `--run-dir`, else `PATH/../surfcensus-run` if its parent is writable, else `./surfcensus-run`. If that default lies inside the subjects directory (a single subject as `PATH`), `<subjects directory>/../surfcensus-run`. `Run directory: /data/surfcensus-run (new)`.
2. Inventory. `Found 48 subjects (fMRIPrep layout, FreeSurfer 7.3.2). 2 cannot be rendered: sub-07 (no surf/lh.pial), sub-31 (no mri/T1.mgz).`
3. Metrics, all stages, with a progress bar; then one line naming the reference used, or the D16 or template message.
4. Selection: every present subject, reason `all`, `inclusion_prob` 1, seed `queue_seed`. Incomplete subjects are in it; they get their `missing surface` render row in the next step and an unratable final label, and are never offered for rating.
5. Render in queue order (`--jobs`, default `min(8, cpu count)`), no video unless the run has it switched on. With `--serve-after N` only the first N are rendered here. The `native` stage follows.
6. Install the protocol, `start_server`, print `Rating app: http://127.0.0.1:8765/ (Ctrl-C to stop)`, open the browser unless `--no-browser`. With `--serve-after`, a background thread with its own `Run` renders the rest in queue order, and the app answers `rendering` when nothing is ready.
7. On Ctrl-C: stop the server through its handle, then the renderer; print `31 rated, 15 to go. Run the same command to continue. Export: surfcensus export --run-dir /data/surfcensus-run`.

Running the command again, with or without the flags, does the same steps with everything unchanged skipped, so it reopens the app within seconds. `--no-serve` stops after step 5 (preparing on a cluster, and the end-to-end test). `review` exits 1 before serving only when no subject can be rendered.

```python
# app/finalize.py
@dataclass(frozen=True)
class FinalizeReport:
    consensus: int                      # subjects given a consensus label
    single: int                         # subjects with one rating
    unratable: int                      # selected subjects whose render failed
    open: tuple[str, ...]               # subject_uids that need adjudication
def sync_unratable(session: Session, plan: QueuePlan, protocol: Protocol) -> int
def finalize(session: Session, plan: QueuePlan, protocol: Protocol) -> FinalizeReport
def freeze_calibration(run_dir: Path, *, subjects: Sequence[str] | None = None) -> Path      # writes calibration.yaml
```

`finalize` rewrites `final_labels` and `final_marks` from the current submissions of `main` assignments (calibration and repeats excluded). A subject with one rating: its values, method `single`. With two or more: where all raters are within one scale step on every criterion, the lower value (T3) and the union of the marks, method `consensus`; where any criterion differs by two steps, or one rater said unratable and another did not, no label, and the subject is listed in `open`. Rows of method `adjudicated` are never touched. `surfcensus finalize` prints the report and, for a local run, the labels of the open subjects, so two pilot raters can sit down with the list; the adjudication screen is milestone 5. This is how the two-rater pilot of phase 0 works on milestone 4 alone (T29): each rater runs `surfcensus review PATH --run-dir DIR --rater NAME --blind` on the same run directory, one after the other; the unblinded screens stay closed to each until they have rated everything; then `finalize`.

```python
# calibration.py
class CalibrationMark(_Frozen):
    mark_class: Literal["A", "B", "C", "X"] = Field(alias="class")
    tkr: Vec3
class CalibrationExample(_Frozen):
    criterion: Literal["A", "B", "C"]
    value: int
    panel: PanelName
    slice: int
    caption: str = ""
class CalibrationSubject(_Frozen):
    subject_uid: str
    values: dict[str, int]              # consensus value per criterion
    unratable: str | None = None
    marks: tuple[CalibrationMark, ...] = ()
    justification: str = ""             # the short written text shown after calibration; written by hand
    examples: tuple[CalibrationExample, ...] = ()
class CalibrationSet(_Frozen):
    version: str
    subjects: tuple[CalibrationSubject, ...]

def load_calibration(run_dir: Path) -> tuple[CalibrationSet, str] | None      # None when the run has no calibration.yaml
def write_calibration(run_dir: Path, calibration: CalibrationSet) -> Path     # atomic; refuses to overwrite
```

`calibration.yaml` is read like `protocol.yaml`: at server start, validated (every subject is in the catalog, values are on the scale, example slices exist in the subject's render), and its SHA-256 is stored with every calibration submission (`submissions.calibration_sha256`). `finalize.freeze_calibration` (behind `surfcensus calibration freeze`) writes it from `final_labels` and `final_marks` (all subjects with a final label, or those named), with empty justifications and no examples, and adds the `selection` rows with reason `calibration`; it refuses to overwrite an existing file. Justifications and examples are then written into the file by hand. In milestone 4 the file only feeds `examples` of `/api/session`; the calibration queue and the consensus feedback are team mode (8.7), as is carrying a set from the pilot run into the census run (8.10).

### 7.8 `export/tables.py`, `surfcensus export`

```python
def export_run(run_dir: Path, out_dir: Path | None = None, *, fmt: Literal["csv", "parquet"] = "csv",
               profile: Literal["analysis", "release"] = "analysis") -> list[Path]
def ratings_frame(run_dir: Path) -> pd.DataFrame          # the `ratings` table below; also behind GET /api/export
```

```
surfcensus export --run-dir DIR [--out DIR] [--format csv|parquet] [--release]
surfcensus export --run-dir DIR --reveal --reveal-out DIR        (milestone 5, 8.1)
```

Reads `PublicCatalog`, the tables and `app.sqlite` (standard-library `sqlite3`); writes into `run_dir/export/` (`run_dir/release/` with `--release`; or `--out`), atomically, sorted, and prints the paths. It works on a run without ratings and on one without `app.sqlite` (the rating columns are then empty). `export_run` has no reveal argument and cannot reach an accession; `--reveal` is handled in `cli.py` (8.1).

| File | One row per | Columns |
|---|---|---|
| `subjects` | subject | `subject_uid`, `dataset`, `subject_label` and `participant` (local run only), the provenance columns of `public_subjects`, `render_status`, `render_reason`, every column of the primary `scores` row, `final_A`, `final_B`, `final_C`, `final_method`, `unratable_reason`, `other`, `n_ratings`, `n_marks` |
| `scores` | subject and reference | as `scores.parquet` (all references) |
| `selection` | subject and reason | `subject_uid`, `reason`, `stratum`, `inclusion_prob`, `seed`. A subject drawn by both samples has two rows with their own probabilities, which is what the weighted analyses need. |
| `ratings` | completed assignment (current submission) | `assignment_id`, `subject_uid`, `rater`, `kind`, `render_key`, `render_current`, `A`, `B`, `C`, `unratable_reason`, `other`, `note`, `revision`, `protocol_version`, `protocol_sha256`, `app_version`, `client_ts`, `server_ts`, `ms_on_subject` |
| `marks` | mark of a current submission | `subject_uid`, `rater`, `assignment_id`, and every column of the `marks` table except `deleted_ts` |
| `final_marks` | final mark | as the table |
| `clusters` | cluster | `clusters.parquet` left-joined with `clusters_native.parquet` |
| `datasets` | dataset | `dataset`, `source`, `pipeline`, `pipeline_version`, `n_subjects`, modal field strength |
| `behaviour` | completed assignment | `assignment_id`, `subject_uid`, `rater`, `ms_on_subject`, `n_frames_viewed`, `n_plays`, `n_pauses`, `n_mode_switches`, `n_back` |
| `manifest.json` | | run id, mode, profile, key fingerprint, surfcensus version, stage hashes, template id, reference ids, protocol and calibration versions and hashes, and the row count and SHA-256 of every exported file |

Two further files are named here and written in milestone 6, because `analysis_plan.md` has the notebooks read only from the export and the spatial analyses need per-vertex data: `vertices.parquet`, key (`subject_uid`, `reference_id`, `hemi`, `fsavg6_vertex`), for rated subjects, with `in_search` and `visible` (bool), `z` (float32), and `dist_A_mm`, `dist_B_mm`, `dist_C_mm` (the distance, in native space, from the nearest native vertex mapping to this template vertex to the nearest final mark of that class; null beyond 15 mm), built from `fsavg/`, the references, `vertex_map.npz` and `final_marks`; and `error_map.parquet`, key (`class`, `hemi`, `fsavg6_vertex`), with `n_marks` and `n_subjects`.

Profiles. `analysis` is the full pseudonymous export for the team's notebooks and is the only profile of a local run. `release` is what a census publishes: `subjects`, `scores` and `selection` hold selected subjects only; `age`, `sex` and `manufacturer` are left out; `field_strength` is the binned value (1.5, 3, 7); `datasets` has no `pipeline_version` and bins `n_subjects` (1-9, 10-29, 30-99, 100+); `inclusion_prob` is rounded to two significant digits; rater names are `R-01`, `R-02`; `note` is left out; `behaviour` is per-assignment summaries as above. The reason is D10: an export that bins the subject count in `datasets` while listing every subject with its pseudonym, exact age and exact sampling probability lets anyone recover the count by a `groupby` and match the age vector against `participants.tsv`. What `release` contains is Tomas's to decide (T26).

### 7.9 Frontend (`app/static/`, plain ES modules, no build step; pages in `app/templates/`)

| File | Responsibility |
|---|---|
| `templates/base.html`, `rate.html`, `review.html`; `static/app.css` | layout of `rater_app.md`; the pages contain no subject data, everything arrives through the API |
| `js/api.js` | `fetch` wrappers for the routes of 7.5; retry on `rendering`; retries a failed submit and keeps the unsent payload in `sessionStorage` |
| `js/viewer.js` | per panel two stacked images (grayscale, contours); both playback modes from the frame tables; play, scrub, speed, 2x zoom, contour toggle; crosshairs and edge letters from the `render` object; preloads the current subject's tiles and prefetches the next two |
| `js/coords.js` | click offset to tile pixel (`px = ox * width / w`), `pix2vox`, `vox2pix`, voxel to tkr; the same arithmetic as `render/geometry.py`. Display only: the server is the authority |
| `js/marks.js` | SVG layer per panel: place (pauses playback), select, delete, undo, hide; a mark shows in the other panels within `mark_show_mm` of their current slice, with opacity falling off |
| `js/rating.js` | the rating bar built from the protocol; note, unratable, other; the active class; whether `Enter` is enabled, and the hint when it is not |
| `js/keys.js` | the keyboard map of `rater_app.md`, with criterion and mark keys taken from the protocol |
| `js/events.js` | run-length frame log, batching, `sendBeacon` |
| `js/rate.js` | page controller: session, next, submit, back, progress line |
| `js/review.js` | sortable table; scatter of composite against final label (inline SVG); subject view with marks and cluster centroids; a plain message when the server answers 403 because the rater still has subjects to rate |
| `js/help.js` | the `?` overlay with the scale definitions and, when the session has any, the reference `examples` (tiles of calibration subjects) |

---

## 8. Milestone 5: census (interfaces and data flow)

Less detail on purpose. The rule: this milestone adds a source, a key, phases the local path never calls, the intensity metric and the team branch of the queue. It changes no table of `catalog.sqlite` or `app.sqlite` and no signature of sections 5 to 7. If a census need seems to require that, it is a design question to raise, not a reason to add a parameter to `review`.

New files: `remote/{__init__,s3,cache,openneuro,gitrefs}.py`, `sources/openneuro_derivatives.py`, `enrich/{__init__,participants,acquisition}.py`, `enrich/columns.yaml`, `select/{__init__,sampling}.py`, `metrics/intensity.py`, `app/auth.py`, `app/adjudication.py`, `app/agreement.py`, `app/errormap.py`, the team pages of 8.7, `calibration_transfer.py`, `workflow/Snakefile`. Completed: `privacy.py`, `log.py` (`RedactingFilter`), `app/queue.py` (team branch), `app/server.py` (mounts the team routers), `render/batch.py` (the worker calls `intensity_rows`), `cli.py`. `pyproject.toml` gains a dependency group `workflow = ["snakemake"]` and the `justfile` a `census` recipe that uses it.

### 8.1 `privacy.py`, completed

```python
DEFAULT_KEY_PATH = Path("~/.config/surfcensus/dataset.key")

def init_key(path: Path = DEFAULT_KEY_PATH) -> HashKey
    # 32 random bytes as 64 hex characters, mode 0600, parent 0700; refuses to overwrite; refuses a path inside a
    # git work tree or inside a directory holding config.yaml and catalog.sqlite; prints the fingerprint and the
    # reminder to back the key up
def load_key(path: Path | None = None) -> HashKey            # RuntimeSettings.key_file by default; warns if group- or world-readable
def assign_pseudonyms(key: HashKey, records: Sequence[DatasetRecord],
                      taken: Mapping[str, tuple[str, str]]) -> dict[tuple[str, str], str]
    # (source, identity) -> pseudonym. 8 hex each; when two pairs collide both get 12 hex, then 16; pseudonyms
    # already in the catalog never change (a newcomer that collides with one gets the longer form alone)
def reveal_table(catalog: Catalog, key: HashKey) -> tuple[pd.DataFrame, pd.DataFrame]
    # datasets: pseudonym, source, accession, derivative_ref, snapshot_tag. subjects: subject_uid, subject_label.
    # Recomputes every pseudonym and uid from the key and fails if any differs from the catalog.
def write_reveal(catalog: Catalog, key: HashKey, out_dir: Path) -> list[Path]
    # reveal_datasets.csv, reveal_subjects.csv and reveal_manifest.json (run id, key fingerprint, row counts),
    # mode 0600. Refuses an out_dir inside the run directory (which contains export/ and release/) or inside a
    # git work tree.
class RedactingFilter(logging.Filter)                        # in log.py
    # applies privacy.scrub to every record's message and arguments with the run's pseudonyms and roots:
    # roots and identities become "<root>", absolute paths their last component, ds\d{6} the pseudonym,
    # sub-<label> tokens "sub-?". Library code logs subject_uids, never labels, so nothing needs mapping back.
```

`surfcensus key init [--path FILE]`. `surfcensus export --reveal --reveal-out DIR` is implemented in `cli.py`: it opens the private catalog, loads the key and calls `write_reveal`; `--reveal-out` is required, so the mapping never lands next to files meant for release, and nothing about it is written into the export. A local run has nothing to reveal (`ConfigError("invalid", ...)`). `reveal_table` and `write_reveal` are the only functions that return accessions for display, and only `cli.py` calls them.

### 8.2 `remote/`

```python
def default_transport() -> httpx.BaseTransport      # remote/__init__.py; the one seam tests replace

@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str
    last_modified: str
@dataclass(frozen=True)
class S3Head:
    size: int
    etag: str
    last_modified: str
    version_id: str | None

class S3Client:                                   # remote/s3.py
    def __init__(self, bucket: str, *, config: RemoteConfig, addressing: Literal["virtual", "path"] | None = None,
                 transport: httpx.BaseTransport | None = None) -> None
        # addressing defaults to "path" for a bucket name with a dot (openneuro.org), else "virtual";
        # transport=None uses default_transport()
    def list(self, prefix: str, *, delimiter: str | None = None) -> Iterator[S3Object | str]   # str: a common prefix
    def head(self, key: str) -> S3Head
    def get(self, key: str, *, byte_range: tuple[int, int] | None = None,
            if_match: str | None = None) -> tuple[bytes, S3Head]
    def download(self, key: str, dest: Path, *, expect: S3Object) -> S3Head
        # If-Match: expect.etag; streams to a temporary file; checks the size, and the MD5 when the ETag has no "-";
        # renames into place. HTTP 412 or a mismatch raises SourceError("changed", ...)

class Cache:                                      # remote/cache.py
    def __init__(self, root: Path, *, max_bytes: int, secret: bytes, retain_roles: Collection[FileRole] = ()) -> None
    def path_for(self, bucket: str, key: str, etag: str, *, retained: bool) -> Path
        # h = privacy.cache_name(secret, bucket, key, etag); name = the file name of key.
        # retained: root/retained/<h[:2]>/<h>/<name>.  leased: root/leases/<host>-<pid>/<h>/<name>.
    def lease(self, client: S3Client, objects: Mapping[FileRole, S3Object], *,
              retain: bool = False) -> AbstractContextManager[dict[FileRole, Path]]
        # blocks while this process's leased bytes plus the request would exceed max_bytes; downloads what is
        # missing; on exit deletes what this lease downloaded, except roles in retain_roles and everything when
        # retain is true, which move to root/retained/ by rename
    def prune(self) -> int
        # removes partial files under root/retained/ and the lease areas of this host whose process no longer
        # exists; never the area of a live process or of another host (those are reported)

def remote_head(url: str, *, transport: httpx.BaseTransport | None = None) -> str | None
    # remote/gitrefs.py: commit of HEAD over smart HTTP, no git binary
def list_datasets(*, transport: httpx.BaseTransport | None = None) -> Iterator[str]      # remote/openneuro.py, GraphQL
def latest_snapshot(accession: str, *, transport: httpx.BaseTransport | None = None) -> str | None
```

- Read-only and anonymous by construction: the transport wrapper refuses any method but GET and HEAD and any request with an `Authorization` header or a signed query string (`PermissionError` before any I/O). The one exception is POST to the OpenNeuro GraphQL endpoint with a query body.
- Retries with exponential backoff and jitter on 5xx, `SlowDown` and connection errors, up to `max_retries`; 403 and 404 are not retried. Concurrency is a process-wide semaphore of `max_concurrency`.
- Retained across phases: the Tier A files (stats, build stamp), the template files, and the roles of `cache.retain_roles`. Everything else lives for one lease. With the default (nothing extra retained) a rendered subject downloads its two `sphere.reg` files twice, once in 4b and once in phase 6 (10.6 MB each time); setting `retain_roles` to the two `sphere_reg` roles trades that for 10.6 MB of disk per subject (T28).
- Leases are per process, in an area of their own, so a process that starts while another is in the middle of a batch (Snakemake runs one per dataset on the same cache) never sees the other's files as leftovers. The byte budget is per process, so under Snakemake the disk bound is `jobs x cache.max_bytes`.
- The cache directory names are keyed with the run secret and keep only the file's own name (`lh.pial.T1`, because `render.json` records an input under the name it is handed, if that is a candidate name of the role). A cache path therefore names no accession and no subject, and cannot be matched to the bucket listing; where one is printed, `scrub` has already reduced it to the file name. The cache belongs to one run.
- `derivative_ref` comes from `remote_head("https://github.com/OpenNeuroDerivatives/<name>")`, stored with `derivative_ref_source = "github-head"`: what the mirror's head was when the census ran, the best available statement about a bucket without a version listing.

### 8.3 `sources/openneuro_derivatives.py`

```python
class OpenNeuroDerivatives:
    kind: SourceKind = "openneuro-derivatives"
    def __init__(self, *, client: S3Client, cache: Cache, prefix: str = "fmriprep/") -> None
    # implements Source; config() returns SourceConfig(kind=..., bucket=client.bucket, prefix=prefix)
```

- `datasets()`: common prefixes of `fmriprep/` matching `ds\d{6}-fmriprep/`; `fs_root` is the first of `sourcedata/freesurfer`, `freesurfer` that has a common prefix other than a template name; neither gives a record with `fs_root = ""` and no subjects. `pipeline_version` from `dataset_description.json`; `template_location` when `fs_root/fsaverage/` has the eight required template files.
- The phase 1 number of `phases.md` is "datasets with at least one FreeSurfer subject", by the pattern (a left pial candidate and `mri/T1.mgz`). The existence of a subject prefix is not that. `discover` therefore lists up to three candidate subject prefixes per dataset (one request each) and stores whether one of them passes the pattern in `datasets.fs_subject_found`; its report gives that count and says it is a probe. The exact count is in the inventory report.
- `subjects()`: one listing per subject prefix (309 to 337 keys, one page); roles matched against `ROLE_CANDIDATES` exactly as for local directories, which already covers `?h.pial.T1` without `?h.pial`.
- `read_bytes`, `lease`, `template_dir` go through `S3Client` and `Cache` with `If-Match` on every GET. On the first full download `version_id` and SHA-256 are written to `files`. Dataset-level files (`dataset_description.json`) are recorded in `dataset_files`.

The embedded source (O9) would be a third class with the same protocol and a path-pattern table.

### 8.4 Commands

All take `--run-dir DIR`; all but `key` and `discover` take `--dataset ID` (accession or pseudonym, repeatable; never stored) and `--jobs J`. Each prints the exit check of `phases.md` as a small table, ordered by pseudonym, and saves it as `logs/<phase>-summary.json`.

| Command | Phase | Unit | Reads | Writes |
|---|---|---|---|---|
| `key init` | | | | the key file, outside the run |
| `discover [--source openneuro-derivatives]` | 1 | source | bucket top level, descriptions, git refs, up to three subject listings per dataset | `datasets`, `dataset_files`; creates the run on first use; needs the key |
| `discover --source local --path PATH` | 1 | | a local directory | one `datasets` row and a `SourceConfig` with its pseudonym; the one way to add a local source (the pilot 7T data) to a census run; needs the key |
| `inventory` | 2 | dataset | listings, Tier A (stats retained) | `subjects`, `files`; needs the key |
| `enrich` | 3 | dataset | raw bucket: `participants.tsv`, T1w sidecars; the snapshot tag | `subjects.age`, `sex`, `field_strength`, `manufacturer`; `datasets.snapshot_tag`; `dataset_files` |
| `metrics --stage regional` | 4a | dataset | retained stats (no network) | regional part |
| `metrics --stage fsaverage` | 4b | dataset | Tier B, leased in batches | `fsavg/*.npz`, vertex_summary part |
| `metrics --stage reference` | 4c | run | maps, enrichment | reference, clusters |
| `metrics --stage score` | | run | tables | `scores.parquet` |
| `select --n-prevalence N --n-detectability M --seed S` | 5 | run | scores | `selection` |
| `render [--all \| --selected]` | 6 | dataset | Tier B and C, leased in batches | `media/`, `renders`, intensity part |
| `render --check-sample` | 6 exit check | run | renders | `logs/render-check-<seed>.{txt,html}` |
| `metrics --stage native` | | run | clusters, vertex maps | `clusters_native.parquet` |
| `metrics --stage merge` | | run | parts | every merged table |
| `calibration import --from FILE` | before 7 | run | a portable calibration file | `calibration.yaml`, `selection` rows |
| `serve` | 7 | | media, selection | `app.sqlite` |
| `raters add NAME [--role R]`, `raters list` | 7 | | | `app.sqlite` |
| `finalize` | 7 | | ratings | `final_labels`, `final_marks` |
| `export [--release]`, `export --reveal --reveal-out DIR` | 8 | | everything public; the reveal: the catalog and the key | `export/`, `release/`; the reveal directory |

Enrichment: sidecars by BIDS inheritance, cheapest first (top-level `T1w.json`, then `sub-<label>/anat/*_T1w.json`, then the session level). `participants.tsv` columns are mapped through the reviewed table `enrich/columns.yaml`; an unmapped file leaves nulls and is listed in the summary. Every file read is recorded in `dataset_files` (bucket, key, size, ETag, version id, SHA-256), which is where the hard rule on traceability is met for dataset-level values and what the `enrich` unit's input signature is built from. `snapshot_tag` is `latest_snapshot(accession)`, null when the GraphQL endpoint cannot be reached.

### 8.5 Streaming batches and the disk bound

Phases 4b and 6 use `batches.run_batches` (5.9) unchanged; for remote data the lease downloads and deletes. Disk in use: at most `cache.batch_subjects x 39 MB` of source files (1.25 GB at the default 32) plus outputs. A subject whose download fails or whose ETag changed gets a failed row (`unreachable`, `source changed`) and the batch goes on. Killing the process loses at most one batch; `Cache.prune()` at the next start removes its leftovers.

Phase 6 does, per subject and in one lease of Tier B and C: render with the pinned template, fill the `vertex_map` counterparts and the D12 pole distance, and compute the intensity rows (8.9) while T1 and the surfaces are on disk. Native coordinates of cluster centroids need nothing from the lease: they come from the vertex map afterwards (stage `native`, 6.9).

Which subjects phase 6 renders is `select.render_scope`: `all` by default, as `phases.md` says ("selection determines the rating sample, not the rendering set"), so that the D12 pole distance and the intensity metric exist for every complete subject, and the 2% by-eye check (`render --check-sample`) is drawn from all of them. For 5,000 subjects that is about 195 GB of transfer, about 25 minutes of compute on 32 cores and about 30 GB of media without video. `selected` renders only the rating sample (T28).

In a census run the template candidate order is the same as locally; `adjacent` is the copy of the first dataset, in pseudonym order, that has one. Every dataset's adjacent `?h.sphere.reg` is compared with the pinned template's by MD5 and the result stored in `datasets.template_sphere_match`.

### 8.6 Selection (`select/sampling.py`)

```python
def prevalence_sample(subjects: pd.DataFrame, n: int, *, min_per_dataset: int, seed: int) -> pd.DataFrame
def detectability_sample(scores: pd.DataFrame, n: int, *, config: SelectConfig, seed: int) -> pd.DataFrame
def run_select(ctx: RunContext, *, n_prevalence: int, n_detectability: int, seed: int) -> PhaseReport
```

Both return `subject_uid`, `reason`, `stratum`, `inclusion_prob`, `seed`. Sampling is without replacement within strata, on subjects sorted by `subject_uid`, with `numpy.random.default_rng([seed, zlib.crc32(stratum.encode())])`, so adding a dataset later does not change the draws in the others. The prevalence sample never reads `scores`. The detectability sample is stratified by decile of `rank` among the primary score rows (not of the composite value, which has ties), samples the top decile at `top_decile_factor` times the base rate, and takes every subject above the `extreme_quantile` of rank with reason `extreme`. A subject drawn by both keeps both rows, each with its own probability. A sample size larger than its population selects everything with reason `all`. The `select` unit's input signature is the SHA-256 of `scores.parquet` and of the sorted present uids, so a rerun with the same sizes and seed on unchanged scores does nothing. `architecture.md` has a single `--n`; the two samples have different sizes and designs, so there are two options.

### 8.7 Team mode, calibration, adjudication, dashboard

`app/auth.py`: `GET /r/{token}` looks up `sha256(token)` in `raters.token_hash` where `active = 1`, sets the token itself as an `HttpOnly`, `SameSite=Strict`, `Secure` cookie and redirects to `/` (404 for an unknown token). Every `/api` route then looks the cookie up the same way on each request. There is no server-side session, no `SessionMiddleware` and no dependency beyond the `app` extra. Tokens are `secrets.token_urlsafe(24)`, shown once by `surfcensus raters add NAME`; only the hash is stored.

Added without changing a milestone 4 route:

| Route | Who | |
|---|---|---|
| `GET /api/calibration/{subject_uid}` | rater, after their own submission for that subject | consensus values, marks and the written justification from `calibration.yaml`; sets `assignments.consensus_served_at`, after which the assignment cannot be revised (409) |
| `GET /api/adjudication/next` | adjudicator | a subject in `FinalizeReport.open`; every rater's ratings, notes and marks |
| `POST /api/adjudication` | adjudicator | final values per criterion and the final mark set (`keep`, `move`, `delete`, `add`, each naming its source marks); writes `final_labels` (method `adjudicated`) and `final_marks` |
| `POST /api/finalize` | lead | `finalize` of 7.7 |
| `GET /api/errormap?class=&hemi=` | lead, adjudicator | per-vertex frequency of final marks on fsaverage6 (through `marked_vertices` with `app.mark_near_mm`) and the mean z-map, as JSON arrays |
| `GET /adjudicate`, `GET /dashboard`, `GET /errormap` | adjudicator; lead; lead and adjudicator | the team pages listed below |
| `GET /api/errormap.png?class=&hemi=&view=medial\|inferior\|posterior` | lead, adjudicator | the static image `rater_app.md` asks for, regenerated on demand: the two maps side by side on the template's white surface, drawn with matplotlib on the Agg canvas (`app/errormap.py`) |

All of these are unblinded routes (7.5): closed to every role while the rater has work left. `/api/stats` gains `agreement` (weighted Cohen's kappa per pair and Krippendorff's alpha over all raters, per criterion; intra-rater agreement from repeats) and `drift` (per rater, the distribution of values in consecutive blocks of 50 assignments). Both statistics are implemented in `app/agreement.py` with numpy, because `scikit-learn` and `krippendorff` are in the `analysis` extra and a lab server installs `surfcensus[app]`; the tests compare them with those packages.

Team pages (`app/templates/` and `app/static/js/`): `adjudicate.html` / `adjudicate.js` (the viewer with every rater's marks, the final-label bar, mark editing); `dashboard.html` / `dashboard.js` (`/api/stats`); `errormap.html` / `errormap.js`; and the calibration feedback panel in `rate.js` (consensus next to the rater's own values, the justification, the consensus marks drawn in a second colour).

### 8.8 `workflow/Snakefile`

Wildcards are pseudonyms; no accession appears in a target, a log name or a rule. Targets are sentinel files `status/<phase>/<pseudonym>.done` written by a thin wrapper after the CLI command exits 0 or 1. The catalog is the truth; sentinels only let Snakemake see it. Every rule is one CLI command.

```
checkpoint discover:   output status/datasets.txt                    (one pseudonym per line, sorted)
rule inventory:        input discover    output status/inventory/{ds}.done     resources: net=1
rule enrich:           input inventory   output status/enrich/{ds}.done        resources: net=1
rule regional:         input inventory   output status/regional/{ds}.done      (no network: stats were retained)
rule fsaverage:        input inventory   output status/fsaverage/{ds}.done     resources: net=1, disk_mb
rule reference:        input all fsaverage, enrich, regional   output status/reference.done   threads: N
rule score:            input reference   output tables/scores.parquet
rule select:           input score       output status/select.done
rule render:           input score (render_scope all) or select (selected)
                                         output status/render/{ds}.done        resources: net=1, disk_mb; threads: J
rule native:           input all render  output tables/clusters_native.parquet        (metrics --stage native)
rule merge:            input all render, native   output status/merge.done            (metrics --stage merge)
```

`net` caps how many datasets download at once. A rule whose command exits 1 (some units failed) writes its sentinel and a `failed/<phase>/<pseudonym>.txt` with the counts, so one bad subject does not stop the census; exit 2 stops it.

### 8.9 `metrics/intensity.py` (phase 4d)

`phases.md` runs the intensity metric inside phase 6, while T1 and the surfaces are on disk; the streaming batches delete them afterwards. D18 puts the module after milestone 4. It is therefore built in this milestone, before phase 6 runs at scale, and it plugs into the hook that exists from milestone 2 (`RenderResult.extras`), so no signature changes. Rendering the census without it would mean downloading Tier C a second time later (about 135 GB for 5,000 subjects).

```python
def intensity_rows(subject_uid: str, t1: Volume, white: Mapping[Hemi, Surface], pial: Mapping[Hemi, Surface],
                   regions: Mapping[Hemi, Mapping[RegionName, Bool[np.ndarray, "n"]] | None],
                   config: IntensityConfig) -> pd.DataFrame
```

Per hemisphere: `T1.mgz` is sampled by nearest voxel at each `depth` along the line from the white to the pial position of every native vertex. `outer_outlier_frac` is the fraction of search-region vertices whose value at the outermost depth lies outside the `percentiles` range of the control region's values at that depth (the subject is its own reference, as `metrics.md` asks, because the sign of the contrast depends on the sequence). `gap_frac` is the fraction of search-region vertices whose white-to-pial distance exceeds `metrics.thickness_max_mm`, which sees the vertices whose stored thickness was clamped. `regions` are the native regions of 6.5 (`native_region`), None for a hemisphere without one, whose row then has nulls.

`intensity.parquet`, key (`subject_uid`, `hemi`): `region_kind` string; `search_n`, `control_n` int32; `outer_outlier_frac`, `gap_frac` float32; `search_median_d<k>`, `control_median_d<k>` float32 for each depth `k`. Its part is written by the render commit and merged by `metrics --stage merge`. It is not a component of the composite (`metrics.md` lists four), and it is one more baseline in the detectability analysis.

### 8.10 Carrying the calibration set from the pilot into the census

The pilot runs on milestone 4, as a local run with its own salt, so its `subject_uid`s mean nothing in the census run. The docs do not say how the frozen set gets from one to the other. Subjects are matched by content, not by name:

```python
def export_calibration(run: Run, out: Path) -> Path
    # the pilot run's calibration.yaml plus, per subject, the size and MD5 of t1, lh_pial and rh_pial (read through
    # leases). Mode 0600; refused inside any run directory: the digests equal the bucket's ETags, so this file
    # identifies the subjects and is handled like the reveal.
def import_calibration(run: Run, source: Path) -> Path
    # finds each subject in the census catalog by size and ETag (remote: the ETag is the MD5 for these objects;
    # a local source: the MD5 is computed for the candidates of equal size), writes the census run's
    # calibration.yaml with census subject_uids, and the selection rows with reason `calibration`. A subject
    # that matches nothing, or more than one, is an error naming its position in the file.
```

Both are in a module of their own, `calibration_transfer.py`, behind `surfcensus calibration export --run-dir PILOT --out FILE` and `surfcensus calibration import --run-dir CENSUS --from FILE`. `calibration.py` itself stays on the public side of the boundary. The transfer module takes a `Run`, is called from `cli.py` only, and is the one place outside inventory that reads private file metadata; it joins the list of modules that may import `store.Catalog` (5.5), and the `ast` test of 5.1 bans importing it from the downstream packages, as it does `reveal_table`.

---

## 9. Tests

Conventions as in milestone 1: the package runs under jaxtyping and beartype; session fixtures are read-only; a test that changes a subject writes its own; network tests carry `@pytest.mark.network`, browser tests `@pytest.mark.browser`; both are excluded by default.

### 9.1 Synthetic fixtures

New modules beside `tests/synth.py`, which these milestones do not edit. All files are written with nibabel (geometry, morph data, annotations round-trip; a `.label` file is written as text and read back by `read_label`).

`tests/synth_tree.py` (milestone 2)

```python
def write_light_subject(path: Path, *, complete: bool = True) -> None
    # the smallest directory that passes is_freesurfer_subject_dir and renders: 16^3 volume, 42-vertex surfaces;
    # complete=False leaves out surf/lh.pial
def write_tree(root: Path, layout: Layout, labels: Sequence[str], *, with_template: bool = False,
               dangling_template: bool = False, incomplete: Collection[str] = (),
               fmriprep_version: str | None = "23.2.0") -> Path
    # SUBJECTS_DIR, fMRIPrep new and legacy, BIDS derivatives; fsaverage and fsaverage5 directories (or a dangling
    # fsaverage link); a dataset_description.json; returns the path a user would pass
```

`tests/synth_template.py` (milestone 3)

```python
@dataclass(frozen=True)
class SyntheticTemplate:
    path: Path
    order: int
    masks: Mapping[str, Mapping[str, np.ndarray]]      # hemi -> name -> the parcel and cortex masks it used
def write_synthetic_template(root: Path, *, name: str = "fsaverage", order: int = 7,
                             with_avg_area: bool = True) -> SyntheticTemplate
    # the eight required template files, and surf/?h.white.avg.area.mgh unless with_avg_area is False.
    # Sphere: icosphere(order) at radius 100 (its vertex order nests, like fsaverage's). White: the synthetic
    # subject's white ellipsoid, pole at the most posterior point. Annotation: Desikan names laid out by rule on
    # the sphere (a posterior cap in four sectors for the occipital parcels, a ring in front of it split into
    # precuneus and fusiform, a frontal cap in three sectors, a medial strip left unlabelled). Cortex label:
    # everything but the medial strip. Average area: the mesh area times a factor that differs between the
    # posterior cap and the rest, so the two area sources can be told apart. scripts/build-stamp.txt.
def write_label(path: Path, vertices: np.ndarray, coords: np.ndarray) -> None
```

The default order is 7, the size of the real fsaverage, so the fixture is a valid input of the default code path: `find_template` with the default config (`mesh = "fsaverage6"`) coarsens it once through the real `coarsen_faces`, and `Run.template()`, `Run.metrics()`, `review()` and `three_plane(template="auto")` work on it unchanged (`icosphere(7)` takes 0.25 s). Tests that want a small mesh pass `MetricsConfig(mesh="fsaverage4")` (2,562 vertices from the same order-7 files, three coarsening steps) or write an order-5 template and ask for `fsaverage4`.

`tests/synth_tierb.py` (milestone 3)

```python
@dataclass(frozen=True)
class ThickPatch:
    centre: tuple[float, float, float]     # template-space direction
    radius_deg: float
    value_mm: float                        # clamped at 5.0
def add_tier_b(subject: SyntheticSubject, template: SyntheticTemplate, *, warp: float = 0.15,
               patch: ThickPatch | None = None, hemis: Sequence[str] = ("lh", "rh"), noise_seed: int | None = None,
               flip_winding: bool = False) -> None
    # surf/?h.sphere.reg: the subject's icosphere at radius 100 with a known smooth tangential displacement (a
    # rotation about a fixed axis by an angle that varies with latitude, amplitude `warp` radians; warp=0 is the
    # identity), so the mapping is known in closed form; flip_winding reverses every face. surf/?h.thickness: an
    # analytic field of the template-space direction (2.5 mm plus a smooth term, optional seeded noise), 0 on the
    # medial strip, the patch if given. label/?h.aparc.annot: each vertex takes the parcel of the template point
    # it maps to, -1 on the medial strip.
def expected_thickness(directions: np.ndarray, patch: ThickPatch | None = None) -> np.ndarray
def add_tier_a(subject: SyntheticSubject, *, log_style: Literal["fs6-fmriprep", "fs7-fmriprep", "plain"] = "fs6-fmriprep",
               flags: Sequence[str] = (), truncations: Mapping[str, tuple[int, int]] | None = None,
               holes: tuple[int, int] = (3, 5), fs_version: str = "6.0.1") -> None
    # stats/?h.aparc.stats and stats/aseg.stats in the real layout, keeping the malformed FreeSurfer 6 measure line
    # and carrying "SUBJECTS_DIR /data/ds999999-fmriprep/..." headers as privacy canaries; scripts/build-stamp.txt;
    # scripts/recon-all.log and, for the fMRIPrep styles, scripts/recon-all-?h.log with several invocations
def synthetic_cohort(root: Path, n: int, template: SyntheticTemplate, *, seed: int,
                     bad: Mapping[int, ThickPatch] | None = None) -> list[SyntheticSubject]
    # n small subjects (subdivisions=3) with Tier A and B; n = 32 crosses the 30-subject rule
```

`tests/conftest.py` gains:

- session fixtures (read-only): `synth_template`; `synth_cohort` (32 subjects, three with a patch, one without `sphere.reg`, one without stats); `small_cohort` (5 subjects, for D16);
- `small_run` (session, read-only): three synthetic subjects in a `write_tree` with a template, one of them with a `ThickPatch`, prepared without `review` and without the app:

  ```python
  run = Run(
      run_dir,
      sources.local(tree),
      config=new_local_config(
          tree,
          render=RenderConfig(panel_px=64),
          metrics=MetricsConfig(min_reference_n=3, min_vertex_n=2),
      ),
  )
  run.inventory()
  run.metrics()
  run.public().replace_selection(all_present, ["all"])
  run.render(video=False)
  ```

  With `min_reference_n=3` the three subjects are scored within the set, so the run has z-maps, a cluster with a native centroid, and `composite_z` scores; the D16 path is covered by `small_cohort`. The fixture needs milestones 2 and 3 only, so the app agents can use it from the first step of milestone 4.
- `run_copy(small_run, tmp_path)` (function scope): `shutil.copytree` of the run directory. Every test that writes (a rating, a render, a config change) uses the copy.

`tests/data/recon_logs/`: short text files cut from four real logs, `scripts/recon-all.log` and `scripts/recon-all-lh.log` of `fmriprep/ds000102-fmriprep/sourcedata/freesurfer/sub-01/` (FreeSurfer 6.0.1) and of `fmriprep/ds003020-fmriprep/sourcedata/freesurfer/sub-UTS01/` (7.3.2) in the OpenNeuroDerivatives bucket, plus a hand-written plain layout. The agent of the provenance package fetches the four logs once (anonymous GET; the session's scratch directory has only three of them and is not part of the repository) and commits the excerpts together with the script that cut them, `tests/data/recon_logs/make_excerpts.py`: the header of each invocation and the build-stamp, holes, truncation and "finished" lines, with every path rewritten to `/data/ds999999-fmriprep/...` and the user and host replaced.

`tests/fakes3.py` (milestone 5)

```python
class FakeS3(httpx.MockTransport):
    def __init__(self, buckets: Mapping[str, Path], *, git_heads: Mapping[str, str] = {},
                 graphql_datasets: Sequence[str] = (), fail: Mapping[str, Sequence[int]] | None = None) -> None
    requests: list[httpx.Request]          # the request log
```

It routes by host (`<bucket>.s3.amazonaws.com`, and by first path segment for a dotted bucket in path style), and serves each directory as a bucket: ListObjectsV2 XML with pagination and delimiter, GET with `Range` and `If-Match`, ETag = MD5, `x-amz-version-id`, injected 503 and `SlowDown` (`fail`: key -> status codes to answer before succeeding). It also answers the smart-HTTP `info/refs` request of `github.com/OpenNeuroDerivatives/<name>` from `git_heads` and the GraphQL POST of `openneuro.org` from `graphql_datasets`. Census tests install it with `monkeypatch.setattr("surfcensus.remote.default_transport", lambda: fake)` or pass it as `Run(..., transport=fake)`; with it `discover` to `export` run offline through the CLI, across the derivatives bucket, the raw bucket and the two HTTP services.

Test modules that need an extra start with `pytest.importorskip`: app tests with `"fastapi"`, `tests/test_browser.py` with `"playwright.sync_api"`, the Snakefile test with `"snakemake"`, the agreement tests' comparison with `"sklearn"` and `"krippendorff"`. So a default run without those extras collects without error.

### 9.2 Tests per module

| Module | Tests |
|---|---|
| imports | every module of the package imports in a fresh interpreter, one process per module (catches import cycles whatever the collection order); `import surfcensus` loads the standard library only |
| `privacy` (M2) | pseudonym equals the D10 formula for a fixed key (golden value); uid is 16 hex and differs by source, identity, pipeline, label and key; the salt round-trips through config; `repr(HashKey)` shows no secret; `render_key` changes with the secret, the tile hash and a digest, and not with anything else; `cache_name` likewise; `scrub`: the reviewer's two real messages (`Empty file: '<full path>'`, `No such file or no access: '<full path>'`) come out with the file name only, `ds000102` becomes its pseudonym, `sub-01` is masked in census mode and kept in local mode, a Windows path, text without any of these is unchanged |
| `config` | defaults round-trip through YAML byte-identically; unknown key and newer schema are errors; changing a hashed field changes that stage's hash and no other; changing `remote`, `cache`, `app` changes none; `stage_hash` is stable against key order; a census config with a local source holds no path |
| `store` | create and reopen; migrations applied from an empty file, rows preserved by a test migration, a newer `user_version` refused; `PublicCatalog.open` read-only applies no migration and refuses an older schema with the documented message; upserts idempotent; `mark_absent`; `relocate` keeps uids; the public views and every frame `PublicCatalog` returns contain none of the private column names, no `dataset_id` and no signature; `public_subjects.subject_label` is null in census mode; `renders()` has no `detail` in census mode; `subject_uids` is ordered by pseudonym then uid, not by discovery order; two processes writing 200 short transactions each lose nothing; `write_table` is byte-identical for shuffled rows and rejects a wrong schema; `upsert_part` replaces exactly the named subjects; `merge_parts` equals a single write and drops absent subjects |
| `_io` | `write_npz` gives the bytes of the milestone 1 vertex-map writer; `atomic_path` leaves nothing behind on an exception |
| `sources/local` | each layout on `write_tree`, with the right root, `fs_root` and pipeline; rule precedence; a single subject as path; templates (by name and by link target) skipped; incomplete and unreadable subjects reported, not dropped; globs (union; no match is an error); duplicate links; old and new `dataset_description.json`, missing and broken; adjacent template found, absent, dangling, incomplete; `config()` round-trips through `from_config`; a given `identity` survives a new root; `snapshot_tree` unchanged after listing, `read_bytes` and `lease`; a changed mtime raises `SourceError("changed")` |
| `inventory/provenance` | the three real build stamps and the 5.3, 7.1.1 and dev forms; 4 and 5 invocations from the two header formats; flags are whole tokens and case-sensitive; `-noT2pial` gives False, `-T2pial` then `-noT2pial` in one invocation gives False, `-T2pial` in a later invocation gives True; `pial.T2` variant; expert, xopts, edit flags; truncations from the per-hemisphere logs, from a two-line single log, absent; holes; `partial=True`; undecodable bytes; an empty file. Property tests: any text parses without raising; no token with `/` or matching `ds\d{6}` survives in `flags` |
| `batches` | order preserved; one failing unit does not stop the batch; leases are released on success, failure and `KeyboardInterrupt`; `jobs=1` and `jobs=2` commit the same results |
| `inventory/subjects`, `run`, CLI | inventory twice: the second reads no file (a counting source); a touched file re-inventories that subject only and leaves its `render_key` unchanged after the re-render; a pial link pointed at another variant of the same size and modification time is inventoried again, loses its digest and is rendered again with another `render_key`; the stamp and the logs read in full have their SHA-256 in `files`, a log read as a head has none, and an interruption leaves no stored subject without them; incomplete subjects counted and given a `missing surface` render row; a vanished subject gets `present = 0`; render through `Run` equals `render_subject` bytes; `jobs=1` and `jobs=2` identical; skipping calls `render_subject` zero times; a failing subject is recorded with a scrubbed detail, does not stop the batch, is retried only with `--retry-failed`, and a second run reports the same failed count and exit code 1; a missing ffmpeg stops the phase with exit 2 and records nothing against a subject; an interruption between batches, then a rerun, completes the rest; a crash between writing a part and recording the ledger row redoes the unit; a second root is refused, `--relocate` accepts it and keeps every uid and render key; `--all` and `--selected`; `render --check-sample` is reproducible for a seed and names it; `tile_hash` field list is pinned; exit codes 0, 1, 2 and the printed summaries; the source tree is unchanged |
| `fsio/stats` | the malformed measure line; both real header variants; wrong field count; holes present and absent; no private header value anywhere in the returned object |
| `fsio/annot`, `fsio/morph` | -1 labels, byte names, missing colour table, `mask` with an unknown name; label indices sorted; native byte order |
| `fsio/registration` | a sphere onto itself: weights (1, 0, 0) at its own vertex and any field reproduced exactly; the known warp: a smooth field survives within a stated tolerance at two resolutions, and barycentric beats nearest neighbour; weights non-negative and summing to 1, every contained point inside its face (hypothesis over random points on a randomly rotated icosphere: all contained); a folded and a zero-area face never match and never raise; a sphere with every face reversed gives the same map as the original and `flipped = True`; the fallback path when faces are removed; the `valid` rule on both sides of `min_valid_weight`; `nearest_source` and `nearest_target` against brute force; identical output across runs |
| `templates` | `coarsen_faces(icosphere(k))` equals the faces of `icosphere(k - 1)` exactly for k = 1 to 5; a non-icosahedral mesh is refused; each mesh name gives its order from an order-7 directory; regions equal the fixture's masks, are subsets of cortex, and `medial`, `inferior`, `lateral` partition `search`; the cut moves the posterior parcels only; a missing parcel is an error; the pole is the most posterior cortex vertex; `area` from the average-area file sums to the file's total and differs from the mesh area by the planted factor, and without the file it is the mesh area, with `area_source` and the id saying which; candidate order with a fake `FREESURFER_HOME`, an adjacent copy, both, neither; an unpinned explicit directory that is unusable does not fall through; a pinned template is used when the environment changes and when the explicit directory has gone (one warning); an explicit directory that differs from the pin stops unless `retemplate`; a pinned run whose cache was deleted and whose candidates are gone raises `changed` and does not return None; the id is equal for two copies and changes with a file or the region config; the cache round-trips exactly; `TemplateNotFound` lists every place |
| `metrics/regional`, `metrics/ceiling` | weights; a missing hemisphere; counts with values at 4.89, 4.9, 5.0 and 0; an empty region gives NaN; the three region kinds; the above-maximum flag |
| `metrics/fsaverage` | the resampled map against `expected_thickness` on the cohort; a constant field stays constant; zeros and non-cortex values do not leak; low coverage raises; more fallbacks than `max_fallback_frac` raise `degenerate sphere`, a flipped sphere does not; the global median excludes the search region; float16 round trip within 0.002 mm and byte identity; `build_reference` against numpy on a small stack, `min_vertex_n`, smoothing leaves a constant map unchanged, the floor; z of a planted patch exceeds the threshold inside the patch (to within one vertex ring) and nowhere else; strata: three datasets of 40, 40 and 10 subjects at one field strength, the 10 are assigned the field-and-version stratum and its reference has 90 members; fall back in the documented order |
| `metrics/clusters` | one planted disc gives one cluster of the right area, `mass`, `excess_mass` and peak; `excess_mass` tends to 0 as the disc's z tends to the threshold; `min_vertices`; two discs joined by one vertex are one cluster, separated they are two; a disc straddling medial and inferior; the threshold boundary; NaN ignored; order and tie rules; `native_centroids` from a vertex map lands within a millimetre of the patch centre on the synthetic pial surface, uses a neighbour when the centroid has no counterpart, and gives no row for a centroid outside the map |
| `metrics/scoring` | `normal_scores`: the 26-zeros case gives the values of section 3, ties share the minimum rank, one subject gives 0, NaN stays NaN; `robust_z`: each step of the cascade with the stated condition; the composite is the maximum and names its component; population `run` against `stratum` on two planted datasets (a dataset whose every subject is bad ranks high under `run`); `n_components` and `partial_score`; `quantile_scores` against a grid; `fallback_rank` on 5 subjects and on 12 with no ceiling vertex (12 distinct ranks); `add_rank` ties; the three bad subjects of the cohort rank 1 to 3 |
| `reference/load` | round trip with `metric_norms` and `parcel_norms`; `applicable_stratum` order; no shipped reference today; the id changes when a member's map changes and its uid does not |
| `metrics/pipeline`, `Run` | the cohort: the tables with the schemas of 6.10; `scores` has one primary row per subject; a second run does nothing; adding a subject redoes the set-level stages only; changing `cluster_z` redoes reference, score and native only; `--dataset` without `--stage` runs the per-subject stages only; no template: the message once, fallback scores, no cluster table; a pinned template that has gone raises, and scores are untouched; `jobs=2` equals `jobs=1` byte for byte in every table; `native` after render fills `clusters_native` and leaves `clusters.parquet` byte-identical; `Run.zmap` equals a direct computation |
| render schema 3 | with the template, `{h}_fsavg6` matches the closed-form inverse of the warp and the pole distance is as constructed; the sphere files are in `inputs`; without it everything is -1 and the tiles are byte-identical; a template added later rewrites two files only (tile mtimes unchanged) and leaves `render_key` unchanged; a schema 2 `render.json` is not accepted as complete; a `sphere.reg` of another vertex count leaves that hemisphere at -1; `three_plane` with `"auto"` next to an unusable `fsaverage` renders without counterparts; `--fsaverage` and `--no-template` |
| `protocol`, `calibration` | the shipped file validates; each rule violated once; the render reasons are covered; `marks_required`; `freeze_calibration` from final labels, refusal to overwrite, validation of uids, values and example slices |
| `app/marks` | every panel and slice: `voxel_to_pixel` then `resolve_mark` returns the voxel (hypothesis); a click at the projected position of a known pial vertex returns that vertex, its hemisphere, a distance under half a voxel and the expected template vertex; two hemispheres 3 mm apart with a mark between them: pial and white vertex are in one hemisphere for classes A and C, and class C takes hemisphere and `fsavg6_vertex` from the white surface; no pial vertex in range but a white one; background gives null vertices; `box_dist_mm` inside and outside the box; frame and slice mismatch; no vertex map; `marked_vertices` against brute force; a snap radius above the vertex-map margin is refused at start |
| `app/db` | a new file gets every table; a test migration preserves rows; a newer schema, another run and another mode are refused |
| `app/queue` | score order with the partial-score block first; `--blind` order for a fixed seed; idempotence; a failed render is never offered; a subject whose `render_key` changed comes back, one whose render was only re-encoded does not; an outdated render is not ready; `rendering` while nothing is ready; two local raters each get every subject; local repeats when `repeat_rate > 0`; (M5) even coverage with three simulated raters, repeats respect the gap |
| `app/finalize` | one rating: `single`; two within one step: the lower value, union of marks; two steps apart and unratable against rated: open, no label; adjudicated rows untouched; a failed render gets its unratable label; the second rater's arrival removes per-submission labels |
| `app/server` (`TestClient(app, base_url="http://127.0.0.1")` on `run_copy`) | session, next, rating, back, review, subject, events, stats: status codes and the shapes of `app/schemas.py`; each rule of 7.5 with the field it names, and a 422 stores nothing; a revision creates revision 2, keeps revision 1 and sets `deleted_ts` on its marks; `back` shows only the most recent assignment; 409 on a changed protocol version; media: whitelisted paths only, `..`, other names, `render.json` and `vertex_map.npz` give 404; another `Host` gives 403; the gate: with `blind`, and with two raters, `/review`, `/api/review`, `/api/stats`, `/api/export` give 403 until the rater has nothing left, and without either they are open; blindness: no response of the rating routes contains a subject label, a path, a `scores` column name, a flag or the key `inputs` (a recursive scan of the JSON), and the `/api/next` payload of a repeat differs from that of a new subject only in ids; `start_server` returns a handle with the real port and `stop` ends it |
| `export` | documented columns; a run without ratings and one without `app.sqlite`; CSV and parquet agree; `import surfcensus.export.tables` works with `sys.modules["sqlmodel"] = None`; `selection` has two rows for a subject drawn twice; `release` leaves out what 7.8 says and `analysis` does not; `manifest.json` hashes match the files |
| `review` | `review(..., serve=False)` on a `write_tree` of three subjects with a template, then again without flags: `config.yaml` byte-identical and `render_subject` called zero times; with `--video` then without: still nothing redone; `--blind` then no flag keeps blind; changing blind after a rating is refused without `--force` and recorded with it; then `export`; without a template; the source tree unchanged |
| `privacy` (M5) | key file permissions and refusals; collision lengthening with a forged pair; existing pseudonyms never change; `reveal_table` fails on a wrong key; `write_reveal` refuses a directory inside the run; `RedactingFilter` on the canary strings; the `ast` test of 5.1 |
| `remote/s3`, `remote/cache` | pagination; a dotted bucket uses path style; non-GET refused before I/O; retry on 503 and `SlowDown`, none on 404; an `If-Match` mismatch raises `changed`; MD5 mismatch; a lease deletes on exit and keeps retained roles; the byte budget blocks (peak size asserted); two `Cache` objects on one root in two processes: `prune` of one leaves the other's leased files; `prune` clears the area of a dead process; no directory name under the cache can be computed without the secret |
| `sources/openneuro_derivatives` | on the fake bucket: both layouts, a dataset without FreeSurfer, a dataset with a subject prefix that fails the pattern (`fs_subject_found` 0), `pial.T1` only, template detection |
| `enrich` | inheritance order of sidecars; the column mapping; an unmapped file; every file read has a `dataset_files` row; the snapshot tag, and null when GraphQL fails |
| `metrics/intensity` | a synthetic subject with a bright shell outside the pial surface over the posterior cap: `outer_outlier_frac` above the control level there and near 0 without it; `gap_frac` with a displaced pial patch; a hemisphere without regions gives nulls; the render commit writes the part |
| `select` | probabilities sum to the sample size; the prevalence sample is unchanged when scores are shuffled; adding a dataset leaves the other strata's draws unchanged; the minimum per dataset; deciles are formed on rank; a subject in both samples has two rows; the seed is recorded |
| team mode | no cookie gives 401; a token gives a cookie that works and whose value appears nowhere in the database; an inactive rater is refused; every unblinded route gives 403 to a lead and to an adjudicator who still have subjects; calibration: consensus only after the rater's own submission, and no revision afterwards; adjudication writes final labels and marks; `finalize` applies the one-step rule; `app/agreement.py` equals `scikit-learn` and `krippendorff` on random ratings; `errormap.png` is a PNG of the expected size |
| `calibration_transfer` | a set exported from a local run is found in a census run over the fake bucket by size and MD5; an unmatched and a doubly matched subject are errors; the portable file is refused inside a run directory |
| census end to end | fake buckets with 3 datasets of 4 subjects plus a local source: `discover` to `export` through the CLI; peak cache size under the budget; a kill in the middle of `render` and a rerun give the same bytes as an uninterrupted run; the canary search of 5.1, including the failure path and the ordering check; an object changed between inventory and metrics yields `source changed` for that subject only |
| `Snakefile` | `snakemake -n` builds the DAG; every rule's command is a `surfcensus` subcommand that exists |

Playwright smoke test of the real page (`tests/test_browser.py`, marked `browser`; the installed Chrome through `channel="chrome"`): `review(tree, serve=False, require_marks=True)` on two synthetic subjects, then `handle = start_server(run_dir, port=0)`, the browser pointed at `handle.url`, and `handle.stop()` at the end.

1. The page loads with no console error and no failed request.
2. Three panels show; each has a grayscale and a contour image of non-zero natural size.
3. `Space` starts playback and the frame slider moves; `Space` again stops it; `ArrowRight` while paused advances exactly one frame; `m` switches the mode.
4. Holding `s` hides the contour layer and releasing shows it.
5. `Enter` is disabled before any rating; pressing `3` sets A to major and makes class A active.
6. A click at a pixel computed from the `render` object for a known vertex puts a marker there; it appears in a second panel once that panel's slice is within range; `Enter` enables only after the click and after B and C have values.
7. `4`, `7`, then `Enter` submits; the progress line counts one; a new subject's tiles load.
8. `Backspace` returns to the first subject with its values and its mark.
9. The mark stored in `app.sqlite` has the expected vertex. The page is tested at two window widths, so `px = ox * width / w` is exercised through the real page scaling.
10. `/review` lists the rated subject with its values (one rater, not blind, so the page is open).

### 9.3 Network tests on the real data

`REAL_SUBJECT_FILES` grows by Tier A and B, with the sizes of the listing of 2026-11-12: `stats/lh.aparc.stats` 6,717; `stats/rh.aparc.stats` 6,717; `stats/aseg.stats` 9,853; `scripts/build-stamp.txt` 58; `scripts/recon-all.log` 220,842; `scripts/recon-all-lh.log` 212,143; `scripts/recon-all-rh.log` 206,247; `surf/lh.thickness` 593,343; `surf/rh.thickness` 587,803; `surf/lh.sphere.reg` 5,343,555; `surf/rh.sphere.reg` 5,293,695; `label/lh.aparc.annot` 1,188,169; `label/rh.aparc.annot` 1,177,089. A fixture `real_template` fetches the eight required template files of the adjacent `fsaverage` (about 39 MB), its two `surf/?h.white.avg.area.mgh` files, and `fsaverage5/surf/lh.sphere.reg`. A fixture `real_subject_fs7` fetches `ds003020` sub-UTS01 (FreeSurfer 7.3.2; `surf/lh.pial.T1` 4,731,523 and no `lh.pial`). Sizes of the files not listed here are recorded when the fixtures are written.

On `ds000102` sub-01 with its template:

- inventory: complete, Tier A and B present; FreeSurfer 6.0.1, major 6; 4 invocations in `recon-all.log`; no T2, FLAIR, hires, expert or edit; finished; holes 41 and 26 in the log and in `aseg.stats`; truncations (39, 225) on `lh` and (93, 434) on `rh`; `recon_flags` holds no `/` and no `ds000102`; the source tree unchanged.
- stats: `MeanThickness` 2.39047 and `NumVert` 140,304 on `lh`; `CortexVol` read despite the malformed line; 34 parcels per hemisphere; ratio 0.811 within 0.001.
- template: 40,962 vertices and 81,920 faces per hemisphere, degrees 5 (12 vertices) and 6; `lh` cortex 37,487; pole vertex 12950; `lh` search 4,856, medial 1,851, inferior 1,405, lateral 1,600, control 5,789; `area_source` is `white.avg.area` and the `lh` areas sum to the file's own total (about 82,220 mm2 by the reviewer's measurement, fixed when the fixture is written), while with the file withheld the total is the white-mesh area, 65,096 mm2 within 1; `coarsen_faces` once more equals the faces of `fsaverage5/surf/lh.sphere.reg` exactly.
- mapping: at most 10 fallback targets per hemisphere (3 and 2 measured); at least 99.5% of the template cortex vertices get a value (37,420 of 37,487 on `lh`); the `lh` map has mean 2.43 mm within 0.02 and a global median between 2.3 and 2.6 mm; whole-hemisphere counts at exactly 5.0 are 16 (`lh`) and 41 (`rh`); the `lh` native search region has 21,484 vertices, none at 4.9 mm or more, so its ceiling fraction is 0.
- render with the template: no -1 among the `fsavg6` entries; `fsaverage_poles` present, the `lh` distance below 3 mm (about 1 mm in the prototype) and both printed (the first data for D12); the tiles byte-identical to a render without the template; time and size bounds as in the existing test.
- agreement with `mri_surf2surf`: skipped until `tests/data/surf2surf_ds000102_sub-01_lh.npy` exists. To produce it, with FreeSurfer 6 or 7 and the subject and its `fsaverage` in one `SUBJECTS_DIR` (flags from memory of the tool's help, to check against it):

  ```
  mri_surf2surf --srcsubject sub-01 --trgsubject ico --trgicoorder 6 --hemi lh \
      --srcsurfval thickness --src_type curv --trgsurfval lh.thickness.fsaverage6.mgh
  ```

  The test reports the mean and maximum absolute difference in cortex and asserts a tolerance that is set once the first comparison has been seen.

On `ds003020` sub-UTS01: the pial variant resolves to `pial.T1`; version 7.3.2; 5 invocations; `t2pial` False although the log contains `-noT2pial`; `lh` truncations (22, 230); thickness maximum exactly 5.0; it renders (the first run of the renderer on FreeSurfer 7 data, which closes the caveat in `CLAUDE.md` or opens a bug).

On the bucket (milestone 5, one request each): `fmriprep/` lists at least 308 dataset prefixes; one subject listing is a single page; a ranged GET returns 206 and the bytes; a GET with a wrong `If-Match` raises `changed`; `x-amz-version-id` is present; the dotted raw bucket lists by path style; `remote_head` returns 40 hex characters; inventory through `LocalSource` on the cached files equals inventory through `OpenNeuroDerivatives`, role by role, apart from `etag_or_mtime`.

End to end on the real subject (`architecture.md` asks for it): the cached files of sub-01 are copied into a `SUBJECTS_DIR` layout with the template next to them; `review(path, serve=False)` then inventories, scores (D16 path: one subject), renders with counterparts and writes the selection; a second call does nothing; `export` writes every table of 7.8; the source tree is unchanged throughout.

---

## 10. Doc changes that go with the code

Made in the same change as the code that implements them. None is made by this contract.

- `CLAUDE.md`, verified facts: the findings of section 3 with their date; "nothing here is checked for FreeSurfer 7" removed for the items now checked on one 7.3.2 subject (clamping, pial keys, log layout) and kept for the renderer; "fsaverage6 by truncation" corrected to vertices by truncation, faces by coarsening; status and planned commands. Hard rules: one sentence added to the pseudonym rule, that no unkeyed hash of bucket metadata (key, size, ETag) leaves `catalog.sqlite` either.
- `data_sources.md`: G1 to G5; the two new roles; tier sizes; versions vary; `?h.pial` keys absent in FreeSurfer 7 derivatives; the ETag is the MD5 for these objects; version id on GET; derivative ref from the GitHub mirror; path-style addressing for the raw bucket; dataset-level files are recorded in `dataset_files`.
- `architecture.md`: catalog tables and views (5.5), `dataset_files`, the run secret; run directory (5.6); `Source` with `lease` and `config`; `Run` and `RunContext`; `privacy.py` owning `subject_uid`, `render_key`, cache names and `scrub`; the cache keyed, with per-process lease areas; new modules (S12); CLI additions (S43), `select` with two sample sizes, `export --release` and `--reveal-out`; thickness-only `fsavg` files; the shipped reference as `.npz` plus JSON; `imageio` removed from the dependency list (and from `pyproject.toml`); a `workflow` dependency group.
- `metrics.md`: G2, G6; the region definitions used; cortex mask by truncation; the missing-value rule; which subjects build a stratum's reference; the null rate of z > 3 at small n and `min_vertex_n`; cluster area from `white.avg.area`; `excess_mass`; the native region for the ceiling fraction and the second, annotation-based fraction; truncation counts as the companion of the ceiling fraction; cluster order; centroids to native space through the vertex map, by nearest vertex; the composite: population, normal scores, minimum-rank ties, partial scores; D16 ordering; the intensity metric's place (milestone 5, inside phase 6).
- `phases.md`: G7; Tier A contents; phase 1's number is a probe and phase 2's is exact; phase 6 renders all complete subjects by default, computes the intensity rows, and its exit check is drawn by `render --check-sample`; phase 7 gives failed renders an unratable final label; phase 8 names the two export profiles.
- `rendering.md`: the vertex map as the code has it (clickable region plus `vertex_map_margin_mm`, `{h}_visible`); schema 3 fields; the template as an input; the two-file update; `sphere.reg` problems are not failures; `render.json` is not served as a file and the API copy has no `inputs`; `tile_hash`.
- `rater_app.md`: `submissions`, `render_key`, `marks.slice`, `mode`, `vertex_surface`, `box_dist_mm`; one hemisphere per mark; who converts pixels; the routes of 7.5 and the rule for unblinded routes; repeats reported as `main`; `back` shows the most recent assignment only; local raters and `finalize`; `calibration.yaml`; the cookie; `--serve-after`.
- `rating_protocol.md`: where consensus marks and justifications live; that a calibration rating cannot be revised once its consensus was shown.
- `analysis_plan.md`: the evaluable vertices (what the tiles show, with the box as a filter); positives defined in native space from the mark and the vertex map, then carried to fsaverage; `vertices.parquet` and `error_map.parquet`; selection weights from the `selection` table.
- `decisions.md`: D20 (privacy boundary by construction: `PublicCatalog`, `RunContext`, the run secret, no unkeyed hash of bucket metadata outside the catalog, ordering by pseudonym), D21 (ledger and input signatures), D22 (template built at run time, pinned by content, never silently dropped), D23 (full logs in Tier A), D24 (render schema 3, `tile_hash`, keyed `render_key`), D25 (z-maps on demand; native centroids from the vertex map), D26 (local `review`: sticky options, no video, re-rating after re-render), D27 (composite: run population, normal scores), D28 (`release` and `analysis` export profiles), D15 amended; O3 and O10 updated with the evidence; the items of section 13 added as open until answered.

---

## 11. Reviewer problems and what was done with each

Every problem was checked against the draft and, where it rests on code or data, against the working tree or by running the numbers again (section 3). All 54 (P1 to P21, SCI-1 to SCI-13, B1 to B20) are right in substance; none was rejected. 49 were taken as proposed or with a small change; 5 (P1, P10, SCI-5, SCI-11, B18) were taken with a different mechanism than the one proposed, for the reason given in their row.

| Id | Verdict | What changed, and where |
|---|---|---|
| P1 | accepted; mechanism differs | `render_key` is keyed and content-based (5.1), stored in `renders.render_key`; `tile_signature` is gone and `input_signature` never leaves the catalog (section 4, 5.5). The key is a per-run secret in `catalog.sqlite` instead of the dataset key, so `render` and `serve` need no key file. `render.json` is not served as a file and the API copy has no `inputs` (7.5); T27. |
| P2 | accepted | `dataset_id` is private: out of both views, parts and work units named by pseudonym, all per-dataset output ordered by pseudonym (section 4, 5.5, 8.8); the canary test checks ordering (5.1). |
| P3 | accepted | `export_run` has no reveal argument; `privacy.write_reveal` is called from `cli.py` with a required `--reveal-out` outside the run (7.8, 8.1). `PublicCatalog.open` (5.5). Phase functions take a `RunContext`, `Run` hides the catalog, and the `ast` test covers attribute access (5.1, 5.9). |
| P4 | accepted | `scores.parquet` key (`subject_uid`, `reference_id`) with `primary` (6.10); T4 reworded. |
| P5 | accepted | `metrics/intensity.py` specified (8.9) and built in milestone 5; the worker hook `RenderResult.extras` exists from milestone 2 (5.9); `gap_frac` goes with it. T31. |
| P6 | accepted | `kind` is `calibration` or `main` only; repeats are reported as `main` (7.5); blindness test extended (9.2). |
| P7 | accepted | Tri-state, sticky options (5.4, 7.7). Changing `blind` after a rating needs `--force` and is recorded as an event. |
| P8 | accepted | One gate for every unblinded route and every role (7.5); several local raters and `finalize` in milestone 4 (7.2, 7.7); T29. |
| P9 | accepted | `privacy.scrub` (5.1) on every stored or printed failure text; no `detail` through `PublicCatalog` in a census run; `RedactingFilter` redefined (8.1); a local source in a census run keeps its root in the catalog only (5.4); canary run with a local source and a failing subject. |
| P10 | accepted; shape differs | A T-item (T26). Instead of thinning the one census export, there are two profiles: `analysis` (full, for the team's notebooks, which `analysis_plan.md` has read only from the export) and `release` (the coarsened table the reviewer describes) (7.8). Thinning the only export would have left the notebooks without the unselected subjects. |
| P11 | accepted | One hemisphere per mark; the white vertex is searched in it (7.3). Merged with SCI-6. |
| P12 | accepted | `dataset_files` in migration 1 (5.5); `enrich` writes it and `snapshot_tag` (8.4); its signature in the stage table (section 4). |
| P13 | accepted | `calibration.yaml`, `calibration.py`, `calibration freeze` (7.7); `examples` in `/api/session` (7.5); no revision after the consensus was served (7.2, 7.5, 8.7). Added: how the set moves from the pilot run to the census run (8.10), which the reviewer's proposal left open. |
| P14 | accepted | `render --all / --selected`, `select.render_scope` defaulting to all, `render --check-sample` (5.9, 5.10, 8.5); T28. |
| P15 | accepted | `selection` export table; `--n-prevalence`, `--n-detectability` (7.8, 8.6). |
| P16 | accepted | `Reference.metric_norms` as percentile grids (which fit the normal-score composite) and `parcel_norms`; `quantile_scores`; `SourceConfig.field_strength`, `--field-strength` (5.4, 6.8). |
| P17 | accepted | Cache names keyed with the run secret, and `scrub` as well (5.1, 8.2). |
| P18 | accepted | Unratable final labels for failed renders (`sync_unratable`, 7.2); `review` selects every present subject and `run_render` records `missing surface` for incomplete ones (5.9, 7.7). |
| P19 | accepted | Inventory retains the stats files, so 4a is offline (5.8, 8.8); `CacheConfig.retain_roles` (5.4, 8.2). |
| P20 | accepted | `vertices.parquet` and `error_map.parquet` named, written in milestone 6 (7.8). |
| P21 | accepted, all eight | (1) nearest vertex stated as intended (6.7); (2) error-map image, drift and the team pages (8.7); (3) `app/agreement.py` in numpy; (4) the phase 1 probe (8.3); (5) `.npz` plus JSON stated (6.8); (6) `--fsaverage`, `--no-template` (6.11); (7) network test of `review` then `export` (9.3); (8) local repeats (7.4). Each is in section 12. |
| SCI-1 | accepted | `ScoringConfig.population`, default `run` (5.4, 6.8); T24. |
| SCI-2 | accepted | Normal scores by default; `robust_z` kept with its conditions written out (6.8); T5 reworded with both failure directions. Numbers reproduced (section 3). |
| SCI-3 | accepted | Minimum-rank ties in `normal_scores` and `fallback_rank`; `add_rank` tie rule; deciles on rank (6.8, 8.6). |
| SCI-4 | accepted | A stratum's reference is built from every subject that matches its definition (6.6), with the reviewer's test. |
| SCI-5 | accepted; mechanism differs | Positives and error maps use the neighbourhood of a mark in native space (`marked_vertices`, 7.3; `vertices.parquet`, 7.8). No `mark_vertices` table: the point and the vertex map are both kept, so the set is recomputed where it is used instead of being stored twice. T30. |
| SCI-6 | accepted | As P11, and class C marks take hemisphere and `fsavg6_vertex` from the white surface; `vertex_surface` records which (7.3). No separate `white_hemi`, since the two vertices now always share `hemi`. |
| SCI-7 | accepted | Optional `white.avg.area.mgh`; `area_source` (6.3); T10 corrected. The file's presence next to the real subject was checked. |
| SCI-8 | accepted | Winding reversed when most faces are inverted, `SphereMap.flipped`; `max_fallback_frac` (6.4, 6.6). |
| SCI-9 | accepted | `excess_mass` beside `mass`, the composite uses the excess by default; `cluster_min_vertices`; `min_vertex_n` 20 (5.4, 6.7); T25. |
| SCI-10 | accepted | `n_components`, `partial_score`, the annotation-based ceiling fraction for every subject (6.8, 6.9, 6.10); partial scores first in the open queue (7.4). |
| SCI-11 | accepted; mechanism differs | `members_sha256` covers each member's map file digest (6.8) instead of the fsaverage input signature, which is private under P1. The effect is the one asked for: changed surfaces give a new `reference_id`. |
| SCI-12 | accepted | As P16(c), with the lookup order of `applicable_stratum` (6.8). |
| SCI-13 | accepted | `box_dist_mm` replaces `in_box` (7.2, 7.3). In the working tree the vertex map now reaches the snap radius beyond everything the tiles show, so every mark keeps its vertex; the app refuses a snap radius above that margin. |
| B1 | accepted | The synthetic template is order 7 by default and passes through the default path; small meshes by name (`fsaverage4`, `fsaverage5`) for fast tests (5.4, 6.3, 9.1); `three_plane("auto")` treats `TemplateError` like not found (6.11). |
| B2 | accepted | `records.py` and `schemas.py` in step 2.0; `TEMPLATE_FILES` in `sources/patterns.py`; `run_inventory` takes its inputs, not a `Run`; `template` and `zmap` join `Run` in milestone 3; `inventory/__init__.py` owned (5, 5.8, 5.9, 14). |
| B3 | accepted | `remote.default_transport`, `transport` arguments, `OpenNeuroDerivatives(client=...)`, `FakeS3` over several buckets and the two HTTP services (5.9, 8.2, 8.3, 9.1). |
| B4 | accepted | `PublicCatalog.open`, `subject_uids`, `set_file_digests` (5.5); `--dataset` resolved by `Run`, to pseudonyms rather than ids because of P2; only `discover` and `inventory` need the key (5.1). |
| B5 | accepted | `from_config` in `run.py`, `sources/__init__.py` lazy, the import rule and the one-process-per-module import test (section 4, 5.3, 9.2). |
| B6 | accepted | `metrics/__init__.py` and `reference/__init__.py` in step 3.0; `ReferenceMaps` in `reference/load.py`; ceiling after templates (6.6, 6.8, 14). |
| B7 | accepted | Shapes and model classes in step 4.0; `small_run` built without `review`, with a cluster; `run_copy`; server after export (7.3, 9.1, 14). |
| B8 | accepted | As P3. |
| B9 | accepted | As P7, with the idempotence test (9.2). |
| B10 | accepted | The file checked per stage, the meaning of `failed`, commit order, `--dataset` with run-level stages, `enrich` and `select` rows, absent subjects dropped at merge (section 4, 5.5). |
| B11 | accepted | The pin comes first; a run with scores raises instead of falling back when its template is gone (6.3). |
| B12 | accepted | `RenderConfig.tile_hash()` (5.9); readiness by `config_hash`; `serve` exits 2 on out-of-date renders (7.4, 7.5). Combined with P1: the key is `HMAC(tile_hash, content digests)`. |
| B13 | accepted | `allowed_hosts`, host name only; `ServerHandle`, `start_server` (7.5); the browser test uses them (9.2). |
| B14 | accepted | `StatsOut`; `prefetch` as objects; `back` without `before`, in the `next` envelope; `examples` (7.5). |
| B15 | accepted | Only `app/` imports `sqlmodel`; `MIGRATIONS` for `app.sqlite`; mode check; busy timeout in a local run (5.5, 7, 7.2). |
| B16 | accepted | The cookie is the token, no `SessionMiddleware`; agreement statistics in numpy; `workflow` group; `importorskip`; `imageio` removed (8, 8.7, 9.1, 10). |
| B17 | accepted | Per-process lease areas named by host and pid, which also covers two commands on one dataset at once; `prune` touches only areas of dead processes (8.2). |
| B18 | accepted; mechanism differs | `clusters.parquet` is immutable after the reference stage and `clusters_native` is its own table, as proposed; `metrics --stage merge` replaces `rule collect`. But the centroids are mapped through `vertex_map.npz` in a stage of their own (6.7, 6.9), not in the render worker: no source file is needed, so a reference rebuilt after rendering (the second pass) costs no download. |
| B19 | accepted | `Source.config()`; `SourceConfig.identity`, `dataset`, `bucket`, `prefix`; `LocalSource(identity=...)`; `discover --source local --path`; the absolute-path rule exempts a local run's `config.yaml` (section 4, 5.2 to 5.4, 5.9, 8.4). |
| B20 | accepted | The four log keys and the committed excerpt script (9.1); `missing surface` rows for every incomplete subject (5.9). |

---

## 12. Where the docs are silent or disagree, and the choice made

From the draft. Contradicted by what was found (the docs need correcting, section 10):

| # | Doc statement | Finding | Choice |
|---|---|---|---|
| G1 | `data_sources.md`: flags come from the "first lines of `recon-all.log`"; Tier A fetches its first 64 kB | Under fMRIPrep the log holds 4 or 5 invocations; 64 kB shows 2 | Parse every invocation of the three logs; fetch them whole by default (`inventory.logs = "full"`); `"head"` stays an option with `finished = None` |
| G2 | `metrics.md`: the truncation line "ends up in `recon-all.log`" | Under fMRIPrep it is in `recon-all-lh.log` / `-rh.log` only | Two new roles `recon_log_lh`, `recon_log_rh` |
| G3 | `data_sources.md` "(to verify)": one fMRIPrep and FreeSurfer version throughout | No: 21.0.1 to 25.1.4; FreeSurfer 6.0.1 and 7.3.2 | Versions per subject and dataset; T7 |
| G4 | `data_sources.md` "(to verify)": how `?h.pial` links appear on S3 | Absent; only `?h.pial.T1` exists | The existing candidate order suffices; `pial_variant` is recorded |
| G5 | `data_sources.md`: Tier A about 15 kB, Tier B about 12 MB, annot "small" | annot is 1.05 to 1.19 MB per hemisphere; Tier B about 14 MB; Tier A with full logs 0.7 to 1 MB | Numbers and cost estimates updated |
| G6 | `metrics.md`: the search region is "shipped as a label with the package"; `metrics.md` and `CLAUDE.md`: fsaverage6 "by truncation" | Shipping conflicts with the condition on D14; truncation gives vertices, not faces | Regions computed at run time from the template; faces by `coarsen_faces` |
| G7 | `phases.md`: phase 4 input is "complete subjects"; `metrics.md`: regional metrics are "available even when surfaces are missing" | The two disagree | `complete` keeps its `phases.md` meaning (renderable). 4a runs on `tier_a`, 4b on `tier_b` |
| G8 | `rater_app.md`: "the client converts a click to voxel coordinates"; its API posts `(panel, frame, px, py)` | Two contracts; `frame` does not determine the slice without the mode | The client posts panel, slice and pixel; the server converts and is the authority; `slice`, `frame`, `mode` are all stored |
| G9 | Hard rule: logs use pseudonyms; `architecture.md`: `--dataset ds000102` | A Snakemake wildcard or config entry with an accession would put it in file names | `--dataset` takes accession or pseudonym; nothing with an accession is stored outside the catalog |
| G10 | `architecture.md`: `fsavg/<uid>.npz` holds "thickness (and z) maps" | z depends on the reference, which changes with the set | Thickness only; z on demand (`Run.zmap`) |
| G11 | D14 order against O10's default | | D14 order at first use, then pinned by content; T13 |
| G12 | `architecture.md`: `review` "renders every subject" before serving; `rendering.md` lists `anim.mp4` in every render | 500 subjects take 10 to 75 minutes; the app never plays the video | Default as documented for serving; `--serve-after N` exists; no video in `review` unless `--video`; T20 |
| G13 | `architecture.md`: a subject is a child "containing `surf/` and `mri/`"; the code requires T1 and a left pial surface | | Both: the first finds candidates, the second (with the other render roles) marks them complete; incomplete ones are reported |

Silent, with the choice made:

| # | Topic | Choice |
|---|---|---|
| S1 | `subject_uid` format | 16 hex of HMAC-SHA256 with a `subject` domain prefix; a collision is an error |
| S2 | "Root path" when local data moves | `datasets.identity` fixed at first inventory; `--relocate` updates `root` only |
| S3 | A single subject directory as `PATH` | Accepted (layout rule 1) |
| S4 | `--subjects` semantics | Globs on directory names, union; they define the run's subject set and are stored; no match is an error |
| S5 | A subject that disappears from the source | Keeps its row, media and ratings with `present = 0` |
| S6 | A directory that cannot be read | Inventoried with `problem`, reported, never dropped silently |
| S7 | Default run directory inside the subjects directory | Moved one level up; a run directory inside the subjects directory is refused |
| S8 | One source per run or several | Several (`RunConfig.sources`); `review` creates one |
| S9 | `files.location`, extra file columns | Relative to the subject directory; `variant`, `last_modified`, `version_id`, `sha256` |
| S10 | Derivative commit, snapshot tag, subject labels | Private in a census run; labels shown in a local run outside the rating screen |
| S11 | Catalog tables not in `architecture.md` | `meta`, `templates`, `work`, `dataset_files`; extra columns in `datasets`, `subjects`, `renders`, `selection`; three public views |
| S12 | Modules not in the package layout | `_io.py`, `log.py`, `records.py`, `schemas.py`, `run.py`, `batches.py`, `review.py`, `protocol.py`, `calibration.py`, `calibration_transfer.py`, `render/batch.py`, `metrics/pipeline.py`, `fsio/morph.py`, `app/{schemas,db,events,finalize,auth,adjudication,agreement,errormap}.py`, `remote/gitrefs.py` |
| S13 | Concurrent writers | One writer per process, short transactions, per-dataset parquet parts merged afterwards; WAL only in a census run |
| S14 | How a phase knows its inputs are unchanged without reading them | `input_signature` from inventory metadata; size and mtime or `If-Match` checked at lease |
| S15 | Failed units | Recorded with a reason; not retried without `--retry-failed`, a changed config or changed inputs |
| S16 | The template needs the white surface too | For the cut, the pole and the areas; `metrics.md` lists sphere, cortex label and annotation |
| S17 | fsaverage6 cortex mask | Truncation of the fsaverage label; differs from FreeSurfer's own lower-resolution labels by a few vertices |
| S18 | Search region, split, control region | T2, T1, T12 |
| S19 | Where the global median is computed | On the stored fsaverage6 map, cortex outside the search region, per hemisphere |
| S20 | Missing values in resampling | Row 16 of section 2 |
| S21 | "Lightly smoothed" MAD, vertices per reference | T10 |
| S22 | The set reference and the scored subject | Included (no leave-one-out) |
| S23 | Field strength strata | Binned to the nearest of 1.5, 3, 7 T (a Prisma reports 2.89) |
| S24 | `vertex_summary.parquet` is named and never defined | Subject by hemisphere: resampling facts and ceiling counts (6.10) |
| S25 | Hemispheres in subject-level metrics | The larger of the two; the regional ratio pools both |
| S26 | Clusters | Positive z only; per region; five kept; order by area (T6); centroid on the sphere, z-weighted |
| S27 | Native coordinates of centroids | From the vertex map of the render, in the `native` stage, for clusters the tiles show; a table of their own (6.7, 6.9) |
| S28 | A thickness maximum other than 5 mm | Flagged (`thickness_above_max`), not handled |
| S29 | Two references for one subject | `clusters` and `scores` hold rows for both; the primary one is the shipped reference locally, the set in the census |
| S30 | Regional means | Weighted by `SurfArea`, both hemispheres pooled |
| S31 | D16 ordering; composite scaling | T11; T5 |
| S32 | The "fixed fsaverage pole vertex" of D12 | The most posterior cortex vertex of the template white surface, computed (12950 on `lh`) |
| S33 | Counterpart direction in `vertex_map.npz` | The template vertex nearest the native vertex on the registered sphere |
| S34 | Schema 3 and existing renders | All earlier renders are rendered again (none is rated); a later template change rewrites two files only; ratings follow `render_key`, which has no schema number in it |
| S35 | `sphere.reg` unusable at render | That hemisphere keeps -1; not a failure |
| S36 | Notes, unratable, revisions in the ratings schema | `submissions` table; `ratings` rows hang off it |
| S37 | Marks outside the swept box | `box_dist_mm`; every mark keeps its vertex, because the vertex map reaches the snap radius beyond what the tiles show; pial and white vertex stored for every class |
| S38 | Re-render after rating | `assignments.render_key`; back in the queue (T19) |
| S39 | Failed renders in the queue | Never queued; listed in review and export with their reason, and given an unratable final label |
| S40 | `protocol.yaml` says marks are required | Local mode overrides through `app.require_marks`; the file is not rewritten |
| S41 | Events and final tables have no key | `event_id`, `final_mark_id` added; `app_meta` ties the database to its run |
| S42 | Final label for "unratable"; one-step disagreements | A `final_labels` row with criterion `unratable`; T3 |
| S43 | Commands and options not in `architecture.md` | `status [--failures]`, `finalize`, `calibration freeze / export / import`, `discover --source local --path`; `--source`, `--relocate`, `--retry-failed`, `--stage` (with `native` and `merge`), `--fsaverage`, `--retemplate`, `--field-strength`, `--require-marks`, `--video`, `--port`, `--jobs`, `--no-serve`, `--serve-after`, `--force`, `render --all / --selected / --check-sample`, `render-subject --fsaverage / --no-template`, `serve --mode / --rater`, `export --out / --release / --reveal-out`, `select --n-prevalence / --n-detectability`, `raters list` |
| S44 | `pydantic-settings` | Used only for `RuntimeSettings` in milestone 5 |
| S45 | `TestClient` with httpx is deprecated in Starlette 1.10 | Tests filter that one warning; if it becomes an error they switch to `httpx.AsyncClient` with `ASGITransport` |
| S46 | The `mri_surf2surf` comparison array | Needs a FreeSurfer install; the test exists and skips until the array is stored (T22) |
| S47 | `render-subject` with an output inside the subject directory (left open by the milestone 1 review) | Refused; built in milestone 1 when the review's fixes were verified |

Added in this revision. Docs that disagree with the code, with each other, or with a hard rule:

| # | Doc statement | Finding | Choice |
|---|---|---|---|
| G14 | `rendering.md`: the vertex map holds vertices "inside the box plus a 10 mm margin"; D15: evaluable vertices are those "inside that subject's rendered box (the vertices in `vertex_map.npz`)" | The working tree selects the map by the clickable region of the tiles plus `vertex_map_margin_mm` (5 mm), marks `{h}_visible`, and is at schema 2. The box and the map are no longer the same set | Resolved in the milestone 1 review: `rendering.md`, D15 and `analysis_plan.md` now state the rule of the code. Milestone 3 is schema 3. Evaluable vertices are the `visible` ones, those whose pial position lies in the region the tiles sweep; a mark can resolve to a vertex just outside it and is kept (`analysis_plan.md`). `box_dist_mm` on marks and the box corners in `render.json` allow the stricter reading as a filter (T30) |
| G15 | `architecture.md`: the cache is "content-addressed, keyed by (bucket, key, etag)"; hard rule: no unkeyed hash that can be inverted by enumeration | Key and ETag are in the public listing, so the directory name identifies the subject | Cache names are an HMAC with the run secret (8.2) |
| G16 | D10 and `architecture.md`: `export --reveal` writes the mapping table | Written into the export directory, it sits next to files meant for release | `--reveal-out DIR` is required and must lie outside the run directory (8.1) |
| G17 | D10: released per-dataset tables report the subject count binned; `phases.md`: the release holds "ratings, scores and provenance" per subject | A per-subject table with pseudonyms gives the exact count, and exact age and sampling probability match `participants.tsv` and the listing | Two export profiles, `analysis` and `release` (7.8); T26 |
| G18 | `metrics.md`: a cluster centroid maps back to native space "via the barycentric mapping" | An integer native vertex is what the renderer and the review screen use, and the barycentric mapping runs from native to template | Nearest vertex, through the vertex map of the render (6.7) |
| G19 | `architecture.md`: the shipped reference is "parquet + JSON metadata" | Per-vertex maps are arrays | `.npz` per stratum plus `index.json` (6.8) |
| G20 | `architecture.md`: `select --n N --seed S` | Two samples with different designs and sizes | `--n-prevalence`, `--n-detectability` (8.6) |
| G21 | `phases.md`: the intensity metric runs inside phase 6; D18: it "follows milestone 4"; `architecture.md` lists `metrics/intensity.py` | Neither says which milestone, and a census rendered without it needs Tier C twice | Built in milestone 5 through a hook that exists from milestone 2 (5.9, 8.9); T31 |
| G22 | `phases.md`: "Selection determines the rating sample, not the rendering set"; phase 6 input "selected (or all) subjects" | The CLI of `architecture.md` has no way to say which | `select.render_scope`, default all; `render --all / --selected` (5.9); T28 |
| G23 | `rater_app.md`: "Repeats look identical to new subjects"; its `assignments.kind` has `repeat` | The API would show the kind | `kind` is stored and never sent; a repeat is reported as `main` (7.5) |
| G24 | Hard rule: raters never see scores or other ratings while rating; `rater_app.md`: review screens for adjudicators "only after their own assignment is done", and the review screen is what "a local user ends up using most" | Nothing covers a lead who also rates, `--blind`, or two local raters | One gate: unblinded routes are closed while the rater has work left; in local mode only under `--blind` or with several raters (7.5) |
| G25 | `metrics.md`: the composite is the maximum over "z-scored subject-level metrics" | Neither the population nor the method is named; within-dataset z removes between-dataset differences, and a robust z breaks on zero-inflated metrics | Whole run, normal scores, minimum-rank ties (6.8); T24, T5 |
| G26 | `architecture.md` and `pyproject.toml` list `imageio` as a core dependency | No module imports it (`imageio_ffmpeg` is the one used) | Removed in milestone 2 |
| G27 | `rater_app.md`: the census error map is "a static image regenerated on demand", next to the mean z-map; the dashboard shows drift | The draft had JSON arrays only and no drift | Both: `/api/errormap` and `/api/errormap.png`; `drift` in `/api/stats` (8.7) |
| G28 | `analysis_plan.md`: a vertex is positive within "5 mm geodesic on fsaverage" of a final mark, the mark being one fsaverage vertex | The nearest vertex of a click is unstable across sulcal banks (7.3) | Positives are defined in native space from the stored point and the vertex map, then carried to fsaverage (7.3, 7.8); T30 |
| G29 | `phases.md` phase 1: the count of "datasets with at least one FreeSurfer subject", by pattern | The pattern needs a subject listing; a prefix is not a subject | `discover` probes up to three subjects per dataset and says so; the exact count comes from inventory (8.3) |
| G30 | `phases.md` phase 4b: Tier B is deleted "unless retained by config or needed for rendering" | No such config existed | `cache.retain_roles` (5.4); T28 |
| G31 | `phases.md` phase 7 exit check: "all sampled subjects have final labels"; `rating_protocol.md`: unratable reasons include the renderer's | A selected subject whose render fails is never offered, so it never gets a label | `sync_unratable` writes the label from the render's reason (7.2) |

Silent, with the choice made:

| # | Topic | Choice |
|---|---|---|
| S48 | What identifies a render outside the catalog | `render_key`: HMAC with a per-run secret over the tile settings and the content digests of the tile inputs (5.1) |
| S49 | Order of datasets in anything written | By pseudonym; `dataset_id` and discovery order are private (section 4) |
| S50 | Exception text in tables and logs | `privacy.scrub` before storing or printing; no `detail` through `PublicCatalog` in a census run (5.1) |
| S51 | Where the source of an enrichment value is recorded | `dataset_files` (5.5, 8.4) |
| S52 | Where consensus marks, justifications and help-screen examples live; how the pilot's calibration set reaches the census run | `calibration.yaml` (7.7); matched by size and MD5 through a private portable file (8.10) |
| S53 | Who draws the 2% of renders for the phase 6 check | `render --check-sample`, seeded and recorded (5.9) |
| S54 | Which subjects a stratum's reference is built from | All that match its definition, assigned to it or not (6.6) |
| S55 | Vertex areas on the template; what "cluster mass" does at the threshold | `white.avg.area.mgh` when present; `excess_mass` beside `mass` (6.3, 6.7); T10, T25 |
| S56 | A sphere wound the other way; a registration with many unresolved targets | Winding reversed once and flagged; above `max_fallback_frac` no map (6.4, 6.6) |
| S57 | Subjects scored on fewer components than the rest | `n_components`, `partial_score`; first in the open queue; a second ceiling fraction on one definition for everyone (6.8, 6.10, 7.4) |
| S58 | Whether a flag given once holds on the next run | Yes: options are tri-state and stored; changing `blind` after a rating needs `--force` (5.4) |
| S59 | Two raters before team mode exists (the pilot) | Named local raters on one run directory, the gate of G24, and `surfcensus finalize` (7.7); T29 |
| S60 | Tier A for the regional stage in the census | The stats files are fetched and retained at inventory (5.8) |
| S61 | Per-vertex data for the notebooks | `vertices.parquet`, `error_map.parquet`, named now, written in milestone 6 (7.8) |
| S62 | Whether `render-subject` output may depend on the environment | `"auto"` stays the default and the template used is recorded; `--fsaverage` and `--no-template` fix it (6.11) |
| S63 | Repeats in local mode ("no repeats unless enabled") | Enabled by `app.repeat_rate > 0` (7.4) |
| S64 | A run whose pinned template is no longer reachable | An error that says how to fix it, never a silent fall-back to D16 (6.3) |
| S65 | Changing `app.sqlite` after ratings exist; opening it in the other mode | SQL migrations as for the catalog; another mode is refused (7.2) |
| S66 | The team session | The rater token itself as the cookie, looked up by hash on each request (8.7) |
| S67 | Where agreement statistics are computed | In the server, in numpy, tested against the `analysis` packages (8.7) |
| S68 | Mesh sizes below fsaverage6 | `fsaverage4` and `fsaverage5` by name, for tests and for O5 (5.4) |
| S69 | A mark's hemisphere when pial and white disagree; which surface gives the fsaverage vertex | The surface the mark's class is about decides; the other is searched in that hemisphere (7.3) |
| S70 | What the release profile leaves out | 7.8; T26 |
| S71 | `render.json` `inputs` (exact sizes and SHA-256) in what raters and a gallery receive | Never sent: the file is not served, and the API copy has no `inputs` (7.5); T27 |
| S72 | A render failure caused by the environment (no ffmpeg) | Stops the phase with exit 2; not recorded against a subject (5.9; the milestone 1 review asked for this) |

---

## 13. Decisions for Tomas, each with the default that work proceeds on

None of these blocks milestones 2 to 4. Each has a default, and each default is a config value or a run-time computation, so a different answer means recomputing, not rewriting. T1 to T23 are the draft's items (T4, T5, T10, T11, T21 and T22 reworded); T24 to T31 are new in this revision. Within each group the order is by how much the answer changes results.

Scientific: these change the numbers in the paper

| # | Question | Default | Why it is yours |
|---|---|---|---|
| T24 | Over which subjects are the composite's components standardised? `metrics.md` says "z-scored" and names no population. | The whole run. Ceiling fraction and the occipital-to-control ratio are raw numbers that compare across datasets; the cluster masses already carry the per-dataset reference. | Within the dataset (the draft's reading) the typical subject of a dataset in which most reconstructions are bad scores 0 and ranks below a mildly odd subject of a clean dataset. Selection and the blind-spot rate are computed census-wide on this score. `scoring.population` switches it. |
| T5 | How are the components put on one scale when most subjects have the value 0 (cluster mass, ceiling fraction)? | Rank-based normal scores with minimum-rank ties; the maximum over components is the composite. | A robust z fails in both directions on such metrics: one subject at 1% ceiling fraction scores z = 144 when just under half the subjects are non-zero, and a real cluster of mass 50 scores 0.27 when just over half are zero (reproduced, section 3). Normal scores are bounded and give each component equal say, at the price that the composite is a rank statistic, not an effect size. `scoring.standardize = "robust_z"` keeps the other. |
| T25 | What is "cluster mass", given that noise alone puts z > 3 clusters in every subject (10 to 15 vertices per hemisphere of the search region at reference sizes of 30 to 15)? | The excess over the threshold, sum of (z - 3), which goes to 0 continuously; the plain sum of z (the wording of `metrics.md`) is stored beside it. No minimum cluster size. | A plain sum jumps from 0 to at least 3 for a single vertex, and small within-set references produce more null clusters than large strata. Alternatives: a minimum extent (`cluster_min_vertices`), or the plain sum (`scoring.mass`). Belongs with O11. |
| T1 | Medial and inferior parts of the search region. A split by surface-normal direction was tried and is poor (fsaverage6 `lh`: 57% of cuneus comes out medial, 67% of lateraloccipital falls in neither part). | By parcel. Medial: cuneus, pericalcarine, posterior precuneus (1,851 vertices). Inferior: lingual, posterior fusiform (1,405). Lateral: lateraloccipital (1,600), recorded and, by the wording of `metrics.md`, not in the composite. | A transverse-sinus capture on the inferior edge of lateraloccipital then reaches the composite only through the ceiling fraction and the regional ratio. Alternative: add the lateral cluster mass to the composite. |
| T2 | Search region on fsaverage6. | lateraloccipital, lingual, cuneus, pericalcarine whole, plus precuneus and fusiform vertices within 45 mm (along y, template white surface) of the template's occipital pole. `lh`: 4,856 vertices. 45 mm is the anterior extent of the render box; 50 mm gives 5,261. | Defines what "occipital" means in every score. |
| T3 | Final label when two raters differ by one step. The primary outcome is "major on A or B", so this rule moves the prevalence estimate. | The lower value, method `consensus` (what the protocol tells one rater to do when unsure). Two steps apart goes to adjudication. | Alternatives: the higher value, or adjudicate every disagreement. |
| T4 | `metrics.md` stratifies the per-vertex reference by dataset first. A dataset in which most subjects have the error (plausible at 7T) looks normal against itself. | Follow the doc for the z-maps. `clusters.parquet` and `scores.parquet` are keyed by `reference_id`, so scoring against a coarser stratum as well is more rows, not a new schema. (T24 already keeps the raw components comparable across datasets.) | Decides whether the first-pass cluster metrics can miss whole datasets. |
| T6 | "Largest outlier cluster": by area or by mass. | By area, as `metrics.md` words it; its excess mass is the composite component. | A small, extreme cluster ranks below a large, mild one. |
| T10 | Reference details `metrics.md` leaves open, and O11. | A vertex needs 20 subjects with a value (at 15 the null rate of z > 3 is 2.2 times the Gaussian one); MAD smoothed by 2 passes of one-ring averaging; MAD floor 0.05 mm; cluster threshold z > 3. Cluster area from FreeSurfer's average of subject areas (`white.avg.area.mgh`) summed onto the fsaverage6 mesh; without that file, from the fsaverage white mesh, which is smaller than real cortex by a factor that varies by parcel (0.59 in lateraloccipital to 1.01 in insula), so areas would not compare across parcels. | All are in the metrics config and its hash. |
| T9 | Ceiling fraction, given that a vertex reads exactly 5.0 only when both searches were truncated. | As `metrics.md`: fraction of native search-region vertices at 4.9 mm or more, or at exactly 0 inside cortex. Counts at exactly 5.0 and at 0 are stored beside it. Region and cortex on native surfaces come from the sphere mapping and the subject's own `aparc.annot` (not `cortex.label`, which would add 12.5 MB per subject). A second fraction on the subject's own occipital parcels is stored for every subject. | Definition of one of four composite components. |
| T30 | Spatial detectability: which vertices are evaluable, and where is the 5 mm tolerance measured? D15 says the rendered box; the vertex map now covers everything the tiles show. `analysis_plan.md` measures the tolerance on fsaverage from one vertex per mark, and that vertex jumps across sulcal banks for 7% of clicks shifted by half a millimetre. | Evaluable: vertices the tiles show (`visible`); the box-only reading is a filter. Positive: a native vertex within the tolerance of a final mark in native space, then carried to fsaverage6. `vertices.parquet` holds the distances, so 3, 5 and 10 mm and the fsaverage-geodesic variant can all be computed. | It defines the positives of the vertex-level ROC. Milestone 6 work; the stored data allow either answer. |
| T11 | D16 does not say how two raw numbers make one ranking. | The larger of the two within-set percentile ranks (ceiling fraction, occipital-to-control ratio), ties at the minimum rank. | |
| T12 | Frontal control region (`metrics.md` names none). | superiorfrontal, rostralmiddlefrontal, caudalmiddlefrontal; 5,789 `lh` fsaverage6 vertices; the same parcels for the regional ratio. | |
| T13 | D14 puts a FreeSurfer install first; O10's default prefers the copy next to the subjects. Evidence for O10: across FreeSurfer 6.0.1 and 7.3.2 derivatives the adjacent `fsaverage` has byte-identical `lh.sphere.reg`, `lh.aparc.annot` and `lh.cortex.label`; `lh.white` differs. | D14 order at first use. After that a run keeps the template it was first scored with, identified by content hash, and every score records that hash. The census pins one template for all datasets. | Resampling is the same everywhere; the region cut and vertex areas depend on the copy. |
| T7 | OpenNeuroDerivatives is not one pipeline version: in 15 datasets looked at, fMRIPrep 21.0.1 to 25.1.4, FreeSurfer 6.0.1 in 4 and 7.3.2 in 11. FreeSurfer version is tied to when a dataset was processed. | Record both versions per subject, stratify by FreeSurfer major version as `metrics.md` says, report by version with the confound stated. | Whether the paper presents version as a finding or a nuisance stratum. |
| T8 | Truncation counts ("N:M truncations") are in `recon-all-lh.log` / `-rh.log` under fMRIPrep, not in `recon-all.log`, and the flags are spread over four or five invocations of which the first 64 kB show two. | Fetch the three logs whole in the census (0.7 to 1 MB per subject, 3 to 5 GB for 5,000, deleted after parsing). Store the counts; keep them out of the composite; use them as a baseline in the detectability analysis. | Cost, and whether the count enters the score. |
| T31 | The intensity metric (phase 4d) has to run while T1 and surfaces are on disk in phase 6. | Built in milestone 5, before the census renders, with `gap_frac` (white-to-pial distance above 5 mm); stored as its own table; not a composite component. | If it were left for later, the census would download Tier C twice (about 135 GB more for 5,000 subjects). Say if it should enter the composite. |
| T14 | Unit of analysis when a dataset has several FreeSurfer directories per participant (sessions, `*.long.*`). | Every directory is one unit; the participant label is stored so the analysis can cluster on it; `*.long.*` directories are flagged, not excluded. | |
| T21 | Phase 5 numbers and O6. | Minimum 5 per dataset; top decile (of rank) sampled at 3 times the base rate; every subject above the 99th percentile; 2 raters per subject; the two sample sizes given separately (`--n-prevalence`, `--n-detectability`). Fixed before `select` runs at scale. | |
| T28 | Does the census render every complete subject or only the rating sample? | Every one, as `phases.md` says: about 195 GB of transfer, 25 minutes on 32 cores and 30 GB of media for 5,000 subjects, and the pole distance (D12) and intensity metric then exist for all. `sphere.reg` files are not retained between phases 4b and 6 (10.6 MB per subject downloaded twice, against 53 GB of disk to keep them). | Cost against completeness. `select.render_scope`, `cache.retain_roles`. |

Legal, privacy, ethics

| # | Question | Default | Why it is yours |
|---|---|---|---|
| T15 | D14: may the wheel carry fsaverage-derived files (spheres, cortex mask, region masks)? The same applies to masks inside a shipped reference. | Ship none. The template is derived at run time from an `fsaverage` directory and cached in the run directory. Without one, scoring falls back as in D16 and marks get no fsaverage vertex, with a message that says how to fix it. | Licence terms have not been read for this project. |
| T26 | What may the released subject table contain? D10 bins the per-dataset subject count, but a per-subject table with pseudonyms gives the exact count by a `groupby`, and exact age, sex, manufacturer and sampling probability match `participants.tsv` and the bucket listing by a join. | Two exports. `analysis` (full, pseudonymous, for the team's notebooks, not for release). `release`: selected subjects only; no age, sex, manufacturer, pipeline version or notes; field strength binned; subject count binned; sampling probabilities rounded to two digits; raters as `R-01`. | D10 calls pseudonyms goodwill, not secrecy, and accepts matching "with effort". This decides how much effort. If you prefer the full table released, D10 should drop the binning so the release does not suggest a protection it lacks. |
| T27 | `render.json` records the exact byte size and SHA-256 of each input, and sizes match the public listing. The draft served the file to raters and to any static media host. | Never sent: the app passes the geometry without `inputs`, and the file stays in the run directory. A gallery or `media_base_url` host gets tiles and video only. | The digests are what ties a render to exact source bytes (a hard rule), so they stay on disk; say if you want them published with the gallery. |
| T16 | Public gallery and released media redistribute T1 slices from OpenNeuro datasets; licences differ per dataset. | Not built in milestone 5; needed before phase 8. | |
| T17 | Rater identity in released data; release of event logs. | Raters exported as `R-01`, `R-02` in a census run; names stay in `app.sqlite`; events exported as per-assignment summaries only. | |
| T18 | Your 7T subjects in the pilot and calibration set: may other raters see them, and may they be reference images on the help screen? | They enter the census run as a second, local source under a pseudonym (`discover --source local`); shown to raters; not in any public gallery until you confirm. | |

Protocol and product

| # | Question | Default | Why it is yours |
|---|---|---|---|
| T29 | How do the two pilot raters work before team mode (milestone 5) exists? | On one run directory, one after the other: `surfcensus review PATH --rater NAME --blind`. Scores, the review screen and the other's ratings stay closed to each until they have rated everything. `surfcensus finalize` then writes the labels on which they agree within one step and lists the rest for discussion; the adjudication screen comes with milestone 5. | If you want both to rate at the same time from two machines, tokens and `serve --mode team` have to come forward from milestone 5 (a few days). |
| T19 | A subject whose surfaces changed after it was rated (edits, recon-all rerun). | It is rendered again and returns to the queue. The earlier rating is kept and marked as belonging to the earlier render; export reports the latest and flags the others. A file that was only touched or moved does not count as changed. | |
| T20 | Local `review`: (a) subject labels in the review screen and in exports; (b) `anim.mp4`; (c) when the app opens. | (a) Shown (the researcher has to find the subject again); rating stays blind. Never in a census run without `--reveal`. (b) Off unless `--video`: the app plays tiles, and the video costs 2.5 of 9 s and 4.3 of 10.2 MB per subject. (c) After every subject is rendered, as `architecture.md` says; `--serve-after N` opens it after the first N in queue order and renders the rest in the background (500 subjects otherwise mean a 10 to 75 minute wait). Say if (c) should be the default. | |

Only you can do these

| # | What | Until then |
|---|---|---|
| T22 | (a) The by-eye Freeview comparison that closes milestone 1. (b) One `mri_surf2surf` run on `ds000102` sub-01 for the resampling reference array (section 9.3). (c) One known-bad 7T subject for the pilot fixtures. (d) A look at the logs of one dataset processed with a T2 or FLAIR, to confirm where `-T2pial` appears. (e) After the pilot: the written justification for each calibration subject and the choice of help-screen examples, typed into `calibration.yaml`. | Tests that need (b) to (d) skip until the files exist; (e) leaves the texts empty. |
| T23 | O1 (name), O2 (reveal), O8 (hosting), O9 (embedded source), O12 (ratings licence). | Unchanged: pseudonyms only, embedded source deferred, `--reveal` exists, needs the key and writes outside the run. |

---

## 14. Order of implementation and work packages

Rules for every milestone.

- A package is a set of files owned by one agent, with the tests it owns. No file has two owners within a milestone. A package that needs a line in a file it does not own (a CLI option, a re-export, a fixture) hands the line to the integrator.
- One integrator per milestone owns the shared files: `cli.py`, `__init__.py`, `run.py`, `config.py`, `records.py`, `schemas.py`, `sources/base.py`, `sources/patterns.py`, `pyproject.toml`, `justfile`, `tests/conftest.py`, `CLAUDE.md`, `docs/`. The integrator works first, alone (step 0: every name that two packages need), and last (step Z: wiring, end-to-end tests, docs).
- Packages in the same wave depend only on step 0 and on earlier waves, so they can be written at the same time.
- Each milestone ends as `CLAUDE.md` asks: `just lint` and `just test` pass, a note in `docs/decisions.md` for every choice made along the way, docs updated in the same change (section 10).

### Milestone 2

| Package | Files | Tests it owns | When |
|---|---|---|---|
| 2.0 integrator | `_io.py`, `log.py`, `records.py`, `schemas.py`, `privacy.py` (5.1), `config.py`, `sources/base.py`, `sources/patterns.py` (`TEMPLATE_FILES`, candidates), `sources/__init__.py` (lazy), `inventory/__init__.py`, `render/encode.py` and `render/vertexmap.py` (helpers to `_io.py`), `render/model.py` (`tile_hash`), `__init__.py`, `pyproject.toml` (`imageio` out) | `tests/synth_tree.py`, `test_io.py`, `test_privacy.py`, `test_config.py`, `test_schemas.py`, `test_imports.py`, additions to `test_patterns.py` | first, alone |
| 2.A local source | `sources/local.py` | `test_local_source.py` | wave 1 |
| 2.B provenance | `inventory/provenance.py` | `test_provenance.py`, `tests/data/recon_logs/` with `make_excerpts.py` | wave 1 (pure text) |
| 2.C catalog | `store.py` | `test_store.py` | wave 1 |
| 2.D batches | `batches.py` | `test_batches.py` | wave 1 |
| 2.E inventory | `inventory/subjects.py` | `test_inventory.py` (builds a `Catalog` and a `LocalSource` directly; no `Run`) | wave 2, after 2.A, 2.B, 2.C |
| 2.F render batch | `render/batch.py`, `render/api.py` (the refusal of 5.10) | `test_render_batch.py` (a `RunContext` built by hand over a `PublicCatalog`), additions to `test_render_api.py` | wave 2, after 2.C, 2.D |
| 2.Z integrator | `run.py`, `cli.py` | `test_run.py`, `test_cli_run.py` (resume, relocate, exit codes, source tree unchanged) | last |

### Milestone 3

| Package | Files | Tests it owns | When |
|---|---|---|---|
| 3.0 integrator | `metrics/__init__.py` (`MetricError`), `reference/__init__.py`, `tests/conftest.py` (`synth_template`, `synth_cohort`, `small_cohort`) | `tests/synth_template.py`, `tests/synth_tierb.py` and a test that the fixtures are what they say (closed-form warp, masks) | first, alone |
| 3.A stats and regional | `fsio/stats.py`, `metrics/regional.py` | `test_stats.py`, `test_regional.py` | wave 1 |
| 3.B annotation and morph | `fsio/annot.py`, `fsio/morph.py` | `test_annot_morph.py` | wave 1 |
| 3.C registration | `fsio/registration.py` | `test_registration.py` | wave 1 |
| 3.D scoring | `metrics/scoring.py` (pandas and scipy only, against `schemas.py`) | `test_scoring.py` | wave 1 |
| 3.E reference files | `reference/load.py` (`ReferenceMaps`, `Reference`, read and write) | `test_reference_load.py` | wave 1 |
| 3.F template | `templates.py` | `test_templates.py` | wave 2, after 3.B |
| 3.G ceiling | `metrics/ceiling.py` | `test_ceiling.py` | wave 3, after 3.B, 3.C, 3.F |
| 3.H maps and clusters | `metrics/fsaverage.py`, `metrics/clusters.py` | `test_fsaverage_maps.py`, `test_clusters.py` | wave 3, after 3.C, 3.E, 3.F |
| 3.I render schema 3 | `render/model.py`, `render/api.py`, `render/vertexmap.py` | `test_render_schema3.py`, updates to the render tests | wave 3, after 3.C, 3.F; beside 3.G and 3.H |
| 3.Z integrator | `metrics/pipeline.py`, `run.py` (`template`, `zmap`, `metrics`, `scores`, `clusters`), `render/batch.py` (template, the `native` stage), `cli.py` (`metrics`, `render-subject` options), `tests/conftest.py` (`small_run`, `run_copy`, the real-data fixtures) | `test_pipeline.py`, `test_cli_metrics.py`, the network tests of 9.3 | last |

### Milestone 4

| Package | Files | Tests it owns | When |
|---|---|---|---|
| 4.0 integrator | `app/__init__.py`, `export/__init__.py`, `app/schemas.py` (every shape of 7.3 and 7.5), `protocol.py` (complete: it is small and `schemas.py` embeds its model), `calibration.py` (models, `load_calibration`, `write_calibration`), `pyproject.toml` and `justfile` (the `browser` group and marker), `tests/data/render_small.json` (a recorded `render` object for the frontend) | `test_protocol.py`, `test_calibration.py`, `test_app_schemas.py` (the JSON examples of 7.5 validate) | first, alone |
| 4.A marks | `app/marks.py` | `test_marks.py` | wave 1 |
| 4.B models and database | `app/models.py`, `app/db.py` | `test_app_db.py` | wave 1 |
| 4.C frontend | `app/static/`, `app/templates/` (against a mock server that replays the shapes of `app/schemas.py`) | a script that checks every route and field the JavaScript uses exists in `app/schemas.py` | wave 1, runs through the milestone |
| 4.D queue | `app/queue.py` (local branch; the team branch raises `NotImplementedError`) | `test_queue.py` | wave 2, after 4.B |
| 4.E export | `export/tables.py` | `test_export.py` | wave 2, after 4.B (table names only; it reads with `sqlite3`) |
| 4.F finalize | `app/finalize.py` (`finalize`, `sync_unratable`, `freeze_calibration`) | `test_finalize.py` | wave 3, after 4.B, 4.D |
| 4.G server and events | `app/server.py`, `app/events.py` | `test_server.py` | wave 4, after 4.A to 4.F |
| 4.Z integrator | `review.py`, `cli.py` (`review`, `serve`, `finalize`, `calibration freeze`, `export`) | `test_review.py`, `test_cli_review.py`, `test_browser.py`; then `--serve-after` with its own test | last |

### Milestone 5

| Package | Files | Tests it owns | When |
|---|---|---|---|
| 5.0 integrator | `remote/__init__.py` (`default_transport`), `config.py` (`RuntimeSettings`), `pyproject.toml` (`workflow` group), `justfile` (`census`) | `tests/fakes3.py` and a test of the fake itself | first, alone |
| 5.A privacy, completed | `privacy.py` (8.1), `log.py` (`RedactingFilter`) | `test_privacy_census.py`, `test_boundary_ast.py` | wave 1 |
| 5.B S3 client | `remote/s3.py` | `test_s3.py` | wave 1 |
| 5.C git refs and GraphQL | `remote/gitrefs.py`, `remote/openneuro.py` | `test_gitrefs.py`, `test_openneuro.py` | wave 1 |
| 5.D selection | `select/__init__.py`, `select/sampling.py` | `test_select.py` | wave 1 (needs milestone 3 only) |
| 5.E intensity | `metrics/intensity.py` | `test_intensity.py` | wave 1 (needs milestone 3 only) |
| 5.F team app | `app/auth.py`, `app/adjudication.py`, `app/agreement.py`, `app/errormap.py`, the team branch of `app/queue.py`, the router mounts in `app/server.py`, the team pages of 8.7 | `test_team.py`, `test_agreement.py`, `test_adjudication.py` | wave 1 (needs milestone 4 only; one agent, because it owns `queue.py` and `server.py` for this milestone) |
| 5.G cache | `remote/cache.py` | `test_cache.py` | wave 2, after 5.B |
| 5.H enrichment | `enrich/` | `test_enrich.py` | wave 2, after 5.B, 5.C |
| 5.I derivatives source | `sources/openneuro_derivatives.py` | `test_openneuro_derivatives.py` | wave 3, after 5.G |
| 5.J calibration transfer | `calibration_transfer.py` | `test_calibration_transfer.py` | wave 4, after 5.I |
| 5.Z integrator | `run.py` (remote kinds in `from_config`), `render/batch.py` (the `intensity` extra), `cli.py` (census commands), `workflow/Snakefile` | `test_census_e2e.py` (canary, kill and resume, changed object), `test_snakefile.py`, the bucket network tests of 9.3 | last |

Packages 5.A to 5.E depend only on milestones 2 and 3, so they can start while milestone 4 is under way. Phase 0 of the census (the pilot) needs milestone 4 only.

Stringing the milestones together: each milestone's step 0 can be written as soon as the previous milestone's step Z has merged, and the only hand-over between milestones is the test suite plus the fixtures named in 9.1 (`small_run` is what milestone 4 builds on, `FakeS3` what milestone 5 does). Nothing in section 13 has to be answered first.
