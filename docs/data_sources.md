# Data sources

Applies to the census. Local users only need the "Files required per subject" section.

## OpenNeuroDerivatives (primary census source)

Bucket `s3://openneuro-derivatives`, HTTPS endpoint `https://openneuro-derivatives.s3.amazonaws.com`, anonymous ListObjectsV2 works (verified 2026-11-01).

Top-level prefixes: `fitlins/`, `fmriprep/`, `mriqc/`, `xcpd/`. The census uses `fmriprep/`, which held 308 dataset prefixes named `dsXXXXXX-fmriprep/` on 2026-11-01.

Layout of one dataset, verified on `fmriprep/ds000102-fmriprep/`:

```
ds000102-fmriprep/
  .datalad/  .reproman/  code/  logs/
  dataset_description.json
  desc-aparcaseg_dseg.tsv  desc-aseg_dseg.tsv
  sub-01.html ...
  sub-01/ ...
  sourcedata/freesurfer/sub-01/{mri,surf,label,stats,scripts,...}
```

The fMRIPrep documentation states that when FreeSurfer runs, its subjects directory is written to `<output_dir>/sourcedata/freesurfer/`. Older fMRIPrep versions wrote it to `<output_dir>/freesurfer/`, next to the `fmriprep/` folder. Discovery must check both.

Why this source comes first: one organization ran one pipeline across many datasets. Differences in failure rate between datasets then reflect acquisition and anatomy more than who ran FreeSurfer with which flags. Pipeline versions can still differ between datasets and must be recorded per dataset (to verify: whether OpenNeuroDerivatives used one fMRIPrep and FreeSurfer version throughout).

Not every dataset is guaranteed to contain FreeSurfer outputs; a run with `--fs-no-reconall` would not. Phase 2 measures this.

## Embedded derivatives in raw OpenNeuro datasets (secondary source)

Raw datasets live in `s3://openneuro.org/dsXXXXXX/` (anonymous HTTPS listing returned 200 on 2026-11-01). Some contain author-supplied `derivatives/` with FreeSurfer outputs, under varied paths: `derivatives/freesurfer/`, `derivatives/fmriprep/sourcedata/freesurfer/`, `derivatives/fmriprep/freesurfer/`, and idiosyncratic ones. FreeSurfer versions, flags and manual edits differ.

This pool answers a different question: what the community actually ships. Report it separately from OpenNeuroDerivatives, never pooled.

Dataset enumeration: the OpenNeuro GraphQL API (endpoint `https://openneuro.org/crn/graphql`, to verify) lists datasets and snapshots. Fallback: list the top level of `s3://openneuro.org/` with a delimiter and probe each `dsXXXXXX/derivatives/` prefix.

Snapshot pinning: record the snapshot tag for each raw dataset and the git commit of each OpenNeuroDerivatives dataset (from the `.datalad` metadata or the GitHub mirror in the OpenNeuroDerivatives organization). The S3 bucket reflects the latest state; the recorded refs make the census reproducible in principle.

## Files required per subject

Sizes from `ds000102-fmriprep/sourcedata/freesurfer/sub-01` (FreeSurfer default 1 mm conformed space):

| Role | Path in subject dir | Size | Needed for |
|---|---|---|---|
| t1 | `mri/T1.mgz` | 5.6 MB | render, intensity metric |
| brainmask | `mri/brainmask.mgz` | 1.7 MB | intensity window (optional) |
| lh/rh_white | `surf/?h.white` | 5.3 MB each | render, vertex metrics |
| lh/rh_pial | `surf/?h.pial` | 5.3 MB each | render, vertex metrics |
| lh/rh_thickness | `surf/?h.thickness` | 0.6 MB each | fsaverage metrics |
| lh/rh_sphere_reg | `surf/?h.sphere.reg` | 5.3 MB each | resampling to fsaverage, mapping marks |
| lh/rh_annot | `label/?h.aparc.annot` | small (to verify) | vertex ROIs |
| lh/rh_stats | `stats/?h.aparc.stats` | 6 kB each | regional metrics |
| aseg_stats | `stats/aseg.stats` | small | Euler number / holes, eTIV |
| talairach | `mri/transforms/talairach.xfm` | <1 kB | approximate MNI placement |
| recon_log | `scripts/recon-all.log` | 220 kB (fetch first 64 kB by Range) | flags |
| recon_log_lh/rh | `scripts/recon-all-?h.log` | 212 and 206 kB | thickness truncation counts |
| build_stamp | `scripts/build-stamp.txt` | <1 kB | FreeSurfer version |

Fetch tiers:

- Tier A, about 15 kB per subject: stats, build stamp, head of recon-all.log. Enough for regional metrics and provenance.
- Tier B, about 12 MB: thickness, sphere.reg and annot for both hemispheres. Enough for fsaverage z-maps and cluster metrics, for every subject.
- Tier C, about 27 MB more: T1, white and pial surfaces. Needed only for rendering, the intensity metric and `vertex_map.npz`. A rendered subject needs Tier B and C together, about 39 MB.

Template: the fsaverage sphere, cortex label and annotation, fetched once per FreeSurfer major version from the `fsaverage/` directory that fMRIPrep places next to the subjects (verified present in `ds000102-fmriprep/sourcedata/freesurfer/`, alongside `fsaverage5/`), or from a local FreeSurfer install, or nilearn (see `metrics.md`).

Note on `?h.pial` in FreeSurfer 7: from 7.0.0 on, recon-all writes `?h.pial.T1` and links `?h.pial` to it, or to `?h.pial.T2` or `?h.pial.FLAIR` when those refinements run (read in `scripts/recon-all` at v7.0.0 to v7.4.1 on 2026-11-12; FreeSurfer 6 writes `?h.pial` as a regular file). How these links appear on S3 (separate keys, copies, or absent) is to verify on a FreeSurfer 7 dataset. The resolver checks `?h.pial`, then `?h.pial.T2`, `?h.pial.FLAIR`, `?h.pial.T1`, and records which it used.

## Provenance fields

- FreeSurfer version: `scripts/build-stamp.txt`.
- recon-all command line and flags: first lines of `scripts/recon-all.log`. Flags of interest: `-T2pial`, `-FLAIRpial`, `-hires`, `-expert`, `-xopts-*`, and any manual-edit reruns (`-autorecon2-wm`, `-autorecon-pial`), which suggest the subject was edited.
- Thickness truncation counts: `scripts/recon-all-lh.log` and `scripts/recon-all-rh.log`, the per-hemisphere logs that recon-all writes under fMRIPrep (see `metrics.md`, "The 5 mm ceiling").
- fMRIPrep version: `dataset_description.json`, `GeneratedBy`.
- Field strength, manufacturer: raw dataset T1w JSON sidecar (`MagneticFieldStrength`, `Manufacturer`); often missing, then null.
- Age, sex: raw dataset `participants.tsv`; column names vary and need a small mapping table.

## Dataset identities

Accessions are needed to fetch data and are stored in `catalog.sqlite` only. Everything shown or exported uses keyed pseudonyms (decision D10 in `decisions.md`).

## Etiquette and load

The buckets are public open-data buckets. Keep concurrency moderate (default 8), use Range requests where possible, cache by ETag, and never re-download unchanged files. Tier B for every subject is about 12 MB × N, about 60 GB for 5,000 subjects. Tier B plus C for every rendered subject is about 39 MB × N, about 195 GB for 5,000. Both are streamed in batches and source files are deleted after use (see `phases.md`); only derived tables, fsaverage maps, media and vertex maps are kept.
