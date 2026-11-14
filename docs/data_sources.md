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

Why this source comes first: one organization ran one pipeline across many datasets. Differences in failure rate between datasets then reflect acquisition and anatomy more than who ran FreeSurfer with which flags. Pipeline versions differ between datasets and are recorded per dataset and per subject: `ds000102-fmriprep` was made with fMRIPrep 21.0.2 and FreeSurfer 6.0.1, `ds003020-fmriprep` with fMRIPrep 23.0.1 and FreeSurfer 7.3.2 (their `dataset_description.json` and the build stamp of one subject each, read 2026-11-12). How the versions are spread over the 308 datasets is measured in phase 2. FreeSurfer version is tied to when a dataset was processed, so it is confounded with everything else that changed over that time.

Not every dataset is guaranteed to contain FreeSurfer outputs; a run with `--fs-no-reconall` would not. Phase 2 measures this.

## Embedded derivatives in raw OpenNeuro datasets (secondary source)

Raw datasets live in `s3://openneuro.org/dsXXXXXX/` (anonymous HTTPS listing returned 200 on 2026-11-01). Some contain author-supplied `derivatives/` with FreeSurfer outputs, under varied paths: `derivatives/freesurfer/`, `derivatives/fmriprep/sourcedata/freesurfer/`, `derivatives/fmriprep/freesurfer/`, and idiosyncratic ones. FreeSurfer versions, flags and manual edits differ.

This pool answers a different question: what the community actually ships. Report it separately from OpenNeuroDerivatives, never pooled.

On disk. A copy of such a dataset (downloaded, or cloned) is read by the local source when its root is given: the subjects are looked for below `derivatives/`, first in `derivatives/freesurfer/`, then in the folders of `derivatives/` that hold subject directories themselves (`derivatives/freesurfer-7.3.2/`), then in `derivatives/<folder>/sourcedata/freesurfer/`, then in `derivatives/<folder>/freesurfer/`, whatever the folder is called (`architecture.md`, Sources). The idiosyncratic paths are not found; the folder that holds the subjects is then given as the path. The rule was written from the three path forms named above and tried on synthetic trees and on two real subjects linked into `derivatives/fmriprep/sourcedata/freesurfer/` of a made-up dataset root (2026-11-14); no real OpenNeuro dataset with embedded FreeSurfer output was listed or downloaded for it.

Dataset enumeration: the OpenNeuro GraphQL API (endpoint `https://openneuro.org/crn/graphql`, to verify) lists datasets and snapshots. Fallback: list the top level of `s3://openneuro.org/` with a delimiter and probe each `dsXXXXXX/derivatives/` prefix.

Snapshot pinning: record the snapshot tag for each raw dataset and the git commit of each OpenNeuroDerivatives dataset (from the `.datalad` metadata or the GitHub mirror in the OpenNeuroDerivatives organization). The S3 bucket reflects the latest state; the recorded refs make the census reproducible in principle.

## Files required per subject

Sizes from `ds000102-fmriprep/sourcedata/freesurfer/sub-01` (FreeSurfer 6.0.1, default 1 mm conformed space; bucket listing of 2026-11-12):

| Role | Path in subject dir | Size | Needed for |
|---|---|---|---|
| t1 | `mri/T1.mgz` | 5.6 MB | render, intensity metric |
| brainmask | `mri/brainmask.mgz` | 1.7 MB | intensity window (optional) |
| lh/rh_white | `surf/?h.white` | 5.3 MB each | render, vertex metrics |
| lh/rh_pial | `surf/?h.pial` | 5.3 MB each | render, vertex metrics |
| lh/rh_thickness | `surf/?h.thickness` | 0.6 MB each | fsaverage metrics |
| lh/rh_sphere_reg | `surf/?h.sphere.reg` | 5.3 MB each | resampling to fsaverage, mapping marks |
| lh/rh_annot | `label/?h.aparc.annot` | 1.2 MB each | cortex and parcels on the native surface |
| lh/rh_stats | `stats/?h.aparc.stats` | 6 kB each | regional metrics |
| aseg_stats | `stats/aseg.stats` | 10 kB | Euler number / holes, eTIV |
| talairach | `mri/transforms/talairach.xfm` | <1 kB | approximate MNI placement |
| recon_log | `scripts/recon-all.log` | 221 kB | flags, whether recon-all finished, holes |
| recon_log_lh/rh | `scripts/recon-all-?h.log` | 212 and 206 kB | flags, thickness truncation counts |
| build_stamp | `scripts/build-stamp.txt` | <1 kB | FreeSurfer version |

Fetch tiers:

- Tier A, 0.7 to 1.0 MB per subject: the three stats files, the build stamp and the three recon-all logs, whole. Enough for regional metrics and provenance. The stats files and the stamp alone are 21 to 23 kB; the logs are the rest.
- Tier B, 13 to 14 MB: thickness, sphere.reg and annot for both hemispheres. Enough for fsaverage z-maps and cluster metrics, for every subject. The annotation is used when present; thickness and sphere.reg are required.
- Tier C, 23 to 29 MB more: T1, brainmask, white and pial surfaces. Needed only for rendering, the intensity metric and `vertex_map.npz`. A rendered subject needs Tier B and C together, 36 to 43 MB.

The ranges are the two subjects whose listings were read on 2026-11-12: `ds000102` sub-01 (FreeSurfer 6.0.1; Tier A 0.66 MB, Tier B 14.2 MB, Tier C 28.5 MB) and `ds003020` sub-UTS01 (FreeSurfer 7.3.2; 0.98, 12.7 and 23.2 MB).

The logs are read whole because their head does not hold what is needed (`inventory.logs = "full"`, the default). Under fMRIPrep, recon-all is called several times per subject: `recon-all.log` of the 6.0.1 subject holds 4 invocations and that of the 7.3.2 subject 5, and each per-hemisphere log holds 2. The first 64 kB of either `recon-all.log` hold 2 invocations, and the closing line that says whether recon-all finished is at the end. `inventory.logs = "head"` reads only the first `head_bytes` of each log; flags that the head does not show and the finished state are then stored as unknown, not as absent.

A subject is `complete` when the files a render cannot do without are present (T1, white and pial of both hemispheres). A directory with `mri/` and `surf/` that lacks some of them is inventoried as incomplete, so that a recon-all run that stopped half-way is counted, not overlooked.

Template: the fsaverage sphere, white surface, cortex label and annotation of both hemispheres, and the two average-area files when they are there (40 MB), from a local FreeSurfer install or from the `fsaverage/` directory that fMRIPrep places next to the subjects (present in `ds000102-fmriprep/sourcedata/freesurfer/` and in `ds003020-fmriprep/sourcedata/freesurfer/`, alongside `fsaverage5/`). One template serves a whole run (decision D22). The two copies named here differ only in `?h.white`, at one vertex that fsaverage6 does not have, and in the creation line of that file (compared 2026-11-13; see `metrics.md`). nilearn is not used.

Note on `?h.pial` in FreeSurfer 7: from 7.0.0 on, recon-all writes `?h.pial.T1` and links `?h.pial` to it, or to `?h.pial.T2` or `?h.pial.FLAIR` when those refinements run (read in `scripts/recon-all` at v7.0.0 to v7.4.1 on 2026-11-12; FreeSurfer 6 writes `?h.pial` as a regular file). On S3 the link is absent: the listing of `ds003020` sub-UTS01 (FreeSurfer 7.3.2, read 2026-11-12) has `surf/?h.pial.T1` and no `surf/?h.pial` key. One subject of one dataset was looked at. The resolver checks `?h.pial`, then `?h.pial.T2`, `?h.pial.FLAIR`, `?h.pial.T1`, and the inventory records which it used (`subjects.pial_variant`).

## Provenance fields

- FreeSurfer version: `scripts/build-stamp.txt`, else the stamp line of `recon-all.log`. `freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a` gives 6.0.1, `freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275` gives 7.3.2. A development stamp without a version number (the `fsaverage` next to `ds000102` has `freesurfer-x86_64-redhat-linux-gnu-dev4-20090216`) gives none.
- recon-all flags: every invocation in `scripts/recon-all.log` and in the two per-hemisphere logs. An invocation is found at its `subjid NAME` line; its arguments are on the last non-blank line above (alone in FreeSurfer 6, after the script path in FreeSurfer 7). Only flag names are stored (`subjects.recon_flags`, a sorted list): tokens that look like a flag and hold no accession. Paths, numbers and subject names are dropped, so the stored provenance names no dataset. Flags of interest: `-T2pial`, `-FLAIRpial`, `-hires`, `-expert`, `-xopts-*`, and any manual-edit reruns (`-autorecon2-wm`, `-autorecon-pial`), which suggest the subject was edited. Each is stored as true, false or unknown. It is false only when all of the subject's logs were read in full and none shows it; fMRIPrep passes `-noT2pial -noFLAIRpial` to FreeSurfer 7.3.2, which reads as false. A pial surface that resolves to `?h.pial.T2` or `?h.pial.FLAIR` sets its flag whatever the logs say. Where `-T2pial` appears in the logs of a dataset that was processed with a T2 image has not been looked at.
- Whether recon-all finished: the closing line of the last invocation in `recon-all.log` (`finished without error` or `exited with ERRORS`).
- Surface holes before topology fixing: the `orig.nofix lhholes = N, rhholes = M` line of `recon-all.log`.
- Thickness truncation counts: `scripts/recon-all-lh.log` and `scripts/recon-all-rh.log`, the per-hemisphere logs that recon-all writes under fMRIPrep (see `metrics.md`, "The 5 mm ceiling"). A plain `recon-all -all` run is expected to write both lines to `recon-all.log`, left hemisphere first; no such log has been looked at.
- fMRIPrep version: `dataset_description.json`, `GeneratedBy` (the entry named fMRIPrep), else `PipelineDescription.Version`, the form of older fMRIPrep versions (from memory; no such file has been looked at).
- Participant: the leading `sub-<label>` of the directory name, so that sessions (`sub-01_ses-02`) and longitudinal directories (`sub-01.long.base`, flagged as such) can be grouped. Each directory is a subject of its own.
- Field strength, manufacturer: raw dataset T1w JSON sidecar (`MagneticFieldStrength`, `Manufacturer`); often missing, then null.
- Age, sex: raw dataset `participants.tsv`; column names vary and need a small mapping table.

## Dataset identities

Accessions are needed to fetch data and are stored in `catalog.sqlite` only. Everything shown or exported uses keyed pseudonyms (decision D10 in `decisions.md`).

## Etiquette and load

The buckets are public open-data buckets. Keep concurrency moderate (default 8), use Range requests where possible, cache by ETag, and never re-download unchanged files. Tier A for every subject is 0.7 to 1.0 MB × N, 3 to 5 GB for 5,000 subjects. Tier B is 13 to 14 MB × N, about 63 to 71 GB for 5,000. Tier B plus C for every rendered subject is 36 to 43 MB × N, about 180 to 215 GB for 5,000. Both are streamed in batches and source files are deleted after use (see `phases.md`); only derived tables, fsaverage maps, media and vertex maps are kept.
