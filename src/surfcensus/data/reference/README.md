# Shipped normative reference

This directory is empty on purpose. It will hold the normative reference that is built from
rater-passed census subjects (`surfcensus reference build`, phase 8): an `index.json`, one
`stratum_<nnn>.npz` per stratum with the per-vertex median, MAD and subject count of globally
normalized thickness on fsaverage6, and `members.parquet`.

`index.json` carries a `schema_version` and the SHA-256 of the fsaverage spheres the
reference was made on; it applies to a run only when the run's template was built from the
same spheres. A shipped reference has no members and no template id.

Until the census has produced ratings there is no `index.json` here, and
`surfcensus.reference.load.shipped_reference()` returns None. Local sets of at least 30
subjects are then scored within the set; smaller sets are ranked on the raw ceiling fraction
and the occipital-to-control thickness ratio (decision D16 in `docs/decisions.md`).
