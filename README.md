# surfcensus

Look at your FreeSurfer surfaces where they most often go wrong.

`surfcensus` scores FreeSurfer reconstructions with occipital cortical-thickness metrics, renders animated sagittal, coronal and axial views around the occipital pole with white and pial contours, and serves them in a keyboard-driven rating app where raters click on errors to mark where they are. Thickness outliers are detected per vertex in fsaverage space, so marks and metrics can be compared location by location. FreeSurfer surfaces often absorb the superior sagittal sinus or the cerebellar tentorium in this region, and these errors are rarely edited.

Status: design stage. See `docs/`.

## Your own data

```
pip install "surfcensus[app]"
surfcensus review /path/to/SUBJECTS_DIR
```

Works on a FreeSurfer `SUBJECTS_DIR`, an fMRIPrep output folder, or a BIDS derivatives folder with `freesurfer/`. No FreeSurfer installation needed. Ratings are exported with `surfcensus export`.

## The OpenNeuro census

The same pipeline, run over FreeSurfer outputs in public OpenNeuro derivatives, to measure how common these errors are and how often quantitative checks miss them. See `docs/phases.md`.
