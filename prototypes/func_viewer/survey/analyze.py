"""Tables from the listings: one row per sampled dataset, one row per run, and summaries.

usage: analyze.py OUT [--without ACCESSION] [--pair FILE]
       (after survey.py OUT and raw_and_versions.py OUT)

Writes OUT/func_survey_30_datasets.csv, func_survey_runs.csv and func_survey_subject_totals.csv
and prints the summaries. Tables and printed text hold accessions and stay out of the
repository. --without names one dataset to leave out of the last scale-up lines (the study left
out the one whose first subject is by far the largest). --pair is a listing made by
pair_listing.py, for a dataset whose first derivative subject is absent from the raw bucket.

Routes per run, in bytes:
  a  fMRIPrep's T1w-space preprocessed series
  b  the BOLD-space preprocessed series plus the transform to T1w space
  c  the raw series plus the transform plus the T1w-space preprocessed series
  c_minimal  the raw series plus the stored head-motion, coregistration and fieldmap transforms,
     which is all that a derivative of the minimal layout offers
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling module, also under -I
from survey import fmriprep_version

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--without", default=None)
ap.add_argument("--pair", default=None)
args = ap.parse_args()
OUT = Path(args.out)
d = json.loads((OUT / "sample_listings.json").read_text())
raws = {r["accession"]: r for r in json.loads((OUT / "raw_listings.json").read_text())}

ID_ENT = ("ses", "task", "acq", "ce", "rec", "dir", "run")
MB = 1e6
OPTIONS = ("option_a_bytes", "option_b_bytes", "option_c_bytes", "option_c_minimal_bytes")
BYTES = (*OPTIONS, "raw_bold_bytes")


def parse(name: str):
    stem, _, ext = name.partition(".")
    parts = stem.split("_")
    ents = {}
    suffix = None
    for p in parts:
        if "-" in p:
            k, v = p.split("-", 1)
            ents[k] = v
        else:
            suffix = p
    return ents, suffix, ext


def run_id(ents: dict) -> str:
    # run-01 in the raw data is run-1 in some derivatives: compare run numbers as numbers
    def norm(k, v):
        return str(int(v)) if k == "run" and v.isdigit() else v

    return "_".join(f"{k}-{norm(k, ents[k])}" for k in ID_ENT if k in ents)


def kind(ents, suffix, ext):
    sp = ents.get("space")
    desc = ents.get("desc")
    if suffix == "bold" and ext == "nii.gz":
        if desc == "preproc":
            if sp is None:
                return "preproc_boldspace"
            if sp == "T1w":
                return "preproc_T1w"
            if sp.startswith("MNI"):
                return "preproc_MNI"
            return f"preproc_{sp}"
        if desc and "AROMA" in desc:
            return "aroma"
        return f"bold_other_{desc}"
    if suffix == "bold" and ext == "dtseries.nii":
        return "cifti"
    if suffix == "bold" and ext == "func.gii":
        return "gifti"
    if suffix == "boldref" and ext == "nii.gz":
        return "boldref_" + (sp or desc or "boldspace")
    if suffix == "mask" and ext == "nii.gz":
        return "mask_" + (sp or "boldspace")
    if suffix == "timeseries" and ext == "tsv":
        return "confounds"
    if suffix == "xfm" and ext in ("txt", "h5"):
        return "xfm"
    if suffix == "dseg" and ext == "nii.gz":
        return "dseg"
    return None


def counted(counter: Counter) -> str:
    return ";".join(f"{k}({v})" for k, v in sorted(counter.items()))


rows = []
run_rows = []
for i, ds in zip(d["indices"], d["datasets"], strict=True):
    acc = ds["prefix"].split("/")[1].split("-")[0]
    sub_prefix = ds["first_subject"]
    sub = sub_prefix.rstrip("/").split("/")[-1]
    ver = fmriprep_version(ds.get("description"))
    by_dir = defaultdict(int)
    sessions = set()
    runs = defaultdict(lambda: defaultdict(int))  # run_id -> kind -> bytes
    run_n = defaultdict(lambda: defaultdict(int))
    xfm_names = Counter()
    mni = Counter()
    kinds_seen = Counter()
    kind_bytes = defaultdict(int)
    func_files = 0
    echo_in_deriv = False
    fmap_ids = set()
    fmap_coeff_bytes = 0
    anat = {}
    for k, s in ds["subject_keys"]:
        rel = k[len(sub_prefix) :].split("/")
        for p in rel[:-1]:
            if p.startswith("ses-"):
                sessions.add(p)
        dn = next((p for p in rel[:-1] if not p.startswith("ses-")), ".")
        by_dir[dn] += s
        ents, suffix, ext = parse(rel[-1])
        if dn == "anat":
            anat_ents = ("space", "from", "to", "hemi", "label", "desc", "den", "res")
            g = "_".join([f"{a}-{ents[a]}" for a in anat_ents if a in ents] + [suffix or ""])
            g = g + "." + ext
            anat[g] = anat.get(g, 0) + s
        if dn == "fmap":
            if "fmapid" in ents:
                fmap_ids.add((ents.get("ses"), ents["fmapid"]))
            if ents.get("desc", "").startswith("coeff"):
                fmap_coeff_bytes += s
        if dn != "func":
            continue
        func_files += 1
        if "echo" in ents:
            echo_in_deriv = True
        kd = kind(ents, suffix, ext)
        if kd is None:
            continue
        rid = run_id(ents)
        kinds_seen[kd] += 1
        kind_bytes[kd] += s
        if kd == "xfm":
            nm = f"from-{ents.get('from')}_to-{ents.get('to')}"
            nm += f"_desc-{ents['desc']}" if "desc" in ents else ""
            if ents.get("to", "").startswith("auto") or ents.get("to", "").startswith("func"):
                nm = "from-boldref_to-<fmapid>"
            xfm_names[nm] += 1
            runs[rid]["xfm:" + nm] += s
        else:
            runs[rid][kd] += s
            run_n[rid][kd] += 1
        if kd == "preproc_MNI":
            mni[ents["space"] + ("_res-" + ents["res"] if "res" in ents else "")] += 1
        if kd == "aroma":
            mni["AROMA:" + ents["space"]] += 1

    # raw bucket
    raw = raws[acc]
    raw_runs = defaultdict(int)
    raw_echoes = defaultdict(int)
    raw_dirs = defaultdict(int)
    raw_dir_n = Counter()
    raw_anat = {}
    raw_fmap_suffix = Counter()
    raw_sbref = 0
    for k, s in raw["keys"]:
        rel = k.split("/")[2:]
        dn = next((p for p in rel[:-1] if not p.startswith("ses-")), ".")
        raw_dirs[dn] += s
        raw_dir_n[dn] += 1
        ents, suffix, ext = parse(rel[-1])
        if dn == "func" and suffix == "bold" and ext in ("nii.gz", "nii"):
            rid = run_id(ents)
            raw_runs[rid] += s
            raw_echoes[rid] += 1
        if dn == "func" and suffix == "sbref" and ext in ("nii.gz", "nii"):
            raw_sbref += 1
        if dn == "anat" and ext in ("nii.gz", "nii"):
            raw_ents = ("acq", "rec", "ce", "inv", "part")
            g = "_".join([f"{a}-{ents[a]}" for a in raw_ents if a in ents] + [suffix])
            raw_anat[g] = raw_anat.get(g, 0) + s
        if dn == "fmap" and ext in ("nii.gz", "nii"):
            raw_fmap_suffix[suffix] += 1

    n_runs = len([r for r in runs if r])
    new_layout = "boldref_hmc" in kinds_seen
    # per-run rows
    for rid in sorted(r for r in runs if r):
        rr = runs[rid]
        raw_b = raw_runs.get(rid)
        hmc = rr.get("xfm:from-orig_to-boldref_desc-hmc", 0)
        coreg = rr.get("xfm:from-boldref_to-T1w_desc-coreg", 0)
        coreg += rr.get("xfm:from-scanner_to-T1w", 0)
        opt_a = rr.get("preproc_T1w") or None
        opt_b = (rr["preproc_boldspace"] + coreg) if rr.get("preproc_boldspace") else None
        opt_c = None
        if raw_b and rr.get("preproc_T1w"):
            opt_c = raw_b + coreg + rr["preproc_T1w"]
        # what a viewer needs when no preprocessed BOLD exists: raw + hmc + coreg (+ fmap
        # coefficients, per subject)
        opt_c_min = None
        if raw_b and new_layout:
            opt_c_min = raw_b + hmc + coreg + rr.get("xfm:from-boldref_to-<fmapid>", 0)
        run_rows.append(
            dict(
                accession=acc,
                fmriprep_version=ver,
                subject=sub,
                run=rid,
                raw_bold_bytes=raw_b,
                raw_echoes=raw_echoes.get(rid),
                preproc_boldspace_bytes=rr.get("preproc_boldspace") or None,
                preproc_T1w_bytes=rr.get("preproc_T1w") or None,
                preproc_MNI_bytes=rr.get("preproc_MNI") or None,
                aroma_bytes=rr.get("aroma") or None,
                confounds_bytes=rr.get("confounds") or None,
                hmc_xfm_bytes=hmc or None,
                coreg_xfm_bytes=coreg or None,
                option_a_bytes=opt_a,
                option_b_bytes=opt_b,
                option_c_bytes=opt_c,
                option_c_minimal_bytes=opt_c_min,
            )
        )

    def per(kd, kinds_seen=kinds_seen, kind_bytes=kind_bytes):
        n = kinds_seen.get(kd, 0)
        return round(kind_bytes[kd] / n / MB, 2) if n else ""

    raw_per_run = ""
    if raw_runs:
        raw_per_run = round(float(np.median(list(raw_runs.values()))) / MB, 1)
    rows.append(
        dict(
            sample_index=i,
            accession=acc,
            fmriprep_version=ver,
            n_subject_dirs=ds["n_subject_dirs"],
            first_subject=sub,
            n_sessions=len(sessions),
            n_runs_derivative=n_runs,
            func_files=func_files,
            func_MB=round(by_dir["func"] / MB, 1),
            fmap_MB=round(by_dir.get("fmap", 0) / MB, 1),
            anat_MB=round(by_dir["anat"] / MB, 1),
            figures_MB=round(by_dir.get("figures", 0) / MB, 1),
            subject_total_MB=round(sum(by_dir.values()) / MB, 1),
            layout="minimal (boldrefs+transforms)" if new_layout else "full (preproc BOLD)",
            n_preproc_boldspace=kinds_seen.get("preproc_boldspace", 0),
            MB_per_preproc_boldspace=per("preproc_boldspace"),
            n_preproc_T1w=kinds_seen.get("preproc_T1w", 0),
            MB_per_preproc_T1w=per("preproc_T1w"),
            n_preproc_MNI=kinds_seen.get("preproc_MNI", 0),
            MB_per_preproc_MNI=per("preproc_MNI"),
            MNI_spaces=";".join(sorted(mni)),
            n_aroma=kinds_seen.get("aroma", 0),
            MB_per_aroma=per("aroma"),
            n_cifti=kinds_seen.get("cifti", 0),
            n_gifti=kinds_seen.get("gifti", 0),
            boldref_kinds=";".join(sorted(k[8:] for k in kinds_seen if k.startswith("boldref_"))),
            mask_kinds=";".join(sorted(k[5:] for k in kinds_seen if k.startswith("mask_"))),
            n_confounds=kinds_seen.get("confounds", 0),
            MB_per_confounds=per("confounds"),
            xfm_names=counted(xfm_names),
            echo_entity_in_derivative=echo_in_deriv,
            n_fieldmaps_derivative=len(fmap_ids),
            fmap_coeff_MB=round(fmap_coeff_bytes / MB, 2),
            anat_preproc_T1w_MB=round(anat.get("desc-preproc_T1w.nii.gz", 0) / MB, 2),
            anat_has_fsnative_to_T1w_xfm=any(k.startswith("from-fsnative_to-T1w") for k in anat),
            raw_subject_present=bool(raw["keys"]),
            raw_n_bold_files=sum(raw_echoes.values()),
            raw_n_runs=len(raw_runs),
            raw_multi_echo=any(v > 1 for v in raw_echoes.values()),
            raw_bold_MB=round(sum(raw_runs.values()) / MB, 1),
            raw_MB_per_run_median=raw_per_run,
            raw_runs_matched_to_derivative=len(set(raw_runs) & {r for r in runs if r}),
            raw_n_sbref=raw_sbref,
            raw_fmap_files=counted(raw_fmap_suffix),
            raw_anat=";".join(f"{k}({v / MB:.1f}MB)" for k, v in sorted(raw_anat.items())),
            raw_subject_MB=round(sum(raw_dirs.values()) / MB, 1),
            raw_other_dirs=";".join(
                sorted(k for k in raw_dirs if k not in ("func", "anat", "fmap", "."))
            ),
        )
    )

df = pd.DataFrame(rows)
runs_df = pd.DataFrame(run_rows)
df.to_csv(OUT / "func_survey_30_datasets.csv", index=False)
runs_df.to_csv(OUT / "func_survey_runs.csv", index=False)
pd.set_option(
    "display.width", 250, "display.max_columns", 60, "display.max_rows", 200,
    "display.max_colwidth", 60,
)  # fmt: skip
for columns in (
    [
        "sample_index", "accession", "fmriprep_version", "n_subject_dirs", "n_sessions",
        "n_runs_derivative", "func_MB", "fmap_MB", "anat_MB", "figures_MB", "layout",
    ],
    [
        "accession", "fmriprep_version", "n_preproc_boldspace", "MB_per_preproc_boldspace",
        "n_preproc_T1w", "MB_per_preproc_T1w", "n_preproc_MNI", "MB_per_preproc_MNI",
        "MNI_spaces", "n_aroma", "MB_per_aroma", "n_cifti", "n_gifti", "n_confounds",
    ],
    [
        "accession", "fmriprep_version", "boldref_kinds", "mask_kinds", "xfm_names",
        "n_fieldmaps_derivative", "fmap_coeff_MB", "echo_entity_in_derivative",
    ],
    [
        "accession", "anat_preproc_T1w_MB", "anat_has_fsnative_to_T1w_xfm",
        "raw_subject_present", "raw_n_bold_files", "raw_n_runs",
        "raw_runs_matched_to_derivative", "raw_multi_echo", "raw_bold_MB",
        "raw_MB_per_run_median", "raw_n_sbref", "raw_fmap_files", "raw_subject_MB",
        "raw_other_dirs",
    ],
):  # fmt: skip
    print(df[columns].to_string())
    print()
print(df[["accession", "raw_anat"]].to_string())


def dist(x, label):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    x = np.asarray(x, dtype=float) / MB
    if len(x) == 0:
        print(f"{label}: no runs")
        return
    q = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
    print(
        f"{label}: n={len(x)} min {q[0]:.1f} p10 {q[1]:.1f} p25 {q[2]:.1f} median {q[3]:.1f}"
        f" p75 {q[4]:.1f} p90 {q[5]:.1f} max {q[6]:.1f} mean {x.mean():.1f} MB"
    )


print("\nPer-run distributions (all runs of the first subjects, pooled)")
for col in (
    *BYTES,
    "preproc_MNI_bytes",
    "aroma_bytes",
    "hmc_xfm_bytes",
    "coreg_xfm_bytes",
    "confounds_bytes",
):
    dist(runs_df[col].tolist(), col)
print("\nPer-run medians per dataset, then distribution over datasets (each dataset counted once)")
for col in BYTES:
    med = runs_df.groupby("accession")[col].median().dropna()
    dist(med.tolist(), col + " (dataset medians)")

print("\nPer-subject totals (first subject), MB")
by_dataset = runs_df.groupby(["accession", "fmriprep_version"])
tot = by_dataset[list(BYTES)].sum(min_count=1) / MB
tot["n_runs"] = by_dataset.size()
print(tot.round(1).to_string())
tot.round(2).to_csv(OUT / "func_survey_subject_totals.csv")

# ---------------------------------------------------------------------------------------------
# Summaries for the report
# ---------------------------------------------------------------------------------------------
print("\n=== fractions over the 30 sampled datasets (first subject) ===")
n = len(df)
has_t1 = df.n_preproc_T1w > 0
has_bs = df.n_preproc_boldspace > 0
has_mni = df.n_preproc_MNI > 0
print("T1w-space preproc BOLD:", int(has_t1.sum()), "/", n)
print("BOLD-space preproc BOLD:", int(has_bs.sum()), "/", n)
print("MNI preproc BOLD:", int(has_mni.sum()), "/", n)
print(
    "only MNI:",
    int((has_mni & ~has_t1 & ~has_bs).sum()),
    " only BOLD space:",
    int((has_bs & ~has_t1 & ~has_mni).sum()),
    " no preprocessed BOLD image at all:",
    int((~has_t1 & ~has_bs & ~has_mni).sum()),
)
print(
    "BOLD-to-T1w transform (from-scanner_to-T1w or from-boldref_to-T1w):",
    int(df.xfm_names.str.contains("to-T1w").sum()),
    "/",
    n,
)
print(
    "per-volume head-motion transform (from-orig_to-boldref desc-hmc):",
    int(df.xfm_names.str.contains("desc-hmc").sum()),
    "/",
    n,
)
print("confounds table:", int((df.n_confounds > 0).sum()), "/", n)
print("BOLD-space brain mask:", int(df.mask_kinds.str.contains("boldspace").sum()), "/", n)
print("fieldmap outputs in derivative:", int((df.n_fieldmaps_derivative > 0).sum()), "/", n)
df["raw_has_fmap"] = df.raw_fmap_files != ""
df["deriv_has_fmap"] = df.n_fieldmaps_derivative > 0
print(pd.crosstab([df.layout, df.raw_has_fmap], df.deriv_has_fmap))
print("raw multi-echo:", df.loc[df.raw_multi_echo, "accession"].tolist())
print(
    "sessions > 0:",
    int((df.n_sessions > 0).sum()),
    " runs/first subject (derivative): min",
    df.n_runs_derivative.min(),
    "median",
    df.n_runs_derivative.median(),
    "mean",
    round(df.n_runs_derivative.mean(), 1),
    "max",
    df.n_runs_derivative.max(),
)
print("derivative func MB per subject, full layout:", df.loc[has_t1, "func_MB"].tolist())
min_func = df.loc[~has_t1, "func_MB"]
min_both = df.loc[~has_t1, "func_MB"] + df.loc[~has_t1, "fmap_MB"]
print(
    "derivative func MB per subject, minimal layout: min",
    min_func.min(),
    "median",
    min_func.median(),
    "max",
    min_func.max(),
)
print("derivative func+fmap MB, minimal: median", min_both.median(), "max", min_both.max())
print(
    "anat MB per subject: min",
    df.anat_MB.min(),
    "median",
    df.anat_MB.median(),
    "max",
    df.anat_MB.max(),
)

if args.pair:
    sup = json.loads(Path(args.pair).read_text())
    sup_raw = sum(s for k, s in sup["raw"] if k.endswith("_bold.nii.gz"))
    sup_x = sum(s for k, s in sup["deriv"] if "/func/" in k and k.endswith("_xfm.txt"))
    print(
        f"\nsupplement {Path(args.pair).stem} (first subject present in both buckets):"
        f" raw BOLD {sup_raw / MB:.1f} MB, transforms {sup_x / 1e3:.1f} kB"
    )

print("\n=== per-subject bytes, GB (first subject of each dataset; n_subject_dirs as weight) ===")
tot = runs_df.groupby("accession")[list(BYTES)].sum(min_count=1) / 1e9
tot = tot.join(
    df.set_index("accession")[["n_subject_dirs", "n_runs_derivative", "fmriprep_version"]]
)
# for the minimal layout the viewer route is raw + transforms; add the fieldmap coefficient
# files per subject
tot["fmap_coeff_GB"] = df.set_index("accession").fmap_coeff_MB / 1e3


def scale_up(x):
    """For 5,000 subjects, GB: by the mean over datasets, the subject-weighted mean, the median."""
    w = tot.loc[x.index, "n_subject_dirs"]
    return (
        f"{5000 * x.mean():.0f} GB (dataset mean),"
        f" {5000 * np.average(x, weights=w):.0f} GB (subject-weighted),"
        f" {5000 * x.median():.0f} GB (median subject)"
    )


for col in OPTIONS:
    x = tot[col].dropna()
    w = tot.loc[x.index, "n_subject_dirs"]
    print(
        f"{col}: n datasets {len(x)}; per subject min {x.min():.2f} median {x.median():.2f}"
        f" mean {x.mean():.2f} max {x.max():.2f} GB;"
        f" subject-weighted mean {np.average(x, weights=w):.2f} GB;"
        f" sum over datasets of (first subject x n subjects) {float((x * w).sum()):.0f} GB"
        f" over {int(w.sum())} subjects"
    )
    per_dataset = {a: round(float(v), 1) for a, v in (x * w).items()}
    print("   per dataset (first subject x n subjects), GB:", per_dataset)
# combined route: T1w preproc where it exists (a), raw + transforms elsewhere
comb = tot.option_a_bytes.fillna(tot.option_c_minimal_bytes)
comb_c = tot.option_c_bytes.fillna(tot.option_c_minimal_bytes)
for label, x in (
    ("a where available, else raw+transforms", comb),
    ("c where available, else raw+transforms", comb_c),
):
    x = x.dropna()
    w = tot.loc[x.index, "n_subject_dirs"]
    print(
        f"{label}: n {len(x)}; per subject median {x.median():.2f} mean {x.mean():.2f} GB;"
        f" subject-weighted mean {np.average(x, weights=w):.2f} GB"
        f" -> 5,000 subjects: {scale_up(x)}"
    )
x = tot.option_c_minimal_bytes.dropna()
print(f"raw+transforms only (minimal layout, n {len(x)}): 5,000 subjects: {scale_up(x)}")
for col in ("option_a_bytes", "option_b_bytes", "option_c_bytes"):
    x = tot[col].dropna()
    line = f"{col} (full layout, n {len(x)}): 5,000 subjects: {scale_up(x)}"
    if args.without in x.index:
        line += f"; without {args.without} mean {5000 * x.drop(args.without).mean():.0f} GB"
    print(line)
tot.round(3).to_csv(OUT / "func_survey_subject_totals.csv")

print("\n=== per-run, by layout and version group ===")
runs_df["group"] = np.where(
    runs_df.fmriprep_version < "23.2", "full (<=23.0.x)", "minimal (>=23.2)"
)
for g, sub in runs_df.groupby("group"):
    print(g, "runs", len(sub), "datasets", sub.accession.nunique())
    for col in BYTES:
        dist(sub[col].tolist(), "  " + col)
by_acc = runs_df.groupby("accession")
both = runs_df.dropna(subset=["raw_bold_bytes", "preproc_T1w_bytes"]).groupby("accession")
print(
    "\nratio BOLD-space preproc / T1w-space preproc per dataset:",
    (by_acc.preproc_boldspace_bytes.sum() / by_acc.preproc_T1w_bytes.sum())
    .dropna()
    .round(2)
    .to_dict(),
)
print(
    "ratio T1w preproc / raw BOLD per dataset:",
    (both.preproc_T1w_bytes.sum() / both.raw_bold_bytes.sum()).round(2).to_dict(),
)
print(
    "ratio MNI res-2 preproc / T1w preproc:",
    (by_acc.preproc_MNI_bytes.sum() / by_acc.preproc_T1w_bytes.sum()).dropna().round(2).to_dict(),
)
print("hmc xfm kB per run: ", runs_df.hmc_xfm_bytes.dropna().div(1e3).describe().round(1).to_dict())
