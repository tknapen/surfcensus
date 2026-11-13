"""Step 4 in numbers: what differs between the raw-rigid and the preprocessed series, and what a
frame can show.

usage: analyze.py DATA OUT_DIR
       (writes OUT_DIR/analysis.json, absdiff_quiet_vs_moved.png and tsd_tsnr_raw_vs_preproc.png)
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))  # the sibling modules, also under -I
from epiview import Panels, compose
from geom import read_itk_ras, resample

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip(), file=sys.stderr)  # too few arguments: say how to run it
    sys.exit(2)
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TRANS = ["trans_x", "trans_y", "trans_z"]
ROT = ["rot_x", "rot_y", "rot_z"]


def rms(x):
    return np.sqrt(np.mean(x**2))


def over_10(diff):
    """Per frame, the fraction of pixels that differ by more than 10 gray levels."""
    return (diff > 10).mean(axis=1)


res = {}
for run in (1, 2):
    raw = nib.load(D / f"raw/run-{run}_bold.nii.gz")
    rd = np.asarray(raw.dataobj, dtype=np.float32)
    pre_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz")
    B = np.asarray(pre_img.dataobj, dtype=np.float32)
    mask_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz")
    mask = np.asarray(mask_img.dataobj) > 0
    m = read_itk_ras(D / f"deriv/run-{run}_from-scanner_to-T1w_mode-image_xfm.txt")
    A = np.stack(
        [
            resample(rd[..., t], raw.affine, B.shape[:3], pre_img.affine, m)
            for t in range(rd.shape[3])
        ],
        axis=3,
    )
    c = pd.read_csv(D / f"deriv/run-{run}_desc-confounds_timeseries.tsv", sep="\t")
    fd = c.framewise_displacement.to_numpy()
    # displacement from the reference of a point 50 mm from the centre (Power's sphere), from
    # the six parameters
    disp = c[TRANS].abs().sum(axis=1).to_numpy() + 50 * c[ROT].abs().sum(axis=1).to_numpy()
    n = A.shape[3]
    med = np.median(B[mask])
    r_ab = np.array([np.corrcoef(A[..., t][mask], B[..., t][mask])[0, 1] for t in range(n)])
    rms_ab = np.array([rms(A[..., t][mask] - B[..., t][mask]) for t in range(n)]) / med * 100
    dA = np.array([rms(A[..., t][mask] - A[..., t - 1][mask]) for t in range(1, n)]) / med * 100
    dB = np.array([rms(B[..., t][mask] - B[..., t - 1][mask]) for t in range(1, n)]) / med * 100
    out = dict(
        max_abs_trans_mm=float(c[TRANS].abs().to_numpy().max()),
        max_abs_rot_deg=float(np.degrees(c[ROT].abs().to_numpy().max())),
        disp50_max_mm=float(disp.max()),
        disp50_median_mm=float(np.median(disp)),
        fd_max=float(np.nanmax(fd)),
        fd_mean=float(np.nanmean(fd)),
        n_fd_over_05=int((fd > 0.5).sum()),
        n_fd_over_02=int((fd > 0.2).sum()),
        corr_A_B_min=float(r_ab.min()),
        corr_A_B_median=float(np.median(r_ab)),
        rms_A_minus_B_pct_median=float(np.median(rms_ab)),
        rms_A_minus_B_pct_max=float(rms_ab.max()),
        corr_rmsAB_with_disp=float(np.corrcoef(rms_ab, disp)[0, 1]),
        framediff_A_pct_median=float(np.median(dA)),
        framediff_B_pct_median=float(np.median(dB)),
        framediff_A_pct_max=float(dA.max()),
        framediff_B_pct_max=float(dB.max()),
        framediff_A_at_fd_peaks=[float(dA[i - 1]) for i in np.flatnonzero(fd > 0.5)],
        framediff_B_at_fd_peaks=[float(dB[i - 1]) for i in np.flatnonzero(fd > 0.5)],
    )
    # temporal SNR, raw-rigid against preprocessed (no detrending)
    tsA = A.mean(3) / np.maximum(A.std(3), 1e-6)
    tsB = B.mean(3) / np.maximum(B.std(3), 1e-6)
    out["tsnr_A_median"] = float(np.median(tsA[mask]))
    out["tsnr_B_median"] = float(np.median(tsB[mask]))
    # spikes: mean of every acquired slice of the raw data over time, robust z
    sl = rd.mean(axis=(0, 1))  # (slices, time)
    keep = sl.mean(axis=1) > 0.2 * sl.mean(axis=1).max()
    centre = np.median(sl, axis=1, keepdims=True)
    z = (sl - centre) / (1.4826 * np.median(np.abs(sl - centre), axis=1, keepdims=True))
    out["slice_spike_max_abs_z"] = float(np.abs(z[keep]).max())
    out["slice_spikes_over_z6"] = int((np.abs(z[keep]) > 6).sum())
    zi, ti = np.unravel_index(np.abs(np.where(keep[:, None], z, 0)).argmax(), z.shape)
    out["slice_spike_where"] = [int(zi), int(ti), float(100 * (sl[zi, ti] / np.median(sl[zi]) - 1))]
    # what the display shows: 8-bit gray levels of the three panels
    hi = float(np.percentile(B[..., ::10][mask], 99.5))
    com = np.array(np.nonzero(mask)).mean(axis=1)
    P = Panels(B.shape[:3], pre_img.affine, 2.0, np.round(com + np.array([4, 0, 1])).astype(int))
    gA = np.stack([P.strip(A[..., t], 0, hi) for t in range(n)]).astype(np.int16)
    gB = np.stack([P.strip(B[..., t], 0, hi) for t in range(n)]).astype(np.int16)
    inside = gB.mean(axis=0) > 40  # pixels of the head
    dAB = np.abs(gA - gB)
    out["display_absdiff_A_B_mean_gray"] = float(dAB[:, inside].mean())
    out["display_frac_pixels_diff_over_10_median"] = float(np.median(over_10(dAB[:, inside])))
    out["display_frac_pixels_diff_over_10_max"] = float(over_10(dAB[:, inside]).max())
    fA = np.abs(np.diff(gA, axis=0))[:, inside]
    fB = np.abs(np.diff(gB, axis=0))[:, inside]
    out["display_framediff_A_mean_gray"] = float(fA.mean())
    out["display_framediff_B_mean_gray"] = float(fB.mean())
    out["display_framediff_A_frac_over_10"] = [
        float(np.median(over_10(fA))),
        float(over_10(fA).max()),
    ]
    out["display_framediff_B_frac_over_10"] = [
        float(np.median(over_10(fB))),
        float(over_10(fB).max()),
    ]
    out["display_framediff_peak_volume_A_B"] = [
        int(over_10(fA).argmax()) + 1,
        int(over_10(fB).argmax()) + 1,
    ]
    res[f"run{run}"] = out
    print(f"run {run}:", json.dumps(out, indent=1))

    if run == 1:
        # pictures of what a gray frame does not show: |A - B| at the largest displacement and
        # at a quiet volume, and the temporal standard deviation of both series (same window)
        t_hi, t_lo = int(disp.argmax()), int(disp.argmin())
        print("largest displacement at volume", t_hi, disp[t_hi], "smallest at", t_lo, disp[t_lo])
        rows = [compose(P.strip(np.abs(A[..., t] - B[..., t]), 0, hi / 8)) for t in (t_lo, t_hi)]
        Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "absdiff_quiet_vs_moved.png")
        sdA, sdB = A.std(3), B.std(3)
        s_hi = float(np.percentile(sdA[mask], 99))
        rows = [
            np.concatenate(
                [compose(P.strip(sdA, 0, s_hi)), compose(P.strip(sdB, 0, s_hi))], axis=1
            ),
            np.concatenate([compose(P.strip(tsA, 0, 120)), compose(P.strip(tsB, 0, 120))], axis=1),
        ]
        Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "tsd_tsnr_raw_vs_preproc.png")
        res["run1"]["tsd_window_hi"] = s_hi

# dropout and brain outline: EPI mean against the T1w brain mask on the EPI grid
t1m = nib.load(D / "deriv/anat_desc-brain_mask.nii.gz")
t1m_data = np.asarray(t1m.dataobj, dtype=np.float32)
for run in (1, 2):
    pre_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-preproc_bold.nii.gz")
    aff = pre_img.affine
    mean = np.asarray(pre_img.dataobj, dtype=np.float32).mean(axis=3)
    # the T1w brain mask on the EPI grid, linear: 1 where a voxel is wholly inside it
    t1soft = resample(t1m_data, t1m.affine, mean.shape, aff, order=1)
    tm = t1soft > 0.99  # voxels wholly inside the T1w brain mask
    em_img = nib.load(D / f"deriv/run-{run}_space-T1w_desc-brain_mask.nii.gz")
    em = np.asarray(em_img.dataobj) > 0
    med = np.median(mean[tm])
    low = tm & (mean < 0.4 * med)
    ijk = np.array(np.nonzero(low)).T
    xyz = ijk @ aff[:3, :3].T + aff[:3, 3]
    ctr = (np.array(np.nonzero(tm)).T @ aff[:3, :3].T + aff[:3, 3]).mean(axis=0)
    rel = xyz - ctr
    below = rel[:, 2] < -15
    o = dict(
        t1w_mask_voxels=int(tm.sum()),
        low_signal_voxels=int(low.sum()),
        low_signal_frac=float(low.sum() / tm.sum()),
        low_below_centre_frac=float(below.mean()) if len(rel) else None,
        low_anterior_inferior_frac=float((below & (rel[:, 1] > 0)).mean()) if len(rel) else None,
        epi_mask_outside_t1w_mask=int((em & ~(t1soft > 0.01)).sum()),
        t1w_inside_not_in_epi_mask=int((tm & ~em).sum()),
    )
    # outline along the phase-encoding axis (y): where the EPI falls below 40% of the brain
    # median, against the T1w mask edge
    ant, post = [], []
    k0 = round(np.array(np.nonzero(tm)).mean(axis=1)[2])
    for k in range(k0 - 3, k0 + 6):
        for i in range(mean.shape[0]):
            a, b = t1soft[i, :, k] > 0.5, mean[i, :, k] > 0.4 * med
            if a.sum() > 25 and b.sum() > 25:
                ya, yb = np.flatnonzero(a), np.flatnonzero(b)
                ant.append((yb.max() - ya.max()) * aff[1, 1])
                post.append((ya.min() - yb.min()) * aff[1, 1])
    o["epi_edge_beyond_t1w_mask_anterior_mm_median"] = float(np.median(ant))
    o["epi_edge_beyond_t1w_mask_posterior_mm_median"] = float(np.median(post))
    o["n_lines"] = len(ant)
    res[f"run{run}_dropout"] = o
    print(f"run {run} dropout/outline:", json.dumps(o))
(OUT / "analysis.json").write_text(json.dumps(res, indent=1))
