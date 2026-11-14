"""A synthetic long run on a fine grid, as local files that pack_run.py takes: to test what the
player does with a run that does not fit in memory decoded.

usage: pack_synth.py DATA OUT [--volumes 1200] [--voxel 2.0] [--tr 0.8] [--seed 1]

DATA is the worked example as pipeline/fetch.py writes it; OUT gets deriv/ and raw/ of one run
named run-1 (about 1.5 GB for the defaults: both series are noise-like and are written whole).
The run is made up; only its anatomy is real, so that the real surfaces lie on it:

* the display grid covers the field of view of the worked example's T1w-space series plus 6 mm
  on every side, in voxels of --voxel mm;
* the preprocessed series is the temporal mean of the worked example's run 1 on that grid, times
  a slow drift, with Rician noise whose sigma is the median temporal SD of the real series
  inside the brain mask; nothing moves in it;
* the raw series is the same picture on a scanner grid that is rotated and shifted against the
  anatomy, moved from volume to volume by a made-up head motion (a slow wander and a jump every
  150 volumes or so), with noise of its own; from-scanner_to-T1w is the exact transform of
  volume 0;
* the first two volumes are brighter and marked non-steady-state in the confounds table, whose
  framewise displacement follows from the made-up motion and whose DVARS column is made up.

Every volume follows from --seed and its index alone, so two calls write the same series.
"""

import gzip
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))  # also under -I
from geom import FLIP, resample

PAD_MM = 6.0
JUMP_EVERY = 150


def rigid(angles_deg, shift, centre):
    """4x4 rotation by three angles (about x, y, z, degrees) around a centre, then a shift."""
    ax, ay, az = np.radians(angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = np.asarray(centre) - m[:3, :3] @ np.asarray(centre) + np.asarray(shift)
    return m


def motion(n, seed):
    """Six made-up motion parameters per volume (3 shifts in mm, 3 angles in degrees): a slow
    wander with a jump every JUMP_EVERY volumes or so. Volume 0 is at rest."""
    rng = np.random.default_rng([seed, 10**6])
    step = rng.normal(0, 1, (n, 6)) * np.array([0.03, 0.03, 0.05, 0.02, 0.02, 0.02])
    jumps = rng.random(n) < 1.0 / JUMP_EVERY
    step[jumps] += rng.normal(0, 1, (int(jumps.sum()), 6)) * np.array(
        [0.5, 0.5, 0.9, 0.4, 0.3, 0.3]
    )
    step[0] = 0
    par = np.zeros((n, 6))
    for t in range(1, n):  # the wander is pulled back to rest, so it stays within a few mm
        par[t] = 0.995 * par[t - 1] + step[t]
    return par


def framewise_displacement(par):
    """Power's framewise displacement from the parameters (angles on a sphere of 50 mm)."""
    d = np.abs(np.diff(par, axis=0))
    return np.concatenate(
        [[np.nan], d[:, :3].sum(axis=1) + (np.radians(d[:, 3:]) * 50).sum(axis=1)]
    )


def write_series(path, shape, affine, tr, volumes):
    """Write an int16 4D NIfTI volume by volume, without holding the series in memory."""
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    hdr.set_data_shape((*shape, len(volumes)))
    hdr.set_zooms((*np.sqrt((affine[:3, :3] ** 2).sum(axis=0)), tr))
    hdr.set_xyzt_units("mm", "sec")
    hdr.set_sform(affine, code=1)
    hdr.set_qform(affine, code=1)
    hdr["vox_offset"] = 352
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as f,
    ):
        hdr.write_to(f)
        f.write(b"\0\0\0\0")
        for vol in volumes:
            f.write(np.ascontiguousarray(vol, "<i2").tobytes(order="F"))


class Synthetic:
    """The made-up run: grids, transform, motion, and any volume of either series."""

    def __init__(self, data, n=1200, voxel=2.0, tr=0.8, seed=1):
        deriv = Path(data) / "deriv"
        pre = nib.load(deriv / "run-1_space-T1w_desc-preproc_bold.nii.gz")
        series = np.asarray(pre.dataobj, np.float32)
        mask = np.asarray(nib.load(deriv / "run-1_space-T1w_desc-brain_mask.nii.gz").dataobj) > 0
        mean, sd = series.mean(axis=3), series.std(axis=3)
        self.n, self.tr, self.seed = int(n), float(tr), int(seed)
        self.sigma = float(np.median(sd[mask]))
        size = np.sqrt((pre.affine[:3, :3] ** 2).sum(axis=0))
        extent = np.array(pre.shape[:3]) * size
        self.shape = tuple(int(x) for x in np.ceil((extent + 2 * PAD_MM) / voxel))
        corner = pre.affine[:3, 3] - size / 2 - PAD_MM  # the low corner of the padded box
        self.affine = np.diag([voxel, voxel, voxel, 1.0])
        self.affine[:3, 3] = corner + voxel / 2
        self.base = np.maximum(resample(mean, pre.affine, self.shape, self.affine, order=3), 0)
        self.mask = (
            resample(mask.astype(np.float32), pre.affine, self.shape, self.affine, order=0) > 0.5
        )
        centre = self.affine[:3, 3] + voxel * (np.array(self.shape) - 1) / 2
        # T1w point -> scanner point at volume 0: the stored transform of the run
        self.t1w_to_scanner = rigid((3.0, -2.0, 4.0), (2.0, -3.0, 5.0), centre)
        self.scanner_centre = self.t1w_to_scanner[:3, :3] @ centre + self.t1w_to_scanner[:3, 3]
        self.raw_shape = tuple(int(2 * round(x / voxel / 2)) for x in (192.0, 208.0, 144.0))
        self.raw_affine = np.diag([voxel, voxel, voxel, 1.0])
        self.raw_affine[:3, 3] = self.scanner_centre - voxel * (np.array(self.raw_shape) - 1) / 2
        self.par = motion(self.n, self.seed)
        self.fd = framewise_displacement(self.par)
        rng = np.random.default_rng([self.seed, 10**6 + 1])
        self.dvars = 1.0 + 0.05 * rng.normal(size=self.n) + 1.5 * np.nan_to_num(self.fd)
        self.dvars[0] = np.nan
        self.drift = 1.0 + 0.01 * np.sin(np.arange(self.n) * self.tr / 97.0)
        self.drift[:2] *= (1.35, 1.10)  # not yet in the steady state
        self.non_steady = (0, 1)

    def _noisy(self, signal, t, stream):
        rng = np.random.default_rng([self.seed, t, stream])
        a = signal * self.drift[t] + rng.normal(0, self.sigma, signal.shape).astype(np.float32)
        b = rng.normal(0, self.sigma, signal.shape).astype(np.float32)
        return np.clip(np.rint(np.sqrt(a * a + b * b)), 0, 32767).astype(np.int16)

    def pre_volume(self, t):
        """Volume t of the preprocessed series, int16 on the display grid."""
        return self._noisy(self.base, t, 0)

    def raw_volume(self, t):
        """Volume t of the raw series, int16 on the scanner grid."""
        move = rigid(self.par[t, 3:], self.par[t, :3], self.scanner_centre)
        # a scanner point -> where its tissue is in the anatomy at volume t
        back = np.linalg.inv(move @ self.t1w_to_scanner)
        signal = resample(self.base, self.affine, self.raw_shape, self.raw_affine, back, order=1)
        return self._noisy(signal, t, 1)

    def itk_text(self):
        m = FLIP @ self.t1w_to_scanner @ FLIP  # RAS -> LPS
        numbers = [*m[:3, :3].ravel(), *m[:3, 3]]
        return (
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: MatrixOffsetTransformBase_double_3_3\n"
            f"Parameters: {' '.join(repr(float(x)) for x in numbers)}\n"
            "FixedParameters: 0 0 0\n"
        )

    def confounds(self):
        def cell(x):
            return "n/a" if np.isnan(x) else f"{x:.6f}"

        names = ["framewise_displacement", "std_dvars"]
        names += [f"non_steady_state_outlier{k:02d}" for k in range(len(self.non_steady))]
        rows = ["\t".join(names)]
        for t in range(self.n):
            flags = ["1" if t == v else "0" for v in self.non_steady]
            rows.append("\t".join([cell(self.fd[t]), cell(self.dvars[t]), *flags]))
        return "\n".join(rows) + "\n"


class _Volumes:
    """The volumes of one series as a sequence that makes each when it is asked for."""

    def __init__(self, n, make):
        self.n, self.make = n, make

    def __len__(self):
        return self.n

    def __iter__(self):
        return (self.make(t) for t in range(self.n))


def write(data, out, **options):
    """Write the synthetic run under OUT; returns the Synthetic."""
    out = Path(out)
    (out / "deriv").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    s = Synthetic(data, **options)
    stem = out / "deriv" / "run-1"
    name = "anat_from-T1w_to-fsnative_mode-image_xfm.txt"
    shutil.copyfile(Path(data) / "deriv" / name, out / "deriv" / name)
    stem.with_name("run-1_from-scanner_to-T1w_mode-image_xfm.txt").write_text(s.itk_text())
    stem.with_name("run-1_desc-confounds_timeseries.tsv").write_text(s.confounds())
    sidecar = {"RepetitionTime": s.tr, "Synthetic": True}
    stem.with_name("run-1_space-T1w_desc-preproc_bold.json").write_text(json.dumps(sidecar) + "\n")
    nib.save(
        nib.Nifti1Image(s.mask.astype(np.uint8), s.affine),
        stem.with_name("run-1_space-T1w_desc-brain_mask.nii.gz"),
    )
    write_series(
        stem.with_name("run-1_space-T1w_desc-preproc_bold.nii.gz"),
        s.shape,
        s.affine,
        s.tr,
        _Volumes(s.n, s.pre_volume),
    )
    write_series(
        out / "raw" / "run-1_bold.nii.gz",
        s.raw_shape,
        s.raw_affine,
        s.tr,
        _Volumes(s.n, s.raw_volume),
    )
    return s


def main():
    args = sys.argv[1:]
    options = {"--volumes": 1200, "--voxel": 2.0, "--tr": 0.8, "--seed": 1}
    positional = []
    while args:
        arg = args.pop(0)
        if arg in options and args:
            options[arg] = type(options[arg])(args.pop(0))
        elif arg.startswith("--"):
            positional = []
            break
        else:
            positional.append(arg)
    if len(positional) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    s = write(
        positional[0],
        positional[1],
        n=options["--volumes"],
        voxel=options["--voxel"],
        tr=options["--tr"],
        seed=options["--seed"],
    )
    above = int(np.nansum(s.fd > 0.5))
    print(
        f"synthetic run: {s.n} volumes, display grid {s.shape} and raw grid {s.raw_shape} of"
        f" {s.affine[0, 0]:g} mm, noise sigma {s.sigma:.1f}; framewise displacement above 0.5 mm"
        f" in {above} volumes, largest {np.nanmax(s.fd):.2f} mm"
    )


if __name__ == "__main__":
    main()
