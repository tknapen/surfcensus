"""One registration per session: the anchor run, the raw stream of any run of the session through
the anchor's transform, and the check that the runs of a session share a scanner frame.

usage: session.py anchor BOLD_JSON_OR_NAME [BOLD_JSON_OR_NAME ...]
       session.py frame MASK ANCHOR_XFM RUN_XFM [RUN_XFM ...] [--limit MM]
       session.py digest RAW_BOLD GRID_IMAGE ANCHOR_XFM [--portable]

anchor  The arguments are the raw BOLD runs of one or more sessions: file names, or paths of
        their JSON sidecars, which are then read for AcquisitionTime. Prints, per session, the
        anchor run and which rule chose it.
frame   MASK is a brain mask in T1w space (NIfTI); the others are from-scanner_to-T1w ITK text
        transforms, the anchor's first. Prints, per run, how far the run lies from where its own
        transform puts it when it is shown through the anchor's, and whether that is flagged.
        Exit status 1 when a run is flagged.
digest  SHA-256 over the raw stream of RAW_BOLD (a raw series) on the grid of GRID_IMAGE (any
        image in T1w space, such as the space-T1w_boldref of a run) through ANCHOR_XFM. Two
        calls, two processes or two machines can compare the line. --portable computes the
        voxel map without numpy's linear algebra (see voxel_map).

Decision of Tomas Knapen, 2026-11-13: "Raw bids means single per session registration to the
anatomy to ensure comparability across sessions, processing stages, and subjects." How it is
read here (the reading is that of the session that wrote this module, not his words): the raw
series is shown in fMRIPrep's T1w space through ONE rigid transform per session, the stored
from-scanner_to-T1w transform of the session's anchor run, applied to every volume of every run
of the session. Nothing is registered anew.

The anchor (pick_anchor), fixed before any image was looked at: the run of the session that was
acquired first, by AcquisitionTime of the raw JSON sidecars when every run of the session has a
readable one; else the first in BIDS name order (name_order: task by name, run index as a
number). Which rule applied is recorded. A dataset without ses- folders has one session per
subject.

Applying the anchor's transform to another run assumes that the runs share the scanner frame of
reference. frame_displacement measures what the stored transforms say about that, and
frame_flag is the rule for a session whose runs do not plausibly share one.
"""

import datetime
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))  # also under -I
from geom import read_itk_ras

# ---- names, sessions and the anchor ----------------------------------------------------------

ORDER_ENTITIES = ("task", "acq", "ce", "rec", "dir")
RULE_TIME, RULE_NAME = "acquisition_time", "name_order"
_INDEX = re.compile(r"\d+")


def parse_name(name):
    """The entities of a BIDS file name or stem: {'sub': '01', 'task': 'rest', 'run': '2'}."""
    stem = str(name).rsplit("/", 1)[-1].split(".", 1)[0]
    out = {}
    for part in stem.split("_"):
        key, sep, label = part.partition("-")
        if sep and key not in out:
            out[key] = label
    return out


def _index(label):
    """A run or echo index as a number; a missing or unreadable one sorts first."""
    return int(label) if label is not None and _INDEX.fullmatch(label) else 0


def session_of(entities):
    """The session label of a run, '' in a dataset without ses- folders."""
    return entities.get("ses", "")


def run_key(entities):
    """A run within its session, the same for the raw file and its derivatives: task and the
    other labels as they are, the run index as a number (run-01 and run-1 are one run), no
    subject, session or echo."""
    parts = [f"{k}-{entities[k]}" for k in ORDER_ENTITIES if k in entities]
    if "run" in entities:
        parts.append(f"run-{_index(entities['run'])}")
    return "_".join(parts)


def name_order(entities):
    """Sort key of the name-order rule: task by name, then acq, ce, rec and dir by name (absent
    first), then the run index as a number (absent first), then the echo index."""
    labels = tuple(entities.get(k, "") for k in ORDER_ENTITIES)
    return (*labels, _index(entities.get("run")), _index(entities.get("echo")))


def acquisition_seconds(value):
    """AcquisitionTime of a sidecar as (kind, seconds), or None when it cannot be read.

    Two forms are read: a time of day H:M:S[.f] (kind 'time'; sidecars that were read write it
    without leading zeros, as in 17:30:2.685000) and an ISO date and time (kind 'datetime'). A
    time of day says nothing about the date, so a session that runs past midnight is ordered
    wrongly; that case is not handled.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if m:
        h, minute, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if h < 24 and minute < 60 and s < 61:
            return "time", h * 3600 + minute * 60 + s
        return None
    try:
        stamp = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if "T" not in text and " " not in text:
        return None  # a date alone orders nothing within a session
    return "datetime", (stamp.replace(tzinfo=None) - datetime.datetime(1970, 1, 1)).total_seconds()


@dataclass(frozen=True)
class Anchor:
    """The anchor run of a session and how it was chosen."""

    name: str  # the run as it was given to pick_anchor
    rule: str  # RULE_TIME or RULE_NAME
    why: str  # one line: what the rule saw


def pick_anchor(runs):
    """The anchor of ONE session.

    runs: (name, AcquisitionTime or None) for every run of the session; name is a BIDS file name
    or stem and AcquisitionTime the value of the raw JSON sidecar. Runs with equal times go by
    name order. A multi-echo run should be given once (one echo, or the name without echo).

    Give the runs that have a stored from-scanner_to-T1w transform: the anchor's transform is
    the session's registration, so a raw run that the derivative has no transform for cannot be
    the anchor, and "every run has an AcquisitionTime" is asked of the runs that are given.
    """
    runs = [(str(name), when) for name, when in runs]
    if not runs:
        raise ValueError("a session without runs has no anchor")
    if len({session_of(parse_name(name)) for name, _ in runs}) > 1:
        raise ValueError("pick_anchor takes the runs of one session")
    by_name = sorted(runs, key=lambda r: (name_order(parse_name(r[0])), r[0]))
    times = [acquisition_seconds(when) for _, when in runs]
    n_read = sum(t is not None for t in times)
    if n_read < len(runs):
        n_given = sum(when is not None for _, when in runs)
        why = f"AcquisitionTime given for {n_given} and readable for {n_read} of {len(runs)} runs"
        return Anchor(by_name[0][0], RULE_NAME, why)
    if len({kind for kind, _ in times}) > 1:
        why = "AcquisitionTime is a time of day in some runs and a date and time in others"
        return Anchor(by_name[0][0], RULE_NAME, why)
    seconds = {name: t[1] for (name, _), t in zip(runs, times, strict=True)}
    first = min(by_name, key=lambda r: seconds[r[0]])  # min keeps the first of equal times
    why = f"earliest AcquisitionTime of {len(runs)} runs ({dict(runs)[first[0]]})"
    return Anchor(first[0], RULE_TIME, why)


def sessions(names):
    """(subject, session) -> the names of its runs in name order, from BIDS file names."""
    out = {}
    for name in names:
        ent = parse_name(name)
        out.setdefault((ent.get("sub", ""), session_of(ent)), []).append(str(name))
    return {
        key: sorted(group, key=lambda n: (name_order(parse_name(n)), n))
        for key, group in sorted(out.items())
    }


def sidecar_time(path):
    """AcquisitionTime of a raw JSON sidecar, None when the file or the field is absent."""
    try:
        value = json.loads(Path(path).read_text()).get("AcquisitionTime")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


# ---- the raw stream of a run through the anchor's transform ----------------------------------


def _rows(m):
    return [[float(x) for x in row] for row in np.asarray(m, dtype=float)]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse_affine(a):
    """Inverse of a 4x4 affine [[L, t], [0, 1]] by cofactors, in Python floats."""
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = (row[:3] for row in a[:3])
    c00, c01, c02 = a11 * a22 - a12 * a21, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20
    det = a00 * c00 + a01 * c01 + a02 * c02
    if det == 0.0 or not math.isfinite(det):
        raise ValueError("the affine of the run cannot be inverted")
    inv = [
        [c00 / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det],
        [c01 / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det],
        [c02 / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det],
    ]
    t = [a[0][3], a[1][3], a[2][3]]
    out = [[*inv[i], -sum(inv[i][k] * t[k] for k in range(3))] for i in range(3)]
    return [*out, [0.0, 0.0, 0.0, 1.0]]


def voxel_map(run_affine, t1w_to_scanner, grid_affine, portable=False):
    """4x4 map from a voxel of the display grid to the voxel of the run that is shown there.

    grid voxel -> T1w point (grid_affine) -> scanner point (t1w_to_scanner, the point map of the
    ANCHOR's from-scanner_to-T1w file, as geom.read_itk_ras returns it) -> voxel of THIS run
    (inverse of the run's own affine).

    By default the product is the expression of pipeline/geom.py's resample, so a run through
    its own transform has the bytes of the feasibility study's per-run stream. That expression
    uses numpy's inverse and matrix product, whose last bits can depend on the LAPACK and BLAS
    that numpy is linked with. portable=True computes the same map in Python floats in a fixed
    order instead. On the worked example the two maps differ by at most 5.3e-15 voxel, and that
    changes single float32 values in 5 of the 146 volumes of run 1 by at most 9.5e-7 (measured
    2026-11-13 on one machine).
    """
    if not portable:
        return np.linalg.inv(run_affine) @ t1w_to_scanner @ grid_affine
    m = _mul(_mul(_inverse_affine(_rows(run_affine)), _rows(t1w_to_scanner)), _rows(grid_affine))
    return np.array(m, dtype=np.float64)


_INSIDE = {}


def resample_voxels(vol, vox, grid_shape, order=3):
    """`vol` at the voxel centres of the display grid, given the voxel map of voxel_map.

    The interpolation of pipeline/geom.py's resample: B-spline of the given order (cubic), edge
    value up to half a voxel beyond the outermost voxel centres, 0 outside the field of view.
    """
    out = ndimage.affine_transform(
        vol,
        vox[:3, :3],
        offset=vox[:3, 3],
        output_shape=tuple(grid_shape),
        order=order,
        mode="nearest",
        output=np.float32,
    )
    key = (vol.shape, tuple(grid_shape), vox.tobytes())
    if key not in _INSIDE:  # nearest-voxel lookup of a volume of ones: 1 inside the field of view
        if len(_INSIDE) > 64:
            _INSIDE.clear()
        _INSIDE[key] = ndimage.affine_transform(
            np.ones(vol.shape, np.uint8),
            vox[:3, :3],
            offset=vox[:3, 3],
            output_shape=tuple(grid_shape),
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)
    out[~_INSIDE[key]] = 0
    return out


def session_volume(
    vol, run_affine, grid_shape, grid_affine, anchor_t1w_to_scanner, order=3, portable=False
):
    """One raw volume of any run of a session on the display grid in T1w space, float32.

    vol and run_affine are the volume and the affine of the run's own image header;
    anchor_t1w_to_scanner is read_itk_ras of the from-scanner_to-T1w file of the session's
    anchor run. With the run's own transform in its place this is the per-run stream of
    pipeline/make_videos.py, bit for bit unless portable is set (voxel_map).
    """
    vox = voxel_map(run_affine, anchor_t1w_to_scanner, grid_affine, portable)
    return resample_voxels(vol, vox, grid_shape, order=order)


def reference_of(raw_path):
    """The BOLD-space reference image that fMRIPrep made of a raw run, when it lies beside it.

    In the layout of DATA (raw/ and deriv/ side by side, as pipeline/fetch.py, session_fetch.py
    and fetch_run.py write it) that is deriv/<stem>_boldref.nii.gz of the derivative run with the
    same session and run_key. None when there is none.
    """
    path = Path(raw_path)
    if path.parent.name != "raw":
        return None
    stem = path.name.split(".nii")[0]
    stem = stem[: -len("_bold")] if stem.endswith("_bold") else stem
    want = (session_of(parse_name(stem)), run_key(parse_name(stem)))
    for ref in sorted((path.parent.parent / "deriv").glob("*_boldref.nii.gz")):
        ref_stem = ref.name[: -len("_boldref.nii.gz")]
        ent = parse_name(ref_stem)
        if "space" not in ent and (session_of(ent), run_key(ent)) == want:
            return ref
    return None


def raw_affine(raw_path, img=None):
    """The voxel-to-scanner affine of a raw series in the frame of its stored transform.

    The stored from-scanner_to-T1w transform maps to the scanner space of fMRIPrep's reference
    image of the run, which fMRIPrep made after validating the raw header. nibabel's img.affine
    prefers the sform; fMRIPrep sometimes keeps the sform and sometimes overwrites it with the
    qform when the two differ. So the affine of the run's BOLD-space reference image decides:
    img.affine when they agree (to 1e-4), else the qform when that agrees, else the reference
    image's affine. Without a reference image, or with one on another grid, img.affine.

    Added in the integration of stage 1 (2026-11-14). Of the 15 raw series of the selected
    sessions, five store a qform and an sform (code 2) that differ by 4 to 25 mm: in four of one
    session fMRIPrep's reference image has the qform, in one of another session the sform. Through
    img.affine alone the raw stream of the first session lay about 22 mm from the anatomy
    (temporal-mean r 0.27 against fMRIPrep's series). The worked example's headers agree.
    """
    img = nib.load(raw_path) if img is None else img
    ref = reference_of(raw_path)
    if ref is None:
        return img.affine
    ref_img = nib.load(ref)
    if tuple(ref_img.shape[:3]) != tuple(img.shape[:3]):
        return img.affine
    if np.allclose(ref_img.affine, img.affine, atol=1e-4):
        return img.affine
    qform = img.header.get_qform()
    if np.allclose(ref_img.affine, qform, atol=1e-4):
        return qform
    return ref_img.affine


def raw_volumes(raw_path):
    """(array proxy of the stored values, affine, slope, intercept) of a raw series.

    The values are scaled per volume, as pipeline/make_videos.py does:
    vol = stored[..., t].astype(float32) * slope + intercept. The affine is the one fMRIPrep
    of the run's stored transform (raw_affine), which is img.affine whenever the reference image
    agrees with it.
    """
    img = nib.load(raw_path)
    return (
        img.dataobj.get_unscaled(),
        raw_affine(raw_path, img),
        float(img.dataobj.slope),
        float(img.dataobj.inter),
    )


def session_stream(
    raw_path, grid_shape, grid_affine, anchor_t1w_to_scanner, volumes=None, portable=False
):
    """The raw stream of one run: yields (t, float32 volume on the display grid) per volume.

    raw_path: the raw BOLD series of the run (3D or 4D NIfTI). volumes: indices to yield, all
    when None. Every volume goes through the same voxel map, so the series is neither
    motion-corrected nor slice-time-corrected nor unwarped. portable: see voxel_map.
    """
    stored, run_affine, slope, inter = raw_volumes(raw_path)
    vox = voxel_map(run_affine, anchor_t1w_to_scanner, grid_affine, portable)
    n = stored.shape[3] if len(stored.shape) > 3 else 1
    for t in range(n) if volumes is None else volumes:
        vol = stored[..., t] if len(stored.shape) > 3 else stored
        yield t, resample_voxels(vol.astype(np.float32) * slope + inter, vox, grid_shape)


# ---- the frame check -------------------------------------------------------------------------

# A run is flagged when the displacement between its own stored transform and the anchor's
# exceeds this somewhere in its brain mask. Where the number comes from is in frame_flag.
FRAME_LIMIT_MM = 10.0


def frame_displacement(run_t1w_to_scanner, anchor_t1w_to_scanner, mask, mask_affine):
    """How far a run lies, shown through the anchor's transform, from where its own puts it.

    Both transforms are point maps T1w -> scanner (read_itk_ras of the from-scanner_to-T1w
    files). The run's own registration says that the tissue at the T1w point p was at the
    scanner point M_run p. Through the anchor's transform that scanner point is drawn at
    q = inv(M_anchor) M_run p. The displacement q - p is evaluated at the centres of the voxels
    of the brain mask (a boolean volume with its affine, in T1w space).

    Returns mean_mm and max_mm over the mask, the rigid parts of inv(M_anchor) M_run
    (rotation_deg about an axis through the centroid of the mask, translation_mm of that
    centroid with its three components translation_ras) and shear_scale, the largest deviation
    of a singular value of its 3x3 part from 1 (0 for a rigid pair), and n_voxels.
    """
    d = np.linalg.inv(anchor_t1w_to_scanner) @ run_t1w_to_scanner
    ijk = np.argwhere(np.asarray(mask) > 0).astype(float)
    if not len(ijk):
        raise ValueError("the brain mask is empty")
    a = np.asarray(mask_affine, float)
    p = ijk @ a[:3, :3].T + a[:3, 3]
    q = p @ d[:3, :3].T + d[:3, 3]
    dist = np.sqrt(((q - p) ** 2).sum(axis=1))
    u, s, vt = np.linalg.svd(d[:3, :3])
    r = u @ vt  # the rotation nearest the 3x3 part
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    angle = math.degrees(math.atan2(float(np.linalg.norm(w)), float(np.trace(r)) - 1.0))
    c = p.mean(axis=0)
    shift = d[:3, :3] @ c + d[:3, 3] - c
    return {
        "mean_mm": float(dist.mean()),
        "max_mm": float(dist.max()),
        "rotation_deg": angle,
        "translation_mm": float(np.linalg.norm(shift)),
        "translation_ras": [float(x) for x in shift],
        "shear_scale": float(np.abs(s - 1.0).max()),
        "n_voxels": len(ijk),
    }


def frame_flag(displacement, limit=FRAME_LIMIT_MM):
    """True when a run does not plausibly share the scanner frame of its session's anchor.

    The rule: the largest displacement in the brain mask (frame_displacement, max_mm) exceeds
    the limit, 10 mm by default.

    Where 10 mm comes from (measured 2026-11-13 with session_frames.py on the first subject of
    70 derivatives made with fMRIPrep 23.0.2 or older: 103 sessions, 77 of them with two or more
    runs, 452 pairs of a run and its anchor): within a session the largest displacement is 2.7
    mm in the median pair and 9.8 mm at the 95th percentile, and 21 pairs, in 12 of the 77
    sessions, exceed 10 mm. Of 33 pairs of anchors of two sessions of one subject (10
    subjects), which are known not to share a frame, 24 exceed 10 mm and 9 do not (6 of the 9
    in one subject). So the limit is about the 95th percentile of what runs of one session
    show, and three quarters of the pairs known to be in different frames lie beyond it.

    What the flag cannot do. The stored transforms cannot tell a head that moved between two
    runs from a subject who was taken out and put back, or from a registration that failed:
    all three give a displacement, and the two distributions overlap. Below the limit a session
    can still hold runs of two frames (9 of 33 known pairs), and above it a run may have moved
    that far in one frame. Either way a flagged run would be drawn more than 10 mm from where
    its own registration puts it, so it is not to be shown as if it shared the anchor's frame.
    """
    return bool(displacement["max_mm"] > limit)


def check_frame(transforms, anchor, masks, limit=FRAME_LIMIT_MM):
    """The frame check of one session.

    transforms: run name -> 4x4 point map T1w -> scanner (read_itk_ras of the run's own
    from-scanner_to-T1w file), for every run of the session. anchor: the name of the anchor
    run. masks: run name -> (boolean brain mask in T1w space, its affine), or one such pair for
    all runs.

    Returns {"anchor", "limit_mm", "runs": {name: frame_displacement plus "flagged"},
    "flagged": [names], "shared_frame_plausible": no run is flagged, "anchor_is_outlier"}. The
    anchor's own entry is all zeros. anchor_is_outlier is True when there are at least two
    other runs, every one of them is flagged, and they lie within the limit of each other:
    then the anchor, or its stored transform, is the odd one, and the whole session would be
    drawn off the anatomy.
    """
    if anchor not in transforms:
        raise ValueError("the anchor is not among the runs")

    def mask_of(name):
        return masks[name] if isinstance(masks, dict) else masks

    runs = {}
    for name, own in transforms.items():
        mask, affine = mask_of(name)
        d = frame_displacement(own, transforms[anchor], mask, affine)
        if name == anchor:
            zero = {k: 0.0 for k in d if k not in ("n_voxels", "translation_ras")}
            d = {**d, **zero, "translation_ras": [0.0, 0.0, 0.0]}
        runs[name] = {**d, "flagged": frame_flag(d, limit)}
    others = [name for name in transforms if name != anchor]
    flagged = [name for name in others if runs[name]["flagged"]]
    outlier = False
    if len(others) >= 2 and len(flagged) == len(others):
        outlier = all(
            not frame_flag(frame_displacement(transforms[a], transforms[b], *mask_of(a)), limit)
            for i, a in enumerate(others)
            for b in others[i + 1 :]
        )
    return {
        "anchor": anchor,
        "limit_mm": float(limit),
        "runs": runs,
        "flagged": flagged,
        "shared_frame_plausible": not flagged,
        "anchor_is_outlier": outlier,
    }


def stream_digest(
    raw_path, grid_shape, grid_affine, anchor_t1w_to_scanner, volumes=None, portable=False
):
    """SHA-256 over the bytes of every volume of session_stream, in order: one line that two
    calls, two processes or two machines can compare."""
    h = hashlib.sha256()
    stream = session_stream(
        raw_path, grid_shape, grid_affine, anchor_t1w_to_scanner, volumes, portable
    )
    for _, vol in stream:
        h.update(np.ascontiguousarray(vol).tobytes())
    return h.hexdigest()


# ---- command line ----------------------------------------------------------------------------


def _anchor_command(args):
    given = {}
    for arg in args:
        path = Path(arg)
        name = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
        given[name] = sidecar_time(path) if path.name.endswith(".json") else None
    for (subject, session), names in sessions(given).items():
        anchor = pick_anchor([(name, given[name]) for name in names])
        parts = (f"sub-{subject}" if subject else "", f"ses-{session}" if session else "")
        label = "_".join(x for x in parts if x)
        print(f"{label or 'session'}: {len(names)} runs; anchor {anchor.name}")
        print(f"  rule {anchor.rule}: {anchor.why}")
    return 0


def _frame_command(args):
    limit = FRAME_LIMIT_MM
    if "--limit" in args:
        at = args.index("--limit")
        limit = float(args[at + 1])
        args = args[:at] + args[at + 2 :]
    if len(args) < 3:
        return None
    img = nib.load(args[0])
    transforms = {f"run {i}": read_itk_ras(path) for i, path in enumerate(args[1:])}
    res = check_frame(transforms, "run 0", (np.asarray(img.dataobj) > 0, img.affine), limit)
    print(f"anchor: {args[1]}")
    for name, path in zip(list(transforms)[1:], args[2:], strict=True):
        d = res["runs"][name]
        print(
            f"{path}: mean {d['mean_mm']:.2f} mm, max {d['max_mm']:.2f} mm in the mask;"
            f" rotation {d['rotation_deg']:.2f} deg, translation {d['translation_mm']:.2f} mm"
            f" (R {d['translation_ras'][0]:+.2f}, A {d['translation_ras'][1]:+.2f},"
            f" S {d['translation_ras'][2]:+.2f}){'; FLAGGED' if d['flagged'] else ''}"
        )
    verdict = "plausible" if res["shared_frame_plausible"] else "NOT plausible"
    print(f"a shared scanner frame is {verdict} at a limit of {limit:g} mm")
    if res["anchor_is_outlier"]:
        print(
            "every other run is flagged and they agree with each other: the anchor is the odd one"
        )
    return 0 if res["shared_frame_plausible"] else 1


def _digest_command(args):
    portable = "--portable" in args
    args = [a for a in args if a != "--portable"]
    if len(args) < 3:
        return None
    grid = nib.load(args[1])
    xfm = read_itk_ras(args[2])
    print(stream_digest(args[0], grid.shape[:3], grid.affine, xfm, portable=portable))
    return 0


def _usage():
    print(__doc__.strip().split("\n\nDecision of")[0], file=sys.stderr)
    sys.exit(2)


def main():
    commands = {"anchor": _anchor_command, "frame": _frame_command, "digest": _digest_command}
    if len(sys.argv) < 3 or sys.argv[1] not in commands:
        _usage()  # too few arguments: say how to run it
    status = commands[sys.argv[1]](sys.argv[2:])
    if status is None:
        _usage()
    sys.exit(status)


if __name__ == "__main__":
    main()
