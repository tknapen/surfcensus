"""The frame check of stage 1 on the worked example (network): ``ds000102`` sub-01, run 2 of
``task-flanker`` against the anchor of its session, run 1 (name order: the raw runs have no
sidecar with ``AcquisitionTime``).

Stage 1 measured (``prototypes/func_viewer/stage1/check_session.py``, 2026-11-13): over the
39,801 voxels of run 2's ``space-T1w`` brain mask, a mean of 0.829 mm and a maximum of 1.087 mm,
a rotation of 0.205 degrees and a translation of 0.821 mm; not flagged at 10 mm.

Fetched once into the pytest cache, anonymously and read-only from the public
OpenNeuroDerivatives bucket (``fmriprep/ds000102-fmriprep/sub-01/func/``): the two
``from-scanner_to-T1w`` transforms (238 and 230 bytes) and the two ``space-T1w`` brain masks
(2,223 and 2,177 bytes), 4.9 kB in all. No series is fetched. The ITK files are read here, with
the convention of nitransforms that stage 1's review read (``y = M (x - c) + t + c`` in LPS,
mapping a point of T1w to the run's scanner space), not with ``func.itk``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from surfcensus.func import session
from surfcensus.records import AnchorCandidate, to_matrix4

pytestmark = pytest.mark.network

URL = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds000102-fmriprep/sub-01/func/"
STEM = "sub-01_task-flanker_run-{n}"
FILES = {  # sizes in bytes, from the bucket listing of 2026-11-14
    f"{STEM.format(n=1)}_from-scanner_to-T1w_mode-image_xfm.txt": 238,
    f"{STEM.format(n=2)}_from-scanner_to-T1w_mode-image_xfm.txt": 230,
    f"{STEM.format(n=1)}_space-T1w_desc-brain_mask.nii.gz": 2_223,
    f"{STEM.format(n=2)}_space-T1w_desc-brain_mask.nii.gz": 2_177,
}
EXPECTED = {"mean_mm": 0.829, "max_mm": 1.087, "rotation_deg": 0.205, "translation_mm": 0.821}
TOLERANCE = 0.002  # stage 1's


@pytest.fixture(scope="module")
def frame_files(request: pytest.FixtureRequest) -> Path:
    import httpx

    root = request.config.cache.mkdir("ds000102-sub-01-func-frame")
    with httpx.Client(timeout=120.0) as client:
        for name, size in FILES.items():
            path = root / name
            if path.is_file() and path.stat().st_size == size:
                continue
            content = client.get(URL + name).raise_for_status().content
            assert len(content) == size, f"{name}: {len(content)} bytes, expected {size}"
            part = path.with_name(path.name + ".part")
            part.write_bytes(content)
            part.replace(path)
    return root


def itk_ras(path: Path) -> np.ndarray:
    params = fixed = None
    for line in path.read_text().splitlines():
        if line.startswith("Parameters:"):
            params = np.array(line.split(":", 1)[1].split(), float)
        elif line.startswith("FixedParameters:"):
            fixed = np.array(line.split(":", 1)[1].split(), float)
    assert params is not None and params.size == 12
    c = fixed if fixed is not None and fixed.size == 3 else np.zeros(3)
    m = np.eye(4)
    m[:3, :3] = params[:9].reshape(3, 3)
    m[:3, 3] = params[9:] + c - m[:3, :3] @ c
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps @ m @ lps


def test_run_2_against_the_anchor_of_the_worked_example(frame_files: Path) -> None:
    runs = ["run-1", "run-2"]
    candidates = [
        AnchorCandidate(run=r, order=("flanker", "", "", "", "", n + 1, 0), acquisition_time=None)
        for n, r in enumerate(runs)
    ]
    anchor = session.pick_anchor(candidates[::-1])
    assert (anchor.run, anchor.rule) == ("run-1", "name_order")
    assert session.acquisition_order(candidates, anchor) == ("run-1", "run-2")
    transforms, masks = {}, {}
    for n, run in enumerate(runs, start=1):
        transforms[run] = to_matrix4(
            itk_ras(frame_files / f"{STEM.format(n=n)}_from-scanner_to-T1w_mode-image_xfm.txt")
        )
        img = nib.load(frame_files / f"{STEM.format(n=n)}_space-T1w_desc-brain_mask.nii.gz")
        masks[run] = (np.asarray(img.dataobj) > 0, img.affine)
    check = session.check_frame(transforms, anchor.run, masks)
    two = check.runs["run-2"]
    assert two.n_voxels == 39_801
    for name, value in EXPECTED.items():
        assert getattr(two, name) == pytest.approx(value, abs=TOLERANCE), name
    assert not two.flagged and check.shared_frame_plausible and not check.anchor_is_outlier
    assert two.shear_scale < 1e-3  # fMRIPrep's 6-dof registration
    # the same numbers through func.itk, the reader of package E1.A
    from surfcensus.func import itk

    for n, run in enumerate(runs, start=1):
        path = frame_files / f"{STEM.format(n=n)}_from-scanner_to-T1w_mode-image_xfm.txt"
        np.testing.assert_allclose(np.array(itk.read_itk_file(path)), transforms[run], atol=1e-9)
