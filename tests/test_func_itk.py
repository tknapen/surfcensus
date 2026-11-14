"""ITK text transforms as RAS point maps (``surfcensus.func.itk``).

Checked three ways: against the closed-form transforms of ``tests/synth_func.py``, against the
convention of nitransforms 24.1.1 written out here (``io/itk.py`` lines 84 to 91,
``ITKLinearTransform.to_ras``: ``LPS . C(+c) . M . C(-c) . LPS``), and against nitransforms
itself where it is importable (it is not a dependency of the package; the test skips without
it).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from synth_func import (  # pyrefly: ignore[missing-import]
    BRAIN_CENTRE,
    FSNATIVE_TO_T1W,
    GENERATIONS,
    SyntheticFuncDataset,
    apply,
    itk_text,
    rigid,
    write_func_dataset,
)

from surfcensus.func.itk import ItkError, read_itk, read_itk_file

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


@pytest.fixture(scope="module", params=GENERATIONS)
def dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SyntheticFuncDataset:
    return write_func_dataset(tmp_path_factory.mktemp("func"), generation=request.param)


def text_of(params: list[float], fixed: list[float], kind: str = "AffineTransform_double_3_3") -> str:
    numbers = " ".join(repr(float(x)) for x in params)
    centre = " ".join(repr(float(x)) for x in fixed)
    return (
        "#Insight Transform File V1.0\n#Transform 0\n"
        f"Transform: {kind}\nParameters: {numbers}\nFixedParameters: {centre}\n"
    )


def nitransforms_convention(params: list[float], fixed: list[float]) -> np.ndarray:
    """``ITKLinearTransform.to_ras`` of nitransforms 24.1.1, in float64."""
    m = np.eye(4)
    m[:3, :3] = np.reshape(params[:9], (3, 3))
    m[:3, 3] = params[9:]
    c_neg, c_pos = np.eye(4), np.eye(4)
    c_neg[:3, 3] = -np.asarray(fixed)
    c_pos[:3, 3] = np.asarray(fixed)
    return LPS @ c_pos @ m @ c_neg @ LPS


def random_transform(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Parameters and a centre that float32 holds exactly, so that nitransforms, which reads
    them as float32, sees the same numbers."""
    a = rigid(rng.uniform(-20, 20, 3), (0, 0, 0), centre=(0, 0, 0))[:3, :3]
    a = a * rng.uniform(0.9, 1.1, (3, 3)) ** 0.1
    t = rng.uniform(-30, 30, 3)
    c = rng.uniform(-80, 80, 3)
    params = [float(np.float32(x)) for x in [*a.ravel(), *t]]
    return params, [float(np.float32(x)) for x in c]


def test_the_transforms_of_the_fixture_are_their_point_maps(
    dataset: SyntheticFuncDataset,
) -> None:
    for run in dataset.runs:
        if "xfm_to_anat" not in run.files:
            continue
        matrix = np.asarray(read_itk_file(run.files["xfm_to_anat"]))
        np.testing.assert_allclose(matrix, run.anat_to_scanner, atol=1e-11)
        # the direction: the file maps a point of the anatomy to the run's scanner world
        point = np.asarray([BRAIN_CENTRE])
        np.testing.assert_allclose(
            apply(matrix, point), apply(run.anat_to_scanner, point), atol=1e-10
        )
        assert not np.allclose(matrix, np.linalg.inv(run.anat_to_scanner), atol=1e-3)
    for path in dataset.fsnative_files.values():
        np.testing.assert_allclose(np.asarray(read_itk_file(path)), FSNATIVE_TO_T1W, atol=1e-12)
        inverse = path.with_name(path.name.replace("from-T1w_to-fsnative", "from-fsnative_to-T1w"))
        np.testing.assert_allclose(
            np.asarray(read_itk_file(inverse)), np.linalg.inv(FSNATIVE_TO_T1W), atol=1e-12
        )


def test_the_centre_and_the_flip_are_those_of_nitransforms() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        params, fixed = random_transform(rng)
        got = np.asarray(read_itk(text_of(params, fixed)))
        np.testing.assert_allclose(got, nitransforms_convention(params, fixed), rtol=0, atol=1e-12)
        # a reader that ignored the centre, or the flip, would be off by millimetres
        assert not np.allclose(got, nitransforms_convention(params, [0.0, 0.0, 0.0]), atol=1e-2)
        no_flip = nitransforms_convention(params, fixed)
        assert not np.allclose(got, LPS @ no_flip @ LPS, atol=1e-2)
    # the worked case: a shift of 1 2 3 in LPS is -1 -2 3 in RAS; a centre does not move a shift
    for centre in ([0.0, 0.0, 0.0], [10.0, -20.0, 30.0]):
        m = read_itk(text_of([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3], centre))
        assert [row[3] for row in m[:3]] == [-1.0, -2.0, 3.0]
    # a rotation about a centre keeps the centre in place (the centre is given in LPS)
    params = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    m = np.asarray(read_itk(text_of(params, [-10.0, -20.0, 5.0])))
    np.testing.assert_allclose(apply(m, np.asarray([[10.0, 20.0, 5.0]])), [[10.0, 20.0, 5.0]])


def test_against_nitransforms_itself() -> None:
    itk = pytest.importorskip("nitransforms.io.itk")
    rng = np.random.default_rng(11)
    for _ in range(50):
        params, fixed = random_transform(rng)
        for kind in ("AffineTransform_double_3_3", "MatrixOffsetTransformBase_double_3_3"):
            text = text_of(params, fixed, kind)
            theirs = itk.ITKLinearTransform.from_string(text).to_ras()
            np.testing.assert_allclose(np.asarray(read_itk(text)), theirs, rtol=0, atol=1e-11)


def test_the_kinds_that_are_read() -> None:
    params = [1.0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 5, 6]
    for kind in (
        "AffineTransform_double_3_3",
        "AffineTransform_float_3_3",
        "MatrixOffsetTransformBase_double_3_3",
    ):
        assert read_itk(text_of(params, [1, 2, 3], kind))[0][3] == -4.0
    m = read_itk(text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]))
    assert all(math.copysign(1.0, x) == 1.0 for row in m for x in row if x == 0.0)
    assert all(type(x) is float for row in m for x in row)
    # blank lines, Windows line ends and a missing "#Transform 0" line are read
    text = text_of(params, [0, 0, 0]).replace("#Transform 0\n", "").replace("\n", "\r\n\n")
    assert read_itk(text)[1][3] == -5.0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Transform: AffineTransform_double_3_3\nParameters: 1 0 0 0 1 0 0 0 1 0 0 0\n"
        "FixedParameters: 0 0 0\n",  # no banner
        "#Insight Transform File V1.0\n",  # no transform
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0])
        + text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).split("\n", 1)[1],
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0], "Euler3DTransform_double_3_3"),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0], "AffineTransform_double_2_2"),
        text_of([1.0, 0, 0, 1, 0, 0], [0, 0], "AffineTransform_double_2_2"),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0]),  # 11 numbers
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0]),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).replace("1.0", "nan", 1),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).replace("1.0", "inf", 1),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).replace("1.0", "1_0", 1),
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).replace("1.0", "١", 1),
        text_of([1e300, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [1e300, 0, 0]),  # overflows
        text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).replace(
            "FixedParameters", "Fixed"
        ),
        "#Insight Transform File V1.0\nTransform: CompositeTransform_double_3_3\n"
        + text_of([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0]).split("\n", 2)[2],
    ],
)
def test_what_is_not_one_affine_transform(text: str) -> None:
    with pytest.raises(ItkError):
        read_itk(text)


def test_the_errors_of_a_file_name_the_file_not_its_path(tmp_path: Path) -> None:
    path = tmp_path / "sub-01_from-T1w_to-fsnative_mode-image_xfm.txt"
    path.write_text("#Insight Transform File V1.0\n")
    with pytest.raises(ItkError) as raised:
        read_itk_file(path)
    assert str(raised.value).startswith(path.name + ": ") and str(tmp_path) not in str(raised.value)
    path.write_bytes(b"#Insight Transform File V1.0\n\xff\n")
    with pytest.raises(ItkError, match="not UTF-8"):
        read_itk_file(path)
    path.write_text(itk_text(np.eye(4), (0, 0, 0), "AffineTransform") + "#" * (1024 * 1024))
    with pytest.raises(ItkError, match="too large"):
        read_itk_file(path)
    with pytest.raises(OSError) as missing:
        read_itk_file(tmp_path / "absent_xfm.txt")
    assert not isinstance(missing.value, ItkError)
    assert "absent_xfm.txt" in str(missing.value) and str(tmp_path) not in str(missing.value)
    assert missing.value.filename is None
