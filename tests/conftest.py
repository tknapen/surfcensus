"""Runtime shape checking for the package, and the synthetic subjects shared by the tests.

The import hook must be installed before anything imports ``surfcensus``, so it comes first.
It decorates every function and dataclass of the package with jaxtyping + beartype. The
PEP 484 numeric tower is on, so an ``int`` passes where ``float`` is annotated, as it does
for the static checker.

``tests/`` is put on ``sys.path`` here: test modules import helpers with ``from synth import ...``.
The session fixtures are shared and must be treated as read-only; a test that needs to change
a subject writes its own with ``write_synthetic_subject(tmp_path, ...)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from jaxtyping import install_import_hook

install_import_hook(
    "surfcensus", "beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))"
)
sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from synth import (  # noqa: E402  # pyrefly: ignore[missing-import]
    SyntheticSubject,
    write_synthetic_subject,
)


@pytest.fixture(scope="session")
def synth_subject(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    """1 mm, 160^3, 10,242 vertices per surface, with brainmask."""
    return write_synthetic_subject(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def synth_subject_hires(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    """0.8 mm, 200^3: the same anatomy in mm on a finer grid."""
    return write_synthetic_subject(
        tmp_path_factory.mktemp("synth_hires"), voxel_size=0.8, shape=(200, 200, 200)
    )


@pytest.fixture(scope="session")
def synth_subject_pial_t1(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    """Pial surfaces only as ``surf/?h.pial.T1`` (no ``?h.pial``), 2,562 vertices per surface."""
    return write_synthetic_subject(
        tmp_path_factory.mktemp("synth_pial_t1"), pial_variant="pial.T1", subdivisions=4
    )


REAL_SUBJECT_URL = (
    "https://openneuro-derivatives.s3.amazonaws.com/"
    "fmriprep/ds000102-fmriprep/sourcedata/freesurfer/sub-01/"
)
REAL_SUBJECT_FILES = {  # sizes in bytes, from the bucket listing of 2026-11-12
    "mri/T1.mgz": 5_576_738,
    "mri/brainmask.mgz": 1_691_388,
    "surf/lh.white": 5_341_363,
    "surf/lh.pial": 5_341_363,
    "surf/rh.white": 5_291_503,
    "surf/rh.pial": 5_291_503,
}


@pytest.fixture(scope="session")
def real_subject(request: pytest.FixtureRequest) -> Path:
    """ds000102 sub-01 (FreeSurfer 6.0.1): the render roles and the brainmask, in the pytest cache.

    Network tests only. Anonymous, read-only GETs; files already present with the expected
    size are not fetched again.
    """
    import httpx

    root = request.config.cache.mkdir("ds000102-sub-01")
    with httpx.Client(timeout=120.0) as client:
        for name, size in REAL_SUBJECT_FILES.items():
            path = root / name
            if path.is_file() and path.stat().st_size == size:
                continue
            content = client.get(REAL_SUBJECT_URL + name).raise_for_status().content
            assert len(content) == size, f"{name}: {len(content)} bytes, expected {size}"
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            part.write_bytes(content)
            part.replace(path)
    return root
