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

import logging
import sys
from collections.abc import Iterator
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


@pytest.fixture(autouse=True)
def _package_logger_as_it_was() -> Iterator[None]:
    """The CLI puts a handler and a level on the package logger; no test inherits them."""
    logger = logging.getLogger("surfcensus")
    level, handlers = logger.level, list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers


BUCKET_URL = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/"
REAL_SUBJECT_URL = BUCKET_URL + "ds000102-fmriprep/sourcedata/freesurfer/sub-01/"
REAL_SUBJECT_FILES = {  # sizes in bytes, from the bucket listing of 2026-11-12
    # Tier C: what a render reads
    "mri/T1.mgz": 5_576_738,
    "mri/brainmask.mgz": 1_691_388,
    "surf/lh.white": 5_341_363,
    "surf/lh.pial": 5_341_363,
    "surf/rh.white": 5_291_503,
    "surf/rh.pial": 5_291_503,
    # Tier A: stats, build stamp, logs
    "stats/lh.aparc.stats": 6_717,
    "stats/rh.aparc.stats": 6_717,
    "stats/aseg.stats": 9_853,
    "scripts/build-stamp.txt": 58,
    "scripts/recon-all.log": 220_842,
    "scripts/recon-all-lh.log": 212_143,
    "scripts/recon-all-rh.log": 206_247,
    # Tier B: what the fsaverage maps are computed from
    "surf/lh.thickness": 593_343,
    "surf/rh.thickness": 587_803,
    "surf/lh.sphere.reg": 5_343_555,
    "surf/rh.sphere.reg": 5_293_695,
    "label/lh.aparc.annot": 1_188_169,
    "label/rh.aparc.annot": 1_177_089,
}
REAL_SUBJECT_FS7_URL = BUCKET_URL + "ds003020-fmriprep/sourcedata/freesurfer/sub-UTS01/"
REAL_SUBJECT_FS7_FILES = {  # sizes in bytes, from the bucket listing of 2026-11-12
    "mri/T1.mgz": 2_597_344,
    "mri/brainmask.mgz": 1_600_363,
    "surf/lh.white": 4_731_523,
    "surf/lh.pial.T1": 4_731_523,  # the subject has no lh.pial and no rh.pial key
    "surf/rh.white": 4_757_443,
    "surf/rh.pial.T1": 4_757_443,
    "stats/lh.aparc.stats": 5_909,
    "stats/rh.aparc.stats": 5_909,
    "stats/aseg.stats": 9_618,
    "scripts/build-stamp.txt": 56,
    "scripts/recon-all.log": 370_305,
    "scripts/recon-all-lh.log": 275_132,
    "scripts/recon-all-rh.log": 312_262,
    "surf/lh.thickness": 525_623,
    "surf/rh.thickness": 528_503,
}


def _fetched(request: pytest.FixtureRequest, name: str, url: str, files: dict[str, int]) -> Path:
    """The files of one subject in the pytest cache. Anonymous, read-only GETs; files
    already present with the expected size are not fetched again."""
    import httpx

    root = request.config.cache.mkdir(name)
    with httpx.Client(timeout=120.0) as client:
        for relative, size in files.items():
            path = root / relative
            if path.is_file() and path.stat().st_size == size:
                continue
            content = client.get(url + relative).raise_for_status().content
            assert len(content) == size, f"{relative}: {len(content)} bytes, expected {size}"
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            part.write_bytes(content)
            part.replace(path)
    return root


@pytest.fixture(scope="session")
def cached_subjects(request: pytest.FixtureRequest) -> Path:
    """Where the network fixtures keep the real subjects, fetched or not: default tests that
    read them skip when they are absent, and fetch nothing."""
    try:
        cache_dir = str(request.config.getini("cache_dir"))
    except ValueError:  # pytest was started without its cache plugin
        cache_dir = ".pytest_cache"
    return request.config.rootpath / cache_dir / "d"


@pytest.fixture(scope="session")
def real_subject(request: pytest.FixtureRequest) -> Path:
    """ds000102 sub-01 (FreeSurfer 6.0.1, fMRIPrep 21.0.2): Tier A, B and C, 43.4 MB, in the
    pytest cache. Network tests only."""
    return _fetched(request, "ds000102-sub-01", REAL_SUBJECT_URL, REAL_SUBJECT_FILES)


@pytest.fixture(scope="session")
def real_subject_fs7(request: pytest.FixtureRequest) -> Path:
    """ds003020 sub-UTS01 (FreeSurfer 7.3.2): Tier A, Tier C and the thickness files, 25.2 MB,
    in the pytest cache. Its pial surfaces exist as ``?h.pial.T1`` only. Network tests only."""
    return _fetched(request, "ds003020-sub-UTS01", REAL_SUBJECT_FS7_URL, REAL_SUBJECT_FS7_FILES)
