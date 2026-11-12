"""The renderer on one real subject: ds000102 sub-01 from OpenNeuroDerivatives (FreeSurfer 6.0.1).

Run with ``uv run pytest -m network tests/test_network_render.py -q -s`` to see the numbers.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]

from surfcensus.render import three_plane
from surfcensus.render.encode import read_image
from surfcensus.render.model import PANEL_ORDER

pytestmark = pytest.mark.network


def test_real_subject_renders(real_subject: Path, tmp_path: Path) -> None:
    source = snapshot_tree(real_subject)
    out = tmp_path / "media"
    start = time.perf_counter()
    info = three_plane(real_subject, out)
    seconds = time.perf_counter() - start
    assert snapshot_tree(real_subject) == source

    assert [len(info.panels[name].slices) for name in PANEL_ORDER] == [61, 54, 61]
    assert info.volume.shape == (256, 256, 256) and info.volume.voxel_size_mm == (1.0, 1.0, 1.0)
    assert info.poles["lh"].tkr == pytest.approx((-17.56, -83.64, -15.74), abs=0.01)
    assert info.poles["rh"].tkr == pytest.approx((9.15, -85.33, -15.48), abs=0.01)
    assert info.box.centre == pytest.approx((-4.34, -84.485, -15.61), abs=0.01)
    assert {role: used.name for role, used in info.inputs.items()} == {
        "t1": "T1.mgz", "brainmask": "brainmask.mgz", "lh_white": "lh.white",
        "rh_white": "rh.white", "lh_pial": "lh.pial", "rh_pial": "rh.pial",
    }  # fmt: skip

    gray, contours, drawn = [], [], 0
    for name in PANEL_ORDER:
        for pos in range(len(info.panels[name].slices)):
            gray.append((out / info.tiles.gray_path(name, pos)).stat().st_size)
            path = out / info.tiles.contours_path(name, pos)
            contours.append(path.stat().st_size)
            drawn += bool(read_image(path)[..., 3].any())
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    total = sum(gray) + sum(contours) + sum(sizes.values())
    assert info.vertex_map is not None
    print(
        f"\nrender time {seconds:.1f} s; {drawn} of {len(gray)} tiles have contours"
        f"\ngray tiles: mean {sum(gray) / len(gray) / 1e3:.1f} kB, sum {sum(gray) / 1e6:.2f} MB"
        f"\ncontour tiles: mean {sum(contours) / len(contours) / 1e3:.1f} kB, "
        f"sum {sum(contours) / 1e6:.2f} MB"
        f"\nanim.mp4 {sizes['anim.mp4'] / 1e6:.2f} MB, "
        f"poster.webp {sizes['poster.webp'] / 1e3:.1f} kB, "
        f"vertex_map.npz {sizes['vertex_map.npz'] / 1e3:.1f} kB "
        f"({info.vertex_map.n_lh} + {info.vertex_map.n_rh} vertices), "
        f"render.json {sizes['render.json'] / 1e3:.1f} kB"
        f"\ntotal {total / 1e6:.2f} MB in {len(gray) + len(contours) + len(sizes)} files"
    )
    # slices behind the poles and below the occipital lobe cross no surface
    assert drawn > 0.8 * len(gray)
    assert 1e6 < total < 15e6
    assert seconds < 120
