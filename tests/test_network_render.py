"""The renderer on one real subject: ds000102 sub-01 from OpenNeuroDerivatives (FreeSurfer 6.0.1).

Run with ``uv run pytest -m network tests/test_network_render.py -q -s`` to see the numbers.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from synth import snapshot_tree  # pyrefly: ignore[missing-import]

from surfcensus.fsio.surfaces import read_surface
from surfcensus.fsio.volumes import read_volume
from surfcensus.render import frames, three_plane
from surfcensus.render.encode import read_image
from surfcensus.render.model import PANEL_ORDER
from surfcensus.render.slicing import MeshSlicer

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
    files = {
        "t1": "mri/T1.mgz", "brainmask": "mri/brainmask.mgz", "lh_white": "surf/lh.white",
        "rh_white": "surf/rh.white", "lh_pial": "surf/lh.pial", "rh_pial": "surf/rh.pial",
    }  # fmt: skip
    assert set(info.inputs) == set(files)
    for role, relative in files.items():  # files of 1.7 to 5.6 MB, hashed whole
        content = (real_subject / relative).read_bytes()
        used = info.inputs[role]
        assert used.name == Path(relative).name
        assert (used.size, used.sha256) == (len(content), hashlib.sha256(content).hexdigest())

    t1 = read_volume(real_subject / "mri" / "T1.mgz")
    slicers = {}
    for role in ("lh_white", "rh_white", "lh_pial", "rh_pial"):
        surface = read_surface(real_subject / files[role])
        slicers[role] = MeshSlicer(surface.vertices, surface.faces)
    size, line_width = info.config.panel_px, info.config.line_width_px
    gray, contours, drawn, covered, expected = [], [], 0, 0.0, 0.0
    for name in PANEL_ORDER:
        panel = info.panels[name]
        for pos in range(len(panel.slices)):
            gray.append((out / info.tiles.gray_path(name, pos)).stat().st_size)
            path = out / info.tiles.contours_path(name, pos)
            contours.append(path.stat().st_size)
            alpha = read_image(path)[..., 3]
            drawn += bool(alpha.any())
            # Every line of every surface is drawn: the lines cover nearly the length of the
            # contours on the tile times the line width. On 152 of the 155 tiles with contours
            # a surface gives more than one line, up to 15. With only the first closed loop of
            # each surface drawn, the worst tile comes to 0.37 and all tiles together to 0.88.
            on_tile = 0.0
            for segments, _ in frames.contour_layers(
                panel, pos, t1.vox2ras_tkr, slicers, info.config
            ):
                ends = np.clip(segments, 0, size)
                on_tile += float(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).sum())
            covered += alpha.sum() / 255
            expected += on_tile * line_width
            if on_tile * line_width > 200:  # measured: 0.952 to 1.005
                assert 0.93 < alpha.sum() / 255 / (on_tile * line_width) < 1.02, (name, pos)
    assert 0.96 < covered / expected < 0.99  # measured: 0.976
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
        f"({info.vertex_map.n_lh} + {info.vertex_map.n_rh} vertices, "
        f"{info.vertex_map.n_lh_visible} + {info.vertex_map.n_rh_visible} visible), "
        f"render.json {sizes['render.json'] / 1e3:.1f} kB"
        f"\ntotal {total / 1e6:.2f} MB in {len(gray) + len(contours) + len(sizes)} files"
    )
    # the numbers of docs/rendering.md: the map follows the tiles (the box plus 10 mm held 56,031)
    assert (info.vertex_map.n_lh, info.vertex_map.n_rh) == (37356, 37298)
    assert (info.vertex_map.n_lh_visible, info.vertex_map.n_rh_visible) == (27543, 27991)
    # slices behind the poles and below the occipital lobe cross no surface
    assert drawn > 0.8 * len(gray)
    assert 1e6 < total < 15e6
    assert seconds < 120
