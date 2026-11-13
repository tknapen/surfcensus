"""Three-plane panels of a volume on an axis-aligned RAS grid, with surface contours.

Display conventions are those of surfcensus (docs/rendering.md): sagittal with posterior at the
left and superior up; coronal and axial radiological (subject's right at the left of the panel),
superior up and anterior up. Voxels are shown by nearest neighbour, so a voxel is a block.
"""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw, ImageFont

from surfcensus.render.frames import ContourRenderer, blend
from surfcensus.render.slicing import MeshSlicer

PIAL, WHITE = "#FFD400", "#FF3B30"
NAMES = ("sagittal", "coronal", "axial")
# per panel: slice axis, horizontal axis, vertical axis, horizontal flipped,
# edge letters (left, right, top, bottom)
_SPEC = {
    "sagittal": (0, 1, 2, False, "PASI"),
    "coronal": (1, 0, 2, True, "RLSI"),
    "axial": (2, 0, 1, True, "RLAP"),
}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # an older Pillow has the bitmap font only
        return ImageFont.load_default()


@dataclass
class Panel:
    name: str
    axis: int
    h: int
    v: int
    flip_h: bool
    letters: str
    width: int
    height: int
    index: int  # voxel index of the slice
    position: float  # world coordinate of the slice plane (mm)
    cols: np.ndarray  # voxel index along h for every pixel column
    rows: np.ndarray  # voxel index along v for every pixel row
    lo_h: float
    hi_h: float
    hi_v: float
    scale: float

    def cut(self, vol):
        """The slice as (height, width), nearest neighbour."""
        # axes in ascending order without the slice axis
        sl = np.take(vol, self.index, axis=self.axis)
        if self.h > self.v:
            sl = sl.T
        return sl[np.ix_(self.cols, self.rows)].T  # sl is [h, v]

    def to_pixel(self, xyz):
        px = (self.hi_h - xyz[..., self.h]) if self.flip_h else (xyz[..., self.h] - self.lo_h)
        py = self.hi_v - xyz[..., self.v]
        return np.stack([px * self.scale, py * self.scale], axis=-1)


class Panels:
    def __init__(self, shape, affine, scale, slices):
        """slices: voxel indices (i, j, k) of the sagittal, coronal and axial slice."""
        a = np.asarray(affine, float)
        diagonal = np.allclose(a[:3, :3], np.diag(np.diag(a[:3, :3])))
        if not diagonal or (np.diag(a[:3, :3]) <= 0).any():
            raise ValueError("the prototype needs an axis-aligned RAS grid")
        self.shape, self.affine, self.scale = tuple(shape), a, float(scale)
        size = np.diag(a[:3, :3])
        lo = a[:3, 3] - size / 2
        hi = lo + size * np.array(shape[:3])
        self.panels = []
        for name, idx in zip(NAMES, slices, strict=True):
            axis, h, v, flip, letters = _SPEC[name]
            w = round((hi[h] - lo[h]) * scale / 2) * 2
            ht = round((hi[v] - lo[v]) * scale / 2) * 2
            ch = (np.arange(w) + 0.5) / scale
            ch = (hi[h] - ch) if flip else (lo[h] + ch)
            cv = hi[v] - (np.arange(ht) + 0.5) / scale
            cols = np.clip(np.floor((ch - lo[h]) / size[h]).astype(int), 0, shape[h] - 1)
            rows = np.clip(np.floor((cv - lo[v]) / size[v]).astype(int), 0, shape[v] - 1)
            self.panels.append(
                Panel(
                    name,
                    axis,
                    h,
                    v,
                    flip,
                    letters,
                    w,
                    ht,
                    int(idx),
                    float(a[axis, 3] + size[axis] * idx),
                    cols,
                    rows,
                    float(lo[h]),
                    float(hi[h]),
                    float(hi[v]),
                    float(scale),
                )
            )
        self.height = max(p.height for p in self.panels)
        self.width = sum(p.width for p in self.panels)

    def contours(self, surfaces, line_px=1.0):
        """One RGBA overlay (height, width, 4) for the whole strip; static, so drawn once."""
        out = np.zeros((self.height, self.width, 4), np.uint8)
        slicers = {k: MeshSlicer(v, f) for k, (v, f) in surfaces.items()}
        x0 = 0
        for p in self.panels:
            layers = []
            for kind, colour in (("white", WHITE), ("pial", PIAL)):
                segs = [
                    p.to_pixel(slicers[h, kind].segments(p.axis, p.position)) for h in ("lh", "rh")
                ]
                segs = [s for s in segs if len(s)]
                if segs:
                    layers.append((np.concatenate(segs), colour))
            with ContourRenderer(p.width, p.height, line_px) as r:
                y0 = (self.height - p.height) // 2
                out[y0 : y0 + p.height, x0 : x0 + p.width] = r.render(layers)
            x0 += p.width
        return out

    def strip(self, vol, lo, hi):
        """Grayscale strip (height, width) uint8 of the three panels, windowed lo..hi."""
        out = np.zeros((self.height, self.width), np.uint8)
        x0 = 0
        k = 255.0 / (hi - lo)
        for p in self.panels:
            g = np.clip((p.cut(vol).astype(np.float32) - lo) * k + 0.5, 0, 255).astype(np.uint8)
            y0 = (self.height - p.height) // 2
            out[y0 : y0 + p.height, x0 : x0 + p.width] = g
            x0 += p.width
        return out

    def letters(self, image):
        """Edge letters of every panel, drawn on an RGB PIL image of the strip."""
        d = ImageDraw.Draw(image)
        f = font(13)
        x0 = 0
        for p in self.panels:
            y0 = (self.height - p.height) // 2
            left, right, top, bottom = p.letters
            for txt, (x, y) in (
                (left, (x0 + 4, y0 + p.height // 2 - 7)),
                (right, (x0 + p.width - 13, y0 + p.height // 2 - 7)),
                (top, (x0 + p.width // 2 - 4, y0 + 2)),
                (bottom, (x0 + p.width // 2 - 4, y0 + p.height - 17)),
            ):
                d.text(
                    (x, y), txt, font=f, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0)
                )
            x0 += p.width
        return image


def compose(gray, overlay=None):
    return np.repeat(gray[..., None], 3, axis=2) if overlay is None else blend(gray, overlay)
