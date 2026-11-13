// Display arithmetic on the `render` object of an assignment: click to tile pixel, tile
// pixel to voxel and back, voxel to tkr RAS, crosshairs, tile names. The same arithmetic as
// surfcensus/render/geometry.py. Display only: the server computes what is stored.
//
// No DOM here, so the functions run under Node (tests/test_frontend.py).

export const PANEL_ORDER = ["sagittal", "coronal", "axial"];

export function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

// Offset of a click inside a tile shown at `shownWidth` x `shownHeight` -> continuous tile
// pixel coordinates (origin top-left, px right, py down): px = ox * width / w.
export function clickToPixel(panel, ox, oy, shownWidth, shownHeight) {
  return [
    clamp((ox * panel.width) / shownWidth, 0, panel.width),
    clamp((oy * panel.height) / shownHeight, 0, panel.height),
  ];
}

// Continuous voxel coordinates (i, j, k) of a point on one tile of a panel.
export function pixelToVoxel(panel, slicePos, px, py) {
  const m = panel.pix2vox;
  const ijk = [0, 0, 0];
  ijk[panel.slice_voxel_axis] = panel.slices[slicePos];
  ijk[panel.h_voxel_axis] = m[0][0] * px + m[0][2];
  ijk[panel.v_voxel_axis] = m[1][1] * py + m[1][2];
  return ijk;
}

// [px, py, voxel coordinate along the slice axis] of a voxel in a panel.
export function voxelToPixel(panel, ijk) {
  const m = panel.vox2pix;
  return [
    m[0][0] * ijk[panel.h_voxel_axis] + m[0][2],
    m[1][1] * ijk[panel.v_voxel_axis] + m[1][2],
    ijk[panel.slice_voxel_axis],
  ];
}

// Position in panel.slices of the tile nearest a voxel coordinate along the slice axis; the
// first of two that are equally near, as numpy's argmin gives it.
export function nearestSlice(panel, sliceCoordinate) {
  let best = 0;
  let bestDistance = Infinity;
  panel.slices.forEach((voxel, position) => {
    const distance = Math.abs(voxel - sliceCoordinate);
    if (distance < bestDistance) {
      best = position;
      bestDistance = distance;
    }
  });
  return best;
}

// A 4 x 4 affine (rows) applied to a point.
export function applyAffine(affine, xyz) {
  return [0, 1, 2].map(
    (row) =>
      affine[row][0] * xyz[0] + affine[row][1] * xyz[1] + affine[row][2] * xyz[2] + affine[row][3],
  );
}

// The inverse of a 4 x 4 affine, through the adjugate of its linear part.
export function invertAffine(affine) {
  const [a, b, c] = [affine[0][0], affine[0][1], affine[0][2]];
  const [d, e, f] = [affine[1][0], affine[1][1], affine[1][2]];
  const [g, h, i] = [affine[2][0], affine[2][1], affine[2][2]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || det === 0) throw new Error("the affine cannot be inverted");
  const linear = [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
  const shift = [affine[0][3], affine[1][3], affine[2][3]];
  const out = linear.map((row) => [
    ...row,
    -(row[0] * shift[0] + row[1] * shift[1] + row[2] * shift[2]),
  ]);
  out.push([0, 0, 0, 1]);
  return out;
}

export function voxelToTkr(render, ijk) {
  return applyAffine(render.volume.vox2ras_tkr, ijk);
}

export function tkrToVoxel(render, xyz) {
  return applyAffine(invertAffine(render.volume.vox2ras_tkr), xyz);
}

// Voxel coordinate of the current slice along each voxel axis, for a frame
// [sagittal, coronal, axial] of a frame table.
export function currentVoxel(render, frame) {
  const current = [0, 0, 0];
  PANEL_ORDER.forEach((name, position) => {
    const panel = render.panels[name];
    current[panel.slice_voxel_axis] = panel.slices[frame[position]];
  });
  return current;
}

// Where the other two planes cross each panel in a frame: name -> [px of the vertical line,
// py of the horizontal line] (geometry.crosshair_pixels).
export function crosshairPixels(render, frame) {
  const current = currentVoxel(render, frame);
  const out = {};
  for (const name of PANEL_ORDER) {
    const [px, py] = voxelToPixel(render.panels[name], current);
    out[name] = [px, py];
  }
  return out;
}

// The pixel column or row that holds a crosshair coordinate, as the composite strip of the
// renderer draws it (frames.composite_strip): one on a pixel boundary must not hang on the
// last bit.
export function crosshairCell(coordinate) {
  return Math.floor(coordinate + 1e-6);
}

// A tile's name from a pattern of `render.tiles` ("tiles/{panel}_{index:03d}.webp"): the
// number padded with zeros to three digits (app.schemas.tile_name).
export function tileName(pattern, panelName, index) {
  return pattern
    .replace("{panel}", panelName)
    .replace("{index:03d}", String(index).padStart(3, "0"));
}

export function tileUrl(mediaBase, pattern, panelName, index, renderKey) {
  return `${mediaBase}${tileName(pattern, panelName, index)}?v=${encodeURIComponent(renderKey)}`;
}

// Distance in mm, along a panel's slice axis, between a tkr point and one of its slices.
export function sliceDistanceMm(panel, slicePos, tkr) {
  return Math.abs(tkr[panel.world_axis] - panel.slice_mm[slicePos]);
}
