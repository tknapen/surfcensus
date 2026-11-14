// The logic of the player that needs no page: slices of a volume, the difference colours, the
// layout, the playback clock, which chunks to hold, and the contour file. It is imported by the
// page, by the worker and by logic_test.mjs (run under Node).

export const PLANES = ['sagittal', 'coronal', 'axial'];

// Per plane: the slice axis, the voxel axis along the panel's horizontal and vertical, whether
// the horizontal runs against the voxel axis, and the letters at the left, right, top and bottom
// edge. The display orientation is that of the surface renders (docs/rendering.md): sagittal
// with posterior at the left, coronal and axial radiological, superior or anterior up. The
// vertical always runs against the voxel axis. The grid is in RAS order (pack_run.py checks).
export const SPEC = [
  { name: 'sagittal', axis: 0, h: 1, v: 2, flipH: false, letters: 'PASI' },
  { name: 'coronal', axis: 1, h: 0, v: 2, flipH: true, letters: 'RLSI' },
  { name: 'axial', axis: 2, h: 0, v: 1, flipH: true, letters: 'RLAP' },
];

export const STREAMS = ['raw', 'pre'];
export const VIEWS = ['raw', 'pre', 'side', 'diff'];
export const KINDS = ['series', 'mean', 'sd'];
export const RATES = [1, 2, 5, 12, 24, 30, 60];
export const DIFF_SCALES = [8, 16, 32, 64, 128];
export const HOLD_MS = 500; // a marker stays on the frame at least this long

export function streamsOf(view) {
  return view === 'raw' ? ['raw'] : view === 'pre' ? ['pre'] : ['raw', 'pre'];
}

export function groupsOf(view) {
  return view === 'side' ? ['raw', 'pre'] : [view];
}

// ---- slices ------------------------------------------------------------------------------

export function planeSize(shape, plane) {
  const s = SPEC[plane];
  return [shape[s.h], shape[s.v]];
}

// The slice `index` of plane `plane` of a volume (voxel (i, j, k) at i + nx * (j + ny * k)) as
// the rows of the panel, top row first: out[r * w + c].
export function extractSlice(vol, shape, plane, index, out) {
  const s = SPEC[plane];
  const stride = [1, shape[0], shape[0] * shape[1]];
  const w = shape[s.h];
  const h = shape[s.v];
  const sh = stride[s.h];
  const sv = stride[s.v];
  const base = index * stride[s.axis];
  let o = 0;
  for (let r = 0; r < h; r++) {
    const row = base + (h - 1 - r) * sv;
    if (s.flipH) {
      for (let c = w - 1; c >= 0; c--) out[o++] = vol[row + c * sh];
    } else {
      for (let c = 0; c < w; c++) out[o++] = vol[row + c * sh];
    }
  }
  return out;
}

export function grayToRgba(slice, rgba) {
  for (let i = 0, o = 0; i < slice.length; i++, o += 4) {
    const g = slice[i];
    rgba[o] = g;
    rgba[o + 1] = g;
    rgba[o + 2] = g;
    rgba[o + 3] = 255;
  }
}

// The colours of a signed difference d = raw - pre in gray levels, -255 to 255, at the scale S
// (gray levels at which the colour saturates). Zero is a dark gray; raw above pre runs through
// red to light yellow, raw below pre through blue to light cyan. With a = min(|d|, S), the
// colour is interpolated in whole numbers between three stops at a = 0, S / 2 and S:
// c = c0 + floor(((c1 - c0) * num * 2 + den) / (2 * den)), num / den the place between two stops.
export const DIFF_STOPS = {
  zero: [24, 24, 24],
  pos: [
    [214, 48, 39],
    [255, 224, 144],
  ],
  neg: [
    [33, 102, 172],
    [166, 226, 255],
  ],
};

export function diffColour(d, scale) {
  const a = Math.min(Math.abs(d), scale);
  if (a === 0) return DIFF_STOPS.zero.slice();
  const [mid, end] = d > 0 ? DIFF_STOPS.pos : DIFF_STOPS.neg;
  let c0;
  let c1;
  let num;
  if (2 * a <= scale) {
    c0 = DIFF_STOPS.zero;
    c1 = mid;
    num = 2 * a;
  } else {
    c0 = mid;
    c1 = end;
    num = 2 * a - scale;
  }
  return [0, 1, 2].map((k) => c0[k] + Math.floor(((c1[k] - c0[k]) * num * 2 + scale) / (2 * scale)));
}

// lut[3 * (d + 255) + channel]
export function diffLut(scale) {
  const lut = new Uint8Array(511 * 3);
  for (let d = -255; d <= 255; d++) lut.set(diffColour(d, scale), 3 * (d + 255));
  return lut;
}

export function diffToRgba(a, b, lut, rgba) {
  for (let i = 0, o = 0; i < a.length; i++, o += 4) {
    const k = 3 * (a[i] - b[i] + 255);
    rgba[o] = lut[k];
    rgba[o + 1] = lut[k + 1];
    rgba[o + 2] = lut[k + 2];
    rgba[o + 3] = 255;
  }
}

// ---- chunks --------------------------------------------------------------------------------

// A chunk as it arrives holds residuals; adding the reference volume (mod 256) gives the values.
export function addReference(buf, ref) {
  const n = ref.length;
  for (let o = 0; o < buf.length; o += n) {
    for (let i = 0; i < n; i++) buf[o + i] = (buf[o + i] + ref[i]) & 255;
  }
  return buf;
}

export function chunkOf(t, chunkVolumes) {
  return Math.floor(t / chunkVolumes);
}

// The order in which the chunks of one run are wanted around the chunk of the cursor: the
// cursor's own, then two ahead for every one behind, wrapping round the ends of the run (the
// player loops). All chunks of the run, each once.
export function chunkOrder(nChunks, cursor) {
  const out = [cursor];
  const seen = new Set(out);
  let ahead = 1;
  let behind = 1;
  const push = (c) => {
    const k = ((c % nChunks) + nChunks) % nChunks;
    if (!seen.has(k)) {
      seen.add(k);
      out.push(k);
    }
  };
  while (out.length < nChunks) {
    push(cursor + ahead++);
    push(cursor + ahead++);
    push(cursor - behind++);
  }
  return out;
}

// Which chunks to hold decoded, most wanted first, within a memory limit.
//   runs: [{ nChunks, chunkBytes }] (chunkBytes: decoded size of a full chunk of one stream)
//   run, cursor: the run on screen and the chunk of its current volume
//   streams: the streams on screen; limitBytes: the limit for decoded chunks
// First the streams on screen of the current run in chunkOrder, then its other stream, then the
// other runs from the next one on. The list stops before the chunk that would exceed the limit,
// but always holds the cursor's chunk and the next of every stream on screen, whatever the limit.
// Returns keys 'run|stream|chunk'.
export function desiredChunks({ runs, run, cursor, streams, limitBytes }) {
  const out = [];
  let bytes = 0;
  const cur = runs[run];
  const order = chunkOrder(cur.nChunks, cursor);
  let full = false;
  order.forEach((c, n) => {
    for (const s of streams) {
      if (full) return;
      if (n >= 2 && bytes + cur.chunkBytes > limitBytes) {
        full = true;
        return;
      }
      out.push(`${run}|${s}|${c}`);
      bytes += cur.chunkBytes;
    }
  });
  const rest = [];
  for (const s of STREAMS) if (!streams.includes(s)) for (const c of order) rest.push([run, s, c]);
  for (let k = 1; k < runs.length; k++) {
    const r = (run + k) % runs.length;
    for (let c = 0; c < runs[r].nChunks; c++) for (const s of STREAMS) rest.push([r, s, c]);
  }
  for (const [r, s, c] of rest) {
    if (full || bytes + runs[r].chunkBytes > limitBytes) break;
    out.push(`${r}|${s}|${c}`);
    bytes += runs[r].chunkBytes;
  }
  return out;
}

// ---- playback ------------------------------------------------------------------------------

// The playback clock. A volume is due at nextDue (ms). At an animation frame at `now`, the next
// volume is shown when it is due within slackMs, and the one after is due 1000 / rate later, so
// every volume is painted and none is skipped: when frames come late, playing takes longer. If
// the clock has fallen more than resyncMs behind (a stall, a hidden tab), it starts anew.
export function playbackStep(nextDue, now, rate, slackMs = 4, resyncMs = 250) {
  if (now + slackMs < nextDue) return { advance: false, nextDue, resynced: false };
  let due = nextDue + 1000 / rate;
  let resynced = false;
  if (now - due > resyncMs) {
    due = now + 1000 / rate;
    resynced = true;
  }
  return { advance: true, nextDue: due, resynced };
}

export function wrap(t, n) {
  return ((t % n) + n) % n;
}

export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

export function stepRate(rate, direction) {
  let at = RATES.indexOf(rate);
  if (at < 0) at = RATES.findIndex((r) => r >= rate);
  return RATES[clamp(at + direction, 0, RATES.length - 1)];
}

// Volumes whose framewise displacement exceeds the limit (a missing value never does).
export function aboveLimit(fd, limit) {
  const out = [];
  fd.forEach((v, t) => {
    if (v !== null && v > limit) out.push(t);
  });
  return out;
}

// ---- layout --------------------------------------------------------------------------------

// Sizes in mm of the three panels of a grid: [[w, h], ...].
export function panelSizesMm(shape, voxel) {
  return SPEC.map((s) => [shape[s.h] * voxel[s.h], shape[s.v] * voxel[s.v]]);
}

// How to lay out nGroups groups of three panels in a box of availW x availH px: the panels of a
// group in a row or a column, the groups beside or under each other, whichever gives the largest
// scale (px per mm). gap is the space between panels and between groups, head the height of a
// group's heading, frame the width of the frame round a group (both sides together).
export function fitLayout(sizes, nGroups, availW, availH, gap = 8, head = 22, frame = 4) {
  let best = null;
  for (const panels of ['row', 'column']) {
    const gw = panels === 'row' ? sizes.reduce((a, s) => a + s[0], 0) : Math.max(...sizes.map((s) => s[0]));
    const gh = panels === 'row' ? Math.max(...sizes.map((s) => s[1])) : sizes.reduce((a, s) => a + s[1], 0);
    const gapsW = (panels === 'row' ? 2 * gap : 0) + frame;
    const gapsH = (panels === 'row' ? 0 : 2 * gap) + head + frame;
    for (const groups of nGroups > 1 ? ['row', 'column'] : ['row']) {
      const nw = groups === 'row' ? nGroups : 1;
      const nh = groups === 'row' ? 1 : nGroups;
      const w = availW - nw * gapsW - (nw - 1) * 2 * gap;
      const h = availH - nh * gapsH - (nh - 1) * gap;
      const scale = Math.min(w / (nw * gw), h / (nh * gh));
      if (!best || scale > best.scale) best = { scale, panels, groups };
    }
  }
  best.scale = clamp(best.scale, 0.1, 8);
  return best;
}

// A point of a slice in voxel coordinates (a along the plane's horizontal voxel axis, b along
// its vertical one; a whole number is a voxel centre) as panel coordinates in voxels from the
// top left corner of the panel.
export function voxelToPanel(plane, shape, a, b) {
  const s = SPEC[plane];
  return [s.flipH ? shape[s.h] - 0.5 - a : a + 0.5, shape[s.v] - 0.5 - b];
}

// The voxel under a panel position (in voxels from the top left corner): [index along h, along v].
export function panelToVoxel(plane, shape, x, y) {
  const s = SPEC[plane];
  const c = clamp(Math.floor(x), 0, shape[s.h] - 1);
  const r = clamp(Math.floor(y), 0, shape[s.v] - 1);
  return [s.flipH ? shape[s.h] - 1 - c : c, shape[s.v] - 1 - r];
}

// ---- contours ------------------------------------------------------------------------------

// The contour file of one plane (pack_contours.py), inflated: lines of points in voxel
// coordinates. cell = slice * nSurfaces + surface; lines lineStart[cell] .. lineStart[cell + 1];
// points of line n: xy[2 * pointStart[n]] .. xy[2 * pointStart[n + 1]].
export function decodeContours(buffer, unit) {
  const head = new Uint32Array(buffer, 0, 4);
  const [nSlices, nSurfaces, nLines, nPoints] = head;
  let at = 16;
  const lineStart = new Uint32Array(buffer, at, nSlices * nSurfaces + 1);
  at += 4 * (nSlices * nSurfaces + 1);
  const pointStart = new Uint32Array(buffer, at, nLines + 1);
  at += 4 * (nLines + 1);
  const delta = new Int16Array(buffer.slice(at, at + 4 * nPoints));
  const xy = new Float32Array(2 * nPoints);
  for (let n = 0; n < nLines; n++) {
    let x = 0;
    let y = 0;
    for (let p = pointStart[n]; p < pointStart[n + 1]; p++) {
      x += delta[2 * p];
      y += delta[2 * p + 1];
      xy[2 * p] = x / unit;
      xy[2 * p + 1] = y / unit;
    }
  }
  return { nSlices, nSurfaces, nLines, nPoints, lineStart, pointStart, xy };
}

export function formatBytes(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(1)} MB`;
}
