// The logic of the EPI page that needs no page: slices of a volume, the timeline of the
// subject's one stream, the slices of another grid at the same place in the anatomy, the
// common intensity scale, the difference colours, the layout, the playback clock, which
// chunks to hold, and the contour file. Imported by the page, by the worker, and by the tests
// under Node (tests/test_epi_frontend.py). Ported from stage 1's player (logic.js).

export const PLANES = ['sagittal', 'coronal', 'axial'];

// Per plane: the slice axis, the voxel axis along the panel's horizontal and vertical, and
// whether the horizontal runs against the voxel axis. The vertical always runs against the
// voxel axis. The letters at the edges come from the grid (GridOut.edge_letters), which
// func.geometry.edge_letters computes for this same drawing rule.
export const SPEC = [
  { name: 'sagittal', axis: 0, h: 1, v: 2, flipH: false },
  { name: 'coronal', axis: 1, h: 0, v: 2, flipH: true },
  { name: 'axial', axis: 2, h: 0, v: 1, flipH: true },
];

export const STREAMS = ['raw', 'pre'];
export const STILLS = ['ref', 'mean', 'sd'];
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

export function wrap(t, n) {
  return ((t % n) + n) % n;
}

export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

// ---- the stream --------------------------------------------------------------------------

// The timeline of a subject (decision D38): every packed run of every session, sessions in
// the order given and runs in the order of their session's runs[], as one stream of volumes.
//   sessions: EpiStreamOut.sessions
// Returns { total, segments } with one segment per packed run: { session, run (indices into
// sessions and their runs), first (index of its first volume in the stream), n, gridKey,
// chunks (the chunk list of its pre stream: first, count within the run), startsSession }.
// A run without a pack (not packed, or not whole on disk) is not on the timeline.
export function buildTimeline(sessions) {
  const segments = [];
  let at = 0;
  sessions.forEach((session, si) => {
    let firstOfSession = true;
    session.runs.forEach((run, ri) => {
      if (!run.pack || !run.n_volumes) return;
      const pre = run.pack.streams.pre.chunks;
      const raw = run.pack.streams.raw.chunks;
      if (pre.length !== raw.length || pre.some((c, k) => c.first !== raw[k].first || c.count !== raw[k].count)) {
        throw new Error(`the streams of run ${run.run_uid} are not chunked alike`);
      }
      segments.push({
        session: si,
        run: ri,
        first: at,
        n: run.n_volumes,
        gridKey: run.grid_key,
        chunks: pre.map((c) => ({ first: c.first, count: c.count })),
        startsSession: firstOfSession,
      });
      firstOfSession = false;
      at += run.n_volumes;
    });
  });
  return { total: at, segments };
}

// The segment that holds volume g of the stream (0 <= g < total).
export function locate(timeline, g) {
  const s = timeline.segments;
  let lo = 0;
  let hi = s.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (s[mid].first <= g) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// The boundaries of the stream: [{ at (the first volume after it), session (true at a session
// boundary) }]. The start of the stream is not a boundary.
export function boundaries(timeline) {
  return timeline.segments.slice(1).map((seg) => ({ at: seg.first, session: seg.startsSession }));
}

// The traces of the stream, one value per volume, null at a missing value, and the volumes
// that are not steady-state, as indices of the stream.
export function streamTraces(timeline, sessions) {
  const fd = new Array(timeline.total).fill(null);
  const dvars = new Array(timeline.total).fill(null);
  const nonSteady = [];
  for (const seg of timeline.segments) {
    const tr = sessions[seg.session].runs[seg.run].pack.traces;
    for (let t = 0; t < seg.n; t++) {
      fd[seg.first + t] = tr.fd.length === seg.n ? tr.fd[t] : null;
      dvars[seg.first + t] = tr.dvars.length === seg.n ? tr.dvars[t] : null;
    }
    for (const t of tr.non_steady) if (t < seg.n) nonSteady.push(seg.first + t);
  }
  return { fd, dvars, nonSteady };
}

// ---- the same place in the anatomy across grids ---------------------------------------------

// The point x where the three slice planes meet: normal_p . x = distance_p (RULES.slices).
// normals: three unit vectors, distances: three numbers. Cramer's rule in doubles.
export function meetPoint(normals, distances) {
  const [a, b, c] = normals;
  const det = (u, v, w) => u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0]);
  const d = det(a, b, c);
  if (!(Math.abs(d) > 1e-12)) throw new Error('the slice planes of the grid do not meet in a point');
  const out = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    // replace column k of the matrix whose rows are the normals by the distances
    const rows = normals.map((n, r) => n.map((v, j) => (j === k ? distances[r] : v)));
    out[k] = det(rows[0], rows[1], rows[2]) / d;
  }
  return out;
}

// The place of slices [s0, s1, s2] of a grid (GridOut): the point where their planes meet.
export function pointOf(grid, slices) {
  return meetPoint(
    PLANES.map((p) => grid.slice_normal[p]),
    PLANES.map((p, n) => grid.slice_mm[p][slices[n]]),
  );
}

// The slices of a grid nearest to the point x: for each plane, the slice whose distance is
// nearest normal . x; of two equally near, the lower index.
export function slicesAt(grid, x) {
  return PLANES.map((p) => {
    const n = grid.slice_normal[p];
    const target = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
    const mm = grid.slice_mm[p];
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < mm.length; k++) {
      const d = Math.abs(mm[k] - target);
      if (d < bestD) {
        best = k;
        bestD = d;
      }
    }
    return best;
  });
}

// The place after a step of one plane to slice `index` of a grid (GridOut): the place moves
// along that plane's normal n only, to the plane's distance d of the slice, place + (d - n .
// place) n, so the place stays where it was along the other two planes. A step on a coarse
// grid therefore does not move the other two planes when the stream comes back to a fine one.
export function movePlace(grid, place, plane, index) {
  const n = grid.slice_normal[PLANES[plane]];
  const d = grid.slice_mm[PLANES[plane]][index];
  const along = d - (n[0] * place[0] + n[1] * place[1] + n[2] * place[2]);
  return [place[0] + along * n[0], place[1] + along * n[1], place[2] + along * n[2]];
}

// ---- intensity --------------------------------------------------------------------------------

// The table by which a run with window (0, hi) is shown under one window (0, top) for the
// whole stream (RULES.common_scale; func.packread.common_scale): b -> floor(b * (hi / top) +
// 0.5), clipped to 0..255. The operations are those of numpy's float64, in that order.
export function commonScale(hi, top) {
  if (!(top >= hi && hi > 0)) throw new Error('top is the largest upper limit of the stream');
  const ratio = hi / top;
  const out = new Uint8Array(256);
  for (let b = 0; b < 256; b++) out[b] = clamp(Math.floor(b * ratio + 0.5), 0, 255);
  return out;
}

export const IDENTITY = Uint8Array.from({ length: 256 }, (_, b) => b);

// The tables of the packed runs, by segment: with normalization on the stored values as they
// are; off, the common scale of the stream (the largest hi of its runs), for the series and
// mean (window) and for the SD stills (sd_window) apart.
//   windows: [[lo, hi], ...] and sdWindows likewise, one per segment
export function intensityTables(windows, sdWindows, normalized) {
  const top = Math.max(...windows.map((w) => w[1]));
  const topSd = Math.max(...sdWindows.map((w) => w[1]));
  return windows.map((w, k) => ({
    value: normalized ? IDENTITY : commonScale(w[1], top),
    sd: normalized ? IDENTITY : commonScale(sdWindows[k][1], topSd),
    top: normalized ? w[1] : top,
    topSd: normalized ? sdWindows[k][1] : topSd,
  }));
}

// What a stored value b of a run with window (lo, hi) stands for (RULES.value).
export function signalOf(b, window) {
  return window[0] + (b * (window[1] - window[0])) / 255;
}

// ---- slices -----------------------------------------------------------------------------------

export function planeSize(shape, plane) {
  const s = SPEC[plane];
  return [shape[s.h], shape[s.v]];
}

// The slice `index` of plane `plane` of a volume (voxel (i, j, k) at i + nx * (j + ny * k)) as
// the rows of the panel, top row first, each value through `table`: out[r * w + c].
export function extractSlice(vol, shape, plane, index, out, table = IDENTITY) {
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
      for (let c = w - 1; c >= 0; c--) out[o++] = table[vol[row + c * sh]];
    } else {
      for (let c = 0; c < w; c++) out[o++] = table[vol[row + c * sh]];
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
// (gray levels at which the colour saturates); stage 1's rule, unchanged.
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

// ---- chunks -----------------------------------------------------------------------------------

// A chunk as it arrives holds residuals; adding the reference volume (mod 256) gives the values.
export function addReference(buf, ref) {
  const n = ref.length;
  for (let o = 0; o < buf.length; o += n) {
    for (let i = 0; i < n; i++) buf[o + i] = (buf[o + i] + ref[i]) & 255;
  }
  return buf;
}

// Every chunk of the stream in stream order: { seg, chunk (its index within the run), first
// (of the stream), count }.
export function streamChunks(timeline) {
  const out = [];
  timeline.segments.forEach((seg, si) => {
    seg.chunks.forEach((c, k) => out.push({ seg: si, chunk: k, first: seg.first + c.first, count: c.count }));
  });
  return out;
}

// The index in streamChunks of the chunk that holds volume g.
export function chunkIndexOf(chunks, g) {
  let lo = 0;
  let hi = chunks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (chunks[mid].first <= g) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// The order in which the chunks of the stream are wanted around the chunk of the cursor: the
// cursor's own, then two ahead for every one behind, wrapping round the ends of the stream
// (the player loops over the whole stream). All chunks, each once. Ahead runs on into the
// next run and the next session, so the chunks after a boundary are asked for before it.
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

// Which chunks to hold decoded, most wanted first, within one memory limit for the stream.
//   sizes: decoded bytes of each chunk of streamChunks (one stream)
//   cursor: the index of the chunk of the volume on screen; streams: the streams on screen
// First the streams on screen in chunkOrder, then the other stream in the same order. The
// list stops before the chunk that would exceed the limit, but always holds the cursor's
// chunk and the next of every stream on screen, whatever the limit. Keys 'index|stream'.
export function desiredChunks({ sizes, cursor, streams, limitBytes }) {
  const order = chunkOrder(sizes.length, cursor);
  const out = [];
  let bytes = 0;
  let full = false;
  order.forEach((c, n) => {
    for (const s of streams) {
      if (full) return;
      if (n >= 2 && bytes + sizes[c] > limitBytes) {
        full = true;
        return;
      }
      out.push(`${c}|${s}`);
      bytes += sizes[c];
    }
  });
  for (const s of STREAMS) {
    if (streams.includes(s)) continue;
    for (const c of order) {
      if (full || bytes + sizes[c] > limitBytes) return out;
      out.push(`${c}|${s}`);
      bytes += sizes[c];
    }
  }
  return out;
}

// ---- playback ---------------------------------------------------------------------------------

// The playback clock of stage 1. A volume is due at nextDue (ms). At an animation frame at
// `now`, the next volume is shown when it is due within slackMs, and the one after is due
// 1000 / rate later, so every volume is painted and none is skipped: when frames come late,
// playing takes longer. If the clock has fallen more than resyncMs behind, it starts anew.
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

// ---- layout -----------------------------------------------------------------------------------

// Sizes in mm of the three panels of a grid: [[w, h], ...].
export function panelSizesMm(shape, voxel) {
  return SPEC.map((s) => [shape[s.h] * voxel[s.h], shape[s.v] * voxel[s.v]]);
}

// The largest size in mm of each panel over the grids of the stream: the boxes of the panels
// keep this size for the whole stream, so that the layout and the scale in px per mm do not
// change at a boundary and a grid with a smaller field of view sits in the middle of its box.
export function streamPanelSizesMm(grids) {
  const out = SPEC.map(() => [0, 0]);
  for (const g of grids) {
    panelSizesMm(g.shape, g.voxel_mm).forEach(([w, h], p) => {
      out[p][0] = Math.max(out[p][0], w);
      out[p][1] = Math.max(out[p][1], h);
    });
  }
  return out;
}

// How to lay out nGroups groups of three panels in a box of availW x availH px (stage 1).
// pad: px on every side of each panel outside its picture (the margin of the edge letters).
export function fitLayout(sizes, nGroups, availW, availH, gap = 8, head = 22, frame = 4, pad = 0) {
  let best = null;
  for (const panels of ['row', 'column']) {
    const gw = panels === 'row' ? sizes.reduce((a, s) => a + s[0], 0) : Math.max(...sizes.map((s) => s[0]));
    const gh = panels === 'row' ? Math.max(...sizes.map((s) => s[1])) : sizes.reduce((a, s) => a + s[1], 0);
    const gapsW = (panels === 'row' ? 2 * gap + 6 * pad : 2 * pad) + frame;
    const gapsH = (panels === 'row' ? 2 * pad : 2 * gap + 6 * pad) + head + frame;
    for (const groups of nGroups > 1 ? ['row', 'column'] : ['row']) {
      const nw = groups === 'row' ? nGroups : 1;
      const nh = groups === 'row' ? 1 : nGroups;
      const w = availW - nw * gapsW - (nw - 1) * 2 * gap;
      const h = availH - nh * gapsH - (nh - 1) * gap;
      const scale = Math.min(w / (nw * gw), h / (nh * gh));
      if (!best || scale > best.scale) best = { scale, panels, groups };
    }
  }
  best.scale = clamp(best.scale, 0.1, 16);
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

// Where volume boundary b (the first volume after it) lies along a track of the stream, as a
// fraction of the travel of a slider from 0 to total - 1: half-way between volumes b - 1 and b.
export function boundaryFraction(b, total) {
  return total > 1 ? clamp((b - 0.5) / (total - 1), 0, 1) : 0;
}

// The colours of the contours: in the series views those of the surface interface; in the
// difference view, whose colours run to red and yellow, another pair, drawn on a dark outline.
export const SURFACE_COLOURS = { white: '#FF3B30', pial: '#FFD400' };
export const DIFF_SURFACE_COLOURS = { white: '#3DFF5A', pial: '#FF4DFF', outline: '#000000' };

// ---- contours ---------------------------------------------------------------------------------

// The contour file of one plane (func.contours), inflated: lines of points in voxel
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

export const CONTOUR_UNIT = 64;

export function formatBytes(n) {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(1)} MB`;
}
