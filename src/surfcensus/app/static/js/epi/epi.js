// The EPI page: one continuous stream of a subject's functional runs (decision D38), raw and
// preprocessed on the same slices with the white and pial contours. State, keys, the playback
// loop and what is written on the page. Ported from stage 1's player (main.js).
//
// The stream is every packed run of every session, in order, as one timeline of volumes. At a
// run or session boundary nothing changes but the pictures: the slices stay at the same place
// in the anatomy (on another grid, the slices nearest to that place), and the view, the
// layers, the rate and the crosshair carry over. What the page says about the run on screen
// changes beside the pictures.
import {
  boundaries,
  buildTimeline,
  clamp,
  CONTOUR_UNIT,
  decodeContours,
  DIFF_SCALES,
  diffLut,
  extractSlice,
  formatBytes,
  groupsOf,
  HOLD_MS,
  intensityTables,
  KINDS,
  locate,
  PLANES,
  planeSize,
  playbackStep,
  pointOf,
  RATES,
  signalOf,
  slicesAt,
  SPEC,
  stepRate,
  streamPanelSizesMm,
  streamsOf,
  streamTraces,
  VIEWS,
  wrap,
} from './logic.js';
import { inflate, sha256Hex, Store } from './store.js';
import { Traces } from './traces.js';
import { Views } from './views.js';

const $ = (id) => document.getElementById(id);
const T_OPEN = performance.now();

const state = {
  g: 0,
  slices: [0, 0, 0],
  place: [0, 0, 0], // the point in the anatomy the slices were last chosen at (mm, RAS)
  view: 'raw',
  kind: 'series',
  playing: false,
  rate: 12,
  contours: true,
  crosshair: true,
  traces: true,
  normalized: false,
  diffScale: 32,
  fdLimit: 0.5,
  active: 0,
  help: false,
};

let stream;
let timeline;
let store;
let views;
let traces;
let tables;
let lut = diffLut(state.diffScale);
let shown = null; // what the panels hold now
let pending = false;
let pendingReason = 'other';
let nextDue = 0;
let seek = null;
let lastFlip = 'pre';
let stalled = false;
let segOnScreen = -1;
let allGrids = [];
const contoursOf = new Map(); // grid key -> [three decoded files] while loading or loaded; null without contours
const sliceBuffers = [];
const waiters = [];
const marker = { fd: null, ns: null, shownKind: '' };
const page = {
  log: [], // { ms, g, seg, grid, view, kind, reason, nz, slices } for every painted volume of a series
  markerLog: [],
  timing: { opened: T_OPEN, stream: null, firstFrame: null, contours: null, allLoaded: null },
  lastSeek: null,
  stalls: 0,
  resyncs: 0,
  errors: [],
  requests: 0,
  fetchedBytes: 0,
};

// ---- the stream -----------------------------------------------------------------------------

function segOf(g) {
  return locate(timeline, g);
}

function segment(k) {
  return timeline.segments[k];
}

function runOf(k) {
  const seg = segment(k);
  return stream.sessions[seg.session].runs[seg.run];
}

function sessionOf(k) {
  return stream.sessions[segment(k).session];
}

function gridOf(k) {
  const seg = segment(k);
  return stream.sessions[seg.session].grids.find((x) => x.grid_key === seg.gridKey);
}

function stateKey(s) {
  return [s.kind === 'series' ? s.g : `${segOf(s.g)}:${s.kind}`, s.view, s.slices.join(','), s.diffScale, s.normalized].join('|');
}

// The contours of a grid belong to its session: two sessions on one grid may have the
// contours of different FreeSurfer subjects, or one of them none.
function contourKey(k) {
  return `${segment(k).session}|${segment(k).gridKey}`;
}

function contoursReady(k) {
  const grid = gridOf(k);
  if (!grid.contours) return true;
  const c = contoursOf.get(contourKey(k));
  return Array.isArray(c) && c.every((x) => x !== null);
}

// whether volume g can be painted now for the streams of the view (and its grid's contours)
function readyFor(g) {
  const k = segOf(g);
  if (!store.stillsNow(k)) return false;
  if (!streamsOf(state.view).every((s) => store.volume(s, g))) return false;
  return !state.contours || contoursReady(k);
}

// ---- painting ---------------------------------------------------------------------------------

function groupLabels() {
  const what = { series: '', mean: 'temporal mean: ', sd: 'temporal SD: ' }[state.kind];
  return {
    raw: `${what}raw data, one placement per session`,
    pre: `${what}preprocessed, as the pipeline wrote it`,
    diff: `${what}raw minus preprocessed`,
  };
}

function setPending(on, reason) {
  pending = on;
  pendingReason = reason || 'other';
  views.stale(on);
}

function volumesFor(k, streams) {
  const stills = store.stillsNow(k);
  if (!stills) return null;
  const out = {};
  for (const s of streams) {
    out[s] = state.kind === 'series' ? store.volume(s, state.g) : stills[s][state.kind];
    if (!out[s]) return null;
  }
  return out;
}

// The grid of the segment on screen, and the slices at the place on it.
function enterSegment(k) {
  if (k === segOnScreen) return;
  const before = segOnScreen;
  segOnScreen = k;
  const grid = gridOf(k);
  if (before < 0 || gridOf(before).grid_key !== grid.grid_key) {
    state.slices = slicesAt(grid, state.place);
    views.setGrid(grid);
    for (let plane = 0; plane < 3; plane++) {
      const [w, h] = planeSize(grid.shape, plane);
      sliceBuffers[plane] = [new Uint8Array(w * h), new Uint8Array(w * h)];
    }
    writeSliceControls();
  }
  writeRun();
  writeLegend();
}

function render(reason = 'other') {
  const k = segOf(state.g);
  enterSegment(k);
  const streams = streamsOf(state.view);
  store.want(streams, state.g);
  views.setGroups(groupsOf(state.view), groupLabels());
  const vols = volumesFor(k, streams);
  writeControls();
  if (!vols) {
    setPending(true, reason);
    if (!store.stillsNow(k)) store.stills(k).then(() => pending && render(pendingReason), fail);
    writeStatus();
    return false;
  }
  setPending(false);
  const grid = gridOf(k);
  const shape = grid.shape;
  const table = state.kind === 'sd' ? tables[k].sd : tables[k].value;
  let nz = 0;
  for (const group of groupsOf(state.view)) {
    for (let plane = 0; plane < 3; plane++) {
      const index = state.slices[SPEC[plane].axis];
      const [a, b] = sliceBuffers[plane];
      if (group === 'diff') {
        extractSlice(vols.raw, shape, plane, index, a, table);
        extractSlice(vols.pre, shape, plane, index, b, table);
        views.paintDiff(group, plane, a, b, lut);
      } else {
        extractSlice(vols[group], shape, plane, index, a, table);
        views.paintGray(group, plane, a);
      }
      if (plane === 0) for (let i = 0; i < a.length; i++) if (a[i]) nz += 1;
    }
  }
  drawOverlay();
  const now = performance.now();
  shown = { g: state.g, seg: k, grid: grid.grid_key, kind: state.kind, view: state.view, slices: state.slices.slice(), normalized: state.normalized, key: stateKey(state) };
  if (state.kind === 'series') {
    page.log.push({ ms: now, g: state.g, seg: k, grid: grid.grid_key, view: state.view, reason, nz, slices: state.slices.slice() });
    if (page.log.length > 40000) page.log.splice(0, 20000);
    if (page.timing.firstFrame === null) page.timing.firstFrame = now;
    if (seek && seek.g === state.g) {
      page.lastSeek = { g: seek.g, ms: now - seek.at };
      seek = null;
    }
    if (reason === 'seek') {
      // a jump ends the markers of where the page was: they belong to another volume
      closeMarker('fd', now);
      closeMarker('ns', now);
    }
    raiseMarkers(now);
  }
  showMarker(now);
  traces.draw(state.g);
  writeStatus(vols, k);
  while (waiters.length) waiters.shift()(shown);
  return true;
}

function drawOverlay() {
  const k = segOf(state.g);
  const loaded = contoursOf.get(contourKey(k));
  const planes = Array.isArray(loaded) ? loaded : [null, null, null];
  views.overlay(state.slices, planes, state.contours, state.crosshair, state.active, contourKey(k));
}

function raiseMarkers(now) {
  const fd = streamData.fd[state.g];
  if (fd !== null && fd > state.fdLimit) {
    if (!marker.fd || marker.fd.g !== state.g) {
      closeMarker('fd', now);
      marker.fd = { g: state.g, value: fd, until: now + HOLD_MS, on: now };
    }
  }
  if (streamData.nonSteadySet.has(state.g)) {
    if (!marker.ns || marker.ns.g !== state.g) {
      closeMarker('ns', now);
      marker.ns = { g: state.g, until: now + HOLD_MS, on: now };
    }
  }
}

function closeMarker(kind, now) {
  const m = marker[kind];
  if (m) page.markerLog.push({ kind, g: m.g, on: m.on, off: now });
  marker[kind] = null;
}

function localVolume(g) {
  const k = segOf(g);
  return { k, t: g - segment(k).first };
}

// A marker stays while its volume is on screen and at least HOLD_MS after it was raised.
function showMarker(now) {
  for (const kind of ['fd', 'ns']) {
    const m = marker[kind];
    const onScreen = m && state.kind === 'series' && state.g === m.g;
    if (m && !onScreen && now >= m.until) closeMarker(kind, now);
  }
  const visible = state.traces;
  let kind = '';
  let text = '';
  if (visible && marker.fd) {
    kind = 'fd';
    text = `FD ${marker.fd.value.toFixed(2)} mm at volume ${localVolume(marker.fd.g).t} of its run`;
  } else if (visible && marker.ns) {
    kind = 'ns';
    text = `non-steady state, volume ${localVolume(marker.ns.g).t} of its run`;
  }
  const signature = kind + text;
  if (signature !== marker.shownKind) {
    marker.shownKind = signature;
    views.marker(kind, text);
  }
}

// ---- what is written on the page ----------------------------------------------------------------

let streamData = null;

function writeSliceControls() {
  const grid = gridOf(segOf(state.g));
  for (const label of $('slices').children) {
    const input = label.querySelector('input');
    input.max = String(grid.shape[Number(input.dataset.plane)] - 1);
  }
}

function writeControls() {
  const k = segOf(state.g);
  const { t } = localVolume(state.g);
  const run = runOf(k);
  for (const b of $('runs').querySelectorAll('button[data-seg]')) b.setAttribute('aria-pressed', String(Number(b.dataset.seg) === k));
  for (const b of $('views-buttons').children) b.setAttribute('aria-pressed', String(b.dataset.view === state.view));
  for (const b of $('kinds-buttons').children) b.setAttribute('aria-pressed', String(b.dataset.kind === state.kind));
  $('normalize').setAttribute('aria-pressed', String(state.normalized));
  $('normalize-text').textContent = state.normalized ? 'on' : 'off';
  const time = $('time');
  time.max = String(timeline.total - 1);
  time.value = String(state.g);
  time.disabled = state.kind !== 'series';
  $('play-text').textContent = state.playing ? 'pause' : 'play';
  $('rate').value = String(state.rate);
  const tr = run.tr_s ? `, ${(t * run.tr_s).toFixed(1)} s into the run` : '';
  $('readout').textContent =
    state.kind === 'series'
      ? `volume ${state.g} of 0 to ${timeline.total - 1} of the stream; ${t} of 0 to ${run.n_volumes - 1} of ${run.name}${tr}`
      : `still of ${run.name} over ${run.pack.stills_volumes} of ${run.n_volumes} volumes`;
  const grid = gridOf(k);
  for (const label of $('slices').children) {
    const input = label.querySelector('input');
    const plane = Number(input.dataset.plane);
    const index = state.slices[plane];
    input.value = String(index);
    const mm = grid.slice_mm[PLANES[plane]][index];
    label.querySelector('output').textContent = `${index} of ${grid.shape[plane] - 1}, ${mm.toFixed(1)} mm`;
  }
  $('contours').checked = state.contours;
  $('crosshair').checked = state.crosshair;
  $('traces-on').checked = state.traces;
  $('diff-scale').value = String(state.diffScale);
  document.body.classList.toggle('no-traces', !state.traces);
}

function writeRun() {
  const k = segOf(state.g);
  const run = runOf(k);
  const session = sessionOf(k);
  const reg = session.registration;
  const rule = reg && reg.rule === 'acquisition_time' ? 'the earliest acquisition time' : 'the first in name order (no acquisition times)';
  const f = run.frame;
  const anchorName = reg ? (session.runs.find((r) => r.run_uid === reg.anchor) || {}).name : null;
  let note;
  if (run.anchor) {
    note = `${session.label}, ${run.name}: the anchor run of the session (${rule}); the raw stream of every run of the session is drawn through its stored registration to the anatomy.`;
  } else if (f) {
    note =
      `${session.label}, ${run.name}: drawn through the registration of the anchor run ${anchorName}, it lies ${f.mean_mm.toFixed(2)} mm on average and ` +
      `${f.max_mm.toFixed(2)} mm at most (over its brain mask) from where its own stored registration puts it: ` +
      `a rotation of ${f.rotation_deg.toFixed(2)} degrees and a shift of ${f.translation_mm.toFixed(2)} mm.`;
  } else {
    note = `${session.label}, ${run.name}.`;
  }
  $('run-note').textContent = note;
  const warnings = [];
  if (f && f.flagged) {
    warnings.push(
      `${run.name} is FLAGGED (raw data): drawn through the session's one registration, it lies more than ${reg.limit_mm} mm somewhere in the brain from where its own registration puts it. ` +
        'The head moved that much since the anchor run, the run may not share its scanner frame, or a stored registration failed. It is shown through the session\'s registration all the same.',
    );
  }
  if (reg && reg.anchor_is_outlier) {
    warnings.push(`Every other run of ${session.label} is flagged against the anchor ${anchorName} and they agree with each other: the anchor, or its stored registration, is the odd one.`);
  }
  if (run.pack.raw_affine === 'header, unchecked') {
    warnings.push(`The raw affine of ${run.name} is unchecked: the pipeline wrote no reference image to compare the header with, so the raw stream may lie off by a disagreement between qform and sform.`);
  }
  if (!run.pack.rigid) warnings.push(`${run.name}: affine registration, not rigid; it is used as stored.`);
  $('warning').hidden = warnings.length === 0;
  $('warning').textContent = warnings.join(' ');
  const missing = session.runs.filter((r) => r.reason !== null);
  $('session-note').textContent = missing.length
    ? `${session.label}: ${missing.length} run(s) not in the stream: ${missing.map((r) => `${r.name} (${r.reason}${r.detail ? `: ${r.detail}` : ''})`).join('; ')}.`
    : `${session.label}: every listed run is in the stream.`;
  writeCorrections();
}

function statementText(s, applied, notApplied, unknown) {
  return s.state === 'applied' ? applied : s.state === 'not applied' ? notApplied : unknown;
}

function correctionsText(run) {
  const c = run.pack.corrections;
  const steps = ['head-motion correction'];
  steps.push(statementText(c.sdc, `distortion correction, ${c.sdc_method || 'method not stated'}`, 'no distortion correction', 'distortion correction not stated'));
  steps.push(statementText(c.slice_timing, 'slice-timing correction', 'no slice-timing correction', 'slice-timing correction not stated'));
  const reg = c.registration.words ? `; registration: ${c.registration.words}${c.dof ? `, ${c.dof} dof` : ''}` : '';
  const source = [c.sdc.source, c.slice_timing.source].filter(Boolean)[0];
  let text = `Preprocessing of ${run.name}: ${steps.join('; ')}${reg}${source ? ` (as stated in the ${source.replace(/_/g, ' ')})` : ''}. The raw stream has none of these, only the session's one placement.`;
  if (c.sdc.state === 'applied') {
    text += ' So the raw stream shows the distortion as acquired against the contours, and the preprocessed stream how well the correction brought the series onto them.';
  }
  return { level: c.sdc.state === 'unknown' || c.slice_timing.state === 'unknown' ? 'unknown' : 'known', text };
}

function writeCorrections() {
  const { level, text } = correctionsText(runOf(segOf(state.g)));
  const box = $('corrections');
  box.textContent = text;
  box.dataset.level = level;
  box.dataset.strong = String(state.view !== 'pre');
}

// What the gray levels on screen stand for, and the scale of the difference, under the
// setting of intensity normalization.
function legendText() {
  const k = segOf(state.g);
  const run = runOf(k);
  const sd = state.kind === 'sd';
  const what = sd ? 'temporal SD' : 'signal';
  const top = sd ? tables[k].topSd : tables[k].top;
  const scope = state.normalized
    ? `Intensity normalization on: each run under its own window; for ${run.name} gray 0 to 255 is ${what} 0 to ${top.toFixed(0)}.`
    : `Intensity normalization off: one window for the whole stream; gray 0 to 255 is ${what} 0 to ${top.toFixed(0)} in every run.`;
  if (state.view !== 'diff') return scope;
  return (
    `${scope} Difference raw minus preprocessed of the gray levels on screen, the effect of preprocessing on this run: the colour saturates at ${state.diffScale} gray levels ` +
    `(${((state.diffScale * top) / 255).toFixed(0)} units of ${what}). Blue: raw lower. Red: raw higher.`
  );
}

function writeLegend() {
  if (!tables || segOnScreen < 0) return;
  const bar = $('legend-bar');
  if (state.view === 'diff') {
    const ctx = bar.getContext('2d');
    const image = ctx.createImageData(255, 10);
    for (let x = 0; x < 255; x++) {
      const d = Math.round(((x - 127) / 127) * state.diffScale);
      for (let y = 0; y < 10; y++) {
        const o = 4 * (y * 255 + x);
        image.data.set(lut.subarray(3 * (d + 255), 3 * (d + 255) + 3), o);
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }
  bar.hidden = state.view !== 'diff';
  $('legend-text').textContent = legendText();
  $('legend').dataset.normalized = String(state.normalized);
}

function measuredRate() {
  const now = performance.now();
  const recent = page.log.filter((e) => e.reason === 'play' && now - e.ms < 2000);
  return recent.length > 1 ? ((recent.length - 1) * 1000) / (recent[recent.length - 1].ms - recent[0].ms) : null;
}

function writeStatus(vols, k) {
  const parts = [];
  if (pending) parts.push(state.kind === 'series' ? `loading volume ${state.g}` : 'loading');
  if (vols && k !== undefined) {
    const [i, j, kk] = state.slices;
    const shape = gridOf(k).shape;
    const at = i + shape[0] * (j + shape[1] * kk);
    const run = runOf(k);
    const win = state.kind === 'sd' ? run.sd_window : run.window;
    const values = Object.entries(vols).map(([s, v]) => `${s === 'pre' ? 'preprocessed' : s} ${v[at]} (${signalOf(v[at], win).toFixed(1)})`);
    parts.push(`voxel (${i}, ${j}, ${kk}): ${values.join(', ')} of 255 stored`);
  }
  if (state.playing) {
    const rate = measuredRate();
    parts.push(`${state.rate} volumes per s asked${rate ? `, ${rate.toFixed(1)} painted` : ''}`);
  }
  const s = store.stats();
  parts.push(`${formatBytes(s.decodedChunkBytes + s.stillBytes)} decoded in memory (chunks limited to ${formatBytes(s.limitBytes)})`);
  if (s.unchecked) parts.push('the SHA-256 of the files could not be checked (no secure context)');
  if (page.errors.length) parts.push(`ERROR: ${page.errors[page.errors.length - 1]}`);
  $('status').textContent = parts.join(' · ');
}

function fail(err) {
  page.errors.push(String(err && err.message ? err.message : err));
  if (store) writeStatus();
}

// ---- changes of state ----------------------------------------------------------------------------

function setTime(g, reason) {
  state.g = clamp(Math.round(g), 0, timeline.total - 1);
  if (state.kind !== 'series') state.kind = 'series';
  if (reason === 'seek') seek = { g: state.g, at: performance.now() };
  render(reason);
}

function jumpToSegment(k) {
  setTime(segment(wrap(k, timeline.segments.length)).first, 'seek');
}

function setView(view) {
  if (view === 'raw' || view === 'pre') lastFlip = view === 'raw' ? 'pre' : 'raw';
  state.view = view;
  writeLegend();
  writeCorrections();
  views.layout();
  render('other');
}

function setKind(kind) {
  state.kind = kind;
  if (kind !== 'series') state.playing = false;
  writeLegend();
  render('other');
}

function placeFromSlices() {
  state.place = pointOf(gridOf(segOf(state.g)), state.slices);
}

function setSlice(plane, index) {
  const grid = gridOf(segOf(state.g));
  state.slices[plane] = clamp(index, 0, grid.shape[plane] - 1);
  placeFromSlices();
  render('other');
}

function setNormalized(on) {
  state.normalized = on;
  tables = intensityTables(windows.value, windows.sd, on);
  writeLegend();
  render('other');
}

function setPlaying(on) {
  state.playing = on;
  if (on) {
    if (state.kind !== 'series') {
      state.kind = 'series';
      writeLegend();
    }
    nextDue = performance.now();
  }
  render('other');
}

function setRate(rate) {
  state.rate = rate;
  nextDue = performance.now();
  writeControls();
}

function setDiffScale(scale) {
  state.diffScale = scale;
  lut = diffLut(scale);
  writeLegend();
  render('other');
}

function setHelp(on) {
  state.help = on;
  $('help').hidden = !on;
  if (on) {
    state.playing = false;
    writeControls();
    $('help-close').focus({ preventScroll: true });
  } else {
    $('help-button').focus();
  }
}

function tick(now) {
  if (state.playing && state.kind === 'series' && !state.help) {
    const step = playbackStep(nextDue, now, state.rate);
    if (step.advance) {
      const next = wrap(state.g + 1, timeline.total);
      if (readyFor(next)) {
        state.g = next;
        nextDue = step.nextDue;
        if (step.resynced) page.resyncs += 1;
        stalled = false;
        render('play');
      } else {
        // the next volume is not loaded: wait for it and do not skip it
        if (!stalled) {
          page.stalls += 1;
          stalled = true;
          $('status').textContent = `loading volume ${next}`;
        }
        store.want(streamsOf(state.view), next);
        nextDue = now;
      }
    }
  }
  if (marker.fd || marker.ns) showMarker(now);
  if (page.timing.allLoaded === null && page.timing.firstFrame !== null && store.idle() && [...contoursOf.values()].every((c) => c === null || c.every((x) => x !== null))) {
    page.timing.allLoaded = now;
    writeStatus();
  }
  requestAnimationFrame(tick);
}

// ---- keys --------------------------------------------------------------------------------------------

let contourKeyDown = 0;

function onKeyDown(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key;
  if (state.help) {
    if (key === 'Escape' || key === '?') {
      event.preventDefault();
      setHelp(false);
    }
    return;
  }
  const target = event.target;
  if (target instanceof HTMLInputElement && target.type === 'number') return;
  if ((key === ' ' || key === 'Enter') && (target instanceof HTMLButtonElement || target instanceof HTMLInputElement)) return;
  if (key.startsWith('Arrow') && (target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target.type === 'range'))) return;
  const step = event.shiftKey ? 10 : 1;
  const actions = {
    ' ': () => setPlaying(!state.playing),
    ArrowLeft: () => {
      state.playing = false;
      setTime(wrap(state.g - step, timeline.total), 'seek');
    },
    ArrowRight: () => {
      state.playing = false;
      setTime(wrap(state.g + step, timeline.total), 'seek');
    },
    Home: () => setTime(0, 'seek'),
    End: () => setTime(timeline.total - 1, 'seek'),
    ArrowUp: () => setSlice(state.active, state.slices[state.active] + (event.shiftKey ? 5 : 1)),
    ArrowDown: () => setSlice(state.active, state.slices[state.active] - (event.shiftKey ? 5 : 1)),
    '[': () => setRate(stepRate(state.rate, -1)),
    ']': () => setRate(stepRate(state.rate, 1)),
    p: () => {
      state.active = (state.active + 1) % 3;
      drawOverlay();
    },
    1: () => setView('raw'),
    2: () => setView('pre'),
    3: () => setView('side'),
    4: () => setView('diff'),
    f: () => setView(state.view === 'raw' ? 'pre' : state.view === 'pre' ? 'raw' : lastFlip === 'raw' ? 'pre' : 'raw'),
    q: () => setKind('series'),
    w: () => setKind('mean'),
    e: () => setKind('sd'),
    n: () => setNormalized(!state.normalized),
    r: () => jumpToSegment(segOf(state.g) + 1),
    R: () => jumpToSegment(segOf(state.g) - 1),
    s: () => {
      if (event.repeat) return;
      contourKeyDown = performance.now();
      state.contours = !state.contours;
      writeControls();
      drawOverlay();
    },
    c: () => {
      state.crosshair = !state.crosshair;
      writeControls();
      drawOverlay();
    },
    t: () => {
      state.traces = !state.traces;
      writeControls();
      views.layout();
      traces.resize();
      render('other');
    },
    '?': () => setHelp(true),
  };
  const action = actions[key];
  if (!action) return;
  event.preventDefault();
  action();
}

function onKeyUp(event) {
  // s held down for a while: the contours were off only while it was held
  if (event.key === 's' && contourKeyDown && performance.now() - contourKeyDown > 400 && !state.help) {
    state.contours = !state.contours;
    writeControls();
    drawOverlay();
  }
  if (event.key === 's') contourKeyDown = 0;
}

// ---- start -------------------------------------------------------------------------------------------

let windows = { value: [], sd: [] };

function buildRuns() {
  const box = $('runs');
  box.replaceChildren();
  stream.sessions.forEach((session, si) => {
    const group = document.createElement('div');
    group.className = 'session';
    group.dataset.session = String(si);
    const head = document.createElement('span');
    head.className = 'session-label';
    head.textContent = session.label;
    group.append(head);
    session.runs.forEach((run, ri) => {
      const k = timeline.segments.findIndex((seg) => seg.session === si && seg.run === ri);
      const b = document.createElement('button');
      b.type = 'button';
      const away = document.createElement('span');
      away.className = 'away';
      if (k < 0) {
        b.disabled = true;
        b.textContent = `${run.name} `;
        away.textContent = run.reason || 'no pack';
        b.title = run.detail ? `not in the stream: ${run.detail}` : 'not in the stream';
      } else {
        b.dataset.seg = String(k);
        b.textContent = `${run.name} `;
        const where = run.anchor ? 'anchor' : run.frame ? `${run.frame.max_mm.toFixed(1)} mm` : '';
        away.textContent = run.frame && run.frame.flagged ? `${where}, flagged` : where;
        b.classList.toggle('flagged', Boolean(run.frame && run.frame.flagged));
        b.title = `jump to volume ${segment(k).first} of the stream`;
        b.addEventListener('click', () => jumpToSegment(k));
      }
      b.append(away);
      group.append(b);
    });
    box.append(group);
  });
}

function buildControls() {
  buildRuns();
  for (const b of $('views-buttons').children) b.addEventListener('click', () => setView(b.dataset.view));
  for (const b of $('kinds-buttons').children) b.addEventListener('click', () => setKind(b.dataset.kind));
  $('normalize').addEventListener('click', () => setNormalized(!state.normalized));
  for (const rate of RATES) $('rate').add(new Option(String(rate), String(rate)));
  for (const scale of DIFF_SCALES) $('diff-scale').add(new Option(String(scale), String(scale)));
  $('rate').addEventListener('change', (event) => {
    setRate(Number(event.target.value));
    event.target.blur();
  });
  $('diff-scale').addEventListener('change', (event) => {
    setDiffScale(Number(event.target.value));
    event.target.blur();
  });
  $('play').addEventListener('click', () => setPlaying(!state.playing));
  $('time').addEventListener('input', (event) => {
    state.playing = false;
    setTime(Number(event.target.value), 'seek');
  });
  for (const label of $('slices').children) {
    const input = label.querySelector('input');
    const plane = Number(input.dataset.plane);
    input.addEventListener('input', () => {
      state.active = plane;
      setSlice(plane, Number(input.value));
    });
  }
  for (const input of document.querySelectorAll('input[type=range]')) input.addEventListener('change', () => input.blur());
  $('contours').addEventListener('change', (event) => {
    state.contours = event.target.checked;
    drawOverlay();
  });
  $('crosshair').addEventListener('change', (event) => {
    state.crosshair = event.target.checked;
    drawOverlay();
  });
  $('traces-on').addEventListener('change', (event) => {
    state.traces = event.target.checked;
    writeControls();
    views.layout();
    traces.resize();
    render('other');
  });
  $('fd-limit').value = String(state.fdLimit);
  $('fd-limit').addEventListener('change', (event) => {
    const value = Number(event.target.value);
    if (Number.isFinite(value) && value >= 0) {
      state.fdLimit = value;
      traces.setData(streamData, state.fdLimit);
      render('other');
    }
  });
  $('help-button').addEventListener('click', () => setHelp(true));
  $('help-close').addEventListener('click', () => setHelp(false));
  $('help').addEventListener('click', (event) => {
    if (event.target === $('help')) setHelp(false);
  });
  const limits = [...new Set(stream.sessions.filter((s) => s.registration).map((s) => s.registration.limit_mm))];
  $('help-limit').textContent = limits.join(', ');
  $('help-hold').textContent = String(HOLD_MS);
  $('help-order').textContent = stream.order;
  $('help-memory').textContent = formatBytes(store.limitBytes);
  $('help-window').textContent = `${stream.rules.window}; ${stream.rules.quantise}. ${stream.rules.common_scale}.`;
}

function availableHeight() {
  const used = ['bar', 'reading', 'runs', 'run-note', 'session-note', 'corrections', 'legend', 'controls', 'status'].reduce((a, id) => a + $(id).offsetHeight, 0);
  const warn = $('warning').hidden ? 0 : $('warning').offsetHeight;
  const tr = state.traces ? $('traces').offsetHeight : 0;
  if (window.innerWidth < 700) return 0.9 * window.innerHeight;
  return Math.max(window.innerHeight - used - warn - tr - 70, 0.4 * window.innerHeight);
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`${url}: status ${response.status}`);
  const buffer = await response.arrayBuffer();
  page.requests += 1;
  page.fetchedBytes += buffer.byteLength;
  return JSON.parse(new TextDecoder().decode(buffer));
}

// The contours of every grid of the stream, small, loaded at the start in parallel.
function loadContours() {
  const jobs = [];
  stream.sessions.forEach((session, si) => {
    for (const grid of session.grids) {
      const key = `${si}|${grid.grid_key}`;
      if (!grid.contours) {
        contoursOf.set(key, null);
        continue;
      }
      const planes = [null, null, null];
      contoursOf.set(key, planes);
      PLANES.forEach((name, plane) => {
        const entry = grid.contours[name];
        jobs.push(
          (async () => {
            const response = await fetch(`${session.media_url}${entry.file}?v=${encodeURIComponent(grid.token)}`);
            if (!response.ok) throw new Error(`${entry.file}: status ${response.status}`);
            const packed = await response.arrayBuffer();
            page.requests += 1;
            page.fetchedBytes += packed.byteLength;
            const digest = await sha256Hex(packed);
            if (digest && digest !== entry.sha256) throw new Error(`${entry.file}: the bytes are not those the session lists`);
            const data = await inflate(packed);
            planes[plane] = decodeContours(data.buffer, stream.contour_unit || CONTOUR_UNIT);
            if (shown && contourKey(shown.seg) === key) drawOverlay();
          })(),
        );
      });
    }
  });
  return Promise.all(jobs).then(() => {
    page.timing.contours = performance.now();
  });
}

async function start() {
  const api = document.querySelector('main[data-api]').dataset.api;
  stream = await fetchJson(api);
  page.timing.stream = performance.now();
  timeline = buildTimeline(stream.sessions);
  if (timeline.total === 0) {
    $('status').textContent = 'No run of this subject is packed: there is nothing to show.';
    return;
  }
  allGrids = stream.sessions.flatMap((s) => s.grids);
  windows = {
    value: timeline.segments.map((seg) => stream.sessions[seg.session].runs[seg.run].window),
    sd: timeline.segments.map((seg) => stream.sessions[seg.session].runs[seg.run].sd_window),
  };
  tables = intensityTables(windows.value, windows.sd, state.normalized);
  const tr = streamTraces(timeline, stream.sessions);
  streamData = { total: timeline.total, fd: tr.fd, dvars: tr.dvars, nonSteady: tr.nonSteady, nonSteadySet: new Set(tr.nonSteady), boundaries: boundaries(timeline) };
  store = new Store(stream, timeline, { limitBytes: stream.memory_mib * 1024 * 1024 });
  store.onchunk = () => {
    if (pending) render(pendingReason);
  };
  store.onerror = fail;
  document.title = `EPI movie: ${stream.label}`;
  $('subject').textContent = stream.label;
  $('reading').textContent = stream.reading;
  state.fdLimit = stream.sessions.find((s) => s.fd_threshold_mm)?.fd_threshold_mm || 0.5;
  const first = gridOf(0);
  state.slices = first.shape.map((n) => Math.floor(n / 2));
  state.place = pointOf(first, state.slices);
  views = new Views($('views'), streamPanelSizesMm(allGrids), {
    pick: (plane, a, b) => {
      state.active = plane;
      state.slices[SPEC[plane].h] = a;
      state.slices[SPEC[plane].v] = b;
      placeFromSlices();
      render('other');
    },
    wheel: (plane, direction) => {
      state.active = plane;
      setSlice(SPEC[plane].axis, state.slices[SPEC[plane].axis] + direction);
    },
    availableHeight,
  });
  traces = new Traces($('traces'), (g) => {
    state.playing = false;
    setTime(g, 'seek');
  });
  buildControls();
  enterSegment(0);
  views.setGroups(groupsOf(state.view), groupLabels());
  traces.setData(streamData, state.fdLimit);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('pointerup', (event) => {
    const button = event.target instanceof Element ? event.target.closest('button') : null;
    if (button && !state.help) button.blur();
  });
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', () => {
    views.layout();
    traces.resize();
    if (shown) {
      drawOverlay();
      traces.draw(state.g);
    }
  });
  loadContours().catch(fail);
  await store.stills(0);
  render('other');
  requestAnimationFrame(tick);
}

// What the browser tests drive and read. Not used by the page itself.
window.__epi = {
  state,
  page,
  get stream() {
    return stream;
  },
  get timeline() {
    return timeline;
  },
  get shown() {
    return shown;
  },
  tables: () => tables.map((t) => ({ value: Array.from(t.value), sd: Array.from(t.sd), top: t.top, topSd: t.topSd })),
  stats: () => ({ ...store.stats(), pageRequests: page.requests, pageFetchedBytes: page.fetchedBytes }),
  // Change the state (g, slices, view, kind, rate, playing, diffScale, contours, crosshair,
  // normalized) and resolve when the panels hold what was asked for.
  set(change) {
    if ('view' in change && VIEWS.includes(change.view)) setView(change.view);
    if ('normalized' in change && change.normalized !== state.normalized) setNormalized(change.normalized);
    if ('g' in change) {
      state.g = clamp(change.g, 0, timeline.total - 1);
      state.kind = 'series';
      seek = { g: state.g, at: performance.now() };
      enterSegment(segOf(state.g));
    }
    if ('kind' in change && KINDS.includes(change.kind)) {
      state.kind = change.kind;
      if (change.kind !== 'series') state.playing = false;
      writeLegend();
    }
    if ('slices' in change) {
      const grid = gridOf(segOf(state.g));
      state.slices = change.slices.map((v, k) => clamp(v, 0, grid.shape[k] - 1));
      placeFromSlices();
    }
    if ('diffScale' in change) setDiffScale(change.diffScale);
    if ('rate' in change) setRate(change.rate);
    for (const k of ['contours', 'crosshair']) if (k in change) state[k] = change[k];
    if ('playing' in change) setPlaying(change.playing);
    else render('g' in change ? 'seek' : 'other');
    return window.__epi.ready();
  },
  ready() {
    if (shown && shown.key === stateKey(state) && !pending) return Promise.resolve(shown);
    return new Promise((resolve) => waiters.push(resolve));
  },
  allLoaded() {
    return page.timing.allLoaded !== null;
  },
  has: (view, g) => streamsOf(view).every((s) => store.volume(s, g) !== null),
  idle: () => store.idle(),
  // the gray levels a panel holds, as the canvas has them, and its size; read through a copy
  // made for reading, so that the page's own canvases stay as the page made them
  pixels(group, plane, layer = 'gray') {
    const canvas = document.querySelector(`#views .epi-group[data-group="${group}"] .epi-panel[data-plane="${plane}"] canvas.${layer}`);
    const copy = document.createElement('canvas');
    copy.width = canvas.width;
    copy.height = canvas.height;
    const ctx = copy.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0);
    const data = ctx.getImageData(0, 0, copy.width, copy.height).data;
    if (layer !== 'gray') {
      let drawn = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i]) drawn += 1;
      return { w: copy.width, h: copy.height, drawn };
    }
    const out = [];
    for (let i = 0; i < data.length; i += 4) out.push([data[i], data[i + 1], data[i + 2]]);
    return { w: copy.width, h: copy.height, rgb: out };
  },
  // where a panel's canvas lies on the page, in CSS px
  rect(group, plane) {
    const canvas = document.querySelector(`#views .epi-group[data-group="${group}"] .epi-panel[data-plane="${plane}"] canvas.gray`);
    const r = canvas.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height, cw: canvas.width, ch: canvas.height };
  },
  boxRect(group, plane) {
    const box = document.querySelector(`#views .epi-group[data-group="${group}"] .epi-panel[data-plane="${plane}"]`);
    const r = box.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  },
  contoursDrawn() {
    const k = segOf(state.g);
    const c = contoursOf.get(contourKey(k));
    return { grid: gridOf(k).grid_key, contours: contourKey(k), loaded: Array.isArray(c) && c.every((x) => x !== null), key: views.overlayKey };
  },
  texts: () => ({
    runNote: $('run-note').textContent,
    warning: $('warning').hidden ? '' : $('warning').textContent,
    corrections: $('corrections').textContent,
    legend: $('legend-text').textContent,
    session: $('session-note').textContent,
    status: $('status').textContent,
    readout: $('readout').textContent,
  }),
};

start().catch((err) => {
  fail(err);
  $('status').textContent = `ERROR: ${String(err && err.message ? err.message : err)}`;
});
