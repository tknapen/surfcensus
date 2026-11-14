// The player page: state, keys, the playback loop and what is written on the page.
import {
  clamp,
  decodeContours,
  DIFF_SCALES,
  diffLut,
  extractSlice,
  formatBytes,
  groupsOf,
  HOLD_MS,
  KINDS,
  PLANES,
  planeSize,
  playbackStep,
  RATES,
  SPEC,
  stepRate,
  streamsOf,
  VIEWS,
  wrap,
} from './logic.js';
import { DEFAULT_LIMIT_MB, inflate, sha256Hex, Store } from './store.js';
import { Traces } from './traces.js';
import { Views } from './views.js';

const PACK_URL = '/pack/';
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const T_OPEN = performance.now();

const state = {
  run: 0,
  t: 0,
  slices: [0, 0, 0],
  view: 'raw',
  kind: 'series',
  playing: false,
  rate: 12,
  contours: true,
  crosshair: true,
  traces: true,
  diffScale: 32,
  fdLimit: 0.5,
  active: 0,
  help: false,
};

let manifest;
let store;
let views;
let traces;
let lut = diffLut(state.diffScale);
let shown = null; // what the panels hold now
let pending = false;
let pendingReason = 'other';
let nextDue = 0;
let seek = null;
let lastFlip = 'pre';
let stalled = false;
const contours = [null, null, null];
const sliceBuffers = [];
const waiters = [];
const marker = { fd: null, ns: null, shownKind: '' };
const page = {
  log: [], // { ms, run, t, view, reason } for every painted volume of a series
  markerLog: [], // { kind, t, on, off } for every marker on the frame
  timing: { opened: T_OPEN, firstFrame: null, contours: null, allLoaded: null },
  lastSeek: null,
  stalls: 0,
  resyncs: 0,
  errors: [],
  requests: 0,
  fetchedBytes: 0,
};

function run() {
  return manifest.runs[state.run];
}

function stateKey(s) {
  return [s.run, s.kind === 'series' ? s.t : -1, s.kind, s.view, s.slices.join(','), s.diffScale].join('|');
}

// ---- painting ------------------------------------------------------------------------------

function groupLabels() {
  const what = { series: '', mean: 'temporal mean: ', sd: 'temporal SD: ' }[state.kind];
  return {
    raw: `${what}raw, session registration`,
    pre: `${what}preprocessed, as stored`,
    diff: `${what}raw minus preprocessed`,
  };
}

function setPending(on, reason) {
  pending = on;
  pendingReason = reason || 'other';
  views.stale(on);
}

function volumesFor(streams) {
  const stills = store.stillsNow(state.run);
  if (!stills) return null;
  const out = {};
  for (const s of streams) {
    out[s] = state.kind === 'series' ? store.volume(state.run, s, state.t) : stills[s][state.kind];
    if (!out[s]) return null;
  }
  return out;
}

function render(reason = 'other') {
  const streams = streamsOf(state.view);
  store.want(state.run, streams, state.t);
  views.setGroups(groupsOf(state.view), groupLabels());
  const vols = volumesFor(streams);
  writeControls();
  if (!vols) {
    setPending(true, reason);
    if (!store.stillsNow(state.run)) store.stills(state.run).then(() => pending && render(pendingReason), fail);
    writeStatus();
    return false;
  }
  setPending(false);
  const shape = manifest.grid.shape;
  for (const group of groupsOf(state.view)) {
    for (let plane = 0; plane < 3; plane++) {
      const index = state.slices[SPEC[plane].axis];
      const [a, b] = sliceBuffers[plane];
      if (group === 'diff') {
        extractSlice(vols.raw, shape, plane, index, a);
        extractSlice(vols.pre, shape, plane, index, b);
        views.paintDiff(group, plane, a, b, lut);
      } else {
        extractSlice(vols[group], shape, plane, index, a);
        views.paintGray(group, plane, a);
      }
    }
  }
  drawOverlay();
  const now = performance.now();
  shown = { run: state.run, t: state.t, kind: state.kind, view: state.view, slices: state.slices.slice(), key: stateKey(state) };
  if (state.kind === 'series') {
    page.log.push({ ms: now, run: state.run, t: state.t, view: state.view, reason });
    if (page.log.length > 40000) page.log.splice(0, 20000);
    if (page.timing.firstFrame === null) page.timing.firstFrame = now;
    if (seek && seek.t === state.t) {
      page.lastSeek = { t: seek.t, ms: now - seek.at };
      seek = null;
    }
    raiseMarkers(now);
  }
  showMarker(now);
  traces.draw(state.t);
  writeStatus(vols);
  while (waiters.length) waiters.shift()(shown);
  return true;
}

function drawOverlay() {
  views.overlay(state.slices, contours, state.contours, state.crosshair, state.active);
}

function raiseMarkers(now) {
  const r = run();
  const fd = r.fd[state.t];
  if (fd !== null && fd > state.fdLimit) {
    if (!marker.fd || marker.fd.t !== state.t) {
      closeMarker('fd', now);
      marker.fd = { t: state.t, value: fd, until: now + HOLD_MS, on: now };
    }
  }
  if (r.non_steady.includes(state.t)) {
    if (!marker.ns || marker.ns.t !== state.t) {
      closeMarker('ns', now);
      marker.ns = { t: state.t, until: now + HOLD_MS, on: now };
    }
  }
}

function closeMarker(kind, now) {
  const m = marker[kind];
  if (m) page.markerLog.push({ kind, t: m.t, on: m.on, off: now });
  marker[kind] = null;
}

// A marker stays while its volume is on screen and at least HOLD_MS after it was raised.
function showMarker(now) {
  for (const kind of ['fd', 'ns']) {
    const m = marker[kind];
    const onScreen = m && state.kind === 'series' && shown && shown.run === state.run && state.t === m.t;
    if (m && !onScreen && now >= m.until) closeMarker(kind, now);
  }
  const visible = state.traces;
  let kind = '';
  let text = '';
  if (visible && marker.fd) {
    kind = 'fd';
    text = `FD ${marker.fd.value.toFixed(2)} mm at volume ${marker.fd.t}`;
  } else if (visible && marker.ns) {
    kind = 'ns';
    text = `non-steady state, volume ${marker.ns.t}`;
  }
  const signature = kind + text;
  if (signature !== marker.shownKind) {
    marker.shownKind = signature;
    views.marker(kind, text);
  }
}

// ---- what is written on the page -----------------------------------------------------------

function writeControls() {
  const r = run();
  for (const b of $('runs').children) b.setAttribute('aria-pressed', String(Number(b.dataset.run) === state.run));
  for (const b of $('views-buttons').children) b.setAttribute('aria-pressed', String(b.dataset.view === state.view));
  for (const b of $('kinds-buttons').children) b.setAttribute('aria-pressed', String(b.dataset.kind === state.kind));
  const time = $('time');
  time.max = String(r.n_volumes - 1);
  time.value = String(state.t);
  time.disabled = state.kind !== 'series';
  $('play-text').textContent = state.playing ? 'pause' : 'play';
  $('rate').value = String(state.rate);
  $('readout').textContent =
    state.kind === 'series'
      ? `volume ${state.t} of 0 to ${r.n_volumes - 1}, ${(state.t * r.tr_s).toFixed(1)} s`
      : `still over ${r.stills_volumes} of ${r.n_volumes} volumes`;
  const g = manifest.grid;
  for (const label of $('slices').children) {
    const input = label.querySelector('input');
    const plane = Number(input.dataset.plane);
    const index = state.slices[plane];
    input.value = String(index);
    const mm = (index - (g.shape[plane] - 1) / 2) * g.voxel_mm[plane];
    const side = ['left', 'right', 'back', 'front', 'down', 'up'][2 * plane + (mm < 0 ? 0 : 1)];
    label.querySelector('output').textContent = `${index} of ${g.shape[plane] - 1}, ${Math.abs(mm).toFixed(1)} mm ${side} of the middle`;
  }
  $('contours').checked = state.contours;
  $('crosshair').checked = state.crosshair;
  $('traces-on').checked = state.traces;
  $('diff-scale').value = String(state.diffScale);
  document.body.classList.toggle('no-traces', !state.traces);
}

function writeRun() {
  const r = run();
  const reg = manifest.registration;
  const rule = reg.rule === 'acquisition_time' ? 'the earliest acquisition time' : 'the first in name order (no acquisition times)';
  const f = r.frame;
  let note;
  if (r.anchor) {
    note = `${r.id} is the anchor run of this session (${rule}): the raw stream of every run is drawn through its stored registration to the anatomy.`;
  } else {
    note =
      `${r.id}, drawn through the registration of the anchor run ${reg.anchor}, lies ${f.mean_mm.toFixed(2)} mm on average and ` +
      `${f.max_mm.toFixed(2)} mm at most (over its brain mask) from where its own stored registration puts it: ` +
      `a rotation of ${f.rotation_deg.toFixed(2)} degrees and a shift of ${f.translation_mm.toFixed(2)} mm.`;
  }
  $('run-note').textContent = note;
  const warnings = [];
  if (f.flagged) {
    warnings.push(
      `${r.id} is FLAGGED: somewhere in the brain it lies more than ${reg.limit_mm} mm from where its own registration puts it. ` +
        'It may not share the scanner frame of the anchor run, or a registration may have failed. Its raw stream is not in register with the anatomy.',
    );
  }
  if (reg.anchor_is_outlier) {
    warnings.push(
      `Every other run of this session is flagged against the anchor ${reg.anchor} and they agree with each other: the anchor, or its stored registration, is the odd one.`,
    );
  }
  $('warning').hidden = warnings.length === 0;
  $('warning').textContent = warnings.join(' ');
}

function writeLegend() {
  const r = run();
  const win = state.kind === 'sd' ? r.sd_window : r.window;
  const what = state.kind === 'sd' ? 'temporal SD' : 'signal';
  const span = win[1] - win[0];
  let text = `Gray: ${what} ${win[0].toFixed(0)} to ${win[1].toFixed(0)} as 0 to 255.`;
  const bar = $('legend-bar');
  if (state.view === 'diff') {
    text =
      `Difference raw minus preprocessed: the colour saturates at ${state.diffScale} gray levels of 255 ` +
      `(${((state.diffScale * span) / 255).toFixed(0)} units of ${what}; window ${win[0].toFixed(0)} to ${win[1].toFixed(0)}). ` +
      'Blue: raw lower. Red: raw higher.';
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
  $('legend-text').textContent = text;
}

function measuredRate() {
  const now = performance.now();
  const recent = page.log.filter((e) => e.reason === 'play' && now - e.ms < 2000);
  return recent.length > 1 ? ((recent.length - 1) * 1000) / (recent[recent.length - 1].ms - recent[0].ms) : null;
}

function writeStatus(vols) {
  const parts = [];
  if (pending) parts.push(state.kind === 'series' ? `loading volume ${state.t}` : 'loading');
  if (vols) {
    const [i, j, k] = state.slices;
    const shape = manifest.grid.shape;
    const at = i + shape[0] * (j + shape[1] * k);
    const values = Object.entries(vols).map(([s, v]) => `${s === 'pre' ? 'preprocessed' : s} ${v[at]}`);
    parts.push(`voxel (${i}, ${j}, ${k}): ${values.join(', ')} of 255`);
  }
  if (state.playing) {
    const rate = measuredRate();
    parts.push(`${state.rate} volumes per s asked${rate ? `, ${rate.toFixed(1)} painted` : ''}`);
  }
  const s = store.stats();
  parts.push(`${formatBytes(s.decodedChunkBytes + s.stillBytes)} decoded in memory (limit ${formatBytes(s.limitBytes)})`);
  if (page.errors.length) parts.push(`ERROR: ${page.errors[page.errors.length - 1]}`);
  $('status').textContent = parts.join(' · ');
}

function fail(err) {
  page.errors.push(String(err && err.message ? err.message : err));
  writeStatus();
}

// ---- changes of state ------------------------------------------------------------------------

function setTime(t, reason) {
  const n = run().n_volumes;
  state.t = clamp(Math.round(t), 0, n - 1);
  if (state.kind !== 'series') state.kind = 'series';
  if (reason === 'seek') seek = { t: state.t, at: performance.now() };
  render(reason);
}

function setRun(index) {
  const n = manifest.runs.length;
  state.run = wrap(index, n);
  state.t = clamp(state.t, 0, run().n_volumes - 1);
  closeMarker('fd', performance.now());
  closeMarker('ns', performance.now());
  writeRun();
  writeLegend();
  traces.setRun(run(), state.fdLimit);
  store.stills(state.run).then(() => render('other'), fail);
  render('other');
}

function setView(view) {
  if (view === 'raw' || view === 'pre') lastFlip = view === 'raw' ? 'pre' : 'raw';
  state.view = view;
  writeLegend();
  views.layout();
  render('other');
}

function setKind(kind) {
  state.kind = kind;
  if (kind !== 'series') state.playing = false;
  writeLegend();
  render('other');
}

function setSlice(plane, index) {
  state.slices[plane] = clamp(index, 0, manifest.grid.shape[plane] - 1);
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
    $('help-close').focus();
  } else {
    $('help-button').focus();
  }
}

function tick(now) {
  if (state.playing && state.kind === 'series' && !state.help) {
    const step = playbackStep(nextDue, now, state.rate);
    if (step.advance) {
      const next = wrap(state.t + 1, run().n_volumes);
      const streams = streamsOf(state.view);
      if (streams.every((s) => store.volume(state.run, s, next))) {
        state.t = next;
        nextDue = step.nextDue;
        if (step.resynced) page.resyncs += 1;
        stalled = false;
        render('play');
      } else {
        // the next volume is not loaded: wait for it and do not skip it
        if (!stalled) {
          page.stalls += 1;
          stalled = true;
          views.stale(true);
          $('status').textContent = `loading volume ${next}`;
        }
        store.want(state.run, streams, next);
        nextDue = now;
      }
    }
  }
  if (marker.fd || marker.ns) showMarker(now);
  if (page.timing.allLoaded === null && page.timing.firstFrame !== null && store.idle() && contours.every((c) => c !== null || !manifest.contours)) {
    page.timing.allLoaded = now;
    writeStatus();
  }
  requestAnimationFrame(tick);
}

// ---- keys --------------------------------------------------------------------------------------

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
  // a focused button, checkbox, select or slider keeps the keys that operate it
  if ((key === ' ' || key === 'Enter') && (target instanceof HTMLButtonElement || target instanceof HTMLInputElement)) return;
  if (key.startsWith('Arrow') && (target instanceof HTMLSelectElement || (target instanceof HTMLInputElement && target.type === 'range'))) return;
  const step = event.shiftKey ? 10 : 1;
  const actions = {
    ' ': () => setPlaying(!state.playing),
    ArrowLeft: () => {
      state.playing = false;
      setTime(wrap(state.t - step, run().n_volumes), 'seek');
    },
    ArrowRight: () => {
      state.playing = false;
      setTime(wrap(state.t + step, run().n_volumes), 'seek');
    },
    Home: () => setTime(0, 'seek'),
    End: () => setTime(run().n_volumes - 1, 'seek'),
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
    r: () => setRun(state.run + 1),
    R: () => setRun(state.run - 1),
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

// ---- start -------------------------------------------------------------------------------------

async function fetchCounted(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: status ${response.status}`);
  const buffer = await response.arrayBuffer();
  page.requests += 1;
  page.fetchedBytes += buffer.byteLength;
  return buffer;
}

function buildControls() {
  manifest.runs.forEach((r, index) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.run = String(index);
    const away = r.anchor ? 'anchor' : `${r.frame.max_mm.toFixed(1)} mm`;
    b.innerHTML = `${r.id} <span class="away"></span>`;
    b.querySelector('.away').textContent = r.frame.flagged ? `${away}, flagged` : away;
    b.classList.toggle('flagged', r.frame.flagged);
    b.title = r.anchor ? 'the anchor run of the session' : 'largest distance from where its own registration puts it';
    b.addEventListener('click', () => setRun(index));
    $('runs').append(b);
  });
  for (const b of $('views-buttons').children) b.addEventListener('click', () => setView(b.dataset.view));
  for (const b of $('kinds-buttons').children) b.addEventListener('click', () => setKind(b.dataset.kind));
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
    input.max = String(manifest.grid.shape[plane] - 1);
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
      traces.setRun(run(), state.fdLimit);
      render('other');
    }
  });
  $('help-button').addEventListener('click', () => setHelp(true));
  $('help-close').addEventListener('click', () => setHelp(false));
  $('help').addEventListener('click', (event) => {
    if (event.target === $('help')) setHelp(false);
  });
  $('help-limit').textContent = String(manifest.registration.limit_mm);
  $('help-hold').textContent = String(HOLD_MS);
  $('help-memory').textContent = formatBytes(store.limitBytes);
  $('help-window').textContent = `The 8 bits: ${manifest.rules.window}; ${manifest.rules.quantise}.`;
}

function availableHeight() {
  const used = ['bar', 'run-note', 'legend', 'controls', 'status'].reduce((a, id) => a + $(id).offsetHeight, 0);
  const warn = $('warning').hidden ? 0 : $('warning').offsetHeight;
  const tr = state.traces ? $('traces').offsetHeight : 0;
  // In a narrow window the views take most of its height and the controls are reached by
  // scrolling; in a wide one everything fits, and the views never get less than 0.4 of it.
  if (window.innerWidth < 700) return 0.9 * window.innerHeight;
  return Math.max(window.innerHeight - used - warn - tr - 70, 0.4 * window.innerHeight);
}

async function loadContours() {
  if (!manifest.contours) return;
  await Promise.all(
    PLANES.map(async (name, plane) => {
      const entry = manifest.contours.planes[name];
      const packed = await fetchCounted(store.url(entry.file));
      const digest = await sha256Hex(packed);
      if (digest && digest !== entry.sha256) throw new Error(`${entry.file}: the bytes are not those the manifest lists`);
      const data = await inflate(packed);
      contours[plane] = decodeContours(data.buffer, manifest.contours.unit);
      if (shown) drawOverlay();
    }),
  );
  page.timing.contours = performance.now();
}

async function start() {
  const text = await fetchCounted(`${PACK_URL}manifest.json`);
  manifest = JSON.parse(new TextDecoder().decode(text));
  if (manifest.format !== 'func-viewer-pack' || manifest.version !== 1) throw new Error('this is not a pack this page can read');
  const version = ((await sha256Hex(text)) || 'unversioned').slice(0, 16);
  const mem = Number(params.get('mem')) || DEFAULT_LIMIT_MB;
  store = new Store(manifest, PACK_URL, version, {
    limitBytes: mem * 1024 * 1024,
    nWorkers: Number(params.get('workers')) || undefined,
  });
  store.onchunk = (r) => {
    if (pending && r === state.run) render(pendingReason);
  };
  store.onerror = fail;
  document.title = `Functional viewer ${manifest.id}`;
  $('pack-id').textContent = manifest.id;
  state.fdLimit = manifest.fd_threshold_mm;
  const shape = manifest.grid.shape;
  state.slices = shape.map((n) => Math.floor(n / 2));
  for (let plane = 0; plane < 3; plane++) {
    const [w, h] = planeSize(shape, plane);
    sliceBuffers.push([new Uint8Array(w * h), new Uint8Array(w * h)]);
  }
  views = new Views($('views'), manifest.grid, {
    pick: (plane, a, b) => {
      state.active = plane;
      state.slices[SPEC[plane].h] = a;
      state.slices[SPEC[plane].v] = b;
      render('other');
    },
    wheel: (plane, direction) => {
      state.active = plane;
      setSlice(SPEC[plane].axis, state.slices[SPEC[plane].axis] + direction);
    },
    availableHeight,
  });
  traces = new Traces($('traces'), (t) => {
    state.playing = false;
    setTime(t, 'seek');
  });
  buildControls();
  writeRun();
  writeLegend();
  views.setGroups(groupsOf(state.view), groupLabels());
  traces.setRun(run(), state.fdLimit);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('pointerup', (event) => {
    const button = event.target instanceof Element ? event.target.closest('button') : null;
    if (button && !state.help) button.blur(); // so that Space plays and does not press it again
  });
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('resize', () => {
    views.layout();
    traces.resize();
    if (shown) {
      drawOverlay();
      traces.draw(state.t);
    }
  });
  loadContours().catch(fail);
  await store.stills(state.run);
  render('other');
  requestAnimationFrame(tick);
}

// What check_player.py drives and reads. Not used by the page itself.
window.__player = {
  state,
  page,
  get manifest() {
    return manifest;
  },
  get shown() {
    return shown;
  },
  stats: () => ({ ...store.stats(), pageRequests: page.requests, pageFetchedBytes: page.fetchedBytes }),
  // Change the state (run, t, slices, view, kind, rate, playing, diffScale, contours, crosshair)
  // and resolve when the panels hold what was asked for.
  set(change) {
    if ('run' in change && change.run !== state.run) setRun(change.run);
    if ('view' in change && VIEWS.includes(change.view)) setView(change.view);
    if ('kind' in change && KINDS.includes(change.kind)) setKind(change.kind);
    if ('slices' in change) state.slices = change.slices.map((v, k) => clamp(v, 0, manifest.grid.shape[k] - 1));
    if ('diffScale' in change) setDiffScale(change.diffScale);
    if ('rate' in change) setRate(change.rate);
    for (const k of ['contours', 'crosshair']) if (k in change) state[k] = change[k];
    if ('t' in change) setTime(change.t, 'seek');
    if ('playing' in change) setPlaying(change.playing);
    else render('other');
    return window.__player.ready();
  },
  ready() {
    if (shown && shown.key === stateKey(state) && !pending) return Promise.resolve(shown);
    return new Promise((resolve) => waiters.push(resolve));
  },
  allLoaded() {
    return page.timing.allLoaded !== null;
  },
  // whether volume t of a run is decoded in memory for every stream of a view
  has(runIndex, view, t) {
    return streamsOf(view).every((stream) => store.volume(runIndex, stream, t) !== null);
  },
  idle: () => store.idle(),
};

start().catch((err) => {
  fail(err);
  $('status').textContent = `ERROR: ${String(err && err.message ? err.message : err)}`;
});
