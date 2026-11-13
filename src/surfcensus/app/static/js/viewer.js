// The three panels of one subject: per panel a grayscale tile and a contour tile, stacked,
// under an SVG layer with the crosshairs (and the marks, drawn by marks.js) and the edge
// letters. Plays both playback modes from the frame tables of the `render` object; scrub,
// speed, 2x zoom, contours on and off. Loads every tile of the subject once and keeps the
// image elements, so no tile is requested twice, and loads the tiles of the subjects that
// come next ahead of time.

import { PANEL_ORDER, clickToPixel, crosshairCell, crosshairPixels, tileUrl } from "./coords.js";

export const SPEEDS = [0.25, 0.5, 1, 2, 4];
export const DEFAULT_SPEED = 2; // position in SPEEDS
export const JUMP_FRAMES = 10;
const SVG = "http://www.w3.org/2000/svg";
const LAYERS = ["gray", "contours"];

export function wrapFrame(frame, count) {
  return ((frame % count) + count) % count;
}

// The frame of a table that shows slices nearest the ones on screen: the first with the
// smallest sum of slice differences. Used when the playback mode changes.
export function nearestFrame(frames, shown) {
  let best = 0;
  let bestDistance = Infinity;
  frames.forEach((frame, position) => {
    const distance = frame.reduce((sum, slice, panel) => sum + Math.abs(slice - shown[panel]), 0);
    if (distance < bestDistance) {
      best = position;
      bestDistance = distance;
    }
  });
  return best;
}

// Every tile of one subject, {url, layer, panel, index}, the tiles of `first` (a frame) in
// front. `tiles` are the two name patterns and `nSlices` the tile count of each panel.
export function subjectTiles(tiles, nSlices, mediaBase, renderKey, first = null) {
  const out = [];
  const add = (panel, index) => {
    for (const layer of LAYERS) {
      const url = tileUrl(mediaBase, tiles[layer], panel, index, renderKey);
      out.push({ url, layer, panel, index });
    }
  };
  if (first) PANEL_ORDER.forEach((panel, position) => add(panel, first[position]));
  for (const panel of PANEL_ORDER) {
    for (let index = 0; index < nSlices[panel]; index += 1) {
      if (!first || first[PANEL_ORDER.indexOf(panel)] !== index) add(panel, index);
    }
  }
  return out;
}

export function sliceCounts(render) {
  const counts = {};
  for (const name of PANEL_ORDER) counts[name] = render.panels[name].slices.length;
  return counts;
}

// Image elements by URL. An element is made, and its file requested, once; while the cache
// holds it the viewer shows that element itself, so the browser has no reason to ask again.
export class TileCache {
  constructor() {
    this.entries = new Map();
    this.requested = 0;
  }

  entry(url, layer) {
    let found = this.entries.get(url);
    if (!found) {
      const image = new Image();
      image.className = `tile tile-${layer}`;
      image.alt = "";
      image.draggable = false;
      image.decoding = "sync";
      const settled = new Promise((resolve) => {
        image.addEventListener("load", () => resolve(true), { once: true });
        image.addEventListener("error", () => resolve(false), { once: true });
      });
      image.src = url;
      this.requested += 1;
      found = { image, settled };
      this.entries.set(url, found);
    }
    return found;
  }

  // Resolves with {loaded, failed} when every tile has loaded or failed.
  async load(tiles, onProgress = null) {
    let loaded = 0;
    let failed = 0;
    await Promise.all(
      tiles.map(async (tile) => {
        if (await this.entry(tile.url, tile.layer).settled) loaded += 1;
        else failed += 1;
        if (onProgress) onProgress(loaded + failed, tiles.length);
      }),
    );
    return { loaded, failed };
  }

  // Drop every element whose URL is not in `urls`.
  keepOnly(urls) {
    for (const url of [...this.entries.keys()]) {
      if (!urls.has(url)) this.entries.delete(url);
    }
  }
}

function svg(name, attributes = {}) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function keyHint(key) {
  return html("kbd", "", key);
}

export class Viewer {
  // `root` is an empty element. Callbacks: onFrame(frame, mode, slices) after every change
  // of what is shown; onState(name, value) for play, pause, mode, speed, zoom, contours;
  // onClick({panel, slice, px, py, frame, mode, target}) for a click on a tile.
  constructor(root, { onFrame = null, onState = null, onClick = null } = {}) {
    this.root = root;
    this.onFrame = onFrame;
    this.onState = onState;
    this.onClick = onClick;
    this.cache = new TileCache();
    this.render = null;
    this.mode = "sync";
    this.frame = 0;
    this.playing = false;
    this.speed = DEFAULT_SPEED;
    this.zoomed = false;
    this.contoursHidden = false;
    this.held = false;
    this.resume = false;
    this.token = 0;
    this.current = new Set();
    this.aheadUrls = new Set();
    this.lastTick = null;
    this.raf = null;
    this.panels = {};
    this.build();
    this.reflect();
  }

  build() {
    this.root.classList.add("viewer");
    const panels = html("div", "panels");
    for (const name of PANEL_ORDER) {
      const figure = html("figure", "panel");
      figure.dataset.panel = name;
      const stage = html("div", "stage");
      const zoomable = html("div", "zoomable");
      const slots = {};
      for (const layer of LAYERS) {
        slots[layer] = html("img", `tile tile-${layer}`);
        slots[layer].alt = "";
        zoomable.append(slots[layer]);
      }
      const overlay = svg("svg", { class: "panel-layer", preserveAspectRatio: "none" });
      const crosshair = svg("g", { class: "crosshair" });
      const vertical = svg("line", { class: "crosshair-v" });
      const horizontal = svg("line", { class: "crosshair-h" });
      crosshair.append(vertical, horizontal);
      const marks = svg("g", { class: "marks" });
      overlay.append(crosshair, marks);
      zoomable.append(overlay);
      const letters = html("div", "letters");
      const edges = {};
      for (const edge of ["left", "right", "top", "bottom"]) {
        edges[edge] = html("span", `edge edge-${edge}`);
        letters.append(edges[edge]);
      }
      stage.append(zoomable, letters);
      stage.addEventListener("click", (event) => this.clicked(name, event));
      figure.append(stage, html("figcaption", "", name));
      panels.append(figure);
      this.panels[name] = {
        figure,
        stage,
        zoomable,
        slots,
        overlay,
        vertical,
        horizontal,
        marks,
        edges,
      };
    }

    const transport = html("div", "transport");
    this.playButton = html("button", "control");
    this.playButton.id = "play";
    this.playButton.type = "button";
    this.playButton.addEventListener("click", () => this.toggle());
    this.slider = html("input");
    this.slider.id = "frame";
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = "0";
    this.slider.step = "1";
    this.slider.value = "0";
    this.slider.setAttribute("aria-label", "frame");
    this.slider.addEventListener("pointerdown", () => this.pause());
    this.slider.addEventListener("input", () => {
      this.pause();
      this.seek(Number(this.slider.value));
    });
    this.frameLabel = html("output", "frame-label");
    this.frameLabel.id = "frame-label";
    this.modeButton = html("button", "control");
    this.modeButton.id = "mode";
    this.modeButton.type = "button";
    this.modeButton.addEventListener("click", () => this.toggleMode());
    this.slowerButton = html("button", "control");
    this.slowerButton.id = "slower";
    this.slowerButton.type = "button";
    this.slowerButton.append(keyHint("["), " slower");
    this.slowerButton.addEventListener("click", () => this.changeSpeed(-1));
    this.speedLabel = html("output", "speed-label");
    this.speedLabel.id = "speed";
    this.fasterButton = html("button", "control");
    this.fasterButton.id = "faster";
    this.fasterButton.type = "button";
    this.fasterButton.append("faster ", keyHint("]"));
    this.fasterButton.addEventListener("click", () => this.changeSpeed(1));
    this.zoomButton = html("button", "control");
    this.zoomButton.id = "zoom";
    this.zoomButton.type = "button";
    this.zoomButton.addEventListener("click", () => this.toggleZoom());
    this.loading = html("span", "loading");
    this.loading.id = "loading";
    transport.append(
      this.playButton,
      this.slider,
      this.frameLabel,
      this.modeButton,
      this.slowerButton,
      this.speedLabel,
      this.fasterButton,
      this.zoomButton,
      this.loading,
    );
    // a control that kept the focus would take Space and the arrow keys for itself
    for (const control of transport.querySelectorAll("button")) {
      control.addEventListener("click", () => control.blur());
    }
    this.slider.addEventListener("change", () => this.slider.blur());
    this.root.append(panels, transport);
    this.panelsBox = panels;
    new ResizeObserver(() => this.fit()).observe(panels);
    this.fit();
  }

  // The side of a panel on screen: a third of the width of the box of the panels, or its
  // height less the caption, whichever is smaller.
  fit() {
    const style = getComputedStyle(this.panelsBox);
    const gap = Number.parseFloat(style.columnGap) || 0;
    const caption = 18;
    const side = Math.floor(
      Math.min((this.panelsBox.clientWidth - 2 * gap) / 3, this.panelsBox.clientHeight - caption),
    );
    this.panelsBox.style.setProperty("--side", `${Math.max(0, side)}px`);
  }

  // --- what is shown --------------------------------------------------------------------

  get frames() {
    return this.render ? this.render.modes[this.mode].frames : [[0, 0, 0]];
  }

  get slices() {
    return this.frames[this.frame];
  }

  slice(panelName) {
    return this.slices[PANEL_ORDER.indexOf(panelName)];
  }

  // Show a subject: `subject` has `render`, `media_base` and `render_key` (an assignment, or
  // the subject of the review screen). The first frame shows as soon as its tiles are there.
  // Resolves with {loaded, failed, current}: `current` is false when another subject was
  // shown in the meantime.
  async show(subject, { autoplay = true, mode = "sync" } = {}) {
    const token = (this.token += 1);
    this.pause(false);
    this.render = subject.render;
    this.mediaBase = subject.media_base;
    this.renderKey = subject.render_key;
    this.mode = mode;
    this.frame = 0;
    for (const name of PANEL_ORDER) {
      const panel = this.render.panels[name];
      const parts = this.panels[name];
      parts.overlay.setAttribute("viewBox", `0 0 ${panel.width} ${panel.height}`);
      parts.vertical.setAttribute("y1", "0");
      parts.vertical.setAttribute("y2", String(panel.height));
      parts.horizontal.setAttribute("x1", "0");
      parts.horizontal.setAttribute("x2", String(panel.width));
      for (const edge of Object.keys(parts.edges)) {
        parts.edges[edge].textContent = panel.edges[edge];
      }
    }
    const config = this.render.config;
    this.root.style.setProperty("--crosshair", config.crosshair_colour);
    this.root.style.setProperty("--crosshair-alpha", String(config.crosshair_alpha));
    const tiles = subjectTiles(
      this.render.tiles,
      sliceCounts(this.render),
      this.mediaBase,
      this.renderKey,
      this.slices,
    );
    this.current = new Set(tiles.map((tile) => tile.url));
    this.evict();
    this.root.dataset.ready = "false";
    this.loading.textContent = "loading";
    this.draw();
    const result = await this.cache.load(tiles, (done, total) => {
      if (token === this.token) this.loading.textContent = `loading ${done} / ${total}`;
    });
    if (token !== this.token) return { ...result, current: false };
    this.loading.textContent = result.failed
      ? `${result.failed} of ${tiles.length} tiles did not load; reload the page`
      : "";
    this.root.dataset.ready = "true";
    if (autoplay && !result.failed) this.play();
    return { ...result, current: true };
  }

  // Load the tiles of the subjects that come next (the `prefetch` of a NextOut).
  prefetch(list) {
    this.aheadUrls = new Set();
    for (const ahead of list) {
      const tiles = subjectTiles(ahead.tiles, ahead.n_slices, ahead.media_base, ahead.render_key);
      for (const tile of tiles) {
        this.aheadUrls.add(tile.url);
        this.cache.entry(tile.url, tile.layer);
      }
    }
    this.evict();
  }

  evict() {
    this.cache.keepOnly(new Set([...this.current, ...this.aheadUrls]));
  }

  draw() {
    if (!this.render) return;
    const slices = this.slices;
    const cross = crosshairPixels(this.render, slices);
    PANEL_ORDER.forEach((name, position) => {
      const parts = this.panels[name];
      for (const layer of LAYERS) {
        const url = tileUrl(
          this.mediaBase,
          this.render.tiles[layer],
          name,
          slices[position],
          this.renderKey,
        );
        const image = this.cache.entry(url, layer).image;
        if (parts.slots[layer] !== image) {
          parts.slots[layer].replaceWith(image);
          parts.slots[layer] = image;
        }
      }
      // the line covers the pixel column (row) that holds the point, as in the poster
      const x = crosshairCell(cross[name][0]) + 0.5;
      const y = crosshairCell(cross[name][1]) + 0.5;
      parts.vertical.setAttribute("x1", String(x));
      parts.vertical.setAttribute("x2", String(x));
      parts.horizontal.setAttribute("y1", String(y));
      parts.horizontal.setAttribute("y2", String(y));
      parts.figure.dataset.slice = String(slices[position]);
    });
    this.reflect();
    if (this.onFrame) this.onFrame(this.frame, this.mode, slices);
  }

  reflect() {
    const count = this.frames.length;
    this.slider.max = String(count - 1);
    this.slider.value = String(this.frame);
    this.frameLabel.textContent = `frame ${this.frame + 1} / ${count}`;
    this.playButton.replaceChildren(keyHint("Space"), this.playing ? " pause" : " play");
    this.modeButton.replaceChildren(
      keyHint("m"),
      this.mode === "sync" ? " synchronized" : " one at a time",
    );
    this.speedLabel.textContent = `${SPEEDS[this.speed]}x`;
    this.zoomButton.replaceChildren(keyHint("z"), this.zoomed ? " zoom 2x" : " zoom 1x");
    const data = this.root.dataset;
    data.playing = String(this.playing);
    data.mode = this.mode;
    data.frame = String(this.frame);
    data.speed = String(SPEEDS[this.speed]);
    data.zoom = this.zoomed ? "2" : "1";
    data.contours = this.contoursHidden ? "hidden" : "shown";
  }

  // --- playback -------------------------------------------------------------------------

  seek(frame) {
    this.frame = wrapFrame(Math.round(frame), this.frames.length);
    this.draw();
  }

  step(by) {
    this.pause();
    this.seek(this.frame + by);
  }

  // While held nothing plays (a dialog covers the panels). What was playing, and what is
  // asked to play meanwhile, plays again at the release.
  hold() {
    if (this.held) return;
    this.resume = this.playing;
    this.pause();
    this.held = true;
  }

  release() {
    if (!this.held) return;
    this.held = false;
    if (this.resume) this.play();
    this.resume = false;
  }

  play() {
    if (this.held) {
      this.resume = true;
      return;
    }
    if (this.playing || !this.render) return;
    this.playing = true;
    this.lastTick = null;
    this.reflect();
    if (this.onState) this.onState("play", { frame: this.frame });
    const tick = (now) => {
      if (!this.playing) return;
      const interval = 1000 / (this.render.config.fps * SPEEDS[this.speed]);
      if (this.lastTick === null) this.lastTick = now;
      const due = Math.floor((now - this.lastTick) / interval);
      if (due >= 1) {
        // a stalled page shows the next frame; it does not jump over the ones it missed
        this.lastTick = due > 3 ? now : this.lastTick + interval;
        this.seek(this.frame + 1);
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  pause(report = true) {
    if (!this.playing) return;
    this.playing = false;
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
    this.reflect();
    if (report && this.onState) this.onState("pause", { frame: this.frame });
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  toggleMode() {
    if (!this.render) return;
    const shown = this.slices;
    this.mode = this.mode === "sync" ? "seq" : "sync";
    this.frame = nearestFrame(this.frames, shown);
    this.lastTick = null;
    if (this.onState) this.onState("mode", { mode: this.mode });
    this.draw();
  }

  changeSpeed(by) {
    const speed = Math.min(SPEEDS.length - 1, Math.max(0, this.speed + by));
    if (speed === this.speed) return;
    this.speed = speed;
    this.lastTick = null;
    this.reflect();
    if (this.onState) this.onState("speed", { speed: SPEEDS[this.speed] });
  }

  toggleZoom() {
    this.zoomed = !this.zoomed;
    if (this.onState) this.onState("zoom", { zoom: this.zoomed ? 2 : 1 });
    // what is drawn over the tiles keeps its size on screen
    this.draw();
  }

  setContoursHidden(hidden) {
    if (hidden === this.contoursHidden) return;
    this.contoursHidden = hidden;
    this.reflect();
    if (this.onState) this.onState("contours", { hidden });
  }

  // --- clicks ---------------------------------------------------------------------------

  clicked(name, event) {
    if (!this.render || !this.onClick) return;
    const panel = this.render.panels[name];
    // the box of the zoomed layer, so the arithmetic is the same at 1x and at 2x
    const box = this.panels[name].zoomable.getBoundingClientRect();
    const [px, py] = clickToPixel(
      panel,
      event.clientX - box.left,
      event.clientY - box.top,
      box.width,
      box.height,
    );
    this.onClick({
      panel: name,
      slice: this.slice(name),
      px,
      py,
      frame: this.frame,
      mode: this.mode,
      target: event.target,
    });
  }
}
