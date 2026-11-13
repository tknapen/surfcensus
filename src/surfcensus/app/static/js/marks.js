// Click marks: the marks of the subject on screen, and an SVG layer per panel that draws
// them. A mark is a point on one tile (the fields of MarkIn); as a 3D point it shows in
// every panel whose current slice passes within `mark_show_mm` of it, fainter with distance.

import { PANEL_ORDER, pixelToVoxel, sliceDistanceMm, tkrToVoxel, voxelToPixel, voxelToTkr } from "./coords.js";

const SVG = "http://www.w3.org/2000/svg";
export const MIN_OPACITY = 0.2;

// 1 on the slice of the point, falling linearly to MIN_OPACITY at `showMm`.
export function markOpacity(distanceMm, showMm) {
  if (showMm <= 0) return 1;
  return 1 - (1 - MIN_OPACITY) * Math.min(1, distanceMm / showMm);
}

// Voxel and tkr position of a mark, from the tile it was placed on.
export function markPosition(render, mark) {
  const vox = pixelToVoxel(render.panels[mark.panel], mark.slice, mark.px, mark.py);
  return { vox, tkr: voxelToTkr(render, vox) };
}

// Where a 3D point shows in a panel at one of its slices: {px, py, distance, opacity}, or
// null when the slice is farther than `showMm` from the point.
export function pointInPanel(render, vox, tkr, panelName, slicePos, showMm) {
  const panel = render.panels[panelName];
  const distance = sliceDistanceMm(panel, slicePos, tkr);
  // a point on the slice itself (a mark on its own tile) is on it whatever the rounding
  if (distance > showMm + 1e-9) return null;
  const [px, py] = voxelToPixel(panel, vox);
  return { px, py, distance, opacity: markOpacity(distance, showMm) };
}

export function markInPanel(render, mark, panelName, slicePos, showMm) {
  const { vox, tkr } = markPosition(render, mark);
  return pointInPanel(render, vox, tkr, panelName, slicePos, showMm);
}

// The marks of one subject, in the order they were placed.
export class MarkStore {
  constructor() {
    this.marks = [];
    this.selected = null;
    this.placed = [];
    this.counter = 0;
  }

  // Start over with the marks of an earlier submission (PreviousOut.marks) or of a draft.
  reset(marks = []) {
    this.marks = marks.map((mark) => ({
      client_id: mark.client_id,
      class: mark.class,
      panel: mark.panel,
      slice: mark.slice,
      px: mark.px,
      py: mark.py,
      frame: mark.frame ?? null,
      mode: mark.mode ?? null,
      ts: mark.ts ?? null,
    }));
    this.selected = null;
    this.placed = [];
    // new marks are n1, n2, ...; the server names stored ones m<mark_id>
    this.counter = 0;
    for (const mark of this.marks) {
      const match = /^n(\d+)$/.exec(mark.client_id);
      if (match) this.counter = Math.max(this.counter, Number(match[1]));
    }
  }

  add(markClass, click, ts) {
    this.counter += 1;
    const mark = /* shape: MarkIn */ {
      client_id: `n${this.counter}`,
      class: markClass,
      panel: click.panel,
      slice: click.slice,
      px: click.px,
      py: click.py,
      frame: click.frame,
      mode: click.mode,
      ts,
    };
    this.marks.push(mark);
    this.placed.push(mark.client_id);
    this.selected = mark.client_id;
    return mark;
  }

  get(clientId) {
    return this.marks.find((mark) => mark.client_id === clientId) ?? null;
  }

  remove(clientId) {
    const mark = this.get(clientId);
    if (!mark) return null;
    this.marks = this.marks.filter((other) => other !== mark);
    this.placed = this.placed.filter((id) => id !== clientId);
    if (this.selected === clientId) this.selected = null;
    return mark;
  }

  // Remove the mark that was placed last on this page and is still there.
  undo() {
    const last = this.placed[this.placed.length - 1];
    return last === undefined ? null : this.remove(last);
  }

  select(clientId) {
    this.selected = this.get(clientId) ? clientId : null;
  }

  count(markClass) {
    return this.marks.filter((mark) => mark.class === markClass).length;
  }

  // The marks as a rating posts them.
  posted() {
    return this.marks.map((mark) => ({ ...mark }));
  }
}

function svg(name, attributes = {}) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

// Draws marks, and on the review screen cluster centroids, into the `marks` group of each
// panel of a Viewer. `getMarks()` gives objects with the fields of MarkIn (and, on the
// review screen, `rater`); `getSelected()` the client_id of the selected one.
export class MarkLayer {
  constructor(viewer, { showMm, getMarks, getSelected = () => null, getPoints = () => [] }) {
    this.viewer = viewer;
    this.showMm = showMm;
    this.getMarks = getMarks;
    this.getSelected = getSelected;
    this.getPoints = getPoints;
    this.hidden = false;
  }

  setHidden(hidden) {
    this.hidden = hidden;
    this.viewer.root.dataset.marks = hidden ? "hidden" : "shown";
    this.draw();
  }

  draw() {
    const render = this.viewer.render;
    if (!render) return;
    const slices = this.viewer.slices;
    const selected = this.getSelected();
    const marks = this.hidden ? [] : this.getMarks();
    const points = this.hidden ? [] : this.getPoints();
    PANEL_ORDER.forEach((name, position) => {
      const group = this.viewer.panels[name].marks;
      // a ring of 1/64 of the tile on screen, at 1x and at 2x
      const radius = render.panels[name].width / 64 / (this.viewer.zoomed ? 2 : 1);
      const drawn = [];
      for (const point of points) {
        const vox = tkrToVoxel(render, point.tkr);
        const at = pointInPanel(render, vox, point.tkr, name, slices[position], this.showMm);
        if (at) drawn.push(this.centroid(point, at, radius));
      }
      for (const mark of marks) {
        const at = markInPanel(render, mark, name, slices[position], this.showMm);
        if (at) drawn.push(this.mark(mark, at, radius, name, mark.client_id === selected));
      }
      group.replaceChildren(...drawn);
    });
  }

  mark(mark, at, radius, panelName, selected) {
    const group = svg("g", {
      class: selected ? "mark mark-selected" : "mark",
      transform: `translate(${at.px} ${at.py})`,
      opacity: at.opacity.toFixed(3),
    });
    group.dataset.class = mark.class;
    group.dataset.clientId = mark.client_id ?? "";
    group.dataset.own = String(mark.panel === panelName);
    group.dataset.distance = at.distance.toFixed(3);
    group.append(
      svg("circle", { class: "mark-hit", r: radius * 1.8 }),
      svg("circle", { class: "mark-halo", r: radius }),
      svg("circle", { class: "mark-ring", r: radius }),
    );
    if (selected) group.append(svg("circle", { class: "mark-chosen", r: radius * 1.7 }));
    const label = svg("text", {
      class: "mark-label",
      x: radius * 1.3,
      y: -radius * 1.3,
      "font-size": radius * 2.4,
    });
    label.textContent = mark.rater ? `${mark.class} ${mark.rater}` : mark.class;
    group.append(label);
    return group;
  }

  centroid(point, at, radius) {
    const group = svg("g", {
      class: "centroid",
      transform: `translate(${at.px} ${at.py})`,
      opacity: at.opacity.toFixed(3),
    });
    const r = radius * 1.3;
    group.append(
      svg("path", { class: "centroid-halo", d: `M 0 ${-r} L ${r} 0 L 0 ${r} L ${-r} 0 Z` }),
      svg("path", { class: "centroid-shape", d: `M 0 ${-r} L ${r} 0 L 0 ${r} L ${-r} 0 Z` }),
    );
    const label = svg("text", {
      class: "mark-label",
      x: radius * 1.6,
      y: radius * 2.6,
      "font-size": radius * 2.4,
    });
    label.textContent = point.label;
    group.append(label);
    return group;
  }
}
