// The panels: per group (a stream, or the difference) three planes, each a box of a fixed
// size for the whole stream that holds a canvas with one pixel per voxel of the grid on screen
// and a second canvas over it for the contours and the crosshair. A grid with a smaller field
// of view sits in the middle of its box; the scale in px per mm is that of the whole stream,
// so at a boundary nothing moves but the pictures. The orientation letters lie in a margin of
// the box (PAD px on every side) outside the canvas, so that they cover no voxel.
import {
  DIFF_SURFACE_COLOURS,
  diffToRgba,
  fitLayout,
  grayToRgba,
  panelSizesMm,
  panelToVoxel,
  planeSize,
  SPEC,
  SURFACE_COLOURS,
  voxelToPanel,
} from './logic.js';

export const PAD = 13;
const EDGES = ['left', 'right', 'top', 'bottom'];

// per surface of a contour file (lh, rh white; lh, rh pial): the colour in a group's panels
function surfaceColours(group) {
  const c = group === 'diff' ? DIFF_SURFACE_COLOURS : SURFACE_COLOURS;
  return [c.white, c.white, c.pial, c.pial];
}

function crosshairColour() {
  const css = getComputedStyle(document.documentElement);
  const colour = css.getPropertyValue('--crosshair').trim() || '#33c7ff';
  const alpha = Number(css.getPropertyValue('--crosshair-alpha')) || 0.45;
  return { colour, alpha };
}

export class Views {
  // sizesMm: logic.streamPanelSizesMm of every grid of the stream
  constructor(root, sizesMm, handlers) {
    this.root = root;
    this.sizes = sizesMm;
    this.handlers = handlers;
    this.grid = null;
    this.groups = new Map(); // key -> { element, heading, badge, panels: [{ box, gray, over, ctx, image }] }
    this.scale = 2;
    this.markerState = ['', ''];
    this.crosshair = crosshairColour();
  }

  // The grid on screen (GridOut). The canvases take its shape and its letters; the boxes stay.
  setGrid(grid) {
    if (this.grid && this.grid.grid_key === grid.grid_key) return;
    this.grid = grid;
    this.overlayKey = null;
    for (const group of this.groups.values()) this.fitGroup(group);
  }

  // keys: the groups to show ('raw', 'pre', 'diff'); labels: key -> heading text
  setGroups(keys, labels) {
    const same = keys.length === this.groups.size && keys.every((k) => this.groups.has(k));
    if (!same) {
      this.root.replaceChildren();
      this.groups.clear();
      for (const key of keys) this.groups.set(key, this.makeGroup(key));
      this.layout();
      this.marker(...this.markerState); // the new groups carry the marker that is up
    }
    for (const key of keys) this.groups.get(key).heading.textContent = labels[key];
  }

  makeGroup(key) {
    const element = document.createElement('section');
    element.className = 'epi-group';
    element.dataset.group = key;
    const head = document.createElement('h2');
    const heading = document.createElement('span');
    const badge = document.createElement('span');
    badge.className = 'badge';
    head.append(heading, badge);
    const row = document.createElement('div');
    row.className = 'epi-panels';
    element.append(head, row);
    const panels = SPEC.map((spec, plane) => {
      const box = document.createElement('div');
      box.className = 'epi-panel';
      box.dataset.plane = String(plane);
      const gray = document.createElement('canvas');
      gray.className = 'gray';
      const over = document.createElement('canvas');
      over.className = 'over';
      box.append(gray, over);
      const letters = EDGES.map((edge) => {
        const letter = document.createElement('span');
        letter.className = `epi-letter ${edge}`;
        box.append(letter);
        return letter;
      });
      const pick = (event) => {
        if (!this.grid) return;
        const [w, h] = planeSize(this.grid.shape, plane);
        const rect = gray.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * w;
        const y = ((event.clientY - rect.top) / rect.height) * h;
        const [a, b] = panelToVoxel(plane, this.grid.shape, x, y);
        this.handlers.pick(plane, a, b);
      };
      box.addEventListener('pointerdown', (event) => {
        box.setPointerCapture(event.pointerId);
        pick(event);
      });
      box.addEventListener('pointermove', (event) => {
        if (event.buttons & 1) pick(event);
      });
      box.addEventListener(
        'wheel',
        (event) => {
          event.preventDefault();
          this.handlers.wheel(plane, event.deltaY > 0 ? -1 : 1);
        },
        { passive: false },
      );
      row.append(box);
      return { box, gray, over, letters, ctx: gray.getContext('2d'), image: null };
    });
    this.root.append(element);
    const group = { element, heading, badge, panels, row };
    this.fitGroup(group);
    return group;
  }

  // The canvases of a group for the grid on screen: their pixels, their place in the boxes.
  fitGroup(group) {
    if (!this.grid) return;
    const { shape, voxel_mm: voxel, edge_letters: letters } = this.grid;
    const mm = panelSizesMm(shape, voxel);
    const dpr = window.devicePixelRatio || 1;
    group.panels.forEach((panel, plane) => {
      const [w, h] = planeSize(shape, plane);
      if (panel.gray.width !== w || panel.gray.height !== h) {
        panel.gray.width = w;
        panel.gray.height = h;
        panel.image = null;
      }
      if (!panel.image) panel.image = panel.ctx.createImageData(w, h);
      const innerW = Math.max(8, Math.round(this.sizes[plane][0] * this.scale));
      const innerH = Math.max(8, Math.round(this.sizes[plane][1] * this.scale));
      const cw = Math.min(innerW, Math.max(4, Math.round(mm[plane][0] * this.scale)));
      const ch = Math.min(innerH, Math.max(4, Math.round(mm[plane][1] * this.scale)));
      panel.box.style.width = `${innerW + 2 * PAD}px`;
      panel.box.style.height = `${innerH + 2 * PAD}px`;
      for (const canvas of [panel.gray, panel.over]) {
        canvas.style.width = `${cw}px`;
        canvas.style.height = `${ch}px`;
        canvas.style.left = `${PAD + Math.floor((innerW - cw) / 2)}px`;
        canvas.style.top = `${PAD + Math.floor((innerH - ch) / 2)}px`;
      }
      const ow = Math.round(cw * dpr);
      const oh = Math.round(ch * dpr);
      if (panel.over.width !== ow || panel.over.height !== oh) {
        panel.over.width = ow;
        panel.over.height = oh;
      }
      const edge = letters[SPEC[plane].name];
      panel.letters.forEach((el, n) => {
        el.textContent = edge ? edge[n] : '';
      });
    });
  }

  // Sizes every box for the space the views have. Returns the scale in px per mm.
  layout() {
    this.overlayKey = null; // resizing a canvas clears it
    const n = this.groups.size || 1;
    const availW = this.root.clientWidth;
    const availH = Math.max(this.handlers.availableHeight(), 200);
    const fit = fitLayout(this.sizes, n, availW, availH, 8, 22, 4, PAD);
    this.scale = fit.scale;
    this.root.dataset.groups = fit.groups;
    for (const group of this.groups.values()) {
      group.row.dataset.panels = fit.panels;
      this.fitGroup(group);
    }
    return fit.scale;
  }

  paintGray(key, plane, slice) {
    const panel = this.groups.get(key).panels[plane];
    grayToRgba(slice, panel.image.data);
    panel.ctx.putImageData(panel.image, 0, 0);
  }

  paintDiff(key, plane, a, b, lut) {
    const panel = this.groups.get(key).panels[plane];
    diffToRgba(a, b, lut, panel.image.data);
    panel.ctx.putImageData(panel.image, 0, 0);
  }

  // The layer over the voxels: contours of the slices on screen and the crosshair of the other
  // two planes. contours: per plane the decoded file of the grid on screen, or null. Drawn again
  // only when the slices, the grid, the switches, the layout or the loaded contours change.
  overlay(slices, contours, showContours, showCrosshair, active, contourKey = '') {
    const shape = this.grid.shape;
    const groups = [...this.groups.keys()].join('+');
    const key = [this.grid.grid_key, contourKey, slices.join(','), contours.map((c) => (c ? 1 : 0)).join(''), showContours, showCrosshair, active, groups].join('|');
    if (key === this.overlayKey) return;
    this.overlayKey = key;
    const dpr = window.devicePixelRatio || 1;
    for (const [groupKey, group] of this.groups.entries()) {
      const colours = surfaceColours(groupKey);
      const outlined = groupKey === 'diff';
      group.panels.forEach((panel, plane) => {
        const spec = SPEC[plane];
        const [w, h] = planeSize(shape, plane);
        const ctx = panel.over.getContext('2d');
        const sx = panel.over.width / w;
        const sy = panel.over.height / h;
        ctx.clearRect(0, 0, panel.over.width, panel.over.height);
        const c = contours[plane];
        if (showContours && c) {
          ctx.lineJoin = 'round';
          const slice = slices[spec.axis];
          const path = (surface) => {
            const cell = slice * c.nSurfaces + surface;
            ctx.beginPath();
            for (let n = c.lineStart[cell]; n < c.lineStart[cell + 1]; n++) {
              for (let p = c.pointStart[n]; p < c.pointStart[n + 1]; p++) {
                const [x, y] = voxelToPanel(plane, shape, c.xy[2 * p], c.xy[2 * p + 1]);
                if (p === c.pointStart[n]) ctx.moveTo(x * sx, y * sy);
                else ctx.lineTo(x * sx, y * sy);
              }
            }
          };
          if (outlined) {
            // a dark outline under every line first, so that no line lies on a colour of the
            // difference without an edge
            ctx.strokeStyle = DIFF_SURFACE_COLOURS.outline;
            ctx.lineWidth = 3.5 * dpr;
            for (let surface = 0; surface < c.nSurfaces; surface++) {
              path(surface);
              ctx.stroke();
            }
          }
          ctx.lineWidth = (outlined ? 1.5 : 1.25) * dpr;
          for (let surface = 0; surface < c.nSurfaces; surface++) {
            ctx.strokeStyle = colours[surface % colours.length];
            path(surface);
            ctx.stroke();
          }
        }
        if (showCrosshair) {
          const [x, y] = voxelToPanel(plane, shape, slices[spec.h], slices[spec.v]);
          ctx.strokeStyle = this.crosshair.colour;
          ctx.globalAlpha = this.crosshair.alpha;
          ctx.lineWidth = dpr;
          ctx.beginPath();
          ctx.moveTo(Math.round(x * sx) + 0.5, 0);
          ctx.lineTo(Math.round(x * sx) + 0.5, panel.over.height);
          ctx.moveTo(0, Math.round(y * sy) + 0.5);
          ctx.lineTo(panel.over.width, Math.round(y * sy) + 0.5);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        panel.box.classList.toggle('active', plane === active);
      });
    }
  }

  // The marker on the frame: text in the heading of every group and a coloured frame.
  marker(kind, text) {
    this.markerState = [kind, text];
    for (const group of this.groups.values()) {
      group.element.classList.toggle('marker-fd', kind === 'fd');
      group.element.classList.toggle('marker-ns', kind === 'ns');
      group.badge.textContent = text;
    }
  }

  stale(on) {
    this.root.classList.toggle('stale', on);
  }
}
