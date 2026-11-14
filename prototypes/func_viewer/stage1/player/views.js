// The panels: per group (a stream, or the difference) three planes, each a canvas with one pixel
// per voxel and a second canvas over it for the contours and the crosshair.
import { diffToRgba, fitLayout, grayToRgba, panelSizesMm, panelToVoxel, planeSize, SPEC, voxelToPanel } from './logic.js';

const SURFACE_COLOURS = ['#FF3B30', '#FF3B30', '#FFD400', '#FFD400']; // lh, rh white; lh, rh pial
const CROSSHAIR = 'rgba(64, 224, 255, 0.55)';

export class Views {
  constructor(root, grid, handlers) {
    this.root = root;
    this.shape = grid.shape;
    this.voxel = grid.voxel_mm;
    this.sizes = panelSizesMm(this.shape, this.voxel);
    this.handlers = handlers;
    this.groups = new Map(); // key -> { element, heading, badge, panels: [{ box, gray, over, image }] }
    this.scale = 2;
  }

  // keys: the groups to show ('raw', 'pre', 'diff'); labels: key -> heading text
  setGroups(keys, labels) {
    const same = keys.length === this.groups.size && keys.every((k) => this.groups.has(k));
    if (!same) {
      this.root.replaceChildren();
      this.groups.clear();
      for (const key of keys) this.groups.set(key, this.makeGroup(key));
      this.layout();
    }
    for (const key of keys) this.groups.get(key).heading.textContent = labels[key];
  }

  makeGroup(key) {
    const element = document.createElement('section');
    element.className = 'group';
    element.dataset.group = key;
    const head = document.createElement('h2');
    const heading = document.createElement('span');
    const badge = document.createElement('span');
    badge.className = 'badge';
    head.append(heading, badge);
    const row = document.createElement('div');
    row.className = 'panels';
    element.append(head, row);
    const panels = SPEC.map((spec, plane) => {
      const box = document.createElement('div');
      box.className = 'panel';
      box.dataset.plane = String(plane);
      const [w, h] = planeSize(this.shape, plane);
      const gray = document.createElement('canvas');
      gray.className = 'gray';
      gray.width = w;
      gray.height = h;
      const over = document.createElement('canvas');
      over.className = 'over';
      box.append(gray, over);
      ['left', 'right', 'top', 'bottom'].forEach((edge, n) => {
        const letter = document.createElement('span');
        letter.className = `letter ${edge}`;
        letter.textContent = spec.letters[n];
        box.append(letter);
      });
      const pick = (event) => {
        const rect = box.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * w;
        const y = ((event.clientY - rect.top) / rect.height) * h;
        const [a, b] = panelToVoxel(plane, this.shape, x, y);
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
      const ctx = gray.getContext('2d');
      return { box, gray, over, ctx, image: ctx.createImageData(w, h) };
    });
    this.root.append(element);
    return { element, heading, badge, panels, row };
  }

  // Sizes every panel for the space the views have. Returns the scale in px per mm.
  layout() {
    const n = this.groups.size || 1;
    const availW = this.root.clientWidth;
    const availH = Math.max(this.handlers.availableHeight(), 200);
    const fit = fitLayout(this.sizes, n, availW, availH);
    this.scale = fit.scale;
    this.root.dataset.groups = fit.groups;
    const dpr = window.devicePixelRatio || 1;
    for (const group of this.groups.values()) {
      group.row.dataset.panels = fit.panels;
      group.panels.forEach((panel, plane) => {
        const w = Math.max(8, Math.round(this.sizes[plane][0] * fit.scale));
        const h = Math.max(8, Math.round(this.sizes[plane][1] * fit.scale));
        panel.box.style.width = `${w}px`;
        panel.box.style.height = `${h}px`;
        panel.over.width = Math.round(w * dpr);
        panel.over.height = Math.round(h * dpr);
      });
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
  // two planes. contours: per plane the decoded file, or null.
  overlay(slices, contours, showContours, showCrosshair, active) {
    for (const group of this.groups.values()) {
      group.panels.forEach((panel, plane) => {
        const spec = SPEC[plane];
        const [w, h] = planeSize(this.shape, plane);
        const ctx = panel.over.getContext('2d');
        const sx = panel.over.width / w;
        const sy = panel.over.height / h;
        const dpr = window.devicePixelRatio || 1;
        ctx.clearRect(0, 0, panel.over.width, panel.over.height);
        const c = contours[plane];
        if (showContours && c) {
          ctx.lineWidth = 1.25 * dpr;
          ctx.lineJoin = 'round';
          const slice = slices[spec.axis];
          for (let surface = 0; surface < c.nSurfaces; surface++) {
            const cell = slice * c.nSurfaces + surface;
            ctx.strokeStyle = SURFACE_COLOURS[surface];
            ctx.beginPath();
            for (let n = c.lineStart[cell]; n < c.lineStart[cell + 1]; n++) {
              for (let p = c.pointStart[n]; p < c.pointStart[n + 1]; p++) {
                const [x, y] = voxelToPanel(plane, this.shape, c.xy[2 * p], c.xy[2 * p + 1]);
                if (p === c.pointStart[n]) ctx.moveTo(x * sx, y * sy);
                else ctx.lineTo(x * sx, y * sy);
              }
            }
            ctx.stroke();
          }
        }
        if (showCrosshair) {
          const [x, y] = voxelToPanel(plane, this.shape, slices[spec.h], slices[spec.v]);
          ctx.strokeStyle = CROSSHAIR;
          ctx.lineWidth = dpr;
          ctx.beginPath();
          ctx.moveTo(Math.round(x * sx) + 0.5, 0);
          ctx.lineTo(Math.round(x * sx) + 0.5, panel.over.height);
          ctx.moveTo(0, Math.round(y * sy) + 0.5);
          ctx.lineTo(panel.over.width, Math.round(y * sy) + 0.5);
          ctx.stroke();
        }
        panel.box.classList.toggle('active', plane === active);
      });
    }
  }

  // The marker on the frame: text in the heading of every group and a coloured frame.
  marker(kind, text) {
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
