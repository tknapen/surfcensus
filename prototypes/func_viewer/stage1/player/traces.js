// The traces under the views: framewise displacement and standardised DVARS of the run on
// screen, bands on the volumes above the threshold and on the non-steady-state volumes, and a
// cursor at the current volume. A click or a drag sets the volume.
import { aboveLimit, clamp } from './logic.js';

const LEFT = 46;
const RIGHT = 10;
const COLOURS = { fd: '#4FC3F7', dvars: '#AED581', limit: '#FF5252', ns: '#FFB300', axis: '#777', text: '#ccc' };

export class Traces {
  constructor(canvas, onScrub) {
    this.canvas = canvas;
    this.base = document.createElement('canvas');
    this.run = null;
    this.limit = 0.5;
    const scrub = (event) => {
      if (!this.run) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const n = this.run.n_volumes;
      onScrub(clamp(Math.floor(((x - LEFT) / (rect.width - LEFT - RIGHT)) * n), 0, n - 1));
    };
    canvas.addEventListener('pointerdown', (event) => {
      canvas.setPointerCapture(event.pointerId);
      scrub(event);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (event.buttons & 1) scrub(event);
    });
  }

  setRun(run, limit) {
    this.run = run;
    this.limit = limit;
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(120, this.canvas.clientWidth);
    const h = Math.max(60, this.canvas.clientHeight);
    for (const c of [this.canvas, this.base]) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    this.w = w;
    this.h = h;
    this.dpr = dpr;
    if (this.run) this.drawBase();
  }

  x(t) {
    return LEFT + ((t + 0.5) / this.run.n_volumes) * (this.w - LEFT - RIGHT);
  }

  drawBase() {
    const ctx = this.base.getContext('2d');
    const { w, h, run } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const n = run.n_volumes;
    const plotW = w - LEFT - RIGHT;
    const band = Math.max(plotW / n, 2);
    const rowH = (h - 6) / 2;
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = COLOURS.ns;
    for (const t of run.non_steady) ctx.fillRect(this.x(t) - band / 2, 0, band, h);
    ctx.fillStyle = COLOURS.limit;
    for (const t of aboveLimit(run.fd, this.limit)) ctx.fillRect(this.x(t) - band / 2, 0, band, h);
    ctx.globalAlpha = 1;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const rows = [
      { values: run.fd, colour: COLOURS.fd, label: 'FD mm', floor: this.limit * 1.2 },
      { values: run.dvars, colour: COLOURS.dvars, label: 'DVARS', floor: 1.5 },
    ];
    rows.forEach((row, k) => {
      const top = k * (rowH + 6);
      const present = row.values.filter((v) => v !== null);
      const max = Math.max(row.floor, ...present) * 1.08;
      const y = (v) => top + rowH - 2 - (v / max) * (rowH - 6);
      ctx.strokeStyle = COLOURS.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(LEFT + 0.5, top + 0.5, plotW - 1, rowH - 1);
      ctx.fillStyle = COLOURS.text;
      ctx.textAlign = 'right';
      ctx.fillText(row.label, LEFT - 5, top + rowH / 2);
      ctx.fillStyle = COLOURS.axis;
      ctx.fillText(max.toFixed(max < 10 ? 1 : 0), LEFT - 5, top + 7);
      if (k === 0) {
        ctx.strokeStyle = COLOURS.limit;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(LEFT, y(this.limit));
        ctx.lineTo(LEFT + plotW, y(this.limit));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = row.colour;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      let pen = false;
      row.values.forEach((v, t) => {
        if (v === null) {
          pen = false;
          return;
        }
        if (pen) ctx.lineTo(this.x(t), y(v));
        else ctx.moveTo(this.x(t), y(v));
        pen = true;
      });
      ctx.stroke();
    });
  }

  draw(t) {
    if (!this.run) return;
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.base, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(this.x(t), 0);
    ctx.lineTo(this.x(t), this.h);
    ctx.stroke();
  }
}
