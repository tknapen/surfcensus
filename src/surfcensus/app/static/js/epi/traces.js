// The traces under the views, over the whole stream of the subject: framewise displacement and
// standardised DVARS of every run, bands on the volumes above the threshold and on the
// non-steady-state volumes, a dashed line at every run boundary and a thick solid one at every
// session boundary, and a cursor at the current volume in a colour that nothing else of the
// traces has (magenta, with a head at the top and the bottom). A click or a drag sets the volume.
import { aboveLimit, clamp } from './logic.js';

const LEFT = 46;
const RIGHT = 10;
export const COLOURS = {
  fd: '#4FC3F7',
  dvars: '#AED581',
  limit: '#FF5252',
  ns: '#FFB300',
  axis: '#777',
  text: '#ccc',
  run: '#c9d1db',
  session: '#ffffff',
  cursor: '#FF3DF0',
};
export const BOUNDARY_STYLE = {
  run: { width: 1, dash: [4, 3] },
  session: { width: 3, dash: [] },
  cursor: { width: 2, dash: [] },
};

export class Traces {
  constructor(canvas, onScrub) {
    this.canvas = canvas;
    this.base = document.createElement('canvas');
    this.data = null;
    this.limit = 0.5;
    const scrub = (event) => {
      if (!this.data) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const n = this.data.total;
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

  // data: { total, fd, dvars, nonSteady, boundaries: [{ at, session }] }
  setData(data, limit) {
    this.data = data;
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
    if (this.data) this.drawBase();
  }

  x(t) {
    return LEFT + ((t + 0.5) / this.data.total) * (this.w - LEFT - RIGHT);
  }

  edge(t) {
    return LEFT + (t / this.data.total) * (this.w - LEFT - RIGHT);
  }

  drawBase() {
    const ctx = this.base.getContext('2d');
    const { w, h, data } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const n = data.total;
    const plotW = w - LEFT - RIGHT;
    const band = Math.max(plotW / n, 2);
    const rowH = (h - 6) / 2;
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = COLOURS.ns;
    for (const t of data.nonSteady) ctx.fillRect(this.x(t) - band / 2, 0, band, h);
    ctx.fillStyle = COLOURS.limit;
    for (const t of aboveLimit(data.fd, this.limit)) ctx.fillRect(this.x(t) - band / 2, 0, band, h);
    ctx.globalAlpha = 1;
    for (const b of data.boundaries) {
      const style = b.session ? BOUNDARY_STYLE.session : BOUNDARY_STYLE.run;
      const x = Math.round(this.edge(b.at)) + (style.width % 2 ? 0.5 : 0);
      if (b.session) {
        // a dark edge on both sides, so that the session line stands out of the bands too
        ctx.strokeStyle = '#000';
        ctx.lineWidth = style.width + 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      ctx.strokeStyle = b.session ? COLOURS.session : COLOURS.run;
      ctx.lineWidth = style.width;
      ctx.setLineDash(style.dash);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const starts = new Set(data.boundaries.map((b) => b.at));
    const rows = [
      { values: data.fd, colour: COLOURS.fd, label: 'FD mm', floor: this.limit * 1.2 },
      { values: data.dvars, colour: COLOURS.dvars, label: 'DVARS', floor: 1.5 },
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
        if (v === null || starts.has(t)) pen = false; // no line across a run boundary
        if (v === null) return;
        if (pen) ctx.lineTo(this.x(t), y(v));
        else ctx.moveTo(this.x(t), y(v));
        pen = true;
      });
      ctx.stroke();
    });
  }

  draw(t) {
    if (!this.data) return;
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.base, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const x = this.x(t);
    ctx.strokeStyle = COLOURS.cursor;
    ctx.fillStyle = COLOURS.cursor;
    ctx.lineWidth = BOUNDARY_STYLE.cursor.width;
    ctx.setLineDash(BOUNDARY_STYLE.cursor.dash);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, this.h);
    ctx.stroke();
    for (const [y, dy] of [
      [0, 6],
      [this.h, -6],
    ]) {
      ctx.beginPath();
      ctx.moveTo(x - 5, y);
      ctx.lineTo(x + 5, y);
      ctx.lineTo(x, y + dy);
      ctx.closePath();
      ctx.fill();
    }
    this.cursorAt = x;
  }
}
