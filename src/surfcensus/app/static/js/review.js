// The review page, unblinded: a sortable table of the subjects with their scores and final
// labels, a scatter of the composite score against the final label, and one subject in the
// viewer with every rater's marks and the centroids of its outlier clusters. The server
// answers 403 while the rater still has subjects to rate; the page then says so.

import * as api from "./api.js";
import { PANEL_ORDER, nearestSlice, tkrToVoxel } from "./coords.js";
import { VIEWER_KEYS, buildKeyMap, installKeys } from "./keys.js";
import { MarkLayer } from "./marks.js";
import { Viewer, nearestFrame } from "./viewer.js";

const SVG = "http://www.w3.org/2000/svg";
const PAGE_ROWS = 200;
// What the tables of one subject show: fields of ReviewRating, ReviewMark and ClusterOut.
const RATING_COLUMNS = ["rater", "kind", "A", "B", "C", "unratable_reason", "other", "note", "revision", "server_ts", "ms_on_subject", "render_current"];
const MARK_COLUMNS = ["rater", "class", "panel", "slice", "hemi", "pial_vertex", "pial_dist_mm", "fsavg6_vertex", "box_dist_mm"];
const CLUSTER_COLUMNS = ["reference_id", "hemi", "region", "rank", "area_mm2", "peak_z"];
const byId = (id) => document.getElementById(id);

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

function svg(name, attributes = {}, text = "") {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== "") element.textContent = text;
  return element;
}

export function cellText(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number" && !Number.isInteger(value)) return value.toPrecision(4);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

// The label a subject is plotted at: the worse of the final A and B (the primary outcome
// of the analysis plan), "unratable", or "not rated".
export function finalLevel(row) {
  if (row.unratable_reason) return "unratable";
  const values = [row.final_A, row.final_B].filter((value) => value !== null && value !== undefined);
  return values.length ? Math.max(...values) : "not rated";
}

// The frame of a table nearest the slices `wanted`; with `own` = [panel name, slice], among
// the frames that show that slice in that panel when there is one.
export function frameShowing(frames, wanted, own = null) {
  if (own) {
    const position = PANEL_ORDER.indexOf(own[0]);
    const showing = frames
      .map((frame, number) => [frame, number])
      .filter(([frame]) => frame[position] === own[1]);
    if (showing.length) {
      const nearest = nearestFrame(showing.map(([frame]) => frame), wanted);
      return showing[nearest][1];
    }
  }
  return nearestFrame(frames, wanted);
}

class ReviewPage {
  constructor() {
    this.page = byId("review");
    this.message = byId("review-message");
    this.sort = null;
    this.order = "asc";
    this.filter = "";
    this.offset = 0;
    this.review = null;
    this.session = null;
    this.subject = null;
    this.viewer = null;
    this.selected = null;
  }

  async start() {
    try {
      this.session = await api.getSession();
    } catch (failure) {
      this.closed(`The session could not be opened: ${failure.message}`);
      return;
    }
    byId("filter-form").addEventListener("submit", (event) => {
      event.preventDefault();
      this.filter = byId("filter").value.trim();
      this.offset = 0;
      this.load();
    });
    byId("page-previous").addEventListener("click", () => this.turn(-1));
    byId("page-next").addEventListener("click", () => this.turn(1));
    installKeys(
      window,
      buildKeyMap(null, VIEWER_KEYS),
      () => null,
      (action, down) => this.perform(action, down),
    );
    await this.load();
  }

  // The page cannot show anything: say why, in words.
  closed(text) {
    this.page.dataset.state = "closed";
    this.message.textContent = text;
    this.message.hidden = false;
    byId("review-body").hidden = true;
  }

  refused(failure) {
    if (failure instanceof api.ApiError && failure.status === 403) {
      this.closed(
        "The review is closed while you still have subjects to rate: it shows scores, " +
          `datasets and other ratings. (${failure.message})`,
      );
    } else {
      this.closed(`The review could not be loaded: ${failure.message}`);
    }
  }

  async load() {
    const query = /* shape: ReviewQuery */ {
      sort: this.sort,
      order: this.order,
      filter: this.filter,
      offset: this.offset,
      limit: PAGE_ROWS,
    };
    let review;
    try {
      review = await api.getReview(query);
    } catch (failure) {
      if (failure instanceof api.ApiError && failure.status === 422 && this.review) {
        // a filter or a sort column the server does not know: keep the table, say what is wrong
        this.message.textContent = failure.message;
        this.message.hidden = false;
        return;
      }
      this.refused(failure);
      return;
    }
    this.review = review;
    this.message.hidden = true;
    this.page.dataset.state = "open";
    byId("review-body").hidden = false;
    this.table();
    this.scatter();
    const first = review.total ? this.offset + 1 : 0;
    const last = this.offset + review.rows.length;
    byId("review-count").textContent = `${review.total} subjects`;
    byId("page-label").textContent = `${first} to ${last} of ${review.total}`;
    byId("page-previous").disabled = this.offset === 0;
    byId("page-next").disabled = last >= review.total;
  }

  turn(by) {
    this.offset = Math.max(0, this.offset + by * PAGE_ROWS);
    this.load();
  }

  sortBy(column) {
    if (this.sort === column) this.order = this.order === "asc" ? "desc" : "asc";
    else {
      this.sort = column;
      this.order = "asc";
    }
    this.offset = 0;
    this.load();
  }

  table() {
    const review = this.review;
    const head = html("tr");
    for (const column of review.columns) {
      const cell = html("th");
      const control = html("button", "sort", column);
      control.type = "button";
      control.dataset.column = column;
      if (column === this.sort) {
        cell.setAttribute("aria-sort", this.order === "asc" ? "ascending" : "descending");
        control.textContent = `${column} ${this.order === "asc" ? "▲" : "▼"}`;
      }
      control.addEventListener("click", () => this.sortBy(column));
      cell.append(control);
      head.append(cell);
    }
    const body = review.rows.map((row) => {
      const line = html("tr");
      line.dataset.subjectUid = row.subject_uid;
      if (row.subject_uid === this.selected) line.setAttribute("aria-selected", "true");
      for (const column of review.columns) line.append(html("td", "", cellText(row[column])));
      line.addEventListener("click", () => this.open(row));
      return line;
    });
    const table = byId("subjects");
    table.tHead.replaceChildren(head);
    table.tBodies[0].replaceChildren(...body);
  }

  // Composite score (x) against the final label (y), one dot per subject of the table.
  scatter() {
    const rows = this.review.rows.filter((row) => typeof row.composite === "number");
    const figure = byId("scatter");
    const width = 640;
    const height = 220;
    const left = 86;
    const right = 14;
    const top = 12;
    const bottom = 34;
    figure.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const scale = this.session.protocol.scale;
    const levels = ["not rated", ...scale.map((level) => level.value), "unratable"];
    const names = ["not rated", ...scale.map((level) => level.label), "unratable"];
    const band = (height - top - bottom) / levels.length;
    const xs = rows.map((row) => row.composite);
    const low = xs.length ? Math.min(...xs) : 0;
    const high = xs.length ? Math.max(...xs) : 1;
    const span = high - low || 1;
    const x = (value) => left + ((value - low) / span) * (width - left - right);
    const y = (position) => height - bottom - (position + 0.5) * band;
    const parts = [];
    levels.forEach((level, position) => {
      parts.push(
        svg("line", { class: "scatter-grid", x1: left, x2: width - right, y1: y(position), y2: y(position) }),
        svg("text", { class: "scatter-label", x: left - 8, y: y(position) + 4, "text-anchor": "end" }, names[position]),
      );
    });
    parts.push(
      svg("text", { class: "scatter-label", x: left, y: height - 18, "text-anchor": "start" }, cellText(low)),
      svg("text", { class: "scatter-label", x: width - right, y: height - 18, "text-anchor": "end" }, cellText(high)),
      svg("text", { class: "scatter-label", x: (left + width - right) / 2, y: height - 4, "text-anchor": "middle" }, "composite score"),
    );
    rows.forEach((row, position) => {
      const level = levels.indexOf(finalLevel(row));
      // a fixed spread within the band, so dots of one label do not hide each other
      const spread = (((position * 7919) % 17) / 16 - 0.5) * band * 0.6;
      const dot = svg("circle", {
        class: row.subject_uid === this.selected ? "scatter-dot scatter-chosen" : "scatter-dot",
        cx: x(row.composite),
        cy: y(level) + spread,
        r: 4,
      });
      dot.dataset.subjectUid = row.subject_uid;
      dot.append(svg("title", {}, row.subject_label ?? row.subject_uid));
      dot.addEventListener("click", () => this.open(row));
      parts.push(dot);
    });
    figure.replaceChildren(...parts);
    byId("scatter-caption").textContent = rows.length
      ? `Composite score against the final label (the worse of A and B) of the ${rows.length} subjects of this page that have a score. A dot opens its subject.`
      : "No subject of this page has a composite score.";
  }

  // --- one subject ----------------------------------------------------------------------

  async open(row) {
    this.selected = row.subject_uid;
    this.table();
    this.scatter();
    let subject;
    try {
      subject = await api.getSubject(row.subject_uid);
    } catch (failure) {
      if (failure instanceof api.ApiError && failure.status === 403) this.refused(failure);
      else byId("subject-message").textContent = `The subject could not be loaded: ${failure.message}`;
      return;
    }
    this.subject = subject;
    byId("subject").hidden = false;
    byId("subject-title").textContent = row.subject_label ?? subject.subject_uid;
    const viewerRoot = byId("viewer");
    viewerRoot.hidden = !subject.render;
    if (subject.render) {
      if (!this.viewer) {
        this.viewer = new Viewer(viewerRoot, { onFrame: () => this.layer.draw() });
        this.layer = new MarkLayer(this.viewer, {
          showMm: this.session.settings.mark_show_mm,
          getMarks: () => this.subject.marks,
          getPoints: () =>
            this.subject.clusters.map((cluster) => ({
              tkr: cluster.tkr,
              label: `${cluster.region} ${cluster.rank}`,
            })),
        });
      }
      byId("subject-message").textContent =
        "Circles are marks (class and rater); diamonds are the centroids of outlier clusters.";
      this.viewer.show(subject, { autoplay: false });
      // start where there is something to see: the first mark, else the largest cluster
      if (subject.marks.length) {
        this.goTo(subject.marks[0].vox, [subject.marks[0].panel, subject.marks[0].slice]);
      }
      else if (subject.clusters.length) this.goTo(tkrToVoxel(subject.render, subject.clusters[0].tkr));
    } else {
      byId("subject-message").textContent = row.render_reason
        ? `No render: ${row.render_reason}.`
        : "No render yet.";
    }
    this.detail(subject);
  }

  // Show the frame of the synchronized mode whose slices are nearest a voxel. With `own`
  // (a panel name and one of its slices), only frames that show that tile are considered:
  // a mark is then seen on the tile it was placed on.
  goTo(vox, own = null) {
    if (!this.viewer || !this.viewer.render) return;
    const render = this.viewer.render;
    const wanted = PANEL_ORDER.map((name) => {
      const panel = render.panels[name];
      return nearestSlice(panel, vox[panel.slice_voxel_axis]);
    });
    this.viewer.pause();
    if (this.viewer.mode !== "sync") this.viewer.toggleMode();
    this.viewer.seek(frameShowing(render.modes.sync.frames, wanted, own));
  }

  detail(subject) {
    const box = byId("subject-detail");
    const parts = [];
    const list = (title, columns, rows, onRow = null) => {
      parts.push(html("h3", "", `${title} (${rows.length})`));
      if (!rows.length) return;
      const table = html("table", "subjects");
      const head = html("tr");
      for (const column of columns) head.append(html("th", "", column));
      table.append(head);
      for (const row of rows) {
        const line = html("tr");
        for (const column of columns) line.append(html("td", "", cellText(row[column])));
        if (onRow) line.addEventListener("click", () => onRow(row));
        table.append(line);
      }
      const wrap = html("div", "table-wrap");
      wrap.append(table);
      parts.push(wrap);
    };
    list("Ratings", RATING_COLUMNS, subject.ratings);
    list("Marks", MARK_COLUMNS, subject.marks, (mark) => this.goTo(mark.vox, [mark.panel, mark.slice]));
    list("Outlier clusters with a native position", CLUSTER_COLUMNS, subject.clusters, (cluster) => {
      if (subject.render) this.goTo(tkrToVoxel(subject.render, cluster.tkr));
    });
    if (subject.scores) {
      parts.push(html("h3", "", "Scores"));
      const scores = html("dl", "scores");
      for (const [key, value] of Object.entries(subject.scores)) {
        scores.append(html("dt", "", key), html("dd", "", cellText(value)));
      }
      parts.push(scores);
    }
    box.replaceChildren(...parts);
  }

  perform(action, down = true) {
    const viewer = this.viewer;
    if (!viewer || !viewer.render) return;
    switch (action.type) {
      case "toggle_play":
        viewer.toggle();
        break;
      case "step":
        viewer.step(action.by);
        break;
      case "speed":
        viewer.changeSpeed(action.by);
        break;
      case "toggle_mode":
        viewer.toggleMode();
        break;
      case "toggle_zoom":
        viewer.toggleZoom();
        break;
      case "hide_contours":
        viewer.setContoursHidden(down);
        break;
      case "hide_marks":
        this.layer.setHidden(down);
        break;
      case "escape":
        if (document.activeElement) document.activeElement.blur();
        break;
      default:
    }
  }
}

if (typeof document !== "undefined") new ReviewPage().start();
