// The rating page: session, next subject, the rating and its marks, submit, back, the
// progress line. Everything about a subject arrives through the API; the page shows its
// tiles and its geometry and nothing else about it.

import * as api from "./api.js";
import { EventLog } from "./events.js";
import { Help } from "./help.js";
import { buildKeyMap, installKeys } from "./keys.js";
import { MarkLayer, MarkStore } from "./marks.js";
import { RatingBar, ReasonPicker, blockers, emptyRating, isPositive, ratingBody, ratingFromPrevious } from "./rating.js";
import { Viewer } from "./viewer.js";

const DRAFT_KEY = "surfcensus.draft";
const byId = (id) => document.getElementById(id);

function readDraft(assignmentId) {
  try {
    const draft = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? "null");
    return draft && draft.assignment_id === assignmentId ? draft : null;
  } catch {
    return null;
  }
}

function writeDraft(draft) {
  try {
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // without storage a draft does not survive a reload; nothing else changes
  }
}

function dropDraft(assignmentId) {
  try {
    if (readDraft(assignmentId)) sessionStorage.removeItem(DRAFT_KEY);
  } catch {
    // nothing to drop
  }
}

class RatePage {
  constructor() {
    this.page = byId("rate");
    this.hint = byId("hint");
    this.message = byId("message");
    this.noteBox = byId("note-box");
    this.note = byId("note");
    this.submitButton = byId("submit");
    this.session = null;
    this.assignment = null;
    this.rating = emptyRating();
    this.marks = new MarkStore();
    this.activeClass = null;
    this.notice = "";
    this.busy = false;
    this.turn = 0;
    this.visibleSince = null;
    this.visibleMs = 0;
  }

  async start() {
    this.setState("loading");
    try {
      this.session = await api.getSession();
    } catch (failure) {
      this.fail(`The session could not be opened: ${failure.message}. Reload the page.`);
      return;
    }
    this.setUp();
    let next = null;
    const unsent = api.unsentRating();
    if (unsent) {
      // a rating that was posted before the page was reloaded and never answered
      this.say("An unsaved rating is being sent again.");
      try {
        const stored = await api.submitRating(unsent, { onRetry: (n) => this.retrying(n) });
        next = stored.next;
        dropDraft(unsent.assignment_id);
      } catch (failure) {
        this.notice = `The unsaved rating was not stored: ${failure.message}`;
      }
    }
    try {
      await this.take(next ?? (await api.getNext()));
    } catch (failure) {
      this.fail(`The next subject could not be fetched: ${failure.message}. Reload the page.`);
    }
  }

  setUp() {
    const session = this.session;
    const protocol = session.protocol;
    this.viewer = new Viewer(byId("viewer"), {
      onFrame: (frame, mode) => {
        this.events.frame(mode, frame);
        this.layer.draw();
      },
      onState: (name, payload) => this.events.add(name, payload),
      onClick: (click) => this.clicked(click),
    });
    this.layer = new MarkLayer(this.viewer, {
      showMm: session.settings.mark_show_mm,
      getMarks: () => this.marks.marks,
      getSelected: () => this.marks.selected,
    });
    this.events = new EventLog({ send: api.postEvents, beacon: api.beaconEvents });
    this.events.start();
    this.bar = new RatingBar(byId("criteria"), protocol, {
      onValue: (criterion, value) => this.perform({ type: "value", criterion, value }),
      onClass: (markClass) => this.perform({ type: "class", class: markClass }),
    });
    this.picker = new ReasonPicker(byId("unratable-picker"), protocol.unratable_reasons, {
      onPick: (reason) => {
        this.rating.unratable = reason;
        this.changed();
      },
      onCancel: () => this.changed(),
    });
    this.help = new Help(byId("help"), session);

    const press = (id, action) =>
      byId(id).addEventListener("click", (event) => {
        event.currentTarget.blur();
        this.perform(action);
      });
    press("unratable", { type: "unratable" });
    press("other", { type: "other" });
    press("note-toggle", { type: "note" });
    press("back", { type: "back" });
    press("help-toggle", { type: "help" });
    press("submit", { type: "submit" });
    this.note.addEventListener("input", () => {
      this.rating.note = this.note.value;
      this.changed(false);
    });
    this.note.addEventListener("blur", () => {
      if (!this.note.value.trim()) this.noteBox.hidden = true;
    });
    byId("review-link").hidden = session.settings.blind;

    installKeys(
      window,
      buildKeyMap(protocol),
      () => (this.help.open ? "help" : this.picker.open ? "unratable" : null),
      (action, down) => this.perform(action, down),
    );
    document.addEventListener("visibilitychange", () => {
      const visible = document.visibilityState === "visible";
      this.events.visible(visible);
      this.clock(visible);
    });
    window.addEventListener("pagehide", () => this.events.leave());
  }

  // --- state of the page ----------------------------------------------------------------

  setState(state) {
    this.page.dataset.state = state;
  }

  say(text) {
    this.message.textContent = text;
    this.message.hidden = !text;
  }

  fail(text) {
    this.setState("error");
    this.say(text);
  }

  retrying(attempt) {
    this.notice = `Not saved yet: the server does not answer (attempt ${attempt}). Trying again.`;
    this.refresh();
  }

  showProgress(progress) {
    const parts = [
      `${progress.done_today} done today`,
      `${progress.done_total} done in total`,
      `${progress.remaining} remaining`,
    ];
    if (progress.rendered < progress.n_subjects) {
      parts.push(`${progress.rendered} of ${progress.n_subjects} rendered`);
    }
    const line = byId("progress-line");
    line.textContent = parts.join(" · ");
    line.dataset.doneToday = String(progress.done_today);
    line.dataset.doneTotal = String(progress.done_total);
    line.dataset.remaining = String(progress.remaining);
  }

  // Time the subject was on a visible page.
  clock(visible) {
    const now = performance.now();
    if (!visible && this.visibleSince !== null) {
      this.visibleMs += now - this.visibleSince;
      this.visibleSince = null;
    } else if (visible && this.visibleSince === null && this.assignment) {
      this.visibleSince = now;
    }
  }

  msOnSubject() {
    const running = this.visibleSince === null ? 0 : performance.now() - this.visibleSince;
    return this.visibleMs + running;
  }

  // --- the queue ------------------------------------------------------------------------

  // Act on a NextOut: show its assignment, wait while nothing is rendered, or say that
  // nothing is left.
  async take(next) {
    const turn = (this.turn += 1);
    this.showProgress(next.progress);
    if (next.status === "rendering") {
      this.leave();
      this.setState("rendering");
      const waiting = (answer) => {
        if (turn !== this.turn) return;
        this.showProgress(answer.progress);
        this.say(
          `Rendering: ${answer.progress.rendered} of ${answer.progress.n_subjects} subjects ` +
            "are ready. The next one opens by itself.",
        );
      };
      waiting(next);
      next = await api.settle(next, { onRendering: waiting });
      if (turn !== this.turn) return;
      this.showProgress(next.progress);
    }
    if (next.status === "done") {
      this.leave();
      this.setState("done");
      byId("review-link").hidden = false;
      this.say(
        `Nothing is left to rate: ${next.progress.done_total} done. ` +
          "Backspace goes back to the last subject.",
      );
      return;
    }
    this.say("");
    this.open(next.assignment);
    this.viewer.prefetch(next.prefetch);
  }

  leave() {
    this.viewer.pause(false);
    this.events.begin(null);
    this.assignment = null;
    this.visibleSince = null;
    this.rating = emptyRating();
    this.marks.reset();
    this.activeClass = null;
    this.note.value = "";
    this.noteBox.hidden = true;
    this.picker.close();
    this.refresh();
  }

  open(assignment) {
    this.assignment = assignment;
    const draft = assignment.previous ? null : readDraft(assignment.assignment_id);
    if (assignment.previous) {
      this.rating = ratingFromPrevious(assignment.previous);
      this.marks.reset(assignment.previous.marks);
    } else if (draft) {
      this.rating = draft.rating;
      this.marks.reset(draft.marks);
    } else {
      this.rating = emptyRating();
      this.marks.reset();
    }
    this.activeClass = null;
    this.note.value = this.rating.note;
    this.noteBox.hidden = !this.rating.note;
    this.picker.close();
    this.events.begin(assignment.assignment_id);
    this.visibleMs = 0;
    this.visibleSince = document.visibilityState === "visible" ? performance.now() : null;
    this.setState("rating");
    const started = performance.now();
    this.viewer.show(assignment, { autoplay: !assignment.previous }).then((result) => {
      if (!result.current) return;
      this.events.add("load", {
        ms: Math.round(performance.now() - started),
        tiles: result.loaded,
        failed: result.failed,
      });
    });
    this.refresh();
  }

  // --- input ----------------------------------------------------------------------------

  counts() {
    const counts = {};
    for (const mark of this.marks.marks) counts[mark.class] = (counts[mark.class] ?? 0) + 1;
    return counts;
  }

  blocked() {
    return blockers(this.session.protocol, this.rating, this.counts(), this.session.require_marks);
  }

  // Something of the rating or the marks changed.
  changed(clearNotice = true) {
    if (clearNotice) this.notice = "";
    this.refresh();
  }

  refresh() {
    if (!this.bar) return;
    const rating = this.rating;
    const rated = Boolean(this.assignment);
    this.bar.update(rating, this.counts(), this.activeClass);
    const blocked = rated ? this.blocked() : [];
    this.submitButton.disabled = !rated || this.busy || blocked.length > 0;
    const unratable = byId("unratable");
    unratable.setAttribute("aria-pressed", String(Boolean(rating.unratable)));
    byId("unratable-reason").textContent = rating.unratable ? `: ${rating.unratable}` : "";
    byId("other").setAttribute("aria-pressed", String(rating.other));
    byId("note-toggle").setAttribute("aria-pressed", String(!this.noteBox.hidden));
    const active = byId("active-class");
    active.dataset.class = this.activeClass ?? "";
    active.textContent = this.activeClass ? `marking ${this.activeClass}` : "no mark class";
    this.viewer.root.dataset.activeClass = this.activeClass ?? "";
    this.layer.draw();
    let text = this.notice;
    if (!text && rated) {
      if (this.busy) text = "Saving.";
      else if (blocked.length) text = blocked.join(" · ");
      else text = "Ready: Enter submits.";
    }
    this.hint.textContent = text;
    this.hint.dataset.kind = this.notice ? "notice" : blocked.length ? "blocked" : "ready";
    // what is entered for a new subject survives a reload, and a look back at the last one
    if (rated && !this.busy && !this.assignment.previous) {
      writeDraft({
        assignment_id: this.assignment.assignment_id,
        rating,
        marks: this.marks.posted(),
      });
    }
  }

  clicked(click) {
    if (!this.assignment || this.busy) return;
    this.viewer.pause();
    const hit = click.target.closest ? click.target.closest(".mark") : null;
    if (hit) {
      this.marks.select(hit.dataset.clientId);
      this.changed();
      return;
    }
    if (!this.activeClass) {
      const keys = [
        ...this.session.protocol.criteria.map((criterion) => criterion.mark_key),
        this.session.protocol.other_mark_key,
      ];
      this.notice = `Choose what to mark first: ${keys.join(", ")}, or rate a criterion above none.`;
      this.refresh();
      return;
    }
    const mark = this.marks.add(this.activeClass, click, new Date().toISOString());
    this.events.add("mark_add", { client_id: mark.client_id, class: mark.class });
    this.changed();
  }

  // An action of the keyboard map (keys.js), from a key or from a button. `down` is given
  // for the two keys that act while held.
  perform(action, down = true) {
    const viewer = this.viewer;
    switch (action.type) {
      case "help":
        this.events.add("help", { open: this.help.toggle() });
        return;
      case "reason":
        this.picker.pick(action.index);
        return;
      case "reason_move":
        this.picker.move(action.by);
        return;
      case "reason_choose":
        this.picker.choose();
        return;
      case "reason_cancel":
        this.picker.cancel();
        return;
      case "escape":
        this.escape();
        return;
      case "back":
        this.back();
        return;
      default:
    }
    if (!this.assignment || this.busy) return;
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
        this.events.add("marks_hidden", { hidden: down });
        break;
      case "value":
        this.rating.values[action.criterion] = action.value;
        this.rating.unratable = null;
        if (isPositive(this.session.protocol, action.value)) this.activeClass = action.criterion;
        this.changed();
        break;
      case "class":
        this.activeClass = action.class;
        this.changed();
        break;
      case "delete_mark": {
        const removed = this.marks.selected ? this.marks.remove(this.marks.selected) : null;
        if (removed) this.events.add("mark_delete", { client_id: removed.client_id });
        this.changed();
        break;
      }
      case "undo_mark": {
        const removed = this.marks.undo();
        if (removed) this.events.add("mark_undo", { client_id: removed.client_id });
        this.changed();
        break;
      }
      case "unratable":
        if (this.rating.unratable) {
          this.rating.unratable = null;
          this.changed();
        } else {
          viewer.pause();
          this.picker.show();
        }
        break;
      case "other":
        this.rating.other = !this.rating.other;
        if (this.rating.other) this.openNote();
        this.changed();
        break;
      case "note":
        if (this.noteBox.hidden || document.activeElement !== this.note) this.openNote();
        else this.closeNote();
        this.changed(false);
        break;
      case "submit":
        this.submit();
        break;
      default:
    }
  }

  openNote() {
    this.noteBox.hidden = false;
    this.note.focus();
  }

  closeNote() {
    this.note.blur();
    if (!this.note.value.trim()) this.noteBox.hidden = true;
  }

  escape() {
    if (this.help.open) this.help.close();
    else if (this.picker.open) this.picker.cancel();
    else if (document.activeElement === this.note) this.closeNote();
    else if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    } else this.marks.select(null);
    this.changed(false);
  }

  // --- submit and back ------------------------------------------------------------------

  async submit() {
    if (!this.assignment || this.busy) return;
    const blocked = this.blocked();
    if (blocked.length) {
      this.notice = `Not submitted: ${blocked.join(" · ")}`;
      this.refresh();
      return;
    }
    const assignment = this.assignment;
    const body = ratingBody(
      assignment,
      this.session.protocol,
      this.rating,
      this.marks.posted(),
      this.msOnSubject(),
      new Date().toISOString(),
    );
    this.busy = true;
    this.notice = "";
    this.viewer.pause();
    this.setState("saving");
    this.refresh();
    // the events first: they belong to an assignment that is still open
    this.events.closeFrames();
    await this.events.flush().catch(() => {});
    try {
      const stored = await api.submitRating(body, { onRetry: (n) => this.retrying(n) });
      this.busy = false;
      this.notice = "";
      dropDraft(assignment.assignment_id);
      await this.take(stored.next);
    } catch (failure) {
      this.busy = false;
      if (!(failure instanceof api.ApiError)) {
        this.fail(`The rating could not be sent: ${failure.message}. Reload the page.`);
        return;
      }
      if (failure.status === 422) {
        // the rating is wrong in a way the page did not see: say where, and keep it
        this.setState("rating");
        this.notice = `Not stored: ${failure.errors
          .map((problem) => `${problem.field}: ${problem.message}`)
          .join(" · ")}`;
        this.refresh();
        return;
      }
      // 409, 403, 404: the page is behind the server (another protocol, another render)
      await this.again(`Not stored (${failure.message}). The subject was loaded again.`);
    }
  }

  // Fetch the session and the next subject again.
  async again(notice) {
    try {
      const session = await api.getSession();
      if (session.protocol_sha256 !== this.session.protocol_sha256) {
        // another protocol: other criteria, other keys. Start the page over.
        window.location.reload();
        return;
      }
      await this.take(await api.getNext());
      this.notice = notice;
      this.refresh();
    } catch (failure) {
      this.fail(`The next subject could not be fetched: ${failure.message}. Reload the page.`);
    }
  }

  async back() {
    if (this.busy || !this.session) return;
    let next;
    try {
      next = await api.getBack();
    } catch (failure) {
      if (failure instanceof api.ApiError && failure.status === 404) {
        this.notice = "There is no earlier subject to go back to.";
      } else {
        this.notice = `Going back did not work: ${failure.message}`;
      }
      this.refresh();
      return;
    }
    this.events.add("back", {});
    this.notice = "";
    await this.take(next);
    this.notice = "This subject was rated before: change what is wrong, then Enter stores it again.";
    this.refresh();
  }
}

new RatePage().start();
