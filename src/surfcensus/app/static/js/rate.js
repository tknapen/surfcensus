// The rating page: session, next subject, the rating and its marks, submit, skip, back, the
// progress line, the tour of the screen. Everything about a subject arrives through the
// API; the page shows its tiles and its geometry and nothing else about it.

import * as api from "./api.js";
import { EventLog } from "./events.js";
import { Help } from "./help.js";
import { buildKeyMap, installKeys, keysFor } from "./keys.js";
import { MarkLayer, MarkStore } from "./marks.js";
import {
  RatingBar,
  ReasonPicker,
  blockers,
  emptyRating,
  isPositive,
  ratingBody,
  ratingFromPrevious,
  rowsLacking,
  skipBody,
} from "./rating.js";
import { Tour, markTourSeen, tourSeen, tourSteps } from "./tour.js";
import { Viewer } from "./viewer.js";

// What is entered for a subject without a rating is kept per assignment, so that it is
// there again after a reload and when a skipped subject comes back.
const DRAFT_KEY = "surfcensus.draft";
const byId = (id) => document.getElementById(id);
const draftKey = (assignmentId) => `${DRAFT_KEY}.${assignmentId}`;

function readDraft(assignmentId) {
  try {
    const draft = JSON.parse(sessionStorage.getItem(draftKey(assignmentId)) ?? "null");
    return draft && draft.assignment_id === assignmentId ? draft : null;
  } catch {
    return null;
  }
}

function writeDraft(draft) {
  try {
    sessionStorage.setItem(draftKey(draft.assignment_id), JSON.stringify(draft));
  } catch {
    // without storage a draft does not survive a reload; nothing else changes
  }
}

function dropDraft(assignmentId) {
  try {
    sessionStorage.removeItem(draftKey(assignmentId));
  } catch {
    // nothing to drop
  }
}

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function words(names) {
  if (names.length < 2) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

class RatePage {
  constructor() {
    this.page = byId("rate");
    this.hint = byId("hint");
    this.message = byId("message");
    this.noteBox = byId("note-box");
    this.note = byId("note");
    this.submitButton = byId("submit");
    this.skipButton = byId("skip");
    this.refusal = byId("refusal");
    this.session = null;
    this.assignment = null;
    this.rating = emptyRating();
    this.marks = new MarkStore();
    this.activeClass = null;
    this.notice = "";
    // Enter was refused for the subject on screen and what it lacked is not all there yet
    this.refused = false;
    this.busy = false;
    this.tourOffered = false;
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
        // what is on screen under the tour is not being looked at
        if (!this.tour.open) this.events.frame(mode, frame);
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
    this.keys = buildKeyMap(protocol);
    this.help = new Help(byId("help"), session, { onTour: () => this.openTour("help") });
    this.tour = new Tour(byId("tour"), tourSteps(protocol, this.keys), {
      onClose: (read) => this.tourClosed(read),
    });

    const press = (id, action) =>
      byId(id).addEventListener("click", (event) => {
        event.currentTarget.blur();
        this.perform(action);
      });
    press("unratable", { type: "unratable" });
    press("other", { type: "other" });
    press("note-toggle", { type: "note" });
    press("back", { type: "back" });
    press("skip", { type: "skip" });
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
      this.keys,
      () => {
        if (this.tour.open) return "tour";
        return this.help.open ? "help" : this.picker.open ? "unratable" : null;
      },
      (action, down) => this.perform(action, down),
    );
    document.addEventListener("visibilitychange", () => this.watched());
    window.addEventListener("resize", () => this.pointRefusal());
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

  retrying(attempt, what = "saved") {
    this.notice = `Not ${what} yet: the server does not answer (attempt ${attempt}). Trying again.`;
    this.refresh();
  }

  showProgress(progress) {
    const parts = [
      `${progress.done_today} done today`,
      `${progress.done_total} done in total`,
      `${progress.remaining} remaining`,
    ];
    if (progress.skipped > 0) parts.push(`${progress.skipped} skipped, waiting`);
    if (progress.rendered < progress.n_subjects) {
      parts.push(`${progress.rendered} of ${progress.n_subjects} rendered`);
    }
    const line = byId("progress-line");
    line.textContent = parts.join(" · ");
    line.dataset.doneToday = String(progress.done_today);
    line.dataset.doneTotal = String(progress.done_total);
    line.dataset.remaining = String(progress.remaining);
    line.dataset.skipped = String(progress.skipped);
  }

  // Whether the subject is being looked at: the page is visible and the tour is not over it.
  watching() {
    return document.visibilityState === "visible" && !(this.tour && this.tour.open);
  }

  // The page was hidden or shown, or the tour opened or closed.
  watched() {
    const watching = this.watching();
    this.events.visible(watching);
    this.clock(watching);
  }

  // Time the subject was looked at (`watching`), for `ms_on_subject`.
  clock(watching) {
    const now = performance.now();
    if (!watching && this.visibleSince !== null) {
      this.visibleMs += now - this.visibleSince;
      this.visibleSince = null;
    } else if (watching && this.visibleSince === null && this.assignment) {
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
    this.refused = false;
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
    this.refused = false;
    this.note.value = this.rating.note;
    this.noteBox.hidden = !this.rating.note;
    this.picker.close();
    this.events.begin(assignment.assignment_id);
    // of this visit: an earlier visit that ended with a skip was sent with the skip
    this.visibleMs = 0;
    this.visibleSince = this.watching() ? performance.now() : null;
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
    if (!this.tourOffered) {
      // the first subject of this page: a rater who was never shown the tour gets it now
      this.tourOffered = true;
      if (!tourSeen(this.session.rater.name)) this.openTour("first");
    }
  }

  // --- the tour ---------------------------------------------------------------------------

  // Open the tour of the screen over the subject. `how`: "first" (by itself, on a rater's
  // first subject), "key" or "help".
  openTour(how) {
    if (!this.assignment || this.busy || this.picker.open || this.tour.open) return;
    if (this.help.open) {
      this.help.close();
      this.events.add("help", { open: false });
    }
    markTourSeen(this.session.rater.name);
    this.events.add("tour_open", { how });
    this.viewer.hold();
    this.tour.show();
    // from here until it closes, no time counts for the subject
    this.watched();
    this.refresh();
  }

  tourClosed(read) {
    this.watched();
    this.viewer.release();
    this.events.add("tour_close", read);
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
    const blocked = rated ? this.blocked() : [];
    // a refusal stands until what it named is there, or the subject changes
    if (!blocked.length) this.refused = false;
    const lacking = this.refused
      ? rowsLacking(this.session.protocol, rating, this.counts(), this.session.require_marks)
      : {};
    this.bar.update(rating, this.counts(), this.activeClass, lacking);
    // a blocked submit is not disabled: pressing it says what is missing
    this.submitButton.disabled = !rated || this.busy;
    this.submitButton.setAttribute("aria-disabled", String(!rated || this.busy || blocked.length > 0));
    this.skipButton.disabled = !rated || this.busy;
    this.showRefusal(blocked, lacking);
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
    // what is entered for a subject without a rating survives a reload, a look back at the
    // last one, and a skip
    if (rated && !this.busy && !this.assignment.previous) {
      writeDraft({
        assignment_id: this.assignment.assignment_id,
        rating,
        marks: this.marks.posted(),
      });
    }
  }

  // Why Enter was refused, beside the submit button, and how to leave the subject without
  // a rating. Hidden unless Enter was refused and the rating is still held back.
  showRefusal(blocked, lacking) {
    const box = this.refusal;
    if (!this.refused || !blocked.length) {
      box.hidden = true;
      box.replaceChildren();
      return;
    }
    const keys = (type) => keysFor(this.keys, (action) => action.type === type);
    const kbd = (names) => names.flatMap((name, at) => [at ? " " : "", html("kbd", "", name)]);
    const noValue = Object.keys(lacking).filter((id) => lacking[id] === "value");
    const noMark = Object.keys(lacking).filter((id) => lacking[id] === "mark");
    const reasons = html("ul", "refusal-reasons");
    const reason = (...parts) => {
      const item = html("li");
      item.append(...parts);
      reasons.append(item);
    };
    if (noValue.length) {
      reason(
        `${words(noValue)} ${noValue.length > 1 ? "have" : "has"} no value yet. Give ` +
          `${noValue.length > 1 ? "each" : "it"} a value, or press `,
        ...kbd(keys("unratable")),
        " if the subject cannot be rated.",
      );
    }
    if (noMark.length) {
      reason(
        `${words(noMark)} ${noMark.length > 1 ? "have" : "has"} no mark yet. Click on the ` +
          "error in one of the views.",
      );
    }
    if (this.rating.other && !this.rating.note.trim()) {
      reason("The other problem needs a note: press ", ...kbd(keys("note")), ".");
    }
    const skip = html("p", "refusal-skip");
    skip.append(
      "To leave this subject for later, press ",
      ...kbd(keys("skip")),
      ": it comes back after the others, and nothing is stored.",
    );
    box.replaceChildren(html("strong", "refusal-title", "Not submitted."), reasons, skip);
    box.hidden = false;
    this.pointRefusal();
  }

  // The arrow of the refusal points at the submit button, wherever the window puts it.
  pointRefusal() {
    if (this.refusal.hidden) return;
    const button = this.submitButton.getBoundingClientRect();
    const box = this.refusal.getBoundingClientRect();
    const at = button.left + button.width / 2 - box.left;
    this.refusal.style.setProperty("--arrow", `${Math.max(16, Math.min(box.width - 16, at))}px`);
  }

  clicked(click) {
    if (!this.assignment || this.busy || this.tour.open) return;
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
      this.notice =
        `Choose what to mark first: ${keys.join(", ")}, ` + "or rate a criterion above none.";
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
        // the tour is about the subject on screen: without one the help screen cannot open it
        this.help.allowTour(Boolean(this.assignment) && !this.busy);
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
      case "tour":
        this.openTour(this.help.open ? "help" : "key");
        return;
      case "tour_step":
        this.tour.step(action.by);
        return;
      case "tour_close":
        this.tour.close();
        return;
      default:
    }
    if (!this.assignment || this.busy || this.tour.open) return;
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
      case "skip":
        this.skip();
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
    if (this.blocked().length) {
      this.refuse();
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

  // Enter was pressed and the rating is not complete: say so where it cannot be missed.
  refuse() {
    this.refused = true;
    this.notice = "";
    this.refresh();
    // again at every refused Enter, so that a second press is answered too
    this.refusal.classList.remove("refusal-again");
    void this.refusal.offsetWidth;
    this.refusal.classList.add("refusal-again");
    this.refusal.scrollIntoView({ block: "nearest" });
  }

  // Leave the subject without a rating. One without a submission is skipped on the server:
  // it comes back after the others, and what was entered stays here as its draft. One
  // that is shown again for revision keeps the rating it has.
  async skip() {
    if (!this.assignment || this.busy) return;
    const assignment = this.assignment;
    if (assignment.previous) {
      await this.again("Left as it was: the earlier rating stands.");
      return;
    }
    const body = skipBody(assignment, this.msOnSubject(), new Date().toISOString());
    this.busy = true;
    this.notice = "";
    this.viewer.pause();
    this.setState("saving");
    this.refresh();
    this.events.closeFrames();
    await this.events.flush().catch(() => {});
    let next;
    try {
      next = await api.skipAssignment(body, { onRetry: (n) => this.retrying(n, "skipped") });
    } catch (failure) {
      this.busy = false;
      if (!(failure instanceof api.ApiError)) {
        this.fail(`The skip could not be sent: ${failure.message}. Reload the page.`);
        return;
      }
      // 409, 403, 404: the page is behind the server
      await this.again(`Not skipped (${failure.message}). The subject was loaded again.`);
      return;
    }
    this.busy = false;
    try {
      await this.take(next);
    } catch (failure) {
      this.fail(`The next subject could not be fetched: ${failure.message}. Reload the page.`);
      return;
    }
    if (!this.assignment) return;
    const back = keysFor(this.keys, (action) => action.type === "back").join(" ");
    this.notice =
      this.assignment.assignment_id === assignment.assignment_id
        ? "No other subject is left, so this one is here again. Rate it to finish."
        : `Skipped: nothing was stored, and that subject comes back later. ${back} returns to it now.`;
    this.refresh();
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
    if (!this.assignment) return;
    this.notice = this.assignment.previous
      ? "This subject was rated before: change what is wrong, then Enter stores it again."
      : "This subject was skipped: rate it now, or skip it again.";
    this.refresh();
  }
}

new RatePage().start();
