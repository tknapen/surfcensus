// Interaction events of the subject on screen (app.schemas.EVENT_TYPES): what was pressed,
// and which frames were looked at for how long. Frame viewing is run-length encoded, not
// logged per frame. Events are posted every few seconds, before a rating is submitted, and
// with a beacon when the page is left.

export const MAX_EVENTS_PER_CALL = 500;
// A `frames` event holds at most this many visits, which keeps its payload under the
// 16 kB that the server stores (a visit is at most about 16 characters).
export const MAX_VISITS_PER_EVENT = 800;
const MAX_QUEUED = 5000;
const FLUSH_MS = 5000;

// Which frame was on screen for how long: visits [frame, ms] in order, one list per stretch
// in one playback mode.
export class FrameLog {
  constructor() {
    this.mode = null;
    this.frame = null;
    this.since = null;
    this.visits = [];
  }

  close(now) {
    if (this.frame === null) return;
    const ms = Math.max(0, Math.round(now - this.since));
    const last = this.visits[this.visits.length - 1];
    if (last && last[0] === this.frame) last[1] += ms;
    else this.visits.push([this.frame, ms]);
    this.since = now;
  }

  take() {
    if (!this.visits.length) return [];
    const payload = { mode: this.mode, visits: this.visits };
    this.visits = [];
    return [payload];
  }

  // The frame on screen is now `frame` of `mode`. Returns the payloads of `frames` events
  // that are complete: when the mode changed, or the list is full.
  show(mode, frame, now) {
    let done = [];
    this.close(now);
    if (mode !== this.mode || this.visits.length >= MAX_VISITS_PER_EVENT) done = this.take();
    this.mode = mode;
    this.frame = frame;
    this.since = now;
    return done;
  }

  // Nothing is on screen from now on (the page is hidden, the subject is left).
  stop(now) {
    this.close(now);
    this.frame = null;
    return this.take();
  }

  // What was seen so far; the frame on screen stays on screen.
  drain(now) {
    this.close(now);
    return this.take();
  }
}

// Bodies of POST /api/events for a queue of {assignment_id, event}: the events of one
// assignment together, in order, at most MAX_EVENTS_PER_CALL per body.
export function eventBatches(queue) {
  const batches = [];
  let batch = null;
  for (const item of queue) {
    if (
      !batch ||
      batch.assignment_id !== item.assignment_id ||
      batch.events.length >= MAX_EVENTS_PER_CALL
    ) {
      batch = /* shape: EventsIn */ { assignment_id: item.assignment_id, events: [] };
      batches.push(batch);
    }
    batch.events.push(item.event);
  }
  return batches;
}

export class EventLog {
  // send(body) posts a batch and resolves when it is stored; it rejects with an error that
  // has a `status` when the server refused it. beacon(body) hands a batch to the browser.
  constructor({
    send,
    beacon,
    now = () => performance.now(),
    wall = () => new Date().toISOString(),
  }) {
    this.send = send;
    this.beacon = beacon;
    this.now = now;
    this.wall = wall;
    this.assignmentId = null;
    this.queue = [];
    this.frames = new FrameLog();
    this.shown = null;
    this.timer = null;
    this.sending = null;
  }

  start() {
    if (this.timer === null) this.timer = setInterval(() => this.flush(), FLUSH_MS);
  }

  // The events from now on belong to this assignment (null: to none, they are dropped).
  begin(assignmentId) {
    this.closeFrames();
    this.assignmentId = assignmentId;
    this.shown = null;
  }

  add(type, payload = {}) {
    if (this.assignmentId === null) return;
    const event = /* shape: EventIn */ { ts: this.wall(), type, payload };
    this.queue.push({ assignment_id: this.assignmentId, event });
    if (this.queue.length > MAX_QUEUED) this.queue.splice(0, this.queue.length - MAX_QUEUED);
  }

  addFrames(payloads) {
    for (const payload of payloads) this.add("frames", payload);
  }

  frame(mode, frame) {
    this.shown = [mode, frame];
    this.addFrames(this.frames.show(mode, frame, this.now()));
  }

  // The page is hidden, or shown again: time while it is hidden is not viewing time.
  visible(isVisible) {
    if (!isVisible) this.addFrames(this.frames.stop(this.now()));
    else if (this.shown) this.addFrames(this.frames.show(this.shown[0], this.shown[1], this.now()));
  }

  closeFrames() {
    this.addFrames(this.frames.stop(this.now()));
  }

  // Post what is queued. Events the server did not answer for are kept for the next call;
  // events it refused are dropped.
  async flush() {
    if (this.sending) await this.sending.catch(() => {});
    this.addFrames(this.frames.drain(this.now()));
    if (!this.queue.length) return;
    const queue = this.queue;
    this.queue = [];
    this.sending = (async () => {
      const batches = eventBatches(queue);
      for (let position = 0; position < batches.length; position += 1) {
        try {
          await this.send(batches[position]);
        } catch (failure) {
          if (failure && typeof failure.status === "number" && failure.status < 500) continue;
          const kept = batches
            .slice(position)
            .flatMap((batch) =>
              batch.events.map((event) => ({ assignment_id: batch.assignment_id, event })),
            );
          this.queue = [...kept, ...this.queue];
          return;
        }
      }
    })();
    try {
      await this.sending;
    } finally {
      this.sending = null;
    }
  }

  // The page is being left: hand everything to the browser.
  leave() {
    this.closeFrames();
    const queue = this.queue;
    this.queue = [];
    for (const batch of eventBatches(queue)) this.beacon(batch);
  }
}
