// The tour of the rating screen: at most five short steps, each beside the part of the
// screen it is about, with that part lit. It opens by itself the first time a rater sees
// the rating screen, and again from the help screen or with its key.
//
// The tour says how one gets from subject to subject, which the screen alone did not: a
// rater who cannot submit (a row has no value) has to learn that there is a skip key.
//
// The keys it names are read from the key map of the session (keys.js), which holds the
// criterion and mark keys of the protocol, so a protocol with other keys gets a tour with
// those keys. It shows nothing about the subject and sends nothing.

import { keysFor } from "./keys.js";

const SEEN_KEY = "surfcensus.tour.seen";
const EDGE = 8;
const GAP = 10;
const SPOT_PAD = 4;

function seenKey(rater) {
  return `${SEEN_KEY}:${rater}`;
}

// Whether this rater was shown the tour in this browser. Without storage: no, so the tour
// opens once per load of the page.
export function tourSeen(rater) {
  try {
    return globalThis.localStorage.getItem(seenKey(rater)) !== null;
  } catch {
    return false;
  }
}

export function markTourSeen(rater) {
  try {
    globalThis.localStorage.setItem(seenKey(rater), "1");
  } catch {
    // a closed or full storage: the tour opens again at the next load, nothing else changes
  }
}

function list(words) {
  if (words.length < 2) return words.join("");
  return `${words.slice(0, -1).join(", ")} or ${words[words.length - 1]}`;
}

// The steps: [{id, title, targets, text}]. `targets` are selectors of the elements the step
// is about; `text` is a list of paragraphs, each a list of strings and {keys: [...]} parts.
// `map` is the key map of the session (buildKeyMap).
export function tourSteps(protocol, map) {
  const key = (type) => ({ keys: keysFor(map, (action) => action.type === type) });
  const stepKey = (by) => ({
    keys: keysFor(map, (action) => action.type === "step" && action.by === by),
  });
  const labels = protocol.scale.map((level) => level.label);
  // the values that say there is an error: every one above the lowest
  const positive = list(labels.slice(1));
  const valueKeys = protocol.criteria.flatMap((criterion, position) => [
    position ? ", " : " ",
    { keys: criterion.keys.map(String) },
    ` for ${criterion.id}`,
  ]);
  const classKeys = {
    keys: keysFor(map, (action) => action.type === "class"),
  };
  return [
    {
      id: "views",
      title: "Three views of one brain",
      targets: ["#viewer .stage"],
      text: [
        [
          "The back of the brain from the side, from behind and from above. The slices " +
            "sweep through it, forward and back.",
        ],
        ["The coloured lines are the surfaces. You judge whether they follow the brain."],
        [
          key("toggle_play"),
          " pauses. ",
          stepKey(-1),
          " and ",
          stepKey(1),
          " move one slice. Hold ",
          key("hide_contours"),
          " to hide the lines.",
        ],
      ],
    },
    {
      id: "rating",
      title: "Rate every row",
      targets: [
        "#criteria .criterion:not(.criterion-other) .criterion-label",
        "#criteria .criterion:not(.criterion-other) .levels",
      ],
      text: [
        [`Each row is one question about the surfaces. Give every row a value: ${list(labels)}.`],
        ["The key of a value is on its button:", ...valueKeys, "."],
      ],
    },
    {
      id: "marks",
      title: "Mark where the error is",
      targets: ["#criteria .class-cell"],
      text: [
        ["Click on an error in any view to mark where it is."],
        [
          `The mark goes to the active row. Rating a row ${positive} makes it active; `,
          classKeys,
          " choose one yourself.",
        ],
      ],
    },
    {
      id: "moving",
      title: "On to the next subject",
      targets: ["#back", "#skip", "#submit"],
      text: [
        [
          key("submit"),
          " stores the rating and shows the next subject. It works once every row has a " +
            "value, or ",
          key("unratable"),
          " says that the subject cannot be rated.",
        ],
        [key("skip"), " skips: nothing is stored, and the subject comes back later."],
        [key("back"), " goes back to the subject before."],
      ],
    },
    {
      id: "help",
      title: "Help is one key away",
      targets: ["#help-toggle"],
      text: [
        [key("help"), " shows what the values mean, every key, and a way back to this tour."],
        [key("tour"), " opens the tour at any time."],
      ],
    },
  ];
}

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function paragraph(parts) {
  const line = html("p");
  for (const part of parts) {
    if (typeof part === "string") {
      line.append(part);
      continue;
    }
    part.keys.forEach((name, position) => {
      if (position) line.append(" ");
      line.append(html("kbd", "", name));
    });
  }
  return line;
}

// The box around the elements of a step, in the coordinates of the window; null when none
// of them is on the page.
export function unionBox(boxes) {
  const shown = boxes.filter((box) => box.width > 0 && box.height > 0);
  if (!shown.length) return null;
  const left = Math.min(...shown.map((box) => box.left));
  const top = Math.min(...shown.map((box) => box.top));
  const right = Math.max(...shown.map((box) => box.right));
  const bottom = Math.max(...shown.map((box) => box.bottom));
  return { left, top, right, bottom, width: right - left, height: bottom - top };
}

// Where the card goes: {left, top} in the window. Under the lit part when there is room,
// else over it, else at the bottom or the top of the window, whichever hides less of it.
// Always inside the window, however small.
export function placeCard(spot, card, view) {
  const width = Math.min(card.width, view.width - 2 * EDGE);
  const centre = spot ? (spot.left + spot.right) / 2 : view.width / 2;
  const left = Math.max(EDGE, Math.min(view.width - EDGE - width, centre - width / 2));
  const lowest = Math.max(EDGE, view.height - EDGE - card.height);
  let top;
  if (!spot) top = (view.height - card.height) / 2;
  else if (spot.bottom + GAP + card.height <= view.height - EDGE) top = spot.bottom + GAP;
  else if (spot.top - GAP - card.height >= EDGE) top = spot.top - GAP - card.height;
  else top = spot.top + spot.bottom > view.height ? EDGE : lowest;
  return { left, top: Math.max(EDGE, Math.min(lowest, top)) };
}

// How far to scroll the page (down is positive) before a step is shown, in a window too
// small for the page: nothing when the part is on screen with room for the card under or
// over it; else so that the part lies at the top of the window with the card under it, or
// at the bottom with the card over it, whichever the page can scroll to. `scroll` is
// {at, max}: where the page is scrolled to, and how far it can be.
export function scrollFor(spot, cardHeight, viewHeight, scroll) {
  const room = (top, bottom) =>
    bottom + GAP + cardHeight <= viewHeight - EDGE || top - GAP - cardHeight >= EDGE;
  const inside = spot.top >= SPOT_PAD && spot.bottom <= viewHeight - SPOT_PAD;
  if (inside && room(spot.top, spot.bottom)) return 0;
  const reach = (by) => Math.max(0, Math.min(scroll.max, scroll.at + by)) - scroll.at;
  const toTop = reach(spot.top - EDGE - SPOT_PAD);
  const toBottom = reach(spot.bottom - (viewHeight - EDGE - SPOT_PAD));
  if (spot.bottom - toTop + GAP + cardHeight <= viewHeight - EDGE) return toTop;
  if (spot.top - toBottom - GAP - cardHeight >= EDGE) return toBottom;
  return toTop;
}

export class Tour {
  // `root` is the empty, hidden #tour element. onOpen() when it opens; onClose({step,
  // steps, finished, ms}) when it closes: `step` is the last step that was shown, from 1.
  constructor(root, steps, { onOpen = null, onClose = null } = {}) {
    this.root = root;
    this.steps = steps;
    this.onOpen = onOpen;
    this.onClose = onClose;
    this.position = 0;
    this.reached = 0;
    this.since = 0;
    this.returnTo = null;
    this.build();
    this.replace = () => this.place();
    this.observer = new ResizeObserver(this.replace);
  }

  get open() {
    return !this.root.hidden;
  }

  build() {
    this.spot = html("div", "tour-spot");
    this.card = html("div", "tour-card");
    this.card.setAttribute("role", "dialog");
    this.card.setAttribute("aria-modal", "true");
    this.card.setAttribute("aria-labelledby", "tour-title");
    this.card.setAttribute("aria-describedby", "tour-text");
    this.card.tabIndex = -1;
    this.count = html("span", "tour-count");
    this.title = html("h2", "tour-title");
    this.title.id = "tour-title";
    this.closeButton = html("button", "tour-close", "×");
    this.closeButton.type = "button";
    this.closeButton.setAttribute("aria-label", "Close the tour");
    this.closeButton.addEventListener("click", () => this.close());
    const head = html("div", "tour-head");
    head.append(this.count, this.title, this.closeButton);
    this.text = html("div", "tour-text");
    this.text.id = "tour-text";
    this.text.setAttribute("aria-live", "polite");
    this.backButton = html("button", "control tour-back");
    this.backButton.type = "button";
    this.backButton.append(html("kbd", "", "←"), " back");
    this.backButton.addEventListener("click", () => this.step(-1));
    this.nextButton = html("button", "control primary tour-next");
    this.nextButton.type = "button";
    this.nextButton.addEventListener("click", () => this.step(1));
    const foot = html("div", "tour-foot");
    const leave = html("span", "tour-leave");
    leave.append(html("kbd", "", "Esc"), " closes");
    foot.append(leave, this.backButton, this.nextButton);
    this.card.append(head, this.text, foot);
    this.root.replaceChildren(this.spot, this.card);
    // focus stays in the card: Tab goes round its buttons
    this.card.addEventListener("keydown", (event) => {
      if (event.key !== "Tab") return;
      const stops = [this.closeButton, this.backButton, this.nextButton].filter(
        (button) => !button.disabled,
      );
      const at = stops.indexOf(document.activeElement);
      const to = event.shiftKey ? at - 1 : at + 1;
      event.preventDefault();
      stops[(to + stops.length) % stops.length].focus();
    });
    this.root.addEventListener("focusout", (event) => {
      if (this.open && !this.card.contains(event.relatedTarget)) {
        queueMicrotask(() => {
          if (this.open && !this.card.contains(document.activeElement)) this.nextButton.focus();
        });
      }
    });
  }

  targets() {
    return this.steps[this.position].targets.flatMap((selector) => [
      ...document.querySelectorAll(selector),
    ]);
  }

  show() {
    if (this.open) return;
    this.returnTo = document.activeElement;
    this.position = 0;
    this.reached = 0;
    this.since = performance.now();
    this.root.hidden = false;
    window.addEventListener("resize", this.replace);
    window.addEventListener("scroll", this.replace, true);
    this.draw();
    this.nextButton.focus({ preventScroll: true });
    if (this.onOpen) this.onOpen();
  }

  // The next step (by 1) or the one before (by -1). A step past the last one closes the
  // tour: it was read to the end.
  step(by) {
    if (!this.open) return;
    const to = this.position + by;
    if (to >= this.steps.length) {
      this.close(true);
      return;
    }
    this.position = Math.max(0, to);
    this.draw();
  }

  draw() {
    const step = this.steps[this.position];
    const last = this.position === this.steps.length - 1;
    this.reached = Math.max(this.reached, this.position);
    this.root.dataset.step = step.id;
    this.count.textContent = `${this.position + 1} of ${this.steps.length}`;
    this.title.textContent = step.title;
    this.text.replaceChildren(...step.text.map(paragraph));
    this.backButton.disabled = this.position === 0;
    this.nextButton.replaceChildren(html("kbd", "", "Enter"), last ? " done" : " next");
    if (document.activeElement === this.backButton && this.backButton.disabled) {
      this.nextButton.focus({ preventScroll: true });
    }
    const targets = this.targets();
    this.observer.disconnect();
    this.observer.observe(this.card);
    for (const target of targets) this.observer.observe(target);
    // a window too small for the page scrolls: bring the part the step is about into it,
    // with room for the card beside it
    const box = unionBox(targets.map((target) => target.getBoundingClientRect()));
    if (box) {
      const page = document.documentElement;
      const scroll = { at: window.scrollY, max: page.scrollHeight - page.clientHeight };
      const by = scrollFor(box, this.card.getBoundingClientRect().height, page.clientHeight, scroll);
      if (by) window.scrollBy(0, by);
    }
    this.place();
  }

  place() {
    if (!this.open) return;
    const view = {
      width: document.documentElement.clientWidth,
      height: document.documentElement.clientHeight,
    };
    const spot = unionBox(this.targets().map((target) => target.getBoundingClientRect()));
    this.spot.hidden = !spot;
    if (spot) {
      this.spot.style.left = `${spot.left - SPOT_PAD}px`;
      this.spot.style.top = `${spot.top - SPOT_PAD}px`;
      this.spot.style.width = `${spot.width + 2 * SPOT_PAD}px`;
      this.spot.style.height = `${spot.height + 2 * SPOT_PAD}px`;
    }
    const card = this.card.getBoundingClientRect();
    const { left, top } = placeCard(spot, card, view);
    this.card.style.left = `${left}px`;
    this.card.style.top = `${top}px`;
  }

  close(finished = false) {
    if (!this.open) return;
    this.root.hidden = true;
    this.observer.disconnect();
    window.removeEventListener("resize", this.replace);
    window.removeEventListener("scroll", this.replace, true);
    const back = this.returnTo;
    this.returnTo = null;
    if (back && back !== document.body && back.isConnected && typeof back.focus === "function") {
      back.focus({ preventScroll: true });
    } else if (document.activeElement && document.activeElement !== document.body) {
      document.activeElement.blur();
    }
    if (this.onClose) {
      this.onClose({
        step: this.reached + 1,
        steps: this.steps.length,
        finished,
        ms: Math.max(0, Math.round(performance.now() - this.since)),
      });
    }
  }
}
