// The keyboard map of docs/rater_app.md. The keys of the viewer and of the page are fixed
// (surfcensus.protocol.FIXED_KEYS); the keys of the criteria and of the mark classes come
// from the protocol of the session. Keys are compared without regard to case.
//
// `keyAction` decides what a key press means, from the event and from where the page is: a
// text field takes every key but Escape, and an open dialog takes its own keys only.

// As KeyboardEvent.key gives them. A press with Shift moves ten frames (the arrows).
export const FIXED_ACTIONS = {
  " ": { type: "toggle_play" },
  ArrowLeft: { type: "step", by: -1 },
  ArrowRight: { type: "step", by: 1 },
  "[": { type: "speed", by: -1 },
  "]": { type: "speed", by: 1 },
  m: { type: "toggle_mode" },
  z: { type: "toggle_zoom" },
  Delete: { type: "delete_mark" },
  h: { type: "hide_marks", hold: true },
  s: { type: "hide_contours", hold: true },
  u: { type: "unratable" },
  o: { type: "other" },
  n: { type: "note" },
  Enter: { type: "submit" },
  k: { type: "skip" },
  Backspace: { type: "back" },
  "?": { type: "help" },
  t: { type: "tour" },
  Escape: { type: "escape" },
};

// How a key is written on the screen.
const KEY_LABELS = { " ": "Space", ArrowLeft: "←", ArrowRight: "→", Escape: "Esc" };

export function keyLabel(key) {
  return KEY_LABELS[key] ?? key;
}

// The keys of a key map whose action passes `test`, as they are written on the screen, in
// the order of the map: the fixed keys first, then those of the protocol.
export function keysFor(map, test) {
  const found = [];
  for (const [key, action] of map) {
    if (test(action)) found.push(keyLabel(key));
  }
  return found;
}

// The keys of the viewer alone, for a page that shows subjects without rating them.
export const VIEWER_KEYS = [" ", "ArrowLeft", "ArrowRight", "[", "]", "m", "z", "h", "s", "Escape"];

// Actions that repeat while the key is held down.
const REPEATING = new Set(["step", "speed"]);
export const JUMP = 10;

export function normalKey(key) {
  return key.length === 1 ? key.toLowerCase() : key;
}

// key -> action. With a protocol: its criterion keys ({type: "value", criterion, value}) and
// mark keys ({type: "class", class}) beside the fixed ones; `only` restricts the fixed keys.
export function buildKeyMap(protocol = null, only = null) {
  const map = new Map();
  for (const [key, action] of Object.entries(FIXED_ACTIONS)) {
    if (!only || only.includes(key)) map.set(normalKey(key), action);
  }
  if (protocol) {
    for (const criterion of protocol.criteria) {
      criterion.keys.forEach((key, position) => {
        const value = protocol.scale[position].value;
        map.set(normalKey(key), { type: "value", criterion: criterion.id, value });
      });
      map.set(normalKey(criterion.mark_key), { type: "class", class: criterion.id });
    }
    map.set(normalKey(protocol.other_mark_key), { type: "class", class: "X" });
  }
  return map;
}

// What a key press does, or null when the page leaves it to the browser. `context` is
// {text, dialog}: `text` is true while a text field has the focus, `dialog` is null,
// "help", "unratable" or "tour".
export function keyAction(event, map, context) {
  const key = normalKey(event.key);
  if (context.dialog === "tour") {
    // the tour takes its own keys and lets no other through to the rating screen
    if (event.ctrlKey || event.metaKey || event.altKey) return null;
    if (key === "Enter" || key === "ArrowRight") return { type: "tour_step", by: 1 };
    if (key === "ArrowLeft") return { type: "tour_step", by: -1 };
    if (key === "Escape" || key === "t") return { type: "tour_close" };
    return null;
  }
  if (context.text) return key === "Escape" ? { type: "escape" } : null;
  if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && key === "z") {
    return context.dialog ? null : { type: "undo_mark" };
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return null;
  if (context.dialog === "help") {
    if (key === "t") return { type: "tour" };
    return key === "Escape" || key === "?" ? { type: "help" } : null;
  }
  if (context.dialog === "unratable") {
    if (/^[1-9]$/.test(key)) return { type: "reason", index: Number(key) - 1 };
    if (key === "ArrowUp") return { type: "reason_move", by: -1 };
    if (key === "ArrowDown") return { type: "reason_move", by: 1 };
    if (key === "Enter") return { type: "reason_choose" };
    if (key === "Escape" || key === "u") return { type: "reason_cancel" };
    return null;
  }
  const action = map.get(key);
  if (!action) return null;
  if (action.type === "step" && event.shiftKey) return { type: "step", by: action.by * JUMP };
  return action;
}

export function isTextField(element) {
  if (!element) return false;
  if (element.isContentEditable) return true;
  const name = element.tagName;
  if (name === "TEXTAREA" || name === "SELECT") return true;
  if (name !== "INPUT") return false;
  return !["range", "button", "checkbox", "radio", "submit", "reset"].includes(element.type);
}

// Listen on `target` (the window). `perform(action)` runs an action; for an action with
// `hold`, `perform(action, true)` when the key goes down and `perform(action, false)` when
// it comes up or the window loses the focus. `getDialog()` gives the open dialog.
export function installKeys(target, map, getDialog, perform) {
  const held = new Map();
  const release = () => {
    for (const action of held.values()) perform(action, false);
    held.clear();
  };
  target.addEventListener("keydown", (event) => {
    const context = { text: isTextField(document.activeElement), dialog: getDialog() };
    const action = keyAction(event, map, context);
    if (!action) return;
    // the browser must not scroll on Space, move a slider or press a focused button as well
    event.preventDefault();
    if (action.hold) {
      const key = normalKey(event.key);
      if (!held.has(key)) {
        held.set(key, action);
        perform(action, true);
      }
      return;
    }
    if (event.repeat && !REPEATING.has(action.type)) return;
    perform(action);
  });
  target.addEventListener("keyup", (event) => {
    const key = normalKey(event.key);
    const action = held.get(key);
    if (action) {
      held.delete(key);
      perform(action, false);
    }
  });
  target.addEventListener("blur", release);
  return release;
}
