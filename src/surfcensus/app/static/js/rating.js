// The rating of the subject on screen: a value per criterion of the protocol, or an
// unratable reason; the other-problem flag and the note; which mark class is active; and
// whether Enter may submit, with the reason when it may not. The rating bar is built from
// the protocol of the session.

export function emptyRating() {
  return { values: {}, unratable: null, other: false, note: "" };
}

// The rating a page starts from when an assignment is shown again (PreviousOut).
export function ratingFromPrevious(previous) {
  return {
    values: { ...previous.values },
    unratable: previous.unratable ?? null,
    other: previous.other,
    note: previous.note,
  };
}

export function lowestValue(protocol) {
  return Math.min(...protocol.scale.map((level) => level.value));
}

// Every scale value above the lowest is positive (Protocol.positive_values).
export function isPositive(protocol, value) {
  return value > lowestValue(protocol);
}

export function levelLabel(protocol, value) {
  const found = protocol.scale.find((level) => level.value === value);
  return found ? found.label : String(value);
}

// Why Enter does not submit: a list of sentences, empty when it does. `counts` maps a mark
// class to the number of marks of it. The rules are those of POST /api/rating, so that a
// rating the page lets through is one the server stores.
export function blockers(protocol, rating, counts, requireMarks) {
  const out = [];
  if (!rating.unratable) {
    const missing = protocol.criteria
      .filter((criterion) => !(criterion.id in rating.values))
      .map((criterion) => criterion.id);
    if (missing.length) {
      out.push(`no value yet for ${missing.join(", ")} (or press u for unratable)`);
    }
    if (requireMarks) {
      for (const criterion of protocol.criteria) {
        const value = rating.values[criterion.id];
        if (value !== undefined && isPositive(protocol, value) && !counts[criterion.id]) {
          out.push(
            `criterion ${criterion.id} is rated ${levelLabel(protocol, value)} but has no mark ` +
              `of class ${criterion.id}: click on the error`,
          );
        }
      }
    }
  }
  if (rating.other && !rating.note.trim()) out.push("an other problem needs a note (press n)");
  return out;
}

// What a row of the rating bar lacks before Enter submits: criterion id -> "value" (no
// value yet) or "mark" (a positive value without a mark, where marks are required). The
// rows of `blockers`, so that a refused Enter can show where to look.
export function rowsLacking(protocol, rating, counts, requireMarks) {
  const lacking = {};
  if (rating.unratable) return lacking;
  for (const criterion of protocol.criteria) {
    const value = rating.values[criterion.id];
    if (value === undefined) lacking[criterion.id] = "value";
    else if (requireMarks && isPositive(protocol, value) && !counts[criterion.id]) {
      lacking[criterion.id] = "mark";
    }
  }
  return lacking;
}

// The body of POST /api/skip: the subject is left without a rating. Nothing of what was
// entered is in it.
export function skipBody(assignment, msOnSubject, clientTs) {
  return /* shape: SkipIn */ {
    assignment_id: assignment.assignment_id,
    client_ts: clientTs,
    ms_on_subject: Math.max(0, Math.round(msOnSubject)),
  };
}

// The body of POST /api/rating. An unratable subject is sent without values and marks.
export function ratingBody(assignment, protocol, rating, marks, msOnSubject, clientTs) {
  const unratable = rating.unratable;
  return /* shape: RatingIn */ {
    assignment_id: assignment.assignment_id,
    protocol_version: protocol.version,
    values: unratable ? {} : { ...rating.values },
    unratable,
    other: rating.other,
    note: rating.note.trim(),
    marks: unratable ? [] : marks,
    client_ts: clientTs,
    ms_on_subject: Math.max(0, Math.round(msOnSubject)),
  };
}

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function button(className, key, text) {
  const element = html("button", className);
  element.type = "button";
  element.append(html("kbd", "", key), ` ${text}`);
  // a button that kept the focus would take Space and Enter for itself
  element.addEventListener("click", () => element.blur());
  return element;
}

function rank(protocol, value) {
  const values = protocol.scale.map((level) => level.value);
  if (value === Math.min(...values)) return "low";
  if (value === Math.max(...values)) return "high";
  return "mid";
}

// The rows of the rating bar: per criterion its scale buttons, each with its key, and the
// key that makes its mark class active, with the number of marks of the class.
export class RatingBar {
  // `root` is the empty #criteria element; onValue(criterionId, value), onClass(markClass).
  constructor(root, protocol, { onValue, onClass }) {
    this.root = root;
    this.protocol = protocol;
    this.levels = [];
    this.classes = [];
    this.rows = [];
    root.replaceChildren();
    for (const criterion of protocol.criteria) {
      const line = html("div", "criterion");
      line.dataset.criterion = criterion.id;
      const label = html("span", "criterion-label");
      // says what the row lacks after a refused Enter; empty otherwise
      const need = html("span", "criterion-need");
      label.append(html("b", "", criterion.id), need, ` ${criterion.label}`);
      label.title = criterion.question;
      this.rows.push({ id: criterion.id, line, need });
      const levels = html("span", "levels");
      protocol.scale.forEach((level, position) => {
        const control = button("level", criterion.keys[position], level.label);
        control.dataset.criterion = criterion.id;
        control.dataset.value = String(level.value);
        control.dataset.rank = rank(protocol, level.value);
        control.title = level.help;
        control.setAttribute("aria-pressed", "false");
        control.addEventListener("click", () => onValue(criterion.id, level.value));
        levels.append(control);
        this.levels.push(control);
      });
      line.append(label, levels, this.classControl(criterion.id, criterion.mark_key, onClass));
      root.append(line);
    }
    const other = html("div", "criterion criterion-other");
    other.dataset.criterion = "X";
    const label = html("span", "criterion-label");
    label.append(html("b", "", "X"), " Anything else worth a mark");
    const control = this.classControl("X", protocol.other_mark_key, onClass);
    other.append(label, html("span", "levels"), control);
    root.append(other);
  }

  classControl(markClass, key, onClass) {
    const control = button("class-key", key, "mark");
    control.dataset.class = markClass;
    control.setAttribute("aria-pressed", "false");
    control.addEventListener("click", () => onClass(markClass));
    const count = html("span", "mark-count");
    count.dataset.class = markClass;
    const wrap = html("span", "class-cell");
    wrap.append(control, count);
    this.classes.push({ markClass, control, count });
    return wrap;
  }

  // `lacking` maps a criterion to what its row lacks ("value" or "mark"); those rows are
  // marked. Empty unless Enter was refused.
  update(rating, counts, activeClass, lacking = {}) {
    this.root.dataset.unratable = String(Boolean(rating.unratable));
    for (const { id, line, need } of this.rows) {
      const lacks = lacking[id] ?? "";
      line.dataset.lacking = lacks;
      need.textContent = lacks ? `needs a ${lacks}` : "";
    }
    for (const control of this.levels) {
      const value = rating.values[control.dataset.criterion];
      const pressed = !rating.unratable && value === Number(control.dataset.value);
      control.setAttribute("aria-pressed", String(pressed));
    }
    for (const { markClass, control, count } of this.classes) {
      control.setAttribute("aria-pressed", String(markClass === activeClass));
      const n = counts[markClass] ?? 0;
      count.textContent = n === 1 ? "1 mark" : `${n} marks`;
      count.dataset.count = String(n);
    }
  }
}

// The reasons a subject can be unratable for, numbered; shown after `u`.
export class ReasonPicker {
  constructor(root, reasons, { onPick, onCancel }) {
    this.root = root;
    this.reasons = reasons;
    this.onPick = onPick;
    this.onCancel = onCancel;
    this.position = 0;
    this.items = [];
    const box = html("div", "dialog");
    box.append(html("h2", "", "Unratable: why?"));
    const list = html("ol", "reasons");
    reasons.forEach((reason, position) => {
      const item = button("reason", String(position + 1), reason);
      item.dataset.reason = reason;
      item.addEventListener("click", () => this.pick(position));
      const entry = html("li");
      entry.append(item);
      list.append(entry);
      this.items.push(item);
    });
    const foot = html("p", "dialog-foot");
    foot.append(
      html("kbd", "", "1"),
      ` to `,
      html("kbd", "", String(reasons.length)),
      " or ",
      html("kbd", "", "↑"),
      html("kbd", "", "↓"),
      " and ",
      html("kbd", "", "Enter"),
      " choose; ",
      html("kbd", "", "Esc"),
      " cancels",
    );
    box.append(list, foot);
    root.replaceChildren(box);
    root.addEventListener("click", (event) => {
      if (event.target === root) this.cancel();
    });
  }

  get open() {
    return !this.root.hidden;
  }

  show(current = null) {
    this.position = Math.max(0, this.reasons.indexOf(current));
    this.root.hidden = false;
    this.reflect();
  }

  close() {
    this.root.hidden = true;
  }

  reflect() {
    this.items.forEach((item, position) => {
      item.setAttribute("aria-selected", String(position === this.position));
    });
  }

  move(by) {
    const n = this.reasons.length;
    this.position = (((this.position + by) % n) + n) % n;
    this.reflect();
  }

  pick(position) {
    if (position < 0 || position >= this.reasons.length) return;
    this.close();
    this.onPick(this.reasons[position]);
  }

  choose() {
    this.pick(this.position);
  }

  cancel() {
    this.close();
    this.onCancel();
  }
}
