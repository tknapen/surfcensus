// The `?` screen: the scale definitions and the criteria of the protocol, the keyboard map,
// a way back to the tour of the screen and, when the session has any, reference examples
// (tiles of calibration subjects).

import { tileUrl } from "./coords.js";

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function keys(...names) {
  const cell = html("td", "keys");
  names.forEach((name, position) => {
    if (position) cell.append(" ");
    cell.append(html("kbd", "", name));
  });
  return cell;
}

function addRow(table, keyCell, text) {
  const line = html("tr");
  line.append(keyCell, html("td", "", text));
  table.append(line);
}

// The rows of the keyboard map: [[keys...], action]. The criterion and mark keys are the
// protocol's; the rest is fixed.
export function keyboardRows(protocol) {
  const labels = protocol.scale.map((level) => level.label).join(", ");
  const rows = [
    [["Space"], "play / pause"],
    [["←", "→"], "previous / next frame (pauses)"],
    [["Shift + ←", "Shift + →"], "jump 10 frames"],
    [["[", "]"], "slower / faster playback"],
    [["m"], "synchronized or one plane at a time"],
    [["z"], "zoom 2x, centred on the box"],
  ];
  for (const criterion of protocol.criteria) {
    rows.push([criterion.keys, `${criterion.id}, ${criterion.label}: ${labels}`]);
  }
  rows.push(
    [["click"], "place a mark of the active class (pauses playback); click a mark to select it"],
    [
      [...protocol.criteria.map((criterion) => criterion.mark_key), protocol.other_mark_key],
      `active mark class: ${protocol.criteria
        .map((criterion) => criterion.id)
        .join(", ")}, X other`,
    ],
    [["Delete"], "remove the selected mark"],
    [["Ctrl/Cmd + Z"], "undo the last mark"],
    [["h"], "hide the marks while held"],
    [["s"], "hide the surface contours while held"],
    [["u"], "unratable, with a reason"],
    [["o"], "other problem (needs a note)"],
    [["n"], "note; Esc leaves the note"],
    [["Enter"], "submit and next (every criterion needs a value, or u)"],
    [["k"], "skip: nothing is stored, the subject comes back after the others"],
    [["Backspace"], "back to the subject before, rated or skipped"],
    [["?"], "this screen"],
    [["t"], "the tour of the screen"],
  );
  return rows;
}

export class Help {
  // `root` is the empty, hidden overlay element. onTour() is called for the button that
  // opens the tour of the screen; without it the screen has no such button.
  constructor(root, session, { onTour = null } = {}) {
    this.root = root;
    this.session = session;
    this.onTour = onTour;
    this.tourAllowed = true;
    this.tourButton = null;
    this.built = false;
    root.addEventListener("click", (event) => {
      if (event.target === root) this.close();
    });
  }

  get open() {
    return !this.root.hidden;
  }

  build() {
    const protocol = this.session.protocol;
    const box = html("div", "dialog dialog-wide");
    if (this.onTour) {
      const tour = html("button", "control help-tour");
      tour.type = "button";
      tour.id = "help-tour";
      tour.append(html("kbd", "", "t"), " show the tour of the screen");
      tour.addEventListener("click", () => this.onTour());
      tour.disabled = !this.tourAllowed;
      this.tourButton = tour;
      box.append(tour);
    }
    box.append(html("h2", "", "Rating scale"));
    box.append(
      html(
        "p",
        "dialog-lead",
        "Judge the surfaces, not the image. When uncertain between two levels, choose the " +
          "lower one and add a note.",
      ),
    );
    const scale = html("table", "help-table");
    for (const level of protocol.scale) {
      const line = html("tr");
      line.append(html("th", "", `${level.value} ${level.label}`), html("td", "", level.help));
      scale.append(line);
    }
    box.append(scale, html("h2", "", "Criteria"));
    const criteria = html("table", "help-table");
    for (const criterion of protocol.criteria) {
      const line = html("tr");
      line.append(
        html("th", "", `${criterion.id} ${criterion.label}`),
        html("td", "", criterion.question),
      );
      criteria.append(line);
    }
    box.append(criteria);
    if (protocol.unratable_reasons.length) {
      box.append(
        html("p", "dialog-lead", `Unratable (u): ${protocol.unratable_reasons.join("; ")}.`),
      );
    }

    if (this.session.examples.length) {
      box.append(html("h2", "", "Reference examples"));
      const gallery = html("div", "examples");
      for (const example of this.session.examples) {
        const figure = html("figure", "example");
        const stack = html("div", "example-tiles");
        for (const layer of ["gray", "contours"]) {
          const image = html("img", `tile tile-${layer}`);
          image.alt = "";
          image.loading = "lazy";
          image.src = tileUrl(
            example.media_base,
            example.tiles[layer],
            example.panel,
            example.slice,
            example.render_key,
          );
          stack.append(image);
        }
        const label = protocol.scale.find((level) => level.value === example.value);
        const caption = html("figcaption");
        caption.append(
          html("b", "", `${example.criterion}: ${label ? label.label : example.value}`),
          example.caption ? ` ${example.caption}` : "",
        );
        figure.append(stack, caption);
        gallery.append(figure);
      }
      box.append(gallery);
    }

    box.append(html("h2", "", "Keys"));
    const map = html("table", "help-table help-keys");
    for (const [names, text] of keyboardRows(protocol)) addRow(map, keys(...names), text);
    box.append(map);
    const foot = html("p", "dialog-foot");
    foot.append(html("kbd", "", "?"), " or ", html("kbd", "", "Esc"), " closes");
    box.append(foot);
    this.root.replaceChildren(box);
    this.built = true;
  }

  // Whether the button that opens the tour can be pressed.
  allowTour(allowed) {
    this.tourAllowed = allowed;
    if (this.tourButton) this.tourButton.disabled = !allowed;
  }

  show() {
    if (!this.built) this.build();
    this.root.hidden = false;
    // with the focus, the screen scrolls on the arrow keys, Space and Page Down
    this.root.tabIndex = -1;
    this.root.focus();
  }

  close() {
    this.root.hidden = true;
    this.root.blur();
  }

  toggle() {
    if (this.open) this.close();
    else this.show();
    return this.open;
  }
}
