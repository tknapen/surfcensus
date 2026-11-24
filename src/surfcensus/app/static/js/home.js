// The start page (templates/home.html): the two interfaces of the app, the surfaces and the
// EPI movies, with how far each is. It says what the rating screen's progress line says of
// the surfaces, and of the functional data only that there is some until the EPI pages are
// open to the rater. It writes nothing and asks nothing of another host.
//
// Names follow the shapes of app/schemas.py, as tests/test_frontend.py reads them: `home` is
// a HomeOut, `surfaces` a HomeSurfacesOut and `functional` a HomeFunctionalOut.

import * as api from "./api.js";

const byId = (id) => document.getElementById(id);

export function plural(n, one, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}

// What the surface card says, from a HomeSurfacesOut.
export function surfacesText(surfaces) {
  if (surfaces.n_subjects === 0) return "No subject of this run can be rated.";
  const parts = [`${surfaces.done_total} rated, ${surfaces.remaining} to go`];
  if (surfaces.skipped > 0) parts.push(`${surfaces.skipped} skipped, waiting to come back`);
  if (surfaces.rendered < surfaces.n_subjects) {
    parts.push(`${surfaces.rendered} of ${plural(surfaces.n_subjects, "subject")} rendered so far`);
  } else {
    parts.push(`${plural(surfaces.n_subjects, "subject")}`);
  }
  return `${parts.join("; ")}.`;
}

// What the EPI card says, from a HomeFunctionalOut.
export function functionalText(functional) {
  if (!functional.offered) return "This run has no functional data to show.";
  if (functional.closed) return `Closed for now: ${functional.closed}.`;
  const parts = [
    `${plural(functional.n_participants, "participant")}, ${plural(functional.n_runs, "run")}`,
    `${functional.n_packed} of ${functional.n_packable} runs that can be shown are packed`,
  ];
  if (functional.n_pending > 0) parts.push(`${functional.n_pending} being packed`);
  if (functional.n_failed > 0) parts.push(`${functional.n_failed} failed`);
  return `${parts.join("; ")}.`;
}

// A link that is shown with its address when there is one, and hidden otherwise.
function setLink(element, href) {
  element.hidden = !href;
  if (href) element.href = href;
}

function show(home) {
  const surfaces = home.surfaces;
  const functional = home.functional;
  byId("home-rater").textContent = `Rater: ${home.rater}${home.blind ? " (blind: subjects come in a shuffled order)" : ""}`;
  byId("surfaces-state").textContent = surfacesText(surfaces);
  byId("rate-open").href = surfaces.url;
  setLink(byId("review-open"), surfaces.review_url);
  byId("functional-state").textContent = functionalText(functional);
  setLink(byId("epi-open"), functional.url);
  setLink(byId("subjects-open"), functional.subjects_url);
  byId("home-functional").dataset.offered = functional.offered ? "true" : "false";
}

async function load() {
  const root = byId("home");
  const message = byId("home-message");
  let home = null;
  try {
    home = await api.getHome();
    show(home);
    message.hidden = true;
    root.dataset.state = "ready";
  } catch (failure) {
    message.textContent = `The page could not be loaded: ${failure.message}`;
    message.hidden = false;
    root.dataset.state = "error";
  }
  // ask again while subjects are rendered or runs packed behind the app, or the server was away
  const poll = home ? home.poll_s : 5;
  if (poll) setTimeout(load, poll * 1000);
}

if (typeof document !== "undefined" && byId("home")) load();
