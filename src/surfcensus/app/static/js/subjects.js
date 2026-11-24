// The subject pages of a run with functional data (templates/subjects.html and subject.html;
// step 3 of docs/epi_interface.md): every subject of the run with the state of its surfaces
// and of its functional runs, and one subject with every session and run of it, each run with
// a link into the EPI page at that run. Both ask again while packs are still being made
// behind the app. They write nothing and ask nothing of another host.
//
// Names follow the shapes of app/epi_schemas.py, as tests/test_frontend.py reads them:
// `listing` is a SubjectsOut, `subjectRow` a SubjectRowOut, `subjectPage` a SubjectPageOut,
// `funcSession` a FuncSessionOut, `funcRun` a FuncRunOut, `funcFrame` a FuncFrameOut,
// `corrections` a CorrectionsInfo and `statement` one of its statements.

import * as api from "./api.js";

const byId = (id) => document.getElementById(id);
const main = () => document.querySelector("main[data-api]");

const SURFACE_TEXT = {
  rated: "rated",
  unratable: "unratable",
  rendered: "rendered, not rated",
  "render failed": "render failed",
  "not rendered": "not rendered",
  none: "no FreeSurfer subject",
};

const RULE_TEXT = {
  acquisition_time: "the first run acquired (acquisition times of the raw sidecars)",
  name_order: "the first run by name (no acquisition time for every run)",
};

const STREAM_RULE_TEXT = {
  "acquisition time": "by acquisition date",
  "session label": "by session label",
};

const AFFINE_TEXT = {
  header: "header, checked against the reference image",
  qform: "qform, as the reference image has it",
  reference: "the reference image's own",
  "header, unchecked": "header, unchecked (no reference image)",
};

export function plural(n, one, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}

function html(name, className = "", text = "") {
  const element = document.createElement(name);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

function link(href, text, className = "") {
  const element = html("a", className, text);
  element.href = href;
  return element;
}

function cell(line, content, className = "") {
  const td = html("td", className);
  if (content instanceof Node) td.append(content);
  else td.textContent = content;
  line.append(td);
  return td;
}

export function surfaceText(subjectRow) {
  const text = SURFACE_TEXT[subjectRow.surface] || subjectRow.surface;
  return subjectRow.surface_reason ? `${text}: ${subjectRow.surface_reason}` : text;
}

export function sessionName(funcSession) {
  return funcSession.label ? `ses-${funcSession.label}` : "no session label";
}

// What a correction statement says and where it was read, as the EPI page says it.
export function statementText(statement, applied, notApplied, unknown) {
  const said =
    statement.state === "applied" ? applied : statement.state === "not applied" ? notApplied : unknown;
  let from;
  if (statement.source) {
    const file = statement.source.replace(/_/g, " ");
    from = statement.words ? `${file}: "${statement.words}"` : file;
  } else if (statement.note) from = statement.note;
  else from = statement.state === "unknown" ? "nothing read" : "stated by no file";
  return `${said} (${from})`;
}

// The corrections the pipeline applied to the preprocessed series of a run, one per line.
export function correctionLines(corrections) {
  if (!corrections) return ["not recorded"];
  const lines = [
    statementText(corrections.motion, "head-motion correction", "no head-motion correction", "head-motion correction not stated"),
    statementText(
      corrections.sdc,
      `distortion correction, ${corrections.sdc_method || "method not stated"}`,
      "no distortion correction",
      "distortion correction not stated",
    ),
    statementText(corrections.slice_timing, "slice-timing correction", "no slice-timing correction", "slice-timing correction not stated"),
  ];
  const statement = corrections.registration;
  if (statement.words) {
    // fMRIPrep's words name the degrees of freedom already ("... BBR) - 6 dof"); say them
    // once, as the EPI page does (js/epi/epi.js)
    const stated = corrections.dof && new RegExp(`\\b${corrections.dof}\\s*dof\\b`).test(statement.words);
    const dof = corrections.dof && !stated ? `, ${corrections.dof} dof` : "";
    const file = statement.source ? statement.source.replace(/_/g, " ") : "stated by no file";
    lines.push(`registration ${statement.words}${dof} (${file})`);
  } else if (statement.note) {
    lines.push(`registration not stated (${statement.note})`);
  }
  return lines;
}

// What the pipeline applied to each run, as the page lists it: the lines of correctionLines,
// and a line for a registration that is not rigid.
function runCorrections(funcRun) {
  const lines = correctionLines(funcRun.corrections);
  if (funcRun.rigid === false) lines.push("affine registration, not rigid: used as stored");
  return lines;
}

// The corrections of a session's runs, said once for the session where every run has them:
// `common` are the lines of every run (each with its source), `own` the other lines of each
// run, in the order of `runs`. A session of one run says them for the run.
export function sharedCorrections(runs) {
  const lines = runs.map(runCorrections);
  if (runs.length < 2) return { common: [], own: lines };
  const common = lines[0].filter((line) => lines.every((other) => other.includes(line)));
  return { common, own: lines.map((own) => own.filter((line) => !common.includes(line))) };
}

export function frameText(funcFrame, limit) {
  if (!funcFrame) return "not checked";
  const text = `${funcFrame.max_mm.toFixed(2)} mm at most, ${funcFrame.mean_mm.toFixed(2)} mean`;
  if (funcFrame.flagged) return `${text}: flagged, beyond the limit${limit ? ` of ${limit} mm` : ""}`;
  return text;
}

// The state of a run as the page says it: what the catalog records, which the heading of the
// column says once (STATE_HEADING).
export const STATE_HEADING = "state (recorded)";

export function stateText(funcRun) {
  switch (funcRun.state) {
    case "packed":
      return "packed";
    case "pending":
      return "waiting for its pack";
    case "failed":
      return `pack failed${funcRun.reason ? `: ${funcRun.reason}` : ""}`;
    case "not selected":
      return "not selected (--epi-runs)";
    default:
      return `cannot be shown${funcRun.reason ? `: ${funcRun.reason}` : ""}`;
  }
}

function setPacking(nPending) {
  const box = byId("subjects-packing");
  box.hidden = nPending === 0;
  if (nPending > 0) {
    box.textContent =
      `Packs are still being made: ${plural(nPending, "run waits for its pack", "runs wait for their packs")}. ` +
      "This page asks again by itself.";
  }
}

function say(text) {
  const box = byId("subjects-message");
  box.textContent = text;
  box.hidden = !text;
}

// --- the list ---------------------------------------------------------------------------

// The counts of a row of the list, by the heading of their column (templates/subjects.html).
export const LIST_COLUMNS = [
  ["sessions", "n_sessions"],
  ["runs", "n_runs"],
  ["packed", "n_packed"],
  ["waiting", "n_pending"],
  ["failed", "n_failed"],
  ["flagged", "n_flagged"],
];

function drawList(listing) {
  const body = byId("subject-rows").tBodies[0];
  const lines = listing.subjects.map((subjectRow) => {
    const line = html("tr", subjectRow.n_flagged ? "flagged-row" : "");
    line.dataset.key = subjectRow.key;
    line.dataset.kind = subjectRow.kind;
    cell(line, link(subjectRow.url, subjectRow.label || subjectRow.key));
    cell(line, subjectRow.dataset);
    cell(line, surfaceText(subjectRow), `surface-${subjectRow.surface.replace(/ /g, "-")}`);
    for (const [heading, field] of LIST_COLUMNS) {
      const flagged = heading === "flagged" && subjectRow[field] ? "flagged" : "";
      cell(line, String(subjectRow[field]), flagged).dataset.column = heading;
    }
    cell(line, subjectRow.epi_url ? link(subjectRow.epi_url, "open the stream") : "no functional runs");
    return line;
  });
  body.replaceChildren(...lines);
  say(listing.subjects.length ? "" : "The run has no subject.");
  setPacking(listing.n_pending);
}

// --- one subject ------------------------------------------------------------------------

function drawSurface(subjectPage) {
  const subjectRow = subjectPage.subject;
  const name = subjectRow.label || subjectRow.key;
  byId("subject-name").textContent = name;
  document.title = `surfcensus: ${name}`;
  const reviewLink = byId("review-link");
  reviewLink.hidden = !subjectPage.review_url;
  if (subjectPage.review_url) reviewLink.href = subjectPage.review_url;
  byId("subjects-link").href = subjectPage.index_url;
  byId("rate-link").href = subjectPage.rate_url;
  byId("epi-index-link").href = subjectPage.epi_index_url;
  const box = byId("surface");
  const parts = [];
  if (subjectRow.kind === "subject") {
    const line = html("p", "", `Surfaces: ${surfaceText(subjectRow)}. `);
    if (subjectPage.review_url) line.append(link(subjectPage.review_url, "the surfaces of this subject on the review screen"));
    parts.push(line);
  } else {
    parts.push(html("p", "", "No FreeSurfer subject: these sessions are shown without surfaces to rate."));
  }
  const runs = plural(subjectRow.n_runs, "run");
  const summary = subjectRow.n_sessions
    ? `${plural(subjectRow.n_sessions, "functional session")}, ${runs}: ${subjectRow.n_packed} packed, ` +
      `${subjectRow.n_pending} waiting, ${subjectRow.n_failed} failed, ${subjectRow.n_flagged} flagged.`
    : "No functional runs: no functional session of the run uses this subject.";
  const line = html("p", "", `${summary} `);
  if (subjectRow.epi_url) line.append(link(subjectRow.epi_url, "open the EPI stream"));
  parts.push(line);
  box.replaceChildren(...parts);
}

// What is said after the anchor's name. An anchor whose pack is still being made is waiting,
// not left out: "not packed" is said only of an anchor that is not packed and is not waiting.
export function anchorSuffix(funcSession) {
  if (funcSession.anchor_packed) return "";
  const anchor = funcSession.runs.find((funcRun) => funcRun.run_uid === funcSession.anchor_run_uid);
  if (anchor && anchor.state === "pending") return " (waiting for its pack)";
  return " (not packed: its registration places the raw stream all the same)";
}

// The place of a session in its participant's stream. The stream counts every present
// session of the participant (stream_sessions); a page can hold only some of them, when some
// sessions use a session-wise FreeSurfer subject (sub-X_ses-Y) and the others sub-X.
export function streamPlace(funcSession, onPage) {
  const total = funcSession.stream_sessions;
  const order = `sessions ordered ${STREAM_RULE_TEXT[funcSession.stream_rule] || funcSession.stream_rule}`;
  if (total <= 1) return `the only session; ${order}`;
  const place = `session ${funcSession.stream_order + 1} of ${total}`;
  if (onPage >= total) return `${place}; ${order}`;
  const here = `${onPage} of them ${onPage === 1 ? "is" : "are"} on this page`;
  return `${place} in this participant's stream; ${here}; ${order}`;
}

// The runs of a session that the frame check flagged, as the catalog records it (the flag of
// the run, under the session's own limit, not a limit of the page).
export function flaggedRuns(funcSession) {
  return funcSession.runs.filter((funcRun) => Boolean(funcRun.frame && funcRun.frame.flagged));
}

function sessionFacts(funcSession, onPage, common) {
  const facts = html("dl", "session-facts");
  const fact = (term, text) => facts.append(html("dt", "", term), html("dd", "", text));
  fact("in the stream", streamPlace(funcSession, onPage));
  if (funcSession.fs_subject) fact("FreeSurfer subject", funcSession.fs_subject);
  const anchorName = funcSession.anchor_name || funcSession.anchor_run_uid;
  let anchorText = anchorName ? `${anchorName}${anchorSuffix(funcSession)}` : "none";
  if (funcSession.anchor_rule) anchorText += `; chosen as ${RULE_TEXT[funcSession.anchor_rule] || funcSession.anchor_rule}`;
  if (funcSession.anchor_note) anchorText += `; ${funcSession.anchor_note}`;
  fact("anchor", anchorText);
  fact(
    "registration",
    "one rigid placement of the raw data for the whole session, that of the anchor run's stored registration",
  );
  const flagged = flaggedRuns(funcSession);
  const checks = [];
  if (flagged.length) {
    checks.push(
      `${plural(flagged.length, "run")} flagged (raw data): ${flagged.map((funcRun) => funcRun.name || funcRun.run_uid).join(", ")}` +
        ` would lie more than ${funcSession.frame_limit_mm === null ? "the limit" : `${funcSession.frame_limit_mm} mm`} from where the session's registration puts ` +
        `${flagged.length === 1 ? "it" : "them"}: the head moved that far between runs, or a stored registration is off`,
    );
  } else {
    checks.push("no run flagged");
  }
  if (funcSession.anchor_is_outlier) {
    checks.push("the anchor is the odd one (raw data): every other run is flagged against it and they agree with each other");
  }
  if (funcSession.shared_frame_plausible === false && !funcSession.anchor_is_outlier) {
    checks.push("a shared frame of the runs is not plausible");
  }
  fact("frame check", `${checks.join("; ")}.`);
  if (common.length) {
    const list = html("ul", "corrections");
    for (const text of common) list.append(html("li", "", text));
    const dd = html("dd");
    dd.append(list);
    facts.append(html("dt", "", "preprocessing applied to every run"), dd);
  }
  if (funcSession.problem) fact("problem", funcSession.problem);
  if (!funcSession.shown) fact("stream", "not in the stream yet: no session.json of it in the run directory");
  return facts;
}

// The address of the EPI page at a run, with the subject page it is opened from, so that the
// EPI page can link back to it (#run=<run_uid>&subject=<key>).
export function runLink(epiUrl, key) {
  return `${epiUrl}&subject=${encodeURIComponent(key)}`;
}

// Whether a run of the session has corrections of its own beside those said for the session:
// the column of them is left out when none has.
export function ownCorrectionsShown(own) {
  return own.some((lines) => lines.length > 0);
}

function runTable(funcSession, key, own) {
  const table = html("table", "subjects runs");
  const head = html("tr");
  const columns = ["#", "run", STATE_HEADING, "volumes", "TR", "displacement under the session's registration", "raw affine"];
  const ownShown = ownCorrectionsShown(own);
  if (ownShown) columns.push("preprocessing applied");
  for (const name of columns) {
    const th = html("th", "", name);
    th.scope = "col";
    head.append(th);
  }
  table.createTHead().append(head);
  const body = table.createTBody();
  funcSession.runs.forEach((funcRun, at) => {
    const flagged = Boolean(funcRun.frame && funcRun.frame.flagged);
    const line = html("tr", flagged ? "flagged-row" : "");
    line.dataset.run = funcRun.run_uid;
    line.dataset.state = funcRun.state;
    cell(line, funcRun.acquisition_order === null ? "" : String(funcRun.acquisition_order + 1));
    // the name of a run in the stream is the link into the EPI page at that run
    const runName = funcRun.name || funcRun.run_uid;
    const name = cell(
      line,
      funcRun.epi_url
        ? link(runLink(funcRun.epi_url, key), runName, "run-name run-link")
        : html("span", "run-name", runName),
      "run-cell",
    );
    if (funcRun.epi_url) name.firstChild.title = "open the EPI page at this run";
    if (funcRun.anchor) name.append(html("span", "anchor-badge", "anchor"));
    if (flagged) name.append(html("span", "flag-badge", "flagged"));
    if (!funcRun.epi_url && funcRun.state === "packed") {
      name.append(html("br"), html("span", "hint", "not in the stream yet"));
    }
    const state = cell(line, stateText(funcRun), `wrap state-${funcRun.state.replace(/ /g, "-")}`);
    if (funcRun.detail) state.append(html("br"), html("span", "hint", funcRun.detail));
    cell(line, funcRun.n_volumes === null ? "" : String(funcRun.n_volumes));
    cell(line, funcRun.tr_s === null ? "" : `${funcRun.tr_s} s`);
    cell(line, frameText(funcRun.frame, funcSession.frame_limit_mm), funcRun.frame && funcRun.frame.flagged ? "flagged wrap" : "wrap");
    cell(line, funcRun.raw_affine ? AFFINE_TEXT[funcRun.raw_affine] || funcRun.raw_affine : "", funcRun.raw_affine === "header, unchecked" ? "flagged wrap" : "wrap");
    if (ownShown) {
      const list = html("ul", "corrections");
      for (const text of own[at]) {
        list.append(html("li", text.startsWith("affine registration, not rigid") ? "flagged" : "", text));
      }
      if (!own[at].length) list.append(html("li", "hint", "as for every run of the session (above)"));
      cell(line, list, "wrap corrections-cell");
    }
    body.append(line);
  });
  const wrap = html("div", "table-wrap runs-wrap");
  wrap.append(table);
  return wrap;
}

function drawPage(subjectPage) {
  drawSurface(subjectPage);
  const counts = {};
  for (const funcSession of subjectPage.sessions) {
    counts[funcSession.participant_uid] = (counts[funcSession.participant_uid] || 0) + 1;
  }
  const sections = subjectPage.sessions.map((funcSession) => {
    const section = html("section", "subject-session");
    section.dataset.session = funcSession.session_uid;
    const title = html("h2", "", sessionName(funcSession));
    if (funcSession.participant) title.append(html("span", "session-of", ` ${funcSession.participant}`));
    const shared = sharedCorrections(funcSession.runs);
    section.append(title, sessionFacts(funcSession, counts[funcSession.participant_uid], shared.common));
    if (funcSession.runs.length) section.append(runTable(funcSession, subjectPage.subject.key, shared.own));
    else section.append(html("p", "hint", "No functional run is listed in this session."));
    return section;
  });
  byId("sessions").replaceChildren(...sections);
  setPacking(subjectPage.subject.n_pending);
}

// --- both -------------------------------------------------------------------------------

async function load() {
  const root = main();
  const poll = Number(root.dataset.poll || 0) * 1000;
  const one = root.dataset.page === "one";
  let pending = 0;
  try {
    if (one) {
      const subjectPage = await api.getSubjectInfo(root.dataset.key);
      drawPage(subjectPage);
      pending = subjectPage.subject.n_pending;
    } else {
      const listing = await api.getSubjects();
      drawList(listing);
      pending = listing.n_pending;
    }
    root.dataset.state = "ready";
  } catch (failure) {
    if (failure instanceof api.ApiError) {
      say(
        failure.status === 403
          ? `The subject pages are closed while you still have subjects to rate. (${failure.message})`
          : failure.message,
      );
      root.dataset.state = "error";
      return;
    }
    say("The server does not answer; this page asks again.");
    pending = 1;
  }
  if (pending > 0 && poll > 0) setTimeout(load, poll);
}

if (typeof document !== "undefined") load();
