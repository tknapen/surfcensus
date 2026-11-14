// The routes of the rater app (surfcensus/app/schemas.py, ROUTES) and nothing else: every
// request of a page goes through `request` or `beacon` here, to a path of the table below,
// on the server the page came from.
//
// Names in the JavaScript follow the shapes of app/schemas.py, and tests/test_frontend.py
// reads the sources by them: `session` is a SessionOut, `next` a NextOut, `assignment` an
// AssignmentOut, `render` a RenderOut, `stored` a RatingOut, and so on (the table is in
// that test). An object literal after a comment `/* shape: Name */` is a body of that shape.

export const ROUTES = {
  session: ["GET", "/api/session"],
  next: ["GET", "/api/next"],
  back: ["GET", "/api/back"],
  rating: ["POST", "/api/rating"],
  skip: ["POST", "/api/skip"],
  events: ["POST", "/api/events"],
  review: ["GET", "/api/review"],
  subject: ["GET", "/api/review/{subject_uid}"],
  exportCsv: ["GET", "/api/export"],
  ratePage: ["GET", "/"],
  reviewPage: ["GET", "/review"],
  // the index of the EPI movies, the one link of the rating page to them (session.epi_url);
  // the EPI pages call their own routes (js/epi/)
  epiPage: ["GET", "/epi"],
};

const UNSENT_KEY = "surfcensus.unsent";
const RETRY_FIRST_MS = 1000;
const RETRY_MAX_MS = 10000;

// The server answered with a status of 400 or above; `errors` is the list of its ErrorsOut.
export class ApiError extends Error {
  constructor(status, errors) {
    super(errors.map((problem) => problem.message).join("; "));
    this.name = "ApiError";
    this.status = status;
    this.errors = errors;
  }
}

// The server did not answer.
export class NetworkError extends Error {
  constructor(cause) {
    super("the server did not answer");
    this.name = "NetworkError";
    this.cause = cause;
  }
}

export function routePath(name, params = {}, query = null) {
  let path = ROUTES[name][1].replace(/\{(\w+)\}/g, (_, key) => encodeURIComponent(params[key]));
  if (query) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== null && value !== undefined && value !== "") search.set(key, String(value));
    }
    const text = search.toString();
    if (text) path += `?${text}`;
  }
  return path;
}

export async function request(name, { params = {}, query = null, body = null } = {}) {
  const method = ROUTES[name][0];
  const options = { method, cache: "no-store", credentials: "same-origin", headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  let text;
  try {
    response = await fetch(routePath(name, params, query), options);
    text = await response.text();
  } catch (cause) {
    throw new NetworkError(cause);
  }
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!response.ok) {
    const errors =
      data && Array.isArray(data.errors) && data.errors.length
        ? data.errors
        : [{ field: "request", message: `the server answered ${response.status}` }];
    throw new ApiError(response.status, errors);
  }
  return data;
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const getSession = () => request("session");
export const getNext = () => request("next");
export const getBack = () => request("back");
export const getReview = (query) => request("review", { query });
export const getSubject = (subjectUid) =>
  request("subject", { params: { subject_uid: subjectUid } });

// The NextOut that is not `rendering`: asks again after `retry_after_s` while nothing is
// rendered, and when the server does not answer.
export async function settle(next, { onRendering = null, sleep = pause } = {}) {
  for (;;) {
    if (next && next.status !== "rendering") return next;
    if (next && onRendering) onRendering(next);
    // never a busy loop, whatever the answer says
    const seconds = next && next.retry_after_s > 0 ? next.retry_after_s : 2;
    await sleep(Math.max(0.25, seconds) * 1000);
    try {
      next = await getNext();
    } catch (failure) {
      if (failure instanceof ApiError) throw failure;
    }
  }
}

function storage() {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

// The rating that was posted and has no answer yet, or null. It is kept in sessionStorage
// from the moment it is sent until the server has stored or refused it, so a reload of the
// page while the server is away does not lose it.
export function unsentRating() {
  try {
    const text = storage()?.getItem(UNSENT_KEY);
    return text ? JSON.parse(text) : null;
  } catch {
    return null;
  }
}

function keepUnsent(body) {
  try {
    storage()?.setItem(UNSENT_KEY, JSON.stringify(body));
  } catch {
    // a full or closed storage: the rating is still sent, it only does not survive a reload
  }
}

function forgetUnsent() {
  try {
    storage()?.removeItem(UNSENT_KEY);
  } catch {
    // nothing to forget
  }
}

// POST /api/rating. A submission that the server could not take (no answer, or a status of
// 500 or above) is sent again until it answers; one that it refuses (ApiError below 500)
// is given up and thrown.
export async function submitRating(body, { onRetry = null, sleep = pause } = {}) {
  keepUnsent(body);
  for (let attempt = 1; ; attempt += 1) {
    try {
      const stored = await request("rating", { body });
      forgetUnsent();
      return stored;
    } catch (failure) {
      if (failure instanceof ApiError && failure.status < 500) {
        forgetUnsent();
        throw failure;
      }
      if (onRetry) onRetry(attempt, failure);
      await sleep(Math.min(RETRY_MAX_MS, RETRY_FIRST_MS * 2 ** (attempt - 1)));
    }
  }
}

// POST /api/skip: the NextOut of the subject that comes next. Sent again, like a rating,
// until the server answers; the server counts a skip once however often it arrives.
export async function skipAssignment(body, { onRetry = null, sleep = pause } = {}) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await request("skip", { body });
    } catch (failure) {
      if (failure instanceof ApiError && failure.status < 500) throw failure;
      if (onRetry) onRetry(attempt, failure);
      await sleep(Math.min(RETRY_MAX_MS, RETRY_FIRST_MS * 2 ** (attempt - 1)));
    }
  }
}

export const postEvents = (body) => request("events", { body });

// POST /api/events while the page is being left; false when the browser did not take it.
export function beaconEvents(body) {
  const blob = new Blob([JSON.stringify(body)], { type: "application/json" });
  return navigator.sendBeacon(routePath("events"), blob);
}
