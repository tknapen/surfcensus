// The index of the EPI movies of a run directory (templates/epi_index.html): every participant
// with functional data, its sessions, how many of its runs are packed, waiting, failed and
// flagged, and the link to its one stream (decision D38). While packs are still being made
// behind the app, the page says so and asks again every few seconds. It writes nothing.

const POLL_MS = 5000;

const $ = (id) => document.getElementById(id);

function cell(row, text, cls) {
  const td = document.createElement('td');
  td.textContent = text;
  if (cls) td.className = cls;
  row.append(td);
  return td;
}

function sessionsText(p) {
  if (p.sessions.length === 0) return 'none';
  return p.sessions
    .map((s) => {
      const name = s.label ? `ses-${s.label}` : 'no session';
      const state = s.shown ? `${s.n_packed} of ${s.n_packable} packed` : 'not shown yet';
      return `${name} (${state})`;
    })
    .join(', ');
}

function draw(index) {
  const body = $('participants').querySelector('tbody');
  body.replaceChildren();
  for (const p of index.participants) {
    const row = document.createElement('tr');
    row.dataset.participant = p.participant_uid;
    cell(row, p.label || p.participant_uid);
    cell(row, sessionsText(p));
    cell(row, String(p.n_runs));
    cell(row, String(p.n_packed));
    cell(row, String(p.n_pending));
    cell(row, String(p.n_failed));
    cell(row, String(p.n_flagged));
    const td = cell(row, '');
    const link = document.createElement('a');
    link.href = p.url;
    link.textContent = p.can_open ? 'open the stream' : 'nothing packed yet';
    if (!p.can_open) link.className = 'epi-not-yet';
    td.append(link);
    body.append(row);
  }
  const packing = $('index-packing');
  if (index.n_pending > 0) {
    const runs = index.n_pending === 1 ? 'run waits' : 'runs wait';
    packing.textContent = `Packs are still being made: ${index.n_pending} ${runs} for its pack. A stream shows what is packed; this page asks again by itself.`;
    packing.hidden = false;
  } else {
    packing.hidden = true;
  }
  const message = $('index-message');
  if (index.participants.length === 0) {
    message.textContent = 'The run has no participant with functional data.';
    message.hidden = false;
  } else {
    message.hidden = true;
  }
  document.querySelector('main').dataset.state = 'ready';
}

async function load() {
  const api = document.querySelector('main[data-api]').dataset.api;
  let index = null;
  try {
    const response = await fetch(api, { headers: { Accept: 'application/json' } });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const said = body && body.errors ? body.errors.map((e) => e.message).join('; ') : `status ${response.status}`;
      $('index-message').textContent = said;
      $('index-message').hidden = false;
      document.querySelector('main').dataset.state = 'error';
      return;
    }
    index = await response.json();
    draw(index);
  } catch {
    $('index-message').textContent = 'The server does not answer.';
    $('index-message').hidden = false;
  }
  if (index === null || index.n_pending > 0) setTimeout(load, POLL_MS);
}

load();
