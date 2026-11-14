// Loads chunks off the main thread: fetch, check the SHA-256 that the session shape lists,
// inflate, add the reference volume of the run and stream. The buffer of values is handed
// back without a copy. Where crypto.subtle is missing (a page that is not in a secure
// context) the digest is not checked, and the answer says so.
import { addReference } from './logic.js';

const refs = new Map();

async function sha256(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function inflate(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function load(job) {
  const t0 = performance.now();
  const response = await fetch(job.url);
  if (!response.ok) throw new Error(`${job.file}: status ${response.status}`);
  const packed = await response.arrayBuffer();
  const checked = Boolean(crypto.subtle);
  if (checked && (await sha256(packed)) !== job.sha256) {
    throw new Error(`${job.file}: the bytes are not those the session lists`);
  }
  const values = await inflate(packed);
  if (values.length !== job.count * job.nVoxels) {
    throw new Error(`${job.file}: ${values.length} bytes, expected ${job.count * job.nVoxels}`);
  }
  addReference(values, refs.get(job.refKey));
  return { values, fetched: packed.byteLength, ms: performance.now() - t0, checked };
}

self.onmessage = async (event) => {
  const job = event.data;
  if (job.type === 'ref') {
    refs.set(job.key, job.data);
    return;
  }
  if (job.type === 'drop') {
    refs.delete(job.key);
    return;
  }
  try {
    const { values, fetched, ms, checked } = await load(job);
    self.postMessage({ id: job.id, ok: true, buffer: values.buffer, fetched, ms, checked }, [values.buffer]);
  } catch (err) {
    self.postMessage({ id: job.id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
