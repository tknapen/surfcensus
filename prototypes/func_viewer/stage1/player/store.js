// The volume store of the page: the stills of a run, and the chunks of its two streams held
// decoded within a memory limit, loaded by workers around the cursor.
import { chunkOf, desiredChunks, STREAMS } from './logic.js';

export const DEFAULT_LIMIT_MB = 512;

export async function inflate(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export class Store {
  constructor(manifest, packUrl, version, { limitBytes, nWorkers } = {}) {
    this.manifest = manifest;
    this.packUrl = packUrl;
    this.version = version;
    this.nVoxels = manifest.grid.shape[0] * manifest.grid.shape[1] * manifest.grid.shape[2];
    this.chunkVolumes = manifest.chunk_volumes;
    this.limitBytes = limitBytes || DEFAULT_LIMIT_MB * 1024 * 1024;
    this.chunks = new Map(); // 'run|stream|chunk' -> Uint8Array of the chunk's volumes
    this.inflight = new Map(); // key -> job id
    this.stillsOf = new Map(); // run -> { raw: { ref, mean, sd }, pre: ... } or a promise of it
    this.desired = [];
    this.desiredKey = '';
    this.onchunk = () => {};
    this.onerror = () => {};
    this.counts = { requests: 0, fetchedBytes: 0, evictions: 0, chunkLoads: 0, chunkMs: 0, errors: 0 };
    this.jobs = 0;
    const n = nWorkers || Math.min(4, Math.max(2, (navigator.hardwareConcurrency || 2) - 1));
    this.workers = [];
    for (let k = 0; k < n; k++) {
      const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      const slot = { worker, busy: null, refs: new Set() };
      worker.onmessage = (event) => this.finished(slot, event.data);
      worker.onerror = (event) => this.failed(slot, event.message || 'the worker failed');
      this.workers.push(slot);
    }
  }

  url(file) {
    return `${this.packUrl}${file}?v=${this.version}`;
  }

  async fetchBin(entry) {
    const response = await fetch(this.url(entry.file));
    if (!response.ok) throw new Error(`${entry.file}: status ${response.status}`);
    const packed = await response.arrayBuffer();
    this.counts.requests += 1;
    this.counts.fetchedBytes += packed.byteLength;
    return inflate(packed);
  }

  // The reference volume and the two stills of both streams of a run; loaded once and kept.
  stills(run) {
    if (!this.stillsOf.has(run)) {
      const streams = this.manifest.runs[run].streams;
      const load = async () => {
        const out = {};
        await Promise.all(
          STREAMS.flatMap((s) =>
            ['ref', 'mean', 'sd'].map(async (what) => {
              out[s] = out[s] || {};
              out[s][what] = await this.fetchBin(streams[s][what]);
            }),
          ),
        );
        this.stillsOf.set(run, out);
        return out;
      };
      this.stillsOf.set(run, load());
    }
    return Promise.resolve(this.stillsOf.get(run));
  }

  stillsNow(run) {
    const s = this.stillsOf.get(run);
    return s && !(s instanceof Promise) ? s : null;
  }

  // The 8-bit values of volume t of a stream of a run, or null while its chunk is not loaded.
  volume(run, stream, t) {
    const c = chunkOf(t, this.chunkVolumes);
    const chunk = this.chunks.get(`${run}|${stream}|${c}`);
    if (!chunk) return null;
    const o = (t - c * this.chunkVolumes) * this.nVoxels;
    return chunk.subarray(o, o + this.nVoxels);
  }

  // Says what is on screen; the store loads what is wanted around it and drops the rest.
  want(run, streams, t) {
    const cursor = chunkOf(t, this.chunkVolumes);
    const key = `${run}|${streams.join('+')}|${cursor}`;
    if (key !== this.desiredKey) {
      this.desiredKey = key;
      const runs = this.manifest.runs.map((r) => ({
        nChunks: r.streams.raw.chunks.length,
        chunkBytes: this.chunkVolumes * this.nVoxels,
      }));
      this.desired = desiredChunks({ runs, run, cursor, streams, limitBytes: this.limitBytes });
      this.desiredSet = new Set(this.desired);
    }
    this.pump();
  }

  decodedChunkBytes() {
    let n = 0;
    for (const chunk of this.chunks.values()) n += chunk.byteLength;
    return n;
  }

  evict(room) {
    let total = this.decodedChunkBytes();
    if (total + room <= this.limitBytes) return;
    for (const key of this.chunks.keys()) {
      if (total + room <= this.limitBytes) break;
      if (!this.desiredSet.has(key)) {
        total -= this.chunks.get(key).byteLength;
        this.chunks.delete(key);
        this.counts.evictions += 1;
      }
    }
  }

  pump() {
    for (const slot of this.workers) {
      if (slot.busy) continue;
      const key = this.desired.find((k) => !this.chunks.has(k) && !this.inflight.has(k));
      if (!key) return;
      const [run, stream, chunk] = key.split('|');
      const stills = this.stillsNow(Number(run));
      if (!stills) {
        // the reference volume comes first; ask again when it is there
        this.stills(Number(run)).then(() => this.pump(), (err) => this.onerror(String(err)));
        this.inflight.set(key, -1);
        Promise.resolve().then(() => this.inflight.delete(key));
        continue;
      }
      const entry = this.manifest.runs[Number(run)].streams[stream].chunks[Number(chunk)];
      this.evict(entry.count * this.nVoxels);
      const refKey = `${run}|${stream}`;
      if (!slot.refs.has(refKey)) {
        slot.worker.postMessage({ type: 'ref', key: refKey, data: stills[stream].ref });
        slot.refs.add(refKey);
      }
      const id = ++this.jobs;
      slot.busy = { id, key };
      this.inflight.set(key, id);
      slot.worker.postMessage({
        type: 'load',
        id,
        url: this.url(entry.file),
        sha256: entry.sha256,
        refKey,
        nVoxels: this.nVoxels,
        count: entry.count,
      });
    }
  }

  finished(slot, message) {
    const job = slot.busy;
    slot.busy = null;
    if (!job || job.id !== message.id) return;
    this.inflight.delete(job.key);
    if (!message.ok) {
      this.counts.errors += 1;
      this.onerror(message.error);
      return;
    }
    this.counts.requests += 1;
    this.counts.fetchedBytes += message.fetched;
    this.counts.chunkLoads += 1;
    this.counts.chunkMs += message.ms;
    const values = new Uint8Array(message.buffer);
    if (this.desiredSet.has(job.key) || this.decodedChunkBytes() + values.byteLength <= this.limitBytes) {
      this.chunks.set(job.key, values);
      const [run, stream, chunk] = job.key.split('|');
      this.onchunk(Number(run), stream, Number(chunk));
    }
    this.pump();
  }

  failed(slot, text) {
    if (slot.busy) this.inflight.delete(slot.busy.key);
    slot.busy = null;
    this.counts.errors += 1;
    this.onerror(text);
  }

  // true when nothing that is wanted is still to come
  idle() {
    return this.inflight.size === 0 && this.desired.every((k) => this.chunks.has(k));
  }

  stats() {
    let stillBytes = 0;
    for (const s of this.stillsOf.values()) {
      if (s instanceof Promise) continue;
      for (const stream of STREAMS) for (const a of Object.values(s[stream])) stillBytes += a.byteLength;
    }
    return {
      ...this.counts,
      limitBytes: this.limitBytes,
      decodedChunkBytes: this.decodedChunkBytes(),
      stillBytes,
      chunksHeld: this.chunks.size,
      chunksWanted: this.desired.length,
      workers: this.workers.length,
      idle: this.idle(),
    };
  }
}
