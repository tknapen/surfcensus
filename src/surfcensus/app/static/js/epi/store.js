// The volume store of the EPI page: the stills of each run on the timeline, and the chunks of
// the two streams of the whole stream held decoded within one memory limit, loaded by workers
// around the cursor and ahead of it across run and session boundaries.
import { chunkIndexOf, desiredChunks, STREAMS, STILLS, streamChunks } from './logic.js';

export async function sha256Hex(buffer) {
  if (!crypto.subtle) return null; // not a secure context: nothing can be checked
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export async function inflate(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export class Store {
  // stream: EpiStreamOut; timeline: logic.buildTimeline of its sessions
  constructor(stream, timeline, { limitBytes, nWorkers } = {}) {
    this.stream = stream;
    this.timeline = timeline;
    this.segs = timeline.segments.map((seg) => {
      const session = stream.sessions[seg.session];
      const run = session.runs[seg.run];
      const grid = session.grids.find((g) => g.grid_key === run.grid_key);
      return { session, run, grid, nVoxels: grid.shape[0] * grid.shape[1] * grid.shape[2] };
    });
    this.chunkList = streamChunks(timeline);
    this.sizes = this.chunkList.map((c) => c.count * this.segs[c.seg].nVoxels);
    this.limitBytes = limitBytes;
    this.chunks = new Map(); // 'index|stream' -> Uint8Array of the chunk's volumes
    this.inflight = new Map(); // key -> job id
    this.stillsOf = new Map(); // segment -> { raw: { ref, mean, sd }, pre: ... } or a promise of it
    this.desired = [];
    this.desiredSet = new Set();
    this.desiredKey = '';
    this.onchunk = () => {};
    this.onerror = () => {};
    this.unchecked = false; // true once a file could not be checked (no crypto.subtle)
    this.counts = { requests: 0, fetchedBytes: 0, evictions: 0, chunkLoads: 0, chunkMs: 0, errors: 0, peakDecoded: 0 };
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

  url(session, file, version) {
    return `${session.media_url}${file}?v=${encodeURIComponent(version)}`;
  }

  // A file of a pack, fetched, checked against its SHA-256 and inflated.
  async fetchFile(session, entry, version, length) {
    const response = await fetch(this.url(session, entry.file, version));
    if (!response.ok) throw new Error(`${entry.file}: status ${response.status}`);
    const packed = await response.arrayBuffer();
    this.counts.requests += 1;
    this.counts.fetchedBytes += packed.byteLength;
    const digest = await sha256Hex(packed);
    if (digest === null) this.unchecked = true;
    else if (digest !== entry.sha256) throw new Error(`${entry.file}: the bytes are not those the session lists`);
    const values = await inflate(packed);
    if (length !== undefined && values.length !== length) {
      throw new Error(`${entry.file}: ${values.length} bytes, expected ${length}`);
    }
    return values;
  }

  // The reference volume and the two stills of both streams of a segment; loaded once, kept.
  stills(seg) {
    if (!this.stillsOf.has(seg)) {
      const { session, run, nVoxels } = this.segs[seg];
      const load = async () => {
        const out = { raw: {}, pre: {} };
        await Promise.all(
          STREAMS.flatMap((s) =>
            STILLS.map(async (what) => {
              out[s][what] = await this.fetchFile(session, run.pack.streams[s][what], run.pack.token, nVoxels);
            }),
          ),
        );
        this.stillsOf.set(seg, out);
        return out;
      };
      const promise = load();
      promise.catch(() => this.stillsOf.delete(seg)); // asked again next time
      this.stillsOf.set(seg, promise);
    }
    return Promise.resolve(this.stillsOf.get(seg));
  }

  stillsNow(seg) {
    const s = this.stillsOf.get(seg);
    return s && !(s instanceof Promise) ? s : null;
  }

  // The 8-bit values of volume g of the stream (of a stream), or null while its chunk is not
  // loaded.
  volume(stream, g) {
    const c = chunkIndexOf(this.chunkList, g);
    const chunk = this.chunks.get(`${c}|${stream}`);
    if (!chunk) return null;
    const n = this.segs[this.chunkList[c].seg].nVoxels;
    const o = (g - this.chunkList[c].first) * n;
    return chunk.subarray(o, o + n);
  }

  // Says what is on screen; the store loads what is wanted around it and drops the rest.
  want(streams, g) {
    const cursor = chunkIndexOf(this.chunkList, g);
    const key = `${streams.join('+')}|${cursor}`;
    if (key !== this.desiredKey) {
      this.desiredKey = key;
      this.desired = desiredChunks({ sizes: this.sizes, cursor, streams, limitBytes: this.limitBytes });
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
      const [index, stream] = key.split('|');
      const item = this.chunkList[Number(index)];
      const stills = this.stillsNow(item.seg);
      if (!stills) {
        // the reference volume comes first; ask again when it is there
        this.stills(item.seg).then(
          () => this.pump(),
          (err) => this.onerror(String(err && err.message ? err.message : err)),
        );
        this.inflight.set(key, -1);
        Promise.resolve().then(() => this.inflight.delete(key));
        continue;
      }
      const { session, run, nVoxels } = this.segs[item.seg];
      const entry = run.pack.streams[stream].chunks[item.chunk];
      const bytes = entry.count * nVoxels;
      // room for this chunk and for every chunk that is still on its way
      this.evict(bytes + this.workers.reduce((a, w) => a + (w.busy ? w.busy.bytes : 0), 0));
      const refKey = `${item.seg}|${stream}`;
      if (!slot.refs.has(refKey)) {
        slot.worker.postMessage({ type: 'ref', key: refKey, data: stills[stream].ref });
        slot.refs.add(refKey);
      }
      const id = ++this.jobs;
      slot.busy = { id, key, bytes };
      this.inflight.set(key, id);
      slot.worker.postMessage({
        type: 'load',
        id,
        file: entry.file,
        url: this.url(session, entry.file, run.pack.token),
        sha256: entry.sha256,
        refKey,
        nVoxels,
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
    if (!message.checked) this.unchecked = true;
    this.counts.requests += 1;
    this.counts.fetchedBytes += message.fetched;
    this.counts.chunkLoads += 1;
    this.counts.chunkMs += message.ms;
    const values = new Uint8Array(message.buffer);
    const wanted = this.desiredSet.has(job.key);
    if (wanted) this.evict(values.byteLength); // the cursor may have moved since it was asked for
    if (wanted || this.decodedChunkBytes() + values.byteLength <= this.limitBytes) {
      this.chunks.set(job.key, values);
      this.counts.peakDecoded = Math.max(this.counts.peakDecoded, this.decodedChunkBytes());
      const [index, stream] = job.key.split('|');
      this.onchunk(Number(index), stream);
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
      chunksInStream: this.chunkList.length * 2,
      workers: this.workers.length,
      unchecked: this.unchecked,
      idle: this.idle(),
    };
  }
}
