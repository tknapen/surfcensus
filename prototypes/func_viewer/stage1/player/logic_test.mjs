// Checks of logic.js that need no page. Run: node logic_test.mjs (check_player.py does).
// Prints one line per check and exits with status 1 when one fails.
import {
  aboveLimit,
  addReference,
  chunkOf,
  chunkOrder,
  decodeContours,
  desiredChunks,
  diffColour,
  diffLut,
  diffToRgba,
  extractSlice,
  fitLayout,
  grayToRgba,
  panelSizesMm,
  panelToVoxel,
  planeSize,
  playbackStep,
  SPEC,
  stepRate,
  voxelToPanel,
} from './logic.js';

let failed = 0;
function check(name, ok, detail = '') {
  if (!ok) failed += 1;
  console.log(`${ok ? 'ok    ' : 'FAILED'} ${name}${detail ? `: ${detail}` : ''}`);
}
const same = (a, b) => a.length === b.length && Array.from(a).every((v, k) => v === b[k]);

// a volume whose value says where it is: value = (i + 7 * j + 31 * k) mod 251
const shape = [5, 4, 3];
const vol = new Uint8Array(60);
const value = (i, j, k) => (i + 7 * j + 31 * k) % 251;
for (let k = 0; k < 3; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 5; i++) vol[i + 5 * (j + 4 * k)] = value(i, j, k);

{
  let ok = true;
  for (let plane = 0; plane < 3; plane++) {
    const s = SPEC[plane];
    const [w, h] = planeSize(shape, plane);
    for (let index = 0; index < shape[s.axis]; index++) {
      const out = extractSlice(vol, shape, plane, index, new Uint8Array(w * h));
      for (let r = 0; r < h; r++) {
        for (let c = 0; c < w; c++) {
          const ijk = [0, 0, 0];
          ijk[s.axis] = index;
          ijk[s.h] = s.flipH ? w - 1 - c : c;
          ijk[s.v] = h - 1 - r;
          if (out[r * w + c] !== value(...ijk)) ok = false;
          const [a, b] = panelToVoxel(plane, shape, c + 0.5, r + 0.5);
          if (a !== ijk[s.h] || b !== ijk[s.v]) ok = false;
          const [x, y] = voxelToPanel(plane, shape, a, b);
          if (x !== c + 0.5 || y !== r + 0.5) ok = false;
        }
      }
    }
  }
  check('every slice of every plane holds the voxels of its place, and panel and voxel coordinates agree', ok);
}

check('sagittal has posterior at the left, coronal and axial the right side at the left',
  SPEC[0].letters === 'PASI' && !SPEC[0].flipH && SPEC[1].letters === 'RLSI' && SPEC[1].flipH && SPEC[2].letters === 'RLAP' && SPEC[2].flipH);

{
  const rgba = new Uint8ClampedArray(12);
  grayToRgba(Uint8Array.of(0, 128, 255), rgba);
  check('gray values become opaque gray pixels', same(rgba, [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]));
}

{
  const lut = diffLut(32);
  const at = (d) => Array.from(lut.subarray(3 * (d + 255), 3 * (d + 255) + 3));
  let monotone = true;
  for (let d = 1; d <= 32; d++) if (at(d)[0] < at(d - 1)[0] || at(-d)[2] < at(-d + 1)[2]) monotone = false;
  check('the difference colours: zero is dark gray, the ends are the last stops, beyond the scale they stay',
    same(at(0), [24, 24, 24]) && same(at(32), [255, 224, 144]) && same(at(-32), [166, 226, 255]) &&
    same(at(16), [214, 48, 39]) && same(at(-16), [33, 102, 172]) && same(at(255), at(32)) && same(at(-255), at(-32)) && monotone);
  check('one step of difference has a colour of its own at every scale',
    [8, 16, 32, 64, 128].every((s) => !same(diffColour(1, s), diffColour(0, s)) && !same(diffColour(-1, s), diffColour(0, s))));
  const rgba = new Uint8ClampedArray(8);
  diffToRgba(Uint8Array.of(10, 200), Uint8Array.of(26, 100), lut, rgba);
  check('a difference is raw minus preprocessed', same(rgba, [...at(-16), 255, ...at(100), 255]));
}

{
  const ref = Uint8Array.of(250, 3, 100);
  const values = Uint8Array.of(1, 2, 3, 255, 0, 128);
  const residual = values.map((v, k) => (v - ref[k % 3] + 256) & 255);
  check('adding the reference volume gives back the values, mod 256', same(addReference(residual, ref), values));
}

{
  const order = chunkOrder(10, 4);
  check('chunks are wanted from the cursor outwards, two ahead for one behind, each once',
    same(order.slice(0, 5), [4, 5, 6, 3, 7]) && new Set(order).size === 10 && order.length === 10);
  check('at the end of a run the order wraps round', same(chunkOrder(10, 9).slice(0, 4), [9, 0, 1, 8]) && same(chunkOrder(1, 0), [0]));
  check('a volume is in the chunk of its index', chunkOf(31, 32) === 0 && chunkOf(32, 32) === 1);
  const runs = [{ nChunks: 10, chunkBytes: 100 }, { nChunks: 5, chunkBytes: 100 }];
  const all = desiredChunks({ runs, run: 0, cursor: 4, streams: ['raw'], limitBytes: 1e9 });
  check('with room for everything, every chunk of every run and stream is wanted once',
    all.length === 30 && new Set(all).size === 30 && all[0] === '0|raw|4' && all[10] === '0|pre|4' && all[20] === '1|raw|0');
  const few = desiredChunks({ runs, run: 0, cursor: 4, streams: ['raw', 'pre'], limitBytes: 650 });
  check('a memory limit keeps the chunks nearest the cursor, both streams on screen',
    same(few, ['0|raw|4', '0|pre|4', '0|raw|5', '0|pre|5', '0|raw|6', '0|pre|6']));
  const tiny = desiredChunks({ runs, run: 0, cursor: 9, streams: ['raw', 'pre'], limitBytes: 1 });
  check('however small the limit, the chunk of the cursor and the next one are held', same(tiny, ['0|raw|9', '0|pre|9', '0|raw|0', '0|pre|0']));
}

{
  // 60 per second on frames that come every 16.7 ms with jitter: one volume per frame
  let due = 0;
  let shown = 0;
  for (let f = 0; f < 600; f++) {
    const now = f * (1000 / 60) + ((f * 7) % 5) - 2;
    const step = playbackStep(due, now, 60);
    if (step.advance) shown += 1;
    due = step.nextDue;
  }
  check('at 60 per second on a 60 Hz display a volume is shown at nearly every frame', shown >= 590, `${shown} of 600`);
  due = 0;
  shown = 0;
  for (let f = 0; f < 600; f++) {
    const step = playbackStep(due, f * (1000 / 60), 12);
    if (step.advance) shown += 1;
    due = step.nextDue;
  }
  check('at 12 per second a volume is shown every fifth frame', shown === 120, `${shown} in 10 s`);
  const late = playbackStep(0, 5000, 30);
  check('a clock that fell far behind starts anew and shows one volume, not a burst', late.advance && late.resynced && late.nextDue > 5000);
  check('the rate steps through the offered rates and stops at the ends', stepRate(12, 1) === 24 && stepRate(60, 1) === 60 && stepRate(1, -1) === 1);
}

check('volumes above the threshold are found, a missing value is not one', same(aboveLimit([null, 0.2, 0.51, 0.5, 2], 0.5), [2, 4]));

{
  const sizes = panelSizesMm([49, 62, 36], [3, 3, 4]);
  check('panel sizes in mm follow the grid', same(sizes.flat(), [186, 144, 147, 144, 147, 186]));
  const wide = fitLayout(sizes, 1, 1380, 500);
  const narrow = fitLayout(sizes, 1, 380, 900);
  const two = fitLayout(sizes, 2, 1380, 500);
  check('a wide window puts the planes in a row, a narrow one in a column',
    wide.panels === 'row' && narrow.panels === 'column' && two.scale < wide.scale && two.scale > 0.5,
    `scales ${wide.scale.toFixed(2)}, ${narrow.scale.toFixed(2)}, ${two.scale.toFixed(2)} px per mm`);
}

{
  // one slice, one surface: a closed square and an open line, as pack_contours.py writes them
  const lines = [[[64, 64], [128, 64], [128, 128], [64, 128], [64, 64]], [[0, 0], [-32, 16]]];
  const nSlices = 2;
  const nSurfaces = 1;
  const points = lines.flat();
  const buffer = new ArrayBuffer(16 + 4 * (nSlices * nSurfaces + 1) + 4 * (lines.length + 1) + 4 * points.length);
  new Uint32Array(buffer, 0, 4).set([nSlices, nSurfaces, lines.length, points.length]);
  new Uint32Array(buffer, 16, 3).set([0, 0, 2]); // slice 0 has no line, slice 1 has both
  new Uint32Array(buffer, 28, 3).set([0, 5, 7]);
  const delta = new Int16Array(buffer, 40, 2 * points.length);
  let n = 0;
  for (const line of lines) {
    line.forEach((p, k) => {
      delta[n++] = p[0] - (k ? line[k - 1][0] : 0);
      delta[n++] = p[1] - (k ? line[k - 1][1] : 0);
    });
  }
  const c = decodeContours(buffer, 64);
  check('a contour file is read back as lines of points in voxels',
    c.nSlices === 2 && c.lineStart[1] === 0 && c.lineStart[2] === 2 && same(c.xy.subarray(0, 10), [1, 1, 2, 1, 2, 2, 1, 2, 1, 1]) && same(c.xy.subarray(10), [0, 0, -0.5, 0.25]));
}

console.log(failed ? `${failed} checks FAILED` : 'all checks ok');
process.exit(failed ? 1 : 0);
