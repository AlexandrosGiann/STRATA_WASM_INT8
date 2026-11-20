// node tests/run_na3d_wasm.mjs <qkv.bin> <out.bin> heads D H W [threads]
// Runs the WASM NA3D kernel (web/engine/strata_mt.js) on a [tokens][3][heads][64] float32 buffer.
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const [,, inp, outp, heads, D, H, W, thr] = process.argv;
const StrataEngine = require('../web/engine/strata_mt.js');
const M = await StrataEngine();
const nthreads = M._st_init(+(thr || 4));
const qkv = new Float32Array(fs.readFileSync(inp).buffer.slice(0));
const n = (+D) * (+H) * (+W) * (+heads) * 64;
const pq = M._st_malloc(qkv.byteLength), po = M._st_malloc(n * 4);
M.HEAPF32.set(qkv, pq >> 2);
const t = performance.now();
M._st_na3d(pq, po, +heads, +D, +H, +W);
const ms = performance.now() - t;
fs.writeFileSync(outp, Buffer.from(M.HEAPF32.slice(po >> 2, (po >> 2) + n).buffer));
console.log(JSON.stringify({ ms, nthreads }));
process.exit(0);
