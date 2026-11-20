// node tests/run_forward_wasm.mjs <weights.bin> <dir> D H W [threads] [act_bits]
// <dir>/x.bin (13*D*H*W f32), <dir>/rope_cos.bin, rope_sin.bin ((H/4*W/4)*64 f32)
// writes <dir>/wasm_out.bin (10*D*H*W), wasm_sem{0,1,23}.bin, wasm_pix{0..3}.bin, wasm_timing.json
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { parseHeader, bindWeights } from '../web/strata_weights.mjs';
const require = createRequire(import.meta.url);
const [,, wfile, dir, D, H, W, thr = '4', bits = '16'] = process.argv;
const StrataEngine = require('../web/engine/strata_mt.js');
const M = await StrataEngine();
const nthreads = M._st_init(+thr);
M._st_set_act_bits(+bits);
const fileBuf = fs.readFileSync(wfile);
const u8 = new Uint8Array(fileBuf.buffer, fileBuf.byteOffset, fileBuf.byteLength);
const header = parseHeader(u8);
const base = M._st_malloc(u8.byteLength);
M.HEAPU8.set(u8, base);
const nb = bindWeights(M, base, header);
const mib = M._st_configure(+D, +H, +W);
if (mib < 0) throw new Error('configure failed ' + mib);
const rd = (f) => new Float32Array(fs.readFileSync(`${dir}/${f}`).buffer.slice(0));
const put = (arr) => { const p = M._st_malloc(arr.byteLength); M.HEAPF32.set(arr, p >> 2); return p; };
const x = put(rd('x.bin')), rc = put(rd('rope_cos.bin')), rs = put(rd('rope_sin.bin'));
M._st_set_rope(rc, rs);
const P = (+D) * (+H) * (+W), N = (+D) * (+H / 4) * (+W / 4);
const dbg = {};
for (const i of [0, 1, 23]) { dbg[`sem${i}`] = [M._st_malloc(N * 1024 * 4), N * 1024]; M._st_set_debug(i, dbg[`sem${i}`][0]); }
for (const i of [0, 1, 2, 3]) { dbg[`pix${i}`] = [M._st_malloc(P * 128 * 4), P * 128]; M._st_set_debug(100 + i, dbg[`pix${i}`][0]); }
const out = M._st_malloc(10 * P * 4);
const t = performance.now();
const rc2 = M._st_forward(x, out);
const ms = performance.now() - t;
if (rc2) throw new Error('forward failed ' + rc2);
const out2 = M._st_malloc(10 * P * 4); M._st_final_only(out2);
{ const a = M.HEAPF32.slice(out >> 2, (out >> 2) + 10 * P), b = M.HEAPF32.slice(out2 >> 2, (out2 >> 2) + 10 * P); let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]) || 0); console.error('final recompute max diff', d, 'alloc failures', M._st_alloc_failures()); }
const save = (f, p, n) => fs.writeFileSync(`${dir}/${f}`, Buffer.from(M.HEAPF32.slice(p >> 2, (p >> 2) + n).buffer));
save('wasm_out.bin', out, 10 * P);
for (const [k, [p, n]] of Object.entries(dbg)) save(`wasm_${k}.bin`, p, n);
const timing = { ms, nthreads, bound: nb, work_mib: mib, heap_mib: M.HEAPU8.byteLength / 2 ** 20,
  cats: Object.fromEntries(['ln','quant','gemm','qknorm','na','depth','elem','embed','adaln','final'].map((k, i) => [k, M._st_tm(i)])), semantic_ms: M._st_timing(0), pixel_ms: M._st_timing(1), attention_ms: M._st_timing(2), linear_and_rest_ms: M._st_timing(3) };
fs.writeFileSync(`${dir}/wasm_timing.json`, JSON.stringify(timing));
console.log(JSON.stringify(timing));
process.exit(0);
