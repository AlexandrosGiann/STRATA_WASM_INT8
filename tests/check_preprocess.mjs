// node tests/check_preprocess.mjs <city> : JS pre-processing (strata_step.mjs prepare) of the
// exported initial state vs. the network input captured from PyTorch (step 1).
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { StrataStep } from '../web/strata_step.mjs';
const require = createRequire(import.meta.url);
const city = process.argv[2] || 'athens';
const dir = `web/data/${city}/`;
const meta = JSON.parse(fs.readFileSync(dir + 'meta.json'));
const rd = (f) => new Float32Array(fs.readFileSync(dir + f).buffer.slice(0));
const M = await require('../web/engine/strata_st.js')();
const T = meta.tile, P = T * T, hw = (T / 4) ** 2;
const geom = rd('geom.bin'); let g = 0; const take = (n) => geom.subarray(g, (g += n));
const r = new StrataStep(M, { T, inMean: meta.in_mean, inVar: meta.in_var, tgtMean: meta.tgt_mean, tgtVar: meta.tgt_var, inEps: meta.in_eps, tgtEps: meta.tgt_eps,
  windFwd: take(4 * P), windInv: take(4 * P), cosLat: take(P), sinLat: take(P), ropeCos: take(hw * 64), ropeSin: take(hw * 64) });
const x = r.prepare(rd('ic.bin'), rd('forcing.bin').subarray(0, 4 * P));
fs.writeFileSync(process.argv[3], Buffer.from(x.slice().buffer));
console.log('ok');
process.exit(0);
