// Classic Web Worker: runs the STRATA v1 INT8 forecast in WebAssembly.
// Messages in:  {type:'run', city, steps, threads, actBits, weights}
// Messages out: {type:'log'|'progress'|'step'|'done'|'error', ...}
'use strict';
const post = (m, t) => self.postMessage(m, t || []);
const log = (text) => post({ type: 'log', text });

async function fetchBin(url, Type = Float32Array) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Type(await r.arrayBuffer());
}

function stats(a, b, mask) {  // a: browser, b: reference
  let n = 0, s = 0, s2 = 0, mx = 0, sb = 0;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const d = a[i] - b[i];
    n++; s += d; s2 += d * d; sb += b[i] * b[i];
    if (Math.abs(d) > mx) mx = Math.abs(d);
  }
  return { n, bias: s / n, rmse: Math.sqrt(s2 / n), max_abs: mx, rel_l2: Math.sqrt(s2 / Math.max(sb, 1e-300)) };
}

// When this file is concatenated after the threaded engine (strata_mt_worker.js, see
// engine/build.sh) the same script also runs as each pthread worker; those are named
// 'em-pthread' and must keep emscripten's own message handler.
if (self.name !== 'em-pthread') self.onmessage = async (ev) => {
  const o = ev.data;
  if (o.type !== 'run') return;
  try {
    const W = await import('./strata_weights.mjs');
    const S = await import('./strata_step.mjs');
    const mt = typeof self.StrataEngine === 'function';  // threaded engine is prepended in strata_mt_worker.js
    if (!mt) importScripts('engine/strata_st.js');
    const t0 = performance.now();
    const M = await self.StrataEngine({ locateFile: (p) => 'engine/' + p });
    log('engine module instantiated');
    const nthreads = M._st_init(mt ? o.threads : 1);
    M._st_set_act_bits(o.actBits);
    log(`engine: ${mt ? 'pthreads' : 'single thread'} (${nthreads} thread${nthreads > 1 ? 's' : ''}), crossOriginIsolated=${self.crossOriginIsolated}, activations ${o.actBits}-bit`);

    // weights: stream straight into WASM memory
    const tw = performance.now();
    const resp = await fetch(o.weights);
    if (!resp.ok) throw new Error(`${o.weights}: HTTP ${resp.status}`);
    const total = +resp.headers.get('content-length');
    if (!total) throw new Error('server did not send Content-Length for the weight file');
    const reader = resp.body.getReader();
    async function* chunks() { for (;;) { const { done, value } = await reader.read(); if (done) return; yield value; } }
    let lastPct = -1;
    const base = await W.streamIntoWasm(M, chunks(), total, (done, tot) => {
      const pct = Math.floor((100 * done) / tot);
      if (pct !== lastPct && pct % 5 === 0) { lastPct = pct; post({ type: 'progress', what: 'weights', pct }); }
    });
    const header = W.parseHeader(M.HEAPU8.subarray(base, base + (1 << 20)));
    const nb = W.bindWeights(M, base, header);
    const weightMs = performance.now() - tw;
    log(`weights: ${(total / 2 ** 20).toFixed(1)} MiB INT8 file, ${nb} tensors bound, ${(weightMs / 1000).toFixed(1)} s`);

    // regional data
    const dir = `data/${o.city}/`;
    const meta = await (await fetch(dir + 'meta.json')).json();
    const T = meta.tile, P = T * T, steps = Math.min(o.steps, meta.steps);
    const [ic, ring, forcing, geom] = await Promise.all(['ic.bin', 'ring.bin', 'forcing.bin', 'geom.bin'].map((f) => fetchBin(dir + f)));
    const refs = {};
    for (const v of meta.ref_variants) refs[v] = await fetchBin(dir + (v === 'fp32' ? 'ref.bin' : `ref_${v}.bin`)).catch(() => null);
    const hw = (T / 4) * (T / 4);
    let g = 0;
    const take = (n) => geom.subarray(g, (g += n));
    const cfg = {
      T, inMean: meta.in_mean, inVar: meta.in_var, tgtMean: meta.tgt_mean, tgtVar: meta.tgt_var,
      inEps: meta.in_eps, tgtEps: meta.tgt_eps,
      windFwd: take(4 * P), windInv: take(4 * P), cosLat: take(P), sinLat: take(P), ropeCos: take(hw * 64), ropeSin: take(hw * 64),
    };
    const runner = new S.StrataStep(M, cfg);
    const ringIdx = Int32Array.from(meta.ring_idx);
    const R = ringIdx.length;
    const interior = new Uint8Array(P).fill(1);
    for (const k of ringIdx) interior[k] = 0;
    log(`region ${meta.city}: ${T}x${T} cells (interior ${T - 2 * meta.halo}, halo ${meta.halo}), 24 levels; work buffers ${runner.workMiB} MiB; WASM memory ${(M.HEAPU8.byteLength / 2 ** 20).toFixed(0)} MiB`);

    const refIdx = Object.fromEntries(meta.ref_channels.map((n, i) => [n, i]));
    const outCh = { T_2m: 144, U_31: 24 + 23, V_31: 48 + 23, precip_liq_surf_mass_flux: 146, precip_ice_surf_mass_flux: 145, ps: 147, PotentialTemperature_31: 23, qv_31: 120 + 23 };
    const NR = meta.ref_channels.length;
    let state = ic;
    const results = [];
    let peakHeap = M.HEAPU8.byteLength;
    log(`υπολογισμός ${steps} βημάτων· σε κινητό με 1 thread κάθε βήμα μπορεί να πάρει πολλά λεπτά (η μπάρα δείχνει την πρόοδο μέσα στο βήμα)`);
    for (let k = 1; k <= steps; k++) {
      const f = forcing.subarray((k - 1) * 4 * P, k * 4 * P);
      const ts = performance.now();
      self.__stProgress = (done, tot) => post({ type: 'sub', step: k, steps, done, total: tot, ms: performance.now() - ts });
      const { full, next, ms } = runner.step(state, f);
      if (meta.bc === 'truth') S.applyRing(next, P, ringIdx, ring.subarray((k - 1) * 145 * R, k * 145 * R));
      const stepMs = performance.now() - ts;
      state = next;
      peakHeap = Math.max(peakHeap, M.HEAPU8.byteLength);
      const fields = {}, cmp = {};
      for (const [name, ch] of Object.entries(outCh)) {
        fields[name] = full.slice(ch * P, (ch + 1) * P);
        cmp[name] = {};
        for (const [v, ref] of Object.entries(refs)) {
          if (!ref) continue;
          const r = ref.subarray(((k - 1) * NR + refIdx[name]) * P, ((k - 1) * NR + refIdx[name] + 1) * P);
          cmp[name][v] = { all: stats(fields[name], r), interior: stats(fields[name], r, interior) };
        }
      }
      const refFields = {};
      if (refs.fp32) for (const name of Object.keys(outCh)) refFields[name] = refs.fp32.slice(((k - 1) * NR + refIdx[name]) * P, ((k - 1) * NR + refIdx[name] + 1) * P);
      const timing = { step_ms: stepMs, network_ms: ms, semantic_ms: M._st_timing(0), pixel_ms: M._st_timing(1), attention_ms: M._st_timing(2) };
      results.push({ step: k, lead_min: 10 * k, timing, cmp });
      post({ type: 'step', step: k, steps, T, halo: meta.halo, timing, cmp, fields, refFields, heapMiB: M.HEAPU8.byteLength / 2 ** 20 });
    }
    post({
      type: 'done',
      summary: {
        city: meta.city, tile: T, halo: meta.halo, steps, bc: meta.bc, threads: nthreads, act_bits: o.actBits,
        engine: mt ? 'wasm-simd128+pthreads' : 'wasm-simd128', weight_file_mib: total / 2 ** 20,
        weight_load_s: weightMs / 1000, init_s: (performance.now() - t0) / 1000,
        wasm_memory_peak_mib: peakHeap / 2 ** 20, work_buffers_mib: runner.workMiB,
        torch_step_seconds: meta.torch_step_seconds, results, userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        deviceMemoryGB: navigator.deviceMemory || null,
      },
    });
  } catch (e) {
    post({ type: 'error', text: String(e && e.stack || e) });
  }
};
