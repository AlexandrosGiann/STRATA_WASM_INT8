'use strict';
// Main thread: UI, map rendering, benchmark hooks (window.__result for the headless harness).
const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const logEl = $('log');
const log = (t, cls) => { const s = document.createElement('span'); if (cls) s.className = cls; s.textContent = t + '\n'; logEl.appendChild(s); logEl.scrollTop = 1e9; console.log('[strata]', t); };

const FIELDS = [
  { key: 'T_2m', label: 'T 2 m (°C)', conv: (v) => v - 273.15, diffUnit: 'K', fmt: 3 },
  { key: 'wind', label: 'Άνεμος κατώτατου επιπέδου (m/s)', conv: (v) => v, diffUnit: 'm/s', fmt: 3 },
  { key: 'precip', label: 'Βροχή (mm/h)', conv: (v) => v * 3.6e6, diffUnit: 'mm/h', fmt: 4 },
];

function colormap(t) {  // perceptual-ish blue -> yellow ramp
  t = Math.min(1, Math.max(0, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.6 * t - 0.3)));
  const g = Math.round(255 * (0.15 + 0.75 * t));
  const b = Math.round(255 * Math.max(0, 0.75 - 0.7 * t));
  return [r, g, b];
}
function diverging(t) {  // t in [-1, 1]
  const a = Math.min(1, Math.abs(t));
  return t < 0 ? [Math.round(255 * (1 - a)), Math.round(255 * (1 - 0.6 * a)), 255] : [255, Math.round(255 * (1 - 0.6 * a)), Math.round(255 * (1 - a))];
}
function draw(canvas, data, T, lo, hi, div) {
  canvas.width = T; canvas.height = T;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(T, T);
  for (let x = 0; x < T; x++)
    for (let y = 0; y < T; y++) {
      const v = data[x * T + y];
      const c = div ? diverging(v / (hi || 1)) : colormap((v - lo) / (hi - lo || 1));
      const o = ((T - 1 - x) * T + y) * 4;  // face x up
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
}
function derived(f) {
  const n = f.T_2m.length, wind = new Float32Array(n);
  for (let i = 0; i < n; i++) wind[i] = Math.hypot(f.U_31[i], f.V_31[i]);
  return { T_2m: f.T_2m, wind, precip: f.precip_liq_surf_mass_flux };
}
function renderMaps(m) {
  const maps = $('maps');
  maps.innerHTML = '';
  const a = derived(m.fields), b = m.refFields.T_2m ? derived(m.refFields) : null;
  for (const F of FIELDS) {
    const va = a[F.key].map(F.conv), vb = b ? b[F.key].map(F.conv) : va;
    let lo = Infinity, hi = -Infinity;
    for (const v of [...va, ...vb]) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (F.key === 'precip') lo = 0;
    const d = va.map((v, i) => v - vb[i]);
    const dm = Math.max(...d.map(Math.abs));
    for (const [arr, cap, div, l, h] of [[va, `${F.label} · browser`, false, lo, hi], [vb, `${F.label} · PyTorch`, false, lo, hi], [d, `Δ (max ${dm.toPrecision(2)} ${F.diffUnit})`, true, 0, dm]]) {
      const fig = document.createElement('figure');
      const c = document.createElement('canvas');
      draw(c, arr, m.T, l, h, div);
      const fc = document.createElement('figcaption');
      fc.textContent = cap + (div ? '' : ` [${l.toFixed(1)}, ${h.toFixed(1)}]`);
      fig.append(c, fc);
      maps.appendChild(fig);
    }
  }
  $('maptitle').textContent = `Χάρτες στο +${m.step * 10} λεπτά: browser INT8 · PyTorch FP32 · διαφορά (πάνω = +x της όψης του κύβου)`;
}
const SHOW = [['T_2m', 'T 2 m (K)', 1], ['U_31', 'U κατώτ. (m/s)', 1], ['V_31', 'V κατώτ. (m/s)', 1], ['precip_liq_surf_mass_flux', 'βροχή (mm/h)', 3.6e6]];
function addRows(m) {
  const tb = $('tbl').tBodies[0];
  for (const [k, lab, sc] of SHOW) {
    const s = m.cmp[k].fp32 && m.cmp[k].fp32.interior;
    if (!s) continue;
    const tr = document.createElement('tr');
    const cells = [`+${m.step * 10}′`, lab, (s.rmse * sc).toExponential(2), (s.max_abs * sc).toExponential(2), (s.bias * sc).toExponential(2), s.rel_l2.toExponential(2), `${(m.timing.step_ms / 1000).toFixed(1)} s`];
    for (const c of cells) { const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); }
    tb.appendChild(tr);
  }
}

async function init() {
  const idx = await (await fetch('data/index.json')).json();
  for (const c of idx.cities) { const o = document.createElement('option'); o.value = c.name; o.textContent = `${c.label} (${c.lat.toFixed(2)}, ${c.lon.toFixed(2)})`; $('city').appendChild(o); }
  $('threads').value = Math.min(navigator.hardwareConcurrency || 4, 8);
  if (idx.cities.some((c) => c.name === 'athens64')) $('city').value = 'athens64';  // smallest region by default (phones)
  for (const k of ['city', 'steps', 'threads', 'bits', 'wfile']) if (qs.get(k)) $(k).value = qs.get(k);
  $('run').onclick = run;
  log(`crossOriginIsolated=${self.crossOriginIsolated} (threads ${self.crossOriginIsolated ? 'διαθέσιμα' : 'μη διαθέσιμα: single-thread'}), cores=${navigator.hardwareConcurrency}, deviceMemory=${navigator.deviceMemory || '?'} GB`);
  if (qs.get('bench')) run();
}

function run() {
  $('run').disabled = true;
  $('tbl').tBodies[0].innerHTML = '';
  const threaded = self.crossOriginIsolated && +$('threads').value > 1;
  const w = new Worker(threaded ? 'strata_mt_worker.js' : 'worker.js');
  const t0 = performance.now();
  const status = $('status');
  let phase = 'φόρτωση';
  const tick = setInterval(() => { status.textContent = `${phase} · ${((performance.now() - t0) / 60000).toFixed(1)} λεπτά από την έναρξη`; }, 1000);
  const fail = (text) => { log(text, 'err'); clearInterval(tick); window.__result = { error: text }; $('run').disabled = false; };
  w.onerror = (e) => { e.preventDefault(); fail(`σφάλμα worker: ${e.message || 'άγνωστο'} (${e.filename || ''}:${e.lineno || ''}) — πιθανόν έλλειψη μνήμης· δοκίμασε την περιοχή 64x64`); };
  w.onmessageerror = () => fail('σφάλμα μηνύματος από τον worker');
  document.addEventListener('visibilitychange', () => { if (document.hidden) log('προσοχή: η καρτέλα πήγε στο παρασκήνιο· σε κινητό ο υπολογισμός μπορεί να παγώσει — κράτα την οθόνη ανοιχτή', 'err'); });
  w.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'log') log(m.text);
    else if (m.type === 'progress') { $('bar').style.width = m.pct + '%'; phase = `βάρη ${m.pct}%`; }
    else if (m.type === 'sub') {
      $('bar').style.width = (100 * ((m.step - 1) + m.done / m.total) / m.steps).toFixed(1) + '%';
      phase = `βήμα ${m.step}/${m.steps}: block ${m.done}/${m.total} (${(m.ms / 1000).toFixed(0)} s)`;
    }
    else if (m.type === 'step') {
      $('bar').style.width = (100 * m.step / m.steps) + '%';
      const c = m.cmp.T_2m.fp32 ? ` · ΔT2m RMSE ${m.cmp.T_2m.fp32.interior.rmse.toExponential(2)} K` : '';
      log(`βήμα ${m.step}/${m.steps} (+${m.step * 10}′): ${(m.timing.step_ms / 1000).toFixed(1)} s (semantic ${(m.timing.semantic_ms / 1000).toFixed(1)} s, pixel ${(m.timing.pixel_ms / 1000).toFixed(1)} s)${c} · WASM ${m.heapMiB.toFixed(0)} MiB`);
      renderMaps(m);
      addRows(m);
    } else if (m.type === 'done') {
      const s = m.summary;
      s.wall_s = (performance.now() - t0) / 1000;
      const st = s.results.map((r) => r.timing.step_ms / 1000);
      s.mean_step_s = st.reduce((a, b) => a + b, 0) / st.length;
      log(`τέλος: ${s.steps} βήματα, μέσος χρόνος/βήμα ${s.mean_step_s.toFixed(1)} s, μέγιστη μνήμη WASM ${s.wasm_memory_peak_mib.toFixed(0)} MiB`);
      window.__result = s;
      clearInterval(tick); phase = 'τέλος';
      status.textContent = `τέλος · ${(s.wall_s / 60).toFixed(1)} λεπτά συνολικά`;
      $('run').disabled = false;
      w.terminate();
    } else if (m.type === 'error') {
      fail(m.text);
    }
  };
  w.postMessage({ type: 'run', city: $('city').value, steps: +$('steps').value, threads: +$('threads').value, actBits: +$('bits').value, weights: `weights/${$('wfile').value}` });
}
init();
