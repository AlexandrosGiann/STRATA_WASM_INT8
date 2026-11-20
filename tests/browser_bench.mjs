// Headless desktop-Chrome benchmark of the demo.
//   node tests/browser_bench.mjs [--city athens] [--steps 6] [--threads 4] [--bits 16] [--out logs/browser_bench.json]
// Starts web/serve.py, opens http://127.0.0.1:<port>/?bench=1..., waits for window.__result,
// and samples the resident memory (RSS) of every Chrome process every 250 ms (Linux /proc).
import { spawn, execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(ROOT, 'tools', 'package.json'));
const puppeteer = require('puppeteer-core');

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const city = arg('city', 'athens'), steps = arg('steps', '6'), threads = arg('threads', '4'), bits = arg('bits', '16'), wfile = arg('weights', 'strata_v1_int8.bin');
const out = arg('out', path.join(ROOT, 'logs', `browser_bench_${city}_t${threads}_a${bits}_${wfile.replace('.bin', '')}.json`));
const port = +arg('port', '8765');
const chrome = arg('chrome', execSync('which google-chrome || which chromium').toString().trim());

const srv = spawn('python3', [path.join(ROOT, 'web', 'serve.py'), '--port', String(port)], { stdio: ['ignore', 'pipe', 'inherit'] });
await new Promise((r) => srv.stdout.once('data', r));
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, protocolTimeout: 0, args: ['--no-sandbox'] });
const rootPid = browser.process().pid;
function treeRssMiB() {  // sum and max RSS over the Chrome process tree
  const kids = (p) => { try { return fs.readFileSync(`/proc/${p}/task/${p}/children`, 'utf8').trim().split(/\s+/).filter(Boolean).map(Number); } catch { return []; } };
  const all = [], st = [rootPid];
  while (st.length) { const p = st.pop(); all.push(p); st.push(...kids(p)); }
  let sum = 0, max = 0;
  for (const p of all) {
    try { const m = /VmRSS:\s+(\d+)/.exec(fs.readFileSync(`/proc/${p}/status`, 'utf8')); if (m) { const v = +m[1] / 1024; sum += v; max = Math.max(max, v); } } catch {}
  }
  return { sum, max };
}
let peak = { sum: 0, max: 0 };
const timer = setInterval(() => { const r = treeRssMiB(); peak = { sum: Math.max(peak.sum, r.sum), max: Math.max(peak.max, r.max) }; }, 250);
const page = await browser.newPage();
page.on('console', (m) => { const t = m.text(); if (t.startsWith('[strata]')) console.log(t); });
page.on('pageerror', (e) => console.error('pageerror', e));
page.on('workercreated', (w) => console.log('[bench] worker created', w.url()));
await page.goto(`http://127.0.0.1:${port}/?bench=1&city=${city}&steps=${steps}&threads=${threads}&bits=${bits}&wfile=${wfile}`);
await page.waitForFunction('window.__result', { timeout: 0, polling: 1000 });
const result = await page.evaluate('window.__result');
clearInterval(timer);
result.chrome = await browser.version();
const shot = arg('shot', null);
if (shot) { await page.setViewport({ width: 1100, height: 900 }); await new Promise((r) => setTimeout(r, 500)); await page.screenshot({ path: shot, fullPage: true }); console.log('screenshot', shot); }
result.rss_peak_mib = { chrome_tree_sum: peak.sum, largest_process: peak.max, note: 'Linux VmRSS sampled every 250 ms' };
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(result, null, 1));
console.log(`wrote ${out}`);
if (result.error) console.error(result.error);
else console.log(`mean step ${result.mean_step_s.toFixed(1)} s, WASM peak ${result.wasm_memory_peak_mib.toFixed(0)} MiB, Chrome RSS peak: largest process ${peak.max.toFixed(0)} MiB, tree ${peak.sum.toFixed(0)} MiB`);
await browser.close();
srv.kill();
process.exit(result.error ? 1 : 0);
