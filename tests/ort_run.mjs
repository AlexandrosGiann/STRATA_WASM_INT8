// node tests/ort_run.mjs <model.onnx> <in.bin> <dims comma> <out.bin> [threads] [reps]
// Runs an ONNX model with onnxruntime-web 1.31.1, execution provider 'wasm'.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(ROOT, 'tools', 'package.json'));
const ortPath = require.resolve('onnxruntime-web');
const ort = await import(ortPath);
const [,, model, inp, dims, outp, thr = '4', reps = '2'] = process.argv;
ort.env.wasm.numThreads = +thr;
const sess = await ort.InferenceSession.create(fs.readFileSync(model), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
const shape = dims.split(',').map(Number);
const x = new Float32Array(fs.readFileSync(inp).buffer.slice(0));
const feeds = { [sess.inputNames[0]]: new ort.Tensor('float32', x, shape) };
let ms = [], out;
for (let r = 0; r < +reps; r++) {
  const t = performance.now();
  out = await sess.run(feeds);
  ms.push(performance.now() - t);
}
const y = out[sess.outputNames[0]];
fs.writeFileSync(outp, Buffer.from(y.data.buffer, y.data.byteOffset, y.data.byteLength));
console.log(JSON.stringify({ ms, threads: +thr, ort: ort.env.versions, out_dims: y.dims }));
process.exit(0);
