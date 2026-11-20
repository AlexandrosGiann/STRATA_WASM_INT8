// One STRATA v1 time step around the WASM network core.
// Mirrors screamcast MixedPredictionAsymmetric.step / forward_from_flat and
// StrataModel._preprocess/_postprocess (third_party/STRATA, commit d56a78c):
//   normalise -> 3-D adapter -> wind rotation -> cos/sin(lat) -> network (WASM)
//   -> inverse wind rotation -> level-mean for 2-D vars -> qv softplus -> precip ReLU
//   -> denormalise -> state update (difference prediction; omega is state prediction)
// Channel layout (ranges_input / ranges_output): 6 x 24-level vars, T_2m,
// then 4 forcings (input) or 3 diagnostics (output).
export const NPROG = 145, NIN = 149, NOUT = 148, D = 24;
const QV0 = 120, OMEGA0 = 96, OMEGA1 = 120;

export class StrataStep {
  constructor(M, cfg) {
    // cfg: {T, inMean, inVar, tgtMean, tgtVar, inEps, tgtEps, windFwd, windInv, cosLat, sinLat, ropeCos, ropeSin}
    Object.assign(this, cfg);
    this.M = M;
    const P = cfg.T * cfg.T;
    this.P = P;
    const r = M._st_configure(D, cfg.T, cfg.T);
    if (r < 0) throw new Error('st_configure failed: ' + r);
    this.workMiB = r;
    this.px = M._st_malloc(13 * D * P * 4);
    this.pout = M._st_malloc(10 * D * P * 4);
    const put = (a) => { const p = M._st_malloc(a.byteLength); M.HEAPF32.set(a, p >> 2); return p; };
    M._st_set_rope(put(cfg.ropeCos), put(cfg.ropeSin));
    this.inStd = Float64Array.from(cfg.inVar, (v) => Math.fround(Math.fround(Math.sqrt(v)) + cfg.inEps));
    this.tgtStd = Float64Array.from(cfg.tgtVar, (v) => Math.fround(Math.fround(Math.sqrt(v)) + cfg.tgtEps));
    this.xnorm = new Float32Array(NIN * P);
  }

  // state: Float32Array(145*P) physical; forcing: Float32Array(4*P) (coszr, phis, sgh30, landfrac)
  // returns {full: Float32Array(148*P), next: Float32Array(145*P), ms}
  // builds the network input (13, 24, T, T) in WASM memory; returns a view of it
  prepare(state, forcing) {
    const { P, M } = this;
    const xn = this.xnorm;
    for (let c = 0; c < NIN; c++) {
      const src = c < NPROG ? state.subarray(c * P, (c + 1) * P) : forcing.subarray((c - NPROG) * P, (c - NPROG + 1) * P);
      const m = this.inMean[c], s = this.inStd[c];
      const dst = xn.subarray(c * P, (c + 1) * P);
      for (let p = 0; p < P; p++) dst[p] = (src[p] - m) / s;
    }
    // 3-D adapter (13, 24, T, T) directly into WASM memory
    const x = new Float32Array(M.HEAPF32.buffer, this.px, 13 * D * P);
    for (let v = 0; v < 6; v++)
      for (let l = 0; l < D; l++) x.set(xn.subarray((v * D + l) * P, (v * D + l + 1) * P), (v * D + l) * P);
    for (let v = 6; v < 11; v++) {  // T_2m + 4 forcings, broadcast over levels
      const ch = 144 + (v - 6);
      const src = xn.subarray(ch * P, (ch + 1) * P);
      for (let l = 0; l < D; l++) x.set(src, (v * D + l) * P);
    }
    for (let l = 0; l < D; l++) { x.set(this.cosLat, (11 * D + l) * P); x.set(this.sinLat, (12 * D + l) * P); }
    const wf = this.windFwd;
    for (let l = 0; l < D; l++) {
      const ou = (1 * D + l) * P, ov = (2 * D + l) * P;
      for (let p = 0; p < P; p++) {
        const u = x[ou + p], v = x[ov + p];
        x[ou + p] = wf[p] * u + wf[P + p] * v;
        x[ov + p] = wf[2 * P + p] * u + wf[3 * P + p] * v;
      }
    }
    return x;
  }

  step(state, forcing) {
    const { P, M } = this;
    const xn = this.xnorm;
    this.prepare(state, forcing);
    const t0 = performance.now();
    const rc = M._st_forward(this.px, this.pout);
    const ms = performance.now() - t0;
    if (rc) throw new Error('st_forward failed: ' + rc);
    const y = new Float32Array(M.HEAPF32.buffer, this.pout, 10 * D * P);
    const wi = this.windInv;
    for (let l = 0; l < D; l++) {
      const ou = (1 * D + l) * P, ov = (2 * D + l) * P;
      for (let p = 0; p < P; p++) {
        const u = y[ou + p], v = y[ov + p];
        y[ou + p] = wi[p] * u + wi[P + p] * v;
        y[ov + p] = wi[2 * P + p] * u + wi[3 * P + p] * v;
      }
    }
    // from_3d: 3-D vars keep levels, 2-D vars (T_2m, precip ice/liq, ps) = level mean
    const res = new Float32Array(NOUT * P);
    res.set(y.subarray(0, 6 * D * P), 0);
    for (let v = 6; v < 10; v++) {
      const ch = 144 + (v - 6);
      for (let p = 0; p < P; p++) {
        let s = 0;
        for (let l = 0; l < D; l++) s += y[(v * D + l) * P + p];
        res[ch * P + p] = s / D;
      }
    }
    // qv softplus constraint (normalised space, forward_from_flat)
    for (let l = 0; l < D; l++) {
      const ci = QV0 + l;
      const qm = this.inMean[ci], qs = this.inStd[ci], dm = this.tgtMean[ci], ds = this.tgtStd[ci];
      for (let p = 0; p < P; p++) {
        const qphys = xn[ci * P + p] * qs + qm;
        const dphys = res[ci * P + p] * ds + dm;
        const qsafe = Math.max(qphys, 1e-14);
        const z = (dphys / qsafe + 1) * 10;
        const sp = z > 20 ? z / 10 : Math.log1p(Math.exp(z)) / 10;  // softplus(beta=10, threshold=20)
        const dc = (1 + (sp - 1)) * qsafe - qphys;
        res[ci * P + p] = (dc - dm) / ds;
      }
    }
    // precip ReLU in physical space
    for (const ch of [145, 146]) {
      const m = this.tgtMean[ch], s = this.tgtStd[ch];
      for (let p = 0; p < P; p++) res[ch * P + p] = (Math.max(res[ch * P + p] * s + m, 0) - m) / s;
    }
    // denormalise and update state
    const full = new Float32Array(NOUT * P);
    for (let c = 0; c < NOUT; c++) {
      const m = this.tgtMean[c], s = this.tgtStd[c];
      const stateDiff = c < NPROG && !(c >= OMEGA0 && c < OMEGA1);
      for (let p = 0; p < P; p++) {
        let v = res[c * P + p] * s + m;
        if (stateDiff) v += state[c * P + p];
        if (c >= QV0 && c < QV0 + D && v < 1e-15) v = 1e-15;
        full[c * P + p] = v;
      }
    }
    return { full, next: full.slice(0, NPROG * P), ms };
  }
}

// Lateral boundary: overwrite the halo ring of `next` with the prescribed values.
// ringIdx: Int32Array of pixel indices; ringVals: Float32Array(145 * ringIdx.length)
export function applyRing(next, P, ringIdx, ringVals) {
  const n = ringIdx.length;
  for (let c = 0; c < NPROG; c++)
    for (let k = 0; k < n; k++) next[c * P + ringIdx[k]] = ringVals[c * n + k];
}
