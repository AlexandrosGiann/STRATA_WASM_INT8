// STRATA v1 network core for WebAssembly (SIMD128 + optional pthreads).
//
// Computes physicsnemo Strata.forward(x, pos) for one tile:
//   x   (13, D, H, W)  normalised, wind-rotated, lat-concatenated input
//   out (10, D, H, W)  network output before the inverse wind rotation
// Everything outside the network (normalisation, wind rotation, constraints,
// state update, boundary conditions) is done by the JavaScript caller.
//
// Numerics:
//   * every large nn.Linear: INT8 weights (per-output-channel scale),
//     activations quantised per token to signed `act_bits` integers,
//     integer dot products (i32x4.dot_i16x8_s), int32 partial sums over K
//     chunks of 1024 (overflow-safe for 16-bit activations), then FP32.
//   * FP32: LayerNorm/RMSNorm, RoPE, neighbourhood + depth attention
//     (scores, softmax, weighted sum), GELU, sigmoid gates, residuals, adaLN
//     modulation, depthwise conv, bilinear upsampling, patch embeddings,
//     final 128->10 layer.
//   * 3-D neighbourhood attention: NATTEN semantics (non-causal, stride 1,
//     dilation 1, window start clamp(i - k/2, 0, L - k) per axis); each query
//     only touches its k^3 neighbours, no tile-wide attention matrix exists.
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif
#if defined(__EMSCRIPTEN_PTHREADS__) || defined(ST_NATIVE_THREADS)
#include <pthread.h>
#define ST_THREADS 1
#endif

// ----------------------------------------------------------------------------
// configuration (STRATA v1)
enum { CIN = 13, COUT = 10, E = 1024, NH = 16, HD = 64, MLPH = 4096, NSEM = 24, PATCH = 4 };
enum { EP = 128, NHP = 2, MLPP = 512, NPIX = 4, KERN = 9, DWK = 5 };

// ----------------------------------------------------------------------------
// thread pool: parallel_for(n, fn, ctx) runs fn(ctx, i0, i1) over chunks
typedef void (*range_fn)(void *ctx, int i0, int i1);
static int g_nthreads = 1;
#ifdef ST_THREADS
static pthread_t g_thr[64];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv_start = PTHREAD_COND_INITIALIZER, g_cv_done = PTHREAD_COND_INITIALIZER;
static volatile int g_gen = 0, g_busy = 0;
static range_fn g_fn;
static void *g_ctx;
static int g_n, g_grain;
static volatile int g_next;
static void run_chunks(void) {
  for (;;) {
    int i0 = __atomic_fetch_add(&g_next, g_grain, __ATOMIC_RELAXED);
    if (i0 >= g_n) break;
    int i1 = i0 + g_grain < g_n ? i0 + g_grain : g_n;
    g_fn(g_ctx, i0, i1);
  }
}
static void *worker(void *arg) {
  (void)arg;
  int seen = 0;
  for (;;) {
    pthread_mutex_lock(&g_mu);
    while (g_gen == seen) pthread_cond_wait(&g_cv_start, &g_mu);
    seen = g_gen;
    pthread_mutex_unlock(&g_mu);
    run_chunks();
    pthread_mutex_lock(&g_mu);
    if (--g_busy == 0) pthread_cond_signal(&g_cv_done);
    pthread_mutex_unlock(&g_mu);
  }
  return 0;
}
#endif
static double now_ms(void);
enum { TM_LN, TM_QUANT, TM_GEMM, TM_QKNORM, TM_NA, TM_DEPTH, TM_ELEM, TM_EMBED, TM_ADALN, TM_FINAL, TM_N };
static double g_tm[TM_N];
static int g_cat = TM_ELEM;
static void parallel_for_raw(int n, int grain, range_fn fn, void *ctx);
static void parallel_for(int n, int grain, range_fn fn, void *ctx) {
  double t = now_ms();
  parallel_for_raw(n, grain, fn, ctx);
  g_tm[g_cat] += now_ms() - t;
}
static void parallel_for_raw(int n, int grain, range_fn fn, void *ctx) {
  if (n <= 0) return;
  if (grain < 1) grain = 1;
#ifdef ST_THREADS
  if (g_nthreads > 1 && n > grain) {
    pthread_mutex_lock(&g_mu);
    g_fn = fn; g_ctx = ctx; g_n = n; g_grain = grain; g_next = 0;
    g_busy = g_nthreads - 1;
    g_gen++;
    pthread_cond_broadcast(&g_cv_start);
    pthread_mutex_unlock(&g_mu);
    run_chunks();
    pthread_mutex_lock(&g_mu);
    while (g_busy > 0) pthread_cond_wait(&g_cv_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
    return;
  }
#endif
  fn(ctx, 0, n);
}

EXPORT int st_init(int nthreads) {
#ifdef ST_THREADS
  if (nthreads > 64) nthreads = 64;
  for (int i = g_nthreads - 1; i < nthreads - 1; i++) pthread_create(&g_thr[i], 0, worker, 0);
  if (nthreads > g_nthreads) g_nthreads = nthreads;
#else
  (void)nthreads;
  g_nthreads = 1;
#endif
  return g_nthreads;
}

static int g_alloc_fail = 0;
EXPORT void *st_malloc(size_t n) {
  void *p = aligned_alloc(64, (n + 63) & ~(size_t)63);
  if (!p) g_alloc_fail++;
  return p;
}
EXPORT int st_alloc_failures(void) { return g_alloc_fail; }
EXPORT void st_free(void *p) { free(p); }

// ----------------------------------------------------------------------------
// weights: bound by name from JavaScript after it parses the file header
typedef struct { const int8_t *w; const float *scale, *bias; int out, in; } QLin;
typedef struct {
  QLin qkv, proj, gate, fc1, fc2;
} SemBlock;
typedef struct {
  QLin qkv, proj, fc1, fc2, ada;
  const float *dw_w, *dw_b;
} PixBlock;
static struct {
  const float *pe_w, *pe_b;  // semantic patch embed (1024, 13, 1, 4, 4)
  const float *ppe_w, *ppe_b;  // pixel patch embed (128, 13)
  const float *fin_w, *fin_b;  // final (10, 128)
  SemBlock sem[NSEM];
  PixBlock pix[NPIX];
} W;

static int ends_with(const char *s, const char *suf) {
  size_t a = strlen(s), b = strlen(suf);
  return a >= b && strcmp(s + a - b, suf) == 0;
}
static void bind_qlin(QLin *q, const char *leaf, const char *field, void *ptr, int out, int in) {
  (void)leaf;
  if (!strcmp(field, "weight")) { q->w = ptr; q->out = out; q->in = in; }
  else if (!strcmp(field, "weight.scale")) q->scale = ptr;
  else if (!strcmp(field, "bias")) q->bias = ptr;
}
// returns 1 if the name was recognised
EXPORT int st_bind(const char *name, void *ptr) {
  int i;
  char rest[128];
  if (!strcmp(name, "strata.backbone.patch_embed.proj.weight")) { W.pe_w = ptr; return 1; }
  if (!strcmp(name, "strata.backbone.patch_embed.proj.bias")) { W.pe_b = ptr; return 1; }
  if (!strcmp(name, "strata.pixel_patch_embed.proj.weight")) { W.ppe_w = ptr; return 1; }
  if (!strcmp(name, "strata.pixel_patch_embed.proj.bias")) { W.ppe_b = ptr; return 1; }
  if (!strcmp(name, "strata.pixel_final_layer.linear.weight")) { W.fin_w = ptr; return 1; }
  if (!strcmp(name, "strata.pixel_final_layer.linear.bias")) { W.fin_b = ptr; return 1; }
  if (sscanf(name, "strata.backbone.blocks.%d.%127s", &i, rest) == 2 && i >= 0 && i < NSEM) {
    SemBlock *b = &W.sem[i];
    struct { const char *p; QLin *q; int out, in; } m[] = {
        {"attn.qkv.", &b->qkv, 3 * E, E}, {"attn.proj.", &b->proj, E, E}, {"attn.gate_proj.", &b->gate, E, E},
        {"mlp.layers.0.", &b->fc1, MLPH, E}, {"mlp.layers.2.", &b->fc2, E, MLPH}};
    for (int k = 0; k < 5; k++) {
      size_t l = strlen(m[k].p);
      if (!strncmp(rest, m[k].p, l)) { bind_qlin(m[k].q, m[k].p, rest + l, ptr, m[k].out, m[k].in); return 1; }
    }
    return 0;
  }
  if (sscanf(name, "strata.pixel_blocks.%d.%127s", &i, rest) == 2 && i >= 0 && i < NPIX) {
    PixBlock *b = &W.pix[i];
    if (!strcmp(rest, "adaln_bilinear_dw_conv.weight")) {  // (1024,1,5,5) -> transposed (25, 1024)
      float *t = st_malloc((size_t)DWK * DWK * E * 4);
      const float *src = ptr;
      for (int c = 0; c < E; c++)
        for (int k = 0; k < DWK * DWK; k++) t[(size_t)k * E + c] = src[(size_t)c * DWK * DWK + k];
      b->dw_w = t;
      return 1;
    }
    if (!strcmp(rest, "adaln_bilinear_dw_conv.bias")) { b->dw_b = ptr; return 1; }
    struct { const char *p; QLin *q; int out, in; } m[] = {
        {"attn.qkv.", &b->qkv, 3 * EP, EP}, {"attn.proj.", &b->proj, EP, EP}, {"mlp.layers.0.", &b->fc1, MLPP, EP},
        {"mlp.layers.2.", &b->fc2, EP, MLPP}, {"adaln_bilinear_dw_proj.", &b->ada, 6 * EP, E}};
    for (int k = 0; k < 5; k++) {
      size_t l = strlen(m[k].p);
      if (!strncmp(rest, m[k].p, l)) { bind_qlin(m[k].q, m[k].p, rest + l, ptr, m[k].out, m[k].in); return 1; }
    }
  }
  (void)ends_with;
  return 0;
}
EXPORT int st_check_bound(void) {
  int missing = 0;
  if (!W.pe_w || !W.pe_b || !W.ppe_w || !W.ppe_b || !W.fin_w || !W.fin_b) missing++;
  for (int i = 0; i < NSEM; i++) {
    QLin *q[] = {&W.sem[i].qkv, &W.sem[i].proj, &W.sem[i].gate, &W.sem[i].fc1, &W.sem[i].fc2};
    for (int k = 0; k < 5; k++) if (!q[k]->w || !q[k]->scale || !q[k]->bias) missing++;
  }
  for (int i = 0; i < NPIX; i++) {
    QLin *q[] = {&W.pix[i].qkv, &W.pix[i].proj, &W.pix[i].fc1, &W.pix[i].fc2, &W.pix[i].ada};
    for (int k = 0; k < 5; k++) if (!q[k]->w || !q[k]->scale || !q[k]->bias) missing++;
    if (!W.pix[i].dw_w || !W.pix[i].dw_b) missing++;
  }
  return missing;
}

// ----------------------------------------------------------------------------
// INT8 linear: y[n][o] = sum_k x[n][k] W[o][k] + b[o]
static int g_act_bits = 16;
EXPORT void st_set_act_bits(int b) { g_act_bits = b; }
static int g_group = 0;  // 0: per-output-channel weight scales; g: scale per (row, g input columns)
EXPORT void st_set_group(int g) { g_group = g; }

typedef struct {
  const float *x; int ldx; int n, in;
  int16_t *xq; float *sx;
} QuantCtx;
static void quant_rows(void *c_, int r0, int r1) {
  QuantCtx *c = c_;
  const float qmax = (float)((1 << (g_act_bits - 1)) - 1);
  for (int r = r0; r < r1; r++) {
    const float *x = c->x + (size_t)r * c->ldx;
    float amax = 0.f;
    for (int k = 0; k < c->in; k++) { float a = fabsf(x[k]); amax = a > amax ? a : amax; }
    float s = amax > 0.f ? amax / qmax : 1.f;
    float inv = 1.f / s;
    c->sx[r] = s;
    int16_t *q = c->xq + (size_t)r * c->in;
    for (int k = 0; k < c->in; k++) q[k] = (int16_t)lrintf(x[k] * inv);
  }
}

typedef struct {
  const QLin *L; const int16_t *xq; const float *sx; int n;
  int o0, o1;  // output column range
  float *y; int ldy;
} GemmCtx;

#define KCH 1024
#ifndef NR
#define NR 2  // outputs per micro-tile (4 tokens x NR outputs); 4x2 keeps 14 SIMD registers live
#endif
#define MB 32  // tokens per task: each pair of weight rows is reused from L1 for MB/4 micro-tiles
static void gemm_rows(void *c_, int b0, int b1) {
  GemmCtx *c = c_;
  const int in = c->L->in;
  for (int mb = b0; mb < b1; mb++)
  for (int o = c->o0; o < c->o1; o += NR) {
    int on = c->o1 - o < NR ? c->o1 - o : NR;
    const int8_t *wr[NR];
    for (int j = 0; j < NR; j++) wr[j] = c->L->w + (size_t)(o + (j < on ? j : 0)) * in;
    for (int t0 = mb * MB; t0 < mb * MB + MB && t0 < c->n; t0 += 4) {
      int tn = c->n - t0 < 4 ? c->n - t0 : 4;
      const int16_t *xr[4];
      for (int t = 0; t < 4; t++) xr[t] = c->xq + (size_t)(t0 + (t < tn ? t : 0)) * in;
      double acc[4][NR];
#ifdef __wasm_simd128__
      v128_t fa[4][NR];
      for (int t = 0; t < 4; t++) for (int j = 0; j < NR; j++) fa[t][j] = wasm_f32x4_splat(0.f);
      const int kch = g_group ? g_group : KCH;
      for (int k0 = 0; k0 < in; k0 += kch) {
        int k1 = k0 + kch < in ? k0 + kch : in;
        v128_t a[4][NR];
        for (int t = 0; t < 4; t++) for (int j = 0; j < NR; j++) a[t][j] = wasm_i32x4_splat(0);
        for (int k = k0; k < k1; k += 8) {
          v128_t x0 = wasm_v128_load(xr[0] + k), x1 = wasm_v128_load(xr[1] + k);
          v128_t x2 = wasm_v128_load(xr[2] + k), x3 = wasm_v128_load(xr[3] + k);
          for (int j = 0; j < NR; j++) {
            v128_t w = wasm_i16x8_load8x8(wr[j] + k);
            a[0][j] = wasm_i32x4_add(a[0][j], wasm_i32x4_dot_i16x8(x0, w));
            a[1][j] = wasm_i32x4_add(a[1][j], wasm_i32x4_dot_i16x8(x1, w));
            a[2][j] = wasm_i32x4_add(a[2][j], wasm_i32x4_dot_i16x8(x2, w));
            a[3][j] = wasm_i32x4_add(a[3][j], wasm_i32x4_dot_i16x8(x3, w));
          }
        }
        if (g_group) {
          for (int j = 0; j < NR; j++) {
            v128_t gs = wasm_f32x4_splat(c->L->scale[(size_t)(o + (j < on ? j : 0)) * (in / g_group) + k0 / g_group]);
            for (int t = 0; t < 4; t++) fa[t][j] = wasm_f32x4_add(fa[t][j], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a[t][j]), gs));
          }
        } else {
          for (int t = 0; t < 4; t++) for (int j = 0; j < NR; j++) fa[t][j] = wasm_f32x4_add(fa[t][j], wasm_f32x4_convert_i32x4(a[t][j]));
        }
      }
      for (int t = 0; t < 4; t++)
        for (int j = 0; j < NR; j++)
          acc[t][j] = (double)wasm_f32x4_extract_lane(fa[t][j], 0) + wasm_f32x4_extract_lane(fa[t][j], 1) +
                      wasm_f32x4_extract_lane(fa[t][j], 2) + wasm_f32x4_extract_lane(fa[t][j], 3);
#else
      for (int t = 0; t < 4; t++)
        for (int j = 0; j < NR; j++) {
          double sum = 0;
          int kch = g_group ? g_group : in;
          for (int k0 = 0; k0 < in; k0 += kch) {
            int64_t part = 0;
            for (int k = k0; k < k0 + kch; k++) part += (int32_t)xr[t][k] * wr[j][k];
            sum += (double)part * (g_group ? c->L->scale[(size_t)(o + j) * (in / g_group) + k0 / g_group] : 1.0);
          }
          acc[t][j] = sum;
        }
#endif
      for (int t = 0; t < tn; t++) {
        float *y = c->y + (size_t)(t0 + t) * c->ldy;
        float s = c->sx[t0 + t];
        for (int j = 0; j < on; j++) {
          int oo = o + j;
          float ws = g_group ? 1.f : c->L->scale[oo];
          y[oo - c->o0] = (float)(acc[t][j] * (double)(s * ws)) + c->L->bias[oo];
        }
      }
    }
  }
}

// y (n x (o1-o0), row stride ldy) = x (n x in, row stride ldx) @ W[o0:o1]^T + b
// xq/sx are scratch (n*in int16, n floats); if xq_ready the caller already quantised x.
static void qlinear(const QLin *L, const float *x, int ldx, int n, float *y, int ldy, int o0, int o1,
                    int16_t *xq, float *sx, int xq_ready) {
  if (!xq_ready) {
    QuantCtx qc = {x, ldx, n, L->in, xq, sx};
    g_cat = TM_QUANT; parallel_for(n, 16, quant_rows, &qc);
  }
  GemmCtx gc = {L, xq, sx, n, o0, o1, y, ldy};
  g_cat = TM_GEMM; parallel_for((n + MB - 1) / MB, 1, gemm_rows, &gc);
}

// ----------------------------------------------------------------------------
// FP32 helpers
static inline float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
static inline float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865476f)); }
static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

static void layernorm_row(const float *x, float *y, int n) {  // no affine, eps 1e-6, biased var
  double m = 0, v = 0;
  for (int i = 0; i < n; i++) m += x[i];
  m /= n;
  for (int i = 0; i < n; i++) { double d = x[i] - m; v += d * d; }
  v /= n;
  float inv = (float)(1.0 / sqrt(v + 1e-6));
  float mf = (float)m;
  for (int i = 0; i < n; i++) y[i] = (x[i] - mf) * inv;
}
static void rmsnorm_inplace(float *x, int n, float eps) {
  double s = 0;
  for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
  float inv = (float)(1.0 / sqrt(s / n + eps));
  for (int i = 0; i < n; i++) x[i] *= inv;
}
static inline float dot64(const float *a, const float *b) {
#ifdef __wasm_simd128__
  v128_t s0 = wasm_f32x4_mul(wasm_v128_load(a), wasm_v128_load(b));
  v128_t s1 = wasm_f32x4_mul(wasm_v128_load(a + 4), wasm_v128_load(b + 4));
  v128_t s2 = wasm_f32x4_mul(wasm_v128_load(a + 8), wasm_v128_load(b + 8));
  v128_t s3 = wasm_f32x4_mul(wasm_v128_load(a + 12), wasm_v128_load(b + 12));
  for (int i = 16; i < 64; i += 16) {
    s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    s2 = wasm_f32x4_add(s2, wasm_f32x4_mul(wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8)));
    s3 = wasm_f32x4_add(s3, wasm_f32x4_mul(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12)));
  }
  v128_t s = wasm_f32x4_add(wasm_f32x4_add(s0, s1), wasm_f32x4_add(s2, s3));
  return wasm_f32x4_extract_lane(s, 0) + wasm_f32x4_extract_lane(s, 1) + wasm_f32x4_extract_lane(s, 2) +
         wasm_f32x4_extract_lane(s, 3);
#else
  float s = 0;
  for (int i = 0; i < 64; i++) s += a[i] * b[i];
  return s;
#endif
}
static inline void axpy64(float *o, float p, const float *v) {
#ifdef __wasm_simd128__
  v128_t pp = wasm_f32x4_splat(p);
  for (int i = 0; i < 64; i += 4) wasm_v128_store(o + i, wasm_f32x4_add(wasm_v128_load(o + i), wasm_f32x4_mul(pp, wasm_v128_load(v + i))));
#else
  for (int i = 0; i < 64; i++) o[i] += p * v[i];
#endif
}

// ----------------------------------------------------------------------------
// attention pieces. qkv rows: [3][heads][64] (stride 3*heads*64)
typedef struct {
  float *qkv; int heads; int n;
  const float *cos, *sin; int hw;  // RoPE (hw x 64) indexed by token % hw, or NULL
} QKNormCtx;
static void qknorm_rows(void *c_, int r0, int r1) {
  QKNormCtx *c = c_;
  const int C = c->heads * HD;
  for (int r = r0; r < r1; r++) {
    float *row = c->qkv + (size_t)r * 3 * C;
    for (int part = 0; part < 2; part++)
      for (int h = 0; h < c->heads; h++) {
        float *v = row + part * C + h * HD;
        rmsnorm_inplace(v, HD, 1e-6f);
        if (c->cos) {
          const float *cs = c->cos + (size_t)(r % c->hw) * HD, *sn = c->sin + (size_t)(r % c->hw) * HD;
          for (int i = 0; i < HD; i += 2) {
            float a = v[i], b = v[i + 1];
            v[i] = a * cs[i] - b * sn[i];
            v[i + 1] = b * cs[i + 1] + a * sn[i + 1];
          }
        }
      }
  }
}

typedef struct {
  const float *qkv; float *out; int heads; int D, H, Wd; float scale;
} NACtx;
static inline int wstart(int i, int L, int k) {
  int s = i - k / 2;
  if (s < 0) s = 0;
  if (s > L - k) s = L - k;
  return s;
}
// vectorised exp (Cephes expf polynomial, |rel err| < 2e-7 on [-87, 88])
#ifdef __wasm_simd128__
static inline v128_t exp4(v128_t x) {
  x = wasm_f32x4_min(wasm_f32x4_max(x, wasm_f32x4_splat(-87.3f)), wasm_f32x4_splat(88.3f));
  v128_t fx = wasm_f32x4_nearest(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504088896341f)));
  x = wasm_f32x4_sub(x, wasm_f32x4_mul(fx, wasm_f32x4_splat(0.693359375f)));
  x = wasm_f32x4_sub(x, wasm_f32x4_mul(fx, wasm_f32x4_splat(-2.12194440e-4f)));
  v128_t z = wasm_f32x4_mul(x, x);
  v128_t y = wasm_f32x4_splat(1.9875691500E-4f);
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.3981999507E-3f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(8.3334519073E-3f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(4.1665795894E-2f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.6666665459E-1f));
  y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(5.0000001201E-1f));
  y = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(y, z), x), wasm_f32x4_splat(1.f));
  v128_t e = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(fx), wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(y, e);
}
#endif
// p[0..n) = exp(s - max) / sum, in place
static void softmax_inplace(float *s, int n) {
  float mx = -INFINITY;
  for (int i = 0; i < n; i++) mx = s[i] > mx ? s[i] : mx;
  float sum = 0.f;
  int i = 0;
#ifdef __wasm_simd128__
  // terms with s - max < -60 (relative weight < 1e-26) are set to 0: avoids
  // denormal arithmetic, which is ~100x slower on x86, in exp and in p*v.
  v128_t m4 = wasm_f32x4_splat(mx), acc = wasm_f32x4_splat(0.f), lo = wasm_f32x4_splat(-60.f);
  for (; i + 4 <= n; i += 4) {
    v128_t z = wasm_f32x4_sub(wasm_v128_load(s + i), m4);
    v128_t e = wasm_v128_and(exp4(z), wasm_f32x4_ge(z, lo));
    wasm_v128_store(s + i, e);
    acc = wasm_f32x4_add(acc, e);
  }
  sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) + wasm_f32x4_extract_lane(acc, 2) +
        wasm_f32x4_extract_lane(acc, 3);
#endif
  for (; i < n; i++) { float z = s[i] - mx; s[i] = z >= -60.f ? expf(z) : 0.f; sum += s[i]; }
  float inv = 1.f / sum;
  for (i = 0; i < n; i++) s[i] *= inv;
}
#define NK3 (KERN * KERN * KERN)
#define NK3P ((NK3 + 3) & ~3)
// 3-D neighbourhood attention. One task = QB consecutive queries along the
// last axis (same d, same i): they share the d/i windows and their last-axis
// windows overlap, so each key/value row of the union is read once for all QB.
#define QB 4
static void na3d_rows(void *c_, int b0, int b1) {
  NACtx *c = c_;
  const int C = c->heads * HD, ld = 3 * C, HW = c->H * c->Wd, nbw = (c->Wd + QB - 1) / QB;
  static _Thread_local float sc[QB * NH * NK3P];  // per-thread scores, no malloc in the hot path
  for (int blk = b0; blk < b1; blk++) {
    int row = blk / nbw, j0 = (blk % nbw) * QB;
    int d = row / c->H, i = row % c->H;
    int nq = c->Wd - j0 < QB ? c->Wd - j0 : QB;
    int sd = wstart(d, c->D, KERN), si = wstart(i, c->H, KERN);
    int sj[QB];
    for (int q = 0; q < nq; q++) sj[q] = wstart(j0 + q, c->Wd, KERN);
    int ju0 = sj[0], ju1 = sj[nq - 1] + KERN;  // union [ju0, ju1)
    size_t tq0 = (size_t)row * c->Wd + j0;
    for (int a = 0; a < KERN; a++)
      for (int b = 0; b < KERN; b++) {
        size_t rbase = ((size_t)(sd + a) * c->H + si + b) * c->Wd;
        for (int jj = ju0; jj < ju1; jj++) {
          const float *kr = c->qkv + (rbase + jj) * ld + C;
          for (int q = 0; q < nq; q++) {
            int e = jj - sj[q];
            if (e < 0 || e >= KERN) continue;
            int m = (a * KERN + b) * KERN + e;
            const float *qr = c->qkv + (tq0 + q) * ld;
            float *sq = sc + (size_t)q * c->heads * NK3P;
            for (int h = 0; h < c->heads; h++) sq[h * NK3P + m] = dot64(qr + h * HD, kr + h * HD) * c->scale;
          }
        }
      }
    for (int q = 0; q < nq; q++)
      for (int h = 0; h < c->heads; h++) softmax_inplace(sc + ((size_t)q * c->heads + h) * NK3P, NK3);
    for (int q = 0; q < nq; q++) memset(c->out + (tq0 + q) * C, 0, (size_t)C * 4);
    for (int a = 0; a < KERN; a++)
      for (int b = 0; b < KERN; b++) {
        size_t rbase = ((size_t)(sd + a) * c->H + si + b) * c->Wd;
        for (int jj = ju0; jj < ju1; jj++) {
          const float *vr = c->qkv + (rbase + jj) * ld + 2 * C;
          for (int q = 0; q < nq; q++) {
            int e = jj - sj[q];
            if (e < 0 || e >= KERN) continue;
            int m = (a * KERN + b) * KERN + e;
            float *o = c->out + (tq0 + q) * C;
            const float *sq = sc + (size_t)q * c->heads * NK3P;
            for (int h = 0; h < c->heads; h++) axpy64(o + h * HD, sq[h * NK3P + m], vr + h * HD);
          }
        }
      }
  }
}
static int na3d_tasks(int D, int H, int Wd) { return D * H * ((Wd + QB - 1) / QB); }
// depth-axis attention: full attention over D tokens of each (i, j) column
static void depth_attn_cols(void *c_, int col0, int col1) {
  NACtx *c = c_;
  const int C = c->heads * HD, ld = 3 * C, HW = c->H * c->Wd, D = c->D;
  float sc[64];
  for (int col = col0; col < col1; col++)
    for (int dq = 0; dq < D; dq++) {
      size_t tq = (size_t)dq * HW + col;
      for (int h = 0; h < c->heads; h++) {
        const float *q = c->qkv + tq * ld + h * HD;
        float mx = -INFINITY;
        for (int dk = 0; dk < D; dk++) {
          float s = dot64(q, c->qkv + ((size_t)dk * HW + col) * ld + C + h * HD) * c->scale;
          sc[dk] = s;
          mx = s > mx ? s : mx;
        }
        (void)mx;
        softmax_inplace(sc, D);
        float o[HD] __attribute__((aligned(16)));
        memset(o, 0, sizeof o);
        for (int dk = 0; dk < D; dk++) axpy64(o, sc[dk], c->qkv + ((size_t)dk * HW + col) * ld + 2 * C + h * HD);
        memcpy(c->out + tq * C + h * HD, o, sizeof o);
      }
    }
}

// ----------------------------------------------------------------------------
// buffers sized for one tile configuration
static struct {
  int D, H, Wd, h, w, N, P;  // N semantic tokens, P pixels
  float *tok, *tmp, *qkv, *att, *gate, *hid;  // semantic
  float *px, *py, *pqkv, *patt;  // pixel
  float *cond, *cond2, *ada;  // per depth slice adaLN
  int16_t *xq; float *sx;
  const float *rcos, *rsin;
  float *dbg[256];
  double t_sem, t_pix, t_na, t_gemm;
} B;

EXPORT void st_set_debug(int which, float *dst) { if (which >= 0 && which < 256) B.dbg[which] = dst; }
EXPORT void st_set_rope(const float *cos_hw, const float *sin_hw) { B.rcos = cos_hw; B.rsin = sin_hw; }

static size_t g_bytes = 0;
static void *xalloc(size_t n) {
  g_bytes += (n + 63) & ~(size_t)63;
  void *p = st_malloc(n);
  if (p) memset(p, 0, n);
  return p;
}
#define TOKCH 512  // semantic MLP token chunk
EXPORT int st_configure(int D, int H, int Wd) {
  if (H % PATCH || Wd % PATCH || D < KERN || H / PATCH < KERN || Wd / PATCH < KERN) return -1;
  B.D = D; B.H = H; B.Wd = Wd; B.h = H / PATCH; B.w = Wd / PATCH;
  B.N = D * B.h * B.w; B.P = D * H * Wd;
  if (B.tok) {
    float *bufs[] = {B.tok, B.tmp, B.hid, B.px, B.py, B.patt, B.cond, B.cond2, B.ada, B.sx};
    for (int i = 0; i < 10; i++) st_free(bufs[i]);
    st_free(B.xq);
  }
  g_bytes = 0;
  int HW = H * Wd;
  size_t qrows = (size_t)B.P > (size_t)B.N ? B.P : B.N;
  size_t qcols = MLPH;
  B.tok = xalloc((size_t)B.N * E * 4);
  // semantic scratch (tmp | qkv | att | gate = 6 x N x E floats) is reused as the
  // pixel-stage qkv (P x 3*EP = N*16 x 384 = 6 x N x E floats): same size by construction.
  float *region = xalloc((size_t)B.N * E * 6 * 4);
  B.tmp = region;
  B.qkv = region + (size_t)B.N * E;
  B.att = region + (size_t)B.N * E * 4;
  B.gate = region + (size_t)B.N * E * 5;
  B.hid = xalloc((size_t)TOKCH * MLPH * 4);
  B.px = xalloc((size_t)B.P * EP * 4);
  B.py = xalloc((size_t)B.P * EP * 4);
  B.pqkv = region;
  B.patt = xalloc((size_t)B.P * EP * 4);
  B.cond = xalloc((size_t)HW * E * 4);
  B.cond2 = xalloc((size_t)HW * E * 4);
  B.ada = xalloc((size_t)HW * 6 * EP * 4);
  // quantisation scratch: the largest (rows x in) used at once
  size_t xq_elems = (size_t)B.N * E;
  if ((size_t)TOKCH * MLPH > xq_elems) xq_elems = (size_t)TOKCH * MLPH;
  if ((size_t)HW * E > xq_elems) xq_elems = (size_t)HW * E;
  if ((size_t)B.P * EP > xq_elems) xq_elems = (size_t)B.P * EP;
  if ((size_t)B.P * MLPP / 8 > xq_elems) xq_elems = (size_t)B.P * MLPP / 8;
  B.xq = xalloc(xq_elems * 2);
  B.sx = xalloc(qrows * 4);
  (void)qcols;
  if (!B.tok || !B.tmp || !B.qkv || !B.att || !B.gate || !B.hid || !B.px || !B.py || !B.pqkv || !B.patt || !B.cond ||
      !B.cond2 || !B.ada || !B.xq || !B.sx)
    return -2;
  return (int)(g_bytes >> 20);  // MiB of work buffers
}

// ----------------------------------------------------------------------------
// semantic stage
typedef struct { const float *x; } PECtx;
static void patch_embed_rows(void *c_, int t0, int t1) {
  const float *x = ((PECtx *)c_)->x;
  const int HW = B.H * B.Wd;
  float col[CIN * PATCH * PATCH];
  for (int t = t0; t < t1; t++) {
    int d = t / (B.h * B.w), i = (t / B.w) % B.h, j = t % B.w;
    for (int c = 0; c < CIN; c++)
      for (int a = 0; a < PATCH; a++)
        for (int b = 0; b < PATCH; b++)
          col[(c * PATCH + a) * PATCH + b] = x[(size_t)c * B.D * HW + (size_t)d * HW + (i * PATCH + a) * B.Wd + j * PATCH + b];
    float *y = B.tok + (size_t)t * E;
    for (int o = 0; o < E; o++) {
      const float *w = W.pe_w + (size_t)o * CIN * PATCH * PATCH;
      float s = 0.f;
      for (int k = 0; k < CIN * PATCH * PATCH; k++) s += w[k] * col[k];
      y[o] = s + W.pe_b[o];
    }
  }
}
typedef struct { const float *x; float *y; int n; } LNCtx;
static void ln_rows(void *c_, int r0, int r1) {
  LNCtx *c = c_;
  for (int r = r0; r < r1; r++) layernorm_row(c->x + (size_t)r * c->n, c->y + (size_t)r * c->n, c->n);
}
typedef struct { float *tok; const float *att, *gate; } GateCtx;
static void sem_gate_add_rows(void *c_, int r0, int r1) {  // tmp = att * sigmoid(gate)
  GateCtx *c = c_;
  for (size_t i = (size_t)r0 * E; i < (size_t)r1 * E; i++) B.tmp[i] = c->att[i] * sigmoidf(c->gate[i]);
}
typedef struct { float *dst; const float *src; size_t n; } AddCtx;
static void add_rows(void *c_, int r0, int r1) {
  AddCtx *c = c_;
  size_t per = c->n;
  for (size_t i = (size_t)r0 * per; i < (size_t)r1 * per; i++) c->dst[i] += c->src[i];
}
typedef struct { float *h; int n; } ActCtx;
static void gelu_tanh_rows(void *c_, int r0, int r1) {
  ActCtx *c = c_;
  for (size_t i = (size_t)r0 * c->n; i < (size_t)r1 * c->n; i++) c->h[i] = gelu_tanh(c->h[i]);
}

static void sem_block(int bi) {
  const SemBlock *b = &W.sem[bi];
  const int N = B.N;
  double t0 = now_ms();
  LNCtx lc = {B.tok, B.tmp, E};
  g_cat = TM_LN; parallel_for(N, 32, ln_rows, &lc);
  // gate and qkv share the same normed input -> quantise once
  QuantCtx qc = {B.tmp, E, N, E, B.xq, B.sx};
  g_cat = TM_QUANT; parallel_for(N, 16, quant_rows, &qc);
  qlinear(&b->gate, B.tmp, E, N, B.gate, E, 0, E, B.xq, B.sx, 1);
  qlinear(&b->qkv, B.tmp, E, N, B.qkv, 3 * E, 0, 3 * E, B.xq, B.sx, 1);
  double t1 = now_ms();
  int depthwise = bi % 2 == 1;
  QKNormCtx kc = {B.qkv, NH, N, depthwise ? 0 : B.rcos, depthwise ? 0 : B.rsin, B.h * B.w};
  g_cat = TM_QKNORM; parallel_for(N, 16, qknorm_rows, &kc);
  NACtx nc = {B.qkv, B.att, NH, B.D, B.h, B.w, 0.125f};
  if (depthwise) { g_cat = TM_DEPTH; parallel_for(B.h * B.w, 1, depth_attn_cols, &nc); }
  else { g_cat = TM_NA; parallel_for(na3d_tasks(B.D, B.h, B.w), 2, na3d_rows, &nc); }
  double t2 = now_ms();
  GateCtx gc = {B.tok, B.att, B.gate};
  g_cat = TM_ELEM; parallel_for(N, 64, sem_gate_add_rows, &gc);
  qlinear(&b->proj, B.tmp, E, N, B.att, E, 0, E, B.xq, B.sx, 0);
  AddCtx ac = {B.tok, B.att, E};
  parallel_for(N, 64, add_rows, &ac);
  // MLP in token chunks
  lc = (LNCtx){B.tok, B.tmp, E};
  g_cat = TM_LN; parallel_for(N, 32, ln_rows, &lc);
  for (int c0 = 0; c0 < N; c0 += TOKCH) {
    int cn = N - c0 < TOKCH ? N - c0 : TOKCH;
    qlinear(&b->fc1, B.tmp + (size_t)c0 * E, E, cn, B.hid, MLPH, 0, MLPH, B.xq, B.sx, 0);
    ActCtx ga = {B.hid, MLPH};
    g_cat = TM_ELEM; parallel_for(cn, 8, gelu_tanh_rows, &ga);
    qlinear(&b->fc2, B.hid, MLPH, cn, B.att + (size_t)c0 * E, E, 0, E, B.xq, B.sx, 0);
  }
  parallel_for(N, 64, add_rows, &ac);
  double t3 = now_ms();
  B.t_na += t2 - t1;
  B.t_gemm += (t1 - t0) + (t3 - t2);
}

// ----------------------------------------------------------------------------
// pixel stage
typedef struct { const float *x; } PPECtx;
static void pixel_embed_rows(void *c_, int p0, int p1) {
  const float *x = ((PPECtx *)c_)->x;
  for (int p = p0; p < p1; p++) {
    float in[CIN];
    for (int c = 0; c < CIN; c++) in[c] = x[(size_t)c * B.P + p];
    float *y = B.px + (size_t)p * EP;
    for (int o = 0; o < EP; o++) {
      float s = 0.f;
      for (int c = 0; c < CIN; c++) s += W.ppe_w[o * CIN + c] * in[c];
      y[o] = s + W.ppe_b[o];
    }
  }
}
// bilinear (align_corners=False) taps, PyTorch area_pixel_compute_source_index
static void bilin_tap(int dst, int in, int out, int *i0, int *i1, float *l0, float *l1) {
  float scale = (float)in / (float)out;
  float src = scale * (dst + 0.5f) - 0.5f;
  if (src < 0) src = 0;
  int a = (int)src;
  int off = a < in - 1 ? 1 : 0;
  *i0 = a; *i1 = a + off;
  *l1 = src - a; *l0 = 1.f - *l1;
}
typedef struct { int d; } SliceCtx;
static void upsample_rows(void *c_, int r0, int r1) {  // cond[p][c] for pixel rows of slice d
  int d = ((SliceCtx *)c_)->d;
  for (int r = r0; r < r1; r++) {
    int y = r / B.Wd, x = r % B.Wd;
    int y0, y1, x0, x1; float ly0, ly1, lx0, lx1;
    bilin_tap(y, B.h, B.H, &y0, &y1, &ly0, &ly1);
    bilin_tap(x, B.w, B.Wd, &x0, &x1, &lx0, &lx1);
    const float *t00 = B.tok + ((size_t)(d * B.h + y0) * B.w + x0) * E, *t01 = B.tok + ((size_t)(d * B.h + y0) * B.w + x1) * E;
    const float *t10 = B.tok + ((size_t)(d * B.h + y1) * B.w + x0) * E, *t11 = B.tok + ((size_t)(d * B.h + y1) * B.w + x1) * E;
    float *o = B.cond + (size_t)r * E;
    for (int c = 0; c < E; c++) o[c] = ly0 * (lx0 * t00[c] + lx1 * t01[c]) + ly1 * (lx0 * t10[c] + lx1 * t11[c]);
  }
}
typedef struct { const PixBlock *b; } DWCtx;
static void dwconv_norm_gelu_rows(void *c_, int r0, int r1) {  // cond2 = gelu(rmsnorm(dwconv(cond)))
  const PixBlock *b = ((DWCtx *)c_)->b;
  for (int r = r0; r < r1; r++) {
    int y = r / B.Wd, x = r % B.Wd;
    float *o = B.cond2 + (size_t)r * E;
    for (int c = 0; c < E; c++) o[c] = b->dw_b[c];
    for (int ky = 0; ky < DWK; ky++) {
      int yy = y + ky - DWK / 2;
      yy = yy < 0 ? 0 : (yy >= B.H ? B.H - 1 : yy);
      for (int kx = 0; kx < DWK; kx++) {
        int xx = x + kx - DWK / 2;
        xx = xx < 0 ? 0 : (xx >= B.Wd ? B.Wd - 1 : xx);
        const float *in = B.cond + ((size_t)yy * B.Wd + xx) * E;
        const float *w = b->dw_w + (size_t)(ky * DWK + kx) * E;
        for (int c = 0; c < E; c++) o[c] += w[c] * in[c];
      }
    }
    rmsnorm_inplace(o, E, 1.1920928955078125e-07f);
    for (int c = 0; c < E; c++) o[c] = gelu_erf(o[c]);
  }
}
// adaLN params for slice d, columns [c0, c1) of the 6*128 projection, into B.ada (HW x (c1-c0))
static void adaln_slice(const PixBlock *b, int d, int c0, int c1) {
  int HW = B.H * B.Wd;
  SliceCtx sc = {d};
  g_cat = TM_ADALN; parallel_for(HW, 16, upsample_rows, &sc);
  DWCtx dc = {b};
  g_cat = TM_ADALN; parallel_for(HW, 8, dwconv_norm_gelu_rows, &dc);
  qlinear(&b->ada, B.cond2, E, HW, B.ada, c1 - c0, c0, c1, B.xq, B.sx, 0);
}
typedef struct { int d; int ld; } ModCtx;
static void modulate1_rows(void *c_, int r0, int r1) {  // py = LN(px)*(1+scale1)+shift1
  ModCtx *c = c_;
  size_t base = (size_t)c->d * B.H * B.Wd;
  for (int r = r0; r < r1; r++) {
    float *y = B.py + (base + r) * EP;
    layernorm_row(B.px + (base + r) * EP, y, EP);
    const float *a = B.ada + (size_t)r * c->ld;  // [shift1 | scale1]
    for (int k = 0; k < EP; k++) y[k] = y[k] * (1.f + a[EP + k]) + a[k];
  }
}
static void modulate2_rows(void *c_, int r0, int r1) {  // px += gate1*z; py = LN(px)*(1+scale2)+shift2
  ModCtx *c = c_;
  size_t base = (size_t)c->d * B.H * B.Wd;
  for (int r = r0; r < r1; r++) {
    float *x = B.px + (base + r) * EP, *y = B.py + (base + r) * EP;
    const float *z = B.patt + (base + r) * EP;
    const float *a = B.ada + (size_t)r * c->ld;  // [gate1 | shift2 | scale2 | gate2]
    for (int k = 0; k < EP; k++) x[k] += a[k] * z[k];
    layernorm_row(x, y, EP);
    for (int k = 0; k < EP; k++) y[k] = y[k] * (1.f + a[2 * EP + k]) + a[EP + k];
  }
}
static void gate2_rows(void *c_, int r0, int r1) {  // px += gate2 * mlp_out (mlp_out in patt slice)
  ModCtx *c = c_;
  size_t base = (size_t)c->d * B.H * B.Wd;
  for (int r = r0; r < r1; r++) {
    float *x = B.px + (base + r) * EP;
    const float *z = B.patt + (base + r) * EP;
    const float *a = B.ada + (size_t)r * c->ld + 3 * EP;
    for (int k = 0; k < EP; k++) x[k] += a[k] * z[k];
  }
}

static void pix_block(int bi) {
  const PixBlock *b = &W.pix[bi];
  const int HW = B.H * B.Wd, P = B.P;
  double t0 = now_ms();
  for (int d = 0; d < B.D; d++) {
    adaln_slice(b, d, 0, 2 * EP);
    ModCtx mc = {d, 2 * EP};
    parallel_for(HW, 32, modulate1_rows, &mc);
  }
  qlinear(&b->qkv, B.py, EP, P, B.pqkv, 3 * EP, 0, 3 * EP, B.xq, B.sx, 0);
  double t1 = now_ms();
  QKNormCtx kc = {B.pqkv, NHP, P, 0, 0, 1};
  g_cat = TM_QKNORM; parallel_for(P, 64, qknorm_rows, &kc);
  NACtx nc = {B.pqkv, B.patt, NHP, B.D, B.H, B.Wd, 0.125f};
  g_cat = TM_NA; parallel_for(na3d_tasks(B.D, B.H, B.Wd), 8, na3d_rows, &nc);
  double t2 = now_ms();
  qlinear(&b->proj, B.patt, EP, P, B.py, EP, 0, EP, B.xq, B.sx, 0);
  memcpy(B.patt, B.py, (size_t)P * EP * 4);  // z = attn output projection
  for (int d = 0; d < B.D; d++) {
    adaln_slice(b, d, 2 * EP, 6 * EP);
    ModCtx mc = {d, 4 * EP};
    parallel_for(HW, 32, modulate2_rows, &mc);
    size_t base = (size_t)d * HW;
    for (int c0 = 0; c0 < HW; c0 += TOKCH) {
      int cn = HW - c0 < TOKCH ? HW - c0 : TOKCH;
      qlinear(&b->fc1, B.py + (base + c0) * EP, EP, cn, B.hid, MLPP, 0, MLPP, B.xq, B.sx, 0);
      ActCtx ga = {B.hid, MLPP};
      parallel_for(cn, 16, gelu_tanh_rows, &ga);
      qlinear(&b->fc2, B.hid, MLPP, cn, B.patt + (base + c0) * EP, EP, 0, EP, B.xq, B.sx, 0);
    }
    parallel_for(HW, 64, gate2_rows, &mc);
  }
  double t3 = now_ms();
  B.t_na += t2 - t1;
  B.t_gemm += (t1 - t0) + (t3 - t2);
}

typedef struct { float *out; } FinCtx;
static void final_rows(void *c_, int p0, int p1) {
  float *out = ((FinCtx *)c_)->out;
  float y[EP];
  for (int p = p0; p < p1; p++) {
    layernorm_row(B.px + (size_t)p * EP, y, EP);
    for (int o = 0; o < COUT; o++) {
      float s = 0.f;
      for (int k = 0; k < EP; k++) s += W.fin_w[o * EP + k] * y[k];
      out[(size_t)o * B.P + p] = s + W.fin_b[o];
    }
  }
}

#ifdef __EMSCRIPTEN__
static double now_ms(void) { return emscripten_get_now(); }
#else
#include <time.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#endif

// progress hook: calls self.__stProgress(done, total) in the calling JS context (the demo worker)
#ifdef __EMSCRIPTEN__
EM_JS(void, js_progress, (int done, int total), {
  if (typeof self !== 'undefined' && self.__stProgress) self.__stProgress(done, total);
});
#else
static void js_progress(int done, int total) { (void)done; (void)total; }
#endif
static void dbg_copy(int which, const float *src, size_t n) {
  if (B.dbg[which]) memcpy(B.dbg[which], src, n * 4);
}

// timings (ms) of the last forward: [semantic, pixel, attention (all), linear+rest]
EXPORT double st_tm(int cat) { return cat >= 0 && cat < TM_N ? g_tm[cat] : 0; }
EXPORT double st_timing(int which) {
  return which == 0 ? B.t_sem : which == 1 ? B.t_pix : which == 2 ? B.t_na : B.t_gemm;
}

EXPORT int st_forward(const float *x, float *out) {
  if (st_check_bound()) return -1;
  if (!B.rcos) return -2;
  B.t_na = B.t_gemm = 0;
  memset(g_tm, 0, sizeof g_tm);
  double t0 = now_ms();
  PECtx pc = {x};
  g_cat = TM_EMBED; parallel_for(B.N, 8, patch_embed_rows, &pc);
  dbg_copy(200, B.tok, (size_t)B.N * E);
  js_progress(0, NSEM + NPIX);
  for (int i = 0; i < NSEM; i++) {
    sem_block(i);
    dbg_copy(i, B.tok, (size_t)B.N * E);
    js_progress(i + 1, NSEM + NPIX);
  }
  double t1 = now_ms();
  PPECtx ppc = {x};
  g_cat = TM_EMBED; parallel_for(B.P, 64, pixel_embed_rows, &ppc);
  dbg_copy(201, B.px, (size_t)B.P * EP);
  for (int i = 0; i < NPIX; i++) {
    pix_block(i);
    dbg_copy(100 + i, B.px, (size_t)B.P * EP);
    js_progress(NSEM + i + 1, NSEM + NPIX);
  }
  FinCtx fc = {out};
  g_cat = TM_FINAL; parallel_for(B.P, 64, final_rows, &fc);
  double t2 = now_ms();
  B.t_sem = t1 - t0;
  B.t_pix = t2 - t1;
  return 0;
}

EXPORT void st_final_only(float *out) {  // debug: re-run the final layer on the current pixel state
  FinCtx fc = {out};
  g_cat = TM_FINAL; parallel_for(B.P, 64, final_rows, &fc);
}
// standalone NA3D entry for the kernel-vs-NATTEN test: qkv rows [3][heads][64]
EXPORT void st_na3d(const float *qkv, float *out, int heads, int D, int H, int Wd) {
  NACtx nc = {qkv, out, heads, D, H, Wd, 0.125f};
  parallel_for(na3d_tasks(D, H, Wd), 4, na3d_rows, &nc);
}
EXPORT double st_heap_mib(void) { return (double)g_bytes / (1 << 20); }
