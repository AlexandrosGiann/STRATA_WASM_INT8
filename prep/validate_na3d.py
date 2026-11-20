"""Validate prep/na3d_chunked.py against NATTEN 0.21.7 (flex-fna, unfused, CPU).

NATTEN is called one head at a time so that its full score matrix fits in RAM.
Shapes: random small cases with odd kernels, the exact STRATA v1 semantic-stage
shape (24x24x24 tokens, 16 heads, dim 64, kernel 9) and a pixel-stage-like
shape (24x32x32, 2 heads, dim 64, kernel 9).

Usage: python prep/validate_na3d.py [--full]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from natten.functional import na3d

sys.path.insert(0, str(Path(__file__).parent))
from na3d_chunked import na3d_chunked, window_start  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="include the 24^3 x 16-head case")
a = ap.parse_args()
torch.manual_seed(0)


def natten_per_head(q, k, v, ks):
    return torch.cat(
        [na3d(q[..., h : h + 1, :], k[..., h : h + 1, :], v[..., h : h + 1, :], kernel_size=ks, backend="flex-fna") for h in range(q.shape[-2])],
        dim=-2,
    )


cases = [((1, 5, 7, 9, 2, 8), 3), ((2, 9, 9, 11, 3, 16), 5), ((1, 10, 12, 9, 2, 32), (3, 5, 9)), ((1, 24, 32, 32, 2, 64), 9)]
if a.full:
    cases.append(((1, 24, 24, 24, 16, 64), 9))
def brute(q, k, v, ks, pts):
    """Direct definition, one query at a time (independent of both codes)."""
    ks3 = (ks,) * 3 if isinstance(ks, int) else ks
    _, X, Y, Z, H, D = q.shape
    st = [window_start(n, kk) for n, kk in zip((X, Y, Z), ks3)]
    res = []
    for b, x, y, z in pts:
        sl = tuple(slice(int(s[i]), int(s[i]) + kk) for s, i, kk in zip(st, (x, y, z), ks3))
        kk_ = k[(b,) + sl].reshape(-1, H, D)
        vv = v[(b,) + sl].reshape(-1, H, D)
        s = torch.einsum("hd,nhd->hn", q[b, x, y, z], kk_) * D**-0.5
        res.append(torch.einsum("hn,nhd->hd", s.softmax(-1), vv))
    return torch.stack(res)


# NATTEN's CPU path (torch flex_attention, unfused) is not deterministic: two
# identical calls occasionally differ at isolated query positions (observed
# 9.3e-4 at one position of the 24x32x32 case). Each case therefore runs
# NATTEN twice; positions where any run disagrees with the chunked kernel are
# re-checked against the direct definition.
for shp, ks in cases:
    q, k, v = (torch.randn(*shp) for _ in range(3))
    t = time.time()
    mine = na3d_chunked(q, k, v, ks)
    t_mine = time.time() - t
    errs = []
    bad = set()
    for rep in range(2):
        t = time.time()
        ref = natten_per_head(q, k, v, ks)
        t_ref = time.time() - t
        d = (ref - mine).abs().amax(dim=(-1, -2))
        errs.append(d.max().item())
        bad |= {tuple(i) for i in torch.nonzero(d > 1e-5).tolist()}
    pts = sorted(bad)[:64]
    bru = (brute(q, k, v, ks, pts) - torch.stack([mine[p] for p in pts])).abs().max().item() if pts else 0.0
    print(f"{shp} k={ks}: max|natten-chunked| run1={errs[0]:.2e} run2={errs[1]:.2e}; "
          f"positions >1e-5: {len(bad)}; chunked vs direct definition there: {bru:.2e} "
          f"(natten {t_ref:.1f}s/run, chunked {t_mine:.1f}s)")
    assert min(errs) < 1e-5 and bru < 1e-5
print("OK")
