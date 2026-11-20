"""Memory-bounded 3-D neighborhood attention (NATTEN semantics) for CPU.

NATTEN 0.21.7 builds libnatten only with CUDA, and refuses compiled Flex
Attention without a CUDA device (`get_device_cc() >= 70`), so on a CPU-only
machine its only backend is *unfused* Flex Attention, which materialises the
full (N x N) score matrix (12 GB per call for the 24x24x24x16-head semantic
stage; impossible for the pixel stage). This module computes the same operator
block by block:

  * non-causal, stride 1, dilation 1, odd kernel k (the STRATA v1 config);
  * per axis the window is [s, s+k) with s = clamp(i - k//2, 0, L - k)
    (NATTEN's flex mask: centre = clamp(i, k//2, L-1-k//2));
  * softmax(q.k * head_dim**-0.5) over the k^3 neighbours, weighted sum of v.

For a block of queries, the union of their windows is a box; scores are a
dense (block x box) matmul with the out-of-window entries masked to -inf.
No full-tile attention matrix is ever built.

`validate_na3d.py` checks it against NATTEN itself (per head, at the real
semantic-stage shape and at a pixel-stage-like shape).
"""

from __future__ import annotations

import torch


def window_start(length: int, kernel: int) -> torch.Tensor:
    i = torch.arange(length)
    return (i - kernel // 2).clamp(0, length - kernel)


def _axis_blocks(length: int, kernel: int, block: int):
    s = window_start(length, kernel)
    for a0 in range(0, length, block):
        a1 = min(length, a0 + block)
        lo = int(s[a0])
        hi = int(s[a1 - 1]) + kernel
        q = torch.arange(a0, a1)
        kv = torch.arange(lo, hi)
        st = s[q][:, None]
        mask = (kv[None, :] >= st) & (kv[None, :] < st + kernel)  # (nq, nkv)
        yield a0, a1, lo, hi, mask


def na3d_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size,
    dilation: int = 1,
    is_causal: bool = False,
    backend=None,
    block: tuple[int, int, int] = (8, 8, 8),
    **_,
) -> torch.Tensor:
    """Drop-in for natten.functional.na3d on (B, X, Y, Z, heads, dim) tensors."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    kd, kh, kw = (int(x) for x in kernel_size)
    if dilation not in (1, (1, 1, 1)) or is_causal:
        raise NotImplementedError("only dilation=1, non-causal is implemented")
    B, X, Y, Z, H, D = q.shape
    if X < kd or Y < kh or Z < kw:
        raise ValueError(f"input {X, Y, Z} smaller than kernel {kd, kh, kw}")
    scale = D**-0.5
    out = torch.empty_like(q)
    # (B, H, X, Y, Z, D) layout for batched matmuls
    qh = q.permute(0, 4, 1, 2, 3, 5)
    kh_ = k.permute(0, 4, 1, 2, 3, 5)
    vh = v.permute(0, 4, 1, 2, 3, 5)
    neg = torch.finfo(q.dtype).min
    for x0, x1, xl, xh, mx in _axis_blocks(X, kd, block[0]):
        for y0, y1, yl, yh, my in _axis_blocks(Y, kh, block[1]):
            for z0, z1, zl, zh, mz in _axis_blocks(Z, kw, block[2]):
                qb = qh[:, :, x0:x1, y0:y1, z0:z1].reshape(B, H, -1, D)
                kb = kh_[:, :, xl:xh, yl:yh, zl:zh].reshape(B, H, -1, D)
                vb = vh[:, :, xl:xh, yl:yh, zl:zh].reshape(B, H, -1, D)
                m = (mx[:, None, None, :, None, None] & my[None, :, None, None, :, None] & mz[None, None, :, None, None, :])
                m = m.reshape(qb.shape[2], kb.shape[2])
                s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
                s = s.masked_fill(~m, neg)
                p = torch.softmax(s, dim=-1)
                ob = torch.matmul(p, vb)  # (B, H, nq, D)
                out[:, x0:x1, y0:y1, z0:z1] = ob.reshape(B, H, x1 - x0, y1 - y0, z1 - z0, D).permute(0, 2, 3, 4, 1, 5)
    return out
