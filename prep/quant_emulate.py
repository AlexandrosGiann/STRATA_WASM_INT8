"""Exact PyTorch emulation of the WASM INT8 linear layers.

For every quantised nn.Linear (see quantize_int8.py) the forward becomes
    s_x[t]  = max_k |x[t, k]| / qmax          (dynamic, per token; qmax = 2**(bits-1) - 1)
    xq[t,k] = round(x[t,k] / s_x[t])          (int, |xq| <= qmax)
    y[t,o]  = (sum_k xq[t,k] * Wq[o,k]) * s_x[t] * s_w[o] + b[o]
with the integer dot product evaluated exactly (float64), as the WASM kernel
does with int32 accumulation. This separates quantisation error from
implementation error when comparing FP32 / INT8-emulated / WASM outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def read_weight_file(path: Path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b"STRATAQ8"
    hlen = int(np.frombuffer(raw[8:12], np.uint32)[0])
    header = json.loads(raw[12 : 12 + hlen])
    start = (12 + hlen + 63) // 64 * 64
    out = {}
    for n, t in header["tensors"].items():
        dt = np.int8 if t["dtype"] == "i8" else np.float32
        cnt = int(np.prod(t["shape"])) if t["shape"] else 1
        out[n] = np.frombuffer(raw, dt, cnt, start + t["offset"]).reshape(t["shape"])
    return header, out


class Int8Linear(torch.nn.Module):
    def __init__(self, wq: np.ndarray, scale: np.ndarray, bias, act_bits: int):
        super().__init__()
        self.register_buffer("wq", torch.from_numpy(wq.astype(np.float64)))
        self.register_buffer("sw", torch.from_numpy(scale.astype(np.float64)))
        self.bias = bias
        self.qmax = float(2 ** (act_bits - 1) - 1)

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).double()
        amax = x2.abs().amax(dim=1, keepdim=True)
        sx = torch.where(amax > 0, amax / self.qmax, torch.ones_like(amax))
        xq = torch.round(x2 / sx)
        if self.sw.dim() == 1:
            acc = (xq @ self.wq.T) * self.sw[None, :]  # exact integer sums in float64
        else:  # group-wise: per (row, group) scale applied to each group's exact integer sum
            o, i = self.wq.shape
            g = i // self.sw.shape[1]
            acc = torch.zeros(xq.shape[0], o, dtype=torch.float64)
            for j in range(i // g):
                acc += (xq[:, j * g : (j + 1) * g] @ self.wq[:, j * g : (j + 1) * g].T) * self.sw[None, :, j]
        y = (acc * sx).float()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], -1)


def apply_int8_emulation(net: torch.nn.Module, weight_file: Path, act_bits: int = 16, scope: str = ""):
    """scope: only emulate INT8 for weights whose name starts with this prefix ("" = all)."""
    header, t = read_weight_file(weight_file)
    n = 0
    for name in [k for k in t if k.endswith(".weight") and t[k].dtype == np.int8 and k.startswith(scope)]:
        mod_path = name[: -len(".weight")]
        parent_path, _, leaf = mod_path.rpartition(".")
        parent = net.get_submodule(parent_path)
        lin = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        new = Int8Linear(t[name], t[name + ".scale"], lin.bias, act_bits)
        if leaf.isdigit():
            parent[int(leaf)] = new
        else:
            setattr(parent, leaf, new)
        n += 1
    print(f"INT8 emulation: replaced {n} linear layers (activations {act_bits}-bit per token)")
    return header
