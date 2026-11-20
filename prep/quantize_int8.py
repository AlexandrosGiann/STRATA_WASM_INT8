"""Create the INT8 STRATA v1 weight file from the official FP32 checkpoint.

Scheme (weight-only storage quantisation, per output channel, symmetric):
    scale[o] = max_k |W[o, k]| / 127,   Wq[o, k] = round(W[o, k] / scale[o]) in [-127, 127]
applied to every large nn.Linear of the network:
    semantic blocks (24x): attn.qkv, attn.proj, attn.gate_proj, mlp.layers.0, mlp.layers.2
    pixel blocks    (4x):  attn.qkv, attn.proj, mlp.layers.0, mlp.layers.2, adaln_bilinear_dw_proj
Kept in FP32: biases, the two patch-embedding Conv3d, the depthwise 5x5 conv,
the final 128->10 linear, normalisation statistics.

File layout (little endian):
    b"STRATAQ8" | u32 header_len | JSON header | zero pad to 64 | tensors (64-aligned)
JSON: {"meta": {...}, "tensors": {name: {"dtype": "i8"|"f32", "shape": [...], "offset": o}}}
Quantised weights W have a companion "<W>.scale" (f32, [out]).

Usage: python prep/quantize_int8.py --out weights/strata_v1_int8.bin
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import strata_env  # noqa: E402

SEM_LIN = ("attn.qkv", "attn.proj", "attn.gate_proj", "mlp.layers.0", "mlp.layers.2")
PIX_LIN = ("attn.qkv", "attn.proj", "mlp.layers.0", "mlp.layers.2", "adaln_bilinear_dw_proj")


def quantized_weight_names(n_sem=24, n_pix=4):
    names = [f"strata.backbone.blocks.{i}.{l}.weight" for i in range(n_sem) for l in SEM_LIN]
    names += [f"strata.pixel_blocks.{i}.{l}.weight" for i in range(n_pix) for l in PIX_LIN]
    return names


def quantize_per_row(w: torch.Tensor, group: int = 0):
    """Symmetric INT8. group=0: one scale per output row; group=g: one scale per
    (row, block of g input columns) -> scale shape [out, in // g]."""
    w = w.double()
    if group:
        o, i = w.shape
        assert i % group == 0
        wg = w.reshape(o, i // group, group)
        amax = wg.abs().amax(dim=2)
        scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
        q = torch.round(wg / scale[..., None]).clamp(-127, 127).to(torch.int8).reshape(o, i)
        return q, scale.float()
    amax = w.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=strata_env.CKPT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", type=int, default=0, help="0 = per output channel; g = per (row, g columns)")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # Load through the official model so legacy key remapping is applied exactly
    # as at inference time; then read the canonical state dict.
    model = strata_env.load_model(128)
    net_sd = model.pipeline.network.state_dict()
    qnames = set(quantized_weight_names())
    assert qnames <= set(net_sd), sorted(qnames - set(net_sd))[:5]

    tensors: dict[str, np.ndarray] = {}
    report = {"int8_weights": 0, "fp32_params": 0}
    for name, t in net_sd.items():
        if not t.is_floating_point():
            continue
        if name.startswith("geometry."):
            continue  # global lat/lon buffers: not model weights
        if name in qnames:
            q, s = quantize_per_row(t, a.group)
            tensors[name] = q.numpy()
            tensors[name + ".scale"] = s.numpy()
            report["int8_weights"] += q.numel()
        else:
            tensors[name] = t.float().numpy()
            report["fp32_params"] += t.numel()
    for key in ("input_norm", "target_norm"):
        for buf in ("mean", "var"):
            tensors[f"{key}.{buf}"] = ckpt[key][buf].float().numpy().reshape(-1)
    norm_eps = {k: float(model.pipeline.__getattr__(k).eps) for k in ("input_norm", "target_norm")}

    header = {"tensors": {}}
    offset = 0
    order = sorted(tensors)
    for n in order:
        arr = tensors[n]
        offset = (offset + 63) // 64 * 64
        header["tensors"][n] = {"dtype": "i8" if arr.dtype == np.int8 else "f32", "shape": list(arr.shape), "offset": offset}
        offset += arr.nbytes
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    header["meta"] = {
        "format": "STRATAQ8 v1",
        "source_checkpoint": str(a.ckpt.name),
        "source_sha256": sha256(a.ckpt),
        "quantization": ("symmetric per-output-channel int8" if not a.group else f"symmetric int8, one scale per output row and {a.group} input columns") + " (scale = absmax/127), round-to-nearest",
        "group_size": a.group,
        "norm_eps": norm_eps,
        "int8_weight_count": report["int8_weights"],
        "fp32_param_count": report["fp32_params"],
        "history": f"{now}: {' '.join([sys.executable, *sys.argv])}",
    }
    hj = json.dumps(header).encode()
    pre = 8 + 4 + len(hj)
    data_start = (pre + 63) // 64 * 64
    with open(a.out, "wb") as f:
        f.write(b"STRATAQ8")
        f.write(np.uint32(len(hj)).tobytes())
        f.write(hj)
        f.write(b"\0" * (data_start - pre))
        pos = 0
        for n in order:
            arr = np.ascontiguousarray(tensors[n])
            o = header["tensors"][n]["offset"]
            f.write(b"\0" * (o - pos))
            f.write(arr.tobytes())
            pos = o + arr.nbytes
    size = a.out.stat().st_size
    print(f"wrote {a.out}: {size/2**20:.1f} MiB; int8 weights {report['int8_weights']:,}, fp32 params {report['fp32_params']:,}")
    print(f"sha256 {sha256(a.out)}")


if __name__ == "__main__":
    main()
