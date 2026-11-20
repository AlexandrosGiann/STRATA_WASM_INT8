"""Whole-network check: WASM engine vs the official physicsnemo Strata forward.

Runs Strata.forward(x, pos) of the loaded STRATA v1 checkpoint on one input
(by default a deterministic random normalised field on the real Athens
lat/lon; with --capture, the real normalised input captured by
run_reference.py) in three ways:
  FP32 PyTorch (official), INT8-emulated PyTorch (quant_emulate.py), WASM.
Prints relative L2 / max errors of every checkpoint (semantic blocks 0, 1, 23,
pixel blocks 0-3, output).

Usage: python tests/engine_test.py [--tile 96] [--capture data/athens/capture.npz] [--act-bits 16]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))
import strata_env  # noqa: E402
import torch  # noqa: E402
from quant_emulate import apply_int8_emulation  # noqa: E402
from region_latlon import face_image_latlon, locate  # noqa: E402

NODE = os.environ.get("NODE", str(next((ROOT / "tools/emsdk/node").glob("*/bin/node"))))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)), float(np.abs(a - b).max())


def run_torch(strata, x, pos):
    cap = {}
    hs = []

    def keep(n):
        return lambda m, i, o: cap.__setitem__(n, (o[0] if isinstance(o, tuple) else o).detach().numpy().copy())

    for i in (0, 1, 23):
        hs.append(strata.backbone.blocks[i].register_forward_hook(keep(f"sem{i}")))
    for i in range(4):
        hs.append(strata.pixel_blocks[i].register_forward_hook(keep(f"pix{i}")))
    rope = {}
    orig = strata.backbone._build_rope_tables

    def rope_cap(p, dhw):
        r = orig(p, dhw)
        rope["cos"], rope["sin"] = r[0].numpy().copy(), r[1].numpy().copy()
        return r

    strata.backbone._build_rope_tables = rope_cap
    with torch.no_grad():
        out = strata(x, pos=pos).numpy()
    strata.backbone._build_rope_tables = orig
    for h in hs:
        h.remove()
    return out, cap, rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--capture", type=Path, default=None)
    ap.add_argument("--act-bits", type=int, default=16)
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/strata_v1_int8.bin")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    model = strata_env.load_model(a.tile)
    net = model.pipeline.network
    strata = net.strata
    T, D = a.tile, 24
    if a.capture:
        c = np.load(a.capture)
        x = torch.from_numpy(c["strata_in"])
        pos = torch.from_numpy(c["pos"])
    else:
        face, cx, cy = locate(37.98, 23.73)
        lat, lon = face_image_latlon(1024, 2, face, cx - T // 2, T, cy - T // 2, T)
        pos = torch.from_numpy(np.stack([np.deg2rad(lat), np.deg2rad(lon)])[None].astype(np.float32))
        x = torch.randn(1, 13, D, T, T)
    out32, cap32, rope = run_torch(strata, x, pos)
    cos, sin = rope["cos"][0, 0], rope["sin"][0, 0]  # (N, 64)
    hw = (T // 4) ** 2
    assert np.array_equal(cos.reshape(D, hw, 64), np.broadcast_to(cos[:hw], (D, hw, 64))), "RoPE varies with depth"
    apply_int8_emulation(net, a.weights, act_bits=a.act_bits)
    out8, cap8, _ = run_torch(strata, x, pos)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x.numpy().astype(np.float32).tofile(td / "x.bin")
        cos[:hw].astype(np.float32).tofile(td / "rope_cos.bin")
        sin[:hw].astype(np.float32).tofile(td / "rope_sin.bin")
        r = subprocess.run(
            [NODE, str(ROOT / "tests/run_forward_wasm.mjs"), str(a.weights), str(td), str(D), str(T), str(T), str(a.threads), str(a.act_bits)],
            capture_output=True, text=True,
        )
        if r.returncode:
            print(r.stdout, r.stderr)
            sys.exit(1)
        timing = json.loads(r.stdout.strip().splitlines()[-1])
        outw = np.fromfile(td / "wasm_out.bin", np.float32).reshape(out32.shape)
        capw = {k: np.fromfile(td / f"wasm_{k}.bin", np.float32) for k in cap32}

    res = {"timing": timing, "checks": {}}
    print(f"{'checkpoint':10s} {'wasm vs int8-emul (relL2, max)':>34s} {'wasm vs fp32 (relL2, max)':>30s} {'int8-emul vs fp32':>22s}")
    for k in list(cap32) + ["out"]:
        f32 = out32 if k == "out" else cap32[k]
        e8 = out8 if k == "out" else cap8[k]
        w = outw if k == "out" else capw[k].reshape(f32.shape)
        a1, a2, a3 = rel(w, e8), rel(w, f32), rel(e8, f32)
        res["checks"][k] = {"wasm_vs_emul": a1, "wasm_vs_fp32": a2, "emul_vs_fp32": a3}
        print(f"{k:10s} {a1[0]:14.2e} {a1[1]:12.2e}      {a2[0]:12.2e} {a2[1]:12.2e}   {a3[0]:10.2e} {a3[1]:10.2e}")
    print(json.dumps(timing))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
