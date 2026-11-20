"""ONNX Runtime Web (execution provider 'wasm') experiments.

A. torch.onnx.export of the whole STRATA network (records whether / why it fails).
B. Dense part of one semantic block (LN -> qkv, gate; LN -> fc1 -> GELU -> fc2) at the
   real 96x96 token count, FP32 ONNX and INT8 (onnxruntime.quantization.quantize_dynamic,
   per-channel QInt8 -> DynamicQuantizeLinear + MatMulInteger), run in ORT Web.
C. 3-D neighbourhood attention expressed with standard ONNX ops (the block-local
   masked-matmul formulation of prep/na3d_chunked.py: no tile-wide attention matrix),
   run in ORT Web and compared with NATTEN-validated output.
Usage: python tests/ort_experiment.py [--parts ABC] [--threads 4]
"""
import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))
import strata_env  # noqa: E402
import torch  # noqa: E402
from na3d_chunked import na3d_chunked  # noqa: E402

NODE = glob.glob(str(ROOT / "tools/emsdk/node/*/bin/node"))[0]
ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="ABC")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", type=Path, default=ROOT / "logs/ort_experiment.json")
a = ap.parse_args()
res = {}
td = Path(tempfile.mkdtemp())


def ort_web(model, x, outshape):
    fi, fo = td / "in.bin", td / "out.bin"
    x.numpy().astype(np.float32).tofile(fi)
    r = subprocess.run([NODE, str(ROOT / "tests/ort_run.mjs"), str(model), str(fi), ",".join(map(str, x.shape)), str(fo), str(a.threads), "2"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return torch.from_numpy(np.fromfile(fo, np.float32).reshape(outshape)), info


def export(mod, args, path):
    torch.onnx.export(mod, args, str(path), opset_version=18, dynamo=False, input_names=["x"], output_names=["y"])


if "A" in a.parts:
    model = strata_env.load_model(40)
    net = model.pipeline.network
    x = torch.randn(1, 13, 24, 40, 40)
    pos = torch.zeros(1, 2, 40, 40)
    out = {}
    for label, dyn in (("torchscript", False), ("dynamo", True)):
        t = time.time()
        try:
            torch.onnx.export(net.strata, (x, {"pos": pos}), str(td / f"full_{label}.onnx"), opset_version=18, dynamo=dyn)
            out[label] = {"ok": True, "s": time.time() - t}
        except Exception as e:  # noqa: BLE001
            out[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
    res["A_full_export"] = out
    print("A:", json.dumps(out, indent=1))
    del model, net

if "B" in a.parts:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = strata_env.load_model(96)
    blk = model.pipeline.network.strata.backbone.blocks[0]

    class Dense(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):  # the five INT8 GEMMs of a block, in sequence
            y = self.b.norm1(x)
            q = self.b.attn.qkv(y)
            g = torch.sigmoid(self.b.attn.gate_proj(y))
            z = self.b.attn.proj(q[:, :1024] * g)
            x = x + z
            return x + self.b.mlp(self.b.norm2(x))

    d = Dense(blk).eval()
    N = 24 * 24 * 24
    x = torch.randn(N, 1024)
    with torch.no_grad():
        ref = d(x)
    export(d, (x,), td / "dense_fp32.onnx")
    quantize_dynamic(str(td / "dense_fp32.onnx"), str(td / "dense_int8.onnx"), weight_type=QuantType.QInt8, per_channel=True)
    out = {}
    for tag in ("fp32", "int8"):
        y, info = ort_web(td / f"dense_{tag}.onnx", x, (N, 1024))
        out[tag] = {"ms": info["ms"], "rel_l2_vs_torch_fp32": float((y - ref).norm() / ref.norm()), "file_mib": (td / f"dense_{tag}.onnx").stat().st_size / 2**20}
    out["ort"] = info["ort"]
    out["gemm_gmac"] = N * 1024 * (3072 + 1024 + 1024 + 4096 * 2) / 1e9
    res["B_dense_block"] = out
    print("B:", json.dumps(out, indent=1))
    del model

if "C" in a.parts:
    class NA(torch.nn.Module):
        def forward(self, qkv):
            q, k, v = qkv.unbind(0)
            return na3d_chunked(q, k, v, 9)

    out = {}
    for (D, H, W), heads in (((24, 24, 24), 16), ((24, 96, 96), 2)):
        qkv = torch.randn(3, 1, D, H, W, heads, 64)
        with torch.no_grad():
            ref = NA()(qkv)
        f = td / f"na_{H}.onnx"
        t = time.time()
        try:
            export(NA(), (qkv,), f)
            nodes = None
            import onnx

            nodes = len(onnx.load(str(f)).graph.node)
            y, info = ort_web(f, qkv, tuple(ref.shape))
            out[f"{D}x{H}x{W}x{heads}h"] = {"export_s": time.time() - t, "onnx_nodes": nodes, "ms": info["ms"], "max_abs_vs_chunked": float((y - ref).abs().max()), "rel_l2": float((y - ref).norm() / ref.norm())}
        except Exception as e:  # noqa: BLE001
            out[f"{D}x{H}x{W}x{heads}h"] = {"error": f"{type(e).__name__}: {str(e)[:400]}", "trace": traceback.format_exc()[-800:]}
        print("C:", json.dumps(out, indent=1))
    res["C_na3d_onnx_ops"] = out

prev = json.loads(a.out.read_text()) if a.out.exists() else {}
prev.update(res)
a.out.write_text(json.dumps(prev, indent=1))
print("wrote", a.out)
