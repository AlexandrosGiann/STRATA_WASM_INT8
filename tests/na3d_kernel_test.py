"""Compare the WASM NA3D kernel with NATTEN 0.21.7 (and the chunked torch op).

Shapes: STRATA v1 semantic stage (24x24x24 tokens, 16 heads, dim 64, k=9) and
a pixel-stage-like grid (24x32x32, 2 heads). NATTEN runs per head, twice (its
CPU flex path is non-deterministic at isolated positions, see validate_na3d.py).
Usage: python tests/na3d_kernel_test.py [--real data/athens/capture_qkv.npy]
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))
from na3d_chunked import na3d_chunked
from natten.functional import na3d
NODE = os.environ.get("NODE", str(next((ROOT / "tools/emsdk/node").glob("*/bin/node"))))
torch.manual_seed(0)

def run(shape, heads, label, qkv=None):
    D, H, W = shape
    if qkv is None:
        qkv = torch.randn(D * H * W, 3, heads, 64)
    q, k, v = (qkv[:, i].reshape(1, D, H, W, heads, 64) for i in range(3))
    with tempfile.TemporaryDirectory() as td:
        fi, fo = Path(td) / "qkv.bin", Path(td) / "out.bin"
        fi.write_bytes(qkv.contiguous().numpy().tobytes())
        r = subprocess.run([NODE, str(ROOT / "tests/run_na3d_wasm.mjs"), fi, fo, str(heads), str(D), str(H), str(W), "4"], capture_output=True, text=True, check=True)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        wasm = torch.from_numpy(np.fromfile(fo, np.float32)).reshape(1, D, H, W, heads, 64)
    ref = na3d_chunked(q, k, v, 9)
    nat = [torch.cat([na3d(q[..., h:h+1, :], k[..., h:h+1, :], v[..., h:h+1, :], kernel_size=9, backend="flex-fna") for h in range(heads)], -2) for _ in range(2)]
    e_nat = [(n - wasm).abs().max().item() for n in nat]
    e_ref = (ref - wasm).abs().max().item()
    rel = ((ref - wasm).norm() / ref.norm()).item()
    print(f"{label}: tokens={D*H*W} heads={heads}: max|natten-wasm| run1={e_nat[0]:.2e} run2={e_nat[1]:.2e}; "
          f"max|chunked-wasm|={e_ref:.2e}; rel L2={rel:.2e}; wasm {info['ms']:.0f} ms ({info['nthreads']} threads)")
    return min(e_nat), e_ref

r1 = run((24, 24, 24), 16, "semantic (random)")
r2 = run((24, 32, 32), 2, "pixel-like (random)")
assert r1[1] < 1e-5 and r2[1] < 1e-5
print("OK")
