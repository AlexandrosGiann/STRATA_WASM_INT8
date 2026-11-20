"""Quick check: WASM NA3D vs the NATTEN-validated chunked op, with timings.
Usage: python tests/na3d_speed.py"""
import glob, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))
from na3d_chunked import na3d_chunked
NODE = glob.glob(str(ROOT / "tools/emsdk/node/*/bin/node"))[0]
torch.manual_seed(0)
for (D, H, W), heads in [((24, 24, 24), 16), ((24, 96, 96), 2)]:
    qkv = torch.randn(D * H * W, 3, heads, 64)
    with tempfile.TemporaryDirectory() as td:
        fi, fo = Path(td) / "i", Path(td) / "o"
        fi.write_bytes(qkv.numpy().tobytes())
        r = subprocess.run([NODE, str(ROOT / "tests/run_na3d_wasm.mjs"), fi, fo, str(heads), str(D), str(H), str(W), "4"], capture_output=True, text=True, check=True)
        w = torch.from_numpy(np.fromfile(fo, np.float32)).reshape(1, D, H, W, heads, 64)
    q, k, v = (qkv[:, i].reshape(1, D, H, W, heads, 64) for i in range(3))
    ref = na3d_chunked(q, k, v, 9)
    print((D, H, W), heads, r.stdout.strip(), f"max {(ref - w).abs().max().item():.2e} relL2 {((ref - w).norm() / ref.norm()).item():.2e}")
