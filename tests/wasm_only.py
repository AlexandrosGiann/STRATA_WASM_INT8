"""Run only the WASM forward on a saved/new input (debugging + timing).

Usage: python tests/wasm_only.py --dir logs/wasm_dbg [--tile 96] [--new] [--threads 4] [--act-bits 16]
--new builds a deterministic random input and the real RoPE tables (official
physicsnemo builder on the Athens lat/lon) into --dir; otherwise reuses --dir.
"""
import argparse, glob, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))
NODE = glob.glob(str(ROOT / "tools/emsdk/node/*/bin/node"))[0]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--tile", type=int, default=96)
ap.add_argument("--new", action="store_true")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--act-bits", type=int, default=16)
a = ap.parse_args()
T, D = a.tile, 24
a.dir.mkdir(parents=True, exist_ok=True)
if a.new:
    import torch
    from physicsnemo.experimental.models.strata.coords import build_stereographic_token_coords
    from physicsnemo.experimental.nn import build_axial_rope_cos_sin_2d_continuous
    from region_latlon import face_image_latlon, locate
    import strata_env  # noqa
    from screamcast.strata_wrappers import _resolve_rope_base
    face, cx, cy = locate(37.98, 23.73)
    lat, lon = face_image_latlon(1024, 2, face, cx - T // 2, T, cy - T // 2, T)
    pos = torch.from_numpy(np.stack([np.deg2rad(lat), np.deg2rad(lon)])[None].astype(np.float32))
    base = _resolve_rope_base(None)
    coords = build_stereographic_token_coords(pos, (4, 4), d_patch=D, length_scale=base * 4)
    cos, sin = build_axial_rope_cos_sin_2d_continuous(coords[..., 0], coords[..., 1], 64, theta=100.0)
    hw = (T // 4) ** 2
    cos[0, :hw].numpy().astype(np.float32).tofile(a.dir / "rope_cos.bin")
    sin[0, :hw].numpy().astype(np.float32).tofile(a.dir / "rope_sin.bin")
    torch.manual_seed(0)
    torch.randn(1, 13, D, T, T).numpy().tofile(a.dir / "x.bin")
r = subprocess.run([NODE, str(ROOT / "tests/run_forward_wasm.mjs"), str(ROOT / "weights/strata_v1_int8.bin"), str(a.dir), str(D), str(T), str(T), str(a.threads), str(a.act_bits)], capture_output=True, text=True)
if r.returncode:
    print(r.stdout, r.stderr); sys.exit(1)
print(r.stdout.strip().splitlines()[-1]); print(r.stderr.strip()[-500:])
out = np.fromfile(a.dir / "wasm_out.bin", np.float32).reshape(10, D, T, T)
bad = ~np.isfinite(out)
print("non-finite outputs:", int(bad.sum()), "per channel", bad.reshape(10, -1).sum(1).tolist())
if bad.any():
    idx = np.argwhere(bad)
    print("first", idx[:5].tolist(), "pixel-major unique pixels:", len({tuple(i[1:]) for i in idx}))
for k in ["pix3"]:
    v = np.fromfile(a.dir / f"wasm_{k}.bin", np.float32)
    print(k, "non-finite:", int((~np.isfinite(v)).sum()), "absmax", float(np.nanmax(np.abs(v))))
