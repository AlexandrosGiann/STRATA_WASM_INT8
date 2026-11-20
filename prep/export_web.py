"""Export one region's sample to the browser demo (web/data/<name>/).

Inputs: data/<name>/sample.nc, capture.npz and reference_fp32.nc (run_reference.py),
the INT8 weight file (normalisation statistics).
Files written (all little-endian float32 unless stated):
  meta.json      tile/halo/steps/channels/norm stats/reference timings
  ic.bin         initial state, 145 x T x T
  ring.bin       halo-ring values per step k=1..N, N x 145 x R  (R ring pixels, ring_idx in meta)
  forcing.bin    per-step forcing as the model saw it (coszr, phis, sgh30, landfrac), N x 4 x T x T
  geom.bin       wind_fwd(4TT) wind_inv(4TT) cos_lat(TT) sin_lat(TT) rope_cos(hw*64) rope_sin(hw*64)
  ref.bin        PyTorch FP32 reference: N x len(ref_channels) x T x T
  ref_int8.bin   (optional) PyTorch INT8-emulated reference, same layout
Usage: python prep/export_web.py --name athens
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import strata_env  # noqa: E402
from quant_emulate import read_weight_file  # noqa: E402

REF_CH = ["T_2m", "U_31", "V_31", "precip_liq_surf_mass_flux", "precip_ice_surf_mass_flux", "ps", "PotentialTemperature_31", "qv_31"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", type=Path, default=strata_env.ROOT / "weights/strata_v1_int8.bin")
    a = ap.parse_args()
    import torch
    import xarray as xr

    d = strata_env.ROOT / "data" / a.name
    out = strata_env.ROOT / "web" / "data" / a.name
    out.mkdir(parents=True, exist_ok=True)
    s = xr.open_dataset(d / "sample.nc")
    c = np.load(d / "capture.npz")
    ref = xr.open_dataset(d / "reference_fp32.nc")
    T, halo = int(s.attrs["tile"]), int(s.attrs["halo"])
    N = int(ref.sizes["step"])
    P = T * T
    prog = s["prognostic"].values.astype(np.float32)
    ring = np.ones((T, T), bool)
    ring[halo : T - halo, halo : T - halo] = False
    ring_idx = np.flatnonzero(ring.reshape(-1)).astype(np.int32)

    prog[0].tofile(out / "ic.bin")
    np.stack([prog[k].reshape(145, P)[:, ring_idx] for k in range(1, N + 1)]).astype(np.float32).tofile(out / "ring.bin")
    c["forcing_per_step"][:N].astype(np.float32).tofile(out / "forcing.bin")
    lat = torch.from_numpy(c["lat_rad"].astype(np.float32))
    hw = (T // 4) ** 2
    geom = [
        c["wind_fwd"].reshape(-1),
        c["wind_inv"].reshape(-1),
        torch.cos(lat).numpy().reshape(-1),
        torch.sin(lat).numpy().reshape(-1),
        c["rope_cos"][0, 0, :hw].reshape(-1),
        c["rope_sin"][0, 0, :hw].reshape(-1),
    ]
    np.concatenate(geom).astype(np.float32).tofile(out / "geom.bin")
    names = list(ref["out_channel"].values)
    idx = [names.index(n) for n in REF_CH]
    ref["output"].values[:, idx].astype(np.float32).tofile(out / "ref.bin")
    variants = {}
    for tag in ("int8_a16", "int8_a8"):
        f = d / f"reference_{tag}.nc"
        if f.exists():
            r8 = xr.open_dataset(f)
            r8["output"].values[:N, idx].astype(np.float32).tofile(out / f"ref_{tag}.bin")
            variants[tag] = list(r8.attrs["step_seconds"])

    header, t = read_weight_file(a.weights)
    meta = {
        "city": s.attrs["city"],
        "city_lat": float(s.attrs["city_lat"]),
        "city_lon": float(s.attrs["city_lon"]),
        "face": int(s.attrs["face"]),
        "x0": int(s.attrs["x0"]),
        "y0": int(s.attrs["y0"]),
        "tile": T,
        "halo": halo,
        "steps": N,
        "init_time": s.attrs["init_time"],
        "bc": ref.attrs["bc"],
        "ring_idx": ring_idx.tolist(),
        "ref_channels": REF_CH,
        "ref_variants": ["fp32"] + list(variants),
        "torch_step_seconds": {"fp32": list(ref.attrs["step_seconds"]), **variants},
        "in_mean": t["input_norm.mean"].tolist(),
        "in_var": t["input_norm.var"].tolist(),
        "tgt_mean": t["target_norm.mean"].tolist(),
        "tgt_var": t["target_norm.var"].tolist(),
        "in_eps": header["meta"]["norm_eps"]["input_norm"],
        "tgt_eps": header["meta"]["norm_eps"]["target_norm"],
        "lat_center": float(np.degrees(s["lat"].values[T // 2, T // 2])),
        "lon_center": float(np.degrees(0) + s["lon"].values[T // 2, T // 2]),
        "landfrac": None,
    }
    lf = s["static"].sel(static_channel="landfrac").values
    meta["landfrac"] = np.round(lf.reshape(-1), 3).tolist()
    (out / "meta.json").write_text(json.dumps(meta))
    tot = sum(f.stat().st_size for f in out.iterdir())
    print(f"exported {out}: {tot/2**20:.1f} MiB")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:16s} {f.stat().st_size/2**20:7.2f} MiB")


if __name__ == "__main__":
    main()
