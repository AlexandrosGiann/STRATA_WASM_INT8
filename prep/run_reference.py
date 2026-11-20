"""FP32 PyTorch reference rollout of the official STRATA v1 model on one region.

Follows notebooks/run_screamcast.py (ScreamcastModel.from_checkpoint,
set_tile_size, create_iterator) on the regional tile of data/<name>/sample.nc.
Lateral boundary condition: before every step the official `front_hook`
overwrites the outer `halo` ring of the prognostic state with the SCREAM
values at that step's valid time (identical rule in the browser). With
--bc free the ring is left to evolve (the notebook's behaviour).

Outputs (data/<name>/):
  reference_fp32.nc   all 148 output channels for steps 1..N (+ history)
  capture.npz         what the model actually saw: per-step forcing channels
                      (coszr, phis, sgh30, landfrac), semantic RoPE tables,
                      wind-rotation coefficients, cos/sin(lat), step-1
                      intermediate activations for layer-by-layer checks.

Usage: python prep/run_reference.py --name athens [--steps 6] [--quant int8.bin]
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import strata_env  # noqa: E402
import torch  # noqa: E402


def ring_mask(tile: int, halo: int) -> torch.Tensor:
    m = torch.ones(tile, tile, dtype=torch.bool)
    m[halo : tile - halo, halo : tile - halo] = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--bc", choices=["truth", "free"], default="truth")
    ap.add_argument("--quant", type=Path, default=None, help="emulate an INT8 weight file (prep/quantize_int8.py)")
    ap.add_argument("--act-bits", type=int, default=16)
    ap.add_argument("--int8-scope", default="", help="emulate INT8 only for weights with this name prefix, e.g. strata.backbone")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-capture", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import xarray as xr

    d = strata_env.ROOT / "data" / a.name
    s = xr.open_dataset(d / "sample.nc")
    tile, halo, face = int(s.attrs["tile"]), int(s.attrs["halo"]), int(s.attrs["face"])
    xs, ys = s["x"].values, s["y"].values
    prog = torch.from_numpy(s["prognostic"].values)  # (steps+1, 145, T, T)
    assert a.steps <= prog.shape[0] - 1

    t = time.time()
    model = strata_env.load_model(tile)
    print(f"model loaded in {time.time()-t:.0f}s")
    tag = a.tag or ("fp32" if a.quant is None else f"int8_a{a.act_bits}")
    if a.quant is not None:
        from quant_emulate import apply_int8_emulation

        apply_int8_emulation(model.pipeline.network, a.quant, act_bits=a.act_bits, scope=a.int8_scope)
    net = model.pipeline.network
    in_names = list(model.input_coords()["variable"])
    assert in_names == list(s["channel"].values), "channel order mismatch"

    ring = ring_mask(tile, halo)
    cap: dict[str, np.ndarray] = {}
    step_inputs = []

    def front_hook(x, coords):
        k = int(coords["lead_time"][0] / np.timedelta64(600, "s"))
        if a.bc == "truth" and k > 0:
            x = x.clone()
            x[0, :, 0][:, ring] = prog[k][:, ring].to(x.dtype)
        return x, coords

    model.front_hook = front_hook

    # --- capture hooks -------------------------------------------------------
    orig_step = model.pipeline.step

    def step_capture(x_flat, index):
        step_inputs.append(x_flat[0, len(in_names) :].clone())  # forcing channels as the model saw them
        return orig_step(x_flat, index)

    model.pipeline.step = step_capture
    handles = []
    if not a.no_capture:
        strata = net.strata
        first = {"done": False}

        def keep(name):
            def hook(mod, inp, out):
                if not first["done"]:
                    cap[name] = (out[0] if isinstance(out, tuple) else out).detach().float().numpy().copy()
            return hook

        def strata_pre(mod, args, kwargs):
            if not first["done"]:
                cap["strata_in"] = args[0].detach().numpy().copy()  # (1, 13, 24, T, T) after norm + wind rotation + lat concat
                cap["pos"] = kwargs["pos"].detach().numpy().copy()

        def strata_post(mod, args, kwargs, out):
            if not first["done"]:
                cap["strata_out"] = out.detach().numpy().copy()  # (1, 10, 24, T, T) before inverse wind rotation
                first["done"] = True

        handles.append(strata.register_forward_pre_hook(strata_pre, with_kwargs=True))
        handles.append(strata.register_forward_hook(strata_post, with_kwargs=True))
        bb = strata.backbone
        handles.append(bb.patch_embed.register_forward_hook(keep("sem_patch_embed")))
        for i in (0, 1, 2, 3, 23):
            handles.append(bb.blocks[i].register_forward_hook(keep(f"sem_block{i}")))
        handles.append(strata.pixel_patch_embed.register_forward_hook(keep("pix_patch_embed")))
        for i in range(len(strata.pixel_blocks)):
            handles.append(strata.pixel_blocks[i].register_forward_hook(keep(f"pix_block{i}")))
        orig_rope = bb._build_rope_tables

        def rope_capture(pos, latent_dhw):
            r = orig_rope(pos, latent_dhw)
            if "rope_cos" not in cap:
                cap["rope_cos"] = r[0].detach().numpy().copy()
                cap["rope_sin"] = r[1].detach().numpy().copy()
            return r

        bb._build_rope_tables = rope_capture

    # --- rollout (as in notebooks/run_screamcast.py) -------------------------
    t0 = np.datetime64(s.attrs["init_time"])
    coords = OrderedDict(model.input_coords())
    coords["face"] = np.array([face])
    coords["x"] = xs
    coords["y"] = ys
    coords["batch"] = np.empty(1)
    coords["time"] = np.array([t0])
    x0 = prog[0][None, :, None]  # (1, C, 1, T, T)
    outs, step_time = [], []
    it = model.create_iterator(x0, coords)
    next(it)
    out_names = None
    for k in range(a.steps):
        t = time.time()
        y, oc = next(it)
        step_time.append(time.time() - t)
        out_names = list(oc["variable"])
        outs.append(y[0, :, 0].numpy().copy())
        print(f"step {k+1}/{a.steps}: {step_time[-1]:.1f}s", flush=True)
    for h in handles:
        h.remove()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    ds = xr.Dataset(
        {"output": (("step", "out_channel", "x", "y"), np.stack(outs))},
        coords={"step": np.arange(1, a.steps + 1), "out_channel": out_names, "x": xs, "y": ys},
        attrs={
            "bc": a.bc,
            "variant": tag,
            "torch": torch.__version__,
            "step_seconds": step_time,
            "history": f"{now}: {' '.join([sys.executable, *sys.argv])}",
        },
    )
    ds.to_netcdf(d / f"reference_{tag}.nc")
    print(f"wrote {d / f'reference_{tag}.nc'}; mean step {np.mean(step_time):.1f}s")

    if not a.no_capture:
        lat = torch.from_numpy(s["lat"].values.astype(np.float32))
        lon = torch.from_numpy(s["lon"].values.astype(np.float32))
        # wind rotation coefficients via the official functions on unit vectors
        from screamcast.tile_geometry import TileGeometry
        from screamcast.wind_rotation import forward_uv_to_tile, inverse_tile_to_uv

        latr, lonr = torch.deg2rad(lat)[None], torch.deg2rad(lon)[None]
        lat0, lon0 = TileGeometry.tile_center(latr, lonr)
        one, zero = torch.ones(1, 1, tile, tile), torch.zeros(1, 1, tile, tile)
        fu = forward_uv_to_tile(one, zero, latr, lonr, lat0, lon0)
        fv = forward_uv_to_tile(zero, one, latr, lonr, lat0, lon0)
        iu = inverse_tile_to_uv(one, zero, latr, lonr, lat0, lon0)
        iv = inverse_tile_to_uv(zero, one, latr, lonr, lat0, lon0)
        cap["wind_fwd"] = torch.stack([fu[0], fv[0], fu[1], fv[1]])[:, 0, 0].numpy()  # u'=a u+b v, v'=c u+d v
        cap["wind_inv"] = torch.stack([iu[0], iv[0], iu[1], iv[1]])[:, 0, 0].numpy()
        cap["forcing_per_step"] = torch.stack(step_inputs).numpy()  # (steps, 4, T, T)
        cap["lat_rad"] = latr[0].numpy()
        np.savez(d / "capture.npz", **cap)
        print("wrote capture.npz:", {k: v.shape for k, v in cap.items()})


if __name__ == "__main__":
    main()
