"""Build a real regional sample for one city from the SCREAM sdecadal store.

Reads only the city's region (interior + halo) through the official
ScreamDataSource (data_catalog.scream()), streaming from Hugging Face:

  * prognostic state (145 channels) at t0 (initial condition) and at
    t0 + k*10 min, k = 1..steps (lateral boundary values for the halo ring and
    truth for reference),
  * diagnostic truth (precip ice/liq, ps) at k = 1..steps,
  * static forcing (phis, sgh30, landfrac) from the released .npy files,
  * cell-centre lat/lon (vectorised official grid code, prep/region_latlon.py).

Output: data/<name>/sample.nc with a `history` provenance attribute.

Usage:
  SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal \
  python prep/fetch_sample.py --name athens --lat 37.98 --lon 23.73
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import strata_env  # noqa: E402,F401  (paths, stubs)
from region_latlon import face_image_latlon, locate  # noqa: E402

NE, NPG = 1024, 2
NSIDE = NE * NPG
LEVELS = np.arange(8, 32)
PROG = ("PotentialTemperature", "U", "V", "z_mid", "omega", "qv", "T_2m")
DIAG = ("precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux", "ps")
STATIC = ("phis", "sgh30", "landfrac")


def channel_names():
    from screamcast.dali_ext_src import ScreamV2
    from screamcast.earth2studio_wrappers import _build_variable_names

    dims = ScreamV2.get_default_variables_dimensions()
    return _build_variable_names(PROG, LEVELS, dims), _build_variable_names(DIAG, LEVELS, dims)


def static_region(face, xs, ys):
    import torch
    from screamcast.cubesphere_transforms import reorder_cubesphere_to_2d_tensor

    per_face = NE * NE * NPG * NPG
    out = []
    for v in STATIC:
        flat = np.load(strata_env.WEIGHTS / f"{v}.npy", mmap_mode="r").reshape(-1)
        f = torch.from_numpy(np.ascontiguousarray(flat[face * per_face : (face + 1) * per_face]).astype(np.float32))
        img = reorder_cubesphere_to_2d_tensor(f, ne=NE, npg=NPG).numpy()
        out.append(img[np.ix_(xs, ys)])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--interior", type=int, default=64)
    ap.add_argument("--halo", type=int, default=16)
    ap.add_argument("--init", default="2020-10-13T00:00")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if "SCREAM_MAIN_ZARR_PATH" not in os.environ:
        sys.exit("set SCREAM_MAIN_ZARR_PATH (e.g. hf://datasets/nvidia/STRATA-SCREAM-sdecadal)")
    tile = a.interior + 2 * a.halo
    if tile % 4:
        sys.exit("interior + 2*halo must be divisible by the patch size 4")

    face, cx, cy = locate(a.lat, a.lon, NE, NPG)
    x0 = int(np.clip(cx - tile // 2, 0, NSIDE - tile))
    y0 = int(np.clip(cy - tile // 2, 0, NSIDE - tile))
    if (x0, y0) != (cx - tile // 2, cy - tile // 2):
        print(f"note: region shifted to stay inside cube face {face} (cross-face regions are not supported)")
    xs, ys = np.arange(x0, x0 + tile), np.arange(y0, y0 + tile)
    lat, lon = face_image_latlon(NE, NPG, face, x0, tile, y0, tile)
    print(f"{a.name}: ({a.lat}, {a.lon}) -> face {face}, pixel ({cx}, {cy}); region x[{x0}:{x0+tile}] y[{y0}:{y0+tile}]")
    print(f"  centre cell lat/lon = {lat[cx-x0, cy-y0]:.3f}, {lon[cx-x0, cy-y0]:.3f}")

    from data_catalog import scream

    prog_names, diag_names = channel_names()
    ds = scream()
    t0 = np.datetime64(a.init)
    times = [t0 + np.timedelta64(10 * k, "m") for k in range(a.steps + 1)]
    jobs = [(k, "prog", i, n) for k in range(a.steps + 1) for i, n in enumerate(prog_names)]
    jobs += [(k, "diag", i, n) for k in range(1, a.steps + 1) for i, n in enumerate(diag_names)]
    prog = np.full((a.steps + 1, len(prog_names), tile, tile), np.nan, np.float32)
    diag = np.full((a.steps + 1, len(diag_names), tile, tile), np.nan, np.float32)

    def fetch(job):
        k, kind, i, n = job
        for attempt in range(5):
            try:
                da = ds({"time": np.array([times[k]]), "variable": [n], "face": np.array([face]), "x": xs, "y": ys})
                return job, da.values[0, 0, 0]
            except Exception as e:  # network hiccups: retry
                err = e
                time.sleep(2 * (attempt + 1))
        raise err

    t = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for done, (job, arr) in enumerate(ex.map(fetch, jobs), 1):
            k, kind, i, _ = job
            (prog if kind == "prog" else diag)[k, i] = arr
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} fields, {time.time()-t:.0f}s", flush=True)
    assert np.isfinite(prog).all() and np.isfinite(diag[1:]).all()
    static = static_region(face, xs, ys)

    out = a.out or (strata_env.ROOT / "data" / a.name)
    out.mkdir(parents=True, exist_ok=True)
    import xarray as xr

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    cmd = " ".join([sys.executable, *sys.argv])
    dset = xr.Dataset(
        {
            "prognostic": (("step", "channel", "x", "y"), prog),
            "diagnostic_truth": (("step", "diag_channel", "x", "y"), diag),
            "static": (("static_channel", "x", "y"), static),
            "lat": (("x", "y"), lat),
            "lon": (("x", "y"), lon),
        },
        coords={
            "step": np.arange(a.steps + 1),
            "valid_time": ("step", np.array(times, dtype="datetime64[ns]")),
            "channel": prog_names,
            "diag_channel": diag_names,
            "static_channel": list(STATIC),
            "x": xs,
            "y": ys,
        },
        attrs={
            "city": a.name,
            "city_lat": a.lat,
            "city_lon": a.lon,
            "face": face,
            "x0": x0,
            "y0": y0,
            "tile": tile,
            "interior": a.interior,
            "halo": a.halo,
            "init_time": str(t0),
            "source": os.environ["SCREAM_MAIN_ZARR_PATH"],
            "history": f"{now}: SCREAM_MAIN_ZARR_PATH={os.environ['SCREAM_MAIN_ZARR_PATH']} {cmd}",
        },
    )
    dset.to_netcdf(out / "sample.nc")
    print(f"wrote {out/'sample.nc'} ({(out/'sample.nc').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
