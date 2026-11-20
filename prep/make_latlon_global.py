"""Write latlon_ne1024pg2.nc (float32 lat/lon over ncol) with the vectorised port.

Equivalent to `generate_all.sh` steps 1-3 (generate_exodus_meshes.py --halo 0,
scrip_convert.py, derive_latlon.py) but in minutes on a laptop instead of a
64-core node. prep/validate_region_latlon.py checks bit-identity with the
official scripts at small resolution.

Usage: python prep/make_latlon_global.py --out weights/strata_v1/latlon_ne1024pg2.nc
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from region_latlon import face_image_latlon  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--ne", type=int, default=1024)
ap.add_argument("--rows", type=int, default=128, help="face-image rows per chunk")
a = ap.parse_args()
ne, npg = a.ne, 2
n = ne * npg
lat = np.empty(6 * n * n, np.float32)
lon = np.empty(6 * n * n, np.float32)
for f in range(6):
    img_lat = np.empty((n, n), np.float32)
    img_lon = np.empty((n, n), np.float32)
    for x0 in range(0, n, a.rows):
        la, lo = face_image_latlon(ne, npg, f, x0, a.rows, 0, n)
        img_lat[x0 : x0 + a.rows] = la
        img_lon[x0 : x0 + a.rows] = lo
    # face image (x=elem_i*npg+gp_i, y=elem_j*npg+gp_j) -> flat (elem_i, elem_j, gp_i, gp_j)
    sl = slice(f * n * n, (f + 1) * n * n)
    lat[sl] = img_lat.reshape(ne, npg, ne, npg).transpose(0, 2, 1, 3).reshape(-1)
    lon[sl] = img_lon.reshape(ne, npg, ne, npg).transpose(0, 2, 1, 3).reshape(-1)
    print(f"face {f} done", flush=True)

with Dataset(a.out, "w") as out:
    out.createDimension("ncol", lat.shape[0])
    v_lat = out.createVariable("lat", "f4", ("ncol",))
    v_lon = out.createVariable("lon", "f4", ("ncol",))
    v_lat.units = "degrees_north"
    v_lon.units = "degrees_east"
    v_lat[:] = lat
    v_lon[:] = lon
    out.title = f"per-column cell-center lat/lon for ne{ne}pg2 (vectorised TempestRemap port)"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out.history = f"{now}: {' '.join([sys.executable, *sys.argv])}"
print(f"wrote {a.out}")
