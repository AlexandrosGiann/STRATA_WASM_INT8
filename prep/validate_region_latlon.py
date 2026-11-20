"""Check prep/region_latlon.py against the official scrip_generation scripts.

Runs the official pure-Python workflow at a small resolution and compares every
cell centre (float32 degrees) with the vectorised port.
Usage: python prep/validate_region_latlon.py --res 32 --work /tmp/dir
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
sys.path.insert(0, str(Path(__file__).parent))
from region_latlon import face_image_latlon

ap = argparse.ArgumentParser(); ap.add_argument("--res", type=int, default=32); ap.add_argument("--work", required=True)
a = ap.parse_args()
w = Path(a.work); w.mkdir(parents=True, exist_ok=True)
sg = Path(__file__).resolve().parents[1] / "third_party/STRATA/data_prep/scrip_generation"
py = sys.executable
subprocess.run([py, "-I", sg / "generate_exodus_meshes.py", "--res", str(a.res), "--np", "2", "--halo", "0", "--out-dir", w], check=True)
subprocess.run([py, "-I", sg / "scrip_convert.py", "--in-exodus", w / f"ne{a.res}pg2.g"], check=True)
subprocess.run([py, "-I", sg / "derive_latlon.py", "--in-scrip", w / f"ne{a.res}pg2_scrip.nc", "--out", w / "latlon.nc"], check=True)
with Dataset(w / "latlon.nc") as d:
    lat = np.asarray(d["lat"][:]); lon = np.asarray(d["lon"][:])
ne, npg = a.res, 2; n = ne * npg
nexact = nulp = 0; maxd = 0.0
for f in range(6):
    flat = slice(f * n * n, (f + 1) * n * n)
    ref_lat = lat[flat].reshape(ne, ne, npg, npg).transpose(0, 2, 1, 3).reshape(n, n)
    ref_lon = lon[flat].reshape(ne, ne, npg, npg).transpose(0, 2, 1, 3).reshape(n, n)
    my_lat, my_lon = face_image_latlon(ne, npg, f, 0, n, 0, n)
    dlon = np.abs(((my_lon.astype(np.float64) - ref_lon + 180) % 360) - 180)
    d = np.maximum(np.abs(my_lat.astype(np.float64) - ref_lat), dlon)
    nexact += int(np.sum((my_lat == ref_lat) & ((my_lon == ref_lon) | (dlon == 0))))
    maxd = max(maxd, float(d.max()))
print(f"ne{ne}pg2: {nexact}/{6*n*n} cells bit-identical, max |diff| = {maxd:.3e} deg")
