"""Centre-crop an existing regional sample to a smaller tile (no new download).

    python prep/crop_sample.py --src athens --name athens64 --interior 32 --halo 16
The crop keeps the same cube face / grid; its own outer `halo` ring then gets
the SCREAM boundary values exactly as in the larger sample.
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import strata_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--interior", type=int, required=True)
ap.add_argument("--halo", type=int, default=16)
a = ap.parse_args()
import xarray as xr  # noqa: E402

s = xr.open_dataset(strata_env.ROOT / "data" / a.src / "sample.nc")
T0, T = int(s.attrs["tile"]), a.interior + 2 * a.halo
assert T % 4 == 0 and T <= T0
o = (T0 - T) // 2
c = s.isel(x=slice(o, o + T), y=slice(o, o + T)).load()
c.attrs.update(tile=T, interior=a.interior, halo=a.halo, x0=int(s.attrs["x0"]) + o, y0=int(s.attrs["y0"]) + o)
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
c.attrs["history"] = s.attrs["history"] + f"\n{now}: {' '.join([sys.executable, *sys.argv])}"
out = strata_env.ROOT / "data" / a.name
out.mkdir(parents=True, exist_ok=True)
c.to_netcdf(out / "sample.nc")
print(f"wrote {out/'sample.nc'}: tile {T} (interior {a.interior}, halo {a.halo}) from {a.src} tile {T0}")
