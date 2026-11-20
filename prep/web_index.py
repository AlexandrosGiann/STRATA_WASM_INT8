"""Write web/data/index.json listing the exported regions.  python prep/web_index.py"""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "web" / "data"
cities = []
for m in sorted(root.glob("*/meta.json")):
    j = json.loads(m.read_text())
    cities.append({"name": m.parent.name, "label": f"{j['city']} {j['tile']}x{j['tile']}", "lat": j["city_lat"], "lon": j["city_lon"], "tile": j["tile"], "steps": j["steps"]})
(root / "index.json").write_text(json.dumps({"cities": cities}, indent=1))
print(cities)
