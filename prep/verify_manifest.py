"""Verify every file of the nvidia/strata download against the SHA-256 table in its MANIFEST.md.
Usage: python3 prep/verify_manifest.py weights/strata_v1"""
import hashlib, re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = re.findall(r"^\| (\S+) \| [\d.]+ MiB \| ([0-9a-f]{64}) \|", (root / "MANIFEST.md").read_text(), re.M)
bad = 0
for f, h in rows:
    d = hashlib.sha256()
    with open(root / f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            d.update(b)
    ok = d.hexdigest() == h
    bad += not ok
    print(("OK  " if ok else "FAIL"), f, d.hexdigest())
print(f"{len(rows) - bad}/{len(rows)} files match MANIFEST.md")
sys.exit(1 if bad else 0)
