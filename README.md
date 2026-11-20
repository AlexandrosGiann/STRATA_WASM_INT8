# STRATA v1 INT8 στον browser (WebAssembly)

Τοπική πρόγνωση με το NVIDIA STRATA v1 για μία πόλη και τα περίχωρά της, μέσα σε browser (και σε κινητό Android),
με βάρη INT8 (W8A16) και δική μας μηχανή WASM (SIMD128 + pthreads, NA3D χωρίς πλήρη πίνακα attention).

Όλη η τεκμηρίωση, οι μετρήσεις και οι ακριβείς εντολές αναπαραγωγής είναι στο **[LOG.md](LOG.md)**.

## Γρήγορη εκκίνηση (demo)

```bash
# βάρη INT8 από το Release v1 του repo (σε private repo: κατέβασέ τα από τη σελίδα Releases) (ή δημιουργία με prep/quantize_int8.py, LOG.md §4/§10)
mkdir -p weights && curl -L -o weights/strata_v1_int8.bin https://github.com/AlexandrosGiann/STRATA_WASM_INT8/releases/download/v1/strata_v1_int8.bin
ln -sf ../../weights/strata_v1_int8.bin web/weights/strata_v1_int8.bin
python3 web/serve.py --port 8000              # http://localhost:8000
python3 web/serve.py --host 0.0.0.0 --port 8080   # τοπικό δίκτυο (LOG.md §9)
```

SHA-256 του `strata_v1_int8.bin`: `45f91e15c314473dfb9b2133be652a8e3d4cc58482af8b104761ec01335d68c4`.

## Δομή

```
prep/     προετοιμασία σε PC (Python): δείγμα περιοχής, αναφορά PyTorch, INT8, export για web
engine/   strata.c + build.sh (Emscripten 6.0.13)
web/      demo: index.html, app.js, worker, engine/*.wasm, data/<πόλη>/ (μόνο περιφερειακά δεδομένα)
tests/    NA3D vs NATTEN, engine vs PyTorch, ONNX Runtime Web, browser benchmark
logs/     όλες οι μετρήσεις
```

Δεν περιλαμβάνονται (αναπαράγονται με το LOG.md §10): `.venv/`, `tools/emsdk`, `third_party/STRATA`,
τα επίσημα βάρη (`weights/strata_v1`) και τα πλήρη δείγματα (`data/`).

## Άδειες

- Κώδικας STRATA (NVlabs/STRATA): Apache-2.0 — `licenses/Apache-2.0_STRATA-code.txt`.
- Βάρη STRATA v1 και τα παράγωγα INT8 βάρη: OpenMDW-1.1 — `licenses/OpenMDW-1.1_STRATA-weights.txt`.
- Τα δεδομένα στο `web/data/` προέρχονται από τα STRATA-SCREAM datasets (CC BY 4.0) — `licenses/STRATA_ATTRIBUTION.md`.
