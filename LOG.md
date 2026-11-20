# STRATA v1 → INT8 → WebAssembly: log εργασίας

Στόχος: τοπική πρόγνωση με το NVIDIA STRATA v1 για μία πόλη και την περιοχή γύρω της, μέσα σε browser
(WebAssembly), με βάρη INT8, και αριθμητική σύγκριση με το αρχικό PyTorch μοντέλο στα 1 και 6 βήματα (10 και 60 λεπτά).

Κατάσταση: **λειτουργεί από άκρη σε άκρη σε desktop Chrome 157 (headless, μέσω puppeteer)**. Δεν υπήρχε διαθέσιμο κινητό·
το benchmark για κινητό είναι έτοιμο (§9). Οι μετρήσεις έγιναν σε φορητό Intel i5-6200U (2 πυρήνες / 4 threads, 2015), 15 GB RAM, χωρίς GPU.

---

## 1. Πηγές, εκδόσεις, επαλήθευση

| Τι | Έκδοση / ταυτότητα |
|---|---|
| Επίσημος κώδικας [NVlabs/STRATA](https://github.com/NVlabs/STRATA) | commit `d56a78caf410bd58dd9082765d2826ed98479374` (2026-11-10) → `third_party/STRATA` |
| Βάρη [nvidia/strata](https://huggingface.co/nvidia/strata) | revision `407ccb1e19a23303ebf114acd8a87655b7463f5a` → `weights/strata_v1/` |
| physicsnemo (Strata model) | commit `fae6bb2b0e96488a052d9627d41e6dc94653f06a` (το pin του `docker/Dockerfile`), 2.3.0a0 |
| earth2grid | commit `6e39b069f0558cd0eff0ec666ede5e125d79b4ff` (CPU build) |
| NATTEN | 0.21.7 (το pin του `docker/build_natten.sh`), χωρίς libnatten (CPU) |
| PyTorch | 2.14.1+cpu (Python 3.12.3) |
| onnx / onnxruntime / onnxruntime-web | 1.23.2 / 1.32.0 / 1.31.1 |
| Emscripten | 6.0.13 (`tools/emsdk`), Node 24.19.0 (του emsdk) |
| Browser για μετρήσεις | Google Chrome 157.0.8103.78 (headless, puppeteer-core 24.43.1) |
| Δεδομένα | `hf://datasets/nvidia/STRATA-SCREAM-sdecadal` (streaming, μόνο η περιοχή) |

**`AGENTS.md`:** στον φάκελο του project δεν υπήρχε `AGENTS.md` (ο φάκελος ήταν άδειος). Διάβασα το `AGENTS.md` του επίσημου repo
και τήρησα: provenance `history` (absolute timestamp + ακριβής εντολή) σε κάθε NetCDF που γράφεται, κανένα binary
στο git, τεστ με ανεξάρτητη αλήθεια αντί για ταυτολογικές αναφορές.

**SHA-256 έναντι `MANIFEST.md`:** και τα 20 αρχεία ταιριάζουν, μεταξύ τους το
`checkpoints/strata_v1.pth` = `a66bc75c1f04ddb43375d4280d71951c3d0874b043ad854b5791bc4d6f316a45` (1265,3 MiB, 331.664.906 παράμετροι).

## 2. Δείγμα δεδομένων (περιοχή πόλης)

`prep/fetch_sample.py --name <πόλη> --lat <φ> --lon <λ>` (παραμετροποιήσιμο με συντεταγμένες):

- Εντοπίζει την όψη του κύβου και το pixel της πόλης με τη γεωμετρία του πλέγματος ne1024pg2.
  Το `prep/region_latlon.py` είναι vectorized μεταφορά των επίσημων `generate_exodus_meshes.py`/`scrip_convert.py`/`derive_latlon.py`
  και είναι **bit-identical** με αυτά σε όλα τα κελιά στα ne8 και ne64 (`prep/validate_region_latlon.py`).
  Με τον ίδιο κώδικα παράγεται σε 27 s το καθολικό `latlon_ne1024pg2.nc`, που χρειάζεται το επίσημο `ScreamcastModel`
  (το επίσημο `generate_all.sh` θέλει κόμβο 64 πυρήνων).
- Κατεβάζει **μόνο την περιοχή** μέσω της επίσημης `ScreamDataSource` (`data_catalog.scream()`):
  - 145 prognostic κανάλια στο t0 (αρχικές συνθήκες) και στα t0+10·k λεπτά, k=1..6, για τις συνθήκες ορίων (halo) και την αλήθεια·
  - διαγνωστικά (βροχή υγρή/στερεή, ps) για k=1..6·
  - στατικά πεδία (`phis`, `sgh30`, `landfrac`) από τα `.npy` της NVIDIA.
  Το Zarr είναι sharded με inner chunk = ολόκληρη όψη, οπότε κατεβαίνει ~1 chunk ανά πεδίο (~14 GB ροή για 1.033 πεδία, ~1,5 ώρα εδώ). Αποθηκεύεται μόνο η περιοχή (38 MB).
- Σημείο έναρξης: `2020-10-13T00:00`. Είναι εκτός της περιόδου εκπαίδευσης· το README ζητά αξιολόγηση από 2020-10-11.
- Περιοχή: tile 96×96 κελιά (~3,2 km) = εσωτερικό 64 (~205 km) + halo 16 γύρω-γύρω· και crop 64×64 = εσωτερικό 32 + halo 16 (`prep/crop_sample.py`).

**Συνθήκες ορίων (ίδιες σε PyTorch και browser):** πριν από κάθε βήμα, ο δακτύλιος halo (16 κελιά) της prognostic κατάστασης παίρνει τις τιμές SCREAM
της τρέχουσας ώρας. Στο PyTorch αυτό γίνεται μέσω του επίσημου `ScreamcastModel.front_hook`, στον browser με τις ίδιες τιμές float32 (`ring.bin`).
Οι συγκρίσεις αναφέρονται στο **εσωτερικό** (χωρίς το halo).

Περιοχές που χρησιμοποιήθηκαν:

| όνομα | πόλη (φ, λ) | όψη, pixel | tile | χαρακτήρας |
|---|---|---|---|---|
| `athens` | Αθήνα (37.98, 23.73) | 0, (1944, 1563) | 96 | ξηρή (μέση βροχή ~1e-5 mm/h) |
| `athens64` | ίδια, crop | 0 | 64 | ξηρή |
| `puntarenas` | Puntarenas, Κόστα Ρίκα (9.98, −84.84) | 3, (1252, 1141) | 96 | **βροχερή**: εσωτ. μέση 3,2 mm/h, 50% κελιά >0,1 mm/h, max 109 mm/h |

## 3. Αναφορά PyTorch (αρχικό μοντέλο)

`prep/run_reference.py`: επίσημο `ScreamcastModel.from_checkpoint(bf16=False)`, `set_tile_size`, `create_iterator`, όπως το notebook `run_screamcast.py`, σε FP32 σε CPU.
Γράφει `reference_fp32.nc` (148 κανάλια × βήμα, με `history`) και `capture.npz` (ό,τι είδε το μοντέλο: coszr ανά βήμα, πίνακες RoPE, ενδιάμεσες ενεργοποιήσεις).

**Η μόνη αλλαγή στην εκτέλεση:** το `na3d` του NATTEN αντικαθίσταται από το `prep/na3d_chunked.py`. Λόγος: το NATTEN 0.21.7 χτίζει `libnatten` μόνο με CUDA
και αρνείται το compiled Flex σε CPU (`get_device_cc() >= 70`). Μένει μόνο το unfused Flex, που υλοποιεί πλήρη πίνακα scores (12 GB για το semantic stage, αδύνατο για το pixel stage).
Το chunked NA3D είναι blocked masked matmul με τα ίδια clamped παράθυρα και επαληθεύτηκε έναντι του ίδιου του NATTEN (`prep/validate_na3d.py`, `logs/validate_na3d.txt`):

| σχήμα (D,H,W, heads, dim) | kernel | max \|NATTEN − chunked\| |
|---|---|---|
| 24×24×24, 16, 64 (πραγματικό semantic stage) | 9 | 2,98e-7 |
| 24×32×32, 2, 64 (τύπου pixel stage) | 9 | 2,38e-7 |
| 3 μικρά τυχαία (k=3, 5, (3,5,9)) | – | ≤ 7,2e-7 |

Εύρημα: **το NATTEN σε CPU (unfused flex) δεν είναι ντετερμινιστικό.** Δύο ίδιες κλήσεις διέφεραν σε μεμονωμένες θέσεις
(9,3e-4 σε 1 θέση· 2,85e-3 σε 3 θέσεις του πραγματικού σχήματος). Στις θέσεις αυτές το chunked ταιριάζει με τον ορισμό (υπολογισμός ένα-ένα query) σε 3e-7.
Γι' αυτό το validation τρέχει το NATTEN δύο φορές και ελέγχει τις διαφωνίες με brute force.

Χρόνος αναφοράς FP32 σε αυτή τη CPU: **236,5 s/βήμα** (96×96), **93,0 s/βήμα** (64×64).

## 4. INT8: μετατροπή, τι είναι INT8, πού FP32

Δεν βρέθηκε δημόσιο INT8 checkpoint του STRATA v1 (το HF repo έχει μόνο το FP32 `.pth`), οπότε το INT8 δημιουργήθηκε από το επίσημο FP32 checkpoint.

`prep/quantize_int8.py --out weights/strata_v1_int8.bin` (αναπαραγώγιμο, ντετερμινιστικό· `--group 128` για group-wise):

- **Αποθηκεύονται σε INT8** (συμμετρικό, per-output-channel, scale = absmax/127, round-to-nearest): όλα τα μεγάλα `nn.Linear`, 140 πίνακες, **331.087.872 βάρη (99,83%)**:
  - semantic blocks ×24: `attn.qkv` (3072×1024), `attn.proj`, `attn.gate_proj` (1024×1024), `mlp.layers.0` (4096×1024), `mlp.layers.2` (1024×4096)·
  - pixel blocks ×4: `attn.qkv`, `attn.proj`, `mlp.layers.0/2`, `adaln_bilinear_dw_proj` (768×1024).
- **Μένουν FP32** (577.034 παράμετροι): biases, τα δύο patch embeddings (Conv3d 13→1024 με patch 1×4×4, 13→128), depthwise conv 5×5 του adaLN,
  τελικό `pixel_final_layer` 128→10, στατιστικά κανονικοποίησης.
- Αρχείο: **`weights/strata_v1_int8.bin` = 319,0 MiB** (334.459.088 bytes, SHA-256 `45f91e15c314473dfb9b2133be652a8e3d4cc58482af8b104761ec01335d68c4`), έναντι 1265,3 MiB FP32.
  Μορφή: `STRATAQ8` + JSON header + tensors στοιχισμένοι σε 64 bytes· ένα αρχείο περιέχει όλα όσα χρειάζεται ο browser.
- Η μετατροπή περνά από το επίσημο μοντέλο (`ScreamcastModel.from_checkpoint`), ώστε να εφαρμοστεί το επίσημο remapping των ονομάτων του checkpoint.

**Πράξεις στη WASM:**

| Πράξη | Ακρίβεια |
|---|---|
| Όλα τα παραπάνω Linear (≈99% των MAC) | **INT8 βάρη × INT16 ενεργοποιήσεις**, ακέραια γινόμενα `i32x4.dot_i16x8_s`, άθροιση int32 ανά 1024 στοιχεία K, μετά FP32· δυναμική κβάντιση ενεργοποιήσεων ανά token |
| NA3D (QKᵀ, softmax, PV), depth attention, RoPE, LayerNorm/RMSNorm, GELU, sigmoid, adaLN modulation, residuals, depthwise conv, bilinear upsample, patch embeds, τελικό layer | FP32 |
| Κανονικοποίηση, περιστροφή ανέμου, qv-softplus, precip-ReLU, denormalize, state update, συνθήκες ορίων (JS) | FP32/FP64 |

**Πού χρειάστηκε FP32/INT16 (μετρημένο):**
- **Ενεργοποιήσεις INT8 (W8A8):** ίδια ταχύτητα με INT16 (282 έναντι 287 s), επειδή η WASM SIMD128 κάνει το ίδιο `dot_i16x8` και για τα δύο, αλλά ~10× μεγαλύτερο σφάλμα στο +10′
  (T2m RMSE 0,063 K έναντι 0,005· U 0,24 έναντι 0,023 m/s· ps 1003 έναντι 48 Pa). Άρα **οι ενεργοποιήσεις μένουν INT16 (W8A16)**.
- **Group-wise INT8 (g=128):** μειώνει το σφάλμα ανακατασκευής βαρών από 1,125% σε 0,760%, με +3% μέγεθος (327,9 MiB) και +11% χρόνο.
  End-to-end στο +10′ ήταν μικτό (T2m 0,0050→0,0036 K, V και βροχή καλύτερα, ps 48→98 Pa χειρότερα), άρα **προεπιλογή μένει το per-channel**·
  το g128 είναι επιλογή στο UI. Το MSE-clipping της κλίμακας δεν βοήθησε (1,125%→1,118%).
- NA3D/softmax σε FP32: σκόπιμα, για να μη μπει σφάλμα σε ένα κομμάτι που δεν είναι το κύριο κόστος.

## 5. Μηχανή WASM

`engine/strata.c` (C → Emscripten, SIMD128, pthreads), ~1000 γραμμές· `engine/build.sh` → `web/engine/strata_{mt,st}.{js,wasm}` (276 KiB / 66 KiB).
Ο πυρήνας υπολογίζει το `physicsnemo Strata.forward(x, pos)`· τα υπόλοιπα του `MixedPredictionAsymmetric.step` γίνονται σε JS (`web/strata_step.mjs`).
Από τον υπολογιστή προϋπολογίζονται με τις επίσημες συναρτήσεις μόνο όσα εξαρτώνται από γεωμετρία/χρόνο:
πίνακες RoPE (stereographic), συντελεστές περιστροφής ανέμου 2×2 ανά pixel, cos/sin(lat), coszr ανά βήμα.

**3D neighborhood attention (NATTEN):** δικός μου WASM kernel. Κάθε query αγγίζει μόνο τους 9³ γείτονές του (clamped παράθυρα όπως το NATTEN)·
δεν υπάρχει πίνακας attention για όλο το tile (μόνο 729 scores ανά query/head). Ομαδοποίηση 4 διαδοχικών queries, που διαβάζουν μία φορά την ένωση των γραμμών K/V.
Έλεγχος έναντι NATTEN (`tests/na3d_kernel_test.py`, `logs/na3d_kernel_test.txt`):

| σχήμα | max \|NATTEN − WASM\| | rel L2 |
|---|---|---|
| 24×24×24, 16 heads (semantic) | 1,6e-6 | 7,3e-7 |
| 24×32×32, 2 heads (pixel-like) | 7,5e-7 | 7,3e-7 |
| 24×96×96, 2 heads (πραγματικό pixel stage, έναντι chunked) | 1,3e-6 | 6,1e-7 |

**Ολόκληρο δίκτυο, πραγματική είσοδος βήματος 1, Αθήνα 96×96** (`tests/engine_test.py --capture`, `logs/engine_test_real_a16.txt`), rel L2:

| checkpoint | WASM vs PyTorch INT8-emulation | INT8-emulation vs FP32 |
|---|---|---|
| semantic block 0 | 7,0e-6 | 8,0e-3 |
| semantic block 23 | 5,8e-4 | 3,0e-2 |
| pixel block 3 | 2,1e-4 | 3,5e-2 |
| έξοδος δικτύου | **2,5e-4** | **1,35e-2** |

Η WASM ακολουθεί πιστά την ακριβή PyTorch emulation του σχήματος INT8 (`prep/quant_emulate.py`). Η απόκλιση από το FP32 είναι σχεδόν εξ ολοκλήρου σφάλμα κβάντισης βαρών.
Η προεπεξεργασία σε JS ταιριάζει με την είσοδο που είδε το PyTorch σε ≤ 9,5e-7 (`tests/check_preprocess.mjs`).

**Bugs που βρέθηκαν και διορθώθηκαν:**
- NaN/σκουπίδια στην έξοδο στο 96×96: `malloc` μέσα στα threads του NA3D → thread-local scratch.
- Denormals στο softmax με πραγματικές ενεργοποιήσεις (NA 4× πιο αργό) → όροι με s−max < −60 τίθενται 0 (σχετικό βάρος < 1e-26).
- Στην Emscripten 6 τα pthreads ξαναφορτώνουν `self.location.href`, άρα τον δικό μου worker· το `mainScriptUrlOrBlob` δεν υπάρχει πια → `web/strata_mt_worker.js` = glue μηχανής + worker, με φύλαξη `self.name !== 'em-pthread'`.
- `TextDecoder` δεν δέχεται views σε SharedArrayBuffer → αντιγραφή του header.

**ONNX Runtime Web (EP `wasm`)**, `tests/ort_experiment.py`, `logs/ort_experiment.json`:
- (A) `torch.onnx.export` ολόκληρου του `Strata` **αποτυγχάνει** και με TorchScript (`RuntimeError: Unsupported value kind: Tensor`) και με dynamo (`TorchExportError` στο `torch.export`, βήμα 1/3).
  Αναπαραγωγή: `python tests/ort_experiment.py --parts A`.
- (C) NA3D με **συμβατές πράξεις ONNX** (blocked masked MatMul/Softmax, χωρίς πίνακα attention για όλο το tile): σωστό (rel L2 2,8e-7), αλλά πιο αργό από τον δικό μου kernel·
  semantic 24³×16h: **6,4 s** (ORT Web) έναντι **2,9 s** (δικός μου kernel)· pixel 24×96×96×2h: **20,4 s** έναντι **7,3 s**. Γράφημα 3.144 / 45.624 κόμβων.
- (B) dense μέρος ενός semantic block (LN→qkv/gate→proj, LN→MLP· 188 GMAC, 13.824 tokens) στο ORT Web 1.31.1, 4 threads:
  FP32 **10,8 s** (rel L2 vs PyTorch 5,8e-7)· INT8 με `quantize_dynamic` (per-channel QInt8 βάρη, `DynamicQuantizeLinear`+`MatMulInteger`, uint8 ενεργοποιήσεις ανά tensor)
  **6,5 s**, αλλά rel L2 **5,4e-2** έναντι FP32. Ο δικός μου INT8 GEMM κάνει το ίδιο φορτίο σε ~**6,1 s** (31 GMAC/s μετρημένα σε όλο το forward)
  με ~0,8% σφάλμα ανά block (INT16 ενεργοποιήσεις ανά token).

Επιλογή: δική μου μηχανή WASM για όλο το δίκτυο. Δεν υπάρχει εξαγώγιμο γράφημα ONNX του μοντέλου, το NA3D σε ORT είναι 2–3× πιο αργό,
και ο δικός μου πυρήνας ελέγχει τη μνήμη (ροή βαρών κατευθείαν στη μνήμη WASM, pixel stage ανά επίπεδο, επαναχρησιμοποίηση buffers).

## 6. Αποτελέσματα: browser (WASM INT8) − PyTorch FP32, ίδιο δείγμα, ίδιες συνθήκες ορίων

Εσωτερικό περιοχής. Άνεμος = U, V στο κατώτατο μοντελικό επίπεδο (31). Πηγή: `logs/browser_bench_*.json`, `python tests/summarize_bench.py <json>`.

### Αθήνα 96×96 (εσωτερικό 64×64)

| lead | πεδίο | RMSE | max \|Δ\| | bias | rel L2 |
|---|---|---|---|---|---|
| +10 min | T 2 m (K) | 0.00505 | 0.0457 | 0.0012 | 1.8e-05 |
| +10 min | U (m/s) | 0.0228 | 0.114 | -0.013 | 0.018 |
| +10 min | V (m/s) | 0.0219 | 0.0917 | 0.011 | 0.015 |
| +10 min | βροχή (mm/h) | 0.00133 | 0.0387 | 8.9e-05 | (ξηρό) |
| +10 min | ps (Pa) | 48 | 337 | -2.9 | 0.00048 |
| +60 min | T 2 m (K) | 0.0281 | 0.278 | 0.0064 | 9.8e-05 |
| +60 min | U (m/s) | 0.0868 | 0.482 | -0.035 | 0.066 |
| +60 min | V (m/s) | 0.122 | 0.476 | 0.062 | 0.079 |
| +60 min | βροχή (mm/h) | 0.00237 | 0.0549 | 0.00021 | (ξηρό) |
| +60 min | ps (Pa) | 54.6 | 300 | 19 | 0.00055 |

Κλίμακα: στο +60′ το ίδιο το FP32 μοντέλο απέχει από την αλήθεια SCREAM T2m 0,097 K, U/V 0,25/0,26 m/s, ps 131 Pa, βροχή 0,0033 mm/h (RMSE).
Δηλαδή η απόκλιση INT8 είναι ~20–50% του σφάλματος του μοντέλου. Η ΔT2m RMSE μεγαλώνει περίπου γραμμικά: 0,005 → 0,010 → 0,014 → 0,019 → 0,023 → 0,028 K.

### Αθήνα 64×64 (εσωτερικό 32×32)

| lead | πεδίο | RMSE | max \|Δ\| | bias | rel L2 |
|---|---|---|---|---|---|
| +10 min | T 2 m (K) | 0.00324 | 0.0108 | 0.00024 | 1.1e-05 |
| +10 min | U (m/s) | 0.0152 | 0.0601 | -0.0068 | 0.016 |
| +10 min | V (m/s) | 0.0162 | 0.0577 | 0.0073 | 0.014 |
| +10 min | βροχή (mm/h) | 0.00197 | 0.0412 | 0.00017 | (ξηρό) |
| +10 min | ps (Pa) | 46.8 | 263 | -18 | 0.00047 |
| +60 min | T 2 m (K) | 0.0202 | 0.0821 | -0.00026 | 7e-05 |
| +60 min | U (m/s) | 0.0736 | 0.269 | -0.017 | 0.084 |
| +60 min | V (m/s) | 0.0796 | 0.427 | 0.021 | 0.066 |
| +60 min | βροχή (mm/h) | 0.0023 | 0.0473 | 0.00016 | (ξηρό) |
| +60 min | ps (Pa) | 48.5 | 250 | 2.4 | 0.00049 |

**Διάσπαση σφάλματος (Αθήνα 64×64, ίδιο rollout):** browser − PyTorch με ακριβή INT8-emulation (`prep/run_reference.py --quant weights/strata_v1_int8.bin`)
= μόνο το σφάλμα υλοποίησης WASM:

| lead | T 2 m (K) | U (m/s) | V (m/s) | βροχή (mm/h) | ps (Pa) |
|---|---|---|---|---|---|
| +10 min, RMSE | 7.5e-05 | 3.5e-04 | 3.4e-04 | 1.7e-05 | 0.87 |
| +60 min, RMSE | 2.8e-04 | 2.4e-03 | 1.7e-03 | 2.5e-05 | 1.16 |

Δηλαδή ~1–3% της συνολικής απόκλισης από το FP32· το υπόλοιπο είναι κβάντιση των βαρών. Δύο ανεξάρτητα τρεξίματα του browser έδωσαν ίδιους αριθμούς (ντετερμινιστικό).

### Puntarenas 96×96 (βροχερή περίπτωση, συναγωγή έως ~100 mm/h)

| lead | πεδίο | RMSE | max \|Δ\| | bias | rel L2 |
|---|---|---|---|---|---|
| +10 min | T 2 m (K) | 0.0317 | 0.281 | -0.02 | 0.00011 |
| +10 min | U (m/s) | 0.0511 | 0.354 | -0.028 | 0.016 |
| +10 min | V (m/s) | 0.0656 | 0.303 | 0.031 | 0.014 |
| +10 min | βροχή (mm/h) | 0.721 | 8.86 | -0.079 | 0.072 |
| +10 min | ps (Pa) | 273 | 921 | -250 | 0.0028 |
| +60 min | T 2 m (K) | 0.124 | 0.549 | -0.095 | 0.00042 |
| +60 min | U (m/s) | 0.2 | 1.47 | -0.11 | 0.061 |
| +60 min | V (m/s) | 0.239 | 1.48 | 0.12 | 0.064 |
| +60 min | βροχή (mm/h) | 1.38 | 17 | -0.2 | 0.13 |
| +60 min | ps (Pa) | 259 | 1090 | -230 | 0.0026 |

Κλίμακα (FP32 − SCREAM, RMSE στο +10′ / +60′): T2m 0,12 / 0,21 K· U 0,14 / 0,38 m/s· V 0,14 / 0,45 m/s· βροχή 2,5 / 3,5 mm/h· ps 273 / 245 Pa.
Μέση βροχή στην περιοχή 3,2 mm/h, max ~104 mm/h.
**Σημαντικό:** το INT8 δίνει εδώ συστηματικό **bias −2,3 έως −2,5 hPa στο ps**. Το ps είναι διαγνωστικό και δεν ανατροφοδοτεί την κατάσταση.
Στην Αθήνα το bias ήταν < 0,2 hPa. Δοκιμή: με το pixel stage σε FP32 (μόνο το semantic backbone σε INT8, PyTorch emulation, 1 βήμα) το bias μένει −245 Pa και η βροχή 0,72 mm/h RMSE.
Άρα το bias προέρχεται από την κβάντιση του semantic backbone, και το FP32 στο pixel stage δεν το διορθώνει (`data/puntarenas/reference_int8_scope_*.nc`).

## 7. Χρόνος και μνήμη (desktop Chrome 157, i5-6200U, 4 threads)

| | Αθήνα 96×96 | Αθήνα 64×64 | Puntarenas 96×96 |
|---|---|---|---|
| χρόνος/βήμα, browser WASM INT8 | **285,4 s** (282–290) | **129,8 s** (125–150)· 2ο τρέξιμο 129,2 | **285,9 s** (282–292) |
| χρόνος/βήμα, PyTorch FP32 CPU (ίδιο μηχάνημα) | 236,5 s | 93,0 s | 240,4 s |
| μνήμη WASM (μέγιστη) | **1.229 MiB** | **808 MiB** | 1.229 MiB |
| μέγιστη RSS μεγαλύτερης διεργασίας Chrome | **1.513 MiB** | **985 MiB** | 1.579 MiB |
| άθροισμα RSS όλων των διεργασιών Chrome | 2.350 MiB | 1.825 MiB | 2.417 MiB |
| φόρτωση βαρών (319 MiB, localhost) | 1,1 s | 1,0 s | 1,1 s |
| δεδομένα περιοχής που κατεβάζει ο browser | 25,4 MiB | 14,7 MiB | 25,4 MiB |

Τα 6 βήματα (60′) θέλουν εδώ ~29 λεπτά στο 96×96 και ~13 λεπτά στο 64×64.
Single-thread εκδοχή (`strata_st.wasm`, χωρίς cross-origin isolation, π.χ. κινητό μέσω LAN): 64×64 **273,8 s/βήμα**, WASM 805 MiB, RSS 931 MiB, με ακριβώς το ίδιο αποτέλεσμα (ΔT2m 3,24e-3 K). Άρα 2,1× επιτάχυνση από 4 threads σε 2 φυσικούς πυρήνες.
Το browser είναι ~1,2–1,4× πιο αργό από το PyTorch FP32 (oneDNN/MKL, AVX2) στην ίδια CPU, με 4× μικρότερα βάρη. Η WASM SIMD είναι 128-bit και δεν υπάρχει i8 dot χωρίς relaxed-SIMD.

Κατανομή χρόνου ενός forward στο 96×96 (ms): GEMM INT8 173.823 · NA3D 60.918 · adaLN (upsample+dwconv+GELU) 18.303 · elementwise 16.645 · κβάντιση ενεργοποιήσεων 9.571 · υπόλοιπα < 3.000.
Το INT8 GEMM τρέχει ~30 GMAC/s σε 2 φυσικούς πυρήνες (≈5 MAC/κύκλο/πυρήνα).

## 8. Δομή παραδοτέων

```
prep/        προετοιμασία σε υπολογιστή (Python): δείγμα, αναφορά, INT8, export για web
engine/      strata.c + build.sh (Emscripten)
web/         το demo: index.html, app.js, worker.js, strata_mt_worker.js, strata_step.mjs, strata_weights.mjs,
             engine/*.wasm, data/<πόλη>/*, weights/ (symlinks), serve.py
tests/       NA3D vs NATTEN, engine vs PyTorch, preprocess check, ORT experiments, browser benchmark
weights/     strata_v1/ (επίσημα, επαληθευμένα), strata_v1_int8.bin, strata_v1_int8_g128.bin
data/        sample.nc, reference_*.nc, capture.npz ανά περιοχή
logs/        όλες οι μετρήσεις (json/txt)
```

## 9. Benchmark για κινητό

Στον υπολογιστή: `python web/serve.py --port 8000` (στέλνει COOP/COEP, που χρειάζονται για threads).

Κινητό Android με USB debugging: `adb reverse tcp:8000 tcp:8000`, και στο Chrome του κινητού:

    http://localhost:8000/?bench=1&city=athens64&steps=6&threads=4&bits=16

(`localhost` είναι secure context, άρα δουλεύουν τα threads.) Το ίδιο με `city=athens` για 96×96 (~1,5 GB RSS· σε κινητό 6 GB μάλλον χωράει, αλλά το 64×64 είναι ασφαλέστερο).
Η σελίδα γράφει στο log χρόνο ανά βήμα, μνήμη WASM και την απόκλιση από το PyTorch· το πλήρες JSON βρίσκεται στο `window.__result`
(στο Chrome DevTools μέσω `chrome://inspect`: `copy(JSON.stringify(__result))`). Χωρίς adb, μέσω `http://<IP-υπολογιστή>:8000` η σελίδα δεν είναι
cross-origin isolated και τρέχει αυτόματα με 1 thread (`python web/serve.py --host 0.0.0.0 --port 8080`· ελέγχθηκε από `http://192.168.1.29:8080`).
Για threads μέσω LAN: στο Chrome του κινητού `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → προσθήκη `http://<IP>:8080` → Enabled → επανεκκίνηση Chrome.
Αν το κινητό δεν συνδέεται και το ufw είναι ενεργό: `sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp`.

## 10. Αναπαραγωγή (ακριβείς εντολές, από τη ρίζα του project)

```bash
# 0. περιβάλλον
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "numpy>=2.1" "pandas<3" "xarray>=2026.7" "zarr>=3" einops scipy cftime python-dotenv pyyaml typer tqdm \
  netCDF4 h5netcdf h5py fsspec aiohttp requests dacite omegaconf loguru tensordict timm jaxtyping warp-lang onnx==1.23.2 \
  onnxruntime==1.32.0 onnxscript psutil huggingface_hub hatchling hatch-vcs s3fs GitPython hydra-core termcolor treelib nvtx \
  pydantic importlib-metadata matplotlib torchmetrics lightning pytorch-lightning pytorch-optimizer tensorboard cartopy lmdb \
  torch-harmonics==0.8.0 tensorly tensorly-torch dask wandb pyarrow nest_asyncio gcsfs pygrib
.venv/bin/pip install --no-deps fme==2026.1.1 earth2studio==0.13.0
# NATTEN 0.21.7 (CPU, χωρίς libnatten), physicsnemo και earth2grid από source, χωρίς build isolation:
curl -LO https://files.pythonhosted.org/packages/c5/10/98fa5bc8e20937c757b5dbb2e7e949af38bc46aa92da804b0ceb07c90a86/natten-0.21.7.tar.gz && tar xzf natten-0.21.7.tar.gz
NATTEN_CUDA_ARCH="" .venv/bin/pip install --no-deps --no-build-isolation ./natten-0.21.7
.venv/bin/pip install --no-deps --no-build-isolation https://github.com/NVIDIA/physicsnemo/archive/fae6bb2b0e96488a052d9627d41e6dc94653f06a.tar.gz
.venv/bin/pip install --no-deps --no-build-isolation https://github.com/NVlabs/earth2grid/archive/6e39b069f0558cd0eff0ec666ede5e125d79b4ff.tar.gz
git clone https://github.com/NVlabs/STRATA third_party/STRATA && git -C third_party/STRATA checkout d56a78caf410bd58dd9082765d2826ed98479374
git clone https://github.com/emscripten-core/emsdk tools/emsdk && tools/emsdk/emsdk install 6.0.13 && tools/emsdk/emsdk activate 6.0.13
(cd tools && npm i puppeteer-core@24 onnxruntime-web@1.31.1)

# 1. βάρη + επαλήθευση SHA-256 (MANIFEST.md) + βοηθητικά αρχεία
.venv/bin/hf download nvidia/strata --revision 407ccb1e19a23303ebf114acd8a87655b7463f5a --local-dir weights/strata_v1
python3 prep/verify_manifest.py weights/strata_v1
.venv/bin/python prep/validate_region_latlon.py --res 64 --work /tmp/ll64
.venv/bin/python prep/make_latlon_global.py --out weights/strata_v1/latlon_ne1024pg2.nc
cp third_party/STRATA/data/scream_vertical_coordinate.nc weights/strata_v1/
.venv/bin/python prep/validate_na3d.py --full

# 2. δείγμα περιοχής (οποιεσδήποτε συντεταγμένες) + αναφορά PyTorch
export SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal
.venv/bin/python prep/fetch_sample.py --name athens --lat 37.98 --lon 23.73
.venv/bin/python prep/crop_sample.py --src athens --name athens64 --interior 32 --halo 16
.venv/bin/python prep/fetch_sample.py --name puntarenas --lat 9.98 --lon -84.84
.venv/bin/python prep/run_reference.py --name athens --steps 6
.venv/bin/python prep/run_reference.py --name athens64 --steps 6
.venv/bin/python prep/run_reference.py --name puntarenas --steps 6
# (μετά το βήμα 3) PyTorch με ακριβή INT8-emulation, για τη διάσπαση σφάλματος / το τεστ mixed precision:
.venv/bin/python prep/run_reference.py --name athens64 --steps 6 --quant weights/strata_v1_int8.bin --act-bits 16 --no-capture
.venv/bin/python prep/run_reference.py --name puntarenas --steps 1 --quant weights/strata_v1_int8.bin --int8-scope strata.backbone --tag int8_scope_strata.backbone --no-capture

# 3. INT8
.venv/bin/python prep/quantize_int8.py --out weights/strata_v1_int8.bin
.venv/bin/python prep/quantize_int8.py --out weights/strata_v1_int8_g128.bin --group 128

# 4. WASM + web data
engine/build.sh
for c in athens athens64 puntarenas; do .venv/bin/python prep/export_web.py --name $c; done && .venv/bin/python prep/web_index.py
mkdir -p web/weights && ln -sf ../../weights/strata_v1_int8.bin web/weights/ && ln -sf ../../weights/strata_v1_int8_g128.bin web/weights/

# 5. έλεγχοι
.venv/bin/python tests/na3d_kernel_test.py
.venv/bin/python tests/engine_test.py --tile 96 --capture data/athens/capture.npz
NODE=$(ls tools/emsdk/node/*/bin/node); $NODE tests/check_preprocess.mjs athens /tmp/x_js.bin
.venv/bin/python tests/ort_experiment.py --parts ABC

# 6. browser
python3 web/serve.py            # → http://127.0.0.1:8000
$NODE tests/browser_bench.mjs --city athens --steps 6 --threads 4 --bits 16          # ή athens64 / puntarenas
python3 tests/summarize_bench.py logs/browser_bench_athens_t4_a16_strata_v1_int8.json
$NODE tests/browser_bench.mjs --city athens64 --steps 6 --out logs/browser_bench_athens64_decomp.json
python3 tests/summarize_bench.py logs/browser_bench_athens64_decomp.json int8_a16   # browser − INT8-emulation
$NODE tests/browser_bench.mjs --city athens --steps 1 --bits 8 --out logs/bench_a8_1step.json                     # W8A8
$NODE tests/browser_bench.mjs --city athens --steps 1 --weights strata_v1_int8_g128.bin --out logs/bench_g128_1step.json
```

## 11. Περιορισμοί / ανοιχτά

- Δεν μετρήθηκε σε πραγματικό κινητό (δεν υπήρχε διαθέσιμο)· οι αριθμοί είναι από φορητό 2015.
- Περιοχές μόνο μέσα σε μία όψη του κύβου (όχι στις ακμές του κύβου).
- Το STRATA v1 βγάζει πάντα μηδενική στερεή βροχόπτωση (γνωστός περιορισμός του checkpoint)· δεν συγκρίνεται.
- Οι ενεργοποιήσεις INT8 (W8A8) δεν είναι ακριβείς αρκετά· με relaxed-SIMD i8 dot θα μπορούσαν να διπλασιάσουν την ταχύτητα, αλλά θέλουν 7-bit/asymmetric σχήμα (δεν δοκιμάστηκε).
