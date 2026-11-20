#!/usr/bin/env bash
# Build the STRATA WASM engine: threaded (pthreads, needs cross-origin isolation)
# and single-threaded variants. Requires emsdk 6.0.13 (tools/emsdk).
set -euo pipefail
cd "$(dirname "$0")"
source ../tools/emsdk/emsdk_env.sh > /dev/null
OUT=../web/engine
mkdir -p "$OUT"
COMMON=(strata.c -O3 -msimd128
  -sMODULARIZE=1 -sEXPORT_NAME=StrataEngine -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=64MB -sSTACK_SIZE=1MB
  -sEXPORTED_FUNCTIONS=_st_init,_st_malloc,_st_free,_st_bind,_st_check_bound,_st_set_act_bits,_st_set_group,_st_configure,_st_set_rope,_st_set_debug,_st_forward,_st_timing,_st_tm,_st_na3d,_st_heap_mib,_st_alloc_failures,_st_final_only,_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP8,stringToNewUTF8,wasmMemory)
emcc "${COMMON[@]}" -pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency -o "$OUT/strata_mt.js"
emcc "${COMMON[@]}" -o "$OUT/strata_st.js"
# threaded demo worker = engine glue + demo worker logic (pthreads re-load this same script)
cat "$OUT/strata_mt.js" ../web/worker.js > ../web/strata_mt_worker.js
ls -la "$OUT" ../web/strata_mt_worker.js
