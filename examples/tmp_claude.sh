#!/bin/bash
# tmp_claude.sh -- PVMG chunk-3b: realistic Ls=8 at weak coupling, volume + disorder scan
# RUN ONLY WHEN GPU DEVICE 0 IS CLEAR (a crash under MPS can kill co-tenant jobs).
# (user runs; Claude reads dwf_pvmg_chunk3b_claude.log)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
LOG=dwf_pvmg_chunk3b_claude.log

export GRID_ALLOC_NCACHE_LARGE=2

COMMON8="--grid 8.8.8.8 --mpi 1.1.1.1 --Ls 8 --M5 1.5 --b 1.0 --c 1.0 --mass 0.01 --device-mem 6000"
COMMON16="--grid 16.16.16.8 --mpi 1.1.1.1 --Ls 8 --M5 1.5 --b 1.0 --c 1.0 --mass 0.01 --device-mem 6000"

{
  echo "==== BUILD ===================================================="
  ./compile_dwf_pvmg_claude.sh dwf_pvmg2_test_claude.cc || { echo "BUILD FAILED"; exit 1; }

  echo "==== RUN 1: g=0.25, 8^4, Ls=8, FIXED-apply A/B ================"
  ./dwf_pvmg2_test_claude ${COMMON8} --restart 64 --gdis 0.25 --phase 3 --fixed \
    || { echo "RUN 1 FAILED"; exit 1; }

  echo "==== RUN 2: g=0.25, 16^3x8, Ls=8 (fixed phase 3 + baselines) =="
  for ARGS in "3 --fixed" "3" "4" "5"; do
    echo "---- RUN 2 phase ${ARGS} ----"
    ./dwf_pvmg2_test_claude ${COMMON16} --restart 24 --gdis 0.25 --phase ${ARGS} \
      || { echo "RUN 2 PHASE ${ARGS} FAILED"; exit 1; }
  done

  echo "==== RUN 3: g=0.5, 16^3x8, Ls=8 (fixed phase 3 + baselines) ==="
  for ARGS in "3 --fixed" "3" "4" "5"; do
    echo "---- RUN 3 phase ${ARGS} ----"
    ./dwf_pvmg2_test_claude ${COMMON16} --restart 24 --gdis 0.5 --phase ${ARGS} \
      || { echo "RUN 3 PHASE ${ARGS} FAILED"; exit 1; }
  done

  echo "==== ALL DONE ================================================="
} 2>&1 | tee "${LOG}"
