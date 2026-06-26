#!/bin/bash
# build_disc_lma_v2_claude.sh
# Builds the LMA v2 pipeline (local-agent 2026-06-26) after its Make.inc targets were
# added here:
#   - disc_lma_eigref_v2_claude            (per-ensemble shift-invert eigref -> HDF5)
#   - disc_lma_bench_v2_claude             (eigensolve gates bench)
#   - disc_lma_estimator_bench_v2_claude   (LMA estimator + DET gate + variance)
#   - disc_multipleGamma_binary_lma_claude (PRODUCTION binary, chunk D)
# All #include the header-only disc_lma_v2_common_claude.h.
# Adding Make.inc targets requires an automake regen (REGEN below) before compiling.
# No file is deleted. Output tees to build_disc_lma_v2_claude.log.
#
# Run this yourself (Claude does not compile); then send me build_disc_lma_v2_claude.log.
# Compile is serial; add -jN to the make line if you want parallelism (Grid TUs are RAM-heavy).
set -u

ROOT=/usr/workspace/lsd/matsumoto5/su4_32c
GRIDSRC=${ROOT}/Grid
BUILD=${ROOT}/build
LOG=${ROOT}/build_disc_lma_v2_claude.log
TARGETS="disc_lma_eigref_v2_claude disc_lma_bench_v2_claude disc_lma_estimator_bench_v2_claude disc_multipleGamma_binary_lma_claude"

cd "${ROOT}" || exit 1
source "${ROOT}/env.sh"

{
  echo "############ start $(date) ############"

  echo "==== [1/4] regenerate examples/Makefile.in from Make.inc (automake) ===="
  ( cd "${GRIDSRC}" && automake examples/Makefile ) || { echo "FAILED: automake"; exit 1; }

  echo "==== [2/4] regenerate build/examples/Makefile (config.status) ===="
  ( cd "${BUILD}" && ./config.status examples/Makefile ) || { echo "FAILED: config.status"; exit 1; }

  echo "==== [3/4] compile + link targets ===="
  make -C "${BUILD}/examples" ${TARGETS} || { echo "FAILED: make"; exit 1; }

  echo "==== [4/4] verify binaries ===="
  ok=1
  for t in ${TARGETS}; do
    bin="${BUILD}/examples/${t}"
    if [ -x "${bin}" ]; then
      ls -l "${bin}"
    else
      echo "MISSING: ${bin}"
      ok=0
    fi
  done
  [ "${ok}" -eq 1 ] && echo "BUILD OK" || { echo "BUILD INCOMPLETE"; exit 1; }

  echo "############ end $(date) ############"
} 2>&1 | tee "${LOG}"
