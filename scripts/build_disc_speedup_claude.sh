#!/bin/bash
# build_disc_speedup_claude.sh
# Builds the two new disc-speedup example targets after their Make.inc entries
# were added:
#   - disc_multipleGamma_binary_mixedprec_claude   (chunk #1, mixed precision)
#   - disc_mrhs_defl_bench_claude                  (chunk #2+#3 benchmark)
# Adding a Make.inc target requires regenerating the build Makefile (automake +
# config.status) before compiling. No file is ever deleted. All output is teed to
# build_disc_speedup_claude.log. Compilation only -- no flux job is submitted.
#
# Run this yourself (Claude does not compile); then send me
# build_disc_speedup_claude.log. Compile is serial; add -jN to the make lines if
# you want parallelism (each Grid TU is RAM-heavy).
set -u

ROOT=/usr/workspace/lsd/matsumoto5/su4_32c
GRIDSRC=${ROOT}/Grid
BUILD=${ROOT}/build
LOG=${ROOT}/build_disc_speedup_claude.log
TARGETS="disc_multipleGamma_binary_mixedprec_claude disc_mrhs_defl_bench_claude disc_multipleGamma_binary_defl_claude disc_lma_bench_claude"

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
