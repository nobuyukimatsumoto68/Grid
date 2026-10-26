#!/bin/bash
# build_disc_hit_claude.sh
# Rebuild the three disc binaries after the DISC_HIT change
# (Grid/examples/disc_second_hit_impl_plan_claude.md): plain, mixed-precision, LMA v2.
# Run on a TUOLUMNE login node (ROCm modules), after git pull. No automake regen needed
# (targets already exist in Make.inc). Output tees to build_disc_hit_claude.log.
# Claude does not compile -- run this yourself and send me the log.
set -u

ROOT=/usr/workspace/lsd/matsumoto5/su4_32c
BUILD=${ROOT}/build
LOG=${ROOT}/build_disc_hit_claude.log
TARGETS="disc_multipleGamma_binary_claude disc_multipleGamma_binary_mixedprec_claude disc_multipleGamma_binary_lma_v2_claude"

cd "${ROOT}" || exit 1
source "${ROOT}/env.sh"

{
  echo "############ start $(date) ############"
  echo "==== [1/2] compile + link ${TARGETS} ===="
  make -C "${BUILD}/examples" ${TARGETS} || { echo "FAILED: make"; exit 1; }

  echo "==== [2/2] verify binaries contain DISC_HIT ===="
  ok=1
  for t in ${TARGETS}; do
    if [ -x "${BUILD}/examples/${t}" ] && grep -q DISC_HIT "${BUILD}/examples/${t}"; then
      echo "OK   ${t}"
    else
      echo "BAD  ${t}"
      ok=0
    fi
  done
  [ "${ok}" -eq 1 ] && echo "BUILD OK" || echo "BUILD INCOMPLETE"
  echo "############ end $(date) ############"
} 2>&1 | tee "${LOG}"
