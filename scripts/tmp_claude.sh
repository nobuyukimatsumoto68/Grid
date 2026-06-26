#!/bin/bash
# Rebuild conn binary after adding NERSC header pre-check (skip bad configs).
# Run on the cluster (ROCm/hipcc). Reads the result back from the log.
set -u

dir=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4
log=/g/g91/matsumoto5/workdir/build_conn_nersc_check_claude.log

{
  echo "==== build conn (Makefile_claude, in-place) $(date) ===="
  source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
  make -C "${dir}" -f Makefile_claude baryons_0000_dirac_claude
  rc=$?
  echo "==== make exit code: ${rc} ===="
  if [ "${rc}" -ne 0 ]; then
    echo "BUILD FAILED"
    exit 1
  fi
  echo "BUILD OK: ${dir}/baryons_0000_dirac_claude"
  ls -l "${dir}/baryons_0000_dirac_claude"
} 2>&1 | tee "${log}"
