#!/bin/bash
#FLUX: -t 120m
#FLUX: --output=flow_24c_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# M5 eigenvalue-flow scan (eye4_anti) on the two LIGHT 24^3x48 ensembles
# (b10p800_m0p0100, b10p840_m0p0500) -- the same configs as the mres check / the 32c
# production point. For the LAST FLOW_NCFG configs of each ensemble it scans M5 over
# M5_LIST and collects the lowest eigenvalues of MdagM = H_W(M5)^2; pick M5 where the
# smallest |H_W| is largest / most stable (fewest near-zero modes -> smallest m_res).
#
# Self-contained -- ONE submission runs the whole scan (loops ensembles x configs x M5,
# one flux run of eye4_anti per point):
#   flux batch run_flow_24c_claude.sh
#
# eye4_anti CLI: eye4_anti <config> <alpha> <beta_cheby> <M5>  [Grid flags]
#   WilsonFermionD mass = -M5, antiperiodic {1,1,1,-1}; ALWAYS writes ./evals.dat (truncated
#   each call) -> we run each call in its own dir and cat-append into one table per config.
# NOT resumable: re-running re-does the scan (COLLECT is truncated). Fine -- the scan is short.

set -u
date

ENV_SH=/usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/eye4_anti

GRID_GEOM="24.24.24.48"
MPI_GEOM="2.2.2.4"           # 32 ranks -> 8 nodes at tasks-per-node=4 (4D Wilson op)
NODES=8
TPN=4
THREADS=8

# index-aligned light ensembles (paired beta <-> mass), 24c conf_nc4nf1_2448 naming
massstrs=(0p0100 0p0500)
betastrs=(10p800 10p840)

FLOW_NCFG=5                  # last N configs per ensemble (each = an 11-point M5 scan)
M5_LIST=$(seq 1.0 0.1 2.0)   # scan around the 1.5 ballpark
ALPHA=0.05                   # Chebyshev lo edge (calibrated via tune_cheby_claude.sh)
BETA_CHEBY=12                # Chebyshev hi edge; RE-CHECK the window at any M5 where the low
                             # spectrum drops toward zero (a too-small/mis-placed window thrashes)

export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
source "${ENV_SH}"
OPTIONS="--comms-concurrent --comms-overlap --shm 2048 --shm-mpi 1"

jmax=${#massstrs[@]}
for((j=0;j<$jmax;j++))
do
  massstr=${massstrs[$j]}
  betastr=${betastrs[$j]}
  cfgname=conf_nc4nf1_2448_b${betastr}_m${massstr}
  latdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgname}
  obsdir=/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}

  if [ ! -d "${latdir}" ]; then
    echo "skip ${cfgname}: no ${latdir}"
    continue
  fi
  mkdir -p "${obsdir}"

  CLIST=$(ls -1 ${latdir}/${cfgname}_lat.* 2>/dev/null | sed 's/.*_lat\.//' | sort -n | tail -n ${FLOW_NCFG})
  echo "============================================================"
  echo "${cfgname}: M5 flow on last ${FLOW_NCFG} configs: ${CLIST}"
  echo "============================================================"

  for conf in ${CLIST}; do
    CFG=${latdir}/${cfgname}_lat.${conf}
    OUTDIR=${obsdir}/flow_${conf}
    mkdir -p "${OUTDIR}"
    COLLECT=${OUTDIR}/flow_${cfgname}_${conf}_claude.dat
    : > "${COLLECT}"   # start fresh (truncate via redirect, no rm)

    for M5 in ${M5_LIST}; do
      echo "--- ${cfgname} conf ${conf}  M5=${M5} ---"
      PARAMS="--grid ${GRID_GEOM} --mpi ${MPI_GEOM} --threads ${THREADS} --accelerator-threads ${THREADS} ${OPTIONS}"
      ( cd "${OUTDIR}" && \
        flux run -N ${NODES} --tasks-per-node=${TPN} --verbose --exclusive \
            --setopt=mpibind=verbose:1 \
            "${APP}" "${CFG}" "${ALPHA}" "${BETA_CHEBY}" "${M5}" ${PARAMS} ) \
        2>&1 | tee "${OUTDIR}/eye4_${cfgname}_${conf}_M5_${M5}_claude.log"

      if [ -f "${OUTDIR}/evals.dat" ]; then
        cat "${OUTDIR}/evals.dat" >> "${COLLECT}"
      fi
    done
    echo "collected flow table: ${COLLECT}"
  done
done

date
echo "ALL 24c FLOW SCANS DONE"
