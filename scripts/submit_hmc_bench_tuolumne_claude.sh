#!/bin/bash
#FLUX: -t 60m
#FLUX: --output=hmc_bench_{{id}}
#FLUX: -q pdebug
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# ONE HMC benchmark run on the pdebug queue: run ${BIN} (tag ${TAG}) for the bench traj from
# the shared starting config and tee the log. If SUMMARY=1 also print the combined
# v4-vs-Hasenbusch table (the hasen job is chained --dependency=afterany on the v4 job, so
# both logs exist by then). pdebug caps at 60m; at ~650-1100s/traj a job may WALL before all
# 5 traj -- that is fine, the tee'd log keeps every COMPLETED trajectory's metrics (raise -t /
# use pbatch for all 5). Env from run_hmc_bench_claude.sh: TAG, BIN, benchroot, SUMMARY.

date
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS="--grid 32.32.32.64 --mpi 2.2.2.4 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ip_bench.xml"

echo "================= BENCH ${TAG} : ${BIN} ================="
cd ${benchroot}/bench_${TAG}
t0=$(date +%s)
flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 ${BIN} ${PARAMS} 2>&1 | tee bench_${TAG}.log
echo "WALL_${TAG}_TOTAL=$(( $(date +%s) - t0 ))s"

# file-scope helper: tabulate the completed-trajectory metrics from one bench log
print_metrics () {
    local tag=$1
    local log=${benchroot}/bench_${tag}/bench_${tag}.log
    local times nt nacc nrej
    times=$(grep -oE "Total time for trajectory \(s\): [0-9.]+" "${log}" 2>/dev/null | grep -oE "[0-9.]+$")
    nt=$(echo "${times}" | grep -c .)
    echo "----- ${tag} -----"
    echo "  completed traj:    ${nt}"
    echo "  per-traj wall (s): $(echo ${times} | tr '\n' ' ')"
    [ "${nt}" -gt 0 ] && echo "  mean traj wall (s): $(echo "${times}" | awk '{s+=$1;n++} END{if(n)printf "%.1f", s/n}')"
    echo "  dH per traj:       $(grep -oE "dH = [-0-9.]+" "${log}" 2>/dev/null | grep -oE "[-0-9.]+$" | tr '\n' ' ')"
    nacc=$(grep -c "Metropolis_test -- ACCEPTED" "${log}" 2>/dev/null)
    nrej=$(grep -c "Metropolis_test -- REJECTED" "${log}" 2>/dev/null)
    echo "  acceptance:        ${nacc} / $((nacc+nrej))"
}

if [ "${SUMMARY}" = "1" ]; then
    echo
    echo "===================== BENCHMARK SUMMARY (v4 vs Hasenbusch) ====================="
    print_metrics v4
    print_metrics hasen
    echo "==============================================================================="
fi
date
