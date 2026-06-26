#!/bin/bash
#
# Benchmark driver (login node): build the v4 and Hasenbusch HMC binaries, set up an A/B
# benchmark from ONE shared starting config (32^3x64, m=0.01, b=10.8), and flux-batch a
# single job that runs each for N_BENCH trajectories and tabulates cost + acceptance.
#   v4         : single EOFA factor det[D(m)/D(pv)], 2-level integrator.
#   hasenbusch : det[D(m)/D(m1)]*det[D(m1)/D(pv)], m1=sqrt(m*pv)=0.1, 3-level integrator.
# Refs: Hasenbusch hep-lat/0107019; EOFA Chen-Chiu arXiv:1403.1683.  See hmc_bench_impl_plan_claude.md.

date; hostname
basedir=$(pwd)
src=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4

source env.sh
make -C ${src} dweofa_mobius_HSDM_v4                  || { echo "BUILD v4 FAILED";         exit 1; }
make -C ${src} dweofa_mobius_HSDM_hasenbusch_claude   || { echo "BUILD hasenbusch FAILED"; exit 1; }
BIN_V4=${src}/bin/dweofa_mobius_HSDM_v4
BIN_HASEN=${src}/bin/dweofa_mobius_HSDM_hasenbusch_claude
[ -x "${BIN_V4}" ]    || { echo "missing ${BIN_V4}";    exit 1; }
[ -x "${BIN_HASEN}" ] || { echo "missing ${BIN_HASEN}"; exit 1; }

# shared starting config (latest thermalizing 32c m=0.01 checkpoint) + bench length
SRCCFG=/p/lustre5/matsumoto5/32_64_0.01/beta10.8m0.01
START=80
N_BENCH=5

xml_template=${basedir}/ip_hmc_mobius_claude.xml
script=submit_hmc_bench_tuolumne_claude.sh
benchroot=/p/lustre5/matsumoto5/hmc_bench_32c_m0.01
mkdir -p ${benchroot}

for v in v4 hasen; do
    rundir=${benchroot}/bench_${v}
    mkdir -p ${rundir}
    # shared starting config, symlinked read-only (saveInterval > Trajectories => no writes)
    ln -sf ${SRCCFG}/ckpoint_lat.${START} ${rundir}/ckpoint_lat.${START}
    ln -sf ${SRCCFG}/ckpoint_rng.${START} ${rundir}/ckpoint_rng.${START}
    # bench XML: CheckpointStart from START, run N_BENCH traj, Metropolis ON from traj 1, no saves
    sed -e "/<gauge_beta>/{s/@BETA@/10.8/}" \
        -e "/<StartTrajectory>/{s/>[0-9]*</>${START}</}" \
        -e "/<Trajectories>/{s/>[0-9]*</>$((START+N_BENCH))</}" \
        -e "/<StartingType>/{s/HotStart/CheckpointStart/}" \
        -e "/<NoMetropolisUntil>/{s/>[0-9]*</>0</}" \
        -e "/<saveInterval>/{s/>[0-9]*</>1000</}" \
        ${xml_template} > ${rundir}/ip_bench.xml
done

echo "bench root: ${benchroot}  (start=${START}, N=${N_BENCH})"
# Two pdebug jobs (each gets the full 60m cap): v4 first, then hasenbusch chained
# --dependency=afterany on v4 (so v4's log is complete when the hasen job prints the
# combined summary). A wall-kill before all ${N_BENCH} traj is fine -- the tee'd logs keep
# every completed trajectory's metrics.
jid=$(flux batch --env=TAG=v4 --env=BIN=${BIN_V4} --env=benchroot=${benchroot} ${basedir}/${script})
echo "  v4 job:    ${jid}"
jid2=$(flux batch --dependency=afterany:${jid} --env=TAG=hasen --env=BIN=${BIN_HASEN} --env=benchroot=${benchroot} --env=SUMMARY=1 ${basedir}/${script})
echo "  hasen job: ${jid2} (afterany:${jid}, prints combined summary)"
