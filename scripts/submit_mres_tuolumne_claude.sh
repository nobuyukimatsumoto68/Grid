#!/bin/bash
#FLUX: -t 120m
#FLUX: --output=mres_{{id}}
#FLUX: -q pbatch
#FLUX: -N 2
#FLUX: -n 8
#FLUX: -g 1
#FLUX: --exclusive
#
# FLUX batch launcher for the residual-mass (mres) measurement on the 24^3x48
# ensembles, via Mobius_mesons_xt. One job per ensemble; processes the LAST 20 saved
# configs serially. Per config: point-source Mobius solve (Ls=16, b/c=1.5/0.5, M5 from
# --env) + ContractJ5q, writing mres.<conf>.h5 into obsdir with datasets PJ5q_t / PJ5q_x
# and 10 meson channels (incl G5_G5). mres = sum_t PJ5q_t / sum_t G5_G5_t at large t.
# Driver run_mres_tuo_claude.sh passes BIN/M5/mass/cfgfilename/latdir/obsdir via --env.
#
# Mobius_mesons_xt_claude CLI: <config> <M5> <mass> <outfile>  (Ls=16 hardcoded; boss-guarded
# HDF5 writer). Configs with a COMPLETE output h5 (has the last dataset GZ_GZ_t) are skipped;
# truncated/partial files are recomputed, so re-submitting resumes safely. Runs on 8 ranks.

Lattice="24.24.24.48"
MpiGrid="2.2.2.1"     # 8 ranks (-N 2 x 4 GPUs); local 12.12.12.48

### four gpus per node
date
here=`pwd`

export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
export HDF5_USE_FILE_LOCKING=FALSE

### environment + binary (binary built by the driver on the login node)
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=${BIN}

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

mkdir -p ${obsdir}

# last 20 saved configs in latdir (highest indices)
CLIST=$(ls ${latdir}/${cfgfilename}_lat.* 2>/dev/null | sed "s|.*_lat\.||" | sort -n | tail -n 20)
echo "configs (last 20): ${CLIST}"

for conf in ${CLIST}; do
    cfg=${latdir}/${cfgfilename}_lat.${conf}
    outfile=${obsdir}/mres.${conf}.h5

    if [ ! -f "${cfg}" ]; then
        echo "skip conf ${conf}: config not found ${cfg}"
        continue
    fi
    # skip only if the output is COMPLETE: a valid h5 containing the LAST-written dataset
    # (GZ_GZ_t). A truncated/partial file (e.g. the old 96-byte aborted writes) fails this
    # and is recomputed -- the boss-guarded binary truncates+rewrites it. (Old existence-only
    # skip would have wrongly kept the corrupt files.)
    if h5ls -r "${outfile}" 2>/dev/null | grep -q "GZ_GZ_t"; then
        echo "skip conf ${conf}: complete output exists ${outfile}"
        continue
    fi

    echo "==== conf ${conf} -> ${outfile} ===="
    # binary positional args: <config> <M5> <mass> <outfile>
    flux run -N 2 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP ${cfg} ${M5} ${mass} ${outfile} $PARAMS
done

echo "--end " `date` `date +%s`
