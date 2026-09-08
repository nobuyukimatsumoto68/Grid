#!/bin/bash -l
# BU SCC SGE batch: measure flowed topological charge Q(tau) on the b2.6 16^4 ensemble, filling in the
# configs that were never scanned. Thin wrapper over grid_flow_topocharge_1node_qsub_claude.sh with
# CONFIGDIR baked in -- CONFIGDIR mode auto-SKIPS any config that already has a flowQ_*.dat, so re-running
# is safe and only the missing configs get flowed. Writes flowQ_dat/flowQ_<cfg>_claude.dat (tau plaq
# Q_clover Q_5Li); the plateau (last row) is the integer Q.
#
# Submit:  qsub grid_flowQ_b26_rescan_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f log/flowQb26.o<jobid>

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N flowQb26
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=4:00:00
#$ -pe omp 16

set -u

ROOT=/projectnb/qfe/nmatsum/dwf
export CONFIGDIR=${CONFIGDIR:-${ROOT}/configs_iwasaki_16_b2.6}
export GRID=${GRID:-16.16.16.16}
export FLOW_ACTION=${FLOW_ACTION:-iwasaki}
export FLOW_EPS=${FLOW_EPS:-0.01}
export FLOW_NSTEP=${FLOW_NSTEP:-400}   # tau up to 4 (16^4 plateau)
export MEAS=${MEAS:-20}

echo "=== flowQ b2.6 rescan $(date)  host ${HOSTNAME:-?}  NSLOTS=${NSLOTS:-?}  CONFIGDIR=${CONFIGDIR} ==="
# run the pipeline body (its own #$ directives are inert when invoked via bash; NSLOTS is inherited)
bash "${ROOT}/Grid/scripts_nm/grid_flow_topocharge_1node_qsub_claude.sh"
echo "=== done $(date) ==="
