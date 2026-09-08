#!/bin/bash
# 16^4 CROSSOVER, m=0.1 FIRST (go down only after this converges). Fixes from the interrupted run:
#  - RESTART: restart-20 STALLS at 16^4 (restart << N_it ~80-100 -> flexible GMRES loses convergence).
#    NO-restart would OOM (fp64 Krylov vector ~100 MiB at 16^4; N_it*100MiB > 12 GiB). So sweep a MID
#    restart that BOTH converges AND fits memory: 80 (~8 GiB Krylov) and 50 (~5 GiB). Watch which converges.
#  - WALL FLUCTUATION: --repeat 4 -> each solve re-run 4x, report the MIN wall (least GPU-contention noise).
#  - Start at m=0.1 only; extend to 0.01/0.001 in a follow-up ONCE m=0.1 converges cleanly.
# --ops rb,m0 = RB-CGNE honest baseline + FGMRES(M0) (no expensive full CGNE). COUNTS load-independent;
# WALL needs a quiet GPU. fp32 M0 path. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/mass_crossover16_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  if [ ! -f "${CFG16}" ]
  then
    echo "CONFIG NOT FOUND: ${CFG16}"
    exit 2
  fi

  echo "======== compile Test_dwf_freeprec_claude.cc (--repeat added) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_masscross16_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_masscross16_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "COMPILE FAILED (rc=${rc})"
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "LINK FAILED (rc=${rc})"
    exit ${rc}
  fi

  echo "======== run: 16^4 m=0.1 --ops rb,m0 --restart-sweep 80,50 --repeat 4 ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --ops rb,m0 --restart-sweep 80,50 --repeat 4 --mass 0.1

  echo "======== SUMMARY ========"
  grep -nE "headline mass m =|Q_5Li|RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE|FFT fraction|M0/Dhop|\[rep " ${LOG} | grep -vE "warning|::write|strlen|\.h\("
  echo "======== done ========"
  echo "READ: does FGMRES(M0) CONVERGE at restart 80 and/or 50 (iters << 20000)? RB-CGNE iters at 16^4"
  echo "      m=0.1 vs the 8^4 181? WALL(min of 4) spread across reps? WALL speedup (RB/FGMRES). If clean,"
  echo "      extend to m=0.01, 0.001 next."
} 2>&1 | tee ${LOG}
