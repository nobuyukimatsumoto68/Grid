#!/bin/bash
# THE 16^4 GATE: the decisive free-prec win-condition test (large-L, where 8^4 was volume-floored).
# Measures FGMRES(M0) vs RB-CGNE at 16^4, and whether F is FFT-bound at 16^4 (PlannedFFT scales
# super-linearly: probe fwd 1043->20851 us at 8->16^4). Two questions answered in ONE run:
#   (1) WIN-CONDITION: does RB-CGNE's iter count GROW at 16^4 (lower gap -> lambda_min~m dominates ->
#       kappa penalty finally bites) while FGMRES(M0) stays flat -> does the honest WALL-vs-RB cross 1.0?
#   (2) FFT-BOUND?: report_timers FFT fraction of M0 at 16^4 (vs 61% at 8^4) -> if F is now FFT-dominated,
#       a faster FFT (the fused multi-stage DofFFT, currently HELD) becomes the lever. This gates that build.
# fp32 default; single GPU (16^4 fits ~4.4 GiB). restart-sweep 30,20,15 finds the wall optimum at 16^4.
#
# *** SET CFG16 to the SCC-provided b6.0 quenched 16^4 NERSC config before running. ***
# *** PURE WALL-CLOCK TEST -- RUN ONLY ON A QUIET GPU (counts load-independent, wall not). *** No rm/kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/gate16_freeprec_claude.log

# ---- FILL THIS IN once the 16^4 config is rsynced from SCC (path to the .nersc file) ----
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  if [ ! -f "${CFG16}" ]
  then
    echo "CONFIG NOT FOUND: ${CFG16} -- set CFG16 to the SCC b6.0 quenched 16^4 NERSC file first."
    exit 2
  fi

  echo "======== compile Test_dwf_freeprec_claude.cc (default: PlannedFFT, fp32) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_gate16_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_gate16_claude
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

  echo "======== run: 16^4 m=0.1 --ops cgne,m0 --restart-sweep 30,20,15 ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --ops cgne,m0 --restart-sweep 30,20,15 --mass 0.1

  echo "======== SUMMARY ========"
  grep -nE "headline mass m =|RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE|fft_fwd|fft_bwd|FFT fraction|M0/Dhop|Dhop\] us" ${LOG} | grep -vE "warning|::write|strlen|\.h\("
  echo "======== done ========"
  echo "READ: (1) RB-CGNE iters at 16^4 vs the 8^4 ~181-239 plateau -- did they GROW? FGMRES(M0) iters vs"
  echo "         ~93-98? WALL speedup (RB-CGNE/FGMRES) >1 = free-prec WINS wall = the crossover reached."
  echo "      (2) FFT fraction of M0 at 16^4 (vs 61% @8^4): if FFT-dominated -> the fused DofFFT build is on."
} 2>&1 | tee ${LOG}
