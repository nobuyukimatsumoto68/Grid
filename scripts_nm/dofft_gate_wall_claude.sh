#!/bin/bash
# DofFFT wire-in GATE + WALL. Builds the free-prec test TWICE -- DEFAULT (PlannedFFT) and with
# -DFREEMOBIUS5D_DOFFT (hybrid: custom radix-L DofFFT on UNSPLIT spacetime dims, PlannedFFT on split) --
# and runs BOTH at 8^4 m=0.1 --ops cgne,m0 --restart 20 (the wall-optimal restart). Compares:
#   [GATE] gate 1 (||F D v - v||) must PASS in the DOFFT build (DofFFT is bit-matched -> F unchanged to eps).
#   [FFT ] report_timers fft_fwd/fft_bwd: DOFFT should be LOWER (unsplit dims skip the pencil transpose).
#   [WALL] FGMRES(M0) WALL + "WALL speedup (RB-CGNE / FGMRES-M0)": does cheaper F push it toward/over 1.0?
# 8^4 fp32: dims x,y UNSPLIT -> DofFFT (2 of 4); z,t split -> PlannedFFT. So a PARTIAL FFT win expected.
#
# *** PURE WALL-CLOCK TEST -- RUN ONLY ON A QUIET GPU. *** Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/dofft_gate_wall_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  SRCF=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc

  for VARIANT in default dofft
  do
    EXTRA=""
    if [ "${VARIANT}" = "dofft" ]
    then
      EXTRA="-DFREEMOBIUS5D_DOFFT"
    fi
    echo "======== [${VARIANT}] compile (EXTRA='${EXTRA}') $(date) ========"
    OBJ=${BUILD}/Test_dwf_freeprec_dofft_${VARIANT}_claude.o
    BIN=${BUILD}/Test_dwf_freeprec_dofft_${VARIANT}_claude
    ${CXX} ${CXXFLAGS} ${EXTRA} -I${SRC} -c ${SRCF} -o ${OBJ}
    rc=$?
    if [ ${rc} -ne 0 ]
    then
      echo "[${VARIANT}] COMPILE FAILED (rc=${rc})"
      exit ${rc}
    fi
    ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
    rc=$?
    if [ ${rc} -ne 0 ]
    then
      echo "[${VARIANT}] LINK FAILED (rc=${rc})"
      exit ${rc}
    fi
    echo "======== [${VARIANT}] run: 8^4 m=0.1 --ops cgne,m0 --restart 20 ========"
    ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops cgne,m0 --restart 20
  done

  echo "======== SUMMARY (default vs dofft) ========"
  grep -nE "\[default\] run|\[dofft\] run|gate 1 .*(PASS|FAIL)|fft_fwd|fft_bwd|FFT fraction|RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE|M0/Dhop" ${LOG} | grep -vE "warning|::write|strlen|\.h\("
  echo "======== done ========"
  echo "READ: DOFFT gate 1 PASS (bit-exact); DOFFT fft_fwd/fft_bwd LOWER than default; compare FGMRES(M0)"
  echo "      WALL + WALL-speedup-vs-RB default vs dofft -- did the cheaper F move it toward/over 1.0?"
} 2>&1 | tee ${LOG}
