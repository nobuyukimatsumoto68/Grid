#!/bin/bash
# DW 2D MASS-SHIFT SCAN (experiment, overnight). Two independent masses in the free-prec block:
#   free-Wilson shift dW  -> M5_prec = 1.8 + dW  (the M5 arg to FreeMobius5DInverse -> free_dw_p kernel;
#                            enters the block DIAGONAL b*D_W+I and off-diag (c*D_W-I)P_-+). BOTH SIGNS.
#   block-solve mass dB   -> m_prec  = dB (ABSOLUTE)  (the DW quark mass -> ONLY the wall corners -m; --mprec).
# Operator FIXED at (m=0.1, M5=1.8). MATCHED reference = (dW=0, dB=0.1) [M5_prec=1.8, m_prec=0.1] ~79 iters.
# dB=0 => massless block (PV-like). --ops m0 (FGMRES(M0)
# only; RB=181 fixed -> count-win = 181/FGMRES offline). no-restart (256; 8^4 fits) for clean N_it.
# Grid: dW in {-0.5..+0.5} step 0.05 (21), dB in {0..0.5} step 0.05 (11) => 231 points, ~overnight.
# Each point echoes "POINT dW dB M5P MP" then the FGMRES line -> parse into the 2D iteration-count table.
# 8^4 DW fits memory. Non-converging points cap at maxit (err_on_no_conv=false). No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/dw_2d_mass_scan_claude.log
CFG8=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

S_WILSON="-0.5 -0.45 -0.4 -0.35 -0.3 -0.25 -0.2 -0.15 -0.1 -0.05 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"
S_BLOCK="0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc (--mprec + --m5prec) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_dwf_freeprec_2d_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_2d_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for dW in ${S_WILSON}
  do
    M5P=$(awk "BEGIN{printf \"%.4f\", 1.8+(${dW})}")
    for dB in ${S_BLOCK}
    do
      MP=$(awk "BEGIN{printf \"%.4f\", (${dB})}")
      echo "POINT dW=${dW} dB=${dB} M5P=${M5P} MP=${MP}"
      ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG8} --ops m0 --mprec ${MP} --m5prec ${M5P} --restart 256
    done
  done

  echo "======== SUMMARY (grep POINT + FGMRES for the 2D table) ========"
  grep -nE "^POINT |FGMRES\(M0\) restart" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: pair each POINT line with the following FGMRES(M0) iters -> a 21x11 (dW,dB) table."
  echo "      matched (dW=0, dB=0.1 -> M5_prec=1.8, m_prec=0.1) is the ~79-iter reference; look for the"
  echo "      (dW,dB) minimum (< 74 = beats the 1D scan; the 1D optimum was m_prec~0.25 at dW=0)."
} 2>&1 | tee ${LOG}
