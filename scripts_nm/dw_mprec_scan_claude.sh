#!/bin/bash
# DW MASS-SHIFT (Tikhonov) scan on the HEADLINE chiral operator. The Wilson m_prec scan showed a heavier
# preconditioner mass markedly improves FGMRES (Tikhonov/mass-floor). DW has NO additive renorm (m_crit=0,
# so m_prec=mm is "matched"), but the Tikhonov effect may STILL help -> could push the DW 8^4 near-parity
# (0.91x vs RB) toward a WALL WIN on the chiral fermion. 8^4 DW FITS memory (no 16^4 Krylov wall).
# Operator fixed at m=0.1; sweep --mprec {0.1,0.15,0.2,0.3,0.5,0.8}. --ops cgne,m0 = full CGNE + RB-CGNE +
# FGMRES(M0). restart 256 (8^4 no-restart fits). Re-invokes per m_prec (re-flows Omega -- fast at 8^4).
# Single GPU; wall needs a quiet GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/dw_mprec_scan_claude.log
CFG8=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
MPRECS="0.1 0.15 0.2 0.3 0.5 0.8"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc (--mprec added) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_dwf_freeprec_dwmp_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_dwmp_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for MP in ${MPRECS}
  do
    echo "======== run: 8^4 DW operator m=0.1, m_prec=${MP}, --ops cgne,m0 --restart 256 ========"
    ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG8} --ops cgne,m0 --mprec ${MP} --restart 256
  done

  echo "======== SUMMARY (per m_prec) ========"
  grep -nE "free-prec built at m_prec|CGNE\(full\):|RB-CGNE:|FGMRES\(M0\) restart|WALL speedup" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: does a heavier m_prec (>0.1) drop FGMRES(M0) below the matched-0.1 value (~79)? If so, the DW"
  echo "      near-parity vs RB improves -- possibly to a WALL WIN on the chiral operator."
} 2>&1 | tee ${LOG}
