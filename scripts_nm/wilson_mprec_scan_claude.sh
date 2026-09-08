#!/bin/bash
# PRECONDITIONER-MASS SCAN (user idea): fix the operator at m_bare=-0.7 (physical ~0.12 at m_crit~-0.82),
# sweep the free-prec mass m_prec to find the EMPIRICAL optimum -- may prefer a slightly HEAVIER m_prec than
# the naive physical 0.12 (Tikhonov/mass-floor: too-light F_W over-amplifies near-zero modes). RB-CGNE is
# computed ONCE (operator fixed); Omega built ONCE. --mprec-list sweeps F_W's mass.
# READ: min-WALL m_prec = the operator's preferred preconditioner mass; is it > 0.12 (heavier preferred)?
# does count-win(RB/FG) or wall-win peak away from the naive value? Single GPU; wall needs a quiet GPU.
# No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_mprec_scan_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_wilson_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_freeprec_mp_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_mp_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  # RE-SCAN m_prec AT THE LIGHT OPERATOR MASS (m=-0.80, physical ~0.02, RB=641). The optimum m_prec~0.8 was
  # found at m=-0.7 only, then FIXED for the crossover -- untested at light mass, where the near-zero modes
  # (and thus the Tikhonov optimum) may SHIFT. If a different m_prec beats the 220 iters we got at fixed 0.8,
  # the crossover verdict improves. Breakeven for a WALL win here: FGMRES < RB/3.6 = 641/3.6 = 178 iters.
  echo "======== run: 16^4 operator m=-0.80, m_prec scan {0.3,0.5,0.8,1.2,2.0,3.0}, restart=80 ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --mass -0.80 --mprec-list "0.3,0.5,0.8,1.2,2.0,3.0" --restart 80

  echo "======== SUMMARY ========"
  grep -nE "preconditioner-mass scan|RB-CGNE\(Wilson\)|m_prec=.*FGMRES|COMPILE FAILED|LINK FAILED" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: pick the m_prec with the LOWEST FGMRES WALL (and highest wall-win). Is it heavier than the"
  echo "      naive 0.12? That sets the preconditioner-mass offset to use in the light-mass crossover scan."
} 2>&1 | tee ${LOG}
