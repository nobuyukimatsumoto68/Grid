#!/bin/bash
# Wilson free-prec HEADLINE at 16^4 (Chunk 1 frame + Chunk 2 RB-CGNE vs FGMRES(M0_W)). Wilson fits fp64
# trivially (Krylov ~12 MiB/vector at 16^4 -> restart-80 ~2 GiB) -- the DW memory wall is gone.
# MASS PHYSICS: Wilson physical quark mass = m - m_crit, m_crit ~ -0.8 at b6.0. So m=0.1 is HEAVY (easy,
# validates the pipeline); the small-physical-mass / large-kappa demo regime is m NEAR m_crit (negative).
# Scan m = 0.1 (validate), -0.5, -0.7 (toward chiral) and watch RB-CGNE iters CLIMB while FGMRES(M0_W)
# stays flatter -> the win-condition on Wilson. Baselines non-erroring (exceptional modes cap, don't abort);
# per-config tune m further if -0.7 spikes. restart 80. Single GPU. Wall needs a quiet GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_headline_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc
MASSES="-0.7 -0.75 -0.78 -0.80"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  if [ ! -f "${CFG16}" ]
  then
    echo "CONFIG NOT FOUND: ${CFG16}"
    exit 2
  fi

  echo "======== compile Test_wilson_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_wilson_freeprec_hl_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_hl_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_claude.cc -o ${OBJ}
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

  # LIGHT-OPERATOR CROSSOVER at FIXED m_prec=0.8 (the Tikhonov optimum from the m_prec scan, ~operator-mass
  # independent). Drive the operator mass toward m_crit(~-0.82) -> physical mass -> ~0 -> RB-CGNE kappa
  # EXPLODES; if FGMRES(M0_W at fixed 0.8) stays FLATTER, count-win blows past the ~3.6x per-iter penalty ->
  # WALL WIN. Omega built ONCE (--mass-list). Physical masses for {-0.7,-0.75,-0.78,-0.80}: ~{0.12,0.07,0.04,0.02}.
  echo "======== run: 16^4 Wilson crossover, masses ${MASSES// /,}, m_prec=0.8 (fixed), restart=80 ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --mass-list "${MASSES// /,}" --mprec 0.8 --restart 80

  echo "======== SUMMARY (per mass) ========"
  grep -nE "cold gate|headline solve|m_prec=|residual proxy|RB-CGNE\(Wilson\)|CGNE\(full\)|FGMRES\(M0_W\)|WALL speedup|Landau functional" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: (1) residual proxy small = M0_W a good approx inverse (frame works). (2) RB-CGNE iters CLIMB"
  echo "      as m -> m_crit (~-0.8) while FGMRES(M0_W) stays flatter -> count + WALL win emerges. Tune m"
  echo "      per config if a mass spikes (exceptional real mode)."
} 2>&1 | tee ${LOG}
