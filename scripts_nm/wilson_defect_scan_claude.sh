#!/usr/bin/env bash
# WILSON free-prec FRAME-QUALITY sweep vs DEFECT FLOW, Grid+GPU 16^4 (user 2026-09-04, the R2 prize).
# Wilson kills the DW 16^4 Krylov MEMORY WALL: no Ls => the fermion is Ls x smaller (~12.6 MB/vec vs ~100 MB
# for DW Ls=8), so full NO-RESTART FGMRES(M0_W) fits the GPU. The frame Omega (flow+Landau) is gauge-only =
# IDENTICAL to what DW would use, so the Landau-functional frame quality (F5 win-predictor) and its Q-
# dependence carry over. PRIMARY output = "frame: ... Landau functional=" (mass-free, printed before any
# solve); FGMRES(M0_W) vs RB-CGNE = secondary.
#
# Dedicated test Test_wilson_freeprec_defect_scan_claude.cc (copy of Test_wilson_freeprec_claude.cc + the
# --flow-eps/--flow-nstep knobs). PANEL (frame recipe per config):
#   BAD  ref: raw ckpoint_lat.240 (Q=-3)          standard flow tau=2 (--flow-nstep 100)
#   GOOD ref: cfg_su3_16161616_b6.0 (trivial Q)   standard flow tau=2 (--flow-nstep 100)
#   DEFECT  : the three 4 t0 checkpoints nowarm/4d/3d  NO reflow (--flow-nstep 0)
# Wilson m_crit caveat: bare --mass here is a first pass (frame quality is mass-FREE; FGMRES count is
# secondary and may want a physical-mass --mprec -- see grid_free_wilson_results_claude.md).
# Single GPU. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_defect_scan_claude.log
CKDIR=${ROOT}/dwf4_qcd_claude/defect_ckpts_claude
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240
TRIV=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
RESTART=256   # Wilson 16^4 fits no-restart (the DW memory wall is gone) -> clean N_it

{
  echo "======== compile Test_wilson_freeprec_defect_scan_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_freeprec_defect_scan_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_defect_scan_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_defect_scan_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  run_one () {
    echo ""
    echo "############## $1 :: $2  (flow-nstep $3) ##############"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --mass ${MASS} --restart ${RESTART} \
        --config "$2" --flow-nstep "$3" --flow-eps 0.02
  }

  run_one "BAD_raw_Q-3"       "${RAW}"                                                 100
  run_one "GOOD_trivialQ"     "${TRIV}"                                                100
  run_one "DEFECT_nowarm_4t0" "${CKDIR}/ckpoint_lat.240_nowarm_s1m1N1_ts14_4t0.nersc"   0
  run_one "DEFECT_4d_4t0"     "${CKDIR}/ckpoint_lat.240_4d_s1m1N1_ts12_4t0.nersc"       0
  run_one "DEFECT_3d_4t0"     "${CKDIR}/ckpoint_lat.240_3d_s1m1N1_ts10_4t0.nersc"       0

  echo ""
  echo "======== SUMMARY (frame quality + Wilson free-prec win per config) ========"
  grep -nE "##############|frame: plaq|Landau functional|RB-CGNE\(Wilson\)|FGMRES\(M0_W\)|WALL speedup" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
