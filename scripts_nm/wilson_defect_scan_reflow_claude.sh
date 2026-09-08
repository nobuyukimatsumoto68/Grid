#!/usr/bin/env bash
# ADDITIONAL Wilson frame-quality run (user 2026-09-04): the smoothness-matched, seam-healed comparison that
# the first sweep lacked. Uses the new --op-reflow-nstep knob (standard-flow the LOADED config before building
# operator+frame). All configs now SMOOTH => matched m_crit (common-mode), so the free-prec win difference
# isolates Q. Same iteration-count / D_W-count metric (ignore wall on the congested GPU).
#   CONV_smooth_Q-3 : raw ckpoint_lat.240 (Q=-3) CONVENTIONALLY flowed to 4 t0  (the MISSING smooth Q!=0 ref)
#   nowarm/4d/3d +1t0: the 4 t0 defect checkpoints REFLOWED +1 t0 (heal the open-boundary seam)
# op-reflow-eps 0.05 (matches the defect flow's eps); 4 t0 = 232 steps, +1 t0 = 58 steps. --flow-nstep 0
# (frame from the reflowed operator directly). Compare against the AS-IS numbers in wilson_defect_scan_claude.log.
# Test = Test_wilson_freeprec_defect_scan_claude.cc (branched copy). Single GPU. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_defect_scan_reflow_claude.log
CKDIR=${ROOT}/dwf4_qcd_claude/defect_ckpts_claude
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
RESTART=256
RFEPS=0.05      # reflow eps (matches the defect flow)
CONV=232        # 4 t0 standard (conventional-flow the raw config to the checkpoints' smoothness)
HEAL=58         # +1 t0 (heal the defect seam)

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
    echo "############## $1 :: $2  (op-reflow $3) ##############"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --mass ${MASS} --restart ${RESTART} \
        --config "$2" --op-reflow-eps ${RFEPS} --op-reflow-nstep "$3" --flow-nstep 0
  }

  run_one "CONV_smooth_Q-3"        "${RAW}"                                                 ${CONV}
  run_one "nowarm_reflow+1t0"      "${CKDIR}/ckpoint_lat.240_nowarm_s1m1N1_ts14_4t0.nersc"  ${HEAL}
  run_one "4d_reflow+1t0"          "${CKDIR}/ckpoint_lat.240_4d_s1m1N1_ts12_4t0.nersc"      ${HEAL}
  run_one "3d_reflow+1t0"          "${CKDIR}/ckpoint_lat.240_3d_s1m1N1_ts10_4t0.nersc"      ${HEAL}

  echo ""
  echo "======== SUMMARY (op-reflow + frame quality + iters) ========"
  grep -nE "##############|OP-REFLOW|frame: plaq|RB-CGNE\(Wilson\)|FGMRES\(M0_W\)" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
