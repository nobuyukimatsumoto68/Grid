#!/bin/bash
# Chunk B validation: fp32 F on the DEFAULT (barrel) path vs double default. Builds TWO binaries -- DEFAULT
# and -DFREEMOBIUS5D_FP32 -- and runs each 8^4 --config --ops m0 back-to-back (same GPU load). Check:
#   (1) cold gate 0a/0b PASS, gate 1 PASS (fp32 tol 5e-5, ~single eps -- exact-in-single);
#   (2) FGMRES M0 converges to 1e-8 (outer double); iteration count within a few of default's 79
#       (single preconditioner only changes iteration count, not final accuracy);
#   (3) [F timers] total drops ~2x (halved bytes: pack+unpack+solve+cuFFT all ComplexF).
# grid_packonce_fft_impl_plan_claude.md Chunk B.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
LOG=${ROOT}/fp32_validate_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  # fp32 is now the DEFAULT (no flags); double is the reference via -DFREEMOBIUS5D_FP64. Confirms the flip:
  # plain build must give the fp32 numbers (gate 3.3e-7, 79 iters, ~2500 us) and FP64 the double reference.
  echo "======== compile FP64 reference (-DFREEMOBIUS5D_FP64) $(date) ========"
  OBJ_D=${BUILD}/Test_dwf_freeprec_fp64_claude.o
  BIN_D=${BUILD}/Test_dwf_freeprec_fp64_claude
  ${CXX} ${CXXFLAGS} -DFREEMOBIUS5D_FP64 -I${SRC} -c ${TEST} -o ${OBJ_D}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "FP64 COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ_D} -o ${BIN_D} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "FP64 LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== compile DEFAULT (no flags = fp32 now) ========"
  OBJ_F=${BUILD}/Test_dwf_freeprec_default_claude.o
  BIN_F=${BUILD}/Test_dwf_freeprec_default_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ_F}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DEFAULT COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ_F} -o ${BIN_F} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DEFAULT LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== run FP64 reference 8^4 --config --ops m0 ========"
  ${BIN_D} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0

  echo "======== run DEFAULT (fp32) 8^4 --config --ops m0 ========"
  ${BIN_F} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "done"
} 2>&1 | tee ${LOG}
