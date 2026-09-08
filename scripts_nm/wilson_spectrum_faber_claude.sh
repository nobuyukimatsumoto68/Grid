#!/usr/bin/env bash
# COMPARE the improved non-Hermitian eigensolvers on the low D_W spectrum of the smooth Q=-3 config
# (raw ckpoint_lat.240 op-reflow to 4 t0), at the HARD Nstop=40 case that previously drifted to true
# residual ~0.3. Two back-to-back runs into the same log:
#   (A) plain IRA on D_W, smallest-|lambda|  -- now with the Chunk-A fixes (Leja shifts + DGKS f^+ reorthog
#       + periodic basis reorthonormalization). Expect the true (RR-cleanup) residuals to drop vs before.
#   (B) FABER-Arnoldi: IRA on the Chebyshev-ellipse filter T_n((D_W-c)/d), largest-modulus wanted; the
#       ellipse (auto-fit to the bulk Ritz estimate) makes the near-zero modes peripheral = Arnoldi's stable
#       case. D_W eigenpairs recovered by the SAME RR-cleanup (exact for D_W). --faber-ord = degree n.
# Both must recover the 3 chiral (chi ~ +-1) modes (index theorem, Q=-3). Driver Test_wilson_frameopt_claude.cc.
# Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_spectrum_faber_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
MPREC=0.3
NSTOP=40
NK=64
NM=100
FORD=12

{
  echo "======== compile Test_wilson_frameopt_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_frameopt_claude.o
  BIN=${BUILD}/Test_wilson_frameopt_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_frameopt_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo ""
  echo "======== (A) improved plain IRA, Nstop=${NSTOP} ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM}

  echo ""
  echo "======== (B) Faber-Arnoldi (ellipse filter, ord=${FORD}), Nstop=${NSTOP} ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --faber --faber-ord ${FORD} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "SPECTRUM|D_W low|FABER|bulk fit|D_W RR mode\[|M0act\[|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
