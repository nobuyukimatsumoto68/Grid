#!/usr/bin/env bash
# DECISIVE test (user 2026-09-05): |Q|=3 => there MUST be 3 chiral modes (chi ~ +-1). Which non-Hermitian
# eigensolver recovers all 3 on the smooth Q=-3 config (ckpoint_lat.240 op-reflow 4t0)?
#   (A) ORIGINAL plain IRA (smallest-|lambda|), now with the Chunk-A Leja+reorthog fixes.
#   (B) FABER-Arnoldi (Chebyshev-ellipse filter, largest-modulus wanted).
# Matched settings Nstop=16 Nk=32 Nm=64 (the setting where plain IRA earlier found 3 chiral: modes 0,1,3
# chi 0.93/0.95/0.90). Mass sweep m = 0.10, 0.05, 0.02 (toward massless: chiral modes -> smaller |lambda|,
# better separated from the bulk). Count RR modes with |chi|>0.5 and read their true residuals.
# Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_chiral_scan_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MPREC=0.3
NSTOP=16
NK=32
NM=64
FORD=12
MASSES="0.10 0.05 0.02"

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

  for M in ${MASSES}; do
    echo ""
    echo "############ mass=${M} : (A) ORIGINAL IRA (smallest-|lambda|) ############"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
        --flow-nstep 0 --spectrum --mass ${M} --mprec ${MPREC} \
        --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM}

    echo ""
    echo "############ mass=${M} : (B) FABER-Arnoldi (ellipse ord=${FORD}) ############"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
        --flow-nstep 0 --spectrum --faber --faber-ord ${FORD} --mass ${M} --mprec ${MPREC} \
        --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM}
  done

  echo ""
  echo "======== SUMMARY (chiral modes: chi with |chi|>0.5 are topological) ========"
  grep -nE "mass=0|ORIGINAL IRA|FABER|bulk fit|D_W RR mode\[|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
