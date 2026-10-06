#!/bin/bash
# compile_dwf_pvmg_claude.sh -- build dwf_pvmg_test_claude.cc against the local Grid build
# (same pattern as Grid_sdm_build/compile_two_baryon_claude.sh: grid-config flags + source-tree includes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID="${GRID:-${ROOT}/../../build}"
if [ -f "${GRID}/bin/grid-config" ]; then
    CONFIG="${GRID}/bin/grid-config"
else
    CONFIG="${GRID}/grid-config"
fi

SRC_NAME="${1:-dwf_pvmg_test_claude.cc}"
SRC="${ROOT}/${SRC_NAME}"
BIN="${ROOT}/$(basename "${SRC_NAME}" .cc)"

CXX="$(${CONFIG} --cxx)"
CXXFLAGS="$(${CONFIG} --cxxflags)"
LDFLAGS="$(${CONFIG} --ldflags)"
LIBS="$(${CONFIG} --libs)"

GRID_SRC="${GRID_SRC:-${ROOT}/..}"
if [ -f "${GRID_SRC}/Grid/Grid.h" ]; then
    CXXFLAGS="-I${GRID_SRC} ${CXXFLAGS}"
fi
if [ -f "${GRID}/Grid/Config.h" ]; then
    CXXFLAGS="-I${GRID}/Grid ${CXXFLAGS}"
fi

echo "Grid build : ${GRID}"
echo "Compiler   : ${CXX}"
echo "Source     : ${SRC}"
echo "Output     : ${BIN}"
${CXX} ${CXXFLAGS} ${LDFLAGS} ${LIBS} "${SRC}" -o "${BIN}"
echo "Done: ${BIN}"
