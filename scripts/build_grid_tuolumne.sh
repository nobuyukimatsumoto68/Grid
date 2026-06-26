#!/bin/bash
# set -e

source env.sh


dir_build=$(pwd)
dir_opt=/usr/workspace/lsd/matsumoto5/opt
dir_log=${dir_build}/log

mkdir -p ${dir_log}

cd ${dir_build}
git clone https://paboyle@github.com/paboyle/Grid

cd Grid
# git checkout a4d11a630f2e416d6735e78aebb6939e0790a639

# e0d5e3c6c7b416d31440066705073a99477db544
# git checkout 79ad567dd5766bd76b99c6d4b12a82cbe919a349
# git checkout a4d11a630f2e416d6735e78aebb6939e0790a639
# git checkout -b june15_2023 e3e1cc19620b8ee9834dfb35491ff6c36857d52c
# git checkout -b may11_2023 876c8f4478886479bcd1e505e5481bf34afb8958
# rm -rf Grid/qcd/action/fermion/instantiation/Sp*
# rm -rf tests/sp2n/*.cc

# find . -type f -name 'FTHMC*' -delete
# find . -type f -name 'FTHMC*' -delete
# find . -type f -name '*FTHMC*' -delete
# rm HMC/HMC2p1f_3GeV.cc
# rm tests/forces/Test_fthmc.cc
# rm tests/forces/Test_gpwilson_force.cc
# mv ${dir_build}/dweofa_mobius_HSDM_v3.cc ${dir_build}/Grid/examples/


time ./bootstrap.sh 2>&1 | tee ${dir_log}/bootstrap.log

cd ..
mkdir -p build; cd build

time ../Grid/configure --prefix=${dir_build}/build/ \
     CXX=$CC \
     MPICXX=$CC \
     CXXFLAGS="-fPIC ${MPI_CFLAGS}  -L/lib64  -fgpu-sanitize --offload-arch=gfx942 -fopenmp" \
     LDFLAGS="-L/lib64 -lamdhip64 -lhipblas -lrocblas ${MPI_LDFLAGS}" \
     --with-lime=${dir_opt}/lime-1.3.2/ \
     --enable-comms=mpi-auto \
     --enable-unified=no \
     --enable-shm=nvlink \
     --enable-tracing=timer \
     --enable-accelerator=hip \
     --enable-gen-simd-width=64 \
     --enable-simd=GPU \
     --enable-openmp \
     --enable-accelerator-cshift \
     --enable-Nc=4 \
     --disable-gparity \
     --disable-fermion-reps \
     2>&1 | tee ${dir_log}/configure.log # -enable-gen-simd-width=64

#########

time make -j 16 2>&1 | tee ${dir_log}/make.log
# time make check -j 8 2>&1 | tee ${dir_log}/make_check.log
time make install 2>&1 | tee ${dir_log}/make_install.log






## USEFUL COMMANDS
# lscpu
# echo | gcc -x c++ -E -Wp,-v - >/dev/null
