#!/bin/bash
# module purge
# #module reset
module load craype-accel-amd-gfx942
# module load cmake/3.24.2	# default
# module load PrgEnv-cray/8.7.0
module load rocm/6.4.1		# for MPI on MI300A

# # 110724
# # ml cce/19.0.0 cray-mpich/8.1.32
# module load cce/20.0.0
# module load cray-mpixlate/1.0.9
# #ml cray-hdf5/1.12.2.11		# default

module list

export GTL_ROOT=/opt/cray/pe/mpich/9.0.1/gtl/lib/ # /opt/cray/pe/mpich/8.1.32/gtl/lib

export MPI_CFLAGS=" -I${MPICH_DIR}/include -I${ROCM_PATH}/include"
export MPI_LDFLAGS="-Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi -Wl,-rpath=${GTL_ROOT} -L${GTL_ROOT} -L${ROCM_PATH}/llvm/lib -Wl,-rpath=${ROCM_PATH}/llvm/lib -L${ROCM_PATH}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx942} ${PE_MPICH_GTL_LIBS_amd_gfx942} "

export MPICH_GPU_SUPPORT_ENABLED=1

export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}


export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${MPICH_DIR}/lib:${GTL_ROOT}:${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

#---------------------------------------------------
# ${ROCM_PATH}/lib: libhipblas.so, librocblas.so
#---------------------------------------------------
# ${CRAY_LD_LIBRARY_PATH}: without this one, following error occurs:
# ./conftest: error while loading shared libraries: libmodules.so.1: cannot open shared object file: No such file or directory
# configure:3671: $? = 127
# configure:3678: error: in `/g/g20/park49/ws_lsd/sungwoo/SU4_sdm/code/build_tioga/build/build_grid':
# configure:3680: error: cannot run C++ compiled programs.
#---------------------------------------------------

CC=hipcc
MPICXX=hipcc

echo "MPI_CFLAGS=${MPI_CFLAGS}"
echo "MPI_LDFLAGS=${MPI_LDFLAGS}"

