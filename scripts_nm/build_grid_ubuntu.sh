#!/bin/bash

# set -e

# add_patch_ventura=true

dir_home=/mnt/hdd_baracuda/
dir_opt=${dir_home}opt/
dir_grid_install=$(pwd)/
dir_grid_build=${dir_grid_install}/build/


# DEPENDENCIES
# mkdir -p ${dir_opt}


# # GMP
# echo " ======= install gmp  ======="
# dir_gmp=${dir_opt}gmp-6.2.1/
# if [ ! -d ${dir_gmp} ]; then
#     cd $dir_opt
#     wget https://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.2.1.tar.bz2 | tar -xj
#     cd $dir_gmp
#     ./configure 2>&1 | tee log_configure
#     make -j 8 2>&1 | tee log_make
#     make check -j 8 2>&1 | tee log_check
#     sudo make install -j 8 2>&1 | tee log_install
#     echo " -- installation finished -- "
#     source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi


# # MPFR
# echo " ======= install mpfr ======="
# dir_mpfr=${dir_opt}mpfr-4.1.0/
# if [ ! -d ${dir_mpfr} ]; then
#     cd $dir_opt
#     wget https://gcc.gnu.org/pub/gcc/infrastructure/mpfr-4.1.0.tar.bz2 | tar -xj
#     cd $dir_mpfr
#     ./configure --with-gmp-include=/usr/local/include --with-gmp-lib=/usr/local/lib 2>&1 | tee log_configure
#     make -j 8 2>&1 | tee log_make
#     make check -j 8 2>&1 | tee log_check
#     sudo make install -j 8 2>&1 | tee log_install
#     echo " -- installation finished -- "
#     source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi


# ****
# # HDF5
# echo " ======= install hdf5 ======="
# dir_hdf5=${dir_opt}hdf5/
# if [ ! -d ${dir_hdf5} ]; then
#     cd $dir_opt
#     git clone https://github.com/mokus0/hdf5.git
#     cd $dir_hdf5
#     ./configure --prefix=/usr/local/hdf5/ --enable-cxx 2>&1 | tee log_configure
#     make -j 8 2>&1 | tee log_make
#     make check -j 8 2>&1 | tee log_check
#     sudo make install -j 8 2>&1 | tee log_install
#     make check-install -j 8 2>&1 | tee log_check_install
#     echo " -- installation finished -- "
#     # source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi
# ****




# # OPENSSL
# echo " ======= install OPENSSL ======="
# dir_openssl=${dir_opt}openssl/
# if [ ! -d ${dir_openssl} ]; then
#     cd $dir_opt
#     git clone https://github.com/openssl/openssl.git
#     cd $dir_openssl
#     ./config --prefix=/opt/openssl --openssldir=/usr/local/ssl 2>&1 | tee log_config
#     make -j 8 2>&1 | tee log_make
#     make test -j 8 2>&1 | tee log_test
#     sudo make install 2>&1 | tee log_install
#     echo " -- installation finished -- "
#     source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi


# # OPENMPI
# echo " ======= install openmpi ======="
# dir_openmpi=${dir_opt}openmpi-4.1.5/
# if [ ! -d ${dir_openmpi} ]; then
#     cd $dir_opt
#     wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2 | tar -xj
#     cd $dir_openmpi
#     ./configure --prefix=/usr/local/openmpi --enable-mpi-cxx 2>&1 | tee log_configure
#     sudo make all install -j 8 2>&1 | tee log_make
#     echo " -- installation finished -- "
#     source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi


# # FFTW
# echo " ======= install FFTW ======="
# dir_fftw=${dir_opt}fftw-3.3.10/
# if [ ! -d ${dir_fftw} ]; then
#     cd $dir_opt
#     wget https://www.fftw.org/fftw-3.3.10.tar.gz | tar -xz
#     cd $dir_fftw
#     ./configure --enable-threads --with-openmp --enable-mpi 2>&1 | tee log_config
#     make -j 8 2>&1 | tee log_make
#     make check -j 8 2>&1 | tee log_check
#     sudo make install -j 8 2>&1 | tee log_install
#     make installcheck -j 8 2>&1 | tee log_installcheck
#     ./configure --enable-float --enable-threads --with-openmp --enable-mpi 2>&1 | tee log_config2
#     make -j 8 2>&1 | tee log_make2
#     make check -j 8 2>&1 | tee log_check2
#     sudo make install -j 8 2>&1 | tee log_install2
#     make installcheck -j 8 2>&1 | tee log_installcheck2
#     echo " -- installation finished -- "
#     source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi


# # LIME
# echo " ======= install LIME ======="
# dir_lime=${dir_opt}lime-1.3.2/
# if [ ! -d ${dir_lime} ]; then
#     cd $dir_opt
#     wget http://usqcd-software.github.io/downloads/c-lime/lime-1.3.2.tar.gz
#     tar xzf lime-1.3.2.tar.gz
#     cd $dir_lime
#     ./configure
#     make -j 8 2>&1 | tee log_make
#     make check -j 8 2>&1 | tee log_check
#     sudo make install -j 8 2>&1 | tee log_install
#     echo " -- installation finished -- "
#     # source ~/.zshrc
# else
#     echo " -- alreday installed. skip -- "
# fi



# ### GRID
# # export CC=/opt/homebrew/bin/gcc-13
export CC="/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc"
export CXX="/usr/local/cuda-12.6/bin//nvcc"
export MPICXX="/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++"
export CXXFLAGS="-std=c++17 -I/usr/local/include -I/mnt/hdd_barracuda/opt/openmpi_cuda/include -I/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/include/"
export LDFLAGS="-L/usr/local/lib -L/opt/openssl/lib -L/mnt/hdd_barracuda/opt/openmpi_cuda/lib/ -L/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/lib/ -lhdf5 -lhdf5_cpp" #
# export LDFLAGS+="-L/usr/local/lib -L/opt/openssl/lib -L/mnt/hdd_barracuda/opt/openmpi_cuda/lib/ -L/usr/local/hdf5/lib/ -lhdf5 -lhdf5_cpp"
# # LINKING LAPACK should also be possible: https://www.google.com/search?client=firefox-b-1-d&q=eigen+lapack
# # LAPACK: sudo port install lapack
# # export LDFLAGS="-L/usr/local/lib -L/opt/openssl/lib -L/usr/local/openmpi/lib/ -L/usr/local/hdf5/lib/ -lhdf5 -lhdf5_cpp -framework Accelerate /opt/local/lib/lapack/liblapacke.dylib"


# echo " ======= install GRID ======="
mkdir -p ${dir_grid_install}; cd ${dir_grid_install}
if [ ! -d Grid ]; then
    git clone git@github.com:nobuyukimatsumoto68/Grid.git
    # git clone https://github.com/paboyle/Grid.git
fi

cd Grid

git switch dwf_prec

./bootstrap.sh 2>&1 | tee bootstrap_log
mkdir -p ${dir_grid_build}; cd ${dir_grid_build}

# -lhdf5_cpp"

time ${dir_grid_install}Grid/configure \
     --prefix=${dir_grid_build} --with-lime=${dir_lime} \
     MPICC="/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc" \
     MPICXX="/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++" \
     CXXFLAGS="-std=c++17 -I/usr/local/include -I/mnt/hdd_barracuda/opt/openmpi_cuda/include -I/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/include/ -ccbin ${MPICXX} -gencode arch=compute_70,code=sm_70 -Xcompiler -fPIC -Xcompiler -fopenmp" \
     LDFLAGS="-L/usr/local/lib -L/opt/openssl/lib -L/mnt/hdd_barracuda/opt/openmpi_cuda/lib/ -L/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/lib/ -lhdf5 -lhdf5_cpp" \
     --enable-comms=mpi \
     --enable-unified=no \
     --enable-shm=no \
     --enable-tracing=timer \
     --enable-accelerator=cuda \
     --enable-gen-simd-width=32 \
     --enable-simd=GPU \
     --enable-accelerator-aware-mpi \
     --enable-openmp \
     --disable-gparity \
     --disable-fermion-reps \
     --enable-accelerator-cshift \
     2>&1 | tee configure.log # -enable-gen-simd-width=64



     # --enable-comms=mpi \
     # --enable-simd=AVX512 \
     # --enable-gen-simd-width=512 \


     # --enable-shm=nvlink \

# CXXFLAGS="-fPIC ${MPI_CFLAGS}  -L/lib64  -fgpu-sanitize --offload-arch=gfx90a -fopenmp" \
    # LDFLAGS="-L/lib64 -lamdhip64 -lhipblas -lrocblas ${MPI_LDFLAGS}" \


# CXXFLAGS="-fPIC ${MPI_CFLAGS} -L/lib64  -fgpu-sanitize --offload-arch=gfx90a -fopenmp" \
    # LDFLAGS="-L/lib64 -lamdhip64 -lhipblas -lrocblas ${MPI_LDFLAGS}" \


# bash ${dir_grid_install}Grid/configure --prefix=${dir_grid_build} --with-lime=${dir_lime} --enable-comms=mpi-auto --
# bash ${dir_grid_install}Grid/configure --prefix=${dir_grid_build} --with-lime=${dir_lime} --enable-comms=mpi --enable-lapack=/opt/local/lib/lapack/
make -j 4 2>&1 | tee log_make
# make check -j 2 2>&1 | tee log_check
make install 2>&1 | tee log_install









# # EIGEN
# echo " ======= install eigen5 ======="
# dir_eigen=${dir_opt}eigen-3.4.0/
# if [ ! -d ${dir_eigen} ]; then
#     cd $dir_opt
#     wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz
#     echo " -- installation finished -- "
# else
#     echo " -- alreday installed. skip -- "
# fi

# source ~/.zshrc



# # cmake
# cd $dir_opt
# wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4.tar.gz
# tar xvzf cmake-3.26.4.tar.gz
# cd cmake-3.26.4
# ./bootstrap
# mkdir install 2>&1 | tee install_log
# make
# sudo make install --prefix="/Users/nobuyukimatsumoto/opt/cmake-3.26.4/install" 2>&1 | tee make_make_install_log


#
#
#
#
### USEFUL COMMANDS
# uname -a
## echo | gcc -x c++ -E -Wp,-v - >/dev/null
