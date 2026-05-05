#include <Grid/Grid.h>
#include <string>
#include <vector>
using namespace Grid;

extern "C"
void grid_init(int NP_T, int NP_X, int NP_Y, int NP_Z,
               int Nt,   int Nx,   int Ny,   int Nz)
{
    // Grid dimension order: x,y,z,t = 0,1,2,3
    std::string prog      = "hirep_io";
    std::string grid_flag = "--grid";
    std::string grid_val  = std::to_string(Nx) + "." + std::to_string(Ny) + "."
                          + std::to_string(Nz) + "." + std::to_string(Nt);
    std::string mpi_flag  = "--mpi";
    std::string mpi_val   = std::to_string(NP_X) + "." + std::to_string(NP_Y) + "."
                          + std::to_string(NP_Z) + "." + std::to_string(NP_T);
    std::vector<char*> fake_argv = {
        const_cast<char*>(prog.c_str()),
        const_cast<char*>(grid_flag.c_str()),
        const_cast<char*>(grid_val.c_str()),
        const_cast<char*>(mpi_flag.c_str()),
        const_cast<char*>(mpi_val.c_str()),
        nullptr
    };
    int    fake_argc     = 5;
    char** fake_argv_ptr = fake_argv.data();
    Grid_init(&fake_argc, &fake_argv_ptr);
    GridLogLayout();
}

// Grid_finalize() is intentionally omitted from any cleanup function:
// it calls MPI_Finalize(), but HiRep's finalize_process() owns MPI teardown.

// Fills out[] with the gauge field in HiRep's in-memory layout:
//   site order T->X->Y->Z (global coords), 4 directions, Nc x Nc complex row-major.
//   out must be pre-allocated to GLB_VOLUME * 4 * 2*Nc*Nc doubles.
extern "C"
void grid_read_config(const char* filename,
                      double*     out,
                      int Nc, int Nt, int Nx, int Ny, int Nz)
{
    // Grid dimension ordering: x,y,z,t = 0,1,2,3
    std::vector<int> latt = {Nx, Ny, Nz, Nt};
    GridCartesian* grid = SpaceTimeGrid::makeFourDimGrid(
        latt,
        GridDefaultSimd(Nd, vComplexD::Nsimd()),
        GridDefaultMpi());

    LatticeGaugeFieldD U(grid);
    FieldMetaData header;
    NerscIO::readConfiguration(U, header, filename);   // swap for IldgIO etc. if needed

    for (int mu = 0; mu < 4; mu++) {
        auto Umu = PeekIndex<LorentzIndex>(U, mu);
        for (int r = 0; r < Nc; r++)
        for (int c = 0; c < Nc; c++) {
            auto Umuij = PeekIndex<ColourIndex>(Umu, r, c);
            for (int t = 0; t < Nt; t++)
            for (int x = 0; x < Nx; x++)
            for (int y = 0; y < Ny; y++)
            for (int z = 0; z < Nz; z++) {
                Coordinate site({x, y, z, t});
                auto Umuij_site = peekSite(Umuij, site);

                double* p = out
                    + (((t*Nx + x)*Ny + y)*Nz + z) * 4 * 2*Nc*Nc
                    + mu * 2*Nc*Nc
                    + (r*Nc + c) * 2;
                auto val = TensorRemove(Umuij_site);
                p[0] = val.real();
                p[1] = val.imag();
            }
        }
    }

    delete grid;
}
