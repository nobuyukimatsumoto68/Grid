#include <Grid/Grid.h>
using namespace Grid;

extern "C"
void grid_init(int *argc, char ***argv)
{
    Grid_init(argc, argv);
}

extern "C"
void grid_finalize()
{
    Grid_finalize();
}

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
