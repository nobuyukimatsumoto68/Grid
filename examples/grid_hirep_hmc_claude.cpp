#include <Grid/Grid.h>
#include <memory>
#include <sstream>
using namespace Grid;

typedef PeriodicGimplR Gimpl;
using GaugeField    = typename Gimpl::GaugeField;
using Rep           = Representations<FundamentalRepresentation>;
using Integrator_t  = MinimumNorm2<Gimpl, NoSmearing<Gimpl>>;
using HMC_t         = HybridMonteCarlo<Integrator_t>;

// Mixed fundamental+adjoint pure-gauge action (SU(Nc))
class WilsonFundAdjointAction : public Action<typename Gimpl::GaugeField> {
public:
    INHERIT_GIMPL_TYPES(Gimpl);

    using Action<GaugeField>::S;
    using Action<GaugeField>::Sinitial;
    using Action<GaugeField>::deriv;
    using Action<GaugeField>::refresh;

    explicit WilsonFundAdjointAction(RealD betaF_, RealD betaA_)
        : betaF(betaF_), betaA(betaA_) {}

    virtual std::string action_name() override { return "WilsonFundAdjointAction"; }

    virtual std::string LogParameters() override {
        std::stringstream ss;
        ss << GridLogMessage << "[WilsonFundAdjointAction] betaF=" << betaF
           << " betaA=" << betaA << std::endl;
        return ss.str();
    }

    void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {}

    virtual RealD S(const GaugeField &U) override {
        GridBase *grid = U.Grid();
        std::vector<GaugeLinkField> Umu(Nd, grid);
        for (int mu = 0; mu < Nd; mu++)
            Umu[mu] = PeekIndex<LorentzIndex>(U, mu);

        ComplexField trP(grid);
        RealD sumF = 0.0, sumA = 0.0;
        for (int mu = 1; mu < Nd; mu++)
        for (int nu = 0; nu < mu;  nu++) {
            WilsonLoops<Gimpl>::traceDirPlaquette(trP, Umu, mu, nu);
            sumF += TensorRemove(sum(trP)).real();
            sumA += TensorRemove(sum(trP * conjugate(trP) - RealD(1.0))).real();
        }
        RealD vol   = U.Grid()->gSites();
        RealD faces = RealD(Nd * (Nd - 1)) / RealD(2.0);
        RealD Nplaq = vol * faces;
        return betaF * (Nplaq - RealD(1.0)/RealD(Nc) * sumF)
             + betaA * (Nplaq - RealD(1.0)/RealD(Nc*Nc - 1) * sumA);
    }

    virtual void deriv(const GaugeField &U, GaugeField &dSdU) override {
        GridBase *grid = U.Grid();
        std::vector<GaugeLinkField> Umu(Nd, grid);
        for (int mu = 0; mu < Nd; mu++)
            Umu[mu] = PeekIndex<LorentzIndex>(U, mu);

        ComplexField  wF(grid), wB(grid);
        GaugeLinkField stapF(grid), stapB(grid), stapSum(grid), dSdU_mu(grid);
        RealD cF = betaF / RealD(Nc);
        RealD cA = RealD(2.0) * betaA / RealD(Nc*Nc - 1);

        for (int mu = 0; mu < Nd; mu++) {
            stapSum = Zero();
            for (int nu = 0; nu < Nd; nu++) {
                if (nu == mu) continue;
                stapF = Cshift(Umu[nu], mu, +1)
                      * adj(Cshift(Umu[mu], nu, +1))
                      * adj(Umu[nu]);
                stapB = adj(Cshift(Cshift(Umu[nu], mu, +1), nu, -1))
                      * adj(Cshift(Umu[mu], nu, -1))
                      * Cshift(Umu[nu], nu, -1);
                ComplexField trF(grid), trB(grid);
                trF = trace(Umu[mu] * stapF);
                trB = trace(Umu[mu] * stapB);
                wF  = ComplexD(cF, 0.0) + ComplexD(cA, 0.0) * conjugate(trF);
                wB  = ComplexD(cF, 0.0) + ComplexD(cA, 0.0) * conjugate(trB);
                stapSum += wF * stapF + wB * stapB;
            }
            dSdU_mu = Ta(Umu[mu] * stapSum) * RealD(0.5);
            PokeIndex<LorentzIndex>(dSdU, dSdU_mu, mu);
        }
    }

private:
    RealD betaF, betaA;
};

// Members destroyed in reverse declaration order: HMC before integrator/RNGs/U.
// TheAction must outlive integrator: ActionLevel::actions is a reference into
// actions_hirep; the integrator's internal copy of ActionSet keeps that
// reference pointing into TheAction, so TheAction must not be destroyed first.
struct HmcState {
    std::unique_ptr<GridCartesian>              UGrid;
    std::unique_ptr<GridSerialRNG>              sRNG;
    std::unique_ptr<GridParallelRNG>            pRNG;
    std::unique_ptr<GaugeField>                 U;
    std::unique_ptr<WilsonFundAdjointAction>    action;
    std::unique_ptr<NoSmearing<Gimpl>>          Sm;
    ActionSet<GaugeField, Rep>                  TheAction;
    std::unique_ptr<Integrator_t>               integrator;
    std::vector<HmcObservable<GaugeField>*>     none;
    std::unique_ptr<HMC_t>                      HMC;
};



extern "C"
HmcState* grid_hmc_init(int NP_T, int NP_X, int NP_Y, int NP_Z,
                        int Nt, int Nx, int Ny, int Nz,
                        double betaF, double betaA,
                        int n_mdSteps, double trajL)
{
    HmcState* S = new HmcState();
    // Build fake argv: --grid Nx.Ny.Nz.Nt  --mpi NP_X.NP_Y.NP_Z.NP_T
    // Grid dimension order is x,y,z,t = 0,1,2,3
    std::string prog      = "hirep_hmc";
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

    S->UGrid.reset(SpaceTimeGrid::makeFourDimGrid(
        GridDefaultLatt(),
        GridDefaultSimd(Nd, vComplex::Nsimd()),
        GridDefaultMpi()));

    S->sRNG = std::make_unique<GridSerialRNG>();
    S->sRNG->SeedFixedIntegers({45, 12, 81, 9});
    S->pRNG = std::make_unique<GridParallelRNG>(S->UGrid.get());
    S->pRNG->SeedFixedIntegers({1, 2, 3, 4});

    S->U = std::make_unique<GaugeField>(S->UGrid.get());
    Gimpl::ColdConfiguration(*(S->pRNG), *(S->U));

    S->action = std::make_unique<WilsonFundAdjointAction>(betaF, betaA);

    S->TheAction.emplace_back(1);               // construct in-place: actions ref is self-contained
    S->TheAction[0].push_back(S->action.get());

    IntegratorParameters MD;
    MD.MDsteps = n_mdSteps;
    MD.trajL   = RealD(trajL);

    S->Sm         = std::make_unique<NoSmearing<Gimpl>>();
    S->integrator = std::make_unique<Integrator_t>(S->UGrid.get(), MD, S->TheAction, *(S->Sm));

    HMCparameters Params;
    Params.NoMetropolisUntil = 0;
    Params.Trajectories      = 1;

    S->HMC = std::make_unique<HMC_t>(Params, *(S->integrator),
                                     *(S->sRNG), *(S->pRNG),
                                     S->none, *(S->U));
    return S;
}

// Runs one HMC trajectory and fills out[] with U in HiRep in-memory layout:
//   T->X->Y->Z site order, 4 directions, Nc x Nc complex row-major.
//   out must be pre-allocated to Nt*Nx*Ny*Nz * 4 * 2*Nc*Nc doubles.
extern "C"
void grid_hmc_step(HmcState* S, double *out)
{
    S->HMC->evolve();

    auto dims   = S->UGrid->GlobalDimensions(); // {Nx, Ny, Nz, Nt}
    const int Nx = dims[0], Ny = dims[1], Nz = dims[2], Nt = dims[3];

    for (int mu = 0; mu < 4; mu++) {
        auto Umu = PeekIndex<LorentzIndex>(*(S->U), mu);
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
}

extern "C"
void grid_hmc_finalize(HmcState* S)
{
    delete S;
    // Grid_finalize() is intentionally omitted: it calls MPI_Finalize(), but
    // HiRep's finalize_process() owns MPI teardown. All Grid heap objects are
    // cleaned up by the HmcState unique_ptr destructors above.
}

