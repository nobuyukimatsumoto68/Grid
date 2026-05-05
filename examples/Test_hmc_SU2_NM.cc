/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_hmc_WilsonFermionGauge.cc

Copyright (C) 2015

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>
Author: Guido Cossu <guido.cossu@ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */

#include <Grid/Grid.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <streambuf>

using namespace Grid;

//typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
// using GimplHMC = PeriodicGaugeImpl<GaugeImplTypes<vComplexD, 2>>;
// using RepsHMC  = Representations<FundamentalRep<2, GroupName::SU>>;
// using HMCWrapper = HMCWrapperTemplate<GimplHMC, MinimumNorm2, RepsHMC, XmlReader>;

typedef PeriodicGimplR Gimpl;


// template <class Gimpl>
class WilsonFundAdjointAction : public Action<typename Gimpl::GaugeField> {
public:  
    INHERIT_GIMPL_TYPES(Gimpl);
  
    using Action<GaugeField>::S;
    using Action<GaugeField>::Sinitial;
    using Action<GaugeField>::deriv;
    using Action<GaugeField>::refresh;
    
    /////////////////////////// constructors
    explicit WilsonFundAdjointAction(RealD betaF_, RealD betaA_): betaF(betaF_), betaA(betaA_){};

    virtual std::string action_name() override {return "WilsonFundAdjointAction";}
  
    virtual std::string LogParameters() override {
        std::stringstream sstream;
        sstream << GridLogMessage << "[WilsonFundAdjointAction] betaF= " << betaF << ", betaA= " << betaA << std::endl;
        return sstream.str();
    }

    void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG) override {}

    virtual RealD S(const GaugeField &U) override {
    
        //Loading in the field
        GridBase *grid = U.Grid();
    
        //Making a field object with Lorentz indices
        std::vector<GaugeLinkField> Umu(Nd, grid);
        for (int mu = 0; mu < Nd; mu++)
        {
            Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
        }
    
        //Declaring a trace field as well as accumulators for plaquettes
        ComplexField trP(grid);
        RealD sumF = 0.0;
        RealD sumA = 0.0;
    
        //Filling the trace and accumulators
        for (int mu = 1; mu < Nd; mu++){
            for (int nu = 0; nu < mu; nu++){
                WilsonLoops<Gimpl>::traceDirPlaquette(trP, Umu, mu, nu);
                
                sumF += TensorRemove(sum(trP)).real();
                sumA += TensorRemove(sum(trP * conjugate(trP) - RealD(1.0))).real();
            }
        }
        RealD vol = U.Grid()->gSites();
        RealD faces = RealD(Nd * (Nd - 1)) / RealD(2.0);
        RealD Nplaq = vol * faces;
        RealD SF = betaF * (Nplaq - RealD(1.0)/RealD(Nc) * sumF);
        RealD SA = betaA * (Nplaq - RealD(1.0)/RealD(Nc*Nc - 1) * sumA);
        return SF + SA;
    }
  
  
    virtual void deriv(const GaugeField &U, GaugeField &dSdU) override {
    
        //Setting up some of the same things as in S
        GridBase *grid = U.Grid();
        
        std::vector<GaugeLinkField> Umu(Nd, grid);
        for (int mu = 0; mu < Nd; mu++){
            Umu[mu] = PeekIndex<LorentzIndex>(U, mu);
        }
        
        //weights
        ComplexField wF(grid);  
        ComplexField wB(grid);
        
        //Declaring upper and lower staples
        GaugeLinkField stapF(grid);
        GaugeLinkField stapB(grid);
        GaugeLinkField stapSum(grid);
        
        //Declaring Lorentz indices for dSdU
        GaugeLinkField dSdU_mu(grid);
        
        RealD cF = betaF / RealD(Nc);
        RealD cA = RealD(2.0) * betaA / RealD(Nc*Nc - 1);
        
        //Entering the calculation
        for (int mu = 0; mu < Nd; mu++){
            
            stapSum = Zero();
            
            for (int nu = 0; nu < Nd; nu++){
                
                if(nu==mu){
                    continue;
                }
                // computing staples
                stapF = Cshift(Umu[nu], mu, +1)
                          * adj( Cshift(Umu[mu], nu, +1) )
                          * adj( Umu[nu] );
    
                stapB = adj( Cshift( Cshift(Umu[nu], mu, +1), nu, -1 ) )
                          * adj( Cshift(Umu[mu], nu, -1) )
                          * Cshift( Umu[nu], nu, -1 );
    
                // compute the plaquette traces directly from staples
                ComplexField trF(grid), trB(grid);
                trF = trace( Umu[mu] * stapF );     // forward plaquette trace at x
                trB = trace( Umu[mu] * stapB );     // backward plaquette trace at x
    
                // weights
                wF = ComplexD(cF, 0.0) + ComplexD(cA, 0.0) * conjugate(trF);
                wB = ComplexD(cF, 0.0) + ComplexD(cA, 0.0) * conjugate(trB);
    
                // sum
                stapSum += wF * stapF + wB * stapB;
                
            }
            dSdU_mu = Ta(Umu[mu] * stapSum) * RealD(0.5);
            PokeIndex<LorentzIndex>(dSdU, dSdU_mu, mu);
        }
    }

private:
    RealD betaF;
    RealD betaA;
};

// template<class Gimpl>
// void SimpleSU2_ForceFD_Cayley(Grid::RealD betaF, Grid::RealD betaA)
// {
//     using namespace Grid;
//     INHERIT_GIMPL_TYPES(Gimpl);

//     // 4^4 lattice
//     std::vector<int> L = {4,4,4,4};
//     Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
//     Coordinate mpi  = GridDefaultMpi();
//     GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(L, simd, mpi);

//     GridParallelRNG pRNG(grid);
//     pRNG.SeedFixedIntegers({1,2,3,4});

//     // Random SU(2) configuration
//     GaugeField U(grid);
//     SU<2>::HotConfiguration(pRNG, U);

//     // Random su(2) direction H
//     GaugeField H(grid);
//     gaussian(pRNG, H);
//     for(int mu=0; mu<Nd; mu++){
//        GaugeLinkField Hmu = PeekIndex<LorentzIndex>(H, mu);
//        Hmu = Ta(Hmu);
//        PokeIndex<LorentzIndex>(H, Hmu, mu);
//     }

//     // Action and force at U
//     WilsonFundAdjointAction<Gimpl> action(betaF, betaA);
//     GaugeField F(grid);
//     action.deriv(U, F);

//     // Inner product: <H,F> = sum over links ReTr(H*F)
//     auto ReTrInner = [&](const GaugeField &A, const GaugeField &B)->RealD {
//         RealD acc = 0.0;
//         for(int mu=0; mu<Nd; mu++){
//             GaugeLinkField Amu = PeekIndex<LorentzIndex>(A, mu);
//             GaugeLinkField Bmu = PeekIndex<LorentzIndex>(B, mu);
//             ComplexField tr(grid);
//             tr = trace(Amu * Bmu);
//             acc += TensorRemove(sum(tr)).real();
//         }
//         return acc;
//     };

//     // SU(2)-valued left update via Cayley transform
//     // U -> (I + (eps/2) H) (I - (eps/2) H)^{-1} U
//     // For H in su(2): H^2 = -h2 * I, with h2 = -1/2 ReTr(H*H).
//     auto LeftUpdate_Cayley = [&](const GaugeField &Uin, const GaugeField &Hin, RealD eps)->GaugeField {
//         GaugeField Uout(grid);
    
//         RealD a = eps * RealD(0.5); // Cayley parameter
    
//         for(int mu=0; mu<Nd; mu++){
//             GaugeLinkField Umu = PeekIndex<LorentzIndex>(Uin, mu);
//             GaugeLinkField Hmu = PeekIndex<LorentzIndex>(Hin, mu);
    
//             GaugeLinkField I(grid); I = 1.0;
    
//             // trHH = Tr(HH) (ComplexField, but should be real for su(2))
//             ComplexField trHH(grid);
//             trHH = trace(Hmu * Hmu);
            
//             // trHH_re as ComplexField (purely real)
//             ComplexField trHH_re(grid);
//             trHH_re = ComplexD(1.0, 0.0) * real(trHH);
            
//             // h2 = -1/2 ReTr(HH) as ComplexField (purely real)
//             ComplexField h2(grid);
//             h2 = ComplexD(-0.5, 0.0) * trHH_re;
            
//             // denom = 1 + a^2 h2
//             ComplexField one(grid);
//             one = ComplexD(1.0, 0.0);
            
//             ComplexField denom(grid);
//             denom = one + ComplexD(a*a, 0.0) * h2;
            
//             // invdenom = 1/denom   (lattice/lattice)
//             ComplexField invdenom(grid);
//             invdenom = one / denom;
            
//             // c0, c1 as ComplexField (lattice/lattice only)
//             ComplexField c0(grid), c1(grid);
            
//             ComplexField twoa(grid);
//             twoa = ComplexD(2.0*a, 0.0);
            
//             c0 = (one - ComplexD(a*a, 0.0) * h2) * invdenom;
//             c1 = twoa * invdenom;
            
//             // X = c0 I + c1 H
//             GaugeLinkField X(grid);
//             X = c0 * I + c1 * Hmu;

    
//             GaugeLinkField Up(grid);
//             Up = X * Umu;
    
//             PokeIndex<LorentzIndex>(Uout, Up, mu);
//         }
//         return Uout;
//     };




//     // eps (of course, the test should fail at large eps)
//     RealD eps_min=1e-4, eps_max=1.0; int N=20;
//     std::vector<RealD> eps_list;
//     for(int i=0;i<=N;i++) eps_list.push_back(eps_min*std::pow(eps_max/eps_min, RealD(i)/N));


//     std::cout << "## SU2_FORCE_FD_Cayley betaF=" << betaF
//               << " betaA=" << betaA
//               << " Nc=" << Nc << " Nd=" << Nd << "\n";
//     std::cout << "## columns: eps deltaS_fd  minus_ReTr(HF)  ratio  relerr\n";

//     // Force-predicted directional variation at U (sign convention: minus)
//     RealD deltaS_force = -ReTrInner(H, F);

//     for(auto eps : eps_list){
//         GaugeField Uplus  = LeftUpdate_Cayley(U, H, +eps);
//         GaugeField Uminus = LeftUpdate_Cayley(U, H, -eps);


//         RealD Splus  = action.S(Uplus);
//         RealD Sminus = action.S(Uminus);

//         RealD deltaS_fd = (Splus - Sminus) / (RealD(2.0) * eps);

//         RealD ratio  = deltaS_force / deltaS_fd;
//         RealD relerr = fabs((deltaS_force - deltaS_fd) / deltaS_fd);

//         std::cout << std::setprecision(16)
//                   << eps << " "
//                   << deltaS_fd << " "
//                   << deltaS_force << " "
//                   << ratio << " "
//                   << relerr << "\n";
//     }

//     delete grid;
// }

// template<class Gimpl>
// void SimpleSU2_ForceFD_FirstOrder(Grid::RealD betaF, Grid::RealD betaA)
// {
//     using namespace Grid;
//     INHERIT_GIMPL_TYPES(Gimpl);

//     // 4^4 lattice
//     std::vector<int> L = {4,4,4,4};
//     Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
//     Coordinate mpi  = GridDefaultMpi();
//     GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(L, simd, mpi);

//     GridParallelRNG pRNG(grid);
//     pRNG.SeedFixedIntegers({1,2,3,4});

//     // Random SU(2) configuration
//     GaugeField U(grid);
//     SU<2>::HotConfiguration(pRNG, U);

//     // Random su(2) direction H
//     GaugeField H(grid);
//     gaussian(pRNG, H);
//     for(int mu=0; mu<Nd; mu++){
//        GaugeLinkField Hmu = PeekIndex<LorentzIndex>(H, mu);
//        Hmu = Ta(Hmu);
//        PokeIndex<LorentzIndex>(H, Hmu, mu);
//     }

//     // Action and force at U
//     WilsonFundAdjointAction<Gimpl> action(betaF, betaA);
//     GaugeField F(grid);
//     action.deriv(U, F);

//     // Inner product: <H,F> = sum over links ReTr(H*F)
//     auto ReTrInner = [&](const GaugeField &A, const GaugeField &B)->RealD {
//         RealD acc = 0.0;
//         for(int mu=0; mu<Nd; mu++){
//             GaugeLinkField Amu = PeekIndex<LorentzIndex>(A, mu);
//             GaugeLinkField Bmu = PeekIndex<LorentzIndex>(B, mu);
//             ComplexField tr(grid);
//             tr = trace(Amu * Bmu);
//             acc += TensorRemove(sum(tr)).real();
//         }
//         return acc;
//     };

//     // First-order "group" displacement: U -> (I + eps H) U
//     auto LeftUpdate_FirstOrder = [&](const GaugeField &Uin, const GaugeField &Hin, RealD eps)->GaugeField {
//         GaugeField Uout(grid);

//         for(int mu=0; mu<Nd; mu++){
//             GaugeLinkField Umu = PeekIndex<LorentzIndex>(Uin, mu);
//             GaugeLinkField Hmu = PeekIndex<LorentzIndex>(Hin, mu);

//             GaugeLinkField I(grid);
//             I = 1.0;

//             GaugeLinkField Up(grid);
//             Up = (I + eps * Hmu) * Umu;

//             PokeIndex<LorentzIndex>(Uout, Up, mu);
//         }
//         return Uout;
//     };

//     // eps (of course, the test should fail at large eps)
//     RealD eps_min=1e-4, eps_max=1.0; int N=20;
//     std::vector<RealD> eps_list;
//     for(int i=0;i<=N;i++) eps_list.push_back(eps_min*std::pow(eps_max/eps_min, RealD(i)/N));


//     std::cout << "## SU2_FORCE_FD_FIRST_ORDER betaF=" << betaF
//               << " betaA=" << betaA
//               << " Nc=" << Nc << " Nd=" << Nd << "\n";
//     std::cout << "## columns: eps deltaS_fd  minus_ReTr(HF)  ratio  relerr\n";

//     // Force-predicted directional variation at U (sign convention: minus)
//     RealD deltaS_force = -ReTrInner(H, F);

//     for(auto eps : eps_list){
//         GaugeField Uplus  = LeftUpdate_FirstOrder(U, H, +eps);
//         GaugeField Uminus = LeftUpdate_FirstOrder(U, H, -eps);

//         RealD Splus  = action.S(Uplus);
//         RealD Sminus = action.S(Uminus);

//         RealD deltaS_fd = (Splus - Sminus) / (RealD(2.0) * eps);

//         RealD ratio  = deltaS_force / deltaS_fd;
//         RealD relerr = fabs((deltaS_force - deltaS_fd) / deltaS_fd);

//         std::cout << std::setprecision(16)
//                   << eps << " "
//                   << deltaS_fd << " "
//                   << deltaS_force << " "
//                   << ratio << " "
//                   << relerr << "\n";
//     }

//     delete grid;
// }


// template<class Gimpl>
// void SimpleSU2_GaugeInvarianceCheck(Grid::RealD betaF, Grid::RealD betaA)
// {
//     using namespace Grid;
//     INHERIT_GIMPL_TYPES(Gimpl);
//     //static_assert(Nc == 2, "This test must be instantiated with an SU(2) Gimpl (Nc=2).");

//     std::cout << "Nc (in HMCWrapper) = " << Nc << std::endl;


//     // 4^4 lattice
//     std::vector<int> L = {4,4,4,4};
//     Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
//     Coordinate mpi  = GridDefaultMpi();
//     GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(L, simd, mpi);

//     GridParallelRNG pRNG(grid);
//     pRNG.SeedFixedIntegers({1,2,3,4});

//     // Random configuration U (use the ImplementationPolicy, not SU<...>)
//     GaugeField U(grid);
//     SU<2>::HotConfiguration(pRNG, U);

//     // Action
//     WilsonFundAdjointAction<Gimpl> action(betaF, betaA);

//     // Random local gauge transform G(x):
//     // build a temporary GaugeField so HotConfiguration can fill it,
//     // then steal one Lorentz component as a site field.
//     GaugeField Gtmp(grid);
//     SU<2>::HotConfiguration(pRNG, Gtmp);

//     GaugeLinkField G(grid);
//     G = PeekIndex<LorentzIndex>(Gtmp, 0);

//     // Gauge-transformed field Ug: U'_mu(x) = G(x) U_mu(x) G^\dagger(x+mu)
//     GaugeField Ug(grid);
//     for(int mu = 0; mu < Nd; mu++){
//         GaugeLinkField Umu = PeekIndex<LorentzIndex>(U,  mu);
//         GaugeLinkField Gf  = Cshift(G, mu, +1);
//         GaugeLinkField Up  = G * Umu * adj(Gf);
//         PokeIndex<LorentzIndex>(Ug, Up, mu);
//     }

//     RealD S0 = action.S(U);
//     RealD Sg = action.S(Ug);
//     RealD dS = Sg - S0;

//     std::cout << "## GAUGE_INVARIANCE_CHECK betaF=" << betaF
//               << " betaA=" << betaA
//               << " Nc=" << Nc << " Nd=" << Nd << "\n";
//     std::cout << std::setprecision(16)
//               << "S(U)   = " << S0 << "\n"
//               << "S(U^G) = " << Sg << "\n"
//               << "dS     = " << dS << "\n";

//     delete grid;
// }

// template <class Gimpl>
// void Test_FundOnly_Equivalence_GridWilsonVsFundAdj(Grid::RealD beta, int ncfg = 10)
// {
//   using namespace Grid;
//   INHERIT_GIMPL_TYPES(Gimpl);

//   // 4^4 lattice
//   std::vector<int> L = {4,4,4,4};
//   Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
//   Coordinate mpi  = GridDefaultMpi();
//   GridCartesian *grid = SpaceTimeGrid::makeFourDimGrid(L, simd, mpi);

//   GridParallelRNG pRNG(grid);
//   pRNG.SeedFixedIntegers({101,202,303,404});

//   // IMPORTANT: fully qualify to force Grid's class (avoid ambiguity)
//   Grid::WilsonGaugeAction<Gimpl>       A(beta);
//   WilsonFundAdjointAction<Gimpl>       B(beta, 0.0);

//   // ReTr inner product on link fields: sum_mu ReTr(X_mu * Y_mu)
//   auto ReTrInner = [&](const typename Gimpl::GaugeField &X,
//                        const typename Gimpl::GaugeField &Y)->RealD {
//     RealD acc = 0.0;
//     for(int mu=0; mu<Nd; mu++){
//       GaugeLinkField Xmu = PeekIndex<LorentzIndex>(X, mu);
//       GaugeLinkField Ymu = PeekIndex<LorentzIndex>(Y, mu);
//       ComplexField tr(grid);
//       tr = trace(Xmu * Ymu);
//       acc += TensorRemove(sum(tr)).real();
//     }
//     return acc;
//   };

//   std::cout << "## FUND_ONLY_EQUIV (Grid::WilsonGaugeAction vs WilsonFundAdjoint betaA=0)\n";
//   std::cout << "## beta=" << beta << " Nc=" << Nc << " Nd=" << Nd << " ncfg=" << ncfg << "\n";
//   std::cout << "## columns: cfg  SA  SB  dS  |dS|  dirA  dirB  ratio(dirA/dirB)  rel(|dirA-dirB|)\n";

//   for(int cfg=0; cfg<ncfg; cfg++){
//     typename Gimpl::GaugeField U(grid);
//     SU<Nc>::HotConfiguration(pRNG, U);

//     // 1) Action values
//     RealD SA = A.S(U);
//     RealD SB = B.S(U);
//     RealD dS = SA - SB;

//     // 2) Force comparison in a random direction H
//     typename Gimpl::GaugeField H(grid);
//     gaussian(pRNG, H);
//     for(int mu=0; mu<Nd; mu++){
//       GaugeLinkField Hmu = PeekIndex<LorentzIndex>(H, mu);
//       Hmu = Ta(Hmu);
//       PokeIndex<LorentzIndex>(H, Hmu, mu);
//     }

//     typename Gimpl::GaugeField FA(grid), FB(grid);
//     A.deriv(U, FA);
//     B.deriv(U, FB);

//     RealD dirA = -ReTrInner(H, FA);
//     RealD dirB = -ReTrInner(H, FB);

//     RealD ratio = dirA / dirB;

//     RealD scale = std::max(RealD(1.0), std::max(fabs(dirA), fabs(dirB)));
//     RealD rel   = fabs(dirA - dirB) / scale;

//     std::cout << std::setprecision(16)
//               << cfg << " "
//               << SA << " "
//               << SB << " "
//               << dS << " "
//               << fabs(dS) << " "
//               << dirA << " "
//               << dirB << " "
//               << ratio << " "
//               << rel << "\n";
//   }

//   delete grid;
// }

// bool getCmdOption(int argc, char** argv,
//                   const std::string& option,
//                   double& value)
// {
//     for(int i = 1; i < argc; ++i) {
//         std::string arg(argv[i]);
//         if(arg.find(option + "=") == 0) {
//             value = std::stod(arg.substr(option.size() + 1));
//             return true;
//         }
//     }
//     return false;
// }


int main(int argc, char **argv)
{
    
    Grid_init(&argc, &argv);
    GridLogLayout();

    // typedef PeriodicGaugeImpl<GaugeImplTypes<S, Dimension > > Gimpl;
    // typedef PeriodicGimplR Gimpl;
    // using Gimpl2 = PeriodicGaugeImpl<GaugeImplTypes<vComplexD, 2>>;
    // WilsonFundAdjoint(beta, 0) == WilsonGaugeAction(1.0) test
    // Test_FundOnly_Equivalence_GridWilsonVsFundAdj<Gimpl2>(1.0, 10);
    
    // // Gauge Invariance Test
    // SimpleSU2_GaugeInvarianceCheck<Gimpl2>(1.0, 0.0);
    // SimpleSU2_GaugeInvarianceCheck<Gimpl2>(0.0, 1.0);
    // SimpleSU2_GaugeInvarianceCheck<Gimpl2>(1.0, 1.0);

    // // Force vs Finite Difference test (First order)
    // SimpleSU2_ForceFD_FirstOrder<Gimpl2>(1.0, 0.0);
    // SimpleSU2_ForceFD_FirstOrder<Gimpl2>(0.0, 1.0);
    // SimpleSU2_ForceFD_FirstOrder<Gimpl2>(1.0, 1.0);

    // // Force vs Finite Difference test (Cayley)
    // SimpleSU2_ForceFD_Cayley<Gimpl2>(1.0, 0.0);
    // SimpleSU2_ForceFD_Cayley<Gimpl2>(0.0, 1.0);
    // SimpleSU2_ForceFD_Cayley<Gimpl2>(1.0, 1.0);

    std::cout << "Entering HMC\n";

    // HMCWrapper TheHMC;
    // TheHMC.Resources.AddFourDimGrid("gauge");

    // CheckpointerParameters CPparams;
    // CPparams.config_prefix = "ckpoint_lat";
    // CPparams.rng_prefix    = "ckpoint_rng";
    // CPparams.saveInterval  = 1;
    // CPparams.format        = "IEEE64BIG";
    // TheHMC.Resources.LoadNerscCheckpointer(CPparams);

    // RNGModuleParameters RNGpar;
    // RNGpar.serial_seeds   = "1 2 3 4 5";
    // RNGpar.parallel_seeds = "6 7 8 9 10";
    // TheHMC.Resources.SetRNGSeeds(RNGpar);

    // Observables
    // typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
    // TheHMC.Resources.AddObservable<PlaqObs>();

    // Action
    // using Gimpl = typename HMCWrapper::ImplPolicy;
    using GaugeField = typename Gimpl::GaugeField;

    // TheHMC.ReadCommandLine(argc, argv);
    // std::cout << GridLogMessage << "About to run HMC with parsed parameters:\n";
    //std::cout << TheHMC.LogParameters();
    //std::cout << TheHMC.Parameters();

    // run parameters
    int n_mdSteps = 20;
    RealD trajL = 1.0;
    // int n_therm = 50;
    // int n_measurements = 10;
    // int n_skip = 5;
    // int n_traj = 10;

    // TheHMC.Parameters.MD.MDsteps    = n_mdSteps;
    // TheHMC.Parameters.MD.trajL      = trajL;
    // TheHMC.Parameters.Trajectories  = n_traj;

    RealD epsilon = trajL / RealD(n_mdSteps);

    // Single (betaF, betaA) point — values passed via --betaF=X --betaA=Y
    // The parameter sweep is driven externally by run_example.sh which submits
    // one independent job per grid point.
    RealD betaF = 1.2;
    RealD betaA = 1.2;
    // getCmdOption(argc, argv, "--betaF", betaF);
    // getCmdOption(argc, argv, "--betaA", betaA);

    // TheHMC.TheAction.clear();
    // auto *action = new WilsonFundAdjointAction<Gimpl>(betaF, betaA);
    auto *action = new WilsonFundAdjointAction(betaF, betaA);

    // ActionLevel<HMCWrapper::Field> Level1(1);
    // Level1.push_back(static_cast<Action<GaugeField>*>(Waction));
    // TheHMC.TheAction.push_back(Level1);

    using Rep = Representations<FundamentalRepresentation>;
    ActionLevel<GaugeField, Rep> Level1(1);
    Level1.push_back(action);
    ActionSet<GaugeField, Rep> TheAction;
    TheAction.push_back(Level1);
    std::cout << "\n============================================================\n"
              << "  HMC run\n"
              << "    betaF       = " << betaF << "\n"
              << "    betaA       = " << betaA << "\n"
              << "    MDsteps     = " << n_mdSteps << "\n"
              << "    trajL       = " << trajL << "\n"
              << "    epsilon     = " << epsilon << "\n"
              // << "    trajectories= " << n_traj << "\n"
              << "============================================================\n";

    GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                     GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                     GridDefaultMpi());
    GridSerialRNG sRNG;            sRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));
    std::vector<int> seeds4({1,2,3,4});
    GridParallelRNG  RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);

    GaugeField U(UGrid);
    Gimpl::ColdConfiguration(RNG4, U);

    IntegratorParameters MD;
    MD.MDsteps = n_mdSteps;
    MD.trajL   = trajL;
    NoSmearing<Gimpl> Sm;
    MinimumNorm2<Gimpl, NoSmearing<Gimpl>> integrator(UGrid, MD, TheAction, Sm);

    std::vector<HmcObservable<GaugeField>*> none;

    HMCparameters Params;
    Params.NoMetropolisUntil = 0;
    Params.Trajectories = 1;

    //template <class IntegratorType>
    HybridMonteCarlo HMC(Params, integrator,
                           sRNG, RNG4,
                           none, U);

    HMC.evolve();

    Grid_finalize();
    return 0;
}
