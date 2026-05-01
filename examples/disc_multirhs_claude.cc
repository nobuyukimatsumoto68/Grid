// #include "reweight.h"
#include <filesystem>

#include <Grid/Grid.h>

using namespace std;
using namespace Grid;


inline bool is_exist (const std::string& name) {
return ( access( name.c_str(), F_OK ) != -1 );
}

// void PointSource(Coordinate &coor,LatticePropagator &source)
// {
//   source=Zero();
//   SpinColourMatrix kronecker; kronecker=1.0;
//   pokeSite(kronecker, source, coor);
// }

void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
                             GridBase *rbgrid, const int tslice, const int eo)
{
  GridBase *grid = source.Grid();
  RealD nrm = 1.0/sqrt(2.0);

  LatticeInteger t(grid);
  LatticeCoordinate(t, Tdir);

  LatticeComplex zz(grid);    zz    = Zero();
  LatticeComplex xi(grid);
  LatticeComplex xi_rb(rbgrid);
  LatticeComplex xi_eo(grid);

  source = Zero();
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      bernoulli(RNG, xi);
      xi = (2.0*xi - Complex(1.0,1.0))*nrm;
      xi = where(t==Integer(tslice), xi, zz);

      xi_eo = Zero();
      pickCheckerboard(eo, xi_rb, xi);
      setCheckerboard(xi_eo, xi_rb);

      // place xi_eo on the diagonal (s,s;c,c) of the source propagator
      auto spin_block = peekSpin(source, s, s);
      pokeColour(spin_block, xi_eo, c, c);
      pokeSpin(source, spin_block, s, s);
    }
  }
}


// void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
//                              const int tslice, const int eo, const int color, const int spin)
// {
//   assert( 0<=color && color<Nc );
//   assert( 0<=spin && spin<Nd );

//   GridBase *grid = source.Grid();
//   // LatticeComplex noise(grid);
//   // LatticeComplex noise_half(grid);
//   LatticePropagator noise(grid);
//   // LatticePropagator noise_half(grid);
//   bernoulli(RNG, noise); // 0,1 50:50 in cplx
//   RealD nrm = 1.0/sqrt(2.0);
//   noise = ( 2.0*noise - Complex(1.0,1.0) )*nrm;

//   LatticeInteger t(grid);
//   LatticeCoordinate(t,Tdir);
//   // LatticeComplex zz(grid); zz=Zero();
//   LatticePropagator zz(grid); zz=Zero();
//   noise = where(t==Integer(tslice), noise, zz);
//   pickCheckerboard( eo, source, noise );

//   // source = source*noise;
// }



template<class Action>
void Solve(Action &D,LatticePropagator &source,LatticePropagator &propagator)
{
  GridBase *UGrid = D.GaugeGrid();
  GridBase *FGrid = D.FermionGrid();

  LatticeFermion src4  (UGrid);
  LatticeFermion src5  (FGrid);
  LatticeFermion result5(FGrid);
  LatticeFermion result4(UGrid);

  ConjugateGradient<LatticeFermion> CG(1.0e-8,100000);
  SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(CG);
  ZeroGuesser<LatticeFermion> ZG; // Could be a DeflatedGuesser if have eigenvectors
  for(int s=0;s<Nd;s++){
    for(int c=0;c<Nc;c++){
      PropToFerm<Action>(src4,source,s,c);

      D.ImportPhysicalFermionSource(src4,src5);

      result5=Zero();
      schur(D,src5,result5,ZG);
      std::cout<<GridLogMessage
               <<"spin "<<s<<" color "<<c
               <<" norm2(src5d) "   <<norm2(src5)
               <<" norm2(result5d) "<<norm2(result5)<<std::endl;

      D.ExportPhysicalFermionSolution(result5,result4);

      FermToProp<Action>(propagator,result4,s,c);
    }
  }
}

// void ChCondPtSrc(std::string file, LatticePropagator &q, Coordinate& coord)
// {
//   LatticeComplex meson_CF( q.Grid() );

//   // Gamma G5(Gamma::Algebra::Gamma5);
//   meson_CF = trace(q);
//   TComplex ChCond;
//   peekSite(ChCond, meson_CF, coord);

//   Complex res = TensorRemove(ChCond); // Yes this is ugly, not figured a work around
//   std::cout << res << std::endl;

//   std::cout << "chcond, " << file << ", " << ChCond << std::endl;
//   {
//     XmlWriter WR(file);
//     write(WR, "MesonFile", res);
//   }
// }

// Complex ChCondStochSrc(LatticePropagator &psi, LatticePropagator &eta)
// {
//   LatticeComplex meson_CF( eta.Grid() );
//   meson_CF = trace(psi*adj(eta));
//   auto ChCond = sum( meson_CF );

//   LatticeComplex identity_CF( eta.Grid() );
//   identity_CF = trace(adj(eta)*eta);
//   auto norm = sum( identity_CF );

//   auto res = ChCond()()/norm()();
//   // std::cout << "chcond, " << file << ", " << res << std::endl;
//   // {
//   //   XmlWriter WR(file);
//   //   write(WR, "MesonFile", res);
//   // }
//   return res;
// }

void TraceField(LatticeComplex& meson_CF,
                const Gamma::Algebra& gam,
                LatticePropagator &psi, LatticePropagator &eta)
{
  // LatticeComplex meson_CF( eta.Grid() );
  meson_CF = trace(Gamma(gam)*psi*adj(eta));
  // auto ChCond = sum( meson_CF );

  // LatticeComplex identity_CF( eta.Grid() );
  // identity_CF = trace(adj(eta)*eta);
  // auto norm = sum( identity_CF );

  // auto res = ChCond()()/norm()();
  // std::cout << "chcond, " << file << ", " << res << std::endl;
  // {
  //   XmlWriter WR(file);
  //   write(WR, "MesonFile", res);
  // }
  // return res;
}


int main (int argc, char ** argv)
{
  const int Ls=16;
  Grid_init(&argc,&argv);

  // Double precision grids
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                   GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  // auto GridPtr   = TheHMC.Resources.GetCartesian();
  // auto GridRBPtr = TheHMC.Resources.GetRBCartesian();


  // std::vector<int> seeds4({1,2,3,4});
  // GridParallelRNG  RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);

  // -------------------------------------
  // for reading data
  // const int conf_min=atoi(argv[1]);
  // const std::string base_dir(argv[2]); // directory of lattice config
  // // -> string path
  // const std::string basedir(argv[3]); // not used
  // const std::string basedir2(argv[4]); // directory for output .bin

  // temporarily
  const double mass = 0.4; // (argv[5]);
  const double beta = 11.08;
  // const int interval=atoi(argv[6]);
  // -> 10, 20
  // const int nbeta=atoi(argv[7]);
  // const int runtype=atoi(argv[8]);
  const int Nt=8;

  // std::vector<std::string> betas;
  // {
  //   for(int i=9; i<9+nbeta; i++) {
  //     std::string str(argv[i]);
  //     betas.push_back(str);
  //   }
  // }

  std::cout << "beta = " << beta << std::endl;
  WilsonGaugeActionR Waction(beta);

  // int conf_max=conf_min;
  std::string dir = "/mnt/baracuda_14/grid_claude/16c";
  std::string obsdir = "/mnt/baracuda_14/grid_claude/16c_obs";
  std::filesystem::create_directories(obsdir);

  RealD M5=1.5;
  RealD b=1.5;// Scale factor b+c=2, b-c=1
  RealD c=0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  FermionAction::ImplParams Params(boundary);

  LatticeGaugeField Umu(UGrid);
  GridParallelRNG  RNG4(UGrid);

  int conf_min, conf_max, interval;
  {
    std::vector<int> confs;
    const std::string prefix = "ckpoint_lat.";
    for(const auto& entry : std::filesystem::directory_iterator(dir)){
      const std::string fname = entry.path().filename().string();
      if(fname.rfind(prefix, 0) == 0)
        confs.push_back(std::stoi(fname.substr(prefix.size())));
    }
    assert(!confs.empty());
    std::sort(confs.begin(), confs.end());
    conf_min = confs.front();
    interval = (confs.size() >= 2) ? confs[1] - confs[0] : 1;
    conf_max = confs.back() + interval;
  }
  std::cout << "conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;


  for(int conf=conf_min; conf<conf_max; conf+=interval){
    {
      const std::string path = dir+"/ckpoint_lat."+std::to_string(conf);
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, path);
      RNG4.SeedUniqueString(path);
    }
    FermionAction FermAct(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c, Params);


    LatticePropagator res(UGrid);
    res = Zero(); // = source;

    // noisy estimator with Z4
    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        // for(int color=0; color<Nc; color++){
        //   for(int spin=0; spin<Nd; spin++){
        LatticePropagator source(UGrid);

        // StochasticDilutedSource(RNG4, source, t, eo, color, spin);
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);
        // std::cout << "# @ " << t << " " << eo << std::endl;
        // std::cout << "# @@ source:" << std::endl
        //           << source << std::endl;

        LatticePropagator StochProp(UGrid);
        Solve(FermAct, source, StochProp);
        // std::cout << "# @@ prop: " << std::endl
        //           << StochProp << std::endl;
        // Complex chcond = ChCondStochSrc( StochProp, stochastic_source );
        LatticeComplex Trace_CF( UGrid );
        //         Gamma::Algebra Gammas[nchannel][2] = {
        //   {Gamma::Algebra::Gamma5      ,Gamma::Algebra::Gamma5},
        //   {Gamma::Algebra::GammaTGamma5,Gamma::Algebra::GammaTGamma5},
        //   {Gamma::Algebra::GammaTGamma5,Gamma::Algebra::Gamma5},
        //   {Gamma::Algebra::Gamma5      ,Gamma::Algebra::GammaTGamma5}
        // };
        Gamma::Algebra gam = Gamma::Algebra::Identity;

        // for(int tp=0; tp<Nt; tp++)
        //   sink = sink + Cshift(source, Tdir, tp);

        // std::cout << "debug. pt1" << std::endl;
        // GridBase *SliceGrid = makeSubSliceGrid(UGrid, Tdir);
        // std::cout << "debug. pt2" << std::endl;
        // // LatticePropagator source_t(SliceGrid);
        // LatticePropagator source_t(SliceGrid);
        // std::cout << "debug. pt3" << std::endl;
        // ExtractSlice(source_t, source, t, Tdir);
        // std::cout << "debug. pt4" << std::endl;
        // for(int tp=0; tp<Nt; tp++)
        //   InsertSlice(source_t, source, tp, Tdir);

        // std::cout << "# @@ sink:" << std::endl
        //           << sink << std::endl;

        TraceField(Trace_CF, gam, StochProp, source);
        // Trace_CF = trace( Gamma(gam)*psi*adj(eta));
        // std::cout << "# @@ trace: " << std::endl
        //           << Trace_CF << std::endl;

        res = res + Trace_CF;
        // break;
      }
      // if(t==10) break;
      //   }
      //   break;
      // }
      // break;
    } // for dilute
    std::cout << "# @@ res = " << std::endl
              << res << std::endl;
    break;
  } // conf

  Grid_finalize();
}




// {
//   SpinColourMatrix tmp;
//   peekSite(tmp, point_source, Origin);
//   std::cout << tmp << std::endl;
// }

// TComplex tmp;
// Coordinate Origin({0,0,0,0});
// peekSite(tmp, check_CF, Origin);
// std::cout << tmp << std::endl;

