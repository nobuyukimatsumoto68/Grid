#include "reweight.h"

#include <filesystem>

#include <Grid/Grid.h>

using namespace std;
using namespace Grid;


inline bool is_exist (const std::string& name) {
return ( access( name.c_str(), F_OK ) != -1 );
}

void PointSource(Coordinate &coor,LatticePropagator &source)
{
  source=Zero();
  SpinColourMatrix kronecker; kronecker=1.0;
  pokeSite(kronecker, source, coor);
}

void StochasticSource(GridParallelRNG &RNG, LatticePropagator &source)
{
  GridBase *grid = source.Grid();
  LatticeComplex noise(grid);
  bernoulli(RNG, noise); // 0,1 50:50 in cplx

  RealD nrm = 1.0/sqrt(2.0);
  noise = ( 2.0*noise - Complex(1.0,1.0) )*nrm;

  source = 1.0;
  source = source*noise;
}

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

void ChCondPtSrc(std::string file, LatticePropagator &q, Coordinate& coord)
{
  LatticeComplex meson_CF( q.Grid() );

  // Gamma G5(Gamma::Algebra::Gamma5);
  meson_CF = trace(q);
  TComplex ChCond;
  peekSite(ChCond, meson_CF, coord);

  Complex res = TensorRemove(ChCond); // Yes this is ugly, not figured a work around
  std::cout << res << std::endl;

  std::cout << "chcond, " << file << ", " << ChCond << std::endl;
  {
    XmlWriter WR(file);
    write(WR, "MesonFile", res);
  }
}

Complex ChCondStochSrc(LatticePropagator &psi, LatticePropagator &eta)
{
  LatticeComplex meson_CF( eta.Grid() );
  meson_CF = trace(psi*adj(eta));
  auto ChCond = sum( meson_CF );

  LatticeComplex identity_CF( eta.Grid() );
  identity_CF = trace(adj(eta)*eta);
  auto norm = sum( identity_CF );

  auto res = ChCond()()/norm()();
  // std::cout << "chcond, " << file << ", " << res << std::endl;
  // {
  //   XmlWriter WR(file);
  //   write(WR, "MesonFile", res);
  // }
  return res;
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
  const int conf_min=atoi(argv[1]);

  const std::string base_dir(argv[2]); // directory of lattice config
  // -> string path

  const std::string basedir(argv[3]); // not used

  const std::string basedir2(argv[4]); // directory for output .bin

  const std::string mass(argv[5]);
  const int interval=atoi(argv[6]);
  // -> 10, 20

  const int nbeta=atoi(argv[7]);
  const int runtype=atoi(argv[8]);
  const int Nt=8;


  std::vector<std::string> betas;
  {
    for(int i=9; i<9+nbeta; i++) {
      std::string str(argv[i]);
      betas.push_back(str);
    }
  }

  for(auto elem : betas) std::cout << elem << std::endl;

  for(std::string beta : betas){
    std::cout << "beta = " << beta << std::endl;
    WilsonGaugeActionR Waction(std::stod(beta));

    int conf_max=conf_min;
    for(int conf=conf_min; conf<conf_min+10000000*interval; conf+=interval){
      char f[200];
      std::string prefix = get_prefix(beta, mass);
      std::sprintf(f,
                   "%s.%d",
                   prefix.data(), conf);

      const std::string f_str = f;
      const std::string path = base_dir+get_configname(beta, mass)+"/"+f_str;
      // std::cout << "path = " << path << std::endl;

      conf_max = conf-interval;
      if( !is_exist( path ) ) break;
    }

    for(int conf=conf_min; conf<conf_max; conf+=interval){
      std::string obs_id=get_configname(beta, mass);

      std::string pathO = basedir2+obs_id+"chcond"+std::to_string(conf)+".xml";
      // if( std::filesystem::exists( pathO ) && (runtype==1) ) continue;
      if( std::ifstream(pathO.c_str()).good() && (runtype==1) ) continue;


      // std::cout << "prefix" << std::endl;
      char f[200];
      std::string prefix = get_prefix(beta, mass);
      std::sprintf(f,
                   "%s.%d",
                   prefix.data(), conf);

      const std::string f_str = f;
      const std::string path = base_dir+obs_id+"/"+f_str;
      FieldMetaData header;
      LatticeGaugeField Umu(UGrid);
      NerscIO::readConfiguration(Umu, header, path);

      RealD M5=1.5;
      RealD b=1.5;// Scale factor b+c=2, b-c=1
      RealD c=0.5;
      std::vector<Complex> boundary = {1,1,1,-1};
      typedef MobiusFermionD FermionAction;
      FermionAction::ImplParams Params(boundary);
      RealD mass_double = stod(mass);
      FermionAction FermAct(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass_double, M5, b, c, Params);

      // noisy estimator with Z4
      LatticePropagator stochastic_source(UGrid);
      GridParallelRNG  RNG4(UGrid);  RNG4.SeedUniqueString(f_str);
      StochasticSource( RNG4, stochastic_source );

      LatticePropagator StochProp(UGrid);
      Solve(FermAct, stochastic_source, StochProp);

      Complex chcond = ChCondStochSrc( StochProp, stochastic_source );

      {
        // std::unique_ptr<XmlWrite> WR;
        // if (UGrid->IsBoss()){
        XmlWriter WR( pathO );
        WR.scientificFormat( true ); // @@@ double check
        WR.setPrecision( std::numeric_limits<Real>::digits10 + 1 );
        // real part only
        write(WR, "chcond", chcond.real() );
      }
    } // conf
  }

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

