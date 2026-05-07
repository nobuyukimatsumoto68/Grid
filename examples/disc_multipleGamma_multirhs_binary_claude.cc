#include <filesystem>
#include <getopt.h>

#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

void ParseArgs(int argc, char** argv,
               double& mass,
               double& beta,
               std::string& dir,
               std::string& obsdir)
{
  const char* const short_opts = ":m:b:d:o:";
  const option long_opts[] = {
    {"mass",   required_argument, nullptr, 'm'},
    {"beta",   required_argument, nullptr, 'b'},
    {"dir",    required_argument, nullptr, 'd'},
    {"obsdir", required_argument, nullptr, 'o'},
    {nullptr,  no_argument,       nullptr,  0}
  };
  opterr = 0;
  int option_index, opt;
  while ((opt = getopt_long(argc, argv, short_opts, long_opts, &option_index)) != -1){
    switch (opt) {
    case 'm': mass   = std::stod(optarg); break;
    case 'b': beta   = std::stod(optarg); break;
    case 'd': dir    = optarg;            break;
    case 'o': obsdir = optarg;            break;
    default: break;
    }
  }
}


inline bool is_exist (const std::string& name) {
return ( access( name.c_str(), F_OK ) != -1 );
}

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

void TraceField(LatticeComplex& meson_CF,
                const Gamma::Algebra& gam,
                LatticePropagator &psi, LatticePropagator &eta)
{
  meson_CF = trace(Gamma(gam)*psi*adj(eta));
}


int main (int argc, char ** argv)
{
  const int Ls=16;
  Grid_init(&argc,&argv);

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                   GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  const double mass = 0.4;
  const double beta = 11.08;
  const int Nt=8;

  std::cout << "beta = " << beta << std::endl;
  WilsonGaugeActionR Waction(beta);

  std::string dir = "/mnt/baracuda_14/grid_claude/16c";
  std::string obsdir = "/mnt/baracuda_14/grid_claude/16c_obs";
  std::filesystem::create_directories(obsdir);

  RealD M5=1.5;
  RealD b=1.5;
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

    const std::vector<Gamma::Algebra> gams = {
      Gamma::Algebra::Identity,
      Gamma::Algebra::Gamma5,
      Gamma::Algebra::GammaX,
      Gamma::Algebra::GammaXGamma5,
    };
    const std::vector<std::string> gam_names = {"id", "g5", "gx", "gxg5"};
    std::vector<LatticeComplex> res(4, LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        LatticePropagator source(UGrid);

        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);

        LatticePropagator StochProp(UGrid);
        Solve(FermAct, source, StochProp);

        LatticeComplex Trace_CF( UGrid );
        for(int ig=0; ig<4; ig++){
          TraceField(Trace_CF, gams[ig], StochProp, source);
          res[ig] = res[ig] + Trace_CF;
        }
      }
    } // for dilute

    for(int ig=0; ig<4; ig++){
      const std::string path = obsdir + "/disc." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacWriter WR(UGrid->IsBoss());
      WR.open(path);
      WR.writeScidacFieldRecord(res[ig], record);
      WR.close();
    }

  } // conf

  Grid_finalize();
}
