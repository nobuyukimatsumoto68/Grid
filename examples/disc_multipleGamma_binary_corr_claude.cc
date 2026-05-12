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
  const char* const short_opts = "+:m:b:d:o:";
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
  double mass   = 0.4;
  double beta   = 11.08;
  std::string dir    = "";
  std::string obsdir = "/mnt/baracuda_14/grid_claude/obs_nc4nf1_2448/obs_nc4nf1_2448_b11p045_m0p4000";
  ParseArgs(argc, argv, mass, beta, dir, obsdir);

  const int Ls=16;
  Grid_init(&argc,&argv);

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                   GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  const int Nt = UGrid->_fdimensions[Tdir];

  std::cout << "mass=" << mass << " beta=" << beta << " Nt=" << Nt << std::endl;
  WilsonGaugeActionR Waction(beta);

  // std::filesystem::create_directories(obsdir);

  RealD M5=1.5;
  RealD b=1.5;
  RealD c=0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  FermionAction::ImplParams Params(boundary);

  LatticeGaugeField Umu(UGrid);
  // GridParallelRNG  RNG4(UGrid);

  int conf_min, conf_max, interval;
  {
    std::vector<int> confs;
    const std::string prefix = "traces.id.";
    for(const auto& entry : std::filesystem::directory_iterator(obsdir)){
      const std::string fname = entry.path().filename().string();
      if(fname.size() <= prefix.size()) continue;
      if(fname.substr(0, prefix.size()) != prefix) continue;
      const std::string numstr = fname.substr(prefix.size());
      if(numstr.empty() || !std::all_of(numstr.begin(), numstr.end(), ::isdigit)) continue;
      confs.push_back(std::stoi(numstr));
    }
    assert(!confs.empty());
    std::sort(confs.begin(), confs.end());
    conf_min = confs.front();
    interval = (confs.size() >= 2) ? confs[1] - confs[0] : 1;
    conf_max = confs.back() + interval;
  }
  std::cout << "conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;


  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity,
    Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,
    Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5,
    Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5,
    Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id", "g5", "gx", "gy", "gz", "gt", "gxg5", "gyg5", "gzg5", "gtg5"};

  std::vector<ComplexD> G(Nt, 0.0);

  for(int conf=conf_min; conf<conf_max; conf+=interval){
    {
      bool all_there = true;
      for(int ig=0; ig<(int)gam_names.size(); ig++){
        const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
        if(!std::filesystem::exists(path)){
          all_there = false;
          break;
        }
      }
      if(!all_there){
        std::cout << GridLogMessage << "skipping conf " << conf << " (output incomplete)" << std::endl;
        continue;
      }
    }

    // std::vector<LatticeComplex> res(gam_names.size(), LatticeComplex(UGrid));
    LatticeComplex lat(UGrid);
    lat = Zero();
    // for(auto &r : lat) r = Zero();
    // for(int ig=0; ig<(int)gam_names.size(); ig++)
    const int ig=0; // test
    {
      const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacReader RD;
      RD.open(path);
      // RD.readScidacFieldRecord(lat[ig], record);
      RD.readScidacFieldRecord(lat, record);
      RD.close();
      std::cout << GridLogMessage
                << "conf " << conf << " " << gam_names[ig]
                << " norm2=" << norm2(lat) << std::endl;
    }

    // FFT
    LatticeComplex ft(UGrid);
    ft = Zero();
    FFT theFFT(UGrid);
    theFFT.FFT_all_dim(ft, lat, FFT::forward);

    // Gtilde[t] = sum_p ft(-p)*ft(p)*exp(i*pt*t), ft(-p)=conj(ft(p)) for real lat
    LatticeComplex Gtilde(UGrid);
    Gtilde = adj(ft) * ft;

    // G = IFFT(A), A[n] = sum_spatial product; computed as backward FFT then sliceSum
    LatticeComplex G_field(UGrid);
    theFFT.FFT_dim(G_field, Gtilde, Tdir, FFT::backward);

    std::vector<TComplex> G_slices(Nt);
    sliceSum(G_field, G_slices, Tdir);

    // std::vector<ComplexD> G(Nt);
    const ComplexD vol = UGrid->_gsites;
    for(int t=0; t<Nt; t++) G[t] += vol * TensorRemove(G_slices[t]);

    // {
    //   const std::string path = dir+"/"+lat_prefix+std::to_string(conf);
    //   FieldMetaData header;
    //   NerscIO::readConfiguration(Umu, header, path);
    //   RNG4.SeedUniqueString(path);
    // }
    // FermionAction FermAct(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c, Params);
  }


  for(int t=0; t<Nt; t++){
    std::cout << t << "\t" << real(G[t]) << "\t" << imag(G[t]) << std::endl;
      // for(int eo=0; eo<=1; eo++){
      //   LatticePropagator source(UGrid);

      //   StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);

      //   LatticePropagator StochProp(UGrid);
      //   Solve(FermAct, source, StochProp);

      //   LatticeComplex Trace_CF( UGrid );
      //   for(int ig=0; ig<(int)gam_names.size(); ig++){
      //     TraceField(Trace_CF, gams[ig], StochProp, source);
      //     res[ig] = res[ig] + Trace_CF;
      //   }
      // }
  } // for dilute

  Grid_finalize();
}
