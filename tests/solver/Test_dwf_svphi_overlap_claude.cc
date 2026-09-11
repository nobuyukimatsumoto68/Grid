/*************************************************************************************
  Test_dwf_svphi_overlap_claude.cc -- overlaps between the SINGULAR vectors (u_i,v_i) and the EIGEN
  vectors (phi_i,chi_i) of the massless 4D D_W[Urot], from the dumped wtop_psi (v_i = right singular)
  and wtop_phi (phi_i = right eigen). Companion to Test_dwf_wtop_eigs_claude.cc.

  v_i : right singular vectors of D_W (= eigenvectors of D_W^H D_W = H_W^2), the wtop_psi dump.
  u_i : left singular vectors, u_i = D_W v_i / sigma_i, sigma_i = ||D_W v_i||.
  phi_i: right eigenvectors of D_W (wtop_phi dump); chi_i = gamma5 phi_i (left, for real lambda).
  mu_i : D_W eigenvalue = <phi_i|D_W phi_i> (phi_i is a true eigenvector; Rayleigh quotient exact).

  Prints (parseable):  SIGMA i sigma_i ;  LAM j |mu_j| ;  MVPHI i j |<v_i|phi_j>| ;  MUPHI i j |<u_i|phi_j>|.
  Frame Urot loaded from the cache (urot_<cfg>_..._gf<gf>.nersc). Massless D_W.

  CLI: --config <NERSC> --ckpt_dir <dir with wtop_psi/phi> [--flowcache <dir>] [--flow_nstep 873]
       [--flow_eps 0.02] [--gf_maxit 4000]
*************************************************************************************/
#include "twolevel_common_claude.h"
#include <sstream>

using namespace Grid;

static std::string urot_file(const std::string& dir, const std::string& cfg_base, double eps, int nstep, int gfmax) {
  std::ostringstream os;
  os << dir << "/urot_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep << "_gf" << gfmax << ".nersc";
  return os.str();
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);
  std::vector<Complex> boundary = {1, 1, 1, -1};

  double flow_eps = 0.02;
  int flow_nstep = 873;
  int gf_maxit = 4000;
  std::string ckpt_dir = "";
  std::string flowcache_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  if (GridCmdOptionExists(argv, argv + argc, "--flowcache")) flowcache_dir = GridCmdOptionPayload(argv, argv + argc, "--flowcache");
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow_eps"));
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) gf_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gf_maxit"));

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
  std::string cfg_base = cfg.substr(cfg.find_last_of('/') + 1);
  std::string urotpath = urot_file(flowcache_dir, cfg_base, flow_eps, flow_nstep, gf_maxit);
  LatticeGaugeFieldD Urot(UGrid);
  FieldMetaData h;
  NerscIO::readConfiguration(Urot, h, urotpath);
  std::cout << GridLogMessage << "[urotcache] LOADED " << urotpath << std::endl;

  WilsonImplD::ImplParams Params(boundary);
  WilsonFermionD DW(Urot, *UGrid, *UrbGrid, 0.0, Params);
  Gamma g5(Gamma::Algebra::Gamma5);

  // read v_i (wtop_psi) and phi_j (wtop_phi)
  std::vector<LatticeFermionD> v;
  std::vector<RealD> sigE;
  int n = read_deflation_vectors_full(v, sigE, UGrid, ckpt_dir, "wtop_psi");
  std::vector<LatticeFermionD> phi;
  std::vector<RealD> lamE;
  int na = read_deflation_vectors_full(phi, lamE, UGrid, ckpt_dir, "wtop_phi");
  std::cout << GridLogMessage << "read " << n << " v_i and " << na << " phi_j" << std::endl;

  // u_i = D_W v_i / sigma_i ; sigma_i = ||D_W v_i|| ; normalize v_i first
  std::vector<LatticeFermionD> u(n, LatticeFermionD(UGrid));
  std::vector<RealD> sigma(n);
  LatticeFermionD DWv(UGrid);
  for (int i = 0; i < n; ++i) {
    RealD nv = std::sqrt(norm2(v[i]));
    v[i] = v[i] * (1.0 / nv);
    DW.M(v[i], DWv);
    sigma[i] = std::sqrt(norm2(DWv));
    u[i] = DWv * (1.0 / sigma[i]);
    std::cout << "SIGMA " << i << " " << sigma[i] << std::endl;
  }

  // mu_j = <phi_j|D_W phi_j> (phi_j normalized) ; |mu_j|
  LatticeFermionD DWphi(UGrid);
  for (int j = 0; j < na; ++j) {
    RealD np = std::sqrt(norm2(phi[j]));
    phi[j] = phi[j] * (1.0 / np);
    DW.M(phi[j], DWphi);
    ComplexD mu = innerProduct(phi[j], DWphi);
    RealD absmu = std::hypot(mu.real(), mu.imag());
    std::cout << "LAM " << j << " " << absmu << std::endl;
  }

  // overlap matrices |<v_i|phi_j>| and |<u_i|phi_j>|
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < na; ++j) {
      ComplexD vp = innerProduct(v[i], phi[j]);
      std::cout << "MVPHI " << i << " " << j << " " << std::hypot(vp.real(), vp.imag()) << std::endl;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < na; ++j) {
      ComplexD up = innerProduct(u[i], phi[j]);
      std::cout << "MUPHI " << i << " " << j << " " << std::hypot(up.real(), up.imag()) << std::endl;
    }
  }

  std::cout << GridLogMessage << "svphi_overlap done" << std::endl;
  Grid_finalize();
  return 0;
}
