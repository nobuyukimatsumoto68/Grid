/*************************************************************************************
  Test_dwf_twolevel_claude.cc
  Two-level deflation CHECK (impl plan: scripts_nm/twolevel_deflation_check_impl_plan_claude.md;
  design: scripts_nm/freeprec_twolevel_deflation_design_claude.md).

  CHUNK 1 (this file, for now): compute + STORE the lowest N (default 24) modes of the 5D Hermitian
  domain-wall operator |D_DW|^2 = MdagM, via Chebyshev-accelerated IRL. These are (a) the deflation
  space (lowest ~10) and (b) the C-diagnostic set (all 24). Reuses the validated recipe from
  Test_dwf_spectrum_transform_claude.cc::run_sv (ODD Chebyshev order + auto cheb_hi = 1.1*lambda_max;
  the two IRL-Chebyshev pitfalls). Prints sigma_i + physical-quark chirality chi_q (|chi_q|>0.8 => a
  topological mode; #{topological} = |Q|). VALIDATE vs the saved 640 data (sigma cluster 0.123-0.145,
  3 chiral modes for |Q|=3).

  Later chunks (NOT here): frame A (full-space direct-Omega opt), frame B (deflated-complement opt via
  the new FrameOptimizerDWF_claude.h), bare C on the 24 modes, GMRES-DR(16,32) inverse count.
*************************************************************************************/
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>          // F = FreeMobius5DInverse, M0 = FreeLimitPreconditioner
#include <Grid/qcd/utils/FrameOptimizerDWF_claude.h>     // chunk 2: DWF direct-Omega optimizer + projector
#include <Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h>  // GMRES-DR(16,32) count
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace Grid;

// Checkpoint a set of deflation eigenvectors (one binary file per vector) + their eigenvalues, so the two
// deflation bases (full D5^H D5 for GMRES-DR(M0); RB-Schur^2 for RB-CGNE) can be analysed offline. This
// tree is built --undef LIME, so ScidacWriter is unavailable -> use the LIME-free BinaryIO path (portable
// lexicographic, SIMD-unpacked; re-read with BinaryIO::readLatticeObject onto the matching grid). Works for
// both the full 5D field (FGrid) and the odd-checkerboard RB field (FrbGrid); on read-back set the cb.
static void save_deflation_vectors(const std::vector<LatticeFermionD>& V,
                                   const std::vector<RealD>& eval,
                                   const std::string& dir,
                                   const std::string& tag) {
  BinarySimpleMunger<SpinColourVectorD, SpinColourVectorD> munge;
  std::string fmt = getFormatString<vSpinColourVectorD>();
  for (int i = 0; i < (int)V.size(); ++i) {
    uint32_t nersc_csum = 0;
    uint32_t scidac_csuma = 0;
    uint32_t scidac_csumb = 0;
    std::string file = dir + "/" + tag + "_v" + std::to_string(i) + ".bin";
    BinaryIO::writeLatticeObject<vSpinColourVectorD, SpinColourVectorD>(
        const_cast<LatticeFermionD&>(V[i]), file, munge, 0, fmt, nersc_csum, scidac_csuma, scidac_csumb);
    std::cout << GridLogMessage << "    saved " << file << "  (scidac_csum "
              << std::hex << scidac_csuma << ":" << scidac_csumb << std::dec << ")" << std::endl;
  }
  std::string efile = dir + "/" + tag + "_eval.txt";
  std::ofstream eos(efile);
  for (int i = 0; i < (int)eval.size(); ++i) {
    eos << i << "  " << std::setprecision(16) << eval[i] << "\n";
  }
  eos.close();
  std::cout << GridLogMessage << "    saved " << efile << "  (" << eval.size() << " eigenvalues)" << std::endl;
}

// F^dag = (D_free^dag)^{-1} = D_free (D_free^dag D_free)^{-1}  (Grid's consistent adjoint, as CGNE uses).
// The Mobius kernel (b!=c) is NOT g5R5-Hermitian ([D_minus, P_pm] != 0; Shamir c=0 IS, verified 3e-15),
// so there is no Gamma5 shortcut for F^dag. CG on the free MdagM is cheap (free operator = mass gap ->
// well-conditioned). Used only by the frame-optimizer gradient (M0^dag term); NOT in the preconditioner.
class FreeMobiusAdjInverse : public LinearFunction<LatticeFermionD> {
  MobiusFermionD& D;
  RealD tol;
  int maxit;
public:
  FreeMobiusAdjInverse(MobiusFermionD& D_, RealD tol_, int maxit_) : D(D_), tol(tol_), maxit(maxit_) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    MdagMLinearOperator<MobiusFermionD, LatticeFermionD> MdagM(D);
    ConjugateGradient<LatticeFermionD> CG(tol, maxit, false);
    LatticeFermionD z(in.Grid());
    z = Zero();
    CG(MdagM, in, z);   // z = (D^dag D)^{-1} in
    D.M(z, out);        // out = D z = (D^dag)^{-1} in = F^dag in
  }
};

// Read back deflation vectors dumped by save_deflation_vectors (full 5D fields + eigenvalues) and return
// them as ODD-checkerboard FrbGrid fields (for RB-CGNE's DeflatedGuesser). Round-trips the deflrb dump so
// the RB-Schur^2 IRL (run A) and the deflated RB-CGNE solve (run B) can be SEPARATE processes -- the
// IRL/Lanczos machinery never co-resides with the solve. Returns the number of vectors read.
static int read_deflation_vectors_odd(std::vector<LatticeFermionD>& Vrb, std::vector<RealD>& Erb,
                                      GridCartesian* FGrid, GridRedBlackCartesian* FrbGrid,
                                      const std::string& dir, const std::string& tag) {
  std::string efile = dir + "/" + tag + "_eval.txt";
  std::ifstream ein(efile);
  Erb.clear();
  int idx = 0;
  RealD ev = 0.0;
  while (ein >> idx >> ev) {
    Erb.push_back(ev);
  }
  ein.close();
  int k = (int)Erb.size();
  BinarySimpleMunger<SpinColourVectorD, SpinColourVectorD> munge;
  std::string fmt = getFormatString<vSpinColourVectorD>();
  Vrb.clear();
  for (int i = 0; i < k; ++i) {
    LatticeFermionD vf(FGrid);
    uint32_t nersc_csum = 0;
    uint32_t scidac_csuma = 0;
    uint32_t scidac_csumb = 0;
    std::string file = dir + "/" + tag + "_v" + std::to_string(i) + ".bin";
    BinaryIO::readLatticeObject<vSpinColourVectorD, SpinColourVectorD>(
        vf, file, munge, 0, fmt, nersc_csum, scidac_csuma, scidac_csumb);
    LatticeFermionD vodd(FrbGrid);
    vodd.Checkerboard() = Odd;
    pickCheckerboard(Odd, vodd, vf);   // extract the odd cb (the RB modes were embedded even=0 on dump)
    Vrb.push_back(vodd);
    std::cout << GridLogMessage << "    read " << file << std::endl;
  }
  std::cout << GridLogMessage << "    read " << k << " " << tag << " vectors + eigenvalues from " << dir << std::endl;
  return k;
}

// Project x onto the orthogonal complement of the orthonormal basis B:  x <- (1 - B B^dag) x.
// (axpy(ret,a,x,y) = a*x + y; ret aliasing y is fine -- elementwise.)  Used for the SVD-deflation
// projectors P_V = 1 - V V^dag and P_U = 1 - U U^dag.
static void project_complement(LatticeFermionD& x, const std::vector<LatticeFermionD>& B) {
  for (int i = 0; i < (int)B.size(); ++i) {
    ComplexD c = innerProduct(B[i], x);
    axpy(x, -c, B[i], x);
  }
}

// Deflated non-Hermitian operator  A_hat = P_U D5 P_V,  with P_V = 1 - V V^dag (right singular subspace)
// and P_U = 1 - U U^dag (left singular subspace, u_i = D5 v_i / sigma_i). The M0-preconditioned GMRES runs
// on A_hat: the near-null singular directions of D5 are projected out so A_hat has smallest singular value
// sigma_{k+1} and is well conditioned. SVD-deflation of the NON-normal D5 (singular, not eigen, vectors --
// exact because D5 maps span V -> span U and the low/complement blocks decouple).
class DeflatedD5Op : public LinearOperatorBase<LatticeFermionD> {
  MobiusFermionD& D;
  const std::vector<LatticeFermionD>& U;
  const std::vector<LatticeFermionD>& V;
public:
  DeflatedD5Op(MobiusFermionD& D_, const std::vector<LatticeFermionD>& U_, const std::vector<LatticeFermionD>& V_)
    : D(D_), U(U_), V(V_) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) {
    LatticeFermionD tmp(in.Grid());
    tmp = in;
    project_complement(tmp, V);   // P_V in
    D.M(tmp, out);                // D5 P_V in
    project_complement(out, U);   // P_U D5 P_V in
  }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { GRID_ASSERT(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { GRID_ASSERT(0); }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
};

// Compute the lowest Nkeep modes of |D_DW|^2 (MdagM). Fills modes[0..Nkeep-1] and lam[0..Nkeep-1]
// (ascending lambda). Chebyshev-IRL exactly as run_sv: ODD order, cheb_hi auto if <=0.
static int compute_low_modes(MobiusFermionD& D, GridCartesian* FGrid, GridParallelRNG& RNG5,
                             int Nkeep, int Nk, int Nm,
                             RealD cheb_lo, RealD cheb_hi, int cheb_ord, RealD resid, int MaxIter,
                             std::vector<LatticeFermionD>& modes, std::vector<RealD>& lam) {
  std::cout << GridLogMessage << "==== chunk1: lowest " << Nkeep << " modes of |D_DW|^2 (Chebyshev-IRL) ====" << std::endl;

  if (cheb_ord % 2 == 0) {
    cheb_ord += 1;   // IRL keeps the LARGEST filtered value -> need ODD order so low modes go large POSITIVE
    std::cout << GridLogMessage << "  cheb_ord even -> bumped to " << cheb_ord << " (IRL needs ODD order)" << std::endl;
  }

  MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermMdagM(D);

  if (cheb_hi <= 0.0) {
    LatticeFermionD pmsrc(FGrid);
    gaussian(RNG5, pmsrc);
    PowerMethod<LatticeFermionD> PM;
    RealD lmax = PM(HermMdagM, pmsrc);
    cheb_hi = 1.1 * lmax;   // MUST bracket the TOP of spec(|D|^2) or the Chebyshev overflows above cheb_hi
    std::cout << GridLogMessage << "  power-method lambda_max(|D|^2) = " << lmax << "  -> cheb_hi = " << cheb_hi << std::endl;
  }
  std::cout << GridLogMessage << "  Chebyshev(" << cheb_lo << ", " << cheb_hi << ", " << cheb_ord
            << ")  Nstop=" << Nkeep << " Nk=" << Nk << " Nm=" << Nm << " resid=" << resid << std::endl;

  Chebyshev<LatticeFermionD> Cheby(cheb_lo, cheb_hi, cheb_ord);
  FunctionHermOp<LatticeFermionD> OpCheby(Cheby, HermMdagM);
  PlainHermOp<LatticeFermionD> Op(HermMdagM);
  ImplicitlyRestartedLanczos<LatticeFermionD> IRL(OpCheby, Op, Nkeep, Nk, Nm, resid, MaxIter);

  std::vector<RealD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(FGrid));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  int Nconv = 0;
  IRL.calc(eval, evec, src, Nconv);

  int nkeep = (Nconv < Nkeep) ? Nconv : Nkeep;
  modes.resize(nkeep, LatticeFermionD(FGrid));
  lam.resize(nkeep);
  for (int i = 0; i < nkeep; ++i) {
    modes[i] = evec[i];
    lam[i] = eval[i];
  }
  std::cout << GridLogMessage << "  Nconv=" << Nconv << "  stored " << nkeep << " modes" << std::endl;
  return nkeep;
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  int Nkeep = 24;
  int Nk = 40;
  int Nm = 64;
  RealD cheb_lo = 0.5;
  RealD cheb_hi = -1.0;   // auto = 1.1*lambda_max
  int cheb_ord = 21;
  RealD sv_resid = 1.0e-5;
  int sv_maxit = 200;

  if (GridCmdOptionExists(argv, argv + argc, "--nmodes")) {
    Nkeep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nmodes"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_lo")) {
    cheb_lo = std::stod(GridCmdOptionPayload(argv, argv + argc, "--cheb_lo"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_ord")) {
    cheb_ord = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--cheb_ord"));
  }
  // --stage selects which --pipeline sub-block runs, to split the peak host memory across separate qsub
  // runs (frame + GMRES-DR(M0) path vs the RB-CGNE + RB-Schur^2 path each balloon; co-resident -> OOM).
  std::string stage = "all";   // all | frame | gmresdr | rbcgne
  if (GridCmdOptionExists(argv, argv + argc, "--stage")) {
    stage = GridCmdOptionPayload(argv, argv + argc, "--stage");
  }
  // --ckpt_dir <dir>: if set, checkpoint the 10 deflation eigenvectors of EACH basis (full D5^H D5 and
  // RB-Schur^2) + eigenvalues into <dir> for offline analysis. Empty -> no checkpoint. (dir must exist.)
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) {
    ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  }
  // --svd_klist <comma list>: deflation sizes k for the SVD-deflation solve (stage svddefl). Runs the
  // two-level solve once per k so k=3 (topological) vs k=10 (low cluster) compare in one job.
  std::string svd_klist = "3,10";
  if (GridCmdOptionExists(argv, argv + argc, "--svd_klist")) {
    svd_klist = GridCmdOptionPayload(argv, argv + argc, "--svd_klist");
  }

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::vector<int> seeds5({1, 2, 3, 4, 5});
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4g(UGrid);
  RNG4g.SeedFixedIntegers({11, 12, 13, 14});

  LatticeGaugeFieldD Umu(UGrid);
  if (GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, cfg);
    std::cout << GridLogMessage << "loaded " << cfg << "  plaq=" << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
  } else if (GridCmdOptionExists(argv, argv + argc, "--hot")) {
    std::cout << GridLogMessage << "no --config: HOT start (random gauge, for the gradient check)" << std::endl;
    SU<Nc>::HotConfiguration(RNG4g, Umu);
  } else {
    std::cout << GridLogMessage << "no --config: cold start (unit gauge)" << std::endl;
    SU<Nc>::ColdConfiguration(Umu);
  }

  MobiusFermionD::ImplParams Params(boundary);
  MobiusFermionD D(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  std::vector<LatticeFermionD> modes;
  std::vector<RealD> lam;
  // rbcgne stage builds its OWN RB-Schur^2 modes and never uses the full 5D modes -> skip the (expensive,
  // ~6 GB Nm=64 basis + 2.3 GB stored) 5D mode solve entirely when only RB-CGNE is requested.
  bool skipmodes = GridCmdOptionExists(argv, argv + argc, "--skipmodes") || (stage == "rbcgne");
  if (!skipmodes) {
  int nkeep = compute_low_modes(D, FGrid, RNG5, Nkeep, Nk, Nm, cheb_lo, cheb_hi, cheb_ord, sv_resid, sv_maxit, modes, lam);

  // Report sigma + physical-quark chirality (identify topological modes: |chi_q|>0.8; #topological = |Q|).
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermionD q4(UGrid);
  LatticeFermionD g5q(UGrid);
  int nchiral = 0;
  RealD index = 0.0;   // net index = sum chi_q ; |Q| = |index| (the +/- pairs cancel, NOT counted)
  std::cout << GridLogMessage << "  mode   sigma            lambda           chi_q" << std::endl;
  for (int i = 0; i < nkeep; ++i) {
    RealD sigma = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
    RealD vn = norm2(modes[i]);
    D.ExportPhysicalFermionSolution(modes[i], q4);
    RealD qn = norm2(q4);
    g5q = g5 * q4;
    RealD chiq = real(innerProduct(q4, g5q)) / qn;
    if (std::abs(chiq) > 0.8) {
      nchiral++;
    }
    index += chiq;
    std::cout << GridLogMessage << "  " << i << "   " << sigma << "   " << lam[i] << "   " << chiq
              << (std::abs(chiq) > 0.8 ? "  (chiral)" : "") << std::endl;
  }
  // |Q| = |net index| = |sum chi_q| (topological); the +/- chiral pairs (non-topological near-zeros)
  // cancel, so the COUNT of |chi_q|>0.8 modes OVER-counts |Q|. Report both.
  std::cout << GridLogMessage << "  net index = sum chi_q = " << index << "  => |Q| ~ "
            << (int)std::lround(std::abs(index)) << "   (#chiral |chi_q|>0.8 = " << nchiral
            << ", counts +/- pairs too)" << std::endl;
  std::cout << GridLogMessage << "chunk1 done: " << nkeep << " low modes computed + stored" << std::endl;
  }  // !skipmodes

  // ---- CHUNK 2 gate: FD-check the DWF frame-optimizer analytic gradient at a random SU(3) frame ----
  // (validates FrameOptimizerDWF_claude.h before trusting the descent.) --fdgate to enable.
  if (GridCmdOptionExists(argv, argv + argc, "--fdgate")) {
    std::cout << GridLogMessage << "==== chunk2 FD gate: DWF frame-optimizer gradient ====" << std::endl;
    FreeMobius5DInverse<WilsonImplD> F(FGrid, Ls, M5, bb, cc, mm, boundary);

    GridParallelRNG RNG4(UGrid);
    std::vector<int> seeds4({6, 7, 8, 9});
    RNG4.SeedFixedIntegers(seeds4);

    // generic random SU(3) 4D frame Om4 = exp(Ta(gaussian)) (not identity -> exercises the gradient).
    LatticeColourMatrixD Mr4(UGrid);
    gaussian(RNG4, Mr4);
    LatticeColourMatrixD X4(UGrid);
    X4 = Ta(Mr4);                       // materialise before expMat (expMat wants a Lattice, not an expr)
    LatticeColourMatrixD Om4(UGrid);
    Om4 = expMat(X4, 1.0, 12);
    LatticeColourMatrixD Om5(FGrid);

    // probes: v gaussian 5D, w = D_DW v.
    int nprobe = 8;
    if (GridCmdOptionExists(argv, argv + argc, "--nprobe")) {
      nprobe = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nprobe"));
    }
    std::vector<LatticeFermionD> vprobe(nprobe, LatticeFermionD(FGrid));
    std::vector<LatticeFermionD> wprobe(nprobe, LatticeFermionD(FGrid));
    for (int p = 0; p < nprobe; ++p) {
      gaussian(RNG5, vprobe[p]);
      D.M(vprobe[p], wprobe[p]);   // w = D_DW v
    }

    // Gate with P = 1 (frame A / full space); P is a fixed linear map so P!=1 needs no separate gate.
    // Draw ONE random direction X, compute the analytic derivative an = 2 Re tr(X G) ONCE, then sweep eps
    // with the SAME X so fd -> an cleanly as eps shrinks (a minimum, then rounding). Needs an fp64 F build
    // for a clean loss. F^dag = true adjoint (D_free.Mdag)^{-1} -- Mobius is NOT g5R5-Herm (see header).
    LatticeGaugeFieldD Uunit(UGrid);
    SU<Nc>::ColdConfiguration(Uunit);
    MobiusFermionD Dfree(Uunit, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
    FreeMobiusAdjInverse Fdag(Dfree, 1.0e-10, 2000);
    std::vector<LatticeFermionD> Vempty;
    LatticeColourMatrixD Gf4(UGrid);
    fodwf_broadcast(Om4, Om5, Ls);
    fodwf_loss_force(F, Fdag, Om5, wprobe, vprobe, Vempty, Ls, Gf4);
    LatticeColourMatrixD MrX(UGrid);
    gaussian(RNG4, MrX);
    LatticeColourMatrixD X(UGrid);
    X = Ta(MrX);
    ComplexD trXG = TensorRemove(sum(trace(X * Gf4)));
    RealD an = 2.0 * real(trXG);
    std::cout << GridLogMessage << "  analytic dL/dt = 2 Re tr(X G) = " << an << "  (fixed X, ||G||^2="
              << norm2(Gf4) << ")" << std::endl;
    for (RealD eps = 1.0e-2; eps >= 1.0e-6; eps *= 0.1) {
      LatticeColourMatrixD Op4 = Om4 * expMat(X, eps, 12);
      fodwf_broadcast(Op4, Om5, Ls);
      RealD Lp = fodwf_loss(F, Om5, wprobe, vprobe, Vempty);
      LatticeColourMatrixD Om4n = Om4 * expMat(X, -eps, 12);
      fodwf_broadcast(Om4n, Om5, Ls);
      RealD Lm = fodwf_loss(F, Om5, wprobe, vprobe, Vempty);
      RealD fd = (Lp - Lm) / (2.0 * eps);
      RealD rel = std::abs(fd - an) / (std::abs(an) + 1.0e-30);
      std::cout << GridLogMessage << "  FD gate eps=" << eps << "  fd=" << fd
                << "  rel.err=" << rel << (rel < 1.0e-3 ? "  PASS" : "") << std::endl;
    }
    fodwf_broadcast(Om4, Om5, Ls);   // restore
    std::cout << GridLogMessage << "chunk2 FD gate done" << std::endl;
  }

  // ---- CHUNKS 3+4: two-level pipeline. Landau warm-start -> frame A (full-space direct-Omega opt) vs
  //      frame B (deflated-complement opt, Vlow = lowest defl_k modes) -> bare C on the 24 modes +
  //      GMRES-DR(16,32) inverse count, for Landau / A / B.  --pipeline to enable (needs the modes).
  if (GridCmdOptionExists(argv, argv + argc, "--pipeline")) {
    int defl_k = 10;
    int fo_iter = 15;
    int fo_nprobe = 6;
    RealD fo_eta = 0.02;
    int flow_nstep = 873;   // s/t0=6 frame (production default)
    RealD flow_eps = 0.02;
    if (GridCmdOptionExists(argv, argv + argc, "--defl_k")) defl_k = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--defl_k"));
    if (GridCmdOptionExists(argv, argv + argc, "--fo_iter")) fo_iter = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo_iter"));
    if (GridCmdOptionExists(argv, argv + argc, "--fo_nprobe")) fo_nprobe = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo_nprobe"));
    if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
    int nkeep = (int)modes.size();
    if (nkeep > 0 && defl_k > nkeep) defl_k = nkeep;   // nkeep==0 for --stage rbcgne (modes skipped)
    std::cout << GridLogMessage << "==== chunks 3+4 pipeline: defl_k=" << defl_k << " fo_iter=" << fo_iter
              << " fo_nprobe=" << fo_nprobe << " flow(" << flow_eps << "x" << flow_nstep << ") ====" << std::endl;

    // Run-mode split (--stage). The frame + GMRES-DR(M0) path (24 stored 5D modes + GMRES-DR recycle
    // space) and the RB-CGNE path (RB-Schur^2 Lanczos basis + CG workspace) each balloon host memory, so
    // co-resident in ONE process they OOM on 16^4. Split into separate qsub runs to halve the peak:
    //   all     : everything (default; needs the most memory)
    //   precond : frame + svddefl (the whole M0/preconditioner side), NO RB-CGNE          <- run 1
    //   rbcgne  : RB-Schur^2 low modes (IRL) -> RB-CGNE plain vs deflated (NO frame/5D modes)  <- run 2
    //   frame   : Landau warm-start -> frame A/B -> C(24 modes) -> per-frame GMRES-DR(16,32)
    //   svddefl : SVD-deflation pre-solve of the topological modes + M0-GMRES on the complement (the
    //             CORRECT two-level solve; supersedes gmresdr). Loops --svd_klist (default 3,10).
    //   gmresdr : OLD frozen-seed harmonic-Ritz deflation (non-convergent, j7510214) -- opt-in reference.
    // The natural 2-run split (avoids co-resident OOM AND redundant flow/mode-solve): precond then rbcgne.
    bool do_svd    = (stage == "all" || stage == "precond" || stage == "svddefl");
    bool do_landau = (stage == "all" || stage == "precond" || stage == "frame" || stage == "gmresdr" || stage == "svddefl");
    bool do_frame  = (stage == "all" || stage == "precond" || stage == "frame");
    bool do_gmres  = (stage == "gmresdr");   // superseded by do_svd; opt-in only
    bool do_rbcgne = (stage == "all" || stage == "rbcgne");
    std::cout << GridLogMessage << "  pipeline stage = " << stage << "  (landau=" << do_landau
              << " frame=" << do_frame << " svddefl=" << do_svd << " gmresdr=" << do_gmres
              << " rbcgne=" << do_rbcgne << ")" << std::endl;

    // Shared solve source from a DEDICATED fixed-seed RNG so b is byte-identical across separated --stage
    // runs (apples-to-apples: RB-CGNE and GMRES-DR must invert the SAME b, even in different jobs).
    LatticeFermionD bsrc(FGrid);
    GridParallelRNG RNGsrc(FGrid);
    RNGsrc.SeedFixedIntegers({20, 21, 22, 23});
    gaussian(RNGsrc, bsrc);

    if (do_landau) {
      // Landau warm-start frame (flow s/t0=6 + Fourier-accelerated Landau gauge-fix); mass-independent.
      LatticeGaugeFieldD Uflowed(UGrid);
      WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
      wf.smear(Uflowed, Umu);
      LatticeColourMatrixD OmL(UGrid);
      FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
          Uflowed, OmL, 0.1 / 16.0, 4000, 1.0e-12, 1.0e-12, true, -1, false);
      std::cout << GridLogMessage << "  Landau frame: flowed-fixed functional="
                << (1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed)) << std::endl;

      FreeMobius5DInverse<WilsonImplD> F(FGrid, Ls, M5, bb, cc, mm, boundary);
      LatticeGaugeFieldD Uunit(UGrid);
      SU<Nc>::ColdConfiguration(Uunit);
      MobiusFermionD Dfree(Uunit, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
      FreeMobiusAdjInverse Fdag(Dfree, 1.0e-10, 2000);

      NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> LinOp(D);

      // Checkpoint the deflation vectors (right singular vectors v_i = lowest defl_k of D5^H D5) once for
      // any M0-side run, for offline analysis. (u_i = D5 v_i/sigma_i are derivable; not saved.)
      if (!ckpt_dir.empty() && !modes.empty()) {
        std::vector<LatticeFermionD> Vck(modes.begin(), modes.begin() + defl_k);
        std::vector<RealD> lam_ck(lam.begin(), lam.begin() + defl_k);
        std::cout << GridLogMessage << "  checkpointing " << defl_k << " full-D5^H-D5 deflation vectors:" << std::endl;
        save_deflation_vectors(Vck, lam_ck, ckpt_dir, "deflfull");
      }

      LatticeColourMatrixD OmA(UGrid);
      OmA = OmL;
      LatticeColourMatrixD OmB(UGrid);
      OmB = OmL;

      if (do_frame) {
        // deflation subspace Vlow = lowest defl_k modes (IRL modes are orthonormal).
        std::vector<LatticeFermionD> Vlow(modes.begin(), modes.begin() + defl_k);
        std::vector<LatticeFermionD> Vempty;

        // probes for the frame optimiser: v gaussian, w = D v.
        std::vector<LatticeFermionD> vpr(fo_nprobe, LatticeFermionD(FGrid));
        std::vector<LatticeFermionD> wpr(fo_nprobe, LatticeFermionD(FGrid));
        for (int p = 0; p < fo_nprobe; ++p) {
          gaussian(RNG5, vpr[p]);
          D.M(vpr[p], wpr[p]);
        }

        // frame A/B direct-Omega opt (SLOW: F^dag = CG on free MdagM). Gated behind --frameopt; default
        // OFF so the run focuses on the deflated-solve comparison. If off, A=B=Landau.
        bool frameopt = GridCmdOptionExists(argv, argv + argc, "--frameopt");
        LatticeColourMatrixD Om5tmp(FGrid);
        if (frameopt) {
          std::cout << GridLogMessage << "  --- frame A (full-space direct-opt) ---" << std::endl;
          fodwf_descend(F, Fdag, OmA, Om5tmp, wpr, vpr, Vempty, Ls, fo_iter, fo_eta, 25);
          std::cout << GridLogMessage << "  --- frame B (deflated-complement opt, k=" << defl_k << ") ---" << std::endl;
          fodwf_descend(F, Fdag, OmB, Om5tmp, wpr, vpr, Vlow, Ls, fo_iter, fo_eta, 25);
        }

        // Chunk 3: bare C(psi_i) = ||psi_i - M0 D psi_i|| / ||psi_i|| on the 24 modes, for Landau / A / B.
        LatticeColourMatrixD frames[3] = {OmL, OmA, OmB};
        const char* fname[3] = {"Landau", "frameA(full)", "frameB(defl)"};
        int nframe = frameopt ? 3 : 1;   // A,B == Landau unless --frameopt -> only report Landau then
        LatticeFermionD Dpsi(FGrid), M0Dpsi(FGrid), rC(FGrid);
        for (int fr = 0; fr < nframe; ++fr) {
          FreeLimitPreconditioner<WilsonImplD> M0(F, frames[fr], FGrid);
          RealD cmax_all = 0.0, csum_all = 0.0, cmax_hi = 0.0, csum_hi = 0.0;
          int nhi = 0;
          for (int i = 0; i < nkeep; ++i) {
            D.M(modes[i], Dpsi);
            M0(Dpsi, M0Dpsi);
            rC = modes[i] - M0Dpsi;
            RealD Ci = std::sqrt(norm2(rC) / norm2(modes[i]));
            csum_all += Ci;
            if (Ci > cmax_all) cmax_all = Ci;
            if (i >= defl_k) {   // the NON-deflated modes = the interesting ones
              csum_hi += Ci;
              nhi++;
              if (Ci > cmax_hi) cmax_hi = Ci;
            }
          }
          std::cout << GridLogMessage << "  [C " << fname[fr] << "] all24: mean=" << csum_all / nkeep
                    << " max=" << cmax_all << " | non-deflated(" << defl_k << "-" << nkeep << "): mean="
                    << (nhi ? csum_hi / nhi : 0.0) << " max=" << cmax_hi << std::endl;
        }

        // Chunk 4: GMRES-DR(16,32) inverse count on D, for Landau / A / B (bare frame effect; m=0.1).
        for (int fr = 0; fr < nframe; ++fr) {
          FreeLimitPreconditioner<WilsonImplD> M0(F, frames[fr], FGrid);
          RecyclingGeneralisedMinimalResidual<LatticeFermionD> RGM(1.0e-8, 20000, M0, 16, 32, false, false);
          LatticeFermionD xg(FGrid);
          xg = Zero();
          RGM(LinOp, bsrc, xg);
          std::cout << GridLogMessage << "  [GMRES-DR(16,32) " << fname[fr] << "] iters=" << RGM.IterationCount
                    << "  D_W applies=" << (long)Ls * RGM.IterationCount << std::endl;
        }
      }  // do_frame

      if (do_svd) {
        // ---- SVD-deflation pre-solve + M0-GMRES on the complement (the CORRECT two-level solve) ----
        // Would-be-topological modes are the lowest RIGHT singular vectors v_i of D5 (= lowest D5^H D5
        // modes); left u_i = D5 v_i/sigma_i. Pre-solve x_lo = V Sigma^-1 U^dag b (exact, NO preconditioner);
        // then M0-GMRES on the deflated complement A_hat = P_U D5 P_V; x = x_lo + P_V x_perp. m=0.1, same b.
        FreeLimitPreconditioner<WilsonImplD> M0L(F, OmL, FGrid);
        std::vector<int> klist;
        std::stringstream kss(svd_klist);
        std::string ktok;
        while (std::getline(kss, ktok, ',')) {
          int kv = std::stoi(ktok);
          if (kv > 0 && kv <= nkeep) klist.push_back(kv);
        }
        for (size_t ki = 0; ki < klist.size(); ++ki) {
          int kd = klist[ki];
          std::vector<LatticeFermionD> Vs(modes.begin(), modes.begin() + kd);
          std::vector<LatticeFermionD> Us;
          Us.reserve(kd);
          for (int i = 0; i < kd; ++i) {
            RealD sig = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
            LatticeFermionD ui(FGrid);
            D.M(Vs[i], ui);            // D5 v_i
            ui = ui * (1.0 / sig);     // u_i = D5 v_i / sigma_i (orthonormal since V orthonormal eigvecs)
            Us.push_back(ui);
          }
          // pre-solve the topological subspace, NO preconditioner: x_lo = sum_i (u_i^dag b / sigma_i) v_i
          LatticeFermionD xlo(FGrid);
          xlo = Zero();
          for (int i = 0; i < kd; ++i) {
            RealD sig = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
            ComplexD ci = innerProduct(Us[i], bsrc) / sig;
            axpy(xlo, ci, Vs[i], xlo);
          }
          // deflated RHS  b_perp = P_U b ; solve A_hat x_perp = b_perp with M0-GMRES(16,32) on the complement
          LatticeFermionD bperp(FGrid);
          bperp = bsrc;
          project_complement(bperp, Us);
          DeflatedD5Op Ahat(D, Us, Vs);
          RecyclingGeneralisedMinimalResidual<LatticeFermionD> RGMs(1.0e-8, 20000, M0L, 16, 32, false, false);
          LatticeFermionD xperp(FGrid);
          xperp = Zero();
          RGMs(Ahat, bperp, xperp);
          project_complement(xperp, Vs);   // keep x_perp perp V (V-component undetermined by A_hat)
          LatticeFermionD xfull(FGrid);
          xfull = xlo + xperp;
          // true residual of the ORIGINAL system D5 x = b (the correctness gate for the two-level split)
          LatticeFermionD chk(FGrid);
          D.M(xfull, chk);
          chk = chk - bsrc;
          RealD relres = std::sqrt(norm2(chk) / norm2(bsrc));
          std::cout << GridLogMessage << "  [SVD-defl(k=" << kd << ") M0-GMRES(16,32)] iters="
                    << RGMs.IterationCount << "  D_W=" << (long)Ls * RGMs.IterationCount
                    << "  true rel.res=" << relres << std::endl;
        }
      }  // do_svd

      if (do_gmres) {
        // OLD (superseded): GMRES-DR(M0) with the D5^H D5 modes FROZEN into the harmonic-Ritz recycle.
        // Non-convergent (j7510214: 20000 iters, res 0.034) -- singular vectors are not eigenvectors of
        // M0 D5, and freezing kills the adaptation. Kept opt-in (--stage gmresdr) for A/B reference only.
        FreeLimitPreconditioner<WilsonImplD> M0L(F, OmL, FGrid);
        std::vector<LatticeFermionD> Vfull(modes.begin(), modes.begin() + defl_k);
        RecyclingGeneralisedMinimalResidual<LatticeFermionD> RGMd(1.0e-8, 20000, M0L, 16, defl_k, false, false);
        RGMd.SetDeflationSubspace(Vfull, true);
        LatticeFermionD xgd(FGrid);
        xgd = Zero();
        RGMd(LinOp, bsrc, xgd);
        std::cout << GridLogMessage << "  [GMRES-DR(M0) deflated(V_full,k=" << defl_k << ")] iters="
                  << RGMd.IterationCount << "  D_W=" << (long)Ls * RGMd.IterationCount << std::endl;
      }  // do_gmres
    }  // do_landau

    if (do_rbcgne) {
      // ---- RB-CGNE + its own deflation (separate --stage to avoid co-resident OOM with the frame path) --
      // RB-CGNE deflated by the RB-Schur^2 low modes (the operator ITS CG runs on) -- apples-to-apples with
      // GMRES-DR(M0) deflated by full D5^H D5; both k=defl_k, each its own squared operator. m=0.1, same b.
      // (a) RB-Schur lowest defl_k modes via Chebyshev-IRL on SchurDiagMooeeOperator (S^dag S) on FrbGrid.
      SchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD> SchurOp(D);
      GridParallelRNG RNGrb(FrbGrid);
      RNGrb.SeedFixedIntegers({7, 8, 9, 10});
      LatticeFermionD pmrb(FrbGrid);
      pmrb.Checkerboard() = Odd;
      gaussian(RNGrb, pmrb);
      PowerMethod<LatticeFermionD> PMrb;
      RealD lmax_rb = PMrb(SchurOp, pmrb);
      Chebyshev<LatticeFermionD> Cheby_rb(0.5, 1.1 * lmax_rb, 21);
      FunctionHermOp<LatticeFermionD> OpCheby_rb(Cheby_rb, SchurOp);
      PlainHermOp<LatticeFermionD> Op_rb(SchurOp);
      ImplicitlyRestartedLanczos<LatticeFermionD> IRLrb(OpCheby_rb, Op_rb, defl_k, 2 * defl_k, 4 * defl_k, 1.0e-5, 300);
      std::vector<RealD> eval_rb(4 * defl_k);
      std::vector<LatticeFermionD> evec_rb(4 * defl_k, LatticeFermionD(FrbGrid));
      for (auto &e : evec_rb) e.Checkerboard() = Odd;
      LatticeFermionD rbsrc(FrbGrid);
      rbsrc.Checkerboard() = Odd;
      gaussian(RNGrb, rbsrc);
      int Nconv_rb = 0;
      IRLrb.calc(eval_rb, evec_rb, rbsrc, Nconv_rb);
      int kdrb = (defl_k < Nconv_rb) ? defl_k : Nconv_rb;
      std::cout << GridLogMessage << "  RB-Schur^2 low modes: Nconv=" << Nconv_rb << " lowest eval="
                << eval_rb[0] << "  (using " << kdrb << ")" << std::endl;
      std::vector<LatticeFermionD> Vrb(evec_rb.begin(), evec_rb.begin() + kdrb);
      std::vector<RealD> Erb(eval_rb.begin(), eval_rb.begin() + kdrb);
      if (!ckpt_dir.empty()) {
        // embed each odd-cb RB mode into a full 5D field (even part zero) -> uniform full-grid format with
        // the full-D5^H-D5 basis, so offline overlap/chirality analysis treats both bases identically.
        std::vector<LatticeFermionD> Vrb_full;
        for (int i = 0; i < kdrb; ++i) {
          LatticeFermionD vf(FGrid);
          vf = Zero();
          setCheckerboard(vf, Vrb[i]);
          Vrb_full.push_back(vf);
        }
        std::cout << GridLogMessage << "  checkpointing " << kdrb << " RB-Schur^2 deflation vectors (embedded full 5D):" << std::endl;
        save_deflation_vectors(Vrb_full, Erb, ckpt_dir, "deflrb");
      }

      // (b) RB-CGNE: plain vs deflated(Vrb).
      ConjugateGradient<LatticeFermionD> CGp(1.0e-8, 60000, false);
      SchurRedBlackDiagMooeeSolve<LatticeFermionD> Sp(CGp);
      LatticeFermionD sp(FGrid);
      sp = Zero();
      Sp(D, bsrc, sp);
      std::cout << GridLogMessage << "  [RB-CGNE plain] iters=" << CGp.IterationsToComplete
                << "  D_W=" << (long)2 * Ls * CGp.IterationsToComplete << std::endl;
      ConjugateGradient<LatticeFermionD> CGd(1.0e-8, 60000, false);
      SchurRedBlackDiagMooeeSolve<LatticeFermionD> Sd(CGd);
      DeflatedGuesser<LatticeFermionD> g_rb(Vrb, Erb);
      LatticeFermionD sd(FGrid);
      sd = Zero();
      Sd(D, bsrc, sd, g_rb);
      std::cout << GridLogMessage << "  [RB-CGNE deflated(k=" << kdrb << ")] iters=" << CGd.IterationsToComplete
                << "  D_W=" << (long)2 * Ls * CGd.IterationsToComplete << std::endl;
    }  // do_rbcgne

    std::cout << GridLogMessage << "chunks 3+4 pipeline done" << std::endl;
  }

  Grid_finalize();
  return 0;
}
