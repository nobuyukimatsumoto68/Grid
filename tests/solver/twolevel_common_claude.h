/*************************************************************************************
  twolevel_common_claude.h -- shared helpers for the SPLIT two-level deflation binaries.

  The monolithic Test_dwf_twolevel_claude.cc was split into four single-purpose binaries
  (Nobu 2026-09-09: "when it's stuffed, things are more unpredictable") that share deflation
  vectors on disk, so no heavy machinery (Lanczos / GMRES-DR recycle) ever co-resides:

    (1) Test_dwf_svddump_rbcgne_claude.cc   -- dump RB-Schur^2 low modes           -> deflrb
    (2) Test_dwf_svddump_deflprec_claude.cc -- dump D5^H D5 low modes              -> deflfull
    (3) Test_dwf_rbcgne_claude.cc           -- read deflrb  -> RB-CGNE plain/deflated
    (4) Test_dwf_deflprec_claude.cc         -- read deflfull -> SVD-deflation + M0-FGMRES

  Design note: scripts_nm/svd_deflation_topological_twolevel_claude.md.
  All vectors are dumped as full 5D fields via LIME-free BinaryIO (this tree is --undef LIME).
*************************************************************************************/
#ifndef TWOLEVEL_COMMON_CLAUDE_H
#define TWOLEVEL_COMMON_CLAUDE_H

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>          // F = FreeMobius5DInverse, M0 = FreeLimitPreconditioner
#include <Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h>  // M0-FGMRES / GMRES-DR
#include <fstream>
#include <iomanip>
#include <sstream>

NAMESPACE_BEGIN(Grid);

// Shamir-Mobius DWF parameters (single source of truth for all four binaries).
static const double TL_M5 = 1.8;
static const int    TL_Ls = 8;
static const double TL_bb = 1.5;
static const double TL_cc = 0.5;
static const double TL_mm = 0.1;

// F^dag = (D_free^dag)^{-1} = D_free (D_free^dag D_free)^{-1}  (Grid's consistent adjoint, as CGNE uses).
// The Mobius kernel (b!=c) is NOT g5R5-Hermitian, so there is no Gamma5 shortcut for F^dag; CG on the free
// MdagM is cheap (mass-gapped -> well conditioned). Used only by the frame-optimizer gradient; not in M0.
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
    CG(MdagM, in, z);
    D.M(z, out);
  }
};

// Compute the lowest Nkeep modes of |D_DW|^2 (MdagM) via Chebyshev-IRL (ODD order, cheb_hi auto if <=0).
// Fills modes[0..nkeep-1] and lam[0..nkeep-1] (ascending lambda). Returns nkeep.
static int compute_low_modes(MobiusFermionD& D, GridCartesian* FGrid, GridParallelRNG& RNG5,
                             int Nkeep, int Nk, int Nm,
                             RealD cheb_lo, RealD cheb_hi, int cheb_ord, RealD resid, int MaxIter,
                             std::vector<LatticeFermionD>& modes, std::vector<RealD>& lam) {
  std::cout << GridLogMessage << "==== lowest " << Nkeep << " modes of |D_DW|^2 (Chebyshev-IRL) ====" << std::endl;

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
    cheb_hi = 1.1 * lmax;
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

// Report singular values + physical-quark chirality chi_q (|chi_q|>0.8 => topological; net index = |Q|).
static void report_chirality(MobiusFermionD& D, GridCartesian* UGrid,
                             const std::vector<LatticeFermionD>& modes, const std::vector<RealD>& lam) {
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermionD q4(UGrid);
  LatticeFermionD g5q(UGrid);
  int nchiral = 0;
  RealD index = 0.0;
  std::cout << GridLogMessage << "  mode   sigma            lambda           chi_q" << std::endl;
  for (int i = 0; i < (int)modes.size(); ++i) {
    RealD sigma = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
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
  std::cout << GridLogMessage << "  net index = sum chi_q = " << index << "  => |Q| ~ "
            << (int)std::lround(std::abs(index)) << "   (#chiral |chi_q|>0.8 = " << nchiral << ")" << std::endl;
}

// Checkpoint a set of deflation vectors (full 5D fields, one binary file each) + eigenvalues. LIME-free
// BinaryIO (this tree is --undef LIME); re-read with read_deflation_vectors_{full,odd}.
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

// Read back a deflation dump as FULL 5D fields (for the deflated preconditioner's right singular vectors).
static int read_deflation_vectors_full(std::vector<LatticeFermionD>& V, std::vector<RealD>& E,
                                       GridCartesian* FGrid, const std::string& dir, const std::string& tag) {
  std::string efile = dir + "/" + tag + "_eval.txt";
  std::ifstream ein(efile);
  E.clear();
  int idx = 0;
  RealD ev = 0.0;
  while (ein >> idx >> ev) {
    E.push_back(ev);
  }
  ein.close();
  int k = (int)E.size();
  BinarySimpleMunger<SpinColourVectorD, SpinColourVectorD> munge;
  std::string fmt = getFormatString<vSpinColourVectorD>();
  V.clear();
  for (int i = 0; i < k; ++i) {
    LatticeFermionD vf(FGrid);
    uint32_t nersc_csum = 0;
    uint32_t scidac_csuma = 0;
    uint32_t scidac_csumb = 0;
    std::string file = dir + "/" + tag + "_v" + std::to_string(i) + ".bin";
    BinaryIO::readLatticeObject<vSpinColourVectorD, SpinColourVectorD>(
        vf, file, munge, 0, fmt, nersc_csum, scidac_csuma, scidac_csumb);
    V.push_back(vf);
  }
  std::cout << GridLogMessage << "    read " << k << " " << tag << " full-5D vectors + eigenvalues from " << dir << std::endl;
  return k;
}

// Read back a deflation dump (full 5D fields, RB modes embedded even=0) and return them as ODD-checkerboard
// FrbGrid fields (for RB-CGNE's DeflatedGuesser).
static int read_deflation_vectors_odd(std::vector<LatticeFermionD>& Vrb, std::vector<RealD>& Erb,
                                      GridCartesian* FGrid, GridRedBlackCartesian* FrbGrid,
                                      const std::string& dir, const std::string& tag) {
  std::vector<LatticeFermionD> Vfull;
  int k = read_deflation_vectors_full(Vfull, Erb, FGrid, dir, tag);
  Vrb.clear();
  for (int i = 0; i < k; ++i) {
    LatticeFermionD vodd(FrbGrid);
    vodd.Checkerboard() = Odd;
    pickCheckerboard(Odd, vodd, Vfull[i]);
    Vrb.push_back(vodd);
  }
  return k;
}

// Project x onto the orthogonal complement of the orthonormal basis B:  x <- (1 - B B^dag) x.
static void project_complement(LatticeFermionD& x, const std::vector<LatticeFermionD>& B) {
  for (int i = 0; i < (int)B.size(); ++i) {
    ComplexD c = innerProduct(B[i], x);
    axpy(x, -c, B[i], x);
  }
}

// Deflated non-Hermitian operator  A_hat = P_U D5 P_V (P_V = 1 - V V^dag right, P_U = 1 - U U^dag left,
// u_i = D5 v_i/sigma_i). M0-FGMRES runs on A_hat: the near-null singular directions are projected out
// (smallest singular value sigma_{k+1}) so it is well conditioned. SVD-deflation of the NON-normal D5.
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

NAMESPACE_END(Grid);
#endif
