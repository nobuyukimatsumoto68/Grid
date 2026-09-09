// Free WILSON preconditioner test.
//  (no --config) Chunk 0 cold gate: ||F_W D_W v - v|| ~ eps at unit gauge/periodic vs Grid WilsonFermionD.
//  (--config <nersc>) Chunk 1+2 headline: flow+Landau -> Omega -> M0_W = Omega^dag F_W Omega; residual
//   proxy ||M0_W D_W[U] v - v||/||v||; then RB-CGNE (SchurRedBlackDiagMooee, honest baseline) vs
//   FGMRES(M0_W) on the AP-time interacting Wilson operator -- N_it + wall + speedup. Per-config mass via
//   --mass. Plan: dwf4_qcd_claude/grid_free_wilson_impl_plan_claude.md.
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>
#include <Grid/qcd/utils/FreeWilsonTwisted_claude.h>
#include <Grid/qcd/utils/QSqueezeFlowAction_claude.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>
#include <Grid/algorithms/iterative/ChebyshevEllipse_claude.h>
#include <sstream>
#include <fstream>
#include <cstdio>

using namespace Grid;

// LinearFunction wrappers so the non-Hermitian Arnoldi (IRA) can be pointed at D_W and at the preconditioned
// M0 D_W (right-precond spectrum: D applied then M0; same spectrum as D M0). Used by the --spectrum mode.
struct DwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  DwLinOp(WilsonFermionD& d) : Dw(d) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) { Dw.M(in, out); }
};
struct M0DwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  FreeLimitPreconditionerW<WilsonImplD>& M0;
  mutable LatticeFermionD tmp;
  M0DwLinOp(WilsonFermionD& d, FreeLimitPreconditionerW<WilsonImplD>& m, GridBase* g) : Dw(d), M0(m), tmp(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Dw.M(in, tmp);
    M0(tmp, out);
  }
};
// H_W = gamma5 D_W (the hermitian Wilson operator): real eigenvalues (both signs), |lambda_HW| = singular
// values of D_W. Used by --opscan.
struct HwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  mutable LatticeFermionD tmp;
  HwLinOp(WilsonFermionD& d, GridBase* g) : Dw(d), tmp(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Gamma g5(Gamma::Algebra::Gamma5);
    Dw.M(in, tmp);
    out = g5 * tmp;            // H_W = gamma5 D_W
  }
};
// M1 D_W (next-order preconditioned operator): spectrum mu should cluster near 1; the quality metric on an
// eigenmode is C = |1 - mu| (>=1 = red flag). Used by --opscan.
struct M1DwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  FreeLimitPreconditionerW1<WilsonImplD>& M1;
  mutable LatticeFermionD tmp;
  M1DwLinOp(WilsonFermionD& d, FreeLimitPreconditionerW1<WilsonImplD>& m, GridBase* g) : Dw(d), M1(m), tmp(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Dw.M(in, tmp);
    M1(tmp, out);             // M1 D_W
  }
};
// Mx D_W (scale-mixed preconditioned operator); C on an eigenmode = |1 - mu|. Used by --mx.
struct MxDwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  FreeLimitPreconditionerWx<WilsonImplD>& Mx;
  mutable LatticeFermionD tmp;
  MxDwLinOp(WilsonFermionD& d, FreeLimitPreconditionerWx<WilsonImplD>& m, GridBase* g) : Dw(d), Mx(m), tmp(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Dw.M(in, tmp);
    Mx(tmp, out);             // Mx D_W
  }
};
// Shift-invert-at-zero operator: out = D_W^{-1} in via CGNE (CG on the Hermitian-PD normal operator
// D_W^dag D_W), so out = (D^dag D)^{-1} D^dag in = D^{-1} in. Fed to IRA (largest-modulus wanted): the
// largest |1/lambda| are D_W's SMALLEST |lambda| -- Arnoldi's easy, STABLE case (diagnosis fix 3). The
// near-zero D_W modes converge fast + accurately; the true lambda is recovered by the RR-cleanup.
struct ShiftInvertZeroOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  MdagMLinearOperator<WilsonFermionD, LatticeFermionD> NormalOp;
  ConjugateGradient<LatticeFermionD> CG;
  mutable LatticeFermionD ddagb;
  ShiftInvertZeroOp(WilsonFermionD& d, GridBase* g, double tol, int maxit)
    : Dw(d), NormalOp(d), CG(tol, maxit, false), ddagb(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Dw.Mdag(in, ddagb);        // D^dag in
    out = Zero();
    CG(NormalOp, ddagb, out);  // out = (D^dag D)^{-1} D^dag in = D^{-1} in
  }
};
// Deflated (two-level) preconditioner: wraps a base preconditioner M and a deflation subspace V (columns =
// the near-zero D_W eigenmodes the IRA found). Additive coarse-grid correction:
//   x_c = V A_c^{-1} V^dag r,   out = x_c + M (r - D_W x_c),   A_c = V^dag D_W V (small dense, Eigen).
// The coarse (deflation) space is inverted EXACTLY, so the near-zero modes are removed from the spectrum
// FGMRES sees -> the topological blockers stop limiting the outer count. Cost/apply = base M + 1 extra D_W.
struct DeflatedPrec : public LinearFunction<LatticeFermionD> {
  LinearFunction<LatticeFermionD>& M;      // base preconditioner (M0 or M1)
  WilsonFermionD& Dw;
  std::vector<LatticeFermionD>& V;         // orthonormal deflation vectors
  std::vector<LatticeFermionD> AV;         // D_W V (precomputed)
  Eigen::MatrixXcd Acinv;                  // (V^dag D_W V)^{-1}
  int nc;
  mutable LatticeFermionD xc, r2, Mr;
  DeflatedPrec(LinearFunction<LatticeFermionD>& M_, WilsonFermionD& d,
               std::vector<LatticeFermionD>& V_, GridBase* g)
    : M(M_), Dw(d), V(V_), xc(g), r2(g), Mr(g) {
    nc = (int)V.size();
    for (int j = 0; j < nc; ++j) {
      LatticeFermionD t(g);
      Dw.M(V[j], t);
      AV.push_back(t);
    }
    Eigen::MatrixXcd Ac(nc, nc);
    for (int i = 0; i < nc; ++i) {
      for (int j = 0; j < nc; ++j) {
        ComplexD o = innerProduct(V[i], AV[j]);
        Ac(i, j) = std::complex<double>(real(o), imag(o));
      }
    }
    Acinv = Ac.inverse();
  }
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Eigen::VectorXcd vr(nc);
    for (int i = 0; i < nc; ++i) {
      ComplexD o = innerProduct(V[i], in);
      vr(i) = std::complex<double>(real(o), imag(o));
    }
    Eigen::VectorXcd y = Acinv * vr;       // A_c^{-1} V^dag r
    xc = Zero();
    for (int i = 0; i < nc; ++i) {
      ComplexD c(y(i).real(), y(i).imag());
      axpy(xc, c, V[i], xc);               // x_c = V y
    }
    Dw.M(xc, r2);                          // D_W x_c
    r2 = in - r2;                          // r - D_W x_c
    M(r2, Mr);                             // base smooth
    out = xc + Mr;
  }
};
// Run right-preconditioned FGMRES on D_W with preconditioner `prec`, return the outer iteration count.
static int run_fgmres(NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD>& LinOp,
                      LinearFunction<LatticeFermionD>& prec, const LatticeFermionD& b,
                      RealD tol, int maxit, int restart) {
  FlexibleGeneralisedMinimalResidual<LatticeFermionD> F(tol, maxit, prec, restart, false);
  LatticeFermionD x(b.Grid());
  x = Zero();
  F(LinOp, b, x);
  return F.IterationCount;
}
// Scan one operator: extract the single most-extreme eigenpair at BOTH ends (LOW = smallest modulus via IRA,
// HIGH = largest modulus via IRA), and on that eigenvector psi evaluate the preconditioner quality
// C = ||(1 - M D_W) psi|| / ||psi|| (= |1 - mu| when psi is an eigenmode of M D_W) for BOTH M0 and M1.
// C>=1 = red flag (that mode amplifies under stationary iteration). `resid` = ||Op psi - eval psi||/||psi||
// (IRA convergence quality of the reported eigenpair).
static void opscan_one(const char* name,
                       LinearFunction<LatticeFermionD>& Op,
                       LinearFunction<LatticeFermionD>& m0op,
                       LinearFunction<LatticeFermionD>& m1op,
                       const LatticeFermionD& src, int Nstop, int Nk, int Nm, int maxit) {
  GridBase* g = src.Grid();
  for (int end = 0; end < 2; ++end) {
    IRAsortCriterion crit = (end == 0) ? IRAsmallestModulus : IRAlargestModulus;
    const char* endname = (end == 0) ? "LOW " : "HIGH";
    std::vector<ComplexD> eval;
    std::vector<LatticeFermionD> evec(Nm + 1, LatticeFermionD(g));
    int Nconv = 0;
    ImplicitlyRestartedArnoldi<LatticeFermionD> ira(Op, Nstop, Nk, Nm, 1.0e-6, maxit, crit);
    ira.calc(eval, evec, src, Nconv);
    if (Nconv <= 0) {
      std::cout << GridLogMessage << "  OPSCAN " << name << " " << endname << ": no converged mode"
                << std::endl;
      continue;
    }
    LatticeFermionD psi(g);
    psi = evec[0];
    RealD npsi = std::sqrt(norm2(psi));
    LatticeFermionD MDpsi(g), r(g), Opsi(g);
    m0op(psi, MDpsi);
    r = psi - MDpsi;
    RealD C0 = std::sqrt(norm2(r)) / npsi;
    m1op(psi, MDpsi);
    r = psi - MDpsi;
    RealD C1 = std::sqrt(norm2(r)) / npsi;
    Op(psi, Opsi);
    axpy(r, -eval[0], psi, Opsi);              // Op psi - eval psi
    RealD resid = std::sqrt(norm2(r)) / npsi;
    double re = real(eval[0]);
    double im = imag(eval[0]);
    double mod = std::sqrt(re * re + im * im);
    std::cout << GridLogMessage << "  OPSCAN " << name << " " << endname << ": eval=(" << re << "," << im
              << ")  |eval|=" << mod << "  (resid " << resid << ")   C_M0=" << C0 << (C0 >= 1.0 ? "(RED)" : "")
              << "   C_M1=" << C1 << (C1 >= 1.0 ? "(RED)" : "") << std::endl;
  }
}
// Scan a preconditioned operator M D_W at BOTH ends and report its own eigenvalue mu + quality C = |1 - mu|
// (= ||(1 - M D_W) psi||/||psi|| on an eigenmode). C>=1 (Re mu<0, or a large Im mu) = red flag. Used by --mx
// to compare M0/M1/Mx_inner/Mx_outer spectra.
static void opscan_mx(const char* name, LinearFunction<LatticeFermionD>& MxDw,
                      const LatticeFermionD& src, int Nstop, int Nk, int Nm, int maxit) {
  GridBase* g = src.Grid();
  for (int end = 0; end < 2; ++end) {
    IRAsortCriterion crit = (end == 0) ? IRAsmallestModulus : IRAlargestModulus;
    const char* endname = (end == 0) ? "LOW " : "HIGH";
    std::vector<ComplexD> eval;
    std::vector<LatticeFermionD> evec(Nm + 1, LatticeFermionD(g));
    int Nconv = 0;
    ImplicitlyRestartedArnoldi<LatticeFermionD> ira(MxDw, Nstop, Nk, Nm, 1.0e-6, maxit, crit);
    ira.calc(eval, evec, src, Nconv);
    if (Nconv <= 0) {
      std::cout << GridLogMessage << "  MXSCAN " << name << " " << endname << ": no converged mode"
                << std::endl;
      continue;
    }
    double re = real(eval[0]);
    double im = imag(eval[0]);
    double C = std::sqrt((1.0 - re) * (1.0 - re) + im * im);
    std::cout << GridLogMessage << "  MXSCAN " << name << " " << endname << ": mu=(" << re << "," << im
              << ")  |mu|=" << std::sqrt(re * re + im * im) << "   C=|1-mu|=" << C << (C >= 1.0 ? "(RED)" : "")
              << std::endl;
  }
}

// ================= DIRECT FRAME OPTIMIZER (ported from dwf4 dwf4_frameopt_claude.h) =================
// Minimize L[Omega] = sum_v ||M0(Omega) D_W v - v||^2, M0 = Omega^dag F Omega, by gradient descent on the
// frame Omega in SU(3) (per config; warm-started at the Landau frame). Analytic gradient, FD-gated. Gradient:
//   a=Om w; b=F a; c=Om^dag b; r=c-v (w=D_W v); s = Om^dag g5 F g5 Om r  (F^dag = g5 F g5, Wilson g5-herm);
//   G(x) = Ta( sum_v [ traceSpin(outer(w,s)) - traceSpin(outer(c,r)) ] ). Descent: Om <- Om exp(+eta G)
// (dL/dt|_{X=G} = -2||G||^2 < 0; validated by the dwf4 8^4 FD gate). See grid_frame_optimizer_impl_plan_claude.md.

template <class Ffree>
static RealD fo_loss(Ffree& F, const LatticeColourMatrixD& Om,
                     const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Omdag = adj(Om);
  LatticeFermionD a(g), b(g), c(g), r(g);
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om * w[p];
    F(a, b);
    c = Omdag * b;
    r = c - v[p];
    L += norm2(r);
  }
  return L;
}

template <class Ffree>
static RealD fo_loss_force(Ffree& F, const LatticeColourMatrixD& Om,
                           const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                           LatticeColourMatrixD& Gf) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Omdag = adj(Om);
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermionD a(g), b(g), c(g), r(g), t(g), fg(g), s(g);
  LatticeColourMatrixD M(g);
  M = Zero();
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om * w[p];
    F(a, b);
    c = Omdag * b;
    r = c - v[p];
    L += norm2(r);
    // s = Om^dag g5 F g5 Om r
    t = Om * r;
    t = g5 * t;
    F(t, fg);
    fg = g5 * fg;
    s = Omdag * fg;
    // M += traceSpin(outer(w,s)) - traceSpin(outer(c,r))  (spin-summed colour outer products)
    M = M + traceSpin(outerProduct(w[p], s)) - traceSpin(outerProduct(c, r));
  }
  Gf = Ta(M);
  return L;
}

// FINITE-DIFFERENCE gate: random su(N) X; central-diff of L along Om exp(+-eps X) vs analytic 2 sum Re tr[X G].
template <class Ffree>
static RealD fo_grad_check(Ffree& F, const LatticeColourMatrixD& Om,
                           const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                           GridParallelRNG& pRNG, RealD eps, RealD& fd, RealD& an) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Gf(g);
  fo_loss_force(F, Om, w, v, Gf);
  LatticeColourMatrixD Mr(g);
  gaussian(pRNG, Mr);
  LatticeColourMatrixD X = Ta(Mr);
  ComplexD tr = TensorRemove(sum(trace(X * Gf)));
  an = 2.0 * real(tr);
  LatticeColourMatrixD Op = Om * expMat(X, eps, 12);
  LatticeColourMatrixD Omn = Om * expMat(X, -eps, 12);
  RealD Lp = fo_loss(F, Op, w, v);
  RealD Lm = fo_loss(F, Omn, w, v);
  fd = (Lp - Lm) / (2.0 * eps);
  return std::abs(fd - an) / (std::abs(an) + 1.0e-30);
}

// backtracking-line-search gradient descent. Updates Om in place; returns final L.
template <class Ffree>
static RealD fo_descend(Ffree& F, LatticeColourMatrixD& Om,
                        const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                        int niter, RealD eta0, int maxls) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Gf(g);
  RealD L = fo_loss_force(F, Om, w, v, Gf);
  RealD L0 = L;
  RealD eta = eta0;
  std::cout << GridLogMessage << "# it     L                  ||G||^2           eta" << std::endl;
  for (int it = 0; it < niter; ++it) {
    RealD gn2 = norm2(Gf);
    std::cout << GridLogMessage << "  " << it << "   " << L << "   " << gn2 << "   " << eta << std::endl;
    if (gn2 < 1.0e-22) {
      break;
    }
    bool acc = false;
    for (int ls = 0; ls < maxls; ++ls) {
      LatticeColourMatrixD Otry = Om * expMat(Gf, eta, 12);
      RealD Lt = fo_loss(F, Otry, w, v);
      if (Lt < L) {
        Om = Otry;
        L = Lt;
        eta *= 1.3;
        acc = true;
        break;
      }
      eta *= 0.5;
    }
    if (!acc) {
      std::cout << GridLogMessage << "  (line search stuck at it=" << it << "; floor)" << std::endl;
      break;
    }
    L = fo_loss_force(F, Om, w, v, Gf);
  }
  std::cout << GridLogMessage << "fo_descend: L " << L0 << " -> " << L << "  ("
            << ((L > 0.0) ? L0 / L : 0.0) << "x lower)" << std::endl;
  return L;
}

// JOINT (Omega, theta) descent: each iteration takes ONE backtracked group step on Omega (analytic
// gradient, at the current theta) AND ONE backtracked finite-difference step on the constant holonomy
// twist theta (8 free params for SU(3): theta[0],theta[1] per direction, theta[2] = -(theta0+theta1)
// enforcing tracelessness). The twisted kernel Ft is rebuilt (setTheta) at each loss eval. This is the
// direct test of whether adding the holonomy to the optimised variables lowers the frame residual
// L = sum_v ||Omega^dag F_theta Omega D_W v - v||^2. (holonomy_frameopt_impl_plan_claude.md)
static RealD fo_descend_joint(FreeWilsonTwistedInverse<WilsonImplD>& Ft, LatticeColourMatrixD& Om,
                              std::vector<std::array<double, Nd>>& theta,
                              const std::vector<LatticeFermionD>& w,
                              const std::vector<LatticeFermionD>& v,
                              int niter, RealD eta0, RealD theta_eta0, int maxls) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Gf(g);
  RealD L = fo_loss_force(Ft, Om, w, v, Gf);
  RealD L0 = L;
  RealD eta = eta0;
  RealD teta = theta_eta0;
  const RealD dth = 1.0e-3;  // FD step for the theta gradient
  std::cout << GridLogMessage << "# joint it     L                  ||G_Om||^2   |theta|" << std::endl;
  for (int it = 0; it < niter; ++it) {
    double tnorm = 0.0;
    for (int c = 0; c < Nc; ++c) {
      for (int mu = 0; mu < Nd; ++mu) {
        tnorm += theta[c][mu] * theta[c][mu];
      }
    }
    std::cout << GridLogMessage << "  " << it << "   " << L << "   " << norm2(Gf) << "   "
              << std::sqrt(tnorm) << std::endl;

    // ---- Omega step (fixed theta) ----
    for (int ls = 0; ls < maxls; ++ls) {
      LatticeColourMatrixD Otry = Om * expMat(Gf, eta, 12);
      RealD Lt = fo_loss(Ft, Otry, w, v);
      if (Lt < L) {
        Om = Otry;
        L = Lt;
        eta *= 1.3;
        break;
      }
      eta *= 0.5;
    }

    // ---- theta step (fixed Omega, FD gradient over the 8 free params) ----
    double grad[2][Nd];
    for (int c = 0; c < 2; ++c) {
      for (int mu = 0; mu < Nd; ++mu) {
        theta[c][mu] += dth;
        theta[2][mu] -= dth;
        Ft.setTheta(theta);
        RealD Lp = fo_loss(Ft, Om, w, v);
        theta[c][mu] -= 2.0 * dth;
        theta[2][mu] += 2.0 * dth;
        Ft.setTheta(theta);
        RealD Lm = fo_loss(Ft, Om, w, v);
        theta[c][mu] += dth;
        theta[2][mu] -= dth;
        grad[c][mu] = (Lp - Lm) / (2.0 * dth);
      }
    }
    // backtracking step on theta
    for (int ls = 0; ls < maxls; ++ls) {
      std::vector<std::array<double, Nd>> th_try = theta;
      for (int mu = 0; mu < Nd; ++mu) {
        th_try[0][mu] -= teta * grad[0][mu];
        th_try[1][mu] -= teta * grad[1][mu];
        th_try[2][mu] = -(th_try[0][mu] + th_try[1][mu]);
      }
      Ft.setTheta(th_try);
      RealD Lt = fo_loss(Ft, Om, w, v);
      if (Lt < L) {
        theta = th_try;
        L = Lt;
        teta *= 1.3;
        break;
      }
      teta *= 0.5;
      if (ls == maxls - 1) {
        Ft.setTheta(theta);  // restore
      }
    }

    // recompute the Omega gradient at the updated theta for the next iteration
    Ft.setTheta(theta);
    L = fo_loss_force(Ft, Om, w, v, Gf);
  }
  std::cout << GridLogMessage << "fo_descend_joint: L " << L0 << " -> " << L << "  ("
            << ((L > 0.0) ? L0 / L : 0.0) << "x lower)" << std::endl;
  return L;
}

// M0 with the holonomy-twisted free kernel: out = Omega^dag F_theta Omega in (for the FGMRES win test).
struct M0TwistW : public LinearFunction<LatticeFermionD> {
  FreeWilsonTwistedInverse<WilsonImplD>& F;
  LatticeColourMatrixD Om;
  LatticeColourMatrixD Omdag;
  M0TwistW(FreeWilsonTwistedInverse<WilsonImplD>& F_, const LatticeColourMatrixD& Om_)
    : F(F_), Om(Om_), Omdag(adj(Om_)) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    LatticeFermionD a(in.Grid());
    LatticeFermionD b(in.Grid());
    a = Om * in;
    F(a, b);
    out = Omdag * b;
  }
};

// one mass: build the mass-dependent operator + M0_W and solve. Omega (xform) is passed in (built ONCE
// per config in main, mass-independent -> reused every mass; the flow+Landau is the expensive part).
static void solve_wilson(LatticeGaugeFieldD& U, GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                         const LatticeColourMatrixD& xform, double mass, double mprec,
                         GridParallelRNG& RNG, int restart, int repeat) {
  // F_W built at m_prec (decoupled from the operator mass). The m_prec scan showed a Tikhonov optimum
  // (~0.8), roughly INDEPENDENT of the operator's physical mass -- so fix m_prec near the optimum and
  // scan the operator mass toward m_crit for the crossover.
  std::cout << GridLogMessage << "==== Wilson headline solve (m_bare=" << mass
            << ", m_prec=" << mprec << ", restart=" << restart << ") ====" << std::endl;
  // ---- AP-time interacting Wilson operator + framed free-Wilson preconditioner (Omega REUSED) ----
  std::vector<Complex> boundary(Nd, Complex(1.0, 0.0));
  boundary[Nd - 1] = Complex(-1.0, 0.0);  // anti-periodic time
  WilsonImplD::ImplParams params(boundary);
  WilsonFermionD Dw(U, *UGrid, *UrbGrid, mass, params);
  FreeWilsonInverse<WilsonImplD> Fw(UGrid, mprec, boundary);
  FreeLimitPreconditionerW<WilsonImplD> M0(Fw, xform, UGrid);
  // M1_W (next-order) preconditioner: needs D_W on the framed config U^L = Omega U Omega^dag (at m_prec).
  LatticeGaugeFieldD Uframed(UGrid);
  Uframed = U;
  LatticeColourMatrixD gtr(UGrid);
  gtr = xform;
  SU<Nc>::GaugeTransform<PeriodicGimplD>(Uframed, gtr);
  WilsonFermionD Dframed(Uframed, *UGrid, *UrbGrid, mprec, params);
  FreeLimitPreconditionerW1<WilsonImplD> M1(Fw, xform, Dframed, UGrid);

  LatticeFermionD bsrc(UGrid);
  gaussian(RNG, bsrc);
  RealD nv = std::sqrt(norm2(bsrc));

  // residual proxy ||M0_W D_W[U] v - v|| / ||v|| (the shared yardstick: how good an approx inverse is M0)
  LatticeFermionD Dv(UGrid);
  Dw.M(bsrc, Dv);
  LatticeFermionD M0Dv(UGrid);
  M0(Dv, M0Dv);
  RealD proxy = std::sqrt(norm2(M0Dv - bsrc)) / nv;
  std::cout << GridLogMessage << "  residual proxy ||M0_W D_W v - v||/||v|| = " << proxy << std::endl;

  RealD tol = 1.0e-8;
  int maxit = 20000;

  // RB-CGNE honest baseline
  ConjugateGradient<LatticeFermionD> CGrb(tol, maxit, false);
  SchurRedBlackDiagMooeeSolve<LatticeFermionD> Schur(CGrb);
  LatticeFermionD xrb(UGrid);
  // repeat the solve, take the MIN wall (discards the GPU warmup / JIT of the first call). Quiet-GPU timing.
  double tw_rb = 1.0e30;
  int rb_iters = 0;
  for (int r = 0; r < repeat; ++r) {
    xrb = Zero();
    double t = -usecond();
    Schur(Dw, bsrc, xrb);
    t += usecond();
    if (t < tw_rb) {
      tw_rb = t;
    }
    rb_iters = CGrb.IterationsToComplete;
  }
  std::cout << GridLogMessage << "  RB-CGNE(Wilson): iters=" << rb_iters
            << "  WALL=" << tw_rb / 1.0e6 << " s  (min of " << repeat << ")   [honest baseline]" << std::endl;

  // vanilla FULL CGNE (CG on the full MdagM, no even-odd) -- the SOFT baseline that many use unpreconditioned.
  MdagMLinearOperator<WilsonFermionD, LatticeFermionD> HermOp(Dw);
  LatticeFermionD bn(UGrid);
  Dw.Mdag(bsrc, bn);
  LatticeFermionD xcg(UGrid);
  xcg = Zero();
  ConjugateGradient<LatticeFermionD> CGf(tol, maxit, false);
  double tw_cgf = -usecond();
  CGf(HermOp, bn, xcg);
  tw_cgf += usecond();
  int cgf_iters = CGf.IterationsToComplete;
  std::cout << GridLogMessage << "  CGNE(full): iters=" << cgf_iters
            << "  WALL=" << tw_cgf / 1.0e6 << " s   [vanilla/soft baseline]" << std::endl;

  // FGMRES right-preconditioned by M0_W
  NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dw);
  FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(tol, maxit, M0, restart, false);
  LatticeFermionD xg(UGrid);
  double tw_fg = 1.0e30;
  int fg_iters = 0;
  for (int r = 0; r < repeat; ++r) {
    xg = Zero();
    double t = -usecond();
    FGMRES(LinOp, bsrc, xg);
    t += usecond();
    if (t < tw_fg) {
      tw_fg = t;
    }
    fg_iters = FGMRES.IterationCount;
  }
  std::cout << GridLogMessage << "  FGMRES(M0_W) restart=" << restart << ": iters=" << fg_iters
            << "  WALL=" << tw_fg / 1.0e6 << " s  (min of " << repeat << ")" << std::endl;

  // FGMRES right-preconditioned by M1_W (next-order). M1 costs +1 interacting D_W[U^L] per apply, so the
  // honest metric is BOTH outer iters (does M1 cut them vs M0) and the added D_W[U^L] applies.
  FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES1(tol, maxit, M1, restart, false);
  LatticeFermionD xg1(UGrid);
  double tw_fg1 = 1.0e30;
  int fg1_iters = 0;
  long m1_ndw = 0;
  for (int r = 0; r < repeat; ++r) {
    xg1 = Zero();
    M1.n_dw = 0;
    double t = -usecond();
    FGMRES1(LinOp, bsrc, xg1);
    t += usecond();
    if (t < tw_fg1) {
      tw_fg1 = t;
    }
    fg1_iters = FGMRES1.IterationCount;
    m1_ndw = M1.n_dw;
  }
  std::cout << GridLogMessage << "  FGMRES(M1_W) restart=" << restart << ": iters=" << fg1_iters
            << "  WALL=" << tw_fg1 / 1.0e6 << " s  (min of " << repeat << ")  [+" << m1_ndw
            << " D_W[U^L] applies]" << std::endl;
  std::cout << GridLogMessage << "  N_it(M0)/N_it(M1) = " << ((fg1_iters > 0) ? (double)fg_iters / fg1_iters : 0.0)
            << "   (>1 = M1 cuts outer iterations)" << std::endl;
  if (tw_fg > 0.0) {
    std::cout << GridLogMessage << "  WALL speedup (RB-CGNE / FGMRES-M0_W)  = " << (tw_rb / tw_fg)
              << "x  [>1 = free-prec beats RB-CGNE (hard bar)]" << std::endl;
    std::cout << GridLogMessage << "  WALL speedup (CGNE-full / FGMRES-M0_W) = " << (tw_cgf / tw_fg)
              << "x  [>1 = free-prec beats VANILLA CGNE (soft bar)]" << std::endl;
  }
}

// PRECONDITIONER-MASS SCAN: fixed operator (m_bare), sweep the free-prec mass m_prec to find the empirical
// optimum (may prefer a slightly HEAVIER m_prec than the naive m - m_crit -- Tikhonov/mass-floor: a
// too-light F_W over-amplifies near-zero modes). RB-CGNE computed ONCE (operator fixed).
static void scan_mprec(LatticeGaugeFieldD& U, GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                       const LatticeColourMatrixD& xform, double mass, const std::vector<double>& mprec_list,
                       GridParallelRNG& RNG, int restart) {
  std::cout << GridLogMessage << "==== preconditioner-mass scan (operator m_bare=" << mass
            << ", restart=" << restart << ") ====" << std::endl;
  std::vector<Complex> boundary(Nd, Complex(1.0, 0.0));
  boundary[Nd - 1] = Complex(-1.0, 0.0);
  WilsonImplD::ImplParams params(boundary);
  WilsonFermionD Dw(U, *UGrid, *UrbGrid, mass, params);
  LatticeFermionD bsrc(UGrid);
  gaussian(RNG, bsrc);
  RealD nv = std::sqrt(norm2(bsrc));
  RealD tol = 1.0e-8;
  int maxit = 20000;

  ConjugateGradient<LatticeFermionD> CGrb(tol, maxit, false);
  SchurRedBlackDiagMooeeSolve<LatticeFermionD> Schur(CGrb);
  LatticeFermionD xrb(UGrid);
  xrb = Zero();
  double tw_rb = -usecond();
  Schur(Dw, bsrc, xrb);
  tw_rb += usecond();
  int rb_iters = CGrb.IterationsToComplete;
  std::cout << GridLogMessage << "  RB-CGNE(Wilson): iters=" << rb_iters << "  WALL=" << tw_rb / 1.0e6
            << " s   [baseline, fixed operator]" << std::endl;

  NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dw);
  for (size_t i = 0; i < mprec_list.size(); ++i) {
    double mp = mprec_list[i];
    FreeWilsonInverse<WilsonImplD> Fw(UGrid, mp, boundary);
    FreeLimitPreconditionerW<WilsonImplD> M0(Fw, xform, UGrid);
    LatticeFermionD Dv(UGrid);
    Dw.M(bsrc, Dv);
    LatticeFermionD M0Dv(UGrid);
    M0(Dv, M0Dv);
    RealD proxy = std::sqrt(norm2(M0Dv - bsrc)) / nv;
    FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(tol, maxit, M0, restart, false);
    LatticeFermionD xg(UGrid);
    xg = Zero();
    double tw_fg = -usecond();
    FGMRES(LinOp, bsrc, xg);
    tw_fg += usecond();
    int fg = FGMRES.IterationCount;
    std::cout << GridLogMessage << "  m_prec=" << mp << ": FGMRES(M0_W) iters=" << fg
              << "  WALL=" << tw_fg / 1.0e6 << " s  proxy=" << proxy
              << "  count-win(RB/FG)=" << (double)rb_iters / (double)fg
              << "  wall-win=" << tw_rb / tw_fg << std::endl;
  }
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG RNG(UGrid);
  RNG.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  double mass = 0.1;
  if (GridCmdOptionExists(argv, argv + argc, "--mass")) {
    mass = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass"));
  }
  int restart = 20;  // FGMRES restart length default (Nobu 2026-09-07); --restart overrides
  if (GridCmdOptionExists(argv, argv + argc, "--restart")) {
    restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--restart"));
  }
  int repeat = 1;   // --repeat N: run each solve N times, report the MIN wall (quiet-GPU timing, drop warmup)
  if (GridCmdOptionExists(argv, argv + argc, "--repeat")) {
    repeat = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--repeat"));
    if (repeat < 1) {
      repeat = 1;
    }
  }
  // frame-flow knobs (defect_scan): the frame Omega = WilsonFlow(flow_eps x flow_nstep) then Landau. For a
  // defect-flowed checkpoint pass --flow-nstep 0 (NO reflow -- the checkpoint IS the frame source). Default
  // 0.02 x 100 = tau 2 (the original hardcoded frame flow for raw configs).
  double flow_eps = 0.02;
  int flow_nstep = 100;
  if (GridCmdOptionExists(argv, argv + argc, "--flow-eps")) {
    flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow-eps"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--flow-nstep")) {
    flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow-nstep"));
  }
  // OPERATOR reflow (defect_scan): standard Wilson flow applied to the LOADED config BEFORE building the
  // operator AND the frame. Use it to (a) conventionally flow a raw config to match the defect checkpoints'
  // smoothness (--op-reflow-nstep 232 = 4 t0, the missing smooth Q!=0 reference), and (b) heal the open-
  // boundary seam of a defect checkpoint with a short reflow (--op-reflow-nstep 58 = +1 t0). 0 = as-is.
  double op_reflow_eps = 0.02;
  int op_reflow_nstep = 0;
  if (GridCmdOptionExists(argv, argv + argc, "--op-reflow-eps")) {
    op_reflow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--op-reflow-eps"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--op-reflow-nstep")) {
    op_reflow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--op-reflow-nstep"));
  }

  // ---- Chunk 0 cold gate: unit gauge, periodic ----
  LatticeGaugeFieldD Uunit(UGrid);
  Uunit = 1.0;
  std::vector<Complex> per(Nd, Complex(1.0, 0.0));
  WilsonImplD::ImplParams pparams(per);
  WilsonFermionD Dwf(Uunit, *UGrid, *UrbGrid, mass, pparams);
  FreeWilsonInverse<WilsonImplD> Fwf(UGrid, mass, per);
  LatticeFermionD v(UGrid);
  gaussian(RNG, v);
  RealD nv = std::sqrt(norm2(v));
  LatticeFermionD Dv(UGrid);
  Dwf.M(v, Dv);
  LatticeFermionD FDv(UGrid);
  Fwf(Dv, FDv);
  RealD e1 = std::sqrt(norm2(FDv - v)) / nv;
  std::cout << GridLogMessage << "==== Wilson cold gate (unit,periodic,m=" << mass << ") ||F_W D_W v - v||/||v|| = "
            << e1 << "  " << ((e1 < 1.0e-10) ? "PASS" : "FAIL") << std::endl;

  // ---- Chunk 1+2 headline (--config <nersc>) ----
  if (GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
    LatticeGaugeFieldD U(UGrid);
    FieldMetaData header;
    NerscIO::readConfiguration(U, header, cfg);
    std::cout << GridLogMessage << "  loaded " << cfg << std::endl;

    // OPERATOR reflow: standard-flow the loaded config in place -> the OPERATOR + frame both use the reflowed
    // config. Conventional-flow a raw config to match smoothness, or heal a defect checkpoint's seam.
    if (op_reflow_nstep > 0) {
      Real plaq_pre = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
      Real q_pre = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U);
      LatticeGaugeFieldD Ur(UGrid);
      WilsonFlow<PeriodicGimplD> wfop(op_reflow_eps, op_reflow_nstep);
      wfop.smear(Ur, U);
      U = Ur;
      Real plaq_post = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
      Real q_post = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U);
      std::cout << GridLogMessage << "  OP-REFLOW: standard flow tau=" << (op_reflow_eps * op_reflow_nstep)
                << " (nstep=" << op_reflow_nstep << ")  plaq " << plaq_pre << " -> " << plaq_post
                << "   Q_5Li " << q_pre << " -> " << q_post << std::endl;
    }

    // frame Omega: flow + Landau -- built ONCE (mass-independent), REUSED for every mass. --flow-nstep 0 =>
    // NO reflow (Uflowed = U): the defect-flowed checkpoint is the frame source directly.
    Real plaq0 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
    LatticeGaugeFieldD Uflowed(UGrid);
    if (flow_nstep > 0) {
      WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
      wf.smear(Uflowed, U);
    } else {
      Uflowed = U;
    }
    LatticeColourMatrixD xform(UGrid);
    FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
        Uflowed, xform, 0.1 / 16.0, 1000, 1.0e-12, 1.0e-12, true, -1, false);
    Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
    Real Qflow = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);
    std::cout << GridLogMessage << "  frame: plaq=" << plaq0 << "  Landau functional=" << landau
              << "  Q_5Li(frame-flow tau=" << (flow_eps * flow_nstep) << ")=" << Qflow
              << "  (flow_nstep=" << flow_nstep << "; Omega built once, reused per mass)" << std::endl;

    // ---- SPECTRUM (--spectrum): IRA (non-Herm Arnoldi) low eigenvalues of D_W and of M0 D_W (Landau frame).
    // "Get the spectrum of D_W first" -- are there ~|Q| near-zero (IR/topological) modes, and does the free-
    // prec cluster them? Smallest-|lambda| wanted. Nstop/Nk/Nm via --spec-nstop/--spec-nk/--spec-nm. ----
    if (GridCmdOptionExists(argv, argv + argc, "--spectrum")) {
      int Nstop = 16, Nk = 32, Nm = 64, spmaxit = 300;
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nstop")) {
        Nstop = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nstop"));
      }
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nk")) {
        Nk = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nk"));
      }
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nm")) {
        Nm = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nm"));
      }
      double mprec = mass;
      if (GridCmdOptionExists(argv, argv + argc, "--mprec")) {
        mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
      }
      std::vector<Complex> bnd(Nd, Complex(1.0, 0.0));
      bnd[Nd - 1] = Complex(-1.0, 0.0);
      WilsonImplD::ImplParams psp(bnd);
      WilsonFermionD Dwsp(U, *UGrid, *UrbGrid, mass, psp);
      LatticeFermionD src(UGrid);
      gaussian(RNG, src);
      std::vector<ComplexD> eval;
      std::vector<LatticeFermionD> evec(Nm + 1, LatticeFermionD(UGrid));
      int Nconv = 0;

      std::cout << GridLogMessage << "==== SPECTRUM  m=" << mass << " mprec=" << mprec
                << "  Nstop=" << Nstop << " Nk=" << Nk << " Nm=" << Nm << " ====" << std::endl;
      // (1) D_W low spectrum. Two subspace generators feed the SAME RR-cleanup below (which uses Dwsp.M,
      // so the recovered lambda/chi/resid are exact for D_W regardless of generator):
      //   default : plain IRA on D_W, smallest-|lambda| wanted (now with the Chunk-A Leja + reorthog fixes).
      //   --faber : IRA on the Chebyshev-ellipse FILTER p_n(D_W) (largest-modulus wanted); the ellipse
      //             encloses the bulk so the near-zero modes become peripheral = Arnoldi's stable case.
      DwLinOp dwop(Dwsp);
      if (GridCmdOptionExists(argv, argv + argc, "--shiftinvert")) {
        double sitol = 1.0e-8;
        int simaxit = 5000;
        if (GridCmdOptionExists(argv, argv + argc, "--si-tol")) {
          sitol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--si-tol"));
        }
        std::cout << GridLogMessage
                  << "-- D_W low spectrum (SHIFT-INVERT IRA: CGNE D_W^{-1}, largest-modulus, sitol=" << sitol
                  << ") --" << std::endl;
        ShiftInvertZeroOp si(Dwsp, UGrid, sitol, simaxit);
        ImplicitlyRestartedArnoldi<LatticeFermionD> iraSI(si, Nstop, Nk, Nm, 1.0e-6, spmaxit,
                                                          IRAlargestModulus);
        iraSI.calc(eval, evec, src, Nconv);
        std::cout << GridLogMessage << "  D_W (shift-invert): Nconv=" << Nconv << std::endl;
      } else if (GridCmdOptionExists(argv, argv + argc, "--faber")) {
        int ford = 12;
        if (GridCmdOptionExists(argv, argv + argc, "--faber-ord")) {
          ford = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--faber-ord"));
        }
        // Estimate the spectrum (Nm Ritz values of a single Nm-step Arnoldi) to fit the bulk ellipse.
        ImplicitlyRestartedArnoldi<LatticeFermionD> iraEst(dwop, Nstop, Nk, Nm, 1.0e-6, spmaxit);
        std::vector<ComplexD> allr;
        iraEst.ritz_estimate(src, allr);
        // Separate WANTED (smallest |lambda|) from BULK. Optional |lambda| cutoff via --faber-lo.
        double locut = -1.0;
        if (GridCmdOptionExists(argv, argv + argc, "--faber-lo")) {
          locut = std::stod(GridCmdOptionPayload(argv, argv + argc, "--faber-lo"));
        }
        // NB: std::abs(ComplexD) is unavailable under the CUDA build (ComplexD = thrust::complex); use the
        // explicit modulus via real()/imag() (the confirmed Grid free functions).
        std::vector<std::pair<double, int> > om(allr.size());
        for (size_t i = 0; i < allr.size(); ++i) {
          double m = std::sqrt(real(allr[i]) * real(allr[i]) + imag(allr[i]) * imag(allr[i]));
          om[i] = std::make_pair(m, (int)i);
        }
        std::sort(om.begin(), om.end());
        double remin = 1.0e300, remax = -1.0e300, immax = 0.0;
        int nb = 0;
        for (size_t s = 0; s < allr.size(); ++s) {
          int i = om[s].second;
          double mi = std::sqrt(real(allr[i]) * real(allr[i]) + imag(allr[i]) * imag(allr[i]));
          bool bulk = (locut > 0.0) ? (mi > locut) : ((int)s >= Nstop);
          if (!bulk) {
            continue;
          }
          double re = real(allr[i]);
          double im = std::abs(imag(allr[i]));
          remin = std::min(remin, re);
          remax = std::max(remax, re);
          immax = std::max(immax, im);
          nb++;
        }
        double cc = 0.5 * (remin + remax);
        double aa = 0.5 * (remax - remin);
        double bb = immax;
        std::complex<double> dstd = std::sqrt(std::complex<double>(aa * aa - bb * bb, 0.0));
        double floor = 1.0e-6 * std::max(1.0, std::max(aa, bb));
        if (std::abs(dstd) < floor) {
          dstd = std::complex<double>(std::max(aa, floor), 0.0);
        }
        ComplexD cf(cc, 0.0);
        ComplexD df(dstd.real(), dstd.imag());
        if (GridCmdOptionExists(argv, argv + argc, "--faber-c")) {
          cf = ComplexD(std::stod(GridCmdOptionPayload(argv, argv + argc, "--faber-c")), 0.0);
        }
        if (GridCmdOptionExists(argv, argv + argc, "--faber-d")) {
          df = ComplexD(std::stod(GridCmdOptionPayload(argv, argv + argc, "--faber-d")), 0.0);
        }
        std::cout << GridLogMessage << "-- D_W low spectrum (FABER: IRA on T_" << ford
                  << "((D_W-c)/d), largest-modulus wanted) --" << std::endl;
        std::cout << GridLogMessage << "   bulk fit (" << nb << " Ritz): Re in [" << remin << "," << remax
                  << "]  max|Im|=" << immax << "  -> c=(" << real(cf) << "," << imag(cf) << ")  d=("
                  << real(df) << "," << imag(df) << ")" << std::endl;
        ChebyshevEllipse<LatticeFermionD> cheb(dwop, cf, df, ford);
        ImplicitlyRestartedArnoldi<LatticeFermionD> iraF(cheb, Nstop, Nk, Nm, 1.0e-6, spmaxit,
                                                         IRAlargestModulus);
        iraF.calc(eval, evec, src, Nconv);
        std::cout << GridLogMessage << "  D_W (Faber): Nconv=" << Nconv << std::endl;
      } else {
        std::cout << GridLogMessage << "-- D_W low spectrum (IRA, smallest |lambda|) --" << std::endl;
        ImplicitlyRestartedArnoldi<LatticeFermionD> iraD(dwop, Nstop, Nk, Nm, 1.0e-6, spmaxit);
        iraD.calc(eval, evec, src, Nconv);
        std::cout << GridLogMessage << "  D_W: Nconv=" << Nconv << std::endl;
      }
      // FIX 1 -- RAYLEIGH-RITZ CLEANUP of the returned subspace: the IRA restart accumulates factorization
      // error (internal estimate optimistic vs the true residual), so extract the BEST eigenpairs from the
      // returned span {evec[0..Nconv-1]} directly: GS-orthonormalize -> Hs = B^dag D_W B (Nconv x Nconv) ->
      // small dense eigensolve -> refined Ritz vectors. Report refined lambda, CHIRALITY chi = Re<psi|g5|psi>
      // /<psi|psi> (|chi|~1 = topological chiral zero mode tied to Q; ~0 = non-chiral bulk), and the TRUE
      // residual ||D psi - lambda psi||/||psi||. Robust to the IRA restart bug (best-of-subspace + honest resid).
      // The refined modes are STORED (rr_mode/rr_lam/rr_chi) so we can then apply M0 D_W to each (below).
      std::vector<LatticeFermionD> rr_mode;
      std::vector<ComplexD> rr_lam;
      std::vector<RealD> rr_chi;
      if (Nconv > 0) {
        int m = Nconv;
        std::vector<LatticeFermionD> B;
        for (int i = 0; i < m; ++i) {
          B.push_back(evec[i]);
        }
        // modified Gram-Schmidt orthonormalize
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < i; ++j) {
            ComplexD o = innerProduct(B[j], B[i]);
            axpy(B[i], -o, B[j], B[i]);
          }
          RealD nn = std::sqrt(norm2(B[i]));
          B[i] = B[i] * (1.0 / nn);
        }
        std::vector<LatticeFermionD> AB;
        for (int i = 0; i < m; ++i) {
          LatticeFermionD t(UGrid);
          Dwsp.M(B[i], t);
          AB.push_back(t);
        }
        Eigen::MatrixXcd Hs(m, m);
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < m; ++j) {
            ComplexD o = innerProduct(B[i], AB[j]);
            Hs(i, j) = std::complex<double>(real(o), imag(o));
          }
        }
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(Hs, true);
        // sort refined eigenvalues by |lambda| ascending
        std::vector<std::pair<double, int> > ord(m);
        for (int k = 0; k < m; ++k) {
          ord[k] = std::make_pair(std::abs(es.eigenvalues()(k)), k);
        }
        std::sort(ord.begin(), ord.end());
        Gamma g5(Gamma::Algebra::Gamma5);
        std::cout << GridLogMessage << "  D_W Rayleigh-Ritz refined low modes (lambda, chi, true resid):"
                  << std::endl;
        for (int s = 0; s < m; ++s) {
          int k = ord[s].second;
          std::complex<double> lam = es.eigenvalues()(k);
          ComplexD lamg(lam.real(), lam.imag());
          LatticeFermionD x(UGrid);
          x = Zero();
          for (int i = 0; i < m; ++i) {
            std::complex<double> c = es.eigenvectors()(i, k);
            ComplexD cg(c.real(), c.imag());
            axpy(x, cg, B[i], x);
          }
          RealD n2 = norm2(x);
          x = x * (1.0 / std::sqrt(n2));  // normalize the stored mode
          LatticeFermionD Ax(UGrid), rr_(UGrid), g5x(UGrid);
          Dwsp.M(x, Ax);
          axpy(rr_, -lamg, x, Ax);
          RealD rres = std::sqrt(norm2(rr_));
          g5x = g5 * x;
          ComplexD chi = innerProduct(x, g5x);
          std::cout << GridLogMessage << "  D_W RR mode[" << s << "]  lambda=(" << lam.real() << ","
                    << lam.imag() << ")  |lambda|=" << std::abs(lam) << "   chi=" << real(chi)
                    << "   ||r||/||psi||=" << rres << std::endl;
          rr_mode.push_back(x);
          rr_lam.push_back(lamg);
          rr_chi.push_back(real(chi));
        }
      }
      // (2) HOW THE PRECONDITIONER ACTS ON THE D_W EIGENMODES. For each D_W eigenmode phi (D_W phi = lambda
      // phi), apply the framed free-prec M0 to D_W phi and report |M0 D_W phi|/|phi| (=1 for a perfect prec
      // M0 = D_W^{-1}) AND the residual |M0 D_W phi - phi|/|phi|. The 3 chiral (topological) modes should be
      // POORLY handled (far from 1) while the non-chiral bulk should be ~1 -- the direct evidence that the
      // free-prec's blocker is the |Q| chiral modes. M0 = Landau frame (xform).
      FreeWilsonInverse<WilsonImplD> Ffo(UGrid, mprec, bnd);
      FreeLimitPreconditionerW<WilsonImplD> M0L(Ffo, xform, UGrid);
      std::cout << GridLogMessage << "-- M0 (Landau frame) acting on D_W eigenmodes --" << std::endl;
      std::cout << GridLogMessage << "   mode : |lambda|      chi        |M0 DW phi|/|phi|   |M0 DW phi - phi|/|phi|"
                << std::endl;
      {
        LatticeFermionD Dphi(UGrid), M0Dphi(UGrid), diff(UGrid);
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          Dwsp.M(rr_mode[k], Dphi);
          M0L(Dphi, M0Dphi);            // M0 D_W phi
          RealD ratio = std::sqrt(norm2(M0Dphi));            // |phi| = 1 (normalized)
          diff = M0Dphi - rr_mode[k];
          RealD res = std::sqrt(norm2(diff));
          std::cout << GridLogMessage << "  M0act[" << k << "]  |lam|=" << std::sqrt(real(rr_lam[k])*real(rr_lam[k])+imag(rr_lam[k])*imag(rr_lam[k]))
                    << "   chi=" << rr_chi[k] << "   |M0DWphi|/|phi|=" << ratio
                    << "   |M0DWphi-phi|/|phi|=" << res << std::endl;
        }
      }
      // (3) SAME test with M1 = next-order (hopping-expansion) correction. M1 = Omega^dag{F - F D(tildeA) F}Omega
      // with tildeA = U^L - 1, U^L = Omega U Omega^dag (framed config), D(tildeA) = D_W[U^L] - D_W^free. Does
      // adding the leading D_W[U^L] correction rescue the chiral sector M0 fails on? (Brower+Izubuchi.)
      LatticeGaugeFieldD Uframed(UGrid);
      Uframed = U;
      LatticeColourMatrixD gtrans(UGrid);
      gtrans = xform;
      SU<Nc>::GaugeTransform<PeriodicGimplD>(Uframed, gtrans);  // U^L = Omega U Omega^dag
      WilsonImplD::ImplParams p1(bnd);
      WilsonFermionD Dframed(Uframed, *UGrid, *UrbGrid, mprec, p1);
      FreeLimitPreconditionerW1<WilsonImplD> M1L(Ffo, xform, Dframed, UGrid);
      std::cout << GridLogMessage << "-- M1 (Landau frame, next-order) acting on D_W eigenmodes --" << std::endl;
      {
        LatticeFermionD Dphi(UGrid), M1Dphi(UGrid), diff(UGrid);
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          Dwsp.M(rr_mode[k], Dphi);
          M1L(Dphi, M1Dphi);            // M1 D_W phi
          RealD ratio = std::sqrt(norm2(M1Dphi));
          diff = M1Dphi - rr_mode[k];
          RealD res = std::sqrt(norm2(diff));
          std::cout << GridLogMessage << "  M1act[" << k << "]  |lam|=" << std::sqrt(real(rr_lam[k])*real(rr_lam[k])+imag(rr_lam[k])*imag(rr_lam[k]))
                    << "   chi=" << rr_chi[k] << "   |M1DWphi|/|phi|=" << ratio
                    << "   |M1DWphi-phi|/|phi|=" << res << std::endl;
        }
      }
      // (4) SAVE the D_W eigenmodes (important -- for deflation/reuse) as a LIME/Scidac field file + a text
      // metadata file (idx, lambda_re, lambda_im, chi). Guarded by --save-modes <path-prefix>.
      if (GridCmdOptionExists(argv, argv + argc, "--save-modes")) {
#ifdef HAVE_LIME
        std::string pref = GridCmdOptionPayload(argv, argv + argc, "--save-modes");
        std::string fn = pref + ".lime";
        ScidacWriter sw(UGrid->IsBoss());
        sw.open(fn);
        emptyUserRecord rec;
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          sw.writeScidacFieldRecord(rr_mode[k], rec);
        }
        sw.close();
        if (UGrid->IsBoss()) {
          std::ofstream meta((pref + ".meta.txt").c_str());
          meta << "# idx  lambda_re  lambda_im  chi" << std::endl;
          for (size_t k = 0; k < rr_mode.size(); ++k) {
            meta << k << "  " << real(rr_lam[k]) << "  " << imag(rr_lam[k]) << "  " << rr_chi[k] << std::endl;
          }
          meta.close();
        }
        std::cout << GridLogMessage << "  SAVED " << rr_mode.size() << " D_W modes -> " << fn
                  << " (+ .meta.txt)" << std::endl;
#else
        std::cout << GridLogMessage << "  --save-modes requested but this build has no LIME -> skipping"
                  << std::endl;
#endif
      }
      // (4c) SCALE-MIXED Mx (--mx, user 2026-09-05): build INNER & OUTER Mx (the M1 hopping correction band-
      // limited to the low kinetic band |sin p|<Acut) and COMPARE the quality C=||(1-M D_W)phi||/||phi|| for
      // M0 / M1 / Mx_in / Mx_out on the D_W eigenmodes, plus the extreme-mode spectrum (does Mx remove M1's UV
      // red-flag mu=0.865+1.238i?). Knobs --mx-acut (default 1.0), --mx-smooth, --mx-width.
      if (GridCmdOptionExists(argv, argv + argc, "--mx")) {
        double mx_width = 0.3;
        bool mx_smooth = GridCmdOptionExists(argv, argv + argc, "--mx-smooth");
        if (GridCmdOptionExists(argv, argv + argc, "--mx-width")) {
          mx_width = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mx-width"));
        }
        // Acut list: --mx-acut-list "0.5,0.7,1.0,1.5" (scan) OR single --mx-acut (default 1.0).
        std::vector<double> acuts;
        if (GridCmdOptionExists(argv, argv + argc, "--mx-acut-list")) {
          std::string s = GridCmdOptionPayload(argv, argv + argc, "--mx-acut-list");
          std::stringstream ss(s);
          std::string tok;
          while (std::getline(ss, tok, ',')) {
            acuts.push_back(std::stod(tok));
          }
        } else if (GridCmdOptionExists(argv, argv + argc, "--mx-acut")) {
          acuts.push_back(std::stod(GridCmdOptionPayload(argv, argv + argc, "--mx-acut")));
        } else {
          acuts.push_back(1.0);
        }
        WilsonImplD::ImplParams pfree(bnd);
        WilsonFermionD Dfree(Uunit, *UGrid, *UrbGrid, mprec, pfree);   // free (unit-gauge) Wilson at m_prec
        // M0 / M1 reference extremes (Acut-independent), once.
        std::cout << GridLogMessage << "== Mx scan (" << (mx_smooth ? "smooth" : "sharp")
                  << "): M0/M1 reference + Acut sweep ==" << std::endl;
        M0DwLinOp m0op(Dwsp, M0L, UGrid);
        M1DwLinOp m1op(Dwsp, M1L, UGrid);
        opscan_mx("M0 D_W   ", m0op, src, 6, 16, 32, spmaxit);
        opscan_mx("M1 D_W   ", m1op, src, 6, 16, 32, spmaxit);
        for (size_t ia = 0; ia < acuts.size(); ++ia) {
          double mx_acut = acuts[ia];
          Ffo.build_bandmask(mx_acut, mx_smooth, mx_width);
          FreeLimitPreconditionerWx<WilsonImplD> MxIn(Ffo, xform, Dframed, Dfree, UGrid, true);
          FreeLimitPreconditionerWx<WilsonImplD> MxOut(Ffo, xform, Dframed, Dfree, UGrid, false);
          // mean C on chiral vs bulk D_W modes (the IR gain), for MxIn/MxOut vs M0/M1
          RealD cch_m1 = 0.0, cch_i = 0.0, cch_o = 0.0, cbu_i = 0.0, cbu_o = 0.0, cbu_m1 = 0.0;
          int nch = 0, nbu = 0;
          LatticeFermionD Dphi(UGrid), t(UGrid), r(UGrid);
          for (size_t k = 0; k < rr_mode.size(); ++k) {
            Dwsp.M(rr_mode[k], Dphi);
            M1L(Dphi, t);
            r = rr_mode[k] - t;
            RealD c1 = std::sqrt(norm2(r));
            MxIn(Dphi, t);
            r = rr_mode[k] - t;
            RealD ci = std::sqrt(norm2(r));
            MxOut(Dphi, t);
            r = rr_mode[k] - t;
            RealD co = std::sqrt(norm2(r));
            if (std::abs(rr_chi[k]) > 0.5) {
              cch_m1 += c1;
              cch_i += ci;
              cch_o += co;
              nch++;
            } else {
              cbu_m1 += c1;
              cbu_i += ci;
              cbu_o += co;
              nbu++;
            }
          }
          std::cout << GridLogMessage << "-- Acut=" << mx_acut << " : mean C(chiral) M1=" << (cch_m1 / nch)
                    << " MxIn=" << (cch_i / nch) << " MxOut=" << (cch_o / nch) << " | mean C(bulk) M1="
                    << (cbu_m1 / nbu) << " MxIn=" << (cbu_i / nbu) << " MxOut=" << (cbu_o / nbu) << std::endl;
          char nmi[32], nmo[32];
          std::snprintf(nmi, sizeof(nmi), "MxIn(A=%.2f) ", mx_acut);
          std::snprintf(nmo, sizeof(nmo), "MxOut(A=%.2f)", mx_acut);
          MxDwLinOp mxinop(Dwsp, MxIn, UGrid);
          MxDwLinOp mxoutop(Dwsp, MxOut, UGrid);
          opscan_mx(nmi, mxinop, src, 6, 16, 32, spmaxit);
          opscan_mx(nmo, mxoutop, src, 6, 16, 32, spmaxit);
        }
      }
      // (4a) OPERATOR SPECTRUM SCAN (--opscan, user 2026-09-05): extract the LOWEST (smallest modulus) and
      // HIGHEST (largest modulus) eigenpair of D_W, H_W=gamma5 D_W, M0 D_W, M1 D_W, and evaluate the
      // preconditioner quality C=||(1-M D_W)psi||/||psi|| (=|1-mu| on an eigenmode of M D_W) for M0 and M1 on
      // each extreme eigenvector. C>=1 (equivalently Re mu<0 for the M D_W eigenvalues) = red flag: the prec
      // amplifies/flips that mode. Answers "does the free-prec ever flip a mode, or only under-contract?".
      if (GridCmdOptionExists(argv, argv + argc, "--opscan")) {
        int os_ns = 6, os_nk = 16, os_nm = 32;
        DwLinOp dwop(Dwsp);
        HwLinOp hwop(Dwsp, UGrid);
        M0DwLinOp m0op(Dwsp, M0L, UGrid);
        M1DwLinOp m1op(Dwsp, M1L, UGrid);
        std::cout << GridLogMessage
                  << "== OPSCAN: lowest & highest eigenpair of D_W, H_W, M0 D_W, M1 D_W (+ C=|1-M D_W|) =="
                  << std::endl;
        opscan_one("D_W   ", dwop, m0op, m1op, src, os_ns, os_nk, os_nm, spmaxit);
        opscan_one("H_W   ", hwop, m0op, m1op, src, os_ns, os_nk, os_nm, spmaxit);
        opscan_one("M0 D_W", m0op, m0op, m1op, src, os_ns, os_nk, os_nm, spmaxit);
        opscan_one("M1 D_W", m1op, m0op, m1op, src, os_ns, os_nk, os_nm, spmaxit);
      }
      // (4b) SOLVER COUNTS incl. DEFLATION (--solve-deflate, user 2026-09-05): actually invert D_W x = b and
      // count outer iterations for RB-CGNE (baseline), FGMRES(M0), FGMRES(M1), and BOTH deflated by the
      // chiral zero modes (DeflatedPrec: exact solve on span{chiral modes} + base prec). Does deflating the
      // topological blocker sector cut the outer count? Honest note: M1 adds 1 D_W[U^L]/apply, deflation
      // adds 1 D_W/apply. Uses the in-memory chiral modes (|chi|>0.5); M0L/M1L are the Landau-frame precs.
      if (GridCmdOptionExists(argv, argv + argc, "--solve-deflate")) {
        int srst = 20;
        if (GridCmdOptionExists(argv, argv + argc, "--solve-restart")) {
          srst = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--solve-restart"));
        }
        // orthonormalize the chiral modes -> deflation basis V (MGS)
        std::vector<LatticeFermionD> V;
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          if (std::abs(rr_chi[k]) > 0.5) {
            V.push_back(rr_mode[k]);
          }
        }
        for (size_t i = 0; i < V.size(); ++i) {
          for (size_t j = 0; j < i; ++j) {
            ComplexD o = innerProduct(V[j], V[i]);
            axpy(V[i], -o, V[j], V[i]);
          }
          RealD nn = std::sqrt(norm2(V[i]));
          V[i] = V[i] * (1.0 / nn);
        }
        std::cout << GridLogMessage << "== SOLVE-DEFLATE: invert D_W x=b, count outer iters (deflation dim="
                  << V.size() << ", restart=" << srst << ") ==" << std::endl;
        LatticeFermionD b(UGrid);
        gaussian(RNG, b);
        RealD stol = 1.0e-8;
        int smaxit = 20000;
        NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dwsp);

        // RB-CGNE honest baseline
        ConjugateGradient<LatticeFermionD> CGrb(stol, smaxit, false);
        SchurRedBlackDiagMooeeSolve<LatticeFermionD> Schur(CGrb);
        LatticeFermionD xrb(UGrid);
        xrb = Zero();
        Schur(Dwsp, b, xrb);
        int rb_it = CGrb.IterationsToComplete;

        // Mx (scale-mixed): the M1 correction band-limited to |sin p| < mx_acut (sweet spot ~0.7). INNER
        // (exact, +D_free) and OUTER (cheap). Fed to the same FGMRES + deflation.
        double mx_acut = 0.7;
        if (GridCmdOptionExists(argv, argv + argc, "--mx-acut")) {
          mx_acut = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mx-acut"));
        }
        WilsonImplD::ImplParams pfree(bnd);
        WilsonFermionD Dfree(Uunit, *UGrid, *UrbGrid, mprec, pfree);
        Ffo.build_bandmask(mx_acut, false, 0.3);
        FreeLimitPreconditionerWx<WilsonImplD> MxIn(Ffo, xform, Dframed, Dfree, UGrid, true);
        FreeLimitPreconditionerWx<WilsonImplD> MxOut(Ffo, xform, Dframed, Dfree, UGrid, false);

        DeflatedPrec M0d(M0L, Dwsp, V, UGrid);
        DeflatedPrec M1d(M1L, Dwsp, V, UGrid);
        DeflatedPrec Mxd(MxIn, Dwsp, V, UGrid);
        int it_m0 = run_fgmres(LinOp, M0L, b, stol, smaxit, srst);
        int it_m1 = run_fgmres(LinOp, M1L, b, stol, smaxit, srst);
        int it_mxi = run_fgmres(LinOp, MxIn, b, stol, smaxit, srst);
        int it_mxo = run_fgmres(LinOp, MxOut, b, stol, smaxit, srst);
        int it_m0d = run_fgmres(LinOp, M0d, b, stol, smaxit, srst);
        int it_m1d = run_fgmres(LinOp, M1d, b, stol, smaxit, srst);
        int it_mxd = run_fgmres(LinOp, Mxd, b, stol, smaxit, srst);

        std::cout << GridLogMessage << "  --- outer iteration counts (tol=" << stol
                  << ", Mx |sin p| cut=" << mx_acut << ") ---" << std::endl;
        std::cout << GridLogMessage << "  RB-CGNE (baseline)     iters=" << rb_it << std::endl;
        std::cout << GridLogMessage << "  FGMRES(M0)             iters=" << it_m0 << std::endl;
        std::cout << GridLogMessage << "  FGMRES(M1)             iters=" << it_m1
                  << "   [+1 D_W[U^L]/apply]" << std::endl;
        std::cout << GridLogMessage << "  FGMRES(MxIn)           iters=" << it_mxi
                  << "   [+1 D_W[U^L] +1 D_free +1 FFT-pair/apply]" << std::endl;
        std::cout << GridLogMessage << "  FGMRES(MxOut)          iters=" << it_mxo
                  << "   [+1 D_W[U^L] +1 FFT-pair/apply]" << std::endl;
        std::cout << GridLogMessage << "  FGMRES(M0 + deflation) iters=" << it_m0d
                  << "   [+1 D_W/apply]" << std::endl;
        std::cout << GridLogMessage << "  FGMRES(M1 + deflation) iters=" << it_m1d
                  << "   [+2 D_W/apply]" << std::endl;
        std::cout << GridLogMessage << "  FGMRES(MxIn + deflation) iters=" << it_mxd
                  << "   [MxIn +1 D_W/apply]" << std::endl;
      }
      // (5) DEFLATED-SUBSPACE FRAME OPT (--deflate-frameopt, user experiment 2026-09-05): determine Omega by
      // descending the M0 L2 loss L=sum||M0(Omega)D_W phi - phi||^2 ONLY on the chiral zero modes (the
      // deflated subspace the IRA found), then build BOTH M0 and M1 from that Omega and compare their action
      // on every mode vs the Landau frame. Probes = |chi|>0.5 modes. (M0-loss gradient is the validated
      // fo_loss_force; M1-loss would need the extra gauge-covariant force through U^L(Omega) -- deferred.)
      if (GridCmdOptionExists(argv, argv + argc, "--deflate-frameopt")) {
        int df_iter = 200;
        double df_eta = 0.1;
        if (GridCmdOptionExists(argv, argv + argc, "--df-iter")) {
          df_iter = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--df-iter"));
        }
        if (GridCmdOptionExists(argv, argv + argc, "--df-eta")) {
          df_eta = std::stod(GridCmdOptionPayload(argv, argv + argc, "--df-eta"));
        }
        std::vector<LatticeFermionD> vlist, wlist;
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          if (std::abs(rr_chi[k]) > 0.5) {
            LatticeFermionD wv(UGrid);
            Dwsp.M(rr_mode[k], wv);
            vlist.push_back(rr_mode[k]);
            wlist.push_back(wv);
          }
        }
        std::cout << GridLogMessage << "== DEFLATE-FRAMEOPT: descend M0 L2 loss on " << vlist.size()
                  << " chiral zero modes (df_iter=" << df_iter << " df_eta=" << df_eta << ") ==" << std::endl;
        LatticeColourMatrixD Omopt(UGrid);
        Omopt = xform;
        fo_descend(Ffo, Omopt, wlist, vlist, df_iter, df_eta, 25);

        // build M0, M1 from the optimized frame (M1 needs U^L rebuilt on Omopt)
        FreeLimitPreconditionerW<WilsonImplD> M0opt(Ffo, Omopt, UGrid);
        LatticeGaugeFieldD UframedOpt(UGrid);
        UframedOpt = U;
        LatticeColourMatrixD gopt(UGrid);
        gopt = Omopt;
        SU<Nc>::GaugeTransform<PeriodicGimplD>(UframedOpt, gopt);
        WilsonFermionD DframedOpt(UframedOpt, *UGrid, *UrbGrid, mprec, p1);
        FreeLimitPreconditionerW1<WilsonImplD> M1opt(Ffo, Omopt, DframedOpt, UGrid);

        std::cout << GridLogMessage
                  << "-- COMPARE |M D_W phi|/|phi| (1=perfect): M0/M1 with Landau vs deflate-opt Omega --"
                  << std::endl;
        LatticeFermionD Dphi(UGrid), t(UGrid);
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          Dwsp.M(rr_mode[k], Dphi);
          M0L(Dphi, t);
          RealD r0L = std::sqrt(norm2(t));
          M0opt(Dphi, t);
          RealD r0O = std::sqrt(norm2(t));
          M1L(Dphi, t);
          RealD r1L = std::sqrt(norm2(t));
          M1opt(Dphi, t);
          RealD r1O = std::sqrt(norm2(t));
          RealD lam = std::sqrt(real(rr_lam[k]) * real(rr_lam[k]) + imag(rr_lam[k]) * imag(rr_lam[k]));
          std::cout << GridLogMessage << "  CMP[" << k << "]  |lam|=" << lam << "  chi=" << rr_chi[k]
                    << "  M0_landau=" << r0L << "  M0_opt=" << r0O
                    << "  M1_landau=" << r1L << "  M1_opt=" << r1O << std::endl;
        }
      }
      Grid_finalize();
      return 0;
    }

    // ---- DIRECT FRAME OPTIMIZER (--frameopt): descend Omega on the TRUE prec mismatch vs the Landau frame ----
    if (GridCmdOptionExists(argv, argv + argc, "--frameopt")) {
      double mprec = mass;
      if (GridCmdOptionExists(argv, argv + argc, "--mprec")) {
        mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
      }
      int nprobe = 4;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-probes")) {
        nprobe = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-probes"));
      }
      int fo_iter = 60;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-iter")) {
        fo_iter = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-iter"));
      }
      double fo_eta = 0.1;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-eta")) {
        fo_eta = std::stod(GridCmdOptionPayload(argv, argv + argc, "--fo-eta"));
      }
      std::vector<Complex> bnd(Nd, Complex(1.0, 0.0));
      bnd[Nd - 1] = Complex(-1.0, 0.0);  // anti-periodic time
      FreeWilsonInverse<WilsonImplD> Ffo(UGrid, mprec, bnd);
      WilsonImplD::ImplParams pfo(bnd);
      WilsonFermionD Dwfo(U, *UGrid, *UrbGrid, mass, pfo);
      std::vector<LatticeFermionD> vlist, wlist;
      for (int p = 0; p < nprobe; ++p) {
        LatticeFermionD vv(UGrid), ww(UGrid);
        gaussian(RNG, vv);
        Dwfo.M(vv, ww);
        vlist.push_back(vv);
        wlist.push_back(ww);
      }
      std::cout << GridLogMessage << "==== frameopt: m=" << mass << " mprec=" << mprec
                << " probes=" << nprobe << " iter=" << fo_iter << " eta=" << fo_eta << " ====" << std::endl;
      RealD fd = 0.0, an = 0.0;
      RealD rel = fo_grad_check(Ffo, xform, wlist, vlist, RNG, 1.0e-4, fd, an);
      std::cout << GridLogMessage << "  FD gate: fd=" << fd << "  an=" << an << "  rel=" << rel
                << ((rel < 1.0e-4) ? "  PASS" : "  CHECK") << std::endl;
      LatticeColourMatrixD Omopt = xform;
      RealD L0 = fo_loss(Ffo, xform, wlist, vlist);
      std::cout << GridLogMessage << "  L(Landau frame) = " << L0 << std::endl;
      fo_descend(Ffo, Omopt, wlist, vlist, fo_iter, fo_eta, 25);
      // compare FGMRES N_it: Landau frame vs optimized frame (same operator, same RHS)
      NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dwfo);
      LatticeFermionD bsrc(UGrid), xg(UGrid);
      gaussian(RNG, bsrc);
      RealD tol = 1.0e-8;
      int maxit = 20000;
      {
        FreeLimitPreconditionerW<WilsonImplD> M0L(Ffo, xform, UGrid);
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0L, restart, false);
        xg = Zero();
        FG(LinOp, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, LANDAU frame):    iters=" << FG.IterationCount << std::endl;
      }
      {
        FreeLimitPreconditionerW<WilsonImplD> M0O(Ffo, Omopt, UGrid);
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0O, restart, false);
        xg = Zero();
        FG(LinOp, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, OPTIMIZED frame): iters=" << FG.IterationCount << std::endl;
      }
      // ---- JOINT (Omega, theta) optimisation: add the constant holonomy twist (Nobu/Taku 2026-09-07) ----
      if (GridCmdOptionExists(argv, argv + argc, "--opt_theta")) {
        double th_eta = 0.05;
        if (GridCmdOptionExists(argv, argv + argc, "--theta-eta")) {
          th_eta = std::stod(GridCmdOptionPayload(argv, argv + argc, "--theta-eta"));
        }
        FreeWilsonTwistedInverse<WilsonImplD> Fto(UGrid, mprec, bnd);  // theta = 0 == FreeWilsonInverse
        std::vector<std::array<double, Nd>> theta(Nc, std::array<double, Nd>());
        for (int c = 0; c < Nc; ++c) {
          for (int mu = 0; mu < Nd; ++mu) {
            theta[c][mu] = 0.0;
          }
        }
        LatticeColourMatrixD Omj = xform;  // warm-start at the Landau frame
        RealD Lbase = fo_loss(Fto, xform, wlist, vlist);
        std::cout << GridLogMessage << "==== opt_theta: joint (Omega,theta) descent (theta_eta=" << th_eta
                  << ") ====" << std::endl;
        std::cout << GridLogMessage << "  L(Landau, theta=0) = " << Lbase << std::endl;
        fo_descend_joint(Fto, Omj, theta, wlist, vlist, fo_iter, fo_eta, th_eta, 25);
        std::cout << GridLogMessage << "  optimised theta (per colour, per dir):" << std::endl;
        for (int c = 0; c < Nc; ++c) {
          std::cout << GridLogMessage << "    c=" << c << ":  " << theta[c][0] << "  " << theta[c][1]
                    << "  " << theta[c][2] << "  " << theta[c][3] << std::endl;
        }
        Fto.setTheta(theta);
        M0TwistW M0J(Fto, Omj);
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0J, restart, false);
        xg = Zero();
        FG(LinOp, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, JOINT Omega+theta frame): iters="
                  << FG.IterationCount << std::endl;
      }
      Grid_finalize();
      return 0;
    }

    // ---- Q-FRAME (--qframe, Nobu 2026-09-07): q^n-flow the ORIGINAL config (NO Wilson flow) to a config
    // U^q with the topological deficit LOCALISED to ~1 site; frame-optimise Omega on D_W(U^q) (where the
    // lump is measure-zero in the volume-summed loss -> the optimiser finds the clean BULK frame,
    // undistracted by the spread lump); then use that Omega to precondition the ORIGINAL D_W(U). Compare
    // FGMRES(M0) on D_W(U) for: Landau(U) frame, Omega opt on U^q. ----
    if (GridCmdOptionExists(argv, argv + argc, "--qframe")) {
      double mprec = mass;
      if (GridCmdOptionExists(argv, argv + argc, "--mprec")) {
        mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
      }
      int qn = 6;
      if (GridCmdOptionExists(argv, argv + argc, "--qn")) {
        qn = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--qn"));
      }
      int qnstep = 2000;
      if (GridCmdOptionExists(argv, argv + argc, "--qflow-nstep")) {
        qnstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--qflow-nstep"));
      }
      double qeps = 0.4;
      if (GridCmdOptionExists(argv, argv + argc, "--qflow-eps")) {
        qeps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--qflow-eps"));
      }
      int nprobe = 4;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-probes")) {
        nprobe = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-probes"));
      }
      int fo_iter = 60;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-iter")) {
        fo_iter = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-iter"));
      }
      double fo_eta = 0.1;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-eta")) {
        fo_eta = std::stod(GridCmdOptionPayload(argv, argv + argc, "--fo-eta"));
      }
      std::vector<Complex> bnd(Nd, Complex(1.0, 0.0));
      bnd[Nd - 1] = Complex(-1.0, 0.0);

      // U^q = q^n flow of the ORIGINAL U (pure ||q||_n force, NO Wilson smoothing)
      std::cout << GridLogMessage << "==== qframe: q^" << qn << " flow (eps=" << qeps << " nstep="
                << qnstep << ") of the ORIGINAL config -> U^q ====" << std::endl;
      LatticeGaugeFieldD Uq(UGrid);
      Uq = U;
      QSqueezeGaugeAction<PeriodicGimplD> QSG(qn);
      WilsonFlow<PeriodicGimplD> qwf(qeps, qnstep);
      qwf.setGaugeAction(&QSG);
      {
        LatticeGaugeFieldD Utmp(UGrid);
        qwf.smear(Utmp, Uq);
        Uq = Utmp;
      }
      Real plaqU = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
      Real plaqUq = WilsonLoops<PeriodicGimplD>::avgPlaquette(Uq);
      std::cout << GridLogMessage << "  plaq(U)=" << plaqU << "  plaq(U^q)=" << plaqUq
                << " (bulk should be ~unchanged)" << std::endl;

      // Landau frame of U^q -> warm-start Omega
      LatticeColourMatrixD OmQ(UGrid);
      {
        LatticeGaugeFieldD Uqcopy(UGrid);
        Uqcopy = Uq;
        FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
            Uqcopy, OmQ, 0.1 / 16.0, 3000, 1.0e-12, 1.0e-12, true, -1, false);
      }

      // probes from D_W(U^q): w = D_W(U^q) v
      WilsonImplD::ImplParams pq(bnd);
      WilsonFermionD Dwq(Uq, *UGrid, *UrbGrid, mass, pq);
      std::vector<LatticeFermionD> vlist, wlist;
      for (int p = 0; p < nprobe; ++p) {
        LatticeFermionD vv(UGrid), ww(UGrid);
        gaussian(RNG, vv);
        Dwq.M(vv, ww);
        vlist.push_back(vv);
        wlist.push_back(ww);
      }

      FreeWilsonInverse<WilsonImplD> Ffo(UGrid, mprec, bnd);
      std::cout << GridLogMessage << "  L(Landau(U^q) frame, on U^q probes) = "
                << fo_loss(Ffo, OmQ, wlist, vlist) << std::endl;
      LatticeColourMatrixD OmOpt = OmQ;
      fo_descend(Ffo, OmOpt, wlist, vlist, fo_iter, fo_eta, 25);

      // precondition the ORIGINAL D_W(U): compare Landau(U) vs the U^q-optimised frame
      WilsonImplD::ImplParams pfo(bnd);
      WilsonFermionD Dwu(U, *UGrid, *UrbGrid, mass, pfo);
      NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOpU(Dwu);
      LatticeFermionD bsrc(UGrid), xg(UGrid);
      gaussian(RNG, bsrc);
      RealD tol = 1.0e-8;
      int maxit = 20000;
      {
        FreeLimitPreconditionerW<WilsonImplD> M0L(Ffo, xform, UGrid);  // Landau(flowed U) baseline
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0L, restart, false);
        xg = Zero();
        FG(LinOpU, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, Landau(U) frame) on D(U):     iters="
                  << FG.IterationCount << std::endl;
      }
      {
        FreeLimitPreconditionerW<WilsonImplD> M0Q(Ffo, OmOpt, UGrid);  // frame optimised on U^q
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0Q, restart, false);
        xg = Zero();
        FG(LinOpU, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, U^q-optimised frame) on D(U): iters="
                  << FG.IterationCount << std::endl;
      }
      Grid_finalize();
      return 0;
    }

    // masses: --mass-list "0.1,-0.5,-0.7" (comma-sep) or the single --mass.
    std::vector<double> masses;
    if (GridCmdOptionExists(argv, argv + argc, "--mass-list")) {
      std::stringstream ss(GridCmdOptionPayload(argv, argv + argc, "--mass-list"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        masses.push_back(std::stod(tok));
      }
    } else {
      masses.push_back(mass);
    }
    // --mprec-list "a,b,c": PRECONDITIONER-MASS SCAN at fixed operator mass (--mass). Takes precedence.
    if (GridCmdOptionExists(argv, argv + argc, "--mprec-list")) {
      std::vector<double> mprec_list;
      std::stringstream ss(GridCmdOptionPayload(argv, argv + argc, "--mprec-list"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        mprec_list.push_back(std::stod(tok));
      }
      scan_mprec(U, UGrid, UrbGrid, xform, mass, mprec_list, RNG, restart);
    } else {
      // preconditioner mass per operator mass: --mprec V = FIXED m_prec = V for all masses (the Tikhonov
      // optimum, ~0.8, is operator-mass-independent -> use this for the light-operator crossover). Else
      // --mcrit C -> m_prec = mass - C (physical mass). Else m_prec = mass (old bare behaviour).
      bool have_mprec = GridCmdOptionExists(argv, argv + argc, "--mprec");
      double mprec_fixed = have_mprec ? std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec")) : 0.0;
      double mcrit = 0.0;
      if (GridCmdOptionExists(argv, argv + argc, "--mcrit")) {
        mcrit = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mcrit"));
      }
      for (size_t i = 0; i < masses.size(); ++i) {
        double mprec = have_mprec ? mprec_fixed : (masses[i] - mcrit);
        solve_wilson(U, UGrid, UrbGrid, xform, masses[i], mprec, RNG, restart, repeat);
      }
    }
  }

  Grid_finalize();
  return 0;
}
