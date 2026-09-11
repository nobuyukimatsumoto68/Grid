/*************************************************************************************
  FrameOptimizerDWF_claude.h
  Direct frame-Omega optimizer for the free-limit DWF preconditioner M0 = Omega^dag F Omega,
  F = FreeMobius5DInverse. Minimise  L[Omega] = sum_v || P (M0(Omega) D_DW - 1) v ||^2  by group
  gradient descent on the 4D frame Omega in SU(3) (broadcast onto the 5D grid), with an OPTIONAL
  deflation projector P = 1 - sum_i psi_i psi_i^dag (empty -> full space, P = 1).

  Port of the Wilson optimizer (Test_wilson_frameopt_claude.cc:220-333, itself from dwf4
  dwf4_frameopt_claude.h) to DWF. Three changes vs Wilson:
   (1) Omega acts on 5D fields as the broadcast Omega5 (phi = Omega5 * in; Omega^dag = adj(Omega5)*.),
       exactly as FreeLimitPreconditioner (FreeMobius5D_claude.h:1532/1541/1551).
   (2) F is Gamma5-R5-Hermitian: F^dag = Gamma5 F Gamma5 with Gamma5 = gamma5 R5 (use G5R5), replacing
       the Wilson g5 F g5.
   (3) The 4D frame Omega(x) is the SAME on every s-slice, so dL/dOmega4(x) = sum_s dL/dOmega5(x,s):
       the 5D gradient colour-matrix is SUMMED over the 5th dim -> 4D before Ta + the group step.

  Gradient (per probe p; w = D_DW v ; c = M0 w = Omega^dag F Omega w ; r = c - v ; rP = P r):
     s   = Omega^dag Gamma5 F Gamma5 Omega rP        ( = M0^dag rP )
     M5 += traceSpin(w (x) s) - traceSpin(c (x) rP)  (5D colour matrix)
     G4  = Ta( sum_s M5 ) ;  descent  Omega4 <- Omega4 exp(+eta G4) ;  rebroadcast Omega5.
  FD-GATE (fodwf_grad_check) before trusting the gradient: central diff of L along Omega4 exp(+-eps X)
  vs the analytic 2 Re tr[X G4] (dwf4 8^4 gate idiom).

  For frame B the probe vectors v are projected into the complement (P v = v) by the caller so the loss is
  L = sum_v || P (M0 D v - v) ||^2.
*************************************************************************************/
#ifndef GRID_FRAME_OPTIMIZER_DWF_CLAUDE_H
#define GRID_FRAME_OPTIMIZER_DWF_CLAUDE_H

namespace Grid {

// P r = r - sum_i psi_i <psi_i, r>   (psi_i orthonormal 5D; empty Vlow -> out = r, P = 1).
static inline void fodwf_project(const std::vector<LatticeFermionD>& Vlow,
                                 const LatticeFermionD& r, LatticeFermionD& out) {
  out = r;
  for (size_t i = 0; i < Vlow.size(); ++i) {
    ComplexD o = innerProduct(Vlow[i], r);
    out = out - o * Vlow[i];
  }
}

// broadcast the 4D frame onto every s-slice of the 5D Omega field (matches FreeLimitPreconditioner).
static inline void fodwf_broadcast(const LatticeColourMatrixD& Om4, LatticeColourMatrixD& Om5, int Ls) {
  for (int s = 0; s < Ls; ++s) {
    InsertSlice(Om4, Om5, s, 0);
  }
}

// M4 = sum_s M5(.,s)  (5th dim = orthog dimension 0 in Grid's 5D layout).
static inline void fodwf_sum_over_s(const LatticeColourMatrixD& M5, LatticeColourMatrixD& M4, int Ls) {
  LatticeColourMatrixD tmp(M4.Grid());
  M4 = Zero();
  for (int s = 0; s < Ls; ++s) {
    ExtractSlice(tmp, M5, s, 0);
    M4 = M4 + tmp;
  }
}

// L[Omega] = sum_p <r_p, P r_p>,  r_p = Omega^dag F Omega w_p - v_p,  w_p = D_DW v_p.
static inline RealD fodwf_loss(LinearFunction<LatticeFermionD>& F, const LatticeColourMatrixD& Om5,
                               const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                               const std::vector<LatticeFermionD>& Vlow) {
  GridBase* g = w[0].Grid();
  LatticeColourMatrixD Om5dag = adj(Om5);
  LatticeFermionD a(g), b(g), c(g), r(g), rP(g);
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om5 * w[p];
    F(a, b);
    c = Om5dag * b;
    r = c - v[p];
    fodwf_project(Vlow, r, rP);
    L += real(innerProduct(r, rP));   // = ||P r||^2 (P Hermitian projector)
  }
  return L;
}

// Loss + analytic 4D gradient Gf4 = Ta(sum_s M5).
// Fdag applies the TRUE F^dag = (D_free^dag)^{-1}. NOTE: the Mobius kernel (b!=c) is NOT g5R5-Hermitian
// ([D_minus, P_pm] != 0), so F^dag != Gamma5 F Gamma5 -- confirmed numerically (Shamir 3e-15, Mobius 0.42)
// and analytically. Fdag = (D_free.Mdag)^{-1} = D_free (D_free^dag D_free)^{-1} (Grid's consistent adjoint,
// as used by CGNE), supplied by the caller (a CG-on-free-MdagM functor).
static inline RealD fodwf_loss_force(LinearFunction<LatticeFermionD>& F,
                                     LinearFunction<LatticeFermionD>& Fdag,
                                     const LatticeColourMatrixD& Om5,
                                     const std::vector<LatticeFermionD>& w,
                                     const std::vector<LatticeFermionD>& v,
                                     const std::vector<LatticeFermionD>& Vlow, int Ls,
                                     LatticeColourMatrixD& Gf4) {
  GridBase* g = w[0].Grid();
  LatticeColourMatrixD Om5dag = adj(Om5);
  LatticeFermionD a(g), b(g), c(g), r(g), rP(g), t(g), fg(g), s(g);
  LatticeColourMatrixD M5(g);
  M5 = Zero();
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om5 * w[p];
    F(a, b);
    c = Om5dag * b;
    r = c - v[p];
    fodwf_project(Vlow, r, rP);
    L += real(innerProduct(r, rP));
    // s = Omega^dag F^dag Omega rP   (= M0^dag rP), F^dag via the true adjoint (Grid Mdag inverse).
    t = Om5 * rP;
    Fdag(t, fg);
    s = Om5dag * fg;
    M5 = M5 + traceSpin(outerProduct(w[p], s)) - traceSpin(outerProduct(c, rP));
  }
  fodwf_sum_over_s(M5, Gf4, Ls);   // 5D colour-matrix gradient -> 4D (sum over s)
  Gf4 = Ta(Gf4);
  return L;
}

// FINITE-DIFFERENCE gate on the 4D frame: random su(N) X; central diff of L along Om4 exp(+-eps X) vs
// analytic 2 Re tr[X G4]. Returns the relative error. Om5 must be the broadcast of Om4 on entry.
static inline RealD fodwf_grad_check(LinearFunction<LatticeFermionD>& F,
                                     LinearFunction<LatticeFermionD>& Fdag,
                                     const LatticeColourMatrixD& Om4,
                                     LatticeColourMatrixD& Om5,
                                     const std::vector<LatticeFermionD>& w,
                                     const std::vector<LatticeFermionD>& v,
                                     const std::vector<LatticeFermionD>& Vlow, int Ls,
                                     GridParallelRNG& pRNG4, RealD eps, RealD& fd, RealD& an) {
  GridBase* g4 = Om4.Grid();
  LatticeColourMatrixD Gf4(g4);
  fodwf_broadcast(Om4, Om5, Ls);
  fodwf_loss_force(F, Fdag, Om5, w, v, Vlow, Ls, Gf4);
  LatticeColourMatrixD Mr(g4);
  gaussian(pRNG4, Mr);
  LatticeColourMatrixD X = Ta(Mr);
  ComplexD tr = TensorRemove(sum(trace(X * Gf4)));
  an = 2.0 * real(tr);
  LatticeColourMatrixD Op4 = Om4 * expMat(X, eps, 12);
  LatticeColourMatrixD Om4n = Om4 * expMat(X, -eps, 12);
  fodwf_broadcast(Op4, Om5, Ls);
  RealD Lp = fodwf_loss(F, Om5, w, v, Vlow);
  fodwf_broadcast(Om4n, Om5, Ls);
  RealD Lm = fodwf_loss(F, Om5, w, v, Vlow);
  fodwf_broadcast(Om4, Om5, Ls);   // restore
  fd = (Lp - Lm) / (2.0 * eps);
  return std::abs(fd - an) / (std::abs(an) + 1.0e-30);
}

// Backtracking line-search gradient descent on the 4D frame Om4 (Om5 kept in sync). Returns final L.
static inline RealD fodwf_descend(LinearFunction<LatticeFermionD>& F,
                                  LinearFunction<LatticeFermionD>& Fdag,
                                  LatticeColourMatrixD& Om4,
                                  LatticeColourMatrixD& Om5,
                                  const std::vector<LatticeFermionD>& w,
                                  const std::vector<LatticeFermionD>& v,
                                  const std::vector<LatticeFermionD>& Vlow, int Ls,
                                  int niter, RealD eta0, int maxls) {
  GridBase* g4 = Om4.Grid();
  LatticeColourMatrixD Gf4(g4);
  fodwf_broadcast(Om4, Om5, Ls);
  RealD L = fodwf_loss_force(F, Fdag, Om5, w, v, Vlow, Ls, Gf4);
  RealD L0 = L;
  RealD eta = eta0;
  std::cout << GridLogMessage << "# it     L                  ||G||^2           eta" << std::endl;
  for (int it = 0; it < niter; ++it) {
    RealD gn2 = norm2(Gf4);
    std::cout << GridLogMessage << "  " << it << "   " << L << "   " << gn2 << "   " << eta << std::endl;
    if (gn2 < 1.0e-22) {
      break;
    }
    bool acc = false;
    for (int ls = 0; ls < maxls; ++ls) {
      LatticeColourMatrixD Otry4 = Om4 * expMat(Gf4, eta, 12);
      fodwf_broadcast(Otry4, Om5, Ls);
      RealD Lt = fodwf_loss(F, Om5, w, v, Vlow);
      if (Lt < L) {
        Om4 = Otry4;
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
    fodwf_broadcast(Om4, Om5, Ls);
    L = fodwf_loss_force(F, Fdag, Om5, w, v, Vlow, Ls, Gf4);
  }
  fodwf_broadcast(Om4, Om5, Ls);
  std::cout << GridLogMessage << "fodwf_descend: L " << L0 << " -> " << L << "  ("
            << ((L > 0.0) ? L0 / L : 0.0) << "x lower)" << std::endl;
  return L;
}

} // namespace Grid
#endif
