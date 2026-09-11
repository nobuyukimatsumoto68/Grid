/*************************************************************************************
  RecyclingGeneralisedMinimalResidual_claude.h
  Deflated / recycling FLEXIBLE right-preconditioned GMRES for the free-limit preconditioner.

  ONE class, a flag picks the variant (identical for a single system):
    RecycleAcrossSolves = false  ->  GMRES-DR(m,k)  (deflated restart, subspace rebuilt each solve)
    RecycleAcrossSolves = true   ->  GCRO-DR        (recycle space persists across operator() calls)

  Solves A x = b with right-preconditioner M0. Deflates the isolated near-null cluster of the FIXED
  preconditioned operator B = A M0 (the ~12+ topological stragglers at Re~0.3 that stall restarted GMRES;
  scripts_nm/freeprec_gaugefix_topology_claude.md sec 8).

  KEY DESIGN (two independent concerns kept separate):
   (1) Deflation math lives in u-space: the harmonic Ritz of B = A M0 over span[U, V] where U are the
       previous recycle vectors and V the Arnoldi Krylov, with C = B U orthonormal. (This is standard
       GMRES-DR / GCRO-DR and was validated to deflate correctly.)
   (2) FLEXIBILITY (reach true 1e-8 with an fp32 M0): never reconstruct x = M0 u with a final M0 apply
       (that floors the residual at the fp32 M0 noise ~1e-7). Instead store the images Z = M0 V and
       Y = M0 U, and build the solution directly from them: x += Y d_u + Z d_v. Y is assembled from the
       SAME stored images ([Y_old, Z] P), so no extra M0 applies. Grid's FGMRES reaches 1e-8 the same way.

  Per cycle (recycle triple U, Y = M0 U, C = B U = A Y with C^H C = I):
    project: x += Y (C^H r);  r -= C (C^H r)                          [r = b - A x kept _|_ C]
    Arnoldi m steps: z_j = M0 v_j (STORE); w = A z_j; Bmat = C^H w; w -= C Bmat; MGS(w,V) -> Hbar, v_{j+1}
      => B V_m = C Bmat + V_{m+1} Hbar_m   (B = A M0)
    r _|_ C  =>  d_v = argmin || Hbar_m d_v - ||r|| e1 ||,  d_u = -Bmat d_v
    x += Y d_u + Z d_v ;  r = b - A x (true residual, one A apply)
    refresh U,Y,C from the combined harmonic Ritz of B over span[U, V]

  Currency: one M0 + one A apply per Arnoldi step (+ one A per cycle for the true residual, ~1/m) ->
  D_W count = Ls * IterationCount, comparable to FlexibleGeneralisedMinimalResidual.

  Sources: Morgan, SIAM J. Matrix Anal. Appl. 24 (2002) 20 (GMRES-DR); Parks, de Sturler, Mackey,
  Johnson, Maiti, SIAM J. Sci. Comput. 28 (2006) 1651 (GCRO-DR). Dense harmonic Ritz: HarmonicRitz_claude.h.
*************************************************************************************/
#ifndef GRID_RECYCLING_GENERALISED_MINIMAL_RESIDUAL_CLAUDE_H
#define GRID_RECYCLING_GENERALISED_MINIMAL_RESIDUAL_CLAUDE_H

#include <Grid/algorithms/iterative/HarmonicRitz_claude.h>

namespace Grid {

template<class Field>
class RecyclingGeneralisedMinimalResidual : public OperatorFunction<Field> {
 public:
  using OperatorFunction<Field>::operator();

  bool    ErrorOnNoConverge;
  RealD   Tolerance;
  Integer MaxIterations;
  Integer RestartLength;      // m : Arnoldi steps per cycle
  Integer DeflationDim;       // k : number of deflation / recycle vectors (may exceed m; grows per cycle)
  bool    RecycleAcrossSolves;
  Integer IterationCount;

  LinearFunction<Field> &Preconditioner;   // M0

  GridStopWatch MatrixTimer;
  GridStopWatch PrecTimer;
  GridStopWatch LinalgTimer;

  // Persistent recycle triple (used when RecycleAcrossSolves == true).
  std::vector<Field> Uvec;   // U   : u-space recycle vectors (for the harmonic Ritz of B = A M0)
  std::vector<Field> Yvec;   // Y = M0 U : solution-space images (for the flexible reconstruction)
  std::vector<Field> Cvec;   // C = B U = A Y, orthonormal (for the residual projection)
  bool haveRecycle;

  // Optional EXTERNAL fixed deflation subspace (e.g. the 10 low modes of D5^H D5): seed the recycle
  // space with these u-space vectors and (if FreezeRecycle) do NOT adaptively update them -- i.e. deflate
  // exactly these modes. Set via SetDeflationSubspace. Empty -> pure adaptive GMRES-DR (default).
  std::vector<Field> SeedU;
  bool FreezeRecycle = false;
  void SetDeflationSubspace(const std::vector<Field> &psi, bool freeze = true) {
    SeedU = psi;
    FreezeRecycle = freeze;
  }

  RecyclingGeneralisedMinimalResidual(RealD   tol,
                                      Integer maxit,
                                      LinearFunction<Field> &Prec,
                                      Integer restart_length,
                                      Integer deflation_dim,
                                      bool    recycle_across_solves = false,
                                      bool    err_on_no_conv        = true)
      : ErrorOnNoConverge(err_on_no_conv)
      , Tolerance(tol)
      , MaxIterations(maxit)
      , RestartLength(restart_length)
      , DeflationDim(deflation_dim)
      , RecycleAcrossSolves(recycle_across_solves)
      , Preconditioner(Prec)
      , haveRecycle(false) {};

  void operator()(LinearOperatorBase<Field> &LinOp, const Field &src, Field &psi) {

    psi.Checkerboard() = src.Checkerboard();
    conformable(psi, src);

    const int m = RestartLength;
    const int k = DeflationDim;

    GridBase *grid = src.Grid();

    RealD ssq = norm2(src);
    RealD rsq = Tolerance * Tolerance * ssq;
    RealD srcnorm = sqrt(ssq);

    PrecTimer.Reset();
    MatrixTimer.Reset();
    LinalgTimer.Reset();
    GridStopWatch SolverTimer;
    SolverTimer.Start();

    if (!RecycleAcrossSolves) {
      haveRecycle = false;
      Uvec.clear();
      Yvec.clear();
      Cvec.clear();
    }

    Field w(grid);
    Field r(grid);

    // Seed the recycle space with an EXTERNAL fixed deflation subspace (SeedU) if provided: build
    // U = SeedU, Y = M0 U, C = B U = A Y, orthonormalise C (MGS), adjust U,Y so C^H C = I & C = B U.
    if (!SeedU.empty() && !haveRecycle) {
      int kk = (int)SeedU.size();
      std::vector<Field> Uraw(kk, grid), Yraw(kk, grid), Craw(kk, grid);
      for (int i = 0; i < kk; ++i) {
        Uraw[i] = SeedU[i];
        Preconditioner(Uraw[i], Yraw[i]);   // Y = M0 U
        LinOp.Op(Yraw[i], Craw[i]);         // C_raw = A Y = B U
      }
      Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(kk, kk);
      Cvec.assign(kk, Field(grid));
      for (int j = 0; j < kk; ++j) {
        Field q = Craw[j];
        for (int i = 0; i < j; ++i) {
          ComplexD rij = innerProduct(Cvec[i], q);
          R(i, j) = rij;
          q = q - rij * Cvec[i];
        }
        RealD nrm = sqrt(norm2(q));
        R(j, j) = ComplexD(nrm, 0.0);
        Cvec[j] = (1.0 / nrm) * q;
      }
      Eigen::MatrixXcd Rinv = R.inverse();
      Uvec.assign(kk, Field(grid));
      Yvec.assign(kk, Field(grid));
      for (int j = 0; j < kk; ++j) {
        Field au(grid), ay(grid);
        au = Zero();
        ay = Zero();
        au.Checkerboard() = src.Checkerboard();   // C1: fresh Zero() defaults to Even; stamp the solve cb
        ay.Checkerboard() = src.Checkerboard();
        for (int i = 0; i < kk; ++i) {
          au = au + ComplexD(Rinv(i, j)) * Uraw[i];
          ay = ay + ComplexD(Rinv(i, j)) * Yraw[i];
        }
        Uvec[j] = au;
        Yvec[j] = ay;
      }
      haveRecycle = true;
      std::cout << GridLogMessage << "RecyclingGMRES: seeded fixed deflation subspace k=" << kk
                << (FreezeRecycle ? " (frozen)" : " (adaptive)") << std::endl;
    }

    // Flexible Arnoldi work space: V (orthonormal Krylov) and Z = M0 V (stored images).
    std::vector<Field> v(m + 1, grid);
    std::vector<Field> z(m, grid);
    for (auto &e : v) {
      e = Zero();
    }
    for (auto &e : z) {
      e = Zero();
    }

    MatrixTimer.Start();
    LinOp.Op(psi, w);
    MatrixTimer.Stop();
    r = src - w;

    IterationCount = 0;
    RealD cp = norm2(r);
    bool converged = (cp <= rsq);

    int maxCycles = MaxIterations / m + 1;

    for (int cyc = 0; cyc < maxCycles && !converged; ++cyc) {

      int kc = haveRecycle ? (int)Cvec.size() : 0;

      // Deflate: project the residual off C so r _|_ C (solution absorbs the Y-component).
      if (kc > 0) {
        LinalgTimer.Start();
        Eigen::VectorXcd cr(kc);
        for (int i = 0; i < kc; ++i) {
          cr(i) = innerProduct(Cvec[i], r);
        }
        for (int i = 0; i < kc; ++i) {
          psi = psi + ComplexD(cr(i)) * Yvec[i];
          r   = r   - ComplexD(cr(i)) * Cvec[i];
        }
        LinalgTimer.Stop();
      }

      RealD beta = sqrt(norm2(r));
      v[0] = (1.0 / beta) * r;

      Eigen::MatrixXcd Hbar = Eigen::MatrixXcd::Zero(m + 1, m);
      Eigen::MatrixXcd Bmat = Eigen::MatrixXcd::Zero(kc, m);   // C^H B V

      int jdone = 0;
      for (int j = 0; j < m; ++j) {
        IterationCount++;
        jdone = j + 1;

        PrecTimer.Start();
        Preconditioner(v[j], z[j]);   // z_j = M0 v_j  (store the flexible image)
        PrecTimer.Stop();
        MatrixTimer.Start();
        LinOp.Op(z[j], w);            // w = A z_j = B v_j
        MatrixTimer.Stop();

        LinalgTimer.Start();
        for (int i = 0; i < kc; ++i) {
          ComplexD cw = innerProduct(Cvec[i], w);
          Bmat(i, j) = cw;
          w = w - cw * Cvec[i];
        }
        for (int i = 0; i <= j; ++i) {
          ComplexD hij = innerProduct(v[i], w);
          Hbar(i, j) = hij;
          w = w - hij * v[i];
        }
        RealD hnext = sqrt(norm2(w));
        Hbar(j + 1, j) = ComplexD(hnext, 0.0);
        v[j + 1] = (1.0 / hnext) * w;
        LinalgTimer.Stop();

        if (IterationCount == MaxIterations) {
          break;
        }
      }

      // GMRES least-squares for d_v: min || Hbar(1:jdone+1,1:jdone) d_v - beta e1 ||.
      Eigen::MatrixXcd Hs = Hbar.topLeftCorner(jdone + 1, jdone);
      Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(jdone + 1);
      rhs(0) = ComplexD(beta, 0.0);
      Eigen::VectorXcd dv = Hs.colPivHouseholderQr().solve(rhs);

      Eigen::VectorXcd du;
      if (kc > 0) {
        du = -(Bmat.leftCols(jdone) * dv);
      }

      // SOLUTION update x = psi += Y d_u + Z d_v  (stored M0 images -> flexible, no fp32-limited re-apply).
      LinalgTimer.Start();
      for (int i = 0; i < kc; ++i) {
        psi = psi + ComplexD(du(i)) * Yvec[i];
      }
      for (int j = 0; j < jdone; ++j) {
        psi = psi + ComplexD(dv(j)) * z[j];
      }
      LinalgTimer.Stop();

      // True residual r = src - A psi (one A apply): honest stopping + re-anchors r (drift can't accumulate).
      MatrixTimer.Start();
      LinOp.Op(psi, w);
      MatrixTimer.Stop();
      r = src - w;
      cp = norm2(r);

      std::cout << GridLogMessage << "RecyclingGMRES: cycle " << cyc << " iters " << IterationCount
                << " residual " << sqrt(cp) << " target " << sqrt(rsq) << std::endl;   // per-cycle curve (discovery diag)

      if (cp <= rsq || IterationCount >= MaxIterations) {
        converged = (cp <= rsq);
        break;
      }

      if (!FreezeRecycle) {   // frozen -> keep the seeded fixed deflation subspace (no harmonic-Ritz update)
        updateRecycleSpace(v, z, Hbar, Bmat, kc, jdone, k);
        haveRecycle = (Cvec.size() > 0);
      }
    }

    SolverTimer.Stop();

    RealD true_residual = sqrt(cp) / srcnorm;

    if (converged) {
      std::cout << GridLogMessage << "RecyclingGMRES(m=" << m << ",k=" << DeflationDim
                << (RecycleAcrossSolves ? ",GCRO-DR" : ",GMRES-DR") << "): Converged on iteration "
                << IterationCount << " true residual " << true_residual
                << " target " << Tolerance << std::endl;
      std::cout << GridLogMessage << "RecyclingGMRES Time: Total " << SolverTimer.Elapsed()
                << "  Precon " << PrecTimer.Elapsed() << "  Matrix " << MatrixTimer.Elapsed()
                << "  Linalg " << LinalgTimer.Elapsed() << std::endl;
    } else {
      std::cout << GridLogMessage << "RecyclingGMRES did NOT converge (iters " << IterationCount
                << ", true residual " << true_residual << ")" << std::endl;
      if (ErrorOnNoConverge) {
        GRID_ASSERT(0);
      }
    }
  }

  // Refresh U, Y = M0 U, C = B U from the k smallest-modulus harmonic Ritz of B = A M0 over span[U, V].
  // Combined u-space \hat U = [U, V], image space \hat C = [C, V_{jdone+1}], B \hat U = \hat C \bar G,
  //   \bar G = [[I_k, Bmat],[0, Hbar]].  Harmonic Ritz GEP: (\bar G^H \bar G) p = theta (\bar G^H \tilde G) p,
  //   \tilde G = \hat C^H \hat U = [[C^H U, C^H V],[V^H U, V^H V=I]].
  // New U_raw = \hat U P; Y_raw = [Y, Z] P (stored images -> flexible); C_raw = \hat C (\bar G P) = B U_raw.
  // QR C_raw -> C (orthonormal), R; then U = U_raw R^{-1}, Y = Y_raw R^{-1} so C = B U, Y = M0 U hold.
  void updateRecycleSpace(std::vector<Field> &v,
                          std::vector<Field> &z,
                          const Eigen::MatrixXcd &Hbar,
                          const Eigen::MatrixXcd &Bmat,
                          int kc, int jdone, int k) {

    GridBase *grid = v[0].Grid();
    int cb = v[0].Checkerboard();   // C1: stamp fresh Zero()-accumulators with the solve checkerboard (RB: Odd)
    const int nv = kc + jdone;
    const int nw = kc + jdone + 1;

    Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(nw, nv);
    for (int i = 0; i < kc; ++i) {
      G(i, i) = ComplexD(1.0, 0.0);
    }
    for (int i = 0; i < kc; ++i) {
      for (int j = 0; j < jdone; ++j) {
        G(i, kc + j) = Bmat(i, j);
      }
    }
    for (int i = 0; i <= jdone; ++i) {
      for (int j = 0; j < jdone; ++j) {
        G(kc + i, kc + j) = Hbar(i, j);
      }
    }

    // \tilde G = [C, V_{jdone+1}]^H [U, V_jdone]: C^H U, C^H V, V^H U, and V^H V = [I_jdone ; 0].
    Eigen::MatrixXcd Gt = Eigen::MatrixXcd::Zero(nw, nv);
    LinalgTimer.Start();
    for (int i = 0; i < kc; ++i) {
      for (int j = 0; j < kc; ++j) {
        Gt(i, j) = innerProduct(Cvec[i], Uvec[j]);        // C^H U
      }
      for (int j = 0; j < jdone; ++j) {
        Gt(i, kc + j) = innerProduct(Cvec[i], v[j]);      // C^H V
      }
    }
    for (int i = 0; i <= jdone; ++i) {
      for (int j = 0; j < kc; ++j) {
        Gt(kc + i, j) = innerProduct(v[i], Uvec[j]);      // V^H U
      }
    }
    for (int j = 0; j < jdone; ++j) {
      Gt(kc + j, kc + j) = ComplexD(1.0, 0.0);            // V^H V = I on the first jdone rows
    }
    LinalgTimer.Stop();

    Eigen::MatrixXcd A_ = G.adjoint() * G;
    Eigen::MatrixXcd B_ = G.adjoint() * Gt;
    Eigen::MatrixXcd M_ = B_.colPivHouseholderQr().solve(A_);
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(M_);
    Eigen::VectorXcd evals = ces.eigenvalues();
    Eigen::MatrixXcd evecs = ces.eigenvectors();

    int kk = k;
    if (kk > nv) {
      kk = nv;
    }
    std::vector<int> idx(nv);
    for (int i = 0; i < nv; ++i) {
      idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
              [&evals](int a, int b) { return std::abs(evals(a)) < std::abs(evals(b)); });

    Eigen::MatrixXcd P(nv, kk);
    for (int j = 0; j < kk; ++j) {
      P.col(j) = evecs.col(idx[j]);
    }
    Eigen::MatrixXcd GP = G * P;   // nw x kk

    // U_raw = [U,V] P ; Y_raw = [Y,Z] P ; C_raw = [C, V_{m+1}] (G P).
    std::vector<Field> Uraw(kk, grid);
    std::vector<Field> Yraw(kk, grid);
    std::vector<Field> Craw(kk, grid);
    LinalgTimer.Start();
    for (int j = 0; j < kk; ++j) {
      Field accu(grid);
      accu = Zero();
      accu.Checkerboard() = cb;
      Field accy(grid);
      accy = Zero();
      accy.Checkerboard() = cb;
      for (int i = 0; i < kc; ++i) {
        accu = accu + ComplexD(P(i, j)) * Uvec[i];
        accy = accy + ComplexD(P(i, j)) * Yvec[i];
      }
      for (int i = 0; i < jdone; ++i) {
        accu = accu + ComplexD(P(kc + i, j)) * v[i];
        accy = accy + ComplexD(P(kc + i, j)) * z[i];   // z = M0 v  -> Y_raw = M0 U_raw (flexible)
      }
      Uraw[j] = accu;
      Yraw[j] = accy;

      Field accc(grid);
      accc = Zero();
      accc.Checkerboard() = cb;
      for (int i = 0; i < kc; ++i) {
        accc = accc + ComplexD(GP(i, j)) * Cvec[i];
      }
      for (int i = 0; i <= jdone; ++i) {
        accc = accc + ComplexD(GP(kc + i, j)) * v[i];
      }
      Craw[j] = accc;
    }
    LinalgTimer.Stop();

    // QR C_raw (MGS) -> C = Q, C_raw = Q R; then U = U_raw R^{-1}, Y = Y_raw R^{-1}.
    Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(kk, kk);
    std::vector<Field> Cnew(kk, grid);
    LinalgTimer.Start();
    for (int j = 0; j < kk; ++j) {
      Field q = Craw[j];
      for (int i = 0; i < j; ++i) {
        ComplexD rij = innerProduct(Cnew[i], q);
        R(i, j) = rij;
        q = q - rij * Cnew[i];
      }
      RealD nrm = sqrt(norm2(q));
      R(j, j) = ComplexD(nrm, 0.0);
      Cnew[j] = (1.0 / nrm) * q;
    }
    LinalgTimer.Stop();

    Eigen::MatrixXcd Rinv = R.inverse();
    std::vector<Field> Unew(kk, grid);
    std::vector<Field> Ynew(kk, grid);
    LinalgTimer.Start();
    for (int j = 0; j < kk; ++j) {
      Field accu(grid);
      accu = Zero();
      accu.Checkerboard() = cb;
      Field accy(grid);
      accy = Zero();
      accy.Checkerboard() = cb;
      for (int i = 0; i < kk; ++i) {
        accu = accu + ComplexD(Rinv(i, j)) * Uraw[i];
        accy = accy + ComplexD(Rinv(i, j)) * Yraw[i];
      }
      Unew[j] = accu;
      Ynew[j] = accy;
    }
    LinalgTimer.Stop();

    Uvec = Unew;
    Yvec = Ynew;
    Cvec = Cnew;
  }
};
}
#endif
