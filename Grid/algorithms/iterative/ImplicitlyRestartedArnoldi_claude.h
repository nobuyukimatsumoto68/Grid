#ifndef GRID_IMPLICITLY_RESTARTED_ARNOLDI_CLAUDE_H
#define GRID_IMPLICITLY_RESTARTED_ARNOLDI_CLAUDE_H

// Implicitly Restarted Arnoldi (IRA) for the low-lying COMPLEX spectrum of a NON-Hermitian operator.
//
// WHY THIS EXISTS:
//   Grid ships only a Hermitian eigensolver (ImplicitlyRestartedLanczos.h). The preconditioned
//   domain-wall operator M0 D_DW = (\Omega^\dagger H_free^{-1} \Omega) H (with H = \gamma5 D_DW and
//   H_free = \gamma5 D_free both Hermitian but INDEFINITE) is a product of two Hermitian factors, so its
//   spectrum is COMPLEX and no Hermitian method applies. This header adds a general (complex) Arnoldi
//   eigensolver to probe the low-lying eigenvalues near 0 that the preconditioner should push toward 1.
//
// ALGORITHM (implicit restart with EXACT shifts), cite prominently:
//   - D. C. Sorensen, "Implicit Application of Polynomial Filters in a k-Step Arnoldi Method",
//     SIAM J. Matrix Anal. Appl. 13(1), 357-385 (1992).
//   - R. B. Lehoucq, D. C. Sorensen, C. Yang, "ARPACK Users' Guide" (SIAM, 1998).
//
//   m-step Arnoldi factorization:  A V_m = V_m H_m + f_m e_m^\dagger, with V_m^\dagger V_m = I,
//   H_m upper Hessenberg = V_m^\dagger A V_m, f_m the residual (||f_m|| = beta_m). Each step is one
//   A-apply + modified Gram-Schmidt (reused idea from Grid's GMRES arnoldiStep and Lanczos step()).
//   RESTART: grow to Nm, take the Nm Ritz values theta_i (eigenvalues of H_m via Eigen), split into Nk
//   WANTED (smallest modulus, near 0) and p = Nm - Nk UNWANTED. Use the p unwanted theta_i as EXACT
//   shifts and apply p implicitly-shifted QR sweeps  (H - theta_j I) = Q_j R_j, H <- Q_j^\dagger H Q_j,
//   V <- V Q_j. Accumulating Q = prod_j Q_j applies the filter polynomial prod_j (A - theta_j I)
//   implicitly, damping the unwanted directions; truncating to the leading Nk columns gives a valid
//   Nk-step factorization to extend again. Ritz residual  ||A x_i - theta_i x_i|| = beta_m |e_k^\dagger y_i|
//   is read off the subdiagonal and the last eigenvector component -- no extra A-applies for the test.
//
// This mirrors Grid's ImplicitlyRestartedLanczos.h idioms (field allocation, step(), basisRotate,
// implicit-QR restart + compressed residual bookkeeping) but in complex arithmetic with a full upper
// Hessenberg instead of a real symmetric tridiagonal. Reuses Eigen (bundled) for the small dense
// Hessenberg eigenproblem (ComplexEigenSolver) and the shifted-QR sweeps (HouseholderQR).

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

// Which end of the spectrum is WANTED. For the M0 D_DW diagnostic we want the eigenvalues nearest 0,
// i.e. smallest modulus.
enum IRAsortCriterion {
  IRAsmallestModulus,
  IRAlargestModulus
};

template<class Field>
class ImplicitlyRestartedArnoldi {
private:
  LinearFunction<Field>& _Op;   // the (non-Hermitian) operator A, applied as _Op(in,out)
  int Nstop;                    // number of converged eigenpairs sought
  int Nk;                       // dimension retained after each restart (the wanted subspace)
  int Nm;                       // maximum Krylov dimension
  RealD eresid;                 // relative residual target (normalized by the spectral-radius estimate)
  int MaxIter;                  // maximum number of restarts
  IRAsortCriterion sortcrit;

  Eigen::MatrixXcd H;           // Nm x Nm projected upper Hessenberg
  RealD beta_m;                 // norm of the current residual f  (= ||f||)
  RealD evalMaxApprox;          // running estimate of the spectral radius (max |Ritz|), for the resid test

public:

  ImplicitlyRestartedArnoldi(LinearFunction<Field>& Op,
                             int _Nstop,
                             int _Nk,
                             int _Nm,
                             RealD _eresid,
                             int _MaxIter,
                             IRAsortCriterion _sortcrit = IRAsmallestModulus)
    : _Op(Op),
      Nstop(_Nstop),
      Nk(_Nk),
      Nm(_Nm),
      eresid(_eresid),
      MaxIter(_MaxIter),
      sortcrit(_sortcrit) {
    GRID_ASSERT(Nstop <= Nk);
    GRID_ASSERT(Nk < Nm);
  }

  ////////////////////////////////
  // Helpers
  ////////////////////////////////
  template<typename T> static RealD normalise(T& v) {
    RealD nn = norm2(v);
    nn = std::sqrt(nn);
    v = v * (1.0 / nn);
    return nn;
  }

  // Order the Ritz values WANTED-first per sortcrit. Sort on (key, index) pairs (no lambda): key is the
  // modulus for smallest-modulus, or its negative for largest-modulus, so ascending sort puts WANTED first.
  std::vector<int> rank_ritz(const std::vector<ComplexD>& theta) const {
    int n = (int)theta.size();
    std::vector<std::pair<double, int> > key(n);
    for (int i = 0; i < n; ++i) {
      double m = std::abs(theta[i]);
      double k = (sortcrit == IRAsmallestModulus) ? m : (-m);
      key[i] = std::make_pair(k, i);
    }
    std::sort(key.begin(), key.end());
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
      idx[i] = key[i].second;
    }
    return idx;
  }

  /* Saad, "Numerical Methods for Large Eigenvalue Problems", Arnoldi (Alg. 6.1):
     w := A v_k
     for i = 0..k :  H(i,k) := (v_i, w) ;  w := w - H(i,k) v_i     // modified Gram-Schmidt
     H(k+1,k) := ||w|| ;  v_{k+1} := w / H(k+1,k)
     A single reorthogonalization pass is added (DGKS "twice is enough"): Arnoldi on a non-normal operator
     loses orthogonality faster than Hermitian Lanczos. For k = Nm-1 the new vector is the residual f
     (kept normalized here; the caller restores ||f|| = beta_m as in Grid's Lanczos step()). */
  void step(std::vector<Field>& evec, Field& w, int k) {
    GRID_ASSERT(k < Nm);
    const RealD tiny = 1.0e-20;

    _Op(evec[k], w);

    for (int i = 0; i <= k; ++i) {
      ComplexD hik = innerProduct(evec[i], w);
      H(i, k) = hik;
      w = w - hik * evec[i];
    }
    // second pass (reorthogonalization); fold the correction into H(i,k)
    for (int i = 0; i <= k; ++i) {
      ComplexD dip = innerProduct(evec[i], w);
      H(i, k) = H(i, k) + dip;
      w = w - dip * evec[i];
    }

    RealD beta = norm2(w);
    beta = std::sqrt(beta);
    if (beta < tiny) {
      std::cout << GridLogMessage << " IRA: beta is tiny " << beta
                << " (invariant subspace reached)" << std::endl;
    }
    w = w * (1.0 / beta);

    if (k < Nm - 1) {
      H(k + 1, k) = ComplexD(beta, 0.0);
      evec[k + 1] = w;
    } else {
      beta_m = beta;
    }
  }

  // Implicit restart: filter out the p = Nm - Nk unwanted Ritz values by EXACT-shift implicitly-shifted
  // QR, compress the Nm-step factorization back to Nk steps, and update the residual f.
  void implicit_restart(std::vector<Field>& evec, Field& f) {
    // Ritz values of the full Nm Hessenberg (values only)
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(H, false);
    std::vector<ComplexD> theta(Nm);
    evalMaxApprox = 0.0;
    for (int i = 0; i < Nm; ++i) {
      theta[i] = es.eigenvalues()(i);
      evalMaxApprox = std::max(evalMaxApprox, std::abs(theta[i]));
    }
    std::vector<int> idx = rank_ritz(theta);  // WANTED = idx[0..Nk-1], UNWANTED = idx[Nk..Nm-1] as shifts

    Eigen::MatrixXcd Qacc = Eigen::MatrixXcd::Identity(Nm, Nm);
    Eigen::MatrixXcd Id = Eigen::MatrixXcd::Identity(Nm, Nm);
    for (int j = Nk; j < Nm; ++j) {
      ComplexD mu = theta[idx[j]];
      Eigen::MatrixXcd Hs = H - mu * Id;
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Hs);
      Eigen::MatrixXcd Q = qr.householderQ();
      H = Q.adjoint() * H * Q;
      Qacc = Qacc * Q;
    }

    // The implicitly shifted QR preserves upper-Hessenberg form; kill rounding below the first subdiagonal.
    for (int i = 0; i < Nm; ++i) {
      for (int j = 0; j < Nm; ++j) {
        if (i > j + 1) {
          H(i, j) = ComplexD(0.0, 0.0);
        }
      }
    }

    // Rotate the Krylov basis V <- V Qacc. basisRotate forms evec_new[jj] = sum_k Qt(jj,k) evec[k], so to
    // realize the column combination evec_new[jj] = sum_k Qacc(k,jj) evec[k] we pass Qt = Qacc^T. Rotate
    // output columns 0..Nk (need column Nk = v_{k+1}^+ for the residual compression below).
    Eigen::MatrixXcd Qrot = Qacc.transpose();
    basisRotate(evec, Qrot, 0, Nk + 1, 0, Nm, Nm);

    // Compressed residual (ARPACK):  f_new = v_{k+1}^+ beta_k^+ + f sigma,
    //   beta_k^+ = H^+(Nk, Nk-1)   (subdiagonal at the k boundary after restart),
    //   sigma    = Qacc(Nm-1, Nk-1) = e_m^\dagger Q e_k.
    ComplexD betak = H(Nk, Nk - 1);
    ComplexD sigma = Qacc(Nm - 1, Nk - 1);
    Field fnew(f.Grid());
    fnew = evec[Nk] * betak + f * sigma;
    f = fnew;

    // Fold the residual norm into the retained factorization (phase absorbed into v_{Nk}); this is the
    // coupling H(Nk, Nk-1) for the next extension, exactly as Grid's Lanczos sets lme[k2-1] = beta_k.
    RealD bk = normalise(f);
    evec[Nk] = f;
    H(Nk, Nk - 1) = ComplexD(bk, 0.0);
    beta_m = bk;
  }

  // Count converged WANTED pairs of the retained Nk-block. Ritz residual estimate for pair (theta_i, y_i)
  // of H_Nk is beta_m * |y_i(Nk-1)| (last eigenvector component), no A-applies. Returns Nconv and, if
  // want_output, also fills eval2/evec2y with the sorted Nk-block Ritz values and eigenvectors.
  int test_convergence(std::vector<ComplexD>& eval2, Eigen::MatrixXcd& Yblock, std::vector<int>& idxk) {
    Eigen::MatrixXcd Hk = H.topLeftCorner(Nk, Nk);
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(Hk, true);

    eval2.resize(Nk);
    for (int i = 0; i < Nk; ++i) {
      eval2[i] = es.eigenvalues()(i);
    }
    Yblock = es.eigenvectors();
    idxk = rank_ritz(eval2);

    RealD scale = (evalMaxApprox > 0.0) ? evalMaxApprox : 1.0;
    int nconv = 0;
    for (int s = 0; s < Nstop; ++s) {
      int i = idxk[s];
      RealD rr = beta_m * std::abs(Yblock(Nk - 1, i));
      RealD rrel = rr / scale;
      std::cout << GridLogMessage << "  IRA Ritz[" << std::setw(3) << s << "] = " << eval2[i]
                << "   |resid|/rho = " << rrel << "   target " << eresid
                << (rrel < eresid ? "  CONV" : "") << std::endl;
      if (rrel < eresid) {
        nconv++;
      }
    }
    return nconv;
  }

  // Main driver, mirroring Grid's ImplicitlyRestartedLanczos::calc().
  //   eval : output complex Ritz values (Nstop), sorted WANTED-first
  //   evec : work space (>= Nm vectors); on return the first Nstop hold the Ritz vectors
  //   src  : starting vector
  void calc(std::vector<ComplexD>& eval, std::vector<Field>& evec, const Field& src, int& Nconv) {
    GridBase* grid = src.Grid();
    GRID_ASSERT(grid == evec[0].Grid());
    GRID_ASSERT(Nm <= (int)evec.size());

    std::cout << GridLogMessage << "**********************************************************" << std::endl;
    std::cout << GridLogMessage << " ImplicitlyRestartedArnoldi::calc()  Nstop=" << Nstop
              << " Nk=" << Nk << " Nm=" << Nm << " MaxRestart=" << MaxIter << std::endl;
    std::cout << GridLogMessage << " seeking the "
              << (sortcrit == IRAsmallestModulus ? "smallest-modulus" : "largest-modulus")
              << " eigenvalues" << std::endl;
    std::cout << GridLogMessage << "**********************************************************" << std::endl;

    H = Eigen::MatrixXcd::Zero(Nm, Nm);
    beta_m = 0.0;
    evalMaxApprox = 0.0;
    Nconv = 0;

    Field f(grid);

    evec[0] = src;
    normalise(evec[0]);

    // initial Nk-step Arnoldi factorization
    for (int k = 0; k < Nk; ++k) {
      step(evec, f, k);
    }

    std::vector<ComplexD> eval2;
    Eigen::MatrixXcd Yblock;
    std::vector<int> idxk;

    int iter;
    for (iter = 0; iter < MaxIter; ++iter) {
      std::cout << GridLogMessage << " ***** IRA restart " << iter << " *****" << std::endl;

      // extend Nk -> Nm
      for (int k = Nk; k < Nm; ++k) {
        step(evec, f, k);
      }
      // step() left f as the normalized residual v_Nm and beta_m = ||f||; restore the true residual field
      f = f * beta_m;

      // filter + compress back to Nk
      implicit_restart(evec, f);

      // convergence on the retained Nk-block
      Nconv = test_convergence(eval2, Yblock, idxk);
      std::cout << GridLogMessage << "  IRA converged " << Nconv << "/" << Nstop << std::endl;
      if (Nconv >= Nstop) {
        break;
      }
    }

    if (Nconv < Nstop) {
      std::cout << GridLogMessage << " IRA: NOT fully converged after " << MaxIter
                << " restarts (" << Nconv << "/" << Nstop << "); returning best estimates" << std::endl;
      // ensure we have a current Nk-block decomposition to extract from
      test_convergence(eval2, Yblock, idxk);
    }

    // Extract Ritz vectors x_i = V_Nk y_i for the Nstop wanted, via basisRotate (Qt row s = y_{idxk[s]}^T).
    Eigen::MatrixXcd Qout = Eigen::MatrixXcd::Zero(Nm, Nm);
    for (int s = 0; s < Nstop; ++s) {
      int i = idxk[s];
      for (int k = 0; k < Nk; ++k) {
        Qout(s, k) = Yblock(k, i);
      }
    }
    basisRotate(evec, Qout, 0, Nstop, 0, Nk, Nm);

    // explicit residual check on the accepted modes (Nstop extra A-applies, end of run only)
    eval.resize(Nstop);
    Field Ax(grid);
    Field r(grid);
    for (int s = 0; s < Nstop; ++s) {
      eval[s] = eval2[idxk[s]];
      _Op(evec[s], Ax);
      r = Ax - eval[s] * evec[s];
      RealD rn = std::sqrt(norm2(r));
      std::cout << GridLogMessage << " IRA eval[" << std::setw(3) << s << "] = " << eval[s]
                << "   ||A x - lambda x|| = " << rn << std::endl;
    }

    std::cout << GridLogMessage << "**********************************************************" << std::endl;
    std::cout << GridLogMessage << " ImplicitlyRestartedArnoldi finished: restarts=" << iter
              << "  Nconv=" << Nconv << "/" << Nstop << std::endl;
    std::cout << GridLogMessage << "**********************************************************" << std::endl;
  }
};

NAMESPACE_END(Grid);
#endif
