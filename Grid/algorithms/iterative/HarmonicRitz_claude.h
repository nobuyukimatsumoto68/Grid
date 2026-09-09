/*************************************************************************************
  HarmonicRitz_claude.h -- dense harmonic-Ritz extraction for deflated/recycling GMRES.

  Eigen-ONLY (no Grid field dependencies) so it can be unit-tested standalone. Used by
  RecyclingGeneralisedMinimalResidual_claude.h (GMRES-DR / GCRO-DR).

  Algorithm source: R. B. Morgan, "GMRES with deflated restarting",
  SIAM J. Matrix Anal. Appl. 24 (2002) 20 -- harmonic Ritz values are the eigenpairs of

      ( H_m + h_{m+1,m}^2  H_m^{-H} e_m e_m^T ) g = \theta g ,                    (Morgan)

  where the m-step Arnoldi on the operator B gives   B V_m = V_{m+1} \bar H_m ,
  \bar H_m is (m+1) x m upper Hessenberg, H_m its square top block, and
  h_{m+1,m} = \bar H_m(m, m-1) the last subdiagonal. We keep the k smallest-|\theta|
  pairs (the near-null cluster to deflate). Rather than forming H_m^{-H} explicitly
  (H_m is near-singular precisely on the modes we want) we solve H_m^H f = e_m by QR.
*************************************************************************************/
#ifndef GRID_HARMONIC_RITZ_CLAUDE_H
#define GRID_HARMONIC_RITZ_CLAUDE_H

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <algorithm>

namespace Grid {

// Compute the k smallest-modulus harmonic Ritz pairs of an m-step Arnoldi.
//   Hbar   : (m+1) x m upper-Hessenberg (unrotated Arnoldi Hessenberg \bar H_m).
//   k      : number of pairs to return (clamped to [1, m]).
//   theta  : (out) k harmonic Ritz values, ascending in |theta|.
//   G      : (out) m x k matrix; column i is g_i (coeffs of the Ritz vector in V_m).
// Returns the actual number of pairs returned (== clamped k).
static inline int harmonicRitzSmallest(const Eigen::MatrixXcd& Hbar,
                                       int k,
                                       std::vector<std::complex<double>>& theta,
                                       Eigen::MatrixXcd& G)
{
  typedef std::complex<double> C;

  const int m = static_cast<int>(Hbar.cols());
  if (k < 1)
  {
    k = 1;
  }
  if (k > m)
  {
    k = m;
  }

  // Square top block H_m and the last subdiagonal h_{m+1,m}.
  Eigen::MatrixXcd H = Hbar.topLeftCorner(m, m);
  C h_last = Hbar(m, m - 1);
  double h2 = std::norm(h_last);   // |h_{m+1,m}|^2

  // f = H^{-H} e_m  via the linear solve  H^H f = e_m  (QR; avoids explicit inverse).
  Eigen::VectorXcd e_m = Eigen::VectorXcd::Zero(m);
  e_m(m - 1) = C(1.0, 0.0);
  Eigen::VectorXcd f = H.adjoint().colPivHouseholderQr().solve(e_m);

  // M = H + h^2 f e_m^T   (rank-1 update; e_m^T selects the last column).
  Eigen::MatrixXcd M = H;
  M += h2 * (f * e_m.transpose());

  // Dense complex eigensolve of the m x m harmonic-Ritz matrix.
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(M);
  Eigen::VectorXcd evals = ces.eigenvalues();
  Eigen::MatrixXcd evecs = ces.eigenvectors();

  // Sort eigenpairs by ascending |theta|.
  std::vector<int> idx(m);
  for (int i = 0; i < m; ++i)
  {
    idx[i] = i;
  }
  std::sort(idx.begin(), idx.end(),
            [&evals](int a, int b) { return std::abs(evals(a)) < std::abs(evals(b)); });

  // Emit the k smallest.
  theta.resize(k);
  G.resize(m, k);
  for (int j = 0; j < k; ++j)
  {
    int src = idx[j];
    theta[j] = evals(src);
    G.col(j) = evecs.col(src);
  }
  return k;
}

} // namespace Grid
#endif
