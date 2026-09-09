// Standalone (non-Grid-build) unit test for HarmonicRitz_claude.h (chunk 1 of the deflated-GMRES plan).
// Compile:
//   g++ -std=c++17 -I Grid/Eigen -I Grid/Grid/algorithms/iterative \
//       Grid/scripts_nm/harmonic_ritz_unittest_claude.cc -o /tmp/hrt_claude && /tmp/hrt_claude
// Tests: (1) the h^2 f e_m^T correction is assembled correctly (compare vs an independently formed M);
//        (2) end-to-end: full Arnoldi on a diagonal B recovers its near-null eigenvalue cluster as the
//        smallest-|theta| harmonic Ritz values.

#include "HarmonicRitz_claude.h"
#include <iostream>
#include <iomanip>
#include <cmath>

typedef std::complex<double> C;

static bool approx(C a, C b, double tol)
{
  return std::abs(a - b) < tol;
}

int main()
{
  int fails = 0;
  std::cout << std::setprecision(8);

  // ---- Test 1: correction assembly + selection vs an independently formed M ----
  // Arbitrary 3-step Arnoldi Hessenberg \bar H (4 x 3), nonzero subdiagonals.
  {
    int m = 3;
    Eigen::MatrixXcd Hbar = Eigen::MatrixXcd::Zero(m + 1, m);
    Hbar(0,0) = C(2.0, 0.3);  Hbar(0,1) = C(0.5,-0.2); Hbar(0,2) = C(0.1, 0.0);
    Hbar(1,0) = C(0.4, 0.0);  Hbar(1,1) = C(1.0, 0.1); Hbar(1,2) = C(0.3,-0.1);
    Hbar(2,1) = C(0.6, 0.0);  Hbar(2,2) = C(0.7, 0.2);
    Hbar(3,2) = C(0.25,0.0);  // h_{m+1,m}

    std::vector<C> theta;
    Eigen::MatrixXcd G;
    Grid::harmonicRitzSmallest(Hbar, m, theta, G);

    // Independent reference: M = H + |h|^2 H^{-H} e_m e_m^T, eig, sort ascending |.|.
    Eigen::MatrixXcd H = Hbar.topLeftCorner(m, m);
    double h2 = std::norm(Hbar(m, m - 1));
    Eigen::VectorXcd e_m = Eigen::VectorXcd::Zero(m);
    e_m(m - 1) = C(1.0, 0.0);
    Eigen::VectorXcd fref = H.adjoint().fullPivLu().solve(e_m);
    Eigen::MatrixXcd Mref = H + h2 * (fref * e_m.transpose());
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(Mref);
    std::vector<C> ref(ces.eigenvalues().data(), ces.eigenvalues().data() + m);
    std::sort(ref.begin(), ref.end(), [](C a, C b){ return std::abs(a) < std::abs(b); });

    std::cout << "[T1] harmonic Ritz values (function vs reference):\n";
    for (int i = 0; i < m; ++i)
    {
      std::cout << "     " << theta[i] << "   ref " << ref[i]
                << (approx(theta[i], ref[i], 1e-9) ? "  ok" : "  MISMATCH") << "\n";
      if (!approx(theta[i], ref[i], 1e-9))
      {
        ++fails;
      }
    }
  }

  // ---- Test 2: end-to-end known near-null cluster on a diagonal B ----
  // B = diag(0.3+0.1i, 0.3-0.1i, 1, 3, 8). Full Arnoldi (m=n) -> harmonic Ritz = eig(B); the two
  // smallest-|theta| must be the 0.3 +- 0.1i near-null pair.
  {
    int n = 5;
    Eigen::VectorXcd d(n);
    d << C(0.3, 0.1), C(0.3, -0.1), C(1.0, 0.0), C(3.0, 0.0), C(8.0, 0.0);
    Eigen::MatrixXcd B = d.asDiagonal();

    // Hand Arnoldi from a start vector with nonzero overlap on every eigenvector.
    Eigen::VectorXcd b(n);
    b << C(1,0), C(1,0), C(1,0), C(1,0), C(1,0);
    std::vector<Eigen::VectorXcd> V;
    V.push_back(b / b.norm());
    Eigen::MatrixXcd Hbar = Eigen::MatrixXcd::Zero(n + 1, n);
    for (int j = 0; j < n; ++j)
    {
      Eigen::VectorXcd w = B * V[j];
      for (int i = 0; i <= j; ++i)
      {
        Hbar(i, j) = V[i].dot(w);   // Eigen dot = conjugate-linear in first arg = <V_i, w>
        w -= Hbar(i, j) * V[i];
      }
      double hn = w.norm();
      Hbar(j + 1, j) = C(hn, 0.0);
      if (hn > 1e-14 && j + 1 < n)
      {
        V.push_back(w / hn);
      }
    }

    std::vector<C> theta;
    Eigen::MatrixXcd G;
    Grid::harmonicRitzSmallest(Hbar, 2, theta, G);

    std::cout << "[T2] two smallest harmonic Ritz (expect 0.3 +- 0.1i):\n";
    bool got_pair = (approx(theta[0], C(0.3, 0.1), 1e-6) || approx(theta[0], C(0.3, -0.1), 1e-6)) &&
                    (approx(theta[1], C(0.3, 0.1), 1e-6) || approx(theta[1], C(0.3, -0.1), 1e-6));
    for (int i = 0; i < 2; ++i)
    {
      std::cout << "     " << theta[i] << "\n";
    }
    std::cout << (got_pair ? "     ok\n" : "     MISMATCH\n");
    if (!got_pair)
    {
      ++fails;
    }
  }

  std::cout << (fails == 0 ? "\nALL PASS\n" : "\nFAILS\n");
  return fails == 0 ? 0 : 1;
}
