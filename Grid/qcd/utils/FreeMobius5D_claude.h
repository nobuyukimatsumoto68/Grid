#ifndef GRID_FREE_MOBIUS_5D_CLAUDE_H
#define GRID_FREE_MOBIUS_5D_CLAUDE_H

// Momentum-space free Mobius domain-wall block for the free-limit preconditioner
// M0 = Omega^dag F Omega  (see dwf4_qcd_claude/grid_dwf_prec_impl_plan_claude.md).
//
// WHY THIS EXISTS (do not delete this reasoning):
//   Grid's DomainWallFermion/MobiusFermion::FreePropagator (DomainWallFermion.h:44) calls the base
//   WilsonFermion5D::MomentumSpacePropagatorHt_5d (implementation/WilsonFermion5DImplementation.h:538),
//   which is Shamir/tanh ONLY: W = 1 - M5 + sk2, parametrized by (M5, Ls, mass) with NO b,c, and it is
//   NOT overridden in Cayley/Mobius. For our headline Mobius (b,c) = (1.5,0.5) -- or even b=c=1 -- that
//   is the WRONG free kernel. Going Shamir -> Mobius changes BOTH the eigenvalues AND the eigenvectors
//   (the s-direction transfer matrix / cosh alpha is Shamir-specific), so we cannot merely rescale
//   Grid's analytic propagator. We instead build and invert the actual (4 Ls)x(4 Ls) Mobius block per
//   momentum (captures values + vectors). REUSE of Grid is limited to the operator-independent Fourier
//   wrapper (FFT + boundary/twist), which is copied in chunk 1's apply.
//
//   CHUNK 0 (this file): the momentum-space SPECTRAL BLOCK builder ("the eigenvalues"), validated
//   against dwf4_qcd_claude/grid_validation_evals_claude.md to ~1e-12 (Test_dwf_freeprec_claude.cc).
//   CHUNK 1: wrap this block-invert inside a copy of Grid's FreePropagator FFT+twist to get the full
//   apply F as a LinearFunction<FermionField>, then the cold gate ||F D_DW^free v - v|| ~ eps.
//
// Port of dwf4 FreeDW5D (dwf4_free_claude.h). Gammas: DeGrand-Rossi (Grid qcd/spin/Gamma.h convention),
// gamma5 = diag(1,1,-1,-1); mu = 0,1,2 space (x,y,z), mu = 3 time.
//
// References: Mobius/Cayley kernel R. Brower, H. Neff, K. Orginos arXiv:1206.5214; free-frame idea
// R. Brower + T. Izubuchi.

#include <array>
#include <complex>
#include <cmath>
#include <vector>
#include <Grid/Grid.h>

namespace Grid {

// DeGrand-Rossi gamma matrices gamma[mu][a*4+b] (Grid convention). Nonzero entries only.
inline const std::array<std::array<std::complex<double>, 16>, 4>& FreeMobiusGammaDR() {
  static const std::array<std::array<std::complex<double>, 16>, 4> g = []() {
    std::array<std::array<std::complex<double>, 16>, 4> t;
    for (int mu = 0; mu < 4; ++mu) {
      t[mu].fill(std::complex<double>(0.0, 0.0));
    }
    const std::complex<double> I(0.0, 1.0);
    // gamma_X (mu=0)
    t[0][0 * 4 + 3] = I;
    t[0][1 * 4 + 2] = I;
    t[0][2 * 4 + 1] = -I;
    t[0][3 * 4 + 0] = -I;
    // gamma_Y (mu=1)
    t[1][0 * 4 + 3] = std::complex<double>(-1.0, 0.0);
    t[1][1 * 4 + 2] = std::complex<double>(1.0, 0.0);
    t[1][2 * 4 + 1] = std::complex<double>(1.0, 0.0);
    t[1][3 * 4 + 0] = std::complex<double>(-1.0, 0.0);
    // gamma_Z (mu=2)
    t[2][0 * 4 + 2] = I;
    t[2][1 * 4 + 3] = -I;
    t[2][2 * 4 + 0] = -I;
    t[2][3 * 4 + 1] = I;
    // gamma_T (mu=3)
    t[3][0 * 4 + 2] = std::complex<double>(1.0, 0.0);
    t[3][1 * 4 + 3] = std::complex<double>(1.0, 0.0);
    t[3][2 * 4 + 0] = std::complex<double>(1.0, 0.0);
    t[3][3 * 4 + 1] = std::complex<double>(1.0, 0.0);
    return t;
  }();
  return g;
}

// gamma5 = diag(1,1,-1,-1), 4x4 row-major.
inline const std::array<std::complex<double>, 16>& FreeMobiusGamma5() {
  static const std::array<std::complex<double>, 16> g5 = []() {
    std::array<std::complex<double>, 16> t;
    t.fill(std::complex<double>(0.0, 0.0));
    t[0 * 4 + 0] = std::complex<double>(1.0, 0.0);
    t[1 * 4 + 1] = std::complex<double>(1.0, 0.0);
    t[2 * 4 + 2] = std::complex<double>(-1.0, 0.0);
    t[3 * 4 + 3] = std::complex<double>(-1.0, 0.0);
    return t;
  }();
  return g5;
}

// The momentum-space free Mobius block. Holds only the scalar parameters; builds the per-momentum
// (4 Ls)x(4 Ls) matrix on demand. This is the piece that REPLACES Grid's Shamir momentum propagator.
struct FreeMobius5DBlock {
  int Ls;
  double M5;
  double b;
  double c;
  double mass;

  FreeMobius5DBlock(int Ls_, double M5_, double b_, double c_, double mass_)
    : Ls(Ls_), M5(M5_), b(b_), c(c_), mass(mass_) {}

  // free D_W(p) 4x4 (Dirac, row-major), DeGrand-Rossi:
  //   D_W(p) = [(Nd - M5) - sum_mu cos p_mu] I + i sum_mu sin p_mu gamma_mu
  void free_dw_p(const std::array<double, 4>& p, std::array<std::complex<double>, 16>& D) const {
    const std::array<std::array<std::complex<double>, 16>, 4>& gmu = FreeMobiusGammaDR();
    double scalar = 4.0 - M5;
    for (int mu = 0; mu < 4; ++mu) {
      scalar -= std::cos(p[mu]);
    }
    for (int a = 0; a < 4; ++a) {
      for (int bb = 0; bb < 4; ++bb) {
        std::complex<double> v = (a == bb) ? std::complex<double>(scalar, 0.0)
                                           : std::complex<double>(0.0, 0.0);
        for (int mu = 0; mu < 4; ++mu) {
          v += std::complex<double>(0.0, std::sin(p[mu])) * gmu[mu][a * 4 + bb];
        }
        D[a * 4 + bb] = v;
      }
    }
  }

  // (4 Ls)x(4 Ls) Mobius block at momentum p (dense, Eigen):
  //   diag (s,s)   : b D_W + I
  //   up   (s,s+1) : (c D_W - I) P_-   [wall s=Ls-1 -> 0 carries factor -m]
  //   down (s,s-1) : (c D_W - I) P_+   [wall s=0 -> Ls-1 carries factor -m]
  // with P_pm = (I pm gamma5)/2. For gamma5 = diag(1,1,-1,-1): P_- keeps spins {2,3}, P_+ keeps {0,1}
  // -- matches dwf4 FreeDW5D. Built from gamma5 directly so it stays correct in any DeGrand-Rossi sign
  // convention (eigenvalues are basis-independent regardless).
  Eigen::MatrixXcd build_block(const std::array<double, 4>& p) const {
    const int n5 = 4 * Ls;
    std::array<std::complex<double>, 16> D;
    free_dw_p(p, D);

    const std::array<std::complex<double>, 16>& g5 = FreeMobiusGamma5();
    std::array<std::complex<double>, 16> Pp;
    std::array<std::complex<double>, 16> Pm;
    for (int k = 0; k < 16; ++k) {
      std::complex<double> id = ((k / 4) == (k % 4)) ? std::complex<double>(1.0, 0.0)
                                                     : std::complex<double>(0.0, 0.0);
      Pp[k] = 0.5 * (id + g5[k]);
      Pm[k] = 0.5 * (id - g5[k]);
    }

    // (c D_W - I)
    std::array<std::complex<double>, 16> cDmI;
    for (int k = 0; k < 16; ++k) {
      std::complex<double> id = ((k / 4) == (k % 4)) ? std::complex<double>(1.0, 0.0)
                                                     : std::complex<double>(0.0, 0.0);
      cDmI[k] = std::complex<double>(c, 0.0) * D[k] - id;
    }

    // up = (c D_W - I) P_- , down = (c D_W - I) P_+   (4x4 matmul, P_pm on the right)
    std::array<std::complex<double>, 16> up;
    std::array<std::complex<double>, 16> dn;
    for (int a = 0; a < 4; ++a) {
      for (int bb = 0; bb < 4; ++bb) {
        std::complex<double> su(0.0, 0.0);
        std::complex<double> sd(0.0, 0.0);
        for (int kk = 0; kk < 4; ++kk) {
          su += cDmI[a * 4 + kk] * Pm[kk * 4 + bb];
          sd += cDmI[a * 4 + kk] * Pp[kk * 4 + bb];
        }
        up[a * 4 + bb] = su;
        dn[a * 4 + bb] = sd;
      }
    }

    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(n5, n5);
    for (int s = 0; s < Ls; ++s) {
      // diagonal: b D_W + I
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          std::complex<double> val = std::complex<double>(b, 0.0) * D[a * 4 + bb];
          if (a == bb) {
            val += std::complex<double>(1.0, 0.0);
          }
          M(s * 4 + a, s * 4 + bb) += val;
        }
      }
      // up neighbour
      int sup = (s + 1 < Ls) ? (s + 1) : 0;
      double wup = (s + 1 < Ls) ? 1.0 : -mass;
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          M(s * 4 + a, sup * 4 + bb) += std::complex<double>(wup, 0.0) * up[a * 4 + bb];
        }
      }
      // down neighbour
      int sdn = (s > 0) ? (s - 1) : (Ls - 1);
      double wdn = (s > 0) ? 1.0 : -mass;
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          M(s * 4 + a, sdn * 4 + bb) += std::complex<double>(wdn, 0.0) * dn[a * 4 + bb];
        }
      }
    }
    return M;
  }
};

// -------------------- CHUNK 1: full apply F = D_DW^free(m)^{-1} --------------------
// Reuses Grid's DomainWallFermion::FreePropagator FFT+twist machinery VERBATIM (DomainWallFermion.h:44);
// the ONLY substitution is the momentum-space step: Grid's Shamir MomentumSpacePropagatorHt_5d -> our
// per-momentum (4 Ls)x(4 Ls) Mobius block solve (build_block + dense inverse, precomputed per momentum).
// Colour-blind: the same block inverse is applied to each colour. Exposed as a LinearFunction so it can
// serve directly as the free preconditioner on unit gauge (Omega = I); the Omega dressing (M0) is chunk 2.
//
// The 5D grid layout has s = dimension 0, spacetime = dims 1..4 (matches FreePropagator's mask[0]=0,
// shift=1). Momentum at 4D site k: p_mu = 2 pi (k_mu + twist_mu)/L_mu, twist_mu carrying the boundary
// phase (boundary={1,1,1,-1} -> twist_t += 1/2). This mirrors MomentumSpacePropagatorHt_5d:605-608.
template <class Impl>
class FreeMobius5DInverse : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  typedef typename Impl::ComplexField ComplexField;
  typedef typename FermionField::scalar_object SiteSpinor;  // = SpinColourVectorD

  GridCartesian* FGrid;  // 5D grid
  int Ls;
  double M5;
  double b;
  double c;
  double mass;
  std::vector<Complex> boundary;      // e.g. {1,1,1,-1}
  Coordinate ldim;                    // 5D dims [Ls, Lx, Ly, Lz, Lt]
  int V4;
  std::vector<Eigen::MatrixXcd> Minv; // per 4D momentum site, precomputed block inverse

  FreeMobius5DInverse(GridCartesian* FGrid_, int Ls_, double M5_, double b_, double c_, double mass_,
                      std::vector<Complex> boundary_)
    : FGrid(FGrid_), Ls(Ls_), M5(M5_), b(b_), c(c_), mass(mass_), boundary(boundary_) {
    Build();
  }

  void Build() {
    ldim = FGrid->_fdimensions;
    assert((int)boundary.size() == Nd);
    // effective twist including the anti-periodic boundary phase
    std::vector<double> tw(Nd, 0.0);
    for (int mu = 0; mu < Nd; ++mu) {
      double bph = std::acos(real(boundary[mu]));  // Grid `real`, not std:: (Complex may be thrust::complex)
      tw[mu] = bph / (2.0 * M_PI);
    }
    int L[4] = {ldim[1], ldim[2], ldim[3], ldim[4]};
    V4 = L[0] * L[1] * L[2] * L[3];
    FreeMobius5DBlock blk(Ls, M5, b, c, mass);
    Minv.resize(V4);
    for (int kt = 0; kt < L[3]; ++kt) {
      for (int kz = 0; kz < L[2]; ++kz) {
        for (int ky = 0; ky < L[1]; ++ky) {
          for (int kx = 0; kx < L[0]; ++kx) {
            int k[4] = {kx, ky, kz, kt};
            std::array<double, 4> p;
            for (int mu = 0; mu < 4; ++mu) {
              p[mu] = 2.0 * M_PI * (k[mu] + tw[mu]) / L[mu];
            }
            int idx = ((kt * L[2] + kz) * L[1] + ky) * L[0] + kx;
            Minv[idx] = blk.build_block(p).inverse();
          }
        }
      }
    }
  }

  // out = F in = D_DW^free(m)^{-1} in
  virtual void operator()(const FermionField& in, FermionField& out) {
    GridBase* g = in.Grid();
    FermionField in_buf(g);
    FermionField in_k(g);
    FermionField prop_k(g);
    FFT theFFT((GridCartesian*)g);

    ComplexField coor(g);
    ComplexField ph(g);
    ph = Zero();
    ComplexD ci(0.0, 1.0);
    int shift = 1;  // fiveD: s is dim 0
    for (int nu = 0; nu < Nd; ++nu) {
      LatticeCoordinate(coor, nu + shift);
      double bph = std::acos(real(boundary[nu]));  // Grid `real`, not std:: (Complex may be thrust::complex)
      ph = ph + bph * coor * (1.0 / (double)(g->_fdimensions[nu + shift]));
    }
    in_buf = exp(ci * ph * (-1.0)) * in;

    std::vector<int> mask(Nd + 1, 1);
    mask[0] = 0;  // do not FFT the s-dimension
    theFFT.FFT_dim_mask(in_k, in_buf, mask, FFT::forward);
    MomentumSpaceSolve(prop_k, in_k);
    theFFT.FFT_dim_mask(out, prop_k, mask, FFT::backward);

    out = out * exp(ci * ph);
  }

  // per-momentum dense block solve (colour-blind), Minv precomputed
  void MomentumSpaceSolve(FermionField& prop_k, const FermionField& in_k) const {
    int L[4] = {ldim[1], ldim[2], ldim[3], ldim[4]};
    int n5 = 4 * Ls;
    prop_k = Zero();
    for (int kt = 0; kt < L[3]; ++kt) {
      for (int kz = 0; kz < L[2]; ++kz) {
        for (int ky = 0; ky < L[1]; ++ky) {
          for (int kx = 0; kx < L[0]; ++kx) {
            int idx = ((kt * L[2] + kz) * L[1] + ky) * L[0] + kx;
            const Eigen::MatrixXcd& Mi = Minv[idx];

            Eigen::MatrixXcd Fin(n5, Nc);
            for (int s = 0; s < Ls; ++s) {
              Coordinate c5(5);
              c5[0] = s;
              c5[1] = kx;
              c5[2] = ky;
              c5[3] = kz;
              c5[4] = kt;
              SiteSpinor v;
              peekSite(v, in_k, c5);
              for (int a = 0; a < 4; ++a) {
                for (int col = 0; col < Nc; ++col) {
                  ComplexD z = v()(a)(col);  // Grid Complex (thrust under CUDA) -> std::complex explicit
                  Fin(s * 4 + a, col) = std::complex<double>(z.real(), z.imag());
                }
              }
            }

            Eigen::MatrixXcd Y = Mi * Fin;

            for (int s = 0; s < Ls; ++s) {
              Coordinate c5(5);
              c5[0] = s;
              c5[1] = kx;
              c5[2] = ky;
              c5[3] = kz;
              c5[4] = kt;
              SiteSpinor w;
              w = Zero();
              for (int a = 0; a < 4; ++a) {
                for (int col = 0; col < Nc; ++col) {
                  std::complex<double> y = Y(s * 4 + a, col);  // std::complex -> Grid Complex explicit
                  w()(a)(col) = ComplexD(y.real(), y.imag());
                }
              }
              pokeSite(w, prop_k, c5);
            }
          }
        }
      }
    }
  }
};

// -------------------- CHUNK 2: free-limit preconditioner M0 = Omega^dag F Omega --------------------
// Omega = the `xform` returned by FourierAcceleratedGaugeFixer::SteepestDescentGaugeFix (orig -> Landau).
// VERIFIED from Grid source (SUn.impl.h:554,GaugeFix.h:156): GaugeTransform does U -> g U g^dag and the
// fermion rotates psi -> g psi (GaugeTransformFundamental: ferm = g*ferm), with xform = g*xform. Gauge
// covariance then gives D_DW[U] = Omega^dag D_DW[U^L] Omega, so with U^L ~ 1 (Landau) M0 = Omega^dag F
// Omega preconditions D_DW[U_original]. Apply: phi = Omega in ; y = F phi ; out = Omega^dag y. The 4D
// frame Omega(x) is broadcast across the Ls slices (acts the same on every s). Map to dwf4 GaugeCovFreeDW.
template <class Impl>
class FreeLimitPreconditioner : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  FreeMobius5DInverse<Impl>& F;
  LatticeColourMatrixD Omega5;  // the 4D frame broadcast onto the 5D grid
  long n_apply;                 // counts M0 applies (= outer FGMRES iters; M0 costs 0 D_W)

  FreeLimitPreconditioner(FreeMobius5DInverse<Impl>& F_, const LatticeColourMatrixD& xform4,
                          GridCartesian* FGrid)
    : F(F_), Omega5(FGrid), n_apply(0) {
    for (int s = 0; s < F.Ls; ++s) {
      InsertSlice(xform4, Omega5, s, 0);  // broadcast Omega(x) onto every s-slice
    }
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y(in.Grid());
    phi = Omega5 * in;      // Omega : colour mat-vec per site (spin untouched)
    F(phi, y);             // free Mobius inverse
    out = adj(Omega5) * y;  // Omega^dag
  }
};

}  // namespace Grid
#endif
