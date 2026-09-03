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
#include <random>
#include <Grid/Grid.h>
#include <Grid/algorithms/BlockingFFT_claude.h>  // SIMD-blocking FFT (used under -DFREEMOBIUS5D_BLOCKING_FFT)
#include <Grid/algorithms/FFT_claude.h>          // cached-plan/pgbuf FFT (DEFAULT; -DFREEMOBIUS5D_GRID_FFT = Grid's uncached)

// fp32 is the DEFAULT F-apply precision (validated 1.51x, 2026-09-02; gate 3.3e-7, FGMRES iters identical
// to double -- grid_packonce_fft_impl_plan_claude.md Chunk B). It runs the barrel pipeline in single on a
// dedicated single-precision 5D grid; the OUTER FGMRES stays double so final accuracy is unchanged. Force
// the double reference path with -DFREEMOBIUS5D_FP64. -DFREEMOBIUS5D_PACKONCE selects the (shelved,
// net-loss) pack-once path instead and takes precedence.
#if !defined(FREEMOBIUS5D_FP64) && !defined(FREEMOBIUS5D_PACKONCE)
#define FREEMOBIUS5D_USE_FP32
#endif

// max n5 = 4*Ls for per-thread device scratch (block-Thomas + pack-once solve). Ls <= 16.
#ifndef FREEMOBIUS5D_PO_NMAX
#define FREEMOBIUS5D_PO_NMAX 64
#endif

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

// Device block-Thomas solve y = T^{-1} r for ONE momentum (colour vector, n5 = 4*Ls). T = D_DW(p,m=0)
// block-tridiagonal: diag A = b D_W + I, super Uup = (c D_W - I)P_- (cols 2,3), sub Udn = (c D_W - I)P_+
// (cols 0,1). D = D_W(p) 4x4 (row-major); Dinv = the Ls pivot inverses Delta_s^{-1} (Ls*16). cC = c as a
// complex of the field type. Templated on the complex scalar C so it serves the double and fp32 solves.
// Same math as block_thomas_solve_host (BT-0, gate 4.3e-16). grid_block_thomas_impl_plan_claude.md.
template <class C>
accelerator_inline void bt_solve_dev(int Ls, C cC, const C* D, const C* Dinv, const C* r, C* y) {
  C one = C(1.0, 0.0);
  C Uup[16];
  C Udn[16];
  for (int a = 0; a < 4; ++a) {
    for (int a2 = 0; a2 < 4; ++a2) {
      C hop = D[a * 4 + a2] * cC;
      if (a == a2) {
        hop = hop - one;
      }
      Uup[a * 4 + a2] = (a2 >= 2) ? hop : C(0.0, 0.0);
      Udn[a * 4 + a2] = (a2 < 2) ? hop : C(0.0, 0.0);
    }
  }
  C rho[FREEMOBIUS5D_PO_NMAX];
  for (int k = 0; k < 4 * Ls; ++k) {
    rho[k] = r[k];
  }
  // forward: rho_s -= Udn (Delta_{s-1}^{-1} rho_{s-1})
  for (int s = 1; s < Ls; ++s) {
    const C* Dprev = Dinv + (s - 1) * 16;
    C t[4];
    for (int a = 0; a < 4; ++a) {
      C acc = C(0.0, 0.0);
      for (int bb = 0; bb < 4; ++bb) {
        acc = acc + Dprev[a * 4 + bb] * rho[(s - 1) * 4 + bb];
      }
      t[a] = acc;
    }
    for (int a = 0; a < 4; ++a) {
      C acc = C(0.0, 0.0);
      for (int bb = 0; bb < 4; ++bb) {
        acc = acc + Udn[a * 4 + bb] * t[bb];
      }
      rho[s * 4 + a] = rho[s * 4 + a] - acc;
    }
  }
  // backward: y_{Ls-1} = Delta_{Ls-1}^{-1} rho_{Ls-1}; y_s = Delta_s^{-1}(rho_s - Uup y_{s+1})
  const C* Dlast = Dinv + (Ls - 1) * 16;
  for (int a = 0; a < 4; ++a) {
    C acc = C(0.0, 0.0);
    for (int bb = 0; bb < 4; ++bb) {
      acc = acc + Dlast[a * 4 + bb] * rho[(Ls - 1) * 4 + bb];
    }
    y[(Ls - 1) * 4 + a] = acc;
  }
  for (int s = Ls - 2; s >= 0; --s) {
    C tmp[4];
    for (int a = 0; a < 4; ++a) {
      C acc = rho[s * 4 + a];
      for (int bb = 0; bb < 4; ++bb) {
        acc = acc - Uup[a * 4 + bb] * y[(s + 1) * 4 + bb];
      }
      tmp[a] = acc;
    }
    const C* Ds = Dinv + s * 16;
    for (int a = 0; a < 4; ++a) {
      C acc = C(0.0, 0.0);
      for (int bb = 0; bb < 4; ++bb) {
        acc = acc + Ds[a * 4 + bb] * tmp[bb];
      }
      y[s * 4 + a] = acc;
    }
  }
}

// Device rank-4 antiperiodic correction for ONE momentum: out = y + m T^{-1} L (I4 - m G_bt)^{-1} R y,
// with y = T^{-1} r already computed. Cm = (I4 - m G_bt)^{-1} (4x4, precomputed). mC = m, cC = c.
template <class C>
accelerator_inline void bt_corner_dev(int Ls, C cC, C mC, const C* D, const C* Dinv, const C* Cm,
                                      const C* y, C* out) {
  C one = C(1.0, 0.0);
  const int srcspin[4] = {2, 3, 0, 1};
  const int tslice[4] = {Ls - 1, Ls - 1, 0, 0};
  const int rrow[4] = {0 * 4 + 2, 0 * 4 + 3, (Ls - 1) * 4 + 0, (Ls - 1) * 4 + 1};
  C w[4];
  for (int i = 0; i < 4; ++i) {
    w[i] = y[rrow[i]];
  }
  C sc4[4];
  for (int a = 0; a < 4; ++a) {
    C acc = C(0.0, 0.0);
    for (int bb = 0; bb < 4; ++bb) {
      acc = acc + Cm[a * 4 + bb] * w[bb];
    }
    sc4[a] = acc;
  }
  C Lvec[FREEMOBIUS5D_PO_NMAX];
  for (int k = 0; k < 4 * Ls; ++k) {
    Lvec[k] = C(0.0, 0.0);
  }
  for (int j = 0; j < 4; ++j) {
    int spn = srcspin[j];
    int st = tslice[j];
    for (int a = 0; a < 4; ++a) {
      C hop = D[a * 4 + spn] * cC;
      if (a == spn) {
        hop = hop - one;
      }
      Lvec[st * 4 + a] = Lvec[st * 4 + a] + sc4[j] * hop;
    }
  }
  C z[FREEMOBIUS5D_PO_NMAX];
  bt_solve_dev(Ls, cC, D, Dinv, Lvec, z);
  for (int k = 0; k < 4 * Ls; ++k) {
    out[k] = y[k] + mC * z[k];
  }
}

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
  int V4loc;                          // LOCAL (per-rank) 4D momentum count; keys Minv
  std::vector<Eigen::MatrixXcd> Minv; // per LOCAL 4D momentum site, precomputed block inverse (host/build ref)

  // block-Thomas per-momentum solve (O(Ls) replacement for the dense Minv; peer PV+mass, algebra in
  // grid_block_thomas_impl_plan_claude.md). BT-0 = host precompute + reference + gate vs dense Minv.
  // Stored per LOCAL momentum idx (host, double). Device UVM copies come in BT-1.
  std::vector<std::array<std::complex<double>, 16>> bt_D;     // D_W(p) 4x4 (row-major a*4+a2)
  std::vector<std::vector<std::complex<double>>> bt_Dinv;     // Ls pivot inverses Delta_s^{-1} (Ls*16)
  std::vector<std::array<std::complex<double>, 16>> bt_Cminv; // (I4 - m G_bt)^{-1} 4x4 (m fixed in F)
  // BT-1 device (UVM, slot-indexed oSite4*Nsimd+lane, like Minv_dev): double path (fp64 / default-double).
  Grid::Vector<Grid::Complex> btD_dev;      // 16 / slot
  Grid::Vector<Grid::Complex> btDinv_dev;   // Ls*16 / slot
  Grid::Vector<Grid::Complex> btCminv_dev;  // 16 / slot
  // device-resident copy of Minv for the on-device batched solve (no host round-trip). UVM (Grid::Vector
  // = uvmAllocator) so it is host-fillable and device-readable. Laid out row-major (4Ls)x(4Ls) per SITE
  // SLOT (oSite4*Nsimd + lane), NOT per idx4, so the kernel needs no coordinate math -- Build() scatters
  // each idx4's inverse into the slot(s) whose local 4D coord maps to it.
  Grid::Vector<Grid::Complex> Minv_dev;

  // precomputed constant twist phase fields exp(-i ph) and exp(+i ph). ph depends only on the boundary
  // phases + coordinates (NOT the input), so it is built ONCE in Build() and the per-apply phase becomes
  // a pointwise multiply instead of recomputing exp(+-i ph) every apply.
  ComplexField phase_neg;
  ComplexField phase_pos;

  // (a) hoisted per-apply scratch buffers + (b) cached-plan/pgbuf FFT -- allocated ONCE (members), reused
  // every apply instead of reallocating/re-planning per apply/per-FFT_dim-call (the working-space knob).
  typedef typename FermionField::vector_object::scalar_type FftScalar;
  FermionField m_in_buf;
  FermionField m_in_k;
  FermionField m_prop_k;
  FFT_claude<FftScalar> m_fft;

#ifdef FREEMOBIUS5D_PACKONCE
  // ---- pack-once fused path (SINGLE RANK): one pack (Grid SIMD -> contiguous) + fused contiguous cuFFT
  // (rank-3 xyz + rank-1 t, cached) + block solve IN the buffer + one unpack. Pays 1 pack + 1 unpack per
  // apply instead of the barrel path's 16 layout conversions. CoreScalar = precision knob (fp32 via
  // -DFREEMOBIUS5D_FP32). Design + the "no-free-lunch" lesson: grid_packonce_fft_impl_plan_claude.md.
#ifdef FREEMOBIUS5D_FP32
  typedef ComplexF CoreScalar;
#else
  typedef ComplexD CoreScalar;
#endif
#ifndef FREEMOBIUS5D_PO_NMAX
#define FREEMOBIUS5D_PO_NMAX 64  // max n5 = 4*Ls for the solve's per-thread gather buffer (Ls <= 16)
#endif
  typedef typename FFTW<CoreScalar>::FFTW_plan CorePlan;
  typedef typename FFTW<CoreScalar>::FFTW_scalar CoreFFTWScalar;
  int po_Lx, po_Ly, po_Lz, po_Lt;      // local spacetime extents (== global, single rank)
  int po_V4;                           // Lx*Ly*Lz*Lt
  int po_Lzyx;                         // Lx*Ly*Lz (one t-slab)
  int po_Ncomp;                        // Ls*4*Nc components (independent FFT batches)
  int po_n5;                           // 4*Ls (the (s,spin) block size the solve couples)
  deviceVector<CoreScalar> m_B;        // contiguous [comp][kt][kz][ky][kx], comp=(s*4+spin)*Nc+colour
  // UVM (uvmAllocator) NOT deviceVector: host-filled in Build(), device-read in the solve kernel -- same
  // reason Minv_dev is a Grid::Vector (a device-only deviceVector host-write segfaults).
  Grid::Vector<CoreScalar> m_Minv_core;// per momentum q, n5 x n5 row-major (CoreScalar) -- keyed q==idx4
  bool po_plans_ready = false;
  CorePlan po_f3, po_f1, po_b3, po_b1; // cached fwd/bwd rank-3 (xyz) + rank-1 (t) plans
#endif

#ifdef FREEMOBIUS5D_USE_FP32
  // ---- fp32 on the DEFAULT (barrel) path: precisionChange in/out to single, run phase + barrel FFT +
  // on-device solve entirely in ComplexF. Halves the pack/unpack/solve/cuFFT bytes uniformly
  // (layout-INDEPENDENT ~2x). The OUTER FGMRES stays double, so final accuracy is unchanged (flexible
  // GMRES; grid_freeprec_cost_benchmark_v2_claude.md Section 10). The single grid has its OWN simd layout,
  // so Minv_dev_f is re-scattered with FGrid_f geometry (not FGrid's). Fields are pointers (need FGrid_f,
  // built in Build()). Single rank (matches the benchmark).
  GridCartesian* UGrid_f = nullptr;
  GridCartesian* FGrid_f = nullptr;
  LatticeComplexF* phase_neg_f = nullptr;
  LatticeComplexF* phase_pos_f = nullptr;
  LatticeFermionF* m_in_buf_f = nullptr;
  LatticeFermionF* m_in_k_f = nullptr;
  LatticeFermionF* m_prop_k_f = nullptr;
  LatticeFermionF* m_out_f = nullptr;
  FFT_claude<ComplexF>* m_fft_f = nullptr;
  Grid::Vector<ComplexF> Minv_dev_f;   // slot-indexed per FGrid_f (oSite4*Nsimd_f + lane), ComplexF
  Grid::Vector<ComplexF> btD_dev_f;     // BT-1 device (fp32), slot-indexed per FGrid_f: 16 / slot
  Grid::Vector<ComplexF> btDinv_dev_f;  // Ls*16 / slot
  Grid::Vector<ComplexF> btCminv_dev_f; // 16 / slot
  typedef typename LatticeFermionF::scalar_object SiteSpinorF;
  // Precomputed precisionChange coordinate maps -- the plain precisionChange(out,in) REBUILDS this map
  // (host loop over lSites + device alloc/copy) EVERY call (~790 us/apply for the pair). Cache once in
  // Build(), reuse via the workspace overload. (F<->D grids differ in Nsimd -> the same-grid fast path
  // never applies; the reshuffle itself is a cheap accelerator_for, the map-build was the cost.)
  precisionChangeWorkspace* pc_in_ws = nullptr;   // in : FGrid (double) -> FGrid_f (single)
  precisionChangeWorkspace* pc_out_ws = nullptr;  // out: FGrid_f (single) -> FGrid (double)
#endif

  // apply-level timers (usecond, always-on; accumulate over applies -- profile the FFT vs the
  // block solve under MPI decomposition; report via report_timers()). FFT internal comm/kernel
  // split (t_shift/t_fft) comes separately from FFT.h under --log Performance.
  double t_phase = 0.0;
  double t_fft_fwd = 0.0;
  double t_solve = 0.0;
  double t_fft_bwd = 0.0;
  long n_apply = 0;

  FreeMobius5DInverse(GridCartesian* FGrid_, int Ls_, double M5_, double b_, double c_, double mass_,
                      std::vector<Complex> boundary_)
    : FGrid(FGrid_), Ls(Ls_), M5(M5_), b(b_), c(c_), mass(mass_), boundary(boundary_),
      phase_neg(FGrid_), phase_pos(FGrid_),
      m_in_buf(FGrid_), m_in_k(FGrid_), m_prop_k(FGrid_), m_fft(FGrid_) {
    Build();
  }

  void Build() {
    assert((int)boundary.size() == Nd);
    // effective twist including the anti-periodic boundary phase
    std::vector<double> tw(Nd, 0.0);
    for (int mu = 0; mu < Nd; ++mu) {
      double bph = std::acos(real(boundary[mu]));  // Grid `real`, not std:: (Complex may be thrust::complex)
      tw[mu] = bph / (2.0 * M_PI);
    }
    // MPI-correct momentum grid: each rank owns a LOCAL 4D sub-block. The momentum VALUE at a site
    // uses the GLOBAL coordinate kglob = rank_offset + local_coord and the GLOBAL extent Lg; only
    // the Minv INDEXING is local. At --mpi 1.1.1.1 (Ll == Lg, off == 0) this reduces to the old
    // single-rank build bit-for-bit. Local 4D lex has x fastest (idx4 = kx + Ll[0]*(ky + ...)),
    // matching Grid IndexFromCoor / the i5 = s + Ls*idx4 order that unvectorizeToLexOrdArray
    // produces on the [Ls,Lx,Ly,Lz,Lt] grid (dim 0 = s fastest).
    const Coordinate& fdim = FGrid->_fdimensions;    // global 5D dims [Ls, Lx, Ly, Lz, Lt]
    const Coordinate& lodim = FGrid->_ldimensions;   // local 5D dims (this rank)
    const Coordinate& pcoor = FGrid->_processor_coor;
    int Lg[4] = {fdim[1], fdim[2], fdim[3], fdim[4]};
    int Ll[4] = {lodim[1], lodim[2], lodim[3], lodim[4]};
    int off[4];
    for (int mu = 0; mu < 4; ++mu) {
      off[mu] = pcoor[mu + 1] * Ll[mu];
    }
    V4loc = Ll[0] * Ll[1] * Ll[2] * Ll[3];
    FreeMobius5DBlock blk(Ls, M5, b, c, mass);
    Minv.resize(V4loc);
    bt_D.resize(V4loc);
    bt_Dinv.resize(V4loc);
    bt_Cminv.resize(V4loc);
    for (int kt = 0; kt < Ll[3]; ++kt) {
      for (int kz = 0; kz < Ll[2]; ++kz) {
        for (int ky = 0; ky < Ll[1]; ++ky) {
          for (int kx = 0; kx < Ll[0]; ++kx) {
            int kc[4] = {kx, ky, kz, kt};
            std::array<double, 4> p;
            for (int mu = 0; mu < 4; ++mu) {
              int kglob = off[mu] + kc[mu];
              p[mu] = 2.0 * M_PI * (kglob + tw[mu]) / Lg[mu];
            }
            int idx = ((kt * Ll[2] + kz) * Ll[1] + ky) * Ll[0] + kx;
            Minv[idx] = blk.build_block(p).inverse();
            std::array<std::complex<double>, 16> D;
            blk.free_dw_p(p, D);
            build_bt_momentum(idx, D);  // block-Thomas pivots + G_bt + Cm_inv (uses free_dw_p, same convention)
          }
        }
      }
    }
    bt_gate();  // BT-0: assert block-Thomas host apply == dense Minv on a few momenta

    // ---- scatter Minv into the device-resident, slot-indexed Minv_dev (Chunk 1 of the on-device
    // batched solve). Slot = oSite4*Nsimd + lane. The local 4D coord of a (oSite4, lane) is
    // lcoor4 = ocoor4 + rdim4 * icoor4 (Grid's lane<->coord convention, same as unvectorizeToLexOrdArray),
    // and idx4 = local lex (x fastest) matching the Minv build above.
    const int n5 = 4 * Ls;
    const Coordinate& rdim5 = FGrid->_rdimensions;   // [Ls, rx, ry, rz, rt] (s: rdim=Ls, simd=1)
    const Coordinate& simd5 = FGrid->_simd_layout;   // [1, sx, sy, sz, st]
    Coordinate rdim4(4);
    Coordinate simd4(4);
    for (int mu = 0; mu < 4; ++mu) {
      rdim4[mu] = rdim5[mu + 1];
      simd4[mu] = simd5[mu + 1];
    }
    int Nsimd = 1;
    int nOsites4 = 1;
    for (int mu = 0; mu < 4; ++mu) {
      Nsimd *= simd4[mu];
      nOsites4 *= rdim4[mu];
    }
    Minv_dev.resize((size_t)nOsites4 * Nsimd * n5 * n5);
    btD_dev.resize((size_t)nOsites4 * Nsimd * 16);
    btDinv_dev.resize((size_t)nOsites4 * Nsimd * Ls * 16);
    btCminv_dev.resize((size_t)nOsites4 * Nsimd * 16);
    for (int oSite4 = 0; oSite4 < nOsites4; ++oSite4) {
      Coordinate ocoor4(4);
      Lexicographic::CoorFromIndex(ocoor4, oSite4, rdim4);
      for (int lane = 0; lane < Nsimd; ++lane) {
        Coordinate icoor4(4);
        Lexicographic::CoorFromIndex(icoor4, lane, simd4);
        int lcoor4[4];
        for (int mu = 0; mu < 4; ++mu) {
          lcoor4[mu] = ocoor4[mu] + rdim4[mu] * icoor4[mu];
        }
#ifdef FREEMOBIUS5D_BLOCKING_FFT
        // The SIMD-blocking FFT places t-momentum kt = icoor_t + sd*ocoor_t at slot (ocoor_t, icoor_t)
        // (digit-reversal vs Grid's freq=coord). Re-key the t-component so Minv_dev matches that ordering;
        // x,y,z are sd=1 -> unchanged. See grid_blocking_fft_impl_plan_claude.md.
        lcoor4[3] = icoor4[3] + simd4[3] * ocoor4[3];
#endif
        int idx4 = ((lcoor4[3] * Ll[2] + lcoor4[2]) * Ll[1] + lcoor4[1]) * Ll[0] + lcoor4[0];
        const Eigen::MatrixXcd& Mi = Minv[idx4];
        size_t base = ((size_t)oSite4 * Nsimd + lane) * n5 * n5;
        for (int r = 0; r < n5; ++r) {
          for (int cc = 0; cc < n5; ++cc) {
            std::complex<double> z = Mi(r, cc);
            Minv_dev[base + (size_t)r * n5 + cc] = Grid::Complex(z.real(), z.imag());
          }
        }
        // block-Thomas per-momentum data (double), same slot
        size_t slot = (size_t)oSite4 * Nsimd + lane;
        for (int k = 0; k < 16; ++k) {
          btD_dev[slot * 16 + k] = Grid::Complex(bt_D[idx4][k].real(), bt_D[idx4][k].imag());
          btCminv_dev[slot * 16 + k] = Grid::Complex(bt_Cminv[idx4][k].real(), bt_Cminv[idx4][k].imag());
        }
        for (int k = 0; k < Ls * 16; ++k) {
          btDinv_dev[slot * Ls * 16 + k] = Grid::Complex(bt_Dinv[idx4][k].real(), bt_Dinv[idx4][k].imag());
        }
      }
    }

#ifdef FREEMOBIUS5D_PACKONCE
    // ---- pack-once geometry + a CoreScalar, momentum-keyed (q == idx4) copy of the block inverses, and
    // the contiguous buffer. Single rank ONLY: local extents == global. comp layout (s,spin,colour) with
    // colour fastest so the solve gathers (s,spin) at stride Nc*V4 for a fixed (spacetime, colour).
    for (int mu = 0; mu < 5; ++mu) {
      assert(FGrid->_processors[mu] == 1 && "FREEMOBIUS5D_PACKONCE is single-rank only");
    }
    po_Lx = Ll[0];
    po_Ly = Ll[1];
    po_Lz = Ll[2];
    po_Lt = Ll[3];
    po_V4 = V4loc;
    po_Lzyx = po_Lx * po_Ly * po_Lz;
    po_n5 = 4 * Ls;
    assert(po_n5 <= FREEMOBIUS5D_PO_NMAX && "raise FREEMOBIUS5D_PO_NMAX for this Ls");
    po_Ncomp = Ls * 4 * Nc;
    m_B.resize((size_t)po_Ncomp * po_V4);
    m_Minv_core.resize((size_t)po_V4 * po_n5 * po_n5);
    for (int q = 0; q < po_V4; ++q) {
      const Eigen::MatrixXcd& Mi = Minv[q];  // q == idx4 (x fastest) by construction
      size_t base = (size_t)q * po_n5 * po_n5;
      for (int r = 0; r < po_n5; ++r) {
        for (int cc = 0; cc < po_n5; ++cc) {
          std::complex<double> z = Mi(r, cc);
          m_Minv_core[base + (size_t)r * po_n5 + cc] = CoreScalar(z.real(), z.imag());
        }
      }
    }
#endif

    // ---- precompute the constant twist phase fields (once). ph = sum_nu bph_nu * x_nu / L_nu, with
    // bph_nu = acos(Re boundary_nu) the anti-periodic boundary phase; s = dim 0 so spacetime is dim 1..4.
    ComplexField coor(FGrid);
    ComplexField ph(FGrid);
    ph = Zero();
    ComplexD ci(0.0, 1.0);
    int shift = 1;  // fiveD: s is dim 0
    for (int nu = 0; nu < Nd; ++nu) {
      LatticeCoordinate(coor, nu + shift);
      double bph = std::acos(real(boundary[nu]));  // Grid `real`, not std:: (Complex may be thrust::complex)
      ph = ph + bph * coor * (1.0 / (double)(FGrid->_fdimensions[nu + shift]));
    }
    phase_neg = exp(ci * ph * (-1.0));
    phase_pos = exp(ci * ph);

#ifdef FREEMOBIUS5D_USE_FP32
    // ---- fp32-default machinery: single-precision 5D grid (its OWN simd layout), single scratch fields,
    // single phase (precisionChange from the double fields), and Minv_dev_f re-scattered with FGrid_f
    // geometry. Minv (idx4-keyed Eigen inverses) is grid-independent -> reuse it, cast to ComplexF.
    {
      Coordinate latt4(4);
      Coordinate mpi4(4);
      for (int mu = 0; mu < 4; ++mu) {
        latt4[mu] = FGrid->_fdimensions[mu + 1];
        mpi4[mu] = FGrid->_processors[mu + 1];
      }
      Coordinate simd_f = GridDefaultSimd(4, vComplexF::Nsimd());
      UGrid_f = SpaceTimeGrid::makeFourDimGrid(latt4, simd_f, mpi4);
      FGrid_f = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid_f);
      phase_neg_f = new LatticeComplexF(FGrid_f);
      phase_pos_f = new LatticeComplexF(FGrid_f);
      m_in_buf_f = new LatticeFermionF(FGrid_f);
      m_in_k_f = new LatticeFermionF(FGrid_f);
      m_prop_k_f = new LatticeFermionF(FGrid_f);
      m_out_f = new LatticeFermionF(FGrid_f);
      m_fft_f = new FFT_claude<ComplexF>(FGrid_f);
      precisionChange(*phase_neg_f, phase_neg);
      precisionChange(*phase_pos_f, phase_pos);
      pc_in_ws = new precisionChangeWorkspace(FGrid_f, FGrid);   // out=single(FGrid_f), in=double(FGrid)
      pc_out_ws = new precisionChangeWorkspace(FGrid, FGrid_f);  // out=double(FGrid), in=single(FGrid_f)

      const int n5 = 4 * Ls;
      const Coordinate& rdim5f = FGrid_f->_rdimensions;
      const Coordinate& simd5f = FGrid_f->_simd_layout;
      Coordinate rdim4f(4);
      Coordinate simd4f(4);
      for (int mu = 0; mu < 4; ++mu) {
        rdim4f[mu] = rdim5f[mu + 1];
        simd4f[mu] = simd5f[mu + 1];
      }
      int Nsimd_f = 1;
      int nOsites4_f = 1;
      for (int mu = 0; mu < 4; ++mu) {
        Nsimd_f *= simd4f[mu];
        nOsites4_f *= rdim4f[mu];
      }
      Minv_dev_f.resize((size_t)nOsites4_f * Nsimd_f * n5 * n5);
      btD_dev_f.resize((size_t)nOsites4_f * Nsimd_f * 16);
      btDinv_dev_f.resize((size_t)nOsites4_f * Nsimd_f * Ls * 16);
      btCminv_dev_f.resize((size_t)nOsites4_f * Nsimd_f * 16);
      for (int oSite4 = 0; oSite4 < nOsites4_f; ++oSite4) {
        Coordinate ocoor4(4);
        Lexicographic::CoorFromIndex(ocoor4, oSite4, rdim4f);
        for (int lane = 0; lane < Nsimd_f; ++lane) {
          Coordinate icoor4(4);
          Lexicographic::CoorFromIndex(icoor4, lane, simd4f);
          int lcoor4[4];
          for (int mu = 0; mu < 4; ++mu) {
            lcoor4[mu] = ocoor4[mu] + rdim4f[mu] * icoor4[mu];
          }
          int idx4 = ((lcoor4[3] * Ll[2] + lcoor4[2]) * Ll[1] + lcoor4[1]) * Ll[0] + lcoor4[0];
          const Eigen::MatrixXcd& Mi = Minv[idx4];
          size_t base = ((size_t)oSite4 * Nsimd_f + lane) * n5 * n5;
          for (int r = 0; r < n5; ++r) {
            for (int cc = 0; cc < n5; ++cc) {
              std::complex<double> z = Mi(r, cc);
              Minv_dev_f[base + (size_t)r * n5 + cc] = ComplexF((float)z.real(), (float)z.imag());
            }
          }
          size_t slot = (size_t)oSite4 * Nsimd_f + lane;
          for (int k = 0; k < 16; ++k) {
            btD_dev_f[slot * 16 + k] = ComplexF((float)bt_D[idx4][k].real(), (float)bt_D[idx4][k].imag());
            btCminv_dev_f[slot * 16 + k] = ComplexF((float)bt_Cminv[idx4][k].real(), (float)bt_Cminv[idx4][k].imag());
          }
          for (int k = 0; k < Ls * 16; ++k) {
            btDinv_dev_f[slot * Ls * 16 + k] = ComplexF((float)bt_Dinv[idx4][k].real(), (float)bt_Dinv[idx4][k].imag());
          }
        }
      }
    }
#endif
  }

#ifdef FREEMOBIUS5D_PACKONCE
  // Pack-once fused apply (SINGLE RANK). One pack (Grid SIMD -> contiguous m_B) + fused contiguous cuFFT
  // (rank-3 xyz + rank-1 t) + block solve IN m_B + one unpack. accelerator_barrier() fences the cuFFT
  // (default stream) against the pack/solve/unpack accelerator_for's. Timers reuse the 4 members:
  // t_phase (phase), t_fft_fwd (pack+fwd cuFFT), t_solve (solve), t_fft_bwd (bwd cuFFT+unpack).
  void apply_packonce(const FermionField& in, FermionField& out) {
    FermionField& in_buf = m_in_buf;

    double tp = -usecond();
    in_buf = phase_neg * in;
    tp += usecond();
    t_phase += tp;

    // ---- lazy cached plans (cuFFT ignores the buffer pointer until exec; created once). rank-3 (x,y,z):
    // n={Lz,Ly,Lx}, howmany=Ncomp*Lt contiguous batches (idist=Lzyx, istride=1). rank-1 (t): per comp a
    // batch of Lzyx t-columns (istride=Lzyx, idist=1), looped over comps with a pointer offset.
    if (!po_plans_ready) {
      CoreFFTWScalar* p0 = (CoreFFTWScalar*)&m_B[0];
      int n3[3] = {po_Lz, po_Ly, po_Lx};
      int n1[1] = {po_Lt};
      po_f3 = FFTW<CoreScalar>::fftw_plan_many_dft(3, n3, po_Ncomp * po_Lt,
                                                   p0, n3, 1, po_Lzyx,
                                                   p0, n3, 1, po_Lzyx,
                                                   FFTW_FORWARD, FFTW_ESTIMATE);
      po_b3 = FFTW<CoreScalar>::fftw_plan_many_dft(3, n3, po_Ncomp * po_Lt,
                                                   p0, n3, 1, po_Lzyx,
                                                   p0, n3, 1, po_Lzyx,
                                                   FFTW_BACKWARD, FFTW_ESTIMATE);
      po_f1 = FFTW<CoreScalar>::fftw_plan_many_dft(1, n1, po_Lzyx,
                                                   p0, n1, po_Lzyx, 1,
                                                   p0, n1, po_Lzyx, 1,
                                                   FFTW_FORWARD, FFTW_ESTIMATE);
      po_b1 = FFTW<CoreScalar>::fftw_plan_many_dft(1, n1, po_Lzyx,
                                                   p0, n1, po_Lzyx, 1,
                                                   p0, n1, po_Lzyx, 1,
                                                   FFTW_BACKWARD, FFTW_ESTIMATE);
      po_plans_ready = true;
    }

    Coordinate rdim5 = FGrid->_rdimensions;
    Coordinate simd5 = FGrid->_simd_layout;
    int Lx = po_Lx;
    int Ly = po_Ly;
    int Lz = po_Lz;
    int V4 = po_V4;
    int Ncc = Nc;
    int Nsimd = (int)FGrid->Nsimd();
    uint64_t oS = FGrid->oSites();

    // ---- PACK once: Grid SIMD field -> contiguous m_B[comp][kt][kz][ky][kx], comp=(s*4+spin)*Nc+colour.
    double tf = -usecond();
    {
      autoView(in_v, in_buf, AcceleratorRead);
      CoreScalar* Bp = &m_B[0];
      accelerator_for(oSite5, oS, 1, {
        Coordinate ocoor(5);
        Lexicographic::CoorFromIndex(ocoor, oSite5, rdim5);
        for (int lane = 0; lane < Nsimd; ++lane) {
          Coordinate icoor(5);
          Lexicographic::CoorFromIndex(icoor, lane, simd5);
          int s = ocoor[0] + rdim5[0] * icoor[0];
          int x = ocoor[1] + rdim5[1] * icoor[1];
          int y = ocoor[2] + rdim5[2] * icoor[2];
          int z = ocoor[3] + rdim5[3] * icoor[3];
          int t = ocoor[4] + rdim5[4] * icoor[4];
          int q = x + Lx * (y + Ly * (z + Lz * t));
          SiteSpinor sp = extractLane(lane, in_v[oSite5]);
          for (int a = 0; a < 4; ++a) {
            for (int col = 0; col < Ncc; ++col) {
              ComplexD zz = sp()(a)(col);
              int comp = (s * 4 + a) * Ncc + col;
              Bp[(size_t)comp * V4 + q] = CoreScalar(zz.real(), zz.imag());
            }
          }
        }
      });
    }

    // ---- forward FFT on m_B (fenced against the pack).
    accelerator_barrier();
    {
      CoreFFTWScalar* Bp = (CoreFFTWScalar*)&m_B[0];
      FFTW<CoreScalar>::fftw_execute_dft(po_f3, Bp, Bp, FFTW_FORWARD);
      for (int comp = 0; comp < po_Ncomp; ++comp) {
        CoreFFTWScalar* pc = Bp + (size_t)comp * po_Lt * po_Lzyx;
        FFTW<CoreScalar>::fftw_execute_dft(po_f1, pc, pc, FFTW_FORWARD);
      }
    }
    accelerator_barrier();
    tf += usecond();
    t_fft_fwd += tf;

    // ---- block solve IN m_B: per (momentum q, colour), gather the n5=(4Ls) (s,spin) values at stride
    // Nc*V4, multiply by Minv_core[q] (n5 x n5), scatter back. q == idx4 keys Minv by construction.
    double ts = -usecond();
    {
      CoreScalar* Bp = &m_B[0];
      const CoreScalar* Mp = &m_Minv_core[0];
      int n5 = po_n5;
      uint64_t nqc = (uint64_t)po_V4 * Ncc;
      accelerator_for(qc, nqc, 1, {
        int q = (int)(qc / Ncc);
        int col = (int)(qc % Ncc);
        const CoreScalar* M = Mp + (size_t)q * n5 * n5;
        CoreScalar xin[FREEMOBIUS5D_PO_NMAX];
        for (int j = 0; j < n5; ++j) {
          int comp = j * Ncc + col;  // j = s*4+spin, so comp = (s*4+spin)*Nc + col
          xin[j] = Bp[(size_t)comp * V4 + q];
        }
        for (int r = 0; r < n5; ++r) {
          CoreScalar acc = CoreScalar(0, 0);
          for (int k = 0; k < n5; ++k) {
            acc = acc + M[r * n5 + k] * xin[k];
          }
          int comp = r * Ncc + col;
          Bp[(size_t)comp * V4 + q] = acc;
        }
      });
    }
    ts += usecond();
    t_solve += ts;

    // ---- backward FFT on m_B (fenced), then unpack once with the 1/(Lx Ly Lz Lt) scale + phase_pos.
    double tb = -usecond();
    accelerator_barrier();
    {
      CoreFFTWScalar* Bp = (CoreFFTWScalar*)&m_B[0];
      FFTW<CoreScalar>::fftw_execute_dft(po_b3, Bp, Bp, FFTW_BACKWARD);
      for (int comp = 0; comp < po_Ncomp; ++comp) {
        CoreFFTWScalar* pc = Bp + (size_t)comp * po_Lt * po_Lzyx;
        FFTW<CoreScalar>::fftw_execute_dft(po_b1, pc, pc, FFTW_BACKWARD);
      }
    }
    accelerator_barrier();

    double scale = 1.0 / ((double)po_Lx * po_Ly * po_Lz * po_Lt);
    {
      autoView(out_v, out, AcceleratorWrite);
      const CoreScalar* Bp = &m_B[0];
      accelerator_for(oSite5, oS, 1, {
        Coordinate ocoor(5);
        Lexicographic::CoorFromIndex(ocoor, oSite5, rdim5);
        for (int lane = 0; lane < Nsimd; ++lane) {
          Coordinate icoor(5);
          Lexicographic::CoorFromIndex(icoor, lane, simd5);
          int s = ocoor[0] + rdim5[0] * icoor[0];
          int x = ocoor[1] + rdim5[1] * icoor[1];
          int y = ocoor[2] + rdim5[2] * icoor[2];
          int z = ocoor[3] + rdim5[3] * icoor[3];
          int t = ocoor[4] + rdim5[4] * icoor[4];
          int q = x + Lx * (y + Ly * (z + Lz * t));
          SiteSpinor w;
          w = Zero();
          for (int a = 0; a < 4; ++a) {
            for (int col = 0; col < Ncc; ++col) {
              int comp = (s * 4 + a) * Ncc + col;
              CoreScalar v = Bp[(size_t)comp * V4 + q];
              w()(a)(col) = ComplexD((double)v.real() * scale, (double)v.imag() * scale);
            }
          }
          insertLane(lane, out_v[oSite5], w);
        }
      });
    }
    tb += usecond();
    t_fft_bwd += tb;

    double tp2 = -usecond();
    out = out * phase_pos;
    tp2 += usecond();
    t_phase += tp2;

    n_apply++;
  }
#endif

#ifdef FREEMOBIUS5D_USE_FP32
  // fp32 default apply: precisionChange to single, run the whole barrel pipeline in ComplexF, change back.
  // Timers: t_phase (precisionChange in/out + phase), t_fft_fwd, t_solve, t_fft_bwd.
  void apply_fp32(const FermionField& in, FermionField& out) {
    Coordinate mask(Nd + 1, 1);
    mask[0] = 0;

    double tp = -usecond();
    precisionChange(*m_in_buf_f, in, *pc_in_ws);  // cached workspace (no per-call map rebuild)
    *m_in_buf_f = (*phase_neg_f) * (*m_in_buf_f);
    tp += usecond();
    t_phase += tp;

    double tf = -usecond();
    m_fft_f->FFT_dim_mask(*m_in_k_f, *m_in_buf_f, mask, FFT::forward);
    tf += usecond();
    t_fft_fwd += tf;

    double ts = -usecond();
#ifdef FREEMOBIUS5D_BLOCK_THOMAS
    MomentumSpaceSolve_bt_dev_f(*m_prop_k_f, *m_in_k_f);   // block-Thomas (GPU LOSS; -DFREEMOBIUS5D_BLOCK_THOMAS)
#else
    MomentumSpaceSolve_dev_f(*m_prop_k_f, *m_in_k_f);      // dense on-device (DEFAULT; fastest on GPU)
#endif
    ts += usecond();
    t_solve += ts;

    double tb = -usecond();
    m_fft_f->FFT_dim_mask(*m_out_f, *m_prop_k_f, mask, FFT::backward);
    tb += usecond();
    t_fft_bwd += tb;

    double tp2 = -usecond();
    *m_out_f = (*m_out_f) * (*phase_pos_f);
    precisionChange(out, *m_out_f, *pc_out_ws);  // cached workspace (no per-call map rebuild)
    tp2 += usecond();
    t_phase += tp2;

    n_apply++;
  }
#endif

  // out = F in = D_DW^free(m)^{-1} in
  virtual void operator()(const FermionField& in, FermionField& out) {
#ifdef FREEMOBIUS5D_PACKONCE
    apply_packonce(in, out);
    return;
#endif
#ifdef FREEMOBIUS5D_USE_FP32
    apply_fp32(in, out);
    return;
#endif
    // (a) reuse the hoisted member scratch buffers (no per-apply allocation).
    FermionField& in_buf = m_in_buf;
    FermionField& in_k = m_in_k;
    FermionField& prop_k = m_prop_k;
    Coordinate mask(Nd + 1, 1);
    mask[0] = 0;  // do not FFT the s-dimension
#ifdef FREEMOBIUS5D_BLOCKING_FFT
    BlockingFFT<FermionField> bfft(FGrid);  // SIMD-blocking FFT (no pack); Minv_dev is re-keyed
#elif defined(FREEMOBIUS5D_GRID_FFT)
    FFT theFFT(FGrid);                      // Grid's UNCACHED FFT (A/B baseline)
#endif

    // phase: precomputed constant twist fields (Build()); per-apply is a pointwise multiply.
    double tp = -usecond();
    in_buf = phase_neg * in;
    tp += usecond();
    t_phase += tp;

    double tf = -usecond();
#ifdef FREEMOBIUS5D_BLOCKING_FFT
    bfft.FFT_spacetime(in_k, in_buf, FFT::forward);
#elif defined(FREEMOBIUS5D_GRID_FFT)
    theFFT.FFT_dim_mask(in_k, in_buf, mask, FFT::forward);
#else
    m_fft.FFT_dim_mask(in_k, in_buf, mask, FFT::forward);  // DEFAULT: cached plan/pgbuf
#endif
    tf += usecond();
    t_fft_fwd += tf;

    double ts = -usecond();
#ifdef FREEMOBIUS5D_HOST_SOLVE
    MomentumSpaceSolve(prop_k, in_k);        // host Eigen path, A/B via -DFREEMOBIUS5D_HOST_SOLVE
#elif defined(FREEMOBIUS5D_BLOCK_THOMAS)
    MomentumSpaceSolve_bt_dev(prop_k, in_k); // block-Thomas O(Ls): CORRECT but a GPU LOSS (solve 2.3-5.4x
                                             // slower -- register spill; -DFREEMOBIUS5D_BLOCK_THOMAS). CPU win.
#else
    MomentumSpaceSolve_dev(prop_k, in_k);    // dense on-device batched matvec (DEFAULT; fastest on GPU)
#endif
    ts += usecond();
    t_solve += ts;

    double tb = -usecond();
#ifdef FREEMOBIUS5D_BLOCKING_FFT
    bfft.FFT_spacetime(out, prop_k, FFT::backward);
#elif defined(FREEMOBIUS5D_GRID_FFT)
    theFFT.FFT_dim_mask(out, prop_k, mask, FFT::backward);
#else
    m_fft.FFT_dim_mask(out, prop_k, mask, FFT::backward);  // DEFAULT: cached plan/pgbuf
#endif
    tb += usecond();
    t_fft_bwd += tb;

    double tp2 = -usecond();
    out = out * phase_pos;
    tp2 += usecond();
    t_phase += tp2;

    n_apply++;
  }

  // Per-apply timing averages (microseconds), plus the FFT fraction of the F apply. The FFT's own
  // comm-vs-kernel split (t_shift vs t_fft) is emitted by FFT.h under --log Performance.
  void report_timers() const {
    if (n_apply == 0) {
      return;
    }
    double n = (double)n_apply;
    double tot = t_phase + t_fft_fwd + t_solve + t_fft_bwd;
    std::cout << GridLogMessage << "[F timers] " << n_apply << " applies, avg us/apply:" << std::endl;
    std::cout << GridLogMessage << "  phase   " << t_phase / n << std::endl;
    std::cout << GridLogMessage << "  fft_fwd " << t_fft_fwd / n << std::endl;
    std::cout << GridLogMessage << "  solve   " << t_solve / n << std::endl;
    std::cout << GridLogMessage << "  fft_bwd " << t_fft_bwd / n << std::endl;
    std::cout << GridLogMessage << "  total   " << tot / n << std::endl;
    std::cout << GridLogMessage << "  FFT fraction (fwd+bwd)/total = " << (t_fft_fwd + t_fft_bwd) / tot << std::endl;
#ifdef FFT_CLAUDE_PROFILE
    m_fft.report();  // internal pack/fftk/unpack split -> picks lever 1 (fp32) vs lever 2 (fuse passes)
#endif
  }

  void reset_timers() {
    t_phase = 0.0;
    t_fft_fwd = 0.0;
    t_solve = 0.0;
    t_fft_bwd = 0.0;
    n_apply = 0;
  }

  // ===== block-Thomas per-momentum solve (BT-0: host precompute + reference + gate) =====
  // T = D_DW(p,m=0) block-tridiagonal; diag A = b D_W + I, super Uup = (c D_W - I)P_- (cols 2,3),
  // sub Udn = (c D_W - I)P_+ (cols 0,1). D_DW(m) = T - m L R (rank-4 antiperiodic corner). See
  // grid_block_thomas_impl_plan_claude.md. C = std::complex<double>.
  void bt_uup_udn(const std::array<std::complex<double>, 16>& D,
                  std::complex<double>* Uup, std::complex<double>* Udn) const {
    for (int a = 0; a < 4; ++a) {
      for (int a2 = 0; a2 < 4; ++a2) {
        std::complex<double> hop = std::complex<double>(c, 0.0) * D[a * 4 + a2];
        if (a == a2) {
          hop -= std::complex<double>(1.0, 0.0);
        }
        Uup[a * 4 + a2] = (a2 >= 2) ? hop : std::complex<double>(0.0, 0.0);
        Udn[a * 4 + a2] = (a2 < 2) ? hop : std::complex<double>(0.0, 0.0);
      }
    }
  }

  // y = T^{-1} r (O(Ls) forward + backward block sweeps; no inversion in the apply). Needs bt_D[idx],
  // bt_Dinv[idx] set.
  void block_thomas_solve_host(int idx, const std::complex<double>* r, std::complex<double>* y) const {
    const int n5 = 4 * Ls;
    std::array<std::complex<double>, 16> Uup;
    std::array<std::complex<double>, 16> Udn;
    bt_uup_udn(bt_D[idx], Uup.data(), Udn.data());
    std::vector<std::complex<double>> rho(r, r + n5);
    for (int s = 1; s < Ls; ++s) {
      const std::complex<double>* Dprev = &bt_Dinv[idx][(s - 1) * 16];
      std::complex<double> t[4];
      for (int a = 0; a < 4; ++a) {
        std::complex<double> acc(0.0, 0.0);
        for (int bb = 0; bb < 4; ++bb) {
          acc += Dprev[a * 4 + bb] * rho[(s - 1) * 4 + bb];
        }
        t[a] = acc;
      }
      for (int a = 0; a < 4; ++a) {
        std::complex<double> acc(0.0, 0.0);
        for (int bb = 0; bb < 4; ++bb) {
          acc += Udn[a * 4 + bb] * t[bb];
        }
        rho[s * 4 + a] -= acc;
      }
    }
    const std::complex<double>* Dlast = &bt_Dinv[idx][(Ls - 1) * 16];
    for (int a = 0; a < 4; ++a) {
      std::complex<double> acc(0.0, 0.0);
      for (int bb = 0; bb < 4; ++bb) {
        acc += Dlast[a * 4 + bb] * rho[(Ls - 1) * 4 + bb];
      }
      y[(Ls - 1) * 4 + a] = acc;
    }
    for (int s = Ls - 2; s >= 0; --s) {
      std::complex<double> tmp[4];
      for (int a = 0; a < 4; ++a) {
        std::complex<double> acc = rho[s * 4 + a];
        for (int bb = 0; bb < 4; ++bb) {
          acc -= Uup[a * 4 + bb] * y[(s + 1) * 4 + bb];
        }
        tmp[a] = acc;
      }
      const std::complex<double>* Ds = &bt_Dinv[idx][s * 16];
      for (int a = 0; a < 4; ++a) {
        std::complex<double> acc(0.0, 0.0);
        for (int bb = 0; bb < 4; ++bb) {
          acc += Ds[a * 4 + bb] * tmp[bb];
        }
        y[s * 4 + a] = acc;
      }
    }
  }

  // Precompute per momentum: bt_D, the Ls pivot inverses bt_Dinv, and bt_Cminv = (I4 - m G_bt)^{-1}.
  void build_bt_momentum(int idx, const std::array<std::complex<double>, 16>& D) {
    const int n5 = 4 * Ls;
    bt_D[idx] = D;
    std::array<std::complex<double>, 16> Uup;
    std::array<std::complex<double>, 16> Udn;
    bt_uup_udn(D, Uup.data(), Udn.data());
    std::array<std::complex<double>, 16> Amat;
    for (int a = 0; a < 4; ++a) {
      for (int a2 = 0; a2 < 4; ++a2) {
        Amat[a * 4 + a2] = std::complex<double>(b, 0.0) * D[a * 4 + a2]
                         + ((a == a2) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0));
      }
    }
    bt_Dinv[idx].assign((size_t)Ls * 16, std::complex<double>(0.0, 0.0));
    // Delta_0^{-1} = A^{-1}
    Eigen::Matrix4cd A;
    for (int a = 0; a < 4; ++a) {
      for (int a2 = 0; a2 < 4; ++a2) {
        A(a, a2) = Amat[a * 4 + a2];
      }
    }
    Eigen::Matrix4cd Ai = A.inverse();
    for (int a = 0; a < 4; ++a) {
      for (int a2 = 0; a2 < 4; ++a2) {
        bt_Dinv[idx][a * 4 + a2] = Ai(a, a2);
      }
    }
    // Delta_s = A - Udn (Delta_{s-1}^{-1} Uup); store Delta_s^{-1}
    for (int s = 1; s < Ls; ++s) {
      const std::complex<double>* Dprev = &bt_Dinv[idx][(s - 1) * 16];
      std::complex<double> M1[16];
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          std::complex<double> acc(0.0, 0.0);
          for (int kk = 0; kk < 4; ++kk) {
            acc += Dprev[a * 4 + kk] * Uup[kk * 4 + bb];
          }
          M1[a * 4 + bb] = acc;
        }
      }
      Eigen::Matrix4cd Delta;
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          std::complex<double> acc = Amat[a * 4 + bb];
          for (int kk = 0; kk < 4; ++kk) {
            acc -= Udn[a * 4 + kk] * M1[kk * 4 + bb];
          }
          Delta(a, bb) = acc;
        }
      }
      Eigen::Matrix4cd Di = Delta.inverse();
      for (int a = 0; a < 4; ++a) {
        for (int bb = 0; bb < 4; ++bb) {
          bt_Dinv[idx][s * 16 + a * 4 + bb] = Di(a, bb);
        }
      }
    }
    // G_bt = R T^{-1} L (4x4); L column j = (c D_W[:,srcspin_j] - e) in block tslice_j; R reads rrow_i.
    const int srcspin[4] = {2, 3, 0, 1};
    const int tslice[4] = {Ls - 1, Ls - 1, 0, 0};
    const int rrow[4] = {0 * 4 + 2, 0 * 4 + 3, (Ls - 1) * 4 + 0, (Ls - 1) * 4 + 1};
    Eigen::Matrix4cd G;
    for (int j = 0; j < 4; ++j) {
      std::vector<std::complex<double>> Lcol(n5, std::complex<double>(0.0, 0.0));
      int sc = srcspin[j];
      int st = tslice[j];
      for (int a = 0; a < 4; ++a) {
        std::complex<double> hop = std::complex<double>(c, 0.0) * D[a * 4 + sc];
        if (a == sc) {
          hop -= std::complex<double>(1.0, 0.0);
        }
        Lcol[st * 4 + a] = hop;
      }
      std::vector<std::complex<double>> yj(n5, std::complex<double>(0.0, 0.0));
      block_thomas_solve_host(idx, Lcol.data(), yj.data());
      for (int i = 0; i < 4; ++i) {
        G(i, j) = yj[rrow[i]];
      }
    }
    // Cm_inv = (I4 - m G_bt)^{-1} (m fixed in F; m=0-safe -> I4 at m=0)
    Eigen::Matrix4cd Cm;
    for (int a = 0; a < 4; ++a) {
      for (int bb = 0; bb < 4; ++bb) {
        Cm(a, bb) = std::complex<double>(-mass, 0.0) * G(a, bb)
                  + ((a == bb) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0));
      }
    }
    Eigen::Matrix4cd Cmi = Cm.inverse();
    for (int a = 0; a < 4; ++a) {
      for (int bb = 0; bb < 4; ++bb) {
        bt_Cminv[idx][a * 4 + bb] = Cmi(a, bb);
      }
    }
  }

  // Full block-Thomas apply: out = D_DW(m)^{-1} r = y + m T^{-1} L (I4 - m G_bt)^{-1} R y, y = T^{-1} r.
  void apply_bt_host(int idx, const std::complex<double>* r, std::complex<double>* out) const {
    const int n5 = 4 * Ls;
    const int srcspin[4] = {2, 3, 0, 1};
    const int tslice[4] = {Ls - 1, Ls - 1, 0, 0};
    const int rrow[4] = {0 * 4 + 2, 0 * 4 + 3, (Ls - 1) * 4 + 0, (Ls - 1) * 4 + 1};
    std::vector<std::complex<double>> y(n5, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> z(n5, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> Lvec(n5, std::complex<double>(0.0, 0.0));
    block_thomas_solve_host(idx, r, y.data());
    std::complex<double> w[4];
    for (int i = 0; i < 4; ++i) {
      w[i] = y[rrow[i]];
    }
    const std::complex<double>* Cmi = bt_Cminv[idx].data();
    std::complex<double> sc4[4];
    for (int a = 0; a < 4; ++a) {
      std::complex<double> acc(0.0, 0.0);
      for (int bb = 0; bb < 4; ++bb) {
        acc += Cmi[a * 4 + bb] * w[bb];
      }
      sc4[a] = acc;
    }
    const std::array<std::complex<double>, 16>& D = bt_D[idx];
    for (int j = 0; j < 4; ++j) {
      int spn = srcspin[j];
      int st = tslice[j];
      for (int a = 0; a < 4; ++a) {
        std::complex<double> hop = std::complex<double>(c, 0.0) * D[a * 4 + spn];
        if (a == spn) {
          hop -= std::complex<double>(1.0, 0.0);
        }
        Lvec[st * 4 + a] += sc4[j] * hop;
      }
    }
    block_thomas_solve_host(idx, Lvec.data(), z.data());
    for (int k = 0; k < n5; ++k) {
      out[k] = y[k] + std::complex<double>(mass, 0.0) * z[k];
    }
  }

  // BT-0 gate: block-Thomas host apply vs the dense Minv on a few momenta (random r). Must match ~eps.
  void bt_gate() const {
    const int n5 = 4 * Ls;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    double maxrel = 0.0;
    int ncheck = (V4loc < 4) ? V4loc : 4;
    for (int idx = 0; idx < ncheck; ++idx) {
      std::vector<std::complex<double>> r(n5);
      std::vector<std::complex<double>> yb(n5);
      for (int k = 0; k < n5; ++k) {
        r[k] = std::complex<double>(uni(rng), uni(rng));
      }
      apply_bt_host(idx, r.data(), yb.data());
      Eigen::VectorXcd rv(n5);
      for (int k = 0; k < n5; ++k) {
        rv(k) = r[k];
      }
      Eigen::VectorXcd yd = Minv[idx] * rv;
      double num = 0.0;
      double den = 0.0;
      for (int k = 0; k < n5; ++k) {
        num += std::norm(yb[k] - yd(k));
        den += std::norm(yd(k));
      }
      double rel = std::sqrt(num / den);
      if (rel > maxrel) {
        maxrel = rel;
      }
    }
    std::cout << GridLogMessage << "[BT-0 gate] block-Thomas vs dense Minv max rel err = " << maxrel
              << (maxrel < 1e-10 ? "  PASS" : "  FAIL") << std::endl;
  }

  // per-momentum dense block solve (colour-blind), Minv precomputed. Gather/scatter the whole 5D field
  // via BULK host<->device transfers (unvectorize/vectorizeFromLexOrdArray) -- one pair per apply, NOT a
  // peekSite/pokeSite per (site,s) (which is ~V4*Ls*2 device syncs/apply, e.g. ~65k at 8^4 -> ~4.6M over
  // an FGMRES solve: a peekSite storm). unvectorizeToLexOrdArray is a strictly LOCAL (per-rank) transfer
  // over lSites (Lattice_transfer.h:1129, _ldimensions/_rdimensions), so we index over the LOCAL momentum
  // count V4loc. The local 5D lex index has s FASTEST: i5 = s + Ls*idx4, with idx4 the LOCAL 4D lex index
  // (x fastest) that also keys the LOCAL Minv -- matches Grid's IndexFromCoor for the [Ls,Lx,Ly,Lz,Lt]
  // grid (dim 0 = s). Validated by the cold gate staying at machine eps (both single- and multi-rank).
  void MomentumSpaceSolve(FermionField& prop_k, const FermionField& in_k) const {
    int n5 = 4 * Ls;
    int V5 = Ls * V4loc;
    std::vector<SiteSpinor> in_h(V5);
    std::vector<SiteSpinor> out_h(V5);
    unvectorizeToLexOrdArray(in_h, in_k);
    for (int idx4 = 0; idx4 < V4loc; ++idx4) {
      const Eigen::MatrixXcd& Mi = Minv[idx4];
      Eigen::MatrixXcd Fin(n5, Nc);
      for (int s = 0; s < Ls; ++s) {
        int i5 = s + Ls * idx4;
        for (int a = 0; a < 4; ++a) {
          for (int col = 0; col < Nc; ++col) {
            ComplexD z = in_h[i5]()(a)(col);
            Fin(s * 4 + a, col) = std::complex<double>(z.real(), z.imag());
          }
        }
      }
      Eigen::MatrixXcd Y = Mi * Fin;
      for (int s = 0; s < Ls; ++s) {
        int i5 = s + Ls * idx4;
        SiteSpinor w;
        w = Zero();
        for (int a = 0; a < 4; ++a) {
          for (int col = 0; col < Nc; ++col) {
            std::complex<double> y = Y(s * 4 + a, col);
            w()(a)(col) = ComplexD(y.real(), y.imag());
          }
        }
        out_h[i5] = w;
      }
    }
    vectorizeFromLexOrdArray(out_h, prop_k);
  }

  // On-device batched block solve (Chunk 2): SAME per-momentum multiply by the precomputed inverse as
  // MomentumSpaceSolve, but fused into ONE accelerator_for on the device field views IN PLACE -- no
  // unvectorize/vectorize host round-trip (the GPU M0 bottleneck, 76% -- grid_freeprec_cost_benchmark_claude.md).
  // Portable GPU+CPU: nsimd=1 (one thread per 4D oSite); the Nsimd SIMD lanes are handled explicitly via
  // extractLane/insertLane because each lane carries its OWN (4Ls)x(4Ls) matrix (per-lane data breaks the
  // usual coalescedRead vectorization). 5D layout: s = dim 0 (simd=1), so oSite5 = s + Ls*oSite4 (s
  // contiguous). Colour-blind (same matrix per colour). Minv_dev is slot-indexed (oSite4*Nsimd + lane).
  void MomentumSpaceSolve_dev(FermionField& prop_k, const FermionField& in_k) const {
    const int Lsloc = Ls;
    const int n5 = 4 * Lsloc;
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites4 = in_k.Grid()->oSites() / (uint64_t)Lsloc;
    const Grid::Complex* Minv_p = &Minv_dev[0];
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite4, nOsites4, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        const Grid::Complex* M = Minv_p + ((uint64_t)oSite4 * Nsimd + lane) * (uint64_t)n5 * n5;
        for (int s = 0; s < Lsloc; ++s) {
          SiteSpinor acc;
          acc = Zero();
          for (int sp = 0; sp < Lsloc; ++sp) {
            SiteSpinor inp = extractLane(lane, in_v[sp + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              for (int bb = 0; bb < 4; ++bb) {
                Grid::Complex m = M[(s * 4 + a) * n5 + (sp * 4 + bb)];
                for (int col = 0; col < Nc; ++col) {
                  acc()(a)(col) += m * inp()(bb)(col);
                }
              }
            }
          }
          insertLane(lane, out_v[s + Lsloc * oSite4], acc);
        }
      }
    });
  }

  // BT-1 on-device block-Thomas solve (double): per (oSite4, lane, colour) gather the (s,spin) vector,
  // apply T^{-1} + the rank-4 corner (bt_solve_dev + bt_corner_dev), scatter back. Reads O(Ls) per
  // momentum (btD/btDinv/btCminv, slot-indexed) vs the dense (4Ls)^2 Minv -> the memory-bound win. The
  // colour loop does read-modify-write of out_v (each colour writes only its 4 spins) to avoid an outbuf.
  void MomentumSpaceSolve_bt_dev(FermionField& prop_k, const FermionField& in_k) const {
    const int Lsloc = Ls;
    const int n5 = 4 * Lsloc;
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites4 = in_k.Grid()->oSites() / (uint64_t)Lsloc;
    const Grid::Complex* D_p = &btD_dev[0];
    const Grid::Complex* Dinv_p = &btDinv_dev[0];
    const Grid::Complex* Cm_p = &btCminv_dev[0];
    Grid::Complex cC(c, 0.0);
    Grid::Complex mC(mass, 0.0);
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite4, nOsites4, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        const Grid::Complex* D = D_p + ((uint64_t)oSite4 * Nsimd + lane) * 16;
        const Grid::Complex* Dinv = Dinv_p + ((uint64_t)oSite4 * Nsimd + lane) * Lsloc * 16;
        const Grid::Complex* Cm = Cm_p + ((uint64_t)oSite4 * Nsimd + lane) * 16;
        for (int col = 0; col < Nc; ++col) {
          Grid::Complex r[FREEMOBIUS5D_PO_NMAX];
          Grid::Complex y[FREEMOBIUS5D_PO_NMAX];
          for (int s = 0; s < Lsloc; ++s) {
            SiteSpinor inp = extractLane(lane, in_v[s + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              r[s * 4 + a] = inp()(a)(col);
            }
          }
          bt_solve_dev(Lsloc, cC, D, Dinv, r, y);
          bt_corner_dev(Lsloc, cC, mC, D, Dinv, Cm, y, r);  // r = out (reuse r as output buffer)
          for (int s = 0; s < Lsloc; ++s) {
            SiteSpinor w = extractLane(lane, out_v[s + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              w()(a)(col) = r[s * 4 + a];
            }
            insertLane(lane, out_v[s + Lsloc * oSite4], w);
          }
        }
      }
    });
  }

#ifdef FREEMOBIUS5D_USE_FP32
  // Single-precision copy of MomentumSpaceSolve_dev (LatticeFermionF + Minv_dev_f). Identical math, half
  // the bytes. Runs on FGrid_f (its own Nsimd_f); Minv_dev_f is slot-indexed for that grid.
  void MomentumSpaceSolve_dev_f(LatticeFermionF& prop_k, const LatticeFermionF& in_k) const {
    const int Lsloc = Ls;
    const int n5 = 4 * Lsloc;
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites4 = in_k.Grid()->oSites() / (uint64_t)Lsloc;
    const ComplexF* Minv_p = &Minv_dev_f[0];
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite4, nOsites4, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        const ComplexF* M = Minv_p + ((uint64_t)oSite4 * Nsimd + lane) * (uint64_t)n5 * n5;
        for (int s = 0; s < Lsloc; ++s) {
          SiteSpinorF acc;
          acc = Zero();
          for (int sp = 0; sp < Lsloc; ++sp) {
            SiteSpinorF inp = extractLane(lane, in_v[sp + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              for (int bb = 0; bb < 4; ++bb) {
                ComplexF m = M[(s * 4 + a) * n5 + (sp * 4 + bb)];
                for (int col = 0; col < Nc; ++col) {
                  acc()(a)(col) += m * inp()(bb)(col);
                }
              }
            }
          }
          insertLane(lane, out_v[s + Lsloc * oSite4], acc);
        }
      }
    });
  }

  // BT-1 on-device block-Thomas solve (single) -- ComplexF copy of MomentumSpaceSolve_bt_dev on FGrid_f.
  void MomentumSpaceSolve_bt_dev_f(LatticeFermionF& prop_k, const LatticeFermionF& in_k) const {
    const int Lsloc = Ls;
    const int n5 = 4 * Lsloc;
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites4 = in_k.Grid()->oSites() / (uint64_t)Lsloc;
    const ComplexF* D_p = &btD_dev_f[0];
    const ComplexF* Dinv_p = &btDinv_dev_f[0];
    const ComplexF* Cm_p = &btCminv_dev_f[0];
    ComplexF cC((float)c, 0.0f);
    ComplexF mC((float)mass, 0.0f);
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite4, nOsites4, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        const ComplexF* D = D_p + ((uint64_t)oSite4 * Nsimd + lane) * 16;
        const ComplexF* Dinv = Dinv_p + ((uint64_t)oSite4 * Nsimd + lane) * Lsloc * 16;
        const ComplexF* Cm = Cm_p + ((uint64_t)oSite4 * Nsimd + lane) * 16;
        for (int col = 0; col < Nc; ++col) {
          ComplexF r[FREEMOBIUS5D_PO_NMAX];
          ComplexF y[FREEMOBIUS5D_PO_NMAX];
          for (int s = 0; s < Lsloc; ++s) {
            SiteSpinorF inp = extractLane(lane, in_v[s + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              r[s * 4 + a] = inp()(a)(col);
            }
          }
          bt_solve_dev(Lsloc, cC, D, Dinv, r, y);
          bt_corner_dev(Lsloc, cC, mC, D, Dinv, Cm, y, r);
          for (int s = 0; s < Lsloc; ++s) {
            SiteSpinorF w = extractLane(lane, out_v[s + Lsloc * oSite4]);
            for (int a = 0; a < 4; ++a) {
              w()(a)(col) = r[s * 4 + a];
            }
            insertLane(lane, out_v[s + Lsloc * oSite4], w);
          }
        }
      }
    });
  }
#endif
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
  double t_omega = 0.0;         // Omega + Omega^dag colour mat-vec time (us, accumulated)
  double t_free = 0.0;          // inner F apply time (us, accumulated)

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
    double to = -usecond();
    phi = Omega5 * in;      // Omega : colour mat-vec per site (spin untouched)
    to += usecond();
    t_omega += to;

    double tfr = -usecond();
    F(phi, y);             // free Mobius inverse
    tfr += usecond();
    t_free += tfr;

    double to2 = -usecond();
    out = adj(Omega5) * y;  // Omega^dag
    to2 += usecond();
    t_omega += to2;
  }

  // Per-apply averages for M0, then the inner F breakdown (FFT vs block solve).
  void report_timers() const {
    if (n_apply == 0) {
      return;
    }
    double n = (double)n_apply;
    std::cout << GridLogMessage << "[M0 timers] " << n_apply << " applies, avg us/apply:" << std::endl;
    std::cout << GridLogMessage << "  omega (fwd+dag) " << t_omega / n << std::endl;
    std::cout << GridLogMessage << "  F (free inv)    " << t_free / n << std::endl;
    F.report_timers();
  }

  void reset_timers() {
    t_omega = 0.0;
    t_free = 0.0;
    n_apply = 0;
    F.reset_timers();
  }
};

// -------------------- CHUNK 4: M1 = leading D_W (hopping-expansion) correction --------------------
// M1 = Omega^dag { F - F D(tildeA) F } Omega, EXACT operator split tildeA = U^L - 1, so
//   D(tildeA) = D_DW[U^L] - D_free,  U^L = Omega U Omega^dag  (the ORIGINAL config framed, NOT flowed).
// Cheap apply (uses D_free F = I, hence D_free y0 = D_free F phi = phi):
//   phi = Omega in ; y0 = F phi ; tmp = D_DW[U^L] y0 ; w = tmp - phi ; y1 = F w ; out = Omega^dag (y0 - y1).
// Cost per M1 apply: 2 free (FFT) inverses (D_W-free) + ONE D_DW[U^L] apply (= Ls D_W). So UNLIKE M0, M1
// is NOT D_W-free -- the honest metric adds Ls per M1 apply (n_dw) to the outer FGMRES D_W count.
// Idea R. Brower + T. Izubuchi; ref dwf4 GaugeCovFreeDW1 (qed2/dwf4_qcd_claude/dwf4_gaugefix_claude.h:294),
// derivation qed2/dwf4_qcd_claude/global_hopping_claude.md.
template <class Impl>
class FreeLimitPreconditioner1 : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  FreeMobius5DInverse<Impl>& F;
  MobiusFermion<Impl>& Dframed;   // D_DW on the framed config U^L = Omega U Omega^dag
  LatticeColourMatrixD Omega5;    // the 4D frame broadcast onto the 5D grid
  long n_apply;                   // counts M1 applies (= outer FGMRES iters)
  long n_dw;                      // counts D_DW[U^L] applies (each = Ls D_W)

  FreeLimitPreconditioner1(FreeMobius5DInverse<Impl>& F_, const LatticeColourMatrixD& xform4,
                           MobiusFermion<Impl>& Dframed_, GridCartesian* FGrid)
    : F(F_), Dframed(Dframed_), Omega5(FGrid), n_apply(0), n_dw(0) {
    for (int s = 0; s < F.Ls; ++s) {
      InsertSlice(xform4, Omega5, s, 0);  // broadcast Omega(x) onto every s-slice
    }
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y0(in.Grid());
    FermionField tmp(in.Grid());
    FermionField w(in.Grid());
    FermionField y1(in.Grid());
    phi = Omega5 * in;      // Omega : colour mat-vec per site (spin untouched)
    F(phi, y0);            // y0 = F phi
    Dframed.M(y0, tmp);    // tmp = D_DW[U^L] y0
    n_dw++;
    w = tmp - phi;         // D(tildeA) y0 = D_DW[U^L] y0 - D_free y0 = D_DW[U^L] y0 - phi
    F(w, y1);             // y1 = F D(tildeA) y0
    out = adj(Omega5) * (y0 - y1);  // Omega^dag ( y0 - F D(tildeA) F phi )
  }
};

}  // namespace Grid
#endif
