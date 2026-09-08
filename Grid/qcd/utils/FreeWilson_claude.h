#ifndef GRID_FREE_WILSON_CLAUDE_H
#define GRID_FREE_WILSON_CLAUDE_H
//
// FreeWilsonInverse: the free-limit Wilson preconditioner F_W = D_W^free(m)^{-1}, FFT-diagonal.
// The Wilson analogue of FreeMobius5DInverse -- NO 5th dimension, so the per-momentum block is a plain
// 4x4 (spin; colour-diagonal in the free limit) instead of the (4Ls)^2 Mobius block. Hence F_W is
// ~PURE FFT + a trivial per-momentum 4x4 matvec, and the 16^4 Krylov MEMORY WALL (Ls=8) disappears
// (Wilson fermion ~1/8 the DW size). Used to demonstrate the large-L/small-m free-prec win-condition on
// Wilson (RB-CGNE kappa^2 penalty vs the free-prec), then port back to DW. Plan:
// dwf4_qcd_claude/grid_free_wilson_impl_plan_claude.md.
//
// Free Wilson operator (momentum space, DeGrand-Rossi gamma) = FreeMobius5DBlock::free_dw_p with M5=-m:
//   D_W(p) = A(p) I + i sum_mu sin(p_mu) gamma_mu,   A(p) = m + sum_mu (1 - cos p_mu)   [= (4-M5) - sum cos]
// so M5 = -m reproduces Grid's free WilsonFermionD(m) (diagonal 4+m). F_W(p) = D_W(p)^{-1} (Eigen 4x4).
// AP-time BC via the same position-space twist phase as the DW class. MPI-correct (local-V4 keying,
// global-coord momentum -- same as the validated DW port). Double + PlannedFFT (Wilson fits in fp64;
// no fp32/BT/DofFFT needed).
//
// Provenance: free Wilson propagator (textbook); gamma convention reused from FreeMobius5D_claude.h
// (calibrate-to-Grid). Mobius/free-frame lineage: Brower-Neff-Orginos arXiv:1206.5214; Brower+Izubuchi.
//
#include <Grid/Grid.h>                            // brings Grid's bundled Eigen (Grid/Eigen/Dense)
#include <Grid/qcd/utils/FreeMobius5D_claude.h>   // FreeMobius5DBlock::free_dw_p + FreeMobiusGammaDR

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FreeWilsonInverse : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  typedef typename FermionField::scalar_object SiteSpinor;      // = SpinColourVector (scalar)
  typedef typename Impl::ComplexField ComplexField;

  GridCartesian* UGrid;
  double mass;
  std::vector<Complex> boundary;

  std::vector<Eigen::MatrixXcd> Minv;   // per LOCAL 4D momentum, 4x4 = F_W(p)
  int V4loc;
  Grid::Vector<Grid::Complex> Minv_dev; // slot-indexed (oSite*Nsimd + lane), 16 complex / slot
  Grid::Vector<Grid::Complex> bandmask_dev; // slot-indexed scalar chi(A(p)) for the Mx band projector

  ComplexField phase_neg;
  ComplexField phase_pos;
  FermionField m_in_buf;
  FermionField m_in_k;
  FermionField m_prop_k;
  PlannedFFT<typename FermionField::vector_object>* m_pfft = nullptr;

  mutable long n_apply = 0;

  FreeWilsonInverse(GridCartesian* UGrid_, double mass_, std::vector<Complex> boundary_)
    : UGrid(UGrid_), mass(mass_), boundary(boundary_),
      phase_neg(UGrid_), phase_pos(UGrid_),
      m_in_buf(UGrid_), m_in_k(UGrid_), m_prop_k(UGrid_) {
    Build();
  }

  void Build() {
    assert((int)boundary.size() == Nd);
    // AP-time boundary twist (added to the momentum value, compensated by the position-space phase).
    std::vector<double> tw(Nd, 0.0);
    for (int mu = 0; mu < Nd; ++mu) {
      double bph = std::acos(real(boundary[mu]));
      tw[mu] = bph / (2.0 * M_PI);
    }
    const Coordinate& fdim = UGrid->_fdimensions;   // global 4D dims
    const Coordinate& lodim = UGrid->_ldimensions;  // local 4D dims (this rank)
    const Coordinate& pcoor = UGrid->_processor_coor;
    int Lg[4] = {fdim[0], fdim[1], fdim[2], fdim[3]};
    int Ll[4] = {lodim[0], lodim[1], lodim[2], lodim[3]};
    int off[4];
    for (int mu = 0; mu < 4; ++mu) {
      off[mu] = pcoor[mu] * Ll[mu];
    }
    V4loc = Ll[0] * Ll[1] * Ll[2] * Ll[3];
    // free_dw_p lives on FreeMobius5DBlock; M5 = -mass makes its diagonal (4-M5) = (4+mass) = Grid Wilson.
    FreeMobius5DBlock blk(1, -mass, 1.0, 1.0, 0.0);
    Minv.resize(V4loc);
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
            std::array<std::complex<double>, 16> D;
            blk.free_dw_p(p, D);
            Eigen::MatrixXcd M4(4, 4);
            for (int a = 0; a < 4; ++a) {
              for (int bb = 0; bb < 4; ++bb) {
                M4(a, bb) = D[a * 4 + bb];
              }
            }
            int idx = ((kt * Ll[2] + kz) * Ll[1] + ky) * Ll[0] + kx;
            Minv[idx] = M4.inverse();
          }
        }
      }
    }
    m_pfft = new PlannedFFT<typename FermionField::vector_object>(UGrid);

    // scatter Minv into the slot-indexed device buffer (slot = oSite*Nsimd + lane), like the DW port.
    const Coordinate& rdim4 = UGrid->_rdimensions;
    const Coordinate& simd4 = UGrid->_simd_layout;
    int Nsimd = 1;
    int nOsites = 1;
    for (int mu = 0; mu < 4; ++mu) {
      Nsimd *= simd4[mu];
      nOsites *= rdim4[mu];
    }
    Minv_dev.resize((size_t)nOsites * Nsimd * 16);
    for (int oSite = 0; oSite < nOsites; ++oSite) {
      Coordinate ocoor4(4);
      Lexicographic::CoorFromIndex(ocoor4, oSite, rdim4);
      for (int lane = 0; lane < Nsimd; ++lane) {
        Coordinate icoor4(4);
        Lexicographic::CoorFromIndex(icoor4, lane, simd4);
        int lcoor4[4];
        for (int mu = 0; mu < 4; ++mu) {
          lcoor4[mu] = ocoor4[mu] + rdim4[mu] * icoor4[mu];
        }
        int idx4 = ((lcoor4[3] * Ll[2] + lcoor4[2]) * Ll[1] + lcoor4[1]) * Ll[0] + lcoor4[0];
        const Eigen::MatrixXcd& Mi = Minv[idx4];
        size_t base = ((size_t)oSite * Nsimd + lane) * 16;
        for (int a = 0; a < 4; ++a) {
          for (int bb = 0; bb < 4; ++bb) {
            std::complex<double> z = Mi(a, bb);
            Minv_dev[base + (size_t)a * 4 + bb] = Grid::Complex(z.real(), z.imag());
          }
        }
      }
    }

    // position-space twist phase (compensates the momentum shift tw). 4D -> shift = 0.
    ComplexField coor(UGrid);
    ComplexField ph(UGrid);
    ph = Zero();
    ComplexD ci(0.0, 1.0);
    for (int nu = 0; nu < Nd; ++nu) {
      LatticeCoordinate(coor, nu);
      double bph = std::acos(real(boundary[nu]));
      ph = ph + bph * coor * (1.0 / (double)(UGrid->_fdimensions[nu]));
    }
    phase_neg = exp(ci * ph * (-1.0));
    phase_pos = exp(ci * ph);
  }

  // per-momentum 4x4 (spin) matvec, colour-looped; on-device batched.
  void MomentumSpaceSolve_dev(FermionField& prop_k, const FermionField& in_k) const {
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites = in_k.Grid()->oSites();
    const Grid::Complex* Minv_p = &Minv_dev[0];
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite, nOsites, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        const Grid::Complex* M = Minv_p + ((uint64_t)oSite * Nsimd + lane) * 16;
        SiteSpinor inp = extractLane(lane, in_v[oSite]);
        SiteSpinor acc;
        acc = Zero();
        for (int a = 0; a < 4; ++a) {
          for (int bb = 0; bb < 4; ++bb) {
            Grid::Complex m = M[a * 4 + bb];
            for (int col = 0; col < Nc; ++col) {
              acc()(a)(col) += m * inp()(bb)(col);
            }
          }
        }
        insertLane(lane, out_v[oSite], acc);
      }
    });
  }

  // out = F_W in = D_W^free(m)^{-1} in
  virtual void operator()(const FermionField& in, FermionField& out) {
    Coordinate mask(Nd, 1);  // 4D: transform ALL spacetime dims
    m_in_buf = phase_neg * in;
    m_pfft->FFT_dim_mask(m_in_k, m_in_buf, mask, FFT::forward);
    MomentumSpaceSolve_dev(m_prop_k, m_in_k);
    m_pfft->FFT_dim_mask(out, m_prop_k, mask, FFT::backward);
    out = out * phase_pos;
    n_apply++;
  }

  // ---- momentum band projector P for the scale-mixed Mx kernel (grid_mx_scale_mixed_note_claude.md) ----
  // Build the per-momentum-slot scalar mask chi(|sin p|). The free Wilson block D_free(p) = A(p) I +
  // i sum_j sin(p_j) gamma_j has eigenvalues A(p) +- i |sin p|, |sin p| = sqrt(sum_j sin^2 p_j) = the KINETIC
  // (Fourier-mode) eigenvalue magnitude (user 2026-09-05: cut on |sin p|, the sin p_j scale, NOT the Wilson
  // real part A(p)). Low band = |sin p| < Acut. Sharp or smooth (tanh). NOTE: |sin p| = 0 at the doublers
  // (p_j = pi corners) as well as p = 0, so corner momenta fall in the low band (Wilson A lifts them in the
  // eigenvalue, but NOT in this cut variable). Same twist / momentum layout as Build().
  void build_bandmask(double Acut, bool smooth, double width) {
    std::vector<double> tw(Nd, 0.0);
    for (int mu = 0; mu < Nd; ++mu) {
      tw[mu] = std::acos(real(boundary[mu])) / (2.0 * M_PI);
    }
    const Coordinate& fdim = UGrid->_fdimensions;
    const Coordinate& lodim = UGrid->_ldimensions;
    const Coordinate& pcoor = UGrid->_processor_coor;
    int Lg[4], Ll[4], off[4];
    for (int mu = 0; mu < 4; ++mu) {
      Lg[mu] = fdim[mu];
      Ll[mu] = lodim[mu];
      off[mu] = pcoor[mu] * Ll[mu];
    }
    const Coordinate& rdim4 = UGrid->_rdimensions;
    const Coordinate& simd4 = UGrid->_simd_layout;
    int Nsimd = 1;
    int nOsites = 1;
    for (int mu = 0; mu < 4; ++mu) {
      Nsimd *= simd4[mu];
      nOsites *= rdim4[mu];
    }
    bandmask_dev.resize((size_t)nOsites * Nsimd);
    for (int oSite = 0; oSite < nOsites; ++oSite) {
      Coordinate ocoor4(4);
      Lexicographic::CoorFromIndex(ocoor4, oSite, rdim4);
      for (int lane = 0; lane < Nsimd; ++lane) {
        Coordinate icoor4(4);
        Lexicographic::CoorFromIndex(icoor4, lane, simd4);
        double s2 = 0.0;
        for (int mu = 0; mu < 4; ++mu) {
          int lc = ocoor4[mu] + rdim4[mu] * icoor4[mu];
          int kglob = off[mu] + lc;
          double p = 2.0 * M_PI * (kglob + tw[mu]) / Lg[mu];
          double sp = std::sin(p);
          s2 += sp * sp;
        }
        double scale = std::sqrt(s2);   // |sin p| = sqrt(sum_j sin^2 p_j), the free kinetic eigenvalue mag
        double chi;
        if (smooth) {
          chi = 0.5 * (1.0 - std::tanh((scale - Acut) / width));
        } else {
          chi = (scale < Acut) ? 1.0 : 0.0;
        }
        bandmask_dev[(size_t)oSite * Nsimd + lane] = Grid::Complex(chi, 0.0);
      }
    }
  }

  // multiply each momentum slot by the scalar mask chi (out = chi * in_k), spin+colour untouched
  void apply_bandmask_dev(FermionField& out, const FermionField& in_k) {
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites = in_k.Grid()->oSites();
    const Grid::Complex* mp = &bandmask_dev[0];
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    accelerator_for(oSite, nOsites, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        Grid::Complex chi = mp[(uint64_t)oSite * Nsimd + lane];
        SiteSpinor inp = extractLane(lane, in_v[oSite]);
        SiteSpinor acc;
        acc = Zero();
        for (int a = 0; a < 4; ++a) {
          for (int col = 0; col < Nc; ++col) {
            acc()(a)(col) = chi * inp()(a)(col);
          }
        }
        insertLane(lane, out_v[oSite], acc);
      }
    });
  }

  // out = P in : twist -> FFT -> multiply by chi(A(p)) -> iFFT -> untwist (same convention as operator()).
  void band_project(const FermionField& in, FermionField& out) {
    Coordinate mask(Nd, 1);
    m_in_buf = phase_neg * in;
    m_pfft->FFT_dim_mask(m_in_k, m_in_buf, mask, FFT::forward);
    apply_bandmask_dev(m_prop_k, m_in_k);
    m_pfft->FFT_dim_mask(out, m_prop_k, mask, FFT::backward);
    out = out * phase_pos;
  }
};

// M0_W = Omega^dag F_W Omega : the framed free-Wilson preconditioner (Omega = flowed-Landau frame,
// applied as a per-site colour mat-vec). Preconditions the interacting WilsonFermionD[U]. Wilson analogue
// of FreeLimitPreconditioner (no s-broadcast -- Omega is already 4D).
template <class Impl>
class FreeLimitPreconditionerW : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  FreeWilsonInverse<Impl>& F;
  LatticeColourMatrixD Omega;
  long n_apply;
  double t_omega = 0.0;
  double t_free = 0.0;

  FreeLimitPreconditionerW(FreeWilsonInverse<Impl>& F_, const LatticeColourMatrixD& xform4,
                           GridCartesian* UGrid)
    : F(F_), Omega(UGrid), n_apply(0) {
    Omega = xform4;
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y(in.Grid());
    double to = -usecond();
    phi = Omega * in;        // Omega : colour mat-vec per site (spin untouched)
    to += usecond();
    t_omega += to;
    double tfr = -usecond();
    F(phi, y);               // free Wilson inverse
    tfr += usecond();
    t_free += tfr;
    double to2 = -usecond();
    out = adj(Omega) * y;    // Omega^dag
    to2 += usecond();
    t_omega += to2;
  }

  void report_timers() const {
    if (n_apply == 0) {
      return;
    }
    double n = (double)n_apply;
    std::cout << GridLogMessage << "[M0_W timers] " << n_apply << " applies, avg us/apply:" << std::endl;
    std::cout << GridLogMessage << "  omega (fwd+dag) " << t_omega / n << std::endl;
    std::cout << GridLogMessage << "  F_W (free inv)  " << t_free / n << std::endl;
  }

  void reset_timers() {
    t_omega = 0.0;
    t_free = 0.0;
    n_apply = 0;
    F.n_apply = 0;
  }
};

// M1_W = Omega^dag { F_W - F_W D(tildeA) F_W } Omega : the leading D_W (hopping-expansion) correction to
// M0_W. EXACT operator split tildeA = U^L - 1 with U^L = Omega U Omega^dag (the framed config), so
//   D(tildeA) = D_W[U^L] - D_W^free,  and the free identity  D_W^free F_W = I  =>  D_W^free y0 = phi.
// Apply: phi=Omega in ; y0=F phi ; tmp=D_W[U^L] y0 ; w=tmp-phi ; y1=F w ; out=Omega^dag (y0 - y1).
// Cost: 2 free (FFT) inverses + ONE interacting D_W[U^L] apply (NOT free -- the honest metric adds one D_W
// per M1 apply, UNLIKE M0). Wilson analog of FreeMobius5D FreeLimitPreconditioner1. Idea R. Brower +
// T. Izubuchi; derivation dwf4_qcd_claude/global_hopping_claude.md.
template <class Impl>
class FreeLimitPreconditionerW1 : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  FreeWilsonInverse<Impl>& F;
  WilsonFermion<Impl>& Dframed;   // interacting Wilson on the framed config U^L = Omega U Omega^dag
  LatticeColourMatrixD Omega;
  long n_apply;
  long n_dw;                      // counts D_W[U^L] applies (each = one Wilson matvec)

  FreeLimitPreconditionerW1(FreeWilsonInverse<Impl>& F_, const LatticeColourMatrixD& xform4,
                            WilsonFermion<Impl>& Dframed_, GridCartesian* UGrid)
    : F(F_), Dframed(Dframed_), Omega(UGrid), n_apply(0), n_dw(0) {
    Omega = xform4;
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y0(in.Grid());
    FermionField tmp(in.Grid());
    FermionField w(in.Grid());
    FermionField y1(in.Grid());
    phi = Omega * in;         // Omega : colour mat-vec per site (spin untouched)
    F(phi, y0);              // y0 = F phi
    Dframed.M(y0, tmp);      // tmp = D_W[U^L] y0
    n_dw++;
    w = tmp - phi;           // D(tildeA) y0 = D_W[U^L] y0 - D_W^free y0 = D_W[U^L] y0 - phi
    F(w, y1);               // y1 = F D(tildeA) y0
    out = adj(Omega) * (y0 - y1);  // Omega^dag ( y0 - F D(tildeA) F phi )
  }
};

// Mx_W = scale-mixed M0/M1 kernel: apply the M1 (hopping) correction ONLY in the low kinetic band
// (|sin p| < Acut = the IR), fall back to M0 in the UV -- keeps M1's IR gain WITHOUT M1's UV red-flag outlier
// (opscan found M1 D_W throws a UV eval to 0.865+1.238i, C=1.25). P = FreeWilsonInverse::band_project (momentum
// low-pass, A(p)<Acut). TWO variants (grid_mx_scale_mixed_note_claude.md):
//   INNER (input-band, exact): P before D(tildeA); correction truly OFF in the UV (Mx->M0). Needs an explicit
//     D_free apply (band-limiting breaks D_free F = I): w = D_W[U^L](P y0) - D_free(P y0).
//   OUTER (output-band, cheap): P after the correction; keeps D_free F=I (w = D_W[U^L] y0 - phi), no extra
//     D_free, but leaks in the UV (D(tildeA) couples momenta). Idea R. Brower + T. Izubuchi lineage; scale
//     split + pre-rotation (Peter Boyle 2026-09-05, grid_prerotated_frame_note_claude.md).
template <class Impl>
class FreeLimitPreconditionerWx : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  FreeWilsonInverse<Impl>& F;
  WilsonFermion<Impl>& Dframed;   // interacting Wilson on U^L = Omega U Omega^dag
  WilsonFermion<Impl>& Dfree;     // free (unit-gauge) Wilson at m_prec (used by INNER only)
  LatticeColourMatrixD Omega;
  bool inner;                     // true = INNER (input-band, +D_free), false = OUTER (output-band)
  long n_apply;
  long n_dw;

  FreeLimitPreconditionerWx(FreeWilsonInverse<Impl>& F_, const LatticeColourMatrixD& xform4,
                            WilsonFermion<Impl>& Dframed_, WilsonFermion<Impl>& Dfree_,
                            GridCartesian* UGrid, bool inner_)
    : F(F_), Dframed(Dframed_), Dfree(Dfree_), Omega(UGrid), inner(inner_), n_apply(0), n_dw(0) {
    Omega = xform4;
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y0(in.Grid());
    FermionField w(in.Grid());
    FermionField y1(in.Grid());
    FermionField t(in.Grid());
    phi = Omega * in;        // frame
    F(phi, y0);             // y0 = F phi
    if (inner) {
      FermionField y0b(in.Grid());
      F.band_project(y0, y0b);   // P y0 (momentum low-pass)
      Dframed.M(y0b, t);         // D_W[U^L] (P y0)
      Dfree.M(y0b, w);           // D_free (P y0)  -- explicit (identity broken by P)
      w = t - w;                 // D(tildeA) (P y0)
      n_dw++;
      F(w, y1);                 // y1 = F D(tildeA) P F phi
      out = adj(Omega) * (y0 - y1);
    } else {
      Dframed.M(y0, t);          // D_W[U^L] y0
      w = t - phi;               // D(tildeA) y0 = D_W[U^L] y0 - D_free y0 (= phi, shortcut)
      n_dw++;
      F(w, y1);                 // y1 = F D(tildeA) F phi
      FermionField y1b(in.Grid());
      F.band_project(y1, y1b);   // P y1 (momentum low-pass of the correction)
      out = adj(Omega) * (y0 - y1b);
    }
  }
};

NAMESPACE_END(Grid);

#endif
