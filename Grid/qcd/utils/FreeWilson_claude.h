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

NAMESPACE_END(Grid);

#endif
