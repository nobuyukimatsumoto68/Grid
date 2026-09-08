// Holonomy-twisted free Wilson inverse: the Wilson analogue of FreeWilsonInverse with a per-COLOUR
// constant holonomy twist theta_mu^(c) added to the momentum. Standalone (does NOT modify the local
// agent's FreeWilson_claude.h); theta = 0 reproduces FreeWilsonInverse bit-for-bit.
//
// Per-colour free block (closed FFT form, freeprec_gaugefix_topology_claude.md Sec. 7):
//   D_W^(c)(p) = A(p+theta^c) I + i sum_mu sin(p_mu + theta_mu^c) gamma_mu.
// So theta is a per-colour momentum shift on top of the AP-time BC shift. Implemented as: per-colour
// momentum blocks Minv_dev[...][col], and a COLOUR-DIAGONAL position phase Phi = diag_c exp(i ph_c)
// (compensating tw + theta_c per colour) applied as a colour mat-vec before/after a SINGLE FFT.
// theta is OPTIMISABLE (setTheta rebuilds; cheap: V4 x Nc 4x4 inverts) -- used by the joint
// (Omega, theta) frame optimisation (holonomy_frameopt_impl_plan_claude.md).
//
// Purpose: add the constant holonomy to the frame norm-optimisation (Nobu/Taku 2026-09-07) -- since
// M0 = Omega^dag F Omega conjugates and cannot change the config's holonomy, giving F a tunable twist
// lets it MATCH the holonomy the frame cannot remove.

#pragma once

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>  // FreeMobius5DBlock::free_dw_p, PlannedFFT

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FreeWilsonTwistedInverse : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  typedef typename FermionField::scalar_object SiteSpinor;
  typedef typename Impl::ComplexField ComplexField;

  GridCartesian* UGrid;
  double mass;
  std::vector<Complex> boundary;
  std::vector<std::array<double, Nd>> theta;  // [Nc][mu] per-colour holonomy angle; default all 0

  int V4loc;
  Grid::Vector<Grid::Complex> Minv_dev;  // slot-indexed, per colour: [((oSite*Nsimd+lane)*Nc+col)*16]
  LatticeColourMatrixD Phi_neg;          // colour-diagonal position phase, exp(-i ph_c)
  LatticeColourMatrixD Phi_pos;          // exp(+i ph_c)
  FermionField m_in_buf;
  FermionField m_in_k;
  FermionField m_prop_k;
  PlannedFFT<typename FermionField::vector_object>* m_pfft = nullptr;

  mutable long n_apply = 0;

  FreeWilsonTwistedInverse(GridCartesian* UGrid_, double mass_, std::vector<Complex> boundary_)
    : UGrid(UGrid_), mass(mass_), boundary(boundary_),
      Phi_neg(UGrid_), Phi_pos(UGrid_),
      m_in_buf(UGrid_), m_in_k(UGrid_), m_prop_k(UGrid_) {
    theta.assign(Nc, std::array<double, Nd>());
    for (int c = 0; c < Nc; ++c) {
      for (int mu = 0; mu < Nd; ++mu) {
        theta[c][mu] = 0.0;
      }
    }
    Build();
  }

  // Set the per-colour holonomy angles and rebuild the kernel (blocks + phase).
  void setTheta(const std::vector<std::array<double, Nd>>& th) {
    theta = th;
    Build();
  }

  void Build() {
    assert((int)boundary.size() == Nd);
    // BC angle per direction (0 spatial-periodic, pi AP-time). Total colour-c angle = bph + theta_c.
    std::vector<double> bph(Nd, 0.0);
    for (int mu = 0; mu < Nd; ++mu) {
      bph[mu] = std::acos(real(boundary[mu]));
    }
    const Coordinate& fdim = UGrid->_fdimensions;
    const Coordinate& lodim = UGrid->_ldimensions;
    const Coordinate& pcoor = UGrid->_processor_coor;
    int Lg[4] = {fdim[0], fdim[1], fdim[2], fdim[3]};
    int Ll[4] = {lodim[0], lodim[1], lodim[2], lodim[3]};
    int off[4];
    for (int mu = 0; mu < 4; ++mu) {
      off[mu] = pcoor[mu] * Ll[mu];
    }
    V4loc = Ll[0] * Ll[1] * Ll[2] * Ll[3];
    FreeMobius5DBlock blk(1, -mass, 1.0, 1.0, 0.0);  // M5=-mass -> Grid free Wilson (same as base)

    // per-colour, per-local-momentum 4x4 inverse block
    std::vector<std::vector<Eigen::MatrixXcd>> Minv(Nc, std::vector<Eigen::MatrixXcd>(V4loc));
    for (int c = 0; c < Nc; ++c) {
      for (int kt = 0; kt < Ll[3]; ++kt) {
        for (int kz = 0; kz < Ll[2]; ++kz) {
          for (int ky = 0; ky < Ll[1]; ++ky) {
            for (int kx = 0; kx < Ll[0]; ++kx) {
              int kc[4] = {kx, ky, kz, kt};
              std::array<double, 4> p;
              for (int mu = 0; mu < 4; ++mu) {
                int kglob = off[mu] + kc[mu];
                double twc = (bph[mu] + theta[c][mu]) / (2.0 * M_PI);
                p[mu] = 2.0 * M_PI * (kglob + twc) / Lg[mu];
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
              Minv[c][idx] = M4.inverse();
            }
          }
        }
      }
    }
    if (m_pfft == nullptr) {
      m_pfft = new PlannedFFT<typename FermionField::vector_object>(UGrid);
    }

    // scatter into the slot-indexed device buffer, with the colour index innermost-but-one
    const Coordinate& rdim4 = UGrid->_rdimensions;
    const Coordinate& simd4 = UGrid->_simd_layout;
    int Nsimd = 1;
    int nOsites = 1;
    for (int mu = 0; mu < 4; ++mu) {
      Nsimd *= simd4[mu];
      nOsites *= rdim4[mu];
    }
    Minv_dev.resize((size_t)nOsites * Nsimd * Nc * 16);
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
        for (int c = 0; c < Nc; ++c) {
          const Eigen::MatrixXcd& Mi = Minv[c][idx4];
          size_t base = (((size_t)oSite * Nsimd + lane) * Nc + c) * 16;
          for (int a = 0; a < 4; ++a) {
            for (int bb = 0; bb < 4; ++bb) {
              std::complex<double> z = Mi(a, bb);
              Minv_dev[base + (size_t)a * 4 + bb] = Grid::Complex(z.real(), z.imag());
            }
          }
        }
      }
    }

    // colour-diagonal position phase: Phi = diag_c exp(i ph_c), ph_c(x) = sum_mu (bph+theta_c)_mu x_mu/Lg
    ComplexD ci(0.0, 1.0);
    Phi_neg = Zero();
    Phi_pos = Zero();
    ComplexField coor(UGrid);
    for (int c = 0; c < Nc; ++c) {
      ComplexField phc(UGrid);
      phc = Zero();
      for (int mu = 0; mu < Nd; ++mu) {
        LatticeCoordinate(coor, mu);
        phc = phc + (bph[mu] + theta[c][mu]) * coor * (1.0 / (double)Lg[mu]);
      }
      ComplexField eneg(UGrid);
      ComplexField epos(UGrid);
      eneg = exp(ci * phc * (-1.0));
      epos = exp(ci * phc);
      pokeColour(Phi_neg, eneg, c, c);
      pokeColour(Phi_pos, epos, c, c);
    }
  }

  // per-momentum, per-COLOUR 4x4 (spin) matvec
  void MomentumSpaceSolve_dev(FermionField& prop_k, const FermionField& in_k) const {
    const int Nsimd = (int)in_k.Grid()->Nsimd();
    const uint64_t nOsites = in_k.Grid()->oSites();
    const Grid::Complex* Minv_p = &Minv_dev[0];
    autoView(in_v, in_k, AcceleratorRead);
    autoView(out_v, prop_k, AcceleratorWrite);
    accelerator_for(oSite, nOsites, 1, {
      for (int lane = 0; lane < Nsimd; ++lane) {
        SiteSpinor inp = extractLane(lane, in_v[oSite]);
        SiteSpinor acc;
        acc = Zero();
        for (int col = 0; col < Nc; ++col) {
          const Grid::Complex* M = Minv_p + (((uint64_t)oSite * Nsimd + lane) * Nc + col) * 16;
          for (int a = 0; a < 4; ++a) {
            for (int bb = 0; bb < 4; ++bb) {
              acc()(a)(col) += M[a * 4 + bb] * inp()(bb)(col);
            }
          }
        }
        insertLane(lane, out_v[oSite], acc);
      }
    });
  }

  // out = F_W^theta in : holonomy-twisted free Wilson inverse
  virtual void operator()(const FermionField& in, FermionField& out) {
    Coordinate mask(Nd, 1);
    m_in_buf = Phi_neg * in;  // colour-diagonal per-colour phase (colour mat-vec)
    m_pfft->FFT_dim_mask(m_in_k, m_in_buf, mask, FFT::forward);
    MomentumSpaceSolve_dev(m_prop_k, m_in_k);
    m_pfft->FFT_dim_mask(out, m_prop_k, mask, FFT::backward);
    out = Phi_pos * out;
    n_apply++;
  }
};

NAMESPACE_END(Grid);
