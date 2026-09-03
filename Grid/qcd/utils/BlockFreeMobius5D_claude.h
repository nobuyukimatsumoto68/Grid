#pragma once
// G0 (quality gate) of the domain-decomposed (additive-Schwarz) free-limit Mobius DWF preconditioner,
// Grid port. Design: qed2/dwf4_qcd_claude/grid_dd_freeprec_impl_plan_claude.md and its dwf4 prototype
// dd_freeprec_impl_plan_claude.md (validated struct BlockFreeDW5D). This file is ADDITIVE -- it does
// NOT modify FreeMobius5D_claude.h; it reuses FreeMobius5DInverse<Impl> as the exact block solver.
//
// Idea: replace the GLOBAL free inverse F = D_free^-1 with a NODE-LOCAL one. Partition the L^4 lattice
// into cubic CORE blocks of side `core`; apply the free inverse INDEPENDENTLY on each block over an
// extended box of side ext = core + 2*halo (RAS overlap) with block-periodic BC, then restrict to the
// core interior. The free operator is translation invariant, so ONE FreeMobius5DInverse built on the
// ext^4 grid is reused for every block. Instantiating it on the ext grid makes its Build() derive the
// block-local momenta + AP twist automatically, so core=L, halo=0 reduces to the exact kernel BIT-FOR-
// BIT (built-in validation, and the check that the block-local AP is consistent).
//
// SCOPE: G0 is a SINGLE-RANK quality gate. The block gather/scatter goes through
// unvectorizeToLexOrdArray/vectorizeFromLexOrdArray (full global array per rank), so it is correct only
// at --mpi 1.1.1.1. The MPI node=block performance version (G1: rank-local grid + Cshift halo, the
// distributed-transpose killer) is a separate later chunk.
//
// References: additive Schwarz / RAS -- X.-C. Cai, M. Sarkis, SIAM J. Sci. Comput. 21 (1999) 792; SAP
// domain decomposition for the lattice Dirac operator -- M. Luscher, hep-lat/0310048. Novelty: the
// Schwarz block solve is the EXACT free-operator inverse via a local FFT (not an iterative block solve).

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>

namespace Grid {

template <class Impl>
class BlockFreeMobius5DInverse : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  typedef typename FermionField::scalar_object SiteSpinor;

  GridCartesian* FGrid;   // full 5D grid (input / output live here)
  int Ls;
  int core;
  int halo;
  int ext;                // core + 2*halo
  int L4[4];              // full 4D spacetime extents (from FGrid, dims 1..4)
  int nb[4];              // blocks per dim = L4/core

  GridCartesian* EUGrid;  // ext 4D grid (non-distributed), owned
  GridCartesian* EFGrid;  // ext 5D grid [Ls, ext^4] (non-distributed), owned
  FreeMobius5DInverse<Impl>* blk;   // exact free inverse on the ext grid (reused per block)

  BlockFreeMobius5DInverse(GridCartesian* FGrid_, int Ls_, double M5, double b, double c, double mass,
                           std::vector<Complex> boundary, int core_, int halo_)
    : FGrid(FGrid_), Ls(Ls_), core(core_), halo(halo_) {
    ext = core_ + 2 * halo_;
    const Coordinate& fd = FGrid->_fdimensions;   // [Ls, Lx, Ly, Lz, Lt]
    for (int mu = 0; mu < 4; ++mu) {
      L4[mu] = fd[mu + 1];
      nb[mu] = L4[mu] / core;
    }
    // G0 is single-rank (the gather uses the full per-rank lex array).
    assert(FGrid->_Nprocessors == 1 &&
           "BlockFreeMobius5DInverse (G0) is single-rank only; run at --mpi 1.1.1.1");
    assert(L4[0] % core == 0 && L4[1] % core == 0 && L4[2] % core == 0 && L4[3] % core == 0 &&
           "core must divide each spacetime extent");
    assert(ext <= L4[0] && "extended block (core + 2 halo) must not exceed the lattice (periodic double-wrap)");

    Coordinate latt4({ext, ext, ext, ext});
    Coordinate simd4 = GridDefaultSimd(Nd, vComplexD::Nsimd());
    Coordinate mpi4({1, 1, 1, 1});
    EUGrid = new GridCartesian(latt4, simd4, mpi4);
    EFGrid = SpaceTimeGrid::makeFiveDimGrid(Ls_, EUGrid);
    blk = new FreeMobius5DInverse<Impl>(EFGrid, Ls_, M5, b, c, mass, boundary);
  }

  ~BlockFreeMobius5DInverse() {
    delete blk;
    delete EFGrid;
    delete EUGrid;
  }

  // full 5D lex index (s fastest), matching unvectorizeToLexOrdArray on a [Ls,Lx,Ly,Lz,Lt] grid
  static inline size_t lex5(int s, int Ls, const int* c4, const int* L) {
    size_t lex4 = (size_t)c4[0] + (size_t)L[0] * (c4[1] + (size_t)L[1] * (c4[2] + (size_t)L[2] * c4[3]));
    return (size_t)s + (size_t)Ls * lex4;
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    std::vector<SiteSpinor> inbuf;
    unvectorizeToLexOrdArray(inbuf, in);          // Ls * V4full, s fastest
    std::vector<SiteSpinor> outbuf(inbuf.size());
    for (size_t i = 0; i < outbuf.size(); ++i) {
      outbuf[i] = Zero();
    }

    int V4e = ext * ext * ext * ext;
    int Le[4] = {ext, ext, ext, ext};
    std::vector<SiteSpinor> ebuf((size_t)Ls * V4e);
    std::vector<SiteSpinor> eob((size_t)Ls * V4e);
    FermionField ein(EFGrid);
    FermionField eout(EFGrid);

    int nblk = nb[0] * nb[1] * nb[2] * nb[3];
    for (int ib = 0; ib < nblk; ++ib) {
      int bc[4];
      int t = ib;
      for (int mu = 0; mu < 4; ++mu) {
        bc[mu] = t % nb[mu];
        t /= nb[mu];
      }
      // gather the extended block (periodic wrap) into the ext grid
      for (int et = 0; et < ext; ++et) {
        for (int ez = 0; ez < ext; ++ez) {
          for (int ey = 0; ey < ext; ++ey) {
            for (int ex = 0; ex < ext; ++ex) {
              int ec[4] = {ex, ey, ez, et};
              int gc[4];
              for (int mu = 0; mu < 4; ++mu) {
                int g = bc[mu] * core - halo + ec[mu];
                g %= L4[mu];
                if (g < 0) {
                  g += L4[mu];
                }
                gc[mu] = g;
              }
              for (int s = 0; s < Ls; ++s) {
                ebuf[lex5(s, Ls, ec, Le)] = inbuf[lex5(s, Ls, gc, L4)];
              }
            }
          }
        }
      }
      vectorizeFromLexOrdArray(ebuf, ein);
      (*blk)(ein, eout);
      unvectorizeToLexOrdArray(eob, eout);
      // scatter the core interior (ext coord in [halo, halo+core)) back to the full output
      for (int et = 0; et < ext; ++et) {
        for (int ez = 0; ez < ext; ++ez) {
          for (int ey = 0; ey < ext; ++ey) {
            for (int ex = 0; ex < ext; ++ex) {
              int ec[4] = {ex, ey, ez, et};
              bool inside = true;
              for (int mu = 0; mu < 4; ++mu) {
                if (ec[mu] < halo || ec[mu] >= halo + core) {
                  inside = false;
                  break;
                }
              }
              if (!inside) {
                continue;
              }
              int gc[4];
              for (int mu = 0; mu < 4; ++mu) {
                gc[mu] = bc[mu] * core + (ec[mu] - halo);
              }
              for (int s = 0; s < Ls; ++s) {
                outbuf[lex5(s, Ls, gc, L4)] = eob[lex5(s, Ls, ec, Le)];
              }
            }
          }
        }
      }
    }
    vectorizeFromLexOrdArray(outbuf, out);
  }
};

// Block preconditioner M0 = Omega^dag F_blk Omega. Mirrors FreeLimitPreconditioner but holds the free
// inverse by the LinearFunction base ref, so it drives BOTH the exact FreeMobius5DInverse and the
// block-local BlockFreeMobius5DInverse (Omega is position-diagonal -> commutes with the block split).
template <class Impl>
class BlockFreeLimitPreconditioner : public LinearFunction<typename Impl::FermionField> {
public:
  typedef typename Impl::FermionField FermionField;
  LinearFunction<FermionField>& F;
  LatticeColourMatrixD Omega5;
  long n_apply;

  BlockFreeLimitPreconditioner(LinearFunction<FermionField>& F_, const LatticeColourMatrixD& xform4,
                               int Ls, GridCartesian* FGrid)
    : F(F_), Omega5(FGrid), n_apply(0) {
    for (int s = 0; s < Ls; ++s) {
      InsertSlice(xform4, Omega5, s, 0);
    }
  }

  virtual void operator()(const FermionField& in, FermionField& out) {
    n_apply++;
    FermionField phi(in.Grid());
    FermionField y(in.Grid());
    phi = Omega5 * in;
    F(phi, y);
    out = adj(Omega5) * y;
  }
};

}  // namespace Grid
