// Maximal-tree (nested-axial) gauge frame for the free-limit DWF preconditioner.
// IDEA: Taku (2026-09-07) -- try a maximal-tree gauge for Omega instead of Landau/Coulomb. Unlike the
// functional-minimisation gauges, a maximal tree EXACTLY trivialises a pure-gauge background (all tree
// links -> 1), concentrating any residual non-triviality in the off-tree links (the holonomies that
// enclose a topological lump). See tests/solver/maxtree_gauge_impl_plan_claude.md.
//
// Construction (Creutz maximal/axial tree; e.g. M. Creutz, "Quarks, Gluons and Lattices"): choose the
// nested-axial spanning tree rooted at the origin. Omega(x) is the ordered product of links along the
// tree path root->x; it is fixed by the requirement U'_mu(y) = Omega(y) U_mu(y) Omega(y+mu)^dag = 1 on
// every tree link, i.e. Omega(y+mu) = Omega(y) U_mu(y). On a PERIODIC lattice the Nd Polyakov loops
// cannot be treed away -> they remain as the residual global holonomy (expected; not an error).
//
// SINGLE-RANK build: Omega is built by a serial cumulative product with a strict dependency order
// (dir 3 line, then dir 2 planes, then dir 1, then dir 0), which does not data-parallelise. Intended
// for the 16^4 free-limit analysis runs (--mpi 1.1.1.1), exactly like the instsize probe. GRID_ASSERTs
// single rank.

#pragma once

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

// Lexicographic site index (dir 0 fastest), matching unvectorizeToLexOrdArray's local ordering.
static inline long maxtree_lex(int x0, int x1, int x2, int x3, int L0, int L1, int L2) {
  return (long)x0 + (long)L0 * ((long)x1 + (long)L1 * ((long)x2 + (long)L2 * (long)x3));
}

template <class Gimpl>
class MaxTreeGaugeFix {
public:
  typedef typename Gimpl::GaugeField GaugeLorentz;
  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename GaugeMat::vector_object vobj;
  typedef typename vobj::scalar_object sobj;  // ColourMatrix scalar (iScalar iScalar iMatrix<C,Nc>)

  // Build Omega = the maximal-tree gauge transform of Umu, apply it in place (Umu -> Omega Umu Omega^dag),
  // and return Omega in xform (same role as the Landau xform for FreeLimitPreconditioner).
  static void GaugeTransform(GaugeLorentz& Umu, GaugeMat& xform) {
    GridBase* grid = Umu.Grid();
    GRID_ASSERT(grid->_Nprocessors == 1);  // serial nested build assumes the whole lattice on one rank
    GRID_ASSERT(Nd == 4);

    Coordinate latt = grid->LocalDimensions();
    int L0 = latt[0];
    int L1 = latt[1];
    int L2 = latt[2];
    int L3 = latt[3];

    long V = (long)L0 * L1 * L2 * L3;

    // BULK host transfer (one per field, GPU-safe -- NOT a per-site peek storm; same idiom as
    // FreeMobius5D_claude.h). Uh[mu] is lex-ordered (dir 0 fastest), matching maxtree_lex().
    std::vector<std::vector<sobj>> Uh(Nd);
    for (int mu = 0; mu < Nd; mu++) {
      GaugeMat Um(grid);
      Um = PeekIndex<LorentzIndex>(Umu, mu);
      unvectorizeToLexOrdArray(Uh[mu], Um);
    }
    std::vector<sobj> Oh(V);

    sobj ident;
    ident = Zero();
    for (int c = 0; c < Nc; c++) {
      ident()()(c, c) = ComplexD(1.0, 0.0);
    }
    Oh[maxtree_lex(0, 0, 0, 0, L0, L1, L2)] = ident;

    // Serial nested-axial build: Omega(next) = Omega(cur) * U_mu(cur) along each tree edge.
    // dir 3: origin line (0,0,0,x3)
    for (int x3 = 1; x3 < L3; x3++) {
      long cur = maxtree_lex(0, 0, 0, x3 - 1, L0, L1, L2);
      long nxt = maxtree_lex(0, 0, 0, x3, L0, L1, L2);
      Oh[nxt] = Oh[cur] * Uh[3][cur];
    }
    // dir 2: planes (0,0,x2,x3)
    for (int x3 = 0; x3 < L3; x3++) {
      for (int x2 = 1; x2 < L2; x2++) {
        long cur = maxtree_lex(0, 0, x2 - 1, x3, L0, L1, L2);
        long nxt = maxtree_lex(0, 0, x2, x3, L0, L1, L2);
        Oh[nxt] = Oh[cur] * Uh[2][cur];
      }
    }
    // dir 1: (0,x1,x2,x3)
    for (int x3 = 0; x3 < L3; x3++) {
      for (int x2 = 0; x2 < L2; x2++) {
        for (int x1 = 1; x1 < L1; x1++) {
          long cur = maxtree_lex(0, x1 - 1, x2, x3, L0, L1, L2);
          long nxt = maxtree_lex(0, x1, x2, x3, L0, L1, L2);
          Oh[nxt] = Oh[cur] * Uh[1][cur];
        }
      }
    }
    // dir 0: (x0,x1,x2,x3)
    for (int x3 = 0; x3 < L3; x3++) {
      for (int x2 = 0; x2 < L2; x2++) {
        for (int x1 = 0; x1 < L1; x1++) {
          for (int x0 = 1; x0 < L0; x0++) {
            long cur = maxtree_lex(x0 - 1, x1, x2, x3, L0, L1, L2);
            long nxt = maxtree_lex(x0, x1, x2, x3, L0, L1, L2);
            Oh[nxt] = Oh[cur] * Uh[0][cur];
          }
        }
      }
    }

    GaugeMat Omega(grid);
    vectorizeFromLexOrdArray(Oh, Omega);
    xform = Omega;
    // Apply: U -> Omega U Omega^dag (same call the c1/M1 frame code uses).
    SU<Nc>::GaugeTransform<Gimpl>(Umu, Omega);
  }

  // Fraction of tree links actually set to identity, as a build self-check: 1 - <(1/Nc)Re tr U_tree>.
  // After GaugeTransform this should be ~0 to machine precision on the tree links; a diagnostic prints
  // the overall link trace (bulk near 1 for a pure-gauge background, off-tree holonomies deviate).
  static RealD linkTrace(const GaugeLorentz& Umu) {
    return WilsonLoops<Gimpl>::linkTrace(Umu);
  }
};

NAMESPACE_END(Grid);
