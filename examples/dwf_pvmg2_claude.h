// dwf_pvmg2_claude.h -- Grid chunk 2 of the PVMG program (see dwf_pvmg_grid_impl_plan_claude.md).
//
// Multigrid inside the s-Fourier Pauli-Villars inverse, b = c only.
// For b = c the twisted-mass Wilson kernel of FourierAcceleratedPV (Lehner/Boyle) has
// s-independent Wilson mass (= -M5) and twist \mu_s = tan(\theta_s/2), \theta_s = \pi(2s+1)/L_s.
// With T_s = D_W(-M5) + i \mu_s \gamma_5 and H = \gamma_5 D_W(-M5) hermitian,
//   T_s^\dagger T_s = H^2 + \mu_s^2
// exactly (\gamma_5-hermiticity cancels the cross terms). Hence ONE near-null subspace and ONE
// coarse operator H_c = P^\dagger H P serve all s-momenta and both twist signs; the coarse
// operator per shift is H_c^2 + \mu_s^2 (scalar shift). This mirrors the 2D testbed
// (../../dwf_1p1_claude/dwf_1p1_mg_claude.h, PVMGPrec on H_T^2 + p_j^2).
//
// Algorithm sources:
//   DD-\alpha AMG: A. Frommer, K. Kahl, S. Krieg, B. Leder, M. Rottmann, arXiv:1303.1377.
//   Inexact deflation / subspace generation: M. Luscher, arXiv:0706.2298.
//   Twisted-mass MG: C. Alexandrou et al., arXiv:1610.02370.
//   MG wiring pattern: Grid tests/solver/Test_dwf_hdcr_2level.cc (P. Boyle).
//   s-Fourier PV inverse: Grid/qcd/action/fermion/FourierAcceleratedPV.h (C. Lehner / P. Boyle);
//   rotate() and the twisted-mass / numerator-rotation formulas below are copied from it.

#pragma once

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/PrecGeneralisedConjugateResidual.h>

NAMESPACE_BEGIN(Grid);

static RealD PVMGInverseApproximation(RealD x) {
  return 1.0 / x;
}

// counting wrapper: forwards to the wrapped operator, tallies fine-operator applies.
// cost = number of Wilson-operator applications per HermOp (2 for MdagM).
template <class Field>
class CountedHermOp : public LinearOperatorBase<Field> {
public:
  LinearOperatorBase<Field>& wrapped;
  long* counter;
  int cost;

  CountedHermOp(LinearOperatorBase<Field>& _wrapped, long* _counter, int _cost)
      : wrapped(_wrapped), counter(_counter), cost(_cost) {}

  void OpDiag(const Field& in, Field& out) {
    wrapped.OpDiag(in, out);
  }
  void OpDir(const Field& in, Field& out, int dir, int disp) {
    wrapped.OpDir(in, out, dir, disp);
  }
  void OpDirAll(const Field& in, std::vector<Field>& out) {
    wrapped.OpDirAll(in, out);
  }
  void Op(const Field& in, Field& out) {
    (*counter) += cost;
    wrapped.Op(in, out);
  }
  void AdjOp(const Field& in, Field& out) {
    (*counter) += cost;
    wrapped.AdjOp(in, out);
  }
  void HermOpAndNorm(const Field& in, Field& out, RealD& n1, RealD& n2) {
    (*counter) += cost;
    wrapped.HermOpAndNorm(in, out, n1, n2);
  }
  void HermOp(const Field& in, Field& out) {
    (*counter) += cost;
    wrapped.HermOp(in, out);
  }
};

// 2-level V-cycle preconditioner for the shifted normal operator H^2 + \mu^2:
// Chebyshev pre-smooth, coarse-grid correction with (H_c^2 + \mu^2)^{-1} by CG, Chebyshev post-smooth.
template <class Fobj, class CComplex, int nbasis, class Field>
class TwistedVcyclePrec : public LinearFunction<Field> {
public:
  using LinearFunction<Field>::operator();
  typedef Aggregation<Fobj, CComplex, nbasis> Aggregates;
  typedef CoarsenedMatrix<Fobj, CComplex, nbasis> CoarseOp;
  typedef typename Aggregates::CoarseVector CoarseVector;

  Aggregates& aggregates;
  CoarseOp& coarseH;
  LinearOperatorBase<Field>& fineOp;   // H^2 + \mu^2 (counted)
  RealD mu2;
  RealD ctol;
  int cmaxit;
  Chebyshev<Field> cheby;
  long coarse_iters;

  TwistedVcyclePrec(Aggregates& _agg, CoarseOp& _Hc, LinearOperatorBase<Field>& _fop,
                    RealD _mu2, RealD lo, RealD hi, int sdeg, RealD _ctol, int _cmaxit)
      : aggregates(_agg), coarseH(_Hc), fineOp(_fop), mu2(_mu2),
        ctol(_ctol), cmaxit(_cmaxit),
        cheby(lo, hi, sdeg, PVMGInverseApproximation),
        coarse_iters(0) {}

  void operator()(const Field& in, Field& out) {
    Field tmp(in.Grid());
    Field cor(in.Grid());
    CoarseVector Csrc(coarseH.Grid());
    CoarseVector Csol(coarseH.Grid());

    // pre-smooth
    cheby(fineOp, in, out);

    // coarse-grid correction on the residual
    fineOp.HermOp(out, tmp);
    tmp = in - tmp;
    aggregates.ProjectToSubspace(Csrc, tmp);
    ShiftedMdagMLinearOperator<CoarseOp, CoarseVector> shifted(coarseH, mu2);
    ConjugateGradient<CoarseVector> ccg(ctol, cmaxit, false);
    Csol = Zero();
    ccg(shifted, Csrc, Csol);
    coarse_iters += ccg.IterationsToComplete;
    aggregates.PromoteFromSubspace(Csol, cor);
    out = out + cor;

    // post-smooth
    fineOp.HermOp(out, tmp);
    tmp = in - tmp;
    cheby(fineOp, tmp, cor);
    out = out + cor;
  }
};

// Slice-wise s-Fourier PV inverse with per-slice 4D twisted-mass CGNE solves,
// MG-preconditioned (PGCR + V-cycle) below p_cut, plain CG above.
template <class M, int nbasis>
class PVSliceMG {
public:
  typedef typename M::Impl_t Impl;
  typedef typename Impl::FermionField Field;
  typedef WilsonTMFermion<Impl> TM4;
  typedef Aggregation<vSpinColourVectorD, vTComplexD, nbasis> Aggregates;
  typedef CoarsenedMatrix<vSpinColourVectorD, vTComplexD, nbasis> CoarseOp;
  typedef typename Aggregates::CoarseVector CoarseVector;

  M& dwfPV;
  LatticeGaugeField& Umu;
  GridCartesian* UGrid;
  GridRedBlackCartesian* UrbGrid;
  int Ls;
  RealD bpar;
  RealD cpar;
  RealD M5;
  RealD mass4;
  std::vector<RealD> mus;                    // \mu_s, s = 0 .. Ls/2-1
  std::vector<std::unique_ptr<TM4> > tmP;    // +\mu_s (slice s)
  std::vector<std::unique_ptr<TM4> > tmM;    // -\mu_s (slice Ls-1-s)
  std::unique_ptr<TM4> tm0;                  // \mu = 0 (subspace / coarsening kernel)

  GridCartesian* CoarseGrid;
  std::unique_ptr<Aggregates> aggregates;
  std::unique_ptr<CoarseOp> coarseH;
  RealD lam_max;                             // \lambda_max(H^2) * safety
  bool mg_ready;

  // knobs (testbed defaults: dwf_1p1_summary_claude.md Sec. 8, tuned sdeg=2, p_cut=0.1-0.25;
  // Grid Chebyshev-with-1/x smoother needs a few more terms, default sdeg=8)
  RealD p_cut;
  int sdeg;
  RealD slo_frac;
  RealD ctol;
  int cmaxit;
  int pgcr_mmax;
  int pgcr_nstep;
  // fixed-application mode: the PV inverse is only a preconditioner, so instead of an inner
  // Krylov solve apply a FIXED approximate inverse per slice -- one V-cycle for |\mu_s| < p_cut,
  // one degree-hdeg Chebyshev of 1/x on [\mu_s^2, \mu_s^2+\lambda_max] above. No residual checks;
  // the map is exactly linear.
  bool fixed_apply;
  int hdeg;

  // counters: fine_applies in 4D Wilson-operator units; coarse iterations separately
  long fine_applies;
  long inner_iters;
  long coarse_iters;
  long pv_calls;

  PVSliceMG(M& _dwfPV, LatticeGaugeField& _Umu)
      : dwfPV(_dwfPV), Umu(_Umu) {
    UGrid = (GridCartesian*)dwfPV.GaugeGrid();
    UrbGrid = (GridRedBlackCartesian*)dwfPV.GaugeRedBlackGrid();
    Ls = dwfPV.FermionGrid()->_fdimensions[0];
    GRID_ASSERT((Ls % 2) == 0);
    get_real_const_bc(dwfPV, bpar, cpar);
    M5 = dwfPV.M5;
    mus.resize(Ls / 2);
    for (int s = 0; s < Ls / 2; s++) {
      // mass/mu formulas copied from FourierAcceleratedPV::pvInv (Lehner/Boyle)
      RealD phase = M_PI / (RealD)Ls * (2.0 * s + 1.0);
      RealD cosp = ::cos(phase);
      RealD sinp = ::sin(phase);
      RealD denom = bpar * bpar + cpar * cpar + 2.0 * bpar * cpar * cosp;
      RealD mass_s = -(bpar * bpar * M5 + cpar * (1.0 - cosp + cpar * M5)
                       + bpar * (-1.0 + cosp + 2.0 * cpar * cosp * M5)) / denom;
      RealD mu_s = (bpar + cpar) * sinp / denom;
      if (s == 0) {
        mass4 = mass_s;
      }
      GRID_ASSERT(::fabs(mass_s - mass4) < 1.0e-10);   // requires b = c
      mus[s] = mu_s;
      tmP.push_back(std::unique_ptr<TM4>(new TM4(Umu, *UGrid, *UrbGrid, mass4, +mu_s)));
      tmM.push_back(std::unique_ptr<TM4>(new TM4(Umu, *UGrid, *UrbGrid, mass4, -mu_s)));
    }
    tm0.reset(new TM4(Umu, *UGrid, *UrbGrid, mass4, 0.0));
    CoarseGrid = NULL;
    lam_max = 0.0;
    mg_ready = false;
    p_cut = 1.0;
    sdeg = 8;
    slo_frac = 0.03;
    ctol = 0.05;
    cmaxit = 1000;
    pgcr_mmax = 16;
    pgcr_nstep = 16;
    fixed_apply = false;
    hdeg = 4;
    fine_applies = 0;
    inner_iters = 0;
    coarse_iters = 0;
    pv_calls = 0;
  }

  void zero_counters(void) {
    fine_applies = 0;
    inner_iters = 0;
    coarse_iters = 0;
    pv_calls = 0;
  }

  // subspace + coarse operator; sub_lofrac/sub_order control the Chebyshev subspace filter
  void setup_mg(GridParallelRNG& RNG4, const std::vector<int>& block, RealD sub_lofrac, int sub_order) {
    Coordinate clatt = UGrid->_fdimensions;
    for (int d = 0; d < 4; d++) {
      GRID_ASSERT((clatt[d] % block[d]) == 0);
      clatt[d] = clatt[d] / block[d];
    }
    CoarseGrid = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());

    MdagMLinearOperator<TM4, Field> mdagm0(*tm0);
    Field noise(UGrid);
    gaussian(RNG4, noise);
    PowerMethod<Field> pm;
    lam_max = pm(mdagm0, noise) * 1.1;
    std::cout << GridLogMessage << "PVSliceMG setup: lam_max(H^2) estimate (incl. 1.1 safety) = " << lam_max << std::endl;

    aggregates.reset(new Aggregates(CoarseGrid, UGrid, 0));
    int nb = nbasis / 2;
    aggregates->CreateSubspaceChebyshev(RNG4, mdagm0, nb, lam_max, sub_lofrac * lam_max,
                                        sub_order, sub_order / 2, sub_order / 2, 0.0);

    // chirality doubling (pattern of Test_dwf_hdcr_2level.cc, with 4D \gamma_5)
    Gamma G5(Gamma::Algebra::Gamma5);
    for (int n = 0; n < nb; n++) {
      aggregates->subspace[n + nb] = G5 * aggregates->subspace[n];
    }
    Field A(UGrid);
    Field B(UGrid);
    for (int n = 0; n < nb; n++) {
      A = aggregates->subspace[n];
      B = aggregates->subspace[n + nb];
      aggregates->subspace[n] = A + B;
      aggregates->subspace[n + nb] = A - B;
    }

    // coarsen the hermitian indefinite H = \gamma_5 D_W(-M5); coarse H_c^2 is formed by
    // squaring on the coarse level (avoids the range-2 stencil of H^2)
    Gamma5HermitianLinearOperator<TM4, Field> hermH(*tm0);
    coarseH.reset(new CoarseOp(*CoarseGrid, 1));
    coarseH->CoarsenOperator(UGrid, hermH, *aggregates);
    mg_ready = true;
    std::cout << GridLogMessage << "PVSliceMG setup: coarse grid " << clatt
              << " nbasis " << nbasis << " done" << std::endl;
  }

  // one 4D twisted-mass solve T x = r via CGNE on H^2 + \mu^2
  void solve_slice(TM4& tm, RealD mu_s, const Field& r, Field& x, RealD tol, int maxit) {
    MdagMLinearOperator<TM4, Field> mdagm(tm);
    CountedHermOp<Field> counted(mdagm, &fine_applies, 2);
    Field rhs(UGrid);
    tm.Mdag(r, rhs);
    fine_applies += 1;
    x = Zero();
    if (fixed_apply && mg_ready) {
      if (::fabs(mu_s) < p_cut) {
        RealD hi = lam_max + mu_s * mu_s;
        TwistedVcyclePrec<vSpinColourVectorD, vTComplexD, nbasis, Field>
            prec(*aggregates, *coarseH, counted, mu_s * mu_s, slo_frac * hi, hi, sdeg, ctol, cmaxit);
        prec(rhs, x);
        coarse_iters += prec.coarse_iters;
      } else {
        RealD lo = 0.9 * mu_s * mu_s;
        RealD hi = lam_max + mu_s * mu_s;
        Chebyshev<Field> cheb(lo, hi, hdeg, PVMGInverseApproximation);
        cheb(counted, rhs, x);
      }
      inner_iters += 1;
      return;
    }
    if (mg_ready && (::fabs(mu_s) < p_cut)) {
      RealD hi = lam_max + mu_s * mu_s;
      TwistedVcyclePrec<vSpinColourVectorD, vTComplexD, nbasis, Field>
          prec(*aggregates, *coarseH, counted, mu_s * mu_s, slo_frac * hi, hi, sdeg, ctol, cmaxit);
      PrecGeneralisedConjugateResidual<Field> pgcr(tol, maxit, counted, prec, pgcr_mmax, pgcr_nstep);
      pgcr(rhs, x);
      inner_iters += pgcr.steps;
      coarse_iters += prec.coarse_iters;
    } else {
      ConjugateGradient<Field> cg(tol, maxit, false);
      cg(counted, rhs, x);
      inner_iters += cg.IterationsToComplete;
    }
  }

  // copied verbatim from FourierAcceleratedPV::rotatePV (Lehner/Boyle), minus timers
  void rotate(const Field& _src, Field& dst, bool forward) {
    typedef typename Field::scalar_type Coeff_t;
    int Lss = dst.Grid()->_fdimensions[0];
    Field _tmp(dst.Grid());
    double phase = M_PI / (double)Lss;
    Coeff_t bzero(0.0, 0.0);
    FFT theFFT((GridCartesian*)dst.Grid());
    if (!forward) {
      for (int s = 0; s < Lss; s++) {
        Coeff_t a(::cos(phase * s), -::sin(phase * s));
        axpby_ssp(_tmp, a, _src, bzero, _src, s, s);
      }
      theFFT.FFT_dim(dst, _tmp, 0, FFT::forward);
    } else {
      theFFT.FFT_dim(_tmp, _src, 0, FFT::backward);
      for (int s = 0; s < Lss; s++) {
        Coeff_t a(::cos(phase * s), ::sin(phase * s));
        axpby_ssp(dst, a, _tmp, bzero, _tmp, s, s);
      }
    }
  }

  // slice-wise M_PV^{-1}: rotate, per-slice twisted-mass solves, numerator rotation, rotate back
  void pvInv(const Field& _src, Field& _dst, RealD tol, int maxit) {
    typedef typename Field::scalar_type Coeff_t;
    pv_calls++;
    Field src_diag(_src.Grid());
    Field dst_diag(_src.Grid());
    Field slice_in(UGrid);
    Field slice_out(UGrid);
    rotate(_src, src_diag, false);
    Gamma G5(Gamma::Algebra::Gamma5);
    for (int s = 0; s < Ls / 2; s++) {
      int sprime = Ls - 1 - s;
      // numerator rotation constants copied from FourierAcceleratedPV::pvInv (Lehner/Boyle)
      RealD phase = M_PI / (RealD)Ls * (2.0 * s + 1.0);
      RealD cosp = ::cos(phase);
      RealD sinp = ::sin(phase);
      Coeff_t pA = Coeff_t(bpar + cpar * cosp, 0.0);
      Coeff_t pB = -Coeff_t(0.0, 1.0) * Coeff_t(cpar * sinp, 0.0);
      Coeff_t den = pA * pA - pB * pB;

      ExtractSlice(slice_in, src_diag, s, 0);
      solve_slice(*tmP[s], +mus[s], slice_in, slice_out, tol, maxit);
      slice_out = (pA / den) * slice_out - (pB / den) * (G5 * slice_out);
      InsertSlice(slice_out, dst_diag, s, 0);

      ExtractSlice(slice_in, src_diag, sprime, 0);
      solve_slice(*tmM[s], -mus[s], slice_in, slice_out, tol, maxit);
      slice_out = (pA / den) * slice_out + (pB / den) * (G5 * slice_out);
      InsertSlice(slice_out, dst_diag, sprime, 0);
    }
    rotate(dst_diag, _dst, true);
  }
};

NAMESPACE_END(Grid);
