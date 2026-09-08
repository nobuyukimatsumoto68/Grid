// Q-squeeze flow action: S = -scale * ||q||_n, the pure topological-charge-density localization
// flow of Grid/scripts_nm/freeprec_gaugefix_topology_claude.md Sec. 6b/6c and
// tests/solver/qsqueeze_impl_plan_claude.md. Gradient flow on this action INCREASES the n-norm of
// the clover charge density q(x), squeezing the topological lump toward the cutoff at an O(1) rate
// (d\rho/dt \propto -1/\rho^5 on the BPST moduli space). The norm-gradient weight
//   w_n(x) = ( q(x) / ||q||_n )^{n-1}        (n EVEN -> smooth, sign correct, bounded |w|<=1)
// concentrates the force at the lump core for large n (soft-max; n = 2, 4, 6 scan).
//
// Sources: F\tilde F (imaginary-theta) force with local weight -- M. D'Elia, F. Negro,
// arXiv:1306.2919; clover-leaf derivative with insertion = Cmunu of Eq. (B.39), Z. Sroczynski PhD
// thesis, ported from Grid's WilsonCloverHelpers.h:42 (fermion Impl -> pure Gimpl primitives);
// clover q(x) = the density inside WilsonLoops::TopologicalCharge (WilsonLoops.h:641).
//
// Conventions: deriv() returns UdSdU in the standard Grid gauge-action convention (validated by the
// --fdcheck mode of Test_dwf_shrinkflow_claude.cc against tests/forces/Test_rect_force.cc:115,
//   dS = -2 sum_mu tr( mom_mu UdSdU_mu ) dt  for  U -> exp(mom dt) U ),
// so WilsonFlow's RK3 (which only calls SG->deriv) integrates this action directly via
// setGaugeAction. scale (default 1) is the overall normalization knob fixed by the FD check.

#pragma once

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

template <class Gimpl>
class QSqueezeGaugeAction : public Action<typename Gimpl::GaugeField> {
public:
  INHERIT_GIMPL_TYPES(Gimpl);
  typedef WilsonLoops<Gimpl> WLoops;

  int qn;        // norm order n (EVEN)
  RealD scale;   // overall factor; FD-validated (folds c0 = 8/(32 pi^2) and the clover 1/8's)

  // diagnostics from the most recent qDensity-based call
  RealD last_qnorm;    // ||q||_n
  RealD last_qmax;     // max_x |q(x)|
  RealD last_Qclover;  // sum_x q(x)
  RealD last_PR;       // participation ratio (sum q^2)^2 / sum q^4 (qStats only)

  QSqueezeGaugeAction(int n, RealD s = 1.0) : qn(n), scale(s) {
    GRID_ASSERT((qn > 1) && (qn % 2 == 0));
  }

  virtual std::string action_name() { return "QSqueezeGaugeAction"; }
  virtual std::string LogParameters() {
    std::stringstream sstream;
    sstream << GridLogMessage << "[QSqueezeGaugeAction] n: " << qn << " scale: " << scale
            << std::endl;
    return sstream.str();
  }
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG) {}

  // Clover topological charge density q(x) (WilsonLoops.h:641 kept site-wise) and the six clover
  // field strengths (order: Fyz, Ftx, Fzx, Fty, Fxy, Ftz -- dual partners adjacent in pairs).
  static void qDensity(const GaugeField& U, std::vector<GaugeLinkField>& F, ComplexField& qfield) {
    GridBase* grid = U.Grid();
    F.resize(6, GaugeLinkField(grid));
    WLoops::FieldStrength(F[0], U, Ydir, Zdir);  // Fyz = Bx
    WLoops::FieldStrength(F[1], U, Tdir, Xdir);  // Ftx = Ex
    WLoops::FieldStrength(F[2], U, Zdir, Xdir);  // Fzx = By
    WLoops::FieldStrength(F[3], U, Tdir, Ydir);  // Fty = Ey
    WLoops::FieldStrength(F[4], U, Xdir, Ydir);  // Fxy = Bz
    WLoops::FieldStrength(F[5], U, Tdir, Zdir);  // Ftz = Ez
    RealD coeff = 8.0 / (32.0 * M_PI * M_PI);
    qfield = coeff * trace(F[0] * F[1] + F[2] * F[3] + F[4] * F[5]);
  }

  // ||q||_n^n = sum_x q^n (n even). Returns the sum; also fills qpow_nm1 = q^{n-1} if requested.
  static RealD qNormN(const ComplexField& qfield, int n, ComplexField* qpow_nm1) {
    ComplexField qp(qfield.Grid());
    qp = qfield;
    for (int k = 2; k <= n - 1; ++k) {
      qp = qp * qfield;
    }
    if (qpow_nm1) {
      *qpow_nm1 = qp;
    }
    ComplexField qpn(qfield.Grid());
    qpn = qp * qfield;  // q^n
    return TensorRemove(sum(qpn)).real();
  }

  void qStats(const GaugeField& U) {
    std::vector<GaugeLinkField> F;
    ComplexField qfield(U.Grid());
    qDensity(U, F, qfield);
    RealD qnn = qNormN(qfield, qn, nullptr);
    last_qnorm = std::pow(qnn, 1.0 / qn);
    last_qmax = std::sqrt(maxLocalNorm2(qfield));
    last_Qclover = TensorRemove(sum(qfield)).real();
    // participation ratio of q^2: PR = (sum q^2)^2 / sum q^4 = effective support volume (in sites)
    // of the charge density -- localization drives PR DOWN at fixed Q.
    ComplexField q2(U.Grid());
    q2 = qfield * qfield;
    ComplexField q4(U.Grid());
    q4 = q2 * q2;
    RealD s2 = TensorRemove(sum(q2)).real();
    RealD s4 = TensorRemove(sum(q4)).real();
    last_PR = (s4 > 0.0) ? (s2 * s2 / s4) : 0.0;
  }

  virtual RealD S(const GaugeField& U) {
    std::vector<GaugeLinkField> F;
    ComplexField qfield(U.Grid());
    qDensity(U, F, qfield);
    RealD qnn = qNormN(qfield, qn, nullptr);
    last_qnorm = std::pow(qnn, 1.0 / qn);
    last_qmax = std::sqrt(maxLocalNorm2(qfield));
    last_Qclover = TensorRemove(sum(qfield)).real();
    return -scale * last_qnorm;
  }

  // Clover-leaf derivative with insertion lambda at the clover centre: Cmunu of
  // WilsonCloverHelpers.h:42 (Sroczynski Eq. B.39), fermion Impl:: -> pure-gauge Gimpl:: port.
  static GaugeLinkField Cmunu(std::vector<GaugeLinkField>& U, GaugeLinkField& lambda, int mu,
                              int nu) {
    conformable(lambda.Grid(), U[0].Grid());
    GaugeLinkField out(lambda.Grid());
    GaugeLinkField tmp(lambda.Grid());
    // insertion in upper staple
    // C1+
    tmp = lambda * U[nu];
    out = Gimpl::ShiftStaple(
        Gimpl::CovShiftForward(tmp, nu,
                               Gimpl::CovShiftBackward(U[mu], mu,
                                                       Gimpl::CovShiftIdentityBackward(U[nu], nu))),
        mu);
    // C2+
    tmp = U[mu] * Gimpl::ShiftStaple(adj(lambda), mu);
    out += Gimpl::ShiftStaple(
        Gimpl::CovShiftForward(U[nu], nu,
                               Gimpl::CovShiftBackward(tmp, mu,
                                                       Gimpl::CovShiftIdentityBackward(U[nu], nu))),
        mu);
    // C3+
    tmp = U[nu] * Gimpl::ShiftStaple(adj(lambda), nu);
    out += Gimpl::ShiftStaple(
        Gimpl::CovShiftForward(U[nu], nu,
                               Gimpl::CovShiftBackward(U[mu], mu,
                                                       Gimpl::CovShiftIdentityBackward(tmp, nu))),
        mu);
    // C4+
    out += Gimpl::ShiftStaple(
               Gimpl::CovShiftForward(U[nu], nu,
                                      Gimpl::CovShiftBackward(
                                          U[mu], mu, Gimpl::CovShiftIdentityBackward(U[nu], nu))),
               mu) *
           lambda;
    // insertion in lower staple
    // C1-
    out -= Gimpl::ShiftStaple(lambda, mu) *
           Gimpl::ShiftStaple(Gimpl::CovShiftBackward(U[nu], nu,
                                                      Gimpl::CovShiftBackward(U[mu], mu, U[nu])),
                              mu);
    // C2-
    tmp = adj(lambda) * U[nu];
    out -= Gimpl::ShiftStaple(
        Gimpl::CovShiftBackward(tmp, nu, Gimpl::CovShiftBackward(U[mu], mu, U[nu])), mu);
    // C3-
    tmp = lambda * U[nu];
    out -= Gimpl::ShiftStaple(
        Gimpl::CovShiftBackward(U[nu], nu, Gimpl::CovShiftBackward(U[mu], mu, tmp)), mu);
    // C4-
    out -= Gimpl::ShiftStaple(Gimpl::CovShiftBackward(U[nu], nu,
                                                      Gimpl::CovShiftBackward(U[mu], mu, U[nu])),
                              mu) *
           lambda;
    return out;
  }

  // dS/dU for S = -scale ||q||_n.
  // d||q||_n = sum_x w_n(x) dq(x), w_n = (q/||q||_n)^{n-1}; dq(x) = c0 [tr(dF_a F_b) + tr(F_a dF_b)]
  // per dual pair; each clover F feels its link derivative through the Cmunu leaf staples with the
  // (weighted) dual partner as insertion. The overall constant (c0, the two 1/8's of FieldStrength,
  // leaf combinatorics) is folded into `knorm` and FIXED BY THE FD CHECK -- do not trust it blind.
  virtual void deriv(const GaugeField& Umu, GaugeField& dSdU) {
    GridBase* grid = Umu.Grid();
    std::vector<GaugeLinkField> F;
    ComplexField qfield(grid);
    qDensity(Umu, F, qfield);
    ComplexField qnm1(grid);
    RealD qnn = qNormN(qfield, qn, &qnm1);
    RealD qnorm = std::pow(qnn, 1.0 / qn);
    last_qnorm = qnorm;
    last_qmax = std::sqrt(maxLocalNorm2(qfield));
    last_Qclover = TensorRemove(sum(qfield)).real();

    // weight field w_n(x) = q^{n-1} / ||q||_n^{n-1}
    ComplexField wfield(grid);
    wfield = qnm1 * (1.0 / std::pow(qnorm, qn - 1));

    std::vector<GaugeLinkField> Ulink(Nd, grid);
    for (int mu = 0; mu < Nd; mu++) {
      Ulink[mu] = PeekIndex<LorentzIndex>(Umu, mu);
    }

    // ordered-plane insertions Lambda_{munu} = w * Fdual_{munu}; antisymmetric under mu<->nu.
    // plane list matching qDensity: (Y,Z)<->(T,X), (Z,X)<->(T,Y), (X,Y)<->(T,Z).
    int pmu[6] = {Ydir, Tdir, Zdir, Tdir, Xdir, Tdir};
    int pnu[6] = {Zdir, Xdir, Xdir, Ydir, Ydir, Zdir};
    int partner[6] = {1, 0, 3, 2, 5, 4};

    std::vector<GaugeLinkField> forcemu(Nd, grid);
    for (int mu = 0; mu < Nd; mu++) {
      forcemu[mu] = Zero();
    }
    GaugeLinkField Lam(grid);
    GaugeLinkField LamNeg(grid);
    for (int p = 0; p < 6; ++p) {
      Lam = wfield * F[partner[p]];
      LamNeg = -1.0 * Lam;
      forcemu[pmu[p]] += Cmunu(Ulink, Lam, pmu[p], pnu[p]);
      forcemu[pnu[p]] += Cmunu(Ulink, LamNeg, pnu[p], pmu[p]);
    }

    // overall: S = -scale ||q||_n; c0 from q, 1/8 from FieldStrength antihermitisation, and the
    // Grid deriv convention factor -- start from the analytic guess, FD check pins it via `scale`.
    RealD c0 = 8.0 / (32.0 * M_PI * M_PI);
    // sign FIXED BY THE FD CHECK (2026-09-06): with +scale the predicted dS matches (S(U')-S(U))
    // for S = -scale ||q||_n; ratio -> 1 as dt -> 0 (the Cmunu/Ta/deriv-convention chain supplies
    // one net minus relative to the naive bookkeeping).
    RealD knorm = scale * c0 * 0.125;
    for (int mu = 0; mu < Nd; mu++) {
      GaugeLinkField dmu(grid);
      dmu = knorm * Ta(Ulink[mu] * forcemu[mu]);
      PokeIndex<LorentzIndex>(dSdU, dmu, mu);
    }
  }
};

NAMESPACE_END(Grid);
