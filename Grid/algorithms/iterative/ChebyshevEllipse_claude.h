#ifndef GRID_CHEBYSHEV_ELLIPSE_CLAUDE_H
#define GRID_CHEBYSHEV_ELLIPSE_CLAUDE_H

// Complex Chebyshev-on-an-ellipse polynomial filter for a NON-Hermitian operator A -- the non-Hermitian
// analog of Grid's Hermitian Chebyshev (Chebyshev.h). Applies p_n(A) = T_n((A - c)/d) via the three-term
// recurrence, with c = ellipse center and d = focal half-distance (the foci are c +- d; d may be pure
// imaginary when the ellipse is taller than wide). |T_n| GROWS outside the ellipse and is bounded (~1)
// inside, so enclosing the UNWANTED bulk spectrum in the ellipse amplifies the WANTED eigenvalues that
// lie OUTSIDE it (our near-zero, near-real Wilson low modes). Feeding p_n(A) to Arnoldi turns the wanted
// modes into the LARGEST-modulus (peripheral) eigenvalues -- Arnoldi's easy, stable case -- WITHOUT
// squaring to |A|^2, so chirality of the D_W modes is preserved.
//
// This is the "Faber-Arnoldi" idea specialized to an ellipse (where Faber polynomials reduce to
// Chebyshev). Cite prominently:
//   - T. A. Manteuffel, "The Tchebychev iteration for nonsymmetric linear systems",
//     Numer. Math. 28 (1977) 307.
//   - Y. Saad, "Chebyshev acceleration techniques for solving nonsymmetric eigenvalue problems",
//     Math. Comp. 42 (1984) 567.
//   - V. Heuveline, M. Sadkane, "Arnoldi-Faber method for large non-Hermitian eigenvalue problems",
//     ETNA 5 (1997) 62 (ellipse = Chebyshev special case).
//
// RECOVERY: the eigenvalues of p_n(A) are NOT those of A. After Arnoldi on p_n(A) converges the wanted
// invariant subspace, recover A's eigenpairs by a Rayleigh-Ritz projection of A itself onto that subspace
// (H_s = B^dag A B), which the caller already does in the --spectrum RR-cleanup block. The filter only
// shapes the subspace; the returned (lambda, chi, residual) are exact for A = D_W.

#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

template<class Field>
class ChebyshevEllipse : public LinearFunction<Field> {
private:
  LinearFunction<Field>& _A;   // base operator (e.g. D_W), applied as _A(in,out)
  ComplexD c;                  // ellipse center
  ComplexD d;                  // focal half-distance (may be pure imaginary)
  int order;                   // Chebyshev degree n

public:
  ChebyshevEllipse(LinearFunction<Field>& A, ComplexD _c, ComplexD _d, int _order)
    : _A(A), c(_c), d(_d), order(_order) {}

  // out = T_n((A - c)/d) in, via  T_0 = I,  T_1 = (A - c)/d,  T_{k+1} = 2 (A - c)/d T_k - T_{k-1}.
  // All scalar*Lattice combinations go through axpy(out, scalar, x, y) = scalar*x + y (the confirmed
  // ComplexD path in Grid), never ComplexD*Lattice which is not defined under the CUDA build.
  void operator()(const Field& in, Field& out) {
    GridBase* grid = in.Grid();
    if (order <= 0) {
      out = in;
      return;
    }
    ComplexD invd = ComplexD(1.0, 0.0) / d;
    ComplexD two_invd = ComplexD(2.0, 0.0) * invd;

    Field zero(grid);
    zero = Zero();
    Field t0(grid), t1(grid), Av(grid), tmp(grid);

    // T_0 = in
    t0 = in;

    // T_1 = (A - c)/d in
    _A(in, Av);                    // Av = A in
    axpy(tmp, -c, in, Av);         // tmp = (A - c) in
    axpy(t1, invd, tmp, zero);     // t1  = (A - c)/d in
    if (order == 1) {
      out = t1;
      return;
    }

    for (int k = 1; k < order; ++k) {
      _A(t1, Av);                  // Av  = A T_k
      axpy(tmp, -c, t1, Av);       // tmp = (A - c) T_k
      axpy(out, two_invd, tmp, zero);        // out = 2 (A - c)/d T_k
      axpy(out, ComplexD(-1.0, 0.0), t0, out);  // out -= T_{k-1}
      t0 = t1;
      t1 = out;
    }
    out = t1;
  }
};

NAMESPACE_END(Grid);
#endif
