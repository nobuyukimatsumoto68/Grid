/*************************************************************************************
    Custom DOF-payload FFT for the free-prec F  --  C0: direct radix-L DFT, one unsplit dim.

    Idea: N. Matsumoto.  Plan: dwf4_qcd_claude/grid_custom_dof_fft_impl_plan_claude.md (sec 4b).

    MECHANISM: a spatial FFT is DIAGONAL in the internal DOF (Ls x spin x colour). Transforming a
    spacetime dim is a twiddle-weighted linear combination of SITES; the whole vObj payload at each
    site rides along identically. So a hand butterfly reads/writes WHOLE vObjs (coalesced, no
    de-interleave, no pencil transpose) -- exactly what cuFFT structurally cannot do.

    C0 SCOPE (this file, CPU-correctness first, GPU-portable by construction): a DIRECT L-point DFT
    along ONE UNSPLIT spacetime dim d (simd_layout[d] == 1, single MPI rank). Because d is unsplit,
    the DFT mixes only oSites (different rdim coord along d); the SIMD lanes (other, split dims) are
    untouched and ride along -- so we operate on whole vObjs via coalescedRead/coalescedWrite and the
    same accelerator_for runs on CPU and GPU.

    CONVENTION -- matched BIT-FOR-BIT to Grid's FFT_dim_execute / PlannedFFT (FFT.h) so the
    per-momentum Minv keying (freq = coord) stays valid:
      forward  (FFTW_FORWARD  = -1): X[k] = sum_x exp(-2 pi i k x / L) v[x],  no scale.
      backward (FFTW_BACKWARD = +1): x[n] = (1/L) sum_k exp(+2 pi i k n / L) V[k].
    Output is in natural (frequency = coordinate) order -- a direct dense L-point DFT, one stage.

    The boundary TWIST (anti-periodic phase) stays OUT of this butterfly -- it lives in
    phase_neg/phase_pos in FreeMobius5D. This is the PLAIN DFT.

    Provenance: DOF-payload / diagonal-in-internal-index observation -- N. Matsumoto. Direct dense DFT
    (L <= 16, one stage) -- standard; mixed-radix Cooley-Tukey deferred to L >= 32 (plan sec 4/4b).
*************************************************************************************/
#ifndef GRID_DOF_FFT_CLAUDE_H
#define GRID_DOF_FFT_CLAUDE_H

// Max direct-DFT length held in per-thread registers (v[L] scratch). C0/C1 target L <= 16; 32 gives
// headroom. L >= 32 wants mixed-radix staging (plan sec 4b) so this stays small.
#ifndef FREEMOBIUS5D_DOF_FFT_LMAX
#define FREEMOBIUS5D_DOF_FFT_LMAX 32
#endif

NAMESPACE_BEGIN(Grid);

// Direct radix-L DFT engine over the SIMD Grid field, DOF as a rider payload. Templated on vobj
// (double + fp32, like PlannedFFT). Precomputes the small L x L DFT matrices once per dim.
template<class vobj>
class DofFFT_claude {
public:
  typedef typename vobj::scalar_type scalar_type;   // ComplexD or ComplexF -- the twiddle type

  static const int forward  = FFTW_FORWARD;   // -1
  static const int backward = FFTW_BACKWARD;  // +1

  GridCartesian* _grid;
  int _Nd;

  // Per-dim dense DFT matrices, row-major [k*L + x], stored device-accessible (UVM). fwd = no scale,
  // bwd = conj with the 1/L normalisation folded in (matches PlannedFFT's div = 1/G on backward).
  std::vector<Grid::Vector<scalar_type>> dft_fwd;
  std::vector<Grid::Vector<scalar_type>> dft_bwd;

  DofFFT_claude(GridCartesian* grid) : _grid(grid) {
    _Nd = grid->Nd();
    dft_fwd.resize(_Nd);
    dft_bwd.resize(_Nd);
    for (int d = 0; d < _Nd; ++d) {
      int L = grid->_fdimensions[d];
      dft_fwd[d].resize((size_t)L * L);
      dft_bwd[d].resize((size_t)L * L);
      double twopi = 2.0 * M_PI;
      for (int k = 0; k < L; ++k) {
        for (int x = 0; x < L; ++x) {
          double ang = twopi * (double)(k * x) / (double)L;
          // forward sign = -1 (FFTW_FORWARD).
          ComplexD wf(std::cos(ang), -std::sin(ang));
          ComplexD wb(std::cos(ang), std::sin(ang));
          wb = wb / (double)L;
          dft_fwd[d][(size_t)k * L + x] = scalar_type(wf.real(), wf.imag());
          dft_bwd[d][(size_t)k * L + x] = scalar_type(wb.real(), wb.imag());
        }
      }
    }
  }

  // Transform a SINGLE unsplit spacetime dim (C0). d must satisfy simd_layout[d] == 1 and a single
  // MPI rank along d. result and source must be distinct fields on _grid.
  void FFT_dim(Lattice<vobj>& result, const Lattice<vobj>& source, int dim, int sign) {
    GRID_ASSERT(source.Grid() == _grid);
    GRID_ASSERT(result.Grid() == _grid);
    GRID_ASSERT(_grid->_simd_layout[dim] == 1);       // C0: unsplit dim only
    GRID_ASSERT(_grid->_processors[dim] == 1);        // C0: single rank along dim

    const int L = _grid->_rdimensions[dim];           // = fdimensions[dim] when unsplit + single rank
    const int Nsimd = (int)vobj::Nsimd();

    // oSite (over _rdimensions, dim0 fastest) = a + coord_d*lower + b*(lower*L). A "line" along dim =
    // fixed (a, b), varying coord_d in [0, L). stride between line elements = lower.
    uint64_t lower = 1;
    for (int j = 0; j < dim; ++j) lower *= (uint64_t)_grid->_rdimensions[j];
    const uint64_t nOsites = _grid->oSites();
    const uint64_t nLines  = nOsites / (uint64_t)L;   // = lower * higher

    const scalar_type* dft = (sign == forward) ? &dft_fwd[dim][0] : &dft_bwd[dim][0];

    autoView(in_v,  source, AcceleratorRead);
    autoView(out_v, result, AcceleratorWrite);
    accelerator_for(g, nLines, Nsimd, {
      // line index g -> (a, b) -> base oSite (coord_d = 0).
      uint64_t a = g % lower;
      uint64_t b = g / lower;
      uint64_t base = a + b * lower * (uint64_t)L;
      // load the L whole-vObj site values along the line into registers.
      typedef decltype(coalescedRead(in_v[0])) calcObj;
      calcObj v[FREEMOBIUS5D_DOF_FFT_LMAX];
      for (int x = 0; x < L; ++x) {
        v[x] = coalescedRead(in_v[base + (uint64_t)x * lower]);
      }
      // dense L-point DFT: acc[k] = sum_x dft[k][x] * v[x]. Natural order (freq k at coord k).
      for (int k = 0; k < L; ++k) {
        calcObj acc = dft[(size_t)k * L + 0] * v[0];
        for (int x = 1; x < L; ++x) {
          acc = acc + dft[(size_t)k * L + x] * v[x];
        }
        coalescedWrite(out_v[base + (uint64_t)k * lower], acc);
      }
    });
  }
};

NAMESPACE_END(Grid);

#endif
