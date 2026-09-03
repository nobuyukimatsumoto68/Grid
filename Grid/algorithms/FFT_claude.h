#ifndef GRID_FFT_CLAUDE_H
#define GRID_FFT_CLAUDE_H

// Cached FFT for the free-prec: Grid's FFT (Grid/algorithms/FFT.h) with the per-call PGBUF allocation and
// cuFFT/FFTW PLAN create+destroy hoisted into PERSISTENT members (allocated/planned once, keyed by
// (sign,G,howmany), destroyed in the dtor). Same barrel-shift/pack math as Grid's FFT_dim -- must give
// bit-identical results (validated by the cold gate + A/B vs Grid FFT). Purpose: remove the ~630 device
// mallocs + plan create/destroys per FGMRES solve on the FASTEST config (Grid packed cuFFT). The FFTW<scalar>
// backend wrapper is reused from Grid's FFT.h (brought in by Grid/Grid.h). Templated on the scalar S.
//
// Grid's Grid/algorithms/FFT.h is NOT edited. Idea: user (working-space reuse knob).

#include <Grid/Grid.h>
#include <map>

namespace Grid {

template <class S>
class FFT_claude {
private:
  double flops;
  double flops_call;
  uint64_t usec;
  deviceVector<S> pgbuf;                                    // persistent pencil working buffer (grows only)
#ifdef FFT_CLAUDE_PROFILE
  // Internal per-FFT_dim split (device-synced). PACK = barrel-shift + pencil collect (+ Cshift comm on
  // MPI); FFTK = cuFFT/FFTW kernel exec; UNPACK = scatter back + 1/G scale. Decides lever 1 (fp32,
  // kernel-dominated) vs lever 2 (fuse passes, pack/unpack-dominated). Guarded -> zero cost by default.
  mutable double tp_pack = 0.0;
  mutable double tp_fftk = 0.0;
  mutable double tp_unpack = 0.0;
  mutable long np_calls = 0;
#endif
  std::map<uint64_t, typename FFTW<S>::FFTW_plan> plans;    // cached plans keyed by (sign,G,howmany)

  static uint64_t plan_key(int sign, int G, int64_t howmany) {
    uint64_t dir = (sign == FFTW_FORWARD) ? 0ULL : 1ULL;
    return (dir << 40) | ((uint64_t)G << 24) | (uint64_t)howmany;
  }

public:
  static const int forward = FFTW_FORWARD;
  static const int backward = FFTW_BACKWARD;

  FFT_claude(GridCartesian* grid) {
    flops = 0;
    usec = 0;
    flops_call = 0;
  }
  ~FFT_claude(void) {
    for (auto& kv : plans) {
      FFTW<S>::fftw_destroy_plan(kv.second);
    }
  }

  double Flops(void) { return flops; }
  double MFlops(void) { return flops / usec; }
  double USec(void) { return (double)usec; }

  template <class vobj>
  void FFT_dim_mask(Lattice<vobj>& result, const Lattice<vobj>& source, Coordinate mask, int sign) {
    const int Ndim = source.Grid()->Nd();
    Lattice<vobj> tmp = source;
    for (int d = 0; d < Ndim; d++) {
      if (mask[d]) {
        FFT_dim(result, tmp, d, sign);
        tmp = result;
      }
    }
  }

  template <class vobj>
  void FFT_dim(Lattice<vobj>& result, const Lattice<vobj>& source, int dim, int sign) {
    typedef typename vobj::scalar_object sobj;
    typedef typename vobj::scalar_type scalar;
    typedef typename vobj::vector_type vector_type;
    static_assert(sizeof(scalar) == sizeof(S), "FFT_claude<S>: field scalar_type must match S");
    typedef typename FFTW<scalar>::FFTW_scalar FFTW_scalar;
    typedef typename FFTW<scalar>::FFTW_plan FFTW_plan;

    const int Ndim = source.Grid()->Nd();
    GridBase* grid = source.Grid();
    conformable(result.Grid(), source.Grid());

    int L = grid->_ldimensions[dim];
    int G = grid->_fdimensions[dim];

    int Ncomp = sizeof(sobj) / sizeof(scalar);
    int64_t Nlow = 1;
    int64_t Nhigh = 1;
    for (int d = 0; d < dim; d++) {
      Nlow *= grid->_ldimensions[d];
    }
    for (int d = dim + 1; d < Ndim; d++) {
      Nhigh *= grid->_ldimensions[d];
    }
    int64_t Nperp = Nlow * Nhigh;

    // PERSISTENT pgbuf (grow-only) instead of a per-call deviceVector<scalar> pgbuf.resize(...).
    size_t need = (size_t)Nperp * Ncomp * G;
    if (pgbuf.size() < need) {
      pgbuf.resize(need);
    }
    scalar* pgbuf_v = &pgbuf[0];

    int rank = 1;
    int n[] = {G};
    int howmany = Ncomp * Nperp;
    int odist, idist, istride, ostride;
    idist = odist = G;
    istride = ostride = 1;
    int *inembed = n, *onembed = n;

    scalar div;
    if (sign == backward) {
      div = 1.0 / G;
    } else if (sign == forward) {
      div = 1.0;
    } else {
      GRID_ASSERT(0);
    }

    // CACHED plan (create once per (sign,G,howmany); NOT destroyed per call). cuFFT/FFTW exec take the
    // current pointer, so a cached plan works even though pgbuf may have grown since it was created.
    uint64_t key = plan_key(sign, G, howmany);
    FFTW_plan p;
    auto it = plans.find(key);
    if (it != plans.end()) {
      p = it->second;
    } else {
      FFTW_scalar* in = (FFTW_scalar*)&pgbuf_v[0];
      FFTW_scalar* out = (FFTW_scalar*)&pgbuf_v[0];
      p = FFTW<scalar>::fftw_plan_many_dft(rank, n, howmany,
                                           in, inembed, istride, idist,
                                           out, onembed, ostride, odist,
                                           sign, FFTW_ESTIMATE);
      plans[key] = p;
    }

    // ----- Barrel shift and collect global pencil (verbatim from Grid FFT_dim) -----
#ifdef FFT_CLAUDE_PROFILE
    accelerator_barrier();
    double _tpk = -usecond();
#endif
    result = source;
    int pc = grid->_processor_coor[dim];

    const Coordinate ldims = grid->_ldimensions;
    const Coordinate rdims = grid->_rdimensions;
    const Coordinate sdims = grid->_simd_layout;

    Coordinate processors = grid->_processors;
    Coordinate pgdims(Ndim);
    pgdims[0] = G;
    for (int d = 0, dd = 1; d < Ndim; d++) {
      if (d != dim) pgdims[dd++] = ldims[d];
    }
    int64_t pgvol = 1;
    for (int d = 0; d < Ndim; d++) pgvol *= pgdims[d];

    const int Nsimd = vobj::Nsimd();
    for (int p2 = 0; p2 < processors[dim]; p2++) {
      autoView(r_v, result, AcceleratorRead);
      accelerator_for(idx, grid->oSites(), vobj::Nsimd(), {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd);
#else
        for (int lane = 0; lane < Nsimd; lane++) {
#endif
          Coordinate icoor;
          Coordinate ocoor;
          Coordinate pgcoor;
          Lexicographic::CoorFromIndex(icoor, lane, sdims);
          Lexicographic::CoorFromIndex(ocoor, idx, rdims);
          pgcoor[0] = ocoor[dim] + icoor[dim] * rdims[dim] + ((pc + p2) % processors[dim]) * L;
          for (int d = 0, dd = 1; d < Ndim; d++) {
            if (d != dim) {
              pgcoor[dd] = ocoor[d] + icoor[d] * rdims[d];
              dd++;
            }
          }
          int64_t pgidx;
          Lexicographic::IndexFromCoor(pgcoor, pgidx, pgdims);
          vector_type* from = (vector_type*)&r_v[idx];
          scalar stmp;
          for (int w = 0; w < Ncomp; w++) {
            int64_t pg_idx = pgidx + w * pgvol;
            stmp = getlane(from[w], lane);
            pgbuf_v[pg_idx] = stmp;
          }
#ifdef GRID_SIMT
        }
#else
        }
#endif
      });
      if (p2 != processors[dim] - 1) {
        Lattice<vobj> temp(grid);
        temp = Cshift(result, dim, L);
        result = temp;
      }
    }

#ifdef FFT_CLAUDE_PROFILE
    accelerator_barrier();
    tp_pack += _tpk + usecond();
    double _tft = -usecond();
#endif
    FFTW_scalar* in = (FFTW_scalar*)pgbuf_v;
    FFTW_scalar* out = (FFTW_scalar*)pgbuf_v;
    FFTW<scalar>::fftw_execute_dft(p, in, out, sign);
#ifdef FFT_CLAUDE_PROFILE
    accelerator_barrier();
    tp_fftk += _tft + usecond();
    double _tup = -usecond();
#endif

    flops_call = 5.0 * howmany * G * log2(G);
    flops = flops_call;

    result = Zero();
    {
      autoView(r_v, result, AcceleratorWrite);
      accelerator_for(idx, grid->oSites(), Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd);
#else
        for (int lane = 0; lane < Nsimd; lane++) {
#endif
          Coordinate icoor(Ndim);
          Coordinate ocoor(Ndim);
          Coordinate pgcoor(Ndim);
          Lexicographic::CoorFromIndex(icoor, lane, sdims);
          Lexicographic::CoorFromIndex(ocoor, idx, rdims);
          pgcoor[0] = ocoor[dim] + icoor[dim] * rdims[dim] + pc * L;
          for (int d = 0, dd = 1; d < Ndim; d++) {
            if (d != dim) {
              pgcoor[dd] = ocoor[d] + icoor[d] * rdims[d];
              dd++;
            }
          }
          int64_t pgidx;
          Lexicographic::IndexFromCoor(pgcoor, pgidx, pgdims);
          vector_type* to = (vector_type*)&r_v[idx];
          scalar stmp;
          for (int w = 0; w < Ncomp; w++) {
            int64_t pg_idx = pgidx + w * pgvol;
            stmp = pgbuf_v[pg_idx];
            putlane(to[w], stmp, lane);
          }
#ifdef GRID_SIMT
        }
#else
        }
#endif
      });
    }
    result = result * div;
#ifdef FFT_CLAUDE_PROFILE
    accelerator_barrier();
    tp_unpack += _tup + usecond();
    np_calls++;
#endif
    // plan NOT destroyed here -- cached in `plans`, freed in the destructor.
  }

#ifdef FFT_CLAUDE_PROFILE
  // Per-FFT_dim-call averages (one M0 apply = 8 FFT_dim calls: 4 dims x fwd+bwd, mask[0]=0).
  void report() const {
    if (np_calls == 0) {
      return;
    }
    double n = (double)np_calls;
    double tot = tp_pack + tp_fftk + tp_unpack;
    std::cout << GridLogMessage << "[FFT_dim split] " << np_calls << " calls, avg us/call:" << std::endl;
    std::cout << GridLogMessage << "  pack   " << tp_pack / n << "  (" << 100.0 * tp_pack / tot << "%)" << std::endl;
    std::cout << GridLogMessage << "  fftk   " << tp_fftk / n << "  (" << 100.0 * tp_fftk / tot << "%)" << std::endl;
    std::cout << GridLogMessage << "  unpack " << tp_unpack / n << "  (" << 100.0 * tp_unpack / tot << "%)" << std::endl;
    std::cout << GridLogMessage << "  total  " << tot / n << std::endl;
    std::cout << GridLogMessage << "  pack+unpack fraction = " << (tp_pack + tp_unpack) / tot << std::endl;
  }
#endif
};

}  // namespace Grid
#endif
