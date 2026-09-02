#ifndef GRID_BLOCKING_FFT_CLAUDE_H
#define GRID_BLOCKING_FFT_CLAUDE_H

// SIMD-blocking Cooley-Tukey FFT for the free-prec F apply -- works ON Grid's vectorized layout with NO
// de-vectorize pack (Grid's L = rd*sd IS a radix-sd blocking). Design + validation plan:
// dwf4_qcd_claude/grid_blocking_fft_impl_plan_claude.md. Idea: N. Matsumoto.
//
// CHUNK A (this file, now): the INTERFACE + a pass-through wrapper around Grid's FFT_dim_mask -- trivially
// correct, establishes the A/B harness (Test_blocking_fft_claude.cc) and the swap-in point in
// FreeMobius5DInverse::operator(). Later chunks replace FFT_spacetime's internals:
//   B: x,y,z (sd=1) via strided cuFFT on the vectorized buffer (lanes ride along, no pack);
//   C: t (sd=2) via coarse length-rd_t FFT + twiddle + a 2-point cross-lane permute butterfly;
//   D: integrate into F, re-validate the cold gate + re-measure m0_breakdown_gpu.
// Grid's Grid/algorithms/FFT.h is NOT edited.

#include <Grid/Grid.h>

namespace Grid {

template <class Field>
class BlockingFFT {
public:
  GridCartesian* grid5;  // 5D grid [Ls, Lx, Ly, Lz, Lt]; s = dim 0 (NOT transformed)

  BlockingFFT(GridCartesian* grid5_) : grid5(grid5_) {}

  // FFT over the 4 spacetime dims (dim 1..4); s = dim 0 is NOT transformed. sign = FFT::forward /
  // FFT::backward. Convention MATCHES Grid FFT_dim_mask (fwd scale 1, bwd scale 1/(Lx Ly Lz Lt)) so the
  // momentum-space block solve (Minv_dev, keyed to Grid's FFT output layout) stays valid.
#ifdef BLOCKINGFFT_PASSTHROUGH
  void FFT_spacetime(Field& out, const Field& in, int sign) {
    FFT theFFT(grid5);
    Coordinate mask(grid5->Nd(), 1);
    mask[0] = 0;
    theFFT.FFT_dim_mask(out, in, mask, sign);
  }
#else
  // SIMD-blocking Cooley-Tukey (this build's layout: simd_layout [1,1,1,1,2], only t (dim 4) is
  // SIMD-split, sd=2; single rank). Two steps, NO de-vectorize pack:
  //   (1) COARSE: one rank-4 strided transform over [rdt, Lz, Ly, Lx] directly on the vectorized buffer,
  //       batch = inner contiguous block B = Ls*Ns*Nc*Nsimd (istride=B, idist=1, howmany=B). Reuses
  //       Grid's FFTW<ComplexD> wrapper (cuFFT on GPU / FFTW on CPU). Transforms x,y,z fully and t's
  //       coarse (length-rdt) factor; the 2 t-lanes ride along as batch (per-lane transform).
  //   (2) t BUTTERFLY: radix-2 across the two t-lanes (lane 0/1 of each oSite), twiddle w = e^{-2pi i k_r/Lt}
  //       on lane 1, then out(k_c=0)=c0+w c1, out(k_c=1)=c0-w c1. Completes t. Output frequency
  //       k_t = k_r + rdt*k_c lands at slot (t_r=k_r, lane=k_c) = the SAME (oSite,lane) Grid's FFT uses.
  void FFT_spacetime(Field& out, const Field& in, int sign) {
    typedef typename Field::vector_object vobj;
    const Coordinate& simd = grid5->_simd_layout;
    const Coordinate& procs = grid5->_processors;
    GRID_ASSERT(simd[0] == 1 && simd[1] == 1 && simd[2] == 1 && simd[3] == 1 && simd[4] == 2);
    for (int d = 0; d < 5; ++d) {
      GRID_ASSERT(procs[d] == 1);
    }
    const Coordinate& ldim = grid5->_ldimensions;
    const int Lx = ldim[1];
    const int Ly = ldim[2];
    const int Lz = ldim[3];
    const int Lt = ldim[4];

    // DIT for Grid's block layout t = ocoor + rd*icoor: forward = (t cross-lane butterfly + twiddle)
    // THEN coarse; backward reverses. (Output t-frequency ordering is k2 + sd*k1 at slot (t_r=k1,
    // lane=k2) -- a digit-reversal vs Grid's natural order; handled by Minv re-keying / a permute, TBD.)
    // in-place at every step on `out` (= in). Simpler; validated by the roundtrip on GPU + CPU.
    out = in;
    if (sign == FFT::forward) {
      ButterflyTwiddle(out, false);
      CoarseCufft(out, sign);
    } else {
      CoarseCufft(out, sign);
      ButterflyTwiddle(out, true);
      typedef typename vobj::scalar_type scalar;
      out = out * scalar(1.0 / ((double)Lx * Ly * Lz * Lt), 0.0);
    }
  }

  // t radix-2 cross-lane butterfly + twiddle (lane = icoor_t, only t is split), IN-PLACE. forward:
  // butterfly then twiddle e^{-2pi i t_r/Lt} on the k2=1 lane; inverse: undo twiddle (conj) then butterfly.
  void ButterflyTwiddle(Field& f, bool inverse) const {
    typedef typename Field::vector_object vobj;
    typedef typename Field::scalar_object sobj;
    typedef typename vobj::scalar_type scalar;
    const int Lt = grid5->_ldimensions[4];
    Coordinate rdim5 = grid5->_rdimensions;
    const double twsign = inverse ? 1.0 : -1.0;
    const double invLt = 1.0 / (double)Lt;
    autoView(fv, f, AcceleratorWrite);
    accelerator_for(ss, grid5->oSites(), 1, {
      Coordinate rc(5);
      Lexicographic::CoorFromIndex(rc, ss, rdim5);
      int tr = rc[4];
      double ang = twsign * 2.0 * M_PI * (double)tr * invLt;
      scalar tw = scalar(cos(ang), sin(ang));
      sobj l0 = extractLane(0, fv[ss]);
      sobj l1 = extractLane(1, fv[ss]);
      sobj o0;
      sobj o1;
      for (int sp = 0; sp < Ns; ++sp) {
        for (int co = 0; co < Nc; ++co) {
          scalar a = l0()(sp)(co);
          scalar b = l1()(sp)(co);
          if (!inverse) {
            o0()(sp)(co) = a + b;
            o1()(sp)(co) = (a - b) * tw;
          } else {
            scalar bt = b * tw;
            o0()(sp)(co) = a + bt;
            o1()(sp)(co) = a - bt;
          }
        }
      }
      insertLane(0, fv[ss], o0);
      insertLane(1, fv[ss], o1);
    });
  }

  // coarse strided transform on the vectorized buffer, IN-PLACE. The 4D FFT is separable, done as
  // cuFFT-LEGAL pieces (cuFFT cufftPlanMany supports rank <= 3 ONLY -- a single rank-4 silently fails,
  // which FFTW allowed so CPU passed): (a) rank-3 (x,y,z) [n={Lz,Ly,Lx}, batch=inner block B, istride=B,
  // idist=1] looped over the rdt t_r slabs; (b) rank-1 (t, length rdt) [batch=B*Lx*Ly*Lz, istride=that].
  void CoarseCufft(Field& f, int sign) const {
    typedef typename Field::vector_object vobj;
    typedef typename vobj::scalar_type scalar;
    typedef typename FFTW<scalar>::FFTW_scalar FFTW_scalar;
    typedef typename FFTW<scalar>::FFTW_plan FFTW_plan;
    const Coordinate& ldim = grid5->_ldimensions;
    const int Lx = ldim[1];
    const int Ly = ldim[2];
    const int Lz = ldim[3];
    const int rdt = grid5->_rdimensions[4];
    const int words = sizeof(vobj) / sizeof(scalar);
    const int B = ldim[0] * words;                   // inner block Ls*Ns*Nc*Nsimd
    const int64_t span = (int64_t)B * Lx * Ly * Lz;  // one t_r slab
    autoView(fv, f, AcceleratorWrite);
    FFTW_scalar* fptr = (FFTW_scalar*)&fv[0];
    // (a) rank-3 (x,y,z), in place, looped over t_r slabs
    int n3[3] = {Lz, Ly, Lx};
    int e3[3] = {Lz, Ly, Lx};
    for (int tr = 0; tr < rdt; ++tr) {
      FFTW_scalar* p3 = fptr + tr * span;
      FFTW_plan pl = FFTW<scalar>::fftw_plan_many_dft(3, n3, B,
                                                      p3, e3, B, 1,
                                                      p3, e3, B, 1,
                                                      sign, FFTW_ESTIMATE);
      FFTW<scalar>::fftw_execute_dft(pl, p3, p3, sign);
      FFTW<scalar>::fftw_destroy_plan(pl);
    }
    // (b) rank-1 (t, length rdt), in place, batch = the inner B*Lx*Ly*Lz block
    int n1[1] = {rdt};
    int e1[1] = {rdt};
    FFTW_plan pl = FFTW<scalar>::fftw_plan_many_dft(1, n1, (int)span,
                                                    fptr, e1, (int)span, 1,
                                                    fptr, e1, (int)span, 1,
                                                    sign, FFTW_ESTIMATE);
    FFTW<scalar>::fftw_execute_dft(pl, fptr, fptr, sign);
    FFTW<scalar>::fftw_destroy_plan(pl);
  }
#endif
};

}  // namespace Grid
#endif
