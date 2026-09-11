# Free-Mobius inverse: current apply, and blocking opportunities for the low-mode correction

Handoff note (Nobu -> local agent). Context: the 5D preconditioner knob study
(`saved/d5_precond_knobs_impl_plan_claude.md`, knobs 2/3/6/7) adds a low-mode correction
`D_approx = (1-P) D_free (1-P) + Sum_i lambda_i phi_i chi_i^H` to the free-Mobius preconditioner
`M0 = F^{-1}` (`FreeMobius5DInverse<Impl>` in `Grid/qcd/utils/FreeMobius5D_claude.h`). Applying it needs
`M0` on a *correction basis* of `~Ls*r` 5D vectors (r = rank of the 4D update C, r <= 48 for 24 modes).
This note documents how `M0` applies the 4D Fourier inverse over `Ls` today, and where blocking (multi-RHS)
wins are available for that precompute.

## Current single-RHS apply (`FreeMobius5DInverse::operator()`, default GPU path)

File: `Grid/qcd/utils/FreeMobius5D_claude.h`, `operator()` ~L965; solve `MomentumSpaceSolve_dev` ~L1359.

1. `in_buf = phase_neg * in`            -- position-space AP-time twist (pointwise).
2. `m_pfft->FFT_dim_mask(in_k, in_buf, mask, forward)` with `mask[0]=0` (dim 0 = s NOT transformed).
   -> ONE batched 4D FFT that transforms every `s`-slice's 4D field at once. `in_k` is `(s, p4)`-indexed.
3. `MomentumSpaceSolve_dev(prop_k, in_k)` -- per 4D momentum `(oSite4, lane)`, apply the precomputed dense
   `(4Ls)x(4Ls)` inverse `Minv` (`n5 = 4*Ls`, spin (x) s): `prop_k[s] = Sum_{s'} Minv[s,s'] in_k[s']` per
   colour. THIS is where the `Ls`-slices + spin couple (the 5th-dim inverse lives entirely in `Minv`,
   precomputed once per momentum in `Build()`). Currently a per-momentum MATVEC, single RHS.
4. `m_pfft->FFT_dim_mask(out, prop_k, mask, backward)` -- batched inverse 4D FFT over `Ls`.
5. `out = out * phase_pos`.

So the "4D Fourier inverse over `Ls` vectors" is already batched across `s` (steps 2/4 are single FFT
calls, no per-`s` loop); the `Ls`-coupling is the per-momentum `(4Ls)^2` block in step 3.

## Reuse for the correction (no new FFT machinery)

The Woodbury precompute needs `M0 U`, U = the `Ls`-lifted correction basis (`A (x) u_k`: each 4D vector
`u_k` spread over the `s`-slices by the Mobius affine structure `A`). Since `M0`'s FFT is batched over
`Ls`, we just PACK the `Ls` lift-slices into a 5D field and call `M0` once -- one batched 4D FFT does all
`Ls` transforms. No per-slice FFT loop, no separate 4D-Fourier code.

## Blocking opportunities (multi-RHS over the correction basis)

The precompute applies `M0` to `nRHS ~ Ls*r` vectors. Block them as `nRHS` columns:

1. **`Minv` matvec -> GEMM (the big one).** In `MomentumSpaceSolve_dev` the per-momentum `(4Ls)x(4Ls)`
   block is applied with the SAME `Minv` for every RHS -> a dense `(4Ls)x(4Ls) . (4Ls x nRHS)` GEMM per
   momentum, not `nRHS` matvecs. The `32x32` block matvec is memory-bound (reads `Minv` once per RHS);
   as a GEMM `Minv` is reused across RHS -> compute-bound, far better GPU utilization (same reason
   multi-RHS Dslash wins). Implementation: add an `nRHS` inner dimension to the field layout and an
   `rhs` loop in the kernel (L1367-1387) that the compiler turns into the GEMM inner product; or call a
   batched GEMM (cublas) with `Minv` as the shared A-matrix.
2. **FFT plan amortization.** One batched FFT over `(Ls * nRHS)` instead of `nRHS` separate `Ls`-batched
   FFTs -- amortizes plan/launch and the `s`-masking overhead. (PlannedFFT already caches the plan;
   batching raises the transform's effective batch count.)
3. **Woodbury inner products as GEMMs.** The one-time inner matrix `V^H (M0 U)` (`rank x rank`) is a block
   of dot products -> one reduction/GEMM, not `rank^2` `innerProduct` calls. (Per-iteration projection
   `V^H (.)` stays vector-wise unless we block-apply, but is also a reduction GEMM if blocked.)
4. **The lift `A (x) C` is itself a block op.** `A` is a tiny `Ls x Ls` s-coupling (with `P_+-`); apply it
   to the `r` 4D vectors as a block (small dense mult), no per-slice loop.

## Net

Precompute (the expensive one-time cost) blocks cleanly into **batched-FFT + per-momentum GEMM**; only the
per-FGMRES-iteration Woodbury projection stays vector-wise (and is a reduction GEMM if blocked). The
existing `MomentumSpaceSolve_dev` just needs an `nRHS` axis; everything else (FFT-over-`Ls`, `Minv`,
phases) is already in place.

## Pointers
- Solve kernel to block: `MomentumSpaceSolve_dev` (L1359) [and the fp32 twin `MomentumSpaceSolve_dev_f`].
- FFT: `FFT_dim_mask(..., mask, ...)`, `mask[0]=0` (L1000/L1025); `PlannedFFT` member `m_pfft`.
- `Minv` build: `Build()` (per-momentum `(4Ls)^2` inverse; `Minv_dev` slot-indexed).
- Design/plan: `saved/d5_precond_knobs_impl_plan_claude.md` (knobs 2/3/6/7, the 4D->5D affine lift + Woodbury).
