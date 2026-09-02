# M1 (leading $D_W$ correction) for the free-limit DWF preconditioner -- impl plan

**Algorithm source (cite in code + here):** next-order / Neumann "hopping-expansion" correction to the
free-limit preconditioner, idea R. Brower and T. Izubuchi. Reference implementation:
`qed2/dwf4_qcd_claude` -- `GaugeCovFreeDW1` (exact split) and `GaugeCovFreeWilsonAlg` (explicit
$D(\tilde A)$), see `dwf4_gaugefix_claude.h:294` and the Grid-port handoff `grid_dwf_prec_handoff_claude.md`
Task 4 (lines 118-133). Derivation: `qed2/dwf4_qcd_claude/global_hopping_claude.md`.

## Goal / physics

We currently have only $M_0 = \Omega^\dagger F \Omega$ (Landau frame; `FreeLimitPreconditioner` in
`Grid/qcd/utils/FreeMobius5D_claude.h`). All R2 results are $M_0$-only. $M_1$ adds the leading fluctuation
of the framed gauge field. Exact operator split (Wilson/DWF hopping is linear in the link):

$$
M_1 = \Omega^\dagger \{ F - F\, D(\tilde A)\, F \} \Omega,
\qquad
D(\tilde A) = D_{DW}[U^L] - D_\text{free},
\qquad
U^L = \Omega\, U\, \Omega^\dagger .
$$

Here $U$ is the ORIGINAL loaded config (we always solve the original operator; the flow is used ONLY to
find $\Omega$). $U^L$ is the original config rotated into the frame -- NOT the flowed config.

## Cheap apply (uses $D_\text{free} F = I$, so $D_\text{free} y_0 = \phi$)

```
phi = Omega * in            // frame rotate
y0  = F(phi)                // free Mobius inverse
tmp = D_DW[U^L] * y0        // one Mobius apply on the framed config
w   = tmp - phi             // D(tildeA) y0 = D_DW[U^L] y0 - D_free y0 = D_DW[U^L] y0 - phi
y1  = F(w)                  // second free inverse
d   = y0 - y1               // y0 - F D(tildeA) F phi
out = adj(Omega) * d        // frame rotate back
```

Cost: per $M_1$ apply = 2 free inverses (FFT, $D_W$-free) + ONE $D_{DW}[U^L]$ apply $= L_s$ $D_W$ applies.
So $M_1$ is NOT $D_W$-free (contrast $M_0$). Metric: FGMRES($M_1$) $D_W$ count
$= L_s \cdot (\text{outer iters}) + L_s \cdot (\text{\# } M_1 \text{ applies})$. dwf4 finding (F3): $M_1$
halves the outer count but is $D_W$-neutral at $L_s{=}8$; its win is memory (half the Krylov dim for
no-restart at large $V$) and finer $a$. So the honest comparison must count $M_1$'s internal $D_W$.

## tildeA variants (start with (i) only)

- (i) $\tilde A = U^L - 1$ (EXACT split): realized simply by building a second `MobiusFermionD` on $U^L$
  and using `tmp = D_DW[U^L] y0 - phi`. No custom gauge field. THIS IS THE FIRST CHUNK.
- (ii) $\tilde A = \mathrm{ta}(U^L)$ (`SU<N>::Ta`, anti-hermitian, dwf4 default -- fewer spectral
  outliers at coarse $a$): needs an explicit $D(\tilde A)$ = DWF hopping with a NON-UNITARY "link"
  field. In Grid this means substituting a hand-built `DoubledGaugeField` into a Mobius operator (its
  hopping apply is just a matmul, accepts non-unitary links). DEFERRED to a later chunk; ask Nobu whether
  we want it given Iwasaki $\beta{=}2.6$ is fairly fine.

## Files

- **Modify** `Grid/qcd/utils/FreeMobius5D_claude.h`: add class `FreeLimitPreconditioner1` (M1) right
  after `FreeLimitPreconditioner` (M0). Preserve M0 untouched (side-by-side, per convention). It holds
  `FreeMobius5DInverse<Impl>& F`, `LatticeColourMatrixD Omega5`, and a reference to the framed Mobius
  operator `MobiusFermion<Impl>& Dframed`; `operator()` does the 6-line apply above; counts its own
  $D_{DW}[U^L]$ applies in `n_dw` (and outer applies in `n_apply`).
- **Modify** `Grid/tests/solver/Test_dwf_freeprec_claude.cc`, `run_headline`: after building the frame
  `xform`, build $U^L = \Omega U \Omega^\dagger$ from the ORIGINAL `U` (Grid `GaugeTransform`, the same
  primitive M0's comment cites at `GaugeFix.h:156` / `SUn.impl.h:554`), build a second `MobiusFermionD
  Dframed(UL, ...)`, construct `FreeLimitPreconditioner1 M1(Ffree, xform, Dframed, FGrid)`, and run a
  THIRD solve: FGMRES right-preconditioned by $M_1$ (same tol/restart as the $M_0$ run). Keep the M0 run
  in place (comment nothing out -- add the M1 block beneath). Print: outer iters, $M_1$ applies, TOTAL
  $D_W$ (outer + internal), and speedup vs CGNE, next to the existing M0 line.

## Validation (mirror dwf4 gate 4)

- Cold gate ($U=1 \Rightarrow \Omega=1$, $U^L=1$): $M_1$ apply must be EXACT ($\|M_1 D v - v\|\sim$ eps),
  FGMRES 1 iter (same as $M_0$; $D(\tilde A)=0$).
- Proxy on a real config: $\|M_1 D v - v\|$ should DROP vs $\|M_0 D v - v\|$ on the smoother configs.
- Headline: report FGMRES($M_1$) TOTAL-$D_W$ speedup alongside $M_0$'s, per config, vs $|Q|$. dwf4 (F3,
  `cgscan_results_claude.md`): $M_1$ HELPS where $M_0$ is weak (coarse/poor frame), mildly OVER-corrects
  where $M_0$ is already strong. At $L_s{=}8$ expect the outer count roughly halved but total $D_W$ near
  break-even -- so watch the outer/Krylov count AND the total $D_W$.

## Open questions for Nobu (before coding)

1. tildeA: start with $U^L - 1$ (exact split, cheap, first chunk) and add $\mathrm{ta}(U^L)$ only if
   wanted? (dwf4 default is `ta`, but exact split is far simpler in Grid and is a correct M1.)
2. Metric: report the HONEST total $D_W$ (outer $L_s\cdot$it $+$ internal $L_s\cdot$applies) as the
   headline number, with the outer-only iteration count also printed? (dwf4 reported both.)
3. Same config set as the M0 R2 scan (16^4 Iwasaki $\beta2.6$), rerun through the freeprec wrapper so we
   get M0 and M1 side-by-side per config? Or a quick 8^4 cold-gate + one-config proof first?
4. Where does $U^L$ come from -- confirm Grid `GaugeTransform(U, xform)` gives $\Omega U \Omega^\dagger$
   with the SAME $\Omega$ convention M0 uses (I will verify against `GaugeFix.h` before wiring).
