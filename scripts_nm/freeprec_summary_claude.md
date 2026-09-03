# Free-limit DWF preconditioner -- R2 data summary (accumulating)

Running table of the R2 topology scan: per config we record the flowed topological charge, the
flowed-fixed Landau functional, and the $D_W$-apply counts + speedups for CGNE, FGMRES($M_0$), and
FGMRES($M_1$). Add a row (or fill the $M_1$ columns) as each `freeprec_<cfg>_claude.log` lands. This is
a data ledger -- APPEND / fill in place, do not recompute known columns.

- Ensemble: $16^4$ quenched Iwasaki $\beta = 2.6$, `configs_iwasaki_16_b2.6/ckpoint_lat.<traj>`.
- Solve: Mobius $D_{DW}$, $(b,c,m,M_5,L_s) = (1.5, 0.5, 0.1, 1.8, 8)$, anti-periodic time, no-restart
  FGMRES (RestartLength 256), tol $10^{-8}$.
- $D_W$ metric: CGNE $= 2 L_s \cdot$iters; FGMRES($M_0$) $= L_s \cdot$iters ($M_0$ is $D_W$-free);
  FGMRES($M_1$) $= L_s \cdot$(outer iters) $+ L_s \cdot$(internal $D_{DW}[U^L]$ applies) -- $M_1$ is NOT
  $D_W$-free (the honest total).
- $Q$ = Iwasaki-flowed $Q_\text{5Li}$ at $\tau \approx 2.0$ (integer plateau); the in-log `Q(orig)` is
  UV-noisy clover on the rough config, NOT the physics $Q$ -- ignore it.
- speedup = CGNE $D_W$ / (that method's $D_W$); $> 1$ means the preconditioner wins.

## Table

| config | $Q$ (5Li, $\tau2$) | Landau func | CGNE $D_W$ | $M_0$ $D_W$ | $M_0$ speedup | $M_1$ $D_W$ | $M_1$ speedup |
|---|---|---|---|---|---|---|---|
| ckpoint_lat.80  | -8 | 0.01619 | 9440 | 3824 | 2.47x | 2800 | 3.37x |
| ckpoint_lat.120 | -7 | 0.01630 | 9488 | 4088 | 2.32x | 2944 | 3.22x |
| ckpoint_lat.160 | -6 | 0.01619 | 9472 | 1832 | 5.17x | 2144 | 4.42x |
| ckpoint_lat.200 | -4 | 0.01442 | 9248 | 2008 | 4.61x | 2368 | 3.91x |
| ckpoint_lat.240 | -3 | 0.01622 | 9312 | 1704 | 5.46x | 1984 | 4.69x |
| ckpoint_lat.280 | -3 | 0.01505 | 9488 | 1424 | 6.66x | 1648 | 5.76x |
| ckpoint_lat.320 | -3 | 0.01479 | 9488 | 1808 | 5.25x | 2112 | 4.49x |
| ckpoint_lat.360 | -3 | 0.01631 | 9440 | 1968 | 4.80x | 2304 | 4.10x |
| ckpoint_lat.400 | -3 | 0.01531 | 9568 | 1960 | 4.88x | 2288 | 4.18x |
| ckpoint_lat.440 | -6 | 0.01463 | 9440 | 1800 | 5.24x | 2128 | 4.44x |
| ckpoint_lat.480 | -6 | 0.01579 | 9344 | 3768 | 2.48x | 2720 | 3.44x |
| ckpoint_lat.520 | -4 | 0.01559 | 9472 | 1736 | 5.46x | 2032 | 4.66x |
| ckpoint_lat.560 | -4 | 0.01571 | 9504 | 1984 | 4.79x | 2336 | 4.07x |
| ckpoint_lat.600 | -5 | 0.01570 | 9328 | 3416 | 2.73x | 2560 | 3.64x |

Not-yet-run in freeprec but flowed: ckpoint_lat.20 ($Q=-3$), .40 ($Q=-8$), .140 ($Q=-6$).

## Findings so far ($M_0$ only)

- The preconditioner WINS at $Q \neq 0$ (2.3-6.7x) -- NOT a clean trivial-sector method. The win
  DEGRADES at high $|Q|$ ($|Q|=7,8 \to$ ~2.3-2.5x vs $|Q|=3 \to$ ~4.8-6.7x): a PARTIAL obstruction.
- The flowed-fixed Landau functional is ~CONSTANT (~0.014-0.016) and does NOT track the win -- so the
  degradation acts through a channel other than the global frame-quality functional (likely the $Q$-tied
  near-zero modes; the spectrum-transform diagnostic will test this).
- Config-to-config scatter is real ($|Q|=6$ spans 2.5-5.2x) -- more configs per $|Q|$ bin would firm it.

## $M_1$ findings (2026-09-02, 14 configs)

CONFIRMS dwf4 F3 on the 16^4 QCD ensemble: $M_1$ HELPS where $M_0$ is WEAK and mildly OVER-corrects where
$M_0$ is already STRONG. Honest metric: $M_1$ total $D_W = 2 L_s\cdot$(outer iters) -- each apply carries
one $D_{DW}[U^L]$ -- so it wins on total $D_W$ only when the outer-count cut beats that 2x per-iter cost.
- WEAK-$M_0$ configs (high $|Q|$ / poor frame): $M_1$ BEATS $M_0$. .80 ($Q{-}8$) 2.47->3.37x; .120 ($-7$)
  2.32->3.22x; .480 ($-6$) 2.48->3.44x; .600 ($-5$) 2.73->3.64x.
- STRONG-$M_0$ configs: $M_1$ mildly WORSE. .280 ($-3$) 6.66->5.76x; .240 5.46->4.69x; .160 5.17->4.42x.
- $M_1$ ALWAYS cuts the outer Krylov count substantially (e.g. .80: M0 478 -> M1 175 outer iters), so its
  memory win (half+ the no-restart Krylov dimension) holds even where the total-$D_W$ ratio does not.
Net R2 read: $M_1$ FLATTENS the win-vs-$Q$ curve -- it lifts the worst (high-$|Q|$) configs toward ~3.3x
and trims the best, i.e. it adds information exactly where the leading frame $M_0$ is obstructed by
topology. Plotted as the red-circle series on freeprec_Qvsimprovement_claude.png.
NB .640/.680 have $M_1$ logs too but no logged CGNE/$M_0$ (outside the M0 scan) -> no ratio yet.
