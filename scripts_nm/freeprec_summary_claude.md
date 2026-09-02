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
| ckpoint_lat.80  | -8 | 0.01619 | 9440 | 3824 | 2.47x | pending | pending |
| ckpoint_lat.120 | -7 | 0.01630 | 9488 | 4088 | 2.32x | pending | pending |
| ckpoint_lat.160 | -6 | 0.01619 | 9472 | 1832 | 5.17x | pending | pending |
| ckpoint_lat.200 | -4 | 0.01442 | 9248 | 2008 | 4.61x | pending | pending |
| ckpoint_lat.240 | -3 | 0.01622 | 9312 | 1704 | 5.46x | pending | pending |
| ckpoint_lat.280 | -3 | 0.01505 | 9488 | 1424 | 6.66x | pending | pending |
| ckpoint_lat.320 | -3 | 0.01479 | 9488 | 1808 | 5.25x | pending | pending |
| ckpoint_lat.360 | -3 | 0.01631 | 9440 | 1968 | 4.80x | pending | pending |
| ckpoint_lat.400 | -3 | 0.01531 | 9568 | 1960 | 4.88x | pending | pending |
| ckpoint_lat.440 | -6 | 0.01463 | 9440 | 1800 | 5.24x | pending | pending |
| ckpoint_lat.480 | -6 | 0.01579 | 9344 | 3768 | 2.48x | pending | pending |
| ckpoint_lat.520 | (unflowed) | 0.01559 | 9472 | 1736 | 5.46x | pending | pending |
| ckpoint_lat.560 | (unflowed) | 0.01571 | 9504 | 1984 | 4.79x | pending | pending |
| ckpoint_lat.600 | (unflowed) | 0.01570 | 9328 | 3416 | 2.73x | pending | pending |

Not-yet-run in freeprec but flowed: ckpoint_lat.20 ($Q=-3$), .40 ($Q=-8$), .140 ($Q=-6$).

## Findings so far ($M_0$ only)

- The preconditioner WINS at $Q \neq 0$ (2.3-6.7x) -- NOT a clean trivial-sector method. The win
  DEGRADES at high $|Q|$ ($|Q|=7,8 \to$ ~2.3-2.5x vs $|Q|=3 \to$ ~4.8-6.7x): a PARTIAL obstruction.
- The flowed-fixed Landau functional is ~CONSTANT (~0.014-0.016) and does NOT track the win -- so the
  degradation acts through a channel other than the global frame-quality functional (likely the $Q$-tied
  near-zero modes; the spectrum-transform diagnostic will test this).
- Config-to-config scatter is real ($|Q|=6$ spans 2.5-5.2x) -- more configs per $|Q|$ bin would firm it.

## $M_1$ (pending)

Expectation (dwf4 F3): $M_1$ roughly HALVES the outer/Krylov count but is ~$D_W$-neutral at $L_s=8$
(the correction is itself ~1 operator apply), HELPS where $M_0$ is weak (high $|Q|$ here?), mildly
OVER-corrects where $M_0$ is already strong. Fill the $M_1$ columns from the FGMRES(M1) line of each
`--ops m1` run; the $M_1$ speedup vs CGNE uses the CGNE column above (already logged).
