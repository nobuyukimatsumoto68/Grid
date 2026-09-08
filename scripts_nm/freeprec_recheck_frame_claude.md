# R2 headline re-check: was the high-|Q| obstruction an under-flowed-frame artifact?

STATUS: DRAFT skeleton (2026-09-03). Numbers to be filled when the flowscan_1 jobs (7433xxx) drain.

## Motivation

The R2 headline table (M0 win = CGNE/M0 D_W-apply ratio vs flowed Q) was built with a SHORT frame flow
(tau~2, i.e. s/t0 ~ 0.4-0.7). The flow-TIME scan showed the M0 optimum is config-dependent and, crucially,
the HARD config 640 (|Q|=3) is STUCK at ~1.4-1.6x for s/t0 <= 0.6 but JUMPS to ~4.4x at s/t0 >= 0.7. So a
poorly-framed (topological) config needs LONGER frame flow. Hypothesis: part of the headline's win
DEGRADATION at high |Q| was an under-flowing artifact, and re-framing at s/t0 ~ 0.7-0.8 will LIFT the
high-|Q| wins and FLATTEN the obstruction.

## What we can compare directly (overlap configs)

R2 headline set: trajs 80,120,160,200,240,280,320,360,400,440,480,600.
Flow-time scan set: trajs 460,480,520,540,560,580,600,640.
Direct overlap = 480, 600 (and 640 has both a headline-style short-frame point and the full s/t0 curve).
=> For the overlap configs, compare headline-frame win vs best-frame (s/t0~0.7-0.8) win from the scan.

## COMPLETE flow-time curves (2026-09-04, all 4 scan configs, M0 = CGNE/M0 D_W-apply ratio)

Headline frame was tau=2 -> s/t0 = 2/t0(2.91) = 0.687. NO overlap between the scan configs {460,520,580,640}
and the headline set {80..480,600}, so this is a LESSON about the frame, not a config-by-config redo.

| cfg | Q_5Li | M0 @ s/t0=0.4 | @ 0.69 (headline) | best (s/t0)   | shape |
|-----|-------|---------------|-------------------|---------------|-------|
| 460 | -6.0  | 3.14          | ~3.15             | 3.41 (0.50)   | flat ~3.1-3.4 (one 1.69 dropout @0.9 = gauge-fix noise) |
| 520 | -4.0  | 3.26          | 4.13              | 4.13 (0.70)   | rises to a peak right at the headline frame |
| 580 | -4.0  | 3.46          | ~3.02 (DIP)       | 3.66 (1.20)   | U: headline sits in the dip; optimum at LONG flow |
| 640 | -3.0  | 1.65          | ~4.36 (CLIFF)     | 4.41 (0.80)   | CLIFF between 0.6 (1.63) and 0.7 (4.36) |

## FINDINGS

1. The CATASTROPHIC under-framing (640: 1.36-1.65x) is confined to s/t0 <= 0.6. The headline frame
   s/t0=0.687 is JUST past the cliff, so the headline was NOT in the catastrophic regime -- earlier worry
   ("headline badly understated high-|Q|") is OVERSTATED.
2. BUT the headline frame is FRAGILE: for 640 the 1.63->4.36 jump happens between s/t0=0.6 and 0.7, so
   0.687 sits right on the cliff edge -- a hair earlier and the win collapses by ~2.7x. Hard/topological
   configs are exactly the ones with this cliff.
3. And it is not always optimal: 580's optimum is at LONG flow (s/t0~1.0-1.2, 3.66x) while the headline
   0.687 lands in its dip (~3.0x) -- a ~20% under-serve.
4. Config-to-config scatter at fixed Q is large (these 4 configs Q=-3..-6 give best 3.4-4.4x, vs headline
   same-Q configs 2.3-6.7x), so single-config statements are weak.

## RECOMMENDATION

Move the default frame from tau=2 (s/t0=0.69, cliff edge) to **s/t0 ~ 0.9-1.0 (tau ~ 2.6-2.9)**: safely
past the hard-config cliff AND at/near the long-flow optimum for the dip configs. Expected effect: removes
the fragility, modest lift on the harder configs. Q is fully plateaued there too (all curves show Q_5Li
integer by s/t0~0.7).

## DECISIVE TEST (hand Nobu the qsub; he submits)

Re-run the freeprec solve on the ACTUAL headline LOW-WIN configs -- 80 (|Q|=8, 2.47x) and 120 (|Q|=7,
2.32x), the ones that built the "obstruction" -- at s/t0~0.9 and compare to their tau=2 headline wins. If
the low win LIFTS, the obstruction was partly frame-fragility; if it STAYS, it is a genuine topological
obstruction. Script: grid_freeprec_wrapper_claude.sh (or the flowscan driver at fixed s/t0=0.9) on 80,120.
