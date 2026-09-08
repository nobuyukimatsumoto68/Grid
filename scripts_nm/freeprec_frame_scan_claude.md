# Frame flow-time scan + DW budget (free-prec, 16^4 Iwasaki b2.6)

Scan of the M0 win vs the frame flow time s/t0 (frame = Wilson-flow to tau then Fourier-Landau -> Omega).
Data: `log/flowscan_ckpoint_lat.<cfg>_16_b2.6_claude.log` (2026-09-03/04). Flow scale $t_0(b2.6)=2.91\,a^2$,
so $s/t_0 = \tau/t_0$ and (at eps=0.02) $\text{nstep}=\tau/\text{eps}$. Solve tol 1e-6. Ls=8, Mobius Shamir
b=1.5 c=0.5 m=0.1 M5=1.8, anti-periodic time.

## M0 win (ratio CGNE/M0, D_W-apply) vs s/t0

| s/t0 | nstep | flowed plaq | 460 (Q-6) | 520 (Q-4) | 580 (Q-4) | 640 (Q-3) |
|------|-------|-------------|-----------|-----------|-----------|-----------|
| 0.40 | 58    | 0.99681     | 3.14      | 3.26      | 3.46      | 1.65      |
| 0.50 | 73    | 0.99768     | 3.41      | 3.40      | 3.08      | 1.36      |
| 0.60 | 87    | 0.99815     | 3.15      | 3.48      | 3.03      | 1.63      |
| 0.69 | 100   | ~0.99843    | ~3.15     | 4.13      | ~3.02     | ~4.36     |  <- OLD default (tau=2)
| 0.70 | 102   | 0.99848     | 3.15      | 4.13      | 3.02      | 4.36      |
| 0.80 | 116   | 0.99869     | 3.07      | 3.92      | 3.08      | 4.41      |
| 0.90 | 131   | 0.99886     | 1.69*     | 3.92      | 3.47      | 4.36      |
| 1.00 | 146   | 0.99898     | 3.14      | 3.94      | 3.51      | 4.36      |  <- NEW default (tau = t0)
| 1.10 | 160   | 0.99908     | 3.15      | 3.92      | 3.51      | 4.36      |
| 1.20 | 175   | 0.99916     | 3.15      | 3.86      | 3.66      | 4.34      |

bare plaquette ~0.6710 (all configs). flowed plaquette is ~config-independent (values above from cfg 640/520).
(*) 460 @ s/t0=0.90 = a single Landau gauge-fix dropout (1.69 between 3.07 and 3.14), not physical.

Key: the hard config 640 has a CLIFF between s/t0=0.6 (1.63) and 0.7 (4.36). The old default tau=2 (s/t0=0.69)
sat right on the cliff edge; 580's optimum is at long flow (s/t0~1.2). s/t0=1.0 is past the cliff, near the
long-flow optimum, and clean for all configs (no dropout).

## DW budget (the D_W-apply metric)

D_W (Mobius) applies to reach the solve tol, Ls=8:
- CGNE (frame-independent) = 2 * Ls * iters   (M and M^dag per CG iter)
- FGMRES(M0), M0 D_W-free   = Ls * iters       (1 D_DW apply per outer iter)
- FGMRES(M1), M1 NOT D_W-free = 2 * Ls * iters  (one internal D_DW[U^L] per M1 apply)

At the new frame s/t0=1.0 (per config):

| cfg | Q_5Li | CGNE iters / D_W | M0 D_W (=CGNE/ratio) | M0 win | M1 D_W | M1 win |
|-----|-------|------------------|----------------------|--------|--------|--------|
| 460 | -6.0  | 370 / 5920       | ~1885                | 3.14x  | ~2540  | ~2.3x  |
| 520 | -4.0  | 376 / 6016       | ~1527                | 3.94x  | ~1810  | ~3.3x  |
| 580 | -4.0  | 377 / 6032       | ~1719                | 3.51x  | ~2030  | ~3.0x  |
| 640 | -3.0  | 375 / 6000       | 1376                 | 4.36x  | 1616   | 3.71x  |

(M0/M1 D_W at s/t0=0.9 were measured directly: 640 M0=1376 M1=1616; 520 M0=1536 M1=1808; 580 M0=1736
M1=2032; 460 M0=3504[dropout] M1=2544. The s/t0=1.0 M0 D_W above are ratio-derived from CGNE.)

## Decision (2026-09-04): default frame moved

Old: tau=2, nstep=100, s/t0=0.69 (cliff edge). NEW: **s/t0=1.0, tau = t0 = 2.91, nstep=146 at eps=0.02,
flowed plaquette ~0.9990**. Changed in:
- `tests/solver/Test_dwf_freeprec_claude.cc:634` (run_headline: 0.02,100 -> 0.02,146; old value commented)
- `scripts_nm/run_spectrum_claude.sh` FLOWNSTEP 58 -> 146
- `scripts_nm/grid_spectrum_gpu_qsub_claude.sh` FLOWNSTEP 100 -> 146
Rebuild the freeprec (MPI) binary for the change to take effect in the R2 solve.

## Open: the decisive re-check

The scan configs {460,520,580,640} do NOT overlap the R2 headline set {80..480,600}, so re-run the headline
LOW-WIN configs 80 (|Q|=8, 2.47x) and 120 (|Q|=7, 2.32x) at the NEW frame to test whether their low win was
frame-fragility (lifts) or genuine topological obstruction (stays). See freeprec_recheck_frame_claude.md.
