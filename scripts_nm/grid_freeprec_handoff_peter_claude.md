# Free-limit DWF preconditioner in Grid — change summary (for Peter)

Branch: `dwf_prec` (fork `git@github.com:nobuyukimatsumoto68/Grid.git`).
Local Grid checkout: `/mnt/baracuda_14/dwms/Grid`; build dir: `/mnt/baracuda_14/dwms/build`.

**Repo layout (nested git).** `/mnt/baracuda_14/dwms` is an OUTER git repo (the research working dir);
the Grid fork at `/mnt/baracuda_14/dwms/Grid` is a SEPARATE, nested git repo (this branch). Everything
in this document lives inside the Grid fork and is tracked by IT (commit/push from `.../dwms/Grid`) --
including `scripts_nm/` (build scripts + this handoff), which is why they were moved inside the fork.
Files under `dwms/` but OUTSIDE `dwms/Grid/` (e.g. the generated `.log`, the dwf4-side impl plan and the
NERSC config bundle in `dwf4_qcd_claude/`) belong to the outer repo, not the fork, and are NOT part of
what you would push to the Grid remote.

**What this is:** a free-limit domain-wall preconditioner $M_0=\Omega^\dagger F\,\Omega$ for a flexible
outer solver (FGMRES), where $F=D_{DW}^{\rm free}(m)^{-1}$ is the free Mobius inverse (FFT-diagonal,
colour-blind) and $\Omega$ is a site-local SU($N$) gauge frame from Landau-fixing the Wilson-flowed
config. Idea: R. Brower + T. Izubuchi. Mobius/Cayley: Brower-Neff-Orginos arXiv:1206.5214.

## File tree (new + edited)

```
/mnt/baracuda_14/dwms/
├── Grid/                                       # the Grid fork (branch dwf_prec)
│   ├── Grid/
│   │   ├── perfmon/PerfCount.h                 # [EDITED, 2 lines] CUDA build fix
│   │   ├── qcd/utils/FreeMobius5D_claude.h     # [NEW, 378 L] the preconditioner (kernel + F + M0)
│   │   └── tests/solver/Test_dwf_freeprec_claude.cc  # [NEW, 282 L] validation ladder + benchmark harness
│   └── scripts_nm/                             # scripts + this handoff (git-traced in the fork; CANONICAL)
│       ├── grid_freeprec_build_claude.sh       # [NEW] standalone build+run handoff (no reconfigure)
│       ├── build_grid_ubuntu.sh                # Grid configure/build (CUDA)
│       └── grid_freeprec_handoff_peter_claude.md  # THIS document
├── grid_freeprec_build_claude.log              # [generated] compile + run output (written to dwms/)
└── dwf4_qcd_claude/
    └── grid_dwf_prec_impl_plan_claude.md       # [NEW, 182 L] design/impl plan + status + derivations
```

## Per-file detail

### `Grid/Grid/perfmon/PerfCount.h` — EDITED (build fix, not physics)
Uncommented the two `__rdtsc`/`__rdpmc` stub definitions in the `#ifdef GRID_CUDA` branch (lines
~54-55). Without this, an `nvcc` (`GRID_CUDA`) build on x86_64 with timers on fails: `cyclecount()`
calls `__rdtsc()` but the CUDA path neither includes `<x86intrin.h>` nor provided the stub. Cycle
counter reads 0 on the GPU build (host intrinsic, unused on device). One-line-class fix; alternative
is `--enable-timers=no` at configure.

### `Grid/Grid/qcd/utils/FreeMobius5D_claude.h` — NEW (the preconditioner)
Header-only, `namespace Grid`. Three pieces:
- `FreeMobius5DBlock` — builds the per-momentum $(4L_s)\times(4L_s)$ free Mobius block (Dirac$\times s$),
  DeGrand-Rossi gammas, $P_\pm=(I\pm\gamma_5)/2$. **This replaces Grid's Shamir-only free kernel:**
  `DomainWallFermion::FreePropagator` -> `WilsonFermion5D::MomentumSpacePropagatorHt_5d` is $W=1-M5+sk2$
  parametrized by $(M5,L_s,m)$ only (no $b,c$), so it is wrong for generic Mobius (even $b=c=1$).
- `FreeMobius5DInverse<Impl>` — the apply $F$. **Reuses Grid's `FreePropagator` FFT+twist verbatim**;
  the ONLY substitution is the momentum-space step (Grid's Shamir analytic propagator -> our
  per-momentum block solve, precomputed inverse, colour-blind). `LinearFunction<FermionField>`.
- `FreeLimitPreconditioner<Impl>` — $M_0(x)=\Omega^\dagger F(\Omega x)$; $\Omega=$ gauge-fixer `xform`
  broadcast across the $L_s$ slices. Also `LinearFunction`. `n_apply` counter (= FGMRES outer iters).

### `Grid/tests/solver/Test_dwf_freeprec_claude.cc` — NEW (validation + benchmark)
- gate 0a: free $D_W(p)$ eigenvalues $=d\pm i\sqrt S$ (vs analytic).
- gate 0b: free Mobius block min/max $|\lambda|$ vs a convention-locked table, all $(b,c,m)$ cases.
- gate 1: cold gate on unit gauge vs Grid's actual `MobiusFermionD`, both directions
  $\lVert FDv-v\rVert$ and $\lVert DFv-v\rVert$.
- gate 2: pure-gauge gate — $U^g=g_0\mathbb1 g_0^\dagger$, Landau-fix, $\lVert M_0 D[U^g]v-v\rVert$.
- chunk-3 harness: WilsonFlow frame + FGMRES($M_0$) vs CGNE $D_W$-apply count (currently on a Hot
  config as a plumbing check; headline needs a thermalized config).

### `Grid/scripts_nm/grid_freeprec_build_claude.sh` — NEW (build/run handoff, CANONICAL)
Compiles the ONE test standalone against the already-built `build/Grid/libGrid.a` via `grid-config`
(no `configure`/`bootstrap`); link driver = `--cxx` with `-x cu`->`-link`. Runs it, tees to
`dwms/grid_freeprec_build_claude.log`. Lives in `Grid/scripts_nm/` so it is tracked by the fork
(alongside `build_grid_ubuntu.sh`, the Grid configure/build script). Uses absolute paths, so it runs
from any cwd.

### `dwf4_qcd_claude/grid_dwf_prec_impl_plan_claude.md` — NEW (design note)
Physics/goal, Grid facilities used, the "reuse Grid's Fourier; only the eigenvalues differ" directive,
the eigenvalue-vs-eigenvector derivation (Shamir->Mobius changes both), ordered chunks, per-chunk
validation status, open questions.

## Validation status (all PASS on $4^4$, $L_s8$, $M5=1.8$, headline Shamir $b{=}1.5,c{=}0.5$)

| gate | quantity | result |
|---|---|---|
| 0a | free $D_W(p)$ eigenvalues vs $d\pm i\sqrt S$ | 5e-16 |
| 0b | free Mobius block min/max $|\lambda|$ vs table | 3e-6 (table-precision limited) |
| 1  | cold gate $\lVert FDv-v\rVert$ / $\lVert DFv-v\rVert$ (vs Grid `MobiusFermionD`) | 3.7e-16 both |
| 2  | pure-gauge $\lVert M_0 D[U^g]v-v\rVert$ (Landau functional 1.2e-12, $Q\approx0$) | 5.9e-12 |
| 3b | raw $D_W(-M5)$ cross-check vs dwf4 (spectral moments $\mathrm{tr}D_W^k$, $k{=}1..6$) on NERSC $4^4\,\beta6$ | 5e-14 rel |

Chunk 3b also loads the NERSC config clean (plaquette match 4e-11 vs header 0.6093762125, checksum
a9d6b482) and runs the headline: **CGNE 141 it / 2256 $D_W$ vs FGMRES($M_0$) 76 it / 608 $D_W$ = 3.71x**.

**KNOWN ISSUE (being fixed):** the $M_0$ frame's gauge fixer (Grid `FourierAcceleratedGaugeFixer`,
$\alpha{=}0.1$) *converges then diverges* on the flowed $\beta6$ config -- dmuAmu $12759\to254$ (it20)
$\to$ ~28000 (stuck). The frame used is the diverged one (flowed-fixed Landau functional 0.128), which
caps the win at 3.71x vs the dwf4 reference ~10x. Fix in progress: lower $\alpha$ / stop at the
converged iterate. The bit-exact operator gate (3b) is unaffected and passes.

Pending: converge the frame (recover ~10x); then the $8^4$ Arnoldi bundle (extremal $D_W$ + lowest
$H_W(-M5)$ + CGNE count); then the R2 topology scan at $16^4+$ (correlate the win with $Q$).

## Build / run

```
bash /mnt/baracuda_14/dwms/Grid/scripts_nm/grid_freeprec_build_claude.sh   # reads build/grid-config; no reconfigure
# results in /mnt/baracuda_14/dwms/grid_freeprec_build_claude.log
```
Prereq: `libGrid.a` built (done) and `make install` populated `build/include`+`build/lib` (the script
runs it). To wire the test into Grid's own `make`: `./scripts/filelist` (or `./bootstrap.sh`) picks up
`tests/solver/Test_*.cc`, then reconfigure.
