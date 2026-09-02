# R2 implementation plan: free-limit preconditioner at large volume with topology

SCC (BU Shared Computing Cluster) port. Branch `dwf_prec`. This plan drives the R2 physics deliverable
on top of the already-validated free-limit Mobius preconditioner (see
`../../qed2/dwf4_qcd_claude/grid_dwf_prec_handoff_claude.md`).

## Physics / goal (one paragraph)

The free-limit preconditioner $M_0 = \Omega^\dagger F \Omega$ (frame $\Omega$ from Landau-fixing the
Wilson-flowed config; $F$ = free Mobius inverse, FFT-diagonal) gives a large $D_W$-apply-count win over
CGNE at $Q=0$ small volume. **Open question R2:** is this a *trivial-sector* method? Wilson flow
**preserves** topological charge, so a $Q\neq0$ config (instanton) is the SU(N) analogue of the
compact-U(1) monopole obstruction (finding F2) and should **obstruct the frame** at $16^4$ and above.
**Deliverable:** at $16^4$ then $24^4$, correlate the FGMRES($M_0$)-vs-CGNE $D_W$-count win with the
**flowed** topological charge $Q$ of each config. Prediction: win at $Q=0$, win shrinks / vanishes at
$Q\neq0$.

Algorithm sources: idea R. Brower + T. Izubuchi; Mobius/Cayley Brower-Neff-Orginos arXiv:1206.5214;
Wilson (gradient) flow M. Luscher arXiv:1006.4518; improved topological charge (5Li / Luscher-Weisz)
per Grid `WilsonLoops::TopologicalCharge5Li`; Fourier-accelerated Landau fixing Davies et al.,
PRD 37 (1988) 1581.

## The binding constraint: no-restart FGMRES GPU memory

No-restart (large-`RestartLength` $R$) FGMRES stores $\sim R{+}3$ fermion fields on the GPU. Per field
(Ls=8, `WilsonImplD`, 192 B/4d-site) and total Krylov basis:

| volume | 1 field | R=128 basis | R=256 basis | fits on |
|---|---|---|---|---|
| $16^4$ | 0.10 GB | 13 GB | 26 GB | **1x A100 (40 GB)** |
| $24^4$ | 0.51 GB | 67 GB | 132 GB | **4x A100 / one 4-GPU node** (R256); 2 GPUs (R128) |
| $32^4$ | 1.61 GB | 211 GB | 417 GB | 8+ GPUs |

This is why we are on the cluster (Nobu). SCC has **25 nodes with 4 GPUs** (+ some 6/8) and a
CUDA-aware **`openmpi/4.1.5_nvidia-2025-25.5`**, so $24^4$ is a **single-node 4-GPU MPI** job (NVLink /
PCIe P2P, no multi-node needed). Restarted GMRES caps memory but can stall (dwf2 spread finding) -- so
memory, not just iteration count, sets $R$; treat $R$ vs #GPUs as the primary scaling knob.

## Files to be created / modified

- `grid_build_scc_mpi_claude.sh` (NEW) -- CUDA-aware multi-GPU MPI build variant of
  `grid_build_scc_claude.sh` (comms=mpi, `-ccbin mpic++`, nvidia openmpi module, accelerator-aware MPI).
- `grid_freeprec_scc_qsub_claude.sh` (MODIFY) -- add a multi-GPU path (`-l gpus=N`, `--mpi` grid,
  `mpirun`/`gpu binding`), and parametrize volume/restart/mass.
- `Test_flowed_topocharge_claude.cc` (NEW, `tests/solver/` or `tests/smearing/`) -- flow a config to a
  reference flow time and print the improved (5Li) + clover $Q$; used to bin configs. (Or extend the
  existing `Test_dwf_freeprec_claude.cc` frame diagnostics, which today print only unsmeared clover
  $Q$ via `WilsonLoops::TopologicalCharge` -- noisy, not integer.)
- config-generation script (NEW) -- SU(3) ensemble with a spread of $Q$ at $16^4$/$24^4$ (see chunk C).
- R2 scan wrapper + analysis/plot scripts (NEW).

## Ordered chunks

### Chunk A -- multi-GPU MPI CUDA build (enables $24^4$)
Files: `grid_build_scc_mpi_claude.sh`, `sourceme` module lines.
- `module load cuda/12.8` + a **CUDA-aware** `openmpi/4.1.5_nvidia-2025-25.5` + `gcc/13.2.0`.
- configure: `--enable-comms=mpi --enable-shm=nvlink --enable-accelerator=cuda --enable-simd=GPU`,
  `CXX=nvcc`, `-ccbin mpic++`, `--enable-accelerator-aware-mpi`, same multi-arch GENCODE, `-ldl -lrt`.
- Validate with a tiny 2-rank job before scaling. Keep the single-GPU comms=none build for $16^4$.

### Chunk B -- $16^4$ single-GPU R2 warmup (current build, no rebuild)
Files: `grid_freeprec_scc_qsub_claude.sh` (GRID=16.16.16.16).
- Run the headline on a real $16^4$ config; confirm the win persists at larger V and that R=256 (~26 GB)
  fits one A100. This is the R2 pipeline dry-run at a volume that still fits one GPU.

### Chunk C -- generate a FEW quenched SU(3) configs (decided with user 2026-09-01)
Files: `grid_gen_quenched_scc_claude.sh` (build + run a stock Grid pure-gauge HMC).
- Use Grid's stock quenched HMC -- NO custom driver: `tests/hmc/Test_hmc_IwasakiGauge.cc` (Iwasaki,
  rectangle-improved via `PlaqPlusRectangleAction`) or `tests/hmc/Test_hmc_WilsonGauge.cc` (Wilson
  plaquette). Build with `make -C tests/hmc Test_hmc_IwasakiGauge`. These write NERSC checkpoints.
- Generate only a HANDFUL of configs (the plan is: pick ONE with nontrivial $Q$ for the preconditioner,
  plus a $Q=0$ control). Quenched = pure gauge, NO fermion force -> cheap, runs on the current
  single-GPU comms=none build; multi-GPU is needed only for the $24^4$ FGMRES SOLVE, not for gen.
- Set $\beta$ (and action) so flow yields near-integer $Q$ and $Q\neq0$ actually appears; confirm the
  test's runtime knobs (volume, $\beta$, trajectory count, checkpoint interval).

### Chunk D -- measure flowed topological charge, pick a $Q\neq0$ config
Files: `grid_flow_topocharge_scc_claude.sh` (build + run a stock Grid flow tool).
- Use `tests/smearing/Test_WilsonFlow.cc` (or `HMC/ComputeWilsonFlow.cc`): flow each checkpoint and read
  $Q(t)$ from `WilsonLoops::TopologicalCharge` / `TopologicalCharge5Li` (improved). Unsmeared clover $Q$
  is too noisy to bin -- flow to a reference time where $Q$ is near-integer and plateaus.
- Output a config -> $Q$ table; select one $Q\neq0$ config (instanton) + one $Q=0$ control for chunk E.

### Chunk E -- R2 scan harness (multi-GPU) + analysis
Files: scan wrapper + plot scripts.
- **Two DIFFERENT flows (dwf_prec, confirmed 2026-09-01):** the FRAME flow that builds $\Omega$ is SHORT
  ($\tau=2$, e.g. `WilsonFlow` eps0.02 x100) -- over-flowing washes out the config you precondition. The
  Q-MEASUREMENT flow (chunk D) is LONG and runs on a COPY. **Solve the ORIGINAL unflowed config.**
- Per config: build $\Omega$ = Landau($\tau{=}2$ flow of a copy). **Landau maxiter scales with volume**
  (peer: 300@$4^4$, 1000@$8^4$; softest mode $\hat p^2_{\min}=2-2\cos(2\pi/L)\approx0.068$ at $L{=}24$,
  ~30x softer than $4^4$ -> **push maxiter well past 1000** and gate on the printed `dmuAmu`/functional
  **plateau**, NOT a fixed count -- the likely $24^4$ snag is frame convergence). `alpha = 0.1/16 =
  0.00625` (Grid's FA force is 16x too big, GaugeFix.h:198). Then run CGNE vs FGMRES($M_0$), record $D_W$
  counts + flowed $Q$ + flowed-fixed Landau functional.
- Bin the win by $Q$; plot win vs $|Q|$ ($Q=0$ vs $Q\neq0$). Deliverable = the obstruction test.
- Accessibility (project convention): distinguish series by marker AND color.

**R2 methodology (dwms-af, 2026-09-01) -- do these or the signal is confounded:**
- **Same-ensemble comparison.** Compare $Q\neq0$ vs $Q=0$ on the SAME ensemble (same Iwasaki action,
  $\beta$, volume, $L_s$, $m$). The win MAGNITUDE is action/$\beta$/volume-dependent, so do NOT benchmark
  against the $8^4$ Wilson-$\beta6$ 10.68x. Establish OUR Iwasaki $24^4$ **$Q=0$ baseline win first**,
  then read $Q\neq0$ configs as a FRACTIONAL change vs that baseline (else roughness confounds topology).
- **Report the flowed-fixed Landau functional PER CONFIG** alongside the $D_W$-count win -- it is the
  MECHANISM (R1), not just a diagnostic. Sharp prediction: if $Q\neq0$ obstructs the frame, the
  functional is HIGHER (worse frame) AND the win LOWER, **correlated**. The functional-vs-$Q$
  correlation is the smoking gun -- cleaner than the win alone (config-to-config Gribov/seed scatter,
  e.g. 3.71 vs 4.21 at $4^4$). Use **a few configs per $Q$ bin** to beat that scatter.

## Open questions to resolve with the user (before coding each chunk)

1. **GPU budget:** OK to target single-node **4-GPU** jobs (`-l gpus=4`) for $24^4$? Any multi-node
   need ($32^4$)? Confirm we may use `openmpi/4.1.5_nvidia-2025-25.5`.
2. **Config source:** generate a fresh quenched SU(3) HMC ensemble (what $\beta$, how many configs per
   volume), or reuse/scale an existing set? Target $Q$ bins $\{0,\pm1,\pm2\}$.
3. **Q definition (RESOLVED 2026-09-01):** no magic flow time -- **plateau method**: flow long on a copy,
   watch $Q(\tau)$, bin as $Q\neq0$ only where the **5Li improved** charge plateaus clearly away from 0
   (5Li plateaus faster/cleaner than clover). All native Grid (`WilsonLoops::TopologicalCharge` /
   `TopologicalCharge5Li`; `WilsonFlow` also auto-prints clover $Q$ per step). dwf4's SU(N) flowed-Q was
   DEFERRED and that session ended -- do not wait on it; the plateau method is self-sufficient.
   (peer's $\tau\sim4$-$8$ numbers were for $\beta6$ Wilson; our Iwasaki $\beta2.6$ differs -> plateau-watch.)
4. **Mass + restart:** $m=0.1$ and $0.02$ (light gives a bigger win but more iters/memory); target
   `RestartLength` vs #GPUs; is restarted GMRES an acceptable fallback if memory-bound?
5. **Volumes / params:** $16^4 \to 24^4$; confirm $L_s=8$, $M_5=1.8$, Shamir $b=1.5,c=0.5$ (and the
   $b=c=1$ control), anti-periodic time BC.
