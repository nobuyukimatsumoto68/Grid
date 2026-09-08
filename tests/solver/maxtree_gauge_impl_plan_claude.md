# Maximal-tree gauge frame -- assessment + implementation plan

Nobu (2026-09-07) wants to test a maximal-tree gauge for the free-limit frame Omega, alongside
Coulomb (Coulomb is done -- a --coulomb flag on the flowscan). Maximal tree is NOT in Grid; this is a
custom implementation. Plan + honest assessment below; CONFIRM before coding.

## What it is
Pick a maximal spanning tree of lattice links; the gauge transform Omega(x) that sets every tree link
to identity is unique up to the root. Omega(x) = ordered product of links along the tree path root->x.
Non-tree links become the gauge-invariant holonomies. On a PERIODIC lattice a tree cannot cover the
Nd non-contractible cycles -> the Polyakov loops in each direction remain (residual global gauge /
holonomy). Standard choice (Creutz maximal axial): U_t=1 on all but one time slice; on that slice
U_z=1 on a line; etc. -- a recursive nested-axial construction.

## Why it MIGHT be interesting here
Unlike Landau (which only drives ||A||->0 approximately, and is prone to Gribov copies), maximal tree
EXACTLY trivialises a pure-gauge configuration (all links ->1). A flowed config at t0 is pure gauge +
lump, so maximal tree would trivialise the bulk exactly and concentrate ALL residual non-triviality
in the non-tree links that enclose the lump. That is spiritually the "avoid the lump" frame Nobu
asked for, achieved exactly for the bulk. So it is a qualitatively different Omega from Landau/Coulomb.

## Why it probably will NOT beat the floor (honest caveat)
Everything measured this session says the M0 win floor (~4-5x, M0-resid ~0.38) is GAUGE-INVARIANT:
- squeezing the lump to a point changed nothing (win/M0-resid flat);
- the Q=0 config framed no better than Q!=0.
The floor is set by the near-zero Dirac modes below the free kernel's gap, which NO choice of Omega
(Landau, Coulomb, maximal tree) can remove -- M0=Omega^dag F Omega is a fixed free kernel conjugated
by a gauge rotation; it cannot manufacture the missing low modes. Also a WARNING specific to maximal
tree: the non-tree links are left at their FULL value (holonomies, can be far from 1), so U^L is NOT
uniformly near-free -- it is near-1 on the tree (1/Nd of links) and arbitrary off-tree. That likely
makes M0 WORSE, not better, because F assumes all links near 1. So my expectation is maximal tree
frames the free kernel POORLY. Worth one test to confirm, not worth heavy investment.

## Implementation (if approved) -- new header Grid/Grid/qcd/utils/MaxTreeGaugeFix_claude.h
`template<class Gimpl> static void MaxTreeGaugeTransform(GaugeLorentz& U, GaugeMat& xform)`:
1. Build Omega by nested axial sweep (all serial cumulative products along each axis; use Cshift +
   CovShift accumulations, root at origin). Careful with periodic wrap (leave the last link per cycle
   -> Polyakov loop remains).
2. Apply: U -> Omega U Omega^dag via SU<Nc>::GaugeTransform (as the c1/M1 code already does).
3. Return xform=Omega for FreeLimitPreconditioner (same interface as the Landau xform).
Wire into the flowscan/shrinkflow driver behind --gauge maxtree (alongside landau/coulomb).
Cost: ~120-180 lines; the nested cumulative product on a distributed lattice is the fiddly part
(serial dependence along each axis breaks data-parallelism -> either a slow Cshift ladder or a
gather-to-boss serial build at 16^4, which is fine single-node). Ref for the construction: any
lattice text on axial/maximal-tree gauge (e.g. Creutz, "Quarks, Gluons and Lattices").

## RECOMMENDATION
Run the Coulomb 747 scan now (free). For maximal tree: I lean AGAINST building it unless you
specifically want the exact-bulk-trivialisation datapoint, because the evidence says frame choice
cannot beat the zero-mode floor, and maximal tree's off-tree links likely make M0 worse. If you want
it anyway as a definitive check, I will implement the header above. Confirm.
