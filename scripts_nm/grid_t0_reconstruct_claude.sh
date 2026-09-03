#!/bin/bash
# Reconstruct the flow scale t0 from EXISTING flow logs -- no re-run, no driver change.
# t^2<E>_plaq(t) = 36 t^2 (1 - plaq(t))  [Grid WilsonFlow.h:161-169, Nc=3; plaq = avgPlaquette column].
# t0/a^2 = flow time where the ENSEMBLE-MEAN t^2<E> crosses a reference (default 0.3, Luscher 1006.4518).
# Averages over all given flowQ dat files (cols: tau plaq Q_clover Q_5Li), so per-config plaquette-E
# scatter (large at smooth configs -- t^2E lives in the 4th decimal of plaq) averages down; t0 is an
# ensemble quantity. If the mean stays below REF, set REF lower (Nobu: "pick one number to define t0").
#
# Usage:
#   bash grid_t0_reconstruct_claude.sh flowQ_dat/flowQ_ckpoint_lat.*_claude.dat
#   REF=0.25 bash grid_t0_reconstruct_claude.sh <dir>/flowQ_*.dat

set -u

REF=${REF:-0.3}
if [ "$#" -lt 1 ]; then
  echo "usage: [REF=0.3] $0 <flowQ dat files...>"
  exit 1
fi

echo "t0 reconstruction: t^2<E>_plaq = 36 t^2 (1-plaq), reference = ${REF}, files = $#"
awk -v ref="${REF}" '
  !/[a-zA-Z]/ && NF>=2 {
    t=$1
    p=$2
    e=36.0*t*t*(1.0-p)
    s[t]+=e
    n[t]++
  }
  END {
    ntg=asorti(s, ord, "@ind_num_asc")
    for (i=1; i<=ntg; i++) {
      t=ord[i]
      m=s[t]/n[t]
      printf "  t=%.2f  <t^2E>=%.4f  (n=%d)\n", t, m, n[t]
      if (pm!="" && pm<ref && m>=ref) {
        t0=pt+(ref-pm)*(t-pt)/(m-pm)
        cross=t0
      }
      pt=t
      pm=m
    }
    if (cross!="") {
      printf "  => t0/a^2 = %.4f   sqrt(t0) = %.4f   (ensemble-mean crossing of %.3f)\n", cross, sqrt(cross), ref
    } else {
      printf "  => mean never reaches %.3f in range (max <t^2E>=%.4f at t=%.2f); lower REF to define t0.\n", ref, pm, pt
    }
  }' "$@"
