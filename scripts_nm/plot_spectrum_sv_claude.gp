# Plot the sv |D_DW|^2 spectrum (singular values + chirality) for a hard config.
# Data: log/spectrum_sv_ckpoint_lat.<cfg>_claude.dat
#   col1 mode  col2 sigma  col3 lambda  col4 chi_q  col5 chi5  col6 wall_s0  col7 wall_sLm1
# Colour-blind convention: two series by BOTH marker AND colour.
#   topological (|chi_q|>0.8) = red filled circle (lc 1 pt 7);  near-zero non-topological = blue filled square (lc 3 pt 5).
# Run:  gnuplot plot_spectrum_sv_claude.gp    (override DAT/CFG with -e "CFG='ckpoint_lat.640'")

if (!exists("CFG")) CFG='ckpoint_lat.640'
DAT = '/projectnb/qfe/nmatsum/dwf/log/spectrum_sv_'.CFG.'_claude.dat'
OUTDIR = '/projectnb/qfe/nmatsum/dwf/log/'

set datafile commentschars "#="
set key top center
set grid

# ---- Plot 1: chirality vs singular value (the topological trio stands out) ----
set terminal pngcairo size 900,650 enhanced font "Helvetica,14"
set output OUTDIR.'spectrum_sv_'.CFG.'_chirality_claude.png'
set title "|D_{DW}|^2 low modes: chirality vs singular value  (".CFG.", 16^4 b2.6)"
set xlabel "singular value {/Symbol s}_i = sqrt({/Symbol l}_i)"
set ylabel "physical-quark chirality  {/Symbol c}_q = <q|{/Symbol g}_5|q>/<q|q>"
set yrange [-1.1:1.1]
plot \
  DAT using 2:(abs($4)>0.8 ? $4 : 1/0) with points lc 1 pt 7 ps 1.8 title "topological (|{/Symbol c}_q|>0.8)", \
  DAT using 2:(abs($4)<=0.8 ? $4 : 1/0) with points lc 3 pt 5 ps 1.6 title "near-zero non-topological"

# ---- Plot 2: the singular-value spectrum vs mode index ----
set output OUTDIR.'spectrum_sv_'.CFG.'_sigma_claude.png'
set title "|D_{DW}|^2 singular-value spectrum  (".CFG.", 16^4 b2.6)"
set xlabel "mode index (ascending {/Symbol l})"
set ylabel "singular value {/Symbol s}_i"
set yrange [*:*]
set autoscale y
plot \
  DAT using 1:(abs($4)>0.8 ? $2 : 1/0) with points lc 1 pt 7 ps 1.8 title "topological ({/Symbol c}_q ~ +1)", \
  DAT using 1:(abs($4)<=0.8 ? $2 : 1/0) with points lc 3 pt 5 ps 1.6 title "near-zero non-topological"

# ---- Plot 3: wall localisation (surface-mode check) ----
set output OUTDIR.'spectrum_sv_'.CFG.'_wall_claude.png'
set title "|D_{DW}|^2 low modes: L_s-wall localisation  (".CFG.")"
set xlabel "mode index"
set ylabel "slice weight  ||{/Symbol y}(s)||^2 / ||{/Symbol y}||^2"
set yrange [0:1]
plot \
  DAT using 1:6 with points lc 7 pt 9 ps 1.6 title "wall s=0", \
  DAT using 1:7 with points lc 2 pt 11 ps 1.6 title "wall s=L_s-1"
