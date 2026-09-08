# Plot the D_DW complex eigenvalues (shift-invert IRA near the physical Wilson branch sigma=-M5=-1.8).
# Data: log/spectrum_dwevals_<cfg>_claude.dat   (cols: i  Re(lambda)  Im(lambda)  |lambda-sigma|)
# gamma5-R5-Hermiticity -> conjugate-paired spectrum (symmetric about Im=0).
# Run:  gnuplot -e "CFG='ckpoint_lat.640'" plot_dwevals_claude.gp

# PREFIX = 'dwevals' (bare D_DW, sigma=-1.8) or 'm0devals' (M0 D_DW, sigma=0)
if (!exists("CFG")) CFG='ckpoint_lat.640'
if (!exists("PREFIX")) PREFIX='dwevals'
if (!exists("SIGMA")) SIGMA=-1.8
if (!exists("OPLABEL")) OPLABEL='D_{DW}'
DAT = '/projectnb/qfe/nmatsum/dwf/log/spectrum_'.PREFIX.'_'.CFG.'_claude.dat'
OUT = '/projectnb/qfe/nmatsum/dwf/log/spectrum_'.PREFIX.'_'.CFG.'_claude.png'

set datafile commentschars "#"
set terminal pngcairo size 850,700 enhanced font "Helvetica,14"
set output OUT
set title OPLABEL." complex eigenvalues near sigma=".sprintf("%.2f",SIGMA)."  (".CFG.", 16^4 b2.6)"
set xlabel "Re {/Symbol l}"
set ylabel "Im {/Symbol l}"
set grid
set key top right
set zeroaxis lt 0 lc rgb "#888888"
# mark the shift target sigma (the physical Wilson real-axis crossing -M5)
set arrow from SIGMA,graph 0 to SIGMA,graph 1 nohead lc rgb "#bbbbbb" dt 2
set label sprintf("{/Symbol s}=%.2f", SIGMA) at SIGMA,graph 0.95 left offset 0.5,0 tc rgb "#888888"
plot DAT using 2:3 with points lc 1 pt 7 ps 1.8 title "eig(D_{DW})"
