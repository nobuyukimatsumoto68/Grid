# plot_flowQ_claude.gp
# Overlay the flowed 5Li topological charge Q_5Li(tau) for a few configs, to read off the flow time
# where Q plateaus near an integer (that tau becomes the Q-binning definition for R2).
#
# usage: gnuplot -e "datdir='/projectnb/qfe/nmatsum/dwf/flowQ_dat'" plot_flowQ_claude.gp
# each flowQ_*.dat (written by Test_flowed_topocharge_claude): col1=tau col2=plaq col3=Q_clover col4=Q_5Li
# Accessibility: each config gets a distinct COLOR and a distinct POINT MARKER (color-blind safe).

if (!exists("datdir")) datdir='.'

set xlabel "flow time tau"
set ylabel "Q_5Li  (improved topological charge)"
set grid xtics ytics
set ytics 1                      # integer gridlines: a good plateau sits near an integer
set key outside right

files = system(sprintf("ls %s/flowQ_*.dat 2>/dev/null", datdir))
nf = words(files)
if (nf == 0) { print sprintf("no flowQ_*.dat found in %s", datdir); exit }

set terminal pngcairo size 1000,650 font ",11"
set output sprintf("%s/flowQ_5Li_claude.png", datdir)

plot for [i=1:nf] word(files,i) using 1:4 with linespoints \
     lc i pt ((i-1)%14)+1 ps 0.8 lw 1.5 \
     title system(sprintf("basename %s .dat | sed 's/flowQ_//'", word(files,i)))

print sprintf("wrote %s/flowQ_5Li_claude.png  (%d configs)", datdir, nf)
