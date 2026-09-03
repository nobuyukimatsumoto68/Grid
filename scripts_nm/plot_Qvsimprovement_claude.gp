# Q vs preconditioner improvement (R2). M0 = blue filled square (lc 3 pt 5); a second series (M1) will
# be red filled circle (lc 1 pt 7) once the --ops m1 runs land -- distinct marker AND colour (color-blind).
# Run: /usr/bin/gnuplot plot_Qvsimprovement_claude.gp   (writes freeprec_Qvsimprovement_claude.png)
set terminal pngcairo size 900,650 enhanced font "Helvetica,15"
set output "freeprec_Qvsimprovement_claude.png"
set title "R2: free-prec vs CGNE D_W-apply ratio vs topological charge (16^4 Iwasaki {/Symbol b}2.6)"
set xlabel "Q  (flowed Q_{5Li} at {/Symbol t}~2)"
set ylabel "D_W-apply ratio  (CGNE / free-prec)"
set xrange [-8.5:-2.5]
set yrange [0:7.5]
set xtics 1
set ytics 1
set grid ytics lc rgb "#dddddd"
set key top right
DAT = "freeprec_Qvsimprovement_claude.dat"
# x = signed Q (col 4). M0 = blue filled square (lc 3 pt 5), M1 = red filled circle (lc 1 pt 7) --
# distinct marker AND colour (color-blind). M1 = leading D_W correction (honest total D_W).
plot DAT using 4:2 with points pt 5 ps 1.6 lc 3 title "free-prec (M0)", \
     DAT using 4:6 with points pt 7 ps 1.6 lc 1 title "free-prec (M1)"
