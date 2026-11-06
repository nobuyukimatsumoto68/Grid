#!/bin/bash
# check_configs_tuo_claude.sh
# READ-ONLY status of the SU(4) Nf=1 gauge-config streams on tuolumne (no rm, no writes, no job control).
# Per ensemble directory: number of <cfg>_lat.<traj> files, first / last trajectory, the trajectory spacings that occur,
# the time of the newest config, and whether its rng file is present. Then the user's flux jobs.
#   24c      : /p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b<beta>_m<mass>/
#   24c cont : /p/lustre5/matsumoto5/conf_nc4nf1_2448_cont/conf_nc4nf1_2448_b<beta>_m<mass>/  (run_prod24c_claude.sh)
#   32c      : /p/lustre5/matsumoto5/conf_nc4nf1_3264/conf_nc4nf1_3264_b<beta>_m<mass>/      (run_prod32c_claude.sh)
# Usage (on tuolumne, any directory):
#   bash check_configs_tuo_claude.sh 2>&1 | tee check_configs_tuo_claude.log

set -u

LUSTRE=${LUSTRE:-/p/lustre5/matsumoto5}
TOPS="conf_nc4nf1_2448 conf_nc4nf1_2448_cont conf_nc4nf1_3264"

echo "=== config check  $(date)  host $(hostname)"
for top in ${TOPS}; do
    echo
    echo "=== ${LUSTRE}/${top}"
    if [ ! -d "${LUSTRE}/${top}" ]; then
        echo "  (missing)"
        continue
    fi
    printf "  %-40s %6s %8s %8s  %-22s %-17s %s\n" "ensemble" "N" "first" "last" "spacings(count)" "newest" "rng(last)"
    for d in "${LUSTRE}/${top}"/*/; do
        cfg=$(basename "${d}")
        trajs=$(ls "${d}" 2>/dev/null | grep "_lat\.[0-9][0-9]*$" | sed "s/.*_lat\.//" | sort -n)
        if [ -z "${trajs}" ]; then
            printf "  %-40s %6s\n" "${cfg}" "0"
            continue
        fi
        n=$(echo "${trajs}" | wc -l)
        first=$(echo "${trajs}" | head -n1)
        last=$(echo "${trajs}" | tail -n1)
        sp=$(echo "${trajs}" | awk 'NR > 1 {print $1 - p} {p = $1}' | sort -n | uniq -c | awk '{printf "%s(%s) ", $2, $1}')
        lastfile=$(ls "${d}" | grep "_lat\.${last}$" | head -n1)
        newest=$(date -r "${d}/${lastfile}" "+%Y-%m-%d %H:%M" 2>/dev/null)
        rngfile=$(echo "${lastfile}" | sed "s/_lat\./_rng./")
        rng="no"
        if [ -e "${d}/${rngfile}" ]; then
            rng="yes"
        fi
        printf "  %-40s %6d %8d %8d  %-22s %-17s %s\n" "${cfg}" "${n}" "${first}" "${last}" "${sp}" "${newest}" "${rng}"
    done
done

echo
echo "=== flux jobs (running / pending, this user)"
if command -v flux > /dev/null 2>&1; then
    flux jobs -o "{id.f58:>12} {name:<32.32} {status_abbrev:>3} {t_remaining!F:>10} {nnodes:>5}" 2>&1
    echo
    echo "=== flux jobs finished in the last 2 days (this user)"
    flux jobs -a --since=-2d -o "{id.f58:>12} {name:<32.32} {result:>10} {t_inactive!d:%m-%d %H:%M}" 2>&1 | head -n 40
else
    echo "  flux not found in PATH"
fi
echo
echo "=== done"
