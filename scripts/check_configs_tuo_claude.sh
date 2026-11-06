#!/bin/bash
# check_configs_tuo_claude.sh
# READ-ONLY status of the SU(4) Nf=1 gauge-config streams on tuolumne (no rm, no writes, no job control).
# Per ensemble directory: number of <cfg>_lat.<traj> files, first / last trajectory, the trajectory spacings that occur,
# the time of the newest config, and whether its rng file is present. Then the user's flux jobs.
# Directories are DISCOVERED below /p/lustre5/matsumoto5 (any name; 24c, 24c _cont, 32c, ...), e.g.
#   conf_nc4nf1_2448/conf_nc4nf1_2448_b<beta>_m<mass>/conf_nc4nf1_2448_b<beta>_m<mass>_lat.<traj>
# Knobs: LUSTRE (search root), DEPTH (find -maxdepth, default 4).
# Usage (on tuolumne, any directory):
#   bash check_configs_tuo_claude.sh 2>&1 | tee check_configs_tuo_claude.log

set -u

LUSTRE=${LUSTRE:-/p/lustre5/matsumoto5}
DEPTH=${DEPTH:-4}
# ORIGINAL (fixed list; 32c was not under conf_nc4nf1_3264): TOPS="conf_nc4nf1_2448 conf_nc4nf1_2448_cont conf_nc4nf1_3264"
# now: DISCOVER every directory below ${LUSTRE} (up to DEPTH levels) holding config files <prefix>lat.<traj>
# (<cfg>_lat.<traj> or ckpoint_lat.<traj>), grouped per directory and file prefix

echo "=== config check  $(date)  host $(hostname)  (search ${LUSTRE}, depth ${DEPTH})"
echo
printf "  %-70s %6s %8s %8s  %-22s %-17s %s\n" "directory : prefix" "N" "first" "last" "spacings(count)" "newest" "rng(last)"
find "${LUSTRE}" -maxdepth "${DEPTH}" -type f -regex '.*lat\.[0-9]+$' -printf '%h\t%f\n' 2>/dev/null \
    | sed 's/lat\.[0-9]*$//' | sort -u | while IFS="$(printf '\t')" read -r d pre; do
    trajs=$(ls "${d}" 2>/dev/null | grep "^${pre}lat\.[0-9][0-9]*$" | sed "s/.*lat\.//" | sort -n)
    n=$(echo "${trajs}" | wc -l)
    first=$(echo "${trajs}" | head -n1)
    last=$(echo "${trajs}" | tail -n1)
    sp=$(echo "${trajs}" | awk 'NR > 1 {print $1 - p} {p = $1}' | sort -n | uniq -c | awk '{printf "%s(%s) ", $2, $1}')
    newest=$(date -r "${d}/${pre}lat.${last}" "+%Y-%m-%d %H:%M" 2>/dev/null)
    rng="no"
    if [ -e "${d}/${pre}rng.${last}" ]; then
        rng="yes"
    fi
    rel=${d#${LUSTRE}/}
    printf "  %-70s %6d %8d %8d  %-22s %-17s %s\n" "${rel} : ${pre}" "${n}" "${first}" "${last}" "${sp}" "${newest}" "${rng}"
done

echo
echo "=== flux jobs (running / pending, this user)"
if command -v flux > /dev/null 2>&1; then
    flux jobs -o "{id.f58:>12} {name:<32.32} {status_abbrev:>3} {t_remaining!F:>10} {nnodes:>5}" 2>&1
    echo
    echo "=== flux jobs finished in the last 2 days (this user)"
    # ORIGINAL (-a also listed the pending jobs): flux jobs -a --since=-2d -o ...
    flux jobs --filter=inactive --since=-2d -o "{id.f58:>12} {name:<32.32} {result:>10} {t_inactive!d:%m-%d %H:%M}" 2>&1 | head -n 40
else
    echo "  flux not found in PATH"
fi
echo
echo "=== done"
