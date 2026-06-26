#!/bin/bash
#
# Finalizer for the OLD 4 ensembles. For each (ensemble, pipeline) it computes
# the MISSING configs and submits ONE 4h job that does only those:
#  - CONN: builds CONFLIST = configs with no mesons_conn.<conf>.h5 and feeds it
#    to submit_meson_momproj_finalize_claude.sh (4h). The binary DOES skip a
#    complete config (~0.65s of Grid time), but each config is a separate flux
#    run, so the MPI init + Grid setup + teardown costs ~6s per skip (measured);
#    re-skipping hundreds of done configs can eat the whole wall before reaching
#    the gaps. Feeding only the missing configs removes that re-skip tax.
#  - DISC: the disc binary loops internally and self-skips existing traces via an
#    instant exists() check (before any read), so there is no relaunch waste --
#    we just submit the existing 4h submit_disc_tuolumne.sh for ensembles that
#    still have gaps.
#
# Each piece is an explicit call at the bottom: COMMENT OUT a call once that
# piece is finished (one-by-one). The helper also self-skips if already complete,
# as a safety. Auto-chains (afterany) on any active same-name job so a re-run
# queues behind a still-running one. Run from workdir on a flux node.
#
# Note: conn "missing" = configs with NO output h5 (fast existence check). A
# partial/incomplete h5 (rare now that the wall blocker stops cleanly) is NOT
# re-fed here; run the normal driver once to repair those (the binary's
# output_complete recomputes partials).

date; hostname
basedir=$(pwd)
source env.sh

M5=1.5
gams="id g5 gx gy gz gt gxg5 gyg5 gzg5 gtg5"
conn_script=submit_meson_momproj_finalize_claude.sh
disc_script=submit_disc_tuolumne.sh

# build both binaries once (incremental: conn via Makefile_claude, disc via automake)
make -C /usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4 -f Makefile_claude baryons_0000_dirac_claude
BIN=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/baryons_0000_dirac_claude
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
make -C ${builddir}/examples disc_multipleGamma_binary_claude

# active same-name job ids -> "--dependency=afterany:ID ..." (auto-chain)
active_deps() {
    local jobname=$1 deps="" id
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    echo "${deps}"
}

# CONN finalizer for one ensemble. args: mass massstr betastr
finalize_conn() {
    local mass=$1 massstr=$2 betastr=$3
    local cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    local latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    local obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"
    mkdir -p ${obsdir}

    local conf missing=""
    for conf in $(ls ${latdir}/${cfgfilename}_lat.* 2>/dev/null | sed "s|.*_lat\.||" | sort -n); do
        [ -f "${obsdir}/mesons_conn.${conf}.h5" ] || missing="${missing} ${conf}"
    done
    local n=$(echo ${missing} | wc -w)
    if [ "${n}" -eq 0 ]; then
        echo "conn ${cfgfilename}: complete, nothing to finalize"
        return
    fi

    local jobname="conn_${betastr}_${massstr}"
    local deps=$(active_deps ${jobname})
    cd ${basedir}; cp ${conn_script} ${obsdir}; cd ${obsdir}
    echo "submit conn ${jobname}: ${n} missing configs${deps:+ deps:${deps# }}"
    flux batch --job-name=${jobname} ${deps} --env=BIN=$BIN --env=M5=$M5 --env=mass=$mass --env=cfgfilename=$cfgfilename --env=latdir=$latdir --env=obsdir=$obsdir --env=CONFLIST="${missing# }" ${conn_script}
    cd ${basedir}
}

# DISC finalizer for one ensemble. args: mass massstr beta betastr
finalize_disc() {
    local mass=$1 massstr=$2 beta=$3 betastr=$4
    local cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    local latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    local obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"
    mkdir -p ${obsdir}

    local conf g n=0
    for conf in $(ls ${latdir}/${cfgfilename}_lat.* 2>/dev/null | sed "s|.*_lat\.||" | sort -n); do
        for g in ${gams}; do
            if [ ! -f "${obsdir}/traces.${g}.${conf}" ]; then n=$((n+1)); break; fi
        done
    done
    if [ "${n}" -eq 0 ]; then
        echo "disc ${cfgfilename}: complete, nothing to finalize"
        return
    fi

    local jobname="disc_${betastr}_${massstr}"
    local deps=$(active_deps ${jobname})
    cd ${basedir}; cp ${disc_script} ${obsdir}; cd ${obsdir}
    echo "submit disc ${jobname}: ${n} missing configs${deps:+ deps:${deps# }}"
    flux batch --job-name=${jobname} ${deps} --env=builddir=$builddir --env=mass=$mass --env=beta=$beta --env=latdir=$latdir --env=obsdir=$obsdir ${disc_script}
    cd ${basedir}
}

# ============ pieces: comment out a line once that piece is finished ============
# CONN: all 4 originals COMPLETE as of 2026-06-22 (b11p045 filled by the finalizer).
# finalize_conn 0.4 0p4000 11p045   # DONE 450/450 (filled 2026-06-22)
# finalize_conn 0.3 0p3000 11p035   # DONE 350/350
# finalize_conn 0.2 0p2000 10p990   # DONE 185/185
# finalize_conn 0.1 0p1000 10p865   # DONE 183/183

# DISC (as of 2026-06-22 b10p990 done; the other three still have gaps):
finalize_disc 0.4 0p4000 11.045 11p045   # 381/450
finalize_disc 0.3 0p3000 11.035 11p035   # 336/350
# finalize_disc 0.2 0p2000 10.99  10p990   # DONE 185/185
finalize_disc 0.1 0p1000 10.865 10p865   # 177/183
