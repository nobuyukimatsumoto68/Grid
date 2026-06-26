#!/bin/bash
# run_disc_m05_check_claude.sh
# m=0.05 is the CROSSOVER between m=0.01 (deflation wins ~5x) and m=0.1 (deflation
# eigensolve-overhead bound -> mixedprec better). This sets up ONE done m=0.05 config
# in scratch and submits BOTH timing runs -- defl and mixedprec -- so we compare
# per-config wall and decide which program m=0.05 belongs to. Touches no production.
# After: compare "conf <n> took <s>s" from disc_defl_val_* vs disc_mp_val_*, and diff
# obs_defl/traces.<gam>.<conf> vs obs_mp/...  vs the .ref copies (correctness).

date; hostname
basedir=$(pwd)
source env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
CONFROOT=/p/lustre5/matsumoto5/conf_nc4nf1_2448
OBSROOT=/p/lustre5/matsumoto5/obs_nc4nf1_2448
SCRATCH=/p/lustre5/matsumoto5/validate_defl
gams="id g5 gx gy gz gt gxg5 gyg5 gzg5 gtg5"

mass=0.05; massstr=0p0500; beta=10.84; betastr=10p840
cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
latdir_real="${CONFROOT}/${cfgfilename}"
obsdir_real="${OBSROOT}/obs_nc4nf1_2448_b${betastr}_m${massstr}"

conf=$(ls ${obsdir_real}/traces.id.* 2>/dev/null | head -n1 | sed 's/.*traces\.id\.//')
if [ -z "${conf}" ]; then echo "no done m0.05 config (need a reference); abort"; exit 1; fi
latfile="${latdir_real}/${cfgfilename}_lat.${conf}"

slat="${SCRATCH}/${betastr}_${massstr}/lat"
sobs_defl="${SCRATCH}/${betastr}_${massstr}/obs_defl"
sobs_mp="${SCRATCH}/${betastr}_${massstr}/obs_mp"
mkdir -p ${slat} ${sobs_defl} ${sobs_mp}
ln -sf ${latfile} ${slat}/${cfgfilename}_lat.${conf}
# one gamma (id) is enough as a correctness reference -- the solve is identical
# across gammas (only the contraction Gamma differs). Timing needs no reference.
if [ -f "${obsdir_real}/traces.id.${conf}" ]; then
    cp -n "${obsdir_real}/traces.id.${conf}" "${sobs_defl}/traces.id.${conf}.ref"
    cp -n "${obsdir_real}/traces.id.${conf}" "${sobs_mp}/traces.id.${conf}.ref"
fi

echo "m0.05 check on conf ${conf}: defl -> ${sobs_defl}, mixedprec -> ${sobs_mp}"
flux batch --job-name=discval_${betastr}_${massstr}_defl \
    --env=builddir=$builddir --env=mass=$mass --env=beta=$beta \
    --env=latdir=$slat --env=obsdir=$sobs_defl submit_disc_defl_validate_claude.sh
flux batch --job-name=discval_${betastr}_${massstr}_mp \
    --env=builddir=$builddir --env=mass=$mass --env=beta=$beta \
    --env=latdir=$slat --env=obsdir=$sobs_mp submit_disc_mixedprec_validate_claude.sh
