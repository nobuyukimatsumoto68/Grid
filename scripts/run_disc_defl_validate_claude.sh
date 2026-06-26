#!/bin/bash
# run_disc_defl_validate_claude.sh
# Set up + submit pdebug single-config VALIDATION/TIMING runs of the SPEED disc
# binary on m=0.1/0.3/0.4 (each on ONE already-done config). For each ensemble it:
#   - picks the first DONE config (one with existing traces),
#   - builds a SCRATCH latdir with just that config's _lat file (symlink),
#   - makes a SCRATCH obsdir and copies the OLD reference traces there as
#     traces.<gam>.<conf>.ref (the new run writes traces.<gam>.<conf> alongside),
#   - submits submit_disc_defl_validate_claude.sh (pdebug, single config).
# Never deletes/overwrites production. To RE-RUN a validation you must remove the
# scratch obsdir's traces.<gam>.<conf> yourself first (the binary self-skips them).
# After it runs: compare traces.<gam>.<conf> vs .ref (must agree ~1e-7) and read the
# "conf <n> took <s>s" / "eigensolve Nconv" lines for the per-config timing.

date; hostname
basedir=$(pwd)
source env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
script=submit_disc_defl_validate_claude.sh

CONFROOT=/p/lustre5/matsumoto5/conf_nc4nf1_2448
OBSROOT=/p/lustre5/matsumoto5/obs_nc4nf1_2448
SCRATCH=/p/lustre5/matsumoto5/validate_defl
gams="id g5 gx gy gz gt gxg5 gyg5 gzg5 gtg5"

masses=(0.1     0.3     0.4)
massstrs=(0p1000 0p3000 0p4000)
betas=(10.865   11.035  11.045)
betastrs=(10p865 11p035 11p045)

jmax=${#masses[@]}
for((j=0;j<$jmax;j++)); do
    cd ${basedir}
    mass=${masses[$j]}; massstr=${massstrs[$j]}
    beta=${betas[$j]};  betastr=${betastrs[$j]}
    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir_real="${CONFROOT}/${cfgfilename}"
    obsdir_real="${OBSROOT}/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    # pick the first DONE config (has reference traces)
    conf=$(ls ${obsdir_real}/traces.id.* 2>/dev/null | head -n1 | sed 's/.*traces\.id\.//')
    if [ -z "${conf}" ]; then echo "no done config for ${cfgfilename}, skip"; continue; fi
    latfile="${latdir_real}/${cfgfilename}_lat.${conf}"
    if [ ! -f "${latfile}" ]; then echo "lat file missing: ${latfile}, skip"; continue; fi

    # scratch latdir (one config) + scratch obsdir (with reference .ref traces)
    slat="${SCRATCH}/${betastr}_${massstr}/lat"
    sobs="${SCRATCH}/${betastr}_${massstr}/obs"
    mkdir -p ${slat} ${sobs}
    ln -sf ${latfile} ${slat}/${cfgfilename}_lat.${conf}
    # one gamma (id) is enough as a correctness reference (same solve for all gammas)
    [ -f "${obsdir_real}/traces.id.${conf}" ] && cp -n "${obsdir_real}/traces.id.${conf}" "${sobs}/traces.id.${conf}.ref"

    echo "validate ${cfgfilename} on conf ${conf}: scratch obs ${sobs}"
    flux batch --job-name=discval_${betastr}_${massstr} \
        --env=builddir=$builddir --env=mass=$mass --env=beta=$beta \
        --env=latdir=$slat --env=obsdir=$sobs $script
done
