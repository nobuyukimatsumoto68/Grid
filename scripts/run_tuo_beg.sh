#!/bin/bash
# set -e


# https://hpc.llnl.gov/banks-jobs/running-jobs/batch-system-cross-reference-guides

masses=(0.4 0.3 0.2 0.1)
beta0s=(11.045 11.035 10.99 10.865) # tick 0.01; three of them
nbeta_m1=(1 1 1 1) # nbeta=1 means one beta; select tick below

basedir=$(pwd)

jjmax=${#masses[@]}
for((jj=0;jj<$jjmax;jj++))
do
    cd $basedir

    m=${masses[$jj]}
    beta0=${beta0s[$jj]}

    rundir=/p/lustre5/matsumoto5/32_64_$m
    # mkdir -p $rundir
    Nt=64
    xml=ip_hmc_mobius.xml
    script=submit_hmc_tuolumne.sh

    mkdir -p ${rundir}
    cp -f $xml ${rundir}
    cp -f $script ${rundir}
    cd ${rundir}

    imax=${nbeta_m1[$jj]}
    echo $imax
    for((i=0;i<$imax;i++))
    do
        beta=$(echo "$beta0 + $i*0.01" | bc -l | sed '$s/0$//') # @@@
        dir=beta${beta}m${m}
        mkdir -p $dir
        mv -f $xml $dir
        mv -f $script $dir
        echo $dir
        cd $dir
        sed -i "/gauge_beta/{s/10.00/${beta}/}" $xml
        sed -i "/mass/{s/0.1/${m}/}" $xml
        # sed -i "/StartingType/{s/ColdStart/CheckpointStart/}" $xml # @@@ because beg file
        nsteps=30 # @@@ TUNING
        echo $nsteps
        sed -i "/MDsteps/{s/30/${nsteps}/}" $xml

        # only in beg file
        date=$(date '+%Y%m%s%N')
        seed=$(($date % 100))
        RANDOM=$seed
        ser=$(echo ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2})
        par=$(echo ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2})
        sed -i "/serial_seeds/{s/21 32 73 64 45/$ser/}" $xml
        sed -i "/serial_seeds/{s/63 79 81 99 10/$par/}" $xml
        cp -f $xml ip_hmc_mobius_0.xml
        flux batch $script # mybatchscript
        cd ..
    done
done
