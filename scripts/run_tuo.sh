#!/bin/bash
# set -e


masses=(0.4 0.3 0.2 0.1)
beta0s=(11.045 11.035 10.99 10.865) # tick 0.01; three of them
nbeta_m1=(1 1 1 1) # nbeta=1 means one beta; select tick below
MDSteps=(16 16 18 20) # nbeta=1 means one beta; select tick below


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
        st=$(ls ckpoint_lat.* | cut -f2 -d'/' | sed 's/ckpoint_lat.//g' | sort --version-sort | tail -n1)
        echo $st
        sed -i "/StartTrajectory/{s/0/${st}/}" $xml
        sed -i "/gauge_beta/{s/10.00/${beta}/}" $xml
        sed -i "/mass/{s/0.1/${m}/}" $xml
        sed -i "/StartingType/{s/ColdStart/CheckpointStart/}" $xml # @@@
        sed -i "/NoMetropolisUntil/{s/20/0/}" $xml # @@@
        nsteps=${MDSteps[$jj]} # @@@ TUNING
        echo $nsteps
        sed -i "/MDsteps/{s/30/${nsteps}/}" $xml

        # date=$(date '+%Y%m%s%N')
        # seed=$(($date % 100))
        # RANDOM=$seed
        # ser=$(echo ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2})
        # par=$(echo ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2} ${RANDOM:0:2})
        # sed -i "/serial_seeds/{s/21 32 73 64 45/$ser/}" $xml
        # sed -i "/serial_seeds/{s/63 79 81 99 10/$par/}" $xml
        # cp -f $xml ip_hmc_mobius_0.xml

        xmls=($(ls ip_hmc_mobius_* | sed 's/ip_hmc_mobius_//' | sed 's/.xml//'))
        next=$(echo $(echo ${xmls[-1]}"+1") | bc)
        cp -f $xml ip_hmc_mobius_${next}.xml
        if [ "$#" -eq 1 ]; then
            # echo "@@@ TO BE IMPLEMENTED @@@"
            flux batch --dependency=afterany:$1 $script
            # flux batch $script # mybatchscript
            # bsub -w "done($1)" < $script # mybatchscript
        else
            flux batch $script # mybatchscript
            # bsub < $script # mybatchscript
        fi
        cd ..
    done
done
