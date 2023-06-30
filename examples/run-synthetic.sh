#! /bin/bash

run () {
    nfeatures=$1
    xdist=$2
    ydist=$3
    ntrain=$4
    split=$5
    fold=$6

    prefix="results/features${nfeatures}-${xdist}-${ydist}-ntrain${ntrain}/$split/"
    if [ $fold == "gibbs" ]; then
        cmd="python -u multistate.py \
            --x-distribution=$xdist \
            --y-deviation=$ydist \
            --n-features=$nfeatures \
            --n-train=$ntrain \
            --split=$split \
            --seed=0 \
            --n-states=100 \
            --n-thinning=1 \
            --batch-size=16 \
            --n-samples=100 \
            --prefix=$prefix"
    else
        cmd="python -u unistate.py \
            --x-distribution=$xdist \
            --y-deviation=$ydist \
            --n-features=$nfeatures \
            --n-train=$ntrain \
            --split=$split \
            --fold=$fold \
            --seed=0 \
            --epochs=10000 \
            --patience=1000 \
            --batch-size=500 \
            --samples=100 \
            --prefix=$prefix"
    fi

    outfile="${prefix}${fold}/log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    start=`date +%s`
    echo $cmd >> $outfile
    $cmd >> $outfile 2>&1
    date >> $outfile
    end=`date +%s`
    runtime=$((end-start))
    echo "runtime: ${runtime} sec" >> $outfile
    echo $outfile

}

for nfeatures in 1; do
    for ntrain in 0200; do
        for xdist in uniform; do
            for ydist in heteroscedastic; do
                for split in 00; do
                    for fold in 0; do
                        run $nfeatures $xdist $ydist $ntrain $split $fold
                    done
                done
            done
        done
    done
done

wait
echo "All jobs finished."
