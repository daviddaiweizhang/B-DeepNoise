#! /bin/bash

run () {
    nfeatures=$1
    xdist=$2
    ydist=$3
    ntrain=$4
    split=$5
    fold=$6

    prefix="results/dnne/features${nfeatures}-${xdist}-${ydist}-ntrain${ntrain}/$split/"
    cmd="python -u dnne_experiment.py \
        --x-distribution=$xdist \
        --y-deviation=$ydist \
        --n-features=$nfeatures \
        --n-train=$ntrain \
        --split=$split \
        --seed=0 \
        --prefix=$prefix"

    outfile="${prefix}log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    $cmd >> $outfile 2>&1
    date >> $outfile
    echo $outfile

}

for nfeatures in 1; do
    for split in {00..19}; do
        for xdist in interval uniform; do
            for ntrain in 4000; do
                for ydist in heteroscedastic skewed multimodal; do
                    run $nfeatures $xdist $ydist $ntrain $split $fold &
                done
                wait
            done
        done
    done
done

wait
echo "All jobs finished."
