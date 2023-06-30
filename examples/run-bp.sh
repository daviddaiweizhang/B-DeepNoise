#! /bin/bash
set -e

run () {
    dataset=$1
    split=$2

    prefix="results/tmp-2345/bp/$dataset/$split/"

    cmd="python -u dnne_experiment.py \
        --dataset=$dataset \
        --split=$split \
        --seed=0 \
        --n-nets=1 \
        --prefix=$prefix"

    outfile="${prefix}log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    $cmd >> $outfile 2>&1
    date >> $outfile
    echo $outfile
}

for dataset in concrete energy bostonHousing yacht; do
    for split in {00..00}; do
        run $dataset $split &
    done
    wait
done

wait
echo "All jobs finished."
