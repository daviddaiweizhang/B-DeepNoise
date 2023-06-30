#! /bin/bash
set -e

run () {
    dataset=$1
    split=$2

    cmd="python -u vi-experiment.py \
        --dataset=$dataset \
        --split=$split \
        --seed=0"

    outfile="results/tmp-2345/vi/$dataset/$split/log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    $cmd >> $outfile 2>&1
    date >> $outfile
    echo $outfile
}

for dataset in yacht bostonHousing energy; do
    for split in {00..00}; do
        run $dataset $split &
    done
done

for dataset in concrete wine-quality-red; do
    for split in {00..00}; do
        run $dataset $split &
    done
done

for dataset in kin8nm power-plant naval-propulsion-plant; do
    for split in {00..00}; do
        run $dataset $split &
    done
done

for dataset in protein-tertiary-structure; do
    for split in {00..00}; do
        run $dataset $split &
    done
done

wait
echo "All jobs finished."
