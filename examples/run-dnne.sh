#! /bin/bash
set -e

run () {
    dataset=$1
    split=$2

    prefix="results/tmp-2345/dnne/$dataset/$split/"

    cmd="python -u dnne_experiment.py \
        --dataset=$dataset \
        --split=$split \
        --prefix=$prefix \
        --seed=0"

    outfile="${prefix}log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    start=`date +%s`
    $cmd >> $outfile 2>&1
    date >> $outfile
    end=`date +%s`
    runtime=$((end-start))
    echo "runtime: ${runtime} sec" >> $outfile
    echo $outfile
}

for dataset in protein-tertiary-structure; do
    for split in {00..00}; do
        run $dataset $split
    done
    wait
done

wait
echo "All jobs finished."
