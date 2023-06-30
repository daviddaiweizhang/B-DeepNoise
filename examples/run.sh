#! /bin/bash

run () {
    dataset=$1
    split=$2
    fold=$3

    prefix="results/$dataset/$split/"

    if [ $fold == "gibbs" ]; then
        cmd="python -u multistate.py \
            --dataset=$dataset \
            --split=$split \
            --seed=0 \
            --n-states=100 \
            --n-thinning=1 \
            --batch-size=16 \
            --n-samples=20 \
            --prefix=$prefix"
    else
        cmd="python -u unistate.py \
            --dataset=$dataset \
            --split=$split \
            --fold=$fold \
            --seed=0 \
            --epochs=10000 \
            --patience=200 \
            --batch-size=1000 \
            --samples=100 \
            --ensemble-only \
            --prefix=$prefix"
    fi

    outfile="results/$dataset/$split/$fold/log.txt"
    mkdir -p `dirname $outfile`
    echo $cmd
    date >> $outfile
    echo $cmd >> $outfile
    $cmd >> $outfile 2>&1
    # until $cmd >> $outfile 2>&1
    # do
    #     sleep $[ ( $RANDOM % 600 )  + 1 ]
    #     echo "Restarting $dataset $split $fold..."
    # done
    date >> $outfile
    echo $outfile

}


for dataset in power-plant; do
    for split in {00..04}; do
        for fold in gibbs; do
            run $dataset $split $fold &
        done
    done
done

for dataset in protein-tertiary-structure; do
    for split in {00..04}; do
        for fold in gibbs; do
            run $dataset $split $fold
        done
    done
    wait
done

wait
echo "All jobs finished."
