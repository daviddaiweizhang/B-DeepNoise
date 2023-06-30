#! /bin/bash

run () {
    dataset=$1
    fold=$2

    prefix="results/vision/$dataset/"

    if [ $fold == "gibbs" ]; then
        cmd="python -u multistate.py \
            --dataset=$dataset \
            --n-layers=4 \
            --n-nodes=250 \
            --seed=0 \
            --n-states=100 \
            --n-thinning=50 \
            --batch-size=1000 \
            --n-samples=100 \
            --prefix=$prefix"
    else
        cmd="python -u unistate.py \
            --dataset=$dataset \
            --n-layers=4 \
            --n-nodes=250 \
            --fold=$fold \
            --seed=0 \
            --epochs=10000 \
            --patience=100 \
            --batch-size=1000 \
            --samples=100 \
            --ensemble-only \
            --prefix=$prefix"
    fi

    outfile="${prefix}${fold}/log.txt"
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


for dataset in mnist; do
    for fold in gibbs; do
        run $dataset $fold
    done
    wait
done

wait
echo "All jobs finished."
