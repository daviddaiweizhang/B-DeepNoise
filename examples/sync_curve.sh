#!/bin/bash
set -e

studyname=$1
seed=0001
if [ $studyname = "normal" ]; then
    seed=0006
fi
for method in dalea hmc vi dnne sgd; do
    host=greatlakes.arc-ts.umich.edu
    user=daiweiz
    workdir=/home/daiweiz/dalea_project/examples/results
    filename=$workdir/${studyname}/$seed/${method}/curve.png
    src=$user@$host:$filename
    dst=tablesfigures/${studyname}/curve_${method}.png
    rsync -am -e ssh --progress $src $dst
    echo $dst
done
