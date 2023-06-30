#!/bin/bash
set -e

studyname=$1  # e.g. 23544744
metric=$2  # e.g. devia_strata

for method in dalea hmc vi dnne sgd; do
    evaluations="results/$studyname/*/$method/evaluation.txt"
    outfile="tablesfigures/$studyname/${metric}_${method}.tsv"
    mkdir -p `dirname $outfile`
    grep "^$metric: " $evaluations | cut -d" " -f2- > $outfile
    echo $outfile
done
