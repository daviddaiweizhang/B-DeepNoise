#!/bin/bash
set -e

./sync_results.sh

for studyname in normal chisq multimodal abcd; do
    for metric in mse yfit_corr cide_corr cover_rate cilen devia_strata; do
        ./tabulate.sh $studyname $metric
    done
done

for studyname in normal chisq multimodal abcd; do
    ./plot_corr.R $studyname yfit_corr
    ./plot_corr.R $studyname cide_corr
    ./plot_cover_rate.R $studyname
    ./plot_devia_strata.R $studyname
    if [ $studyname != abcd ]; then
        ./plot_design.py $studyname
        ./sync_curve.sh $studyname
        ./plot_mse.R $studyname
        ./plot_cilen.R $studyname
    fi
done

./plot_cond_dist.py
