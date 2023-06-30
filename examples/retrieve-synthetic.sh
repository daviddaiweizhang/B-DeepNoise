for xdist in uniform interval; do
    for ydist in heteroscedastic skewed multimodal; do
        for ntrain in 0500 1000 2000 4000; do
            dataset=features1-${xdist}-${ydist}-ntrain${ntrain}
            echo $dataset
            pattern="results/${dataset}/*/gibbs/eval-synthetic.txt"
            cat $pattern |
                grep qq |
                cut -d" " -f2 |
                Rscript -e 'x=scan("stdin"); round(c(mean(x), sd(x))*1000)';
        done
    done
done
