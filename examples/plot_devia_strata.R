#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "normal")
metric = ifelse(length(args) >= 2, args[2], "devia")

method_list = c("dalea", "hmc", "vi", "dnne")
n_methods = length(method_list)
df_list = list()
for(method in method_list){
    infile = sprintf(
        "tablesfigures/%s/%s_strata_%s.tsv",
        studyname, metric, method)
    x = read.table(infile)
    x = x**2  # MSE
    stride = 100 / ncol(x)
    qstart = (0:(ncol(x)-1)) * stride
    qstop = qstart + stride - 1
    strata_names = sprintf("%02d-%02d", qstart, qstop)
    colnames(x) = strata_names
    df_list[[method]] = x
}

noisetype = list(
    normal="Gaussian",
    chisq="chi-squared",
    multimodal="Gaussian mixture")

for(method in method_list){
    outfile = sprintf(
        "tablesfigures/%s/mse_strata_%s.png",
        studyname, method)
    main = paste0(toupper(method))
    if(method == "dalea"){
        yaxt = "s"
        mar = c(8, 3.5, 2, 0.5)
        width = 920
    } else {
        yaxt = "n"
        mar = c(8, 0.5, 2, 0.5)
        width= 800
    }
    png(outfile, width, 1000)
    par(cex=2, mar=mar)
    boxplot(
        log10(df_list[[method]]),
        main=main,
        ylim=c(-3.5, 0.5),
        cex.main=2.5, cex.axis=3, las=2,
        yaxt=yaxt, ylab=""
    )
    dev.off()
    print(outfile)
}
