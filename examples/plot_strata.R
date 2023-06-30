#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "normal")
metric = ifelse(length(args) >= 2, args[2], "devia")
outfile = sprintf(
    "tablesfigures/%s/%s_strata.png",
    studyname, metric)

method_list = c("dalea", "hmc")
n_methods = length(method_list)
df_list = list()
for(method in method_list){
    infile = sprintf(
        "tablesfigures/%s/%s_strata_%s.tsv",
        studyname, metric, method)
    x = read.table(infile)
    colnames(x) = paste0("stratum_", 1:ncol(x))
    df_list[[method]] = x
}
x_min = max(sapply(df_list, min))
x_max = max(sapply(df_list, max))
png(outfile, n_methods * 1000, 1000)
par(mfrow=c(1, n_methods), cex=2)
for(method in method_list){
    main = toupper(method)
    boxplot(
        df_list[[method]],
        ylim=c(x_min*0.9, x_max*1.1), main=main,
        ylab=metric,
        xlab="quantile of posterior CI width")
}
dev.off()
print(outfile)
