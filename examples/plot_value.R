#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "normal")
metric = ifelse(length(args) >= 2, args[2], "devia")

method_list = c("dalea", "hmc")
n_methods = length(method_list)
df_list = list()
for(method in method_list){
    infile = sprintf(
        "tablesfigures/%s/%s_%s.tsv",
        studyname, metric, method)
    x = scan(infile)
    df_list[[toupper(method)]] = x
}
df = data.frame(df_list)

outfile_box = sprintf(
    "tablesfigures/%s/%s_box.png",
    studyname, metric)
png(outfile_box, 1000, 1000)
par(cex=2)
boxplot(df, main=metric)
dev.off()
print(outfile_box)

outfile_scatter = sprintf(
    "tablesfigures/%s/%s_scatter.png",
    studyname, metric)
png(outfile_scatter, 1000, 1000)
plot(df[,1], df[,2], main=metric)
abline(0, 1)
dev.off()
print(outfile_scatter)
