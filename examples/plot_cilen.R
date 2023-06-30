#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "normal")
metric = ifelse(length(args) >= 2, args[2], "cilen")

method_list = c("dalea", "hmc", "vi", "dnne")
n_methods = length(method_list)
df_list = list()
for(method in method_list){
    infile = sprintf(
        "tablesfigures/%s/%s_%s.tsv",
        studyname, metric, method)
    x = scan(infile)
    df_list[[method]] = x
}
df = data.frame(df_list)
print(median(df[,1]) / median(df[,2]))

outfile_box = sprintf(
    "tablesfigures/%s/%s_box.png",
    studyname, metric)
png(outfile_box, 1000, 1000)
par(cex=2)
noisetype = list(
    normal="Gaussian",
    chisq="chi-squared",
    multimodal="Gaussian mixture")
main = paste(noisetype[[studyname]], "noise")
colnames(df) = toupper(colnames(df))
boxplot(
    df, main=main, ylim=c(0.6, 4.0),
    cex.main=2, cex.axis=2)
dev.off()
print(outfile_box)
