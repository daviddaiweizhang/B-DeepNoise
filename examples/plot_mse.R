#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "normal")
metric = ifelse(length(args) >= 2, args[2], "mse")

method_list = c("dalea", "hmc", "vi", "dnne", "sgd")
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
noisetype = list(
    normal="Gaussian",
    chisq="chi-squared",
    multimodal="Gaussian mixture")
main = paste(noisetype[[studyname]], "noise")
if(studyname == "normal"){
    yaxt = "s"
    mar = c(7,4.2,2,1)
    width = 920
} else {
    yaxt = "n"
    mar = c(7,0.5,2,0.5)
    width = 800
}
png(outfile_box, width, 1000)
par(mar=mar, cex=2)
boxplot(
    df, main=main, ylim=c(0.03, 1.6), log="y",
    cex.main=2.5, cex.axis=2, las=2, yaxt=yaxt)
dev.off()
print(outfile_box)
