#!/usr/bin/env Rscript

rm(list=ls())

args = commandArgs(trailingOnly=T)
studyname = ifelse(length(args) >= 1, args[1], "abcd")
metric = ifelse(length(args) >= 2, args[2], "cide_corr")

method_list = c("dalea", "hmc", "vi", "dnne")
if(metric == "yfit_corr"){
    method_list = c(method_list, "sgd")
}
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
las = 2
width = 1000
yaxt = "s"
if(metric == "yfit_corr"){
    df = df^2
    if(studyname == "abcd"){
        main = NULL
        ylim = NULL
        mar=c(8,5,2,1)
    } else {
        main = expression("Testing r"^2)
        ylim = c(0.35, 1.00)
        mar=c(7,5,2,1)
    }
} else if(metric == "cide_corr"){
    if(studyname == "abcd"){
        print(colSums(df > 0))
        main = NULL
        ylim = NULL
        mar=c(8,5.5,2,1)
    } else {
        ylim = c(-0.15, 0.55)
        main = paste(noisetype[[studyname]], "noise")
        if(studyname == "normal"){
            width = 920
            yaxt = "s"
            mar=c(7,4,2,1)
        } else {
            width = 800
            yaxt = "n"
            mar = c(7,0.5,2,0.5)
        }
    }
}
png(outfile_box, width, 1000)
par(cex=2, mar=mar)
boxplot(
    df, main=main, ylim=ylim, cex.main=2.5, cex.axis=2, las=las,
    yaxt=yaxt)
dev.off()
print(outfile_box)
