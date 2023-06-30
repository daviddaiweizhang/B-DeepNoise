rm(list=ls())
set.seed(123)

interleave = function(x){
    stopifnot(is.factor(x))
    idx = unlist(lapply(levels(x), function(le) 1:sum(x==le)))
    order(x)[order(idx)]
}

factors.to.integers = function(x){
    w = model.matrix(~., x)
    data.frame(apply(w[, 2:ncol(w), drop=F], 2, as.integer))
}

infile = "data/neuroimaging/abcd/raw/BioStats_phenos_rest_nback_include.csv"
outfile = "data/neuroimaging/abcd/covariates.tsv"

ids = c("Subject", "site_num")
features = c(
             "g", "p", "Age", "Female",
             "HighestParentalEducation", "HouseholdMaritalStatus",
             "HouseholdIncome", "RaceEthnicity",
             "tfmri_nb_all_beh_c2b_rate")
x = read.csv(infile, header=T, na.strings=c("NA", ""))
x = x[x$Include.nBack, c(ids, features)]
x = x[!duplicated(x),]
x = x[complete.cases(x),]
rownames(x) = x$Subject
x = x[,c(ids, features)]
x = x[sample(nrow(x)),]

x$Female = as.integer(x$Female == "yes")
x$HouseholdMaritalStatus = as.integer(x$HouseholdMaritalStatus == "yes")

x$HighestParentalEducation = factor(x$HighestParentalEducation, c("< HS Diploma", "HS Diploma/GED", "Some College", "Bachelor", "Post Graduate Degree"))
x$HighestParentalEducation = as.integer(x$HighestParentalEducation)
x$HouseholdIncome = factor(x$HouseholdIncome, c("[<50K]", "[>=50K & <100K]", "[>=100K]"))
x$HouseholdIncome = as.integer(x$HouseholdIncome)
x$RaceEthnicity = factor(x$RaceEthnicity, c("Other", "White", "Hispanic", "Black", "Asian"))
x$RaceEthnicity = x$RaceEthnicity[interleave(x$RaceEthnicity)]

# convert factors to integers
is.fa = colnames(x) %in% colnames(Filter(is.factor, x))
if(sum(is.fa) > 0){
    w = cbind(x[,!is.fa], factors.to.integers(x[,is.fa, drop=F]))
} else {
    w = x
}

w$site_num = as.factor(w$site_num)

# rename column names
colnames(w)[colnames(w) == "sitenum"] = "siteid"
colnames(w)[colnames(w) == "Subject"] = "subjectid"
colnames(w)[colnames(w) == "tfmri_nb_all_beh_c2b_rate"] = "CorrectRate2bk"
# clean column names
colnames(w) = gsub("[^[:alnum:]]", "", colnames(w))

write.table(w, outfile, col.names=T, row.names=F, quote=F, sep="\t")
print(outfile)
