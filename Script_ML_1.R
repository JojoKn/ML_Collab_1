###ML for Econ, Assignment 1###

#####Required packages####
library(readr)
library(stargazer)
library(tidyr)
library(tidyverse)
library(leaps)


#####Loading the data####
peru <- read_csv("peru_for_ml_course.csv")
View(peru)

####Random Split of the data####
set.seed(12345)

train<-sample(c(TRUE, FALSE), nrow(peru), replace=TRUE, prob = c(0.5, 0.5))
test<-!train

#Check balanced?
sum(train)
sum(test)

#Generate the train dataset and the test dataset
peru_train<-peru[train,]
peru_test<-peru[test,]


####Generating a Matrix containing the relevant predictors####

#Acccording to paper, all variables coded with "d" in the beginning contain the relevant
#regressors.
regressors<-as.matrix(peru_train[, grep("^d_*", colnames(peru))])

#The online appendix provides information which of the dummy variables are excluded to 
#avoid multicollinearity issues
regressors<-subset(regressors, select = -c(d_crowd_lessthan1, d_lux_0, d_fuel_other,
                                            d_water_other, d_wall_other, d_roof_other,
                                            d_floor_other, d_insurance_0, d_h_educ_none))

####Running a regression####

reg_1<-lm(lnpercapitaconsumption~regressors, data=peru_train)
summary_reg_1<-summary(reg_1)

#Generate output
stargazer(reg_1)

####Obtaining the MSE####

train_mse<-mean(summary_reg_1$residuals^2)


####Obtaining MSE for hold-out sample####

#Getting the coefficients from the first regression
coef_reg_1<-as.vector(coef(reg_1))

#Setting up Matrix with precitors
regressors_predict<-as.matrix(peru_test[, grep("^d_*", colnames(peru))])
regressors_predict<-subset(regressors_predict, select = -c(d_crowd_lessthan1, d_lux_0, d_fuel_other, 
                                                            d_water_other, d_wall_other, d_roof_other, 
                                                            d_floor_other, d_insurance_0, d_h_educ_none))
intercept<-rep(1, nrow(regressors_predict))
regressors_predict<-as.matrix(cbind(intercept, regressors_predict))

#Calculating Predictions via Matrix product

y_hat_test<-regressors_predict%*%coef_reg_1

#Getting the test MSE

test_mse<-mean((peru_test$lnpercapitaconsumption - y_hat_test)^2, na.rm=TRUE)

#Result: train MSE and test MSE are very close together.


####Applying a machine learning technique of choice####

#Trash 1: Forward Model

reg_1_fwd<-regsubsets(lnpercapitaconsumption~regressors, data=peru_train, 
                      method="forward",
                      nvmax=63,
                      really.big = TRUE)
summary_reg_1_fwd<-summary(reg_1_fwd)

names(summary_reg_1_fwd)

par(mfrow=c(2,2))
plot(summary_reg_1_fwd$rss, xlab="Number of Variables", ylab="RSS", type="l")
a<-which.min(summary_reg_1_fwd$rss)
points(a, summary_reg_1_fwd$rss[a], col="red", cex=2, pch=20)

plot(summary_reg_1_fwd$adjr2, xlab="Number of Variables", ylab="Adjusted R-Squared", type="l")
b<-which.max(summary_reg_1_fwd$adjr2)
points(b, summary_reg_1_fwd$adjr2[b], col="red", cex=2, pch=20)

plot(summary_reg_1_fwd$cp, xlab="Number of Variables", ylab="Cp", type="l")
c<-which.min(summary_reg_1_fwd$cp)
points(c, summary_reg_1_fwd$cp[c], col="red", cex=2, pch=20)

plot(summary_reg_1_fwd$bic, xlab="Number of Variables", ylab="BIC", type="l")
d<-which.min(summary_reg_1_fwd$bic)
points(d, summary_reg_1_fwd$bic[d], col="red", cex=2, pch=20)

dev.off()

par(mfrow=c(2,2))
plot(reg_1_fwd, scale="r2")
plot(reg_1_fwd, scale="adjr2")
plot(reg_1_fwd, scale="Cp")
plot(reg_1_fwd, scale="bic")
dev.off

#Choosing model with minimum BIC
coef_reg_1_fwd<-as.vector(coef(reg_1_fwd, d))
colnames(coef_reg_1_fwd)

vars<-summary_reg_1_fwd$which[d,]
vars_names<-names(vars)
vars_names<-gsub("regressors", "", x=vars_names)
vars_names<-vars_names[-1]

regressors_predict_fwd<-as.matrix(peru_test[, vars_names])
regressors_predict_fwd<-as.matrix(cbind(intercept, regressors_predict_fwd))

y_hat_fwd<-regressors_predict_fwd%*%as.vector(coef_reg_1_fwd)

dim(regressors_predict_fwd)
dim(coef_reg_1_fwd)
length(coef_reg_1_fwd)

###Correct dimension issue

