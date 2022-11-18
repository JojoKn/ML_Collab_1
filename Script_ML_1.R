###ML for Econ, Assignment 1###

###Reproduction####

#####Required packages####
library(readr)
library(stargazer)
library(tidyr)
library(tidyverse)
library(leaps)
library(lubridate)
library(xgboost)
library(tree)
library(rpart)
library(rpart.plot)
library(caret) #Das ist gut.

#####Loading the data####
peru <- read_csv("peru_for_ml_course.csv")
peru<-subset(peru, is.na(peru$lnpercapitaconsumption)==FALSE)
View(peru)

####Random Split of the data####
set.seed(12345)

train<-sample(c(TRUE, FALSE), nrow(peru), replace=TRUE, prob = c(0.8, 0.2))
test<-!train

#Check ratio
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



###Trash 1: Forward Model####

reg_1_fwd<-regsubsets(lnpercapitaconsumption~regressors, data=peru_train, 
                      method="forward",
                      nvmax=63,
                      really.big = TRUE)
summary_reg_1_fwd<-summary(reg_1_fwd)

names(summary_reg_1_fwd)

####Plotting different information criteria and searching for optimal model####

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

####Specific Plot according to book contained in package####

par(mfrow=c(2,2))
plot(reg_1_fwd, scale="r2")
plot(reg_1_fwd, scale="adjr2")
plot(reg_1_fwd, scale="Cp")
plot(reg_1_fwd, scale="bic")
dev.off()

####Choosing model with minimum BIC####
coef_reg_1_fwd<-as.vector(coef(reg_1_fwd, d))

####Obtaining the relevant variables####
vars<-summary_reg_1_fwd$which[d,]
vars<-vars[vars==TRUE]
vars_names<-names(vars)
vars_names<-gsub("regressors", "", x=vars_names)
vars_names<-vars_names[-1]

####Building Predictor Matrix####

#For training data
regressors_fwd<-as.matrix(peru_train[, vars_names])
regressors_fwd<-as.matrix(cbind(rep(1,nrow(peru_train)), regressors_fwd))

#For Test Data
regressors_predict_fwd<-as.matrix(peru_test[, vars_names])
regressors_predict_fwd<-as.matrix(cbind(intercept, regressors_predict_fwd))

####Predicting the outcome variable####

y_hat_fwd<-regressors_fwd%*%as.vector(coef_reg_1_fwd)
y_hat_test_fwd<-regressors_predict_fwd%*%as.vector(coef_reg_1_fwd)

####Calculating test MSE####

train_mse_fwd<-mean((peru_train$lnpercapitaconsumption - y_hat_fwd)^2, na.rm=TRUE)
test_mse_fwd<-mean((peru_test$lnpercapitaconsumption - y_hat_test_fwd)^2, na.rm=TRUE)

#Absolutely marginal improvement over the previous test MSE and slightly worse train MSE.

###Trash 2: Growing a regression tree####

train_y<-peru_train[,1]
train_x<-peru_train[, grep("^d_*", colnames(peru))]
train_data<-data.frame(train_y,train_x )

test_y<-peru_test[,1]
test_x<-peru_test[,grep("^d_*", colnames(peru))]
test_data<-data.frame(test_y, test_x)

tree1<-tree(lnpercapitaconsumption ~ ., data = train_data)
summary(tree1)
plot(tree1)
text(tree1, pretty=0)

tree1_predict<-predict(tree1, test_data)
summary(tree1_predict)

cv_tree1<-cv.tree(tree1, FUN=prune.tree, K=100)
cv_tree1

plot(cv_tree1$size, cv_tree1$dev, type="b")

prune_tree1<-prune.tree(tree1, best=8)
plot(prune_tree1)
text(prune_tree1, pretty=0)

hist(peru$lnpercapitaconsumption, freq=FALSE)
#Predictions seem rather high. And using very little variables in tree

prune_tree1_predict<-predict(prune_tree1, test_data)

mean((test_data$lnpercapitaconsumption-prune_tree1_predict)^2)
mean((test_data$lnpercapitaconsumption-tree1_predict)^2)
#Not surprisingly, as CV lead to tree with same size as before, the MSE is still the same
#However, significantly higher than in Linear Regression

###Trash 2.5: Boosting the tree using xgBoost####

#define predictor and response variables in training set
train_x = data.matrix(train_data[, -1])
train_y = train_data[,1]

#define predictor and response variables in testing set
test_x = data.matrix(test_data[, -1])
test_y = test_data[, 1]

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model = xgb.train(data = xgb_train, 
                  max.depth = ncol(test_x), 
                  watchlist=watchlist, 
                  nrounds = 1000)

#define final model
model_xgboost = xgboost(data = xgb_train,  
                        max.depth = ncol(test_x), 
                        nrounds = 51, 
                        verbose = 0)
summary(model_xgboost)

#use model to make predictions on test data
pred_y = predict(model_xgboost, xgb_test)

mean((test_data$lnpercapitaconsumption-pred_y)^2) #MSE
min(model$evaluation_log$train_rmse)
caret::RMSE(test_y, pred_y)
#Poor performance on the test dataset?

hist(pred_y, freq=FALSE)
hist(test_y, freq = FALSE)
#Distributions look very similar
hist(prune_tree1_predict)
#Pruning result histogram looks strange




