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
library(caret) #Large package for regression and classification trees
library(DiagrammeR) #Needed for plotting with xgBoost
library(ParBayesianOptimization) #Parameter Tuning with xgBoost
library(mlbench)
library(doParallel)


#####Loading the data####
peru_full <- read_csv("peru_for_ml_course.csv")
peru<-subset(peru_full, is.na(peru$lnpercapitaconsumption)==FALSE)
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



###Idea 1: Forward Model####

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

###Idea 2: Growing a regression tree####

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

###Idea 2.5: Boosting the tree using xgBoost####

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

#Setting the tuning parameters:
max.depth <- 2 #Maximum depth of each tree. Set to zero for no constraints
nrounds <- 1000 #Number of boosting iterations
eta <-0.1 #Standard is 0.3; learning rate
gamma<- 0 #Standard is 0; Minimum loss reduction required
subsample <- 1 #Standard is 1; prevent overfitting by randomly sampling the from training
min_child_weight <- 1 #Standard is 1; Minimum sum of instance weight (hessian) needed in a child
lambda <- 1 #Standard is 1; L2 regularization term on weights
alpha <- 0 #Standard is 0; L1 regularization term on weights.


#fit XGBoost model and display training and testing data at each iteration
model = xgb.train(data = xgb_train, 
                  max.depth = max.depth, 
                  watchlist = watchlist, 
                  nrounds = nrounds,
                  eta = eta,
                  gamma = gamma,
                  subsample = subsample, 
                  min_child_weight = min_child_weight)

par(mfrow=c(1,2))
plot(seq(1:nrounds), model$evaluation_log$train_rmse, type = "l")
plot(seq((nrounds/2):nrounds), model$evaluation_log$train_rmse[(nrounds/2):nrounds], type = "l")
dev.off()


par(mfrow=c(1,2))
plot(seq(1:nrounds), model$evaluation_log$test_rmse, type = "l")
plot(seq((nrounds/2):nrounds), model$evaluation_log$test_rmse[(nrounds/2):nrounds], type = "l")
dev.off()

#use model to make predictions on test data
#pred_y = predict(model_xgboost, xgb_test)
#pred_y_train = predict(model_xgboost, xgb_train)

pred_y = predict(model, xgb_test)
pred_y_train = predict(model, xgb_train)

#MSE
mean((train_data$lnpercapitaconsumption-pred_y_train)^2)
mean((test_data$lnpercapitaconsumption-pred_y)^2)

min(model$evaluation_log$train_rmse)

#Distribution
par(mfrow=c(1,2))
hist(pred_y, freq=FALSE)
hist(test_y, freq = FALSE)
dev.off()
#Distributions look very similar
hist(prune_tree1_predict)
#Pruning result histogram looks strange

#Diagnostic tools
#Plot single tree out of the many iterations (not very helpful)
xgb.plot.tree(model=model_xgboost, trees=100)
#Plot tree over all iterations including importance of the different variables
xgb.plot.multi.trees(model=model)

# get information on how important each feature is
importance_matrix <- xgb.importance(model = model)

# and plot it
xgb.plot.importance(importance_matrix)


#From the fitting procedure, the best performance on the test data set is reached with 
#max.depth=2; everything else reduces greatly the MSE on the training set, but not on the 
#test set. From there:
#Parameter Tuning with Bayesian Optimization?

#https://www.r-bloggers.com/2022/01/using-bayesian-optimisation-to-tune-a-xgboost-model-in-r/

obj_func <- function(eta, max_depth, min_child_weight, subsample, lambda, alpha) {
  
  xgb_train = xgb.DMatrix(data = train_x, label = train_y)
  
  param <- list(
    
    # Hyperparameters 
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    lambda = lambda,
    alpha = alpha,
    
    # Tree model; default booster
    booster = "gbtree",
    
    # Regression problem; default objective function
    objective = "reg:squarederror",
    
    # Use RMSE
    eval_metric = "rmse")
  
  xgbcv <- xgb.cv(params = param,
                  data = xgb_train,
                  nround = 100,
                  nfold=10,
                  prediction = TRUE,
                  early_stopping_rounds = 5,
                  verbose = 1,
                  maximize = F)
  
  lst <- list(
    
    # First argument must be named as "Score"
    # Function finds maxima so inverting the output
    Score = -min(xgbcv$evaluation_log$test_rmse_mean),
    
    # Get number of trees for the best performing model
    nrounds = xgbcv$best_iteration
  )
  
  return(lst)
}

#Setting bounds

bounds <- list(eta = c(0.0001, 0.3),
               max_depth = c(1L, 10L),
               min_child_weight = c(1, 50),
               subsample = c(0.1, 1),
               lambda = c(1, 10),
               alpha = c(0, 10))

#Setting Seed for reproducibility
set.seed(1234)

#Initializing the process to run in parallel
cl <- makeCluster(8)
registerDoParallel(cl)
clusterExport(cl,c("train_x", "train_y"))
clusterEvalQ(cl,expr= {
  library(xgboost)
})


#Bayesian Optimzation. Plot gives back the progress of the optimization. If lower plot (utility)
#is approaching zero, one can be optimistic that optimal parameter values were identified
#(see the instructions manual for bayesOpt)
bayes_out <- bayesOpt(FUN = obj_func, 
                      bounds = bounds, 
                      initPoints = length(bounds) + 2, 
                      iters.n = 30,
                      verbose=2,
                      plotProgress = TRUE,
                      parallel = TRUE)

# Show relevant columns from the summary object 
bayes_out$scoreSummary[1:5, c(3:8, 13)]
# Get best parameters
data.frame(getBestPars(bayes_out))

opt_params2 <- append(list(booster = "gbtree", 
                          objective = "reg:squarederror", 
                          eval_metric = "rmse"), 
                     getBestPars(bayes_out))
# Run cross validation 
xgbcv2 <- xgb.cv(params = opt_params,
                data = xgb_train,
                nround = 100,
                nfold=10,
                prediction = TRUE,
                early_stopping_rounds = 5,
                verbose = 1,
                maximize = F)
# Get optimal number of rounds
nrounds2 = xgbcv2$best_iteration
# Fit a xgb model
model2 <- xgboost(data = xgb_train, 
                  params = opt_params, 
                  maximize = F, 
                  early_stopping_rounds = 5, 
                  nrounds = nrounds2, 
                  verbose = 1
                  )

pred_y_opt = predict(model2, xgb_test)
pred_y_train_opt = predict(model2, xgb_train)

mean((train_data$lnpercapitaconsumption-pred_y_train_opt)^2)
mean((test_data$lnpercapitaconsumption-pred_y_opt)^2)

#As a result, the Bayesian Parameter Tuning has indeed improved the predictions even further.

#Documentation on Parameters to set with xgBoost: 
# https://xgboost.readthedocs.io/en/latest/parameter.html

#For putting in the numbers into the evaluation tool, these are the predictions on the
#Out-of-sample data:
oos<-data.matrix(peru_full[is.na(peru_full$lnpercapitaconsumption), grep("^d_*", colnames(peru_full))])
xgb_oos<-xgb.DMatrix(data = oos)
pred_y_oos = predict(model2, xgb_oos)
write(pred_y_oos, "Prediction_xgboost.txt", ncolumns=1)

#Miscellaneous
summary(is.na(peru_full))
table(is.na(peru_full))

importance_matrix = xgb.importance(colnames(xgb_train), model = model2)
xgb.plot.importance(importance_matrix)
