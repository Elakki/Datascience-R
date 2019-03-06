
# Loading required library 
library(corrplot)
library(randomForest)
library(gmodels)
library(tigerstats)
library(MASS)
library(caTools)
library(data.table)
library(ROCR)
library(ineq)
library(rpart)
library(rpart.plot)


#Accessing the file
setwd("C:/BABI/R")

ead <- read.csv("HR_Employee_Attrition_Data.csv")

#EDA 
str(ead)

summary(ead)

# removing the unwanted columns
eadf <- subset(ead, select = -c(Over18,EmployeeCount,EmployeeNumber,StandardHours))

rowPerc(xtabs(~PerformanceRating+Attrition,data=eadf))

rowPerc(xtabs(~Gender+Attrition,data=eadf))

# Correlation
corrplot(cor(ead[,-c(2,3,5,7,8,12,16,18,22,23)]))


# Removing the highly correlated columns
eadf <- subset(eadf, select = -c(JobLevel,YearsInCurrentRole,YearsSinceLastPromotion))

# Randomforest to find importanct of variables 
eadf.mrf = randomForest(Attrition ~ ., data=eadf, ntree = 50, mtry = 5, importance = TRUE, method="class")

impVar <- round(randomForest::importance(eadf.mrf), 2)

impVar[order(impVar[,3], decreasing=TRUE),]

# Removing the low scoring variables
eadf <- subset(eadf, select = -c(PerformanceRating,Department,Gender))

# Create training and testing sets
set.seed(111)
split = sample.split(eadf$Attrition, SplitRatio = 0.70)

eadf.dev = subset(eadf, split == TRUE)
eadf.test = subset(eadf, split == FALSE)

c(nrow(eadf.dev), nrow(eadf.test))
rowPerc(xtabs(~Attrition,data=eadf.dev))
rowPerc(xtabs(~Attrition,data=eadf.test))


# Neural Network Model
library(neuralnet)

#Convert catrogircal variables as numeric to do the nerual network model 
eadf.dev$BusinessTravel <- as.numeric(eadf.dev$BusinessTravel)
eadf.dev$EducationField <- as.numeric(eadf.dev$EducationField)
eadf.dev$OverTime <- as.numeric(eadf.dev$OverTime)
eadf.dev$JobRole <- as.numeric(eadf.dev$JobRole)
eadf.dev$MaritalStatus <- as.numeric(eadf.dev$MaritalStatus)

eadf.dev$Attrition <- ifelse(eadf.dev$Attrition =="No",0,1)

#Scalling the model for better performance 
eadf.dev.scale <- scale(eadf.dev[-c(2)])
eadf.dev.scale <- cbind(eadf.dev[2], eadf.dev.scale)

# NN model 
eadf.nm <- neuralnet(formula = Attrition ~ ., data = eadf.dev.scale[-2],
                     hidden = 3,
                 err.fct = "sse",
                 linear.output = FALSE,
                 lifesign = "full",
                 lifesign.step = 100,
                 threshold = 0.1,
                 stepmax = 2000
                 )

eadf.dev$Prob = eadf.nm$net.result[[1]]

quantile(eadf.nm$net.result[[1]] , c(0,1,5,10,25,50,75,90,95,99,100)/100)


# Converting the cateogrical to numeric in test data 
eadf.test$BusinessTravel <- as.numeric(eadf.test$BusinessTravel)
eadf.test$EducationField <- as.numeric(eadf.test$EducationField)
eadf.test$OverTime <- as.numeric(eadf.test$OverTime)
eadf.test$JobRole <- as.numeric(eadf.test$JobRole)
eadf.test$MaritalStatus <- as.numeric(eadf.test$MaritalStatus)

eadf.test$Attrition <- ifelse(eadf.test$Attrition =="No",0,1)

# scalling test data 
eadf.test.scale <- scale(eadf.test[-c(2)])

eadf.test.scale <- cbind(eadf.test[2], eadf.test.scale)

# Prdict the target variable 

eadf.test$probnn <-compute (eadf.nm,eadf.test.scale[,-2])$net.result
eadf.test$predictionnn <- ifelse(eadf.test$prob>0.5,1,0)

# Confusion Matrix

confusionnm <- table(eadf.test$Attrition,eadf.test$predictionnn)

confusionnm

TP <- confusionnm[2, 2]
TN = confusionnm[1, 1]
FP = confusionnm[1, 2]
FN = confusionnm[2, 1]

ClassificationAccuracynm <- round((TP + TN) /(TP + TN + FP + FN),4)
sensitivitynm <- round(TP / (FN + TP),4)
specificitynm <-round(TN / (TN + FP),4)
precisionnm  <- round(TP / (TP + FP),4)

ClassificationAccuracynm
sensitivitynm
specificitynm
precisionnm

## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
    ifelse(x<deciles[2], 2,
    ifelse(x<deciles[3], 3,
    ifelse(x<deciles[4], 4,
    ifelse(x<deciles[5], 5,
    ifelse(x<deciles[6], 6,
    ifelse(x<deciles[7], 7,
    ifelse(x<deciles[8], 8,
    ifelse(x<deciles[9], 9, 10
    ))))))))))
}

eadf.test$deciles <- decile(eadf.test$probnn)
tmp_DT = data.table(eadf.test)
rank <- tmp_DT[, list(
  cnt = length(Attrition), 
  cnt_resp = sum(Attrition), 
  cnt_non_resp = sum(Attrition == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);

print(rank)

# Model performance 
gininm = ineq(eadf.test$predictionnn, type="Gini")
pred <- ROCR::prediction(eadf.test$probnn, eadf.test$Attrition)
perf <- performance(pred, "tpr", "fpr")
KSnm <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
aucnm <- as.numeric(auc@y.values)

KSnm <- round(KSnm,4)
aucnm <- round(aucnm,4)
gininm <- round(gininm,4)

KSnm
aucnm
gininm


# CART Model 

r.ctrl = rpart.control(minsplit=50, minbucket = 5, cp = 0, xval = 10)

eadf.cart <- rpart(formula =  Attrition~ ., data = eadf.dev [-26], method = "class", control = r.ctrl)

rpart.plot(eadf.cart)


## to find how the tree performs and derive cp value
printcp(eadf.cart)
plotcp(eadf.cart)


#Post purning
eadf.prcart<- prune(eadf.cart, cp= 0.0085 ,"CP")
printcp(eadf.prcart)
rpart.plot(eadf.prcart)

#Predicition 
eadf.test$predict.class <- predict(eadf.prcart, eadf.test, type="class")
eadf.test$predict.score <- predict(eadf.prcart, eadf.test)

# confustion Matrix 
confusioncar <- table(eadf.test$Attrition, eadf.test$predict.class)
confusioncar 

TP <- confusioncar[2, 2]
TN = confusioncar[1, 1]
FP = confusioncar[1, 2]
FN = confusioncar[2, 1]

ClassificationAccuracycar <- round((TP + TN) /(TP + TN + FP + FN),4)
sensitivitycar <- round(TP / (FN + TP),4)
specificitycar <-round(TN / (TN + FP),4)
precisioncar  <- round(TP / (TP + FP),4)

ClassificationAccuracycar
sensitivitycar
specificitycar
precisioncar


ginicar = ineq(eadf.test$predict.score[,2], type="Gini")

predcar <- ROCR::prediction(eadf.test$predict.score[,2], eadf.test$Attrition)
perfcar <- performance(predcar, "tpr", "fpr")
KScar <- max(attr(perfcar, 'y.values')[[1]]-attr(perfcar, 'x.values')[[1]])
auccar <- performance(predcar,"auc"); 
aucar <- as.numeric(auccar@y.values)

ginicar<- round(ginicar,4)
KScar <- round (KScar,4)
aucar <- round(aucar,4)
ginicar
KScar
aucar


Modelmeasures <- data.frame (measurename = c("ClassificationAccuracy","sensitivity","specificity","precision","KS","AUC","Gini" ),
                             Neuralnetworks = c (ClassificationAccuracynm,sensitivitynm,specificitynm,precisionnm,KSnm,aucnm,gininm),
                              cart= c (ClassificationAccuracycar,sensitivitycar,specificitycar,precisioncar,KScar,aucar,ginicar))
Modelmeasures

# Emsemple Model 

predic <- data.frame(pcart= eadf.test$predict.class,pnn =  as.factor(eadf.test$predictionnn) )

predic$ensmbleavg <- (as.numeric(predic$pcart)+as.numeric(predic$pnn)-2)/2

predic$enclass <- ifelse (predic$ensmbleavg > 0.5,1,0)

eadf.test$enclass <-predic$enclass

confusionen <- table(eadf.test$Attrition,eadf.test$enclass)

confusionen

TP <- confusionen[2, 2]
TN <- confusionen[1, 1]
FP <- confusionen[1, 2]
FN <- confusionen[2, 1]

ClassificationAccuracyen <- round((TP + TN) /(TP + TN + FP + FN),4)
sensitivityen <- round(TP / (FN + TP),4)
specificityen <- round(TN / (TN + FP),4)
precisionen <- round(TP / (TP + FP),4)

ClassificationAccuracyen
sensitivityen
specificityen
precisionen

Modelmeasures <- data.frame (measurename = c("ClassificationAccuracy","sensitivity","specificity","precision","KS","AUC","Gini" ),
                             Neuralnetworks = c (ClassificationAccuracynm,sensitivitynm,specificitynm,precisionnm,KSnm,aucnm,gininm),
                              cart= c (ClassificationAccuracycar,sensitivitycar,specificitycar,precisioncar,KScar,aucar,ginicar),
                              ensemple =c(ClassificationAccuracyen,sensitivityen,specificityen,precisionen,'NA','NA','NA'))
Modelmeasures


