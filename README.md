# practical-machine-learning
rm(list=ls())
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
set.seed(12345)

UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training<-read.csv(url(UrlTrain))
testing<- read.csv(url(UrlTest))
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
TrainSet<-training[inTrain,]
TestSet<-training[-inTrain,]
dim(TrainSet)
dim(TestSet)
NZV<- nearZeroVar(TrainSet)
TrainSet<- TrainSet[,-NZV]
TestSet<-TestSet[,-NZV]
dim(TrainSet)
dim(TestSet)
AllNA<- sapply(TrainSet,function(x)mean(is.na(x)))> 0.95
TrainSet<-TrainSet[,AllNA==FALSE]
TestSet<-TestSet[,AllNA==FALSE]
dim(TrainSet)
dim(TestSet)
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
dim(TestSet)
corrMatrix<- cor(TrainSet[,-54])
corrplot(corrMatrix,order="FPC",method="color", type="lower",tl.cex=0.8,tl.col=rgb(0,0,0))
#now we cleaned th data and saw how variables are correlated between each other
# we will use prediction models now with three methods; 1- RANDOM FOREST, 2-DECISION TREE, 3-GENERALIZED BOOSTED MODEL.
# we will make a plot of the confusion matrix to visualize the accurancy
# method 1: RANDOM FOREST :
set.seed(32165)
controlRF<- trainControl(method = "cv", number = 3,verboseIter = FALSE)
modFitRandForest<- train(classe~ ., data=TrainSet,method="rf",trControl=controlRF )
modFitRandForest$finalModel
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
# method2: decision tree
set.seed(32165)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
                  
# method3 :Genereliized Boosted Model
set.seed(32165)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
