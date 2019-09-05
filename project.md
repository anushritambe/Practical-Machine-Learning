## **'Project: Practical Machine Learning'**

Anushri

date: "05/09/2019"

==================================================================

## Introduction
   Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.In this project, main goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har.
   
   
## Summary
   In this project we need to predict the manner in which 6 participants did the exercise. This is the "classe" variable in the training set.  The algorithm described here is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz .
   
## 1.Loading required packages and data

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

  

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.6.1
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.6.1
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.6.1
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.6.1
```

```
## Loaded gbm 2.1.5
```

```r
train= read.csv("/Users/Anushree Tambe/Desktop/Coursera/ML/Project/Practical-Machine-Learning/pml-training.csv") 
test= read.csv("/Users/Anushree Tambe/Desktop/Coursera/ML/Project/Practical-Machine-Learning/pml-testing.csv")
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

## 2.Cleaning Data
In this we are removing coloumns which are having near zero variance as well as those columns which contains more than 95% zero or NA values as it wont affect much on final result.Also we are removing some unnecessary columns which we are not required for analysis.


```r
trainNZ<-nearZeroVar(train)
length(trainNZ)
```

```
## [1] 60
```

```r
train<-train[,-trainNZ]
dim(train)
```

```
## [1] 19622   100
```

```r
nacol <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[ , nacol == FALSE]
dim(train)
```

```
## [1] 19622    59
```

```r
overview<-str(train)
```

```
## 'data.frame':	19622 obs. of  59 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
##First five columns are not required
train <- train[, -(1:5)]
dim(train)
```

```
## [1] 19622    54
```

## 3.Partitioning Train data into training and testing set


```r
set.seed(123)
inTrain  <- createDataPartition(train$classe, p=0.8, list=FALSE)
training <- train[inTrain, ]
testing  <- train[-inTrain, ]
dim(training)
```

```
## [1] 15699    54
```

```r
dim(testing)
```

```
## [1] 3923   54
```

## 4.Prediction Models

### 4.1 Decision Tree Model


```r
set.seed(1234)
modtree <- train(classe ~ .,method="rpart", data = training)
predtree<-predict(modtree,testing)
fancyRpartPlot(modtree$finalModel)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png)

```r
confusionMatrix(predtree,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 972 190 113 102  24
##          B  18 258  21 119  54
##          C 125 311 550 395 213
##          D   0   0   0   0   0
##          E   1   0   0  27 430
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5633          
##                  95% CI : (0.5477, 0.5789)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4423          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8710  0.33992   0.8041   0.0000   0.5964
## Specificity            0.8472  0.93300   0.6777   1.0000   0.9913
## Pos Pred Value         0.6938  0.54894   0.3450      NaN   0.9389
## Neg Pred Value         0.9429  0.85491   0.9425   0.8361   0.9160
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2478  0.06577   0.1402   0.0000   0.1096
## Detection Prevalence   0.3571  0.11981   0.4063   0.0000   0.1167
## Balanced Accuracy      0.8591  0.63646   0.7409   0.5000   0.7938
```

The predictive accuracy of the decision tree model is 56.33 % which is very low.

### 4.2 Generalized Boosted Model (GBM)


```r
set.seed(1234)
traincontrolgbm <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modgbm<- train(classe ~ ., data=training, method = "gbm",trControl = traincontrolgbm, verbose = FALSE)
predgbm<-predict(modgbm,testing)
confusionMatrix(predgbm,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    5    0    0    0
##          B    0  745    6    2    1
##          C    0    8  676   10    3
##          D    0    1    1  629    9
##          E    0    0    1    2  708
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9875          
##                  95% CI : (0.9835, 0.9907)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9842          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9816   0.9883   0.9782   0.9820
## Specificity            0.9982   0.9972   0.9935   0.9966   0.9991
## Pos Pred Value         0.9955   0.9881   0.9699   0.9828   0.9958
## Neg Pred Value         1.0000   0.9956   0.9975   0.9957   0.9960
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1899   0.1723   0.1603   0.1805
## Detection Prevalence   0.2858   0.1922   0.1777   0.1631   0.1812
## Balanced Accuracy      0.9991   0.9894   0.9909   0.9874   0.9905
```

The predictive accuracy of the Generalised Boosted model is 98.75% which is relatively higher but still we will check the accuracy with random forest.

### 4.3 Random Forest Model


```r
set.seed(1234)
traincontrolrf<- trainControl(method="cv", number=3, verboseIter=FALSE)
modrf<-train(classe ~ .,method="rf", data = training, trControl=traincontrolrf )
predrf<-predict(modrf,testing)
confusionMatrix(predrf,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    4    0    0    0
##          B    0  755    0    0    0
##          C    0    0  684    2    0
##          D    0    0    0  641    4
##          E    0    0    0    0  717
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9975          
##                  95% CI : (0.9953, 0.9988)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9968          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   1.0000   0.9969   0.9945
## Specificity            0.9986   1.0000   0.9994   0.9988   1.0000
## Pos Pred Value         0.9964   1.0000   0.9971   0.9938   1.0000
## Neg Pred Value         1.0000   0.9987   1.0000   0.9994   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1925   0.1744   0.1634   0.1828
## Detection Prevalence   0.2855   0.1925   0.1749   0.1644   0.1828
## Balanced Accuracy      0.9993   0.9974   0.9997   0.9978   0.9972
```

The predictive accuracy of Roandom Forest is 99.75% which is higher than all earlier models.

## 5.Conclusion
   As the accuracy level of Random Forest is 99.75%,it is best model for the given data. Hence this  model is selected and applied to make predictions on the 20 data points from the original testing dataset.
   

```r
predict(modrf, test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

