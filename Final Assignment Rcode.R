# 5740 Final Assignment Qianqian Fang, Zhengyu Lu

## Importing libraries
install.packages("randomForest")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("caret")
install.packages("varImp")
install.packages("pls")
install.packages("klaR")
install.packages("elasticnet")

library(MASS)
library(randomForest)
library(ggplot2)
library(corrplot)
library(caret)
library(gbm)
library(e1071)
library(tree)
library(varImp)
library(glmnet)
library(leaps)
library(pls)
library(kknn)
library(elasticnet)

#White Wine
whitequality <- read.csv("winequality-white.csv", sep = ";")
dim(whitequality)
head(whitequality)
sum(is.na(whitequality))
summary(whitequality)

##Quality frequency
ggplot(whitequality,aes(quality)) + geom_histogram(stat="count") +
  xlab("Quality of white wines") + ylab("Number of white wines")

##correlation
M <- cor(whitequality)
corrplot(M, method = "number")

##scatter plot
pairs(whitequality, panel=panel.smooth)

##box plot
ggplot(whitequality, aes(as.factor(quality),fixed.acidity))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),volatile.acidity))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),citric.acid))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),residual.sugar))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),chlorides))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),free.sulfur.dioxide))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),total.sulfur.dioxide))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),density))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),pH))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),sulphates))+ geom_boxplot()
ggplot(whitequality, aes(as.factor(quality),alcohol))+ geom_boxplot()

##Create training and testing dataset)
set.seed(1)
index <- sort(sample(nrow(whitequality),nrow(whitequality)*0.85))
train <- whitequality[index,]
test <- whitequality[-index,]

##linear regression
lm.fit <- lm(quality~., data = train)
summary(lm.fit)
pre.lm <- predict(lm.fit, test)
mean((pre.lm-test$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
lmcv <- train(quality ~., data = whitequality, 
               method = "lm",
               trControl = train_control)
print(lmcv)

## Correlation+Linear Regression
lm.fit.correlation <- lm(quality~fixed.acidity+volatile.acidity+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+alcohol, data=train)
pre.lm.correlation <- predict(lm.fit.correlation, test)
mean((pre.lm.correlation-test$quality)^2)

## Best subset+linear regresssion
regfit.full <- regsubsets(quality~., train, nvmax=11) 
summary(regfit.full)
reg.summary=summary(regfit.full)
par(mfrow=c(2,2))
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
which.min(reg.summary$bic)

###adjr2 model
bslm.fit <- lm(quality~fixed.acidity+volatile.acidity+residual.sugar+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=train)
summary(bslm.fit)
bspre.lm <- predict(bslm.fit, test)
mean((bspre.lm-test$quality)^2)

###bic model
bslmbic.fit <- lm(quality~fixed.acidity+volatile.acidity+residual.sugar+free.sulfur.dioxide+density+pH+sulphates+alcohol, data=train)
summary(bslmbic.fit)
bsprebic.lm <- predict(bslmbic.fit, test)
mean((bsprebic.lm-test$quality)^2)

#glm
glm.fit <- glm(quality~., data=train)
summary(glm.fit)
pre.glm <- predict(glm.fit, test)
mean((pre.glm-test$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
glmcv <- train(quality ~., data = whitequality, 
               method = "glm",
               trControl = train_control)
print(glmcv)

#LDA
lda.fit <- lda(quality~., train)
lda.predict <- predict(lda.fit, test)
table(lda.predict$class, test$quality)
mean(lda.predict$class !=test$quality)
summary(lda.fit)

##Naive Bayes
nb.fit <- naiveBayes(quality~., data = train)
nb.predict <- predict(nb.fit, test)
table(nb.predict,test$quality)
mean(nb.predict!=test$quality)

#knn
model_knn <- train(
  quality ~.,
  data = train,
  method = 'knn'
)
model_knn
plot(model_knn)
pre.knn <- predict(model_knn, test)
mean((pre.knn-test$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
knncv <- train(quality ~., data = whitequality, 
               method = "kknn",
               trControl = train_control)
print(knncv)

##Lasso
x=model.matrix(quality~., whitequality)[,-1]  
y=whitequality$quality
set.seed(1)
cv.out=cv.glmnet(x,y,alpha=1) 
bestlam=cv.out$lambda.min
bestlam
lasso.mod=glmnet(x,y,alpha=1,lambda=bestlam)
lasso.coef=coef(lasso.mod)[,1]
lasso.coef[lasso.coef!=0]
pre.lasso = predict(lasso.mod, s = bestlam, newx = x)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
lscv <- train(quality ~., data = whitequality, 
               method = "lasso",
               trControl = train_control)
print(lscv)

##PCR
set.seed(2)
pcr.fit=pcr(quality~., data=train,scale=TRUE,validation="CV")
summary(pcr.fit) 
validationplot(pcr.fit,val.type="MSEP")#refit the model with M=9

pcr.fit=pcr(quality~., data=train,scale=TRUE,ncomp=9) 
summary(pcr.fit)
coef(pcr.fit)
pre.pcr <- predict(pcr.fit,test)
mean((pre.pcr-test$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
pcrcv <- train(quality ~., data = whitequality, 
               method = "pcr",
               trControl = train_control)
print(pcrcv)

## Regression Tree
set.seed(1)
tree.fit=tree(quality~.,train)
summary(tree.fit)
plot(tree.fit)
text(tree.fit,pretty=0)
pre.tree <- predict(tree.fit,test)
plot(pre.tree,test$quality,xlim = c(4,8))
abline(0,1)
mean((pre.tree-test$quality)^2)

## Random Forest
set.seed(1)
rf.fit<- randomForest(quality ~.,data=train, importance=TRUE)
pre.rf <- predict(rf.fit, test)
plot(pre.rf, test$quality)
abline(0,1)
mean((pre.rf-test$quality)^2)
importance(rf.fit)
varImpPlot(rf.fit)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
rfcv <- train(quality ~., data = whitequality, 
              method = "rf",
              trControl = train_control)
print(rfcv)

## Boosting
set.seed(1)
boost.fit=gbm(quality~.,data=train,distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.fit)
pre.boost=predict(boost.fit,newdata=test,n.trees=5000)
mean((pre.boost-test$quality)^2)

#Red Wine
redquality <- read.csv("winequality-red.csv", sep = ";")
dim(redquality)
head(redquality)
sum(is.na(redquality))
summary(redquality)

##Quality frequency
ggplot(redquality,aes(quality)) + geom_histogram(stat="count") +
  xlab("Quality of red wines") + ylab("Number of red wines")

##correlation
N <- cor(redquality)
corrplot(N, method = "number")

##scatter plot
pairs(redquality, panel=panel.smooth)

##box plot
ggplot(redquality, aes(as.factor(quality),fixed.acidity))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),volatile.acidity))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),citric.acid))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),residual.sugar))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),chlorides))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),free.sulfur.dioxide))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),total.sulfur.dioxide))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),density))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),pH))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),sulphates))+ geom_boxplot()
ggplot(redquality, aes(as.factor(quality),alcohol))+ geom_boxplot()

##Create training and testing dataset)
set.seed(1)
index <- sort(sample(nrow(redquality),nrow(redquality)*0.85))
train2 <- redquality[index,]
test2 <- redquality[-index,]

##linear regression
lm.fit2 <- lm(quality~., data = train2)
summary(lm.fit2)
pre.lm2 <- predict(lm.fit2, test2)
mean((pre.lm2-test2$quality)^2)

set.seed(1)
train_control2 <- trainControl(method = "cv",
                               number = 10)
lmcv2 <- train(quality ~., data = redquality, 
               method = "lm",
               trControl = train_control)
print(lmcv2)

## Correlation+Linear Regression
lm.fit.correlation2 <- lm(quality~volatile.acidity+citric.acid+total.sulfur.dioxide+density+sulphates+alcohol, data=train2)
pre.lm.correlation2 <- predict(lm.fit.correlation2, test2)
mean((pre.lm.correlation2-test2$quality)^2)

##Best subset+linear regresssion
regfit.full2 <- regsubsets(quality~., train2, nvmax=11) 
summary(regfit.full2)
reg.summary2=summary(regfit.full2)
par(mfrow=c(2,2))
plot(reg.summary2$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary2$adjr2)
which.min(reg.summary2$bic)

###adjr2 model
bslm.fit2 <- lm(quality~volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+pH+sulphates+alcohol, data=train2)
summary(bslm.fit2)
bspre.lm2 <- predict(bslm.fit2, test2)
mean((bspre.lm2-test2$quality)^2)

###BIC model
bslmbic.fit2 <- lm(quality~volatile.acidity+chlorides+total.sulfur.dioxide+pH+sulphates+alcohol, data=train2)
summary(bslmbic.fit2)
bsprebic.lm2 <- predict(bslmbic.fit2, test2)
mean((bsprebic.lm2-test2$quality)^2)

#glm
glm.fit2 <- glm(quality~., data=train2)
summary(glm.fit2)
pre.glm2 <- predict(glm.fit2, test2)
mean((pre.glm2-test2$quality)^2)

set.seed(1)
train_control2 <- trainControl(method = "cv",
                               number = 10)
glmcv2 <- train(quality ~., data = redquality, 
                method = "glm",
                trControl = train_control)
print(glmcv2)

#LDA
lda.fit2 <- lda(quality~., train2)
lda.predict2 <- predict(lda.fit2, test2)
table(lda.predict2$class, test2$quality)
mean(lda.predict2$class !=test2$quality)
summary(lda.fit2)

##Naive Bayes
nb.fit2 <- naiveBayes(quality~., data = train2)
nb.predict2 <- predict(nb.fit2, test2)
table(nb.predict2,test2$quality)
mean(nb.predict2!=test2$quality)

#knn
model_knn2 <- train(
  quality ~.,
  data = train2,
  method = 'knn'
)
model_knn2
plot(model_knn2)
pre.knn2 <- predict(model_knn2, test2)
mean((pre.knn2-test2$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
knncv2 <- train(quality ~., data = redquality, 
                method = "kknn",
                trControl = train_control)
print(knncv2)

##Lasso
x2=model.matrix(quality~., redquality)[,-1]  
y2=redquality$quality
set.seed(1)
cv.out2=cv.glmnet(x2,y2,alpha=1) 
bestlam2=cv.out2$lambda.min
bestlam2
lasso.mod2=glmnet(x2,y2,alpha=1,lambda=bestlam2)
lasso.coef2=coef(lasso.mod2)[,1]
lasso.coef2[lasso.coef2!=0]
pre.lasso2 = predict(lasso.mod2, s = bestlam2, newx = x2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
lscv2 <- train(quality ~., data = redquality, 
               method = "lasso",
               trControl = train_control)
print(lscv2)

##PCR
set.seed(2)
pcr.fit2=pcr(quality~., data=train2,scale=TRUE,validation="CV")
summary(pcr.fit2) 
validationplot(pcr.fit2,val.type="MSEP")#refit the model with M=6

pcr.fit2=pcr(quality~., data=train2,scale=TRUE,ncomp=6) 
summary(pcr.fit2)
coef(pcr.fit2)
pre.pcr2 <- predict(pcr.fit2,test2)
mean((pre.pcr2-test2$quality)^2)

set.seed(1)
train_control <- trainControl(method = "cv",
                              number = 10)
pcrcv2 <- train(quality ~., data = redquality, 
                method = "pcr",
                trControl = train_control)
print(pcrcv2)

## Regression Tree
set.seed(1)
tree.fit2=tree(quality~.,train2)
summary(tree.fit2)
plot(tree.fit2)
text(tree.fit2,pretty=0)
pre.tree2 <- predict(tree.fit2,test2)
plot(pre.tree2,test2$quality,xlim = c(4,8))
abline(0,1)
mean((pre.tree2-test2$quality)^2)

## Random Forest
set.seed(1)
rf.fit2<- randomForest(quality ~.,data=train2, importance=TRUE)
pre.rf2 <- predict(rf.fit2, test2)
plot(pre.rf2, test2$quality)
abline(0,1)
mean((pre.rf2-test2$quality)^2)
importance(rf.fit2)
varImpPlot(rf.fit2)

set.seed(1)
train_control2 <- trainControl(method = "cv",
                               number = 10)
rfcv2 <- train(quality ~., data = redquality, 
               method = "rf",
               trControl = train_control)
print(rfcv2)

## Boosting
set.seed(1)
boost.fit2=gbm(quality~.,data=train2,distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.fit2)
pre.boost2=predict(boost.fit2,newdata=test2,n.trees=5000)
mean((pre.boost2-test2$quality)^2)
