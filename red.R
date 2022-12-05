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

##Red Wine
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

train_control2 <- trainControl(method = "cv",
                                number = 10)
lmcv2 <- train(quality ~., data = train2, 
               method = "lm",
               trControl = train_control)
print(lmcv2)
pre.lmcv2 <- predict(lmcv2,test2) 
mean((pre.lmcv2-test2$quality)^2)

#glm
glm.fit2 <- glm(quality~., data=train2)
summary(glm.fit2)
pre.glm2 <- predict(glm.fit2, test2)
mean((pre.glm2-test2$quality)^2)

train_control2 <- trainControl(method = "cv",
                               number = 10)
glmcv2 <- train(quality ~., data = test2, 
                method = "glm",
                trControl = train_control)
print(glmcv2)
pre.glmcv2 <- predict(glmcv2,test2) 
mean((pre.glmcv2-test2$quality)^2)

## Correlation+Linear Regression

lm.fit.correlation2 <- lm(quality~volatile.acidity+citric.acid+total.sulfur.dioxide+density+sulphates+alcohol, data=train2)
pre.lm.correlation2 <- predict(lm.fit.correlation2, test2)
mean((pre.lm.correlation2-test2$quality)^2)

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

train_control <- trainControl(method = "cv",
                              number = 10)
knncv2 <- train(quality ~., data = train2, 
                method = "kknn",
                trControl = train_control)
print(knncv2)
pre.knncv2 <- predict(knncv2,test2) 
mean((pre.knncv2-test2$quality)^2)

##best subset+linear regresssion
regfit.full2 <- regsubsets(quality~., train2, nvmax=11) 
summary(regfit.full2)
reg.summary2=summary(regfit.full2)
par(mfrow=c(2,2))
plot(reg.summary2$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary2$adjr2)

bslim.fit2 <- lm(quality~fixed.acidity+volatile.acidity+residual.sugar+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=train2)
summary(bslim.fit2)
bspre.lm2 <- predict(bslim.fit2, test2)
mean((bspre.lm2-test2$quality)^2)

##Lasso
xtrain2=model.matrix(quality~., train2)[,-1]  
xtest2=model.matrix(quality~., test2)[,-1]  
ytrain2=train2$quality
set.seed(1)
cv.out2=cv.glmnet(xtrain2,ytrain2,alpha=1) 
bestlam2=cv.out2$lambda.min
bestlam2

lasso.mod2=glmnet(xtrain2,ytrain2,alpha=1,lambda=bestlam2)
lasso.coef2=coef(lasso.mod2)[,1]
lasso.coef2[lasso.coef2!=0]

pre.lasso2 = predict(lasso.mod2, s = bestlam2, newx = xtest2)
mean((pre.lasso2-test2$quality)^2)

train_control <- trainControl(method = "cv",
                              number = 10)
lscv2 <- train(quality ~., data = train2, 
              method = "lasso",
              trControl = train_control)

print(lscv2)
pre.lscv2 <- predict(lscv2,test2) 
mean((pre.lscv2-test2$quality)^2)

##PCR
set.seed(2)
pcr.fit2=pcr(quality~., data=train2,scale=TRUE,validation="CV")
summary(pcr.fit2) 
validationplot(pcr.fit2,val.type="MSEP")#refit the model with M=9

pcr.fit2=pcr(quality~., data=train2,scale=TRUE,ncomp=9) 
summary(pcr.fit2)
pcr.fit2
coef(pcr.fit2)
pre.pcr2 <- predict(pcr.fit2,test2)
mean((pre.pcr2-test2$quality)^2)

train_control <- trainControl(method = "cv",
                              number = 10)
pcrcv2 <- train(quality ~., data = train2, 
               method = "pcr",
               trControl = train_control)

print(pcrcv2)
pre.pcrcv2 <- predict(pcrcv2,test2) 
mean((pre.lscv2-test2$quality)^2)

## Regression Tree
set.seed(2)
tree.fit2=tree(quality~.,train2)
summary(tree.fit2)
plot(tree.fit2)
text(tree.fit2,pretty=0)
pre.tree2 <- predict(tree.fit2,test2)
plot(pre.tree2,test2$quality,xlim = c(4,8))
abline(0,1)
mean((pre.tree2-test2$quality)^2)

## Random Forest
rf.fit2<- randomForest(quality ~.,data=train2, importance=TRUE)
pre.rf2 <- predict(rf.fit2, test2)
plot(pre.rf2, test2$quality)
abline(0,1)
mean((pre.rf2-test2$quality)^2)
importance(rf.fit2)
varImpPlot(rf.fit2)

train_control2 <- trainControl(method = "cv",
                               number = 10)
rfcv2 <- train(quality ~., data = train2, 
               method = "rf",
               trControl = train_control)
print(rfcv2)
pre.rfcv2 <- predict(rfcv2,test2) 
mean((pre.rfcv2-test2$quality)^2)


## Boosting
set.seed(1)
boost.fit2=gbm(quality~.,data=train2,distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.fit2)
pre.boost2=predict(boost.fit2,newdata=test2,n.trees=5000)
mean((pre.boost2-test2$quality)^2)