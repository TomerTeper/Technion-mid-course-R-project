library(ggplot2)
library(reshape2)
library(readxl)
library(class)
data.redWine = read.csv("winequality-red.csv")
## plot var ggplot(data = D) + geom_point(mapping = aes(x = balance, y = income, colour = default:student))

## change qulity to 1 if qulity > 6.5 else 0 good/bed wine
data.redWine$quality[data.redWine$quality<6.5]=0
data.redWine$quality[data.redWine$quality>6.5]=1
sum(apply(data.redWine,2, function(x) is.na(x)))

sum(data.redWine$quality)
length(data.redWine$quality)
summary(data.redWine)

## looking data var
par(mfrow=c(2,2))
plot(data.redWine$quality)
plot(data.redWine$fixed.acidity)
plot(data.redWine$volatile.acidity)
plot(data.redWine$citric.acid)
plot(data.redWine$residual.sugar)
plot(data.redWine$chlorides)
plot(data.redWine$free.sulfur.dioxide)
plot(data.redWine$total.sulfur.dioxide)
plot(data.redWine$density)
plot(data.redWine$pH)
plot(data.redWine$sulphates)
plot(data.redWine$alcohol)



## heatmap cor for data
cormat <- round(cor(data.redWine),2)
head(cormat)
melted = melt(cormat)
head(melted)
ggplot(data = melted, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

## train and test
train.size = floor(0.7*nrow(data.redWine))
train.index = sample(1:nrow(data.redWine), train.size)
train.set = data.redWine[train.index,]
test.set = data.redWine[-train.index,]

## knn
data.knn = data.redWine # backup

data.knn$quality = factor(data.knn$quality) #factor of y

# normalizing numeric variables

num.vars = sapply(data.knn, is.numeric)
data[num.vars] = sapply(data[num.vars], scale)
summary(data)

#choosing training set and testing set


train.x = train.set
test.x = test.set

# removing the dependent variable from train and test sets

depvar = names(train.x)=="quality"
train.x = train.x[!depvar]
test.x = test.x[!depvar]

# creating vectors of dependent variable for both sets

train.y = data.knn$quality[train.index]
test.y = data.knn$quality[-train.index]
length(train.y)
# knn   

knn.1 =  knn(train = train.x, test = test.x, cl = train.y, k = 1)
knn.3 =  knn(train = train.x, test = test.x, cl = train.y, k = 3) 
knn.5 =  knn(train = train.x, test = test.x, cl = train.y, k = 5)

# Create Confusion matrix

table(knn.1 ,test.y)
table(knn.3 ,test.y)
table(knn.5 ,test.y)

#table

# proportion of correct classification for k = 1, 3, 5
100 * sum(test.y == knn.1)/length(test.y)  # For knn = 1
100 * sum(test.y == knn.3)/length(test.y)  # For knn = 3
100 * sum(test.y == knn.5)/length(test.y)  # For knn = 5


summary(knn.5)

## linar reg
reg1 = lm(quality ~ . , data = train.set)
summary(reg1)
coef(reg1)

predicted = predict(reg1, newdata = test.set)
test.mse = mean((test.set$quality - predicted)^2)
par(mfrow=c(1,1))
plot(test.set$quality ~ predicted)

par(mfrow=c(2,2))
plot(reg1)

## logistic reg
install.packages("InformationValue")
library(InformationValue)

install.packages("mccr")
library(mccr)

install.packages("ROCit")
library(ROCit)

library(ISLR)

data.logReg = data.redWine
levels(data.logReg$quality) = c(0,1)
data.logReg$quality = as.numeric(data.logReg$quality)

names(data.logReg)
dim(data.logReg)
summary(data.logReg)

pairs(data.logReg)
cor(data.logReg)
attach(data.logReg)


library(pastecs)

stat.desc(data.logReg,p = 0.975)

aggregate(data.logReg, list(data.logReg$quality),mean)

train.size = floor(nrow(data.logReg)*0.7)
train.index = sample(1:nrow(data.logReg),train.size, replace = F)
train.set = data.logReg[train.index,]
test.set = data.logReg[-train.index,]



glm.fits=glm(quality~.,data=train.set,family=binomial)
summary(glm.fits)


coef(glm.fits)
glm.probs=predict(glm.fits, test.set,type="response")

plotROC(test.set$quality, glm.probs)

confusionMatrix(test.set$quality, glm.probs, threshold = 0.5)

optCutOff = optimalCutoff(test.set$quality, glm.probs, optimiseFor = "Both")
confusionMatrix(test.set$quality, glm.probs, threshold = optCutOff)

sens = sensitivity(test.set$quality, glm.probs, threshold = 0.5)

specificity(test.set$quality, glm.probs, threshold = 0.5)

ppv = precision(test.set$quality, glm.probs, threshold = 0.5)

npv(test.set$quality, glm.probs, threshold = 0.5)

F1 = 2/((1/sens) + (1/ppv))

Prediction1 = glm.probs
Prediction1 = ifelse(Prediction1>=0.5, 1,  0) 

mccr(test.set$quality, Prediction1)##

sens = sensitivity(test.set$quality, glm.probs, threshold = optCutOff)

specificity(test.set$quality, glm.probs, threshold = optCutOff)

ppv = precision(test.set$quality, glm.probs, threshold = optCutOff)

npv(test.set$quality, glm.probs, threshold = optCutOff)

F1 = 2/((1/sens) + (1/ppv))

Prediction1 = glm.probs
Prediction1 = ifelse(Prediction1>=optCutOff, 1,  0) 

mccr(test.set$quality, Prediction1)##
## tree
library(randomForest)
library(tree)
library(MASS)

data.tree = data.redWine

train = sample(1:nrow(data.tree), nrow(data.tree)/2)

tree.redWine = tree(quality ~ .,data.tree, subset = train)
summary(tree.redWine)

## Plotting Regression Tree 
par(mfrow=c(1,1))
plot(tree.redWine)
text(tree.redWine,pos = 2)

## Cross - Validation Analysis dont work

cv.redWine = cv.tree(tree.redWine, K = 10)
plot(cv.redWine$size,cv.redWine$dev,type='b')

prune.redWine=prune.tree(tree.redWine,best=5)

plot(prune.redWine)
text(prune.redWine)

# Prediction #

yhat=predict(tree.redWine,newdata=data.tree[-train,])
redWine.test=data.tree[-train,"quality"]
plot(yhat,redWine.test)
abline(0,1)
mean((yhat-redWine.test)^2)


## random forest
data.rf = data.redWine
train.size = floor(nrow(data.rf)*0.7)
train.index = sample(1:nrow(data.rf),train.size, replace = F)
train.set = data.rf[train.index,]
test.set = data.rf[-train.index,]


rf = randomForest(formula = quality~., data = train.set, ntree=1500 )

# In order to find the optimal number of trees in random forest

opt_tree = which.min(rf$mse)
plot(rf$mse)

pred = predict(rf, test.set)



print(rf)
importance(rf)
varImpPlot(rf)


plot(pred, test.set$quality)
abline(0,1)
mean((pred-test.set$quality)^2)

## Cross - Validation Random Forest ##

rf_cv = rfcv(data.rf[train.index,!names(data.rf)=="quality"],data.rf$quality[train.index], step = 0.8, cv.fold = 10 )

mtry_min = rf_cv$n.var[rf_cv$error.cv==min(rf_cv$error.cv)]

rf2 = randomForest(formula = quality~., data = data.rf[train.index,], 
                  ntree=opt_tree, mtry = mtry_min )

pred = predict(rf2, data.rf[-train.index,])
table(data.rf[-train.index, "quality"],pred)

mean((pred-data.rf[-train.index,"quality"])^2)

## best subset

library(ISLR)
library(leaps) 

data.best = data.redWine

train.index = sample(1:nrow(data.best), floor(0.7*nrow(data.best)))
train = data.best[train.index,]
test = data.best[-train.index,]


fit.best = regsubsets(quality ~ . , train, nvmax = 19, method = "seqrep")
summary(fit.best) 


test.mse = rep(NA,11)
for (i in 1:11)
{
  coefficients.by.size = coef(fit.best, id = i)
  pred = test.matrix[,names(coefficients.by.size)] %*% coefficients.by.size
  test.mse[i] = mean((test$quality - pred)^2)
}

test.mse
num_of_coef = 1:11
plot(num_of_coef,test.mse)
which.min(test.mse)
coef(fit.best,which.min(test.mse))

##

fit.best.summary = summary(fit.best)
names(fit.best.summary)

fit.best.summary$rsq
fit.best.summary$adjr2



par(mfrow = c(1,1)) 

plot(fit.best.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "b")
points(which.min(fit.best.summary$cp),fit.best.summary$cp[which.min(fit.best.summary$cp)], col = "red", cex = 2, pch = 20)

plot(fit.best.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R2", type = "b")
points(which.max(fit.best.summary$adjr2),fit.best.summary$adjr2[which.max(fit.best.summary$adjr2)], col = "red", cex = 2, pch = 20)

plot(fit.best.summary$rsq, xlab = "Number of Variables", ylab = "R Squared", type = "b")
points(which.max(fit.best.summary$rsq),fit.best.summary$rsq[which.max(fit.best.summary$rsq)], col = "red", cex = 2, pch = 20)

plot(fit.best.summary$bic, xlab = "Number of Variables", ylab = "Bayesian Information Criteria", type = "b")
points(which.min(fit.best.summary$bic),fit.best.summary$bic[which.min(fit.best.summary$bic)], col = "red", cex = 2, pch = 20)


par(mfrow = c(1,1))

plot(full.fit ,scale ="r2")
plot(full.fit ,scale ="adjr2")
plot(full.fit ,scale ="bic")
plot(full.fit ,scale ="Cp")

beta_rsq = coef(full.fit, which.max(fit.best.summary$rsq))
beta_adjr2 = coef(full.fit, which.max(fit.best.summary$adjr2))
beta_bic = coef(full.fit,which.min(fit.best.summary$bic))
beta_Cp = coef(full.fit,which.min(fit.best.summary$cp))


## elastic net
library(ISLR)
library(Matrix)
library(glmnet)

data.elsNet = data.redWine
train.size = floor(nrow(data.elsNet)*0.7)
train.index = sample(1:nrow(data.elsNet),train.size)
train.set = data.elsNet[train.index,]
test.set = data.elsNet[-train.index,]

train.mat = model.matrix(quality~., data=train.set) 
test.mat = model.matrix(quality~., data=test.set)
grid = 10 ^ seq(4, -2, length=100)
alpha_vec = seq(0, 1, by = 0.01)
lambda.best.en = c()
mse.en = c()

for (i in 1:length(alpha_vec))
{
  mod.ElasticNet = cv.glmnet(x = train.mat, y = train.set[, "quality"], alpha= alpha_vec[i], lambda=grid)
  lambda.best = mod.ElasticNet$lambda.1se
  lambda.best.en[i] = lambda.best 
  
  ElasticNet.pred = predict(mod.ElasticNet, newx=test.mat, s=lambda.best)
  mse.en[i] = mean((test.set[, "quality"] - ElasticNet.pred)^2)
}

alpha_best = alpha_vec[which.min(mse.en)]
alpha_best

lambda.best1 = lambda.best.en [ which.min(mse.en)]

mod.ElasticNet = glmnet(x = train.mat, y = train.set[, "quality"], 
                        alpha= alpha_best, lambda=lambda.best1)


ElasticNet.pred = predict(mod.ElasticNet, newx=test.mat, s=lambda.best1)
mse.en = mean((test.set[, "quality"] - ElasticNet.pred)^2) 



