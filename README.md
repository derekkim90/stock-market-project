# stock-market-project
Capstone Project Ryerson-Stock Market Forecasting

Derek Kim 
Stock Market Forecasting using svm, knn, ordinal regression, random forest

---
title: "Predicting Stock Market"
author: "Derek Kim"
date: "February 6, 2019"
output:
  word_document: default
  html_document:
    df_print: paged
  pdf_document: default
---

```{r setup, include=FALSE}
{set.seed(333)}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(cache = TRUE)
```

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:


```{r}
library(bitops)
library(RCurl)
library(randomForest)
library(e1071)
library(caret)
library(corrplot)
library(quantmod)
library(TTR)
library(pageviews)
library(lubridate)
library(gtools)
library(mosaic)
library(FSelectorRcpp)
library(FSelector)
library (devtools)
devtools::install_github('software-analytics/Rnalytica')
library(Rnalytica)
library(foreign)
library(nnet)
library(ggplot2)
library(reshape2)
library(class)
library(gmodels)
library(MASS)
library(Hmisc)
library(SuperLearner)
```

Data Prep
```{r} 
#from personal pc data<- read.csv(file = "D:/Chrome Downloads/Capstone Project/datasets/MSFT.csv")

data <- read.csv(text=getURL("https://raw.githubusercontent.com/derekkim90/stock-market-project/master/MSFT.csv"))
msft<- data.frame(data)
sum(is.na(msft))

msft$Date<- as.Date(msft$Date)
msft$Volume<-as.numeric(msft$Volume)
str(msft)

#simple moving average is a trend indicator calculated as an average price over a particular period
sma<-SMA(msft$Adj.Close, n = 14)


#exponential moving average is a type of moving average where weights of past prices decrease exponentially:
ema<-EMA(msft$Adj.Close, n = 14, wilder = FALSE)


#true range, average true range, true high, true low - provides information about the degree of price volatility.
atr <- ATR(msft[,c("High","Low","Adj.Close")], n=14) 


# Average directional movement index indicates the strength of a trend in price time series. It is a combination of the negative and positive directional movements indicators, DI + n and DIn , computed over a period of n past days corresponding to the input window length
adx<- ADX(msft[,c("High","Low","Adj.Close")], n=14) 


#commodity channel index is an oscillator used to deter- mine whether a stock is overbought or oversold. It assesses the relationship between an asset price, its moving average and de- viations from that average
cci <- CCI(msft[,c("High","Low","Adj.Close")], n= 14) 


#rate of change shows the relative difference between the closing price on the day of forecast and the closing price n days previously, where n is equal to the input window length
roc<- ROC(msft[,c("Adj.Close")], n= 14) 


#relative strength index compares the size of recent gains to recent losses, it is intended to reveal the strength or weak- ness of a price trend from a range of closing prices over a time period
price <- msft[,"Adj.Close"] 
rsi <- RSI(price, n= 14)


#The William's %R oscillator shows the relationship between the current closing price and the high and low prices over the latest n days equal to the input window length
wpr <- WPR(msft[,c("High","Low","Adj.Close")], n = 14) 


#Stochastic %K is a technical momentum indicator that compares a close price and its price interval during a period of n past days and gives a signal meaning that a stock is oversold or over- bought:
sto<- stoch(msft[,c("High","Low","Adj.Close")], nFastK = 14, nFastD = 3, nSlowD = 3) 


mydata <- cbind(sma,ema, atr,adx,cci,roc,rsi,wpr,sto, msft)

#Select data for the past 10 years of stock price information
data <-mydata[mydata$Date >= "2009-02-12",]
```

```{r}
hist(data$sma, main = "Simple Moving Averages (SMA)")
hist(data$ema, main = "Exponential Moving Average")
hist(data$atr, main = "Average True Range of the Series")
hist(data$tr, main = "True Range of the Series")
hist(data$trueHigh, main = "True High of the Series")
hist(data$trueLow, main = "True Low")
hist(data$DIp, main = "Positive Direction Index")
hist(data$DIn, main = "Negative Direction Index")
hist(data$DX, main = "Direction Index")
hist(data$ADX, main = "Average Direction Index")
hist(data$cci, main = "Commodity Channel Index")
hist(data$roc, main = "Rate of Change")
hist(data$rsi, main = "Relative Strength Index")
hist(data$wpr, main = "William's % R")
hist(data$fastK, main = "Number of periods for fast %K")
hist(data$fastD, main = "Number of periods for fast %D")
hist(data$slowD, main = "Number of periods for slow %D")
hist(data$Open, main = "Stock Open Prices")
hist(data$High, main = "Stock High Prices")
hist(data$Low, main = "Stock Low Prices")
hist(data$Close, main = "Stock Close Prices")
hist(data$Adj.Close, main = "Stock Adj. Close Prices")
hist(data$Volume, main = "Stock Volume")



boxplot(data$sma, data$ema, names= c("SMA", "EMA"))
boxplot(data$atr, data$tr, names = c("ATR", "TR"))
boxplot(data$trueHigh, data$trueLow, names= c("True High", "True Low"))
boxplot(data$DIp, data$DIn, names= c("DIp", "DIn"))
boxplot(data$DX, data$ADX, names = c("DX", "ADX"))
boxplot(data$cci, main = ("CCI"))
boxplot(data$roc, main = ("ROC"))
boxplot(data$rsi, main = ("RSI"))
boxplot(data$wpr, data$fastK, data$fastD, data$slowD, names=c("WPR", "fastK", "fastD","slowD"))
boxplot(data$Open, data$High, data$Low, data$Close, data$Adj.Close, names=c("Open", "High", "Low", "Close", "Adj. Close"))
boxplot(data$Volume, main = ("Volume"))

```

```{r}
#adding a new variable to display the degree of stock movement relative to previous day and the difference between the two days

data_movement <-mydata[mydata$Date >= "2009-02-11",] #had to obtain data of one extra row prior to "2019-02-12" to match dimensions for next code
data$move<- (diff(data_movement$Adj.Close))
boxplot(data$move)

#Outliers based on 1.5*IQR
x <- data

outliers<- x$move[which(x$move %in% boxplot.stats(x$move)$out)]

Outliers <- c()
max<- quantile(x$move, 0.75)+(IQR(x$move)*1.5)
min<- quantile(x$move, 0.25)-(IQR(x$move)*1.5)
max
min
idx <- which(x$move < min | x$move > max)

# Append the outliers list
Outliers <- c(Outliers, idx) 
Outliers

# REMOVING OUTLIERS
y <- x[-which(x$move %in% outliers),]
boxplot(y$move)
hist(y$move)

a<-quantile (y$move, prob = 0)
b<-quantile (y$move, prob = 0.15)
c<-quantile (y$move, prob = 0.85)
d<-quantile (y$move, prob = 1)

y$Grade<-NULL
y$Grade [y$move>=a & y$move<b] <- "Very Low"
y$Grade [y$move>=b & y$move<0] <- "Down"
y$Grade [y$move==0] <- "No Move"
y$Grade [y$move>0 & y$move<=c] <- "Up"
y$Grade [y$move>c & y$move<=d] <- "Very High"


# CONVERT OUTLIERS TO MIN/MAX range
data.out<-x

data.out$move<-replace(data.out$move, data.out$move<min, min)
data.out$move<-replace(data.out$move, data.out$move>max, max)
boxplot(data.out$move)
hist((data.out$move))

e<-quantile (data.out$move, prob = 0)
f<-quantile (data.out$move, prob = 0.15)
g<-quantile (data.out$move, prob = 0.85)
h<-quantile (data.out$move, prob = 1)

data.out$Grade<-NULL
data.out$Grade [data.out$move>=e & data.out$move<f] <- "Very Low"
data.out$Grade [data.out$move>=f & data.out$move<0] <- "Down"
data.out$Grade [data.out$move==0] <- "No Move"
data.out$Grade [data.out$move>0 & data.out$move<=g] <- "Up"
data.out$Grade [data.out$move>g & data.out$move<=h] <- "Very High"

print(c(a,b,c,d,e,f,g,h))
```

```{r}
#correlation plot (numeric only)
nodate.y<-y

nodate.y$Grade<-NULL
nodate.y$Date<-as.numeric(nodate.y$Date)
m<-cor(nodate.y, method = "pearson")
corrplot(m, method = "number", type = "upper", number.digits = 1, number.cex = 0.7)

#correlation plot without date variable and outliers adjusted (numeric only)
nodate.data.out<-data.out

nodate.data.out$Grade<-NULL
nodate.data.out$Date<-as.numeric(nodate.data.out$Date)
n<-cor(nodate.data.out, method = "pearson")
corrplot(n, method = "number", type = "upper", number.digits = 1, number.cex = 0.7)
```

```{r}
#Feature selection method Gain Ratio vs AutoSpearman

#Gain ratio on 'Grade' with outliers removed
removed<-y
removed$Date<-as.factor(removed$Date)
removed$move<-NULL
weight<- gain.ratio(Grade~., removed)
subset <- cutoff.k(weight, 6)
subgr <- as.simple.formula(subset, "Grade")
subgr
#Gain ratio on 'Grade' with adjusted outlier values
adjust<-data.out
adjust$Date<-as.factor(adjust$Date)
adjust$move<-NULL
weight2<- gain.ratio(Grade~., adjust)
subset <- cutoff.k(weight2, 6)
subgr1 <- as.simple.formula(subset, "Grade")
subgr1

#Feature selection using AutoSpearman with Outliers removed
indep<-removed
indep$Grade<-NULL
indep$Date<-as.numeric(as.factor(indep$Date))

removed$Date<-as.numeric((as.factor(removed$Date)))
auto<-AutoSpearman(dataset = removed, metrics=colnames(indep))
auto


#Feature selection using AutoSpearman with Outliers adjusted
indep1<-adjust
indep1$Grade<-NULL
indep1$Date<-as.numeric(as.factor(indep1$Date))
adjust$Date<-as.numeric(as.factor(adjust$Date))
auto1<-AutoSpearman(dataset = adjust, metrics=colnames(indep1))
auto1
```

```{r}
#                     Random Forest with Outliers Removed

trainindex<-createDataPartition(removed$Grade, p=0.85, list= FALSE)
training<-removed[trainindex,]
test<-removed[-trainindex,]
dim(training)
dim(test)

traindata<- trainControl(method = "cv", number = 10) #10 fold cross validation

modeldata<- train(Grade~., data = training, trControl = traindata, method = "rf") #23 predictors having better accuracy and kappa
modeldatagr<- train(Grade~cci+sma+ema+DIp+Close+Adj.Close, data = training, trControl = traindata, method = "rf")
modelauto<- train(Grade~DIn+DX+ADX+cci+fastK+Volume, data = training, trControl = traindata, method = "rf")

print(modeldata)
print(modeldatagr)
print(modelauto)

predictdata<-predict(modeldata, test)
predictdatagr<- predict(modeldatagr, test)
predictauto<- predict(modelauto, test)

b1<-confusionMatrix(predictdata,as.factor(test$Grade))
b2<-confusionMatrix(predictdatagr,as.factor(test$Grade))
b3<-confusionMatrix(predictauto, as.factor(test$Grade))

as.table(b1)
as.matrix(b1, what="overall")
as.matrix(b1, what= "classes")

as.table(b2)
as.matrix(b2, what="overall")
as.matrix(b2, what= "classes")

as.table(b3)
as.matrix(b3, what="overall")
as.matrix(b3, what= "classes")
```
Grade ~ cci + sma + ema + DIp + Close + Adj.Close
<environment: 0x0000000014971710>
Grade ~ sma + Date + ema + trueHigh + High + trueLow
<environment: 0x0000000004353060>
[1] "DIn"    "DX"     "ADX"    "cci"    "fastK"  "Volume"
[1] "tr"     "DX"     "ADX"    "cci"    "fastK"  "Volume"
```{r}
#                     Random Forest with Outliers Adjusted

trainindex2<-createDataPartition(adjust$Grade, p=0.85, list= FALSE)

training2<-adjust[trainindex2,]
test2<-adjust[-trainindex2,]
dim(training2)
dim(test2)

traindata2<- trainControl(method = "cv", number = 10) #10 fold cross validation

modeldata2<- train(Grade~., data = training2, trControl = traindata2, method = "rf")
modeldata2gr<- train(Grade~sma+Date+ema+trueHigh+High+trueLow, data = training2, trControl = traindata2, method = "rf")
modelauto2<- train(Grade~tr+DX+ADX+cci+fastK+Volume, data = training2, trControl = traindata2, method = "rf")

print(modeldata2) # having all predictors has better accuracy
print(modeldata2gr)
print(modelauto2)

predictdata2<-predict(modeldata2, test2)
predictdata2gr<-predict(modeldata2gr,test2)
predictauto2<-predict(modelauto2, test2)

a1<-confusionMatrix(predictdata2,as.factor(test2$Grade))
a2<-confusionMatrix(predictdata2gr,as.factor(test2$Grade))
a3<-confusionMatrix(predictauto2, as.factor(test2$Grade))

as.table(a1)
as.matrix(a1, what="overall")
as.matrix(a1, what= "classes")

as.table(a2)
as.matrix(a2, what="overall")
as.matrix(a2, what= "classes")

as.table(a3)
as.matrix(a3, what="overall")
as.matrix(a3, what= "classes")


```


```{r}
#SVM with Outliers removed

Trainsvm <- training
Testsvm <- test
str(Trainsvm)
Trainsvm$Grade<- as.factor(Trainsvm$Grade)

# all features vs gain ratio vs  AutoSpearman
modelsvm <- svm(Grade~., data=Trainsvm, kernel='polynomial')
modelsvmgr<- svm(Grade~cci+sma+ema+DIp+Close, data=Trainsvm, kernel='polynomial')
modelautofeat<-svm(Grade~DIn+DX+ADX+cci+fastK+Volume, data=Trainsvm, kernel='polynomial')
#Predict Output
predssvm <- predict(modelsvm,Testsvm)
predssvmgr<-predict(modelsvmgr, Testsvm)
predsauto <- predict(modelautofeat, Testsvm)

c1<-confusionMatrix(predssvm, as.factor(Testsvm$Grade))
c2<-confusionMatrix(predssvmgr, as.factor(Testsvm$Grade))
c3<-confusionMatrix(predsauto, as.factor(Testsvm$Grade))


as.table(c1)
as.matrix(c1, what="overall")
as.matrix(c1, what= "classes")

as.table(c2)
as.matrix(c2, what="overall")
as.matrix(c2, what= "classes")

as.table(c3)
as.matrix(c3, what="overall")
as.matrix(c3, what= "classes")


```

```{r}
#SVM with Outliers adjusted

Trainsvm2<- training2
Testsvm2<- test2
Trainsvm2$Grade<-as.factor(Trainsvm2$Grade)

# create model
model2svm <- svm(Grade~., data=Trainsvm2, kernel='polynomial')#all features
model2svmgr<-svm(Grade~sma+ema+trueHigh+High+trueLow, data=Trainsvm2, kernel='polynomial') #features from gain ratio
model2svmauto <-svm(Grade~tr+DX+ADX+cci+Volume, data=Trainsvm2, kernel='polynomial') #feature selection based on AutoSpearman
#Predict Output
preds2svm <- predict(model2svm,Testsvm2)
preds2svmgr<-predict(model2svmgr, Testsvm2)
predsauto2svm<- predict(model2svmauto, Testsvm2)

d1<-confusionMatrix(preds2svm, as.factor(Testsvm2$Grade))
d2<-confusionMatrix(preds2svmgr, as.factor(Testsvm2$Grade))
d3<-confusionMatrix(predsauto2svm, as.factor(Testsvm2$Grade))

as.table(d1)
as.matrix(d1, what="overall")
as.matrix(d1, what= "classes")

as.table(d2)
as.matrix(d2, what="overall")
as.matrix(d2, what= "classes")

as.table(d3)
as.matrix(d3, what="overall")
as.matrix(d3, what= "classes")

```

```{r}
#Ordinal Regression without Outliers
regdata<-training
regtestdata<-test
regdata$Grade<-as.factor(regdata$Grade)
regtestdata$Grade<- as.factor(regtestdata$Grade)

reg<-polr(Grade~sma+ema+tr+atr+trueHigh+DIp+DIn+DX+ADX+cci+roc+rsi+wpr+fastD+slowD+Date+Open+High+Low+Close+Adj.Close+Volume, data=regdata, Hess= TRUE)
predall<-predict(reg, regtestdata)
o1<-confusionMatrix(predall, regtestdata$Grade)
as.table(o1)
as.matrix(o1, what="overall")
as.matrix(o1, what= "classes")

tgr<-polr(Grade~Date+cci+sma+ema+DIp+Close, data=regdata, Hess= TRUE)
predtgr<-predict(tgr, test)
o2<-confusionMatrix(predtgr, regtestdata$Grade)
as.table(o2)
as.matrix(o2, what="overall")
as.matrix(o2, what= "classes")

tauto<-polr(Grade~DIn+DX+ADX+cci+fastK+Volume, data=regdata, Hess= TRUE)
predtauto<-predict(tauto, regtestdata)
o3<-confusionMatrix(predtauto, regtestdata$Grade)
as.table(o3)
as.matrix(o3, what="overall")
as.matrix(o3, what= "classes")

#Ordinal Regression with Outliers adjusted
regdata2<-training2
regtestdata2<-test2
regdata2$Grade<-as.factor(regdata2$Grade)
regtestdata2$Grade<-as.factor(regtestdata2$Grade)

reg1<-polr(Grade~sma+ema+tr+atr+trueHigh+DIp+DIn+DX+ADX+cci+roc+rsi+wpr+fastD+slowD+Date+Open+High+Low+Close+Adj.Close+Volume, data=regdata2, Hess= TRUE)
predall1<-predict(reg1, regtestdata2)
o4<-confusionMatrix(predall1, regtestdata2$Grade)
as.table(o4)
as.matrix(o4, what="overall")
as.matrix(o4, what= "classes")

ugr<- polr(Grade~Date+sma+ema+trueHigh+High+trueLow, data=regdata2, Hess= TRUE)
predugr<-predict(ugr, regtestdata2)
o5<-confusionMatrix(predugr, regtestdata2$Grade)
as.table(o5)
as.matrix(o5, what="overall")
as.matrix(o5, what= "classes")

uauto1<-polr(Grade~tr+DX+ADX+cci+fastK+Volume, data=regdata2, Hess= TRUE)
preduauto1<-predict(uauto1, regtestdata2)
o6<-confusionMatrix(preduauto1, regtestdata2$Grade)
as.table(o6)
as.matrix(o6, what="overall")
as.matrix(o6, what= "classes")


```

```{r}
#knn without outliers
set.seed(100)
normalize <- function(x) {
return ((x - min(x)) / (max(x) - min(x))) }

prc_n <- as.data.frame(lapply(removed[1:length(removed)-1], normalize))
nrow(removed)
print(2289*0.8)

prc_train <- prc_n[1:1831,]
prc_test <- prc_n[1832:2289,]

prc_train_labels <- removed[1:1831, 25]
prc_test_labels <- removed[1832:2289, 25]

#determine optimal k neighbours for each condition
knumber<-train(Grade ~ ., data = training, method = "knn", trControl = traindata, preProcess = c("center","scale"),tuneLength = 20)
knumber#k=27
plot(knumber)

knumbergr<-train(Grade ~ Date + cci + sma + ema + DIp + Close, data = training, method = "knn", trControl = traindata, preProcess = c("center","scale"),tuneLength = 20)
knumbergr #k=43
plot(knumbergr)

knumberauto<- train(Grade ~Date + sma + ema + trueHigh + High + trueLow, data = training, method = "knn", trControl = traindata, preProcess = c("center","scale"),tuneLength = 20) 
knumberauto #k=19
plot(knumber)

prc_test_pred1 <- knn(train = prc_train, test = prc_test, cl = prc_train_labels, k=25)

k1<-confusionMatrix(as.factor(prc_test_pred1), as.factor(prc_test_labels))
as.table(k1)
as.matrix(k1, what="overall")
as.matrix(k1, what= "classes")
#knn without outliers with features selected via gain ratio
prc_traingr<- prc_n[1:1831,c(1,2,7,11,18,22)]
prc_testgr<- prc_n[1832:2289,c(1,2,7,11,18,22)]

prc_test_predgr1 <- knn(train = prc_traingr, test = prc_testgr, cl = prc_train_labels, k=39)

k2<-confusionMatrix(as.factor(prc_test_predgr1), as.factor(prc_test_labels))
as.table(k2)
as.matrix(k2, what="overall")
as.matrix(k2, what= "classes")

#knn without outliers with features via AutoSpearman
prc_trainas<- prc_n[1:1831,c(8:11,15,24)]
prc_testas<- prc_n[1832:2289,c(8:11,15,24)]

prc_test_predas1 <- knn(train = prc_trainas, test = prc_testas, cl = prc_train_labels, k=21)

k3<-confusionMatrix(as.factor(prc_test_predas1), as.factor(prc_test_labels))
as.table(k3)
as.matrix(k3, what="overall")
as.matrix(k3, what= "classes")

```

```{r}
#knn with outliers adjusted
set.seed(101)
prc_a <- as.data.frame(lapply(adjust[1:length(adjust)-1], normalize))
nrow(adjust)
print(2517*0.8)

prca_train <- prc_a[1:2014,]
prca_test <- prc_a[2015:2517,]

prca_train_labels <- adjust[1:2014, 25]
prca_test_labels <- adjust[2015:2517, 25]

knumber1<-train(Grade ~ ., data = training2, method = "knn", trControl = traindata2, preProcess = c("center","scale"),tuneLength = 20)
knumber1#k=25
plot(knumber1)

knumbergr1<-train(Grade ~ Date + cci + sma + ema + DIp + Close, data = training2, method = "knn", trControl = traindata2, preProcess = c("center","scale"),tuneLength = 20)
knumbergr1 #k=27
plot(knumbergr1)

knumberauto1<- train(Grade ~Date + sma + ema + trueHigh + High + trueLow, data = training2, method = "knn", trControl = traindata2, preProcess = c("center","scale"),tuneLength = 20) 
knumberauto1 #k=23
plot(knumberauto1)

prca_test_pred2 <- knn(train = prca_train, test = prca_test, cl = prca_train_labels, k=25)

k4<-confusionMatrix(as.factor(prca_test_pred2), as.factor(prca_test_labels))
as.table(k4)
as.matrix(k4, what="overall")
as.matrix(k4, what= "classes")

#knn with outliers with features selected via gain ratio
prca_traingr2<- prc_a[1:2014,c(1,2,5,6,18,20)]
prca_testgr2<- prc_a[2015:2517,c(1,2,5,6,18,20)]

prca_test_predgr2 <- knn(train = prca_traingr2, test = prca_testgr2, cl = prca_train_labels, k=27)

k5<-confusionMatrix(as.factor(prca_test_predgr2), as.factor(prca_test_labels))
as.table(k5)
as.matrix(k5, what="overall")
as.matrix(k5, what= "classes")

#knn with outliers with features via AutoSpearman
prca_train2as<- prc_a[1:2014,c(3,9,10,11,15,24)]
prca_test2as<- prc_a[2015:2517,c(3,9,10,11,15,24)]

prca_test_predas2 <- knn(train = prca_train2as, test = prca_test2as, cl = prca_train_labels, k=23)


k6<-confusionMatrix(as.factor(prca_test_predas2), as.factor(prca_test_labels))
as.table(k6)
as.matrix(k6, what="overall")
as.matrix(k6, what= "classes")
```




