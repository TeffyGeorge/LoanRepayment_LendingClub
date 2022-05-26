###########################################################################
###########################################################################
###     TITLE:  LOAN REPAYMENT                                          ###
###     AUTHOR: JOYAL JOBY CHULLY                                       ###
###             SAILESH DOGIPARTHI                                      ###
###             SHUBHAM                                                 ###
###             SRIRAM KAUSHIK KANURI                                   ###
###             TEFFY ANNIE GEORGE                                      ###
###     DATE :  APRIL 19, 2022                                          ###
###                                                                     ###
###########################################################################
###########################################################################


##########################  LOAN REPAYMENT   ##############################
set.seed(42)
library(caret)
library(pROC)

setwd("C:\\Spring 2022\\BUAN 6356\\Project")
loan_backup = read.csv("loan_data.csv")
loan = loan_backup
View(loan)


str(loan)
View(loan)

##########################  DATA CLEANING   ##############################

# No Missing Values
# Correlation Validated
install.packages("corrplot")
library(corrplot)
corrplot.mixed(cor(loan[,-c(1,2,14)]),order = 'AOE')

# Imbalanced Data
# Accuracy Not Enough

# Categorical Feature Transformation
# Converting purpose to factor
label=c("debt_consolidation","credit_card","all_other","home_improvement","small_business","major_purchase","educational")
loan$purpose <- as.numeric(factor(loan$purpose))

######################   DATA STANDARDIZATION   ###########################
loan[,c(3,4,5,6,7,8,9,10,11,12,13)] = sapply(loan_backup[,c(3,4,5,6,7,8,9,10,11,12,13)],scale)

# df loan is now scaled/normalized/standardized

# Saving the normalized one in loan_std
loan_std=loan
loan=loan_std
str(loan)
summary(loan)


############################  PCA   ######################################
#removing the numerical variables
loan_num=loan[,-c(1,2,14)]
str(loan_num)
pcs=prcomp(na.omit(loan_num),scale=F)
summary(pcs)
###PCA result indicates the usage of all numeric variables for modelling

############################  DATA SPLIT   ###############################
##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

############################  MODELLING   ################################

############################  DECISION TREE   ############################
library(rpart)
library(rpart.plot)
ct = rpart(not.fully.paid~.,data=train.set,control=rpart.control(maxdepth = 4,minsplit=10, minbucket = 3, cp=0.002),method="class")
prp(ct)

ct = rpart(not.fully.paid~.,data=train.set,control=rpart.control(maxdepth = 5,minbucket = 3, cp =0.002),method="class")
prp(ct)

ct.pred.train = predict(ct,train.set,type="class")
library(caret)
confusionMatrix(ct.pred.train, as.factor(train.set$not.fully.paid))
training.predict <- predict(ct,train.set,type = "prob")
library(pROC)
r.decisiontree.train <- roc(train.set$not.fully.paid,training.predict[,2])
plot.roc(r.decisiontree.train)
auc(r.decisiontree.train)

##For validation set
ct.pred.valid = predict(ct,valid.set,type="class")
confusionMatrix(ct.pred.valid, as.factor(valid.set$not.fully.paid))
valid.predict <- predict(ct,valid.set,type = "prob")
r.decisiontree.valid <- roc(valid.set$not.fully.paid,valid.predict[,2])
plot.roc(r.decisiontree.valid)
auc(r.decisiontree.valid)

#######################   BASIC LOGISTIC MODEL   #########################

##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

str(train.set)
#######Modelling
logit.reg <- glm(not.fully.paid ~ ., data = train.set, family = "binomial")
summary(logit.reg)
logit.reg.train=predict(logit.reg,train.set,type="response")
##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.train > 0.5, 1, 0)), as.factor(train.set$not.fully.paid))
##ROC curve
r_train <- roc(train.set$not.fully.paid,logit.reg.train,auc=T)
plot.roc(r_train)
auc(r_train)

#######Validation
logit.reg.valid = predict(logit.reg,valid.set,type="response")

##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.valid > 0.5, 1, 0)), as.factor(valid.set$not.fully.paid))
##ROC curve
r_valid <- roc(valid.set$not.fully.paid,logit.reg.valid,auc=T)
plot.roc(r_valid)
auc(r_valid)

#####################    SYSTEMATIC LOGISTIC MODEL   #########################

##Remove int.rate from standardized df
loan=loan_std[,-3]

##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

#######Modelling
logit.reg <- glm(not.fully.paid ~ ., data = train.set, family = "binomial")
summary(logit.reg)
logit.reg.train=predict(logit.reg,train.set,type="response")
##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.train > 0.5, 1, 0)), as.factor(train.set$not.fully.paid))
##ROC curve
r_train <- roc(train.set$not.fully.paid,logit.reg.train,auc=T)
plot.roc(r_train)
auc(r_train)

#######Validation
logit.reg.valid = predict(logit.reg,valid.set,type="response")

##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.valid > 0.5, 1, 0)), as.factor(valid.set$not.fully.paid))
##ROC curve
r_valid <- roc(valid.set$not.fully.paid,logit.reg.valid,auc=T)
plot.roc(r_valid)
auc(r_valid)

##############################################################################


####Remove int.rate and dti from standardized df
loan=loan_std[,-c(3,6)]

##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

#######Modelling
logit.reg <- glm(not.fully.paid ~ ., data = train.set, family = "binomial")
summary(logit.reg)
logit.reg.train=predict(logit.reg,train.set,type="response")
##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.train > 0.5, 1, 0)), as.factor(train.set$not.fully.paid))
##ROC curve
r_train <- roc(train.set$not.fully.paid,logit.reg.train,auc=T)
plot.roc(r_train)
auc(r_train)

#######Validation
logit.reg.valid = predict(logit.reg,valid.set,type="response")

##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.valid > 0.5, 1, 0)), as.factor(valid.set$not.fully.paid))
##ROC curve
r_valid <- roc(valid.set$not.fully.paid,logit.reg.valid,auc=T)
plot.roc(r_valid)
auc(r_valid)


############################################################################

####Remove int.rate,dti and days.with.cr.line from standardized df
loan=loan_std[,-c(3,6,8)]

##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

#######Modelling
logit.reg <- glm(not.fully.paid ~ ., data = train.set, family = "binomial")
summary(logit.reg)
logit.reg.train=predict(logit.reg,train.set,type="response")
##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.train > 0.5, 1, 0)), as.factor(train.set$not.fully.paid))
##ROC curve
r_train <- roc(train.set$not.fully.paid,logit.reg.train,auc=T)
plot.roc(r_train)
auc(r_train)

#######Validation
logit.reg.valid = predict(logit.reg,valid.set,type="response")

##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.valid > 0.5, 1, 0)), as.factor(valid.set$not.fully.paid))
##ROC curve
r_valid <- roc(valid.set$not.fully.paid,logit.reg.valid,auc=T)
plot.roc(r_valid)
auc(r_valid)

################################################################################

####Remove int.rate,dti, days.with.cr.line and revol.util from standardized df
loan=loan_std[,-c(3,6,8,10)]

##Splitting the df to training and validation
train.index = sample(c(1:nrow(loan)),nrow(loan)*0.7)
train.set = loan[train.index,]###training dataset
valid.set = loan[-train.index,]##Validation dataset

#######Modelling
logit.reg <- glm(not.fully.paid ~ ., data = train.set, family = "binomial")
summary(logit.reg)
logit.reg.train=predict(logit.reg,train.set,type="response")
##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.train > 0.5, 1, 0)), as.factor(train.set$not.fully.paid))
##ROC curve
r_train <- roc(train.set$not.fully.paid,logit.reg.train,auc=T)
plot.roc(r_train)
auc(r_train)

#######Validation
logit.reg.valid = predict(logit.reg,valid.set,type="response")

##Confusion matrix
confusionMatrix(as.factor(ifelse(logit.reg.valid > 0.5, 1, 0)), as.factor(valid.set$not.fully.paid))
##ROC curve
r_valid <- roc(valid.set$not.fully.paid,logit.reg.valid,auc=T)
plot.roc(r_valid)
auc(r_valid)



################################   ODDS RATIO   ###############################
# ODDS RATIO
coef=data.frame(coefficients=logit.reg$coefficients)
coef$ebeta=exp(coef$coefficients)
coef$inv_ebeta=1/coef$ebeta
coef=coef[-c(1,3,4,5,6,7,8),]
coef1=coef[order(-coef$inv_ebeta),]
coef1


################################    EDA    ####################################
# EDA

#Density Plot of numerical variables
str(loan)
library(ggplot2)


################################  BOX PLOT ####################################

loan_box<-loan[,c(-1,-2,-14)]

par(mfrow=c(4,3))
for(i in 1:13)
 boxplot(loan_box[i],main=colnames(loan_box[i]))

#checking missing values
as.data.frame(colSums(is.na(loan)))

################################  DENSITY PLOT ##################################
par(mfrow = c(5,2))

ggplot(data = loan, aes(x = int.rate)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of int.rate') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = installment)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of installment') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = log.annual.inc)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of log.annual.inc') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = dti)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of dti') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = fico)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of fico') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = days.with.cr.line)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of days.with.cr.line') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = revol.bal)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of revol.bal') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = revol.util)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of revol.util') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 

ggplot(data = loan, aes(x = inq.last.6mths)) +
  geom_density(fill = 'cyan', color = 'cyan') +
  labs(title = 'Density Plot of inq.last.6mths') +
  theme(text = element_text(family = 'Gill Sans', color = "#444444")
        ,panel.background = element_rect(fill = '#444B5A')
        ,panel.grid.minor = element_line(color = '#4d5566')
        ,panel.grid.major = element_line(color = '#586174')
        ,plot.title = element_text(size = 24)
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(vjust = .5, angle = 0)
        ,axis.title.x = element_text(hjust = .5)
  ) 


########################## COUNT PLOT not.fully.paid ###############################
#Count Plot of not.fully.paid
ggplot(loan, aes(x=not.fully.paid, y=not.fully.paid / sum(not.fully.paid ), fill=not.fully.paid)) + geom_bar()
library(dplyr)
library(scales)
plotdata <- loan %>%
  count(not.fully.paid) %>%
  mutate(pct = n / sum(n),
         pctlabel = paste0(round(pct*100), "%"))

# plot the bars as percentages, 
# in decending order with bar labels
ggplot(plotdata, 
       aes(x = reorder(not.fully.paid, -pct),
           y = pct)) + 
  geom_bar(stat = "identity", 
           fill = "indianred3", 
           color = "black") +
  geom_text(aes(label = pctlabel), 
            vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  labs(x = "not.fully.paid", 
       y = "Percent", 
       title  = "not.fully.paid")

plotdata <- loan %>%
  count(inq.last.6mths) %>%
  mutate(pct = n / sum(n),
         pctlabel = paste0(round(pct*100), "%"))

################################  OTHER PLOTS ####################################
# plot the bars as percentages, 
# in decending order with bar labels
ggplot(plotdata, 
       aes(x = reorder(inq.last.6mths, -pct),
           y = pct)) + 
  geom_bar(stat = "identity", 
           fill = "indianred3", 
           color = "black") +
  geom_text(aes(label = pctlabel), 
            vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  labs(x = "inq.last.6mths", 
       y = "Percent", 
       title  = "inq.last.6mths")

plotdata <- loan %>%
  count(delinq.2yrs) %>%
  mutate(pct = n / sum(n),
         pctlabel = paste0(round(pct*100), "%"))

# plot the bars as percentages, 
# in decending order with bar labels
ggplot(plotdata, 
       aes(x = reorder(delinq.2yrs, -pct),
           y = pct)) + 
  geom_bar(stat = "identity", 
           fill = "indianred3", 
           color = "black") +
  geom_text(aes(label = pctlabel), 
            vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  labs(x = "delinq.2yrs", 
       y = "Percent", 
       title  = "delinq.2yrs")

plotdata <- loan %>%
  count(pub.rec) %>%
  mutate(pct = n / sum(n),
         pctlabel = paste0(round(pct*100), "%"))

# plot the bars as percentages, 
# in decending order with bar labels
library(ggthemes)
ggplot(loan, aes(x = purpose)) +
  geom_bar(fill = "tomato2") +
ggtitle("Frequency Distribution  of purpose") +
  labs(x = "purpose",y = "Frequency") +
  theme(axis.title.x = element_text(colour = "DarkGreen",size= 8),
        axis.title.y = element_text(colour = "DarkGreen",size= 8),
        axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(color = "Maroon",
                                  size = 10,
                                  family = "Arial",
                                  hjust = 0.5))+
  theme_solarized_2(light = FALSE,base_size = 15, base_family = "serif") +
  coord_flip()

