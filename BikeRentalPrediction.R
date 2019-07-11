## Clear all the global variables and set working directory
rm(list=ls(all=T))
setwd("D:/Shad_Data/Edwisor/Projects/BikeRentalProject")

# For Reading Error Metrics in normal form, rather than in exponential
options(scipen = 999)

# Set Seed for getting constant results
set.seed(12345)

## Read Data
bike_df <- read.csv('day.csv', sep = ',')

## Load Libraries
x = c("ggplot2", "GGally", "plotly", "usdm", "xgboost","rattle", "Matrix","corrgram", "caret", "randomForest", "C50", "dummies", "e1071", "Information",
      "rpart", "gbm", 'DataCombine')

#install.packages(x)
lapply(x, require, character.only = TRUE)


##################################Missing Values Analysis###############################################
missing_val = data.frame(apply(bike_df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(bike_df)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "R_Missing_perc.csv", row.names = F)

ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage") + theme_bw()

## Observation : No Missing value found in dataset


###########################################Explore the data##########################################
# Structure of Dataset
str(bike_df)

# Shape of Dataset
dim(bike_df)

# Display Top 5 records of dataset
head(bike_df)

################# Converting appropriate required Datatypes ####################

#Converting colmnns into categorical factors as they contain unqiue values
category_column_names = c('season','yr','mnth','holiday','weekday','weathersit', 'workingday')

for(i in category_column_names){
  bike_df[,i] = as.factor(bike_df[,i])
}

#Converting date into datetime format
bike_df$dteday = as.Date(bike_df$dteday , "%d%m%Y")

#Converting rest variables into Numeric for standardization
for(i in 1:ncol(bike_df)){
  if(class(bike_df[,i]) == 'integer'){
    bike_df[,i] = as.numeric(bike_df[,i])
  }
}

######################### Graphical Analysis ##########################

#Distribution of Response Variable 'cnt'
fit <- density(bike_df$cnt)
plot_ly(x = bike_df$cnt, type = "histogram", name = "Count Distribution") %>%
  add_trace(x = fit$x, y = fit$y, type = "scatter", mode = "lines", fill = "tozeroy", yaxis = "y2", name = "Density") %>% 
  layout(yaxis2 = list(overlaying = "y", side = "right"))

#Bike Rental Count on Monthly Basis
df <- aggregate(cnt ~ mnth, bike_df, sum)
plot_ly(x = ~mnth, y = ~cnt , data = df, type = "bar", text = ~cnt , marker = list(color = 'rgb(158,202,225)',line = list(color = 'rgb(8,48,107)', width = 1.5)))


#Bike Rental Count on Seasonal Basis
df <- aggregate(cnt ~ season, bike_df, sum)
plot_ly(df, x = ~season, y = ~cnt ,type = "scatter", mode = "lines+markers")


#Bike Rental Count on basis of Weather Type
df <- aggregate(cnt ~ weathersit, bike_df, sum)
plot_ly(df, x = ~weathersit, y = ~cnt ,type = "scatter", mode = "lines+markers")


#Bike Rental Count on basis of Day
df <- aggregate(cnt ~ weekday, bike_df, sum)
plot_ly(df, x = ~weekday, y = ~cnt ,type = "box") %>%
  add_trace( x = ~weekday, y = ~cnt, type="scatter", mode = "lines+markers")



##################################Feature Selection################################################
## Correlation Plot 
numeric_index = sapply(bike_df,is.numeric) #selecting only numeric
corrgram(bike_df[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
ggpairs(bike_df[,c('temp','hum','windspeed','cnt')])

##### Chi Square Test of Independance for Categorical Variables
# Season vs Month
chi = chisq.test(table(bike_df$season,bike_df$mnth))
c(chi$statistic, chi$p.value)
barplot(table(bike_df$season,bike_df$mnth))

# season vs weathersit
chi = chisq.test(table(bike_df$season,bike_df$weathersit))
c(chi$statistic, chi$p.value)
barplot(table(bike_df$yr,bike_df$mnth))

# holiday vs weekday
chi = chisq.test(table(bike_df$holiday,bike_df$weekday))
c(chi$statistic, chi$p.value)
barplot(table(bike_df$holiday,bike_df$weekday))


##################### Dimension Reduction ########################
#'atemp' since it shows alot of correlation with 'temp' variable
#'casual' and 'registered' since its aggregation sum results to 'cnt' and we have to predict 'cnt' using other important aspects
#'dteday' since it doesnt help in prediction and also we have covered all the necessary factors in 'season','yr','mnth'
bike_df = subset(bike_df, 
                 select = -c(atemp,casual,registered,dteday,instant))


############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(bike_df,is.numeric) #selecting only numeric

numeric_data = bike_df[,numeric_index]

column_names = colnames(numeric_data)

for (i in 1:length(column_names))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (column_names[i]), x = "cnt"), data = subset(bike_df))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=column_names[i],x="cnt")+
           ggtitle(paste("Box plot for",column_names[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

# # #Remove outliers using boxplot method
# #loop to remove from all variables
# for(i in column_names){
#   val = bike_df[,i][bike_df[,i] %in% boxplot.stats(bike_df[,i])$out]
#   bike_df = bike_df[which(!bike_df[,i] %in% val),]
# }

# #Replace all outliers with NA and impute
for(i in column_names){
  val = bike_df[,i][bike_df[,i] %in% boxplot.stats(bike_df[,i])$out]
  
  bike_df[,i][bike_df[,i] %in% val] = NA
}

bike_df = knnImputation(bike_df, k = 3)



################## Model Development ######################
## Split it into train and test
train_index = sample(1:nrow(bike_df), 0.8 * nrow(bike_df))
train = bike_df[train_index,]
test = bike_df[-train_index,]

## Train the model using rPart
# method = anova for regression, method = class for classification
DT_regressor = rpart(cnt ~ ., data = train, method = "anova")

### Testing the built model using test data
test_predictions = predict(DT_regressor, test[,-11])

## Method to calculate MAPE ( Mean Absolute Percentage Error )

mape = function(yact, ypred){
  mean(abs((yact - ypred)/yact))*100
}

## Method to calculate RMSE ( Root Mean Square Error )

rmse = function(yact, ypred){
  mse = mean((yact-ypred)**2)
  sqrt(mse)
}


## Method to calculate Accuracy of model
accuracy = function(mape){
  accuracy = abs(100 - mean(mape))
  accuracy
}

mape(test[,11], test_predictions)

rmse(test[,11], test_predictions)

accuracy(mape(test[,11], test_predictions))

### Alternate Method 
regr.eval(test[,11], test_predictions, stats = c("mae","mse","rmse","mape"))

##### Visualize Tree
fancyRpartPlot(DT_regressor)

##### Tune decision tree on basis of Complexity paramter (cp)
printcp(DT_regressor)

plotcp(DT_regressor)

ptree<- prune(DT_regressor, cp= DT_regressor$cptable[which.min(DT_regressor$cptable[,"xerror"]),"CP"])

fancyRpartPlot(ptree, uniform=TRUE, main="Pruned Classification Tree")

### Testing the built model using test data on pruned tree
pruned_test_predictions = predict(ptree, test[,-11])

mape(test[,11], pruned_test_predictions)

rmse(test[,11], pruned_test_predictions)

accuracy(mape(test[,11], pruned_test_predictions))

################### Random Forest #######################

RF_Regressor = randomForest(cnt ~ . , data = train , ntree=1000, importance = TRUE)

importance(RF_Regressor, type = 1)

### Testing the built model using test data on pruned tree
RF_predictions = predict(RF_Regressor, test[,-11])

mape(test[,11], RF_predictions)

rmse(test[,11], RF_predictions)

accuracy(mape(test[,11], RF_predictions))

plot(RF_Regressor)


######################## Linear Regression ########################

vifcor(bike_df[,8:11], th = 0.9)

#run regression model
lm_model = lm(cnt ~., data = train)

#Summary of the model
summary(lm_model)


## Plot Residual vs fitted , Normal Q-Q Plot, Scale Location Plots
#plot(lm_model)

#Predict
Linear_predictions = predict(lm_model, test[,-11])

#Calculate Evualtion Metrics for Linear Regression
mape(test[,11], Linear_predictions)

rmse(test[,11], Linear_predictions)

accuracy(mape(test[,11], Linear_predictions))

# perform step-wise feature selection
lm_tune_model <- step(lm_model)

# summarize the selected model
summary(lm_tune_model)

## Plot Residual vs fitted , Normal Q-Q Plot, Scale Location Plots
#plot(lm_tune_model)

#Predict
predictions_LR_tuned = predict(lm_tune_model, test[,-11])

#Calculate Evualtion Metrics for Tuned Linear Regression Model
mape(test[,11], predictions_LR_tuned)

rmse(test[,11], predictions_LR_tuned)

accuracy(mape(test[,11], predictions_LR_tuned))


################## XGBOOST ########################
train_matrix <- sparse.model.matrix(cnt ~ .-1, data = train)
test_matrix <- sparse.model.matrix(cnt ~ .-1, data = test)

xgb <- xgboost(data = as.matrix(train_matrix),
               label = as.matrix(train$cnt),
               booster = "gbtree", 
               objective = "reg:linear", 
               max.depth = 8, 
               eta = 0.5, 
               nthread = 2, 
               nround = 100, 
               min_child_weight = 1, 
               subsample = 0.75, 
               colsample_bytree = 1, 
               num_parallel_tree = 3)

xgb_predictions <- predict(xgb, test_matrix)

#Calculate Evualtion Metrics for XGBoost
mape(test[,11], xgb_predictions)

rmse(test[,11], xgb_predictions)

accuracy(mape(test[,11], xgb_predictions))

#### Conclusion : XGboost shows the best accuracy and less MAPE and RMSE from among all