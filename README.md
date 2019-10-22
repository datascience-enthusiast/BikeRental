
# BIKE-RENTAL
This project predicts the daily count of bike rentals in a city based on the environmental conditions.

# PROBLEM STATEMENT
The aim of this project is to predict the count of bike rentals based on the seasonal and environmental settings. By predicting the count, it would be possible to help accommodate in managing the number of bikes required on a daily basis, and being prepared for high demand of bikes during peak periods.

# DATA
- instant     : Record index
- dteday      : Date
- season      : Season (1:springer, 2:summer, 3:fall, 4:winter)
- yr          : Year (0: 2011, 1:2012)
- mnth        : Month (1 to 12)
- hr          : Hour (0 to 23)
- holiday     : weather day is holiday or not (extracted fromHoliday Schedule)
- weekday     : Day of the week
- workingday  : If day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit  : (extracted fromFreemeteo)1: Clear, Few clouds, Partly cloudy, Partly cloudy 2: Mist + Cloudy, Mist + Broken clouds, Mist +                    Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy                        Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp        : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly                    scale)
- atemp       : Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_maxt_min), t_min=-16, t_max=+50 (only in                   hourly scale)
- hum         : Normalized humidity. The values are divided to 100 (max)
- windspeed   : Normalized wind speed. The values are divided to 67 (max)
- casual      : count of casual users
- registered  : count of registered users
- cnt         : count of total rental bikes including both casual and registered

### It is a regression Problem.
## All the steps implemented in this project
1. Data Pre-processing.
2. Data Visualization.
3. Outlier Analysis.
4. Missing value Analysis.
5. Feature Selection.
 -  Correlation analysis.
 -  Chi-Square test.
 -  Analysis of Variance(Anova) Test
 -  Multicollinearity Test.
6. Feature Scaling.
 -  Normalization.
7. Splitting into Train and Test Dataset.
8. Dimensionality Reduction using PCA technique.
9. Hyperparameter Optimization.
10. Model Development
I. Linear Regression  
II. Decision Tree 
III. Random Forest
IV. XGBOOST
11. Model Performance- Without PCA.
12. Model Performance- With PCA.
13. Conclusion
14. Python Code
15. R code.

# You can view the Project Report for more details
