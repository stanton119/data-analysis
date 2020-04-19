# Transport for London Cycle Data Exploration


Data exploration:  
``data_exploration.py``

Regression analysis:  
``pending``

Utility functions:  
``data_proc.py``



## Todo
* Add new features to regression model
* Use prophet to remove the time trends
* Model the change to expected journey counts from the time trends
* Add is_holiday feature
* Add short term weather trend - in case people cycle based on yesterdays weather and not todays forcast
* Fit the regression model on bootstrap samples to get sampling distribution and covariances for each regression coefficient
* XBM model
  * Ignore month/year as feature, but can use hour/day
