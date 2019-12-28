# TFL Cycle Analysis

## Todo
* Regression analysis

## Dataset
The dataset counts the number of journeys made per hour in each day of 2015-2017.  
## Findings
Seasonal trends in number of journeys per day  
![](images/against_time.png)

### Weekly data
Weekends have fewer journeys per day  
![](images/journeys_per_week.png)

Journeys by hour peak during morning and evening rush hour. They have a different distribution on weekends.  
![](images/journeys_per_hour_boxplot.png)

Splitting by hour/day of the week. Shows Friday evening has fewer journeys and Thursday/Friday evening peak is more widely distributed
![](images/journeys_per_hour_week.png) ![](images/journeys_per_hour_week_prop.png)

### Monthly data
Winters are not as popular for cycling  
![](images/journeys_per_month.png)

Splitting by hour/month of the year shows the winter months have fewer journeys and have a tighter distribution at evening rush hour. During the summer months a higher proportion of journeys are made latter into the evening.  
![](images/journeys_per_hour_month.png) ![](images/journeys_per_hour_month_prop.png)

### Weather data
Weather features are engineered by averaging the various weather measures over the whole day.  
Better conditions generally correlate with high number of journeys. This is likely part confounded by the seasonality seen.  
![](images/weather_codes.png)

### Regression analysis
Looking at the correlation of features to journey numbers there are multiple fairly strong correlations  
![](images/pairplot.png)

