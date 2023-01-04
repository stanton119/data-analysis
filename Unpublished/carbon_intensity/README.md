# Forecast intensity further ahead

Features:
* Wind speed - faster wind = more wind
* Cloud cover - clear sky = more solar in the daytime
* Temperate - higher temperates = less heating
* Fuel type availability
* Time of year
* Time of day
* Is weekend

Mid output
* Wind output - f(wind speed)
* Solar output - f(cloud cover, time of day, time of year)
* Net demand forecast = demand - wind output - solar output
* Interconnector forecast - f(time of year, time of day, is weekend)

Outputs:
* Intensity

Data sources
* Wind speed - historically?

Notes:
* To forecast regional values we likely need the country level

Todo:
* Seasonal decomposition of intensity