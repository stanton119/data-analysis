"""
Possible bug in fb prophet for multiplicative seasonalities.
"""

# %%
import numpy as np
import pandas as pd
import prophet

# %%
df = pd.DataFrame()
df['ds'] = pd.date_range(start='2019-01-01',end='2021-01-01',freq='4H')
df['daily_effect'] = np.cos(df['ds'].dt.hour/24 * 2*np.pi)*-0.5+0.5
df['yearly_effect'] = np.cos(df['ds'].dt.dayofyear/365 * 2*np.pi)*-0.3 + 0.7
df['y'] = df['daily_effect'] * df['yearly_effect']

df.set_index('ds')['y'].plot()

df.set_index('ds').loc['2020-01-01':'2020-01-05',:].plot()
df.set_index('ds').loc['2020-07-01':'2020-07-05',:].plot()

# %%
m = prophet.Prophet(growth='flat',weekly_seasonality=False,seasonality_mode='multiplicative')
m.fit(df)
# %%
df_future = m.make_future_dataframe(periods=365*6,freq='4H', include_history=False)
df_forecast = m.predict(df_future)

df_forecast['yhat2'] = df_forecast['trend'] * (1 + df_forecast['yearly']) * (1 + df_forecast['daily'])

# %%
m.plot(df_forecast)
fig = m.plot_components(df_forecast)


# %%
df_forecast.set_index('ds')['yhat'].plot()
df_forecast.set_index('ds')['yhat2'].plot()



# %%
df['y'] = np.log(df['y']+1)
# %%
m = prophet.Prophet(growth='flat',weekly_seasonality=False,seasonality_mode='additive')
m.fit(df)
# %%
df_future = m.make_future_dataframe(periods=365*6,freq='4H', include_history=False)
df_forecast = m.predict(df_future)

df_forecast['yhat'] = np.exp(df_forecast['yhat']) - 1
df_forecast['yhat_lower'] = np.exp(df_forecast['yhat_lower']) - 1
df_forecast['yhat_upper'] = np.exp(df_forecast['yhat_upper']) - 1

# proposed aggregation
df_forecast['yhat2'] = df_forecast['trend'] * (1 + df_forecast['yearly']) * (1 + df_forecast['daily'])

# %%
m.plot(df_forecast)
fig = m.plot_components(df_forecast)


# %%
df_forecast.set_index('ds')['yhat'].plot()
df_forecast.set_index('ds')['yhat2'].plot()
