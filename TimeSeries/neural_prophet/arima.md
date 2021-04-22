# Neural prophet vs ARIMA
How fast is fitting long AR models using Neural prophet
In this quick test we will fit AR based models with different lags and see how long they take to fit.

To start, import some relevant libraries:


```python
from operator import le
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-whitegrid")

import logging

logging.basicConfig(level=logging.WARNING)

import sys
from pathlib import Path

import statsmodels.tsa.arima.model
from fbprophet import Prophet
from neuralprophet import NeuralProphet

sys.path.insert(0, str(Path(os.getcwd()).parent / "utilities"))
import data
import utilities
```

    ERROR:fbprophet.plot:Importing plotly failed. Interactive plots will not work.


The data used for this experiment is weather data - daily temperatures over a few years with no other covariates.
We format the data appropriately for the Facebook Prophet API,
with a datetime column `ds` and an output column `y`.
We split the data into the usual train and test sets.


```python
df = data.get_weather_data()
df_train, df_test = utilities.split_ts(df)

ax = df_train["y"].plot(figsize=(10, 6), label="train")
df_test["y"].plot(ax=ax, label="test")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Temp (°C)")

df_train = df_train.reset_index()
df_test = df_test.reset_index()

df_train.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-01-02</td>
      <td>-4.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-01-03</td>
      <td>-4.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-01-04</td>
      <td>-1.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-01-05</td>
      <td>-1.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-01-06</td>
      <td>-12.63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2009-01-07</td>
      <td>-21.09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2009-01-08</td>
      <td>-10.78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2009-01-09</td>
      <td>-13.91</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009-01-10</td>
      <td>-13.24</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009-01-11</td>
      <td>-10.75</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](arima_files/arima_3_1.png)
    


Now let's fit two sets of models. The first is an ARIMA model from `statsmodels` the second is an approximate AR model using `NeuralProphet`.
Each iteration we increase the number of lags used in the model, and we time how long each takes to fit.
The neural prophet model has the various seasonalities and change point parts disabled so it closer to a raw AR model.


```python
fit_time_ar = []
fit_time_np = []
mae_np = []
lag_range = range(1,25)
logging.getLogger("nprophet").setLevel(logging.WARNING)
for lag in lag_range:
    # fit statsmodels
    t1 = process_time()
    model_arima = statsmodels.tsa.arima.model.ARIMA(endog=df_train.set_index('ds'), order=(lag,0,0), freq='1D').fit()
    fit_time_ar.append(process_time() - t1)

    # fit neuralprophet
    t1 = process_time()
    model_nprophet_ar = NeuralProphet(
        growth="off",
        n_changepoints=0,
        n_forecasts=1,
        n_lags=lag,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
    )
    mae_np.append(model_nprophet_ar.fit(df_train, freq="D"))
    fit_time_np.append(process_time() - t1)
```

    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
    100%|██████████| 100/100 [00:00<00:00, 579.28it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     92%|█████████▏| 92/100 [00:00<00:00, 608.31it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     93%|█████████▎| 93/100 [00:00<00:00, 519.62it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     93%|█████████▎| 93/100 [00:00<00:00, 747.87it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     80%|████████  | 80/100 [00:00<00:00, 773.60it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     95%|█████████▌| 95/100 [00:00<00:00, 605.78it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     96%|█████████▌| 96/100 [00:00<00:00, 701.00it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     96%|█████████▌| 96/100 [00:00<00:00, 588.88it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     84%|████████▍ | 84/100 [00:00<00:00, 577.14it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     86%|████████▌ | 86/100 [00:00<00:00, 533.53it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
    100%|██████████| 100/100 [00:00<00:00, 602.81it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     87%|████████▋ | 87/100 [00:00<00:00, 861.91it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     86%|████████▌ | 86/100 [00:00<00:00, 880.36it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 748.06it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     93%|█████████▎| 93/100 [00:00<00:00, 807.52it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     89%|████████▉ | 89/100 [00:00<00:00, 803.41it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     92%|█████████▏| 92/100 [00:00<00:00, 844.05it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     94%|█████████▍| 94/100 [00:00<00:00, 797.13it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     96%|█████████▌| 96/100 [00:00<00:00, 684.42it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     98%|█████████▊| 98/100 [00:00<00:00, 821.19it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     85%|████████▌ | 85/100 [00:00<00:00, 442.58it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     87%|████████▋ | 87/100 [00:00<00:00, 790.10it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     75%|███████▌  | 75/100 [00:00<00:00, 823.47it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     78%|███████▊  | 78/100 [00:00<00:00, 723.23it/s]


Plotting these results we can see that the fitting time for neuralprophet is fairly flat.
As expected with fitting ARIMA models, the time of fitting increases rapidly with the number of lags.
This means neuralprophet allows us to fit longer AR based models which would not be otherwise possible.


```python
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(lag_range[:-1], fit_time_ar[:-1], '-x',label='ARIMA')
ax.plot(lag_range[:-1], fit_time_np[:-1], '-x',label='NeuralProphet')
fig.legend()
ax.set_xlabel('AR lag')
ax.set_ylabel('Fitting time (s)')
plt.show()
```


    
![png](arima_files/arima_7_0.png)
    


As such we can fit models with significantly longer lags.
This may not be necessary most of the time, but if we wanted to, we could do it!


```python
lag = 300
t1 = process_time()
logging.getLogger("nprophet").setLevel(logging.INFO)
model_nprophet_ar = NeuralProphet(
    growth="off",
    n_changepoints=0,
    n_forecasts=1,
    n_lags=lag,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    epochs = 100
)
loss_epoch = model_nprophet_ar.fit(df_train, freq="D")
print("\n")
print("\n")
print(f"Time taken: {process_time() - t1} s")
```

    INFO: nprophet.config - set_auto_batch_epoch: Auto-set batch_size to 32
    INFO:nprophet.config:Auto-set batch_size to 32
     84%|████████▍ | 84/100 [00:00<00:00, 782.63it/s]
    INFO: nprophet - _lr_range_test: learning rate range test found optimal lr: 2.85E-02
    INFO:nprophet:learning rate range test found optimal lr: 2.85E-02
    Epoch[100/100]: 100%|██████████| 100/100 [00:10<00:00,  9.55it/s, SmoothL1Loss=0.00179, MAE=2.23, RegLoss=0]
    
    
    
    
    Time taken: 9.842839999999995 s


Neuralprophet is based on an underlying pytorch model and trains using gradient descent.
The fitting time is related to the number of epochs. This is chosen automatically in the package.
The learning rate is by default chosen with pytorch lightning's learning rate finder.
The automatic learning rate can be a bit noisy and stop training early.
As such it may be necessary to plot the loss against epoch.
As a single example, the auto epoch number would be 27 in the following graph, where we train for 100.


```python
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(loss_epoch['SmoothL1Loss'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()
```


    
![png](arima_files/arima_11_0.png)
    


Even though, we can train for much more epochs and still successfully train a very long AR model.

## Prediction results
The models will not be identical. We can train a smaller AR model and inspect it's coefficients:


```python
lag = 5
model_arima = statsmodels.tsa.arima.model.ARIMA(endog=df_train.set_index('ds'), order=(lag,0,0), freq='1D').fit()

# fit neuralprophet
model_nprophet_ar = NeuralProphet(
    growth="off",
    n_changepoints=0,
    n_forecasts=1,
    n_lags=lag,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    epochs = 100
)
model_nprophet_ar.fit(df_train, freq="D")
```

    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
    INFO: nprophet.config - set_auto_batch_epoch: Auto-set batch_size to 32
    INFO:nprophet.config:Auto-set batch_size to 32
    100%|██████████| 100/100 [00:00<00:00, 666.78it/s]
    INFO: nprophet - _lr_range_test: learning rate range test found optimal lr: 2.31E+00
    INFO:nprophet:learning rate range test found optimal lr: 2.31E+00
    Epoch[100/100]: 100%|██████████| 100/100 [00:10<00:00,  9.88it/s, SmoothL1Loss=0.002, MAE=2.35, RegLoss=0]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SmoothL1Loss</th>
      <th>MAE</th>
      <th>RegLoss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.183347</td>
      <td>21.514868</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.006181</td>
      <td>4.135442</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003016</td>
      <td>2.880831</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002606</td>
      <td>2.692866</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.002437</td>
      <td>2.604848</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.002067</td>
      <td>2.387542</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.002079</td>
      <td>2.405267</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.002022</td>
      <td>2.359091</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.002015</td>
      <td>2.369480</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.002002</td>
      <td>2.352312</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(10,6), ncols=2)
ax[0].plot(model_arima.params.iloc[1:-1].to_numpy(), label='ARIMA')
ax[1].plot(np.flip(model_nprophet_ar.model.ar_weights.detach().numpy()).flatten(), label='np')
plt.show()
```


    
![png](arima_files/arima_15_0.png)
    


Their coefficients differ slightly.
This in turn causes their predictions to be slightly different:


```python
pred_arima = model_arima.predict(start=df_train['ds'].iloc[-1], end=df_train['ds'].iloc[-1] + pd.Timedelta('100D'))

pred_nprophet = df_train.copy()
for idx in range(100):
    future_nprophet = model_nprophet_ar.make_future_dataframe(
        df=pred_nprophet,
    )
    temp = model_nprophet_ar.predict(future_nprophet)
    temp['y'] = temp[['y','yhat1']].fillna(0).sum(axis=1)
    temp = temp[['ds','y']]
    pred_nprophet = pred_nprophet.append(temp.iloc[-1])
pred_nprophet = pred_nprophet.iloc[-101:].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
pred_arima.plot(ax=ax, label='ARIMA')
pred_nprophet.set_index('ds').plot(ax=ax, label='np')
df_train.set_index('ds').iloc[-200:].plot(ax=ax)
```




    <AxesSubplot:xlabel='ds'>




    
![png](arima_files/arima_17_1.png)
    

