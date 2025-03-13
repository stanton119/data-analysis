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




    
![svg](arima_files/arima_3_1.svg)
    


Now let's fit two sets of models. The first is an ARIMA model from `statsmodels` the second is an approximate AR model using `NeuralProphet`.
Each iteration we increase the number of lags used in the model, and we time how long each takes to fit.
The neural prophet model has the various seasonalities and change point parts disabled so it closer to a raw AR model.
We also need to change the loss function to standard MSE.


```python
fit_time_ar = []
fit_time_np = []
mae_np = []
lag_range = range(1, 25)
logging.getLogger("nprophet").setLevel(logging.WARNING)
for lag in lag_range:
    # fit statsmodels
    t1 = process_time()
    model_arima = statsmodels.tsa.arima.model.ARIMA(
        endog=df_train.set_index("ds"), order=(lag, 0, 0), freq="1D"
    ).fit()
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
        loss_func="MSE",
        normalize="off",
    )
    mae_np.append(model_nprophet_ar.fit(df_train, freq="D"))
    fit_time_np.append(process_time() - t1)
```

    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 587.23it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     91%|█████████ | 91/100 [00:00<00:00, 730.38it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 756.66it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 752.12it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     84%|████████▍ | 84/100 [00:00<00:00, 767.28it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 763.74it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     92%|█████████▏| 92/100 [00:00<00:00, 692.78it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     89%|████████▉ | 89/100 [00:00<00:00, 788.81it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 796.34it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     81%|████████  | 81/100 [00:00<00:00, 757.69it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 780.54it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     86%|████████▌ | 86/100 [00:00<00:00, 711.94it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     80%|████████  | 80/100 [00:00<00:00, 811.55it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     84%|████████▍ | 84/100 [00:00<00:00, 783.74it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     83%|████████▎ | 83/100 [00:00<00:00, 697.32it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     87%|████████▋ | 87/100 [00:00<00:00, 704.59it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     81%|████████  | 81/100 [00:00<00:00, 829.23it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     84%|████████▍ | 84/100 [00:00<00:00, 761.93it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     83%|████████▎ | 83/100 [00:00<00:00, 739.54it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     87%|████████▋ | 87/100 [00:00<00:00, 791.21it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 488.08it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 770.79it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     88%|████████▊ | 88/100 [00:00<00:00, 789.60it/s]
    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     86%|████████▌ | 86/100 [00:00<00:00, 729.68it/s]


Plotting these results we can see that the fitting time for neuralprophet is fairly flat.
As expected with fitting ARIMA models, the time of fitting increases rapidly with the number of lags.
This means neuralprophet allows us to fit longer AR based models which would not be otherwise possible.


```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lag_range[:-1], fit_time_ar[:-1], "-x", label="ARIMA")
ax.plot(lag_range[:-1], fit_time_np[:-1], "-x", label="NeuralProphet")
fig.legend()
ax.set_xlabel("AR lag")
ax.set_ylabel("Fitting time (s)")
plt.show()
```


    
![svg](arima_files/arima_7_0.svg)
    


## Prediction results
To ensure these results are meaningful we need to check the models they are fitting are comparible.
Let's train a smaller neural prophet so we can check for similarity with an actual AR model.


```python
lag = 5
model_arima = statsmodels.tsa.arima.model.ARIMA(
    endog=df_train.set_index("ds"), order=(lag, 0, 0), freq="1D"
).fit()

# fit neuralprophet
model_nprophet_ar = NeuralProphet(
    growth="off",
    n_changepoints=0,
    n_forecasts=1,
    n_lags=lag,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    epochs=100,
    loss_func="MSE",
    normalize="off",
)
model_nprophet_ar.fit(df_train, freq="D")
```

    /Users/Rich/Developer/miniconda3/envs/neural_prophet_env/lib/python3.8/site-packages/statsmodels/tsa/base/tsa_model.py:524: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      warnings.warn('No frequency information was'
     86%|████████▌ | 86/100 [00:00<00:00, 630.39it/s]





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
      <th>MSELoss</th>
      <th>MAE</th>
      <th>RegLoss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.424664</td>
      <td>3.443962</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.272208</td>
      <td>2.658695</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.073363</td>
      <td>2.393834</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.814939</td>
      <td>2.356173</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.030052</td>
      <td>2.372665</td>
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
      <td>8.739055</td>
      <td>2.340827</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>8.709106</td>
      <td>2.333304</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>8.679108</td>
      <td>2.335190</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>8.694123</td>
      <td>2.333388</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>8.666990</td>
      <td>2.332289</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
ax[0].plot(model_arima.params.iloc[1:-1].to_numpy())
ax[0].set_title("ARIMA")
ax[0].set_xlabel("AR Lag")
ax[0].set_ylabel("Coefficient")
ax[1].plot(np.flip(model_nprophet_ar.model.ar_weights.detach().numpy()).flatten())
ax[1].set_title("np")
ax[1].set_xlabel("AR Lag")
plt.show()
```


    
![svg](arima_files/arima_10_0.svg)
    


Their coefficients are nearly identical. As such predictions from each model are nearly the same:


```python
pred_arima = model_arima.predict(
    start=df_train["ds"].iloc[-1], end=df_train["ds"].iloc[-1] + pd.Timedelta("100D")
)

pred_nprophet = df_train.copy()
for idx in range(100):
    future_nprophet = model_nprophet_ar.make_future_dataframe(
        df=pred_nprophet,
    )
    temp = model_nprophet_ar.predict(future_nprophet)
    temp["y"] = temp[["y", "yhat1"]].fillna(0).sum(axis=1)
    temp = temp[["ds", "y"]]
    pred_nprophet = pred_nprophet.append(temp.iloc[-1])
pred_nprophet = pred_nprophet.iloc[-101:].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
pred_arima.plot(ax=ax, label="ARIMA")
pred_nprophet.set_index("ds")["y"].plot(ax=ax, label="np")
df_train.set_index("ds")["y"].iloc[-200:].plot(ax=ax, label="actual")
ax.set_ylabel("Temp (°C)")
fig.legend()
plt.show()
```


    
![svg](arima_files/arima_12_0.svg)
    


## Long lags
Due to the significantly faster fitting time we can train models with significantly longer lags.
This may not be necessary most of the time, but if we wanted to, we could do it!
For example in the following we train an AR model with 500 lags.


```python
lag = 500
t1 = process_time()
model_nprophet_ar = NeuralProphet(
    growth="off",
    n_changepoints=0,
    n_forecasts=1,
    n_lags=lag,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    epochs=100,
    loss_func="MSE",
    normalize="off",
)
loss_epoch = model_nprophet_ar.fit(df_train, freq="D")
print("\n")
print("\n")
print(f"Time taken: {process_time() - t1} s")
```

     69%|██████▉   | 69/100 [00:00<00:00, 779.16it/s]
                                                                                                     
    
    
    
    Time taken: 7.613330999999988 s


Neuralprophet is based on an underlying pytorch model and trains using gradient descent.
The fitting time is related to the number of epochs. This is chosen automatically in the package.
The learning rate is by default chosen with pytorch lightning's learning rate finder.
The automatic learning rate can be a bit noisy and stop training early.
As such it may be necessary to plot the loss against epoch.
As a single example, the auto epoch number would be 27 in the following graph, where we train for 100.


```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(loss_epoch["MSELoss"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()
```


    
![svg](arima_files/arima_16_0.svg)
    


Even though, we can train for much more epochs and still successfully train a very long AR model.

## Seasonality via AR models?
With a very long AR model we could potentially capture seasonalities which would be impossible before.
On this daily weather data we have a yearly seasonality, we can't capture that with a normal AR model.
However by using neuralprophet we actually can...


```python
pred_arima = model_arima.predict(
    start=df_train["ds"].iloc[-1], end=df_train["ds"].iloc[-1] + pd.Timedelta("200D")
)

pred_nprophet = df_train.copy()
for idx in range(200):
    future_nprophet = model_nprophet_ar.make_future_dataframe(
        df=pred_nprophet,
    )
    temp = model_nprophet_ar.predict(future_nprophet)
    temp["y"] = temp[["y", "yhat1"]].fillna(0).sum(axis=1)
    temp = temp[["ds", "y"]]
    pred_nprophet = pred_nprophet.append(temp.iloc[-1])
pred_nprophet = pred_nprophet.iloc[-201:].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
pred_arima.plot(ax=ax, label="ARIMA")
pred_nprophet.set_index("ds")["y"].plot(ax=ax, label="np")
df_train.set_index("ds")["y"].iloc[-1500:].plot(ax=ax, label="actual")
ax.set_ylabel("Temp (°C)")
fig.legend()
plt.show()
```


    
![svg](arima_files/arima_18_0.svg)
    


The prediction shows we are able to capture some of the yearly seasonality!
The prediction is quite noisy. The pytorch based approach also allows us to build in regularisation into the AR coefficients.
This should reduce the models ability to model the noise and as such hopefully produce more smooth predictions.
This in turn means we don't have to worry as much about getting the correct AR lag count when specifying the model
Here we set the `ar_sparsity` to a *very* low value, which pushes most of the AR coefficients close to 0.


```python
model_nprophet_ar_sparse = NeuralProphet(
    growth="off",
    n_changepoints=0,
    n_forecasts=1,
    n_lags=lag,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    epochs=100,
    loss_func="MSE",
    normalize="off",
    ar_sparsity=0.0001,  # *** #
)
model_nprophet_ar_sparse.fit(df_train, freq="D")
```

     65%|██████▌   | 65/100 [00:00<00:00, 798.22it/s]





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
      <th>MSELoss</th>
      <th>MAE</th>
      <th>RegLoss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48.809326</td>
      <td>5.547075</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.614197</td>
      <td>5.474120</td>
      <td>1.891101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.352211</td>
      <td>5.397998</td>
      <td>3.777936</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.840831</td>
      <td>5.304262</td>
      <td>5.659053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.259937</td>
      <td>5.211620</td>
      <td>7.524754</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>8.483594</td>
      <td>2.326365</td>
      <td>27.624358</td>
    </tr>
    <tr>
      <th>96</th>
      <td>8.484436</td>
      <td>2.326009</td>
      <td>26.968793</td>
    </tr>
    <tr>
      <th>97</th>
      <td>8.483948</td>
      <td>2.325840</td>
      <td>26.380927</td>
    </tr>
    <tr>
      <th>98</th>
      <td>8.484425</td>
      <td>2.325925</td>
      <td>25.855129</td>
    </tr>
    <tr>
      <th>99</th>
      <td>8.484677</td>
      <td>2.325938</td>
      <td>25.359007</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>



Comparing the coefficients shows the sparse model is more conservative.


```python
fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
ax[0].plot(
    np.flip(model_nprophet_ar.model.ar_weights.detach().numpy()).flatten(), label="np"
)
ax[1].plot(
    np.flip(model_nprophet_ar_sparse.model.ar_weights.detach().numpy()).flatten(),
    label="np",
)
plt.show()
```


    
![svg](arima_files/arima_22_0.svg)
    



```python
pred_nprophet_sparse = df_train.copy()
for idx in range(200):
    future_nprophet = model_nprophet_ar_sparse.make_future_dataframe(
        df=pred_nprophet_sparse,
    )
    temp = model_nprophet_ar_sparse.predict(future_nprophet)
    temp["y"] = temp[["y", "yhat1"]].fillna(0).sum(axis=1)
    temp = temp[["ds", "y"]]
    pred_nprophet_sparse = pred_nprophet_sparse.append(temp.iloc[-1])
pred_nprophet_sparse = pred_nprophet_sparse.iloc[-201:].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
pred_nprophet.set_index("ds")["y"].plot(ax=ax, label="np")
pred_nprophet_sparse.set_index("ds")["y"].plot(ax=ax, label="np_sparse")
df_train.set_index("ds")["y"].iloc[-1500:].plot(ax=ax, label="actual")
ax.set_ylabel("Temp (°C)")
fig.legend()
plt.show()
```


    
![svg](arima_files/arima_23_0.svg)
    

