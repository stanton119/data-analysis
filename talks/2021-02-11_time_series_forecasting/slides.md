# Time series forecasting
2021-02-11
(Adapted slides)

----
Outline:

* Literature review
* Python time series libraries
* My questions

---
## Literature review

----
### Traditional methods
[Rob J. Hyndman](https://otexts.com/fpp2/)
* Exponential smoothing
* ARIMA
  * Long lags can be very slow
* TBATS model
  * Seasonalities can change with time, no covariates


----
### Traditional methods

* Dynamic harmonic regression
  * Seasonal decomposition type models
* Hierarchical forecasting
  * Proportion out time series

----
### M Competitions

[Wikipedia](https://en.wikipedia.org/wiki/Makridakis_Competitions)
* Traditional statistical approaches generally win
* M3 - Theta model - very fast, similar to prophet with no covariates
* M4 - winner - [Uber first ML hybrid solution to win](https://eng.uber.com/m4-forecasting-competition/)
  * HoltWinters + RNN
* Univariate based

----
### M5 Competition

* [Kaggle based](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/163415), multivariate Walmart data
* Ensembles everywhere
* 8k submissions, fair amount of noise
* Top results, accuracy and uncertainty competitions:
  * GBM ensembles
  * deepAR
  * traditional
* [XGBoost/LightGBM has been commonly used in forecasting](https://github.com/microsoft/LightGBM/blob/master/examples/README.md#machine-learning-challenge-winning-solutions)

----
### Prophet

* [fbprophet](https://facebook.github.io/prophet/)
* [NeuralProphet](https://github.com/ourownstory/neural_prophet)
  * Pytorch implementation
  * Very similar API
  * Option for auto regressive effects
  * SGD - long history faster to fit
* [Structure time series (tensorflow)](https://blog.tensorflow.org/2019/03/structural-time-series-modeling-in.html)
  * Slow compared to prophet

----
### ML solutions

* [NBeats](https://arxiv.org/abs/1905.10437)
  * Very deep stack of fully-connected layers
* [DeepAR](https://arxiv.org/abs/1704.04110)
  * Probabilistic LSTM networks
* [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
  * NLP solutions tricky as time series typically much longer

----
### ML solutions

* [Octopus energy model](https://tech.octopus.energy/data-discourse/PyData2019/TimeSeries.html)
  * NN to predict energy consumption
  * Embedding for each customer
  * Forecast individual customers
  * Bayesian solution
* [Datadog](https://youtu.be/0zpg9ODE6Ww?t=2032)
  * Median based robust long term trend estimation

----
### Python Implementations

* [StatsModels](https://www.statsmodels.org/stable/user-guide.html#time-series-analysis)
* [Darts](https://github.com/unit8co/darts)
* [Pytorch forecasting](https://github.com/jdb78/pytorch-forecasting)
* [GluonTS](https://github.com/awslabs/gluon-ts)
  * Amazon - mxnet library
  * Pytorch-TS
* [SkTime](https://github.com/alan-turing-institute/sktime)
* [Keras](https://github.com/philipperemy/n-beats)
* All different APIs...
* All different dependencies...

----
## My questions

* Stationarity requirement for ML solutions?
* Quantile loss function to get confidence intervals in the absense of a PDF
* NN - embeddings to represent related time series?