# Survival models

Survival models are useful for time-to-event scenarios.

Models to try:
1. Pytorch probabilistic model
   1. predicts binary has a customer churned at day N following a treatment
   2. feature is only the number of days since the intervention
   3. and reward is churned by day N or not

Dataset
1. we need to generate days until churn for a set of customers
2. truncate to the reward horizon of day N
3. Features - days since intervention
   1. should we include features that are not relevant to our treatment

Scenario:
1. Assume we are Spotify/Netflix selling a subscription to a music service
2. Customers sign up and cancel or churn at some point in the future
3. We want to predict the number of days until a customer churns

Todo:
1. Model churn rate as not smooth, spikes at different dates.

References:
1. https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html

Setup:
```
uv add lifelines polars numpy seaborn matplotlib scikit-learn
uv add ipykernel --dev
```

## Survival models

When dealing with time to event data we observe censored data. For example an online service measuring the number of days until a customer churns after signing up to a subscription.
The event we are measuring is a customer churning.
If we collect data after 90 days, we observe churns only up to day 90, for anyone that churns after 90 days we do not observe. These unobserved customers are right-censored.

Treating these right-censored the same as those fully observed customers will bias our analysis.
For example averaging the days until churn will always report an underbiased value, as it will never be greater than 90.
Similarly including only observed customers creates a biased dataset, as we are selecting for customers who churned earlier.

We can formulate this as a survival problem. The standard framework is presented as follows:
Given that $T$ is the time an event takes place, survival functions represent the probability of an event happening after a given time:

$$
S(t) = Pr(T>t)
$$

The function must be monotonically decreasing to 0.

The hazard function represents the probability of an event happening at time $t$, given that the event has not yet happened:

$$
h(t) = \mathrm{lim}_{\delta t  \rightarrow 0} \frac{Pr(t<T<t+\delta t|T>t)}{\delta t}
$$

Its also represented as:

$$
h(t) = \frac{-S'(t)}{S(t)}
$$

We can use this to represent the survival function as a function of the hazard function:

$$
\begin{align}
\frac{S'(t)}{S(t)} & = -h(t) &\\
\int \frac{S'(t)}{S(t)} dt & = - \int h(t) dt &\\
\ln |S(t)| & = - \int h(t) dt + C &\\
S(t) & =\exp \left(- \int h(t) dt + C \right)
\end{align}
$$

At $t=0$, $S(t)=1$ as no events have yet happened. Therefore, $C=0$:

$$
S(t)  =\exp \left(- \int_0^t h(u) du \right)
$$

This makes intuitive sense - the survival function at time $t$ is the cumulative sum of hazards up to $t$. Hence the hazard integral has an alternative name, the cumulative hazard, $H(t)$:

$$
S(t)  =\exp \left(- H(t) \right)
$$


In our case the hazard function relates to the risk of churn on any given day.
The survival function is the probability of not having churned yet at a given day.

## Simulating data
First lets create a data generating function.

We simulate customers subscribing and then churning from a subscription service.


```python
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

n_customers = 1000
max_days = 90
np.random.seed(42)

customer_days = np.tile(np.arange(0, max_days), [n_customers, 1])


# Generate churn dates as base churn percent per day + exponentially decaying churn rate since subscription start
def churn_rate_per_day(x):
    return 0.001 + 0.05 * np.exp(-x / 20)


# simulate churn events per day
churn_rates = churn_rate_per_day(customer_days)
churn_events = np.random.binomial(1, churn_rates)
# find first churn event
churn_day_events = (churn_events & (np.cumsum(churn_events, axis=1) == 1)).astype(int)
churn_days = np.argmax(churn_day_events, axis=1) + 1
# if sum of churn_events is 0, set churn_days to max_days
churn_days[churn_day_events.sum(axis=1) == 0] = max_days

df = pl.DataFrame(
    {
        "customer_id": range(n_customers),
        "days_until_churn": churn_days,
        "observed": churn_days < max_days,
    }
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.head()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 3)</small><table border="1" class="dataframe"><thead><tr><th>customer_id</th><th>days_until_churn</th><th>observed</th></tr><tr><td>i64</td><td>i64</td><td>bool</td></tr></thead><tbody><tr><td>29</td><td>90</td><td>false</td></tr><tr><td>535</td><td>16</td><td>true</td></tr><tr><td>695</td><td>90</td><td>false</td></tr><tr><td>557</td><td>90</td><td>false</td></tr><tr><td>836</td><td>14</td><td>true</td></tr></tbody></table></div>



We model churn per day as an exponential decay. Where we see higher churn events early in the subscription:

$$
h(t) = 0.001 + 0.05 \exp \left( \frac{-t}{20} \right)
$$

We denote the cumumlative hazards function as:

$$
\begin{align}
H(t) & = \int h(t) dt &\\
H(t) & = \int 0.001 + 0.05 \exp \left( \frac{-t}{20} \right) &\\
H(t) & = 0.001t + (0.05)(-20) \exp\left(\frac{-t}{20}\right) + C &\\
H(t) & = 0.001t - \exp\left(\frac{-t}{20}\right) + 1
\end{align}
$$

Where $C=1$ to ensure $H(0)=0$.

The churn rate is shown as:


```python
x = np.arange(100)
y = churn_rate_per_day(x)
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(x=x, y=y, ax=ax)
ax.set(title="Churn rate against days since sign up")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/3878014295.py:6: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_5_1.png)
    


We see the days until churn reduce in proportion as customers stay longer, using the empical CDF:


```python
fig, ax = plt.subplots(figsize=(6, 4))
sns.ecdfplot(data=train_df, x="days_until_churn", ax=ax)
ax.set(title="Churn rate against days since sign up")
fig.show()
df
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/4179589568.py:4: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1_000, 3)</small><table border="1" class="dataframe"><thead><tr><th>customer_id</th><th>days_until_churn</th><th>observed</th></tr><tr><td>i64</td><td>i64</td><td>bool</td></tr></thead><tbody><tr><td>0</td><td>90</td><td>false</td></tr><tr><td>1</td><td>90</td><td>false</td></tr><tr><td>2</td><td>90</td><td>false</td></tr><tr><td>3</td><td>90</td><td>false</td></tr><tr><td>4</td><td>31</td><td>true</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>995</td><td>90</td><td>false</td></tr><tr><td>996</td><td>48</td><td>true</td></tr><tr><td>997</td><td>8</td><td>true</td></tr><tr><td>998</td><td>90</td><td>false</td></tr><tr><td>999</td><td>30</td><td>true</td></tr></tbody></table></div>




    
![png](survival_models_files/survival_models_7_2.png)
    


For survival modeling we can use the [lifelines package](https://lifelines.readthedocs.io/en/latest/index.html).

We can plot the empirical CDF as the life times for each customer via the lifelines package, where all blue customers have not yet churned as of 90 days.:


```python
from lifelines.plotting import plot_lifetimes

CURRENT_TIME = 90
fig, ax = plt.subplots(figsize=(6, 4))
plot_lifetimes(df["days_until_churn"], event_observed=df["observed"], ax=ax)
ax.set_xlim(0, CURRENT_TIME * 1.2)
ax.vlines(CURRENT_TIME, 0, n_customers, lw=2, linestyles="--")
ax.set_xlabel("time")
ax.set_title(f"Days until churn, at $t={CURRENT_TIME}$")
fig.show()
```

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/.venv/lib/python3.12/site-packages/lifelines/plotting.py:773: UserWarning: For less visual clutter, you may want to subsample to less than 25 individuals.
      warnings.warn(
    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/661472789.py:10: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_9_1.png)
    


### Non parametric estimators
To make statistical inferences, we want to estimate the survival function, $S(t)$, from the collected data.

A univariate way is the Kaplan Meier estimator. This is a non-parametric estimator, for which the estimator mean is the same as the empirical inverse CDF:


```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=train_df["days_until_churn"], event_observed=train_df["observed"])

empirical_cdf = (
    train_df.group_by("days_until_churn")
    .len()
    .sort("days_until_churn")
    .with_columns(
        ((len(train_df) - (pl.col("len").cum_sum())) / len(train_df)).alias("cdf")
    )
    .filter(pl.col("days_until_churn") < 90)
)

fig, ax = plt.subplots(figsize=(6, 4))
kmf.plot_survival_function(at_risk_counts=True, ax=ax)
sns.lineplot(
    data=empirical_cdf, x="days_until_churn", y="cdf", ax=ax, label="empirical_cdf"
)
ax.set(title="Kaplan Meier - Survival function")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/326303029.py:22: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_11_1.png)
    


The estimator variance increases over timeline as we fewer customers at higher days until churn as days increase.

Similarly, to estimate the cumulative Hazards function, $H(t)$, we can use the Nelson Aalan estimator:


```python
from lifelines import NelsonAalenFitter

naf = NelsonAalenFitter()
naf.fit(durations=train_df["days_until_churn"], event_observed=train_df["observed"])

fig, ax = plt.subplots(figsize=(6, 4))
naf.plot_cumulative_hazard(at_risk_counts=True, ax=ax)
ax.set(title="Nelson Aalen - Cumulative hazards function")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/1938030963.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_14_1.png)
    


### Parametric models
The above estimators are non-parametric, they assume no functional form.
This limits their ability to learn the hazards function at any time point given knowledge or other time points and also it limits the abilitiy to extrapolate past 90 days.

Examples of parametric survival estimators include the Weibull and exponential distributions. We can fit the distribution parameters and then extrapolate to longer time horizons.


```python
from lifelines import WeibullFitter, ExponentialFitter

wf = WeibullFitter().fit(
    durations=train_df["days_until_churn"],
    event_observed=train_df["observed"],
    timeline=np.arange(150),
)
exf = ExponentialFitter().fit(
    durations=train_df["days_until_churn"],
    event_observed=train_df["observed"],
    timeline=np.arange(150),
)
```


```python
fig, ax = plt.subplots(figsize=(6, 4))
wf.plot_survival_function(ax=ax)
exf.plot_survival_function(ax=ax)

kmf.plot_survival_function(ax=ax)
sns.lineplot(
    data=empirical_cdf, x="days_until_churn", y="cdf", ax=ax, label="empirical_cdf"
)
ax.set(title="Kaplan Meier - Survival function")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/1596966498.py:10: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_17_1.png)
    


Note the distribution choice strongly reflects how well it fits the observed data, so any extrapolation is dependent on how well the distribution assumptions reflect the data generating process.
For example here, neither Weibull or exponential distributions fit that well. The exponential distribution assumes a constant hazard rate against time ($h(t)=\frac{1}{\lambda}$).
Our churn rate/hazard rate reduces over time, so an exponential distribution will be a poor fit for this.

Given our knowledge of the ground truth hazard function we can create the same distribution and fit through lifelines.
We use the cumulative hazards function derived above.
Lifelines fits the distribution parameters through the scipy optimize function:


```python
from lifelines.fitters import ParametricUnivariateFitter

import autograd.numpy as np


class ExponentialHazardFitter(ParametricUnivariateFitter):
    _fitted_parameter_names = ["alpha_0_", "alpha_1_"]
    _bounds = [(0, None), (0, None)]

    def _cumulative_hazard(self, params, times):
        alpha_0, alpha_1 = params[0], params[1]
        return alpha_0 - np.exp(-times / alpha_1) + 1


ehf = ExponentialHazardFitter().fit(
    durations=train_df["days_until_churn"],
    event_observed=train_df["observed"],
    timeline=np.arange(200),
)
print("fitted parameters:")
print(ehf.alpha_0_, ehf.alpha_1_)
```

    fitted parameters:
    1.0000000000000023e-09 21.837275241776833


    /Users/stantoon/Documents/VariousProjects/github/data-analysis/.venv/lib/python3.12/site-packages/lifelines/fitters/__init__.py:1011: ApproximationWarning: 
    The Hessian for ExponentialHazardFitter's fit was not invertible. We will instead approximate it using the pseudo-inverse.
    
    It's advisable to not trust the variances reported, and to be suspicious of the fitted parameters too. Perform plots of the cumulative hazard to help understand the latter's bias.
    
      warnings.warn(warning_text, exceptions.ApproximationWarning)


The distribution (green) fits reasonably well to the original parameters of (0.001 and 20). We can see this in an improve survival function and cumulative hazards function:


```python
fig, ax = plt.subplots(figsize=(6, 4))
wf.plot_survival_function(ax=ax)
exf.plot_survival_function(ax=ax)
ehf.plot_survival_function(ax=ax)

kmf.plot_survival_function(at_risk_counts=True, ax=ax)
ax.set(title="Survival functions")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/2545171641.py:8: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_21_1.png)
    



```python
fig, ax = plt.subplots(figsize=(6, 4))
wf.plot_cumulative_hazard(ax=ax)
exf.plot_cumulative_hazard(ax=ax)

ehf.plot_cumulative_hazard(ax=ax)
naf.plot_cumulative_hazard(at_risk_counts=True, ax=ax)
ax.set(title="Cumulative hazards functions")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_60909/1079980104.py:8: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](survival_models_files/survival_models_22_1.png)
    


The survival and cumulative hazard functions match much closer to the empirical data.

## Survival regression

For building hazards that are conditioned on covariates we can use the Cox proportional hazards model, which uses the following form:
$$
h(t|x) = b_0(t) \exp({\beta x})
$$

It is made up of a base hazard rate and an exponential term which creates a ratio.
We are measuring the proportional change in hazards due to covariates.

Lifelines requires a dataset with duration and event columns and covariates.
For rows with an observed event the duration corresponds to the time of the event.
For rows without an observed event yet, the duration corresponds to the lifetime so far.

In the following dataset, these are represented by the week (duration) and arrest (event) columns:


```python
from lifelines.datasets import load_rossi

rossi = load_rossi()
rossi.head()
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
      <th>week</th>
      <th>arrest</th>
      <th>fin</th>
      <th>age</th>
      <th>race</th>
      <th>wexp</th>
      <th>mar</th>
      <th>paro</th>
      <th>prio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The Cox model is fit as:


```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(rossi, duration_col="week", event_col="arrest", show_progress=True)

cph.print_summary()
```

    Iteration 1: norm_delta = 5.09e-01, step_size = 0.9500, log_lik = -675.38063, newton_decrement = 1.68e+01, seconds_since_start = 0.0
    Iteration 2: norm_delta = 1.39e-01, step_size = 0.9500, log_lik = -659.79004, newton_decrement = 9.92e-01, seconds_since_start = 0.0
    Iteration 3: norm_delta = 1.80e-02, step_size = 0.9500, log_lik = -658.76197, newton_decrement = 1.42e-02, seconds_since_start = 0.0
    Iteration 4: norm_delta = 1.83e-04, step_size = 1.0000, log_lik = -658.74766, newton_decrement = 1.32e-06, seconds_since_start = 0.0
    Iteration 5: norm_delta = 1.97e-08, step_size = 1.0000, log_lik = -658.74766, newton_decrement = 1.34e-14, seconds_since_start = 0.0
    Convergence success after 5 iterations.



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
  <tbody>
    <tr>
      <th>model</th>
      <td>lifelines.CoxPHFitter</td>
    </tr>
    <tr>
      <th>duration col</th>
      <td>'week'</td>
    </tr>
    <tr>
      <th>event col</th>
      <td>'arrest'</td>
    </tr>
    <tr>
      <th>baseline estimation</th>
      <td>breslow</td>
    </tr>
    <tr>
      <th>number of observations</th>
      <td>432</td>
    </tr>
    <tr>
      <th>number of events observed</th>
      <td>114</td>
    </tr>
    <tr>
      <th>partial log-likelihood</th>
      <td>-658.75</td>
    </tr>
    <tr>
      <th>time fit was run</th>
      <td>2025-02-26 15:51:06 UTC</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style="min-width: 12px;"></th>
      <th style="min-width: 12px;">coef</th>
      <th style="min-width: 12px;">exp(coef)</th>
      <th style="min-width: 12px;">se(coef)</th>
      <th style="min-width: 12px;">coef lower 95%</th>
      <th style="min-width: 12px;">coef upper 95%</th>
      <th style="min-width: 12px;">exp(coef) lower 95%</th>
      <th style="min-width: 12px;">exp(coef) upper 95%</th>
      <th style="min-width: 12px;">cmp to</th>
      <th style="min-width: 12px;">z</th>
      <th style="min-width: 12px;">p</th>
      <th style="min-width: 12px;">-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fin</th>
      <td>-0.38</td>
      <td>0.68</td>
      <td>0.19</td>
      <td>-0.75</td>
      <td>-0.00</td>
      <td>0.47</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>-1.98</td>
      <td>0.05</td>
      <td>4.40</td>
    </tr>
    <tr>
      <th>age</th>
      <td>-0.06</td>
      <td>0.94</td>
      <td>0.02</td>
      <td>-0.10</td>
      <td>-0.01</td>
      <td>0.90</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>-2.61</td>
      <td>0.01</td>
      <td>6.79</td>
    </tr>
    <tr>
      <th>race</th>
      <td>0.31</td>
      <td>1.37</td>
      <td>0.31</td>
      <td>-0.29</td>
      <td>0.92</td>
      <td>0.75</td>
      <td>2.50</td>
      <td>0.00</td>
      <td>1.02</td>
      <td>0.31</td>
      <td>1.70</td>
    </tr>
    <tr>
      <th>wexp</th>
      <td>-0.15</td>
      <td>0.86</td>
      <td>0.21</td>
      <td>-0.57</td>
      <td>0.27</td>
      <td>0.57</td>
      <td>1.30</td>
      <td>0.00</td>
      <td>-0.71</td>
      <td>0.48</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>mar</th>
      <td>-0.43</td>
      <td>0.65</td>
      <td>0.38</td>
      <td>-1.18</td>
      <td>0.31</td>
      <td>0.31</td>
      <td>1.37</td>
      <td>0.00</td>
      <td>-1.14</td>
      <td>0.26</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>paro</th>
      <td>-0.08</td>
      <td>0.92</td>
      <td>0.20</td>
      <td>-0.47</td>
      <td>0.30</td>
      <td>0.63</td>
      <td>1.35</td>
      <td>0.00</td>
      <td>-0.43</td>
      <td>0.66</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>prio</th>
      <td>0.09</td>
      <td>1.10</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.15</td>
      <td>1.04</td>
      <td>1.16</td>
      <td>0.00</td>
      <td>3.19</td>
      <td>&lt;0.005</td>
      <td>9.48</td>
    </tr>
  </tbody>
</table><br><div>
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
  <tbody>
    <tr>
      <th>Concordance</th>
      <td>0.64</td>
    </tr>
    <tr>
      <th>Partial AIC</th>
      <td>1331.50</td>
    </tr>
    <tr>
      <th>log-likelihood ratio test</th>
      <td>33.27 on 7 df</td>
    </tr>
    <tr>
      <th>-log2(p) of ll-ratio test</th>
      <td>15.37</td>
    </tr>
  </tbody>
</table>
</div>


We should check the assumptions of the Cox's model are applicable here.

We not that there are issues in applying to this dataset.
I will ignore this for now to focus on the use of the lifelines library.


```python
cph.check_assumptions(rossi)
```

    The ``p_value_threshold`` is set at 0.01. Even under the null hypothesis of no violations, some
    covariates will be below the threshold by chance. This is compounded when there are many covariates.
    Similarly, when there are lots of observations, even minor deviances from the proportional hazard
    assumption will be flagged.
    
    With that in mind, it's best to use a combination of statistical tests and visual tests to determine
    the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)``
    and looking for non-constant lines. See link [A] below for a full example.
    



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
  <tbody>
    <tr>
      <th>null_distribution</th>
      <td>chi squared</td>
    </tr>
    <tr>
      <th>degrees_of_freedom</th>
      <td>1</td>
    </tr>
    <tr>
      <th>model</th>
      <td>&lt;lifelines.CoxPHFitter: fitted with 432 total ...</td>
    </tr>
    <tr>
      <th>test_name</th>
      <td>proportional_hazard_test</td>
    </tr>
  </tbody>
</table>
</div><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>test_statistic</th>
      <th>p</th>
      <th>-log2(p)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">age</th>
      <th>km</th>
      <td>11.03</td>
      <td>&lt;0.005</td>
      <td>10.12</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>11.45</td>
      <td>&lt;0.005</td>
      <td>10.45</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">fin</th>
      <th>km</th>
      <td>0.02</td>
      <td>0.89</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.02</td>
      <td>0.90</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">mar</th>
      <th>km</th>
      <td>0.60</td>
      <td>0.44</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.71</td>
      <td>0.40</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">paro</th>
      <th>km</th>
      <td>0.12</td>
      <td>0.73</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.13</td>
      <td>0.71</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">prio</th>
      <th>km</th>
      <td>0.02</td>
      <td>0.88</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>0.02</td>
      <td>0.89</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">race</th>
      <th>km</th>
      <td>1.44</td>
      <td>0.23</td>
      <td>2.12</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>1.43</td>
      <td>0.23</td>
      <td>2.11</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">wexp</th>
      <th>km</th>
      <td>7.48</td>
      <td>0.01</td>
      <td>7.32</td>
    </tr>
    <tr>
      <th>rank</th>
      <td>7.31</td>
      <td>0.01</td>
      <td>7.19</td>
    </tr>
  </tbody>
</table>


    
    
    1. Variable 'age' failed the non-proportional test: p-value is 0.0007.
    
       Advice 1: the functional form of the variable 'age' might be incorrect. That is, there may be
    non-linear terms missing. The proportional hazard test used is very sensitive to incorrect
    functional forms. See documentation in link [D] below on how to specify a functional form.
    
       Advice 2: try binning the variable 'age' using pd.cut, and then specify it in `strata=['age',
    ...]` in the call in `.fit`. See documentation in link [B] below.
    
       Advice 3: try adding an interaction term with your time variable. See documentation in link [C]
    below.
    
    
    2. Variable 'wexp' failed the non-proportional test: p-value is 0.0063.
    
       Advice: with so few unique values (only 2), you can include `strata=['wexp', ...]` in the call in
    `.fit`. See documentation in link [E] below.
    
    ---
    [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
    [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
    [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
    [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
    [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
    





    []



We can show the predicted baseline hazard.
The baseline hazard is only fit to the max duration in the dataset.
It does not get extrapolated.


```python
cph.baseline_hazard_.plot()
```




    <Axes: >




    
![png](survival_models_files/survival_models_32_1.png)
    


We can predict survival at different time horizons.
This is predicted for all durations that were available in the dataset.
It can be conditioned on each row's lifetime so far to get survival rates for subsequent time frames.


```python
cph.predict_survival_function(
    rossi.head(5), conditional_after=rossi.head()["week"]
).transpose()  # survival rates at weeks after their current lifetime
cph.predict_survival_function(
    rossi.head(5), times=[0, 10, 30, 52], conditional_after=[0] * 5
).transpose()  # survival rates conditioned on 0 current lifetime, evaluated at given time frames
cph.predict_survival_function(rossi.head(5), times=[0, 10, 30, 52]).transpose()
cph.predict_survival_function(rossi.head(5)).transpose()
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
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
      <th>6.0</th>
      <th>7.0</th>
      <th>8.0</th>
      <th>9.0</th>
      <th>10.0</th>
      <th>...</th>
      <th>42.0</th>
      <th>43.0</th>
      <th>44.0</th>
      <th>45.0</th>
      <th>46.0</th>
      <th>47.0</th>
      <th>48.0</th>
      <th>49.0</th>
      <th>50.0</th>
      <th>52.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.997616</td>
      <td>0.995230</td>
      <td>0.992848</td>
      <td>0.990468</td>
      <td>0.988085</td>
      <td>0.985699</td>
      <td>0.983305</td>
      <td>0.971402</td>
      <td>0.966614</td>
      <td>0.964223</td>
      <td>...</td>
      <td>0.784733</td>
      <td>0.774567</td>
      <td>0.769460</td>
      <td>0.764349</td>
      <td>0.754116</td>
      <td>0.751552</td>
      <td>0.746427</td>
      <td>0.733641</td>
      <td>0.725969</td>
      <td>0.715699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993695</td>
      <td>0.987411</td>
      <td>0.981162</td>
      <td>0.974941</td>
      <td>0.968739</td>
      <td>0.962552</td>
      <td>0.956370</td>
      <td>0.926001</td>
      <td>0.913958</td>
      <td>0.907978</td>
      <td>...</td>
      <td>0.526079</td>
      <td>0.508214</td>
      <td>0.499383</td>
      <td>0.490641</td>
      <td>0.473429</td>
      <td>0.469176</td>
      <td>0.460745</td>
      <td>0.440128</td>
      <td>0.428038</td>
      <td>0.412181</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.994083</td>
      <td>0.988183</td>
      <td>0.982314</td>
      <td>0.976468</td>
      <td>0.970639</td>
      <td>0.964820</td>
      <td>0.959004</td>
      <td>0.930402</td>
      <td>0.919043</td>
      <td>0.913399</td>
      <td>...</td>
      <td>0.547334</td>
      <td>0.529874</td>
      <td>0.521230</td>
      <td>0.512664</td>
      <td>0.495770</td>
      <td>0.491589</td>
      <td>0.483296</td>
      <td>0.462975</td>
      <td>0.451031</td>
      <td>0.435335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999045</td>
      <td>0.998089</td>
      <td>0.997133</td>
      <td>0.996176</td>
      <td>0.995216</td>
      <td>0.994254</td>
      <td>0.993287</td>
      <td>0.988460</td>
      <td>0.986508</td>
      <td>0.985531</td>
      <td>...</td>
      <td>0.907577</td>
      <td>0.902855</td>
      <td>0.900469</td>
      <td>0.898071</td>
      <td>0.893242</td>
      <td>0.892026</td>
      <td>0.889587</td>
      <td>0.883460</td>
      <td>0.879752</td>
      <td>0.874752</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.997626</td>
      <td>0.995250</td>
      <td>0.992878</td>
      <td>0.990507</td>
      <td>0.988135</td>
      <td>0.985758</td>
      <td>0.983374</td>
      <td>0.971520</td>
      <td>0.966752</td>
      <td>0.964370</td>
      <td>...</td>
      <td>0.785530</td>
      <td>0.775396</td>
      <td>0.770304</td>
      <td>0.765209</td>
      <td>0.755007</td>
      <td>0.752451</td>
      <td>0.747341</td>
      <td>0.734592</td>
      <td>0.726942</td>
      <td>0.716702</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



We can show the effect of covariates on the survival function:


```python
cph.plot_partial_effects_on_outcome(covariates="age", values=[20, 40])
```




    <Axes: >




    
![png](survival_models_files/survival_models_36_1.png)
    

