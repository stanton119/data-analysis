# DoubleML with binary variables and marginal effects

When using DoubleML, what do we do with binary outcomes or binary treatments? It seems like we just plough ahead and we the same models we would have if we used continuous outcomes.

Here we explore the effects of using appropriate models and 'less' appropriate models to see if there's an actual difference.

Ref:
https://github.com/py-why/EconML/issues/204

## Create dummy data

We create a data generating function to produce some dummy features with confounding and generate random outcomes from a linear model.


```python
import numpy as np
import pandas as pd


def print_results(array):
    print([f"{_x:.3f}" for _x in array])


def logit(p):
    return np.log(p) - np.log(1 - p)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def no_confound(x: np.array):
    return np.zeros(x.shape[0])


def linear_confound(x: np.array, a: float = 1.0, b: float = 0.5):
    return a * x[:, 0] + b


def generate_treatment_data(
    n_samples: int = 1000,
    n_features: int = 4,
    treatment_binary: bool = False,
    seed: int = None,
    confounding_fcn: callable = None,
    treatment_noise: float = 0.1,
):
    if confounding_fcn is None:
        confounding_fcn = no_confound

    rand = np.random.default_rng(seed)

    # generate random features
    x = rand.normal(
        loc=rand.normal(size=n_features),
        scale=rand.exponential(size=n_features),
        size=(n_samples, n_features),
    )

    t_x = confounding_fcn(x)
    if treatment_binary:
        t = rand.binomial(n=1, p=inv_logit(t_x), size=n_samples)
    else:
        t = treatment_noise * rand.normal(size=n_samples) + t_x

    x = np.concatenate([t[:, np.newaxis], x], axis=1)

    t_col = "t"
    x_cols = [f"x_{idx+1}" for idx in range(n_features)]

    return pd.DataFrame(data=x, columns=[t_col] + x_cols), t_col, x_cols


def generate_outcome_data(
    x: pd.DataFrame,
    outcome_binary: bool = False,
    outcome_noise: float = 0.1,
    seed: int = None,
    bias: float = None,
    weights: np.array = None,
):
    rand = np.random.default_rng(seed)

    n_samples, n_features = x.shape
    if bias is None:
        bias = rand.normal()
    if weights is None:
        weights = rand.normal(size=(n_features, 1))
    y = bias + np.dot(x, weights) + outcome_noise * rand.normal()

    if outcome_binary:
        y_avg = inv_logit(y)
        y = rand.binomial(n=1, p=y_avg, size=(n_samples, 1))
    else:
        y_avg = None

    return y, bias, weights, y_avg
```

## Linear case

We start with the linear case.

We create data from a linear model where a linear regression model would be ideal.

We:
1. generate the data
2. fit a linear regression model with all the features and treatment
3. fit a linear regression model with only the treatment, ignoring the confounding features


```python
import statsmodels.api as sm

# generate data
x_df, t_col, x_cols = generate_treatment_data(
    treatment_binary=False, confounding_fcn=linear_confound, seed=0
)
y, bias, weights, _ = generate_outcome_data(x=x_df, outcome_binary=False, seed=0)

print("True weights")
print_results(weights.flatten())


# fit models
linear_model = sm.OLS(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, all features:")
print_results(np.array(linear_model.params)[1:])

linear_model = sm.OLS(y, sm.add_constant(x_df[t_col])).fit()
print("Est weights, missing confounders, biased results:")
print_results(np.array(linear_model.params)[1:])
```

    True weights
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est weights, all features:
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est weights, missing confounders, biased results:
    ['0.479']


Linear regression works as expected and when we don't include the confounder features we get a biased estimate for the treatment uplift.

### Estimate marginal effects
The marginal effect represents the change in `y` given a unit change in the treatment `t` (or the features `x` as well).

We calculate this by estimating derivatives in `y` for each feature to represent the average marginal effects.
For the linear data model, these marginal effects are the same as the data generating weights.
When we estimate them, the results are, as expected, asymptotically equal to the estimate linear regression coefficients.


```python
def get_marginal_effects(x, d_x: float = 1.0, model_fcn=None):
    d_y = []
    for col in x.columns:
        _x = x.copy()
        _x[col] = _x[col] + d_x
        d_y.append((model_fcn(_x) - model_fcn(x)).mean() / d_x)

    return d_y


print("True est. marginal effects:")
print_results(
    get_marginal_effects(
        x=x_df,
        model_fcn=lambda x: generate_outcome_data(
            x, outcome_binary=False, bias=bias, weights=weights, seed=0
        )[0],
    )
)
```

    True est. marginal effects:
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']


### Double ML

Assume all features are confounders.
Assume no features for CATE.

We can use GBMs for the models for `y` and `t` without creating bias.
We get similar results if we use linear regression for these.

As the data is generated from a linear model, linear regression is the ideal model to use, so we get slightly worse results with GBMs.


```python
import econml.dml
import sklearn.ensemble
import sklearn.linear_model


est = econml.dml.LinearDML(
    model_t=sklearn.linear_model.LinearRegression(),
    model_y=sklearn.linear_model.LinearRegression(),
    random_state=0,
)
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("True weights")
print_results(weights.flatten())

print("Est marginal effect - linear models")
display(est.const_marginal_ate_inference())


est = econml.dml.LinearDML(
    model_t=sklearn.ensemble.GradientBoostingRegressor(random_state=0),
    model_y=sklearn.ensemble.GradientBoostingRegressor(random_state=0),
    random_state=0,
)
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("Est marginal effect - GBMs")
display(est.const_marginal_ate_inference())
```

    True weights
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est marginal effect - linear models



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>        <th>zstat</th>        <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.132</td>       <td>0.0</td>     <td>-1232558224036509.5</td>   <td>0.0</td>     <td>-0.132</td>        <td>-0.132</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.132</td>          <td>-0.132</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
       <td>0.0</td>         <td>-0.132</td>         <td>-0.132</td>    
</tr>
</table>


    Est marginal effect - GBMs


    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>  <th>zstat</th> <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.092</td>      <td>0.033</td>    <td>-2.816</td>  <td>0.005</td>    <td>-0.156</td>        <td>-0.028</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.092</td>          <td>-0.092</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
      <td>0.033</td>        <td>-0.156</td>         <td>-0.028</td>    
</tr>
</table>


So we have agreement for the treatment effect over linear regression, doubleML and this matches the average marginal effect.

## Binary outcome case

In this case the linear model coefficients are not the same as the average marginal effect.

Here, the marginal effect is the average change in probability of observing a positive outcome, also known as percentage basis points.

Here we see that the linear regression coefficients do not match the data weights, as expected.
The logistic regression model is the appropriate model to use and we recover the data weights without bias.

Side note: As the binary outcome introduces a lot of noise we increase the number of samples to get reasonable accuracy.


```python
# generate data
x_df, t_col, x_cols = generate_treatment_data(
    treatment_binary=False, confounding_fcn=linear_confound, n_samples=int(1e6), seed=0
)
y, bias, weights, y_avg = generate_outcome_data(x=x_df, outcome_binary=True, seed=0)

print("True weights")
print_results(weights.flatten())

# fit models
linear_model = sm.OLS(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, linear regression:")
print_results(np.array(linear_model.params)[1:])

logit_model = sm.Logit(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, logistic regression:")
print_results(np.array(logit_model.params)[1:])
```

    True weights
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est weights, linear regression:
    ['-0.024', '0.143', '0.024', '-0.124', '0.084']
    Optimization terminated successfully.
             Current function value: 0.657852
             Iterations 5
    Est weights, logistic regression:
    ['-0.104', '0.613', '0.103', '-0.533', '0.360']


However if we find the average marginal effect from the logistic model we get the linear regression coefficients.

We use the average `y` to calculate the true marginal effects as it reduces the noise a lot.


```python
print("True est. marginal effects:")
marginal_effects = get_marginal_effects(
    x=x_df,
    model_fcn=lambda x: generate_outcome_data(
        x, outcome_binary=True, bias=bias, weights=weights, seed=0
    )[-1],
)
print_results(marginal_effects)


print("Average marginal effect, logistic regression:")
print_results(logit_model.get_margeff().margeff)

print("Est weights, linear regression:")
print_results(np.array(linear_model.params)[1:])
```

    True est. marginal effects:
    ['-0.030', '0.148', '0.024', '-0.119', '0.084']
    Average marginal effect, logistic regression:
    ['-0.024', '0.143', '0.024', '-0.124', '0.084']
    Est weights, linear regression:
    ['-0.024', '0.143', '0.024', '-0.124', '0.084']


Therefore if we are interested in the average marginal effects we don't need to use logistic regression.

This opens the door to use doubleML with regressors to estimate the average marginal effect even if we have binary outcomes.


```python
if 1:
    est = econml.dml.LinearDML(
        model_t=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
        model_y=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
        random_state=0,
    )
else:
    est = econml.dml.LinearDML(
        model_t=sklearn.linear_model.LinearRegression(),
        model_y=sklearn.linear_model.LinearRegression(),
        random_state=0,
    )
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("True est. marginal effects:")
print_results(marginal_effects)

print("Est marginal effect")
display(est.const_marginal_ate_inference())
```

    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    True est. marginal effects:
    ['-0.030', '0.148', '0.024', '-0.119', '0.084']
    Est marginal effect



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>  <th>zstat</th> <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.024</td>      <td>0.005</td>    <td>-5.018</td>   <td>0.0</td>     <td>-0.033</td>        <td>-0.015</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.024</td>          <td>-0.024</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
      <td>0.005</td>        <td>-0.033</td>         <td>-0.015</td>    
</tr>
</table>


## Binary treatment case

If we have a binary treatment the final model should be a linear regression.

Here the marginal effects are equal to the data weights.


```python
# generate data
x_df, t_col, x_cols = generate_treatment_data(
    treatment_binary=True, confounding_fcn=linear_confound, n_samples=int(1e6), seed=0
)
y, bias, weights, y_avg = generate_outcome_data(x=x_df, outcome_binary=False, seed=0)

print("True weights")
print_results(weights.flatten())


# fit models
linear_model = sm.OLS(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, linear regression:")
print_results(np.array(linear_model.params)[1:])

print("True est. marginal effects:")
marginal_effects = get_marginal_effects(
    x=x_df,
    model_fcn=lambda x: generate_outcome_data(
        x, outcome_binary=False, bias=bias, weights=weights, seed=0
    )[0],
)
print_results(marginal_effects)
```

    True weights
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est weights, linear regression:
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    True est. marginal effects:
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']


And we see that using regression based double ML returns the same marginal effect back:


```python
if 1:
    est = econml.dml.LinearDML(
        model_t=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
        model_y=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
        random_state=0,
    )
else:
    est = econml.dml.LinearDML(
        model_t=sklearn.linear_model.LinearRegression(),
        model_y=sklearn.linear_model.LinearRegression(),
        random_state=0,
    )
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("True est. marginal effects:")
print_results(marginal_effects)

print("Est marginal effect")
display(est.const_marginal_ate_inference())
```

    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    True est. marginal effects:
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    Est marginal effect



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>   <th>zstat</th>   <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.132</td>       <td>0.0</td>     <td>-1795.605</td>   <td>0.0</td>     <td>-0.132</td>        <td>-0.132</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.132</td>          <td>-0.132</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
       <td>0.0</td>         <td>-0.132</td>         <td>-0.132</td>    
</tr>
</table>


## Binary outcome and treatment case

We see that the marginal effect from a logistic model, linear regression coefficient and the data marginal effect match well.


```python
# generate data
x_df, t_col, x_cols = generate_treatment_data(
    treatment_binary=True, confounding_fcn=linear_confound, n_samples=int(1e6), seed=0
)
y, bias, weights, y_avg = generate_outcome_data(x=x_df, outcome_binary=True, seed=0)

print("True weights")
print_results(weights.flatten())

print("True est. marginal effects:")
marginal_effects = get_marginal_effects(
    x=x_df,
    model_fcn=lambda x: generate_outcome_data(
        x, outcome_binary=True, bias=bias, weights=weights, seed=0
    )[-1],
)
print_results(marginal_effects)


# fit models
linear_model = sm.OLS(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, linear regression:")
print_results(np.array(linear_model.params)[1:])

logit_model = sm.Logit(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, logistic regression:")
print_results(np.array(logit_model.params)[1:])

print("Average marginal effect, logistic regression:")
print_results(logit_model.get_margeff().margeff)
```

    True weights
    ['-0.132', '0.640', '0.105', '-0.536', '0.362']
    True est. marginal effects:
    ['-0.030', '0.147', '0.024', '-0.118', '0.083']
    Est weights, linear regression:
    ['-0.030', '0.148', '0.024', '-0.123', '0.083']
    Optimization terminated successfully.
             Current function value: 0.654005
             Iterations 5
    Est weights, logistic regression:
    ['-0.130', '0.641', '0.104', '-0.533', '0.358']
    Average marginal effect, logistic regression:
    ['-0.030', '0.148', '0.024', '-0.123', '0.083']


And we see that using regression based double ML returns the same marginal effect back:


```python
est = econml.dml.LinearDML(
    model_t=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
    model_y=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0),
    random_state=0,
)
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("True est. marginal effects:")
print_results(marginal_effects)

print("Est marginal effect")
display(est.const_marginal_ate_inference())
```

    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    True est. marginal effects:
    ['-0.030', '0.147', '0.024', '-0.118', '0.083']
    Est marginal effect



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>  <th>zstat</th>  <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
     <td>-0.03</td>      <td>0.001</td>    <td>-28.912</td>   <td>0.0</td>     <td>-0.032</td>        <td>-0.028</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>         <td>-0.03</td>           <td>-0.03</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
      <td>0.001</td>        <td>-0.032</td>         <td>-0.028</td>    
</tr>
</table>


Does the choise of classifier or regressor matter?

This should work, but our residuals from each model would be discrete and therefore less informative. We get answers which are close here.


```python
est = econml.dml.LinearDML(
    model_t=sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
    model_y=sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
    random_state=0,
)
est.fit(Y=y, T=x_df[t_col], X=None, W=x_df.drop(columns=t_col))

print("True est. marginal effects:")
print_results(marginal_effects)

print("Est marginal effect")
display(est.const_marginal_ate_inference())
```

    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    True est. marginal effects:
    ['-0.030', '0.147', '0.024', '-0.118', '0.083']
    Est marginal effect



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>  <th>zstat</th> <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.009</td>      <td>0.001</td>    <td>-8.054</td>   <td>0.0</td>     <td>-0.011</td>        <td>-0.007</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.009</td>          <td>-0.009</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
      <td>0.001</td>        <td>-0.011</td>         <td>-0.007</td>    
</tr>
</table>


Does the use of discrete treatment matter?

This calls the `predict_proba` on the `model_t` object. It also apply one hot encoding to the treatment column.
We get very similar answers to using regressor GBMs. Therefore we can likely get away with using a single model definition for all binary and continuous model problems.

> There was a bug that required the `_set_transformed_treatment_names` function in econml to be adjusted.


```python
est = econml.dml.LinearDML(
    model_t=sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
    model_y=sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
    random_state=0,
    discrete_treatment=True,
    categories=[0, 1],
)
est.fit(Y=y, T=x_df[t_col].to_numpy(), X=None, W=x_df.drop(columns=t_col).to_numpy())

print("True est. marginal effects:")
print_results(marginal_effects)

print("Est marginal effect")
display(est.const_marginal_ate_inference())
```

    `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    True est. marginal effects:
    ['-0.030', '0.147', '0.024', '-0.118', '0.083']
    Est marginal effect



<table class="simpletable">
<caption>Uncertainty of Mean Point Estimate</caption>
<tr>
  <th>mean_point</th> <th>stderr_mean</th>  <th>zstat</th>  <th>pvalue</th> <th>ci_mean_lower</th> <th>ci_mean_upper</th>
</tr>
<tr>
    <td>-0.031</td>      <td>0.001</td>    <td>-23.085</td>   <td>0.0</td>     <td>-0.034</td>        <td>-0.028</td>    
</tr>
</table>
<table class="simpletable">
<caption>Distribution of Point Estimate</caption>
<tr>
  <th>std_point</th> <th>pct_point_lower</th> <th>pct_point_upper</th>
</tr>
<tr>
     <td>0.0</td>        <td>-0.031</td>          <td>-0.031</td>     
</tr>
</table>
<table class="simpletable">
<caption>Total Variance of Point Estimate</caption>
<tr>
  <th>stderr_point</th> <th>ci_point_lower</th> <th>ci_point_upper</th>
</tr>
<tr>
      <td>0.001</td>        <td>-0.034</td>         <td>-0.028</td>    
</tr>
</table>


# TODO

## Similar results with S-Learners

If we use an S-learner and find the ATE as E{Y|T=do=1} - E{Y|T=do=0} we should observe the same marginal effect as a logistic regression and linear regression.


```python
import econml.metalearners

print("True est. marginal effects:")
print_results(marginal_effects)

est = econml.metalearners.SLearner(
    overall_model=sklearn.linear_model.LinearRegression(),
    # categories=[0,1]
)
T = x_df[t_col].to_numpy()
X = x_df.drop(columns=t_col).to_numpy()
est.fit(Y=y, T=T, X=X)

print("Est marginal effect - linear regression")
display(est.ate(x_df.drop(columns=t_col).to_numpy()))

est = econml.metalearners.SLearner(
    overall_model=sklearn.linear_model.LogisticRegression(penalty=None),
    # categories=[0,1]
)
est.fit(Y=y, T=x_df[t_col].to_numpy(), X=x_df.drop(columns=t_col).to_numpy())

print("Est marginal effect - logistic regression")
display(est.ate(x_df.drop(columns=t_col).to_numpy()))
```

    True est. marginal effects:
    ['-0.030', '0.147', '0.024', '-0.118', '0.083']


    `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.


    Est marginal effect - linear regression



    array([-0.02994736])


    `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().


    Est marginal effect - logistic regression



    array([-0.086629])



```python
# coefs match
est.overall_model.coef_
# est.overall_model.predict_proba(x_df.drop(columns=t_col).to_numpy())

X = x_df.drop(columns=t_col).to_numpy()
T = x_df[t_col].to_numpy()
np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
```


    ---------------------------------------------------------------------------

    AxisError                                 Traceback (most recent call last)

    Cell In[42], line 7
          5 X = x_df.drop(columns=t_col).to_numpy()
          6 T = x_df[t_col].to_numpy()
    ----> 7 np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)


    File <__array_function__ internals>:180, in sum(*args, **kwargs)


    File ~/miniconda3/envs/project_env/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2298, in sum(a, axis, dtype, out, keepdims, initial, where)
       2295         return out
       2296     return res
    -> 2298 return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
       2299                       initial=initial, where=where)


    File ~/miniconda3/envs/project_env/lib/python3.10/site-packages/numpy/core/fromnumeric.py:86, in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
         83         else:
         84             return reduction(axis=axis, out=out, **passkwargs)
    ---> 86 return ufunc.reduce(obj, axis, dtype, out, **passkwargs)


    AxisError: axis 1 is out of bounds for array of dimension 1



```python
x_df.drop(columns=t_col).to_numpy().shape
```




$\displaystyle \left( 1000000, \  4\right)$




```python
logit_model = sm.Logit(y, sm.add_constant(x_df[[t_col] + x_cols])).fit()
print("Est weights, logistic regression:")
print_results(np.array(logit_model.params)[1:])

print("Average marginal effect, logistic regression:")
print_results(logit_model.get_margeff().margeff)
```

    Optimization terminated successfully.
             Current function value: 0.654005
             Iterations 5
    Est weights, logistic regression:
    ['-0.130', '0.641', '0.104', '-0.533', '0.358']
    Average marginal effect, logistic regression:
    ['-0.030', '0.148', '0.024', '-0.123', '0.083']



```python

```
