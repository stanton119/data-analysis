# Bimonial regression alternatives to Logistic regression

If we have a very large dataset it can be computationally expensive to train a logistic regression model.
If we have categorical features or can convert features into categories we can summiarise the data and fit a binomial regression model instead, which are more data efficient.

Here we we show their equivalence and how they scale.


First lets create a data generating function


```python
from typing import Tuple
import numpy as np
import polars as pl


def generate_data(
    n_samples: int = 1000,
    n_features: int = 4,
    seed: int = None,
) -> Tuple[pl.DataFrame, np.array]:
    rand = np.random.default_rng(seed)

    # binary covariates
    covariates = rand.integers(0, 2, size=(n_samples, n_features))

    weights = rand.normal(size=(n_features, 1))
    outcome_logit = np.dot(covariates, weights) + rand.normal(
        scale=0.1, size=(n_samples, 1)
    )

    def inv_logit(p):
        return np.exp(p) / (1 + np.exp(p))

    outcome_prob = inv_logit(outcome_logit)
    outcome = rand.binomial(n=1, p=outcome_prob)

    df = pl.concat(
        [
            pl.DataFrame(outcome, schema=["y"]),
            pl.DataFrame(outcome_prob, schema=["y_prob"]),
            pl.DataFrame(covariates, schema=[f"x_{idx}" for idx in range(n_features)]),
        ],
        how="horizontal",
    )

    return df, weights


df, weights = generate_data(n_samples=1000, n_features=2, seed=0)
print("Data sample:")
display(df.head())

print("weights:")
print(weights.flatten())
```

    Data sample:



<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>y</th><th>y_prob</th><th>x_0</th><th>x_1</th></tr><tr><td>i64</td><td>f64</td><td>i64</td><td>i64</td></tr></thead><tbody><tr><td>0</td><td>0.768634</td><td>1</td><td>1</td></tr><tr><td>0</td><td>0.539129</td><td>1</td><td>0</td></tr><tr><td>0</td><td>0.464949</td><td>0</td><td>0</td></tr><tr><td>1</td><td>0.433964</td><td>0</td><td>0</td></tr><tr><td>1</td><td>0.708297</td><td>0</td><td>1</td></tr></tbody></table></div>


    weights:
    [0.08365798 0.8965755 ]


Lets fit a logistic regression model first. We predict the outcome column `y` from the `x` columns and we retrieve estimated weights that are close to the true weights:


```python
import statsmodels.api as sm

logit_model = sm.Logit(
    df["y"].to_pandas(), df.drop(columns=["y", "y_prob"]).to_pandas()
)
logit_model_results = logit_model.fit()
logit_model_results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.643229
             Iterations 5





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>  <td>  1000</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   998</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 10 Aug 2023</td> <th>  Pseudo R-squ.:     </th>  <td>0.03417</td> 
</tr>
<tr>
  <th>Time:</th>                <td>10:41:17</td>     <th>  Log-Likelihood:    </th> <td> -643.23</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -665.99</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>1.512e-11</td>
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>      <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x_0</th> <td>    0.0825</td> <td>    0.105</td> <td>    0.786</td> <td> 0.432</td> <td>   -0.123</td> <td>    0.288</td>
</tr>
<tr>
  <th>x_1</th> <td>    0.8720</td> <td>    0.109</td> <td>    8.003</td> <td> 0.000</td> <td>    0.658</td> <td>    1.086</td>
</tr>
</table>



Lets fit with a Binomial model. First we need to group the data into counts of success and failure against each combination of the input categories:


```python
def group_data(df: pl.DataFrame):
    x_cols = [col for col in df.columns if "x_" in col]
    return df.groupby(x_cols).agg(
        [
            pl.col("y").sum().alias("success"),
            (pl.count() - pl.col("y").sum()).alias("failure"),
        ]
    )


df_group = group_data(df)
df_group
```




<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (4, 4)</small><table border="1" class="dataframe"><thead><tr><th>x_0</th><th>x_1</th><th>success</th><th>failure</th></tr><tr><td>i64</td><td>i64</td><td>i64</td><td>i64</td></tr></thead><tbody><tr><td>0</td><td>0</td><td>102</td><td>115</td></tr><tr><td>1</td><td>0</td><td>138</td><td>118</td></tr><tr><td>0</td><td>1</td><td>193</td><td>74</td></tr><tr><td>1</td><td>1</td><td>183</td><td>77</td></tr></tbody></table></div>



Now we can fit a Binomial regression model. We get identical results to the logistic regression model, including confidence intervals.


```python
binom_model = sm.GLM(
    df_group[["success", "failure"]].to_pandas(),
    df_group.drop(columns=["success", "failure"]).to_pandas(),
    family=sm.families.Binomial(),
)
binom_model_results = binom_model.fit()
binom_model_results.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>   <td>['success', 'failure']</td> <th>  No. Observations:  </th>  <td>     4</td> 
</tr>
<tr>
  <th>Model:</th>                     <td>GLM</td>          <th>  Df Residuals:      </th>  <td>     2</td> 
</tr>
<tr>
  <th>Model Family:</th>           <td>Binomial</td>        <th>  Df Model:          </th>  <td>     1</td> 
</tr>
<tr>
  <th>Link Function:</th>            <td>Logit</td>         <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                   <td>IRLS</td>          <th>  Log-Likelihood:    </th> <td> -12.717</td>
</tr>
<tr>
  <th>Date:</th>               <td>Thu, 10 Aug 2023</td>    <th>  Deviance:          </th> <td>  1.9577</td>
</tr>
<tr>
  <th>Time:</th>                   <td>10:41:17</td>        <th>  Pearson chi2:      </th>  <td>  1.96</td> 
</tr>
<tr>
  <th>No. Iterations:</th>             <td>4</td>           <th>  Pseudo R-squ. (CS):</th>  <td> 1.000</td> 
</tr>
<tr>
  <th>Covariance Type:</th>        <td>nonrobust</td>       <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>      <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x_0</th> <td>    0.0825</td> <td>    0.105</td> <td>    0.786</td> <td> 0.432</td> <td>   -0.123</td> <td>    0.288</td>
</tr>
<tr>
  <th>x_1</th> <td>    0.8720</td> <td>    0.109</td> <td>    8.003</td> <td> 0.000</td> <td>    0.658</td> <td>    1.086</td>
</tr>
</table>



## Fitting time/data scaling

Now that we've shown their equivalence how does each model scale with data size?

We can test the fitting time on different number of samples to find out. We will not include the time for grouping the data as this can commonly be done via external tools such as data warehouses.


```python
import timeit
import tqdm


times = []
n_features = 10
n_trials = 10
for n_samples in tqdm.tqdm([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]):
    for _trial in range(n_trials):
        _n_samples = int(n_samples)
        df, weights = generate_data(
            n_samples=_n_samples, n_features=10, seed=_n_samples + _trial
        )

        if _n_samples < 1e6:
            t1 = timeit.default_timer()
            logit_model = sm.Logit(
                df["y"].to_pandas(), df.drop(columns=["y", "y_prob"]).to_pandas()
            )
            logit_model_results = logit_model.fit(disp=0)
            times.append(["logit", _n_samples, timeit.default_timer() - t1])

        df_group = group_data(df=df)
        t1 = timeit.default_timer()
        binom_model = sm.GLM(
            df_group[["success", "failure"]].to_pandas(),
            df_group.drop(columns=["success", "failure"]).to_pandas(),
            family=sm.families.Binomial(),
        )
        binom_model_results = binom_model.fit()
        times.append(["binomial", _n_samples, timeit.default_timer() - t1])

times = pl.DataFrame(times, schema=["model_type", "n_samples", "fit_time"])
```

      0%|          | 0/11 [00:00<?, ?it/s]

    /Users/stantoon/miniconda3/envs/project_env/lib/python3.10/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    100%|██████████| 11/11 [00:33<00:00,  3.08s/it]


Plotting the results show that the Logistic regression model scales exponentially whilst the Binomial is approximately flat.
The shaded region represents the confidence interval of the mean fit time.


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=times, x="n_samples", y="fit_time", hue="model_type", ax=ax, marker="x"
)
ax.set(xscale="log", ylabel="Fit time (s)", title="Fit time against data length")
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_90987/2954676713.py:4: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      plt.style.use("seaborn-whitegrid")
    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_90987/2954676713.py:11: UserWarning: Matplotlib is currently using module://matplotlib_inline.backend_inline, which is a non-GUI backend, so cannot show the figure.
      fig.show()



    
![png](binomial_regression_files/binomial_regression_12_1.png)
    


The data size passed to the Binomial model always has the same upper group which is the number of possible groups in the data. If we have $n$ features we have $2^n$ possible groups. The counts for each group simply increase when our data size is larger, so we don't expect any change in fitting time. Therefore the Binomial model can scale very well to large problems if we have categorical features. If we don't have categorical features this won't apply, but we can always approximate continuous features with binary categories, such as through various sklearn transformers ([link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html), [link2](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)). What's convenient here is that it allows us to do the heavy lifting and aggregation on a data warehouse and output a small dataset for modelling. Therefore we are not limited by the memory of our device used for building models.

Note: there are alternative approaches for logistic regression to handle large data, such as fitting on a sample, though this will bias the confidence intervals to be larger than reality.
