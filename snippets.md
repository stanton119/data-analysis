# Snippets

## Standard imports
```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl

plt.style.use("seaborn-v0_8-whitegrid")
pl.Config.set_fmt_str_lengths(30)
```

## IO
Loading from S3
For datasets that are small enough to save to disk:
```
!aws s3 sync "s3://bucket_name/path/to/data/" "../data/"
```
then import locally via pandas/polars:
```python
import pyarrow.parquet as pq

df = pq.read_table("../data/set1", partitioning="hive", filters=[('partition_date','>=','2023-02-01'), ('partition_date','<=','2023-02-10')])
df_pl = pl.from_arrow(df)
df_pl
```
is faster (and cheaper) than loading from s3 directly.

## Plotting
Create quick matplotlib figure:
```python
fig, ax = plt.subplots(figsize=(10, 6))
```

Style matplotlib
```python
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
```

Rotate matplotlib axis labels:
```python
ax.tick_params(axis="x", labelrotation=90)
```
Horizontal/vertical line:
```python
ax.axhline(y=0, linestyle="--", color="k", alpha=0.5)
```

## Docker
```
CMD ["val"]
ENTRYPOINT ["python3", "script.py"]
```
Becomes
```
python3 script.py val
```

## Sklearn
Customer transformer:
```python
class CustomTransformer(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    def __init__(self):
        self.latent_space = None

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        return df.drop(columns=["col"])

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return ["List", "of", "column", "names"]
```

To output Pandas DataFrames use:
```python
transformer = transformer.set_output(transform="pandas")
```

Column transformer for Pandas DataFrames
```python
estimator = sklearn.pipeline.make_pipeline(
    sklearn.compose.ColumnTransformer(
        [
            (
                "col1_trans",
                sklearn.pipeline.make_pipeline(
                    sklearn.decomposition.PCA(n_components=5),
                ),
                ["col1"],
            ),
            (
                "col2_trans",
                sklearn.pipeline.make_pipeline(
                    sklearn.decomposition.PCA(n_components=10),
                ),
                ["col2"],
            ),
        ]
    ),
    sklearn.decomposition.PCA(n_components=10),
    sklearn.ensemble.HistGradientBoostingClassifier(**model_kws)
)
```

## Statsmodels
Linear regression:
```
import statsmodels.api as sm

sm_results = sm.OLS(y_train, x_train).fit()
sm_results.summary()
```
Get coefficients and standard deviations:
```
sm_results.params
np.sqrt(np.diag(sm_results.normalized_cov_params))
```


## PyTorch
Dataloader from numpy array:
```python
dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=False
)
```

To initialise the output layer bias:
```python
self.classifier.bias.data.fill_(output_bias)
```

### Dataset and Dataloaders
The [dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) converts to tensors when called.
```
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
batch = next(iter(dataloader))
```


## Polars
Polars increase string display:
```python
pl.Config.set_fmt_str_lengths(30)
```

Load partitioned parquet:
```python
df = pl.read_parquet(
    "path/to/dataset/",
    use_pyarrow=True,
    pyarrow_options=dict(
        partitioning="hive",
        filters=[
            ("partition_date", ">=", "2022-11-01"),
            ("partition_date", "<=", "2022-11-10"),
            ("partition_2", "=", "val1"),
        ],
    ),
)
```

How to apply a common set of statements to a dataframe via a function:
```python
df = pl.DataFrame(
    {
        "col_a": ["a", "b", "a", "b", "c"],
        "col_b": [1, 2, 1, 3, 3],
        "col_c": [5, 4, 3, 2, 1],
    }
)
df.groupby("col_a").agg(
    [
        pl.count().alias("count"),
        pl.col("col_b").sum().alias("col_b_sum"),
        pl.col("col_b").mean().alias("col_b_mean"),
        pl.col("col_c").sum().alias("col_c_sum"),
        pl.col("col_c").mean().alias("col_c_mean"),
    ]
)
```
We can define the columns in a function and use that within a groupby aggregatin or select statement:
```python
def agg_data():
    return [
        pl.count().alias("count"),
        pl.col("col_b").sum().alias("col_b_sum"),
        pl.col("col_b").mean().alias("col_b_mean"),
        pl.col("col_c").sum().alias("col_c_sum"),
        pl.col("col_c").mean().alias("col_c_mean"),
    ]


df.groupby("col_a").agg(agg_data())
df.select(agg_data())
```

## Debugging

Profiling within Notebooks [ref](https://stackoverflow.com/questions/44734297/how-to-profile-python-3-5-code-line-by-line-in-jupyter-notebook-5):
* install `line_profiler` via `pip` or `conda`
* add the extension to the notebook `%load_ext line_profiler`
* create a function to profile, e.g. `prof_function`
* run profile profiler `%lprun -f prof_function prof_function()`

Memory profile within Notebooks:
* `pip install memory_profiler`
* `%load_ext memory_profiler`
* `%memit prof_function()` - where `prof_function` is defined as an import from a file

### Logging

Default logging setup within a module
```python
import logging
logger = logging.getLogger(__name__)
...
logger.info("Message")
```

Default logging setup with a notebook calling modules
```python
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING)
```

Change logging level on another logger
```python
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.WARNING)
```

### Exception handling
```python
import traceback

try:
    1/0
except Exception as e:
    traceback.print_exc()
```
or via logging:
```python
import logging

try:
    1/0
except Exception:
    logging.exception("An exception was thrown!")
```

### Timing execution
```
import timeit
t1 = timeit.default_timer()
logger.info(f"Time taken: {timeit.default_timer() - t1}s")
```

## Data analysis
* Check for missing: https://github.com/ResidentMario/missingno
* General profiling: ydata-profiling

## Environment variables
```bash
pip install python-dotenv
```

`.env` file as:
```
key=value
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv('key'))
```

## Argparsers

```python
parser = argparse.ArgumentParser()
parser.add_argument("--param", type=int)
parser.add_argument("--param2", type=str, default="default_str")
parser.add_argument(
    "--bool_arg", type=bool, default=False, action=argparse.BooleanOptionalAction
)
args = parser.parse_args()
args.param...:
```

## Decorators
```python
import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        print("do something pre function")
        result = func(*args, **kwargs)
        print("do something post function")
        return result
    return wrapper_decorator

@decorator
def test_fun(a:int=0)->int:
    return a+1

test_fun(1)
```

## Other
Run bash without getting interrupted:
```bash
nohup sh shell_script.sh &
```

Check memory usage:
```bash
top
```
Sort by memory: `shift+m`.
Change units: `shift+e`

Check disk space:
```bash
df -h
```