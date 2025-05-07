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

Move legend outside:
```python
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

Add annotation:
```python
plt.annotate("Caption", (x_val, y_val), bbox=dict(boxstyle="round", fc="1"))
```

Histograms of multiple columns:
```python
plot_df = df.select(
    pl.col(
        [
            "col1",
            ...
            "colN",
        ]
    )
).melt()
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=plot_df, x="value", hue="variable", stat="probability", common_norm=False, ax=ax)
fig.show()
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

Iterate a dataloader to get predictions:
```python
y_est, y_true = [], []
for idx, batch in enumerate(dataloader):
    _x, _y = batch
    _y_est = model(_x)
    y_est.append(_y_est.detach().numpy())
    y_true.append(_y.detach().numpy())

y_est = np.concatenate(y_est)
y_true = np.concatenate(y_true)
```

### PyTorch Lightning
We can wrap PyTorch models in a Lightning wrapper as:
```python
class TrainModule(pl.LightningModule):
    def __init__(self, model):
        self.model = model
        ...

    def forward(self, x):
        return self.model(x)
```

## Polars
Polars increase string display:
```python
pl.Config.set_fmt_str_lengths(30)
```
Show more rows display:
```python
with pl.Config(set_tbl_rows=50):
    display(pl)
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

Load parquet from S3 (required `fsspec`):
```python
df = pl.read_parquet(
    "s3://bucket/path/file.parquet",
    use_pyarrow=True,
    storage_options=dict(profile="AWS-profile-name"),
    hive_partitioning=True,
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

Run dataframe functions
```python
def fun(df):
    return df.with_columns(pl.len())

df.pipe(fun)
```

Applying a Python function to each row, we can create a struct and use apply:
```python
df.with_columns(
    pl.struct(["col1", "col2"]).apply(
        lambda x: python_function(x["col1"], x["col2"])
    )
)
```

Applying a groupby with a Python function, we need to pass and return a dataframe:
```python
def _python_wrapper(df: pl.DataFrame)->pl.DataFrame:
    value = python_function(
        x=df["col1"],
        y=df["col2"],
    )
    return pl.DataFrame(data={"group_id": df["group_id"].head(1), "value": value})

df = df.groupby("group_id").apply(_python_wrapper)
```

Applying to each group any transform:
```python
for df_group in df.partition_by("group_col"):
    df_group...
```

## Debugging

Profiling within Notebooks using `pyinstrument`:
```python
import pyinstrument
with pyinstrument.profile():
    ...code...
```

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

## Testing
Fixtures:
```python
@pytest.fixture
def fixture_1():
    return {"a":1, "b":2}

def test_1(fixture_1):
    assert f(fixture_1)
```

Parameterise:
```python
@pytest.mark.parametrize(
    "param1, param2",
    [
        ("a", 1),
    ],
)
def test_1(param1, param2):
    assert f(param1) == param2
```

Patch:
```python
from unittest.mock import patch
@patch("module.const", "value")
def test_1():
    assert f()

@patch("module.function_name")
def test_2(function_name):
    function_name.return_value = {"a": 1}
    assert f()
```

## Python setup - UV
Setup project:
```
uv init project-name
uv init # if project folder already exists
uv python install 3.12 # if needed
```

Add to pyproject.toml to make buildable package:
```
[tool.uv]
package = true
```

Add dependency:
```
uv add package
```

Run scripts:
```
uv run python script.py
```

Tools (formatters etc., which don't need the project package installed) are run as:
```
uvx ruff format .
uvx black .
uvx isort .
```

Testing is inside the project venv as it needs the project installed:
```
uv add pytest --dev # add to dev group
uv run pytest
```

Notebook usage:
```
uv add ipykernel --dev
```

## AWS

Get credentials from profile name:
```python
import boto3
session = boto3.session.Session(profile_name=profile_name)
credentials = session.get_credentials().get_frozen_credentials()

storage_options={
    "aws_access_key_id": credentials.access_key,
    "aws_secret_access_key": credentials.secret_key,
    "aws_session_token": credentials.token,
    "aws_region": session.region_name,
}
```

## LaTex
Align equations in a notebook:
```
$$
\begin{align}
x & = y &\\
  & = 1 &\\
  & = 2
\end{align}
$$
```

## Other
Run bash without getting interrupted:
```bash
nohup sh shell_script.sh &
```

Or better through tmux:
```bash
# starts new session
tmux new -s session_name
sh shell_script.sh
#  crtl+b, d to escape the session
tmux ls
tmux attach -t session_name
tmux kill-session -t session_name
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
Check folder size:
```bash
du -sh -- * .[^.]* | sort -h
```