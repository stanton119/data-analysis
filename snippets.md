# Snippets

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
plt.style.use("seaborn-whitegrid")
```

Rotate matplotlib axis labels:
```python
ax.tick_params(axis="x", labelrotation=90)
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
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
```

Change logging level on another logger
```python
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.WARNING)
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

## Other
Run bash without getting interrupted:
```bash
nohup sh shell_script.sh &
```