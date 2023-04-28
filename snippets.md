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

## Other