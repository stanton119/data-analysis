Install as:
```
uv add pytorch-lightning mlflow
```

```
dependencies = [
    "matplotlib>=3.10.0",
    "mlflow>=2.19.0",
    "nbformat>=5.10.4",
    "numpy>=2.2.0",
    "pandas>=2.2.3",
    "plotly>=5.24.1",
    "polars>=1.17.1",
    "pyarrow>=18.1.0",
    "pytorch-lightning>=2.4.0",
    "requests>=2.32.3",
    "scikit-learn>=1.6.0",
    "seaborn>=0.13.2",
    "statsmodels>=0.14.4",
    "torch>=2.5.1",
]
```

Run as:
```
uv run mlflow ui
```

To clear deleted runs:
```
uv run mlflow gc
```

Todo:

1. negative sampling
   1. use lack of rating as a negative event - people dont watch movies by random, they select ones they are interested in
   2. positive rating = 1, low rating or no rating = 0
   3. balance negative samples against positive
