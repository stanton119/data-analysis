# Read me

## Refactor

2. refactor into scripts
   1. train
   2. evaluate
   3. models/
      1. init to have model factory
   4. config.yaml
      1. training runs and parameters
   5. mlflow setup
      1. train - starts run_id, stores artifact and training metrics
      2. evaluate - loads run_id and stores in metadata. evaluation using new run id to allow multiple test runs
      3. same experiment for all dataset sizes within the same project
         1. dataset_name = "small_sample_v1"  # Change to "full_dataset_v1" when using full data
            mlflow.log_param("dataset_name", dataset_name) # or
            mlflow.set_tag("dataset", dataset_name)

python train.py --model resnet --dataset mnist --epochs 5 --lr 0.001
python evaluate.py --dataset mnist --mlflow_model_uri "runs:/<run_id>/model"


my_project/
│── train.py                 # Training script with MLflow
│── evaluate.py              # Evaluation script with MLflow
│── models/
│   ├── __init__.py          # Model registry
│   ├── resnet.py            # ResNet model definition
│   ├── mlp.py               # MLP model definition
│   ├── transformer.py       # Transformer model definition
│── utils/
│   ├── dataset_loader.py    # Function to load datasets
│── configs/
│   ├── resnet.yaml          # Hyperparameters for ResNet
│   ├── mlp.yaml             # Hyperparameters for MLP
│── experiments/             # (Optional) Stores MLflow runs


## Setup
Dependencies are managed through UV and the pyproject.toml.

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

Run MLFlow UI as:
```
uv run mlflow ui
```

To clear deleted runs:
```
uv run mlflow gc
```

## Summary
1. Need intercept on customers and items to control for bias.

## Todo

1. negative sampling
   1. use lack of rating as a negative event - people dont watch movies by random, they select ones they are interested in
   2. positive rating = 1, low rating or no rating = 0
   3. balance negative samples against positive

Resources:
1. https://developers.google.com/machine-learning/recommendation/dnn/softmax