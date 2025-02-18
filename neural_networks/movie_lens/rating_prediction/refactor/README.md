# ml-project/ml-project/README.md

# ML Project

This project is a machine learning framework that integrates MLflow for tracking experiments and managing models. It includes various components for data processing, model training, evaluation, and exploratory data analysis.

Run MLFlow UI as:
```
uv run mlflow ui --backend-store-uri experiments
```

To clear deleted runs:
```
uv run mlflow gc
```

```
export UV_PROJECT_ENVIRONMENT=/Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/.venv
```

## Usage
```bash
uv run python src/train.py --config configs/nn_colab_filter_non_linear.yaml
```

## Insights summary
1. Need intercept on customers and items to control for bias.

## Todo

1. negative sampling
   1. use lack of rating as a negative event - people dont watch movies by random, they select ones they are interested in
   2. positive rating = 1, low rating or no rating = 0
   3. balance negative samples against positive

Resources:
1. https://developers.google.com/machine-learning/recommendation/dnn/softmax

## Project Structure

```
ml-project
├── data
│   ├── raw                # Raw data files
│   └── processed          # Processed data files ready for modeling
├── models
│   ├── __init__.py       # Initializes the models package
│   ├── resnet.py         # ResNet model class
│   ├── mlp.py            # MLP model class
│   └── transformer.py     # Transformer model class
├── notebooks
│   └── exploratory_data_analysis.ipynb  # Jupyter notebook for EDA
├── src
│   ├── __init__.py       # Initializes the src package
│   ├── data_preprocessing.py  # Data preprocessing functions
│   ├── train.py          # Training script with MLflow logging
│   ├── evaluate.py       # Evaluation functions with MLflow logging
│   └── utils.py          # Utility functions
├── experiments
│   └── mlruns            # Directory for MLflow experiment results
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── setup.py              # Setup script for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the exploratory data analysis notebook:
   - Open `notebooks/exploratory_data_analysis.ipynb` in Jupyter Notebook.

UPdating the project depedencies
```
uv add ...
```


```
uv run python src/train.py
```


## Usage

- Use the `src/train.py` script to train your model. It includes MLflow logging for tracking metrics and parameters.
- Evaluate your model using the `src/evaluate.py` script, which also logs evaluation metrics with MLflow.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.