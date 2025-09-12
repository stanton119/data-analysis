# ml-project/ml-project/README.md

# Recsys

This project is a recommendation system using neural collaborative filtering with MLflow for experiment tracking.

## Dataset Processing

The MovieLens dataset is processed as follows:
1. **Binary conversion**: Ratings ≥4.0 become positive (1), others negative (0)
2. **Negative sampling**: For each positive interaction, generates 4 random negative samples (items user hasn't interacted with)
3. **Label encoding**: Users and items are encoded to sequential integers starting from 0
4. **Train/test split**: 80/20 split with shuffling

This creates a balanced binary classification problem for implicit feedback recommendation.

## Usage
```bash
uv run python src/recsys/train.py --config configs/nn_colab_filter_non_linear.yaml
```

Run MLFlow UI as:
```bash
uv run mlflow ui --backend-store-uri experiments
```

To clear deleted runs:
```bash
uv run mlflow gc
```

## Insights summary
1. Need intercept on customers and items to control for bias.
2. Movie Lens 100k
   1. Inner product models perform well. This is a small dataset and is expected.
   2. We dont have any side features to take advantage of here, so pure colab filtering is the only option.
   3. We havent tuned embedding sizes.

## Todo

1. negative sampling
   1. use lack of rating as a negative event - people dont watch movies by random, they select ones they are interested in
   2. positive rating = 1, low rating or no rating = 0
   3. balance negative samples against positive
2. Models
   1. RQ-VAE
   2. Diffusion based recsys
3. Update readme project structure

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