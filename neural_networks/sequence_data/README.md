# Learning over Sequences with Neural Networks

This project explores how to build machine learning models that can effectively learn from a combination of tabular features and sequence data to predict a target variable. We will investigate different neural network architectures for learning over sequences and combining them with tabular inputs.

## Sequence Encoding Approaches

Here is a summary of different approaches to encode sequence data:

### 1. Statistical Aggregation

*   **Description:** This is the simplest and fastest approach. It involves computing a single vector representation of the sequence by calculating statistical properties across the time steps. Common aggregations include:
    *   **Mean:** The average value of all elements in the sequence.
    *   **Max:** The maximum value of all elements in the sequence.
    *   **Sum:** The sum of all elements in the sequence.
*   **Pros:**
    *   Extremely fast and computationally cheap.
    *   No parameters to learn.
    *   Can provide a reasonable baseline.
*   **Cons:**
    *   Completely ignores the order of the elements in the sequence, losing all temporal information.
    *   Can be sensitive to outliers (especially max and sum).
*   **Latency/Speed:** Very low latency. This is the fastest method.

### 2. Fixed Positional Weighting

*   **Description:** In this approach, the weights are not learned but are pre-defined based on the position of the element in the sequence. This is a heuristic method that embeds a specific bias. A common example is applying a **linear decay**, where the most recent elements receive the highest weights.
*   **Pros:**
    *   **No Parameters:** Requires no training and adds no parameters to the model.
    *   **Extremely Simple:** Trivial to implement.
    *   **Encodes Recency Bias:** Useful for tasks where more recent information is known to be more important.
*   **Cons:**
    *   **Not Flexible:** The weighting scheme is a fixed assumption that may not be optimal for the task. It cannot adapt to the data.
*   **Latency/Speed:** Extremely low latency. It's computationally equivalent to a simple weighted average.

### 3. Learned Static Positional Weighting

*   **Description:** In this approach, the model learns a set of static weights, one for each position in the sequence. These weights are then applied uniformly across all sequences to create a weighted sum. Unlike attention, these weights are not dependent on the content of the individual sequence elements during inference; they are fixed once learned.
*   **Pros:**
    *   **Data-Driven:** The model learns the optimal static weights for the task from the data.
    *   **Efficient:** Very fast and computationally light, similar to fixed positional weighting but with learned parameters.
    *   **Simple to Understand:** The learned weights directly indicate the importance of each position.
*   **Cons:**
    *   **Context-Independent:** The weights are static and do not adapt to the specific content of each sequence. This means it cannot capture dynamic relationships within sequences.
*   **Latency/Speed:** Very low latency, only slightly more complex than fixed positional weighting.

### 4. Learned Weighted Average (Simple Attention)

*   **Description:** Instead of treating all sequence elements equally, you let the model *learn* the importance of each position. A small neural network computes a "score" for each element, these scores are passed through a `softmax` function to create weights, and the final representation is a weighted sum.
*   **Pros:**
    *   **Data-Driven:** The model learns the optimal weights for the task from the data itself.
    *   **Efficient:** Very fast and computationally light.
    *   **Interpretability:** You can inspect the learned weights to understand which parts of the sequence the model is focusing on.
*   **Cons:**
    *   **Context-Independent:** The weight for each element is calculated based on that element alone, without considering the context of other elements.
*   **Latency/Speed:** Very low latency, only slightly more complex than statistical aggregation.

### 5. 1D Convolutional Neural Networks (CNNs)

*   **Description:** 1D CNNs apply convolutional filters to the sequence, capturing local patterns or "motifs." By stacking convolutional layers, they can learn hierarchical features. The output is typically passed through a pooling layer (e.g., max-pooling) to produce a fixed-size embedding.
*   **Pros:**
    *   Highly parallelizable, making them much faster than RNNs for both training and inference.
    *   Effective at detecting local patterns in the sequence.
*   **Cons:**
    *   The receptive field is limited by the filter size and network depth, which can make it difficult to capture long-range dependencies.
*   **Latency/Speed:** Medium latency. Faster than RNNs, but slower than simple aggregation.

### 6. Recurrent Neural Networks (RNNs)

*   **Description:** RNNs are designed to process sequential data. They iterate through the sequence, maintaining a hidden state that captures information from previous time steps. The final hidden state, or an aggregation of all hidden states, is used as the sequence embedding.
    *   **LSTM (Long Short-Term Memory):** A type of RNN that uses gates to control the flow of information, making it better at capturing long-range dependencies.
    *   **GRU (Gated Recurrent Unit):** A simplified version of the LSTM with fewer parameters, often performing similarly.
*   **Pros:**
    *   Explicitly designed for sequential data and captures temporal dependencies.
    *   LSTMs and GRUs are effective at learning long-range patterns.
*   **Cons:**
    *   Inherently sequential nature makes them slow to train and run, as they cannot be easily parallelized over the time dimension.
    *   Can still struggle with very long sequences.
*   **Latency/Speed:** High latency, especially for long sequences, due to the recurrent, step-by-step processing.

### 7. Transformer Encoder

*   **Description:** The Transformer architecture uses a self-attention mechanism to weigh the importance of different elements in the sequence. The encoder part of the Transformer can be used to create a powerful sequence embedding.
*   **Pros:**
    *   Excellent at capturing long-range dependencies, as the self-attention mechanism can connect any two positions in the sequence.
    *   Highly parallelizable, similar to CNNs.
    *   Often achieves state-of-the-art performance on many sequence-based tasks.
*   **Cons:**
    *   The self-attention mechanism has a computational complexity that is quadratic with respect to the sequence length (O(n^2)), making it memory and compute-intensive for very long sequences.
    *   Generally requires more data to train effectively than RNNs or CNNs.
*   **Latency/Speed:** Can be faster than RNNs for training due to parallelization, but inference latency can be high for very long sequences due to the quadratic complexity.

## Project Structure

```
.
├── data/                               # Directory for storing raw or processed data.
├── notebooks/                          # Directory for Jupyter notebooks used for exploration and analysis.
├── pyproject.toml                      # Project configuration file for build system and dependencies.
├── README.md                           # This file, providing an overview of the project.
├── src/                                # Source code directory.
│   ├── __init__.py                     # Makes `src` a Python package.
│   ├── data_generator.py               # Contains functions to generate dummy tabular and sequence data for training and testing.
│   ├── dataset.py                      # Defines the `TabularSequenceDataset` for PyTorch and a utility function `create_dataloaders`.
│   ├── models.py                       # Contains PyTorch model definitions, starting with `StatisticalAggregationModel`.
│   └── train.py                        # Script for training the models using PyTorch Lightning.
└── uv.lock                             # Lock file for `uv` package manager, ensuring reproducible environments.
```


## Setup

1.  Create and activate a virtual environment:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    uv pip install -e .
    ```

## Usage

To train a model using the dummy data:

```bash
uv run python src/train.py --model_name <MODEL_NAME> --num_epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE> --num_samples <NUM_SAMPLES> --sequence_length <SEQUENCE_LENGTH> --num_features <NUM_FEATURES>
```

Replace `<MODEL_NAME>` with one of the following:
- `StatisticalAggregationModel`
- `FixedPositionalWeightingModel`
- `LearnedStaticPositionalWeightingModel`
- `LearnedWeightedAverageModel`

Example:
```bash
uv run python src/train.py --model_name StatisticalAggregationModel --num_epochs 10 --batch_size 32
```

Train all models:
```bash
uv run python src/train.py --model_name StatisticalAggregationModel
uv run python src/train.py --model_name FixedPositionalWeightingModel
uv run python src/train.py --model_name LearnedStaticPositionalWeightingModel
uv run python src/train.py --model_name LearnedWeightedAverageModel
```

To view the MLflow UI (after running `src/train.py` at least once):

```bash
uv run mlflow ui
```
Then open your web browser and navigate to `http://localhost:5000` (or the address shown in your terminal).



## Prompts
Main prompts used to create the analysis:

> I want to explore how to create machine learning models that can learn from tabular features and sequences in order to predict a target variable.
> Please create a python module to generate dummy data that I can use to train a model.

> Read the README.md and create a pytorch data loader to take the output of the data_generator.py output.

> Read the README.md. Detail the multiple ways you can encode a sequence with a neural network.
> Include simple solutions like statistical mean/max/sum aggregation.
> Add the summary of the approaches to the README.md
> For each approach add a short descriptions, pro/cons and latency/speed expectation.

> Read the README.md. implement approach 1. Make a python module for 'models' if there isn't one already.
> Create a pytorch model which implements the idea.

> Read the README.md. I want to create a training module which will generate dummy data and use it to train a model from the models module. Use pytorch-lightning for training.

> Read the README.md. Fill in the section on Project Structure. It should list the files in the project folder. For the files in src/ add a one line description of what each one does

> Read the README.md. Update the section on Project Structure by reading through the project directories and list them.

> Read the README.md. I want to track the model training loss to see how each model is converging. How to best track this? MLflow?
