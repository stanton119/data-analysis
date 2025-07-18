# Mixture of Experts for Multi-Task Learning

This project explores mixture of experts models applied to multi-task learning, with a focus on comparing different architectures across varying levels of task correlation. The implementation is based on the paper ["Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) (Ma et al., 2018).

## Project Overview

Multi-task learning aims to improve model performance by training on multiple related tasks simultaneously. This project implements and compares four key architectures:

1. **Single model per task**: Independent models for each task
2. **Shared bottom layer**: Hard parameter sharing across tasks
3. **Mixture of Experts (OMoE)**: Single gating network routing to multiple experts
4. **Multi-gate Mixture of Experts (MMoE)**: Task-specific gating networks

The project includes:
- Implementation of all four model architectures
- Data loaders for the UCI Census Income dataset
- Synthetic data generators with controllable task correlation
- Experiment framework to compare models across correlation levels
- MLflow integration for experiment tracking and visualization


Tasks:
1. reproduce the paper - https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
2. apply to OPM
3. compare with single model baselines
   1. NN + GBM
4. synthesise experiment data, with tasks with different correlation levels


## Multi-Task Learning Approaches

Here is a summary of different approaches for multi-task learning:

### 1. Single model per task (Traditional approach)
*   **Reference:** Traditional machine learning approach, predating multi-task learning
*   **Description:** Train a separate model for each task independently. Each model is optimized for its specific task without sharing any parameters with other tasks.
*   **Pros:** 
    - Simple to implement and maintain
    - No interference between tasks
    - Can use task-specific architectures optimized for each task
    - Easy to debug and interpret
*   **Cons:** 
    - No knowledge sharing between tasks
    - Requires more parameters overall
    - Cannot leverage correlations between related tasks
    - Higher computational and memory requirements for deployment
*   **Latency/Speed:** Very low latency per model. This is the fastest method for individual task inference, but requires running multiple models for all tasks.

### 2. Shared bottom layer (Hard parameter sharing) [1997]
*   **Reference:** Caruana, R. "Multitask Learning." Machine Learning, 28(1), 41-75 (1997)
*   **Description:** A single shared network for the lower layers that branches into task-specific output layers. All tasks share the same feature extraction layers but have separate task-specific heads.
*   **Pros:**
    - Parameter efficient - reduces overall model size
    - Enables knowledge transfer between related tasks
    - Helps prevent overfitting through implicit regularization
    - Single model deployment for multiple tasks
*   **Cons:**
    - Negative transfer can occur when tasks are unrelated
    - Difficult to balance learning across tasks with different complexities
    - Training can be dominated by easier tasks or tasks with more data
    - Finding optimal shared representation is challenging
*   **Latency/Speed:** Medium latency. Faster than separate models but slower than single-task models due to the shared computation.

### 3. OMoE (One-gate Mixture of Experts) [1991/2017]
*   **Reference:** Jacobs, R. A., et al. "Adaptive Mixtures of Local Experts." Neural Computation (1991); Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR (2017)
*   **Description:** Uses a single gating network to route inputs to different expert networks, with each expert specializing in different aspects of the input space. All tasks share the same gating mechanism.
*   **Pros:**
    - More flexible than hard parameter sharing
    - Can handle tasks with different relationships
    - Experts can specialize in different input patterns
    - More parameter efficient than separate models
*   **Cons:**
    - Single gate may not optimally route for all tasks
    - Still potential for negative transfer
    - More complex to train than shared bottom models
    - Requires careful tuning of expert capacity
*   **Latency/Speed:** Medium to high latency, depending on the number of experts and implementation details.

### 4. MMoE (Multi-gate Mixture of Experts) [2018]
*   **Reference:** Ma, J., et al. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." KDD (2018)
*   **Description:** Extends OMoE by using task-specific gating networks, allowing each task to utilize experts differently. Each task has its own gating network to determine the importance of each expert for that specific task.
*   **Pros:**
    - Task-specific routing provides more flexibility
    - Better handles tasks with different relationships
    - Reduces negative transfer between unrelated tasks
    - Can model complex task relationships
*   **Cons:**
    - More parameters due to multiple gates
    - More complex to implement and train
    - Requires careful balancing of shared vs. task-specific components
    - May need more data to train effectively
*   **Latency/Speed:** Higher latency compared to other approaches due to multiple gating networks and expert computations.

### 5. Soft parameter sharing [2017]
*   **Reference:** Duong, L., et al. "Multi-task Learning over Graph Structures." AAAI (2015); Misra, I., et al. "Cross-stitch Networks for Multi-task Learning." CVPR (2016); Yang, Y. and Hospedales, T. "Trace Norm Regularised Deep Multi-Task Learning." ICLR Workshop (2017)
*   **Description:** Each task has its own model with separate parameters, but the distance between the parameters of the models is regularized to encourage similarity.
*   **Pros:**
    - More flexible than hard parameter sharing
    - Allows task-specific adaptations while encouraging similarity
    - Can better handle tasks with different but related structures
    - Easier to balance task-specific needs
*   **Cons:**
    - More parameters than hard sharing approaches
    - Requires careful tuning of regularization strength
    - More complex optimization process
    - Higher memory requirements
*   **Latency/Speed:** Similar to single models per task, as each task has its own model, but with additional regularization overhead during training.

### 6. Progressive Neural Networks [2016]
*   **Reference:** Rusu, A. A., et al. "Progressive Neural Networks." arXiv:1606.04671 (2016)
*   **Description:** Start with a model for the first task, then add lateral connections from this model to new columns for each new task. Previous task parameters are frozen when training new tasks.
*   **Pros:**
    - Eliminates catastrophic forgetting
    - Explicit knowledge transfer between tasks
    - Maintains performance on all tasks
    - Can leverage pre-trained models effectively
*   **Cons:**
    - Model size grows linearly with the number of tasks
    - Increased inference time for later tasks
    - Complex architecture that's difficult to scale
    - Higher memory requirements
*   **Latency/Speed:** Latency increases with each additional task, as later tasks require computation through multiple columns.

### 7. Cross-stitch Networks [2016]
*   **Reference:** Misra, I., et al. "Cross-stitch Networks for Multi-task Learning." CVPR (2016)
*   **Description:** Starts with task-specific networks but allows learning optimal combination of shared and task-specific representations at each layer through cross-stitch units.
*   **Pros:**
    - Learns optimal sharing at each layer
    - Flexible architecture that adapts to task relationships
    - Can discover which layers benefit most from sharing
    - Balance between shared and task-specific features
*   **Cons:**
    - Additional parameters for cross-stitch units
    - More complex to implement and train
    - Requires careful initialization
    - May need more data to learn effective cross-connections
*   **Latency/Speed:** Medium to high latency, depending on the network size and number of cross-stitch units.

### 8. Adversarial Multi-task Learning [2017]
*   **Reference:** Liu, P., et al. "Adversarial Multi-task Learning for Text Classification." ACL (2017)
*   **Description:** Uses adversarial training to encourage task-invariant shared representations while maintaining task-specific information in private encoders.
*   **Pros:**
    - Better separation of shared and task-specific features
    - Can reduce negative transfer
    - More robust shared representations
    - Improved performance on diverse tasks
*   **Cons:**
    - More complex training procedure with adversarial components
    - Harder to optimize and balance different loss terms
    - Requires careful tuning of adversarial strength
    - May be unstable during training
*   **Latency/Speed:** Higher training complexity but inference speed comparable to shared bottom models.

## Datasets

This project uses two types of datasets:

### 1. UCI Census Income Dataset

The [UCI Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income) is used for multi-task learning with two tasks:
- Income prediction (binary classification: >50K or ≤50K)
- Marital status prediction (binary classification: married or not married)

### 2. Synthetic Datasets

The project includes functions to generate synthetic datasets with controllable task correlation:

- **Simple synthetic data**: Linear combination of features with controlled correlation between tasks
- **MMoE synthetic data**: Replicates the synthetic data generation process from the MMoE paper, using sinusoidal components to create non-linear relationships

## Project Structure

```
.
├── data/                               # Directory for storing raw or processed data.
│   └── census_income/                  # UCI Census Income dataset files.
├── notebooks/                          # Directory for Jupyter notebooks used for exploration and analysis.
├── pyproject.toml                      # Project configuration file for build system and dependencies.
├── README.md                           # This file, providing an overview of the project.
├── src/                                # Source code directory.
│   ├── __init__.py                     # Makes `src` a Python package.
│   ├── data_sources.py                 # Functions to load and create datasets (UCI Census, synthetic data).
│   ├── torch_datasets.py               # PyTorch dataset classes and dataloader utilities.
│   ├── models.py                       # PyTorch model implementations for all architectures.
│   └── train.py                        # Training script with experiment framework and MLflow integration.
└── uv.lock                             # Lock file for `uv` package manager, ensuring reproducible environments.
```

## Setup (WRONG)

1. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -e .
   ```

3. Download the UCI Census Income dataset and place it in the `data/census_income/` directory:
   ```bash
   mkdir -p data/census_income
   cd data/census_income
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
   cd ../..
   ```

## Usage

### Training Individual Models

To train a specific model on a dataset:

```bash
python src/train.py --model_name <MODEL_NAME> --dataset <DATASET> --num_epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE>
```

Replace `<MODEL_NAME>` with one of the following:
- `SingleTaskModel`: Independent model for a single task
- `SharedBottomModel`: Hard parameter sharing model
- `MixtureOfExperts`: One-gate Mixture of Experts (OMoE)
- `MultiGateMixtureOfExperts`: Multi-gate Mixture of Experts (MMoE)

Replace `<DATASET>` with one of:
- `synthetic`: Generated synthetic data with controllable correlation
- `uci_census`: UCI Census Income dataset

Example:
```bash
uv run python src/train.py --model_name SharedBottomModel --dataset uci_census --num_epochs 50 --batch_size 64
```

For SingleTaskModel, you need to specify the target task:
```bash
uv run python src/train.py --model_name SingleTaskModel --target_task income_binary --dataset uci_census
```

### Running the Correlation Experiment

To compare all models across different correlation levels:

```bash
uv run python src/train.py --run_correlation_experiment --num_epochs 30
```

This will:
1. Generate synthetic datasets with varying task correlations (0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5)
2. Train all model types on each dataset
3. Log results to MLflow for comparison

### Viewing Results

To view the experiment results in MLflow:

```bash
uv run mlflow ui
```

## References

1. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) (Ma et al., 2018)
   - Key findings:
     - Shared bottom layers perform better with higher correlation between tasks
     - As correlation between tasks reduces, OMoE models perform worse
     - MMoE models have a different gating function per task and are more robust

2. [Multitask Learning](https://link.springer.com/article/10.1023/A:1007379606734) (Caruana, 1997)
   - Pioneered the concept of hard parameter sharing for multi-task learning
   - Demonstrated how learning related tasks simultaneously can improve generalization

3. [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) (Jacobs et al., 1991)
   - Original formulation of the mixture of experts approach
   - Introduced the concept of gating networks to combine expert outputs

4. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
   - Modern implementation of mixture of experts for large-scale models
   - Introduced techniques for scaling MoE to billions of parameters

5. [Cross-stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539) (Misra et al., 2016)
   - Proposed learning optimal combination of task-specific features
   - Introduced cross-stitch units to allow flexible sharing between tasks

6. [Progressive Neural Networks](https://arxiv.org/abs/1606.04671) (Rusu et al., 2016)
   - Introduced a method to transfer knowledge between tasks while avoiding catastrophic forgetting
   - Proposed lateral connections between task-specific columns

7. [Adversarial Multi-task Learning for Text Classification](https://aclanthology.org/P17-1001/) (Liu et al., 2017)
   - Applied adversarial training to multi-task learning
   - Improved separation between shared and task-specific features

8. [Trace Norm Regularised Deep Multi-Task Learning](https://openreview.net/forum?id=rknkBhdrIB) (Yang & Hospedales, 2017)
   - Proposed soft parameter sharing through trace norm regularization
   - Demonstrated improved performance on diverse tasks

9. [UCI Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)
   - Used for multi-task learning with income prediction and marital status prediction

## Future Work

1. Implement the remaining multi-task learning approaches:
   - Soft parameter sharing
   - Progressive Neural Networks
   - Cross-stitch Networks
   - Adversarial Multi-task Learning

2. Extend the experiment framework to:
   - Compare performance on more real-world datasets
   - Analyze the impact of task weights on training dynamics
   - Visualize expert specialization in MoE and MMoE models

3. Optimize model architectures:
   - Implement expert capacity constraints
   - Explore different expert and gate network architectures
   - Add regularization techniques to improve generalization
