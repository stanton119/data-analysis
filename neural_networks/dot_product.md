# Representing the dot/inner product through an MLP

Dot products are commonly used in recommender systems to combine user and item embeddings.

The dot product of two vectors (A and B) is the sum of the products of their corresponding elements:
$$
\mathbf{a} \cdot \mathbf{b} = \langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i=1}^{n} a_i b_i
$$


Dense layers in a neural network work on the weighted sum of inputs. They dont directly capture interactions between features. We can concat two vector and push through a dense layer. The output of a dense layer is given as:
$$
g([\mathbf{a} , \mathbf{b}]) = \sigma(\mathbf{W} \cdot ([\mathbf{a} , \mathbf{b}]) + \mathbf{c})
$$

Where $[,]$ is the concatenation operation and $\mathbf{W}$ and $\mathbf{c}$ are the weight matrix and bias of the dense layer, respectively.

In this notebook, we will explore how to represent the dot product using a neural network

## TODO
1. Create a model training framework
2. Deep vs shallow
3. data loader to create random dot product samples
4. overfit small sample, with high epoch count
   1. is learning rate correct?

## Setup
```
uv add pytorch-lightning
uv run mlflow ui --backend-store-uri experiments
```

## Model definition


```python
from typing import List
import pytorch_lightning as pyl
import torch
import torch.nn as nn


class Model(pyl.LightningModule):
    def __init__(
        self,
        dimension: int,
        layer_sizes: List[int] = [32, 16],
        learning_rate: float = 5e-3,
    ):
        super().__init__()

        if layer_sizes is None:
            self.output = nn.Linear(dimension * 2, 1)
        else:
            layer_sizes = [dimension * 2] + layer_sizes + [1]
            layers = []
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(nn.ReLU())
            self.output = nn.Sequential(*layers, nn.Linear(layer_sizes[-1], 1))

        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, a, b):
        return self.output(torch.cat([a, b], dim=1))

    def training_step(self, batch, batch_idx):
        a, b, y = batch
        y_hat = self(a, b)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a, b, y = batch
        y_hat = self(a, b)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        a, b, y = batch
        y_hat = self(a, b)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```

## Generate data
We will make a dataset that creates a random pair of vectors and their dot product.
As the data is randomly generate each batch we don't need to be concerned with overfitting.
Therefore we have only a train dataloader.


```python
from torch.utils.data import Dataset, DataLoader
import torch


class RandomVectorDataset(Dataset):
    def __init__(self, dimension: int, num_samples: int, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        self.dimension = dimension
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = torch.randn(self.dimension)
        b = torch.randn(self.dimension)
        y = torch.dot(a, b)
        return a, b, y


dataset = RandomVectorDataset(dimension=2, num_samples=10)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

for batch in dataloader:
    a, b, y = batch
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"Dot Product: {y}")
    break  # Remove this break to iterate through the entire dataset
```

    a: tensor([[ 0.4740,  0.1978],
            [-2.4661,  0.3623],
            [ 0.3930,  0.4327],
            [ 0.6688, -0.7077]])
    b: tensor([[ 1.1561,  0.3965],
            [ 0.3765, -0.1808],
            [-1.3627,  1.3564],
            [-0.3267, -0.2788]])
    Dot Product: tensor([ 0.6265, -0.9941,  0.0513, -0.0212])


## Train models


```python
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger


def setup_experiment():
    mlf_logger = MLFlowLogger(experiment_name="dot_product", tracking_uri="experiments")
    return mlf_logger


def train(model, dataloader):
    early_stopping = EarlyStopping(monitor="train_loss", patience=2, mode="min")
    mlf_logger = setup_experiment()
    trainer = pyl.Trainer(
        max_epochs=30,
        logger=mlf_logger,
        log_every_n_steps=1,
        callbacks=early_stopping,
    )
    trainer.fit(model, dataloader)

    return trainer.test(model, dataloader)


def train_loop(dimension: int, layers: List[int]):
    # Create dataset and dataloader
    num_samples = 1_000_000
    num_samples = 1_000
    batch_size = 1024
    dataset = RandomVectorDataset(dimension=dimension, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model(dimension=dimension, layer_sizes=layers, learning_rate=1e-3)
    return train(model=model, dataloader=dataloader)
```


```python
# loss = train_loop(dimension=4, layers=[4, 2])
# loss = train_loop(dimension=4, layers=[4])
# loss = train_loop(dimension=4, layers=[16, 8, 4, 2])
# loss = train_loop(dimension=2, layers=[4, 2])
# loss = train_loop(dimension=2, layers=[4])
# loss = train_loop(dimension=2, layers=[16, 8, 4, 2])
# loss = train_loop(dimension=2, layers=[32, 32, 16, 16, 8, 8, 4, 4, 2, 2])
loss = train_loop(dimension=2, layers=[4] * 30)
# loss = train_loop(dimension=2, layers=None)
```

    GPU available: True (mps), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    
      | Name   | Type       | Params | Mode 
    ----------------------------------------------
    0 | output | Sequential | 607    | train
    ----------------------------------------------
    607       Trainable params
    0         Non-trainable params
    607       Total params
    0.002     Total estimated model params size (MB)
    64        Modules in train mode
    0         Modules in eval mode


    Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 14.49it/s, v_num=d9c5, train_loss_step=2.260, train_loss_epoch=2.260]
    Testing DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 121.78it/s]
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         test_loss_epoch        2.4767603874206543
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


## Results


```python
# load all mlflow final epoch loss
# plot against architecture and dimension
```

### Conclusion


