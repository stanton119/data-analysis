# Trying out PyTorch Lightning
In this post I was trying out PyTorch Lightning to see if it's a library that should be used by default alongside PyTorch.
I will create the same nonlinear probabilistic network from before, but this time using Lightning.
Hence the first few steps are the same as previously shown.

Import stuff:


```python
import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
```

Let's generate some data with non-linearities that would pose some issues for a linear regression solution:


```python
# amount of noise that is added is a function of x
n = 2000
x = np.random.uniform(-10, 10, size=n)
noise_std = np.sin(x * 0.4) + 1
y = -0.5 + 1.3 * x + 3 * np.cos(x * 0.5) + np.random.normal(loc=0, scale=noise_std)

x_train = x[: n // 2]
x_test = x[n // 2 :]
y_train = y[: n // 2]
y_test = y[n // 2 :]

plt.plot(x, y, ".")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Weird looking data")
plt.show()
```


    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_3_0.svg)
    


Next we prep data for PyTorch by converting to Tensors and
creating dataloaders to sort out training batches for us.


```python
from torch.utils.data import TensorDataset, DataLoader

x_train_t = torch.Tensor(x_train[:, np.newaxis])
y_train_t = torch.Tensor(y_train[:, np.newaxis])
x_test_t = torch.Tensor(x_test[:, np.newaxis])
y_test_t = torch.Tensor(y_test[:, np.newaxis])

dataset_train = TensorDataset(x_train_t, y_train_t)
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_test = TensorDataset(x_test_t, y_test_t)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)
```

Next we define various helper functions to help train our models.
In this case pytorch lightning has taken away much of the boiler plate, so
the number of functions required for training is essentially just our loss.


```python
def loss_fn_loglike(y_hat, y):
    negloglik = -y_hat.log_prob(y)
    return torch.mean(negloglik)


def plot_loss(loss_train, loss_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.array(loss_train), label="Train loss")
    ax.plot(np.array(loss_test), label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NegLogLike")
    ax.set_title("Training Overview")
    ax.legend()

    plt.show()


def plot_results(x, y, y_est_mu, y_est_std=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, ".", label="y")
    plt.plot(x, y_est_mu, "-y", label="y_est_mu")
    if y_est_std is not None:
        plt.plot(x, y_est_mu + 2 * y_est_std, "-r", label="mu+2std")
        plt.plot(x, y_est_mu - 2 * y_est_std, "-r", label="mu-2std")
    plt.legend()
    plt.show()


def plot_model_results(model, x, y):
    si = np.argsort(x)
    x = x[si]
    y = y[si]
    y_hat = model(torch.Tensor(x[:, np.newaxis]))
    y_est_mu = y_hat.mean.detach().numpy()
    y_est_std = y_hat.scale.detach().numpy()
    plot_results(x, y, y_est_mu, y_est_std)
```

Helper functions we need for the pytorch implementation are much longer:


```python
def train_loop(x, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    y_hat = model(x)
    loss = loss_fn(y_hat, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test_loop(x, y, model, loss_fn):
    with torch.no_grad():
        y_hat = model(x)
        test_loss = loss_fn(y_hat, y).item()

    return test_loss


def train_loop_dl(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        loss = train_loop(x, y, model, loss_fn, optimizer)
    return loss


def test_loop_dl(dataloader, model, loss_fn):
    size = len(dataloader_train.dataset)
    test_loss = 0

    for x, y in dataloader:
        _test_loss = test_loop(x, y, model, loss_fn)
        test_loss += _test_loss * len(x)

    test_loss /= size
    return test_loss


def train_dl(model, dataloader_train, dataloader_test, loss_fn, optimizer, epochs=20):
    loss_train = []
    loss_test = []
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop_dl(dataloader_train, model, loss_fn, optimizer)
        loss_train.append(test_loop_dl(dataloader_train, model, loss_fn))
        loss_test.append(test_loop_dl(dataloader_test, model, loss_fn))
        # if t+1 % 5 == 0:
        #     print(
        #         f"Epoch {t+1}, train loss: {loss_train[-1]:>7f}, test loss: {loss_test[-1]:>7f}"
        #     )
    return loss_train, loss_test


def train(
    model,
    x_train_t,
    y_train_t,
    x_test_t,
    y_test_t,
    loss_fn,
    optimizer,
    epochs=100,
):
    loss_train = []
    loss_test = []
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(x_train_t, y_train_t, model, loss_fn, optimizer)
        loss_train.append(test_loop(x_train_t, y_train_t, model, loss_fn))
        loss_test.append(test_loop(x_test_t, y_test_t, model, loss_fn))
        # print(
        #     f"train loss: {loss_train[-1]:>7f}, test loss: {loss_test[-1]:>7f}"
        # )
    return loss_train, loss_test
```

## PyTorch approach
In pytorch we define our model as a class and train it as follows:


```python
class DeepNormalModel(torch.nn.Module):
    def __init__(self, n_inputs: int = 1, n_hidden: int = 10):
        super().__init__()

        self.hidden = torch.nn.Linear(n_inputs, n_hidden)
        self.mean_linear = torch.nn.Linear(n_hidden, 1)
        self.scale_linear = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        outputs = self.hidden(x)
        # outputs = torch.relu(outputs)
        outputs = torch.sigmoid(outputs)

        mean = self.mean_linear(outputs)
        scale = torch.nn.functional.softplus(self.scale_linear(outputs))

        return torch.distributions.Normal(mean, scale)
```


```python
model_dnm = DeepNormalModel(1)

learning_rate = 0.05
optimizer = torch.optim.Adam(model_dnm.parameters(), lr=learning_rate)

loss_train, loss_test = train_dl(
    model_dnm,
    dataloader_train,
    dataloader_test,
    loss_fn_loglike,
    optimizer,
    epochs=100,
)

plot_loss(loss_train, loss_test)
plot_model_results(model_dnm, x_train, y_train)
```


    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_12_0.svg)
    



    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_12_1.svg)
    


The training code is hidden inside the helper functions.

## PyTorch Lightning approach
In PyTorch Lightning we subclass the `LightningModule` and bring in a couple of the above PyTorch helpers into the class definition.
The training code becomes shorter as the Lightning Trainer class does most of the work for us...


```python
import pytorch_lightning as pl


class DeepNormalModelPL(pl.LightningModule):
    def __init__(self, n_inputs: int = 1, n_hidden: int = 10, learning_rate=0.05):
        super().__init__()

        self.learning_rate = learning_rate
        self.hidden = torch.nn.Linear(n_inputs, n_hidden)
        self.mean_linear = torch.nn.Linear(n_hidden, 1)
        self.scale_linear = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        outputs = self.hidden(x)
        # outputs = torch.relu(outputs)
        outputs = torch.sigmoid(outputs)

        mean = self.mean_linear(outputs)
        scale = torch.nn.functional.softplus(self.scale_linear(outputs))

        return torch.distributions.Normal(mean, scale)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_fn_loglike(y_hat, y)
        return loss


model_lightning = DeepNormalModelPL()
trainer = pl.Trainer(max_epochs=200, progress_bar_refresh_rate=20)
trainer.fit(model_lightning, dataloader_train, dataloader_test)

plot_model_results(model_lightning, x_train, y_train)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    /Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: you passed in a val_dataloader but have no validation_step. Skipping validation loop
      warnings.warn(*args, **kwargs)
    
      | Name         | Type   | Params
    ----------------------------------------
    0 | hidden       | Linear | 20    
    1 | mean_linear  | Linear | 11    
    2 | scale_linear | Linear | 11    
    ----------------------------------------
    42        Trainable params
    0         Non-trainable params
    42        Total params
    0.000     Total estimated model params size (MB)
    Epoch 3:   0%|          | 0/16 [00:00<?, ?it/s, loss=3.46, v_num=0]/Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      warnings.warn(*args, **kwargs)
    Epoch 199: 100%|██████████| 16/16 [00:00<00:00, 352.62it/s, loss=1.03, v_num=0]



    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_14_1.svg)
    


The trainer object is fairly extensive and gives us a lot of extras for free, some of which are explored below.

### Tensorboard
Lightning by defaults allows us to can display the loss in tensorboard:


```python
%load_ext tensorboard
%tensorboard --logdir lightning_logs/
```


    Reusing TensorBoard on port 6006 (pid 42170), started 1:18:31 ago. (Use '!kill 42170' to kill it.)


### Fit with early stopping
To introduce early stopping we add a callback to the trainer object.
We need to add a `validation_step` which logs the validation loss in order to use it with early stopping.
As such we redefine the model class, we do that here by inheriting the previous class:


```python
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class DeepNormalModelPLES(DeepNormalModelPL):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_fn_loglike(y_hat, y)
        self.log("val_loss", loss)
        return loss


model_lightning = DeepNormalModelPLES()
trainer = pl.Trainer(
    max_epochs=200,
    callbacks=[EarlyStopping(monitor="val_loss")],
    progress_bar_refresh_rate=20,
)
trainer.fit(model_lightning, dataloader_train, dataloader_test)

plot_model_results(model_lightning, x_train, y_train)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    
      | Name         | Type   | Params
    ----------------------------------------
    0 | hidden       | Linear | 20    
    1 | mean_linear  | Linear | 11    
    2 | scale_linear | Linear | 11    
    ----------------------------------------
    42        Trainable params
    0         Non-trainable params
    42        Total params
    0.000     Total estimated model params size (MB)
    Validation sanity check:   0%|          | 0/2 [00:00<?, ?it/s]/Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: Your val_dataloader has `shuffle=True`, it is best practice to turn this off for validation and test dataloaders.
      warnings.warn(*args, **kwargs)
    /Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      warnings.warn(*args, **kwargs)
    Epoch 0:   0%|          | 0/32 [00:00<?, ?it/s] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 0: 100%|██████████| 32/32 [00:00<00:00, 266.12it/s, loss=16.7, v_num=1]
    Epoch 1:   0%|          | 0/32 [00:00<?, ?it/s, loss=16.7, v_num=1]/Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      warnings.warn(*args, **kwargs)
    
    Validating: 0it [00:00, ?it/s][A
    Epoch 1: 100%|██████████| 32/32 [00:00<00:00, 284.45it/s, loss=3.8, v_num=1] 
    Epoch 2:   0%|          | 0/32 [00:00<?, ?it/s, loss=3.8, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 2: 100%|██████████| 32/32 [00:00<00:00, 360.78it/s, loss=3.28, v_num=1]
    Epoch 3:   0%|          | 0/32 [00:00<?, ?it/s, loss=3.28, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 3: 100%|██████████| 32/32 [00:00<00:00, 362.45it/s, loss=3.14, v_num=1]
    Epoch 4:   0%|          | 0/32 [00:00<?, ?it/s, loss=3.14, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 4: 100%|██████████| 32/32 [00:00<00:00, 383.09it/s, loss=3.04, v_num=1]
    Epoch 5:   0%|          | 0/32 [00:00<?, ?it/s, loss=3.04, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 5: 100%|██████████| 32/32 [00:00<00:00, 406.81it/s, loss=3, v_num=1]
    Epoch 6:   0%|          | 0/32 [00:00<?, ?it/s, loss=3, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 6: 100%|██████████| 32/32 [00:00<00:00, 333.49it/s, loss=2.92, v_num=1]
    Epoch 7:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.92, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 7: 100%|██████████| 32/32 [00:00<00:00, 385.59it/s, loss=2.86, v_num=1]
    Epoch 8:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.86, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 8: 100%|██████████| 32/32 [00:00<00:00, 360.02it/s, loss=2.83, v_num=1]
    Epoch 9:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.83, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 9: 100%|██████████| 32/32 [00:00<00:00, 343.45it/s, loss=2.77, v_num=1]
    Epoch 10:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.77, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 10: 100%|██████████| 32/32 [00:00<00:00, 400.57it/s, loss=2.73, v_num=1]
    Epoch 11:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.73, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 11: 100%|██████████| 32/32 [00:00<00:00, 419.53it/s, loss=2.67, v_num=1]
    Epoch 12:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.67, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 12: 100%|██████████| 32/32 [00:00<00:00, 440.97it/s, loss=2.62, v_num=1]
    Epoch 13:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.62, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 13: 100%|██████████| 32/32 [00:00<00:00, 437.74it/s, loss=2.56, v_num=1]
    Epoch 14:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.56, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 14: 100%|██████████| 32/32 [00:00<00:00, 455.77it/s, loss=2.5, v_num=1]
    Epoch 15:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.5, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 15: 100%|██████████| 32/32 [00:00<00:00, 442.24it/s, loss=2.44, v_num=1]
    Epoch 16:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.44, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 16: 100%|██████████| 32/32 [00:00<00:00, 418.80it/s, loss=2.36, v_num=1]
    Epoch 17:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.36, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 17: 100%|██████████| 32/32 [00:00<00:00, 437.46it/s, loss=2.31, v_num=1]
    Epoch 18:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.31, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 18: 100%|██████████| 32/32 [00:00<00:00, 373.15it/s, loss=2.25, v_num=1]
    Epoch 19:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.25, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 19: 100%|██████████| 32/32 [00:00<00:00, 326.78it/s, loss=2.21, v_num=1]
    Epoch 20:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.21, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 20: 100%|██████████| 32/32 [00:00<00:00, 415.31it/s, loss=2.17, v_num=1]
    Epoch 21:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.17, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 21: 100%|██████████| 32/32 [00:00<00:00, 361.21it/s, loss=2.12, v_num=1]
    Epoch 22:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.12, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 22: 100%|██████████| 32/32 [00:00<00:00, 430.19it/s, loss=2.08, v_num=1]
    Epoch 23:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.08, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 23: 100%|██████████| 32/32 [00:00<00:00, 399.83it/s, loss=2.06, v_num=1]
    Epoch 24:   0%|          | 0/32 [00:00<?, ?it/s, loss=2.06, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 24: 100%|██████████| 32/32 [00:00<00:00, 399.78it/s, loss=2, v_num=1]
    Epoch 25:   0%|          | 0/32 [00:00<?, ?it/s, loss=2, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 25: 100%|██████████| 32/32 [00:00<00:00, 353.78it/s, loss=1.96, v_num=1]
    Epoch 26:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.96, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 26: 100%|██████████| 32/32 [00:00<00:00, 416.80it/s, loss=1.94, v_num=1]
    Epoch 27:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.94, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 27: 100%|██████████| 32/32 [00:00<00:00, 432.86it/s, loss=1.9, v_num=1]
    Epoch 28:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.9, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 28: 100%|██████████| 32/32 [00:00<00:00, 424.47it/s, loss=1.87, v_num=1]
    Epoch 29:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.87, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 29: 100%|██████████| 32/32 [00:00<00:00, 437.68it/s, loss=1.82, v_num=1]
    Epoch 30:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.82, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 30: 100%|██████████| 32/32 [00:00<00:00, 366.12it/s, loss=1.8, v_num=1]
    Epoch 31:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.8, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 31: 100%|██████████| 32/32 [00:00<00:00, 360.04it/s, loss=1.76, v_num=1]
    Epoch 32:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.76, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 32: 100%|██████████| 32/32 [00:00<00:00, 420.46it/s, loss=1.73, v_num=1]
    Epoch 33:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.73, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 33: 100%|██████████| 32/32 [00:00<00:00, 430.64it/s, loss=1.73, v_num=1]
    Epoch 34:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.73, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 34: 100%|██████████| 32/32 [00:00<00:00, 421.05it/s, loss=1.69, v_num=1]
    Epoch 35:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.69, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 35: 100%|██████████| 32/32 [00:00<00:00, 431.34it/s, loss=1.69, v_num=1]
    Epoch 36:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.69, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 36: 100%|██████████| 32/32 [00:00<00:00, 431.77it/s, loss=1.68, v_num=1]
    Epoch 37:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.68, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 37: 100%|██████████| 32/32 [00:00<00:00, 407.38it/s, loss=1.68, v_num=1]
    Epoch 38:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.68, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 38: 100%|██████████| 32/32 [00:00<00:00, 392.62it/s, loss=1.65, v_num=1]
    Epoch 39:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.65, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 39: 100%|██████████| 32/32 [00:00<00:00, 416.92it/s, loss=1.64, v_num=1]
    Epoch 40:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.64, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 40: 100%|██████████| 32/32 [00:00<00:00, 355.97it/s, loss=1.63, v_num=1]
    Epoch 41:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.63, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 41: 100%|██████████| 32/32 [00:00<00:00, 406.88it/s, loss=1.66, v_num=1]
    Epoch 42:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.66, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 42: 100%|██████████| 32/32 [00:00<00:00, 435.02it/s, loss=1.63, v_num=1]
    Epoch 43:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.63, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 43: 100%|██████████| 32/32 [00:00<00:00, 381.70it/s, loss=1.6, v_num=1]
    Epoch 44:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.6, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 44: 100%|██████████| 32/32 [00:00<00:00, 397.26it/s, loss=1.63, v_num=1]
    Epoch 45:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.63, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 45: 100%|██████████| 32/32 [00:00<00:00, 432.90it/s, loss=1.62, v_num=1]
    Epoch 46:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.62, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 46: 100%|██████████| 32/32 [00:00<00:00, 430.90it/s, loss=1.61, v_num=1]
    Epoch 47:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.61, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 47: 100%|██████████| 32/32 [00:00<00:00, 428.56it/s, loss=1.56, v_num=1]
    Epoch 48:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.56, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 48: 100%|██████████| 32/32 [00:00<00:00, 425.92it/s, loss=1.58, v_num=1]
    Epoch 49:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.58, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 49: 100%|██████████| 32/32 [00:00<00:00, 449.94it/s, loss=1.53, v_num=1]
    Epoch 50:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.53, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 50: 100%|██████████| 32/32 [00:00<00:00, 383.68it/s, loss=1.54, v_num=1]
    Epoch 51:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.54, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 51: 100%|██████████| 32/32 [00:00<00:00, 415.41it/s, loss=1.53, v_num=1]
    Epoch 52:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.53, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 52: 100%|██████████| 32/32 [00:00<00:00, 421.32it/s, loss=1.52, v_num=1]
    Epoch 53:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.52, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 53: 100%|██████████| 32/32 [00:00<00:00, 432.49it/s, loss=1.54, v_num=1]
    Epoch 54:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.54, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 54: 100%|██████████| 32/32 [00:00<00:00, 416.66it/s, loss=1.54, v_num=1]
    Epoch 55:   0%|          | 0/32 [00:00<?, ?it/s, loss=1.54, v_num=1]
    Validating: 0it [00:00, ?it/s][A
    Epoch 55: 100%|██████████| 32/32 [00:00<00:00, 429.36it/s, loss=1.5, v_num=1]
    Epoch 55: 100%|██████████| 32/32 [00:00<00:00, 401.71it/s, loss=1.5, v_num=1]



    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_18_1.svg)
    


The model stopped training earlier :)

### Auto learning rate finder
Setting `auto_lr_find=True` in the trainer object sets a more optimal learning rate for the training.
This should generally make learning faster if we don't have much knowledge of the optimal parameter.


```python
model_lightning = DeepNormalModelPLES()
trainer = pl.Trainer(
    max_epochs=200,
    callbacks=[EarlyStopping(monitor="val_loss")],
    auto_lr_find=True,
    progress_bar_refresh_rate=20,
)
trainer.tune(model_lightning, dataloader_train, dataloader_test)
trainer.fit(model_lightning, dataloader_train, dataloader_test)

plot_model_results(model_lightning, x_train, y_train)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    
      | Name         | Type   | Params
    ----------------------------------------
    0 | hidden       | Linear | 20    
    1 | mean_linear  | Linear | 11    
    2 | scale_linear | Linear | 11    
    ----------------------------------------
    42        Trainable params
    0         Non-trainable params
    42        Total params
    0.000     Total estimated model params size (MB)
    /Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: Your val_dataloader has `shuffle=True`, it is best practice to turn this off for validation and test dataloaders.
      warnings.warn(*args, **kwargs)
    /Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      warnings.warn(*args, **kwargs)
    /Users/Rich/Developer/miniconda3/envs/pytorch_env/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      warnings.warn(*args, **kwargs)
    Finding best initial lr:  81%|████████  | 81/100 [00:00<00:00, 255.42it/s]Restored states from the checkpoint file at /Users/Rich/Developer/Github/VariousDataAnalysis/PyTorchStuff/pytorch_lightning/lr_find_temp_model.ckpt
    Learning rate set to 0.09120108393559097
    
      | Name         | Type   | Params
    ----------------------------------------
    0 | hidden       | Linear | 20    
    1 | mean_linear  | Linear | 11    
    2 | scale_linear | Linear | 11    
    ----------------------------------------
    42        Trainable params
    0         Non-trainable params
    42        Total params
    0.000     Total estimated model params size (MB)
    Epoch 1:   0%|          | 0/32 [00:00<00:00, -929.59it/s]   
    Validating: 0it [00:00, ?it/s][A
    Epoch 1: 100%|██████████| 32/32 [00:00<00:00, 276.12it/s, loss=5.65, v_num=2]
    Epoch 2:   0%|          | 0/32 [00:00<00:00, -1997.29it/s, loss=5.65, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 2: 100%|██████████| 32/32 [00:00<00:00, 305.29it/s, loss=3.08, v_num=2] 
    Epoch 3:   0%|          | 0/32 [00:00<00:00, -2736.61it/s, loss=3.08, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 3: 100%|██████████| 32/32 [00:00<00:00, 292.03it/s, loss=3, v_num=2]    
    Epoch 4:   0%|          | 0/32 [00:00<00:00, -1141.20it/s, loss=3, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 4: 100%|██████████| 32/32 [00:00<00:00, 301.23it/s, loss=2.92, v_num=2]
    Epoch 5:   0%|          | 0/32 [00:00<00:00, -987.05it/s, loss=2.92, v_num=2]  
    Validating: 0it [00:00, ?it/s][A
    Epoch 5: 100%|██████████| 32/32 [00:00<00:00, 299.95it/s, loss=2.82, v_num=2]
    Epoch 6:   0%|          | 0/32 [00:00<00:00, -1522.99it/s, loss=2.82, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 6: 100%|██████████| 32/32 [00:00<00:00, 336.70it/s, loss=2.73, v_num=2] 
    Epoch 7:   0%|          | 0/32 [00:00<00:00, -633.68it/s, loss=2.73, v_num=2]  
    Validating: 0it [00:00, ?it/s][A
    Epoch 7: 100%|██████████| 32/32 [00:00<00:00, 336.13it/s, loss=2.63, v_num=2]
    Epoch 8:   0%|          | 0/32 [00:00<00:00, -2923.54it/s, loss=2.63, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 8: 100%|██████████| 32/32 [00:00<00:00, 397.88it/s, loss=2.51, v_num=2] 
    Epoch 9:   0%|          | 0/32 [00:00<00:00, -3846.81it/s, loss=2.51, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 9: 100%|██████████| 32/32 [00:00<00:00, 367.82it/s, loss=2.39, v_num=2] 
    Epoch 10:   0%|          | 0/32 [00:00<00:00, -1771.99it/s, loss=2.39, v_num=2]
    Validating: 0it [00:00, ?it/s][A
    Epoch 10: 100%|██████████| 32/32 [00:00<00:00, 352.21it/s, loss=2.24, v_num=2] 
    Epoch 11:   0%|          | 0/32 [00:00<00:00, -2592.81it/s, loss=2.24, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 11: 100%|██████████| 32/32 [00:00<00:00, 369.37it/s, loss=2.08, v_num=2] 
    Epoch 12:   0%|          | 0/32 [00:00<00:00, -1836.12it/s, loss=2.08, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 12: 100%|██████████| 32/32 [00:00<00:00, 392.09it/s, loss=1.93, v_num=2] 
    Epoch 13:   0%|          | 0/32 [00:00<00:00, -2194.82it/s, loss=1.93, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 13: 100%|██████████| 32/32 [00:00<00:00, 355.90it/s, loss=1.8, v_num=2]  
    Epoch 14:   0%|          | 0/32 [00:00<00:00, -1796.53it/s, loss=1.8, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 14: 100%|██████████| 32/32 [00:00<00:00, 374.48it/s, loss=1.72, v_num=2]
    Epoch 15:   0%|          | 0/32 [00:00<00:00, -2017.46it/s, loss=1.72, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 15: 100%|██████████| 32/32 [00:00<00:00, 360.68it/s, loss=1.66, v_num=2] 
    Epoch 16:   0%|          | 0/32 [00:00<00:00, -1575.42it/s, loss=1.66, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 16: 100%|██████████| 32/32 [00:00<00:00, 377.03it/s, loss=1.63, v_num=2] 
    Epoch 17:   0%|          | 0/32 [00:00<00:00, -2518.60it/s, loss=1.63, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 17: 100%|██████████| 32/32 [00:00<00:00, 347.28it/s, loss=1.59, v_num=2] 
    Epoch 18:   0%|          | 0/32 [00:00<00:00, -1981.56it/s, loss=1.59, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 18: 100%|██████████| 32/32 [00:00<00:00, 366.15it/s, loss=1.56, v_num=2] 
    Epoch 19:   0%|          | 0/32 [00:00<00:00, -1962.09it/s, loss=1.56, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 19: 100%|██████████| 32/32 [00:00<00:00, 395.22it/s, loss=1.5, v_num=2]  
    Epoch 20:   0%|          | 0/32 [00:00<00:00, -1722.04it/s, loss=1.5, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 20: 100%|██████████| 32/32 [00:00<00:00, 373.36it/s, loss=1.47, v_num=2]
    Epoch 21:   0%|          | 0/32 [00:00<00:00, -3592.04it/s, loss=1.47, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 21: 100%|██████████| 32/32 [00:00<00:00, 368.30it/s, loss=1.44, v_num=2] 
    Epoch 22:   0%|          | 0/32 [00:00<00:00, -3131.64it/s, loss=1.44, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 22: 100%|██████████| 32/32 [00:00<00:00, 378.17it/s, loss=1.37, v_num=2] 
    Epoch 23:   0%|          | 0/32 [00:00<00:00, -2440.91it/s, loss=1.37, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 23: 100%|██████████| 32/32 [00:00<00:00, 357.78it/s, loss=1.32, v_num=2] 
    Epoch 24:   0%|          | 0/32 [00:00<00:00, -1977.82it/s, loss=1.32, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 24: 100%|██████████| 32/32 [00:00<00:00, 362.73it/s, loss=1.31, v_num=2] 
    Epoch 25:   0%|          | 0/32 [00:00<00:00, -2804.30it/s, loss=1.31, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 25: 100%|██████████| 32/32 [00:00<00:00, 384.91it/s, loss=1.35, v_num=2] 
    Epoch 26:   0%|          | 0/32 [00:00<00:00, -2130.53it/s, loss=1.35, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 26: 100%|██████████| 32/32 [00:00<00:00, 349.55it/s, loss=1.25, v_num=2] 
    Epoch 27:   0%|          | 0/32 [00:00<00:00, -3529.57it/s, loss=1.25, v_num=2] 
    Validating: 0it [00:00, ?it/s][A
    Epoch 27: 100%|██████████| 32/32 [00:00<00:00, 323.33it/s, loss=1.52, v_num=2] 
    Finding best initial lr: 100%|██████████| 100/100 [00:02<00:00, 33.79it/s]



    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_20_1.svg)
    


The model stopped training even earlier :)

### Reloading models
The `trainer` object saves checkpoints as it goes, so it's easy to reload the model at a later date.
We reload it using the `load_from_checkpoint` which is inherited from the base lightning class
and show that the results are the same as before:


```python
# test checkpoint reload
model_lightning_load = DeepNormalModelPL.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=199-step=3199.ckpt"
)
plot_model_results(model_lightning_load, x_train, y_train)
```


    
![svg](pytorch_lightning_regression_files/pytorch_lightning_regression_22_0.svg)
    


## Conclusion
PyTorch Lightning seems pretty handy! It gives some of the features of Keras which makes the code much neater.
I think I'll be using PyTorch Lightning along side PyTorch going forwards!
