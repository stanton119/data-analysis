# %% [markdown]
# # Image compression - part 1. - PCA
# In this post I will be briefly looking at PCA as a means to compress images.
# Images are a matrix of pixels, where each value corresponds to some brightness.
# Image compression typically involves representing those pixels in fewer dimensions than the original.
#
# PCA is a means of dimensionality reduction commonly used in statistical applications.
# I assume the reader is already familiar with the PCA algorithm and here I will look at applying it to images.
# I may explore further techniques in the future, so this is a preliminary part 1...
#
# We'll do this analysis on the ever popular and omnipresent MNIST dataset.
# First let's download the required dataset.
# %%
from pathlib import Path
import torch
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

mnist_train_data = torchvision.datasets.MNIST(
    Path() / "data", train=True, download=True, transform=transform
)
mnist_train = torch.utils.data.DataLoader(mnist_train_data, batch_size=64)
# %% [markdown]
# We can show some images to check we've got that correct.
# %%
# inspect first batch
dataiter = iter(mnist_train)
images, labels = dataiter.next()

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

fig, ax = plt.subplots(figsize=(8, 16), ncols=4)
for col in range(0, 4):
    ax[col].imshow(images[col, 0])
    ax[col].set_title(str(labels[col].numpy()))

# %%
# build autoencoder with dense layers

import pytorch_lightning as pl


class AutoEncoderDense(pl.LightningModule):
    def __init__(self, n_inputs: int = 1, n_latent: int = 5):
        super().__init__()
        self.train_log = []
        self.n_latent = n_latent

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_latent),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.reshape(-1, 1, 28, 28)

    def configure_optimizers(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,  # weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = torch.nn.MSELoss()(x_hat, x)

        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss


# %%
model_dense = AutoEncoderDense()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model_dense, mnist_train)

# %%
# plot mse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(model_dense.train_log)
ax.set_title("Training error")
ax.set_xlabel("Batches")
ax.set_ylabel("MSE")

# %%
fig, ax = plt.subplots(figsize=(20, 20), ncols=6, nrows=2)
images_hat = model_dense(images)


for col in range(6):
    ax[0, col].imshow(images[col, 0])
    ax[0, col].set_title(str(labels[col].numpy()))

    ax[1, col].imshow(images_hat[col, 0].detach())
    ax[1, col].set_title(str(labels[col].numpy()))

# %%
# train with varying latent space
# training speed doesn't seem to change significantly with latent space size
# save each network
# might be a better way to build a network with space 200 and truncate from there.
# or expand the previous model and transfer learn.
latent_space_dim = [5, 10, 50, 200]
model_path = Path() / "models"
model_path.mkdir(exist_ok=True)

for n_latent in latent_space_dim:
    print(f"training: {n_latent}")
    model_dense = AutoEncoderDense(n_latent=n_latent)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model_dense, mnist_train)
    torch.save(model_dense, model_path / f"dense_{n_latent}.pt")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model_dense.train_log)
    ax.set_title(f"Training error: {n_latent}")
    ax.set_xlabel("Batches")
    ax.set_ylabel("MSE")


# %%
# build autoencoder with cnn layers


# %%
# Compare MSE to PCA of same latent dimension.

# use whole training dataset
dataloader = torch.utils.data.DataLoader(
    dataset=mnist_train_data, batch_size=len(mnist_train_data)
)
images_all, labels_all = next(iter(dataloader))

# dense model error
mse_train_dense = []
for n_latent in latent_space_dim:
    print(f"mse: {n_latent}")
    model_dense = torch.load(model_path / f"dense_{n_latent}.pt")
    images_all_hat = model_dense(images_all)
    _loss = torch.nn.MSELoss()(images_all_hat, images_all)
    mse_train_dense.append(_loss.detach().numpy())

# %%
# get PCA comparison

import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.metrics


# convert to 1D
images_flat = images_all[:, 0].reshape(-1, 784).numpy()
images_flat.shape

print(f"training components: {latent_space_dim[-1]}")
pca = sklearn.decomposition.PCA(n_components=latent_space_dim[-1])
images_flat_hat = pca.inverse_transform(pca.fit_transform(images_flat))


def transform_truncated(pca, X, n_components):
    X = pca._validate_data(X, dtype=[np.float64, np.float32], reset=False)
    if pca.mean_ is not None:
        X = X - pca.mean_
    X_transformed = np.dot(X, pca.components_[:n_components, :].T)
    if pca.whiten:
        X_transformed /= np.sqrt(pca.explained_variance_)
    return X_transformed


def inv_transform(pca, X, n_components):
    return np.dot(X, pca.components_[:n_components, :]) + pca.mean_


def inv_forward_transform(pca, X, n_components):
    return inv_transform(
        pca, transform_truncated(pca, X, n_components), n_components
    )


# get pca mse
mse_train_pca = []
for n_latent in latent_space_dim:
    print(f"mse: {n_latent}")
    images_flat_hat = inv_forward_transform(
        pca, X=images_flat, n_components=n_latent
    )
    _loss = sklearn.metrics.mean_squared_error(images_flat_hat, images_flat)
    mse_train_pca.append(_loss)

# %%

# reconstruction mse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(latent_space_dim, mse_train_dense, label="dense")
ax.plot(latent_space_dim, mse_train_pca, label="pca")
ax.set_title("Reconstruction error")
ax.set_xlabel("Latent space size")
ax.set_ylabel("MSE")


# %%
# Compare images to PCA of same latent dimension.

# %%
# Run same analysis on test set to check for overfitting

# %% [markdown]
# We can show some images at various number of components to see what the images look like
# %%
"""
truth 1, 2, 3, 4...
pca5  1, 2, 3, 4...
pca10 1, 2, 3, 4...
pca50 1, 2, 3, 4...
"""

fig, ax = plt.subplots(figsize=(20, 20), ncols=6, nrows=5)

for row, n_components in enumerate([5, 10, 50, 200]):
    images_hat = inv_forward_transform(
        pca, X=images_flat, n_components=n_components
    ).reshape(-1, 28, 28)

    for col in range(6):
        ax[0, col].imshow(images_all[col, 0])
        ax[0, col].set_title(str(labels_all[col].numpy()))

        ax[row + 1, col].imshow(images_hat[col])
        ax[row + 1, col].set_title(str(labels_all[col].numpy()))
# %% [markdown]
# As the components increase the digits look visably clearer.
# Some digits look worse than others.
# We can plot the MSE against the digit to see which are hard to construct:
# %%
# MSE against label
loss_label = []
for row, n_components in enumerate([5, 10, 50, 200]):
    images_flat_hat = inv_forward_transform(
        pca, X=images_flat, n_components=n_components
    )

    _loss_label = []
    for label in range(0, 10):
        filt = labels_all == label
        _loss = sklearn.metrics.mean_squared_error(
            images_flat_hat[filt], images_flat[filt]
        )
        _loss_label.append(_loss)
    loss_label.append(_loss_label)

df_loss = pd.DataFrame(
    loss_label, index=[5, 10, 50, 200], columns=range(0, 10)
).transpose()
fig, ax = plt.subplots(figsize=(10, 6))
df_loss.plot(ax=ax)
ax.set_title("Reconstruction error by digit number")
ax.set_xlabel("Digit label")
ax.set_ylabel("MSE")
# %% [markdown]
# As expected increasing the number of components we use makes all the digits improve.
# Looks like '1' is quite easy compared to the rest, whilst '2' is harder.
