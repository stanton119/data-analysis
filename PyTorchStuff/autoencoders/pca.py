# %%
from pathlib import Path
import torch
import torchvision

# transforms
# prepare transforms standard to MNIST
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]  # , torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)

# data
mnist_train_data = torchvision.datasets.MNIST(
    Path() / "data", train=True, download=True, transform=transform
)
mnist_train = torch.utils.data.DataLoader(mnist_train_data, batch_size=64)

# %%
# inspect first batch
dataiter = iter(mnist_train)
images, labels = dataiter.next()

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(images[0, 0])
ax.set_title(str(labels[0].numpy()))

# %%
# PCA on images
# convert to 1D


# small dataset, so can load it all
dataloader = torch.utils.data.DataLoader(
    dataset=mnist_train_data, batch_size=len(mnist_train_data)
)
images_all, labels_all = next(iter(dataloader))

images_flat = images_all[:, 0].reshape(-1, 784).numpy()
images_flat.shape

# %%
# fit PCA
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.metrics


component_range = np.array(list(range(1, 11)) + [50, 100, 200, 500])

print(f"training components: {component_range.max()}")
pca = sklearn.decomposition.PCA(n_components=component_range.max())
images_flat_hat = pca.inverse_transform(pca.fit_transform(images_flat))


# estimate reconstruction loss
def transform_truncated(pca, X, n_components):
    # forwards
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


mse_train_pca = []
for n in component_range:
    print(f"mse: {n}")
    images_flat_hat = inv_forward_transform(pca, X=images_flat, n_components=n)
    _loss = sklearn.metrics.mean_squared_error(images_flat_hat, images_flat)
    mse_train_pca.append(_loss)

# %% PCA results
# explained variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_)
)
ax.set_title("Explained variance")
ax.set_xlabel("No. components")
ax.set_ylabel("% Variance")

# reconstruction mse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(component_range, mse_train_pca)
ax.set_title("Reconstruction error")
ax.set_xlabel("No. components")
ax.set_ylabel("MSE")

# %%
# Show some results
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
