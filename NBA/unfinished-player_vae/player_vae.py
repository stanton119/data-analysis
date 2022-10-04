# %% [markdown]
# # How do player styles relate to each other
#
# In this post we look at build a variational auto-encoder to find the similarities between different players.
#
# Data comes from [basketball-reference.com](https://www.basketball-reference.com)
#
# Steps:
# Find each players best season, based on max points/game?
# Collect normal + advanced stats for high dimension stats
# Build VAE with small latent space
# Explore the latent space:
#   * Label a subset of players
#   * Find anomolies, by players away from the centroids, and by poor recreations
#
# %% [markdown]
# ## Data preparation
# We collect the data using `pandas.read_html` before transforming and cleaning.
#
# First import some stuff:
# %%
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import basketball_reference as br

# %% Get season summary data
df_season_summary = br.get_season_summary(season=2020)
df_season_summary

# %%
# scale input data and convert to numpy arrays
# need to avoid MSE weighting towards large variables
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

scaler = preprocessing.StandardScaler()
col_names = list(df_season_summary.iloc[:, 5:].columns)
x = scaler.fit_transform(df_season_summary.iloc[:, 5:].to_numpy())
x_train, x_test = train_test_split(x, test_size=0.3)

# %% Find best latent dimension
# Baseline from PCA
# MSE vs dimension
from sklearn import decomposition, metrics


def explained_var(x, x_hat):
    return (
        np.diag(np.cov(x_hat.transpose())).sum()
        / np.diag(np.cov(x.transpose())).sum()
    )


x_var = {}
mse_train_pca = {}
mse_test_pca = {}

for n in range(1, x_train.shape[1] + 1):
    print(f"training: {n}")
    pca = decomposition.PCA(n_components=n)
    x_train_hat = pca.inverse_transform(pca.fit_transform(x_train))
    x_test_hat = pca.inverse_transform(pca.transform(x_test))

    x_var[n] = explained_var(x_train, x_train_hat)

    mse_train_pca[n] = metrics.mean_squared_error(x_train_hat, x_train)
    mse_test_pca[n] = metrics.mean_squared_error(x_test_hat, x_test)

# %% PCA results
# explained variance
df = pd.DataFrame.from_dict(x_var, orient="index")
ax = df.plot(title="Explained variance", figsize=(10, 6))
ax.set_xlabel("No. components")
ax.set_ylabel("% Variance")

# explained variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    range(1, x_train.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_)
)
ax.set_title("Explained variance")
ax.set_xlabel("No. components")
ax.set_ylabel("% Variance")


# %% PCA weights at 10 components
pca = decomposition.PCA(n_components=10)
pca.fit_transform(x_train)
sns.heatmap(np.abs(pca.components_))
components = pd.DataFrame(data=pca.components_, columns=col_names)
components.abs().sum().sort_values(ascending=False)
sns.heatmap(components.abs())


# %%
# set up Pytorch data objects
from torch.utils.data import TensorDataset, DataLoader

x_train_t = torch.Tensor(x_train)
x_test_t = torch.Tensor(x_test)

dataset_train = TensorDataset(x_train_t)
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

dataset_test = TensorDataset(x_test_t)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)


# %% NN model
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from typing import List


class ModelVAE(pl.LightningModule):
    def __init__(
        self,
        n_inputs: int = 1,
        n_hidden: List[int] = [10, 10],
        n_latent_space: int = 2,
        learning_rate=0.05,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], n_hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[1], n_latent_space),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent_space, n_hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[1], n_hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden[0], n_inputs),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss)
        return loss


# %%
n_latent_space = 5
model_lightning = ModelVAE(
    n_inputs=x_train_t.shape[1], n_latent_space=n_latent_space
)
trainer = pl.Trainer(
    max_epochs=200,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
    auto_lr_find=True,
    progress_bar_refresh_rate=20,
)
trainer.tune(model_lightning, dataloader_train)
trainer.fit(
    model=model_lightning,
    train_dataloader=dataloader_train,
    val_dataloaders=dataloader_test,
)

# %%
import time

x_hat = model_lightning(x_train_t)
for idx in range(20):
    plt.plot(x_train[idx], label="original")
    plt.plot(x_hat[idx].detach().numpy(), label="filtered")
    plt.legend()
    plt.show()
    time.sleep(0.3)

# %%
x_hat = model_lightning(x_train_t)
for idx in range(20):
    plt.plot(scaler.inverse_transform(x_train[idx]), label="original")
    plt.plot(
        scaler.inverse_transform(x_hat[idx].detach().numpy()), label="filtered"
    )
    plt.legend()
    plt.show()
    time.sleep(0.3)

# %% Find latent representation
with torch.no_grad():
    z_hat = model_lightning.encoder(torch.cat([x_train_t, x_test_t]))
z_hat_np = z_hat.detach().numpy()

# sns.scatterplot(x=z_hat_np[:, 0], y=z_hat_np[:, 1], hue=z_hat_np[:, 2])
# sns.scatterplot(x=z_hat_np[:, 0], y=z_hat_np[:, 1])

# plot with hiplot? facebookk many dimension plot?
# plot original data with that would be interesting

# %% Annotate players
df_results = df.copy()
for idx in range(n_latent_space):
    df_results[f"latent_space_{idx}"] = z_hat_np[:, idx]

df_results["Pos"].value_counts()
filt = df_results["Pos"].str.contains("-")
df_results.loc[filt, "Pos"] = df_results.loc[filt, "Pos"].apply(lambda x: x[:2])


# %%
sns.scatterplot(
    data=df_results, x="latent_space_0", y="latent_space_1", hue="Pos"
)
# %% top players
top_players = df_results.sort_values(["MP"], ascending=False)["Player"].head(10)
filt = df_results["Player"].isin(top_players)

sns.scatterplot(
    data=df_results.loc[filt], x="latent_space_0", y="latent_space_1", hue="Pos"
)

# %%
import hiplot as hip

# plot_cols =
hip.Experiment.from_dataframe(df_results.iloc[:, 5:]).display(
    force_full_width=True
)


# %%
df_results.sort_values(["MP"], ascending=False).head(10)
df_results["Player"].head()
# add position as colour
df_season_summary["Pos"].value_counts()

# %% Find best latent dimension
# MSE vs dimension

# MSE in scaled space
mse_train_t = {}
mse_test_t = {}
mse_train_np = {}
mse_test_np = {}


def explained_var(x, x_hat):
    return (
        np.diag(np.cov(x_hat.transpose())).sum()
        / np.diag(np.cov(x.transpose())).sum()
    )


# %% Autoencoder
for n in range(5, 6):
    print(f"training: {n}")
    model_lightning = ModelVAE(
        n_inputs=x_train_t.shape[1], n_latent_space=n, learning_rate=0.01
    )
    trainer = pl.Trainer(
        max_epochs=200,
        # callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        # auto_lr_find=True,
        progress_bar_refresh_rate=20,
    )
    # trainer.tune(model_lightning, dataloader_train)
    trainer.fit(
        model=model_lightning,
        train_dataloader=dataloader_train,
        # val_dataloaders=dataloader_test,
    )

    with torch.no_grad():
        x_train_hat = model_lightning(x_train_t)
        x_test_hat = model_lightning(x_test_t)

    mse_train_t[n] = model_lightning.loss_fn(x_train_hat, x_train_t)
    mse_test_t[n] = model_lightning.loss_fn(x_test_hat, x_test_t)

# %%
explained_var(x_train_t.detach().numpy(), x_train_hat.detach().numpy())


# %% results
# df = pd.DataFrame()
# df["AE_train"] = mse_train.values()
# df["AE_test"] = mse_test.values()
# df["PCA_train"] = mse_train_np.values()
# df["PCA_test"] = mse_test_np.values()


df = pd.concat(
    map(
        lambda x: pd.DataFrame.from_dict(x, orient="index"),
        [mse_train_t, mse_test_t, mse_train_np, mse_test_np],
    ),
    axis=1,
)
df.columns = ["AE_train", "AE_test", "PCA_train", "PCA_test"]


df.plot()

# %%
# implement PCA results in NN?
class ModelPCA(pl.LightningModule):
    def __init__(
        self, n_inputs: int = 1, n_latent_space: int = 2, learning_rate=0.02,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_latent_space),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent_space, n_inputs),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss


# %%
mse_train_lin_t = {}
mse_test_lin_t = {}

for n in range(3, 10):
    print(f"training: {n}")
    model_lightning = ModelPCA(n_inputs=x_train_t.shape[1], n_latent_space=n)
    trainer = pl.Trainer(
        max_epochs=200,
        # auto_lr_find=True,
        progress_bar_refresh_rate=20,
    )
    # trainer.tune(model_lightning, dataloader_train)
    trainer.fit(
        model=model_lightning, train_dataloader=dataloader_train,
    )

    with torch.no_grad():
        x_train_hat = model_lightning(x_train_t)
        x_test_hat = model_lightning(x_test_t)

    mse_train_lin_t[n] = model_lightning.loss_fn(x_train_hat, x_train_t)
    mse_test_lin_t[n] = model_lightning.loss_fn(x_test_hat, x_test_t)
# %%
df = pd.concat(
    map(
        lambda x: pd.DataFrame.from_dict(x, orient="index"),
        [
            mse_train_t,
            mse_test_t,
            mse_train_lin_t,
            mse_test_lin_t,
            mse_train_np,
            mse_test_np,
        ],
    ),
    axis=1,
)
df.columns = [
    "AE_train",
    "AE_test",
    "Lin_train",
    "Lin_test",
    "PCA_train",
    "PCA_test",
]


df.plot()
df[["AE_train", "Lin_train", "PCA_train"]].plot()
df[["AE_test", "Lin_test", "PCA_test"]].plot()

# %%
