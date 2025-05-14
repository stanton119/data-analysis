# NN inner product of embeddings

Collaborative filter:
Movies and customer embeddings.
Predict the rates as inner product of the two embeddings.


References:
1. http://d2l.ai/chapter_recommender-systems/mf.html


```python
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

import sys
from pathlib import Path

sys.path.append(str(Path().absolute().parent))

import utilities
```

### Loading and transforming MovieLens


```python
ratings_df, user_id_mapping, movie_id_mapping = utilities.load_torch_data()
ratings_df
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (2_544_549, 6)</small><table border="1" class="dataframe"><thead><tr><th>userId</th><th>movieId</th><th>rating</th><th>timestamp</th><th>userIdMapped</th><th>movieIdMapped</th></tr><tr><td>i64</td><td>i64</td><td>f64</td><td>i64</td><td>u32</td><td>u32</td></tr></thead><tbody><tr><td>1</td><td>5952</td><td>4.0</td><td>1147868053</td><td>0</td><td>46</td></tr><tr><td>1</td><td>296</td><td>5.0</td><td>1147880044</td><td>0</td><td>7</td></tr><tr><td>2</td><td>5952</td><td>5.0</td><td>1141415528</td><td>1</td><td>46</td></tr><tr><td>2</td><td>150</td><td>4.0</td><td>1141415790</td><td>1</td><td>5</td></tr><tr><td>2</td><td>3578</td><td>5.0</td><td>1141415803</td><td>1</td><td>42</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>162541</td><td>1136</td><td>4.5</td><td>1240953505</td><td>157480</td><td>25</td></tr><tr><td>162541</td><td>296</td><td>3.5</td><td>1240953510</td><td>157480</td><td>7</td></tr><tr><td>162541</td><td>1197</td><td>4.5</td><td>1240953534</td><td>157480</td><td>27</td></tr><tr><td>162541</td><td>593</td><td>4.0</td><td>1240953576</td><td>157480</td><td>21</td></tr><tr><td>162541</td><td>47</td><td>4.5</td><td>1240953606</td><td>157480</td><td>2</td></tr></tbody></table></div>



Make into torch datasets


```python
dataloader_train, dataloader_val, dataloader_test = utilities.get_data_loaders(
    ratings_df=ratings_df  # , sample=int(1e5)
)
```

MLflow logging


```python
from pytorch_lightning.loggers import MLFlowLogger
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mlf_logger = MLFlowLogger(
    experiment_name="movie_lens_rating_prediction",
    tracking_uri="mlruns",
    run_name="nn_inner_" + timestamp,
)
```

### Model definition

Fit a model to learn embeddings


```python
import pytorch_lightning as pyl
import torch
import torch.nn as nn


class MovieEmbeddingModel(pyl.LightningModule):
    def __init__(
        self, n_users, n_movies, embedding_dim=10, learning_rate: float = 5e-3
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.output = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.max_rating = 5.0
        self.min_rating = 0.5
        self.output_bias = torch.logit(
            torch.tensor((3.9 - self.min_rating) / (self.max_rating - self.min_rating))
        )
        self.output.bias.data.fill_(self.output_bias)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        output = self.output((user_embeds * movie_embeds).sum(dim=1).view(-1, 1))
        rating = (
            self.sigmoid(output) * (self.max_rating - self.min_rating) + self.min_rating
        )
        return rating.flatten()

    def training_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```

Check it works with a small sample


```python
n_users = ratings_df["userIdMapped"].unique().count()
n_movies = ratings_df["movieIdMapped"].unique().count()
batch = next(iter(dataloader_train))
model = MovieEmbeddingModel(n_users, n_movies)
model(batch[0], batch[1])
```




    tensor([4.4032, 3.4078, 3.0105,  ..., 3.5740, 4.3763, 1.4483],
           grad_fn=<ViewBackward0>)



Add early stopping and enforce a minimum of 6 epochs (it takes a few epochs to start improving over a mean baseline)


```python
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints", save_top_k=1, monitor="val_loss"
)


class EarlyStoppingWithMinEpochs(Callback):
    def __init__(self, min_epochs, **kwargs):
        super().__init__()
        self.min_epochs = min_epochs
        self.early_stopping = EarlyStopping(**kwargs)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.min_epochs - 1:
            self.early_stopping.on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.early_stopping.on_train_end(trainer, pl_module)


early_stop_callback = EarlyStoppingWithMinEpochs(
    min_epochs=6, monitor="val_loss", patience=2, mode="min"
)
```

Train the model

Using an embedding size of 5.


```python
model = MovieEmbeddingModel(n_users, n_movies, embedding_dim=5)
trainer = pyl.Trainer(
    max_epochs=40,
    logger=mlf_logger,
    log_every_n_steps=1,
    callbacks=[early_stop_callback, checkpoint_callback],
)
trainer.fit(model, dataloader_train, dataloader_val)
```

    GPU available: True (mps), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    /Users/rich/Developer/Github/VariousDataAnalysis/.venv/lib/python3.12/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:654: Checkpoint directory /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/checkpoints exists and is not empty.
    
      | Name            | Type      | Params | Mode 
    ------------------------------------------------------
    0 | user_embedding  | Embedding | 787 K  | train
    1 | movie_embedding | Embedding | 250    | train
    2 | output          | Linear    | 2      | train
    3 | sigmoid         | Sigmoid   | 0      | train
    ------------------------------------------------------
    787 K     Trainable params
    0         Non-trainable params
    787 K     Total params
    3.151     Total estimated model params size (MB)
    4         Modules in train mode
    0         Modules in eval mode


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]

    /Users/rich/Developer/Github/VariousDataAnalysis/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


                                                                               

    /Users/rich/Developer/Github/VariousDataAnalysis/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Epoch 12: 100%|██████████| 435/435 [00:17<00:00, 25.22it/s, v_num=f9d7, train_loss_step=0.499, val_loss_step=0.694, val_loss_epoch=0.745, train_loss_epoch=0.498]


Evaluate test set performance


```python
trainer.test(model, dataloader_test)
```

    /Users/rich/Developer/Github/VariousDataAnalysis/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Testing DataLoader 0: 100%|██████████| 63/63 [00:01<00:00, 38.67it/s]
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
         test_loss_epoch        0.7545304894447327
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────





    [{'test_loss_epoch': 0.7545304894447327}]



Store model


```python
mlf_logger.experiment.log_artifact(
    mlf_logger.run_id, checkpoint_callback.best_model_path
)
```

Plotting train/val epoch loss


```python
plot_df = utilities.training_logs_to_df(run_id=mlf_logger.run_id)
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=plot_df, x="epoch", y="loss", hue="dataset", ax=ax)
fig.show()
```

    /var/folders/ky/4qby95090jbbq38_mh94x72r0000gn/T/ipykernel_44162/2428333827.py:4: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](nn_inner_sigmoid_files/nn_inner_sigmoid_22_1.png)
    


### Check prediction results

We will use a baseline prediction from mean ratings to sanity check


```python
train_data, val_data, test_data = utilities.split_train_test(ratings_df=ratings_df)
train_data = train_data.with_columns(pl.col("rating").mean().alias("baseline"))

preds = model(
    torch.tensor(train_data["userIdMapped"]), torch.tensor(train_data["movieIdMapped"])
)
train_data = train_data.with_columns(pl.lit(preds.detach().numpy()).alias("nn"))

train_data
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1_781_184, 8)</small><table border="1" class="dataframe"><thead><tr><th>userId</th><th>movieId</th><th>rating</th><th>timestamp</th><th>userIdMapped</th><th>movieIdMapped</th><th>baseline</th><th>nn</th></tr><tr><td>i64</td><td>i64</td><td>f64</td><td>i64</td><td>u32</td><td>u32</td><td>f64</td><td>f32</td></tr></thead><tbody><tr><td>5150</td><td>79132</td><td>2.5</td><td>1459431773</td><td>5008</td><td>49</td><td>3.979833</td><td>3.290113</td></tr><tr><td>79646</td><td>150</td><td>3.5</td><td>1554224374</td><td>77175</td><td>5</td><td>3.979833</td><td>3.97892</td></tr><tr><td>63956</td><td>318</td><td>4.0</td><td>948005318</td><td>61981</td><td>8</td><td>3.979833</td><td>4.665577</td></tr><tr><td>66248</td><td>1291</td><td>4.5</td><td>1189775094</td><td>64204</td><td>33</td><td>3.979833</td><td>3.922437</td></tr><tr><td>57849</td><td>527</td><td>4.5</td><td>1322993994</td><td>56060</td><td>16</td><td>3.979833</td><td>4.368731</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>7160</td><td>780</td><td>3.0</td><td>848676694</td><td>6942</td><td>23</td><td>3.979833</td><td>2.874167</td></tr><tr><td>108304</td><td>527</td><td>4.0</td><td>1419181402</td><td>104945</td><td>16</td><td>3.979833</td><td>3.819207</td></tr><tr><td>150532</td><td>380</td><td>3.5</td><td>1181551976</td><td>145835</td><td>13</td><td>3.979833</td><td>3.00745</td></tr><tr><td>142477</td><td>260</td><td>5.0</td><td>832244201</td><td>138046</td><td>6</td><td>3.979833</td><td>4.625381</td></tr><tr><td>141844</td><td>1136</td><td>4.5</td><td>1111183788</td><td>137436</td><td>25</td><td>3.979833</td><td>4.480333</td></tr></tbody></table></div>



We are beating the mean baseline in the training dataset


```python
print(
    "baseline loss: ",
    nn.MSELoss()(
        torch.tensor(train_data["baseline"]), torch.tensor(train_data["rating"])
    ),
)
print(
    "NN loss: ",
    nn.MSELoss()(torch.tensor(train_data["nn"]), torch.tensor(train_data["rating"])),
)
```

    baseline loss:  tensor(0.9037)
    NN loss:  tensor(0.4672)


The histogram of predictions looks reasonable


```python
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=train_data, x="rating", ax=ax)
sns.histplot(data=train_data, x="nn", ax=ax)
fig.show()
```

    /var/folders/ky/4qby95090jbbq38_mh94x72r0000gn/T/ipykernel_44162/2656779169.py:4: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](nn_inner_sigmoid_files/nn_inner_sigmoid_29_1.png)
    

