# Word2Vec - Finding similar movies in the MovieLens datset

Can we find an embedding space built on the movielens dataset to compare distance between titles?

Consider each person's viewing history as a series of titles.
Build a model to predict the next title watched based on the ones before it.
Or build a model to predict the middle title based on the ones either side.

Represent the titles as an embedding vector.
Build a dense layer on top of the embedding vector to predict the next title.

Follow an approach similar to Word2Vec.
Instead of treating each word as an entity or token, we use each title.
We treat the vocabulary as the set of titles.

1. Load the movielens dataset
2. Convert each movie title to an integer token
3. Create an embedding layer on the tokens

References:
* https://en.wikipedia.org/wiki/Word2vec

Start by importing stuff:


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl

plt.style.use("seaborn-v0_8-whitegrid")

import sys
from pathlib import Path

sys.path.append(str(Path().absolute().parent))

import utilities
```

## Load data

Load the movielens dataset.


```python
ratings_df = utilities.load_ratings()
ratings_df
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (25_000_095, 4)</small><table border="1" class="dataframe"><thead><tr><th>userId</th><th>movieId</th><th>rating</th><th>timestamp</th></tr><tr><td>i64</td><td>i64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>1</td><td>5952</td><td>4.0</td><td>1147868053</td></tr><tr><td>1</td><td>2012</td><td>2.5</td><td>1147868068</td></tr><tr><td>1</td><td>2011</td><td>2.5</td><td>1147868079</td></tr><tr><td>1</td><td>1653</td><td>4.0</td><td>1147868097</td></tr><tr><td>1</td><td>1250</td><td>4.0</td><td>1147868414</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>162541</td><td>1259</td><td>4.5</td><td>1240953609</td></tr><tr><td>162541</td><td>1266</td><td>5.0</td><td>1240953613</td></tr><tr><td>162541</td><td>1556</td><td>1.0</td><td>1240953650</td></tr><tr><td>162541</td><td>293</td><td>4.0</td><td>1240953789</td></tr><tr><td>162541</td><td>1201</td><td>3.0</td><td>1240953800</td></tr></tbody></table></div>



Each person has rated at least 20 titles.


```python
ratings_df.group_by("userId").count().sort("count", descending=True)
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_44973/2997066704.py:1: DeprecationWarning: `GroupBy.count` is deprecated. It has been renamed to `len`.
      ratings_df.group_by("userId").count().sort("count", descending=True)





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (162_541, 2)</small><table border="1" class="dataframe"><thead><tr><th>userId</th><th>count</th></tr><tr><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>72315</td><td>32202</td></tr><tr><td>80974</td><td>9178</td></tr><tr><td>137293</td><td>8913</td></tr><tr><td>33844</td><td>7919</td></tr><tr><td>20055</td><td>7488</td></tr><tr><td>&hellip;</td><td>&hellip;</td></tr><tr><td>162277</td><td>20</td></tr><tr><td>162304</td><td>20</td></tr><tr><td>162351</td><td>20</td></tr><tr><td>162370</td><td>20</td></tr><tr><td>162371</td><td>20</td></tr></tbody></table></div>



Not all movieIds are sequential, we have 59k IDs with values up to 209k. We will need to tokenize them before training.


```python
print(ratings_df["movieId"].max())
ratings_df.group_by("movieId").count().sort("count", descending=True)
```

    209171


    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_44973/3773860017.py:2: DeprecationWarning: `GroupBy.count` is deprecated. It has been renamed to `len`.
      ratings_df.group_by("movieId").count().sort("count", descending=True)





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (59_047, 2)</small><table border="1" class="dataframe"><thead><tr><th>movieId</th><th>count</th></tr><tr><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>356</td><td>81491</td></tr><tr><td>318</td><td>81482</td></tr><tr><td>296</td><td>79672</td></tr><tr><td>593</td><td>74127</td></tr><tr><td>2571</td><td>72674</td></tr><tr><td>&hellip;</td><td>&hellip;</td></tr><tr><td>203286</td><td>1</td></tr><tr><td>183621</td><td>1</td></tr><tr><td>134186</td><td>1</td></tr><tr><td>102109</td><td>1</td></tr><tr><td>186101</td><td>1</td></tr></tbody></table></div>



We will make a small subset of data for initial building and testing our models

We limit to the top 50 movies and users with at least 20 ratings.


```python
top_movie_ids = utilities.get_most_frequent_movies(ratings_df)
ratings_df = ratings_df.join(top_movie_ids, on="movieId", how="inner")


user_id_counts = (
    ratings_df.group_by("userId").len().filter(pl.col("len") >= 20)[["userId"]]
)
ratings_df = ratings_df.join(user_id_counts, on="userId", how="inner")
ratings_df.shape
```




    (1511948, 4)



## Build model architecture

We will build the embedding using word2vec. There are two forms:

1. Continuous Bag of Words (CBOW):
In CBOW, the model predicts the current word (target word) based on the context words within a fixed window size.

1. Skip-gram:
In Skip-gram, the model predicts context words (surrounding words) given the current word (target word).

CBOW is generally faster to train compared to Skip-gram, especially when using small training datasets.
In our case the words are actually movies.

We will use CBOW for this test as hopefully its faster and more appropriate on a smallish training dataset.
Therefore we need to define a window of entities to train over. We will start with 3, so predict the current entity given the one before and the one after.


### Data preparation

We need to get sequenences of tokens. So we next create tokens from the movieIds as they do not start from 0 and are not sequential.


```python
ratings_df, user_id_mapping, movie_id_mapping = utilities.map_users_and_movies(
    ratings_df
)
ratings_df = ratings_df.rename({"movieIdMapped": "token"})
display(ratings_df.head(4))
```


<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (4, 6)</small><table border="1" class="dataframe"><thead><tr><th>userId</th><th>movieId</th><th>rating</th><th>timestamp</th><th>userIdMapped</th><th>token</th></tr><tr><td>i64</td><td>i64</td><td>f64</td><td>i64</td><td>u32</td><td>u32</td></tr></thead><tbody><tr><td>2</td><td>5952</td><td>5.0</td><td>1141415528</td><td>0</td><td>46</td></tr><tr><td>2</td><td>150</td><td>4.0</td><td>1141415790</td><td>0</td><td>5</td></tr><tr><td>2</td><td>3578</td><td>5.0</td><td>1141415803</td><td>0</td><td>42</td></tr><tr><td>2</td><td>380</td><td>1.0</td><td>1141415808</td><td>0</td><td>13</td></tr></tbody></table></div>


Then for each userId we convert the tokens in to a list.


```python
import tqdm


def get_sequences_from_df(ratings_df: pl.DataFrame):
    sequences = []
    for _user_id in tqdm.tqdm(ratings_df["userId"].unique()):
        sequences.append(
            ratings_df.filter(pl.col("userId") == _user_id)["token"].to_list()
        )
    return sequences
```


```python
sequences = get_sequences_from_df(ratings_df)
print(sequences[0][:10])
```

    100%|██████████| 49276/49276 [00:03<00:00, 14016.76it/s]

    [46, 5, 42, 13, 16, 0, 39, 37, 27, 4]


    


Split sequences randomly to train/test sets


```python
import random


def split_train_test(data, test_ratio=0.2):
    data_copy = data.copy()
    random.shuffle(data_copy)

    split_index = int(len(data_copy) * (1 - test_ratio))
    train_set = data_copy[:split_index]
    test_set = data_copy[split_index:]

    return train_set, test_set


# Example usage:
data = [
    ["sample1", "feature1", "feature2"],
    ["sample2", "feature1", "feature2"],
    ["sample3", "feature1", "feature2"],
    ["sample4", "feature1", "feature2"],
    ["sample5", "feature1", "feature2"],
]

sequences_train, sequences_test = split_train_test(sequences, test_ratio=0.2)
len(sequences_train), len(sequences_test)
```




    (39420, 9856)



We can now create the CBOW dataset from the sequences.
We will use a small embedding size and a window of only 1.

So in a sequence of (w1, w2, w3) we get training tuples as: predict w2 given [w1, w3].

We create a `collate_fn` to collect all the sequences and combine into tensors for training.

```
Sequence Data (e.g., Text, Time Series):
For sequence data, such as text or time series, the shape of a batch is often: (batch_size, sequence_length, input_dim).
batch_size is the number of sequences in the batch.
sequence_length is the length of each sequence.
input_dim represents the dimensionality of each element in the sequence (e.g., word embeddings for text).
```


```python
from torch.utils.data import Dataset, DataLoader
import torch


class CBOWDataset(Dataset):
    def __init__(self, sequences: list[int], window_size: int):
        self.sequences = sequences
        self.window_size = window_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        data = []
        for i in range(len(sequence)):
            target_word = sequence[i]
            context = []
            for j in range(
                max(0, i - self.window_size),
                min(len(sequence), i + self.window_size + 1),
            ):
                if j != i:
                    context.append(sequence[j])
            data.append((context, target_word))
        return data


def collate_fn(batch):
    # join all training examples into single tensor
    data = []
    targets = []
    for batch_item in batch:
        for _item in batch_item[1:-1]:
            data.append(_item[0])
            targets.append(_item[1])

    return torch.tensor(data), torch.tensor(targets).view(-1, 1)
```


```python
vocab_size = ratings_df["movieId"].unique().count()
window_size = 1

# Create dataset and dataloader
dataset_train = CBOWDataset(sequences_train, window_size)
dataloader_train = DataLoader(
    dataset_train, batch_size=64, shuffle=False, collate_fn=collate_fn
)
dataset_test = CBOWDataset(sequences_test, window_size)
dataloader_test = DataLoader(
    dataset_test, batch_size=64, shuffle=False, collate_fn=collate_fn
)
```

The training data is made up of pairs of tokens and the token from the middle. Here is a preview:


```python
dataset_train.__getitem__(0)[:5]
```




    [([26], 8), ([8, 28], 26), ([26, 16], 28), ([28, 6], 16), ([16, 10], 6)]



Following the dataloader we get.


```python
batch = next(iter(dataloader_train))
batch
```




    (tensor([[ 8, 28],
             [26, 16],
             [28,  6],
             ...,
             [ 5, 47],
             [11, 46],
             [47, 45]]),
     tensor([[26],
             [28],
             [16],
             ...,
             [11],
             [47],
             [46]]))



### Model architecture

The input to the model is a one-hot encoded vector representing the context words.
The hidden layer is a projection layer (embedding layer) that converts the one-hot encoded vectors into dense embedding vectors.
The output layer predicts the probability distribution of the target word given the context.
The model is trained to minimize the difference between the predicted probabilities and the actual word (softmax output).


The input to the model is a series of tokens representing the movies.
(These are converted to one-hot encoded vectors. Not needed?)
The tokens are converted into embedding vectors.

The embedding vectors of the context words are averaged to obtain a single context vector.
This context vector represents the overall context of the surrounding words.
The context vector is then passed through a linear transformation followed by a softmax activation function to produce a probability distribution over the entire vocabulary.

We then use a dense layer(s) to find the probability of each element in the vocabulary.
We compare against the true target word and use cross entropy loss to train the model weights, including the embedding layer.


With large vocabularies cross entropy loss can be expensive. There are approximations which are faster. We will stick with the full computation as we have a small vocab.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pytl


class CBOWModel(pytl.LightningModule):
    def __init__(
        self, vocab_size: int, embedding_dim: int, learning_rate: float = 1e-2
    ):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.learning_rate = learning_rate
        self.train_log_error = []
        self.val_log_error = []

    def forward(self, context):
        embedded_context = self.embeddings(context)
        # sum over context to get single embedding vector
        sum_embedded_context = torch.sum(embedded_context, dim=1)
        output = self.linear(sum_embedded_context)
        return output

    def training_step(self, batch, batch_idx):
        context, target = batch
        context = context.squeeze(1)
        target = target.squeeze(1)
        output = self(context)
        loss = nn.CrossEntropyLoss()(output, target)

        self.train_log_error.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context, target = batch
        context = context.squeeze(1)
        target = target.squeeze(1)
        output = self(context)
        loss = nn.CrossEntropyLoss()(output, target)

        self.val_log_error.append(loss.item())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
```

Test with a single training example

We have at least 64*20 sequences in the training example and they return a value for each title in the vocabulary (50).


```python
embedding_dim = 5
model = CBOWModel(vocab_size, embedding_dim)
batch = next(iter(dataloader_train))
model(batch[0]).shape
```




    torch.Size([1838, 50])



Custom logger to store metrics in python dictionaries


```python
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class DictLogger(Logger):
    def __init__(self):
        super().__init__()
        self.metrics = {}

    @property
    def name(self):
        return "DictLogger"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append((step, v))

    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        pass
```

Add early stopping and enforce a minimum of 6 epochs (it takes a few epochs to start improving over a mean baseline)


```python
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback


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
    min_epochs=6, monitor="val_loss", patience=3, mode="min"
)
```

Train the model


```python
embedding_dim = 20
model = CBOWModel(vocab_size, embedding_dim)
logger = DictLogger()
trainer = pytl.Trainer(
    max_epochs=20,
    logger=logger,
    log_every_n_steps=1,
    callbacks=[early_stop_callback],
)
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_test)
```

    GPU available: True (mps), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs
    /Users/stantoon/Documents/VariousProjects/github/data-analysis/.venv/lib/python3.12/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:654: Checkpoint directory /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/embeddings/DictLogger/1.0/checkpoints exists and is not empty.
    
      | Name       | Type      | Params | Mode 
    -------------------------------------------------
    0 | embeddings | Embedding | 1.0 K  | train
    1 | linear     | Linear    | 1.1 K  | train
    -------------------------------------------------
    2.0 K     Trainable params
    0         Non-trainable params
    2.0 K     Total params
    0.008     Total estimated model params size (MB)
    2         Modules in train mode
    0         Modules in eval mode


    Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


                                                                               

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Epoch 11: 100%|██████████| 616/616 [00:08<00:00, 71.62it/s, v_num=1.0, train_loss_step=3.630, val_loss_step=3.620, val_loss_epoch=3.550, train_loss_epoch=3.550]


Plotting train/val epoch loss

Looks like we are learning reasonably well.


```python
def training_logs_to_df(logger, name: str = None):
    df = (
        pl.concat(
            [
                pl.DataFrame(
                    logger.metrics["train_loss_epoch"],
                    orient="row",
                    schema=["batch", "train_loss"],
                ),
                pl.DataFrame(
                    logger.metrics["val_loss_epoch"],
                    orient="row",
                    schema=["batch", "val_loss"],
                ).drop("batch"),
            ],
            how="horizontal",
        )
        .with_row_index(name="epoch", offset=1)
        .unpivot(index=["epoch", "batch"], variable_name="dataset", value_name="loss")
    )
    if name:
        df = df.with_columns(pl.lit(name).alias("name"))
    return df


plot_df = training_logs_to_df(logger, name="nn_inner")


fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=plot_df, x="epoch", y="loss", hue="dataset", ax=ax)
fig.show()
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_44973/3638243156.py:31: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](word2vec_files/word2vec_37_1.png)
    


### Baseline model

A baseline model can be a uniform distribution over the vocab.
We use this to ensure our model is learning something meaningful.

Loss for the model


```python
y_est, y_true = [], []
for idx, batch in enumerate(dataloader_train):
    _x, _y = batch
    _y_est = model(_x)
    y_est.append(_y_est.detach().numpy())
    y_true.append(_y.detach().numpy())

y_est = np.concatenate(y_est)
y_true = np.concatenate(y_true)

nn.CrossEntropyLoss()(torch.tensor(y_est), torch.tensor(y_true).squeeze(1))
```




    tensor(3.5518)



Loss for a uniform distribution

CrossEntropyLoss applies a softmax first, so the fill value doesn't matter here.


```python
y_est_uniform = np.full(shape=y_est.shape, fill_value=1.0)
nn.CrossEntropyLoss()(torch.tensor(y_est_uniform), torch.tensor(y_true).squeeze(1))
```




    tensor(3.9120, dtype=torch.float64)



### Get embeddings

Get movie embeddings from the model layers and store for later use.


```python
from pathlib import Path

movie_embeddings_df = ratings_df["movieId", "token"].unique().sort("movieId")
movie_embeddings_df = pl.concat(
    [movie_embeddings_df, pl.DataFrame(model.embeddings.weight.detach().numpy())],
    how="horizontal",
).drop("token")
display(movie_embeddings_df)

Path("../data/embeddings").mkdir(parents=True, exist_ok=True)
movie_embeddings_df.write_parquet("../data/embeddings/word2vec_20.parquet")
```


<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (50, 21)</small><table border="1" class="dataframe"><thead><tr><th>movieId</th><th>column_0</th><th>column_1</th><th>column_2</th><th>column_3</th><th>column_4</th><th>column_5</th><th>column_6</th><th>column_7</th><th>column_8</th><th>column_9</th><th>column_10</th><th>column_11</th><th>column_12</th><th>column_13</th><th>column_14</th><th>column_15</th><th>column_16</th><th>column_17</th><th>column_18</th><th>column_19</th></tr><tr><td>i64</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td><td>f32</td></tr></thead><tbody><tr><td>1</td><td>0.170268</td><td>0.005285</td><td>0.173711</td><td>-0.514811</td><td>-0.717947</td><td>-0.426253</td><td>0.095356</td><td>0.199622</td><td>0.449421</td><td>0.635752</td><td>0.000855</td><td>-0.27404</td><td>-0.381104</td><td>0.076468</td><td>0.117747</td><td>0.058355</td><td>-0.195489</td><td>-0.429596</td><td>0.263023</td><td>0.047042</td></tr><tr><td>32</td><td>-0.508879</td><td>-0.034648</td><td>-0.601913</td><td>-0.150512</td><td>-0.586849</td><td>-0.113532</td><td>-0.202662</td><td>0.321145</td><td>0.165959</td><td>0.050083</td><td>-0.078404</td><td>0.570016</td><td>-0.045625</td><td>-0.14668</td><td>0.304445</td><td>-0.128906</td><td>-0.543937</td><td>-0.180518</td><td>-0.055841</td><td>0.024215</td></tr><tr><td>47</td><td>-0.031499</td><td>0.355408</td><td>-0.286842</td><td>-0.124137</td><td>0.329915</td><td>0.005808</td><td>0.354786</td><td>0.341577</td><td>0.198219</td><td>0.110213</td><td>0.412576</td><td>-0.344803</td><td>-0.210823</td><td>-0.445296</td><td>0.594245</td><td>-0.515395</td><td>0.356284</td><td>0.213535</td><td>-0.142556</td><td>0.032458</td></tr><tr><td>50</td><td>-0.324363</td><td>0.060988</td><td>0.007004</td><td>-0.091061</td><td>0.266039</td><td>-0.519057</td><td>0.006314</td><td>-0.527639</td><td>-0.69752</td><td>0.073614</td><td>0.83331</td><td>0.505465</td><td>-0.053987</td><td>-0.890503</td><td>0.081994</td><td>-0.145882</td><td>-0.237505</td><td>0.496448</td><td>-0.316936</td><td>0.07419</td></tr><tr><td>110</td><td>0.246857</td><td>-0.459349</td><td>0.304814</td><td>0.228468</td><td>0.538425</td><td>-0.34115</td><td>-0.468971</td><td>0.226764</td><td>-0.623984</td><td>0.341686</td><td>-0.251205</td><td>0.095348</td><td>0.19869</td><td>0.210442</td><td>-0.183482</td><td>-0.010722</td><td>-0.408665</td><td>-0.622425</td><td>0.272464</td><td>-0.16026</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>4993</td><td>-0.006215</td><td>0.108027</td><td>-0.323896</td><td>-0.340742</td><td>-0.128894</td><td>-0.168293</td><td>-0.813256</td><td>-0.417005</td><td>0.524438</td><td>0.162587</td><td>-0.490397</td><td>-0.189253</td><td>-0.293136</td><td>-0.726565</td><td>-1.080682</td><td>0.441643</td><td>0.52161</td><td>-0.08282</td><td>0.690598</td><td>-0.103665</td></tr><tr><td>5952</td><td>-0.558613</td><td>-0.147518</td><td>0.790067</td><td>-0.84381</td><td>-0.155056</td><td>0.188955</td><td>0.141412</td><td>0.286123</td><td>-0.389624</td><td>-0.765076</td><td>-0.872002</td><td>0.416089</td><td>-0.080197</td><td>0.419377</td><td>-0.193393</td><td>0.610264</td><td>0.568712</td><td>0.614561</td><td>-0.481618</td><td>-0.443433</td></tr><tr><td>7153</td><td>0.006625</td><td>0.832853</td><td>-0.093344</td><td>0.031611</td><td>-0.413336</td><td>-0.008178</td><td>-0.171429</td><td>0.01716</td><td>-0.150033</td><td>-0.930672</td><td>0.746316</td><td>-0.609427</td><td>0.974754</td><td>0.796484</td><td>-0.476131</td><td>0.371503</td><td>-0.033553</td><td>0.367805</td><td>0.717005</td><td>-0.464248</td></tr><tr><td>58559</td><td>-0.44002</td><td>0.347676</td><td>-0.26734</td><td>-0.046302</td><td>-0.116599</td><td>-0.620522</td><td>-0.21927</td><td>0.047042</td><td>-0.054698</td><td>-0.469698</td><td>-0.5836</td><td>0.256216</td><td>0.519925</td><td>0.131071</td><td>-0.462281</td><td>-0.363096</td><td>0.646975</td><td>0.42879</td><td>0.039636</td><td>-0.365519</td></tr><tr><td>79132</td><td>0.323736</td><td>0.066623</td><td>0.029123</td><td>-0.560737</td><td>-0.148516</td><td>0.312844</td><td>-0.171787</td><td>0.226215</td><td>-1.063323</td><td>-0.222305</td><td>0.139809</td><td>-0.532508</td><td>-0.199893</td><td>-0.619747</td><td>-0.588239</td><td>0.640908</td><td>0.050276</td><td>0.12261</td><td>-0.061714</td><td>-0.458745</td></tr></tbody></table></div>


# Appendix

## TODO
* Larger dataset
* Using

### Ratings based
The above Word2Vec approach will be assuming that people watched films that are similar in succession.

We have explicit ratings in the dataset we can use for a better indication.
We are assuming that people rate highly movies that are similar.

We also have genre information, how can we exploit this?

### Text dataset
Can we confirm the above approach with text data.

### Predict sequences

Given a starting movie, predict the sequences of movies the user will rate next.


```python
movies_df = utilities.load_movies()
movies_df = movies_df.join(movie_id_mapping, on="movieId", how="inner")
movies_df
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (50, 4)</small><table border="1" class="dataframe"><thead><tr><th>movieId</th><th>title</th><th>genres</th><th>movieIdMapped</th></tr><tr><td>i64</td><td>str</td><td>str</td><td>u32</td></tr></thead><tbody><tr><td>1</td><td>&quot;Toy Story (1995)&quot;</td><td>&quot;Adventure|Animation|Children|C…</td><td>0</td></tr><tr><td>32</td><td>&quot;Twelve Monkeys (a.k.a. 12 Monk…</td><td>&quot;Mystery|Sci-Fi|Thriller&quot;</td><td>1</td></tr><tr><td>47</td><td>&quot;Seven (a.k.a. Se7en) (1995)&quot;</td><td>&quot;Mystery|Thriller&quot;</td><td>2</td></tr><tr><td>50</td><td>&quot;Usual Suspects, The (1995)&quot;</td><td>&quot;Crime|Mystery|Thriller&quot;</td><td>3</td></tr><tr><td>110</td><td>&quot;Braveheart (1995)&quot;</td><td>&quot;Action|Drama|War&quot;</td><td>4</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>4993</td><td>&quot;Lord of the Rings: The Fellows…</td><td>&quot;Adventure|Fantasy&quot;</td><td>45</td></tr><tr><td>5952</td><td>&quot;Lord of the Rings: The Two Tow…</td><td>&quot;Adventure|Fantasy&quot;</td><td>46</td></tr><tr><td>7153</td><td>&quot;Lord of the Rings: The Return …</td><td>&quot;Action|Adventure|Drama|Fantasy&quot;</td><td>47</td></tr><tr><td>58559</td><td>&quot;Dark Knight, The (2008)&quot;</td><td>&quot;Action|Crime|Drama|IMAX&quot;</td><td>48</td></tr><tr><td>79132</td><td>&quot;Inception (2010)&quot;</td><td>&quot;Action|Crime|Drama|Mystery|Sci…</td><td>49</td></tr></tbody></table></div>



We make sequence of 'Lord of the Rings: The Fellows...' and 'Lord of the Rings: The Return ...'.
The model predicts the most likely middle movie as 'Lord of the Rings: The Two Tow...'


```python
y_est = model(torch.tensor([[45, 47]]))
movies_df.filter(pl.col("movieIdMapped") == y_est.argmax())
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1, 4)</small><table border="1" class="dataframe"><thead><tr><th>movieId</th><th>title</th><th>genres</th><th>movieIdMapped</th></tr><tr><td>i64</td><td>str</td><td>str</td><td>u32</td></tr></thead><tbody><tr><td>5952</td><td>&quot;Lord of the Rings: The Two Tow…</td><td>&quot;Adventure|Fantasy&quot;</td><td>46</td></tr></tbody></table></div>



## Larger dataset


```python
top_movie_ids = (
    ratings_df.group_by("movieId")
    .count()
    .sort("count", descending=True)
    .head(200)[["movieId"]]
)
ratings_med_df = ratings_df.join(top_movie_ids, on="movieId", how="inner")

user_id_counts = (
    ratings_med_df.group_by("userId").count().filter(pl.col("count") >= 20)[["userId"]]
)
ratings_med_df = ratings_med_df.join(user_id_counts, on="userId", how="inner")
ratings_med_df.shape
```

    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_44973/2374773301.py:3: DeprecationWarning: `GroupBy.count` is deprecated. It has been renamed to `len`.
      .count()
    /var/folders/_v/nlh4h1yx2n1gd6f3szjlgxt40000gr/T/ipykernel_44973/2374773301.py:10: DeprecationWarning: `GroupBy.count` is deprecated. It has been renamed to `len`.
      ratings_med_df.group_by("userId").count().filter(pl.col("count") >= 20)[["userId"]]





    (1511948, 6)




```python
ratings_med_df, token_mapping = convert_movies_to_tokens(ratings_med_df)
display(ratings_med_df.head(4))

sequences = get_sequences_from_df(ratings_med_df)
print(sequences[0][:10])

vocab_size = ratings_med_df["movieId"].unique().count()
window_size = 1

# Create dataset and dataloader
dataset = CBOWDataset(sequences, window_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

embedding_dim = 20
model = CBOWModel(vocab_size, embedding_dim)
trainer = pytl.Trainer(max_epochs=1)
trainer.fit(model, dataloader)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(moving_average(model.train_log_error, 100), label="train")
ax.set_title(f"Training error")
ax.set_xlabel("Batches")
ax.set_ylabel("LL")
ax.legend()
fig.show()

embedding_matrix = model.embeddings.weight
similarities = calculate_cosine_similarity(embedding_matrix.cpu()).detach().numpy()
similarities[np.triu_indices(similarities.shape[0], k=1)] = np.nan

labels = token_mapping.join(names_df, on="movieId").sort("token")["title"].to_list()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    data=similarities,
    ax=ax,
    xticklabels=labels,
    yticklabels=labels,
)
fig.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[26], line 1
    ----> 1 ratings_med_df, token_mapping = convert_movies_to_tokens(ratings_med_df)
          2 display(ratings_med_df.head(4))
          4 sequences = get_sequences_from_df(ratings_med_df)


    NameError: name 'convert_movies_to_tokens' is not defined


Strong correlations


```python
indices = top_n_indices(similarities)
print(similarities[indices[0][0], indices[0][1]])
print(labels[indices[0][0]], labels[indices[0][1]])
print(similarities[indices[1][0], indices[1][1]])
print(labels[indices[1][0]], labels[indices[1][1]])
print(similarities[indices[2][0], indices[2][1]])
print(labels[indices[2][0]], labels[indices[1][1]])
indices
```

    0.9033904
    American Pie (1999) Austin Powers: International Man of Mystery (1997)
    0.8816099
    One Flew Over the Cuckoo's Nest (1975) Casablanca (1942)
    0.8643362
    Donnie Darko (2001) Casablanca (1942)





    [(134, 103),
     (78, 68),
     (156, 151),
     (107, 104),
     (141, 96),
     (198, 188),
     (119, 100),
     (156, 149),
     (109, 74),
     (196, 193)]




```python
raise NotImplementedError
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    Cell In[23], line 1
    ----> 1 raise NotImplementedError


    NotImplementedError: 

