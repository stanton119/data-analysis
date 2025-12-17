# Exploratory Data Analysis


## Dataloader exploration


```python
import recsys.movie_lens
```


```python
recsys.movie_lens.PROJECT_ROOT, recsys.movie_lens.DEFAULT_DATA_DIR
```




    (PosixPath('/Users/rich/Developer/Github/VariousDataAnalysis/recommenders/recsys'),
     PosixPath('/Users/rich/Developer/Github/VariousDataAnalysis/recommenders/recsys/data/raw'))




```python
positive_ratings, num_users, num_items, user_encoder, item_encoder = (
    recsys.movie_lens.load_movielens()
)
```

    Loaded 55375 positive interactions
    Users: 942, Items: 1447



```python
(
    train_loader,
    val_loader,
    test_loader,
    num_users,
    num_items,
    user_feature_dims,
    item_feature_dims,
) = recsys.movie_lens.get_dataloaders()
```

    Loaded 55375 positive interactions
    Users: 942, Items: 1447
    Loaded user features with dims: {'continuous_dim': 2, 'categorical_dims': {'occupation': 21, 'zip_code': 795}}
    Loaded item features with dims: {'continuous_dim': 19, 'categorical_dims': {'year': 71}}


Positive and negative samples are based on index


```python
train_loader.dataset[0]
```




    {'user_id': tensor(297),
     'item_id': tensor(465),
     'rating': tensor(1.),
     'user_features': {'continuous': tensor([0.4400, 0.0000]),
      'categorical': {'occupation': tensor(6), 'zip_code': tensor(6)}},
     'item_features': {'continuous': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
              0.]),
      'categorical': {'year': tensor(35)}}}




```python
train_loader.dataset[0], train_loader.dataset[1], train_loader.dataset[
    2
], train_loader.dataset[3]
```




    ({'user_id': tensor(297),
      'item_id': tensor(465),
      'rating': tensor(1.),
      'user_features': {'continuous': tensor([0.4400, 0.0000]),
       'categorical': {'occupation': tensor(6), 'zip_code': tensor(6)}},
      'item_features': {'continuous': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
               0.]),
       'categorical': {'year': tensor(35)}}},
     {'user_id': tensor(297),
      'item_id': tensor(1427),
      'rating': tensor(0.),
      'user_features': {'continuous': tensor([0.4400, 0.0000]),
       'categorical': {'occupation': tensor(6), 'zip_code': tensor(6)}},
      'item_features': {'continuous': tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0.]),
       'categorical': {'year': tensor(65)}}},
     {'user_id': tensor(297),
      'item_id': tensor(803),
      'rating': tensor(0.),
      'user_features': {'continuous': tensor([0.4400, 0.0000]),
       'categorical': {'occupation': tensor(6), 'zip_code': tensor(6)}},
      'item_features': {'continuous': tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
               0.]),
       'categorical': {'year': tensor(68)}}},
     {'user_id': tensor(297),
      'item_id': tensor(1310),
      'rating': tensor(0.),
      'user_features': {'continuous': tensor([0.4400, 0.0000]),
       'categorical': {'occupation': tensor(6), 'zip_code': tensor(6)}},
      'item_features': {'continuous': tensor([0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0.]),
       'categorical': {'year': tensor(66)}}})



Number of user/items


```python
train_loader.dataset.num_users, train_loader.dataset.num_items
```




    (942, 1447)



Dataloader outputs


```python
batch = next(iter(train_loader))
batch
```




    {'user_id': tensor([605, 364, 653,  ..., 196,  13, 620]),
     'item_id': tensor([295, 891, 976,  ..., 254, 605,  25]),
     'rating': tensor([1., 0., 0.,  ..., 0., 0., 0.]),
     'user_features': {'continuous': tensor([[0.2800, 0.0000],
              [0.2900, 0.0000],
              [0.2700, 1.0000],
              ...,
              [0.5500, 0.0000],
              [0.4500, 0.0000],
              [0.1700, 0.0000]]),
      'categorical': {'occupation': tensor([14,  9, 18,  ..., 19, 17, 18]),
       'zip_code': tensor([496, 172, 574,  ..., 537, 416, 470])}},
     'item_features': {'continuous': tensor([[0., 1., 0.,  ..., 1., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.],
              ...,
              [0., 1., 1.,  ..., 0., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.]]),
      'categorical': {'year': tensor([69, 70, 63,  ..., 69, 27, 67])}}}



## Models


```python
from recsys.models import get_model
import yaml

model_config = """
# architecture: "NeuralInnerProduct"
# embedding_dim: 10
# include_bias: true

# architecture: "NeuralColabFilter"
# mf_dim: 8
# mlp_dim: 32
# layers: [64, 32, 16]
# include_bias: true

# architecture: "WideAndDeep"
# embedding_dim: 64
# deep_layers: [512, 256, 128]
# dropout: 0.2
# include_bias: true

# architecture: "DCN"
# embedding_dim: 2
# cross_layers: 3
# deep_layers: [512, 256, 128]
# dropout: 0.2

# architecture: "DCNV2"
# embedding_dim: 64
# cross_layers: 3
# deep_layers: [512, 256, 128]
# dropout: 0.2

# architecture: "FactorizationMachines"
# embedding_dim: 10
# include_bias: true

# architecture: "AutoInt"
# embedding_dim: 64
# num_heads: 8
# num_layers: 3
# dropout: 0.1
# include_bias: true
# categorical_features: null
# num_user_continuous: 0
# num_item_continuous: 0

architecture: "AutoInt"
embedding_dim: 64
num_heads: 8
num_layers: 3
dropout: 0.1
include_bias: true
categorical_features:
    occupation: 21
    zip_code: 3439
    year: 81
num_user_continuous: 2
num_item_continuous: 19
"""

model_config = yaml.safe_load(model_config)
model_config["num_users"] = train_loader.dataset.num_users
model_config["num_items"] = train_loader.dataset.num_items

model = get_model(**model_config)

y_est = model(batch)
y_est
```




    tensor([[ 0.4598],
            [-0.0996],
            [ 0.4548],
            ...,
            [ 0.3924],
            [-0.2679],
            [-0.0149]], grad_fn=<AddmmBackward0>)




```python
y_est.shape, batch["user_id"].shape, batch["item_id"].shape, batch["rating"].shape
```




    (torch.Size([1024, 1]),
     torch.Size([1024]),
     torch.Size([1024]),
     torch.Size([1024]))



## Training


```python
import recsys.train
import yaml

config = """
dataset:
  version: "100k"
  batch_size: 4096
  test_split: 0.2
  num_negatives: 4

model:
  # architecture: "NeuralColabFilter"
  # embedding_dim: 10
  # num_users: 943
  # num_items: 1682

  # architecture: "NeuralColabFilter"
  # mf_dim: 8
  # mlp_dim: 32
  # layers: [64, 32, 16]
  # include_bias: true

  architecture: "AutoInt"
  embedding_dim: 64
  num_heads: 8
  num_layers: 3
  dropout: 0.1
  include_bias: true
  categorical_features: null
  num_user_continuous: 0
  num_item_continuous: 0


training:
  epochs: 1
  learning_rate: 0.001
  loss_function: "mse"

logging:
  experiment_name: "movielens_rating_prediction"
  run_name: "debug"
"""
config = yaml.safe_load(config)
recsys.train.main(config)
```

    INFO:recsys.train:Configuration: {'dataset': {'version': '100k', 'batch_size': 4096, 'test_split': 0.2, 'num_negatives': 4}, 'model': {'architecture': 'AutoInt', 'embedding_dim': 64, 'num_heads': 8, 'num_layers': 3, 'dropout': 0.1, 'include_bias': True, 'categorical_features': None, 'num_user_continuous': 0, 'num_item_continuous': 0}, 'training': {'epochs': 1, 'learning_rate': 0.001, 'loss_function': 'mse'}, 'logging': {'experiment_name': 'movielens_rating_prediction', 'run_name': 'nn_colab_filter_linear'}}
    INFO:recsys.train:Loading dataset: {'version': '100k', 'batch_size': 4096, 'test_split': 0.2, 'num_negatives': 4}
    INFO:recsys.train:Loading model: AutoInt
    GPU available: True (mps), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs


    Loaded 55375 positive interactions
    Users: 942, Items: 1447
    Loaded user features with dims: {'continuous_dim': 2, 'categorical_dims': {'occupation': 21, 'zip_code': 795}}
    Loaded item features with dims: {'continuous_dim': 19, 'categorical_dims': {'year': 71}}


    /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:654: Checkpoint directory /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/notebooks/checkpoints exists and is not empty.
    
      | Name         | Type                   | Params | Mode 
    ----------------------------------------------------------------
    0 | model        | Model                  | 203 K  | train
    1 | criterion    | MSELoss                | 0      | train
    2 | train_auc    | BinaryAUROC            | 0      | train
    3 | val_auc      | BinaryAUROC            | 0      | train
    4 | test_auc     | BinaryAUROC            | 0      | train
    5 | train_pr_auc | BinaryAveragePrecision | 0      | train
    6 | val_pr_auc   | BinaryAveragePrecision | 0      | train
    7 | test_pr_auc  | BinaryAveragePrecision | 0      | train
    ----------------------------------------------------------------
    203 K     Trainable params
    0         Non-trainable params
    203 K     Total params
    0.813     Total estimated model params size (MB)
    33        Modules in train mode
    0         Modules in eval mode


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]

    /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:425: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


                                                                               

    /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Epoch 0: 100%|██████████| 48/48 [00:23<00:00,  2.01it/s, v_num=64ec, train_loss_step=0.180, val_loss_step=0.160, val_loss_epoch=0.159, train_loss_epoch=0.243]

    /Users/rich/Developer/Github/VariousDataAnalysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/torchmetrics/utilities/prints.py:43: UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
      warnings.warn(*args, **kwargs)  # noqa: B028
    `Trainer.fit` stopped: `max_epochs=1` reached.


    Epoch 0: 100%|██████████| 48/48 [00:23<00:00,  2.01it/s, v_num=64ec, train_loss_step=0.180, val_loss_step=0.160, val_loss_epoch=0.159, train_loss_epoch=0.243]

