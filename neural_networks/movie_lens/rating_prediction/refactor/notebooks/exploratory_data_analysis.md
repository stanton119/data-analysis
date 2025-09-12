# Exploratory Data Analysis


## Dataloader exploration


```python
import recsys.movie_lens
```


```python
recsys.movie_lens.PROJECT_ROOT, recsys.movie_lens.DEFAULT_DATA_DIR
```




    (PosixPath('/Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor'),
     PosixPath('/Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/data/raw'))




```python
positive_ratings, num_users, num_items, user_encoder, item_encoder = (
    recsys.movie_lens.load_movielens()
)
```

    Loaded 55375 positive interactions
    Users: 942, Items: 1447



```python
train_loader, val_loader, test_loader, num_users, num_items = (
    recsys.movie_lens.get_dataloaders()
)
```

    Loaded 55375 positive interactions
    Users: 942, Items: 1447
    Loaded item features with shape: (1447, 20)
    Loaded user features with shape: (942, 4)


Positive and negative samples are based on index


```python
train_loader.dataset[0]
```




    {'user_id': tensor(297),
     'item_id': tensor(465),
     'rating': tensor(1.),
     'user_continuous': tensor([0.4400, 0.0000]),
     'user_occupation': tensor(6),
     'user_zip': tensor(6),
     'item_genres': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
             0.]),
     'item_year': tensor(63)}




```python
train_loader.dataset[0], train_loader.dataset[1], train_loader.dataset[
    2
], train_loader.dataset[3]
```




    ({'user_id': tensor(297),
      'item_id': tensor(465),
      'rating': tensor(1.),
      'user_features': tensor([0.4400, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'item_features': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
              0.])},
     {'user_id': tensor(297),
      'item_id': tensor(856),
      'rating': tensor(0.),
      'user_features': tensor([0.4400, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'item_features': tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
              0.])},
     {'user_id': tensor(297),
      'item_id': tensor(605),
      'rating': tensor(0.),
      'user_features': tensor([0.4400, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'item_features': tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0.])},
     {'user_id': tensor(297),
      'item_id': tensor(1302),
      'rating': tensor(0.),
      'user_features': tensor([0.4400, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
              0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'item_features': tensor([0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0.])})



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




    {'user_id': tensor([286, 523, 359,  ..., 803, 191, 384]),
     'item_id': tensor([ 505, 1405,  939,  ...,  565, 1384,   65]),
     'rating': tensor([0., 1., 0.,  ..., 0., 0., 0.])}



## Models


```python
from recsys.models import get_model
import yaml

model_config = """
# architecture: "NeuralInnerProduct"
# embedding_dim: 10
# include_bias: true

# architecture: "NeuMF"
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

architecture: "FactorizationMachines"
embedding_dim: 10
include_bias: true
"""

model_config = yaml.safe_load(model_config)
model_config["num_users"] = train_loader.dataset.num_users
model_config["num_items"] = train_loader.dataset.num_items

model = get_model(**model_config)

y_est = model(batch["user_id"], batch["item_id"])
y_est
```




    tensor([[-0.0412],
            [-0.0107],
            [-0.0132],
            ...,
            [ 0.0002],
            [-0.0291],
            [ 0.0176]], grad_fn=<AddBackward0>)




```python
y_est.shape, batch["user_id"].shape, batch["item_id"].shape
```




    (torch.Size([1024, 1]), torch.Size([1024]), torch.Size([1024]))



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
  architecture: "NeuralColabFilter"
  embedding_dim: 10
  num_users: 943
  num_items: 1682

training:
  epochs: 1
  learning_rate: 0.001
  loss_function: "mse"

logging:
  experiment_name: "movielens_rating_prediction"
  run_name: "nn_colab_filter_linear"
"""
config = yaml.safe_load(config)
recsys.train.main(config)
```

    INFO:recsys.train:Configuration: {'dataset': {'version': '100k', 'batch_size': 4096, 'test_split': 0.2, 'num_negatives': 4}, 'model': {'architecture': 'NeuralColabFilter', 'embedding_dim': 10, 'num_users': 943, 'num_items': 1682}, 'training': {'epochs': 1, 'learning_rate': 0.001, 'loss_function': 'mse'}, 'logging': {'experiment_name': 'movielens_rating_prediction', 'run_name': 'nn_colab_filter_linear'}}
    INFO:recsys.train:Loading model: NeuralColabFilter
    INFO:recsys.train:Loading dataset: {'version': '100k', 'batch_size': 4096, 'test_split': 0.2, 'num_negatives': 4}
    GPU available: True (mps), used: True
    TPU available: False, using: 0 TPU cores
    HPU available: False, using: 0 HPUs


    Loaded 55375 positive interactions
    Users: 942, Items: 1447


    
      | Name  | Type  | Params | Mode 
    ----------------------------------------
    0 | model | Model | 28.9 K | train
    ----------------------------------------
    28.9 K    Trainable params
    0         Non-trainable params
    28.9 K    Total params
    0.116     Total estimated model params size (MB)
    6         Modules in train mode
    0         Modules in eval mode


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Sanity Checking DataLoader 0:   0%|          | 0/2 [00:00<?, ?it/s]

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/torch/nn/modules/loss.py:616: UserWarning: Using a target size (torch.Size([4096])) that is different to the input size (torch.Size([4096, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return F.mse_loss(input, target, reduction=self.reduction)


                                                                               

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.


    Epoch 0: 100%|██████████| 55/55 [00:06<00:00,  8.84it/s, v_num=8b61, train_loss_step=0.892]

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/torch/nn/modules/loss.py:616: UserWarning: Using a target size (torch.Size([316])) that is different to the input size (torch.Size([316, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return F.mse_loss(input, target, reduction=self.reduction)


    Epoch 0: 100%|██████████| 55/55 [00:07<00:00,  7.32it/s, v_num=8b61, train_loss_step=0.892, val_loss_step=0.830, val_loss_epoch=0.807, train_loss_epoch=0.927]

    /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/movie_lens/rating_prediction/refactor/.venv/lib/python3.12/site-packages/torch/nn/modules/loss.py:616: UserWarning: Using a target size (torch.Size([2127])) that is different to the input size (torch.Size([2127, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return F.mse_loss(input, target, reduction=self.reduction)
    `Trainer.fit` stopped: `max_epochs=1` reached.


    Epoch 0: 100%|██████████| 55/55 [00:07<00:00,  7.30it/s, v_num=8b61, train_loss_step=0.892, val_loss_step=0.830, val_loss_epoch=0.807, train_loss_epoch=0.927]

