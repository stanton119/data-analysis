"""Generating sine waves.

Generate some sine wave training data.
Start with predicting the next few samples with a dense network.
Extend to RNN architectures to generate long sequences.

References:
https://github.com/pytorch/examples/tree/master/time_sequence_prediction
https://github.com/osm3000/Sequence-Generation-Pytorch
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
# %% generate sine data
rng = np.random.default_rng()


def gen_sine_wave(freq: float = 1.0):
    time_steps = 2000
    samp_freq = 1000
    x = np.arange(time_steps) * 2 * np.pi * freq / samp_freq
    sin_wave = np.sin(x)
    return sin_wave


plt.plot(gen_sine_wave(1))
plt.plot(gen_sine_wave(2))

# %%
# create training data
seq_len_train = 10
seq_len_predict = 5


def split_data_to_xy(all_data):
    x = []
    y = []
    for idx in range(len(all_data)):
        if idx + seq_len_train + seq_len_predict > len(all_data):
            break
        x.append(all_data[idx : idx + seq_len_train])
        y.append(all_data[idx + seq_len_train : idx + seq_len_train + seq_len_predict])

    x = np.array(x)
    y = np.array(y)
    return x, y


def generate_multiple_freq_data(n_freq: int = 10, split_fcn=split_data_to_xy):
    x, y = [], []
    for idx in range(10):
        _x, _y = split_fcn(gen_sine_wave(rng.uniform() * 100))
        x.append(_x)
        y.append(_y)
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


def plot_result(idx):
    plt.plot(np.arange(0, x.shape[1]), x[idx], label="train")
    plt.plot(np.arange(x.shape[1], x.shape[1] + y.shape[1]), y[idx], label="actual")
    try:
        plt.plot(
            np.arange(x.shape[1], x.shape[1] + y_est.shape[1]), y_est[idx], label="est"
        )
    except NameError:
        pass
    plt.legend()
    plt.show()


x, y = generate_multiple_freq_data(n_freq=20)

for idx in range(5):
    plot_result(int(rng.uniform() * x.shape[0]))

# %%
import torch

def create_dataloaders(x: np.array, y: np.array, train_frac: float = 0.8):
    train_idx = int(x.shape[0] * train_frac)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x[:train_idx, :].astype(np.float32)),
        torch.tensor(y[:train_idx, :].astype(np.float32)),
    )
    eval_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x[train_idx + 1 :, :].astype(np.float32)),
        torch.tensor(y[train_idx + 1 :, :].astype(np.float32)),
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=128,
    )

    dataloader_eval = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        shuffle=True,
        batch_size=128,
    )
    return dataloader_train, dataloader_eval


dataloader_train, dataloader_eval = create_dataloaders(x=x, y=y)

# %%
# train dense network
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class DenseModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        n_inputs: int = 2,
        n_outputs: int = 1,
        hidden_layer_size: int = 20,
    ):
        super().__init__()
        self.train_log_error = []
        self.val_log_error = []
        self.learning_rate = learning_rate

        self.dense_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_inputs, out_features=hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_layer_size, out_features=hidden_layer_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_layer_size, out_features=n_outputs),
        )

    def forward(self, x):
        return self.dense_net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = torch.nn.functional.l1_loss(y_hat, y)
        l2_loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("l1_loss", l1_loss, on_epoch=True)
        self.log("l2_loss", l2_loss, on_epoch=True)
        self.train_log_error.append(l2_loss.detach().numpy())
        return l2_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = torch.nn.functional.l1_loss(y_hat, y)
        l2_loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("val_l1_loss", l1_loss, on_epoch=True)
        self.log("val_l2_loss", l2_loss, on_epoch=True)
        self.val_log_error.append(l2_loss.detach().numpy())
        return l2_loss


# fit network
model = DenseModel(n_inputs=x.shape[1], n_outputs=y.shape[1])
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[EarlyStopping(monitor="val_l2_loss", patience=30)],
)
trainer.fit(model, dataloader_train, dataloader_eval)

plt.plot(model.train_log_error)
plt.plot(model.val_log_error)

# %%
# predict model
y_est = model(torch.tensor(x.astype(np.float32))).detach().numpy().astype(np.float64)

for idx in range(5):
    plot_result(int(rng.uniform() * x.shape[0]))


# %%
# we cant generate longer sequences than what we trained the model with
# changing the architecture to RNN based allows us to generate infinite sequences.

# %%
# training set for one period ahead
seq_len_train = 10
seq_len_predict = 1

x, y = generate_multiple_freq_data(n_freq=20)

dataloader_train, dataloader_eval = create_dataloaders(x=x, y=y)

# fit network
model = DenseModel(n_inputs=x.shape[1], n_outputs=y.shape[1])
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[EarlyStopping(monitor="val_l2_loss", patience=30)],
)
trainer.fit(model, dataloader_train, dataloader_eval)

plt.plot(model.train_log_error)
plt.plot(model.val_log_error)

# %%
# predict model
pred_len = 100


def recursive_predict(x):
    _x = torch.tensor(x.copy().astype(np.float32))

    for idx in range(pred_len):
        _y_est = model(_x[idx : len(x) + idx])
        _x = torch.cat((_x, _y_est))
    return _x[len(x) :].detach().numpy().astype(np.float64)


y_est = np.zeros((x.shape[0], pred_len))
for idx in range(5):
    r_idx = int(rng.uniform() * x.shape[0])
    y_est[r_idx] = recursive_predict(x[r_idx])
    plot_result(r_idx)

# %%
# Performance is bad for the case of low frequency data.
# It seems not to have enough context to generate forward and changes the frequency.
# %%
# RNNs
# No need to input lags as training data as the RNN memory will account for it
# Training data can be 1 datapoint, test = subsequent data point

X = []
y = []
# for i in range(0, y_data.shape[0] - 1, seq_len):
for i in range(0, y_data.shape[0] - 1):
    if i + seq_len < y_data.shape[0]:
        X.append(y_data[i : i + seq_len])
        y.append(
            y_data[i + 1 : i + seq_len + 1]
        )  # next sequence (including the next point in the last)
        # y.append(y_data[i + seq_len])  # next point only
X = np.array(X)
y = np.array(y)


def split_data_to_xy_rnn(all_data):
    x = []
    y = []
    for idx in range(len(all_data)):
        if idx + seq_len_train + 1 > len(all_data):
            break
        x.append(all_data[idx : idx + seq_len_train])
        y.append(all_data[idx + 1 : idx + seq_len_train + 1])

    x = np.array(x)
    y = np.array(y)
    return x, y

x, y = generate_multiple_freq_data(n_freq=20, split_fcn=split_data_to_xy_rnn)

for idx in range(5):
    plot_result(int(rng.uniform() * x.shape[0]))


# %%
class SequenceModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        hidden_layer_size: int = 51,
    ):
        super().__init__()
        self.train_log_error = []
        self.val_log_error = []
        self.learning_rate = learning_rate

        self.hidden_layer_size = hidden_layer_size
        self.lstm1 = torch.nn.LSTMCell(1, hidden_layer_size)
        self.lstm2 = torch.nn.LSTMCell(hidden_layer_size, hidden_layer_size)
        self.linear = torch.nn.Linear(hidden_layer_size, 1)

    def forward(self, x, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_layer_size)
        c_t = torch.zeros(input.size(0), self.hidden_layer_size)
        h_t2 = torch.zeros(input.size(0), self.hidden_layer_size)
        c_t2 = torch.zeros(input.size(0), self.hidden_layer_size)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = torch.nn.functional.l1_loss(y_hat, y)
        l2_loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("l1_loss", l1_loss, on_epoch=True)
        self.log("l2_loss", l2_loss, on_epoch=True)
        self.train_log_error.append(l2_loss.detach().numpy())
        return l2_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = torch.nn.functional.l1_loss(y_hat, y)
        l2_loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log("val_l1_loss", l1_loss, on_epoch=True)
        self.log("val_l2_loss", l2_loss, on_epoch=True)
        self.val_log_error.append(l2_loss.detach().numpy())
        return l2_loss


model = SequenceModel()

x[:2].reshape(-1, 2, 1).shape
# (seq, batch, feature)
# normal = (batch, feature)
lstm1 = torch.nn.LSTMCell(1, hidden_layer_size)
torch.tensor(x[:2].reshape(2, 1, -1)).split(1, dim=1)

y_est = model(torch.tensor(x[:2].reshape(1, 2, -1).astype(np.float32))).detach().numpy().astype(np.float64)
y_est = model(torch.tensor(x[:1].reshape(1, -1, 1).astype(np.float32))).detach().numpy().astype(np.float64)
y_est = model(torch.tensor(x[:1].astype(np.float32))).detach().numpy().astype(np.float64)

dir(dataloader_train.dataset)
dataloader_train.dataset.tensors[0].shape
dataloader_train.dataset.tensors[1].shape
