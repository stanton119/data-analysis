# %%
"""Classify the language given a surname.
Tutorial from: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
Converting for PyTorch-Lightning.
"""

# %%
# prep data

"""Load list of names from each file.
Populates a dictionary: key = language, values = List[names]."""

import random
import string
import glob
import os
import unicodedata
import torch

all_letters = string.ascii_lowercase + " .,;'"
n_letters = len(all_letters)

# %%
# one hot encode each letter
# convert name into shape: (sequence length, batch size, ohe size)
# sequence length = length of the word
# batch size = 1 generally, as sequence lengths are variable
# ohe size = 57 = encoding size


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    line = line.lower()
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor("J"))

print(lineToTensor("Jones").size())
# %%
# create dataset


class SeqData(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        filepath: str,
        # category_lines,
        is_train: bool = True,
        train_split: float = 0.8,
        data_len: int = None,
    ):
        # load data files from txt
        category_lines = {}
        for _filepath in self._list_files(filepath + "*.txt"):
            category = os.path.splitext(os.path.basename(_filepath))[0]
            lines = self._read_file(_filepath)
            category_lines[category] = lines

        # convert to dictionary
        words = []
        labels = []
        unique_labels = list(category_lines.keys())

        for label in category_lines:
            for word in category_lines[label]:
                words.append(word)
                labels.append(label)
        self.unique_labels = unique_labels
        self.is_train = is_train

        # randomise order to mix labels
        random.seed(0)
        self.index_order = list(range(len(words)))
        random.shuffle(self.index_order)
        words = [words[idx] for idx in self.index_order]
        labels = [labels[idx] for idx in self.index_order]

        if data_len is not None:
            words = words[:data_len]
            labels = labels[:data_len]

        # split train/test
        train_idx = int(len(words) * train_split)
        if self.is_train:
            self.words = words[:train_idx]
            self.labels = labels[:train_idx]
        else:
            self.words = words[train_idx:]
            self.labels = labels[train_idx:]
        self.current_idx = 0

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return lineToTensor(self.words[index]), self.unique_labels.index(
            self.labels[index]
        )

    @classmethod
    def _list_files(cls, path):
        return glob.glob(path)

    @classmethod
    def _read_file(cls, filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [cls._unicode_to_ascii(line) for line in lines]

    @classmethod
    def _unicode_to_ascii(cls, s):
        # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in all_letters
        )


train_dataset = SeqData(filepath="data/names/")
n_categories = len(train_dataset.unique_labels)
x, y = next(iter(train_dataset))
x.shape

# %%
# create dataloaders
dataloader_all = torch.utils.data.DataLoader(
    dataset=SeqData(filepath="data/names/", is_train=True, data_len=None, train_split=1),
    shuffle=True,
    batch_size=1,
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=SeqData(filepath="data/names/", is_train=True, data_len=None),
    shuffle=True,
    batch_size=1,
)

dataloader_eval = torch.utils.data.DataLoader(
    dataset=SeqData(filepath="data/names/", is_train=False, data_len=None),
    shuffle=True,
    batch_size=1,
)

batch = next(iter(dataloader_train.dataset))
batch
batch[0].shape

# %%
# create RNN module with dense layers
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class RNN(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.train_log_error = []
        self.val_log_error = []
        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.loss = nn.NLLLoss()

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor, hidden=None):
        # take array of sequences, return output class probs, hidden layer
        if hidden is None:
            hidden = self.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self._forward(line_tensor[i], hidden)

        return output

    def _forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def _eval_batch(self, batch):
        # accepts only one word tensor at a time, batch size of 1
        x, y = batch

        output = self.forward(x[0])
        return self.loss(output, y)

        # loss = torch.Tensor([0])
        # for idx in range(x.shape[0]):
        #     output = self.forward(x[idx])
        #     _loss = self.loss(output, y)
        #     loss += _loss
        # return loss

    def training_step(self, batch, batch_idx):
        loss = self._eval_batch(batch)

        self.train_log_error.append(loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_batch(batch)

        self.log("val_loss", loss, on_epoch=True)
        self.val_log_error.append(loss.detach().numpy())
        return loss

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
model = RNN(n_letters, n_hidden, n_categories, learning_rate=1e-5)

model(lineToTensor("Jones"))


# %%
# train
import matplotlib.pyplot as plt
import numpy as np

# fit network
trainer = pl.Trainer(
    max_epochs=4,
    # callbacks=[EarlyStopping(monitor="val_loss", patience=30)],
)
# trainer.fit(model, dataloader_train, dataloader_eval)
trainer.fit(model, dataloader_all)

# %%
# plot training loss
def moving_average(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


plt.plot(moving_average(np.array(model.train_log_error)))
plt.plot(moving_average(np.array(model.val_log_error)))

# %%
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)  # max index
    category_i = top_i[0].item()
    return train_dataset.unique_labels[category_i], category_i


def predict(input_line, n_predictions=3):
    print("\n> %s" % input_line)
    with torch.no_grad():
        output = model(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("(%.2f) %s" % (value, train_dataset.unique_labels[category_index]))
            predictions.append([value, train_dataset.unique_labels[category_index]])


y_est = model(lineToTensor("Schneider"))
categoryFromOutput(y_est)
# y_est


predict("Dovesky")
predict("Jackson")
predict("Satoshi")
predict("Ferguson")
predict("Stanton")
predict("Aakbah")
predict("Miele")
predict("Schneider")
