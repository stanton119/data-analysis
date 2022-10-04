"""
sentiment analysis
text classification

https://www.youtube.com/watch?v=Y_hzMnRXjhI

dataset:
kaggle
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home

News headlines from The Onion = sarcastic
News headlines from Huffington Post = not sarcastic


Dataset is processed to single sequence of tokens and list of offsets.
"""
# %%
# loading data to dataset
# dataset needs __len__ and __get_item__ methods
import torch
import json
from pathlib import Path
import random


class SarcasmData(torch.utils.data.dataset.Dataset):
    """Load sentiment data. Takes only tweet and sentiment from data."""

    def __init__(
        self,
        filepath: Path,
    ):
        self.sentences = []
        self.labels = []

        with open(filepath, "r") as f:
            for line in f:
                json_contents = json.loads(line)
                self.sentences.append(json_contents["headline"])
                self.labels.append(json_contents["is_sarcastic"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


full_dataset = SarcasmData(
    filepath=Path("data/Sarcasm_Headlines_Dataset.json").absolute()
)

# split train and test sets
num_train = int(len(full_dataset) * 0.8)
train_dataset, test_dataset = torch.utils.data.dataset.random_split(
    full_dataset,
    [num_train, len(full_dataset) - num_train],
    generator=torch.Generator().manual_seed(42),
)

x, y = train_dataset[random.randint(0, len(train_dataset))]
x, y

# %%
# normalise string to tokens
# build tokenizer to take sentence and split to List[str]
# build vocab to generate List[int]

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
train_iter = iter(train_dataset)


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

vocab(["here", "is", "an", "example"])
vocab(tokenizer(x))

len(vocab)

# %%
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    """Transform batch from dataset for the text/label pipelines.
    Creates lists of labels, text tokens and offsets."""

    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# %%
# define model
# embedding layer -> linear layer
import torch


class TextClassificationModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# %%
# create model and check layer dimensions
from torchinfo import summary

num_class = len(train_dataset.sentiment_map)
vocab_size = len(vocab)
embed_size = 16
model = TextClassificationModel(vocab_size, embed_size, num_class).to(device)

tokens = [7, 43, 67, 7, 43, 68]
offsets = [0, 3]
summary(model, input_data=(torch.tensor(tokens), torch.tensor(offsets)))

# %%
# pytorch training methods
import time


def train(dataloader):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += loss
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    # epoch, idx, len(dataloader), total_acc / total_count
                    epoch,
                    idx,
                    len(dataloader),
                    total_loss / total_count,
                )
            )
            total_acc, total_count, total_loss = 0, 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    total_loss = 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_loss += loss
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    # return total_acc / total_count
    return total_loss / total_count


# %%
# Train model
# Split train dataset into train/validation

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = torch.utils.data.dataset.random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = torch.utils.data.DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = torch.utils.data.DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)

    # if total_accu is not None and total_accu > accu_val:
    #     scheduler.step()
    # else:
    #     total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

# %%
# test performance
print("Checking the results of train dataset.")
accu_train = evaluate(train_dataloader)
print("train accuracy {:8.3f}".format(accu_train))

print("Checking the results of valid dataset.")
accu_valid = evaluate(valid_dataloader)
print("valid accuracy {:8.3f}".format(accu_valid))

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

# %%
def predict_text(text: str):
    with torch.no_grad():
        log_probit = model(torch.tensor(text_pipeline(text)), torch.tensor([0]))
    proba = torch.nn.Softmax(dim=1)(log_probit)[0]

    print(text)
    print("--------------")

    for key, val in train_dataset.sentiment_map.items():
        print(key, proba[val].item())


predict_text(train_dataset[20][0])
predict_text(test_dataset[30][0])

# %%
# pytorch lightning training

# create RNN module with dense layers
import pytorch_lightning as pl


class TextClassificationModelPL(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_class,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()
        self.train_log_error = []
        self.val_log_error = []
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
        )
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.learning_rate,
        # )
        return optimizer

    def training_step(self, batch, batch_idx):
        (label, text, offsets) = batch

        output = self.forward(text, offsets)
        loss = self.loss(output, label)

        self.train_log_error.append(loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        (label, text, offsets) = batch

        output = self.forward(text, offsets)
        loss = self.loss(output, label)

        self.log("val_loss", loss, on_epoch=True)
        self.val_log_error.append(loss.detach().numpy())
        return loss


from torchinfo import summary

num_class = len(train_dataset.sentiment_map)
vocab_size = len(vocab)
embed_size = 16
model = TextClassificationModelPL(vocab_size, embed_size, num_class).to(device)

tokens = [7, 43, 67, 7, 43, 68]
offsets = [0, 3]
summary(model, input_data=(torch.tensor(tokens), torch.tensor(offsets)))

# %%
# train
import matplotlib.pyplot as plt
import numpy as np

# fit network
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[
        pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=30)
    ],
)
# trainer.fit(model, dataloader_train, dataloader_eval)
trainer.fit(model, train_dataloader, valid_dataloader)


# %%
# plot training loss
def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


plt.plot(moving_average(np.array(model.train_log_error)))
plt.plot(moving_average(np.array(model.val_log_error)))

# %%
# test performance
print("Checking the results of train dataset.")
accu_train = evaluate(train_dataloader)
print("train accuracy {:8.3f}".format(accu_train))

print("Checking the results of valid dataset.")
accu_valid = evaluate(valid_dataloader)
print("valid accuracy {:8.3f}".format(accu_valid))

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

# %%
def predict_text(text: str):
    with torch.no_grad():
        log_probit = model(torch.tensor(text_pipeline(text)), torch.tensor([0]))
    proba = torch.nn.Softmax(dim=1)(log_probit)[0]

    print(text)
    print("--------------")

    for key, val in train_dataset.sentiment_map.items():
        print(key, proba[val].item())


import random


predict_text(train_dataset[random.randint(0, len(train_dataset))][0])
predict_text(test_dataset[random.randint(0, len(test_dataset))][0])
