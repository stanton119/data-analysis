"""
sentiment analysis
text classification

https://www.youtube.com/watch?v=Y_hzMnRXjhI

dataset:
kaggle
https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home

News headlines from The Onion = sarcastic
News headlines from Huffington Post = not sarcastic

Dataset is processed with padding to get training matrix
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


vocab = build_vocab_from_iterator(
    yield_tokens(train_iter), specials=["<unk>"], min_freq=3
)
vocab.set_default_index(vocab["<unk>"])

vocab(["here", "is", "an", "example"])
vocab(tokenizer(x))

len(vocab)

# %%
import numpy as np

max_seq_length = 100
cost_fcn = "bin"


def text_pipeline(text):
    text_tokens = vocab(tokenizer(text[:max_seq_length]))
    padded_tokens = np.zeros((1, max_seq_length), dtype=np.int64)
    padded_tokens[0, 0 : len(text_tokens)] = text_tokens
    return torch.from_numpy(padded_tokens)


def label_pipeline(label):
    return torch.tensor([label])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    """Transform batch from dataset for the text/label pipelines.
    Creates numpy array of text tokens and labels.
    Use only the first N word token"""

    token_list = []
    label_list = []
    for _text, _label in batch:
        token_list.append(text_pipeline(_text))
        label_list.append(label_pipeline(_label))

    if cost_fcn == "bin":
        # class probs
        labels = torch.cat(label_list).type(torch.DoubleTensor)[:, None]
    else:
        # class indices
        labels = torch.cat(label_list).type(torch.int64)
    text_tokens = torch.cat(token_list, dim=0)

    return text_tokens.to(device), labels.to(device)


# %%
BATCH_SIZE = 64

num_train = int(len(train_dataset) * 0.8)
split_train_, split_valid_ = torch.utils.data.dataset.random_split(
    train_dataset,
    [num_train, len(train_dataset) - num_train],
    generator=torch.Generator().manual_seed(42),
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

text_tokens, labels = next(iter(train_dataloader))
text_tokens, labels

# %%
# pytorch lightning training
import pytorch_lightning as pl


class TextClassificationModelPL(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_class,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=embed_size, mode="mean"
        )

        self.fc1 = torch.nn.Linear(embed_size, 24)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(24, num_class)

        self.train_log_error = []
        self.val_log_error = []
        self.learning_rate = learning_rate
        self.loss = torch.nn.BCEWithLogitsLoss()  # requires one output column
        # self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, text_tokens):
        embedded = self.embedding(text_tokens)
        fc1ed = self.fc1(embedded)
        relued = self.relu(fc1ed)
        return self.fc2(relued)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        (text_tokens, labels) = batch

        output = self.forward(text_tokens)
        loss = self.loss(output, labels)

        self.train_log_error.append(loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        (text_tokens, labels) = batch

        output = self.forward(text_tokens)
        loss = self.loss(output, labels)

        self.log("val_loss", loss, on_epoch=True)
        self.val_log_error.append(loss.detach().numpy())
        return loss


from torchinfo import summary

num_class = 1 if cost_fcn == "bin" else 2
vocab_size = len(vocab)
embed_size = 16
model = TextClassificationModelPL(
    vocab_size=vocab_size, embed_size=embed_size, num_class=num_class
).to(device)

tokens = [[7, 43, 67, 7, 43, 68], [1, 2, 3, 4, 5, 6]]
summary(model, input_data=(torch.tensor(tokens)))

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
import random


def predict_text(text: str):
    with torch.no_grad():
        log_probit = model(text_pipeline(text))
    if cost_fcn == "bin":
        proba = torch.nn.Sigmoid()(log_probit).item()
    else:
        proba = torch.nn.Softmax(dim=1)(log_probit)[0][1].item()

    print(text)
    print("p(is sarcastic)= ", proba)


def check_prediction(data_tuple):
    predict_text(data_tuple[0])
    print("is sarcastic: ", data_tuple[1])
    print("--------------")


check_prediction(split_train_[random.randint(0, len(split_train_))])
check_prediction(train_dataset[random.randint(0, len(train_dataset))])
check_prediction(test_dataset[random.randint(0, len(test_dataset))])


# %%
# test performance
def evaluate(dataloader):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataloader.dataset,
        batch_size=len(dataloader.dataset),
        collate_fn=collate_batch,
    )
    (text_tokens, labels) = next(iter(dataloader))

    with torch.no_grad():
        output = model.forward(text_tokens)
    loss = model.loss(output, labels)

    if cost_fcn == "bin":
        # assume 0.5 prob threshold
        output = torch.nn.Sigmoid()(output) > 0.5
        accuracy = ((output == labels).sum() / len(labels)).item()
    else:
        accuracy = ((output.argmax(1) == labels).sum() / len(labels)).item()
    print(f"train loss, accuracy = {loss:8.3f}, {accuracy:8.3f}")

    return loss, accuracy


print("Checking the results of train dataset.")
evaluate(train_dataloader)

print("Checking the results of valid dataset.")
evaluate(valid_dataloader)

print("Checking the results of test dataset.")
evaluate(test_dataloader)

# %%
# PR curves

train_dataset

log_probit = model(text_pipeline(text))
proba = torch.nn.Softmax(dim=1)(log_probit)[0]
