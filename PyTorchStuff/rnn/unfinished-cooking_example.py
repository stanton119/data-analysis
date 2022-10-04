# %%
import numpy as np


# %%
class OutputRotation:
    """
    Every step we move to the next output class.
    """

    def __init__(self) -> None:
        self.current_class = 0
        self.n_output_classes = 3

    def get_next_class(self, x: int) -> int:
        self.current_class = (self.current_class + 1) % self.n_output_classes
        return self.current_class

    def get_ohe(self) -> np.array:
        out = np.zeros((1, self.n_output_classes))
        out[0, self.current_class] = 1
        return out

    def get_ohe_multiple_samples(self, x: np.array) -> np.array:
        out = []
        for val in x:
            self.get_next_class(val)
            out.append(self.get_ohe())
        return np.array(out)


class ConditionalOutputRotation(OutputRotation):
    """
    If input value is 1, repeat previous output.
    Otherwise move to next output class.
    """

    def get_next_class(self, x: int) -> int:
        if x == 1:
            self.current_class = (self.current_class + 1) % self.n_output_classes
        return self.current_class


# %%
# get training data
n_data_points = 1000
rng = np.random.default_rng()
x = rng.integers(0, 2, (n_data_points, 1))

rotation_obj = OutputRotation()
y = rotation_obj.get_ohe_multiple_samples(x)

# %%
x[:5]
y[:5]

# %%


np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

import matplotlib.pyplot as plt

plt.plot(data)

# %%
# train model
import torch

rnn = torch.nn.RNN(
    input_size=0, hidden_size=rotation_obj.n_output_classes, num_layers=1, bias=True, batch_first=False
)


# %%
# create dataloaders
input = torch.Tensor([[[]]])

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(df_train_x.to_numpy()[:train_idx, :].astype(np.float32)),
    torch.tensor(df_train_y.to_numpy()[:train_idx, :].astype(np.float32)),
)
eval_dataset = torch.utils.data.TensorDataset(
    torch.tensor(df_train_x.to_numpy()[train_idx + 1 :, :].astype(np.float32)),
    torch.tensor(df_train_y.to_numpy()[train_idx + 1 :, :].astype(np.float32)),
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=2000,
)
# %%
import pytorch_lightning as pl


class RNNNetwork(pl.LightningModule):
    def __init__(self, n_inputs: int = 1, n_output_classes: int = 5):
        super().__init__()
        self.train_log = []

        self.model = torch.nn.RNN(
            input_size=n_inputs, hidden_size=n_output_classes, num_layers=1, bias=True, batch_first=False
        )
        self.loss = torch.nn.CrossEntropyLoss()
        # self.loss = torch.nn.NLLLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_est = self(x)
        loss = self.loss(y_est, y)

        self.log("loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss


model = RNNNetwork(n_inputs=0, n_output_classes=rotation_obj.n_output_classes)
print(model.summarize())
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, dataloader_train)


# %%
# one to many generation model
# seed with the current output class, generate a sequence of the next outputs
# generate multiple outputs
input = torch.Tensor([[[]]])
hn = None
outputs = []
for idx in range(10):
    output, hn = rnn(input, hn)
    outputs.append(hn.detach().numpy().flatten())
outputs = np.array(outputs)
outputs

hn[0][0][0].item()



# %%

# %%
import torch

rnn = torch.nn.RNN(10, 20, 2)
rnn
y
output, hn = rnn(input, hn)

# generate multiple outputs
hn = None
outputs = []
for idx in range(10):
    output, hn = rnn(input, hn)
    outputs.append(hn.detach().numpy().flatten())

dir(hn)
hn.float()[0][0][0]
hn.detach().numpy().flatten()

# All hidden states associated with a sequence, for all sequences in the batch
# Just the very last hidden state for a sequence, for all sequences in the batch
# In [ ]:
out_all, out_last = rnn(input)

# %%
rnn = torch.nn.RNN(10, 20, 2)
rnn
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
output.shape
hn.shape
