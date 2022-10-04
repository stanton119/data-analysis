# %%
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
# %%
# generate noisy sinusoid

n = 1000
t = np.linspace(0, 4, n)
bias = 0.1
amp = 1
freq = 0.5
phase = 0.0 * np.pi
y = amp * np.sin(2 * np.pi * freq * t + phase) + bias

noise_amp = 0.2
z = y + noise_amp * np.random.randn(n)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, z, label="observed")
ax.plot(t, y, label="true")
fig.legend()
plt.show()

# %%
class SinModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.train_log_error = []
        self.train_log_params = []
        self.learning_rate = learning_rate

        self._amp = torch.nn.Parameter(torch.randn(()))
        self._freq = torch.nn.Parameter(torch.randn(()))
        self._phase = torch.nn.Parameter(torch.randn(()))
        self.bias = torch.nn.Parameter(torch.tensor(np.mean(z)))

    @property
    def amp(self):
        return torch.nn.functional.softplus(self._amp)  # ensure amp > 0

    @property
    def freq(self):
        return torch.nn.functional.softplus(self._freq)  # ensure freq > 0

    @property
    def phase(self):
        return np.pi * torch.sigmoid(self._phase)  # ensure freq > 0

    def forward(self, x):
        y = (
            self.amp * torch.sin(2 * np.pi * self.freq * x + self.phase)
            + self.bias
        )
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = torch.nn.MSELoss()(x_hat, y)

        self.log("loss", loss)
        self.train_log_error.append(loss.detach().numpy())
        self.train_log_params.append(
            [self.amp, self.freq, self.phase, self.bias]
        )
        return loss


# %%

# create dataloader from whole dataset
dataloader = torch.utils.data.DataLoader(dataset=[t, z], batch_size=n)
t_batch, z_batch = next(iter(dataloader))

# %%
model = SinModel(learning_rate = 1e-2)
print(model.summarize())
# %%
trainer = pl.Trainer(max_epochs=int(1e3), auto_lr_find=False)
# trainer.tune(model, dataloader)
trainer.fit(model, dataloader)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(model.train_log_error)
plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    torch.tensor(model.train_log_params).detach().numpy(),
    label=["amp", "freq", "phase", "bias"],
)
fig.legend()
plt.show()

# %%
z_est = model(t_batch)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, z, label="observed")
ax.plot(t, y, label="true")
ax.plot(t, z_est.detach(), label="est")
fig.legend()
plt.show()
# %%
model.freq
model.amp

# %%
# fft
n = 1000
t = np.linspace(0, 4, n)
bias = 0.0
amp = 1
freq = 1.1
phase = 0.0 * np.pi
y = amp * np.sin(2 * np.pi * freq * t + phase) + bias

noise_amp = 0.0
z = y + noise_amp * np.random.randn(n)

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(t, z, label="observed")
# ax.plot(t, y, label="true")
# fig.legend()
# plt.show()

#
y_fft = np.fft.fft(y-np.mean(y))

# plt.plot(np.abs(y_fft))
plt.plot(np.abs(y_fft[:200]))

# np.abs(y_fft[:10])
# np.abs(y_fft[-10:])
# %%
