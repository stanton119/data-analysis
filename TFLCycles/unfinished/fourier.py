





# %% Time results using fft
import numpy as np
import scipy.fftpack

# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N * T, N)
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

plt.plot(x, y, ".")

fig, ax = plt.subplots()
ax.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
plt.show()

temp.loc[:, ["datetimeint", "count"]]
norm_count = (temp["count"] - temp["count"].mean()).to_numpy()
norm_count.shape
yf = scipy.fftpack.fft(norm_count,)
xf = temp["datetimeint"]
temp["datetimeint"].max() - temp["datetimeint"].min()
# np.linspace(0.0, 1.0/(2.0*T), int(N/2))

fig, ax = plt.subplots()
ax.plot(xf, 2.0 / N * np.abs(yf))
ax.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
plt.show()

plt.plot(temp["datetimeint"].diff())

temp["datetimeint"].diff()[1]

temp["datetimeint"][:2]

Y = np.fft.fft(norm_count)
freq = np.fft.fftfreq(len(norm_count), temp["datetimeint"].diff()[1])

plt.figure()
plt.plot(freq, np.abs(Y), ".")
plt.figure()
plt.plot(freq, np.angle(Y))
plt.show()

# %% [markdown]
# Convert to jupyter notebook -> Export current (no output)
# # Convert to markdown file
# `jupyter nbconvert data_proc.ipynb --to markdown`
