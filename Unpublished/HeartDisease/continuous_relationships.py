# %% [markdown]
# # Probabilistic regressions on continuous variables
#

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import validation

plt.style.use("seaborn-whitegrid")


# %%
# Fetch data
dir_path = Path(__file__).parent
heart_data = pd.read_csv(dir_path / "data" / "heart.csv")
# data retrieved from: https://www.kaggle.com/ronitf/heart-disease-uci
print(heart_data.shape)
heart_data.head()

# %% Train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    heart_data.drop(columns="trestbps"), heart_data["trestbps"]
)

# standardise data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

scaler.inverse_transform(x_train_s)

# %% Linear regression
from sklearn.linear_model import LinearRegression

model_linreg = LinearRegression()
model_linreg = model_linreg.fit(X=x_train_s, y=y_train)
y_train_est = model_linreg.predict(x_train_s)
y_test_est = model_linreg.predict(x_test)

model_linreg.coef_
model_linreg.intercept_
model_linreg.get_params()

# %% Explainable gbm
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

model_ebm = ExplainableBoostingRegressor()
model_ebm.fit(X=x_train_s, y=y_train)

y_train_est = model_ebm.predict(x_train_s)
y_test_est = model_ebm.predict(x_test)

ebm_global = model_ebm.explain_global(name="EBM")
show(ebm_global)

# %% NN
import tensorflow as tf

model_nn = tf.keras.Sequential(
    [
        # tf.keras.layers.Dense(10),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1),
    ]
)

model_nn.compile(
    loss="mse",
    # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=["mae"],
)

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, monitor="val_loss", restore_best_weights=True
)

history_nn = model_nn.fit(
    x_train_s,
    y_train.to_numpy().reshape(-1, 1),
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
model_nn.summary()
plt.plot(history_nn.history["loss"])
plt.plot(history_nn.history["val_loss"])


def predict_fcn_nn(x):
    return model_nn(x).numpy().flatten()


y_train_est = predict_fcn_nn(x_train_s)
y_test_est = predict_fcn_nn(x_test_s)

model_nn.evaluate(x_test_s, y_test.to_numpy().reshape(-1, 1))
# %% NN - TFP, beta distribution
import tensorflow as tf
import tensorflow_probability as tfp

model_nn_normal = tf.keras.Sequential(
    [
        # tf.keras.layers.Dense(10),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(2),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., :1],
                scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
            )
        ),
    ]
)


def negloglik(y, distr):
    return -distr.log_prob(y)


model_nn_normal.compile(
    loss=negloglik,
    # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    # optimizer=tf.optimizers.SGD(learning_rate=0.05),
    optimizer=tf.optimizers.SGD(learning_rate=0.1),
)

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, monitor="val_loss", restore_best_weights=True
)

history_nn = model_nn_normal.fit(
    x_train_s,
    y_train.to_numpy().astype(float).reshape(-1, 1),
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
model_nn_normal.summary()
plt.plot(history_nn.history["loss"])
plt.plot(history_nn.history["val_loss"])


def predict_fcn_nn_normal(x):
    return model_nn_normal(x).mean().numpy().flatten()


y_train_est = predict_fcn_nn_normal(x_train_s)
y_test_est = predict_fcn_nn_normal(x_test_s)

# %% Check individual outputs
idx = 10
test_point = x_test.iloc[[idx], :]
test_point
y_test_est = model_nn_normal(scaler.transform(test_point))
x_t = np.linspace(0, 200, 50)
y_t = y_test_est.prob(x_t).numpy().flatten()

plt.plot(x_t, y_t)
plt.show()

# %% Results
from interpret.perf import RegressionPerf
for (name, x, y) in [("train", x_train_s, y_train), ("test", x_test_s, y_test)]:
    blackbox_linreg_perf = RegressionPerf(model_linreg.predict).explain_perf(
        x, y, name=f"SkLearnLog_{name}"
    )
    show(blackbox_linreg_perf)
    blackbox_ebm_perf = RegressionPerf(model_ebm.predict).explain_perf(
        x, y, name=f"EBM_{name}"
    )
    show(blackbox_ebm_perf)
    blackbox_nn_perf = RegressionPerf(predict_fcn_nn).explain_perf(
        x, y, name=f"NN_{name}"
    )
    show(blackbox_nn_perf)
    blackbox_nn_beta_perf = RegressionPerf(predict_fcn_nn_normal).explain_perf(
        x, y, name=f"NN_normal_{name}"
    )
    show(blackbox_nn_beta_perf)


# %%
for (x, y) in [(x_train_s, y_train), (x_test_s, y_test)]:
    plt.plot(model_linreg.predict(x), y, ".", label="SkLearnLog")
    plt.plot(model_ebm.predict(x), y, ".", label="EBM")
    plt.plot(predict_fcn_nn(x), y, ".", label="NN")
    plt.plot(predict_fcn_nn_normal(x), y, ".", label="NN_normal")
    plt.plot(
        np.linspace(np.min(y), np.max(y), 100),
        np.linspace(np.min(y), np.max(y), 100),
        ".",
    )

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predict_fcn_nn_normal(x_test_s).argmax(axis=1))
accuracy_score(y_train, predict_fcn_nn_normal(x_train_s).argmax(axis=1))
