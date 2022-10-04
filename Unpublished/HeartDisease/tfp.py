# %% [markdown]
# # Heart Disease Data Exploration
#

# %%
import os
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

# %% Clean data
# No missing/inf values
print("Missing:\n", heart_data.isna().sum(), "\n")
print("Inf:\n", (np.abs(heart_data) == np.inf).sum(), "\n")

heart_data.describe()

# %% Train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    heart_data.drop(columns="target"), heart_data["target"]
)

# standardise data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)

# %% Logistic regression
from sklearn.linear_model import LogisticRegression

model_logreg = LogisticRegression(max_iter=1000)
model_logreg = model_logreg.fit(X=x_train_s, y=y_train)
y_train_est = model_logreg.predict(x_train_s)
y_test_est = model_logreg.predict(x_test)

model_logreg.coef_
model_logreg.intercept_
model_logreg.get_params()

# %% Explainable gbm
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

model_ebm = ExplainableBoostingClassifier()
model_ebm.fit(X=x_train_s, y=y_train)

y_train_est = model_ebm.predict(x_train_s)
y_test_est = model_ebm.predict(x_test)

ebm_global = model_ebm.explain_global(name="EBM")
show(ebm_global)

# %% NN
import tensorflow as tf

model_nn = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        # tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model_nn.compile(
    loss="categorical_crossentropy",
    # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    optimizer=tf.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

# standardise data, one hot encode output
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False).fit(y_train.to_numpy().reshape(-1, 1))

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, monitor="val_loss", restore_best_weights=True
)

history_nn = model_nn.fit(
    x_train_s,
    ohe.transform(y_train.to_numpy().reshape(-1, 1)),
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
model_nn.summary()
plt.plot(history_nn.history["loss"])
plt.plot(history_nn.history["val_loss"])

y_train_est = model_nn.predict(x_train_s).argmax(axis=1)
y_test_est = model_nn.predict(x_test_s).argmax(axis=1)

model_nn.evaluate(
    x_test_s, ohe.transform(y_test.to_numpy().reshape(-1, 1))
)
# %% NN - TFP, beta distribution
import tensorflow as tf
import tensorflow_probability as tfp

model_nn_beta = tf.keras.Sequential(
    [
        # tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(2),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.BetaBinomial(
                total_count=1,
                concentration1=1e-3 + tf.math.softplus(0.05 * t[..., :1]),
                concentration0=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
            )
        ),
    ]
)


def negloglik(y, distr):
    return -distr.log_prob(y)


model_nn_beta.compile(
    loss=negloglik,
    # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    # optimizer=tf.optimizers.SGD(learning_rate=0.05),
    optimizer=tf.optimizers.SGD(learning_rate=0.1),
)

# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, monitor="val_loss", restore_best_weights=True
)

history_nn = model_nn_beta.fit(
    x_train_s,
    y_train.to_numpy().astype(float).reshape(-1, 1),
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
model_nn_beta.summary()
plt.plot(history_nn.history["loss"])
plt.plot(history_nn.history["val_loss"])


def predict_fcn_nn_beta(x):
    return np.concatenate(
        [model_nn_beta(x).prob(0.0), model_nn_beta(x).prob(1.0),], axis=1,
    )


y_train_est = predict_fcn_nn_beta(x_train_s).argmax(axis=1)
y_test_est = predict_fcn_nn_beta(x_test_s).argmax(axis=1)

model_nn_beta.evaluate(
    x_test_s, y_test.to_numpy().astype(float).reshape(-1, 1)
)

# %% Check individual outputs
idx = 4
test_point = x_test.iloc[[idx],:]
test_point
y_test_est = model_nn_beta(scaler.transform(test_point))
x_t = np.linspace(0, 1, 50)
y_t = y_test_est.prob(x_t).numpy().flatten()

plt.plot(x_t, y_t)
plt.show()

# %% Results - Train
from interpret.perf import ROC

blackbox_logreg_perf = ROC(model_logreg.predict_proba).explain_perf(
    x_train_s, y_train, name="SkLearnLog"
)
show(blackbox_logreg_perf)
blackbox_ebm_perf = ROC(model_ebm.predict_proba).explain_perf(
    x_train_s, y_train, name="EBM"
)
show(blackbox_ebm_perf)
blackbox_nn_perf = ROC(model_nn.predict).explain_perf(
    x_train_s, y_train, name="NN"
)
show(blackbox_nn_perf)
blackbox_nn_beta_perf = ROC(predict_fcn_nn_beta).explain_perf(
    x_train_s, y_train.astype(float), name="NN_beta"
)
show(blackbox_nn_beta_perf)

# %% Test
blackbox_logreg_perf = ROC(model_logreg.predict_proba).explain_perf(
    x_test_s, y_test, name="SkLearnLog"
)
show(blackbox_logreg_perf)
blackbox_ebm_perf = ROC(model_ebm.predict_proba).explain_perf(
    x_test_s, y_test, name="EBM"
)
show(blackbox_ebm_perf)
blackbox_nn_perf = ROC(model_nn.predict_proba).explain_perf(
    x_test_s, y_test, name="NN"
)
show(blackbox_nn_perf)
blackbox_nn_beta_perf = ROC(predict_fcn_nn_beta).explain_perf(
    x_test_s, y_test.astype(float), name="NN_beta"
)
show(blackbox_nn_beta_perf)

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predict_fcn_nn_beta(x_test_s).argmax(axis=1))
accuracy_score(y_train, predict_fcn_nn_beta(x_train_s).argmax(axis=1))
