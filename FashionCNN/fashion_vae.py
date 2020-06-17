# %%
"""
Build conv net for fashion mnist

"""

# %%
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


tfk = tf.keras
tfkl = tf.keras.layers

# %%
datasets, datasets_info = tfds.load(
    name="fashion_mnist", with_info=True, as_supervised=False, data_dir="data"
)
datasets_info.features['label'].names

# %%
fig = tfds.show_examples(datasets["train"], datasets_info)

# %% convert labels to one hot encoding
from tensorflow.keras.utils import to_categorical

# get first 1000 images
train_data = datasets["train"].batch(1000)
train_data = iter(train_data)

batch_1 = next(train_data)
X_train = batch_1["image"].numpy()
Y_train = to_categorical(batch_1["label"].numpy())

batch_2 = next(train_data)
X_test = batch_2["image"].numpy()
Y_test = to_categorical(batch_2["label"].numpy())

input_shape = datasets_info.features["image"].shape
num_classes = datasets_info.features["label"].num_classes

# %%
model = tfk.Sequential(
    [
        tfk.layers.Convolution2D(
            32, kernel_size=5, input_shape=input_shape, activation=tfk.activations.relu
        ),
        tfk.layers.MaxPooling2D(pool_size=2),
        tfk.layers.Convolution2D(
            32, kernel_size=4, activation=tfk.activations.relu, padding='same'
        ),
        tfk.layers.MaxPooling2D(pool_size=2),
        tfk.layers.Convolution2D(
            32, kernel_size=3, activation=tfk.activations.relu
        ),
        # tfk.layers.MaxPooling2D(pool_size=5),
        tfk.layers.Flatten(),
        tfk.layers.Dropout(rate=0.2),
        tfk.layers.Dense(num_classes),
        tfk.layers.Activation("softmax"),
    ]
)


# %%
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.summary()

# %%
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=64,
    epochs=100,
    verbose=1
)

# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.legend()
plt.show()



# %% Predict 5 images from test set
n_images = 10
test_images = X_test[np.random.randint(0, high=X_test.shape[0], size=n_images)]
predictions = model(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    pred_name = datasets_info.features['label'].names[np.argmax(predictions.numpy()[i])]
    pred_conf = predictions.numpy()[i][np.argmax(predictions.numpy()[i])]
    print(f"{pred_name}: {pred_conf:.2f}")


# %% Predict whole test set
all_predictions = model(X_test)
pred_conf = np.max(all_predictions, axis=1)
pred_label = np.argmax(all_predictions, axis=1)
true_label = np.argmax(Y_test, axis=1)


# %% Uncertain images
idx = np.argsort(pred_conf)
for i in idx[:10]:
    plt.imshow(np.reshape(X_test[i], [28, 28]), cmap='gray')
    predictions = model(X_test[np.newaxis, i])
    plt.show()
    pred_name = datasets_info.features['label'].names[np.argmax(predictions.numpy())]
    pred_conf = predictions.numpy().flatten()[np.argmax(predictions.numpy())]
    print(f"{pred_name}: {pred_conf:.2f}")



# %%
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(true_label, pred_label, normalize='true')

plt.imshow(conf_matrix, cmap='gray')
plt.show()

# %%
# Commonly getting confused between T-shirts/shirts and pullovers/coats

