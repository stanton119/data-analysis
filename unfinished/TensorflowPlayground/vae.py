# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# %%
if 1:
    datasets, datasets_info = tfds.load(
        name="mnist",
        with_info=True,
        as_supervised=False,
        data_dir=os.path.join(os.getcwd(), "data"),
    )
else:
    datasets, datasets_info = tfds.load(
        name="fashion_mnist",
        with_info=True,
        as_supervised=False,
        data_dir=os.path.join(os.getcwd(), "data"),
    )

def _preprocess(sample):
    image = (
        tf.cast(sample["image"], tf.float32) / 255.0
    )  # Scale to unit interval.
    image = image < tf.random.uniform(tf.shape(image))  # Randomly binarize.
    return image, image


train_dataset = (
    datasets["train"]
    .map(_preprocess)
    .batch(256)
    .prefetch(tf.data.experimental.AUTOTUNE)
    .shuffle(int(10e3))
)
eval_dataset = (
    datasets["test"]
    .map(_preprocess)
    .batch(256)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# %%
input_shape = datasets_info.features["image"].shape
encoded_size = 4 # 16
base_depth = 32

# %%
prior = tfd.Independent(
    tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1
)


# %%
encoder = tfk.Sequential(
    [
        tfkl.InputLayer(input_shape=input_shape),
        tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        tfkl.Conv2D(
            base_depth,
            5,
            strides=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2D(
            base_depth,
            5,
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2D(
            2 * base_depth,
            5,
            strides=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2D(
            2 * base_depth,
            5,
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2D(
            4 * encoded_size,
            7,
            strides=1,
            padding="valid",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Flatten(),
        tfkl.Dense(
            tfpl.MultivariateNormalTriL.params_size(encoded_size),
            activation=None,
        ),
        tfpl.MultivariateNormalTriL(
            encoded_size,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior),
        ),
    ]
)

# %%
decoder = tfk.Sequential(
    [
        tfkl.InputLayer(input_shape=[encoded_size]),
        tfkl.Reshape([1, 1, encoded_size]),
        tfkl.Conv2DTranspose(
            2 * base_depth,
            7,
            strides=1,
            padding="valid",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2DTranspose(
            2 * base_depth,
            5,
            strides=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2DTranspose(
            2 * base_depth,
            5,
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2DTranspose(
            base_depth,
            5,
            strides=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2DTranspose(
            base_depth,
            5,
            strides=2,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2DTranspose(
            base_depth,
            5,
            strides=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        ),
        tfkl.Conv2D(
            filters=1, kernel_size=5, strides=1, padding="same", activation=None
        ),
        tfkl.Flatten(),
        tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
    ]
)

# %%
vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

# %%
negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss=negloglik)

vae.summary()

# %%
history = vae.fit(train_dataset, epochs=5, validation_data=eval_dataset)
plt.plot(history.history['loss'])

# %%
# We'll just examine ten random digits.
x = next(iter(eval_dataset))[0][:10]
xhat = vae(x)
assert isinstance(xhat, tfd.Distribution)

# %%
def display_imgs(x, y=None):
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    plt.ioff()
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(n, 1))
    if y is not None:
        fig.suptitle(np.argmax(y, axis=1))
    for i in range(n):
        axs.flat[i].imshow(x[i].squeeze(), interpolation="none", cmap="gray")
        axs.flat[i].axis("off")
    plt.show()
    plt.close()
    plt.ion()


# %%
print("Originals:")
display_imgs(x)

print("Decoded Random Samples:")
display_imgs(xhat.sample())

print("Decoded Modes:")
display_imgs(xhat.mode())

print("Decoded Means:")
display_imgs(xhat.mean())


# %%
# Now, let's generate ten never-before-seen digits.
z = prior.sample(10)
xtilde = decoder(z)
assert isinstance(xtilde, tfd.Distribution)


# %%
print("Randomly Generated Samples:")
display_imgs(xtilde.sample())

print("Randomly Generated Modes:")
display_imgs(xtilde.mode())

print("Randomly Generated Means:")
display_imgs(xtilde.mean())

# %% Encoded space is 16 dimensions, input/output = 28x28
encoder(x).shape
encoder(x).mean()


prior.sample(1000).numpy().mean(axis=0)

xtilde = decoder(np.random.normal(size=(2, 16)))
