
# %% Load data
# Use tensorflow datasets to stream data from disk to prevent whole dataset being stored in memory

# Construct a tf.data.Dataset
(ds_train, ds_test), ds_info = tfds.load(
    name="mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir=os.path.join(os.getcwd(), "data"),
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

tfds.as_numpy(ds_train)

ds_info
ds_train['image']
for elem in ds_train:
  print(elem.numpy())

len(elem)
elem[0].shape
elem[1].shape

train_horses, train_zebras = dataset['trainA'], dataset['trainB']

#load dataset in to numpy array 
ds_train.
.batch(1000).get_next()
train_A=train_horses.batch(1000).make_one_shot_iterator().get_next()[0].numpy()



ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_train.



# # Build your input pipeline
# ds_train = ds_train.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# for example in ds.take(1):
#     image, label = example["image"], example["label"]

# %%
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# %%

history = model.fit(
    ds_train, epochs=6, validation_data=ds_test,
)

history = model.fit(
    ds_train, epochs=6, validation_data=ds_test, batch_size=64
)
