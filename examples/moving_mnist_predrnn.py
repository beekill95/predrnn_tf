# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     notebook_metadata_filter: title
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import keras
from keras import layers, utils, losses, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from predrnn_tf.layers import SpatialTemporalLSTMCell, StackedSpatialTemporalLSTMCell

# %% [markdown]
# # Moving MNIST with PredRNN
# ## Dataset
#
# Following [Keras's tutorial](https://keras.io/examples/vision/conv_lstm/).

# %%
# We'll work with this many data.
# Adjust this if you want to work with more data.
data_size = 1000

# Download and load the dataset.
fpath = utils.get_file(
    "moving_mnist.npy",
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
)
# Data will have shape of (timeframes, samples, H, W).
dataset = np.load(fpath)
print(f'Before swapping axes: {dataset.shape=}')

# Thus, we swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(dataset, 0, 1)
print(f'After swapping axes: {dataset.shape=}')

# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:data_size, ...]
print(f'After dataset selection: {dataset.shape=}')
# Add a channel dimension since the images are grayscale.
dataset = np.expand_dims(dataset, axis=-1)

# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# %% [markdown]
# ### Visualization

# %%
# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# Plot each of the sequential images for one random data example.
data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
for idx, ax in enumerate(axes.flat):
    ax.imshow(np.squeeze(train_dataset[data_choice][idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Print information and display the figure.
print(f"Displaying frames for example {data_choice}.")
plt.show()

# %% [markdown]
# ## PredRNN
# ### Model Construction

# %%
# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We'll use a stacked of three spatial temporal cells.
# Also, to be comparable with [Keras's tutorial](https://keras.io/examples/vision/conv_lstm/),
# we'll also use 64 filters.
cells = StackedSpatialTemporalLSTMCell([
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
])
rnn = layers.RNN(cells, return_sequences=True)

out = layers.Conv3D(
    filters=1,
    kernel_size=(3, 3, 3),
    activation="sigmoid",
    padding="same",
)

# Construct the model.
x = rnn(inp)
x = out(x)
predrnn = keras.Model(inp, x)
predrnn.summary()

# %% [markdown]
# ### Model Training

# %%
predrnn.compile(
    loss=losses.binary_crossentropy,
    optimizer=optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

# Fit the model to the training data.
predrnn.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
)

# %% [markdown]
# ## Results
# ### Frame Prediction Visualizations

# %%
# Select a random example from the validation dataset.
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = predrnn.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()
