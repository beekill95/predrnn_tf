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

from predrnn_tf.layers import (
    SpatialTemporalLSTMCell,
    StackedSpatialTemporalLSTMCell,
    PredRNNCell,
)
from predrnn_tf.scheduled_sampling import LinearScheduledSamplingLayer
from predrnn_tf.callbacks import UpdateReversedScheduleSamplingProbCallback

# %% tags=["parameters"]
# Notebook's parameters, to be run with `papermill`.
model_save_path = './saved_models/moving_mnist_predrnn_all'

# Model to be run in this notebook,
# two possible options are: 'predrnn' and 'convlstm'
#model_type = 'predrnn'
model_type = 'convlstm'

# Whether should we use reverse sampling or not.
# Only possible with predrnn model.
use_reversed_sampling = True

# %%
print(f'Saving model at {model_save_path=}')

# %% [markdown]
# # Moving MNIST
#
# In this experiment,
# we'll try different combinations of ConvLSTM/PredRNN
# and with/without reversed scheduled sampling to see
# how these perform on moving MNIST dataset.
#
# ## Dataset
#
# Following [Keras's tutorial](https://keras.io/examples/vision/conv_lstm/).

# %%
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

print(f'After dataset selection: {dataset.shape=}')
# Add a channel dimension since the images are grayscale.
dataset = np.expand_dims(dataset, axis=-1)

# Normalize data into 0-1 range.
dataset = dataset / 255.

# Split into train and test dataset.
# Here, we'll use the last 2000 samples for testing.
test_dataset = dataset[-2000:, ...]
dataset = dataset[:-2000, ...]

# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)
x_test, y_test = create_shifted_frames(test_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))

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
# ## Model Construction
# ### ConvLSTM

# %%
def create_conv_lstm_model(input_shape: tuple):
    # Copied and modified from Keras's tutorial:
    # https://keras.io/examples/vision/conv_lstm/
    inp = layers.Input(shape=input_shape)

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="sigmoid",
    )(x)

    return keras.models.Model(inp, x)


# %% [markdown]
# ### PredRNN

# %%
def create_predrnn_model(input_shape: tuple, use_reversed_sampling: bool = False):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=input_shape)

    # We'll use a stacked of three spatial temporal cells.
    # Also, to be comparable with [Keras's tutorial](https://keras.io/examples/vision/conv_lstm/),
    # we'll also use 64 filters.
    cells = StackedSpatialTemporalLSTMCell([
        SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
        SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
        SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
    ])
    predrnn_cell = PredRNNCell(
        cells,
        layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation="sigmoid",
            padding="same",
        )
    )

    # Initialize reversed sampling.
    if use_reversed_sampling:
        cell = LinearScheduledSamplingLayer(
            predrnn_cell,
            epsilon_s=0.,
            epsilon_e=1.,
            alpha=1e-5,
            reversed_iterations_start=20000,
            name='ss_cell',
        )
    else:
        cell = predrnn_cell

    rnn = layers.RNN(cell, return_sequences=True)

    # Construct the model.
    x = rnn(inp)
    return keras.Model(inp, x)


# %% [markdown]
# ## Model Training

# %%
# Create model according to the parameters.
input_shape = (None, *x_train.shape[2:])
model = (create_predrnn_model(input_shape, use_reversed_sampling)
         if model_type == 'predrnn'
         else create_conv_lstm_model(input_shape))
model.summary()

# %%
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
model.compile(
    #loss=losses.binary_crossentropy,
    loss=losses.mse,
    optimizer=optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
model_callbacks = [early_stopping, reduce_lr]
model_callbacks = ([*model_callbacks,
                    UpdateReversedScheduleSamplingProbCallback(model.get_layer('rnn').cell)]
                   if model_type == 'predrnn' and use_reversed_sampling
                   else model_callbacks)

# Define modifiable training hyperparameters.
epochs = 100
batch_size = 8

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=model_callbacks,
)

# Save the model.
model.save(model_save_path)

# %% [markdown]
# ## Results
# ### Performance on test data

# %%
model.evaluate(x_test, y_test, batch_size=batch_size)

# %% [markdown]
# ### Frame Prediction Visualizations

# %%
def predict_and_visualize_future_frames(predrnn, example, use_true_frames: bool = True):
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    predicted_frames = []

    # Predict a new set of 10 frames.
    for i in range(10):
        if use_true_frames:
            frames = example[i:i+10, ...]
        elif len(predicted_frames):
            frames = np.concatenate((frames, predicted_frames[-1]), axis=0)

        # Extract the model's prediction and post-process it.
        new_prediction = predrnn.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        predicted_frames.append(predicted_frame)

    # Construct a figure for the original and new frames.
    _, axes = plt.subplots(2, 10, figsize=(20, 4), layout='constrained')

    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    # Plot the new frames.
    # new_frames = frames[10:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(predicted_frames[idx]), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")


# %% [markdown]
# #### Training Dataset

# %%
# Select a random example from the training dataset.
example = train_dataset[np.random.choice(range(len(train_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example)

# %%
example = train_dataset[np.random.choice(range(len(train_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example, use_true_frames=False)

# %% [markdown]
# #### Validation Dataset

# %%
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example)

# %%
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example, use_true_frames=False)

# %% [markdown]
# #### Test Dataset

# %%
example = test_dataset[np.random.choice(range(len(test_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example)

# %%
example = test_dataset[np.random.choice(range(len(test_dataset)), size=1)[0]]
predict_and_visualize_future_frames(model, example, use_true_frames=False)
