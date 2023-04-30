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
from keras import layers, losses, optimizers
import numpy as np
import tensorflow as tf

from predrnn_tf.layers import SpatialTemporalLSTMCell, StackedSpatialTemporalLSTMCell

# %% [markdown]
# # Save and Load PredRNN
# ## Model Creation

# %%
# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, 64, 64, 1))

# We'll use a stacked of three spatial temporal cells.
# Also, to be comparable with [Keras's tutorial](https://keras.io/examples/vision/conv_lstm/),
# we'll also use 64 filters.
cells = StackedSpatialTemporalLSTMCell([
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
    SpatialTemporalLSTMCell(64, (3, 3), decouple_loss=False),
])
rnn1 = layers.RNN(cells, return_sequences=True)

out = layers.Conv3D(
    filters=1,
    kernel_size=(3, 3, 3),
    activation="sigmoid",
    padding="same",
)

# Construct the model.
x = rnn1(inp)
x = out(x)
predrnn = keras.Model(inp, x)
predrnn.summary()

# Model compilation.
predrnn.compile(
    loss=losses.binary_crossentropy,
    optimizer=optimizers.Adam(),
)


# %%
x = tf.random.normal((1, 20, 64, 64, 1))
y = predrnn.predict(x)

# %% [markdown]
# ## Model Saving

# %%
predrnn.save('/tmp/predrnn_test_save')

# %% [markdown]
# ## Model Loading

# %%
predrnn_loaded = keras.models.load_model('/tmp/predrnn_test_save')
predrnn_loaded.summary()

# %%
y_loaded = predrnn_loaded.predict(x)

# %%
# Check that the model before & after saving gives the same result.
np.all(np.isclose(y, y_loaded))
