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
from keras import layers
from predrnn_tf.layers import SpatialTemporalLSTMCell, StackedSpatialTemporalLSTMCell

# %%
cells = StackedSpatialTemporalLSTMCell([
    SpatialTemporalLSTMCell(32, 3),
    SpatialTemporalLSTMCell(64, 3),
    SpatialTemporalLSTMCell(128, 3),
])

x = keras.Input(shape=(None, 32, 32, 3))
rnn = layers.RNN(cells)
y = rnn(x)

# %%
model = keras.Model(x, y)
model.summary()
