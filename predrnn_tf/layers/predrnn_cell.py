from __future__ import annotations

from keras import layers
import tensorflow as tf

from .spatial_temporal_lstm_cell import SpatialTemporalLSTMCell
from .stacked_spatial_temporal_lstm_cell import StackedSpatialTemporalLSTMCell


@tf.keras.utils.register_keras_serializable()
class PredRNNCell(layers.Layer):
    def __init__(self,
                 cell: SpatialTemporalLSTMCell | StackedSpatialTemporalLSTMCell,
                 out_conv: layers.Conv2D,
                 **kwargs):
        super().__init__(**kwargs)

        self._cell = cell
        self._out = out_conv

    def call(self, inputs, states, training=None):
        o, s = self._cell(inputs, states, training=training)
        o = self._out(o)
        return o, s

    def get_config(self):
        return {
            **super().get_config(),
            'cell': self._cell,
            'out_conv': self._out,
        }

    @classmethod
    def from_config(cls, config):
        cell = layers.deserialize(config.pop('cell'))
        out = layers.deserialize(config.pop('out_conv'))
        return cls(cell=cell, out_conv=out, **config) # pyright: ignore
