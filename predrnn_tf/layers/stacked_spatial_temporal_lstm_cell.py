from __future__ import annotations

from itertools import chain
from keras import backend as K, layers
import tensorflow as tf

from .. import utils
from .types import *
from .spatial_temporal_lstm_cell import SpatialTemporalLSTMCell


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class StackedSpatialTemporalLSTMCell(layers.Layer):
    def __init__(self, cells: list[SpatialTemporalLSTMCell], **kwargs):
        """
        Construct a stacked spatial temporal LSTM cell based on the
        [paper](https://dl.acm.org/doi/abs/10.5555/3294771.3294855)
        and the [paper](https://arxiv.org/abs/2103.09504).
        """
        super().__init__(**kwargs)
        self._cells = cells

    @property
    def output_size(self):
        return self._cells[-1].output_size

    @property
    def state_size(self):
        return self._state_size

    def build(self, input_shape):
        cells = self._cells
        prev_m_nb_channels = None
        self._Wms = []
        for i, cell in enumerate(cells):
            with tf.name_scope(f'cell_{i}'):
                cell.build(input_shape)
                input_shape = cell.output_size

                # Create weight to handle the mismatched dimensions when M is moved from the previous cell
                # to the current cell.
                cur_m_nb_channels = cell.get_channels(cell.state_size[-1])
                if prev_m_nb_channels is not None and prev_m_nb_channels != cur_m_nb_channels:
                    Wm = self.add_weight(
                        f'stacked_Wm_{i}', shape=(1, 1, prev_m_nb_channels, cur_m_nb_channels))
                    self._Wms.append(Wm)
                else:
                    self._Wms.append(None)

                # Update the current number of channels of the M memory state.
                prev_m_nb_channels = cur_m_nb_channels

        self._state_size = tuple(chain(*[c.state_size for c in self._cells]))

        # We also need to take care the dimensions mismatched between
        # the last cell's memory state and the first cell's memory state.
        first_cell_M_nb_channels = cells[0].get_channels(self._state_size[2])
        last_cell_M_nb_channels = cells[-1].get_channels(self._state_size[-1])
        if first_cell_M_nb_channels != last_cell_M_nb_channels:
            self._Wms[0] = self.add_weight(
                'stacked_Wm_0', shape=(1, 1, last_cell_M_nb_channels, first_cell_M_nb_channels))

        self.built = True

    def call(self, inputs, states, training=None):
        """
        Perform the calculation based on the the data flow described in the papers.
        In particular, the hidden states and cell states are transfered in the temporal dimension,
        while the spatial-temporal memory is transfered in the zigzag direction.

        Parameters
        inputs: input tensor to the cell of shape (batch_size, channels, H, W) or (batch_size, H, W, channels).
        states: the list containing cell states of the previous time step of shape.
            The list contains the cell states of each cell: (h1, c1, m1, h2, c2, m2, ...).

        Returns
            hidden state, [h1, c1, m1, h2, c2, m2, ...]
        """
        # The lenght of the states must be divisible by 3.
        nb_cell_states, remaining_states = divmod(len(states), 3)
        assert remaining_states == 0, "Invalid states: the lenght of states must be a multiple of 3."
        assert nb_cell_states == len(self._cells), "Invalid states: there are not enough states for cells."

        # The spatial temporal memory of the last cell in the previous time step.
        previous_m = states[-1]

        # Call each cell and store the resulting states.
        out_states = []
        x = inputs
        for cell, (h, c, _), Wm in zip(self._cells, utils.triplet(states), self._Wms):
            # Need to adjust the M's dimension if necessary.
            if Wm is not None:
                previous_m = K.conv2d(previous_m, Wm, data_format=cell.data_format)

            # Instead of using the hidden and cell state of the same cell in the previous time step,
            # the spatial temporal memory is from the previous cell of the same time step,
            # or the last cell of the previous time step.
            x, s = cell.call(x, (h, c, previous_m), training=training)

            # Store the spatial temporal memory cell so we could pass it to the next cell.
            previous_m = s[-1]

            # Store the cell's states.
            out_states.extend(s)

        return x, out_states

    def get_config(self):
        return {
            **super().get_config(),
            'cells': [layers.serialize(c) for c in self._cells],
        }

    @classmethod
    def from_config(cls, config):
        cells = [layers.deserialize(c) for c in config.pop('cells')]
        return cls(cells=cells, **config)
