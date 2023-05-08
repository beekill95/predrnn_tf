from __future__ import annotations

from keras import layers
import tensorflow as tf


class ReversedScheduledSamplingLayer(layers.Layer):
    def __init__(self, cell, iterations: int = 0, **kwargs):
        """
        This class implements (reversed) scheduled sampling
        as described in the [paper](https://arxiv.org/abs/2103.09504).

        Parameters
        cell:
            an RNN cell that satisfies all the requirements
            described in https://keras.io/api/layers/recurrent_layers/rnn/#rnn-class.
        """
        super().__init__(**kwargs)

        assert iterations >= 0
        self._cell = cell
        self._iterations = iterations

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        assert iterations >= 0
        self._iterations = iterations

    @property
    def state_size(self):
        cell_output_size = self.output_size
        cell_state_size = self._cell.state_size
        # The state size of this layer will be the cell's state size
        # and the cell's output size.
        return (*cell_state_size, cell_output_size[1:])

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def epsilon_k(self):
        print('parent')
        return 0.

    def build(self, input_shape):
        self._cell.build(input_shape)
        self.built = True

    def call(self, inputs, states, training=None):
        """
        Perform scheduled sampling on the inputs.

        Parameters
        inputs: an input tensor has shape
            (batch_size, channels, H, W) or (batch_size, H, W, channels).
        states: a tuple containing (inner cell's states, inner cell's output for the previous time step).

        Returns
            inner cell's output, (inner cell's states, inner cell's output).
        """
        inner_cell_states = states[:-1]

        # Inner cell's output at the previous timestep.
        # The shape should be the same as the inputs (batch_size, C, H, W)
        # or (batch_size, H, W, C).
        inner_cell_previous_output = states[-1]

        if not training:
            o, s = self._cell.call(inputs, inner_cell_states, trainning=training)
            return o, (*s, o)

        # Probability of choosing the true inputs.
        batch_size = inputs.shape[0]
        prob = tf.random.uniform((batch_size, ))

        # We use the true inputs with probability epsilon_k,
        # otherwise we use the output at the previous timestep.
        inputs = tf.where(prob <= self.epsilon_k, inputs, inner_cell_previous_output)

        o, s = self._cell.call(inputs, inner_cell_states, training=training)
        return o, (*s, o)

    def get_config(self):
        return {
            **super().get_config(),
            'cell': self._cell,
            'iterations': self._iterations,
        }


def from_config(cls, config):
    cell = layers.deserialize(config.pop('cell'))
    iterations = config.pop('iterations')
    return cls(cell=cell, iterations=iterations, **config)
