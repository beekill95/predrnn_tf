from __future__ import annotations

from keras import layers
import tensorflow as tf


class ReversedScheduledSamplingLayer(layers.Layer):
    def __init__(self, cell, *,
                 iterations: int = 0,
                 reversed_iterations_start: int = 0,
                 orig_sampling_prob: float = 0.5,
                 **kwargs):
        """
        This class implements (reversed) scheduled sampling
        as described in the [paper](https://arxiv.org/abs/2103.09504).

        Parameters
        cell:
            an RNN cell that satisfies all the requirements
            described in https://keras.io/api/layers/recurrent_layers/rnn/#rnn-class.
        iterations: int
            The current iterations for this sampling layer, default to 0.
        reversed_iterations_start: int
            The iterations at which we start to use reversed scheduled sampling,
            before that, we use the original sampling.
        orig_sampling_prob: float
            The probability of choosing true samples for original sampling,
            default to 0.5.
        """
        super().__init__(**kwargs)

        assert iterations >= 0
        self._cell = cell
        self._iterations = iterations
        self._reversed_iterations_start = reversed_iterations_start
        self._orig_sampling_prob = orig_sampling_prob

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations: int):
        assert iterations >= 0
        self._iterations = iterations

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def epsilon_k(self):
        iterations = self._iterations
        return (self._orig_sampling_prob
                if iterations < self._reversed_iterations_start
                else self.get_reversed_sampling_prob(iterations))

    def get_reversed_sampling_prob(self, iterations):
        return 0.

    def build(self, input_shape):
        self._cell.build(input_shape)
        self.built = True

        cell_output_size = self._cell.output_size
        cell_state_size = self._cell.state_size
        self._output_size = cell_output_size
        self._state_size = (*cell_state_size, cell_output_size)

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
            o, s = self._cell(inputs, inner_cell_states, training=training)
            return o, (*s, o)

        # Probability of choosing the true inputs.
        batch_size = tf.shape(inputs)[0]
        prob = tf.random.uniform((batch_size, ))

        # We use the true inputs with probability epsilon_k,
        # otherwise we use the output at the previous timestep.
        inputs = tf.where(prob <= self.epsilon_k, inputs, inner_cell_previous_output)

        o, s = self._cell(inputs, inner_cell_states, training=training)
        return o, (*s, o)

    def get_config(self):
        return {
            **super().get_config(),
            'cell': layers.serialize(self._cell),
            'iterations': self._iterations,
            'reversed_iterations_start': self._reversed_iterations_start,
            'orig_sampling_prob': self._orig_sampling_prob,
        }


def from_config(cls, config):
    cell = layers.deserialize(config.pop('cell'))
    return cls(cell=cell, **config)
