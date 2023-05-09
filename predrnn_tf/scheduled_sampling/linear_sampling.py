from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class LinearScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_e: float,
                 epsilon_s: float,
                 alpha: float,
                 iterations: int = 0,
                 reversed_iterations_start: int = 0,
                 orig_sampling_prob: float = 0.5,
                 **kwargs):
        """
        Linear scheduled sampling:
            epsilon_k = min(epsilon_s + alpha * iterations, epsilon_e)
        """
        super().__init__(cell,
                         iterations=iterations,
                         reversed_iterations_start=reversed_iterations_start,
                         orig_sampling_prob=orig_sampling_prob,
                         **kwargs)

        assert alpha > 0
        assert 0. <= epsilon_s < 1.
        assert epsilon_s < epsilon_e <= 1.

        self._epsilon_e = epsilon_e
        self._epsilon_s = epsilon_s
        self._alpha = alpha

    def get_reversed_sampling_prob(self, iterations):
        return min(self._epsilon_s + self._alpha * iterations,
                   self._epsilon_e)

    def get_config(self):
        return {
            **super().get_config(),
            'epsilon_e': self._epsilon_e,
            'epsilon_s': self._epsilon_s,
            'alpha': self._alpha,
        }

    @classmethod
    def from_config(cls, config):
        return from_config(cls, config)
