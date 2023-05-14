from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class ExponentialScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_s: float,
                 epsilon_e: float,
                 alpha: float,
                 iterations: int = 0,
                 reversed_iterations_start: int = 0,
                 **kwargs):
        """
        Exponential scheduled sampling:
            epsilon_k = epsilon_e - (epsilon_e - epsilon_s) * exp(-k / alpha)
        """
        super().__init__(cell,
                         iterations=iterations,
                         reversed_iterations_start=reversed_iterations_start,
                         epsilon_s=epsilon_s,
                         epsilon_e=epsilon_e,
                         **kwargs)
        assert alpha > 0

        self._alpha = alpha

    def get_reversed_sampling_prob(self, iterations):
        curve = tf.exp(-iterations / self._alpha)
        return self.epsilon_e - (self.epsilon_e - self.epsilon_s) * curve

    def get_config(self):
        return {
            **super().get_config(),
            'alpha': self._alpha,
        }

    @classmethod
    def from_config(cls, config):
        return from_config(cls, config)
