from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class LinearScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_s: float,
                 epsilon_e: float,
                 alpha: float,
                 iterations: int = 0,
                 reversed_iterations_start: int = 0,
                 **kwargs):
        """
        Linear scheduled sampling:
            epsilon_k = min(epsilon_s + alpha * iterations, epsilon_e)
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
        return min(self.epsilon_s + self._alpha * iterations,
                   self.epsilon_e)

    def get_config(self):
        return {
            **super().get_config(),
            'alpha': self._alpha,
        }

    @classmethod
    def from_config(cls, config):
        return from_config(cls, config)
