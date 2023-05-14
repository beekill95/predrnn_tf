from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class SigmoidScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_s: float,
                 epsilon_e: float,
                 alpha: float,
                 beta: int,
                 iterations: int = 0,
                 reversed_iterations_start: int = 0,
                 **kwargs):
        """
        Sigmoid scheduled sampling:
            epsilon_k = epsilon_s + (epsilon_e - epsilon_s) * (1. / (1. + exp((beta - k) / alpha)))
        """
        super().__init__(cell,
                         iterations=iterations,
                         reversed_iterations_start=reversed_iterations_start,
                         epsilon_s=epsilon_s,
                         epsilon_e=epsilon_e,
                         **kwargs)

        assert alpha > 0
        assert beta > 0

        self._alpha = alpha
        self._beta = beta

    def get_reversed_sampling_prob(self, iterations):
        curve = 1. / (1. + tf.exp((self._beta - iterations) / self._alpha))
        return self.epsilon_s + (self.epsilon_e - self.epsilon_s) * curve

    def get_config(self):
        return {
            **super().get_config(),
            'alpha': self._alpha,
            'beta': self._beta,
        }

    @classmethod
    def from_config(cls, config):
        return from_config(cls, config)
