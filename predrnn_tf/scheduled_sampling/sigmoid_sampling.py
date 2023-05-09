from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable()
class SigmoidScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_e: float,
                 epsilon_s: float,
                 alpha: float,
                 beta: int,
                 iterations: int = 0,
                 **kwargs):
        """
        Sigmoid scheduled sampling:
            epsilon_k = epsilon_s + (epsilon_e - epsilon_s) * (1. / (1. + exp((beta - k) / alpha)))
        """
        super().__init__(cell, iterations, **kwargs)

        assert alpha > 0
        assert 0. <= epsilon_s < 1.
        assert epsilon_s < epsilon_e <= 1.
        assert beta > 0

        self._epsilon_e = epsilon_e
        self._epsilon_s = epsilon_s
        self._alpha = alpha
        self._beta = beta

    @property
    def epsilon_k(self):
        curve = 1. / (1. + tf.exp((self._beta - self._iterations) / self._alpha))
        return self._epsilon_s + (self._epsilon_e - self._epsilon_s) * curve

    def get_config(self):
        return {
            **super().get_config(),
            'epsilon_e': self._epsilon_e,
            'epsilon_s': self._epsilon_s,
            'alpha': self._alpha,
            'beta': self._beta,
        }

    @classmethod
    def from_config(cls, config):
        return from_config(cls, config)
