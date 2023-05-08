from __future__ import annotations

import tensorflow as tf

from .reversed_sampling import ReversedScheduledSamplingLayer, from_config


@tf.keras.utils.register_keras_serializable()
class ExponentialScheduledSamplingLayer(ReversedScheduledSamplingLayer):
    def __init__(self, cell, *,
                 epsilon_e: float,
                 epsilon_s: float,
                 alpha: float,
                 iterations: int = 0,
                 **kwargs):
        """
        Exponential scheduled sampling:
            epsilon_k = epsilon_e - (epsilon_e - epsilon_s) * exp(-k / alpha)
        """
        super().__init__(cell, iterations, **kwargs)

        assert alpha > 0
        assert 0. < epsilon_e < 1.
        assert 0. < epsilon_s < 1.

        self._epsilon_e = epsilon_e
        self._epsilon_s = epsilon_s
        self._alpha = alpha

    @property
    def epsilon_k(self):
        curve = tf.exp(-self._iterations / self._alpha)
        return self._epsilon_e - (self._epsilon_e - self._epsilon_s) * curve

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
