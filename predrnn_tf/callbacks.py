from __future__ import annotations

from keras import callbacks

from .scheduled_sampling.reversed_sampling import ReversedScheduledSamplingLayer


class UpdateReversedScheduleSamplingProbCallback(callbacks.Callback):
    def __init__(self, layer: ReversedScheduledSamplingLayer):
        super().__init__()
        self._layer = layer
        self._iterations = layer.iterations

    def on_batch_begin(self, **_):
        self._iterations += 1
        self._layer.iterations = self._iterations
