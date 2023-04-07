from __future__ import annotations

from keras import activations, layers
from keras.utils import conv_utils
from typing import Literal, LiteralString, Callable


class SpatialTemporalLSTMCell(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple[int, int] | int,
                 padding: Literal["same", "valid"] = "same",
                 activation: LiteralString | Callable = 'tanh',
                 stride: tuple[int, int] | int = (1, 1),
                 data_format: Literal['channel_first', 'channel_last'] | None = None,
                 **kwargs):
        """
        Construct a Spatial temporal LSTM cell as described in the
        [paper](https://dl.acm.org/doi/abs/10.5555/3294771.3294855)
        and the [paper](https://arxiv.org/abs/2103.09504).

        Parameters
        filters: int
            Number of filters in the convolution layers.
        kernel_size: (int, int) or int
            Size of convolution layers.
        padding: "same" or "valid"
        activation: String or Function
            The activation applied after the convolution layers.
        stride: (int, int) or int
            Stride of convolution operation.
        data_format: "channel_first" or "channel_last" or None
            The data format of the input and state tensors.
        """
        super().__init__(**kwargs)
        self._filters = filters
        self._kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self._data_format = conv_utils.normalize_data_format(data_format)
        self._activation = activations.get(activation)
        self._padding = padding
        self._stride = conv_utils.normalize_tuple(stride, 2, 'stride')

    def build(self, input_shape):
        # First, we will need to calculate the shape of our output.
        self.output_size = self._calculate_output_shape(input_shape)
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs)

    def get_config(self):
        return super().get_config()

    def _calculate_output_shape(self, input_shape):
        if self._data_format == "channel_first":
            in_spatial_dim = input_shape[2:]
        else:
            in_spatial_dim = input_shape[1:-1]
            
        return [conv_utils.conv_output_length(
            in_spatial_dim[i],
            filter_size=self._kernel_size[i],
            padding=self._padding,
            stride=self._stride[i]
        ) for i in range(len(in_spatial_dim))]
