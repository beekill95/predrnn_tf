from __future__ import annotations

from keras import activations, constraints, initializers, layers, regularizers
from keras.utils import conv_utils
import tensorflow as tf
from typing import Literal, Callable


RegularizerType = str | regularizers.Regularizer
ConstraintType = str | constraints.Constraint
InitializerType = str


class SpatialTemporalLSTMCell(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple[int, int] | int,
                 *,
                 padding: Literal["same", "valid"] = "same",
                 activation: str | Callable = 'tanh',
                 stride: tuple[int, int] | int = (1, 1),
                 data_format: Literal['channel_first', 'channel_last'] | None = None,
                 recurrent_activation: str | Callable = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 kernel_regularizer: RegularizerType | None = None,
                 recurrent_regularizer: RegularizerType | None = None,
                 bias_regularizer: RegularizerType | None = None,
                 kernel_constraint: ConstraintType | None = None,
                 recurrent_constraint: ConstraintType | None = None,
                 bias_constraint: ConstraintType | None = None,
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

        # Convolion layers.
        self._filters = filters
        self._kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self._data_format = conv_utils.normalize_data_format(data_format)
        self._padding = padding
        self._stride = conv_utils.normalize_tuple(stride, 2, 'stride')

        # Activations.
        self._activation = activations.get(activation)
        self._recurrent_activation = activations.get(recurrent_activation)

        # Weight initializers.
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._recurrent_initializer = initializers.get(recurrent_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

        # Weight regularizers.
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)

        # Weight constraints.
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._recurrent_constraint = constraints.get(recurrent_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        self._output_size = None
        self._state_size = (filters, filters, filters)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def build(self, input_shape):
        # First, we will need to calculate the shape of our output.
        self._output_size = self._calculate_output_shape(input_shape)

        # The original lstm cell's weights for the input X.
        (self._Wxg, self._Wxi, self._Wxf,
         # Weights for the spatial temporal memory.
         self._Wxg_, self._Wxi_, self._Wxf_,
         # Weights for the output.
         self._Wxo) = self._add_weights_X(input_shape)

        # The original lstm cell's weights for the hidden state.
        (self._Whg,
         self._Whi,
         self._Whf,
         self._Who) = self._add_weights_H(input_shape)

        # Weight for the cell state.
        self._Wco = self._add_weights_C(input_shape)

        # Weights for the new spatial temporal memory state.
        (self._Wmg,
         self._Wmi,
         self._Wmf,
         self._Wmo) = self._add_weights_M(input_shape)

        # Weight for combining cell and spatial temporal memory state.
        self._W11 = self._add_weight_11()

        # Biases.
        (self._bg, self._bi, self._bf,
         self._bg_, self._bi_, self._bf_,
         self._bo) = self._add_biases()

    def call(self, inputs, states, training=None):
        """
        Perform calculation for a spatial temporal lstm cell.

        Parameters
        inputs: an input tensor or hidden state of the previous cell.
        states: a tuple containing (hidden state, cell state, spatial temporal memory state).
        
        Returns
            hidden state, (hidden state, cell state, memory state).
        """
        # First, get the state of the previous time step (hidden state & cell state),
        # and the memory state of the previous layer.
        Ht_1, Ct_1, Ml_1 = states

        # Then, we will calculate the values of the original lstm cell's gates.
        g = self._activation(
                self._input_conv(inputs, self._Wxg)
                + self._recurrent_conv(Ht_1, self._Whg)
                + self._bg) # pyright: ignore
        i = self._recurrent_activation(
                self._input_conv(inputs, self._Wxi)
                + self._recurrent_conv(Ht_1, self._Whi)
                + self._bi) # pyright: ignore
        f = self._recurrent_activation(
                self._input_conv(inputs, self._Wxf)
                + self._recurrent_conv(Ht_1, self._Whf)
                + self._bf) # pyright: ignore

        # New cell state.
        Ct = f * Ct_1 + i * g

        # Calculate gates with the new spatial temporal memory state.
        g_ = self._activation(
                self._input_conv(inputs, self._Wxg_)
                + self._recurrent_conv(Ml_1, self._Wmg)
                + self._bg_) # pyright: ignore
        i_ = self._recurrent_activation(
                self._input_conv(inputs, self._Wxi_)
                + self._recurrent_conv(Ml_1, self._Wmi)
                + self._bi_) # pyright: ignore
        f_ = self._recurrent_activation(
                self._input_conv(inputs, self._Wxf_)
                + self._recurrent_conv(Ml_1, self._Wmf)
                + self._bf_) # pyright: ignore

        # New spatial temporal state.
        Ml = f_ * Ml_1 + i_ * g_

        # Calculate output gate.
        o = self._recurrent_activation(
                self._input_conv(inputs, self._Wxo)
                + self._recurrent_conv(Ht_1, self._Who)
                + self._recurrent_conv(Ct, self._Wco)
                + self._recurrent_conv(Ml, self._Wmo)
                + self._bo) # pyright: ignore

        # New hidden state, which is also our output.
        channel_dim = -1 if self._data_format == "channel_last" else 2
        CM = tf.concat([Ct, Ml], axis=channel_dim)
        Ht = o * self._recurrent_activation(
                self._recurrent_conv(CM, self._W11)) # pyright: ignore

        # Return the output and cell states.
        return Ht, (Ht, Ct, Ml)

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "data_format": self._data_format,
            "padding": self._padding,
            "stride": self._stride,
            "activation": self._activation,
            "recurrent_activation": self._recurrent_activation,
            "kernel_initializer": self._kernel_initializer,
            "recurrent_initializer": self._recurrent_initializer,
            "bias_initializer": self._bias_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "recurrent_regularizer": self._recurrent_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "kernel_constraint": self._kernel_constraint,
            "recurrent_constraint": self._recurrent_constraint,
            "bias_constraint": self._bias_constraint,
        }

    def _calculate_output_shape(self, input_shape):
        if self._data_format == "channel_first":
            in_spatial_dim = input_shape[3:]
        else:
            in_spatial_dim = input_shape[1:-1]

        return [conv_utils.conv_output_length(
            in_spatial_dim[i],
            filter_size=self._kernel_size[i],
            padding=self._padding,
            stride=self._stride[i],
            ) for i in range(len(in_spatial_dim))]

    def _input_conv(self, x, w):
        return tf.nn.conv2d(
            x, w,
            strides=self._stride,
            padding=self._padding,
            data_format=self._data_format)

    def _recurrent_conv(self, x, w):
        return tf.nn.conv2d(
            x, w,
            strides=(1, 1),
            padding='same',
            data_format=self._data_format)

    def _add_weights_X(self, input_shape):
        nb_weights = 7
        shape = self._kernel_size + (self._in_channels(input_shape), self._filters * nb_weights)
        weights = self.add_weight(
            name='Wx',
            shape=shape,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint)
        return tf.split(weights, nb_weights, axis=-1)

    def _add_weights_H(self, input_shape):
        nb_weights = 4
        shape = self._kernel_size + (self._in_channels(input_shape), self._filters * nb_weights)
        weights = self.add_weight(
            name='Wh',
            shape=shape,
            initializer=self._recurrent_initializer,
            regularizer=self._recurrent_regularizer,
            constraint=self._recurrent_constraint)
        return tf.split(weights, nb_weights, axis=-1)

    def _add_weights_M(self, input_shape):
        nb_weights = 4
        shape = self._kernel_size + (self._in_channels(input_shape), self._filters * nb_weights)
        weights = self.add_weight(
            name='Wm',
            shape=shape,
            initializer=self._recurrent_initializer,
            regularizer=self._recurrent_regularizer,
            constraint=self._recurrent_constraint)
        return tf.split(weights, nb_weights, axis=-1)

    def _add_weights_C(self, input_shape):
        shape = self._kernel_size + (self._in_channels(input_shape), self._filters)
        return self.add_weight(
            name='Wc',
            shape=shape,
            initializer=self._recurrent_initializer,
            regularizer=self._recurrent_regularizer,
            constraint=self._recurrent_constraint)

    def _add_weight_11(self):
        return self.add_weight(
            name='W11',
            shape=(1, 1, 2*self._filters, self._filters),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint)

    def _add_biases(self):
        nb_biases = 7
        biases = self.add_weight(
            name='b',
            shape=(self._filters * nb_biases),
            initializer=self._bias_initializer,
            regularizer=self._bias_regularizer,
            constraint=self._bias_constraint)
        return tf.split(biases, nb_biases)

    def _in_channels(self, input_shape):
        channel_dim = -1 if self._data_format == "channel_last" else 2
        return input_shape[channel_dim]

