from __future__ import annotations

from keras import (
    activations,
    backend as K,
    constraints,
    initializers,
    layers,
    regularizers,
)
from keras.utils import conv_utils
import tensorflow as tf
from typing import Literal, Callable

from .types import *


@tf.keras.utils.register_keras_serializable('predrnn_tf')
class SpatialTemporalLSTMCell(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple[int, int] | int,
                 *,
                 padding: Literal["same", "valid"] = "same",
                 activation: str | Callable = 'tanh',
                 stride: tuple[int, int] | int = (1, 1),
                 data_format: Literal['channels_first', 'channels_last'] | None = None,
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
                 decouple_loss: bool = False,
                 **kwargs):
        """
        Construct a Spatial temporal LSTM cell as described in the
        [paper](https://dl.acm.org/doi/abs/10.5555/3294771.3294855)
        and the [paper](https://arxiv.org/abs/2103.09504).

        Notes:
            * decouple_loss = True is not working right now due to a bug in keras.

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

        # Should decouple loss be included.
        self._decouple_loss = decouple_loss

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def is_channels_first(self):
        return self._data_format == 'channels_first'

    @property
    def channels_dim(self):
        return 1 if self.is_channels_first else -1

    @property
    def data_format(self):
        return self._data_format

    def build(self, input_shape):
        # First, we will need to calculate the shape of our output.
        output_size = self._calculate_output_shape(input_shape)
        self._output_size = output_size
        # State size should not contain batch dimensions,
        # or else it will cause error when using as cell inside RNN layer.
        self._state_size = (output_size[1:], output_size[1:], output_size[1:])

        # The original lstm cell's weights for the input X.
        self._Wx = self._add_weights_X(input_shape)

        # The original lstm cell's weights for the hidden state.
        self._Wh = self._add_recurrent_weights('Wh', 4)

        # Weight for the cell state.
        self._Wco = self._add_recurrent_weights('Wco', 1)

        # Weights for the new spatial temporal memory state.
        self._Wm = self._add_recurrent_weights('Wm', 4)

        # Weight for combining cell and spatial temporal memory state.
        self._W11 = self._add_weight_11()

        # Biases.
        self._b = self._add_biases()

        self.built = True

    def call(self, inputs, states, training=None):
        """
        Perform calculation for a spatial temporal lstm cell.

        Parameters
        inputs: an input tensor or hidden state of the previous cell,
            the tensor has shape (batch_size, channels, H, W) or (batch_size, H, W, channels).
        states: a tuple containing (hidden state, cell state, spatial temporal memory state).
        
        Returns
            hidden state, (hidden state, cell state, memory state).
        """
        # First, get the state of the previous time step (hidden state & cell state),
        # and the memory state of the previous layer.
        Ht_1, Ct_1, Ml_1 = states

        # The original lstm cell's weights for the input X.
        (Wxg, Wxi, Wxf,
         # Weights for the spatial temporal memory.
         Wxg_, Wxi_, Wxf_,
         # Weights for the output.
         Wxo) = tf.split(self._Wx, 7, axis=-1)

        # The original lstm cell's weights for the hidden state.
        (Whg, Whi, Whf, Who) = tf.split(self._Wh, 4, axis=-1)

        # The original lstm cell's weights for the hidden state.
        (Wmg, Wmi, Wmf, Wmo) = tf.split(self._Wm, 4, axis=-1)

        # Biases.
        (bg, bi, bf,
         bg_, bi_, bf_,
         bo) = tf.split(self._b, 7, axis=-1)

        # Then, we will calculate the values of the original lstm cell's gates.
        g = self._activation(
                self._input_conv(inputs, Wxg)
                + self._recurrent_conv(Ht_1, Whg)
                + bg) # pyright: ignore
        i = self._recurrent_activation(
                self._input_conv(inputs, Wxi)
                + self._recurrent_conv(Ht_1, Whi)
                + bi) # pyright: ignore
        f = self._recurrent_activation(
                self._input_conv(inputs, Wxf)
                + self._recurrent_conv(Ht_1, Whf)
                + bf) # pyright: ignore

        # New cell state.
        Ct = f * Ct_1 + i * g

        # Calculate gates with the new spatial temporal memory state.
        g_ = self._activation(
                self._input_conv(inputs, Wxg_)
                + self._recurrent_conv(Ml_1, Wmg)
                + bg_) # pyright: ignore
        i_ = self._recurrent_activation(
                self._input_conv(inputs, Wxi_)
                + self._recurrent_conv(Ml_1, Wmi)
                + bi_) # pyright: ignore
        f_ = self._recurrent_activation(
                self._input_conv(inputs, Wxf_)
                + self._recurrent_conv(Ml_1, Wmf)
                + bf_) # pyright: ignore

        # New spatial temporal state.
        Ml = f_ * Ml_1 + i_ * g_

        # Calculate output gate.
        o = self._recurrent_activation(
                self._input_conv(inputs, Wxo)
                + self._recurrent_conv(Ht_1, Who)
                + self._recurrent_conv(Ct, self._Wco)
                + self._recurrent_conv(Ml, Wmo)
                + bo) # pyright: ignore

        # New hidden state, which is also our output.
        CM = K.concatenate([Ct, Ml], axis=self.channels_dim)
        Ht = o * self._recurrent_activation(
                self._recurrent_conv(CM, self._W11)) # pyright: ignore

        # Add decouple loss.
        if self._decouple_loss and training:
            # Here, we'll let W_decouple = I
            delta_c = i * g
            delta_m = i_ * g_
            self.add_loss(self._calc_decouple_loss(delta_c, delta_m))

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
            "kernel_initializer": layers.serialize(self._kernel_initializer),
            "recurrent_initializer": layers.serialize(self._recurrent_initializer),
            "bias_initializer": layers.serialize(self._bias_initializer),
            "kernel_regularizer": layers.serialize(self._kernel_regularizer),
            "recurrent_regularizer": layers.serialize(self._recurrent_regularizer),
            "bias_regularizer": layers.serialize(self._bias_regularizer),
            "kernel_constraint": layers.serialize(self._kernel_constraint),
            "recurrent_constraint": layers.serialize(self._recurrent_constraint),
            "bias_constraint": layers.serialize(self._bias_constraint),
            "decouple_loss": self._decouple_loss,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _calculate_output_shape(self, input_shape):
        batch_size = input_shape[0]
        in_spatial_dim = input_shape[2:] if self.is_channels_first else input_shape[1:-1]
        spatial_output_size = tuple(conv_utils.conv_output_length(in_spatial_dim[i],
                                                                  filter_size=self._kernel_size[i],
                                                                  padding=self._padding,
                                                                  stride=self._stride[i])
                                    for i in range(len(in_spatial_dim)))
        return tf.TensorShape((batch_size, self._filters, *spatial_output_size)
                              if self.is_channels_first
                              else (batch_size, *spatial_output_size, self._filters))

    def _input_conv(self, x, w):
        return K.conv2d(
            x, w,
            strides=self._stride,
            padding=self._padding,
            data_format=self._data_format)

    def _recurrent_conv(self, x, w):
        return K.conv2d(
            x, w,
            strides=(1, 1),
            padding='same',
            data_format=self._data_format)

    def _calc_decouple_loss(self, delta_c, delta_m):
        """
        Calculate decouple loss.

        Parameters
        delta_c: a tensor of shape (batch_size, channels, h, w) for "channels_first"
            or (batch_size, channels, h, w) for "channels_last".
        delta_m: a tensor of shape (batch_size, h, w, channels) for "channels_first"
            or (batch_size, h, w, channels) for "channels_last".
        """
        def norm(tensor, axis=-1):
            return K.sqrt(K.sum(tensor * tensor, axis=axis))

        batch_size = K.shape(delta_c)[0]
        channels = self.get_channels(K.shape(delta_c))
        if self.is_channels_first:
            delta_c = K.reshape(delta_c, (batch_size, channels, -1))
            delta_m = K.reshape(delta_m, (batch_size, channels, -1))

            dot_prod = K.sum(delta_c * delta_m, axis=-1)
            delta_c_l2 = norm(delta_c, axis=-1)
            delta_m_l2 = norm(delta_m, axis=-1)
        else:
            delta_c = K.reshape(delta_c, (batch_size, -1, channels))
            delta_m = K.reshape(delta_m, (batch_size, -1, channels))

            dot_prod = K.sum(delta_c * delta_m, axis=1)
            delta_c_l2 = norm(delta_c, axis=1)
            delta_m_l2 = norm(delta_m, axis=1)

        return K.mean(K.abs(dot_prod) / (delta_c_l2 * delta_m_l2))
        
    def _add_weights_X(self, input_shape):
        nb_weights = 7
        shape = self._kernel_size + (self.get_channels(input_shape), self._filters * nb_weights)
        return self.add_weight(
            name=f'Wx',
            shape=shape,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint)

    def _add_recurrent_weights(self, name: str, nb_weights: int):
        shape = self._kernel_size + (self._filters, self._filters * nb_weights)
        return self.add_weight(
            name=name,
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
        return self.add_weight(
                name='biases',
                shape=(self._filters * nb_biases,),
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                constraint=self._bias_constraint)

    def get_channels(self, input_shape):
        return input_shape[self.channels_dim]
