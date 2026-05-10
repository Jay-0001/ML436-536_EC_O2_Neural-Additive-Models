"""Credit-fraud-specific NAM model definitions.

This file keeps only the NAM-related model code from the authors' `models.py`.
The unrelated DNN baseline is intentionally omitted.
"""

from typing import Union

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

TfInput = Union[np.ndarray, tf.Tensor]


def exu(x, weight, bias):
  """ExU hidden unit modification."""
  return tf.exp(weight) * (x - bias)


def relu(x, weight, bias):
  """ReLU activation."""
  return tf.nn.relu(weight * (x - bias))


def relu_n(x, n: int = 1):
  """ReLU activation clipped at n."""
  return tf.clip_by_value(x, 0, n)


class ActivationLayer(tf.keras.layers.Layer):
  """Custom activation Layer to support ExU hidden units."""

  def __init__(
      self,
      num_units,
      name=None,
      activation="exu",
      trainable=True,
  ):
    super(ActivationLayer, self).__init__(trainable=trainable, name=name)
    self.num_units = num_units
    self._trainable = trainable
    if activation == "relu":
      self._activation = relu
      self._beta_initializer = "glorot_uniform"
    elif activation == "exu":
      self._activation = lambda x, weight, bias: relu_n(exu(x, weight, bias))
      self._beta_initializer = tf.initializers.truncated_normal(
          mean=4.0,
          stddev=0.5,
      )
    else:
      raise ValueError(f"{activation} is not a valid activation")

  def build(self, input_shape):
    self._beta = self.add_weight(
        name="beta",
        shape=[input_shape[-1], self.num_units],
        initializer=self._beta_initializer,
        trainable=self._trainable,
    )
    self._c = self.add_weight(
        name="c",
        shape=[1, self.num_units],
        initializer=tf.initializers.truncated_normal(stddev=0.5),
        trainable=self._trainable,
    )
    super(ActivationLayer, self).build(input_shape)

  @tf.function
  def call(self, x):
    center = tf.tile(self._c, [tf.shape(x)[0], 1])
    return self._activation(x, self._beta, center)


class FeatureNN(tf.keras.layers.Layer):
  """Neural network used for one input feature in the NAM."""

  def __init__(
      self,
      num_units,
      dropout=0.5,
      trainable=True,
      shallow=True,
      feature_num=0,
      name_scope="model",
      activation="exu",
  ):
    super(FeatureNN, self).__init__()
    self._num_units = num_units
    self._dropout = dropout
    self._trainable = trainable
    self._tf_name_scope = name_scope
    self._feature_num = feature_num
    self._shallow = shallow
    self._activation = activation

  def build(self, input_shape):
    self.hidden_layers = [
        ActivationLayer(
            self._num_units,
            trainable=self._trainable,
            activation=self._activation,
            name=f"activation_layer_{self._feature_num}",
        )
    ]
    if not self._shallow:
      self._h1 = tf.keras.layers.Dense(
          64,
          activation="relu",
          use_bias=True,
          trainable=self._trainable,
          name=f"h1_{self._feature_num}",
          kernel_initializer="glorot_uniform",
      )
      self._h2 = tf.keras.layers.Dense(
          32,
          activation="relu",
          use_bias=True,
          trainable=self._trainable,
          name=f"h2_{self._feature_num}",
          kernel_initializer="glorot_uniform",
      )
      self.hidden_layers += [self._h1, self._h2]
    self.linear = tf.keras.layers.Dense(
        1,
        use_bias=False,
        trainable=self._trainable,
        name=f"dense_{self._feature_num}",
        kernel_initializer="glorot_uniform",
    )
    super(FeatureNN, self).build(input_shape)

  @tf.function
  def call(self, x, training):
    with tf.name_scope(self._tf_name_scope):
      for layer in self.hidden_layers:
        x = tf.nn.dropout(
            layer(x),
            rate=tf.cond(training, lambda: self._dropout, lambda: 0.0),
        )
      x = tf.squeeze(self.linear(x), axis=1)
    return x


class NAM(tf.keras.Model):
  """Neural Additive Model for the credit fraud experiment."""

  def __init__(
      self,
      num_inputs,
      num_units,
      trainable=True,
      shallow=True,
      feature_dropout=0.0,
      dropout=0.0,
      **kwargs,
  ):
    super(NAM, self).__init__()
    self._num_inputs = num_inputs
    if isinstance(num_units, list):
      assert len(num_units) == num_inputs
      self._num_units = num_units
    elif isinstance(num_units, int):
      self._num_units = [num_units for _ in range(self._num_inputs)]
    self._trainable = trainable
    self._shallow = shallow
    self._feature_dropout = feature_dropout
    self._dropout = dropout
    self._kwargs = kwargs

  #feature network for each feature!
  def build(self, input_shape):
    self.feature_nns = [None] * self._num_inputs
    #featureNN class defined above
    for i in range(self._num_inputs):
      self.feature_nns[i] = FeatureNN(
          num_units=self._num_units[i],
          dropout=self._dropout,
          trainable=self._trainable,
          shallow=self._shallow,
          feature_num=i,
          **self._kwargs,
      )
    self._bias = self.add_weight(
        name="bias",
        initializer=tf.keras.initializers.Zeros(),
        shape=(1,),
        trainable=self._trainable,
    )
    self._true = tf.constant(True, dtype=tf.bool)
    self._false = tf.constant(False, dtype=tf.bool)

  def call(self, x, training=True):
    individual_outputs = self.calc_outputs(x, training=training)
    stacked_out = tf.stack(individual_outputs, axis=-1)
    training = self._true if training else self._false
    dropout_out = tf.nn.dropout(
        stacked_out,
        rate=tf.cond(training, lambda: self._feature_dropout, lambda: 0.0),
    )
    out = tf.reduce_sum(dropout_out, axis=-1)
    return out + self._bias

  def _name_scope(self):
    tf_name_scope = self._kwargs.get("name_scope", None)
    name_scope = super(NAM, self)._name_scope()
    if tf_name_scope:
      return tf_name_scope + "/" + name_scope
    return name_scope

  def calc_outputs(self, x, training=True):
    training = self._true if training else self._false
    list_x = tf.split(x, self._num_inputs, axis=-1)
    return [
        self.feature_nns[i](x_i, training=training)
        for i, x_i in enumerate(list_x)
    ]
