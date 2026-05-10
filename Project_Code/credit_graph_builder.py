"""Training graph helpers for the credit fraud NAM experiment.

This file is a classification-only adaptation of the authors' graph builder.
Regression and unrelated baseline code paths were removed on purpose.
"""

import functools
from typing import Callable, Dict, Union

import numpy as np
from sklearn import metrics as sk_metrics
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import credit_nam_models as models

TfInput = models.TfInput
LossFunction = Callable[[tf.keras.Model, TfInput, TfInput], tf.Tensor]
GraphOpsAndTensors = Dict[str, Union[tf.Tensor, tf.Operation, tf.keras.Model]]


def cross_entropy_loss(model, inputs, targets):
  """Cross-entropy loss for binary classification."""
  predictions = model(inputs, training=True)
  logits = tf.stack([predictions, tf.zeros_like(predictions)], axis=1)
  labels = tf.stack([targets, 1 - targets], axis=1)
  loss_vals = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels,
      logits=logits,
  )
  return tf.reduce_mean(loss_vals)


def feature_output_regularization(model, inputs):
  """Penalizes the L2 norm of the prediction of each feature net."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  per_feature_norm = [
      tf.reduce_mean(tf.square(outputs)) for outputs in per_feature_outputs
  ]
  return tf.add_n(per_feature_norm) / len(per_feature_norm)


def weight_decay(model, num_networks=1):
  """Penalizes the L2 norm of weights in each feature net."""
  l2_losses = [tf.nn.l2_loss(x) for x in model.trainable_variables]
  return tf.add_n(l2_losses) / num_networks


def penalized_cross_entropy_loss(
    model,
    inputs,
    targets,
    output_regularization,
    l2_regularization=0.0,
):
  """Cross-entropy loss with output penalty and L2 regularization."""
  loss = cross_entropy_loss(model, inputs, targets)
  reg_loss = 0.0
  if output_regularization > 0:
    reg_loss += output_regularization * feature_output_regularization(
        model,
        inputs,
    )
  if l2_regularization > 0:
    reg_loss += l2_regularization * weight_decay(
        model,
        num_networks=len(model.feature_nns),
    )
  return loss + reg_loss


def generate_predictions(pred_tensor, dataset_init_op, sess):
  """Iterates over the prediction tensor to compute predictions."""
  sess.run(dataset_init_op)
  y_pred = []
  while True:
    try:
      y_pred.extend(sess.run(pred_tensor))
    except tf.errors.OutOfRangeError:
      break
  return y_pred


def pr_auc_score(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates area under the precision-recall curve."""
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_pred)
  return sk_metrics.auc(recall, precision)


def grad(model, inputs, targets, loss_fn=cross_entropy_loss, train_vars=None):
  """Calculates gradients of `loss_fn` with respect to train vars."""
  loss_value = loss_fn(model, inputs, targets)
  if train_vars is None:
    train_vars = model.trainable_variables
  return loss_value, tf.gradients(loss_value, train_vars)


#upsampling the fraud class
def create_balanced_dataset(x_train, y_train, batch_size):
  """Creates balanced batches by upsampling the minority fraud class."""

  def partition_dataset(x_values, y_values):
    neg_mask = (y_values == 0)
    x_train_neg = x_values[neg_mask]
    y_train_neg = np.zeros(len(x_train_neg), dtype=np.float32)
    x_train_pos = x_values[~neg_mask]
    y_train_pos = np.ones(len(x_train_pos), dtype=np.float32)
    return (x_train_pos, y_train_pos), (x_train_neg, y_train_neg)

  pos, neg = partition_dataset(x_train, y_train)
  pos_dataset = tf.data.Dataset.from_tensor_slices(pos).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(pos[0]))
  )
  neg_dataset = tf.data.Dataset.from_tensor_slices(neg).apply(
      tf.data.experimental.shuffle_and_repeat(buffer_size=len(neg[0]))
  )
  dataset = tf.data.experimental.sample_from_datasets([pos_dataset, neg_dataset])
  return dataset.batch(batch_size)


def create_iterators(datasets, batch_size):
  """Creates `tf.data` iterators for train-time evaluation datasets."""
  tf_datasets = [
      tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
      for data in datasets
  ]
  input_iterator = tf.data.Iterator.from_structure(
      tf_datasets[0].output_types,
      tf_datasets[0].output_shapes,
  )
  init_ops = [input_iterator.make_initializer(data) for data in tf_datasets]
  x_batch = input_iterator.get_next()
  return x_batch, init_ops


def create_nam_model(
    x_train,
    dropout,
    feature_dropout=0.0,
    num_basis_functions=1000,
    units_multiplier=2,
    activation="exu",
    name_scope="model",
    shallow=True,
    trainable=True,
):
  """Creates the credit-fraud NAM model."""
  num_unique_vals = [
      len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])
  ]
  num_units = [
      min(num_basis_functions, value_count * units_multiplier)
      for value_count in num_unique_vals
  ]
  num_inputs = x_train.shape[-1]
  return models.NAM(
      num_inputs=num_inputs,
      num_units=num_units,
      dropout=np.float32(dropout),
      feature_dropout=np.float32(feature_dropout),
      activation=activation,
      shallow=shallow,
      trainable=trainable,
      name_scope=name_scope,
  )


def build_graph(
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate,
    batch_size,
    output_regularization,
    dropout,
    decay_rate,
    shallow,
    l2_regularization=0.0,
    feature_dropout=0.0,
    num_basis_functions=1000,
    units_multiplier=2,
    activation="exu",
    name_scope="model",
    trainable=True,
):
  """Constructs the classification graph for credit fraud NAM training."""
  ds_tensors = create_balanced_dataset(x_train, y_train, batch_size)
  x_batch, (train_init_op, test_init_op) = create_iterators(
      (x_train, x_test),
      batch_size,
  )

  nn_model = create_nam_model(
      x_train=x_train,
      dropout=dropout,
      feature_dropout=feature_dropout,
      activation=activation,
      num_basis_functions=num_basis_functions,
      shallow=shallow,
      units_multiplier=units_multiplier,
      trainable=trainable,
      name_scope=name_scope,
  )

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.Variable(learning_rate, trainable=False)
  lr_decay_op = learning_rate.assign(decay_rate * learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  predictions = nn_model(x_batch, training=False)
  train_vars = nn_model.trainable_variables
  loss_fn = functools.partial(
      penalized_cross_entropy_loss,
      output_regularization=output_regularization,
      l2_regularization=l2_regularization,
  )
  y_pred = tf.nn.sigmoid(predictions)

  iterator = ds_tensors.make_initializable_iterator()
  x1, y1 = iterator.get_next()
  loss_tensor, grads = grad(nn_model, x1, y1, loss_fn, train_vars)
  update_step = optimizer.apply_gradients(
      zip(grads, train_vars),
      global_step=global_step,
  )
  avg_loss, avg_loss_update_op = tf.metrics.mean(
      loss_tensor,
      name="avg_train_loss",
  )
  tf.summary.scalar("avg_train_loss", avg_loss)

  running_mean_vars = tf.get_collection(
      tf.GraphKeys.LOCAL_VARIABLES,
      scope="avg_train_loss",
  )
  running_vars_initializer = tf.variables_initializer(var_list=running_mean_vars)

  train_metric = functools.partial(
      pr_auc_score,
      y_true=y_train,
      pred_tensor=y_pred,
      dataset_init_op=train_init_op,
  )
  test_metric = functools.partial(
      pr_auc_score,
      y_true=y_test,
      pred_tensor=y_pred,
      dataset_init_op=test_init_op,
  )

  summary_op = tf.summary.merge_all()

  graph_tensors = {
      "train_op": [update_step, avg_loss_update_op],
      "lr_decay_op": lr_decay_op,
      "summary_op": summary_op,
      "iterator_initializer": iterator.initializer,
      "running_vars_initializer": running_vars_initializer,
      "nn_model": nn_model,
      "global_step": global_step,
  }
  eval_metric_scores = {"test": test_metric, "train": train_metric}
  return graph_tensors, eval_metric_scores


def build_eval_graph(
    x_model_reference,
    x_eval,
    y_eval,
    batch_size,
    num_basis_functions=1000,
    units_multiplier=2,
    activation="exu",
    name_scope="model",
    shallow=True,
):
  """Constructs an evaluation-only graph for a saved credit-fraud NAM."""
  x_batch, (eval_init_op,) = create_iterators((x_eval,), batch_size)
  nn_model = create_nam_model(
      x_train=x_model_reference,
      dropout=0.0,
      feature_dropout=0.0,
      activation=activation,
      num_basis_functions=num_basis_functions,
      shallow=shallow,
      units_multiplier=units_multiplier,
      trainable=True,
      name_scope=name_scope,
  )
  predictions = nn_model(x_batch, training=False)
  y_pred = tf.nn.sigmoid(predictions)
  saver = tf.train.Saver(var_list=nn_model.trainable_variables)
  eval_metric = functools.partial(
      pr_auc_score,
      y_true=y_eval,
      pred_tensor=y_pred,
      dataset_init_op=eval_init_op,
  )
  return {
      "nn_model": nn_model,
      "saver": saver,
      "dataset_init_op": eval_init_op,
  }, eval_metric
