"""Training script for the credit fraud Neural Additive Model experiment."""

import json
import operator
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import credit_data_utils
import credit_graph_builder

FLAGS = flags.FLAGS
_N_FOLDS = 5

flags.DEFINE_string(
    "data_path",
    None,
    "Optional local path to creditcard.csv. Defaults to ../credit_dataset/creditcard.csv",
)
flags.DEFINE_integer("training_epochs", None, "Number of training epochs.")
flags.DEFINE_float("learning_rate", 0.0157, "Learning rate.")
flags.DEFINE_float("output_regularization", 0.0, "Feature output penalty.")
flags.DEFINE_float("l2_regularization", 4.95e-6, "L2 weight decay.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_string("logdir", None, "Directory for checkpoints and summaries.")
flags.DEFINE_float("decay_rate", 0.995, "Per-epoch learning-rate decay.")
flags.DEFINE_float("dropout", 0.8, "Hidden-unit dropout rate.")
flags.DEFINE_integer("data_split", 1, "Validation split index to use.")
flags.DEFINE_integer("tf_seed", 1, "TensorFlow random seed.")
flags.DEFINE_float(
    "feature_dropout",
    0.0,
    "Probability of dropping whole feature subnetworks.",
)
flags.DEFINE_integer(
    "num_basis_functions",
    1024,
    "Maximum number of basis functions per feature net.",
)
flags.DEFINE_integer(
    "units_multiplier",
    2,
    "Multiplier used when sizing feature nets from unique feature values.",
)
flags.DEFINE_integer(
    "max_checkpoints_to_keep",
    1,
    "Maximum number of recent checkpoints to keep.",
)
flags.DEFINE_integer(
    "save_checkpoint_every_n_epochs",
    10,
    "Save and evaluate every N epochs.",
)
flags.DEFINE_integer("n_models", 1, "Number of NAMs to train.")
flags.DEFINE_integer("num_splits", 3, "Number of validation splits to sample.")
flags.DEFINE_integer("fold_num", 1, "Outer CV fold index to use.")
flags.DEFINE_string(
    "activation",
    "exu",
    "Activation to use in feature nets: `relu` or `exu`.",
)
flags.DEFINE_boolean("debug", False, "Enable TensorBoard summary logging.")
flags.DEFINE_boolean(
    "keep_tf_artifacts",
    False,
    "Keep TensorFlow graph/event artifacts such as graph.pbtxt and tfevents.",
)
flags.DEFINE_boolean(
    "shallow",
    True,
    "Whether to use shallow one-hidden-layer feature nets.",
)
flags.DEFINE_integer(
    "early_stopping_epochs",
    60,
    "Stop if validation AUC has not improved for this many epochs.",
)


def _write_fold_summary(
    logdir,
    mean_train_prauc,
    mean_train_auc,
    mean_validation_prauc,
    mean_validation_auc,
    mean_test_prauc,
    mean_test_auc,
):
  """Writes a structured summary for one completed fold run."""
  summary = {
      "fold_num": FLAGS.fold_num,
      "data_split": FLAGS.data_split,
      "num_splits": FLAGS.num_splits,
      "mean_train_prauc": mean_train_prauc,
      "mean_train_auc": mean_train_auc,
      "mean_validation_prauc": mean_validation_prauc,
      "mean_validation_auc": mean_validation_auc,
      "mean_test_prauc": mean_test_prauc,
      "mean_test_auc": mean_test_auc,
      "source": "final_training",
      "logdir": logdir,
  }
  summary_path = os.path.join(
      logdir,
      f"fold_{FLAGS.fold_num}_summary.json",
  )
  with tf.io.gfile.GFile(summary_path, "w") as summary_file:
    summary_file.write(json.dumps(summary, indent=2))


def _get_train_and_lr_decay_ops(graph_tensors_and_ops, early_stopping):
  """Returns active training and learning-rate decay ops."""
  train_ops = [
      graph["train_op"]
      for index, graph in enumerate(graph_tensors_and_ops)
      if not early_stopping[index]
  ]
  lr_decay_ops = [
      graph["lr_decay_op"]
      for index, graph in enumerate(graph_tensors_and_ops)
      if not early_stopping[index]
  ]
  return train_ops, lr_decay_ops


def _update_latest_checkpoint(checkpoint_dir, best_checkpoint_dir):
  """Copies the latest checkpoint into the best-checkpoint directory."""
  for filename in tf.io.gfile.glob(os.path.join(best_checkpoint_dir, "model.*")):
    tf.io.gfile.remove(filename)
  copied_prefixes = set()
  for name in tf.io.gfile.glob(os.path.join(checkpoint_dir, "model.*")):
    tf.io.gfile.copy(
        name,
        os.path.join(best_checkpoint_dir, os.path.basename(name)),
        overwrite=True,
    )
    basename = os.path.basename(name)
    if basename.startswith("model.ckpt-"):
      parts = basename.split(".")
      if len(parts) >= 2:
        copied_prefixes.add(parts[0])
  if copied_prefixes:
    latest_prefix = sorted(copied_prefixes)[-1]
    checkpoint_text = (
        f'model_checkpoint_path: "{latest_prefix}"\n'
        f'all_model_checkpoint_paths: "{latest_prefix}"\n'
    )
    with tf.io.gfile.GFile(
        os.path.join(best_checkpoint_dir, "checkpoint"), "w"
    ) as checkpoint_file:
      checkpoint_file.write(checkpoint_text)


def _resolve_best_checkpoint_path(best_checkpoint_dir):
  """Resolves the best checkpoint path even if the state file is missing."""
  checkpoint_path = tf.train.latest_checkpoint(best_checkpoint_dir)
  if checkpoint_path is not None:
    return checkpoint_path

  index_files = tf.io.gfile.glob(
      os.path.join(best_checkpoint_dir, "model.ckpt-*.index")
  )
  if not index_files:
    return None

  latest_index = sorted(index_files)[-1]
  return latest_index[:-len(".index")]


def _create_computation_graph(
    x_train,
    y_train,
    x_validation,
    y_validation,
    batch_size,
):
  """Builds one or more NAM computation graphs."""
  graph_tensors_and_ops = []
  metric_scores = []
  for model_index in range(FLAGS.n_models):
    graph_tensors_and_ops_n, metric_scores_n = credit_graph_builder.build_graph(
        x_train=x_train,
        y_train=y_train,
        x_test=x_validation,
        y_test=y_validation,
        activation=FLAGS.activation,
        learning_rate=FLAGS.learning_rate,
        batch_size=batch_size,
        shallow=FLAGS.shallow,
        output_regularization=FLAGS.output_regularization,
        l2_regularization=FLAGS.l2_regularization,
        dropout=FLAGS.dropout,
        num_basis_functions=FLAGS.num_basis_functions,
        units_multiplier=FLAGS.units_multiplier,
        decay_rate=FLAGS.decay_rate,
        feature_dropout=FLAGS.feature_dropout,
        trainable=True,
        name_scope=f"model_{model_index}",
    )
    graph_tensors_and_ops.append(graph_tensors_and_ops_n)
    metric_scores.append(metric_scores_n)
  return graph_tensors_and_ops, metric_scores


def _create_graph_saver(graph_tensors_and_ops, logdir, num_steps_per_epoch):
  """Creates savers, checkpoint directories, and saver hooks."""
  saver_hooks, model_dirs, best_checkpoint_dirs = [], [], []
  save_steps = num_steps_per_epoch * FLAGS.save_checkpoint_every_n_epochs
  save_steps *= FLAGS.n_models
  for model_index in range(FLAGS.n_models):
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            var_list=graph_tensors_and_ops[model_index]["nn_model"].trainable_variables,
            save_relative_paths=True,
            max_to_keep=FLAGS.max_checkpoints_to_keep,
        )
    )
    model_dir = os.path.join(logdir, f"model_{model_index}")
    best_checkpoint_dir = os.path.join(model_dir, "best_checkpoint")
    tf.io.gfile.makedirs(best_checkpoint_dir)
    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=model_dir,
        save_steps=save_steps,
        scaffold=scaffold,
        save_graph_def=False,
    )
    saver_hooks.append(saver_hook)
    model_dirs.append(model_dir)
    best_checkpoint_dirs.append(best_checkpoint_dir)
  return saver_hooks, model_dirs, best_checkpoint_dirs


def _cleanup_tensorflow_artifacts(model_dirs):
  """Removes oversized TensorFlow event/graph files unless explicitly kept."""
  if FLAGS.keep_tf_artifacts or FLAGS.debug:
    return
  patterns = ["events.out.tfevents*", "graph.pbtxt"]
  for model_dir in model_dirs:
    for pattern in patterns:
      for path in tf.io.gfile.glob(os.path.join(model_dir, pattern)):
        tf.io.gfile.remove(path)


def _update_metrics_and_checkpoints(
    sess,
    epoch,
    metric_scores,
    curr_best_epoch,
    best_validation_prauc,
    best_train_prauc,
    best_validation_auc,
    best_train_auc,
    model_dir,
    best_checkpoint_dir,
):
  """Evaluates validation PR AUC and updates best-checkpoint bookkeeping."""
  compare_metric = operator.gt
  validation_prauc = metric_scores["test_pr"](sess)
  validation_auc = metric_scores["test_auc"](sess)
  if FLAGS.debug:
    tf.logging.info(
        "Epoch %d Val PRAUC %.4f AUC %.4f",
        epoch,
        validation_prauc,
        validation_auc,
    )
  if compare_metric(validation_prauc, best_validation_prauc):
    curr_best_epoch = epoch
    best_validation_prauc = validation_prauc
    best_train_prauc = metric_scores["train_pr"](sess)
    best_validation_auc = validation_auc
    best_train_auc = metric_scores["train_auc"](sess)
    _update_latest_checkpoint(model_dir, best_checkpoint_dir)
  return (
      curr_best_epoch,
      best_validation_prauc,
      best_train_prauc,
      best_validation_auc,
      best_train_auc,
  )


def _evaluate_best_checkpoint_on_test(
    x_model_reference,
    x_test,
    y_test,
    best_checkpoint_dirs,
):
  """Restores the best checkpoint for each model and scores the outer test fold."""
  batch_size = min(FLAGS.batch_size, x_test.shape[0])
  test_prauc_scores = np.zeros(FLAGS.n_models)
  test_auc_scores = np.zeros(FLAGS.n_models)
  for model_index in range(FLAGS.n_models):
    checkpoint_path = _resolve_best_checkpoint_path(
        best_checkpoint_dirs[model_index]
    )
    if checkpoint_path is None:
      raise ValueError(
          "No best checkpoint found for test evaluation at "
          f"{best_checkpoint_dirs[model_index]}. "
          "Make sure training ran long enough to save a checkpoint."
      )

    tf.reset_default_graph()
    with tf.Graph().as_default():
      tf.compat.v1.set_random_seed(FLAGS.tf_seed)
      eval_graph_tensors, test_metrics = credit_graph_builder.build_eval_graph(
          x_model_reference=x_model_reference,
          x_eval=x_test,
          y_eval=y_test,
          batch_size=batch_size,
          activation=FLAGS.activation,
          num_basis_functions=FLAGS.num_basis_functions,
          units_multiplier=FLAGS.units_multiplier,
          shallow=FLAGS.shallow,
          name_scope=f"model_{model_index}",
      )
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        eval_graph_tensors["saver"].restore(sess, checkpoint_path)
        test_prauc_scores[model_index] = test_metrics["pr_auc"](sess)
        test_auc_scores[model_index] = test_metrics["roc_auc"](sess)
        tf.logging.info(
            "Model %d: Outer Test PRAUC %.4f, AUC %.4f",
            model_index,
            test_prauc_scores[model_index],
            test_auc_scores[model_index],
        )
  return test_prauc_scores, test_auc_scores


def training(x_train, y_train, x_validation, y_validation, x_test, y_test, logdir):
  """Trains the credit-fraud NAM and returns best train/val/test metrics."""
  tf.logging.info("Started training with logdir %s", logdir)
  batch_size = min(FLAGS.batch_size, x_train.shape[0])
  num_steps_per_epoch = x_train.shape[0] // batch_size
  best_train_prauc = np.zeros(FLAGS.n_models)
  best_validation_prauc = np.zeros(FLAGS.n_models)
  best_train_auc = np.zeros(FLAGS.n_models)
  best_validation_auc = np.zeros(FLAGS.n_models)
  curr_best_epoch = np.full(FLAGS.n_models, np.inf)
  early_stopping = [False] * FLAGS.n_models

  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.compat.v1.set_random_seed(FLAGS.tf_seed)
    graph_tensors_and_ops, metric_scores = _create_computation_graph(
        x_train,
        y_train,
        x_validation,
        y_validation,
        batch_size,
    )
    train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
      graph_tensors_and_ops,
      early_stopping,
    )
    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    saver_hooks, model_dirs, best_checkpoint_dirs = _create_graph_saver(
        graph_tensors_and_ops,
        logdir,
        num_steps_per_epoch,
    )
    if FLAGS.debug:
      summary_writer = tf.summary.FileWriter(os.path.join(logdir, "tb_log"))

    with tf.train.MonitoredSession(hooks=saver_hooks) as sess:
      for model_index in range(FLAGS.n_models):
        sess.run([
            graph_tensors_and_ops[model_index]["iterator_initializer"],
            graph_tensors_and_ops[model_index]["running_vars_initializer"],
        ])
      for epoch in range(1, FLAGS.training_epochs + 1):
        if not all(early_stopping):
          for _ in range(num_steps_per_epoch):
            sess.run(train_ops)
          sess.run(lr_decay_ops)
        else:
          tf.logging.info("All models early stopped at epoch %d", epoch)
          break

        for model_index in range(FLAGS.n_models):
          if early_stopping[model_index]:
            sess.run(increment_global_step)
            continue

          if FLAGS.debug:
            global_summary, current_global_step = sess.run([
                graph_tensors_and_ops[model_index]["summary_op"],
                graph_tensors_and_ops[model_index]["global_step"],
            ])
            summary_writer.add_summary(global_summary, current_global_step)

          if epoch % FLAGS.save_checkpoint_every_n_epochs == 0:
            (
                curr_best_epoch[model_index],
                best_validation_prauc[model_index],
                best_train_prauc[model_index],
                best_validation_auc[model_index],
                best_train_auc[model_index],
            ) = _update_metrics_and_checkpoints(
                sess,
                epoch,
                metric_scores[model_index],
                curr_best_epoch[model_index],
                best_validation_prauc[model_index],
                best_train_prauc[model_index],
                best_validation_auc[model_index],
                best_train_auc[model_index],
                model_dirs[model_index],
                best_checkpoint_dirs[model_index],
            )
            if curr_best_epoch[model_index] + FLAGS.early_stopping_epochs < epoch:
              tf.logging.info("Early stopping at epoch %d", epoch)
              early_stopping[model_index] = True
              train_ops, lr_decay_ops = _get_train_and_lr_decay_ops(
                  graph_tensors_and_ops,
                  early_stopping,
              )
          sess.run(graph_tensors_and_ops[model_index]["running_vars_initializer"])

  tf.logging.info("Finished training.")
  for model_index in range(FLAGS.n_models):
    tf.logging.info(
        "Model %d: Best Epoch %d, Train PRAUC %.4f, Train AUC %.4f, Validation PRAUC %.4f, Validation AUC %.4f",
        model_index,
        curr_best_epoch[model_index],
        best_train_prauc[model_index],
        best_train_auc[model_index],
        best_validation_prauc[model_index],
        best_validation_auc[model_index],
    )
  test_prauc_scores, test_auc_scores = _evaluate_best_checkpoint_on_test(
      x_model_reference=x_train,
      x_test=x_test,
      y_test=y_test,
      best_checkpoint_dirs=best_checkpoint_dirs,
  )
  _cleanup_tensorflow_artifacts(model_dirs)
  tf.logging.info(
      "Mean Outer Test PRAUC %.4f, Mean Outer Test AUC %.4f",
      np.mean(test_prauc_scores),
      np.mean(test_auc_scores),
  )
  return (
      np.mean(best_train_prauc),
      np.mean(best_train_auc),
      np.mean(best_validation_prauc),
      np.mean(best_validation_auc),
      np.mean(test_prauc_scores),
      np.mean(test_auc_scores),
  )


def create_test_train_fold(fold_num):
  """Loads credit fraud data and creates the outer fold split."""
  data_x, data_y, _ = credit_data_utils.load_dataset(data_path=FLAGS.data_path)
  tf.logging.info("Dataset: Credit, Size: %d", data_x.shape[0])
  tf.logging.info("Cross-val fold: %d/%d", FLAGS.fold_num, _N_FOLDS)
  (x_train_all, y_train_all), test_dataset = credit_data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=fold_num,
      num_folds=_N_FOLDS,
  )
  data_gen = credit_data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      FLAGS.num_splits,
  )
  return data_gen, test_dataset


def single_split_training(data_gen, test_dataset, logdir):
  """Uses one validation split for training."""
  for _ in range(FLAGS.data_split):
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)
  x_test, y_test = test_dataset
  curr_logdir = os.path.join(
      logdir,
      f"fold_{FLAGS.fold_num}",
      f"split_{FLAGS.data_split}",
  )
  return training(
      x_train,
      y_train,
      x_validation,
      y_validation,
      x_test,
      y_test,
      curr_logdir,
  )


def main(argv):
  del argv
  tf.logging.set_verbosity(tf.logging.INFO)
  data_gen, test_dataset = create_test_train_fold(FLAGS.fold_num)
  (
      mean_train_prauc,
      mean_train_auc,
      mean_validation_prauc,
      mean_validation_auc,
      mean_test_prauc,
      mean_test_auc,
  ) = single_split_training(data_gen, test_dataset, FLAGS.logdir)
  _write_fold_summary(
      logdir=FLAGS.logdir,
      mean_train_prauc=mean_train_prauc,
      mean_train_auc=mean_train_auc,
      mean_validation_prauc=mean_validation_prauc,
      mean_validation_auc=mean_validation_auc,
      mean_test_prauc=mean_test_prauc,
      mean_test_auc=mean_test_auc,
  )
  tf.logging.info(
      "Completed fold %d with held-out test PRAUC %.4f and AUC %.4f",
      FLAGS.fold_num,
      mean_test_prauc,
      mean_test_auc,
  )


if __name__ == "__main__":
  flags.mark_flag_as_required("logdir")
  flags.mark_flag_as_required("training_epochs")
  app.run(main)
