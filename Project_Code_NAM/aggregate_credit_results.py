"""Aggregates held-out PRAUC and ROC AUC across the 5 outer folds.

This script prefers evaluating the saved best checkpoints for each final fold
run so the aggregate report can contain both PRAUC and ROC AUC, even for runs
that were originally logged with PRAUC only.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import credit_data_utils
import credit_graph_builder


FLAGS = flags.FLAGS
_N_FOLDS = 5
_TRAIN_PATTERN = re.compile(r"Train PRAUC ([0-9]*\.?[0-9]+)")
_TRAIN_AUC_PATTERN = re.compile(r"Train AUC ([0-9]*\.?[0-9]+)")
_VALIDATION_PATTERN = re.compile(r"Validation PRAUC ([0-9]*\.?[0-9]+)")
_VALIDATION_AUC_PATTERN = re.compile(r"Validation AUC ([0-9]*\.?[0-9]+)")
_TEST_PATTERN = re.compile(r"held-out test PRAUC ([0-9]*\.?[0-9]+)")
_TEST_AUC_PATTERN = re.compile(
    r"held-out test PRAUC [0-9]*\.?[0-9]+(?: and|, Mean Outer Test AUC| AUC) ?AUC? ?([0-9]*\.?[0-9]+)"
)

flags.DEFINE_string(
    "runs_dir",
    None,
    "Directory containing fold run outputs. Defaults to ../runs",
)
flags.DEFINE_string(
    "tuned_hparams_dir",
    None,
    "Directory containing tuned per-fold JSON files. Defaults to ../Tuned_Hyperparameters",
)
flags.DEFINE_string(
    "output_json",
    None,
    "Optional path for writing the aggregate summary JSON.",
)
flags.DEFINE_string(
    "data_path",
    None,
    "Optional local path to creditcard.csv, required for checkpoint evaluation.",
)
flags.DEFINE_integer("batch_size", 1024, "Batch size used for evaluation.")
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
flags.DEFINE_string(
    "activation",
    "exu",
    "Activation to use in feature nets: `relu` or `exu`.",
)
flags.DEFINE_boolean(
    "shallow",
    True,
    "Whether to use shallow one-hidden-layer feature nets.",
)
flags.DEFINE_integer("tf_seed", 1, "TensorFlow random seed for evaluation.")
flags.DEFINE_boolean(
    "prefer_checkpoint_eval",
    True,
    "Whether to evaluate saved best checkpoints before falling back to logs.",
)


def _project_root() -> Path:
  return Path(__file__).resolve().parent.parent


def _resolved_runs_dir() -> Path:
  if FLAGS.runs_dir:
    return Path(FLAGS.runs_dir)
  return _project_root() / "runs"


def _resolved_tuned_hparams_dir() -> Path:
  if FLAGS.tuned_hparams_dir:
    return Path(FLAGS.tuned_hparams_dir)
  return _project_root() / "Tuned_Hyperparameters"


def _load_json(json_path: Path):
  return json.loads(json_path.read_text(encoding="utf-8"))


def _extract_last_metric(text: str, pattern: re.Pattern):
  matches = pattern.findall(text)
  if not matches:
    return None
  return float(matches[-1])


def _resolve_checkpoint_path(checkpoint_dir: Path):
  """Resolves the checkpoint prefix even if the state file is incomplete."""
  checkpoint_path = tf.train.latest_checkpoint(str(checkpoint_dir))
  if checkpoint_path is not None:
    checkpoint_path = checkpoint_path.replace("\\", "/")
    if tf.io.gfile.exists(checkpoint_path + ".index"):
      return checkpoint_path

  index_files = tf.io.gfile.glob(str(checkpoint_dir / "model.ckpt-*.index"))
  if not index_files:
    return None
  latest_index = sorted(index_files)[-1]
  return latest_index[:-len(".index")]


def _find_final_training_summary(runs_dir: Path, fold_num: int):
  """Returns the most recent final-training summary for the requested fold."""
  candidates = sorted(runs_dir.rglob(f"fold_{fold_num}_summary.json"))
  if not candidates:
    return None
  summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
  data = _load_json(summary_path)
  data["summary_path"] = str(summary_path)
  return data


def _find_final_checkpoint_dir(runs_dir: Path, fold_num: int):
  """Locates the best-checkpoint directory for a final fold run."""
  checkpoint_dir = (
      runs_dir
      / f"credit_fold{fold_num}_final"
      / f"fold_{fold_num}"
      / "split_1"
      / "model_0"
      / "best_checkpoint"
  )
  if checkpoint_dir.exists():
    return checkpoint_dir
  return None


def _evaluate_final_checkpoint(runs_dir: Path, fold_num: int):
  """Evaluates a saved final-run checkpoint directly on the held-out fold."""
  if not FLAGS.data_path:
    return None

  checkpoint_dir = _find_final_checkpoint_dir(runs_dir, fold_num)
  if checkpoint_dir is None:
    return None

  checkpoint_path = _resolve_checkpoint_path(checkpoint_dir)
  if checkpoint_path is None:
    return None

  data_x, data_y, _ = credit_data_utils.load_dataset(data_path=FLAGS.data_path)
  (x_train_all, y_train_all), (x_test, y_test) = credit_data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=fold_num,
      num_folds=_N_FOLDS,
  )
  data_gen = credit_data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      n_splits=3,
  )
  (x_train, _), _ = next(data_gen)

  batch_size = min(FLAGS.batch_size, x_test.shape[0])
  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.compat.v1.set_random_seed(FLAGS.tf_seed)
    eval_graph_tensors, eval_metrics = credit_graph_builder.build_eval_graph(
        x_model_reference=x_train,
        x_eval=x_test,
        y_eval=y_test,
        batch_size=batch_size,
        activation=FLAGS.activation,
        num_basis_functions=FLAGS.num_basis_functions,
        units_multiplier=FLAGS.units_multiplier,
        shallow=FLAGS.shallow,
        name_scope="model_0",
    )
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      try:
        eval_graph_tensors["saver"].restore(sess, checkpoint_path)
      except Exception as exc:  # pragma: no cover - environment-dependent restore
        raise RuntimeError(
            "Failed to restore the saved NAM checkpoint during aggregation. "
            "Run the aggregation command from the same Conda environment used "
            "for training, with the same TensorFlow major/minor version. "
            f"Current executable: {sys.executable}\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Original error: {exc}"
        ) from exc
      test_prauc = eval_metrics["pr_auc"](sess)
      test_auc = eval_metrics["roc_auc"](sess)

  return {
      "fold_num": fold_num,
      "test_prauc": test_prauc,
      "test_auc": test_auc,
      "source": "final_training_checkpoint",
      "summary_path": str(checkpoint_dir),
  }


def _find_final_training_terminal_summary(runs_dir: Path, fold_num: int):
  """Parses a legacy terminal log from a final fold run when JSON is absent."""
  candidates = sorted(runs_dir.rglob("terminal*.txt"))
  for terminal_path in sorted(
      candidates, key=lambda path: path.stat().st_mtime, reverse=True
  ):
    text = terminal_path.read_text(encoding="utf-8", errors="ignore")
    if f"Completed fold {fold_num}" not in text:
      continue
    if f"credit_tune_fold{fold_num}" in str(terminal_path):
      continue
    test_prauc = _extract_last_metric(text, _TEST_PATTERN)
    test_auc = _extract_last_metric(text, _TEST_AUC_PATTERN)
    if test_prauc is None and test_auc is None:
      continue
    return {
        "fold_num": fold_num,
        "mean_train_prauc": _extract_last_metric(text, _TRAIN_PATTERN),
        "mean_train_auc": _extract_last_metric(text, _TRAIN_AUC_PATTERN),
        "mean_validation_prauc": _extract_last_metric(text, _VALIDATION_PATTERN),
        "mean_validation_auc": _extract_last_metric(text, _VALIDATION_AUC_PATTERN),
        "mean_test_prauc": test_prauc,
        "mean_test_auc": test_auc,
        "source": "final_training_terminal",
        "summary_path": str(terminal_path),
    }
  return None


def _find_tuning_summary(tuned_dir: Path, fold_num: int):
  """Returns the tuned-hyperparameter summary for the requested fold."""
  json_path = tuned_dir / f"credit_fold_{fold_num}_best_hparams.json"
  if not json_path.exists():
    return None
  data = _load_json(json_path)
  data["summary_path"] = str(json_path)
  return data


def _resolve_fold_result(runs_dir: Path, tuned_dir: Path, fold_num: int):
  """Resolves the best available test metrics artifact for one fold."""
  final_summary = _find_final_training_summary(runs_dir, fold_num)
  if final_summary is not None:
    return {
        "fold_num": fold_num,
        "test_prauc": final_summary.get("mean_test_prauc"),
        "test_auc": final_summary.get("mean_test_auc"),
        "source": "final_training",
        "summary_path": final_summary["summary_path"],
    }

  if FLAGS.prefer_checkpoint_eval:
    checkpoint_summary = _evaluate_final_checkpoint(runs_dir, fold_num)
    if checkpoint_summary is not None:
      return checkpoint_summary

  legacy_final_summary = _find_final_training_terminal_summary(runs_dir, fold_num)
  if legacy_final_summary is not None:
    return {
        "fold_num": fold_num,
        "test_prauc": legacy_final_summary.get("mean_test_prauc"),
        "test_auc": legacy_final_summary.get("mean_test_auc"),
        "source": legacy_final_summary["source"],
        "summary_path": legacy_final_summary["summary_path"],
    }

  tuning_summary = _find_tuning_summary(tuned_dir, fold_num)
  if tuning_summary is not None:
    return {
        "fold_num": fold_num,
        "test_prauc": tuning_summary.get("best_test_prauc_from_same_run"),
        "test_auc": tuning_summary.get("best_test_auc_from_same_run"),
        "source": "tuning_best_trial",
        "summary_path": tuning_summary["summary_path"],
    }

  return {
      "fold_num": fold_num,
      "test_prauc": None,
      "test_auc": None,
      "source": "missing",
      "summary_path": None,
  }


def _sample_std(values):
  if len(values) < 2:
    return 0.0
  mean_value = sum(values) / len(values)
  variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
  return math.sqrt(variance)


def _build_aggregate_summary():
  runs_dir = _resolved_runs_dir()
  tuned_dir = _resolved_tuned_hparams_dir()
  fold_results = [
      _resolve_fold_result(runs_dir, tuned_dir, fold_num)
      for fold_num in range(1, _N_FOLDS + 1)
  ]
  available_prauc_values = [
      fold_result["test_prauc"]
      for fold_result in fold_results
      if fold_result["test_prauc"] is not None
  ]
  available_auc_values = [
      fold_result["test_auc"]
      for fold_result in fold_results
      if fold_result["test_auc"] is not None
  ]
  mean_prauc = (
      sum(available_prauc_values) / len(available_prauc_values)
      if available_prauc_values else None
  )
  std_prauc = _sample_std(available_prauc_values) if available_prauc_values else None
  mean_auc = (
      sum(available_auc_values) / len(available_auc_values)
      if available_auc_values else None
  )
  std_auc = _sample_std(available_auc_values) if available_auc_values else None
  return {
      "fold_results": fold_results,
      "available_prauc_fold_count": len(available_prauc_values),
      "mean_test_prauc": mean_prauc,
      "std_test_prauc": std_prauc,
      "available_auc_fold_count": len(available_auc_values),
      "mean_test_auc": mean_auc,
      "std_test_auc": std_auc,
  }


def main(argv):
  del argv
  summary = _build_aggregate_summary()
  if FLAGS.output_json:
    output_path = Path(FLAGS.output_json)
  else:
    output_path = _project_root() / "runs" / "credit_results_summary.json"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  app.run(main)
