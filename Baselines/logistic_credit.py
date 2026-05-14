"""Sklearn Logistic Regression baseline for the Credit Fraud experiment.

This baseline follows the paper at a high level:
  - use sklearn Logistic Regression
  - use the same Credit Fraud preprocessing as the NAM pipeline
  - use an outer 5-fold held-out evaluation
  - tune hyperparameters with grid search on the outer training pool

Unlike the NAM pipeline, this baseline uses sklearn's GridSearchCV with
repeated stratified shuffle splits on the outer training fold instead of
checkpoint-based early stopping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from absl import app
from absl import flags
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_CODE_DIR = PROJECT_ROOT / "Project_Code_NAM"
if str(PROJECT_CODE_DIR) not in sys.path:
  sys.path.insert(0, str(PROJECT_CODE_DIR))

import credit_data_utils  # pylint: disable=wrong-import-position


FLAGS = flags.FLAGS
_N_FOLDS = 5

flags.DEFINE_string(
    "data_path",
    None,
    "Optional local path to creditcard.csv. Defaults to ../credit_dataset/creditcard.csv",
)
flags.DEFINE_string("logdir", None, "Directory for fold outputs.")
flags.DEFINE_integer("fold_num", 1, "Outer CV fold index to use.")
flags.DEFINE_integer(
    "num_splits",
    3,
    "Number of inner stratified shuffle splits for grid search.",
)
flags.DEFINE_float(
    "validation_size",
    0.125,
    "Fraction of the outer training pool reserved for validation in each inner split.",
)
flags.DEFINE_integer("random_seed", 1, "Random seed for reproducibility.")
flags.DEFINE_integer(
    "max_iter",
    1000,
    "Maximum number of solver iterations for Logistic Regression.",
)
flags.DEFINE_enum(
    "refit_metric",
    "roc_auc",
    ["roc_auc", "average_precision"],
    "Metric used by GridSearchCV to refit the best model.",
)
flags.DEFINE_list(
    "c_values",
    ["0.01", "0.1", "1.0", "10.0", "100.0"],
    "Grid of inverse regularization strengths to evaluate.",
)
flags.DEFINE_list(
    "class_weights",
    ["none", "balanced"],
    "Grid of class weight strategies to evaluate: `none`, `balanced`.",
)
flags.DEFINE_list(
    "solvers",
    ["liblinear", "lbfgs"],
    "Grid of sklearn solver names to evaluate.",
)
flags.DEFINE_boolean(
    "save_model",
    True,
    "Whether to persist the best fitted Logistic Regression model.",
)


def _parse_c_values():
  return [float(value) for value in FLAGS.c_values]


def _parse_class_weights():
  parsed = []
  for value in FLAGS.class_weights:
    lowered = value.lower()
    if lowered == "none":
      parsed.append(None)
    elif lowered == "balanced":
      parsed.append("balanced")
    else:
      raise ValueError(
          "Unsupported class weight value. Use `none` or `balanced`: "
          f"{value}"
      )
  return parsed


def _build_param_grid():
  return {
      "C": _parse_c_values(),
      "class_weight": _parse_class_weights(),
      "solver": FLAGS.solvers,
      "penalty": ["l2"],
  }


def _load_credit_data():
  data_x, data_y, _ = credit_data_utils.load_dataset(data_path=FLAGS.data_path)
  return credit_data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=FLAGS.fold_num,
      num_folds=_N_FOLDS,
  )


def _create_inner_cv():
  return StratifiedShuffleSplit(
      n_splits=FLAGS.num_splits,
      test_size=FLAGS.validation_size,
      random_state=FLAGS.random_seed,
  )


def _create_estimator():
  return LogisticRegression(
      max_iter=FLAGS.max_iter,
      random_state=FLAGS.random_seed,
  )


def _score_predictions(y_true, y_score):
  return {
      "test_auc": roc_auc_score(y_true, y_score),
      "test_prauc": average_precision_score(y_true, y_score),
  }


def _write_summary(logdir, summary):
  summary_path = Path(logdir) / f"fold_{FLAGS.fold_num}_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_terminal_summary(summary):
  print(json.dumps(summary, indent=2))
  print(
      "Completed Logistic Regression fold "
      f"{summary['fold_num']} with held-out test AUC "
      f"{summary['test_auc']:.4f} and PRAUC {summary['test_prauc']:.4f}"
  )


def run_fold(logdir: Path):
  """Runs one outer fold for the Logistic Regression baseline."""
  (x_train, y_train), (x_test, y_test) = _load_credit_data()
  cv = _create_inner_cv()
  grid_search = GridSearchCV(
      estimator=_create_estimator(),
      param_grid=_build_param_grid(),
      scoring={
          "roc_auc": "roc_auc",
          "average_precision": "average_precision",
      },
      refit=FLAGS.refit_metric,
      cv=cv,
      n_jobs=-1,
      verbose=0,
      return_train_score=False,
  )
  grid_search.fit(x_train, y_train)

  y_score = grid_search.predict_proba(x_test)[:, 1]
  test_metrics = _score_predictions(y_test, y_score)
  summary = {
      "fold_num": FLAGS.fold_num,
      "num_splits": FLAGS.num_splits,
      "validation_size": FLAGS.validation_size,
      "refit_metric": FLAGS.refit_metric,
      "best_params": grid_search.best_params_,
      "best_validation_auc": grid_search.cv_results_["mean_test_roc_auc"][
          grid_search.best_index_
      ],
      "best_validation_prauc": grid_search.cv_results_[
          "mean_test_average_precision"
      ][grid_search.best_index_],
      "test_auc": test_metrics["test_auc"],
      "test_prauc": test_metrics["test_prauc"],
      "source": "logistic_regression_final",
      "logdir": str(logdir),
      "model_type": "sklearn_logistic_regression",
  }

  if FLAGS.save_model:
    model_path = logdir / f"fold_{FLAGS.fold_num}_best_model.joblib"
    joblib.dump(grid_search.best_estimator_, model_path)
    summary["model_path"] = str(model_path)

  _write_summary(logdir, summary)
  _write_terminal_summary(summary)


def main(argv):
  del argv
  logdir = Path(FLAGS.logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  run_fold(logdir)


if __name__ == "__main__":
  flags.mark_flag_as_required("logdir")
  app.run(main)
