"""XGBoost baseline for the Credit Fraud experiment.

This baseline follows the same outer 5-fold evaluation structure used in the
project. Unlike the NAM reproduction, it does not run a dedicated inner
hyperparameter search because the paper does not document a specific XGBoost
search protocol in the same way it documents NAM tuning.

The script:
  - reuses the same Credit Fraud preprocessing as the NAM pipeline
  - trains on the 4 outer training folds
  - evaluates on the held-out outer fold
  - reports both ROC AUC and PR AUC
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from absl import app
from absl import flags
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


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
flags.DEFINE_integer("random_seed", 1, "Random seed for reproducibility.")
flags.DEFINE_integer("n_estimators", 300, "Number of boosting rounds.")
flags.DEFINE_integer("max_depth", 6, "Maximum depth of each tree.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
flags.DEFINE_float("subsample", 0.8, "Row subsample ratio per tree.")
flags.DEFINE_float(
    "colsample_bytree",
    0.8,
    "Feature subsample ratio per tree.",
)
flags.DEFINE_float("reg_lambda", 1.0, "L2 regularization strength.")
flags.DEFINE_integer("min_child_weight", 1, "Minimum child weight.")
flags.DEFINE_float("gamma", 0.0, "Minimum loss reduction for a split.")
flags.DEFINE_boolean(
    "use_scale_pos_weight",
    True,
    "Whether to set scale_pos_weight from the outer training fold imbalance.",
)
flags.DEFINE_boolean(
    "save_model",
    True,
    "Whether to persist the fitted XGBoost model as JSON.",
)


def _load_credit_data():
  data_x, data_y, _ = credit_data_utils.load_dataset(data_path=FLAGS.data_path)
  return credit_data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=FLAGS.fold_num,
      num_folds=_N_FOLDS,
  )


def _compute_scale_pos_weight(y_train):
  neg_count = float((y_train == 0).sum())
  pos_count = float((y_train == 1).sum())
  if pos_count == 0:
    return 1.0
  return neg_count / pos_count


def _create_estimator(scale_pos_weight):
  return XGBClassifier(
      n_estimators=FLAGS.n_estimators,
      max_depth=FLAGS.max_depth,
      learning_rate=FLAGS.learning_rate,
      subsample=FLAGS.subsample,
      colsample_bytree=FLAGS.colsample_bytree,
      reg_lambda=FLAGS.reg_lambda,
      min_child_weight=FLAGS.min_child_weight,
      gamma=FLAGS.gamma,
      objective="binary:logistic",
      eval_metric="auc",
      random_state=FLAGS.random_seed,
      n_jobs=-1,
      tree_method="hist",
      scale_pos_weight=scale_pos_weight,
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
      "Completed XGBoost fold "
      f"{summary['fold_num']} with held-out test AUC "
      f"{summary['test_auc']:.4f} and PRAUC {summary['test_prauc']:.4f}"
  )


def run_fold(logdir: Path):
  """Runs one outer fold for the XGBoost baseline."""
  (x_train, y_train), (x_test, y_test) = _load_credit_data()
  scale_pos_weight = (
      _compute_scale_pos_weight(y_train) if FLAGS.use_scale_pos_weight else 1.0
  )
  model = _create_estimator(scale_pos_weight=scale_pos_weight)
  model.fit(x_train, y_train)

  y_score = model.predict_proba(x_test)[:, 1]
  test_metrics = _score_predictions(y_test, y_score)
  summary = {
      "fold_num": FLAGS.fold_num,
      "test_auc": test_metrics["test_auc"],
      "test_prauc": test_metrics["test_prauc"],
      "scale_pos_weight": scale_pos_weight,
      "params": {
          "n_estimators": FLAGS.n_estimators,
          "max_depth": FLAGS.max_depth,
          "learning_rate": FLAGS.learning_rate,
          "subsample": FLAGS.subsample,
          "colsample_bytree": FLAGS.colsample_bytree,
          "reg_lambda": FLAGS.reg_lambda,
          "min_child_weight": FLAGS.min_child_weight,
          "gamma": FLAGS.gamma,
          "tree_method": "hist",
          "use_scale_pos_weight": FLAGS.use_scale_pos_weight,
      },
      "source": "xgboost_final",
      "logdir": str(logdir),
      "model_type": "xgboost_classifier",
  }

  if FLAGS.save_model:
    model_path = logdir / f"fold_{FLAGS.fold_num}_model.json"
    model.save_model(model_path)
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
