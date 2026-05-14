"""Bayesian hyperparameter tuning driver for the Credit Fraud NAM experiment.

This script tunes the four appendix regularization hyperparameters:
  - output penalty
  - weight decay
  - dropout
  - feature dropout

It delegates actual model training to `train_credit_nam.py`, parses the
validation PRAUC from each run, and records the best trial configuration.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from absl import app
from absl import flags

try:
  import optuna
except ImportError as exc:  # pragma: no cover - user environment dependency
  raise ImportError(
      "Optuna is required for Bayesian hyperparameter tuning. "
      "Install it with `pip install optuna` inside your environment."
  ) from exc


FLAGS = flags.FLAGS
_VALIDATION_PATTERN = re.compile(r"Validation PRAUC ([0-9]*\.?[0-9]+)")
_VALIDATION_AUC_PATTERN = re.compile(r"Validation AUC ([0-9]*\.?[0-9]+)")
_TEST_PATTERN = re.compile(r"held-out test PRAUC ([0-9]*\.?[0-9]+)")
_TEST_AUC_PATTERN = re.compile(r"held-out test PRAUC [0-9]*\.?[0-9]+ and AUC ([0-9]*\.?[0-9]+)")

flags.DEFINE_string(
    "data_path",
    None,
    "Optional local path to creditcard.csv. Defaults to ../credit_dataset/creditcard.csv",
)
flags.DEFINE_string(
    "logdir",
    None,
    "Directory where Bayesian tuning artifacts should be written.",
)
flags.DEFINE_integer(
    "training_epochs",
    100,
    "Maximum number of epochs for each tuning trial.",
)
flags.DEFINE_integer("fold_num", 1, "Outer CV fold index to use.")
flags.DEFINE_integer("data_split", 1, "Validation split index to use.")
flags.DEFINE_integer("num_splits", 3, "Number of validation splits to sample.")
flags.DEFINE_integer("batch_size", 1024, "Batch size.")
flags.DEFINE_float("learning_rate", 0.0157, "Fixed learning rate for tuning runs.")
flags.DEFINE_float("decay_rate", 0.995, "Per-epoch learning-rate decay.")
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
flags.DEFINE_integer(
    "save_checkpoint_every_n_epochs",
    10,
    "Save and evaluate every N epochs.",
)
flags.DEFINE_integer(
    "early_stopping_epochs",
    60,
    "Stop if validation metric has not improved for this many epochs.",
)
flags.DEFINE_integer("n_models", 1, "Number of NAMs to train.")
flags.DEFINE_integer("tf_seed", 1, "TensorFlow random seed.")
flags.DEFINE_integer("trials", 10, "Number of Bayesian optimization trials.")
flags.DEFINE_boolean(
    "run_best_config",
    True,
    "Whether to launch one final fold run with the best hyperparameters.",
)
flags.DEFINE_string(
    "tuned_hparams_dir",
    None,
    "Directory where the best tuned hyperparameters should be stored.",
)


def _project_root() -> Path:
  return Path(__file__).resolve().parent.parent


def _training_script() -> Path:
  return Path(__file__).resolve().parent / "train_credit_nam.py"


def _resolved_tuned_hparams_dir() -> Path:
  """Returns the directory used to store final tuned hyperparameters."""
  if FLAGS.tuned_hparams_dir:
    return Path(FLAGS.tuned_hparams_dir)
  return _project_root() / "Tuned_Hyperparameters"


def _build_training_command(
    trial_logdir: Path,
    output_regularization: float,
    l2_regularization: float,
    dropout: float,
    feature_dropout: float,
):
  """Builds the subprocess command for one tuning trial."""
  python_exe = sys.executable
  command = [
      python_exe,
      str(_training_script()),
      "--logdir",
      str(trial_logdir),
      "--training_epochs",
      str(FLAGS.training_epochs),
      "--fold_num",
      str(FLAGS.fold_num),
      "--data_split",
      str(FLAGS.data_split),
      "--num_splits",
      str(FLAGS.num_splits),
      "--batch_size",
      str(FLAGS.batch_size),
      "--learning_rate",
      str(FLAGS.learning_rate),
      "--decay_rate",
      str(FLAGS.decay_rate),
      "--dropout",
      str(dropout),
      "--feature_dropout",
      str(feature_dropout),
      "--output_regularization",
      str(output_regularization),
      "--l2_regularization",
      str(l2_regularization),
      "--num_basis_functions",
      str(FLAGS.num_basis_functions),
      "--units_multiplier",
      str(FLAGS.units_multiplier),
      "--activation",
      FLAGS.activation,
      "--save_checkpoint_every_n_epochs",
      str(FLAGS.save_checkpoint_every_n_epochs),
      "--early_stopping_epochs",
      str(FLAGS.early_stopping_epochs),
      "--n_models",
      str(FLAGS.n_models),
      "--tf_seed",
      str(FLAGS.tf_seed),
      "--shallow={}".format(str(FLAGS.shallow)),
  ]
  if FLAGS.data_path:
    command.extend(["--data_path", FLAGS.data_path])
  return command


def _extract_metric(output_text: str, pattern: re.Pattern, metric_name: str) -> float:
  """Extracts the last occurrence of a metric from subprocess output."""
  matches = pattern.findall(output_text)
  if not matches:
    raise ValueError(f"Unable to parse {metric_name} from trial output.")
  return float(matches[-1])


def _run_trial(
    trial_number: int,
    output_regularization: float,
    l2_regularization: float,
    dropout: float,
    feature_dropout: float,
):
  """Runs one training trial and returns validation/test metrics."""
  trial_logdir = Path(FLAGS.logdir) / f"trial_{trial_number:03d}"
  trial_logdir.mkdir(parents=True, exist_ok=True)
  command = _build_training_command(
      trial_logdir=trial_logdir,
      output_regularization=output_regularization,
      l2_regularization=l2_regularization,
      dropout=dropout,
      feature_dropout=feature_dropout,
  )
  completed = subprocess.run(
      command,
      cwd=str(_project_root()),
      capture_output=True,
      text=True,
      check=False,
  )
  output_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
  (trial_logdir / "terminal_output.txt").write_text(output_text, encoding="utf-8")
  if completed.returncode != 0:
    raise RuntimeError(
        "Training trial failed. Check "
        f"{trial_logdir / 'terminal_output.txt'} for details."
    )

  validation_prauc = _extract_metric(
      output_text,
      _VALIDATION_PATTERN,
      "validation PRAUC",
  )
  validation_auc = _extract_metric(
      output_text,
      _VALIDATION_AUC_PATTERN,
      "validation AUC",
  )
  test_prauc = _extract_metric(
      output_text,
      _TEST_PATTERN,
      "held-out test PRAUC",
  )
  test_auc = _extract_metric(
      output_text,
      _TEST_AUC_PATTERN,
      "held-out test AUC",
  )
  return validation_prauc, validation_auc, test_prauc, test_auc, trial_logdir


def _objective(trial: optuna.Trial):
  """Optuna objective function for Bayesian tuning."""
  output_regularization = trial.suggest_float(
      "output_regularization",
      0.0,
      0.1,
  )
  l2_regularization = trial.suggest_float(
      "l2_regularization",
      1e-6,
      1e-4,
      log=True,
  )
  dropout = trial.suggest_categorical(
      "dropout",
      [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  )
  feature_dropout = trial.suggest_categorical(
      "feature_dropout",
      [0.0, 0.05, 0.1, 0.2],
  )

  validation_prauc, validation_auc, test_prauc, test_auc, trial_logdir = _run_trial(
      trial.number,
      output_regularization=output_regularization,
      l2_regularization=l2_regularization,
      dropout=dropout,
      feature_dropout=feature_dropout,
  )
  trial.set_user_attr("validation_auc", validation_auc)
  trial.set_user_attr("test_prauc", test_prauc)
  trial.set_user_attr("test_auc", test_auc)
  trial.set_user_attr("trial_logdir", str(trial_logdir))
  return validation_prauc


def _write_study_summary(study: optuna.Study):
  """Writes the best trial metadata to disk."""
  summary = {
      "fold_num": FLAGS.fold_num,
      "data_split": FLAGS.data_split,
      "num_splits": FLAGS.num_splits,
      "best_validation_prauc": study.best_value,
      "best_validation_auc_from_same_run": study.best_trial.user_attrs.get("validation_auc"),
      "best_trial_number": study.best_trial.number,
      "best_params": study.best_trial.params,
      "best_test_prauc_from_same_run": study.best_trial.user_attrs.get("test_prauc"),
      "best_test_auc_from_same_run": study.best_trial.user_attrs.get("test_auc"),
      "best_trial_logdir": study.best_trial.user_attrs.get("trial_logdir"),
      "source": "tuning_best_trial",
  }
  summary_path = Path(FLAGS.logdir) / "best_hparams.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  tuned_dir = _resolved_tuned_hparams_dir()
  tuned_dir.mkdir(parents=True, exist_ok=True)
  tuned_summary_path = tuned_dir / f"credit_fold_{FLAGS.fold_num}_best_hparams.json"
  tuned_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


def _run_best_config(study: optuna.Study):
  """Runs one final fold evaluation using the best hyperparameters."""
  best_dir = Path(FLAGS.logdir) / "best_run"
  best_params = study.best_trial.params
  validation_prauc, validation_auc, test_prauc, test_auc, trial_logdir = _run_trial(
      trial_number=999,
      output_regularization=best_params["output_regularization"],
      l2_regularization=best_params["l2_regularization"],
      dropout=best_params["dropout"],
      feature_dropout=best_params["feature_dropout"],
  )
  final_summary = {
      "validation_prauc": validation_prauc,
      "validation_auc": validation_auc,
      "test_prauc": test_prauc,
      "test_auc": test_auc,
      "run_dir": str(trial_logdir),
  }
  best_dir.mkdir(parents=True, exist_ok=True)
  (best_dir / "final_run_summary.json").write_text(
      json.dumps(final_summary, indent=2),
      encoding="utf-8",
  )
  print(json.dumps(final_summary, indent=2))


def main(argv):
  del argv
  Path(FLAGS.logdir).mkdir(parents=True, exist_ok=True)
  _resolved_tuned_hparams_dir().mkdir(parents=True, exist_ok=True)
  sampler = optuna.samplers.TPESampler(seed=FLAGS.tf_seed)
  study = optuna.create_study(direction="maximize", sampler=sampler)
  study.optimize(_objective, n_trials=FLAGS.trials)
  _write_study_summary(study)
  if FLAGS.run_best_config:
    _run_best_config(study)


if __name__ == "__main__":
  flags.mark_flag_as_required("logdir")
  app.run(main)
