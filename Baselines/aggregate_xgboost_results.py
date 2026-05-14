"""Aggregates held-out XGBoost results across 5 outer folds."""

from __future__ import annotations

import json
import math
from pathlib import Path

from absl import app
from absl import flags


FLAGS = flags.FLAGS
_N_FOLDS = 5

flags.DEFINE_string(
    "runs_dir",
    None,
    "Directory containing XGBoost fold outputs. Defaults to ../runs/xgboost",
)
flags.DEFINE_string(
    "output_json",
    None,
    "Optional path for the aggregate summary JSON.",
)


def _project_root() -> Path:
  return Path(__file__).resolve().parent.parent


def _resolved_runs_dir() -> Path:
  if FLAGS.runs_dir:
    return Path(FLAGS.runs_dir)
  return _project_root() / "runs" / "xgboost"


def _sample_std(values):
  if len(values) < 2:
    return 0.0
  mean_value = sum(values) / len(values)
  variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
  return math.sqrt(variance)


def _find_fold_summary(runs_dir: Path, fold_num: int):
  candidates = sorted(runs_dir.rglob(f"fold_{fold_num}_summary.json"))
  if not candidates:
    return {
        "fold_num": fold_num,
        "test_auc": None,
        "test_prauc": None,
        "source": "missing",
        "summary_path": None,
    }
  summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  return {
      "fold_num": fold_num,
      "test_auc": summary.get("test_auc"),
      "test_prauc": summary.get("test_prauc"),
      "source": summary.get("source", "xgboost_final"),
      "summary_path": str(summary_path),
  }


def _build_summary():
  runs_dir = _resolved_runs_dir()
  fold_results = [
      _find_fold_summary(runs_dir, fold_num)
      for fold_num in range(1, _N_FOLDS + 1)
  ]
  auc_values = [
      fold_result["test_auc"]
      for fold_result in fold_results
      if fold_result["test_auc"] is not None
  ]
  prauc_values = [
      fold_result["test_prauc"]
      for fold_result in fold_results
      if fold_result["test_prauc"] is not None
  ]
  return {
      "fold_results": fold_results,
      "available_auc_fold_count": len(auc_values),
      "mean_test_auc": sum(auc_values) / len(auc_values) if auc_values else None,
      "std_test_auc": _sample_std(auc_values) if auc_values else None,
      "available_prauc_fold_count": len(prauc_values),
      "mean_test_prauc": (
          sum(prauc_values) / len(prauc_values) if prauc_values else None
      ),
      "std_test_prauc": _sample_std(prauc_values) if prauc_values else None,
  }


def main(argv):
  del argv
  summary = _build_summary()
  if FLAGS.output_json:
    output_path = Path(FLAGS.output_json)
  else:
    output_path = _project_root() / "runs" / "xgboost_results_summary.json"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  app.run(main)
