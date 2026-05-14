"""Credit fraud data loading and splitting utilities for NAM experiments.

This file is a trimmed adaptation of the authors' `data_utils.py` and keeps
only the code paths needed for the credit card fraud dataset.
"""

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

DatasetType = Tuple[np.ndarray, np.ndarray]

_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "credit_dataset" / "creditcard.csv"
)


class CustomPipeline(Pipeline):
  """Custom sklearn Pipeline to transform data."""

  def apply_transformation(self, x):
    """Applies all transforms to the data, without applying last estimator."""
    xt = x
    for _, transform in self.steps[:-1]:
      xt = transform.fit_transform(xt)
    return xt


def load_credit_data(data_path: str = None):
  """Loads the Credit Card Fraud Detection dataset from a local CSV file."""
  csv_path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
  if not csv_path.exists():
    raise FileNotFoundError(
        "Credit fraud dataset not found. "
        f"Expected CSV at: {csv_path}"
    )

  df = pd.read_csv(csv_path).dropna()
  train_cols = df.columns[0:-1]
  label = df.columns[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      "problem": "classification",
      "X": x_df,
      "y": y_df,
  }


def transform_data(df: pd.DataFrame):
  """One-hot encodes categoricals and scales every feature to [-1, 1]."""
  column_names = df.columns
  new_column_names: List[str] = []
  is_categorical = np.array([dt.kind == "O" for dt in df.dtypes])
  categorical_cols = df.columns.values[is_categorical]
  numerical_cols = df.columns.values[~is_categorical]

  for index, is_cat in enumerate(is_categorical):
    col_name = column_names[index]
    if is_cat:
      new_column_names += [
          f"{col_name}: {val}" for val in sorted(set(df[col_name]))
      ]
    else:
      new_column_names.append(col_name)

  cat_ohe_step = (
      "ohe",
      OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
  )
  cat_pipe = Pipeline([cat_ohe_step])
  num_pipe = Pipeline([("identity", FunctionTransformer(validate=True))])
  transformers = [
      ("cat", cat_pipe, categorical_cols),
      ("num", num_pipe, numerical_cols),
  ]
  column_transform = ColumnTransformer(transformers=transformers)

  pipe = CustomPipeline([
      ("column_transform", column_transform),
      ("min_max", MinMaxScaler((-1, 1))),
      ("dummy", None),
  ])
  transformed = pipe.apply_transformation(df)
  return transformed, new_column_names


def load_dataset(data_path: str = None):
  """Loads and transforms the credit fraud dataset."""
  dataset = load_credit_data(data_path=data_path)
  data_x, data_y = dataset["X"].copy(), dataset["y"].copy()
  data_x, column_names = transform_data(data_x)
  data_x = data_x.astype("float32")
  if not isinstance(data_y, np.ndarray):
    data_y = pd.get_dummies(data_y).values
    data_y = np.argmax(data_y, axis=-1)
  data_y = data_y.astype("float32")
  return data_x, data_y, column_names


def get_train_test_fold(
    data_x: np.ndarray,
    data_y: np.ndarray,
    fold_num: int,
    num_folds: int,
    random_state: int = 42,
):
  """Returns one outer train/test fold using stratified K-fold splitting."""
  stratified_k_fold = StratifiedKFold(
      n_splits=num_folds,
      shuffle=True,
      random_state=random_state,
  )
  assert 1 <= fold_num <= num_folds, "Pass a valid fold number."
  for train_index, test_index in stratified_k_fold.split(data_x, data_y):
    if fold_num == 1:
      x_train, x_test = data_x[train_index], data_x[test_index]
      y_train, y_test = data_y[train_index], data_y[test_index]
      return (x_train, y_train), (x_test, y_test)
    fold_num -= 1


def split_training_dataset(
    data_x: np.ndarray,
    data_y: np.ndarray,
    n_splits: int,
    test_size: float = 0.125,
    random_state: int = 1337,
) -> Iterator[Tuple[DatasetType, DatasetType]]:
  """Yields train/validation splits from the outer training fold."""
  stratified_shuffle_split = StratifiedShuffleSplit(
      n_splits=n_splits,
      test_size=test_size,
      random_state=random_state,
  )
  split_gen = stratified_shuffle_split.split(data_x, data_y)

  for train_index, validation_index in split_gen:
    x_train, x_validation = data_x[train_index], data_x[validation_index]
    y_train, y_validation = data_y[train_index], data_y[validation_index]
    assert x_train.shape[0] == y_train.shape[0]
    yield (x_train, y_train), (x_validation, y_validation)
