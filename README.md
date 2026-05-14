# ML436-536_EC_O2_Neural-Additive-Models

This repository contains our reproduction work for the paper:

- Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., and Hinton, G. E. *Neural Additive Models: Interpretable Machine Learning with Neural Nets*. NeurIPS 2021.

Primary reference files in this workspace:

- [Full paper with appendix](../NAM_Full_Paper.pdf)
- [Original author code copy](./Author_Code)

## Primary Experiment Goal

Our primary experiment is to reproduce the **single-task Credit Fraud NAM result** from the paper and compare it against:

- Logistic Regression
- XGBoost
- Neural Additive Models (NAM)

The immediate focus of this repository is the NAM pipeline for the Credit Fraud dataset, including:

- credit-fraud-specific data loading and preprocessing
- fold-based training and validation
- held-out fold evaluation
- checkpointing and best-model restoration
- Bayesian hyperparameter tuning over the paper-aligned regularization parameters

## Metric Note

There is a metric inconsistency between the paper appendix text and the main
results table.

- The appendix wording suggests precision-recall AUC for imbalanced
  classification.
- The main single-task results table labels the Credit Fraud result as `AUC`.
- The original author code uses `ROC AUC` for classification and `RMSE` for
  regression.

For this reason, the paper-faithful reproduction target for Credit Fraud in
this repository is:

- `ROC AUC` for comparison against the published table

We also compute:

- `PR AUC` as an additional diagnostic metric for the imbalanced fraud dataset

If the reproduced `ROC AUC` matches the paper table while `PR AUC` is much
lower, that is expected and does not by itself indicate a failed reproduction.

## Project Code

The trimmed experiment code used for our reproduction lives in:

- [Project_Code/credit_data_utils.py](./Project_Code/credit_data_utils.py)
- [Project_Code/credit_nam_models.py](./Project_Code/credit_nam_models.py)
- [Project_Code/credit_graph_builder.py](./Project_Code/credit_graph_builder.py)
- [Project_Code/train_credit_nam.py](./Project_Code/train_credit_nam.py)
- [Project_Code/tune_credit_nam.py](./Project_Code/tune_credit_nam.py)
- [Project_Code/aggregate_credit_results.py](./Project_Code/aggregate_credit_results.py)

These files are adapted from the authors' released implementation, but narrowed to the Credit Fraud experiment so the training flow is easier to inspect and reproduce.

## Tuned Hyperparameters

Best tuned hyperparameters are written to:

- [Tuned_Hyperparameters](./Tuned_Hyperparameters)

The tuning script stores one JSON file per fold so the final fold runs can reuse the selected settings directly.

## How To Run

Create and activate a Conda environment first, then run commands from the repository root.

### 0. Create the Conda environment

```powershell
conda create -n nam python=3.9 -y
conda activate nam
python -m pip install --upgrade pip
python -m pip install tensorflow==2.15.1 numpy pandas scikit-learn absl-py optuna pypdf
```

Use a placeholder path to the Credit Fraud dataset in the commands below:

- `<PATH_TO_CREDITCARD_CSV>`

Example:

- `path\to\creditcard.csv`

### 1. Smoke test one training run

```powershell
python Project_Code\train_credit_nam.py `
  --data_path "<PATH_TO_CREDITCARD_CSV>" `
  --logdir runs\credit_verify `
  --training_epochs 5 `
  --fold_num 1 `
  --data_split 1 `
  --num_splits 3 `
  --save_checkpoint_every_n_epochs 5
```

### 2. Tune hyperparameters for one fold

```powershell
python Project_Code\tune_credit_nam.py `
  --data_path "<PATH_TO_CREDITCARD_CSV>" `
  --logdir runs\credit_tune_fold1 `
  --training_epochs 10 `
  --fold_num 1 `
  --data_split 1 `
  --num_splits 3 `
  --trials 5 `
  --save_checkpoint_every_n_epochs 10 `
  --run_best_config=false
```

### 3. Train the final model on one fold with chosen hyperparameters

Run `Project_Code\train_credit_nam.py` again using the selected values from the JSON file in `Tuned_Hyperparameters`.

Use the same `fold_num`, `data_split`, and `num_splits` values that were used during tuning for that fold so the final fold run is aligned with the tuned validation setup.

```powershell
python Project_Code\train_credit_nam.py `
  --data_path "<PATH_TO_CREDITCARD_CSV>" `
  --logdir runs\credit_fold1_final `
  --training_epochs 100 `
  --fold_num 1 `
  --data_split 1 `
  --num_splits 3 `
  --save_checkpoint_every_n_epochs 10 `
  --output_regularization <BEST_OUTPUT_REG> `
  --l2_regularization <BEST_WEIGHT_DECAY> `
  --dropout <BEST_DROPOUT> `
  --feature_dropout <BEST_FEATURE_DROPOUT>
```

### 4. Aggregate the 5-fold held-out test metrics

After running the fold workflow, aggregate the saved fold results with:

```powershell
python Project_Code\aggregate_credit_results.py `
  --data_path "<PATH_TO_CREDITCARD_CSV>"
```

The script evaluates the saved best checkpoints from the final fold runs and
writes both:

- held-out test `ROC AUC`
- held-out test `PR AUC`

to the final summary JSON.

## End-to-End Training Flow

1. Load and preprocess the Credit Fraud dataset.
2. Split the data into 5 outer folds.
3. Pick one fold as the held-out test fold.
4. Generate `num_splits` candidate train/validation splits from the remaining 4 folds and choose one with `data_split`.
5. Use Bayesian tuning to search over:
   - output regularization
   - weight decay
   - dropout
   - feature dropout
6. Select the configuration with the best validation PRAUC for that chosen inner split and save it to `Tuned_Hyperparameters`.
7. Train the NAM for the target fold again using the tuned hyperparameters and the same inner-split settings.
8. Restore the best validation checkpoint from that final fold run.
9. Evaluate on the held-out outer test fold and record both final test `PR AUC` and `ROC AUC`.
10. Repeat the tune-then-train workflow for folds 1 through 5.
11. Run `Project_Code\aggregate_credit_results.py` to average the 5 held-out test metrics and compute the standard deviation.
