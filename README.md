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

## Project Code

The trimmed experiment code used for our reproduction lives in:

- [Project_Code/credit_data_utils.py](./Project_Code/credit_data_utils.py)
- [Project_Code/credit_nam_models.py](./Project_Code/credit_nam_models.py)
- [Project_Code/credit_graph_builder.py](./Project_Code/credit_graph_builder.py)
- [Project_Code/train_credit_nam.py](./Project_Code/train_credit_nam.py)
- [Project_Code/tune_credit_nam.py](./Project_Code/tune_credit_nam.py)

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
  --training_epochs 20 `
  --fold_num 1 `
  --data_split 1 `
  --num_splits 3 `
  --trials 10 `
  --save_checkpoint_every_n_epochs 10
```

### 3. Train the final model on one fold with chosen hyperparameters

Run `Project_Code\train_credit_nam.py` again using the selected values from the JSON file in `Tuned_Hyperparameters`.

## End-to-End Training Flow

1. Load and preprocess the Credit Fraud dataset.
2. Split the data into 5 outer folds.
3. Pick one fold as the held-out test fold.
4. Split the remaining 4 folds into train and validation data.
5. Use Bayesian tuning to search over:
   - output regularization
   - weight decay
   - dropout
   - feature dropout
6. Select the configuration with the best validation PRAUC.
7. Train the NAM for the target fold with that configuration.
8. Restore the best validation checkpoint.
9. Evaluate on the held-out outer test fold and record the final test PRAUC.
10. Repeat for folds 1 through 5 and average the held-out test PRAUC values.
