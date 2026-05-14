# Tuning Notes

This directory stores fold-specific hyperparameter outputs used in the Credit
Fraud reproduction experiments for:

- Neural Additive Models (NAMs)
- Logistic Regression baseline
- XGBoost baseline

The notes below condense the tuning and reporting choices used in the project
and are intended to support the final experiment report.

## Hyperparameter Tuning

### NAM

We tuned the regularization hyperparameters exposed by the trimmed NAM training
pipeline:

- `output_regularization`
- `l2_regularization`
- `dropout`
- `feature_dropout`

The search strategy was **Bayesian optimization** using Optuna's TPE sampler in
[Project_Code_NAM/tune_credit_nam.py](/W:/Binghamton/The%203rd%20Semester_Deci/Machine%20Learning%20/Extra%20credit%20/Neural%20Additive%20Models/ML436-536_EC_O2_Neural-Additive-Models/Project_Code_NAM/tune_credit_nam.py).

Search space:

- `output_regularization`: continuous range `[0.0, 0.1]`
- `l2_regularization`: log-scale range `[1e-6, 1e-4]`
- `dropout`: categorical values `{0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`
- `feature_dropout`: categorical values `{0.0, 0.05, 0.1, 0.2}`

Fixed NAM settings during tuning:

- `learning_rate = 0.0157`
- `batch_size = 1024`
- `decay_rate = 0.995`
- `activation = exu`
- `shallow = True`
- `num_basis_functions = 1024`

Validation protocol:

- 5-fold outer cross validation for final evaluation
- for one chosen outer fold, the remaining 4 folds form the training pool
- from that pool, `num_splits = 3` candidate stratified train/validation splits
  are generated
- one selected inner split (`data_split = 1`) is used during a given tuning run
- Optuna maximizes **validation PR AUC** on that inner split

Final tuned values:

- one JSON file is stored per fold in this directory
- each file records the best hyperparameters found for that outer fold

Brief justification:

- the paper explicitly uses Bayesian optimization for NAM hyperparameters
- the tuned parameters are regularization controls that strongly affect generalization
  on the highly imbalanced Credit Fraud dataset
- limiting the search to these four parameters kept the search feasible while
  staying close to the paper's appendix setup

### Logistic Regression

For Logistic Regression we use the sklearn implementation, consistent with the
paper. The search strategy is **grid search**, implemented in
[Baselines/logistic_credit.py](/W:/Binghamton/The%203rd%20Semester_Deci/Machine%20Learning%20/Extra%20credit%20/Neural%20Additive%20Models/ML436-536_EC_O2_Neural-Additive-Models/Baselines/logistic_credit.py).

Hyperparameters tuned:

- `C` (inverse regularization strength)
- `class_weight`
- `solver`

Grid:

- `C`: `{0.01, 0.1, 1.0, 10.0, 100.0}`
- `class_weight`: `{None, balanced}`
- `solver`: `{'liblinear', 'lbfgs'}`

Validation protocol:

- same 5-fold outer evaluation structure as NAM
- inside the outer training pool, sklearn `GridSearchCV` uses repeated
  `StratifiedShuffleSplit`
- default project setting: `num_splits = 3`, `validation_size = 0.125`
- best model is refit using `roc_auc` by default

Brief justification:

- the paper states that Logistic/Linear Regression uses sklearn and grid search
- `C` and `class_weight` are the most important practical controls for this
  imbalanced classification problem
- the solver grid is small and stable, which keeps the baseline reproducible

### XGBoost

For XGBoost we use the maintained Python package implementation through
[Baselines/xgboost_credit.py](/W:/Binghamton/The%203rd%20Semester_Deci/Machine%20Learning%20/Extra%20credit%20/Neural%20Additive%20Models/ML436-536_EC_O2_Neural-Additive-Models/Baselines/xgboost_credit.py).

Search strategy:

- **fixed-config baseline**

Hyperparameters used:

- `n_estimators = 300`
- `max_depth = 6`
- `learning_rate = 0.1`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 1.0`
- `min_child_weight = 1`
- `gamma = 0.0`
- `tree_method = hist`
- `scale_pos_weight = (# negatives / # positives)` from the outer training fold

Validation protocol:

- same 5-fold outer evaluation structure as NAM and Logistic Regression
- train on the 4 outer training folds
- evaluate on the held-out outer fold

Brief justification:

- the paper references XGBoost as a comparison baseline but does not spell out
  a dedicated XGBoost hyperparameter search procedure in the same detail as NAM
- XGBoost is a strong, mature baseline that often performs well with robust
  default-style settings
- using a stable fixed configuration keeps the baseline reproducible and avoids
  inventing an undocumented paper-specific search space

## Regularization

### NAM

The NAM model uses several forms of regularization:

- **dropout** inside feature subnetworks
- **feature dropout**, which drops whole feature subnet contributions
- **L2 weight decay**
- **feature output regularization**, which penalizes large per-feature outputs

Strength selection:

- these strengths were chosen by Bayesian optimization per outer fold
- the final values are stored in the fold-specific JSON files in this directory

Why appropriate:

- Credit Fraud is extremely imbalanced and relatively easy to overfit
- NAMs contain many learnable feature subnetworks, so controlling feature-level
  and weight-level complexity is important
- dropout and L2-style penalties are consistent with the original paper

### Logistic Regression

The main regularization mechanism is the standard sklearn Logistic Regression
penalty:

- **L2 regularization**

Strength selection:

- controlled through `C`
- smaller `C` means stronger regularization
- chosen through grid search on the inner validation splits

Why appropriate:

- Logistic Regression is a linear baseline, so the key overfitting control is
  the magnitude of the coefficient penalty
- this is the standard and interpretable regularization choice for the model

### XGBoost

XGBoost regularization comes from the tree ensemble itself and the fixed
booster settings:

- tree depth limit through `max_depth`
- shrinkage through `learning_rate`
- row and feature subsampling through `subsample` and `colsample_bytree`
- L2 regularization through `reg_lambda`
- class imbalance handling through `scale_pos_weight`

Strength selection:

- not tuned separately in this reproduction
- chosen as conservative, widely used baseline settings for tabular binary
  classification

Why appropriate:

- Credit Fraud is highly imbalanced
- XGBoost is already a strong nonlinear baseline, so a modest fixed
  regularization profile is a reasonable comparison point when the paper does
  not document a dedicated XGBoost tuning protocol

## Performance Reporting

### Primary metric

For the Credit Fraud reproduction, the primary paper-facing metric is:

- **ROC AUC**

Justification:

- the main results table in the NAM paper reports the Credit Fraud result as `AUC`
- the original author classification code uses `ROC AUC`

### Additional diagnostic metric

We also report:

- **PR AUC**

Justification:

- the dataset is highly imbalanced
- PR AUC is informative for fraud detection, even though it is not the main
  metric used for matching the published table value

### Baselines

The main simple baseline used for comparison is:

- **Logistic Regression**
- **XGBoost**

This is appropriate because:

- it is explicitly named in the paper's comparison set
- it is a standard intelligible baseline for tabular binary classification
- it is a strong nonlinear tabular baseline commonly used as a reference model

An additional conceptual baseline for context is the severe class imbalance of
the dataset itself:

- a naive majority-class classifier would achieve poor discriminative ranking
  despite high raw accuracy, which is why AUC-style metrics are more appropriate

### Variability reporting

Performance is reported over **5 outer cross-validation folds**.

For NAM, Logistic Regression, and XGBoost, the final aggregate reports should include:

- mean test ROC AUC across folds
- standard deviation of test ROC AUC across folds
- optionally mean and standard deviation of test PR AUC

This variability estimate is important because:

- the dataset is imbalanced
- results can vary across fold assignments
- the paper reports mean and standard deviation across cross-validation folds

## Final Values

- NAM fold-specific best hyperparameters are stored as
  `credit_fold_<n>_best_hparams.json`
- Logistic Regression fold-specific best settings are stored inside each fold
  summary JSON written under its run directory
- XGBoost fold-specific fixed settings and the derived `scale_pos_weight` are
  stored inside each fold summary JSON written under its run directory

For the report, use:

1. the tuned NAM values from this directory
2. the Logistic Regression best parameters from each fold summary
3. the XGBoost fold summaries as the documented fixed baseline configuration
4. the 5-fold aggregate mean and standard deviation from the corresponding
   aggregation scripts
