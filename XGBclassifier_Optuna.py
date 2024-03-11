###########################################################
# Optimise XGBoost for Binary Classification using Optuna #
###########################################################
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os

# define variables
num_trials = 100  # number of trials for Optuna optimisation
input_path = ""  # path to where the train and test sets are stored
output_path = ""  # path where everything should be saved
cols_to_scale = []  # columns to scale


######################################################
# Define the XGBOOST model and hyperparameter ranges # Feel free to change hyperparamter ranges
######################################################
def xgboost_objective(trial):
    hyperparams = {
        'use_label_encoder': False,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-9, 1),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.1, 1),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1),
        'eta': trial.suggest_loguniform('eta', 1e-8, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1, 10),  # 1e-8, 10)
        'nthread': -1
    }
    return XGBClassifier(**hyperparams, random_state=42)


base_model = {'model': xgboost_objective}


# Define the cross-validation scheme
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)


#########################################################
# Load data, annotate targets, and separate out X and y #
#########################################################
X = pd.read_csv(f"{input_path}/train.csv")
Xtest = pd.read_csv(f"{input_path}/test.csv")

# Scale shapley values data
scaler = StandardScaler(with_std=True, with_mean=True)

# Scale the selected columns
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
Xtest[cols_to_scale] = scaler.transform(Xtest[cols_to_scale])

# Split x and y
y = X["targ_stat"]
X.drop(columns=["targ_stat"], inplace=True)

ytest = Xtest["targ_stat"]
Xtest.drop(columns=["targ_stat"], inplace=True)

del scaler, cols_to_scale


############################
# Optimise hyperparameters #
############################
# Define the objective
def base_objective(trial):
    # Define the pipeline for the optuna model with oversampling
    pipeline = Pipeline(steps=[
         ('model', base_model['model'](trial))
    ])

    # Define the scores list
    scores = []

    # Loop over the cross-validation folds
    for train_idx, test_idx in cv.split(X, y):
        # Split the data into train and test sets
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]

        # Fit the model and make predictions
        pipeline.fit(X_train.drop(columns=["gene"]), y_train)
        y_pred = pipeline.predict_proba(X_test.drop(columns=["gene"]))

        # Evaluate the predictions
        score = roc_auc_score(y_test, y_pred[:, 1])
        scores.append(score)

    # Calculate the mean score
    mean_score = np.mean(scores)

    # Report the score to Optuna
    return mean_score


# Define the study - study_name argument has to be unique
if os.path.exists(f'study_xgboost.db'):
    sampler = optuna.samplers.TPESampler(seed=0)  # sets a seed for reproducibility
    study = optuna.load_study(study_name="study_xgboost", sampler=sampler,   # allows checkpointing of optimsiation
                              storage=f"sqlite:///{output_path}/study_xgboost.db")
else:
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(study_name="study_xgboost", sampler=sampler, direction="maximize",
                                storage=f"sqlite:///{output_path}/study_xgboost.db")

# Optimize the hyperparameters
study.optimize(base_objective, n_trials=num_trials)
print(f"optimised study")

# Save the best parameters to pkl file - Optuna databases to pkl dumps are not readable across different versions
with open(f"{output_path}/study_xgboost.pkl",'wb') as f:
    pickle.dump(study, f)
print(f"saved study to to .pkl file")

###################################################################
# Get predictions from each model using optimised hyperparameters #
###################################################################
# Define the pipeline for the best parameters model with undersampling
param_dict = study.best_params
param_dict['use_label_encoder'] = False
param_dict['objective'] = 'binary:logistic'
param_dict['eval_metric'] = 'auc'
param_dict['nthread'] = -1

pipeline = Pipeline([('model', XGBClassifier(**param_dict, random_state=42))])

# Define the predictions list
pred = pd.DataFrame(columns=['gene', "xgboost"])
tr_pred = pd.DataFrame(columns=["gene", "xgboost"])
hout_ts_pred = pd.DataFrame(columns=["gene", "xgboost"])

# Loop over the cross-validation folds and predict with the best parameters
for train_idx, test_idx in cv.split(X, y):

    # Split the data into train and test sets
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_test, y_test = X.iloc[test_idx, :], y[test_idx]

    # Fit the model and make predictions
    pipeline.fit(X_train.drop(columns=["gene"]), y_train)

    # TRAIN set: save predictions to file
    tr_predictions = pipeline.predict_proba(X_train.drop(columns=["gene"]))[:, 1]
    tr_pr = pd.DataFrame({"gene": X_train["gene"], "xgboost": tr_predictions})

    # TEST set: save the predictions for the stacked ensemble
    predictions = pipeline.predict_proba(X_test.drop(columns=["gene"]))[:, 1]
    pr = pd.DataFrame({'gene': X_test["gene"], "xgboost": predictions})

    # HELD-OUT TEST set: save predictions to file
    hout_ts_predictions = pipeline.predict_proba(Xtest.drop(columns=["gene"]))[:, 1]
    hout_ts_pr = pd.DataFrame({"gene": Xtest["gene"], "xgboost": hout_ts_predictions})

    # Concatenate to save to file
    tr_pred = pd.concat([tr_pred, tr_pr], axis=0, ignore_index=True)
    pred = pd.concat([pred, pr], axis=0, ignore_index=True)
    hout_ts_pred = pd.concat([hout_ts_pred, hout_ts_pr], axis=0, ignore_index=True)

print(f"computed predictions on CV fold")

# TRAIN set: save to file
tr_pred.to_csv(fr"{output_path}/xgboost_trainSetCVfolds_predProba.csv", index=False)

# TEST set: annotate with genes and predictor
pred.to_csv(fr"{output_path}/xgboost_testSetCVfolds_predProba.csv", index=False)

# HELD-OUT TEST set: save to file
hout_ts_pred.to_csv(fr"{output_path}/xgboost_heldOutTestSetCVfolds_predProba.csv", index=False)

print(f"finished")

