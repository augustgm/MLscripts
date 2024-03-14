###################################################################
#      Optimise XGBoost for Binary Classification using Optuna    #
# Perform undersampling of the majority class using desired ratio #
###################################################################
import pickle
import joblib
import optuna
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os
import gc

# define variables
num_trials = 100  # number of trials for Optuna optimisation
train_filename = ""  # training data csv file (assumed to have a column of sample IDs and the target variable)
hout_filename = ""  # held-out data csv file (assumed to have a column of sample IDs and the target variable)
input_path = ""  # path to where the train and test sets are stored
output_path = ""  # path where everything should be saved
cols_to_scale = []  # columns to scale
sample_col = ""  # column name containing sample ID
target_col = ""  # column name containing the target to be predicted (e.g. response status etc.)
rus_strat = 1.0  # float defining the undersampling of the majority class (1:1=1.0; 2:1=0.5 etc.)


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


def serialise_model(model_obj, model_type, output_path, del_xgb=False):
    """
    Save scikit learn or XGBoost sklearn API models
    :param model_obj: the actual model to be saved
    :param model_type: string stating what kind of model it is (e.g. "xgboost" or "elastic_net", "random_forest" etc.
    :param output_path: the path to the location where the model should be saved.
    Note that the appropriate file ending should not be specified.
    :param del_xgb: bool specifying whether to delete the XGB model booster and invoke garbage collection or not. Useful
    when using GPUs to free memory.
    :return: None
    """
    if model_type == "xgboost":
        # https://mljar.com/blog/xgboost-save-load-python/
        # https://stackabuse.com/bytes/how-to-save-and-load-xgboost-models/
        model_obj.save_model(f"{output_path}.json")
        if del_xgb:
            model_obj._Booster.__del__()
            gc.collect()
    else:  # https://scikit-learn.org/stable/model_persistence.html
        joblib.dump(model_obj, f"{output_path}.joblib")
    return None


# Define the cross-validation scheme
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)


#########################################################
# Load data, annotate targets, and separate out X and y #
#########################################################
X = pd.read_csv(f"{input_path}/{train_filename}")
X_hout = pd.read_csv(f"{input_path}/{hout_filename}")

# Scale shapley values data
scaler = StandardScaler(with_std=True, with_mean=True)

# Scale the selected columns
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
X_hout[cols_to_scale] = scaler.transform(X_hout[cols_to_scale])

# Split x and y
y = X[target_col]
X.drop(columns=[target_col], inplace=True)

y_hout = X_hout[target_col]
X_hout.drop(columns=[target_col], inplace=True)

del scaler, cols_to_scale


############################
# Optimise hyperparameters #
############################
# Define the objective
def base_objective(trial):

    # Define the model objective
    model = xgboost_objective(trial)

    # Define the scores list
    scores = []

    # Loop over the cross-validation folds
    for train_idx, test_idx in cv.split(X, y):
        # Split the data into train and test sets
        X_train, y_train = X.iloc[train_idx, :], y[train_idx]
        X_test, y_test = X.iloc[test_idx, :], y[test_idx]

        # Undersample
        rus = RandomUnderSampler(sampling_strategy=rus_strat, random_state=42, replacement=False)
        X_train, y_train = rus.fit_resample(X_train, y_train)

        # Fit the model and make predictions
        model.fit(X_train.drop(columns=[sample_col]), y_train)
        y_pred = model.predict_proba(X_test.drop(columns=[sample_col]))

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
                              storage=f"sqlite:///{output_path}/study_xgboost_rus{rus_strat}.db")
else:
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(study_name="study_xgboost", sampler=sampler, direction="maximize",
                                storage=f"sqlite:///{output_path}/study_xgboost_rus{rus_strat}.db")

# Optimize the hyperparameters
study.optimize(base_objective, n_trials=num_trials)
print(f"optimised study")

# Save the best parameters to pkl file - Optuna databases to pkl dumps are not readable across different versions
with open(f"{output_path}/study_xgboost.pkl", 'wb') as f:
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

# Define the predictions list
tr_pred = pd.DataFrame(columns=[sample_col, "pred_proba"])
pred = pd.DataFrame(columns=[sample_col, "pred_proba"])
hout_ts_pred = pd.DataFrame(columns=[sample_col, "pred_proba"])

# Loop over the cross-validation folds and predict with the best parameters
for fold_id, train_idx, test_idx in enumerate(cv.split(X, y)):
    model = XGBClassifier(**param_dict, random_state=42)

    # Split the data into train and test sets
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_test, y_test = X.iloc[test_idx, :], y[test_idx]

    # Undersample
    rus = RandomUnderSampler(sampling_strategy=rus_strat, random_state=42, replacement=False)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Fit the model and make predictions
    model.fit(X_train.drop(columns=[sample_col]), y_train)

    # TRAIN set: save predictions to file
    tr_predictions = model.predict_proba(X_train.drop(columns=[sample_col]))[:, 1]
    tr_pr = pd.DataFrame({sample_col: X_train[sample_col], "CV_fold": fold_id, "pred_proba": tr_predictions})

    # TEST set: save the predictions for the stacked ensemble
    predictions = model.predict_proba(X_test.drop(columns=[sample_col]))[:, 1]
    pr = pd.DataFrame({sample_col: X_test[sample_col], "CV_fold": fold_id, "pred_proba": predictions})

    # HELD-OUT TEST set: save predictions to file
    hout_ts_predictions = model.predict_proba(X_hout.drop(columns=[sample_col]))[:, 1]
    hout_ts_pr = pd.DataFrame({sample_col: X_hout[sample_col], "CV_fold": fold_id, "pred_proba": hout_ts_predictions})

    # Concatenate to save to file
    tr_pred = pd.concat([tr_pred, tr_pr], axis=0, ignore_index=True)
    pred = pd.concat([pred, pr], axis=0, ignore_index=True)
    hout_ts_pred = pd.concat([hout_ts_pred, hout_ts_pr], axis=0, ignore_index=True)

    # Save trained model to file
    serialise_model(model_obj=model, model_type="xgboost", del_xgb=False,
                    output_path=f"{output_path}/xgboost_optimTrainedModel_rus{rus_strat}_CVfold{fold_id}.json")

print(f"computed predictions on CV fold")

# TRAIN set: save to file
tr_pred.to_csv(fr"{output_path}/xgboost_rus{rus_strat}_trainSetCVfolds_predProba.csv", index=False)

# TEST set: annotate with genes and predictor
pred.to_csv(fr"{output_path}/xgboost_rus{rus_strat}_testSetCVfolds_predProba.csv", index=False)

# HELD-OUT TEST set: save to file
hout_ts_pred.to_csv(fr"{output_path}/xgboost_rus{rus_strat}_heldOutTestSetCVfolds_predProba.csv", index=False)

print(f"finished")

