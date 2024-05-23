#@title **Location** of the dataset
path =  "../data/HomeCredit/"
process_path = "../data/ProcessedData/"
save_path = "../tests/"
ri_datasets_path = "../data/riData/"

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
import joblib
import os
import math
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp
from lightgbm import LGBMClassifier
from pathlib import Path
from sklearn.metrics import (roc_auc_score)
from sklearn.model_selection import KFold


# %%
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                            f1_score, precision_score, recall_score,
                            roc_auc_score, roc_curve)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelSpreading

# %%
import credit_pipeline.data_exploration as dex
import credit_pipeline.training as tr
import credit_pipeline.reject_inference as ri

import sys
sys.path.append("../")
from submodules.topsis_python import topsis as top

seed_number = 13582
low_AR = 0
high_AR = 100
weights = [1,10]
criterias = np.array([True, True])
n_iterations = 50
p_value = 0.07
N_splits = 5
main_seed = seed_number

logpath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/log.txt'))
logpath.parent.mkdir(parents=True, exist_ok=True)


# Configure logging to file
logging.basicConfig(filename=logpath, 
                    filemode='w',  # Overwrite the file each time the application runs
                    level=logging.DEBUG,  # Capture all levels of logging
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format for the timestamp

# Create and configure a custom logger for detailed (DEBUG level) logging
detailed_logger = logging.getLogger('detailed')
detailed_logger.setLevel(logging.DEBUG)  # Set this logger to capture everything

# Create a file handler for the custom logger (optional if you want all logs in the same file)
file_handler = logging.FileHandler(logpath)
file_handler.setLevel(logging.DEBUG)

# You might want to use the same formatter for consistency
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add the file handler to the detailed logger
detailed_logger.addHandler(file_handler)

detailed_logger.debug(logpath)

# %%
#@title Read dataset
df_o = pd.read_csv(path+'application_train.csv')    #HomeCredit training dataset

# %%
detailed_logger.debug(seed_number)

# %% [markdown]
# #Params

# %%
params_dict = ri.params_dict

# %%
params_dict['LightGBM_2'] = {'boosting_type': 'gbdt', 'class_weight': None,
              'colsample_bytree': 0.22534977954592625, 'importance_type': 'split',
              'learning_rate': 0.052227873762946964, 'max_depth': 5,
              'min_child_samples': 26, 'min_child_weight': 0.001,
              'min_split_gain': 0.0, 'n_estimators': 159, 'n_jobs': -1,
              'num_leaves': 12, 'objective': None, 'random_state': seed_number,
              'reg_alpha': 0.7438345471808012, 'reg_lambda': 0.46164693905368515,
                'verbose': -1, 'subsample': 0.8896599304061413,
              'subsample_for_bin': 200000, 'subsample_freq': 0,
              'is_unbalance': True}

# %% [markdown]
# #<font color='orange'>Helper Functions</font>
# 

# %% [markdown]
# #<font color='red'>Definition of Train and Test Val, and Unl</font>

# %%

kf = KFold(n_splits=N_splits, random_state=main_seed, shuffle = True)   #80-20 split for train-test
hist_dict = {}
data_dict = {}
run = False or True

for fold_number, (train_index, test_index) in enumerate(kf.split(df_o)):
    try:
        #diferent seed for each iteration
        seed_number = main_seed+fold_number

        df_train = df_o.iloc[train_index]
        df_test = df_o.iloc[test_index]
        
        val_split = int(df_train.shape[0] * 0.2)  #80-20 split for train-validation
        df_val = df_train.iloc[:val_split]
        df_train = df_train.iloc[val_split:]

        df_train, policy_model = ri.fit_policy(df_train)

        X_train, X_test, X_val = df_train, df_test, df_val
        y_train, y_test, y_val = df_train["TARGET"], df_test["TARGET"], df_val["TARGET"]

        X_train_acp, X_train_rej, y_train_acp, y_train_rej = ri.accept_reject_split(X_train, y_train, policy_clf=policy_model)
        X_test_acp, X_test_rej, y_test_acp, y_test_rej = ri.accept_reject_split(X_test, y_test, policy_clf=policy_model)
        X_val_acp, X_val_rej, y_val_acp, y_val_rej = ri.accept_reject_split(X_val, y_val, policy_clf=policy_model)
        
        data_dict[fold_number] = [X_train_acp, X_train_rej, y_train_acp, y_train_rej,
                                X_test_acp, X_test_rej, y_test_acp,
                                X_val_acp, X_val_rej, y_val_acp]

        models_dict = {}
        benchmark = tr.create_pipeline(X_train_acp, y_train_acp,
                                    LGBMClassifier(**params_dict['LightGBM_2']))
        benchmark.fit(X_train_acp, y_train_acp)

        
        models_dict['BM'] = benchmark

        models_dict.update(
            ri.augmentation_with_soft_cutoff(X_train_acp, y_train_acp, X_train_rej, seed = seed_number))
        models_dict.update(
            ri.augmentation(X_train_acp, y_train_acp, X_train_rej, mode='up', seed = seed_number))
        models_dict.update(
            ri.augmentation(X_train_acp, y_train_acp, X_train_rej, mode='down', seed = seed_number))
        models_dict.update(
            ri.fuzzy_augmentation(X_train_acp, y_train_acp, X_train_rej, seed = seed_number))
        models_dict.update(
            ri.extrapolation(X_train_acp, y_train_acp, X_train_rej, seed = seed_number))
        models_dict.update(
            ri.parcelling(X_train_acp, y_train_acp, X_train_rej, seed = seed_number))
        models_dict.update(
            ri.label_spreading(X_train_acp, y_train_acp, X_train_rej, seed = seed_number))
        if run and fold_number != 0:
            detailed_logger.info(f"running technique extrapolation with p={p_value} for {n_iterations} iterations")
            models_dict.update(
                ri.trusted_non_outliers(X_train_acp, y_train_acp, X_train_rej,
                                        X_val_acp, y_val_acp, iterations=n_iterations,p = p_value, output=-1,
                                        seed=seed_number, technique='extrapolation'), )
            detailed_logger.info(f"running technique label spreading with p={p_value} for {n_iterations} iterations")
            models_dict.update(
                ri.trusted_non_outliers(X_train_acp, y_train_acp, X_train_rej,
                                        X_val_acp, y_val_acp, iterations=n_iterations,p = p_value, output=-1,
                                        seed=seed_number, technique='LS'))
                
        hist_dict[fold_number] = models_dict
            # metrics_dict[fold_number] = ri.get_metrics_RI(models_dict, X_test_acp, y_test_acp, X_val_acp, y_val_acp, X_test_rej)
        detailed_logger.debug(f"fold_number = {fold_number}")

    except Exception as e:
        detailed_logger.exception("An error occurred.")
        break


# #save data
try:
    filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/DATA.joblib'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_dict, filepath)
except Exception as e:
        detailed_logger.exception("An error occurred.")
        


#save models
try:
    filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/models.joblib'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hist_dict, filepath)
except Exception as e:
        detailed_logger.exception("An error occurred.")


# %%
# main_seed = 9555
seed_number = main_seed

# %%

# %%
hist_kick = {}

kf = KFold(n_splits=N_splits, random_state=main_seed, shuffle = True)   #80-20 split for train-test


metrics_dict = {}
try:
    for fold_number in range(N_splits):

        (X_train_acp, X_train_rej, y_train_acp, y_train_rej, 
        X_test_acp, X_test_rej, y_test_acp,X_val_acp, X_val_rej, y_val_acp) = data_dict[fold_number]

        seed_number = main_seed+fold_number
        filepath_ex = Path(os.path.join(ri_datasets_path,f'TN-{seed_number}-{p_value}.joblib'))
        filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}-{p_value}.joblib'))
        
        datasets_ex = joblib.load(filepath_ex)
        datasets_ls = joblib.load(filepath_ls)

        output_ex, best_values_ex = ri.evaluate_by_AUC_AUK(datasets_ex, X_val_acp, y_val_acp, X_val_rej, weights, criterias, low_AR, high_AR)
        output_ls, best_values_ls = ri.evaluate_by_AUC_AUK(datasets_ls, X_val_acp, y_val_acp, X_val_rej, weights, criterias, low_AR, high_AR)
        print(f'Best values for extrapolation: {best_values_ex} at iteration {output_ex}')
        print(f'Best values for label spreading: {best_values_ls} at iteration {output_ls}')
        print(f'--------{fold_number}--------')
        
        if output_ex != 0:
            hist_dict[fold_number]['TN'] = datasets_ex[f'TN_{output_ex}']
        else:
            hist_dict[fold_number]['TN'] = datasets_ex['BM']
        
        if output_ls != 0:
            hist_dict[fold_number]['TN+'] = datasets_ls[f'TN_{output_ls}']
        else:
            hist_dict[fold_number]['TN+'] = datasets_ls['BM']
        
        # Initialize a dictionary to hold all the basic metrics for EX
        df_TN = ri.get_metrics_RI(datasets_ex, X_test_acp, y_test_acp, X_unl=X_test_rej)
        df_auc_ex = df_TN.loc['AUC', :]
        df_kick_ex = ri.area_under_the_kick(datasets_ex, X_test_acp, y_test_acp, X_test_rej, low_AR, high_AR)

        # Initialize a dictionary to hold all the basic metrics for LS
        df_TNplus = ri.get_metrics_RI(datasets_ls,  X_test_acp, y_test_acp, X_unl=X_test_rej)
        df_auc_ls = df_TNplus.loc['AUC', :]
        df_kick_ls = ri.area_under_the_kick(datasets_ls, X_test_acp, y_test_acp, X_test_rej, low_AR, high_AR)

        metrics_dict[fold_number] = ri.get_metrics_RI(hist_dict[fold_number], X_test_acp, y_test_acp, X_unl = X_test_rej,
                                                acp_rate=0.5, threshold_type='none')
except Exception as e:
    detailed_logger.exception("An error occurred.")
# break

try:
    mean_metrics = sum([metrics_dict[i] for i in range(N_splits)])/N_splits
    hist_kick[AR] = mean_metrics.loc[['Kickout', "Overall AUC"]]
except Exception as e:
    detailed_logger.exception("An error occurred.")

# break

# %%
print(hist_kick)
# %%
try:
    df_kick = pd.concat(hist_kick, axis=0)
except Exception as e:
        detailed_logger.exception("An error occurred.")

#save metrics
try:
    filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/metrics.joblib'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(df_kick, filepath)
except Exception as e:
        detailed_logger.exception("An error occurred.")

param_exec = {
    'low_AR' : low_AR,
    'high_AR' : high_AR,
    'weights' : weights,
    'iterations' : n_iterations,
    'p' : p_value,
    'N_splits' : N_splits,
    'main_seed' : main_seed,
}

#save params
try:
    filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/params.joblib'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(param_exec, filepath)
except Exception as e:
        detailed_logger.exception("An error occurred.")



