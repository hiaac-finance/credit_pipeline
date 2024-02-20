#@title **Location** of the dataset
path =  "../data/HomeCredit/"
process_path = "../data/ProcessedData/"
save_path = "../tests/"
ri_datasets_path = "../data/riData/"

# %%
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
import joblib
import os
import math

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

main_seed = 123456

logpath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/logdata.txt'))
logpath.parent.mkdir(parents=True, exist_ok=True)


# Configure logging to file
logging.basicConfig(filename=logpath, 
                    filemode='w',  # Overwrite the file each time the application runs
                    level=logging.DEBUG,  # Capture all levels of logging
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format for the timestamp


filepath_param = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/params.joblib'))
param_exec = joblib.load(filepath_param)

low_AR = param_exec['low_AR']
high_AR = 99#param_exec['high_AR']
weights = param_exec['weights']
iterations = param_exec['iterations']
p = param_exec['p']
N_splits = param_exec['N_splits']
seed_number  = param_exec['main_seed']

logging.info(param_exec)

# sys.exit()

# seed_number = 13582
# low_AR = 3
# high_AR = 99
# weights = [1,10]
# iterations = 50
# p = 0.1
# N_splits = 5
main_seed = seed_number



filepath_data = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/DATA.joblib'))
filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/models.joblib'))
# filepath_metric = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/metrics.joblib'))
    
data_dict = joblib.load(filepath_data)
hist_dict = joblib.load(filepath)
# hist_met = joblib.load(filepath_metric)


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

hist_kick = []

for i in range(low_AR,high_AR):
    AR = i/100.
    metrics_dict = {}

    logging.info(f'%%%%%%%%%%%%%%% [AR = {AR}] %%%%%%%%%%%%%%%')
    try:
        for fold_number in range(N_splits):

            (X_train_acp, X_train_rej, y_train_acp, y_train_rej, 
            X_test_acp, X_test_rej, y_test_acp,X_val_acp, X_val_rej, y_val_acp) = data_dict[fold_number]

            seed_number = main_seed+fold_number
            filepath_ex = Path(os.path.join(ri_datasets_path,f'TN-{seed_number}.joblib'))
            filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}.joblib'))
            
            datasets_ex = joblib.load(filepath_ex)
            datasets_ls = joblib.load(filepath_ls)

            logging.info(f'-------- [fold number = {fold_number}] --------')
            def evaluate_best_it(dsets, X_val_acp, y_val_acp, X_val_rej, AR):
                values = []

                for it in list(dsets.keys()):
                    auc_value = roc_auc_score(y_val_acp, dsets[it].predict_proba(X_val_acp)[:,1])
                    kick_value = ri.calculate_kickout_metric(dsets['BM'], dsets[it], X_val_acp, y_val_acp, X_val_rej, acp_rate=AR)[0]
                    it_values = [auc_value, kick_value]
                    # print(f'kick_value for {it} = {kick_value}')
                    values.append(it_values)

                values = np.array(values)
                weights = [1,10]
                criterias = np.array([True, True])
                t = top.Topsis(values, weights, criterias)
                t.calc()
                output = t.rank_to_best_similarity()[0] - 1
                return output
            
            output_ex = evaluate_best_it(datasets_ex, X_val_acp, y_val_acp, X_val_rej, AR)
            logging.info(f'best iteration extrapolation: {output_ex}')
            output_ls = evaluate_best_it(datasets_ls, X_val_acp, y_val_acp, X_val_rej, AR)
            logging.info(f'best iteration label spreading: {output_ls}')

            if output_ex != 0:
                hist_dict[fold_number]['TN'] = datasets_ex[f'TN_{output_ex}']
            else:
                hist_dict[fold_number]['TN'] = datasets_ex['BM']
            
            if output_ls != 0:
                hist_dict[fold_number]['TN+'] = datasets_ls[f'TN_{output_ls}']
            else:
                hist_dict[fold_number]['TN+'] = datasets_ls['BM']


            metrics_dict[fold_number] = ri.get_metrics_RI(hist_dict[fold_number], X_test_acp, y_test_acp, X_unl = X_test_rej,
                                                        acp_rate=AR, threshold_type='none')
        
    except Exception as e:
        logging.exception("An error occurred.")
    
    # break
    mean_metrics = sum([metrics_dict[i] for i in range(N_splits)])/N_splits
    hist_kick.append(mean_metrics.loc[['Kickout']])
    df_kick = pd.concat(hist_kick, axis=0)

    if i == low_AR:
        hist_auc = [mean_metrics.loc[['Overall AUC']]]
        df_auc = pd.concat(hist_auc, axis=0)

        filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/auc.joblib'))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(df_auc, filepath)
    # break

    filepath = Path(os.path.join(ri_datasets_path,f'Exp_{main_seed}/kickout-{low_AR}_{i}.joblib'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(df_kick, filepath)

# param_exec = {
#     'low_AR' : low_AR,
#     'high_AR' : high_AR,
#     'weights' : weights,
#     'iterations' : iterations,
#     'p' : p,
#     'N_splits' : N_splits,
#     'main_seed' : main_seed,
# }

# filepath = Path(os.path.join(ri_datasets_path,f'params-{main_seed}.joblib'))
# filepath.parent.mkdir(parents=True, exist_ok=True)
# joblib.dump(param_exec, filepath)
