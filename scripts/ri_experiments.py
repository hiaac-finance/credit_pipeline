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

from submodules.topsis_python import topsis as top


# %%
#@title Read dataset
df_o = pd.read_csv(path+'application_train.csv')    #HomeCredit training dataset


# %%
#@title Set seed
new_seed = True #@param {type:"boolean"}

if new_seed:
    seed_number = secrets.randbelow(10000000000) #to name the results files

    while seed_number <100:
        seed_number = secrets.randbelow(10000000000)
else:
    seed_number = 15444

main_seed = seed_number

print(seed_number)

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
N_splits=5
kf = KFold(n_splits=N_splits, random_state=main_seed)   #80-20 split for train-test
hist_dict = {}
data_dict = {}
run = False or True

for fold_number, (train_index, test_index) in enumerate(kf.split(df_o)):
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
    if run:
        models_dict.update(
            ri.trusted_non_outliers(X_train_acp, y_train_acp, X_train_rej,
                                    X_val_acp, y_val_acp, iterations=50,p = 0.07, output=-1,
                                    seed=seed_number, technique='extrapolation'))
        models_dict.update(
            ri.trusted_non_outliers(X_train_acp, y_train_acp, X_train_rej,
                                    X_val_acp, y_val_acp, iterations=50,p = 0.07, output=-1,
                                    seed=seed_number, technique='LS'))
            
    hist_dict[fold_number] = models_dict
        # metrics_dict[fold_number] = ri.get_metrics_RI(models_dict, X_test_acp, y_test_acp, X_val_acp, y_val_acp, X_test_rej)
    print(fold_number)
    # break


# %%
# main_seed = 9555
seed_number = main_seed

# %%

# %%
hist_kick = []

kf = KFold(n_splits=N_splits, random_state=main_seed)   #80-20 split for train-test



for i in range(40,45):
    AR = i/100.
    metrics_dict = {}
    for fold_number in range(N_splits):

        (X_train_acp, X_train_rej, y_train_acp, y_train_rej, 
         X_test_acp, X_test_rej, y_test_acp,X_val_acp, X_val_rej, y_val_acp) = data_dict[fold_number]

        seed_number = main_seed+fold_number
        filepath_ex = Path(os.path.join(ri_datasets_path,f'TN-{seed_number}.joblib'))
        filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}.joblib'))
        
        datasets_ex = joblib.load(filepath_ex)
        datasets_ls = joblib.load(filepath_ls)

        def evaluate_best_it(dsets, X_val_acp, y_val_acp, X_val_rej, AR):
            values = []

            for it in list(dsets.keys()):
                auc_value = roc_auc_score(y_val_acp, dsets[it].predict_proba(X_val_acp)[:,1])
                kick_value = ri.calculate_kickout_metric(dsets['BM'], dsets[it], X_val_acp, y_val_acp, X_val_rej, acp_rate=AR)[0]
                it_values = [auc_value, kick_value]
                values.append(it_values)

            values = np.array(values)
            weights = [1,10]
            criterias = np.array([True, True])
            t = top.Topsis(values, weights, criterias)
            t.calc()
            output = t.rank_to_best_similarity()[0] - 1
            print(f'best iteration: {output}')
            return output
        
        output_ex = evaluate_best_it(datasets_ex, X_val_acp, y_val_acp, X_val_rej, AR)
        output_ls = evaluate_best_it(datasets_ls, X_val_acp, y_val_acp, X_val_rej, AR)

        hist_dict[fold_number]['TN'] = datasets_ex[f'TN_{output_ex}']
        hist_dict[fold_number]['TN+'] = datasets_ls[f'TN_{output_ls}']


        metrics_dict[fold_number] = ri.get_metrics_RI(hist_dict[fold_number], X_test_acp, y_test_acp, X_unl = X_test_rej,
                                                    acp_rate=AR, threshold_type='none')
    # break
    mean_metrics = sum([metrics_dict[i] for i in range(N_splits)])/N_splits
    hist_kick.append(mean_metrics.loc[['Kickout']])

    # break

# %%
# %%
df_kick = pd.concat(hist_kick, axis=0)

filepath = Path(os.path.join(ri_datasets_path,f'DATA-{main_seed}.joblib'))
filepath.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(data_dict, filepath)

# %%



num_plots = len(df_kick.columns)
num_cols = math.ceil(math.sqrt(num_plots))
num_rows = math.ceil(num_plots / num_cols)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, column_name in enumerate(df_kick.columns):
    column_data = df_kick[column_name]
    row = i // num_cols
    col = i % num_cols

    axs[row, col].plot(df_kick['TN'].values, label='TN')
    axs[row, col].plot(column_data.values, label=column_name)
    axs[row, col].set_xlabel('Acceptance Rate')
    axs[row, col].set_ylabel('Kickout value')
    axs[row, col].legend(loc='lower center')

# Hide unused subplots
for i in range(num_plots, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    if num_rows == 1 and num_cols == 1:
        axs.axis('off')
    else:
        axs[row, col].axis('off')

plt.tight_layout()
plt.savefig(f'all_kick_by_AR_{main_seed}',  dpi=150)
plt.show()



