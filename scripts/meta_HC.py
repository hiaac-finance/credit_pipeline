
#@title **Location** of the dataset
path =  "../data/HomeCredit/"
process_path = "../data/ProcessedData/"
save_path = "../tests/"
ri_datasets_path = "../data/riData/"
backup_image_folder = "../../backup/Images/"

 
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
import joblib
from pathlib import Path
import os
import time
import re
import logging



from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score)
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.feature_selection import r_regression


 
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                            f1_score, precision_score, recall_score,
                            roc_auc_score, roc_curve)

 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelSpreading

 
import credit_pipeline.data_exploration as dex
import credit_pipeline.training as tr
import credit_pipeline.reject_inference as ri
import credit_pipeline.evaluate as ev

from submodules.topsis_python import topsis as top


#ARGs and constants
parser = argparse.ArgumentParser(description='Simple Script')
parser.add_argument('-ar', '--ar_range', type=int, nargs=2, default=[0, 100], help='Low and High AR value', metavar=('LOW_AR', 'HIGH_AR'))
parser.add_argument('-wt', '--weights', type=int, nargs=2, default=[1, 1], help='weights of metrics for Topsis', metavar=('Weight_AUC, Weight_Kickout'))
parser.add_argument('--seed', type=int, help='Seed number')
parser.add_argument('-y', '--year', type=int, default=2009, help='Year')
parser.add_argument('-p', '--percent_bad', type=float, default=0.07, help='Percentage bad added')
parser.add_argument('-ths', '--threshold', type=float, default=0.5, help='thrsehold for accept-reject policy')
parser.add_argument('-s', '--size', type=int, default=1000, help='Percentage bad added')
parser.add_argument('-c', '--contamination', type=float, default=0.12, help='Percentage bad added')
parser.add_argument('-ut', '--use_test', action='store_true', help='Use test set to evaluate')
parser.add_argument('-tri', '--train_ri', action='store_true', default=True, help='Train others RI models')
parser.add_argument('-re', '--reuse_exec', action='store_true', default=False, help='Reuse trained models')
parser.add_argument('-tn', '--train_tn', action='store_true', default=True, help='Train Trusted Non-Outliers models')
parser.add_argument('-ev', '--eval_ri', action='store_true', default=True, help='Evaluate models')
args = parser.parse_args()

# sys.exit()
train_tn = args.train_tn
reuse_exec = args.reuse_exec
eval_ri = args.eval_ri
use_test = args.use_test


if args.percent_bad == 0:
    p_value = 'auto'

if args.percent_bad:
    p_value = args.percent_bad

if args.size:
    size = args.size

if args.year:
    year = args.year

if args.contamination:
    contamination_threshold = args.contamination

#Accept rate
if args.ar_range:
    low_AR, high_AR = args.ar_range

#Weights
if args.weights:
    Weight_AUC, Weight_Kickout = args.weights

#TopSis
weights = [Weight_AUC, Weight_Kickout]
criterias = np.array([True, True])


n_iter = 50
# size = 250
# # p_value = 0.06
# contamination_threshold = 0.12
year = 2000
# tr_policy = 0.05 + (year - 2000)/10
tr_policy = args.threshold

#Set seed
if args.seed:
    seed_number = args.seed
else:
    seed_number = secrets.randbelow(1_000_000)
    while seed_number <100000:
        seed_number = secrets.randbelow(1_000_000)
print(seed_number)
main_seed = seed_number

metadata = {'seed': str(main_seed),
    'year': str(year),
    'p': str(p_value),
    'size': str(size),
    'contamination': str(contamination_threshold),
    'tr_policy': str(tr_policy),
  }


today = time.strftime("%Y-%m-%d")

logpath = Path(os.path.join(ri_datasets_path,f'HC_py/logs_{today}/log-{tr_policy}-{seed_number}-{size}-{p_value}-{contamination_threshold}.log'))
logpath.parent.mkdir(parents=True, exist_ok=True)

print(logpath)

logging.getLogger().handlers = []
logging.basicConfig(filename=logpath, 
                    filemode='w',  
                    level=logging.WARNING,  # This sets the threshold for the root logger
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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

# detailed_logger.debug(metadata)

detailed_logger.debug(args)

#Read Dataset
    
#Accepts

df_o = pd.read_csv(path+'application_train.csv')    #HomeCredit training dataset

# #@title Set seed
# new_seed = False #@param {type:"boolean"}

# if new_seed:
#     seed_number = secrets.randbelow(1_000_000) #to name the results files

#     while seed_number <100000:
#         seed_number = secrets.randbelow(1_000_000)
# else:
#     seed_number = 123123

main_seed = seed_number

print(seed_number)

#@title Create develoment train and test
df_train, df_test = tr.create_train_test(df_o, seed=seed_number)

params_dict = ri.params_dict
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


# print(year, tr_policy)
# low_AR, high_AR = 0, 100
# weights = [1,1]
# criterias = [True, True]

detailed_logger.debug(f'tr_policy: {tr_policy}')
# sys.exit()

df_train, df_val = train_test_split(
            df_train, test_size=0.2, random_state=seed_number)

df_train, policy_model = ri.fit_policy(df_train)
detailed_logger.debug(f'Policy model fitted')

X_train, y_train = df_train, df_train["TARGET"]
X_val, y_val = df_val, df_val["TARGET"]
X_test, y_test = df_test, df_test["TARGET"]


X_train_acp, X_train_rej, y_train_acp, y_train_rej = ri.accept_reject_split(X_train, y_train, policy_clf=policy_model, threshold = tr_policy)
X_test_acp, X_test_rej, y_test_acp, y_test_rej = ri.accept_reject_split(X_test, y_test, policy_clf=policy_model, threshold = tr_policy)
X_val_acp, X_val_rej, y_val_acp, y_val_rej = ri.accept_reject_split(X_val, y_val, policy_clf=policy_model, threshold = tr_policy)

detailed_logger.debug(f'Accept-Reject split done')
detailed_logger.debug(f'X_train_acp: {X_train_acp.shape}, X_train_rej: {X_train_rej.shape}')

if use_test:
    X_eval = X_test_acp.copy()
    y_eval = y_test_acp.copy()
    R_eval = X_test_rej.copy()
    detailed_logger.debug(f'Using test set for evaluation')
# else:
#     X_eval = pd.concat([X_val_acp.copy(), X_val_rej.copy()], axis=0)
#     y_eval = pd.concat([y_val_acp.copy(), y_val_rej.copy()], axis=0)
#     R_eval = X_val_rej.copy()
# else:
#     X_eval = X_val_rej.copy()
#     y_eval = y_val_rej.copy()
#     X_eval_1, X_eval_2, y_eval_1, y_eval_2 = train_test_split(
#             X_eval,y_eval, test_size=0.5, random_state=seed_number)
#     X_eval = X_eval_1.copy()
#     y_eval = y_eval_1.copy()
#     R_eval = X_eval_2.copy()   
else:
    X_eval = X_val_acp.copy()
    y_eval = y_val_acp.copy()
    R_eval = X_val_rej.copy()
    detailed_logger.debug(f'Using validation set for evaluation')



# dex.get_shapes(X_train, X_train_acp, X_train_rej, X_test, X_test_acp, X_test_rej, X_val, X_val_acp, X_val_rej)
# sys.exit()

models_dict = {}

filepath_models = Path(os.path.join(ri_datasets_path,f'HC/Models/RI/models-{seed_number}-{tr_policy}.joblib'))
if filepath_models.exists():
    models_dict = joblib.load(filepath_models)
    detailed_logger.debug(f'Models loaded with shape: {len(models_dict.keys())}')
else:
    detailed_logger.debug(f'Models fitting started with seed {seed_number}')
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
    
    filepath_models.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models_dict, filepath_models)
    detailed_logger.debug(f'Models fitted')


filepath_ex = Path(os.path.join(ri_datasets_path,f'HC/Models/TN-{seed_number}/{tr_policy}-{size}-{p_value}-{contamination_threshold}.joblib'))
datapath_ex = Path(os.path.join(ri_datasets_path,f'HC/Data/TN-/{seed_number}/{tr_policy}-{size}-{p_value}-{contamination_threshold}.parquet'))

detailed_logger.debug(f'filepath_ex: {filepath_ex}')
detailed_logger.debug(f'datapath_ex: {datapath_ex}')

if filepath_ex.exists() and reuse_exec:
    models_ex = joblib.load(filepath_ex)
    detailed_logger.debug(f'TN loaded with shape: {len(models_ex.keys())}')
else:
    detailed_logger.debug(f'TN fitting started with seed {seed_number}')
    TNmodels, TNdata = ri.trusted_non_outliers(X_train_acp, y_train_acp, X_train_rej,
                            X_val_acp, y_val_acp, size=size, iterations=n_iter,p = 0.07, output=-1,return_all=True,
                            save_log=False, seed=seed_number, technique='extrapolation')
    detailed_logger.debug(f'TN fitted')
    filepath_ex.parent.mkdir(parents=True, exist_ok=True)
    datapath_ex.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(TNmodels, filepath_ex)

    #------------------------------------
    detailed_logger.debug(f'starting compactation of TN data')
    train_list = [pd.DataFrame({**item, 'group': i}) for i, item in enumerate(TNdata['X'])]
    last_df = TNdata['unl'][-1].copy()
    last_df.loc[:, 'group'] = -1

    df = pd.concat(train_list, axis=0, ignore_index=False)
    df = pd.concat([df, last_df])

    min_values = df.group.groupby(level=0).min()
    last_value = TNdata['y'][-1]

    result_df = pd.DataFrame({
        'first_it': min_values,
        'label': last_value
    }, index=min_values.index)
    result_df = result_df.sort_values(by=['first_it'], ascending=True)
    result_df.fillna(-1, inplace=True)
    result_df = result_df.astype(int)

    result_df.to_parquet(datapath_ex)
    print(f'data saved to {datapath_ex}')
    #------------------------------------
    models_ex = TNmodels


if train_tn:
    filepath_ex = Path(os.path.join(ri_datasets_path,f'Models/TN-{"HC"}/{tr_policy}-{seed_number}/{size}-{p_value}-{contamination_threshold}-{tr_policy}.joblib'))
    datapath_ex = Path(os.path.join(ri_datasets_path,f'Data/TN-{"HC"}/{tr_policy}-{seed_number}/{size}-{p_value}-{contamination_threshold}-{tr_policy}.parquet'))
    # filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}-{year}-{size}-{p_value}-{contamination_threshold}.joblib'))

    if filepath_ex.exists() and reuse_exec:
        models_ex = joblib.load(filepath_ex)
        detailed_logger.debug(f'TN loaded with shape: {len(models_ex.keys())}')

if eval_ri:
    detailed_logger.debug(f'Evaluation started')
    df_RI = ri.get_metrics_RI(models_dict, X_eval, y_eval, X_unl=R_eval, acp_rate=0.5)
    df_AUK = ri.area_under_the_kick(models_dict, X_eval, y_eval, R_eval, low_AR, high_AR).mean()
    df_AUC = df_RI.loc['AUC']
    df_KS = df_RI.loc['KS']
    RI_unb = ri.get_metrics_RI(models_dict, pd.concat([X_val_acp, X_val_rej],axis=0), pd.concat([y_val_acp, y_val_rej],axis=0))
    
    detailed_logger.debug(f'RI evaluated')
    if train_tn:
        detailed_logger.debug(f'Evaluation of TN started')
        # Initialize a dictionary to hold all the basic metrics for EX
        df_TN = ri.get_metrics_RI(models_ex, X_eval, y_eval, X_unl=R_eval, acp_rate=0.5)
        df_auc_ex = df_TN.loc['AUC', :]
        df_ks_ex = df_TN.loc['KS', :]
        df_kick_ex = ri.area_under_the_kick(models_ex, X_eval, y_eval, R_eval, low_AR, high_AR)
        output_ex, best_values_ex = ri.evaluate_by_AUC_AUK(models_ex,  X_eval, y_eval, R_eval, weights, criterias, low_AR, high_AR)
        TN_unb = ri.get_metrics_RI(models_ex, pd.concat([X_val_acp, X_val_rej],axis=0), pd.concat([y_val_acp, y_val_rej],axis=0))

        if use_test:
            detailed_logger.debug(f'Evaluation of TN on test set started')
            file_path = Path(os.path.join(ri_datasets_path,f'Data/TEST/TN-{year}-{tr_policy}-results.csv'))
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            detailed_logger.debug(f'Evaluation of TN on validation set started')
            file_path = Path(os.path.join(ri_datasets_path,f'Data/VAL/TN-{year}-{tr_policy}-results.csv'))
            file_path.parent.mkdir(parents=True, exist_ok=True)


        # Define the new data to be added
        new_data = {
            "seed": seed_number,
            "model": "TN-EX",
            "size": size,
            "p_value": p_value,
            "contamination": contamination_threshold,
            "AUC": df_TN.loc['AUC', f'TN_{output_ex}'],
            "KS": df_TN.loc['KS', f'TN_{output_ex}'],
            "Kickout": df_TN.loc['Kickout', f'TN_{output_ex}'],
            "AUK": df_kick_ex.loc[:, f'TN_{output_ex}'].mean(),
            "unb_AUC": TN_unb.loc['AUC', f'TN_{output_ex}'],
            "Best": output_ex,
            "Size_train_acp": X_train_acp.shape[0],
            "Size_train_rej": X_train_rej.shape[0],
        }

        # Convert new data to a DataFrame
        new_df = pd.DataFrame([new_data]).round(3)
        
        for col in df_RI.columns:
            new_data = {
                    "seed": seed_number,
                    "model": col,
                    "size": -1,
                    "p_value": -1,
                    "contamination": -1,
                    "AUC": df_RI.loc['AUC', col],
                    "KS": df_RI.loc['KS', col],
                    "Kickout": df_RI.loc['Kickout', col],
                    "AUK": df_AUK.loc[col],
                    "unb_AUC": RI_unb.loc['AUC', col],
                    "Best": -1,
                    "Size_train_acp": X_train_acp.shape[0],
                    "Size_train_rej": X_train_rej.shape[0],
                }
            col_df = pd.DataFrame([new_data]).round(3)
            new_df = pd.concat([new_df, col_df], ignore_index=True)
        detailed_logger.debug(f'Evaluation of TN done')
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the existing CSV file
            existing_df = pd.read_csv(file_path)
            # Append the new data to the existing data
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If the file does not exist, the new data itself is the updated data
            updated_df = new_df

        # Save the updated DataFrame back to the CSV file
        updated_df.to_csv(file_path, index=False)
        detailed_logger.debug(f'Evaluation of TN saved')