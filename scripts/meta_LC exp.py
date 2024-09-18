
#@title **Location** of the dataset
path =  "../data/LendingClub/"
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
parser.add_argument('-p', '--percent_bad', type=float, default=0.2, help='Percentage bad added')
parser.add_argument('-s', '--size', type=int, default=1000, help='Percentage bad added')
parser.add_argument('-c', '--contamination', type=float, default=0.12, help='Percentage bad added')
parser.add_argument('-ut', '--use_test', action='store_true', help='Use test set to evaluate')
parser.add_argument('-tri', '--train_ri', action='store_true', default=True, help='Train others RI models')
parser.add_argument('-re', '--reuse_exec', action='store_true', default=False, help='Reuse trained models')
parser.add_argument('-tn', '--train_tn', action='store_true', default=True, help='Train Trusted Non-Outliers models')
parser.add_argument('-ev', '--eval_ri', action='store_true', default=False, help='Evaluate models')
args = parser.parse_args()

if args.percent_bad == 0:
    p_value = 'auto'

if args.percent_bad:
    p_value = args.percent_bad

if args.size:
    size = args.size

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


#Set seed
if args.seed:
    seed_number = args.seed
else:
    seed_number = secrets.randbelow(1_000_000)
    while seed_number <100:
        seed_number = secrets.randbelow(1_000_000)
print(seed_number)
main_seed = seed_number
year = args.year

metadata = {'seed': str(main_seed),
    'year': str(year),
    'p': str(p_value),
    'size': str(size),
    'contamination': str(contamination_threshold),
  }


today = time.strftime("%Y-%m-%d")

logpath = Path(os.path.join(ri_datasets_path,f'LC_py/logs_{today}/log-{seed_number}-{year}-{size}-{p_value}-{contamination_threshold}.log'))
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

detailed_logger.debug(metadata)

detailed_logger.debug(args)

#Read Dataset
    
#Accepts

load_path = f'{ri_datasets_path}Load/{main_seed}_{year}'

if Path(f'{load_path}').exists():
    df_train = pd.read_csv(f'{load_path}/A_train.csv', index_col=0)
    df_val = pd.read_csv(f'{load_path}/A_val.csv', index_col=0)
    df_test = pd.read_csv(f'{load_path}/A_test.csv', index_col=0)
    R_train = pd.read_csv(f'{load_path}/R_train.csv', index_col=0)
    R_val = pd.read_csv(f'{load_path}/R_val.csv', index_col=0)
    R_test = pd.read_csv(f'{load_path}/R_test.csv', index_col=0)
    X_train = df_train.loc[:, df_train.columns != "target"]
    y_train = df_train["target"]
    X_val = df_val.loc[:, df_val.columns != "target"]
    y_val = df_val["target"]
    X_test = df_test.loc[:, df_test.columns != "target"]
    y_test = df_test["target"]

    detailed_logger.debug(f'Data loaded from {load_path}')


if args.use_test:
    X_eval = X_test.copy()
    y_eval = y_test.copy()
    R_eval = R_test.copy()
else:
    X_eval = X_val.copy()
    y_eval = y_val.copy()
    R_eval = R_val.copy()

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


if args.train_tn:
    filepath_ex = Path(os.path.join(ri_datasets_path,f'Models/TN-{year}/{seed_number}/{size}-{p_value}-{contamination_threshold}.joblib'))
    datapath_ex = Path(os.path.join(ri_datasets_path,f'Data/TN-{year}/{seed_number}/{size}-{p_value}-{contamination_threshold}.parquet'))
    # filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}-{year}-{size}-{p_value}-{contamination_threshold}.joblib'))

    if filepath_ex.exists() and args.reuse_exec:
        models_ex = joblib.load(filepath_ex)
        detailed_logger.debug(f'TN loaded with shape: {len(models_ex.keys())}')

if args.eval_ri:
    if args.train_tn:
        # Initialize a dictionary to hold all the basic metrics for EX
        df_TN = ri.get_metrics_RI(models_ex, X_eval, y_eval, X_unl=R_eval)
        df_auc_ex = df_TN.loc['AUC', :]
        df_ks_ex = df_TN.loc['KS', :]
        df_kick_ex = ri.area_under_the_kick(models_ex, X_eval, y_eval, R_eval, low_AR, high_AR)
        #Should always be done on val and not test
        output_ex, best_values_ex = ri.evaluate_by_AUC_AUK(models_ex,  X_val, y_val, R_val, weights, criterias, low_AR, high_AR)
        
        if args.use_test:
            file_path = Path(os.path.join(ri_datasets_path,f'Data/TEST/TN-{year}-results.csv'))
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = Path(os.path.join(ri_datasets_path,f'Data/VAL/TN-{year}-results.csv'))
            file_path.parent.mkdir(parents=True, exist_ok=True)


        # Define the new data to be added (example data)
        new_data = {
            "seed": seed_number,
            "size": size,
            "p_value": p_value,
            "contamination": contamination_threshold,
            "AUC": df_TN.loc['AUC', f'TN_{output_ex}'],
            "KS": df_TN.loc['KS', f'TN_{output_ex}'],
            "Kickout": df_TN.loc['Kickout', f'TN_{output_ex}'],
            "AUK": df_kick_ex.loc[:, f'TN_{output_ex}'].mean()
        }

        # Convert new data to a DataFrame
        new_df = pd.DataFrame([new_data]).round(3)


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

    else:
        detailed_logger.debug(f'No TN fitted')

else:
    detailed_logger.debug(f'No evaluation requested')