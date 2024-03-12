
# %%
 
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
import os
import re
import logging



from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from pathlib import Path
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

# %%
# ARGs and constants
parser = argparse.ArgumentParser(description='Simple Script')
parser.add_argument('-ar', '--ar_range', type=int, nargs=2, default=[0, 100], help='Low and High AR value', metavar=('LOW_AR', 'HIGH_AR'))
parser.add_argument('-wt', '--weights', type=int, nargs=2, default=[1, 1], help='weights of metrics for Topsis', metavar=('Weight_AUC, Weight_Kickout'))
parser.add_argument('--seed', type=int, default=0, help='Seed number')
parser.add_argument('-y', '--year', type=int, default=2009, help='Year')
parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold value')
parser.add_argument('-p', '--percent_bad', type=float, default=0.2, help='Percentage bad added')
parser.add_argument('-ut', '--use_test', action='store_true', help='Use test set to evaluate')
parser.add_argument('-tri', '--train_ri', action='store_true',default= True, help='Train others RI models')
parser.add_argument('-re', '--reuse_exec', action='store_true',default= True, help='Reuse trained models')
parser.add_argument('-tn', '--train_tn', action='store_true', default= True, help='Train Trusted Non-Outliers models')
args = parser.parse_args({})

# %%

#Accept rate
if args.ar_range:
    low_AR, high_AR = args.ar_range

#Weights
if args.weights:
    Weight_AUC, Weight_Kickout = args.weights

#TopSis
weights = [Weight_AUC, Weight_Kickout]
criterias = np.array([True, True])


threshold = args.threshold


#Set seed
if args.seed:
    seed_number = args.seed
    
else:
    seed_number = secrets.randbelow(1_000_000)
    while seed_number <100:
        seed_number = secrets.randbelow(1_000_000)
print(seed_number)
seed_number = 302461
main_seed = seed_number


logpath = Path(os.path.join(ri_datasets_path,f'LC_py/log_{main_seed}'))
logpath.parent.mkdir(parents=True, exist_ok=True)

logging.getLogger().handlers = []
# Configure logging to file
logging.basicConfig(filename=logpath, 
                    filemode='w',  # Overwrite the file each time the application runs
                    level=logging.DEBUG,  # Capture all levels of logging
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format for the timestamp


logging.debug(logpath)

#Read Dataset
    
#Accepts

la = ["issue_d", "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate"]
lb = ["installment", "emp_length", "annual_inc", "verification_status", "loan_status", 
    "purpose", "addr_state", "dti", "delinq_2yrs"]
lc = ["inq_last_6mths", "open_acc", "home_ownership", "revol_bal", "revol_util",
    "total_acc", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_pymnt_inv",
        "last_pymnt_amnt", "last_fico_range_high", "last_fico_range_low"]

selected_columns_a = la+lb+lc
# Define the chunk size for reading the CSV file
chunksize = 100000  # Adjust this value based on your requirements
# Initialize an empty list to store filtered chunks
filtered_chunks = []
# Read the CSV file in chunks based on the defined chunk size
for chunk in pd.read_csv(path+'accepted_2007_to_2018Q4.csv', chunksize=chunksize, usecols=selected_columns_a):
    # Filter the current chunk based on the criteria
    filtered_chunk = chunk[chunk['issue_d'].str.contains(str(args.year), na=False)]
    # Append the filtered chunk to the list
    filtered_chunks.append(filtered_chunk)
# Concatenate all filtered chunks into a single DataFrame
df_a = pd.concat(filtered_chunks)
logging.debug(f'Accepts read with shape: {df_a.shape}')
# Now filtered_df contains only the rows that match the specified criteria
logging.debug(f'Selected columns: {selected_columns_a}')
    
#Rejects
selected_columns_r = ["Application Date", "Debt-To-Income Ratio","State", "Risk_Score", "Amount Requested", "Employment Length"]
# Define the chunk size for reading the CSV file
chunksize = 100000  # Adjust this value based on your requirements
# Initialize an empty list to store filtered chunks
filtered_chunks = []
# Read the CSV file in chunks based on the defined chunk size
for chunk in pd.read_csv(path+'rejected_2007_to_2018Q4.csv', chunksize=chunksize, usecols=selected_columns_r):
    # Filter the current chunk based on the criteria
    chunk["Application Date"] = chunk["Application Date"].astype(str)
    filtered_chunk = chunk[chunk["Application Date"].str.contains(str(args.year), na=False)]
    # filtered_chunk = filtered_chunk[~filtered_chunk["Application Date"].str.contains("2013-10|2013-11|2013-12", na=False)]
    # Append the filtered chunk to the list
    filtered_chunks.append(filtered_chunk)
# Concatenate all filtered chunks into a single DataFrame
df_r = pd.concat(filtered_chunks)
# Now filtered_df contains only the rows that match the specified criteria
logging.debug(f'Rejects read with shape: {df_r.shape}')
# Log the rejected columns
logging.debug(f'Rejected columns: {df_r.columns.tolist()}')
    
#rejected fix names
df_r["emp_length"] = df_r["Employment Length"]
df_r["addr_state"] = df_r["State"]
df_r["dti"] = df_r["Debt-To-Income Ratio"]
df_r["dti"] = pd.to_numeric(df_r['dti'].str.replace('%', ''))
df_r["loan_amnt"] = df_r["Amount Requested"]
df_r["risk_score"] = df_r["Risk_Score"]
df_r["issue_d"] = df_r["Application Date"]

#accepted fix names
df_a["risk_score"] = df_a.loc[:,["last_fico_range_high","last_fico_range_low"]].mean(axis=1)
df_a["target"] = np.where((df_a.loan_status == 'Current') |
                        (df_a.loan_status == 'Fully Paid') |
                        (df_a.loan_status== "Issued") |
                        (df_a.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 0, 1)


    
for c in ["Amount Requested", "Employment Length", "State",
                "Debt-To-Income Ratio", "Amount Requested","Risk_Score", "Application Date"]:
    try:
        df_r = df_r.drop(c, axis = 1)
    except Exception as e:
        pass
for c in ['last_fico_range_high', 'last_fico_range_low', 'loan_status']:
    try:
        df_a = df_a.drop(c, axis = 1)
    except Exception as e:
        pass


# columns based on Shih et al. (2022)
r_cols = df_r.columns.to_list()
pearson_a = ['int_rate', 'dti', 'delinq_2yrs', 'emp_length', 'annual_inc', 'inq_last_6mths', 'term',
'home_ownership','revol_util', 'risk_score', 'target', 'issue_d']
union_list = r_cols.copy()
for item in pearson_a:
    if item not in union_list:
        union_list.append(item)
# Now union_list contains all elements from r_cols first, followed by those unique to pearson_a
df_a = df_a.loc[:, union_list]
        
logging.debug("Now union_list contains all elements from r_cols first, followed by those unique to pearson_a")

    
#Fix dtype of variable emp_length (Object -> number)
try:
    df_a['emp_length'] = df_a['emp_length'].map(lambda x: "0" if x == '< 1 year' else x)
    df_a['emp_length'] = df_a['emp_length'].map(lambda x : int(re.search(r'\d+', x).group()), na_action='ignore')

    df_r['emp_length'] = df_r['emp_length'].map(lambda x: "0" if x == '< 1 year' else x)
    df_r['emp_length'] = df_r['emp_length'].map(lambda x : int(re.search(r'\d+', x).group()), na_action='ignore')
except Exception as e:
    print(e)
try:
    df_a['term'] = df_a['term'].map(lambda x : int(re.search(r'\d+', x).group()))
except Exception as e:
    print(e)

    
#add missing columns to df_r
input_columns = df_a.columns.difference(df_r.columns).to_list()
input_columns.remove('target')

for col in input_columns:
    df_r.insert(df_r.columns.shape[0], col, np.nan)

logging.debug('Data preprocessing complete!')

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

# %%
    
try:
    train_rej = df_r[~df_r['issue_d'].str.contains(f"{args.year}-10|{args.year}-11|{args.year}-12", na=False)]
    train_acp = df_a[~df_a['issue_d'].str.contains(f"Oct-{args.year}|Nov-{args.year}|Dec-{args.year}", na=False)]

    test_rej = df_r[df_r['issue_d'].str.contains(f"{args.year}-10|{args.year}-11|{args.year}-12", na=False)]
    test_acp = df_a[df_a['issue_d'].str.contains(f"Oct-{args.year}|Nov-{args.year}|Dec-{args.year}", na=False)]
except Exception as e:
    print(e)

train_r, train_a, test_r, test_a = train_rej.copy(), train_acp.copy(), test_rej.copy(), test_acp.copy()

for df in [train_r, train_a, test_r, test_a]:
    try:
        df.drop('issue_d', axis = 1, inplace=True)
    except Exception as e:
        pass

logging.debug(f'Train-Test split done')

    
X_train = train_a.loc[:, train_a.columns != "target"]
y = train_a["target"]
X_test = test_a.loc[:, test_a.columns != "target"]
y_test = test_a["target"]

    
knn_inputer = tr.create_pipeline(X_train,y, None, do_EBE=True, crit = 0, do_KNN=True)
knn_inputer.fit(X_train,y)
X_train_knn = knn_inputer[:-3].transform(X_train)
X_test = knn_inputer[:-3].transform(X_test)
R_train_knn = knn_inputer[:-3].transform(train_r)
R_test = knn_inputer[:-3].transform(test_r)
logging.debug(f'KNN input done')


X_train, X_val, y_train, y_val = train_test_split(
    X_train_knn, y, test_size=0.3, random_state=main_seed, shuffle=True)
R_train, R_val = train_test_split(
    R_train_knn, test_size=0.3, random_state=main_seed, shuffle=True)
logging.debug(f'Train-Val split done')


if args.use_test:
    X_eval = X_test.copy()
    y_eval = y_test.copy()
    R_eval = R_test.copy()
else:
    X_eval = X_val.copy()
    y_eval = y_val.copy()
    R_eval = R_val.copy()
    
# %%
models_dict = {}

# Acp classifier benchmark
benchmark = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
benchmark.fit(X_train, y_train)

#RI models
models_dict['BM'] = benchmark

logging.debug(f'benchmark fitted')

# %%

if args.train_ri:
    models_dict.update(
        ri.augmentation_with_soft_cutoff(X_train, y_train, R_train, seed = seed_number))
    logging.debug(f'augmentation_with_soft_cutoff fitted')
    models_dict.update(
        ri.augmentation(X_train, y_train, R_train, mode='up', seed = seed_number))
    logging.debug(f'augmentation upward fitted')
    models_dict.update(
        ri.fuzzy_augmentation(X_train, y_train, R_train, seed = seed_number))
    logging.debug(f'fuzzy_augmentation fitted')
    models_dict.update(
        ri.extrapolation(X_train, y_train, R_train, seed = seed_number))
    logging.debug(f'extrapolation fitted')
    models_dict.update(
        ri.parcelling(X_train, y_train, R_train, seed = seed_number))
    logging.debug(f'parcelling fitted')
    models_dict.update(
        ri.label_spreading(X_train, y_train, R_train, seed = seed_number))
    logging.debug(f'label_spreading fitted')

# %%

if args.train_tn:
    filepath_ex = Path(os.path.join(ri_datasets_path,f'TN-{seed_number}.joblib'))
    filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}.joblib'))

    if filepath_ex.exists() and args.reuse_exec:
        print(filepath_ex.exists())
        models_ex = joblib.load(filepath_ex)
        logging.debug(f'TN loaded')
    else:
        ri.trusted_non_outliers(X_train=X_train, y_train=y_train, X_unl=R_train,
                                        X_val=X_val, y_val=y_val, iterations=50, p=args.percent_bad, acp_rate=0.5,
                                        technique='extrapolation', seed=seed_number, output=-1)
        logging.debug(f'TN fitted')
        models_ls = joblib.load(filepath_ls)
    if filepath_ls.exists() and args.reuse_exec:
        print(filepath_ls.exists())
        models_ls = joblib.load(filepath_ls)
        logging.debug(f'TN+ loaded')
    else:
        ri.trusted_non_outliers(X_train=X_train, y_train=y_train, X_unl=R_train,
                                        X_val=X_val, y_val=y_val, iterations=50, p=args.percent_bad, acp_rate=0.5,
                                        technique='LS', seed=seed_number, output=-1)
        logging.debug(f'TN+ fitted')

# %%

# Initialize a dictionary to hold all the basic metrics
df_metrics = ri.get_metrics_RI(models_dict, X_eval, y_eval, X_unl=R_eval)
if args.use_test:
    filepath = Path(os.path.join(ri_datasets_path, f'metrics_bm_test/Exp_{main_seed}.csv'))
else:
    filepath = Path(os.path.join(ri_datasets_path, f'metrics_bm_val/Exp_{main_seed}.csv'))
filepath.parent.mkdir(parents=True, exist_ok=True)
df_metrics.round(4).to_csv(filepath, index=True)

# Store kickout values for all ARs
# Initialize a dictionary to hold the kickout values for each model
kick_by_model_dict = {}
# Iterate over each model in the models_dict


# %%

for mname, mmodel in models_dict.items():
    # Calculate the pre-kickout probabilities for the benchmark model and the current model
    p_acp, p_all = ri.pre_kickout(models_dict['BM'], models_dict[mname], X_eval, R_eval)    
    # Initialize a dictionary to hold the kickout values for each AR
    ar_dict = {}
    # Iterate over the range of low_AR to high_AR
    for a in range(low_AR, high_AR):
        # Calculate the AR value
        AR = a/100
        # Calculate the kickout value using the faster_kickout function
        kick_value = ri.faster_kickout(y_eval, p_acp, p_all, acp_rate=AR)[0]
        # Store the kickout value in the ar_dict
        ar_dict[a] = kick_value

    # Store the ar_dict in the kick_by_model_dict for the current model
    kick_by_model_dict[mname] = ar_dict

df_kick_by_model = pd.DataFrame(kick_by_model_dict)

# %%

if args.use_test:
    filepath = Path(os.path.join(ri_datasets_path,f'df_kick_by_model_test/Exp_{main_seed}.csv'))
else:
    filepath = Path(os.path.join(ri_datasets_path,f'df_kick_by_model_val/Exp_{main_seed}.csv'))
filepath.parent.mkdir(parents=True, exist_ok=True)
df_kick_by_model.round(4).to_csv(filepath, index=True)

# %%


models_ex = joblib.load(filepath_ex)
models_ls = joblib.load(filepath_ls)

# %%

# output are dicts containing the best model for each AR 
output_ex = ri.evaluate_best_it(models_ex, X_val, y_val, R_val, low_AR, high_AR, weights, criterias)
# print(output_ex)
logging.debug("Extrapolation Mode Analized")
output_ls = ri.evaluate_best_it(models_ls, X_val, y_val, R_val, low_AR, high_AR, weights, criterias)
# print(output_ls)
logging.debug("Label Spreading Mode Analized")

models_dict_ex = {'BM': models_ex['BM']}
models_dict_ls = {'BM': models_ls['BM']}

# %%

# populate the models_dict with the best models for each AR
for a in range(low_AR, high_AR):
    models_dict_ex[a] = models_ex[f'TN_{output_ex[a]}']
    models_dict_ls[a] = models_ls[f'TN_{output_ls[a]}']


# %%

def calculate_metrics_best_model(models_dict, X_eval, R_eval, y_eval, low_AR, high_AR):
    # in this loop we will, for each AR, calculate the metrics for the best model
    for a in range(low_AR, high_AR):
        AR = a/100
        keys_to_extract = ['BM', a]
        # sub_dict is a dict containing only the models in keys_to_extract because we only
        # want to evaluate these models
        sub_dict = {k: models_dict[k] for k in keys_to_extract if k in models_dict}
        ar_df = ri.get_metrics_RI(sub_dict, X_eval, y_eval, X_unl = R_eval, acp_rate=AR)

        # this if is needed because in the first iteration we need to create the dataframes
        if a == low_AR:
            df = ar_df.loc[:,'BM']

        # we drop because each iteration will add the BM row again
        ar_df = ar_df.drop('BM', axis=1)

        # this dataframe will contain the metrics for each model in each AR
        # being each model the best model for the respective AR
        df = pd.concat([df, ar_df], axis=1)
        
    return df

# %%

def calculate_kickout_values(models_dict, X_eval, R_eval, y_eval, low_AR, high_AR):
    TN_dict = {}
    # in this loop we will, for each AR, calculate the kickout value for each model
    for it in models_dict.keys():
        # Generate predictions for model 'BM' and current model 'it'
        p_acp, p_all = ri.pre_kickout(models_dict['BM'], models_dict[it], X_eval, R_eval)
        
        ar_dict = {}
        for a in range(low_AR, high_AR):
            AR = a / 100
            # Calculate kickout value
            kick_value = ri.faster_kickout(y_eval, p_acp, p_all, acp_rate=AR)[0]
            ar_dict[AR] = kick_value

    # Store the results for the current 'it'
        TN_dict[it] = ar_dict

    ARs = sorted(ar_dict.keys())
    tdf = pd.DataFrame(TN_dict, index=ARs)
    
    # tdf contains the kickout values for each model in each AR
    return tdf

# %%

# Calculate the metrics for the best model for each AR
df_ex = calculate_metrics_best_model(models_dict_ex, X_eval, R_eval, y_eval, low_AR, high_AR)
logging.debug(f'Extrapolation Mode Metrics Calculated')
df_ls = calculate_metrics_best_model(models_dict_ls, X_eval, R_eval, y_eval, low_AR, high_AR)
logging.debug(f'Label Spreading Mode Metrics Calculated')


if args.use_test:
    filepath_ex = Path(os.path.join(ri_datasets_path,f'Best_ex_by_ar/test/Exp_{main_seed}.csv'))
    filepath_ls= Path(os.path.join(ri_datasets_path,f'Best_ls_by_ar/test/Exp_{main_seed}.csv'))
else:
    filepath_ex = Path(os.path.join(ri_datasets_path,f'Best_ex_by_ar/val/Exp_{main_seed}.csv'))
    filepath_ls= Path(os.path.join(ri_datasets_path,f'Best_ls_by_ar/val/Exp_{main_seed}.csv'))

filepath_ex.parent.mkdir(parents=True, exist_ok=True)
df_ex.round(4).to_csv(filepath_ex, index=True)
filepath_ls.parent.mkdir(parents=True, exist_ok=True)
df_ls.round(4).to_csv(filepath_ls, index=True)

# %%

# Calculate the kickout values for the best model for each AR
tdf_ex = calculate_kickout_values(models_dict_ex, X_eval, R_eval, y_eval, low_AR, high_AR)
tdf_ls = calculate_kickout_values(models_dict_ls, X_eval, R_eval, y_eval, low_AR, high_AR)

if args.use_test:
    filepath_ex = Path(os.path.join(ri_datasets_path, f'Kickout_ex/test/Exp_{main_seed}.csv'))
    filepath_ls = Path(os.path.join(ri_datasets_path, f'Kickout_ls/test/Exp_{main_seed}.csv'))
else:
    filepath_ex = Path(os.path.join(ri_datasets_path, f'Kickout_ex/val/Exp_{main_seed}.csv'))
    filepath_ls = Path(os.path.join(ri_datasets_path, f'Kickout_ls/val/Exp_{main_seed}.csv'))

tdf_ex.round(4).to_csv(filepath_ex, index=True)
logging.debug(f'Kickout_ex saved')

tdf_ls.round(4).to_csv(filepath_ls, index=True)
logging.debug(f'Kickout_ls saved')
    

logging.debug(f'execution finished')

