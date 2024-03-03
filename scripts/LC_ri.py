

# %%
#@title **Location** of the dataset
path =  "../data/LendingClub/"
process_path = "../data/ProcessedData/"
save_path = "../tests/"
ri_datasets_path = "../data/riData/"
backup_image_folder = "../../backup/Images/"

# %%
import argparse
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
import credit_pipeline.evaluate as ev

from submodules.topsis_python import topsis as top

import wandb

wandb.login()
wandb.errors.term._show_warnings = False

def script():
    parser = argparse.ArgumentParser(description='Simple Script')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold value')
    parser.add_argument('-w', '--wandb', action='store_true', help='Updates results to WandB')
    
    args = parser.parse_args()
    low_AR,high_AR = 20,80
    weights = [1,1]
    criterias = np.array([True, True])

# %%
    # %%
    #@title Set seed

    if args.seed:
        seed_number = args.seed

    else:
        seed_number = secrets.randbelow(1_000_000)
        while seed_number <100:
            seed_number = secrets.randbelow(1_000_000)
    
    print(seed_number)
    main_seed = seed_number

    threshold = args.threshold

    # %%
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

    # %% [markdown]
    # #Read Dataset


    # %%
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
        filtered_chunk = chunk[chunk['issue_d'].str.contains("2009", na=False)]
        # filtered_chunk = filtered_chunk[~filtered_chunk['issue_d'].str.contains("Oct-2013|Nov-2013|Dec-2013", na=False)]
        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    df_a = pd.concat(filtered_chunks)

    logging.debug(f'Accepts read with shape: {df_a.shape}')

    # Now filtered_df contains only the rows that match the specified criteria

    # %%
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
        filtered_chunk = chunk[chunk["Application Date"].str.contains("2009", na=False)]
        # filtered_chunk = filtered_chunk[~filtered_chunk["Application Date"].str.contains("2013-10|2013-11|2013-12", na=False)]
        # Append the filtered chunk to the list
        filtered_chunks.append(filtered_chunk)

    # Concatenate all filtered chunks into a single DataFrame
    df_r = pd.concat(filtered_chunks)

    # Now filtered_df contains only the rows that match the specified criteria
    logging.debug(f'Rejects read with shape: {df_r.shape}')

    # %%
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


    # %%
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


    # %%
    # #Fix dtype of variable emp_length (Object -> number)
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


    # %%
    #add missing columns to df_r
    input_columns = df_a.columns.difference(df_r.columns).to_list()
    input_columns.remove('target')

    for col in input_columns:
        df_r.insert(df_r.columns.shape[0], col, np.nan)

    logging.debug(f'Data preprocessing complete!')

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

    params_dict['RandomForest_1'] = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None,
                            'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2',
                            'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
                            'min_samples_leaf': 9, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0,
                            'n_estimators': 173, 'n_jobs': -1, 'oob_score': False, 'random_state': seed_number,
                            'verbose': 0, 'warm_start': False},


    # %%
    try:
        train_rej = df_r[~df_r['issue_d'].str.contains("2009-10|2009-11|2009-12", na=False)]
        train_acp = df_a[~df_a['issue_d'].str.contains("Oct-2009|Nov-2009|Dec-2009", na=False)]

        test_rej = df_r[df_r['issue_d'].str.contains("2009-10|2009-11|2009-12", na=False)]
        test_acp = df_a[df_a['issue_d'].str.contains("Oct-2009|Nov-2009|Dec-2009", na=False)]
    except Exception as e:
        print(e)

    # %%
    train_r, train_a, test_r, test_a = train_rej.copy(), train_acp.copy(), test_rej.copy(), test_acp.copy()

    for df in [train_r, train_a, test_r, test_a]:
        try:
            df.drop('issue_d', axis = 1, inplace=True)
        except Exception as e:
            pass

    logging.debug(f'Train-Test split done')

    # %%
    X_train = train_a.loc[:, train_a.columns != "target"]
    y = train_a["target"]
    X_test = test_a.loc[:, test_a.columns != "target"]
    y_test = test_a["target"]

    # %%
    knn_inputer = tr.create_pipeline(X_train,y, None, do_EBE=True, crit = 0, do_KNN=True)
    knn_inputer.fit(X_train,y)
    X_train_knn = knn_inputer[:-3].transform(X_train)
    X_test_knn = knn_inputer[:-3].transform(X_test)
    R_train_knn = knn_inputer[:-3].transform(train_r)
    R_test_knn = knn_inputer[:-3].transform(test_r)


    logging.debug(f'KNN input done')


    # %%
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_knn, y, test_size=0.3, random_state=main_seed, shuffle=True)
    R_train, R_val = train_test_split(
        R_train_knn, test_size=0.3, random_state=main_seed, shuffle=True)


    logging.debug(f'Train-Val split done')

    # %%
    models_dict = {}

    # Acp classifier
    benchmark = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    benchmark.fit(X_train, y_train)

    #RI models
    models_dict['BM'] = benchmark

    logging.debug(f'benchmark fitted')

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

    ri.trusted_non_outliers(X_train=X_train, y_train=y_train, X_unl=R_train,
                                       X_val=X_val, y_val=y_val, iterations=50,p=0.2, acp_rate=0.5,
                                       technique='extrapolation', seed=seed_number, output=-1)
    logging.debug(f'TN fitted')
    
    ri.trusted_non_outliers(X_train=X_train, y_train=y_train, X_unl=R_train,
                                       X_val=X_val, y_val=y_val, iterations=50,p=0.2, acp_rate=0.5,
                                       technique='LS', seed=seed_number, output=-1)
    logging.debug(f'TN+ fitted')

    # Assuming `models_dict` is a dictionary of your models
    # Initialize a dictionary to hold all the metrics
    df_metrics = ri.get_metrics_RI(models_dict, X_val, y_val, X_unl = R_val)
    filepath = Path(os.path.join(ri_datasets_path,f'metrics_bm/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.round(4).to_csv(filepath, index=True)

    kick_by_model_dict = {}
    for mname, mmodel in models_dict.items():
        p_acp, p_all = ri.pre_kickout(models_dict['BM'], models_dict[mname], X_val, R_val)
        ar_dict = {}
        for a in range(low_AR, high_AR):
            AR = a/100
            kick_value = ri.faster_kickout(y_val, p_acp, p_all, acp_rate=AR)[0]
            ar_dict[a] = kick_value

        kick_by_model_dict[mname] = ar_dict

    df_kick_by_model = pd.DataFrame(kick_by_model_dict)
    filepath = Path(os.path.join(ri_datasets_path,f'df_kick_by_model/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_kick_by_model.round(4).to_csv(filepath, index=True)

    if args.wandb:
        for mname, mmodel in models_dict.items():
            wandb.init(project="Lending Club Script", name=mname,
                    group=mname,
                    config={'name':mname, 
                            'threshold': threshold,
                            'type' : 'simples'})
            y_prob = mmodel.predict_proba(X_val)[:,1]
            y_pred = (y_prob >= threshold).astype('int')

            # Compute metrics for each model
            metrics_values = {
                f'Overall AUC': roc_auc_score(y_val, y_prob),
                f'Balanced Accuracy': balanced_accuracy_score(y_val, y_pred),
                f'Accuracy': accuracy_score(y_val, y_pred),
                f'Precision': precision_score(y_val, y_pred),
                f'Recall': recall_score(y_val, y_pred),
                f'F1': f1_score(y_val, y_pred),
            }
            # Log all metrics at once
            wandb.log(metrics_values)


            wandb.finish()   

        for mname, mmodel in models_dict.items():
            wandb.init(project="Lending Club Script", name=mname,
                    group=mname,
                    config={'name':mname, 
                            'type' : 'kickout',
                            'threshold': threshold})

            p_acp, p_all = ri.pre_kickout(models_dict['BM'], models_dict[mname], X_val, R_val)
            ar_dict = {}
            if mname != 'BM':
                for a in range(low_AR, high_AR):
                    AR = a/100
                    kick_value = ri.faster_kickout(y_val, p_acp, p_all, acp_rate=AR)[0]
                    ar_dict[a] = kick_value

                    metrics_values = { 'AR' : AR,
                        'Kickout': kick_value}

                    wandb.log(metrics_values, step = a)

            wandb.finish()   
    
    filepath_ex = Path(os.path.join(ri_datasets_path,f'TN-{seed_number}.joblib'))
    filepath_ls = Path(os.path.join(ri_datasets_path,f'TN+-{seed_number}.joblib'))

    models_ex = joblib.load(filepath_ex)
    models_ls = joblib.load(filepath_ls)
    def evaluate_best_it(models, X_val, y_val, R_val, low_AR, high_AR, weights, criterias):
        output_dict = {}
        values = []

        for it in list(models.keys()):
            ar_dict = {}
            auc_value = roc_auc_score(y_val, models[it].predict_proba(X_val)[:,1])
            p_acp, p_all = ri.pre_kickout(models['BM'], models[it], X_val, R_val)
            for a in range(low_AR, high_AR):
                AR = a/100
                kick_value = ri.faster_kickout(y_val, p_acp, p_all, acp_rate=AR)[0]
                ar_dict[a] = kick_value
            it_values = [auc_value, ar_dict]
            values.append(it_values)
            
        values = np.array(values)
        
        for a in range(low_AR, high_AR):
            values_by_ar = [[j,k] for j,k in zip(values[:,0],[i[a] for i in values[:,1]])]
            t = top.Topsis(values_by_ar, weights, criterias)
            t.calc()
            output = t.rank_to_best_similarity()[0] - 1
            output_dict[a] = output
            logging.debug(f'best iteration for AR {a} : {output}')
        return output_dict

    output_ex = evaluate_best_it(models_ex, X_val, y_val, R_val, low_AR, high_AR, weights, criterias)
    output_ls = evaluate_best_it(models_ls, X_val, y_val, R_val, low_AR, high_AR, weights, criterias)

    models_dict_ex = {'BM': models_ex['BM']}
    models_dict_ls = {'BM': models_ls['BM']}

    for a in range(low_AR, high_AR):
        models_dict_ex[a] = models_ex[f'TN_{output_ex[a]}']
        models_dict_ls[a] = models_ls[f'TN_{output_ls[a]}']

    for a in range(low_AR, high_AR):
        AR = a/100
        keys_to_extract = ['BM', a]
        sub_dict_ex = {k: models_dict_ex[k] for k in keys_to_extract if k in models_dict_ex}
        ar_ex = ri.get_metrics_RI(sub_dict_ex, X_val, y_val, X_unl = R_val,
                                                        acp_rate=AR, )
        
        sub_dict_ls = {k: models_dict_ls[k] for k in keys_to_extract if k in models_dict_ls}
        ar_ls = ri.get_metrics_RI(sub_dict_ls, X_val, y_val, X_unl = R_val,
                                                        acp_rate=AR, )
        
        if a == low_AR:
            df_ex = ar_ex.loc[:,'BM']
            df_ls = ar_ls.loc[:,'BM']

        ar_ex = ar_ex.drop('BM', axis=1)
        ar_ls = ar_ls.drop('BM', axis=1)
        
        df_ex = pd.concat([df_ex, ar_ex], axis=1)
        df_ls = pd.concat([df_ls, ar_ls], axis=1)
    
    filepath = Path(os.path.join(ri_datasets_path,f'Best_ex_by_ar/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_ex.round(4).to_csv(filepath, index=True)

    filepath = Path(os.path.join(ri_datasets_path,f'Best_ls_by_ar/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_ls.round(4).to_csv(filepath, index=True)

    TN_dict = {}
    for it in models_ex.keys():
        # Generate predictions for model 'BM' and current model 'it'
        p_acp, p_all = ri.pre_kickout(models_ex['BM'], models_ex[it], X_val, R_val)

        ar_dict = {}
        for a in range(low_AR, high_AR):
            AR = a / 100
            # Calculate kickout value
            kick_value = ri.faster_kickout(y_val, p_acp, p_all, acp_rate=AR)[0]
            ar_dict[AR] = kick_value

        # Store the results for the current 'it'
        TN_dict[it] = ar_dict

    ARs = sorted(ar_dict.keys())
    tdf = pd.DataFrame(TN_dict, index=ARs)

    filepath = Path(os.path.join(ri_datasets_path,f'Kickout_ex/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tdf.round(4).to_csv(filepath, index=True)

    TN_plus_dict = {}
    for it in models_ls.keys():
        # Generate predictions for model 'BM' and current model 'it'
        p_acp, p_all = ri.pre_kickout(models_ls['BM'], models_ls[it], X_val, R_val)

        ar_dict = {}
        for a in range(low_AR, high_AR):
            AR = a / 100
            # Calculate kickout value
            kick_value = ri.faster_kickout(y_val, p_acp, p_all, acp_rate=AR)[0]
            ar_dict[AR] = kick_value

        # Store the results for the current 'it'
        TN_plus_dict[it] = ar_dict

    ARs = sorted(ar_dict.keys())
    tpdf = pd.DataFrame(TN_plus_dict, index=ARs)

    filepath = Path(os.path.join(ri_datasets_path,f'Kickout_ls/Exp_{main_seed}.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tpdf.round(4).to_csv(filepath, index=True)


if __name__ == "__main__":
    script()