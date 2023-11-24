import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import credit_pipeline.training as tr

from sklearn.model_selection import train_test_split

seed_number = 880

params_dict = {
    'RandomForest_1' : {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None,
                        'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2',
                        'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
                        'min_samples_leaf': 9, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0,
                        'n_estimators': 173, 'n_jobs': None, 'oob_score': False, 'random_state': seed_number,
                        'verbose': 0, 'warm_start': False},

    'RandomForest_2' : {'n_estimators': 113, 'criterion': 'entropy', 'max_depth': 9,
                        'min_samples_split': 9,'min_samples_leaf': 9,
                        'max_features': 'sqrt', 'bootstrap': False, 'random_state': seed_number,},

    'RandomForest_3' : {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None,
                        'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto',
                        'max_leaf_nodes': None, 'max_samples': None,
                        'min_impurity_decrease': 0.0, 'min_samples_leaf': 6,
                        'min_samples_split': 12, 'min_weight_fraction_leaf': 0.0,
                        'n_estimators': 168, 'n_jobs': None, 'oob_score': False,
                        'random_state': seed_number, 'verbose': 0, 'warm_start': False},

    'DecisionTree_1' : {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
                'max_depth': None, 'max_features': None, 'max_leaf_nodes': 25,
                'min_impurity_decrease': 0.0, 'min_samples_leaf': 100,
                'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0,
                'random_state': seed_number, 'splitter': 'best'},

    'LightGBM_1' : {'boosting_type': 'gbdt', 'class_weight': None,
                'colsample_bytree': 0.29280447167767465, 'importance_type': 'split',
                'learning_rate': 0.025693783679932296, 'max_depth': 6,
                'min_child_samples': 48, 'min_child_weight': 0.001,
                'min_split_gain': 0.0, 'n_estimators': 194, 'n_jobs': -1,
                'num_leaves': 35, 'objective': None, 'random_state': seed_number,
                'reg_alpha': 0.42692653558951865, 'reg_lambda': 0.7009056503658567,
                'verbose': -1, 'subsample': 0.9673695418639782,
                'subsample_for_bin': 200000, 'subsample_freq': 0,
                'is_unbalance': True},


    'LightGBM_2' : {'boosting_type': 'gbdt', 'class_weight': None,
              'colsample_bytree': 0.22534977954592625, 'importance_type': 'split',
              'learning_rate': 0.052227873762946964, 'max_depth': 5,
              'min_child_samples': 26, 'min_child_weight': 0.001,
              'min_split_gain': 0.0, 'n_estimators': 159, 'n_jobs': -1,
              'num_leaves': 12, 'objective': None, 'random_state': seed_number,
              'reg_alpha': 0.7438345471808012, 'reg_lambda': 0.46164693905368515,
                'verbose': -1, 'subsample': 0.8896599304061413,
              'subsample_for_bin': 200000, 'subsample_freq': 0,
              'is_unbalance': True},



    'LG_balanced' : { 'solver': 'liblinear','penalty': 'l1',
            'random_state': seed_number,'max_iter': 10000,   'class_weight': 'balanced'},

    'LG_1' : { 'solver': 'liblinear','penalty': 'l2',
            'random_state': seed_number,'max_iter': 1000, },

    'LabelSpreading_1' : {'alpha': 0.9212289329319412, 'gamma': 0.024244533484333246,
                        'kernel': 'knn', 'max_iter': 50, 'n_jobs': None,
                          'n_neighbors': 10, 'tol': 0.001, },
    'LabelSpreading_2' : {'alpha': 0.2, 'gamma': 20,
                        'kernel': 'knn', 'max_iter': 30, 'n_jobs': None,
                          'n_neighbors': 7, 'tol': 0.001,},

    }

#Cherry picked columns for AR policy
cherry_cols = ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
        "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
        "FLAG_OWN_REALTY", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
        "NAME_HOUSING_TYPE", "OWN_CAR_AGE", "FLAG_MOBIL", "FLAG_EMP_PHONE",
        "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE",
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
        "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE",
        "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
        "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI",
        "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI",
        "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
        "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE", "FLAG_DOCUMENT_3",
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]

#Columns to use on RI study
cols_RI = ['AMT_CREDIT', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
        'EXT_SOURCE_3', 'REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'REG_CITY_NOT_WORK_CITY', 'AMT_GOODS_PRICE',
        'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE', 'NAME_CONTRACT_TYPE']

def fit_policy(dataset, test_size=0.2, random_state=880, show_eval_results = False):
    df_train, df_policy = train_test_split(
        dataset, test_size=0.2, random_state=random_state)
    
    X_pol = df_policy.loc[:, cherry_cols]
    y_pol = df_policy["TARGET"]
    X_train_pol, X_val_pol, y_train_pol, y_val_pol = train_test_split(
                            X_pol, y_pol, test_size=0.2, random_state=random_state)
    
    policy_clf = tr.create_pipeline(X_train_pol, y_train_pol, 
                                    LogisticRegression(**params_dict['LG_balanced']), onehot=False, do_EBE=True)
    policy_clf.fit(X_train_pol, y_train_pol)

    if show_eval_results:
        val_prob = policy_clf.predict_proba(X_val_pol)[:,1]
        print(roc_auc_score(y_val_pol, val_prob))
    
    return  df_train, policy_clf


def accept_reject_split(X,y, policy_clf = None, threshold = 0.4):
    rej_prob = policy_clf.predict_proba(X)[:,1]

    X_accepts = X[rej_prob < threshold][ri.cols_RI]
    X_rejects = X[rej_prob >= threshold][ri.cols_RI]
    y_accepts = y[rej_prob < threshold]
    y_rejects = y[rej_prob >= threshold]

    return X_accepts, X_rejects, y_accepts, y_rejects

#Metrics

def calculate_kickout_metric(C1, C2, X_test_acp, y_test_acp, X_test_unl, acp_rate = 0.15):
    # Calculate predictions and obtain subsets A1_G and A1_B
    num_acp_1 = int(len(X_test_acp) * acp_rate) #number of Accepts
    y_prob_1 = C1.predict_proba(X_test_acp)[:, 1]
    threshold = np.percentile(y_prob_1, 100 - (num_acp_1 / len(y_prob_1)) * 100)
    y_pred_1 = (y_prob_1 > threshold).astype('int')
    A1 = X_test_acp[y_pred_1 == 0]
    A1_G = X_test_acp[(y_pred_1 == 0) & (y_test_acp == 0)]
    A1_B = X_test_acp[(y_pred_1 == 0) & (y_test_acp == 1)]

    X_test_holdout = pd.concat([X_test_acp, X_test_unl])

    # Calculate predictions on X_test_holdout and obtain subset A2
    num_Acp_2 = int(len(X_test_holdout) * acp_rate) #number of Accepts
    y_prob_2 = C2.predict_proba(X_test_holdout)[:, 1]
    threshold = np.percentile(y_prob_2, 100 - (num_Acp_2 / len(y_prob_2)) * 100)
    y_pred_2 = (y_prob_2 > threshold).astype('int')
    A2 = X_test_holdout[y_pred_2 == 0]

    # Calculate indices of kicked-out good and bad samples
    indices_KG = np.setdiff1d(A1_G.index, A2.index)
    indices_KB = np.setdiff1d(A1_B.index, A2.index)

    # Calculate the count of kicked-out good and bad samples
    KG = A1_G.loc[indices_KG].shape[0]
    KB = A1_B.loc[indices_KB].shape[0]

    # Calculate the share of bad cases in A1
    p_B = A1_B.shape[0] / A1.shape[0]

    # Calculate the number of bad cases selected by the original model
    SB = A1_B.shape[0]

    # Calculate the kickout metric value
    kickout = ((KB / p_B) - (KG / (1 - p_B))) / (SB / p_B)

    return kickout, KG, KB

def risk_score_threshold(model, X, y, plot = False, defaul_acceped = 0.04):
    #calculate probabilities on validation set
    y_probs = model.predict_proba(X)[:,1]
    #sort index of probabilies on ascending order
    sorted_clients = np.argsort(y_probs)
    #calculate the comulative mean of the probabilities
    cum_mean = np.cumsum(y.iloc[sorted_clients])/np.arange(1, y.shape[0]+1)
    #turn to zero the first 1000 values to reduce noise
    cum_mean[:1000] = np.zeros(1000)
    #get the minimum threshold value that accepts until 4 percent default rate
    thr_0 = y_probs[sorted_clients][np.argmin(abs(cum_mean-defaul_acceped))]

    #plot the threshold x cum_mean graph
    if plot:
        plt.plot(y_probs[sorted_clients], cum_mean, '.')
    return thr_0

def calculate_approval_rate(C1, X_val, y_val, X_test):
    threshold = risk_score_threshold(C1, X_val, y_val)
    y_prob = C1.predict_proba(X_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)  # Convert probabilities to binary predictions
    n_approved = (y_pred == 0).sum()

    return n_approved/X_test.shape[0]
