from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import NoneType
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                            f1_score, precision_score, recall_score,
                            roc_auc_score, roc_curve)
from scipy.stats import ks_2samp
from sklearn.semi_supervised import LabelSpreading

import credit_pipeline.training as tr


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

def fit_policy(dataset, eval_size=0.2, random_state=880, show_eval_results = False):
    df_train, df_policy = train_test_split(
        dataset, test_size=0.2, random_state=random_state)
    
    X_pol = df_policy.loc[:, cherry_cols]
    y_pol = df_policy["TARGET"]
    X_train_pol, X_val_pol, y_train_pol, y_val_pol = train_test_split(
                            X_pol, y_pol, test_size=eval_size, random_state=random_state)
    
    policy_clf = tr.create_pipeline(X_train_pol, y_train_pol, 
                                    LogisticRegression(**params_dict['LG_balanced']), onehot=False, do_EBE=True)
    policy_clf.fit(X_train_pol, y_train_pol)

    if show_eval_results:
        val_prob = policy_clf.predict_proba(X_val_pol)[:,1]
        print(roc_auc_score(y_val_pol, val_prob))
    
    return  df_train, policy_clf


def accept_reject_split(X,y, policy_clf, threshold = 0.4):
    rej_prob = policy_clf.predict_proba(X)[:,1]

    X_accepts = X[rej_prob < threshold][cols_RI]
    X_rejects = X[rej_prob >= threshold][cols_RI]
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

def get_metrics_RI(name_model_dict, X, y, X_v = None, y_v = None,
                   X_unl = None, threshold_type = 'ks', acp_rate = 0.15):
    def get_best_threshold_with_ks(model, X, y):
        y_probs = model.predict_proba(X)[:,1]
        fpr, tpr, thresholds = roc_curve(y, y_probs)
        return thresholds[np.argmax(tpr - fpr)]

    models_dict = {}
    for name, model in name_model_dict.items():
        if isinstance(model, list):
            y_prob = model[0].predict_proba(X)[:,1]
            threshold_model = model[1]
            y_pred = (y_prob >= threshold_model).astype('int')
        else:
            if threshold_type == 'default':
                threshold = 0.5
            elif threshold_type == 'ks':
                if np.any(X_v):
                    threshold = get_best_threshold_with_ks(model, X_v, y_v)
                else:
                    threshold = get_best_threshold_with_ks(model, X, y)
            elif threshold_type == 'risk':
                if np.any(X_v):
                    threshold = risk_score_threshold(model, X_v, y_v)
                else:
                    threshold = risk_score_threshold(model, X, y)

            y_prob = model.predict_proba(X)[:,1]
            y_pred = (y_prob >= threshold).astype('int')

        models_dict[name] = (y_pred, y_prob)

    def evaluate_ks(y_real, y_proba):
        ks = ks_2samp(y_proba[y_real == 0], y_proba[y_real == 1])
        return ks.statistic

    def get_metrics_df(models_dict, y_true,):
        metrics_dict = {
            "Overall AUC": (
                lambda x: roc_auc_score(y_true, x), False),
            "KS": (
                lambda x: evaluate_ks(y_true, x), False),
            "------": (lambda x: "", True),
            "Balanced Accuracy": (
                lambda x: balanced_accuracy_score(y_true, x), True),
            "Accuracy": (
                lambda x: accuracy_score(y_true, x), True),
            "Precision": (
                lambda x: precision_score(y_true, x), True),
            "Recall": (
                lambda x: recall_score(y_true, x), True),
            "F1": (
                lambda x: f1_score(y_true, x), True),
            "-----": (lambda x: "", True),
        }
        df_dict = {}
        for metric_name, (metric_func, use_preds) in metrics_dict.items():
            df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores)
                                    for model_name, (preds, scores) in models_dict.items()]
        return df_dict

    df_dict = get_metrics_df(models_dict, y)
    if isinstance(X_v, NoneType):
        if isinstance(X_unl, NoneType) or ('original' not in name_model_dict):
            del df_dict["-----"]

    if np.any(X_v):
        df_dict['Approval Rate'] = []
    if np.any(X_unl) and 'original' in name_model_dict:
        df_dict['Kickout'] = []
        df_dict['KG'] = []
        df_dict['KB'] = []

    for name, model in name_model_dict.items():
        if name != 'original':
            if isinstance(model, list):
                if np.any(X_v):
                    df_dict['Approval Rate'].append(calculate_approval_rate(model[0], X_v, y_v, X))
                if np.any(X_unl) and 'original' in name_model_dict:
                    kickout, kg, kb = calculate_kickout_metric(
                        name_model_dict['original'][0], model[0], X, y, X_unl, acp_rate)
                    df_dict['Kickout'].append(kickout*10)
                    df_dict['KG'].append(kg)
                    df_dict['KB'].append(kb)
            else:
                if np.any(X_v):
                    df_dict['Approval Rate'].append(calculate_approval_rate(model, X_v, y_v, X))
                if np.any(X_unl) and 'original' in name_model_dict:
                    if isinstance(name_model_dict["original"], list):
                        original = name_model_dict["original"][0]  # Assuming "original" is a list
                    else:
                        original = name_model_dict["original"]

                    kickout, kg, kb = calculate_kickout_metric(original, model, X, y, X_unl, acp_rate)
                    df_dict['Kickout'].append(kickout*10)
                    df_dict['KG'].append(kg)
                    df_dict['KB'].append(kb)
        else:
            if np.any(X_v):
                if isinstance(model,list):
                    df_dict['Approval Rate'].append(calculate_approval_rate(model[0], X_v, y_v, X))
                else:
                    df_dict['Approval Rate'].append(calculate_approval_rate(model, X_v, y_v, X))
            if np.any(X_unl) and 'original' in name_model_dict:
                df_dict['Kickout'].append(0)
                df_dict['KG'].append(0)
                df_dict['KB'].append(0)

    metrics_df = pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())
    return metrics_df


#Iterative Pipeline With EBE
def trusted_non_outliers(X_train, y_train, X_unl, 
                        contamination_threshold = 0.1,
                        size = 1000,
                        clf_class = LGBMClassifier,
                        clf_params = params_dict['LightGBM_2'],
                        seed = 880,
                        ):
    # get_shapes([X_train, y_train, X_unl, y_unl, X_test, y_test])
    params_dict['LightGBM_2'].update({'random_state': seed})

    if X_unl.shape[0] < 1:
        return X_train, y_train, X_unl

    iso_params = {"contamination":contamination_threshold, "random_state":seed}

    rotulator = tr.create_pipeline(X_train, y_train, clf_class(**clf_params),
                                    onehot=True, normalize=True, do_EBE=True)
    rotulator.fit(X_train, y_train)

    def retrieve_confident_samples(number, size):
        # Fits outlier detection based on bad payers on the train set
        iso = tr.create_pipeline(X_train[y_train == number], y_train[y_train == number],
                                                IsolationForest(**iso_params), do_EBE=True, crit = 0)
        iso.fit(X_train[y_train == number], y_train[y_train == number])
        # Retrieve the samples marked as non-outliers for training
        unl_scores = iso.predict(X_unl)
        X_retrieved = X_unl[unl_scores == 1]
        # Label the non-outliers based on the train set
        y_ret_prob = rotulator.predict_proba(X_retrieved)[:, 1]
        y_retrieved = (y_ret_prob >= 0.5).astype('int')
        y_retrieved = pd.Series(y_retrieved, index=X_retrieved.index)

        # Return empty dataframes if size is 0
        if size == 0:
            return X_retrieved.iloc[[]], y_retrieved.iloc[[]]

        # Only add the most confident predictions to the new training set
        size = size if size < len(y_ret_prob) else int(len(y_ret_prob)/2)
        if number == 0:
            # Get indices of lowest probabilities of defaulting
            confident_indices = np.argpartition(y_ret_prob, size)[:size]

        elif number == 1:
            # Get indices of highest probabilities of defaulting
            confident_indices = np.argpartition(y_ret_prob, -1*size)[-1*size:]

        X_retrieved = X_retrieved.iloc[confident_indices]
        y_retrieved = y_retrieved.iloc[confident_indices]

        # print(f'size_{number} = {size}')

        return X_retrieved, y_retrieved

    p = 0.07#y_train.mean()
    c_0 = int(size-size*p) #number of negative (0) samples to add
    c_1 = int(size*p)      #number of positive (1) samples to add

    X_retrieved_0, y_retrieved_0 = retrieve_confident_samples(0, c_0)
    X_retrieved_1, y_retrieved_1 = retrieve_confident_samples(1, c_1)

    #---------------------------------------------------------------------------

    intersection = X_retrieved_0.index.intersection(X_retrieved_1.index)
    if len(intersection) > 0:
        X_retrieved_0 = X_retrieved_0.drop(intersection)
        y_retrieved_0 = y_retrieved_0.drop(intersection)

        # print('intersection', len(X_retrieved_0.index.intersection(X_retrieved_1.index)))
    #---------------------------------------------------------------------------

    # Concat the datasets
    X_retrieved = pd.concat([X_retrieved_0, X_retrieved_1])
    y_retrieved = pd.concat([y_retrieved_0, y_retrieved_1])

    # Keep the samples marked as outliers in the unlabeled/rejected set
    X_kept = X_unl.loc[~X_unl.index.isin(X_retrieved.index)]

    #---------------------------------------------------------------------------
    # Add the retrieved samples to the new training dataset
    X_train_updated = pd.concat([X_train, X_retrieved])
    y_train_updated = pd.concat([y_train, y_retrieved])
    # print('y_train:', y_train.mean(), 'y_retrieved:', y_retrieved.mean(), 'y_train_updated:', y_train_updated.mean())
    y_train_updated = pd.Series(y_train_updated, index=X_train_updated.index)

    # dex.get_shapes(X_train_updated, y_train_updated, X_kept, y_kept)

    # Return the fitted classifier, updated training and unlabeled sets
    return X_train_updated, y_train_updated, X_kept




#---------Other Strategies----------

def augmentation_with_soft_cutoff(X_train, y_train, X_unl, seed = seed_number):
    """[Augmentation with Soft Cutoff] (Siddiqi, 2012)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    """
    params_dict['LightGBM_2'].update({'random_state': seed})

    #--------------Get Data----------------
    #Create dataset based on Approved(1)/Decline(0) condition
    X_aug_train = pd.concat([X_train, X_unl])

    train_y = np.ones(X_train.shape[0]) #Approved gets 1
    unl_y = np.zeros(X_unl.shape[0])    #Rejected get 0

    #Concat train_y and unl_y
    y_aug_train = pd.Series(np.concatenate([train_y, unl_y]), index = X_aug_train.index)

    #--------------Get Weights----------------
    classifier_AR = tr.create_pipeline(X_aug_train, y_aug_train, LGBMClassifier(**params_dict['LightGBM_2']))
    classifier_AR.fit(X_aug_train, y_aug_train)

    #Get the probabilitie of being approved
    prob_A = classifier_AR.predict_proba(X_aug_train)[:,1]
    prob_A_series = pd.Series(prob_A, index = X_aug_train.index)

    n_scores_interv = 100

    #Sort the probabilities of being accepted
    asc_prob_A = np.argsort(prob_A)
    asc_prob_A_series = prob_A_series.iloc[asc_prob_A]
    # #Split the probs in intervals
    score_interv = np.array_split(asc_prob_A_series,n_scores_interv)

    #Create array for accepts weights
    weights_SC = y_train.copy()
    for s in score_interv:
        #Get index of accepts in s
        acceptees = np.intersect1d(s.index, weights_SC.index)
        if len(acceptees) >= 1:
            #Augmentation Factor (Weight) for the split
            AF = y_aug_train.loc[s.index].mean() #A/(A+R)
            AF_split = np.power(AF ,-1)
            weights_SC.loc[acceptees] = AF_split

    ##--------------Fit classifier----------------
    augmentation_classifier_SC = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    augmentation_classifier_SC.fit(X_train, y_train, classifier__sample_weight = weights_SC)

    return {'aug_SC': augmentation_classifier_SC}

def augmentation(X_train, y_train, X_unl, mode = 'up', seed = seed_number):
    """[Augmentation,Reweighting] (Anderson, 2022), (Siddiqi, 2012)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    mode : str, optional
        _description_, by default 'up'
    """
    params_dict['LightGBM_2'].update({'random_state': seed})
    #--------------Get Data----------------
    #Create dataset based on Approved(1)/Decline(0) condition
    X_aug_train = pd.concat([X_train, X_unl])

    train_y = np.ones(X_train.shape[0]) #Approved gets 1
    unl_y = np.zeros(X_unl.shape[0])    #Rejected get 0

    #Concat train_y and unl_y
    y_aug_train = pd.Series(np.concatenate([train_y, unl_y]), index = X_aug_train.index)

    #--------------Get Weights----------------
    weight_classifier = tr.create_pipeline(X_aug_train, y_aug_train, LGBMClassifier(**params_dict['LightGBM_2']))
    weight_classifier.fit(X_aug_train, y_aug_train)

    #Weights are the probabilitie of being approved
    weights = weight_classifier.predict_proba(X_aug_train)[:,1]

    #Upward: ŵ = w/p(A)
    acp_weights_up = 1/weights[:X_train.shape[0]]

    #Downward: ŵ = w * (1 - p(A))
    acp_weights_down = 1 * (1 - weights[:X_train.shape[0]])

    ##--------------Fit classifier----------------
    if mode == 'up':
        augmentation_classifier_up = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
        augmentation_classifier_up.fit(X_train, y_train, classifier__sample_weight = acp_weights_up)

        return {'aug_up': augmentation_classifier_up}
    elif mode == 'down':
        augmentation_classifier_down = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
        augmentation_classifier_down.fit(X_train, y_train, classifier__sample_weight = acp_weights_down)

        return {'aug_down': augmentation_classifier_down}


def fuzzy_augmentation(X_train, y_train, X_unl, seed = seed_number):
    """[Fuzzy-Parcelling](Anderson, 2022)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    """
    params_dict['LightGBM_2'].update({'random_state': seed})

    #--------------Get Dataset----------------
    X_fuzzy_train = pd.concat([X_train, X_unl, X_unl])
    X_fuzzy_train.index = range(X_fuzzy_train.shape[0])
    good_y = np.zeros(X_unl.shape[0])
    bad_y = np.ones(X_unl.shape[0])

    y_fuzzy_rej = pd.Series(np.concatenate([good_y, bad_y]))
    y_fuzzy_train = pd.concat([y_train, y_fuzzy_rej])
    y_fuzzy_train.index = range(X_fuzzy_train.shape[0])

    #--------------Get Weights----------------
    weight_clf = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    weight_clf.fit(X_train, y_train)
    unl_0_weights = weight_clf.predict_proba(X_unl)[:,0]
    unl_1_weights = weight_clf.predict_proba(X_unl)[:,1]

    train_weights = np.ones(y_train.shape[0])

    fuzzy_weights = np.concatenate([train_weights, unl_0_weights, unl_1_weights])

    #--------------Fit classifier----------------
    fuzzy_classifier = tr.create_pipeline(X_fuzzy_train, y_fuzzy_train, LGBMClassifier(**params_dict['LightGBM_2']))
    fuzzy_classifier.fit(X_fuzzy_train, y_fuzzy_train, classifier__sample_weight = fuzzy_weights)

    return {'fuzzy': fuzzy_classifier}


def extrapolation(X_train, y_train, X_unl, mode = "bad", seed = seed_number):
    """[extrapolation, hard cutoff, Simple Augmentation] (Siddiqi, 2012)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    mode : str, optional
        _description_, by default "bad"
    """

    params_dict['LightGBM_2'].update({'random_state': seed})
    #--------------Fit classifier----------------
    #Fit classifier on Accepts Performance
    default_classifier = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    default_classifier.fit(X_train, y_train)

    y_prob_unl = default_classifier.predict_proba(X_unl)[:,1]
    
    y_label_unl = (y_prob_unl >= 0.5).astype(int)
    y_label_unl_s = pd.Series(y_label_unl, index = X_unl.index)

    #--------------Create new Dataset----------------
    if mode == "bad":
        new_X_train = pd.concat([X_train,X_unl[y_label_unl == 1]])
        new_y_train = pd.concat([y_train,y_label_unl_s[y_label_unl == 1]])
    elif mode == "all":
        new_X_train = pd.concat([X_train,X_unl])
        new_y_train = pd.concat([y_train,y_label_unl_s])
    elif mode == "confident":
        new_X_train = pd.concat([X_train,X_unl[y_prob_unl>0.8], X_unl[y_prob_unl<0.15]])
        new_y_train = pd.concat([y_train,y_label_unl_s[y_prob_unl>0.8], y_label_unl_s[y_prob_unl<0.15]])

    #--------------Fit classifier----------------
    extrap_classifier = tr.create_pipeline(new_X_train, new_y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    extrap_classifier.fit(new_X_train, new_y_train)
    # +'-'+mode
    return {'ext': extrap_classifier}


def parcelling(X_train, y_train, X_unl, n_scores_interv = 100, prejudice = 3, seed = seed_number):
    """[Parcelling] (Siddiqi, 2012)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    n_scores_interv : int, optional
        _description_, by default 100
    prejudice : int, optional
        _description_, by default 3
    """
    params_dict['LightGBM_2'].update({'random_state': seed})

    #--------------Create new Dataset----------------
    #Create dataset with Approved and Rejected
    X_aug_train = pd.concat([X_train, X_unl])

    train_y = y_train.copy().array
    unl_y = np.zeros(X_unl.shape[0])    #Placeholder value

    #Concat train_y and unl_y
    y_aug_train = pd.Series(np.concatenate([train_y, unl_y]), index = X_aug_train.index)

    #--------------Fit classifier on Accepts Performance----------------
    default_classifier = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    default_classifier.fit(X_train, y_train)

    prob_def = default_classifier.predict_proba(X_aug_train)[:,1]
    prob_def_s = pd.Series(prob_def, index = X_aug_train.index)

    #--------------Assing Good and Bad labels for rejects----------------

    #Sort the probabilities of being bad payers
    asc_prob_def = np.argsort(prob_def)
    asc_prob_def_s = prob_def_s.iloc[asc_prob_def]

    #Split the probs in intervals
    score_interv = np.array_split(asc_prob_def_s,n_scores_interv)

    for s in score_interv:
        #Get index of accepts in s
        acceptees = np.intersect1d(s.index, X_train.index)
        rejects = np.intersect1d(s.index, X_unl.index)
        if len(acceptees) >= 1:
            #percent of bad in acceptees
            bad_rate = y_train.loc[acceptees].mean()
            #adjusted percent of bad to rejects
            bad_rate = bad_rate*prejudice
            #percent of good in acceptees
            good_rate = (1-bad_rate)
            #expected number of good in rejects
            good_rate_in_R = int(good_rate*len(rejects))
            #randomize rejects
            random_rejects = np.random.default_rng(seed=seed).permutation(rejects)
            #select good_rate_in_R as good
            as_good = random_rejects[:good_rate_in_R]
            y_aug_train.loc[as_good] = 0
            #select the left as bad
            as_bad = random_rejects[good_rate_in_R:]
            y_aug_train.loc[as_bad] = 1

    #--------------Fit classifier---------------
    parcelling_classifier = tr.create_pipeline(X_aug_train, y_aug_train, LGBMClassifier(**params_dict['LightGBM_2']))
    parcelling_classifier.fit(X_aug_train, y_aug_train)

    return {'parcelling': parcelling_classifier}


def label_spreading(X_train, y_train, X_unl, seed = seed_number):
    """[Label Spreading] (Zhou, 2004)(Kang, 2021)

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_
    X_unl : _type_
        _description_
    """

    params_dict['LightGBM_2'].update({'random_state': seed})
    #--------------Create dataset with Approved and Rejected---------------
    X_train_ls, y_train_ls, X_unl_ls = X_train.copy(), y_train.copy(), X_unl.copy()

    X_combined = pd.concat([X_train_ls, X_unl_ls])

    y_unl_ls = np.array([-1]*X_unl_ls.shape[0])
    y_combined = np.concatenate([y_train_ls.array, y_unl_ls])
    y_combined = pd.Series(y_combined, index=X_combined.index)

    n_labeled_points = y_train_ls.shape[0]
    indices = np.arange(y_combined.shape[0])

    #--------------Predict labels on the unlabeled data---------------
    lp_model = tr.create_pipeline(X_combined, y_combined, LabelSpreading(**params_dict['LabelSpreading_2']))
    lp_model.fit(X_combined, y_combined)
    predicted_labels = lp_model['classifier'].transduction_[indices[n_labeled_points:]]

    y_label_pred_s = pd.Series(predicted_labels, index=X_unl_ls.index)

    #--------------Fit classifier---------------
    #Create a new classifier pipeline using labeled and unlabeled data, and fit it
    new_X_train = pd.concat([X_train, X_unl_ls])
    new_y_train = pd.concat([y_train, y_label_pred_s])

    clf_LS = tr.create_pipeline(new_X_train, new_y_train, LGBMClassifier(**params_dict['LightGBM_2']))
    clf_LS.fit(new_X_train, new_y_train,)

    return {'labelSpr': clf_LS}
