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

def load_policy(path = 'drive'):
    if path == 'drive':
        policy_model_path = "drive/Shareddrives/H.IAAC - AI in Finance/Models/AR_policy.joblib"
    else:
        policy_model_path = path

    loaded_policy = joblib.load(policy_model_path)
    policy_model, policy_samples = loaded_policy.values()

    return policy_model, policy_samples

def remove_policy_samples(data, policy_samples):
    return data[~data['SK_ID_CURR'].isin(policy_samples)]

def accept_reject_split(data, threshold = 0.4, path = 'drive'):
    if path == 'drive':
        policy_model_path = "drive/Shareddrives/H.IAAC - AI in Finance/Models/AR_policy.joblib"
    else:
        policy_model_path = path

    loaded_policy = joblib.load(policy_model_path)
    policy_model, policy_samples = loaded_policy.values()

    rej_prob = policy_model.predict_proba(data)[:,1]
    accepts = data[rej_prob < threshold]
    rejects = data[rej_prob >= threshold]

    return accepts, rejects
