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
