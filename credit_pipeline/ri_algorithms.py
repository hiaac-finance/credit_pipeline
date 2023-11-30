#Iterative Pipeline With EBE
def trusted_non_outliers(contamination_threshold, size,
                                X_train, y_train, X_unl, seed = seed_number):
    # get_shapes([X_train, y_train, X_unl, y_unl, X_test, y_test])

    if X_unl.shape[0] < 1:
        return X_train, y_train, X_unl

    iso_params = {"contamination":contamination_threshold, "random_state":seed}

    rotulator = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']),
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


def augmentation_with_soft_cutoff(X_train, y_train, X_unl):
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

    return ('aug_SC', augmentation_classifier_SC)

def augmentation(X_train, y_train, X_unl, mode = 'up'):
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

        return 'aug_up', augmentation_classifier_up
    elif mode == 'down':
        augmentation_classifier_down = tr.create_pipeline(X_train, y_train, LGBMClassifier(**params_dict['LightGBM_2']))
        augmentation_classifier_down.fit(X_train, y_train, classifier__sample_weight = acp_weights_down)

        return 'aug_down', augmentation_classifier_down


def fuzzy_augmentation(X_train, y_train, X_unl):
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

    return 'fuzzy', fuzzy_classifier


def extrapolation(X_train, y_train, X_unl, mode = "bad"):
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


def parcelling(X_train, y_train, X_unl, n_scores_interv = 100, prejudice = 3):
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
            random_rejects = np.random.permutation(rejects)
            #select good_rate_in_R as good
            as_good = random_rejects[:good_rate_in_R]
            y_aug_train.loc[as_good] = 0
            #select the left as bad
            as_bad = random_rejects[good_rate_in_R:]
            y_aug_train.loc[as_bad] = 1

    #--------------Fit classifier---------------
    parcelling_classifier = tr.create_pipeline(X_aug_train, y_aug_train, LGBMClassifier(**params_dict['LightGBM_2']))
    parcelling_classifier.fit(X_aug_train, y_aug_train)

    return 'parcelling', parcelling_classifier


def label_spreading(X_train, y_train, X_unl):
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

    return 'labelSpr', clf_LS


