import numpy as np
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading as SkLabelSpreading

from credit_pipeline.training import create_pipeline


class RejectInferenceAlg:
    def __init__(self, clf, reweighting=True):
        self.clf = clf
        self.reweighting = reweighting
    
    def fit(self, X, y, X_unl):
        X, y, sample_weights = self.update_data(X, y, X_unl)
        self.pipeline = create_pipeline(X, y, self.clf)
        self.pipeline.fit(X, y, classifier__sample_weight=sample_weights)

        return self
        
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.pipeline.predict_proba(X)
        return probs

    def update_data(self, X, y, X_unl):
        """Update the training data with the unlabeled data.
        Should be implemented in subclasses.
        """
        return X, y, pd.Series(np.ones(X.shape[0]), index=X.index)

class AugSoftCutoff(RejectInferenceAlg):
    def __init__(self, clf):
        super().__init__(clf, reweighting=True)


    def update_data(self, X, y, X_unl):
        X_complete = pd.concat([X, X_unl])
        y_complete = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        y_complete = pd.Series(y_complete, index=X_complete.index)

        accept_clf = create_pipeline(X_complete, y_complete, self.clf)
        accept_clf.fit(X_complete, y_complete)
        
        #Get the probabilitie of being approved
        prob_A = accept_clf.predict_proba(X_complete)[:,1]
        prob_A_series = pd.Series(prob_A, index = X_complete.index)

        n_scores_interv = 100

        #Sort the probabilities of being accepted
        asc_prob_A = np.argsort(prob_A)
        asc_prob_A_series = prob_A_series.iloc[asc_prob_A]
        # #Split the probs in intervals
        score_interv = np.array_split(asc_prob_A_series,n_scores_interv)

        #Create array for accepts weights
        sample_weights = pd.Series(np.ones(X.shape[0]), index=X.index)
        for s in score_interv:
            #Get index of accepts in s
            acceptees = np.intersect1d(s.index, sample_weights.index)
            if len(acceptees) >= 1:
                #Augmentation Factor (Weight) for the split
                AF = y_complete.loc[s.index].mean() #A/(A+R)
                AF_split = np.power(AF ,-1)
                sample_weights.loc[acceptees] = AF_split

        return X, y, sample_weights



class AugUpDown(RejectInferenceAlg):
    def __init__(self, clf, method = "up"):
        super().__init__(clf)
        assert method in ["up", "down"], "Method must be 'up' or 'down'."
        self.method = method


    def update_data(self, X, y, X_unl):
        X_complete = pd.concat([X, X_unl])
        y_complete = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        accept_clf = create_pipeline(X_complete, y_complete, self.clf)
        accept_clf.fit(X_complete, y_complete)

        #Weights are the probabilitie of being approved
        weights = accept_clf.predict_proba(X_unl)[:,1]

        if self.method == "up":
            #Upward: ŵ = w/p(A)
            sample_weights = 1/weights[:X.shape[0]]
        elif self.method == "down":
            #Downward: ŵ = w * (1 - p(A))
            sample_weights = 1 * (1 - weights[:X.shape[0]])
    
        return X, y, sample_weights
    

class AugFuzzy(RejectInferenceAlg):
    def __init__(self, clf):
        super().__init__(clf)
        self.estimator = clf


    def update_data(self, X, y, X_unl):
        X_fuzzy = pd.concat([X, X_unl, X_unl])
        X_fuzzy.index = range(X_fuzzy.shape[0])
        good_y = np.zeros(X_unl.shape[0])
        bad_y = np.ones(X_unl.shape[0])

        y_fuzzy_rej = pd.Series(np.concatenate([good_y, bad_y]))
        y_fuzzy = pd.concat([y, y_fuzzy_rej])
        y_fuzzy.index = range(X_fuzzy.shape[0])

        weight_clf = create_pipeline(X, y, self.clf)
        weight_clf.fit(X, y)

        unl_0_weights = weight_clf.predict_proba(X_unl)[:,0]
        unl_1_weights = weight_clf.predict_proba(X_unl)[:,1]

        train_weights = np.ones(y.shape[0])
        sample_weights = np.concatenate([train_weights, unl_0_weights, unl_1_weights])

        return X_fuzzy, y_fuzzy, pd.Series(sample_weights, index=X_fuzzy.index)

class Extrapolation(RejectInferenceAlg):
    def __init__(self, clf, augmentation_type = "only_1"):
        super().__init__(clf)
        assert augmentation_type in ["only_1", "all", "confident"], "augmentation_type must be 'only_1', 'all' or 'confident'."
        self.augmentation_type = augmentation_type

    def update_data(self, X, y, X_unl):
        default_clf = create_pipeline(X, y, self.clf)
        default_clf.fit(X, y)

        y_prob_unl = default_clf.predict_proba(X_unl)[:,1]
        if self.augmentation_type == "only_1":
            X_combined = pd.concat([X, X_unl[y_prob_unl >= 0.5]])
            n_new = (y_prob_unl >= 0.5).sum()
            y_combined = np.concatenate([y, np.ones(n_new)])
            y_combined = pd.Series(y_combined, index=X_combined.index)
        elif self.augmentation_type == "all":
            X_combined = pd.concat([X, X_unl])
            y_combined = np.concatenate([y, (y_prob_unl >= 0.5).astype(int)])
            y_combined = pd.Series(y_combined, index=X_combined.index)
        elif self.augmentation_type == "confident":
            X_combined = pd.concat([X, X_unl[(y_prob_unl > 0.8) | (y_prob_unl < 0.15)]])
            y_new = y_prob_unl[(y_prob_unl > 0.8) | (y_prob_unl < 0.15)] >= 0.5
            y_combined = np.concatenate([y, y_new.astype(int)])
        y_combined = pd.Series(y_combined, index=X_combined.index)
        
        return X_combined, y_combined, pd.Series(np.ones(X_combined.shape[0]), index=X_combined.index)





class LabelSpreading(RejectInferenceAlg):
    def __init__(self, clf):
        super().__init__(clf)
    

    def update_data(self, X, y, X_unl):
        X_combined = pd.concat([X, X_unl])
        y_unl = np.ones(X_unl.shape[0]) * -1  # Unlabeled data gets -1
        y_combined = np.concatenate([y, y_unl])
        y_combined = pd.Series(y_combined, index=X_combined.index)

        clf_aux = create_pipeline(
            X_combined, 
            y_combined, 
            SkLabelSpreading(
                **{'alpha': 0.2, 'gamma': 20,
                'kernel': 'knn', 'max_iter': 30, 'n_jobs': None,
                'n_neighbors': 7, 'tol': 0.001,}
            )
        )
        clf_aux.fit(X_combined, y_combined)
        predicted_labels = clf_aux['classifier'].transduction_

        # create new y with the original labels and the predicted labels for unlabeled data
        n_labels = y.shape[0]
        y_combined = np.concatenate([y.array, predicted_labels[n_labels:]])
        y_combined = pd.Series(y_combined, index=X_combined.index)

        return X_combined, y_combined, pd.Series(np.ones(X_combined.shape[0]), index=X_combined.index)
    



from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score


def calculate_kickout_metric(C1, C2, X_test, y_test, X_unl_holdout, acp_rate = 0.15):
    # Calculate predictions and obtain subsets A1_G and A1_B
    num_Acp_1 = int(len(X_test) * acp_rate) #number of Accepts
    y_prob_1 = C1.predict_proba(X_test)[:, 1]
    threshold = np.percentile(y_prob_1, 100 - (num_Acp_1 / len(y_prob_1)) * 100)
    y_pred_1 = (y_prob_1 > threshold).astype('int')
    A1 = X_test[y_pred_1 == 0]
    A1_G = X_test[(y_pred_1 == 0) & (y_test == 0)]
    A1_B = X_test[(y_pred_1 == 0) & (y_test == 1)]

    # Calculate predictions on X_test_holdout and obtain subset A2
    num_Acp_2 = int(len(X_unl_holdout) * acp_rate) #number of Accepts
    y_prob_2 = C2.predict_proba(X_unl_holdout)[:, 1]
    threshold = np.percentile(y_prob_2, 100 - (num_Acp_2 / len(y_prob_2)) * 100)
    y_pred_2 = (y_prob_2 > threshold).astype('int')
    A2 = X_unl_holdout[y_pred_2 == 0]

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


def get_best_threshold_with_ks(model, X, y):
    y_probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    return thresholds[np.argmax(tpr - fpr)]


def risk_score_threshold(model, X, y, defaul_acceped = 0.04):
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

    return thr_0



def calculate_approval_rate(C1, X_val, y_val, X_test):
    threshold = risk_score_threshold(C1, X_val, y_val)
    y_prob = C1.predict_proba(X_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)  # Convert probabilities to binary predictions
    n_approved = (y_pred == 0).sum()

    return n_approved/X_test.shape[0]




#@title Function to get metrics for Reject Inference

def get_metrics_RI(name_model_dict, X, y, X_v = False, y_v = False,
                   X_unl_holdout = False, threshold_type = 'default', acp_rate = 0.15):
    models_dict = {}
    for name, model in name_model_dict.items():
        if type(model) == list:
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
    if np.any(X_v) == False:
        if np.any(X_unl_holdout) == False or 'original' not in name_model_dict:
            del df_dict["-----"]

    if np.any(X_v):
        df_dict['Approval Rate'] = []
    if np.any(X_unl_holdout) and 'original' in name_model_dict:
        df_dict['Kickout'] = []
        df_dict['KG'] = []
        df_dict['KB'] = []

    for name, model in name_model_dict.items():
        if name != 'original':
            if type(model) == list:
                if np.any(X_v):
                    df_dict['Approval Rate'].append(calculate_approval_rate(model[0], X_v, y_v, X))
                if np.any(X_unl_holdout) and 'original' in name_model_dict:
                    kickout, kg, kb = calculate_kickout_metric(name_model_dict['original'][0], model[0], X, y, X_unl_holdout, acp_rate)
                    df_dict['Kickout'].append(kickout*10)
                    df_dict['KG'].append(kg)
                    df_dict['KB'].append(kb)
            else:
                if np.any(X_v):
                    df_dict['Approval Rate'].append(calculate_approval_rate(model, X_v, y_v, X))
                if np.any(X_unl_holdout) and 'original' in name_model_dict:
                    try:
                        original = name_model_dict['original'][0]
                        kickout, kg, kb = calculate_kickout_metric(original, model, X, y, X_unl_holdout, acp_rate)
                    except:
                        original = name_model_dict['original']
                        kickout, kg, kb = calculate_kickout_metric(original, model, X, y, X_unl_holdout, acp_rate)
                    df_dict['Kickout'].append(kickout*10)
                    df_dict['KG'].append(kg)
                    df_dict['KB'].append(kb)
        else:
            if np.any(X_v):
                if type(model) == list:
                    df_dict['Approval Rate'].append(calculate_approval_rate(model[0], X_v, y_v, X))
                else:
                    df_dict['Approval Rate'].append(calculate_approval_rate(model, X_v, y_v, X))
            if np.any(X_unl_holdout) and 'original' in name_model_dict:
                df_dict['Kickout'].append(0)
                df_dict['KG'].append(0)
                df_dict['KB'].append(0)

    metrics_df = pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())
    return metrics_df
