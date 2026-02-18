from typing import Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.semi_supervised import LabelSpreading


class RejectInference:
    """
    Base class for the implementation of Reject Inference methods
    """

    def init(self, base_estimator: BaseEstimator, reject_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.reject_estimator = reject_estimator

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "RejectInference":
        """Standard call for reject inference methods.
        Feature matrix should include in sequence data from labaled and unlabeled population.
        Labels y are only for the labeled population, and the unlabeled population should be labeled as -1.

        All procedures will update the training set (X, y) or define sample weights using the method _preprocess.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix of the labeled data.
        y : Union[np.ndarray, pd.Series]
            Labels.

        Returns
        -------
        RejectInference
            The fitted RejectInference object.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X_unl = X[y == -1]
        X = X[y != -1]
        y = y[y != -1]
        X_updated, y_updated, sample_weights = self._preprocess(X, y, X_unl)
        self.base_estimator.fit(X_updated, y_updated, sample_weight=sample_weights)
        return self

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class RejectUpward(RejectInference):
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        self.reject_estimator.fit(X_concat[indices], y_concat[indices])

        sample_weights = self.reject_estimator.predict_proba(X)[
            :, 1
        ]  # probability of being accepted
        sample_weights = 1 / sample_weights  # upweighting the accepted samples

        return X, y, sample_weights


class RejectDownward(RejectInference):
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        self.reject_estimator.fit(X_concat[indices], y_concat[indices])

        sample_weights = self.reject_estimator.predict_proba(X)[
            :, 1
        ]  # probability of being accepted
        sample_weights = 1 - sample_weights

        return X, y, sample_weights


class RejectSoftCutoff(RejectInference):
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        self.reject_estimator.fit(X_concat[indices], y_concat[indices])

        prob_accept = self.reject_estimator.predict_proba(X_concat)[:, 1]
        intervals = np.percentile(prob_accept, np.linspace(0, 100, num=100))
        sample_weights = np.ones(X_concat.shape[0])
        for i in range(len(intervals) - 1):
            in_interval = (prob_accept >= intervals[i]) & (
                prob_accept < intervals[i + 1]
            )
            if np.sum(in_interval) > 0:
                accept_rate = np.mean(y_concat[in_interval])  # A/(A+R)
                sample_weights[in_interval] = 1 / accept_rate if accept_rate > 0 else 1

        sample_weights = sample_weights[
            : X.shape[0]
        ]  # only keep weights for the accepted samples

        return X, y, sample_weights


class FuzzyParcelling(RejectInference):
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.base_estimator.fit(X, y)
        prob_1 = self.reject_estimator.predict_proba(X_unl)[:, 1]

        X_concat = np.concatenate([X, X_unl, X_unl])
        y_concat = np.concatenate(
            [y, np.zeros(X_unl.shape[0]), np.ones(X_unl.shape[0])]
        )
        sample_weights = np.concatenate([np.ones(X.shape[0]), 1 - prob_1, prob_1])
        return X_concat, y_concat, sample_weights


class RejectExtrapolation(RejectInference):
    def init(
        self,
        base_estimator: BaseEstimator,
        reject_estimator: BaseEstimator,
        mode: str = "positive",
    ):
        super().init(base_estimator, reject_estimator)
        self.mode = mode

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.base_estimator.fit(X, y)
        prob_1 = self.reject_estimator.predict_proba(X_unl)[:, 1]
        if self.mode == "positive":
            selected = prob_1 > 0.5
        elif self.mode == "all":
            selected = np.ones_like(prob_1, dtype=bool)
        elif self.mode == "confident":
            selected = (prob_1 > 0.8) | (prob_1 < 0.15)

        Y_unl = (prob_1 > 0.5).astype(int)
        X_new = np.concatenate([X, X_unl[selected]])
        y_new = np.concatenate([y, Y_unl[selected]])
        sample_weights = np.ones(X_new.shape[0])
        return X_new, y_new, sample_weights


class RejectSpreading(RejectInference):
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([y, -1 * np.ones(X_unl.shape[0])])

        label_spreading = LabelSpreading(kernel="knn")
        label_spreading.fit(X_concat, y_concat)
        y_new = label_spreading.transduction_
        sample_weights = np.ones(X_concat.shape[0])
        return X_concat, y_new, sample_weights


def get_threshold_bad_rate(y_true: np.array, y_probs: np.array, bad_rate: float = 0.25) -> float:
    """Calculate the threshold that maximizes the true positives while keeping the false positives below the bad_rate.

    Parameters
    ----------
    y_true : np.array
        True labels.
    y_probs : np.array
        Predicted probabilities.
    bad_rate : float, optional
        The maximum allowed false positive rate (bad rate), by default 0.25

    Returns
    -------
    float
        The threshold that satisfies the bad rate constraint.
    """
    # Sort the predicted probabilities in descending order
    sorted_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Calculate cumulative false positives and true positives
    fp = np.cumsum(1 - y_true_sorted)
    tp = np.cumsum(y_true_sorted)

    # Calculate false positive rate (FPR)
    fpr = fp / np.sum(1 - y_true_sorted) if np.sum(1 - y_true_sorted) > 0 else 0

    # Find the threshold that keeps FPR below bad_rate
    valid_thresholds = fpr <= bad_rate
    if not np.any(valid_thresholds):
        return 0.5  # Default threshold if no valid thresholds found

    # Return the highest threshold that satisfies the constraint
    max_valid_idx = np.max(np.where(valid_thresholds)[0])
    return y_probs[sorted_indices[max_valid_idx]]


# def calculate_kickout_metric(C1, C2, X_test_acp, y_test_acp, X_test_unl, acp_rate=0.5):
#     """
#     [Deprecated. Use pre_kickout and faster_kickout instead.] \\
#     Calculate the kickout metric for a reject inference model.

#     Parameters
#     ----------
#     C1 : sklearn.pipeline.Pipeline
#         The classifier to predict the accepts
#     C2 : sklearn.pipeline.Pipeline
#         The classifier to predict the accepts and rejects
#     X_test_acp : pd.DataFrame
#         The features of the accepts
#     y_test_acp : pd.Series
#         The target of the accepts
#     X_test_unl : pd.DataFrame
#         The features of the unlabeled samples
#     acp_rate : float, optional
#         The acceptance rate, by default 0.5

#     Returns
#     -------
#     float, int, int
#         The kickout metric, the number of kicked-out good samples, and the number of kicked-out bad samples
#     """

#     warnings.warn("This function is deprecated.", DeprecationWarning)

#     # Calculate predictions and obtain subsets A1_G and A1_B
#     num_acp_1 = int(len(X_test_acp) * acp_rate)  # number of Accepts
#     y_prob_1 = C1.predict_proba(X_test_acp)[:, 1]
#     threshold = np.percentile(y_prob_1, 100 - (num_acp_1 / len(y_prob_1)) * 100)
#     y_pred_1 = (y_prob_1 > threshold).astype("int")
#     A1 = X_test_acp[y_pred_1 == 0]
#     A1_G = X_test_acp[(y_pred_1 == 0) & (y_test_acp == 0)]
#     A1_B = X_test_acp[(y_pred_1 == 0) & (y_test_acp == 1)]

#     X_test_holdout = pd.concat([X_test_acp, X_test_unl])

#     # Calculate predictions on X_test_holdout and obtain subset A2
#     num_Acp_2 = int(len(X_test_holdout) * acp_rate)  # number of Accepts
#     y_prob_2 = C2.predict_proba(X_test_holdout)[:, 1]
#     threshold = np.percentile(y_prob_2, 100 - (num_Acp_2 / len(y_prob_2)) * 100)
#     y_pred_2 = (y_prob_2 > threshold).astype("int")
#     A2 = X_test_holdout[y_pred_2 == 0]

#     # Calculate indices of kicked-out good and bad samples
#     indices_KG = np.setdiff1d(A1_G.index, A2.index)
#     indices_KB = np.setdiff1d(A1_B.index, A2.index)

#     # Calculate the count of kicked-out good and bad samples
#     KG = A1_G.loc[indices_KG].shape[0]
#     KB = A1_B.loc[indices_KB].shape[0]

#     if KG == 0 and KB == 0:
#         return 0, 0, 0

#     # Calculate the share of bad cases in A1
#     p_B = (
#         (A1_B.shape[0] / A1.shape[0])
#         if (A1.shape[0] != 0 and A1_B.shape[0] != 0)
#         else 1e-8
#     )

#     if p_B == 1e-8 and KG > 0:
#         return -1, KG, KB

#     # Calculate the number of bad cases selected by the BM model
#     SB = A1_B.shape[0] if A1_B.shape[0] != 0 else 1e-8
#     # print('p_B, KG, KB, SB',p_B, KG, KB, SB)

#     kickout = ((KB / p_B) - (KG / (1 - p_B))) / (SB / p_B)

#     return kickout, KG, KB


# def pre_kickout(C1, C2, X_test_acp, X_test_unl):
#     """
#     Calculate the probabilities of the accepts and all samples on the reject inference model.

#     Parameters
#     ----------
#     C1 : object
#         The trained classifier for accepts.
#     C2 : object
#         The trained classifier for all samples.
#     X_test_acp : DataFrame
#         The feature matrix of the accepts samples.
#     X_test_unl : DataFrame
#         The feature matrix of the unlabeled samples.

#     Returns
#     -------
#     y_prob_acp : Series
#         The predicted probabilities of the accepts samples.
#     y_prob_all : Series
#         The predicted probabilities of all samples.

#     """

#     y_prob_acp = C1.predict_proba(X_test_acp)[:, 1]
#     X_test_holdout = pd.concat([X_test_acp, X_test_unl])
#     y_prob_all = C2.predict_proba(X_test_holdout)[:, 1]

#     y_prob_acp = pd.Series(y_prob_acp, index=X_test_acp.index)
#     y_prob_all = pd.Series(y_prob_all, index=X_test_holdout.index)
#     return y_prob_acp, y_prob_all


# def faster_kickout(y_test_acp, y_prob_acp, y_prob_all, acp_rate=0.5):
#     """
#     Calculate the kickout metric for a reject inference model.

#     Parameters
#     ----------
#     y_test_acp : array-like
#         The true labels for the acceptance samples in the ACP dataset.
#     y_prob_acp : array-like
#         The predicted probabilities for the acceptance samples in the ACP dataset.
#     y_prob_all : array-like
#         The predicted probabilities for all samples in the holdout dataset.
#     acp_rate : float, optional
#         The acceptance rate used to determine the threshold for acceptance predictions.
#         Default is 0.5.

#     Returns
#     -------
#     kickout : float
#         The kickout metric, which measures the difference between the number of bad cases
#         selected by the reject inference model and the number of good cases selected.
#     KG : int
#         The count of kicked-out good samples.
#     KB : int
#         The count of kicked-out bad samples.
#     """
#     # Calculate predictions and obtain subsets A1_G and A1_B
#     num_acp_1 = int(len(y_prob_acp) * acp_rate)  # number of Accepts
#     threshold = np.percentile(y_prob_acp, 100 - (num_acp_1 / len(y_prob_acp)) * 100)
#     y_pred_acp = (y_prob_acp > threshold).astype("int")
#     A1 = y_prob_acp[y_pred_acp == 0]
#     A1_G = y_prob_acp[(y_pred_acp == 0) & (y_test_acp == 0)]
#     A1_B = y_prob_acp[(y_pred_acp == 0) & (y_test_acp == 1)]

#     # X_test_holdout = pd.concat([X_test_acp, X_test_unl])

#     # Calculate predictions on X_test_holdout and obtain subset A2
#     num_Acp_2 = int(len(y_prob_all) * acp_rate)  # number of Accepts
#     threshold = np.percentile(y_prob_all, 100 - (num_Acp_2 / len(y_prob_all)) * 100)
#     y_pred_all = (y_prob_all > threshold).astype("int")
#     A2 = y_prob_all[y_pred_all == 0]

#     # Calculate indices of kicked-out good and bad samples
#     indices_KG = np.setdiff1d(A1_G.index, A2.index)
#     indices_KB = np.setdiff1d(A1_B.index, A2.index)

#     # Calculate the count of kicked-out good and bad samples
#     KG = A1_G.loc[indices_KG].shape[0]
#     KB = A1_B.loc[indices_KB].shape[0]

#     # if not any good or bad cases were kicked out
#     if KG == 0 and KB == 0:
#         return 0, 0, 0

#     # Calculate the share of bad cases in A1
#     p_B = (
#         (A1_B.shape[0] / A1.shape[0])
#         if (A1.shape[0] != 0 and A1_B.shape[0] != 0)
#         else 1e-8
#     )

#     # if good cases were kicked out but no bad cases were kicked out when there were no bad cases in A1
#     if p_B == 1e-8 and KG > 0:
#         return -1, KG, KB

#     # Calculate the number of bad cases selected by the BM model
#     SB = A1_B.shape[0] if A1_B.shape[0] != 0 else 1e-8

#     # Calculate the kickout metric using the formula
#     kickout = ((KB / p_B) - (KG / (1 - p_B))) / (SB / p_B)

#     return kickout, KG, KB


# # TBD
# def get_metrics_RI(
#     name_model_dict,
#     X,
#     y,
#     X_v=None,
#     y_v=None,
#     X_unl=None,
#     threshold_type="ks",
#     acp_rate=0.5,
# ):
#     """
#     Calculate the metrics for the reject inference models.

#     Parameters
#     ----------
#     name_model_dict : dict
#         A dictionary containing the names and the models of the reject inference models.
#     X : pd.DataFrame
#         The feature matrix of the test set.
#     y : pd.Series
#         The true labels of the test set.
#     X_v : pd.DataFrame, optional
#         The feature matrix of the validation set. Default is None.
#     y_v : pd.Series, optional
#         The true labels of the validation set. Default is None.
#     X_unl : pd.DataFrame, optional
#         The feature matrix of the unlabeled set. Default is None.
#     threshold_type : str, optional
#         The type of threshold to use. Default is 'ks'.
#     acp_rate : float, optional
#         The acceptance rate. Default is 0.5.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing the metrics of the reject inference models.
#     """

#     def get_best_threshold_with_ks(model, X, y):
#         """
#         Calculate the best threshold based on the KS statistic.

#         Parameters
#         ----------
#         model : sklearn.pipeline.Pipeline
#             The trained risk score model.
#         X : pd.DataFrame
#             The feature matrix of the validation set.
#         y : pd.Series
#             The true labels of the validation set.

#         Returns
#         -------
#         float
#             The best threshold for the risk score model.
#         """
#         y_probs = model.predict_proba(X)[:, 1]
#         fpr, tpr, thresholds = roc_curve(y, y_probs)
#         return thresholds[np.argmax(tpr - fpr)]

#     models_dict = {}
#     for name, model in name_model_dict.items():
#         if isinstance(model, list):
#             y_prob = model[0].predict_proba(X)[:, 1]
#             threshold_model = model[1]
#             y_pred = (y_prob >= threshold_model).astype("int")
#         else:
#             if threshold_type == "default":
#                 threshold = 0.5
#             elif threshold_type == "ks":
#                 if np.any(X_v):
#                     threshold = get_best_threshold_with_ks(model, X_v, y_v)
#                 else:
#                     threshold = get_best_threshold_with_ks(model, X, y)
#             elif threshold_type == "risk":
#                 if np.any(X_v):
#                     threshold = risk_score_threshold(model, X_v, y_v)
#                 else:
#                     threshold = risk_score_threshold(model, X, y)
#             else:
#                 threshold = 0.5

#             y_prob = model.predict_proba(X)[:, 1]
#             y_pred = (y_prob >= threshold).astype("int")

#         models_dict[name] = (y_pred, y_prob)

#     def evaluate_ks(y_real, y_proba):
#         """
#         Calculate the Kolmogorov-Smirnov statistic.

#         Parameters
#         ----------
#         y_real : array-like
#             The true labels.
#         y_proba : array-like
#             The predicted probabilities.

#         Returns
#         -------
#         float
#             The Kolmogorov-Smirnov statistic
#         """
#         ks = ks_2samp(y_proba[y_real == 0], y_proba[y_real == 1])
#         return ks.statistic

#     def get_metrics_df(models_dict, y_true, use_threshold):
#         """
#         Calculate the metrics for the reject inference models.

#         Parameters
#         ----------
#         models_dict : dict
#             A dictionary containing the names and the models of the reject inference models.
#         y_true : pd.Series
#             The true labels of the test set.
#         use_threshold : bool
#             If True, use the threshold to calculate the metrics.

#         Returns
#         -------
#         dict
#             A dictionary containing the metrics of the reject inference.
#         """
#         if use_threshold:
#             metrics_dict = {
#                 "AUC": (lambda x: roc_auc_score(y_true, x), False),
#                 "KS": (lambda x: evaluate_ks(y_true, x), False),
#                 "Balanced_Accuracy": (
#                     lambda x: balanced_accuracy_score(y_true, x),
#                     True,
#                 ),
#                 "Accuracy": (lambda x: accuracy_score(y_true, x), True),
#                 "Precision": (lambda x: precision_score(y_true, x), True),
#                 "Recall": (lambda x: recall_score(y_true, x), True),
#                 "F1": (lambda x: f1_score(y_true, x), True),
#             }
#         else:
#             metrics_dict = {
#                 "AUC": (lambda x: roc_auc_score(y_true, x), False),
#                 "KS": (lambda x: evaluate_ks(y_true, x), False),
#             }
#         df_dict = {}
#         for metric_name, (metric_func, use_preds) in metrics_dict.items():
#             df_dict[metric_name] = [
#                 metric_func(preds) if use_preds else metric_func(scores)
#                 for model_name, (preds, scores) in models_dict.items()
#             ]
#         return df_dict

#     if threshold_type != "none":
#         use_threshold = True
#     else:
#         use_threshold = False

#     df_dict = get_metrics_df(models_dict, y, use_threshold)

#     if np.any(X_v):
#         df_dict["Approval_Rate"] = []
#     if np.any(X_unl) and "BM" in name_model_dict:
#         df_dict["Kickout"] = []
#         df_dict["KG"] = []
#         df_dict["KB"] = []

#     for name, model in name_model_dict.items():
#         if name != "BM":
#             if isinstance(model, list):
#                 if np.any(X_v):
#                     a_r = calculate_approval_rate(model[0], X_v, y_v, X)
#                     # acp_rate = a_r
#                     df_dict["Approval_Rate"].append(a_r)
#                 if np.any(X_unl) and "BM" in name_model_dict:
#                     kickout, kg, kb = calculate_kickout_metric(
#                         name_model_dict["BM"][0], model[0], X, y, X_unl, acp_rate
#                     )
#                     df_dict["Kickout"].append(kickout)
#                     df_dict["KG"].append(kg)
#                     df_dict["KB"].append(kb)
#             else:
#                 if np.any(X_v):
#                     a_r = calculate_approval_rate(model, X_v, y_v, X)
#                     # acp_rate = a_r
#                     df_dict["Approval_Rate"].append(a_r)
#                 if np.any(X_unl) and "BM" in name_model_dict:
#                     if isinstance(name_model_dict["BM"], list):
#                         benchmark = name_model_dict["BM"][0]  # Assuming "BM" is a list
#                     else:
#                         benchmark = name_model_dict["BM"]

#                     kickout, kg, kb = calculate_kickout_metric(
#                         benchmark, model, X, y, X_unl, acp_rate
#                     )
#                     df_dict["Kickout"].append(kickout)
#                     df_dict["KG"].append(kg)
#                     df_dict["KB"].append(kb)
#         else:
#             if np.any(X_v):
#                 if isinstance(model, list):
#                     a_r = calculate_approval_rate(model[0], X_v, y_v, X)
#                     df_dict["Approval_Rate"].append(a_r)
#                 else:
#                     a_r = calculate_approval_rate(model, X_v, y_v, X)
#                     df_dict["Approval_Rate"].append(a_r)
#             if np.any(X_unl) and "BM" in name_model_dict:
#                 df_dict["Kickout"].append(0)
#                 df_dict["KG"].append(0)
#                 df_dict["KB"].append(0)

#     metrics_df = pd.DataFrame.from_dict(
#         df_dict, orient="index", columns=models_dict.keys()
#     )

#     return metrics_df


# # ---------Other Strategies----------


# def augmentation_with_soft_cutoff(X_train, y_train, X_unl, seed=seed_number):
#     """[Augmentation with Soft Cutoff] (Siddiqi, 2012)

#     Parameters
#     ----------
#     X_train : _type_
#         _description_
#     y_train : _type_
#         _description_
#     X_unl : _type_
#         _description_
#     """
#     params_dict["LightGBM_2"].update({"random_state": seed})

#     # --------------Get Data----------------
#     # Create dataset based on Approved(1)/Decline(0) condition
#     X_aug_train = pd.concat([X_train, X_unl])

#     train_y = np.ones(X_train.shape[0])  # Approved gets 1
#     unl_y = np.zeros(X_unl.shape[0])  # Rejected get 0

#     # Concat train_y and unl_y
#     y_aug_train = pd.Series(np.concatenate([train_y, unl_y]), index=X_aug_train.index)

#     # --------------Get Weights----------------
#     classifier_AR = tr.create_pipeline(
#         X_aug_train, y_aug_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     classifier_AR.fit(X_aug_train, y_aug_train)

#     # Get the probabilitie of being approved
#     prob_A = classifier_AR.predict_proba(X_aug_train)[:, 1]
#     prob_A_series = pd.Series(prob_A, index=X_aug_train.index)

#     n_scores_interv = 100

#     # Sort the probabilities of being accepted
#     asc_prob_A = np.argsort(prob_A)
#     asc_prob_A_series = prob_A_series.iloc[asc_prob_A]
#     # #Split the probs in intervals
#     score_interv = np.array_split(asc_prob_A_series, n_scores_interv)

#     # Create array for accepts weights
#     weights_SC = y_train.copy()
#     for s in score_interv:
#         # Get index of accepts in s
#         acceptees = np.intersect1d(s.index, weights_SC.index)
#         if len(acceptees) >= 1:
#             # Augmentation Factor (Weight) for the split
#             AF = y_aug_train.loc[s.index].mean()  # A/(A+R)
#             AF_split = np.power(AF, -1)
#             weights_SC.loc[acceptees] = AF_split

#     ##--------------Fit classifier----------------
#     augmentation_classifier_SC = tr.create_pipeline(
#         X_train, y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     augmentation_classifier_SC.fit(
#         X_train, y_train, classifier__sample_weight=weights_SC
#     )

#     return {"A-SC": augmentation_classifier_SC}


# def augmentation(X_train, y_train, X_unl, mode="up", seed=seed_number):
#     """[Augmentation,Reweighting] (Anderson, 2022), (Siddiqi, 2012)

#     Parameters
#     ----------
#     X_train : _type_
#         _description_
#     y_train : _type_
#         _description_
#     X_unl : _type_
#         _description_
#     mode : str, optional
#         _description_, by default 'up'
#     """
#     params_dict["LightGBM_2"].update({"random_state": seed})
#     # --------------Get Data----------------
#     # Create dataset based on Approved(1)/Decline(0) condition
#     X_aug_train = pd.concat([X_train, X_unl])

#     train_y = np.ones(X_train.shape[0])  # Approved gets 1
#     unl_y = np.zeros(X_unl.shape[0])  # Rejected get 0

#     # Concat train_y and unl_y
#     y_aug_train = pd.Series(np.concatenate([train_y, unl_y]), index=X_aug_train.index)

#     # --------------Get Weights----------------
#     # weight_classifier = tr.create_pipeline(X_aug_train, y_aug_train, LGBMClassifier(**params_dict['LightGBM_2']))
#     weight_classifier = tr.create_pipeline(
#         X_aug_train, y_aug_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     weight_classifier.fit(X_aug_train, y_aug_train)

#     # Weights are the probabilitie of being approved
#     weights = weight_classifier.predict_proba(X_aug_train)[:, 1]

#     # Upward: ŵ = w/p(A)
#     acp_weights_up = 1 / weights[: X_train.shape[0]]

#     # Downward: ŵ = w * (1 - p(A))
#     acp_weights_down = 1 * (1 - weights[: X_train.shape[0]])
#     ##--------------Fit classifier----------------
#     if mode == "up":
#         augmentation_classifier_up = tr.create_pipeline(
#             X_train, y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#         )
#         augmentation_classifier_up.fit(
#             X_train, y_train, classifier__sample_weight=acp_weights_up
#         )

#         return {"A-UW": augmentation_classifier_up}
#     elif mode == "down":
#         augmentation_classifier_down = tr.create_pipeline(
#             X_train, y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#         )
#         augmentation_classifier_down.fit(
#             X_train, y_train, classifier__sample_weight=acp_weights_down
#         )

#         return {"A-DW": augmentation_classifier_down}


# def fuzzy_augmentation(X_train, y_train, X_unl, seed=seed_number):
#     """[Fuzzy-Parcelling](Anderson, 2022)

#     Parameters
#     ----------
#     X_train : _type_
#         _description_
#     y_train : _type_
#         _description_w
#     X_unl : _type_
#         _description_
#     """
#     params_dict["LightGBM_2"].update({"random_state": seed})

#     # --------------Get Dataset----------------
#     X_fuzzy_train = pd.concat([X_train, X_unl, X_unl])
#     X_fuzzy_train.index = range(X_fuzzy_train.shape[0])
#     good_y = np.zeros(X_unl.shape[0])
#     bad_y = np.ones(X_unl.shape[0])

#     y_fuzzy_rej = pd.Series(np.concatenate([good_y, bad_y]))
#     y_fuzzy_train = pd.concat([y_train, y_fuzzy_rej])
#     y_fuzzy_train.index = range(X_fuzzy_train.shape[0])

#     # --------------Get Weights----------------
#     weight_clf = tr.create_pipeline(
#         X_train, y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     weight_clf.fit(X_train, y_train)
#     unl_0_weights = weight_clf.predict_proba(X_unl)[:, 0]
#     unl_1_weights = weight_clf.predict_proba(X_unl)[:, 1]

#     train_weights = np.ones(y_train.shape[0])

#     fuzzy_weights = np.concatenate([train_weights, unl_0_weights, unl_1_weights])

#     # --------------Fit classifier----------------
#     fuzzy_classifier = tr.create_pipeline(
#         X_fuzzy_train, y_fuzzy_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     fuzzy_classifier.fit(
#         X_fuzzy_train, y_fuzzy_train, classifier__sample_weight=fuzzy_weights
#     )

#     return {"A-FU": fuzzy_classifier}


# def extrapolation(X_train, y_train, X_unl, mode="C", seed=seed_number):
#     """[extrapolation, hard cutoff, Simple Augmentation] (Siddiqi, 2012)

#     Parameters
#     ----------
#     X_train : _type_
#         _description_
#     y_train : _type_
#         _description_
#     X_unl : _type_
#         _description_
#     mode : str, optional
#         _description_, by default "bad"
#     """

#     params_dict["LightGBM_2"].update({"random_state": seed})
#     # --------------Fit classifier----------------
#     # Fit classifier on Accepts Performance
#     default_classifier = tr.create_pipeline(
#         X_train, y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     default_classifier.fit(X_train, y_train)

#     y_prob_unl = default_classifier.predict_proba(X_unl)[:, 1]

#     y_label_unl = (y_prob_unl >= 0.5).astype(int)
#     y_label_unl_s = pd.Series(y_label_unl, index=X_unl.index)

#     # --------------Create new Dataset----------------
#     if mode == "B":
#         new_X_train = pd.concat([X_train, X_unl[y_label_unl == 1]])
#         new_y_train = pd.concat([y_train, y_label_unl_s[y_label_unl == 1]])
#     elif mode == "A":
#         new_X_train = pd.concat([X_train, X_unl])
#         new_y_train = pd.concat([y_train, y_label_unl_s])
#     elif mode == "C":
#         new_X_train = pd.concat(
#             [X_train, X_unl[y_prob_unl > 0.8], X_unl[y_prob_unl < 0.15]]
#         )
#         new_y_train = pd.concat(
#             [y_train, y_label_unl_s[y_prob_unl > 0.8], y_label_unl_s[y_prob_unl < 0.15]]
#         )
#         print(f"new_X_train shape: {new_X_train.shape[0]}")
#         print(f"old_X_train shape: {X_train.shape[0]}")
#     else:
#         return {}

#     # --------------Fit classifier----------------
#     extrap_classifier = tr.create_pipeline(
#         new_X_train, new_y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     extrap_classifier.fit(new_X_train, new_y_train)

#     return {"E-" + mode: extrap_classifier}


# def label_spreading(X_train, y_train, X_unl, return_labels=False, seed=seed_number):
#     """[Label Spreading] (Zhou, 2004)(Kang, 2021)

#     Parameters
#     ----------
#     X_train : _type_
#         _description_
#     y_train : _type_
#         _description_
#     X_unl : _type_
#         _description_
#     """

#     params_dict["LightGBM_2"].update({"random_state": seed})
#     # --------------Create dataset with Approved and Rejected---------------
#     X_train_ls, y_train_ls, X_unl_ls = X_train.copy(), y_train.copy(), X_unl.copy()

#     X_combined = pd.concat([X_train_ls, X_unl_ls])

#     y_unl_ls = np.array([-1] * X_unl_ls.shape[0])
#     y_combined = np.concatenate([y_train_ls.array, y_unl_ls])
#     y_combined = pd.Series(y_combined, index=X_combined.index)

#     # --------------Predict labels on the unlabeled data---------------
#     lp_model = tr.create_pipeline(
#         X_combined, y_combined, LabelSpreading(**params_dict["LabelSpreading_2"])
#     )
#     lp_model.fit(X_combined, y_combined)
#     predicted_labels = lp_model["classifier"].transduction_[y_combined == -1]

#     y_label_pred_s = pd.Series(predicted_labels, index=X_unl_ls.index)

#     # --------------Fit classifier---------------
#     # Create a new classifier pipeline using labeled and unlabeled data, and fit it
#     new_X_train = pd.concat([X_train_ls, X_unl_ls])
#     new_y_train = pd.concat([y_train_ls, y_label_pred_s])

#     clf_LS = tr.create_pipeline(
#         new_X_train, new_y_train, LGBMClassifier(**params_dict["LightGBM_2"])
#     )
#     clf_LS.fit(
#         new_X_train,
#         new_y_train,
#     )

#     if return_labels:
#         return lp_model["classifier"].label_distributions_[y_combined == -1]

#     return {"LSP": clf_LS}
