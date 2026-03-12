from typing import Dict, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


def false_positive_rate(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate false positive rate. The y_true and y_pred must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        False positive rate.
    """
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn)


def demographic_parity(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate demographic parity. The y_true and y_pred must be 0 or 1, with 1 being the benefit class.
    Z is the sensitive attribute and also must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels, they are not used in the calculation.
    y_pred : array-like
        Predicted labels.
    z : array-like
        Sensitive attribute.
    """
    return np.mean(y_pred[z == 1]) - np.mean(y_pred[z == 0])


def equal_opportunity(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
):
    """Calculate equal opportunity. The y_true and y_pred must be 0 or 1, with 1 being the benefit class.
    Z is the sensitive attribute and also must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels.
    y_pred : array-like
        Predicted labels.
    z : array-like
        Sensitive attribute.
    """
    return recall_score(y_true[z == 1], y_pred[z == 1]) - recall_score(
        y_true[z == 0], y_pred[z == 0]
    )


def average_odds(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate average odds fairness metric. The y_true and y_pred must be 0 or 1, with 1 being the benefit class.
    Z is the sensitive attribute and also must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels.
    y_pred : array-like
        Predicted labels.
    z : array-like
        Sensitive attribute.
    """
    term1 = equal_opportunity(y_true, y_pred, z)
    term2 = false_positive_rate(y_true[z == 1], y_pred[z == 1]) - false_positive_rate(
        y_true[z == 0], y_pred[z == 0]
    )
    return 0.5 * (term1 + term2)


def average_precision_value_difference(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate average precision value fairness metric. The y_true and y_pred must be 0 or 1, with 1 being the benefit class.
    Z is the sensitive attribute and also must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels.
    y_pred : array-like
        Predicted labels.
    z : array-like
        Sensitive attribute.
    """
    return average_odds(y_pred, y_true, z)  # Note that y_pred and y_true are swapped


def geometric_mean_accuracy(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Calculate geometric mean accuracy fairness metric. The y_true and y_pred must be 0 or 1, with 1 being the benefit class.
    Z is the sensitive attribute and also must be 0 or 1.

    Parameters
    ----------
    y_true : array-like
        Real labels.
    y_pred : array-like
        Predicted labels.
    z : array-like
        Sensitive attribute.
    """
    return np.sqrt(
        accuracy_score(y_true[z == 1], y_pred[z == 1])
        * accuracy_score(y_true[z == 0], y_pred[z == 0])
    )


def kickout(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_base: Union[np.ndarray, pd.Series],
    y_pred_rej: Union[np.ndarray, pd.Series],
):

    # Wrong predictions by the base model
    wrong = (y_true == 1) & (y_pred_base == 0)
    # Correct predictions by the reject model
    kb = np.mean(y_pred_rej[wrong])

    # Correct predictions by the base model
    correct = (y_true == 0) & (y_pred_base == 0)
    # Wrong predictions by the reject model
    kg = np.mean(y_pred_rej[correct])

    return kb - kg


def create_eod_scorer(z: Union[np.ndarray, pd.Series], benefit_class: int = 1):
    """Create a scorer for equal opportunity difference. The scorer can be used in hyperparameter tuning.

    Parameters
    ----------
    z : array-like
        Sensitive attribute to be used in the scorer, it must be in the same order as the data used to generate the prediction.
    benefit_class : int, optional
        Label of positive class to calculate metric, by default 1
    """

    def eod_scorer(y_true, y_pred):
        y_true_ = (y_true == benefit_class).astype("float")
        y_pred_ = (y_pred == benefit_class).astype("float")
        return equal_opportunity(y_true_, y_pred_, z)

    return eod_scorer


def create_fairness_scorer(
    fairness_goal: float,
    z: Union[np.ndarray, pd.Series],
    M: int = 10,
    benefit_class: int = 1,
):
    """Create a scorer for fairness metrics. The scorer can be used in hyperparameter tuning.

    It will return the value of the roc auc if the fairness goal is reached, otherwise it will return a low value.

    Parameters
    ----------
    fairness_goal : float
        Value of the fairness metric to be reached. The lower the value, the more fair the model.
    z : array-like
        Sensitive attribute to be used in the scorer, it must be in the same order as the data used to generate the prediction.
    M : int, optional
        Penalty for not reaching the fairness goal, by default 10
    benefit_class : int, optional
        Label of positive class to calculate metric, by default 1
    """

    def fairness_scorer(y_true, y_pred, y_score_pred):
        y_true_ = (y_true == benefit_class).astype("float")
        y_pred_ = (y_pred == benefit_class).astype("float")
        fairness_score = np.abs(equal_opportunity(y_true_, y_pred_, z))
        if fairness_score <= fairness_goal:
            return roc_auc_score(y_true, y_score_pred)
        else:
            return roc_auc_score(y_true, y_score_pred) - M * abs(
                fairness_score - fairness_goal
            )

    return fairness_scorer


def get_metrics(
    name_model_dict: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Calculate metrics for a set of models. The metrics are returned in a dataframe. The input must be a dict with model names, and the values can be a model or a tuple with model and threshold. If the threshold is not provided, the default value is 0.5.


    Parameters
    ----------
    name_model_dict : Dict[str, Any]
        Dict with model names and values must be either the model or a list with the model and threshold
    X : np.ndarray
        Features matrix
    y : np.ndarray
        True labels
    threshold : float, optional
        Threshold to be used for all models, if individual thresholds are not provided, by default 0.5

    Returns
    -------
    pd.DataFrame
        Dataframe with metrics for each model.
    """
    models_dict = {}
    for name, model in name_model_dict.items():
        if type(model) == list:
            model_ = model[0]
            threshold_ = model[1]
        else:
            model_ = model
            threshold_ = threshold

        if hasattr(model_, "predict_proba"):
            y_prob = model_.predict_proba(X)[:, 1]
            y_pred = (y_prob >= threshold_).astype("int")
        else:
            y_prob = None
            y_pred = model_.predict(X)

        models_dict[name] = (y_pred, y_prob)

    def get_metrics_df(
        models_dict,
        y_true,
    ):
        metrics_score_dict = {
            "AUC": lambda x: roc_auc_score(y_true, x) if x is not None else None,
            "Brier Score": lambda x: (
                brier_score_loss(y_true, x) if x is not None else None
            ),
        }
        metrics_dict = {
            "Balanced Accuracy": lambda x: balanced_accuracy_score(y_true, x),
            "Accuracy": lambda x: accuracy_score(y_true, x),
            "Precision": lambda x: precision_score(y_true, x, zero_division=0),
            "Recall": lambda x: recall_score(y_true, x),
            "F1": lambda x: f1_score(y_true, x),
        }
        df_dict = {}
        for metric_name, metric_func in metrics_score_dict.items():
            df_dict[metric_name] = [
                metric_func(scores) for (_, scores) in models_dict.values()
            ]
        for metric_name, metric_func in metrics_dict.items():
            df_dict[metric_name] = [
                metric_func(preds) for (preds, _) in models_dict.values()
            ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    metrics = get_metrics_df(models_dict, y)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics


def get_fairness_metrics(
    name_model_dict: Dict[str, Any],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    z: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
    benefit_class: int = 1,
):
    """Calculate fairness metrics for a set of models. The metrics are returned in a dataframe.
    The input must be a dict with model names, and the values are the model predictions.

    Parameters
    ----------
    name_model_dict : dict
        Dict with model names as keys and model or tuple with model and threshold as values.
    X : array-like
        Model features.
    y : array-like
        Ground truth labels.
    z : array-like
        Protected attribute.
    benefit_class : int
        Value that indicates the benefit class.
    threshold : float, optional
        Threshold to be used for all models, if individual thresholds are not provided, by default 0.5

    Returns
    -------
    pd.DataFrame
        Dataframe with fairness metrics for each model.
    """
    models_dict = {}
    for model_name, pred in name_model_dict.items():
        models_dict[model_name] = (pred == benefit_class).astype("int")

    # transform y to array if it is a pandas series
    if type(y) == pd.Series:
        y = y.values
    if type(z) == pd.Series:
        z = z.values
    y_true = (y == benefit_class).astype("int")

    def get_metrics_df(models_dict):
        metrics_dict = {
            "DPD": lambda x: demographic_parity(y_true, x, z),
            "EOD": lambda x: equal_opportunity(y_true, x, z),
            "AOD": lambda x: average_odds(y_true, x, z),
            "APVD": lambda x: average_precision_value_difference(y_true, x, z),
            "GMA": lambda x: geometric_mean_accuracy(y_true, x, z),
            "balanced_accuracy": lambda x: balanced_accuracy_score(y_true, x),
        }

        df_dict = {}
        for metric_name, metric_func in metrics_dict.items():
            df_dict[metric_name] = [
                metric_func(preds) for preds in models_dict.values()
            ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    metrics = get_metrics_df(models_dict)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics


def get_threshold_bad_rate(
    y_true: np.array, y_probs: np.array, bad_rate: float = 0.25
) -> float:
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


def get_reject_inference_metrics(
    name_model_dict: Dict[str, Tuple[np.array, np.array]],
    y: np.array,
    bad_rate: float = 0.5,
):
    """Calculate metrics for reject inference methods in comparison to a base model.

    Parameters
    ----------
    name_model_dict : Dict[str, Tuple[np.array, np.array]]
        Dict with model names as keys and tuples of (predicted probabilities for labeled data, predicted probabilities for unlabeled data) as values. The base model must be included in the dict with the key "base".
    y : np.array
        True labels.
    bad_rate : float, optional
        The maximum allowed false positive rate (bad rate), by default 0.5

    Returns
    -------
    _type_
        _description_
    """
    assert (
        "base" in name_model_dict
    ), "Base model must be provided in the input dict with key 'base'"

    # Get the predictions for the base model of the labeled population
    y_probs_base, _ = name_model_dict["base"]
    base_threshold = get_threshold_bad_rate(y, y_probs_base, bad_rate=bad_rate)
    y_pred_base = (y_probs_base >= base_threshold).astype("int")

    # Then, calculate the kickout metric for each model
    metrics = []
    for model_name, model in name_model_dict.items():
        # model contains the predictions for labeled and unlabeled data
        y_probs, y_probs_unl = model
        threshold = get_threshold_bad_rate(y, y_probs, bad_rate=bad_rate)
        y_pred = (y_probs >= threshold).astype("int")
        y_pred_unl = (y_probs_unl >= threshold).astype("int")

        metrics.append(
            {
                "model": model_name,
                "approval_rate": np.mean(np.concatenate([y_pred, y_pred_unl])),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "kickout": kickout(y, y_pred_base, y_pred),
            }
        )

    metrics = pd.DataFrame(metrics)
    return metrics
