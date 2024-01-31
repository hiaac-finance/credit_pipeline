import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


def plot_confusion_matrix(y_true, y_pred):
    """
    Compute and plot confusion matrix given true values and predictions.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy of the model is: ", accuracy_score(y_true, y_pred))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_prob_distribution(y_true, y_prob):
    """
    Compute and distributions of probs by target given true values and probabilities.
    """
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label="1", density=True)
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label="0", density=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.legend(loc="upper center")
    plt.show()


def plot_prob_distribution_from_list(y_true, y_prob_list):
    """
    Compute and distributions of probs by target given
    true values and a list of probabilities.
    """
    num_plots = len(y_prob_list)
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for i, y_prob in enumerate(y_prob_list):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1 and num_cols == 1:
            axs.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label="1", density=True)
            axs.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label="0", density=True)
            axs.set_xlabel("Predicted probability")
            axs.set_ylabel("Frequency")
            axs.legend(loc="upper center")
        else:
            axs[row, col].hist(
                y_prob[y_true == 1], bins=20, alpha=0.5, label="1", density=True
            )
            axs[row, col].hist(
                y_prob[y_true == 0], bins=20, alpha=0.5, label="0", density=True
            )
            axs[row, col].set_xlabel("Predicted probability")
            axs[row, col].set_ylabel("Frequency")
            axs[row, col].legend(loc="upper center")

    # Hide unused subplots
    for i in range(len(y_prob_list), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        if num_rows == 1 and num_cols == 1:
            axs.axis("off")
        else:
            axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def false_positive_rate(y_true, y_pred):
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


def demographic_parity(y_true, y_pred, z):
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


def equal_opportunity(y_true, y_pred, z):
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


def average_odds(y_true, y_pred, z):
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


def average_precision_value_difference(y_true, y_pred, z):
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


def geometric_mean_accuracy(y_true, y_pred, z):
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

def create_eod_scorer(z, benefit_class=1):
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

def create_fairness_scorer(fairness_goal, z, M = 10, benefit_class=1):
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
            return roc_auc_score(y_true, y_score_pred) - M * abs(fairness_score - fairness_goal)
    return fairness_scorer


def get_metrics(name_model_dict, X, y, threshold=0.5):
    """Calculate metrics for a set of models. The metrics are returned in a dataframe. The input must be a dict with model names, and the values can be a model or a tuple with model and threshold. If the threshold is not provided, the default value is 0.5.

    :param name_model_dict:
    :type name_model_dict: dict
    :param X: model features
    :type X: array-like
    :param y: ground truth labels
    :type y: array-like
    :param threshold: threshold used in all models with model-specific threshold is not provided, defaults to 0.5
    :type threshold: float, optional
    :return: dataframe with columns as metrics and rows as models
    :rtype: pandas.DataFrame
    """
    models_dict = {}
    for name, model in name_model_dict.items():
        if type(model) == list:
            model_ = model[0]
            threshold_ = model[1]
        else:
            model_ = model
            threshold_ = threshold

        if hasattr(model[0], "predict_proba"):
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
            "AUC" : lambda x: roc_auc_score(y_true, x) if x is not None else None,
            "Brier Score" : lambda x: brier_score_loss(y_true, x) if x is not None else None,
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


def get_fairness_metrics(name_model_dict, X, y, z, threshold=0.5, benefit_class=1):
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
    """
    models_dict = {}
    for name, model in name_model_dict.items():
        if type(model) == list:
            model_ = model[0]
        else:
            model_ = model
        
        y_prob = None
        y_pred = model_.predict(X)
        y_pred = (y_pred == benefit_class).astype("float")
            
        models_dict[name] = (y_pred, y_prob)

    # transform y to array if it is a pandas series
    if type(y) == pd.Series:
        y = y.values
    if type(z) == pd.Series:
        z = z.values
    y_true = (y == benefit_class).astype("float")

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
                metric_func(preds) for (preds, _) in models_dict.values()
            ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    metrics = get_metrics_df(models_dict)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics
