import numpy as np
import pandas as pd
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


def false_positive_rate(y_true, y_pred):
    """Calculate false positive rate.

    :param y_true: real labels
    :type y_true: array-like
    :param y_pred: prediction labels
    :type y_pred: array-like
    :return: false positive rate
    :rtype: float
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def get_metrics(name_model_dict, X, y, threshold=0.5):
    """Calculate metrics for a set of models. The metrics are returned in a dataframe. The input must be a dict with model names, and the values can be a model or a tuple with model and threshold. If the threshold is not provided, the default value is 0.5.

    :param name_model_dict: dict with model names as keys and model or tuple with model and threshold as values
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
            y_prob = model[0].predict_proba(X)[:, 1]
            threshold_model = model[1]
            y_pred = (y_prob >= threshold_model).astype("int")
        else:
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= threshold).astype("int")

        models_dict[name] = (y_pred, y_prob)

    def get_metrics_df(
        models_dict,
        y_true,
    ):
        metrics_dict = {
            "AUC": (lambda x: roc_auc_score(y_true, x), False),
            "Balanced Accuracy": (lambda x: balanced_accuracy_score(y_true, x), True),
            "Accuracy": (lambda x: accuracy_score(y_true, x), True),
            "Precision": (lambda x: precision_score(y_true, x, zero_division=0), True),
            "Recall": (lambda x: recall_score(y_true, x), True),
            "F1": (lambda x: f1_score(y_true, x), True),
            "Brier Score": (lambda x: brier_score_loss(y_true, x), False),
        }
        df_dict = {}
        for metric_name, (metric_func, use_preds) in metrics_dict.items():
            df_dict[metric_name] = [
                metric_func(preds) if use_preds else metric_func(scores)
                for model_name, (preds, scores) in models_dict.items()
            ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    metrics = get_metrics_df(models_dict, y)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics


def get_fairness_metrics(models_dict, y, z, benefit_class=1):
    """Calculate fairness metrics for a set of models. The metrics are returned in a dataframe. The input must be a dict with model names, and the values are the model predictions.

    :param models_dict: dict with model names as keys and model predictions as values
    :type models_dict: dict
    :param y: ground truth labels
    :type y: array-like
    :param z: sensitive attribute
    :type z: array-like
    :param benefit_class: benifit class of labels, defaults to 1
    :type benefit_class: int, optional
    :return: dataframe with columns as metrics and rows as models
    :rtype: pandas.DataFrame
    """
    models_dict_benefit = {}
    for name, y_pred in models_dict.items():
        models_dict_benefit[name] = (
            (y == benefit_class).astype("float"),
            (y_pred == benefit_class).astype("float"),
        )

    def get_metrics_df(models_dict_benefit):
        df_dict = {}
        df_dict["DPD"] = [
            np.mean(preds[z == 1]) - np.mean(preds[z == 0])
            for ground, preds in models_dict_benefit.values()
        ]
        df_dict["EOD"] = [
            recall_score(ground[z == 1], preds[z == 1])
            - recall_score(ground[z == 0], preds[z == 0])
            for ground, preds in models_dict_benefit.values()
        ]
        df_dict["AOD"] = [
            0.5
            * (
                recall_score(ground[z == 1], preds[z == 1])
                - recall_score(ground[z == 0], preds[z == 0])
            )
            + 0.5
            * (
                false_positive_rate(ground[z == 1], preds[z == 1])
                - false_positive_rate(ground[z == 0], preds[z == 0])
            )
            for ground, preds in models_dict_benefit.values()
        ]
        df_dict["APVD"] = [
            0.5
            * (
                recall_score(preds[z == 1], ground[z == 1])
                - recall_score(preds[z == 0], ground[z == 0])
            )
            + 0.5
            * (
                false_positive_rate(preds[z == 1], ground[z == 1])
                - false_positive_rate(preds[z == 0], ground[z == 0])
            )
            for ground, preds in models_dict_benefit.values()
        ]
        df_dict["GMA"] = [
            np.sqrt(
                accuracy_score(ground[z == 1], preds[z == 1])
                * accuracy_score(ground[z == 0], preds[z == 0])
            )
            for ground, preds in models_dict_benefit.values()
        ]
        df_dict["balanced_accuracy"] = [
            balanced_accuracy_score(ground, preds)
            for ground, preds in models_dict_benefit.values()
        ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict_benefit.keys()
        ).T

    metrics = get_metrics_df(models_dict_benefit)
    metrics = metrics.reset_index().rename(columns={"index": "model"})
    return metrics
