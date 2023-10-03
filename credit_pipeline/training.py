import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
)
from optuna.samplers import TPESampler
import optuna

import credit_pipeline.data_exploration as dex


def false_positive_rate(y_true, y_pred):
    """Calculate false positive rate.

    :param y_true: real labels
    :param y_pred: prediction labels
    :return: float in [0,1] with false positive rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def need_EBE(dataframe, cat_cols, crit=3):
    need_ebe = []
    dont_ebe = []
    for c in cat_cols:
        if len(dataframe[c].unique()) > crit:
            need_ebe.append(c)
        else:
            dont_ebe.append(c)
    return need_ebe, dont_ebe


def create_pipeline(
    X,
    y,
    classifier,
    do_EBE=False,
    crit=3,
    ):  
    num_cols = dex.list_by_type(X, ["float64", "int32", "int64"])
    cat_cols = dex.list_by_type(X, ["O"])
    if do_EBE:
        do_ebe_cols, dont_ebe_cols = need_EBE(X, cat_cols, crit)
    else:
        do_ebe_cols, dont_ebe_cols = [], cat_cols

    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("ss", StandardScaler())]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    class EBE(BaseEstimator, TransformerMixin):
        def __init__(self, k=None):
            self.k = k

        def fit(self, X, y):
            aux_dict = pd.Series(y).groupby(X).agg(["mean", "count"]).to_dict()
            self._aux_dict = aux_dict
            self._ave = y.mean()
            self._count = y.count()
            return self

        def transform(self, X, y=None):
            X_copy = X.copy()
            fit_unique = set(self._aux_dict["mean"].keys())
            X_unique = set(X.unique())
            unknown_values = X_unique - fit_unique
            X_copy.loc[X_copy.isin(unknown_values)] = np.nan

            group_ave = X_copy.replace(self._aux_dict["mean"])
            group_count = X_copy.replace(self._aux_dict["count"])
            Xt = (
                (group_ave * group_count + self.k * self._ave) / (self.k + group_count)
            ).values.reshape(-1, 1)

            return Xt

    def selector(X, col):
        return X[col]

    def build_ebe_pipeline(col, k=1):
        pipe = Pipeline(
            [
                ("sel", FunctionTransformer(selector, kw_args={"col": col})),
                ("ebe", EBE(k=k)),
            ]
        )
        return pipe

    ebe_pipe = FeatureUnion(
        transformer_list=[(col, build_ebe_pipeline(col)) for col in do_ebe_cols]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, dont_ebe_cols),
            ("ebe", ebe_pipe, do_ebe_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("classifier", classifier),
        ]
    )

    return pipeline


def objective(
    trial, model_class, param_space, X_train, y_train, X_val, y_val, seed_number=0
):
    """
    Objective function for optimizing machine learning models using Optuna.

    This function serves as the objective for an Optuna study, aiming to find the best hyperparameters
    for a machine learning model from a given parameter space. It initializes the model with hyperparameters
    suggested by Optuna, trains it, and then evaluates it using a specified scoring metric.

    :param trial: The trial instance from Optuna.
    :type trial: optuna.trial.Trial
    :param model_class: The class of the machine learning model to be trained.
    :type model_class: class
    :param param_space: Dictionary defining the hyperparameter search space. For categorical hyperparameters,
                        provide a list of possible values. For continuous hyperparameters, provide a tuple of
                        (min, max). For integer hyperparameters, provide a tuple of (min, max, step).
    :type param_space: dict
    :param X_train: Training data features.
    :type X_train: pandas.DataFrame or numpy.ndarray
    :param y_train: Training data target.
    :type y_train: pandas.Series or numpy.ndarray
    :param scoring_metric: The metric function to evaluate the performance of the model.
    :type scoring_metric: callable
    :return:
        - Score of the model on the training data using the provided scoring metric.
    :rtype: float

    :Example:

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.metrics import mean_squared_error
    >>> param_space = {
    ...     'n_estimators': (10, 100, 10),
    ...     'max_depth': (3, 10),
    ...     'min_samples_split': [2, 3, 4]
    ... }
    >>> objective(trial, RandomForestRegressor, param_space, X_train, y_train, mean_squared_error)

    Note:
        Ensure that the required libraries (`optuna`, desired machine learning model, and metrics) are installed before using this function.
    """
    # Extract hyperparameters from the parameter space
    params = {}
    for name, values in param_space.items():
        if values["type"] == "int":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_int(name, **values_cp)
        elif values["type"] == "categorical":
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_categorical(name, **values_cp)
        elif values["type"] == "float":  # corrected this line
            values_cp = {n: v for n, v in values.items() if n != "type"}
            params[name] = trial.suggest_float(name, **values_cp)

    params["random_state"] = seed_number

    # Initialize and train the model
    model = create_pipeline(X_train, y_train, model_class(**params))
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, predictions)

    return score


def optimize_model(
    model_class,
    param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=100,
    seed_number=0,
):
    """
    Optimize hyperparameters of a machine learning model using Optuna.

    This function creates an Optuna study to search for the best hyperparameters for a given machine learning
    model from a specified parameter space. The objective of the study is to maximize the performance of the
    model on the training data using the provided scoring metric.

    :param model_class: The class of the machine learning model to be trained.
    :type model_class: class
    :param param_space: Dictionary defining the hyperparameter search space. For categorical hyperparameters,
                        provide a list of possible values. For continuous hyperparameters, provide a tuple of
                        (min, max). For integer hyperparameters, provide a tuple of (min, max, step).
    :type param_space: dict
    :param X_train: Training data features.
    :type X_train: pandas.DataFrame or numpy.ndarray
    :param y_train: Training data target.
    :type y_train: pandas.Series or numpy.ndarray
    :param scoring_metric: The metric function to evaluate the performance of the model.
    :type scoring_metric: callable
    :param n_trials: Number of trials for optimization. Default is 100.
    :type n_trials: int
    :return:
        - An Optuna study object containing the results of the optimization.
    :rtype: optuna.study.Study

    :Example:

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.metrics import mean_squared_error
    >>> param_space = {
    ...     'n_estimators': (10, 100, 10),
    ...     'max_depth': (3, 10),
    ...     'min_samples_split': [2, 3, 4]
    ... }
    >>> study = optimize_model(RandomForestRegressor, param_space, X_train, y_train, mean_squared_error)

    Note:
        Ensure that the required libraries (`optuna`, desired machine learning model, and metrics) are installed before using this function.
    """
    sampler = TPESampler(seed=seed_number)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            model_class,
            param_space,
            X_train,
            y_train,
            X_val,
            y_val,
            seed_number=0,
        ),
        n_trials=n_trials,
    )

    best_params = study.best_params
    model = create_pipeline(X_train, y_train, model_class(**best_params))
    model.fit(X_train, y_train)
    return study, model


def optimize_models(models, param_spaces, X_train, y_train, X_val, y_val, n_trials=50):
    optimized_models = {}
    for model_name, model_class in models.items():
        print(f"Optimizing {model_name}...")
        study = optimize_model(
            model_class,
            param_spaces[model_name],
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials,
        )
        best_params = study.best_params
        print(f"Best parameters for {model_name}: {best_params}")  # Diagnostic print
        optimized_model = model_class(**best_params)
        print(
            f"Optimized model instance for {model_name}: {optimized_model}"
        )  # Diagnostic print
        optimized_model.fit(X_train, y_train)
        optimized_models[model_name] = optimized_model
    return optimized_models


def ks_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    return opt_threshold


def get_metrics(name_model_dict, X, y, threshold=0.5):
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

    return get_metrics_df(models_dict, y)


def get_fairness_metrics(name_model_dict, X, y, z, benefit_label=1, threshold=0.5):
    """Calculate fairness metrics for a set of models. The metrics are returned in a dataframe.

    :param name_model_dict: dict with model names as keys and models as values
    :param X: dataframe with features
    :param y: ground truth labels
    :param z: binary protected attributed, unprivileged group is 1
    :param benefit_label: label of benefit prediction, defaults to 1
    :param threshold: threshold to transform scores to binary prediction, if is going to use a threshold for each model, the values for name_model_dict should be tuples with model as first value and thershold a secondary vale, defaults to 0.5
    :return: dataframe with columns as fairness metrics and rows as models
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

        models_dict[name] = (y_pred == benefit_label).astype("float")

    def get_metrics_df(models_dict):
        df_dict = {}
        df_dict["DPD"] = [
            np.mean(preds[z == 1]) - np.mean(preds[z == 0])
            for preds in models_dict.values()
        ]
        df_dict["EOD"] = [
            recall_score(y[z == 1], preds[z == 1])
            - recall_score(y[z == 0], preds[z == 0])
            for preds in models_dict.values()
        ]
        df_dict["AOD"] = [
            0.5
            * (
                recall_score(y[z == 1], preds[z == 1])
                - recall_score(y[z == 0], preds[z == 0])
            )
            + 0.5
            * (
                false_positive_rate(y[z == 1], preds[z == 1])
                - false_positive_rate(y[z == 0], preds[z == 0])
            )
            for preds in models_dict.values()
        ]
        df_dict["APVD"] = [
            0.5
            * (
                recall_score(preds[z == 1], y[z == 1])
                - recall_score(preds[z == 0], y[z == 0])
            )
            + 0.5
            * (
                false_positive_rate(preds[z == 1], y[z == 1])
                - false_positive_rate(preds[z == 0], y[z == 0])
            )
            for preds in models_dict.values()
        ]
        df_dict["GMA"] = [
            np.sqrt(
                accuracy_score(y[z == 1], preds[z == 1])
                * accuracy_score(y[z == 0], preds[z == 0])
            )
            for preds in models_dict.values()
        ]
        return pd.DataFrame.from_dict(
            df_dict, orient="index", columns=models_dict.keys()
        ).T

    return get_metrics_df(models_dict)
