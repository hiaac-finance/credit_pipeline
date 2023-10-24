import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
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


def columns_by_type(dataframe, types_cols=["numeric"], debug=False):
    list_cols = []
    if types_cols == ["numeric"]:
        types_cols = ["b", "i", "u", "f", "c"]
    elif types_cols == ["categorical"]:
        types_cols = ["O", "S", "U"]

    # Iterate through each column in the DataFrame
    for c in dataframe.columns:
        col = dataframe[c]

        # Check if the column's data type matches any of the specified data types
        if (col.dtype.kind in types_cols) or (col.dtype in types_cols):
            list_cols.append(c)

            # Print debugging information if debug flag is enabled
            if debug:
                print(c, " : ", col.dtype)
                print(col.unique()[:10])
                print("---------------")

    return list_cols


def create_pipeline(
    X,
    y,
    classifier=None,
    cat_cols="infer",
    onehot=True,
    onehotdrop=False,
    normalize=True,
    do_EBE=False,
    crit=3,
):
    if cat_cols == "infer":
        num_cols = columns_by_type(X, ["numeric"])
        cat_cols = columns_by_type(X, ["categorical"])
    else:
        num_cols = X.columns.difference(cat_cols).tolist()

    # check if need EBE
    if do_EBE:
        do_ebe_cols = []
        dont_ebe_cols = []
        for c in cat_cols:
            if len(X[c].unique()) >= crit:
                do_ebe_cols.append(c)
            else:
                dont_ebe_cols.append(c)
    else:
        do_ebe_cols, dont_ebe_cols = [], cat_cols

    # 1 Fill NaNs
    numeric_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean"))]
    )
    categorical_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    fill_pipe = ColumnTransformer(
        transformers=[
            ("num", numeric_nan_fill_transformer, num_cols),
            ("cat", categorical_nan_fill_transformer, cat_cols),
        ],
        verbose_feature_names_out=False,
    )
    fill_pipe.set_output(transform="pandas")

    # 2: Ordinal encoder
    encoder_pipe = ColumnTransformer(
        transformers=[
            ("num", "passthrough", make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=777, dtype = np.int64),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        verbose_feature_names_out=False,
    )
    encoder_pipe.set_output(transform="pandas")

    # 3 Scalling
    scaling_pipe = ColumnTransformer(
        [
            ("scaler", StandardScaler(), num_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaling_pipe.set_output(transform="pandas")

    # 4 Onehot encoder
    onehot_pipe = ColumnTransformer(
        transformers=[
            (
                "onehot_encoder",
                OneHotEncoder(
                    drop="if_binary", sparse_output=False, handle_unknown="ignore"
                )
                if onehotdrop
                else OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_cols
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    onehot_pipe.set_output(transform="pandas")

    # Combine all the transformers in a single pipeline
    pipeline = Pipeline(
        steps=[
            ("fill", fill_pipe),
            ("le", encoder_pipe),
            ("ss", scaling_pipe if normalize else None),
            ("hot", onehot_pipe if onehot else None),
            ("classifier", classifier),
        ],
    )

    return pipeline


def objective(
    trial, model_class, pipeline_params, param_space, X_train, y_train, X_val, y_val, seed_number=0
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
    model = create_pipeline(X_train, y_train, model_class(**params), **pipeline_params)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, predictions)

    return score


def optimize_model(
    model_class,
    pipeline_params,
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed_number)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            model_class,
            pipeline_params,
            param_space,
            X_train,
            y_train,
            X_val,
            y_val,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    # get params 
    model = create_pipeline(
        X_train, y_train, model_class(random_state=seed_number, **best_params), **pipeline_params
    )
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


def get_fairness_metrics(models_dict, y, z, benefit_class=1):
    """Calculate fairness metrics for a set of models. The metrics are returned in a dataframe.

    :param model_dict: dict with model names as keys and classification as values (not score)
    :param y: ground truth labels
    :param z: binary protected attributed, unprivileged group is 1
    :param benefit_label: label of benefit prediction, defaults to 1
    :param threshold: threshold to transform scores to binary prediction, if is going to use a threshold for each model, the values for name_model_dict should be tuples with model as first value and thershold a secondary vale, defaults to 0.5
    :return: dataframe with columns as fairness metrics and rows as models
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

    return get_metrics_df(models_dict_benefit)
