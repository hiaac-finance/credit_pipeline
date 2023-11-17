import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
from sklearn.metrics import roc_auc_score, roc_curve
from optuna.samplers import TPESampler
import optuna

import credit_pipeline.data_exploration as dex

hyperparam_spaces = {
    "LogisticRegression": {
        "C": {"low": 0.001, "high": 10, "log": True, "type": "float"},
        "max_iter": {"low": 1000, "high": 1000, "step": 1, "type": "int"},
        "penalty": {"choices": ["l1", "l2"], "type": "categorical"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "solver": {"choices": ["liblinear"], "type": "categorical"},
    },
    "RandomForestClassifier": {
        "n_estimators": {"low": 10, "high": 150, "step": 20, "type": "int"},
        "max_depth": {"low": 2, "high": 10, "type": "int"},
        "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
        "min_samples_leaf": {"low": 1, "high": 51, "step": 5, "type": "int"},
        "max_features": {"low": 0.1, "high": 1.0, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
    },
    "LGBMClassifier": {
        "learning_rate": {"low": 0.01, "high": 1.0, "type": "float", "log": True},
        "num_leaves": {"low": 5, "high": 100, "step": 5, "type": "int"},
        "max_depth": {"low": 2, "high": 10, "type": "int"},
        "min_child_samples": {"low": 1, "high": 51, "step": 5, "type": "int"},
        "colsample_bytree": {"low": 0.1, "high": 1.0, "type": "float"},
        "reg_alpha": {"low": 0.0, "high": 1.0, "type": "float"},
        "reg_lambda": {"low": 0.0, "high": 1.0, "type": "float"},
        "n_estimators": {"low": 5, "high": 100, "step": 5, "type": "int"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "verbose": {"choices": [-1], "type": "categorical"},
    },
    "MLPClassifier": {
        "hidden_layer_sizes": {
            "choices": [
                [128, 64, 32],
                [128, 64, 32, 16],
                [256, 128, 64, 32, 16],
            ],
            "type": "categorical",
        },
        "alpha": {"low": 0.0001, "high": 0.01, "type": "float", "log": True},
        "learning_rate": {
            "choices": ["constant", "invscaling", "adaptive"],
            "type": "categorical",
        },
        "learning_rate_init": {"low": 0.001, "high": 0.1, "type": "float", "log": True},
        "early_stopping": {"choices": [True], "type": "categorical"},
        "max_iter": {"choices": [50], "type": "categorical"},
    },
}


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
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=777,
                    dtype=np.int64,
                ),
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
                cat_cols,
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
    trial,
    model_class,
    pipeline_params,
    param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    cv,
    seed_number=0,
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

    if cv is None:
        # Initialize and train the model
        model = create_pipeline(
            X_train, y_train, model_class(**params), **pipeline_params
        )
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        predictions = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, predictions)
    else:
        score = []
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed_number)
        for train_idx, test_idx in kf.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
            model = create_pipeline(
                X_train_fold, y_train_fold, model_class(**params), **pipeline_params
            )
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict_proba(X_test_fold)[:, 1]
            score.append(roc_auc_score(y_test_fold, predictions))
        score = np.mean(score)

    return score


def optimize_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_class,
    pipeline_params={},
    param_space=None,
    cv=None,
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

    if param_space is None:
        if model_class.__name__ in hyperparam_spaces.keys():
            param_space = hyperparam_spaces[model_class.__name__]
        else:
            raise ValueError("No hyperparameter space provided")

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
            cv,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    # get params
    model = create_pipeline(
        X_train,
        y_train,
        model_class(random_state=seed_number, **best_params),
        **pipeline_params,
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
