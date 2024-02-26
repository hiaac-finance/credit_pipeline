import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.metrics import roc_auc_score, roc_curve
from optuna.samplers import TPESampler
import optuna


hyperparam_spaces = {
    "LogisticRegression": {
        "C": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "tol": {"low": 1e-6, "high": 1e-3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "max_iter": {"low": 1000, "high": 1000, "step": 1, "type": "int"},
        "penalty": {"choices": ["l1", "l2"], "type": "categorical"},
        "solver": {"choices": ["liblinear"], "type": "categorical"},
    },
    "RandomForestClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
        "min_samples_split": {"low": 2, "high": 256, "type": "int"},
        "min_samples_leaf": {"low": 1, "high": 256, "type": "int"},
        "max_features": {"low": 0.1, "high": 1.0, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
    },
    "LGBMClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        "num_leaves": {"low": 4, "high": 256, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "min_child_samples": {"low": 1, "high": 256, "type": "int"},
        "colsample_bytree": {"low": 0.1, "high": 1.0, "type": "float"},
        "reg_alpha": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "reg_lambda": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "verbose": {"choices": [-1], "type": "categorical"},
    },
    "MLPClassifier": {
        "hidden_layer_sizes": {
            "choices": [
                [32],
                [64],
                [128],
                [32, 16],
                [64, 32],
                [64, 32, 16],
            ],
            "type": "categorical",
        },
        "learning_rate_init": {"low": 0.001, "high": 0.1, "type": "float", "log": True},
        "early_stopping": {"choices": [True], "type": "categorical"},
        "max_iter": {"choices": [50], "type": "categorical"},
    },
}



class EBE(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.feature_names_in_ = []
        self.n_features, self.n_items = X.shape[1], X.shape[0]
        self._aux_dict_main = {}
        self.mean = {}
        self.unique_values = {col: X[col].unique() for col in X.columns}
        for i in range(self.n_features):
            Xi = X.iloc[:, i]
            X_name = X.iloc[:, i].name

            y = pd.Series(y, index=X.index)
            aux_dict = pd.Series(y).groupby(Xi).agg(["mean", "count"]).to_dict()
            self._aux_dict_main[X_name] = aux_dict
            self.feature_names_in_.append(X_name)
            self.mean[X_name] = y.mean()
        return self

    def transform(self, X, y=None):
        Xt_list = []
        for i in range(self.n_features):
            Xi = X.iloc[:, i]
            X_name = X.iloc[:, i].name
            X_copy = Xi.copy()
            mean = self._aux_dict_main[X_name]["mean"]
            count = self._aux_dict_main[X_name]["count"]

            mode = pd.Series(mean.values()).mode()[0]
            fit_unique = set(self.unique_values[X_name])
            X_unique = set(Xi.unique())
            unknown_values = list(X_unique - fit_unique)
            X_copy = X_copy.replace(unknown_values, mode)

            group_ave = X_copy.replace(mean)
            group_count = X_copy.replace(count)

            Xt = (
                (group_ave * group_count + self.k * self.mean[X_name])
                / (self.k + group_count)
            ).values.reshape(-1, 1)
            Xt_list.append(Xt)
        Xt_array = np.hstack(Xt_list)
        return Xt_array

    def get_feature_names_out(self, input_features=None):
        if isinstance(input_features, pd.DataFrame):
            return [col for col in input_features.columns]
        else:
            return [col for col in input_features]


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
    do_EBE=True,
    crit=3,
    do_KNN = False,
    n_neighbors = 2,
):
    # def create_pipeline(X, y, classifier = None, onehot = True, onehotdrop = False,
    #                 normalize = True, do_EBE = False, crit = 3, identify_columns = True,
    #                 num_cols = [], cat_cols = [], ebe_cols = []):
    if cat_cols == "infer":
        num_cols = [
            col for col in X.columns if X[col].dtype.kind in ["b", "i", "u", "f", "c"]
        ]
        cat_cols = [col for col in X.columns if X[col].dtype.kind in ["O", "S", "U"]]
        ebe_cols = [col for col in cat_cols if X[col].nunique() >= crit if do_EBE]
        cat_cols = [item for item in cat_cols if item not in ebe_cols]
    else:
        num_cols = X.columns.difference(cat_cols).tolist()
        ebe_cols = [col for col in cat_cols if X[col].nunique() >= crit if do_EBE]
        cat_cols = [item for item in cat_cols if item not in ebe_cols]

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
        
    knn_nan_fill_transformer = Pipeline(
        [("imputer", KNNImputer(n_neighbors=n_neighbors))]
    )
    numeric_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean"))]
    )
    categorical_nan_fill_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    if do_KNN:
        fill_pipe = ColumnTransformer(
            transformers=[
                ("knn", knn_nan_fill_transformer, X.columns),
            ],
        verbose_feature_names_out=False,)
    else:
        fill_pipe = ColumnTransformer(
            transformers=[
                ("num", numeric_nan_fill_transformer, num_cols),
                ("cat", categorical_nan_fill_transformer, cat_cols),
                ("ebe", categorical_nan_fill_transformer, ebe_cols),
            ],
            verbose_feature_names_out=False,)
    fill_pipe.set_output(transform="pandas")

    #2: Ordinal encoder
    encoder_pipe = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            (
                "ebe",
                EBE()
                if do_EBE
                else OrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                ebe_cols,
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
    if do_KNN:
        pipeline = Pipeline(
            steps=[
                ("le", encoder_pipe),
                ("fill", fill_pipe),
                ("ss", scaling_pipe if normalize else None),
                ("hot", onehot_pipe if onehot else None),
                ("classifier", classifier),
            ],
        )
    else:
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
    fit_params,
    score_func,
    param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    cv,
    seed_number,
):
    """Objective function for optimizing machine learning models using Optuna. This function serves as the objective for an Optuna study, aiming to find the best hyperparameters
    for a machine learning model from a given parameter space. It initializes the model with hyperparameters
    suggested by Optuna, trains it, and then evaluates it using a specified scoring metric.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial instance from optuna.create_study().
    model_class : class with sklearn API
        The class of the machine learning model to be trained.
    pipeline_params : dict
        Parameters to call pipeline.
    fit_params : dict
        Parameters that are used when calling fit.
    score_func : function or str
        Scoring function to use, must be according to sklearn API. If string, it must be one in ["roc_auc"].
    param_space : dict
        Description of parameter spaces.
    X_train : array-like
        Training data features.
    y_train : array-like
        Training data target.
    X_val : array-like or None
        Validation data features.
    y_val : array-like or None
        Validation data target.
    cv : int
        Number of folds for cross-validation if validation set is not provided.
    seed_number : int or None
        Random seed.

    Returns
    -------
    float
        The score of the model on the validation set.

    """
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

    if type(score_func) == str and score_func == "roc_auc":
        score_func = roc_auc_score

    if X_val is not None and y_val is not None:
        # Initialize and train the model
        model = create_pipeline(
            X_train, y_train, model_class(**params), **pipeline_params
        )
        model.fit(X_train, y_train, **fit_params)

        # Predict and evaluate the model
        predictions = model.predict_proba(X_val)[:, 1]
        score = score_func(y_val, predictions)
    else:
        if fit_params != {}:
            raise ValueError("fit_params can only be specified with validation set")
        score = []
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed_number)
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = create_pipeline(
                X_train_fold, y_train_fold, model_class(**params), **pipeline_params
            )
            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict_proba(X_val_fold)[:, 1]
            score.append(score_func(y_val_fold, predictions))
        score = np.mean(score)

    return score


def optimize_model(
    model_class,
    param_space,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    cv=5,
    pipeline_params={},
    fit_params={},
    score_func="roc_auc",
    n_trials=None,
    timeout=None,
    seed_number=None,
    n_jobs=1,
):
    """Optimize hyperparameters of a machine learning model using Optuna.
    This function creates an Optuna study to search for the best hyperparameters for a given machine learning model from a specified parameter space. The objective of the study is to maximize the ROC score of the model on the validation score. It can work with a provided validation set or with cross-validation.
    Parameter spaces for LogisticRegression, RandomForestClassifier, LGBMClassifier, and MLPClassifier are provided by default. For any model, a custom parameter space can be provided.

    Parameters
    ----------

    model_class : class with sklearn API
        The class of the machine learning model to be trained.
    param_space: dict with param spaces or string "suggest"
        description of parameter spaces, pass string "suggest" to use default spaces
    X_train: pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train: array-like
        Training data target.
    X_val: pandas.DataFrame or numpy.ndarray, default=None
        Validation data features.
    y_val: array-like, default=None
        Validation data target.
    cv: int, optional, default=5
        Number of folds for cross-validation
    pipeline_params: dict, optional, default={}
        Parameters to call pipeline.
    fit_params: dict, optional, default={}
        Parameters that are used when calling fit.
    score_func: str, default="roc_auc"
        Scoring function to use, must be according to sklearn API.
    n_trials: int, default=100
        Number of trials.
    timeout: int, optional, default=None
        Number of seconds
    seed_number: int, default=None
        Random seed number.

    Returns
    -------
    (optuna.study.Study, sklearn.pipeline.Pipeline) - study and model
    """
    if param_space == "suggest":
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
            fit_params,
            score_func,
            param_space,
            X_train,
            y_train,
            X_val,
            y_val,
            cv,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )

    # Train model with best hyperparameters
    best_params = study.best_params
    model = create_pipeline(
        X_train,
        y_train,
        model_class(random_state=seed_number, **best_params),
        **pipeline_params,
    )
    model.fit(X_train, y_train)
    return study, model


def optimize_model_fast(
    model_class,
    param_space,
    X_train,
    y_train,
    X_val,
    y_val,
    pipeline_params={},
    fit_params={},
    score_func="roc_auc",
    n_trials=None,
    timeout=None,
    seed_number=None,
    n_jobs=1,
):
    """Optimize hyperparameters of a machine learning model (with Scikit API) using Optuna.
    This is a faster version of the optimization as it only fits the final model, and not the complete pipeline.
    The objective is to maximize the metric "score_func" at the validation dataset. "score_func" must be the string "roc_auc" or a function that will be called with validation prediction.
    Parameter spaces for LogisticRegression, RandomForestClassifier, LGBMClassifier, and MLPClassifier are provided by default. For any model, a custom parameter space can be provided.

    Parameters
    ----------

    model_class : class with sklearn API
        The class of the machine learning model to be trained.
    param_space: dict with param spaces or string "suggest"
        description of parameter spaces, pass string "suggest" to use default spaces
    X_train: pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train: array-like
        Training data target.
    X_val: pandas.DataFrame or numpy.ndarray
        Validation data features.
    y_val: array-like
        Validation data target.
    pipeline_params: dict, optional, default={}
        Parameters to call pipeline.
    fit_params: dict, optional, default={}
        Parameters that are used when calling fit.
    score_func: str, default="roc_auc"
        Scoring function to use, must be according to sklearn API.
    n_trials: int, default=100
        Number of trials.
    timeout: int, optional, default=None
        Number of seconds
    seed_number: int, default=None
        Random seed number.

    Returns
    -------
    (optuna.study.Study, sklearn.pipeline.Pipeline) - study and model
    """

    def objective_fast_(
        trial,
        model_class,
        fit_params,
        score_func,
        param_space,
        X_train,
        y_train,
        X_val,
        y_val,
        seed_number,
    ):
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

        try:
            params["random_state"] = seed_number
            model = model_class(**params)
        except:
            del params["random_state"]
            model = model_class(**params)

        
        model.fit(X_train, y_train, **fit_params)

        # Predict and evaluate the model
        y_val_score = model.predict_proba(X_val)[:, 1]
        if type(score_func) == str and score_func == "roc_auc":
            score = roc_auc_score(y_val, y_val_score)
        else:
            y_val_pred = y_val_score > ks_threshold(y_val, y_val_score)
            score = score_func(y_val, y_val_pred, y_val_score)

        return score

    if param_space == "suggest":
        if model_class.__name__ in hyperparam_spaces.keys():
            param_space = hyperparam_spaces[model_class.__name__]
        else:
            raise ValueError("No hyperparameter space provided")

    # pipeline to preprocess
    preprocess = create_pipeline(
        X_train,
        y_train,
        None,
        **pipeline_params,
    )
    preprocess.fit(X_train, y_train)
    X_train_preprocessed = preprocess[:-1].transform(X_train)
    X_val_preprocessed = preprocess[:-1].transform(X_val)

    fit_params_wt_pip = fit_params.copy()
    for key in fit_params.keys():
        fit_params_wt_pip[key.split("__")[1]] = fit_params_wt_pip.pop(key)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed_number)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_fast_(
            trial,
            model_class,
            fit_params_wt_pip,
            score_func,
            param_space,
            X_train_preprocessed,
            y_train,
            X_val_preprocessed,
            y_val,
            seed_number=seed_number,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )

    # Train model with best hyperparameters
    best_params = study.best_params
    try:
        best_params["random_state"] = seed_number
        model = create_pipeline(
            X_train,
            y_train,
            model_class(random_state=seed_number, **best_params),
            **pipeline_params,
        )
        model.fit(X_train, y_train, **fit_params)
    except:
        del best_params["random_state"]
        model = create_pipeline(
            X_train,
            y_train,
            model_class(**best_params),
            **pipeline_params,
        )
        model.fit(X_train, y_train, **fit_params)

    return study, model


def ks_threshold(y_true, y_score):
    """Identify the threshold that maximizes the Kolmogorov-Smirnov statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    return opt_threshold


def create_train_test(dataset, final=False, seed=880, dev_test_size=0.2):
    """Splits a dataset betweeen a train set and development or deployment test.

    Parameters:
    - dataset (DataFrame or array-like): The dataset to be split into train and test sets.
    - final (bool): If True, indicates that the holdout test will be.
                    If False, indicates that the development test will be returned (default: False).
    - seed (int): Seed value for random state for development test (default: 880).
    - dev_test_size (float): The proportion of the dataset to include in the development
    test split (default: 0.2).

    Returns:
    - Tuple: (train_set, test_set) - The training and testing datasets.

    """
    # Do not change the following parameters neither the value of final to avoid data leakage
    train, holdout = train_test_split(dataset, test_size=0.2, random_state=880)
    if final:
        return train, holdout
    else:
        dev_train, dev_test = train_test_split(
            train, test_size=dev_test_size, random_state=seed
        )
        return dev_train, dev_test
