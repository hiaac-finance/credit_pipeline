import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from fairgbm import FairGBMClassifier
from sklego.linear_model import DemographicParityClassifier, EqualOpportunityClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split, KFold
from credit_pipeline import data, training, evaluate
from credit_pipeline.models import MLPClassifier


import warnings

warnings.filterwarnings("ignore")

MODEL_CLASS_LIST = [
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
    LGBMClassifier,
]
FAIRNESS_CLASS_LIST = [
    "Reweighing",
    "DemographicParityClassifier",
    "EqualOpportunityClassifier",
    "FairGBMClassifier",
    "ThresholdOptimizer",
]

PROTECTED_ATTRIBUTES = {
    "german": "Gender",
    "taiwan": "SEX",
    "homecredit": "CODE_GENDER",
}

HOMECREDIT_PARAM_SPACE = training.hyperparam_spaces.copy()
HOMECREDIT_PARAM_SPACE["LogisticRegression"]["solver"] = {
    "choices": ["saga"],
    "type": "categorical",
}
HOMECREDIT_PARAM_SPACE["LogisticRegression"]["max_iter"] = {
    "low": 100,
    "high": 100,
    "step": 1,
    "type": "int",
}
HOMECREDIT_PARAM_SPACE["MLPClassifier"]["batch_size"] = {
    "choices": [1024],
    "type": "categorical",
}


FAIRNESS_PARAM_SPACES = {}
FAIRNESS_PARAM_SPACES["FairGBMClassifier"] = training.hyperparam_spaces[
    "LGBMClassifier"
].copy()
FAIRNESS_PARAM_SPACES["FairGBMClassifier"]["multiplier_learning_rate"] = {
    "low": 0.01,
    "high": 1,
    "type": "float",
}
del FAIRNESS_PARAM_SPACES["FairGBMClassifier"]["class_weight"]
FAIRNESS_PARAM_SPACES["EqualOpportunityClassifier"] = {
    "covariance_threshold": {"low": 0, "high": 1, "type": "float"},
    "max_iter": {"choices": [1000], "type": "categorical"},
    "C": {"low": 0.01, "high": 100, "type": "float"},
}
FAIRNESS_PARAM_SPACES["DemographicParityClassifier"] = FAIRNESS_PARAM_SPACES[
    "EqualOpportunityClassifier"
].copy()
FAIRNESS_GOAL = {"german": 0.1, "taiwan": 0.1, "homecredit": 0.1}


def load_split(dataset_name, fold, seed=0):
    """Function that loads the dataset and splits it into train and test. Following, splits the train set into train and validation using 10-fold cross validation.

    Parameters
    ----------
        dataset_name : string
            Name of the dataset in ["german", "taiwan", "homecredit"]
        fold : int
            Fold number in the 10-fold cross validation
        seed : int, optional
            Random seed. Defaults to 0.

    Returns
    -------
        pandas.DataFrame: train, validation and test sets
    """
    df = data.load_dataset(dataset_name)
    train, test = training.create_train_test(df, final=True, seed=seed)
    X_train_ = train.drop(columns=["DEFAULT"])
    Y_train_ = train["DEFAULT"]
    X_test = test.drop(columns=["DEFAULT"])
    Y_test = test["DEFAULT"]
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for i, (train_index, val_index) in enumerate(kf.split(X_train_)):
        if i == fold:
            X_train = X_train_.iloc[train_index]
            Y_train = Y_train_.iloc[train_index]
            X_val = X_train_.iloc[val_index]
            Y_val = Y_train_.iloc[val_index]
            break
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def experiment_credit_models(args):
    """Function that run experiments from Section 2.4 of the paper.
    It will fit Logistic, MLP, Random Forest and LightGBM models to any of the datasets in the data folder.
    It will save the models and the metrics in the results folder.

    Parameters
    ----------
        args (dict): arguments for the experiment
    """
    path = f"../results/credit_models/{args['dataset']}"

    for fold in range(10):
        Path(f"{path}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(
            args["dataset"], fold, args["seed"]
        )
        # Workaround to obtain the protected attribute as a binary column
        if args["dataset"] == "homecredit":  # Small fix to not apply EBE to gender
            pipeline_preprocess = training.create_pipeline(X_train, Y_train, crit=4)
        else:
            pipeline_preprocess = training.create_pipeline(X_train, Y_train)
        pipeline_preprocess.fit(X_train, Y_train)
        X_train_preprocessed = pipeline_preprocess.transform(X_train)
        A_train = X_train_preprocessed[PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"]
        X_test_preprocessed = pipeline_preprocess.transform(X_test)
        A_test = X_test_preprocessed[PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"]
        del X_train_preprocessed, X_test_preprocessed, pipeline_preprocess

        for model_class in MODEL_CLASS_LIST:
            print("Model: ", model_class.__name__)
            param_space = "suggest"
            if args["dataset"] == "homecredit":
                param_space = HOMECREDIT_PARAM_SPACE[model_class.__name__]

            study, model = training.optimize_model_fast(
                model_class,
                param_space,
                X_train,
                Y_train,
                X_val,
                Y_val,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
                seed_number=args["seed"],
                pipeline_params={"crit": 4} if args["dataset"] == "homecredit" else {},
                n_jobs=args["n_jobs"],
            )
            Y_pred = model.predict_proba(X_train)[:, 1]
            threshold = training.ks_threshold(Y_train, Y_pred)
            model_dict = {model_class.__name__: [model, threshold]}
            metrics = evaluate.get_metrics(model_dict, X_test, Y_test)
            fairness_metrics = evaluate.get_fairness_metrics(
                model_dict, X_test, Y_test, A_test
            )

            # save results
            print(f"Finished training with ROC {study.best_value:.2f}")
            joblib.dump(model, f"{path}/{fold}/{model_class.__name__}.pkl")
            joblib.dump(
                study,
                f"{path}/{fold}/{model_class.__name__}_study.pkl",
            )
            metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_metrics.csv",
                index=False,
            )
            fairness_metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_fairness_metrics.csv",
                index=False,
            )

    metrics = [
        pd.read_csv(f"{path}/{fold}/{model_class.__name__}_metrics.csv")
        for fold in range(10)
        for model_class in MODEL_CLASS_LIST
    ]
    metrics = pd.concat(metrics)
    metrics_mean = metrics.groupby("model").mean()
    metrics_std = metrics.groupby("model").std()
    metrics = metrics_mean.join(metrics_std, lsuffix="_mean", rsuffix="_std")
    metrics.to_csv(f"{path}/metrics.csv")

    fairness_metrics = [
        pd.read_csv(f"{path}/{fold}/{model_class.__name__}_fairness_metrics.csv")
        for fold in range(10)
        for model_class in MODEL_CLASS_LIST
    ]
    fairness_metrics = pd.concat(fairness_metrics)
    fairness_metrics_mean = fairness_metrics.groupby("model").mean()
    fairness_metrics_std = fairness_metrics.groupby("model").std()
    fairness_metrics = fairness_metrics_mean.join(
        fairness_metrics_std, lsuffix="_mean", rsuffix="_std"
    )
    fairness_metrics.to_csv(f"{path}/fairness_metrics.csv")


def experiment_fairness(args):
    """Function that run experiments from Section 3 of the paper.
    It will evaluate the fairness of the models trained in the previous experiment.
    It will also fit Reweighting, Demographic Parity and Equalized Odds classifier, FairGBM and Threshold Optimizer.
    It will save the models and the metrics in the results folder.

    Args:
        args (dict): arguments for the experiment
    """
    path = f"../results/fair_models/{args['dataset']}"
    for fold in range(10):
        Path(f"{path}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(
            args["dataset"], fold, args["seed"]
        )
        # Workaround to obtain the protected attribute as a binary column
        if args["dataset"] == "homecredit":  # Small fix to not apply EBE to gender
            pipeline_preprocess = training.create_pipeline(X_train, Y_train, crit=4)
        else:
            pipeline_preprocess = training.create_pipeline(X_train, Y_train)
        pipeline_preprocess.fit(X_train, Y_train)
        X_train_preprocessed = pipeline_preprocess.transform(X_train)
        A_train = X_train_preprocessed[PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"]
        X_val_preprocessed = pipeline_preprocess.transform(X_val)
        A_val = X_val_preprocessed[PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"]
        X_test_preprocessed = pipeline_preprocess.transform(X_test)
        A_test = X_test_preprocessed[PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"]

        scorer_validation = evaluate.create_fairness_scorer(
            FAIRNESS_GOAL[args["dataset"]], A_val, benefit_class = 0
        )

        if "Reweighing" in FAIRNESS_CLASS_LIST:
            # Reweighting
            print("Model: Reweighing")
            df_rw = pd.DataFrame(X_train_preprocessed)
            df_rw["DEFAULT"] = Y_train
            X_train_aif = BinaryLabelDataset(
                df=df_rw,
                label_names=["DEFAULT"],
                protected_attribute_names=[
                    PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"
                ],
            )
            rw = Reweighing(
                unprivileged_groups=[{PROTECTED_ATTRIBUTES[args["dataset"]] + "_0": 0}],
                privileged_groups=[{PROTECTED_ATTRIBUTES[args["dataset"]] + "_0": 1}],
            )
            rw.fit(X_train_aif)
            rw_weights = rw.transform(X_train_aif).instance_weights
            for model_class in MODEL_CLASS_LIST:
                print("Model: ", model_class.__name__)
                if args["dataset"] == "homecredit":
                    param_space = HOMECREDIT_PARAM_SPACE[model_class.__name__]
                else:
                    param_space = "suggest"
                study, model = training.optimize_model_fast(
                    model_class,
                    param_space,
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    fit_params={"classifier__sample_weight": rw_weights},
                    score_func=scorer_validation,
                    n_trials=args["n_trials"],
                    timeout=args["timeout"],
                    n_jobs=args["n_jobs"],
                )
                Y_train_score = model.predict_proba(X_train)[:, 1]
                threshold = training.ks_threshold(Y_train, Y_train_score)
                model_dict = {"rw_" + model_class.__name__: [model, threshold]}
                metrics = evaluate.get_metrics(model_dict, X_test, Y_test)
                fairness_metrics = evaluate.get_fairness_metrics(
                    model_dict, X_test, Y_test, A_test, benefit_class = 0
                )

                joblib.dump(model, f"{path}/{fold}/rw_{model_class.__name__}.pkl")
                joblib.dump(
                    study,
                    f"{path}/{fold}/rw_{model_class.__name__}_study.pkl",
                )
                metrics.to_csv(
                    f"{path}/{fold}/rw_{model_class.__name__}_metrics.csv",
                    index=False,
                )
                fairness_metrics.to_csv(
                    f"{path}/{fold}/rw_{model_class.__name__}_fairness_metrics.csv",
                    index=False,
                )

                print(f"Finished training with ROC {study.best_value:.2f}")

        for model_class in [DemographicParityClassifier, EqualOpportunityClassifier]:
            if not model_class.__name__ in FAIRNESS_CLASS_LIST:
                continue

            print("Model: ", model_class.__name__)
            param_space = FAIRNESS_PARAM_SPACES[model_class.__name__]
            param_space["sensitive_cols"] = {
                "choices": [PROTECTED_ATTRIBUTES[args["dataset"]] + "_0"],
                "type": "categorical",
            }
            if model_class.__name__ == "EqualOpportunityClassifier":
                param_space["positive_target"] = {
                    "choices": [1],
                    "type": "categorical",
                }
            study, model = training.optimize_model_fast(
                model_class,
                param_space,
                X_train,
                Y_train,
                X_val,
                Y_val,
                score_func=scorer_validation,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
                n_jobs=args["n_jobs"],
            )
            Y_train_score = model.predict_proba(X_train)[:, 1]
            threshold = training.ks_threshold(Y_train, Y_train_score)
            model_dict = {model_class.__name__: [model, threshold]}
            metrics = evaluate.get_metrics(model_dict, X_test, Y_test)
            fairness_metrics = evaluate.get_fairness_metrics(
                model_dict, X_test, Y_test, A_test,  benefit_class = 0
            )
            joblib.dump(model, f"{path}/{fold}/{model_class.__name__}.pkl")
            joblib.dump(
                study,
                f"{path}/{fold}/{model_class.__name__}_study.pkl",
            )
            metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_metrics.csv",
                index=False,
            )
            fairness_metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_fairness_metrics.csv",
                index=False,
            )

            print(f"Finished training with ROC {study.best_value:.2f}")

        if "FairGBMClassifier" in FAIRNESS_CLASS_LIST:
            model_class = FairGBMClassifier
            print("Model: ", model_class.__name__)
            study, model = training.optimize_model_fast(
                model_class,
                FAIRNESS_PARAM_SPACES[model_class.__name__],
                X_train,
                Y_train,
                X_val,
                Y_val,
                fit_params={"classifier__constraint_group": A_train},
                score_func=scorer_validation,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
                n_jobs=args["n_jobs"],
            )
            Y_train_score = model.predict_proba(X_train)[:, 1]
            threshold = training.ks_threshold(Y_train, Y_train_score)
            model_dict = {model_class.__name__: [model, threshold]}
            metrics = evaluate.get_metrics(model_dict, X_test, Y_test)
            fairness_metrics = evaluate.get_fairness_metrics(
                model_dict, X_test, Y_test, A_test,  benefit_class = 0
            )

            joblib.dump(model, f"{path}/{fold}/{model_class.__name__}.pkl")
            joblib.dump(
                study,
                f"{path}/{fold}/{model_class.__name__}_study.pkl",
            )
            metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_metrics.csv",
                index=False,
            )
            fairness_metrics.to_csv(
                f"{path}/{fold}/{model_class.__name__}_fairness_metrics.csv",
                index=False,
            )

            print(f"Finished training with ROC {study.best_value:.2f}")

        if "ThresholdOptimizer" in FAIRNESS_CLASS_LIST:
            model_class_ = MODEL_CLASS_LIST
        else:
            model_class_ = []

        for model_class in model_class_:
            path_ = path
            path_ = path_.replace("fair_models", "credit_models")
            model = joblib.load(f"{path_}/{fold}/{model_class.__name__}.pkl")
            print("Model: ", model_class.__name__)
            model = model.steps[-1][1]
            thr_opt = ThresholdOptimizer(
                estimator=model,
                constraints="true_positive_rate_parity",
                objective="balanced_accuracy_score",
                prefit=True,
                predict_method="predict_proba",
            )
            thr_opt.fit(X_train_preprocessed, Y_train, sensitive_features=A_train)

            class Thr_helper:
                def __init__(self, model, sensitive_features):
                    self.model = model
                    self.sensitive_features = sensitive_features

                def predict(self, X):
                    return self.model.predict(
                        X, sensitive_features=self.sensitive_features
                    )

            thr_opt_helper = Thr_helper(thr_opt, A_test)
            model_dict = {"thr_" + model_class.__name__: [thr_opt_helper, None]}
            metrics = evaluate.get_metrics(model_dict, X_test_preprocessed, Y_test)
            fairness_metrics = evaluate.get_fairness_metrics(
                model_dict, X_test_preprocessed, Y_test, A_test,  benefit_class = 0
            )
            joblib.dump(thr_opt, f"{path}/{fold}/thr_{model_class.__name__}.pkl")
            metrics.to_csv(
                f"{path}/{fold}/thr_{model_class.__name__}_metrics.csv",
                index=False,
            )
            fairness_metrics.to_csv(
                f"{path}/{fold}/thr_{model_class.__name__}_fairness_metrics.csv",
                index=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="credit_models",
        help="name of the experiment to run",
        choices=["credit_models", "fairness"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        help="name of the dataset to run experiments",
        choices=["german", "taiwan", "homecredit"],
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for the experiment"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="number of trials for the hyperparameter optimization",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="timeout in seconds for the hyperparameter optimization",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="number of jobs to run in parallel for the hyperparameter optimization",
    )

    args = vars(parser.parse_args())
    if args["n_trials"] is not None:
        args["timeout"] = None

    if args["experiment"] == "credit_models":
        experiment_credit_models(args)
    elif args["experiment"] == "fairness":
        experiment_fairness(args)
