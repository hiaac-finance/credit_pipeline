import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from fairgbm import FairGBMClassifier
from sklego.linear_model import DemographicParityClassifier, EqualOpportunityClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split, KFold
from credit_pipeline import data, training, evaluate


import warnings

warnings.filterwarnings("ignore")

MODEL_CLASS_LIST = [
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
    LGBMClassifier,
]


def load_split(dataset_name, fold, validation=False, seed=0):
    """Function that loads the dataset and splits it into train and test.

    :param dataset_name: name of the dataset
    :type dataset_name: string
    :param fold: fold number in the 10-fold cross validation
    :type fold: int
    :param validation: whether to split train into train and validation, defaults to False
    :type validation: bool, optional
    :param seed: random seed, defaults to 0
    :type seed: int, optional
    :return: train and test sets, and validation set if validation is True
    :rtype: pandas.DataFrame
    """
    df = data.load_dataset(dataset_name)
    X = df.drop("DEFAULT", axis=1)
    Y = df.DEFAULT
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i == fold:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            if validation:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=1 / 9, random_state=seed
                )
            else:
                X_val, Y_val = None, None
            break
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def experiment_credit_models(args):
    """Function that run experiments from Section 2.4 of the paper.
    It will fit Logistic, MLP, Random Forest and LightGBM models to any of the datasets in the data folder.
    It will save the models and the metrics in the results folder.

    :param args: arguments for the experiment
    :type args: dict
    """

    for fold in range(10):
        Path(f"../results/{args['dataset']}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(
            args["dataset"], fold, args["validation"], args["seed"]
        )
        for model_class in MODEL_CLASS_LIST:
            print("Model: ", model_class.__name__)
            study, model = training.optimize_model(
                model_class,
                "suggest",
                X_train,
                Y_train,
                X_val,
                Y_val,
                cv=None if args["validation"] else 5,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
            )
            Y_pred = model.predict_proba(X_val)[:, 1]
            threshold = training.ks_threshold(Y_val, Y_pred)
            model_dict = {model_class.__name__: [model, threshold]}
            metrics = evaluate.get_metrics(model_dict, X_test, Y_test)

            # save results
            print(f"Finished training with ROC {study.best_value:.2f}")
            joblib.dump(
                model, f"../results/{args['dataset']}/{fold}/{model_class.__name__}.pkl"
            )
            joblib.dump(
                study,
                f"../results/{args['dataset']}/{fold}/{model_class.__name__}_study.pkl",
            )
            metrics.to_csv(
                f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv",
                index=False,
            )

    metrics = [
        pd.read_csv(f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv")
        for fold in range(10)
        for model_class in MODEL_CLASS_LIST
    ]
    metrics = pd.concat(metrics)
    metrics.groupby("model").mean().to_csv(
        f"../results/{args['dataset']}/metrics_mean.csv"
    )
    metrics.groupby("model").std().to_csv(
        f"../results/{args['dataset']}/metrics_std.csv"
    )


def experiment_fairness(args):
    """Function that run experiments from Section 3 of the paper.
    It will evaluate the fairness of the models trained in the previous experiment.
    It will also fit Reweighting, Demographic Parity and Equalized Odds classifier, FairGBM and Threshold Optimizer.
    It will save the models and the metrics in the results folder.

    :param args: arguments for the experiment
    :type args: dict
    """
    for fold in range(10):
        Path(f"../results/{args['dataset']}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(
            args["dataset"], fold, args["seed"]
        )

        # Reweighting
        print("Model: Reweighting")
        pipeline_preprocess = training.create_pipeline(X_train, Y_train)
        pipeline_preprocess.fit(X_train, Y_train)
        X_train_preprocessed = pipeline_preprocess.transform(X_train)
        df_rw = pd.DataFrame(X_train_preprocessed)
        df_rw["DEFAULT"] = Y_train
        X_train_aif = BinaryLabelDataset(
            df=df_rw, label_names=["DEFAULT"], protected_attribute_names=...  # TODO
        )
        rw = Reweighing(unprivileged_groups=..., privileged_groups=...)  # TODO
        rw.fit(X_train_aif)
        rw_weights = rw.transform(X_train_aif).instance_weights
        for model_class in MODEL_CLASS_LIST:
            study, model = training.optimize_model(
                X_train,
                Y_train,
                X_val,
                Y_val,
                model_class,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
            )

            # SAVE RESULTS # TODO

        for model_class in [DemographicParityClassifier, EqualOpportunityClassifier]:
            sensitive_col_idx = ...
            param_space = {
                "sensitive_col_idx": sensitive_col_idx,
                "positive_target": 0,
            }

            study, model = training.optimize_model(
                X_train,
                Y_train,
                X_val,
                Y_val,
                model_class,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
            )

            # SAVE RESULTS # TODO

        study, model = training.optimize_model(X_train, Y_train, FairGBMClassifier)


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
        default=None,
        help="number of trials for the hyperparameter optimization",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="timeout in seconds for the hyperparameter optimization",
    )
    parser.add_argument(
        "--validation",
        type=bool,
        default=False,
        help="whether to split train into train and validation or use 5-fold cross validation",
    )

    args = vars(parser.parse_args())
    if args["n_trials"] is not None:
        args["timeout"] = None

    if args["experiment"] == "credit_models":
        experiment_credit_models(args)
    elif args["experiment"] == "fairness":
        experiment_fairness(args)
