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


def load_split(dataset_name, fold, seed=0):
    """Function that loads the dataset and splits it into train and test.
    It splits the train set into train and validation using 10-fold cross validation.

    Args:
        dataset_name (string): name of the dataset in ["german", "taiwan", "homecredit"]
        fold (int): fold number in the 10-fold cross validation
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        pandas.DataFrame: train, validation and test sets
    """
    df = data.load_dataset(dataset_name)
    train, test = training.create_train_test(df, final=False, seed=seed)
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

    :param args: arguments for the experiment
    :type args: dict
    """

    for fold in range(10):
        Path(f"../results/{args['dataset']}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(
            args["dataset"], fold, args["seed"]
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
                cv=None,
                n_trials=args["n_trials"],
                timeout=args["timeout"],
            )
            Y_pred = model.predict_proba(X_train)[:, 1]
            threshold = training.ks_threshold(X_train, Y_pred)
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
        pd.read_csv(
            f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv"
        )
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

    args = vars(parser.parse_args())
    if args["n_trials"] is not None:
        args["timeout"] = None

    if args["experiment"] == "credit_models":
        experiment_credit_models(args)
    elif args["experiment"] == "fairness":
        experiment_fairness(args)
