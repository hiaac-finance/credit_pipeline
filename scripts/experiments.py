import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from credit_pipeline import data, training, evaluate


import warnings

warnings.filterwarnings("ignore")


def load_split(dataset_name, fold, seed=0):
    """Function that loads the dataset and splits it into train, val and test.

    :param dataset_name: name of the dataset
    :type dataset_name: string
    :param fold: fold number in the 10-fold cross validation
    :type fold: int
    :param seed: random seed, defaults to 0
    :type seed: int, optional
    :return: train, val and test sets
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
            # split train into val
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=1 / 9, random_state=seed
            )
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
        for model_class in [
            LogisticRegression,
            MLPClassifier,
            RandomForestClassifier,
            LGBMClassifier,
        ]:
            print("Model: ", model_class.__name__)
            study, model = training.optimize_model(
                X_train,
                Y_train,
                X_val,
                Y_val,
                model_class,
                n_trials=args["n_trials"],
            )
            print(f"Finished training with ROC {study.best_value:.2f}")
            joblib.dump(
                model, f"../results/{args['dataset']}/{fold}/{model_class.__name__}.pkl"
            )
            joblib.dump(
                study,
                f"../results/{args['dataset']}/{fold}/{model_class.__name__}_study.pkl",
            )
            Y_pred = model.predict_proba(X_val)[:, 1]
            threshold = training.ks_threshold(Y_val, Y_pred)
            model_dict = {model_class.__name__: [model, threshold]}
            metrics = evaluate.get_metrics(model_dict, X_test, Y_test)
            metrics.to_csv(
                f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv",
                index=False,
            )

    # summaryze the metrics
    metrics = []
    for fold in range(10):
        for model_class in [
            LogisticRegression,
            MLPClassifier,
            RandomForestClassifier,
            LGBMClassifier,
        ]:
            metrics.append(
                pd.read_csv(
                    f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv"
                )
            )
    metrics = pd.concat(metrics)
    metrics.groupby("model").mean().to_csv(
        f"../results/{args['dataset']}/metrics_mean.csv"
    )
    metrics.groupby("model").std().to_csv(
        f"../results/{args['dataset']}/metrics_std.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="credit_models",
        help="name of the experiment to run",
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

    args = vars(parser.parse_args())
    if args["experiment"] == "credit_models":
        experiment_credit_models(args)
