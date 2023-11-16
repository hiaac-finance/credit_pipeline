import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from credit_pipeline import training
from credit_pipeline import data

import joblib
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

def load_split(dataset_name, fold, seed = 0):
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
                X_train, Y_train, test_size=1/9, random_state=seed
            )
            break
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def experiment_credit_models(args):
    """Function that runs experimentation with three credit datasets
    to obtain methods with highest ROC score.

    :param configs: _description_
    """

    for fold in range(10):
        Path(f"../results/{args['dataset']}/{fold}").mkdir(parents=True, exist_ok=True)
        print("Fold: ", fold)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_split(args["dataset"], fold, args["seed"])
        for model_class in [LogisticRegression, MLPClassifier,RandomForestClassifier, LGBMClassifier]:
            print("Model: ", model_class.__name__)
            study, model = training.optimize_model(
                X_train,
                Y_train,
                X_val,
                Y_val,
                model_class,
                n_trials = args["n_trials"],
            )
            print(f"Finished training with ROC {study.best_value:.2f}")
            joblib.dump(model, f"../results/{args['dataset']}/{fold}/{model_class.__name__}.pkl")
            joblib.dump(study, f"../results/{args['dataset']}/{fold}/{model_class.__name__}_study.pkl")
            Y_pred = model.predict_proba(X_val)[:, 1]
            threshold = training.ks_threshold(Y_val, Y_pred)
            model_dict = {
                model_class.__name__ : [model, threshold]
            }
            metrics = training.get_metrics(model_dict, X_test, Y_test)
            metrics.to_csv(f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv", index = False)


    # summaryze the metrics
    metrics = []
    for fold in range(10):
        for model_class in [LogisticRegression, MLPClassifier,RandomForestClassifier, LGBMClassifier]:
            metrics.append(
                pd.read_csv(f"../results/{args['dataset']}/{fold}/{model_class.__name__}_metrics.csv")
            )
    metrics = pd.concat(metrics)
    metrics.groupby("model").mean().to_csv(f"../results/{args['dataset']}/metrics_mean.csv")
    metrics.groupby("model").std().to_csv(f"../results/{args['dataset']}/metrics_std.csv")







if __name__ == "__main__":
    args = {
        "dataset" : "german",
        "n_trials" : 10, 
        "seed" : 0,
    }

    experiment_credit_models(args)