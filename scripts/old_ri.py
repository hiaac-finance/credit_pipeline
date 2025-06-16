import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from credit_pipeline import training, ri_models, reject_inference, data_exploration
import sys


SEED = 42

lgbm_params = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 0.22534977954592625,
    "importance_type": "split",
    "learning_rate": 0.052227873762946964,
    "max_depth": 5,
    "min_child_samples": 26,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 159,
    "n_jobs": -1,
    "num_leaves": 12,
    "objective": None,
    "random_state": SEED,
    "reg_alpha": 0.7438345471808012,
    "reg_lambda": 0.46164693905368515,
    "verbose": -1,
    "subsample": 0.8896599304061413,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "is_unbalance": True,
}

METHOD_NAMES = [
    "BM",
    "aug_sc",
    "aug_up",
    "aug_down",
    "aug_fuzzy",
    "extrapolation-only_1",
    "extrapolation-all",
    "extrapolation-confident",
    "ls",
]


def load_data(seed, fold):
    tr_policy = 0.4
    dataset = "../data/HomeCredit/application_train.csv"
    data = pd.read_csv(dataset)

    # Subamostragem estratificada (caso TARGET seja binária)
    # data = data.sample(frac=0.2, random_state=fold)

    # Divide de forma reprodutível
    train_val, test = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=seed+fold)
    train, val = train_test_split(train_val, test_size=0.2, stratify=train_val["TARGET"], random_state=seed+fold)

    # Política só treinada com dados de treino
    train, policy_clf = reject_inference.fit_policy(train, random_state=seed+fold)

    # Remove TARGET de X
    X_train, y_train = train.drop(columns=["TARGET"]), train["TARGET"]
    X_val, y_val = val.drop(columns=["TARGET"]), val["TARGET"]
    X_test, y_test = test.drop(columns=["TARGET"]), test["TARGET"]

    # Aplicação da política
    X_train_acp, X_train_rej, y_train_acp, y_train_rej = reject_inference.accept_reject_split(X_train, y_train, policy_clf, tr_policy)
    X_val_acp, X_val_rej, y_val_acp, y_val_rej = reject_inference.accept_reject_split(X_val, y_val, policy_clf, tr_policy)
    X_test_acp, X_test_rej, y_test_acp, y_test_rej = reject_inference.accept_reject_split(X_test, y_test, policy_clf, tr_policy)

    return (
        X_train_acp,#.reset_index(drop=True),
        y_train_acp,#.reset_index(drop=True),
        X_val_acp,#.reset_index(drop=True),
        y_val_acp,#.reset_index(drop=True),
        X_test_acp,#.reset_index(drop=True),
        y_test_acp,#.reset_index(drop=True),
        X_train_rej,#.reset_index(drop=True),
        X_test_rej,#.reset_index(drop=True),
    )



def evaluate_ri(models_dict, X, y, X_v, y_v, X_unl, seed):
    # print(models_dict)
    results = reject_inference.get_metrics_RI(
        models_dict,
        X, y,
        X_v, y_v,
        X_unl,
        threshold_type='default',
        acp_rate=0.25,
    )

    print(results.round(3))

    output = {}
    for model in models_dict.keys():
        auc = results.loc['AUC', model]
        Approval_Rate = results.loc['Approval_Rate', model]
        kickout = results.loc['Kickout', model]
        output[model] = {
            "seed": seed,
            "auc": auc.round(3),
            "ar" : Approval_Rate.round(3),
            "kickout": kickout.round(3),
        }

    return output


def gt_model(method_name, X, y, X_unl, seed):
    if method_name == "BM":
        benchmark = training.create_pipeline(X, y, LGBMClassifier(**lgbm_params))
        benchmark.fit(X,y)
        return {"BM": benchmark}
    elif method_name == "aug_sc":
        return reject_inference.augmentation_with_soft_cutoff(X, y, X_unl, seed)
    elif method_name == "aug_up":
        return reject_inference.augmentation(X, y, X_unl, mode="up", seed=seed)
    elif method_name == "aug_down":
        return reject_inference.augmentation(X, y, X_unl, mode="down", seed=seed)
    elif method_name == "aug_fuzzy":
        return reject_inference.fuzzy_augmentation(X, y, X_unl, seed)
    elif method_name.startswith("extrapolation"):
        extrapolation_type = method_name.split("-")[-1]
        mode = {"all":"A",  "only_1": "B","confident":"C"}
        return reject_inference.extrapolation(X, y, X_unl, mode=mode[extrapolation_type], seed=seed)
    elif method_name == "ls":
        return reject_inference.label_spreading(X, y, X_unl, seed=seed)


def experiment(seed, n_folds = 1):
    path = "../results/reject_inference_old"
    n_folds = n_folds

    fold_metrics = pd.DataFrame()
    for fold in range(n_folds):
        X_train, y_train, X_val, y_val, X_test, y_test, X_unl, X_unl_test = load_data(
            seed,  fold
        )

        print(data_exploration.get_shapes(X_train, y_train, X_val, y_val, X_test, y_test, X_unl, X_unl_test))

        X_train_ri = pd.concat([X_train, X_unl], axis=0)
        y_train_ri = pd.Series(
            np.concatenate([y_train.values, [-1] * len(X_unl)]),
            index=X_train_ri.index)

        models_dict = {}
        for method in METHOD_NAMES:
            model = gt_model(method, X_train, y_train, X_unl, seed = seed+fold)
            models_dict.update(model)

            # save to disk
            print(f"Saving model for fold {fold}, method {method}")
        
        metrics = evaluate_ri(models_dict, X_test, y_test, X_val, y_val, X_unl, seed = seed+fold)
        metrics_df = pd.DataFrame(metrics)

        print(metrics_df)
        pathfile = f"{path}/{seed}/{fold}"
        if not os.path.exists(f"{pathfile}"):
            os.makedirs(f"{pathfile}")
        # joblib.dump(
        #     pipeline,
        #     f"{path}/{fold}/{method}.pkl",
        # )
        metrics_df.to_csv(
            f"{pathfile}/ri_metrics.csv",
            index=False,
        )


if __name__ == "__main__":
    experiment(seed = SEED, n_folds=25)
