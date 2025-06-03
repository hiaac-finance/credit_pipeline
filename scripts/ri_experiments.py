import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from credit_pipeline import training, ri_models, reject_inference, data_exploration
import sys


SEED = 0

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


def load_data(fold):
    tr_policy = 0.4
    dataset = "../data/HomeCredit/application_train.csv"
    data = pd.read_csv(dataset)

    # Subamostragem estratificada (caso TARGET seja binária)
    data = data.groupby("TARGET", group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=fold))

    # Divide de forma reprodutível
    train_val, test = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=fold)
    train, val = train_test_split(train_val, test_size=0.2, stratify=train_val["TARGET"], random_state=fold)

    # Política só treinada com dados de treino
    train, policy_clf = reject_inference.fit_policy(train, random_state=fold)

    # Remove TARGET de X
    X_train, y_train = train.drop(columns=["TARGET"]), train["TARGET"]
    X_val, y_val = val.drop(columns=["TARGET"]), val["TARGET"]
    X_test, y_test = test.drop(columns=["TARGET"]), test["TARGET"]

    # Aplicação da política
    X_train_acp, X_train_rej, y_train_acp, y_train_rej = reject_inference.accept_reject_split(X_train, y_train, policy_clf, tr_policy)
    X_test_acp, X_test_rej, y_test_acp, y_test_rej = reject_inference.accept_reject_split(X_test, y_test, policy_clf, tr_policy)

    return (
        X_train_acp.reset_index(drop=True),
        y_train_acp.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
        X_test_acp.reset_index(drop=True),
        y_test_acp.reset_index(drop=True),
        X_train_rej.reset_index(drop=True),
        X_test_rej.reset_index(drop=True),
    )



def evaluate_ri(models_dict, X, y, X_v, y_v, X_unl):
    # print(models_dict)
    results = reject_inference.get_metrics_RI(
        models_dict,
        X, y,
        X_v, y_v,
        X_unl,
        threshold_type='default'
    )

    print(results)

    output = []
    for model in models_dict.keys():
        try:
            auc = results.loc['AUC', model]
        except KeyError:
            auc = None  # ou float('nan') se preferir

        try:
            kickout = results.loc['Kickout', model]
        except KeyError:
            kickout = None

        output.append({
            "method": model,
            "auc": auc,
            "kickout": kickout
        })

    return output


def gt_model(method_name):
    base_model = LGBMClassifier(**lgbm_params)
    accept_model = LGBMClassifier(**lgbm_params)
    if method_name == "BM":
        return base_model
    elif method_name == "aug_sc":
        return ri_models.AugSoftCutoff(base_model, accept_model)
    elif method_name == "aug_up":
        return ri_models.AugUpDown(base_model, accept_model, "up")
    elif method_name == "aug_down":
        return ri_models.AugUpDown(base_model, accept_model, "down")
    elif method_name == "aug_fuzzy":
        return ri_models.AugFuzzy(base_model, accept_model)
    elif method_name.startswith("extrapolation"):
        extrapolation_type = method_name.split("-")[-1]
        return ri_models.Extrapolation(base_model, accept_model, extrapolation_type)
    elif method_name == "ls":
        return ri_models.LabelSpreading(base_model, accept_model)


def experiment():
    path = "../results/reject_inference"
    n_folds = 1

    for fold in range(n_folds):
        X_train, y_train, X_val, y_val, X_test, y_test, X_unl, X_unl_test = load_data(
            fold
        )

        print(data_exploration.get_shapes(X_train, y_train, X_val, y_val, X_test, y_test, X_unl, X_unl_test))

        X_train_ri = pd.concat([X_train, X_unl], axis=0)
        y_train_ri = pd.concat([y_train, pd.Series([-1] * X_unl.shape[0])], axis=0)
        models_dict = {}
        for method in METHOD_NAMES:
            model = gt_model(method)
            pipeline = training.create_pipeline(X_train_ri, y_train_ri, model)
            pipeline.fit(X_train_ri, y_train_ri)
            models_dict[method] = pipeline

            metrics = evaluate_ri(models_dict, X_train, y_train, X_val, y_val, X_unl)
            # save to disk
            print(f"Saving model for fold {fold}, method {method}")
            print(metrics)
            # if not os.path.exists(f"{path}/{fold}"):
            #     os.makedirs(f"{path}/{fold}")
            # joblib.dump(
            #     pipeline,
            #     f"{path}/{fold}/{method}.pkl",
            # )
            # metrics.to_csv(
            #     f"{path}/{fold}/{method}_metrics.csv",
            #     index=False,
            # )


if __name__ == "__main__":
    experiment()
