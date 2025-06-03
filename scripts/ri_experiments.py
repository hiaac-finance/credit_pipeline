import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from credit_pipeline import training, ri_models


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
    "baseline",
    "aug_sc",
    "aug_up",
    "aug_down",
    "aug_fuzzy",
    "extrapolation_only_1",
    "extrapolation_all",
    "extrapolation_confident",
    "ls",
]


def load_data(fold):
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def evaluate_ri(models_dict):
    return [{"method": model, "auc": 1, "kickout": 0} for model in models_dict.keys()]


def gt_model(method_name):
    base_model = LGBMClassifier(**lgbm_params)
    accept_model = LGBMClassifier(**lgbm_params)
    if method_name == "baseline":
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
        augmentation_type = method_name.split("_")[-1]
        return ri_models.Extrapolation(base_model, accept_model, augmentation_type)
    elif method_name == "ls":
        return ri_models.LabelSpreading(base_model, accept_model)


def experiment():
    path = "../results/reject_inference"
    n_folds = 10

    for fold in range(n_folds):
        X_train, y_train, X_val, y_val, X_test, y_test, X_unl, X_unl_test = load_data(
            fold
        )

        X_train_ri = pd.concat([X_train, X_unl], axis=0)
        y_train_ri = pd.concat([y_train, pd.Series([-1] * X_unl.shape[0])], axis=0)
        models_dict = {}
        for method in METHOD_NAMES:
            model = get_model(method)
            pipeline = trainig.create_pipeline(X_train_ri, y_train_ri, model)
            pipeline.fit(X_train_ri, y_train_ri)
            models_dict[method] = pipeline

            metrics = evaluate_ri(models_dict)
            # save to disk
            print(f"Saving model for fold {fold}, method {method}")
            joblib.dump(
                pipeline,
                f"{path}/{fold}/{method}.pkl",
            )
            metrics.to_csv(
                f"{path}/{fold}/{method}_metrics.csv",
                index=False,
            )


if __name__ == "__main__":
    experiment()
