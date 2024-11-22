import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator, ClassifierMixin
from lime import lime_tabular
import shap
import cfmining.algorithms as alg
from cfmining.action_set import ActionSet
import cfmining.criteria as crit
import dice_ml
import json


class PartialDependencePipeline:
    """Wrapper class to calculate Partial Dependence or Individual Conditional Expectation for a given pipeline. It returns the output with the adjusted scale for features.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline to be explained.
    grid_resolution : int, optional
        Resolution to set values of the feature explained, by default 20
    kind : str, optional
        Option to set between PDP and ICE, use "average" if wants to calculate PDP otherwise use "individual", by default "average"
    """

    def __init__(self, pipeline, grid_resolution=20, kind="average"):
        self.preprocess = pipeline[:-1]
        self.model = pipeline[-1]
        self.grid_resolution = grid_resolution
        self.kind = kind

        class ModelWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.classes_ = [0, 1]  # little trick to be valited by sklearn

            def fit(self, X, y):
                return self

            def predict(self, X):
                return self.model.predict_proba(X)[:, 1]

            def predict_proba(self, X):
                return self.model.predict_proba(X)[:, 1]

        self.model_wrapper = ModelWrapper(self.model)

    def __call__(self, X, features):
        """Calculates the Partial Dependence or Individual Conditional Expectation for a given feature.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to be explained.
        features : list
            List of features to be explained, must be in the columns of X.
            !Important: only works with numerical features.

        Returns
        -------
        dict
            Dictionary containing the values of the feature and the prediction for each value.
        """
        X_preprocess = self.preprocess.transform(X)
        importance = partial_dependence(
            self.model_wrapper,
            X_preprocess,
            features,
            kind=self.kind,
            grid_resolution=self.grid_resolution,
            percentiles=(0.05, 0.95),
            method="brute",
        )

        # get deciles for the feature
        deciles = []
        for feature in features:
            deciles.append(np.percentile(X_preprocess[feature], np.arange(4, 96, 2)))

        # transform back to original scale
        for i, feature in enumerate(features):
            scaled_features = self.preprocess[2].transformers_[0][2]
            if feature in scaled_features:
                idx = scaled_features.index(feature)
                scaler = self.preprocess[2].transformers_[0][1]
                mu = scaler.mean_[idx]
                sigma = scaler.scale_[idx]
                importance["values"][i] = importance["values"][i] * sigma + mu
                deciles[i] = deciles[i] * sigma + mu

        return {
            "values": importance["values"],
            "prediction": importance[self.kind],
            "deciles": deciles,
        }


class ShapPipelineExplainer:
    """Explainer designed to calculate shapley values for a given pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline to be explained.
    background_samples : pd.DataFrame
        Background samples to be used in the explainer.
    method_explain : str, optional
        Method of the prediction to explain, must be in ["prob", "pred"], by default "prob". If "prob", the explainer will use the probability of the positive class as the prediction. If "pred", the explainer will use the prediction of the positive class as the prediction.
    threshold : float, optional
        Threshold utilized if explanation is calculated based on the prediction, by default 0.5
    """

    def __init__(
        self, pipeline, background_samples, method_explain="prob", threshold=0.5
    ):
        self.method_explain = method_explain
        self.threshold = threshold
        self.preprocess = pipeline[:3]
        self.model = pipeline[3:]
        self.categoric_features = self.preprocess[1].transformers_[1][2].copy()
        self.categoric_features += self.preprocess[1].transformers_[2][2].copy()
        self.categories_mapping = [
            dict(enumerate(x))
            for x in self.preprocess[1].transformers_[1][1].categories_
        ]
        self.categories_mapping += [
            dict(enumerate(x))
            for x in self.preprocess[1].transformers_[2][1].unique_values.values()
        ]
        X_preprocess = self.preprocess.transform(background_samples)
        self.feature_names = X_preprocess.columns.tolist()

        if self.method_explain == "prob":
            wrap_model = lambda x: self.model.predict_proba(x)[:, 1]
        elif self.method_explain == "pred":
            wrap_model = lambda x: self.model.predict_proba(x)[:, 1] > self.threshold

        self.explainer = shap.Explainer(
            wrap_model,
            masker=X_preprocess,
            algorithm="permutation",
        )

    def __call__(self, X):
        """Calculates the shapley values for the given input.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to be explained.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the shapley values for each feature.
        """
        X_preprocess = self.preprocess.transform(X)
        shap_values = self.explainer(X_preprocess).values
        explanation_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            explanation_dict[feature_name] = shap_values[:, i]
        return pd.DataFrame(explanation_dict)[self.feature_names]

    def plot_explanation(self, X):
        def filter_columns(importances, top_k=5):
            if len(importances) == 1:
                v = np.abs(importances.values[0])
                return importances.columns[v.argsort()[::-1]][0:top_k].tolist()
            else:
                v = np.mean(np.abs(importances.values), axis=0)
                return importances.columns[v.argsort()[::-1]][0:top_k].tolist()

        X_preprocess = self.preprocess.transform(X)
        prob = self.model.predict_proba(X_preprocess)[0, 1]
        explanation = self(X)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        plt.suptitle(f"prob = {prob:.2f}")
        pos_color = "#80b1d3"
        neg_color = "#fccde5"

        # get most important features
        important_features = filter_columns(explanation, 7)
        important_features = important_features[::-1]
        imp = explanation[important_features].values[0]
        importance_dict = dict(zip(important_features, imp))

        # create barplot
        axs[0].barh(
            important_features,
            imp,
            color=[neg_color if x < 0 else pos_color for x in imp],
        )
        # draw a line in 0
        axs[0].axvline(0, color="#606060", lw=1)

        # remove axis
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["bottom"].set_visible(False)

        # add text to bars
        for j, feature in enumerate(important_features):
            value = importance_dict[feature]
            axs[0].text(
                value,
                j,
                f"{value:.2f}",
                ha="right" if value < 0 else "left",
                va="center",
                color="black",
                fontsize=12,
            )
        # increase xlim to have space for the text
        xrange = imp.max() - imp.min()
        pad = 0.23
        xmin = imp.min() - pad * xrange
        xmax = imp.max() + pad * xrange
        axs[0].set_xlim(xmin, xmax)
        axs[0].set_title("SHAP Values")
        axs[0].set_xticks([])

        # get feature values
        important_features = important_features[::-1]
        X_preprocess.columns = self.feature_names
        values = X_preprocess[important_features].values[0].tolist()
        # return features to original scale
        scaled_features = self.preprocess[2].transformers_[0][2]
        for i, feature in enumerate(important_features):
            if feature in scaled_features:
                scaler = self.preprocess[2].transformers_[0][1]
                idx = scaled_features.index(feature)
                mu = scaler.mean_[idx]
                sigma = scaler.scale_[idx]
                values[i] = values[i] * sigma + mu

        # return categorical features to string values
        for i, feature in enumerate(important_features):
            if feature in self.categoric_features:
                idx = self.categoric_features.index(feature)
                text = self.categories_mapping[idx][int(values[i])]
                text = (
                    text[:10] + "..." if len(text) > 10 else text
                )  # truncate text if too long
                values[i] = text

        # plot a table with feature values
        values = [np.round(x, 2) if isinstance(x, float) else x for x in values]
        table_data = np.array([values]).T
        table = axs[1].table(
            cellText=table_data,
            rowLabels=important_features,
            loc="center",
            colWidths=[0.3],
            fontsize=14,
        )
        axs[1].axis("off")

        # color the table
        for j, feature in enumerate(important_features):
            value = importance_dict[feature]
            table[(j, 0)].set_facecolor(neg_color if value < 0 else pos_color)
            table[(j, -1)].set_facecolor(neg_color if value < 0 else pos_color)

        plt.tight_layout()
        # plt.show()


class LimePipelineExplainer:
    def __init__(
        self, pipeline, background_samples, method_explain="prob", threshold=0.5
    ):
        self.method_explain = method_explain
        self.threshold = threshold
        self.preprocess = pipeline[:3]
        self.model = pipeline[3:]
        X_preprocess = self.preprocess.transform(background_samples)
        self.categoric_features = self.preprocess[1].transformers_[1][2].copy()
        self.categoric_features += self.preprocess[1].transformers_[2][2].copy()
        self.feature_names = X_preprocess.columns.tolist()
        self.categoric_features_idx = [
            i for i, f in enumerate(self.feature_names) if f in self.categoric_features
        ]
        self.categories_mapping = {}
        for i, idx in enumerate(self.categoric_features_idx):
            categoric_feature = self.feature_names[idx]
            if categoric_feature in self.preprocess[1].transformers_[1][2]:
                self.categories_mapping[idx] = (
                    self.preprocess[1]
                    .transformers_[1][1]
                    .categories_[i]
                    .tolist()
                    .copy()
                )
            else:
                self.categories_mapping[idx] = (
                    self.preprocess[1]
                    .transformers_[2][1]
                    .unique_values[categoric_feature]
                    .tolist()
                    .copy()
                )

        self.explainer = lime_tabular.LimeTabularExplainer(
            X_preprocess.values,
            feature_names=self.feature_names,
            class_names=["1"],
            mode="classification",
            categorical_features=self.categoric_features_idx,
            categorical_names=self.categories_mapping,
            discretize_continuous=False,
        )

    def __call__(self, X):

        def pred_fn(X):
            # transform X back to a dataframe
            X_df = pd.DataFrame(X, columns=self.feature_names)
            if self.method_explain == "prob":
                return self.model.predict_proba(X_df)
            elif self.method_explain == "pred":
                return self.model.predict_proba(X_df) > self.threshold

        X_preprocess = self.preprocess.transform(X)
        n = X_preprocess.shape[0]
        explanation_dict = dict([(f, np.zeros(n)) for f in self.feature_names])
        for i in range(n):
            explanation = self.explainer.explain_instance(
                X_preprocess.values[i, :].flatten(),
                pred_fn,
                num_features=len(self.feature_names),
            )

            for f, v in explanation.as_list():
                if "=" in f:
                    f = f.split("=")[0]
                explanation_dict[f][i] = v
        return pd.DataFrame(explanation_dict)

    def explain_instance(self, X):

        def pred_fn(X):
            # transform X back to a dataframe
            X_df = pd.DataFrame(X, columns=self.feature_names)
            if self.method_explain == "prob":
                return self.model.predict_proba(X_df)[:, 1]
            elif self.method_explain == "pred":
                return self.model.predict(X_df)[:, 1] > self.threshold

        X_preprocess = self.preprocess.transform(X)
        explanation = self.explainer.explain_instance(
            X_preprocess.values.flatten(), pred_fn
        )
        return explanation

    def plot_explanation(self, X):
        def filter_columns(importances, top_k=5):
            if len(importances) == 1:
                v = np.abs(importances.values[0])
                return importances.columns[v.argsort()[::-1]][0:top_k].tolist()
            else:
                v = np.mean(np.abs(importances.values), axis=0)
                return importances.columns[v.argsort()[::-1]][0:top_k].tolist()

        X_preprocess = self.preprocess.transform(X)
        prob = self.model.predict_proba(X_preprocess)[0, 1]
        explanation = self(X)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        plt.suptitle(f"prob = {prob:.2f}")
        pos_color = "#80b1d3"
        neg_color = "#fccde5"

        # get most important features
        important_features = filter_columns(explanation, 7)
        important_features = important_features[::-1]
        imp = explanation[important_features].values[0]
        importance_dict = dict(zip(important_features, imp))

        # create barplot
        axs[0].barh(
            important_features,
            imp,
            color=[neg_color if x < 0 else pos_color for x in imp],
        )
        # draw a line in 0
        axs[0].axvline(0, color="#606060", lw=1)

        # remove axis
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["bottom"].set_visible(False)

        # add text to bars
        for j, feature in enumerate(important_features):
            value = importance_dict[feature]
            axs[0].text(
                value,
                j,
                f"{value:.2f}",
                ha="right" if value < 0 else "left",
                va="center",
                color="black",
                fontsize=12,
            )
        # increase xlim to have space for the text
        xrange = imp.max() - imp.min()
        pad = 0.23
        xmin = imp.min() - pad * xrange
        xmax = imp.max() + pad * xrange
        axs[0].set_xlim(xmin, xmax)
        axs[0].set_title("Coefficients")
        axs[0].set_xticks([])

        # get feature values
        important_features = important_features[::-1]
        X_preprocess.columns = self.feature_names
        values = X_preprocess[important_features].values[0].tolist()
        # return features to original scale
        scaled_features = self.preprocess[2].transformers_[0][2]
        for i, feature in enumerate(important_features):
            if feature in scaled_features:
                scaler = self.preprocess[2].transformers_[0][1]
                idx = scaled_features.index(feature)
                mu = scaler.mean_[idx]
                sigma = scaler.scale_[idx]
                values[i] = values[i] * sigma + mu

        # return categorical features to string values
        for i, feature in enumerate(important_features):
            if feature in self.categoric_features:
                idx = self.feature_names.index(feature)
                text = self.categories_mapping[idx][int(values[i])]
                text = (
                    text[:10] + "..." if len(text) > 10 else text
                )  # truncate text if too long
                values[i] = text

        # plot a table with feature values
        values = [np.round(x, 2) if isinstance(x, float) else x for x in values]
        table_data = np.array([values]).T
        table = axs[1].table(
            cellText=table_data,
            rowLabels=important_features,
            loc="center",
            colWidths=[0.3],
            fontsize=14,
        )
        axs[1].axis("off")

        # color the table
        for j, feature in enumerate(important_features):
            value = importance_dict[feature]
            table[(j, 0)].set_facecolor(neg_color if value < 0 else pos_color)
            table[(j, -1)].set_facecolor(neg_color if value < 0 else pos_color)

        plt.tight_layout()
        # plt.show()


class MAPOCAM:
    """Wrapper function for MAPOCAM algorithm, fit expects an individual and can be called multiple times.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline with preprocessing and model.
    X : pd.DataFrame
        Dataframe with model features.
    mutable_features : list
        List of features that can be used on conterfactuals.
    target : int
        Target class to be explained.
    max_changes : int, optional
        Maximum number of features to change on counterfactuals, by default 3.
    criteria : str, optional
        Criteria to optimize with counterfactuals, by default "percentile".
    step_size : float, optional
        Step size of grid of features, by default 0.01
    threshold : float, optional
        Threshold use for prediction, by default 0.5.
    """

    def __init__(
        self,
        pipeline,
        X,
        mutable_features,
        target,
        max_changes=3,
        criteria="percentile",
        step_size=0.01,
        threshold=0.5,
    ):
        class HelperClassifier:
            def __init__(
                self,
                pipeline,
                target=1,
                monotone=False,
                threshold=0.5,
                feat_importance=None,
            ):
                self.model = pipeline[2:]
                self.feature_names = pipeline[:2].get_feature_names_out().tolist()
                self.target = target
                self.monotone = monotone
                self.threshold = threshold
                self.feat_importance = feat_importance

            def predict_proba(self, X):
                X_ = pd.DataFrame([X], columns=self.feature_names)
                prob = self.model.predict_proba(X_)[0, self.target]
                return prob

        self.preprocess = pipeline[:2]
        self.model = pipeline[2:]
        self.all_features = self.preprocess.get_feature_names_out().tolist()
        self.mutable_features = mutable_features
        categoric_features = (
            pipeline[1].transformers_[1][2].copy()
            + pipeline[1].transformers_[2][2].copy()
        )
        for col in mutable_features:
            assert not col in categoric_features

        feat_importance = [0 for _ in self.all_features]
        # check if model is a logistic regression
        from sklearn.linear_model import LogisticRegression

        is_logistic = False
        if isinstance(pipeline[-1], LogisticRegression):
            is_logistic = True
            # utilize coefficients as feature importance
            model_features = self.model[-2].get_feature_names_out().tolist()
            coefs = pipeline[-1].coef_[0]
            for i, col in enumerate(self.all_features):
                if col in mutable_features:
                    idx = model_features.index(col)
                    feat_importance[i] = coefs[idx]

        X_preprocess = self.preprocess.transform(X)
        # little fix for cols with 0 variance
        for col in X_preprocess.columns:
            lb = np.percentile(X_preprocess[col], 1)
            ub = np.percentile(X_preprocess[col], 99)
            if lb == ub:
                X_preprocess[col] = X_preprocess[col] + np.random.normal(
                    0, 0.0001, X_preprocess.shape[0]
                )

        self.action_set = ActionSet(
            X=X_preprocess
        )  # , default_bounds=(0, 100, "percentile")
        # )

        for i, feat in enumerate(self.action_set):
            if not feat.name in self.mutable_features:
                feat.mutable = False
            direction = 1 if feat_importance[i] < 0 else -1
            feat.step_size = step_size
            feat.flip_direction = direction
            feat.step_direction = direction
            feat.update_grid()

        self.predictor = HelperClassifier(
            pipeline,
            target,
            monotone=is_logistic,
            threshold=threshold,
            feat_importance=np.abs(feat_importance),
        )
        if criteria == "percentile":
            perc_calc = crit.PercentileCalculator(action_set=self.action_set)
            self.criteria = lambda x: crit.PercentileCriterion(x, perc_calc)
        elif criteria == "percentile_changes":
            perc_calc = crit.PercentileCalculator(action_set=self.action_set)
            self.criteria = lambda x: crit.PercentileChangesCriterion(x, perc_calc)
        elif criteria == "non_dom":
            self.criteria = lambda x: crit.NonDomCriterion(
                [feat.flip_direction for feat in self.action_set]
            )
        else:
            raise ValueError(
                "criteria must be in ['percentile', 'percentile_changes', 'non_dom']"
            )
        self.max_changes = max_changes

    def fit(self, individual):
        individual_ = self.preprocess.transform(individual).values.flatten()
        method = alg.MAPOCAM(
            self.action_set,
            individual_,
            self.predictor,
            max_changes=self.max_changes,
            compare=self.criteria(individual_),
        )
        method.fit()
        return method.solutions


class Dice:
    """Wrapper function for Dice algorithm, fit expects an individual and can be called multiple times.


    Parameters
    ----------
    X : DataFrame
        Dataframe with model features
    Y : np.ndarray or pd.Series
        Classifier target
    pipeline : sklearn.pipeline.Pipeline
        Pipeline used to preprocess the data
    n_cfs : int
        Number of counterfactuals to generate
    mutable_features: list
        List of features that can be used on conterfactuals
    sparsity_weight: float
        Parameter, weight for sparsity in optimization problem
    clean_solutions: bool
        If True, the solutions will be cleaned to avoid small changes in the features
    """

    def __init__(self, X, Y, pipeline, n_cfs, mutable_features, categoric_mutable_features, sparsity_weight=0.2, clean_solutions=False):
        self.total_CFs = n_cfs
        self.sparsity_weight = sparsity_weight
        self.clean_solutions = clean_solutions
        self.mutable_features = mutable_features
        X_preprocess = pipeline[:2].transform(X)
        self.features = X_preprocess.columns.tolist()
        self.categoric_features = (
            pipeline[1].transformers_[1][2].copy()
            + pipeline[1].transformers_[2][2].copy()
            + categoric_mutable_features
        )
        self.categoric_features = list(set(self.categoric_features))
        self.continuous_features = [
            col for col in self.features if col not in self.categoric_features
        ]
        self.preprocess = pipeline[:2]
        self.model = pipeline[2:]
        dice_model = dice_ml.Model(
            model=self.model, backend="sklearn", model_type="classifier"
        )
        self.permitted_range = {}
        for col in self.features:
            if col in self.categoric_features:
                self.permitted_range[col] = X_preprocess[col].unique().tolist()
                if len(self.permitted_range[col]) == 1:
                    self.permitted_range[col] = [self.permitted_range[col][0], self.permitted_range[col][0]]
            else:
                self.permitted_range[col] = np.percentile(X_preprocess[col], [1, 99]).tolist()
        X_extended = X_preprocess.copy()
        X_extended["target"] = Y
        dice_data = dice_ml.Data(
            dataframe=X_extended,
            continuous_features=self.continuous_features,
            outcome_name="target",
            permitted_range=self.permitted_range,
        )
        self.exp = dice_ml.Dice(dice_data, dice_model)

    def fit(self, individual):
        if type(individual) == np.ndarray:
            individual = pd.DataFrame(data=[individual], columns=self.features)
        individual_ = self.preprocess.transform(individual)
        dice_exp = self.exp.generate_counterfactuals(
            individual_,
            total_CFs=self.total_CFs,
            desired_class="opposite",
            sparsity_weight=self.sparsity_weight,
            proximity_weight=1,
            features_to_vary=self.mutable_features,
        )
        solutions = json.loads(dice_exp.to_json())["cfs_list"][0]
        solutions = [solution[:-1] for solution in solutions]
        if len(solutions) > 0:
            if isinstance(solutions[0], np.ndarray):
                solutions = [s.tolist() for s in solutions]

            if self.clean_solutions:
                solutions = self.get_clean_solutions(individual_, solutions)
        return solutions
    
    def get_clean_solutions(self, individual, solutions):
        individual_dtypes = [individual.dtypes[feat] for feat in individual.columns]
        clean_solutions = []

        for i, sol in enumerate(solutions):
            sol = pd.DataFrame([sol], columns = individual.columns)
            for j, dtype in enumerate(individual_dtypes):
                # adjust dtypes. if numeric, verify if change is at least 0.01%
                if dtype == "int64":
                    sol.iloc[:, j] = sol.iloc[:, j].astype(int)
                    change = abs(sol.iloc[0, j] - individual.iloc[0, j]) / abs(individual.iloc[0, j])
                    if change < 1e-4:
                        sol.iloc[:, j] = individual.iloc[0, j]
                elif dtype == "float64":
                    sol.iloc[:, j] = sol.iloc[:, j].astype(float)
                    change = abs(sol.iloc[0, j] - individual.iloc[0, j]) / abs(individual.iloc[0, j])
                    if change < 1e-4:
                        sol.iloc[:, j] = individual.iloc[0, j]
                else:
                    sol.iloc[:, j] = sol.iloc[:, j].astype(str)

                # if feature is not mutable, keep the original value
                feat_name = self.features[j]
                if not feat_name in self.mutable_features:
                    sol.iloc[:, j] = individual.iloc[0, j]

            clean_solutions.append(sol.values[0].tolist())
        return clean_solutions




def display_cfs(individual, cfs, pipeline, show_change=False):
    """Display the counterfactuals generated by the explainer.

    Parameters
    ----------
    individual : pd.DataFrame
        Individual to be explained.
    cfs : list of np.ndarray
        List of counterfactuals generated by the explainer.
    pipeline : sklearn.pipeline.Pipeline
        Pipeline used to preprocess the data.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the original individual and the counterfactuals.
    """
    preprocess = pipeline[:2]
    individual_ = preprocess.transform(individual).values.flatten()
    feature_names = preprocess.get_feature_names_out().tolist()
    df = pd.DataFrame(
        [individual_] + cfs,
        columns=feature_names,
        index=["Original"] + [f"CF {i}" for i in range(len(cfs))],
    )

    altered_columns = []
    for col in df.columns:
        if df[col].max() != df[col].min():
            altered_columns.append(col)
    df = df[altered_columns]

    # calculate the change
    for col in altered_columns:
        df[col + "_"] = df[col] - df[col].iloc[0]

    for col in df.columns:
        if df[col].max() < 10:
            df[col] = df[col].round(3)
        else:
            df[col] = df[col].round(0).astype(int)

    for col in altered_columns:

        def temp(x):
            if x > 0:
                return "+"
            elif x < 0:
                return "-"
            else:
                return ""

        sign = df[col + "_"].apply(temp)
        if show_change:
            df[col] = (
                df[col].astype(str) + " (" + sign + df[col + "_"].astype(str) + ")"
            )

    # sort row by the number of changes
    n_changes = (df[altered_columns] != df[altered_columns].iloc[0]).sum(axis=1)
    df["n_changes"] = n_changes
    df = df.sort_values("n_changes", ascending=True)
    df = df.drop(columns="n_changes")

    for i in range(1, len(cfs) + 1):
        for col in altered_columns:
            # check if the change is equal to 0
            change = df[col + "_"].iloc[i]
            if change == 0:
                df[col].iloc[i] = "---"

    return df[altered_columns].T
