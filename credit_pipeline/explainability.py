import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator, ClassifierMixin
from lime import lime_tabular
import shap


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

    def __call__(self, X, feature):
        """Calculates the Partial Dependence or Individual Conditional Expectation for a given feature.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to be explained.
        feature : str
            Feature to be explained, must be in the columns of X.
            !Important: only works for one feature at a time and with numerical features.

        Returns
        -------
        dict
            Dictionary containing the values of the feature and the prediction for each value.
        """
        X_preprocess = self.preprocess.transform(X)
        importance = partial_dependence(
            self.model_wrapper,
            X_preprocess,
            [feature],
            kind=self.kind,
            grid_resolution=self.grid_resolution,
            percentiles=(0.05, 0.95),
            method="brute",
        )

        # get deciles for the feature
        deciles = np.percentile(X_preprocess[feature], np.arange(4, 96, 2))

        # transform back to original scale
        scaled_features = self.preprocess[2].transformers_[0][2]
        if feature in scaled_features:
            idx = scaled_features.index(feature)
            scaler = self.preprocess[2].transformers_[0][1]
            mu = scaler.mean_[idx]
            sigma = scaler.scale_[idx]
            importance["values"][0] = importance["values"][0] * sigma + mu
            deciles = deciles * sigma + mu

        return {
            "values": importance["values"][0],
            "prediction": importance[self.kind][0],
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
        self.categoric_features = self.preprocess[1].transformers_[1][2]
        self.categories_mapping = [
            dict(enumerate(x))
            for x in self.preprocess[1].transformers_[1][1].categories_
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
        axs[0].set_title("SHAP")

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
                values[i] = self.categories_mapping[idx][int(values[i])]

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
        plt.show()


class LimePipelineExplainer:
    def __init__(
        self, pipeline, background_samples, method_explain="prob", threshold=0.5
    ):
        self.method_explain = method_explain
        self.threshold = threshold
        self.preprocess = pipeline[:3]
        self.model = pipeline[3:]
        X_preprocess = self.preprocess.transform(background_samples)
        self.categoric_features = self.preprocess[1].transformers_[1][2]
        self.feature_names = X_preprocess.columns.tolist()
        self.categoric_features_idx = [
            i for i, f in enumerate(self.feature_names) if f in self.categoric_features
        ]
        self.categories_mapping = {}
        for i, idx in enumerate(self.categoric_features_idx):
            self.categories_mapping[idx] = (
                self.preprocess[1].transformers_[1][1].categories_[i].tolist()
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
        axs[0].set_title("SHAP")

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
                values[i] = self.categories_mapping[idx][int(values[i])]

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
        plt.show()
