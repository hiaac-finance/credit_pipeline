from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from .evaluate import demographic_parity, equal_opportunity
from fairgbm import FairGBMClassifier


class FairModel(BaseEstimator, ClassifierMixin):
    """
    Base class for fairness-aware models.
    """

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ):
        """
        Fit the model to the data.


        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        y : Union[pd.Series, np.ndarray]
            Target variable.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PreProcessingFair(FairModel):
    """
    Base class for pre-processing fairness models.

    Parameters
    ----------
    estimator : ClassifierMixin
        The base classifier to be used with fairness adjustments. Must follow the Sklearn API and have sample_weight in the fit method.
    """

    def __init__(self, estimator: ClassifierMixin):
        self.estimator = estimator

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "PreProcessingFair":
        """
        Fit the model to the data.


        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        y : Union[pd.Series, np.ndarray]
            Target variable.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations.
        """
        self.classes_ = np.unique(y)
        X_, y_, sample_weights = self._preprocess(X, y, sensitive_attributes)
        self.estimator.fit(X_, y_, sample_weight=sample_weights)  # type: ignore
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict binary labels from data input.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.


        Returns
        -------
        np.ndarray
            Array of binary labels.
        """
        return self.estimator.predict(X)  # type: ignore

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities of each label from data input.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.

        Returns
        -------
        np.ndarray
            Array with shape (n, 2) with probabilities.
        """
        return self.estimator.predict_proba(X)  # type: ignore

    def _preprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ]:
        """Preprocess the data for fairness adjustments.

        Parameters
        ----------
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations
        y : Union[pd.Series, np.ndarray]
            Target variable.

        Returns
        -------
        tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]
            Preprocessed feature matrix, target variable, and sample weights.
        """
        return X, y, np.ones_like(y)


class InProcessingFair(FairModel):
    """
    Base class for in-processing fairness models.
    """

    pass


class PostProcessingFair(FairModel):
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "PostProcessingFair":
        """Base fit method of post-processing algorithms.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        y : Union[pd.Series, np.ndarray]
            Target variable.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations


        Returns
        -------
        PostProcessingFair
            Base model.
        """
        self.classes_ = np.unique(y)
        self.estimator.fit(X, y)  # type: ignore
        self._fit_postprocess(X, y, sensitive_attributes)
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Predict binary labels from input data, and alter predictions based on the sensitive attribute.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations

        Returns
        -------
        np.ndarray
            Array of binary labels.
        """
        pred = self.estimator.predict_proba(X)  # type: ignore
        return self._apply_postprocess(pred, sensitive_attributes)

    def _fit_postprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ):
        """Base method to fit post-process methodologie. Must be implemented.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        y : Union[pd.Series, np.ndarray]
            Binary labels.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations
        """
        pass

    def _apply_postprocess(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Base method of transformation in post-process. Must be implemented.

        Parameters
        ----------
        predictions : np.ndarray
            Array with probabilities of each label.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness consideration

        Returns
        -------
        np.ndarray
            Array of binary labels.
        """
        return predictions


class Reweighing(PreProcessingFair):
    """
    Reweighing pre-processing fairness model based on Kamiran and Calders (2012).
    """

    def _preprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ]:
        weights = {}

        for s_val in np.unique(sensitive_attributes):
            for y_val in np.unique(y):
                observed = np.mean((sensitive_attributes == s_val) & (y == y_val))
                expected = np.mean(sensitive_attributes == s_val) * np.mean(y == y_val)
                if observed == 0:
                    weights[(s_val, y_val)] = 0.0
                else:
                    weights[(s_val, y_val)] = expected / observed

        weights_array = np.array(
            [weights[(s, label)] for s, label in zip(sensitive_attributes, y)]
        )
        return X, y, weights_array


class FairGBM(InProcessingFair):
    def __init__(self, **fairgbm_params):
        self._model = FairGBMClassifier(**fairgbm_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "FairGBM":
        self.classes_ = np.unique(y)
        self._model.fit(X, y, constraint_group=sensitive_attributes)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self._model.predict_proba(X)


class ThresholdOpt(PostProcessingFair):
    def __init__(
        self,
        estimator: BaseEstimator,
        perf_metric: str = "balanced_accuracy",
        fair_metric: str = "demographic_parity",
        constraint_value: float = 0.05,
        n_thresholds: int = 25,
    ):
        """Method for optimizing thresholds to improve fairness.

        Parameters
        ----------
        estimator : BaseEstimator
            Base model.
        perf_metric : str, optional
            Performance metric to optimize, by default "balanced_accuracy"
        fair_metric : str, optional
            Fairness metric to optimize, by default "demographic_parity"
        constraint_value : float, optional
            Maximum allowed value for the fairness metric, by default 0.05
        n_thresholds : int, optional
            Number of thresholds to evaluate, by default 25
        """
        self.estimator = estimator
        self.constraint_value = constraint_value
        self.n_thresholds = n_thresholds
        if perf_metric == "balanced_accuracy":
            self.perf_metric = balanced_accuracy_score
        elif perf_metric == "accuracy":
            self.perf_metric = accuracy_score

        if fair_metric == "demographic_parity":
            self.fair_metric = demographic_parity
        elif fair_metric == "equal_opportunity":
            self.fair_metric = equal_opportunity

    def _fit_postprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ):
        """Fit the post-processing method by finding optimal thresholds for each group.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix.
        y : Union[pd.DataFrame, np.ndarray]
            Binary labels.
        sensitive_attributes : Union[pd.Series, np.ndarray]
            Sensitive attributes for fairness considerations

        Returns
        -------
        None
        """

        # search for two thresholds that minimize fairness criteria and maximizes performance criteria
        preds = self.estimator.predict_proba(X)[:, 1]  # type: ignore
        thresholds_g0 = np.quantile(preds, np.linspace(0, 1, self.n_thresholds))
        thresholds_g1 = np.quantile(preds, np.linspace(0, 1, self.n_thresholds))

        acc_matrix = np.zeros((self.n_thresholds, self.n_thresholds))
        fair_matrix = np.zeros((self.n_thresholds, self.n_thresholds))

        for t0, thresh_0 in enumerate(thresholds_g0):
            for t1, thresh_1 in enumerate(thresholds_g1):
                pred = np.where(
                    sensitive_attributes == 0, preds > thresh_0, preds > thresh_1
                )
                acc = self.perf_metric(y, pred)
                fair = self.fair_metric(y, pred, sensitive_attributes)
                acc_matrix[t0, t1] = acc
                fair_matrix[t0, t1] = fair

        acc_matrix[fair_matrix > self.constraint_value] = 0
        selected_threshold = np.where(acc_matrix == acc_matrix.max())
        self.thresh0 = thresholds_g0[selected_threshold[0][0]]
        self.thresh1 = thresholds_g1[selected_threshold[1][0]]
        return

    def _apply_postprocess(
        self,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray,
    ) -> np.ndarray:
        pred = np.where(
            sensitive_attributes == 0,
            predictions[:, 1] > self.thresh0,
            predictions[:, 1] > self.thresh1,
        )
        return pred
