import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from fairgbm import FairGBMClassifier


class FairModel(BaseEstimator, ClassifierMixin):
    """
    Base class for fairness-aware models.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series):
        """
        Fit the model to the data.

        Parameters:
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        sensitive_attributes : pd.Series.
            Sensitive attributes for fairness considerations.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class PreProcessingFair(FairModel):
    """
    Base class for pre-processing fairness models.

    Parameters
    ----------
    estimator : ClassifierMixin
        The base classifier to be used with fairness adjustments, must follow the Sklearn API and have sample_weight in the fit method.
    """

    def __init__(self, estimator: ClassifierMixin):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series):
        """
        Fit the pre-processing fairness model.

        Parameters:
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        sensitive_attributes : pd.DataFrame.
            Sensitive attributes for fairness considerations.
        """
        X_, y_, sample_weights = self._preprocess(X, y, sensitive_attributes)
        self.estimator.fit(X_, y_, sample_weight=sample_weights)  # type: ignore
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)  # type: ignore

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)  # type: ignore

    def _preprocess(
        self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess the data for fairness adjustments.

        Parameters
        ----------
        sensitive_attributes : pd.Series
            Sensitive attributes for fairness considerations;
        y : pd.Series
            Target variable.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            Preprocessed feature matrix, target variable, and sample weights.
        """
        return X, y, pd.Series(1, index=y.index)


class InProcessingFair(FairModel):
    """
    Base class for in-processing fairness models.
    """

    pass


class PostProcessingFair(FairModel):
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series):
        self.estimator.fit(X, y)  # type: ignore
        self._fit_postprocess(X, y, sensitive_attributes)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.estimator.predict(X)  # type: ignore
        return self._apply_postprocess(pred)

    def _fit_postprocess(
        self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series
    ):
        pass

    def _apply_postprocess(self, predictions: np.ndarray) -> np.ndarray:
        return predictions


class Reweighing(PreProcessingFair):
    """
    Reweighing pre-processing fairness model based on Kamiran and Calders (2012).
    """
    def _preprocess(
        self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        weights = {}

        for s_val in sensitive_attributes.unique():
            for y_val in y.unique():
                observed = np.mean((sensitive_attributes == s_val) & (y == y_val))
                expected = np.mean(sensitive_attributes == s_val) * np.mean(y == y_val)
                if observed == 0:
                    weights[(s_val, y_val)] = 0.0
                else:
                    weights[(s_val, y_val)] = expected / observed

        weights_array = np.array(
            [weights[(s, label)] for s, label in zip(sensitive_attributes, y)]
        )
        return X, y, pd.Series(weights_array, index=y.index)


class FairGBM(InProcessingFair):
    def __init__(self, **fairgbm_params):
        self._model = FairGBMClassifier(**fairgbm_params)

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sensitive_attributes: pd.Series
    ) -> "FairGBM":
        self._model.fit(X, y, constraint_group=sensitive_attributes)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)
