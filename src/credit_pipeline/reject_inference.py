from typing import Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.semi_supervised import LabelSpreading


class RejectInference(ClassifierMixin, BaseEstimator):
    """
    Base class for the implementation of Reject Inference methods.
    """

    def __init__(self, base_estimator: BaseEstimator, reject_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.reject_estimator = reject_estimator

    def _wrap(self, X: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
        """
        Helper method to wrap a numpy array back into a pandas DataFrame 
        if feature names were provided during fit.
        """
        if getattr(self, "feature_names_in_", None) is not None:
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return X

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "RejectInference":
        """Standard call for reject inference methods.
        Feature matrix should include in sequence data from labaled and unlabeled population.
        Labels y are only for the labeled population, and the unlabeled population should be labeled as -1.

        All procedures will update the training set (X, y) or define sample weights using the method _preprocess.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix of the labeled data.
        y : Union[np.ndarray, pd.Series]
            Labels.

        Returns
        -------
        RejectInference
            The fitted RejectInference object.
        """
        # Store feature names to prevent LightGBM/Scikit-Learn warnings
        self.feature_names_in_ = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else None

        # Extract underlying arrays for fast, warning-free internal slicing
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        self.classes_ = np.unique(
            y_arr[y_arr != -1]
        )  # only consider the classes in the labeled data
        
        X_unl = X_arr[y_arr == -1]
        X_lab = X_arr[y_arr != -1]
        y_lab = y_arr[y_arr != -1]
        
        X_updated, y_updated, sample_weights = self._preprocess(X_lab, y_lab, X_unl)
        
        # Re-wrap into DataFrame before final fit
        self.base_estimator.fit(self._wrap(X_updated), y_updated, sample_weight=sample_weights)
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities using the base estimator.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return self.base_estimator.predict_proba(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict labels using the base estimator.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return self.base_estimator.predict(X)

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class RejectUpward(RejectInference):
    """
    Upward Augmentation strategy for reject inference.
    
    This strategy adjusts the weights of accepted applicants to represent the 
    rejected population. In the upward approach, the weights of the accepted 
    clients are updated to be indirectly proportional to the chances of a 
    client being accepted.
    """
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        
        # Wrap before fitting reject_estimator
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_concat[indices])

        sample_weights = self.reject_estimator.predict_proba(self._wrap(X))[
            :, 1
        ]  # probability of being accepted
        sample_weights = 1 / sample_weights  # upweighting the accepted samples
        
        return X, y, sample_weights


class RejectDownward(RejectInference):
    """
    Downward Augmentation strategy for reject inference.
    
    This strategy adjusts the weights of accepted applicants to represent the 
    rejected population. In the downward approach, the weights of the accepted 
    clients are updated to be directly proportional to the chances of a client 
    being rejected.
    """
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        
        # Wrap before fitting
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_concat[indices])

        sample_weights = self.reject_estimator.predict_proba(self._wrap(X))[
            :, 1
        ]  # probability of being accepted
        sample_weights = 1 - sample_weights

        return X, y, sample_weights


class RejectSoftCutoff(RejectInference):
    """
    Soft Cut-off Augmentation strategy for reject inference.
    
    This variation of augmentation adjusts the weights of accepted samples based 
    on a score of the group that the sample was placed in. For each group, the 
    augmentation factor is inversely proportional to the probability of a sample 
    in that group being from an accepted client.
    """
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])

        # shuffle data
        indices = np.arange(X_concat.shape[0])
        np.random.shuffle(indices)
        
        # Wrap before fitting
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_concat[indices])

        prob_accept = self.reject_estimator.predict_proba(self._wrap(X_concat))[:, 1]
        intervals = np.percentile(prob_accept, np.linspace(0, 100, num=100))
        sample_weights = np.ones(X_concat.shape[0])
        for i in range(len(intervals) - 1):
            in_interval = (prob_accept >= intervals[i]) & (
                prob_accept < intervals[i + 1]
            )
            if np.sum(in_interval) > 0:
                accept_rate = np.mean(y_concat[in_interval])  # A/(A+R)
                sample_weights[in_interval] = 1 / accept_rate if accept_rate > 0 else 1

        sample_weights = sample_weights[
            : X.shape[0]
        ]  # only keep weights for the accepted samples

        return X, y, sample_weights


class FuzzyParcelling(RejectInference):
    """
    Fuzzy-Parcelling strategy for reject inference.
    
    A variation of augmentation where rejected samples are duplicated. Half are 
    labeled as bad payers and the other half as good payers. The half labeled 
    as good receives a weight equal to the probability of being accepted, while 
    the half labeled as bad receives a weight equal to the probability of being 
    rejected. The weight of the accepted samples remains one.
    """
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Wrap before fitting base estimator internally
        self.base_estimator.fit(self._wrap(X), y)
        prob_1 = self.reject_estimator.predict_proba(self._wrap(X_unl))[:, 1]

        X_concat = np.concatenate([X, X_unl, X_unl])
        y_concat = np.concatenate(
            [y, np.zeros(X_unl.shape[0]), np.ones(X_unl.shape[0])]
        )
        sample_weights = np.concatenate([np.ones(X.shape[0]), 1 - prob_1, prob_1])
        return X_concat, y_concat, sample_weights


class RejectExtrapolation(RejectInference):
    """
    Extrapolation strategy for reject inference.
    
    Often called hard cutoff, simple augmentation, or parceling. This strategy 
    uses a model built on the accepted population to infer the labels of the 
    rejected samples. The rejected samples with their newly inferred labels are 
    then combined with the accepted population to train a new model.
    """
    def __init__(
        self,
        base_estimator: BaseEstimator,
        reject_estimator: BaseEstimator,
        mode: str = "positive",
    ):
        super().__init__(base_estimator, reject_estimator)
        self.mode = mode

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Wrap before fitting base estimator internally
        self.base_estimator.fit(self._wrap(X), y)
        prob_1 = self.reject_estimator.predict_proba(self._wrap(X_unl))[:, 1]
        if self.mode == "positive":
            selected = prob_1 > 0.5
        elif self.mode == "all":
            selected = np.ones_like(prob_1, dtype=bool)
        elif self.mode == "confident":
            selected = (prob_1 > 0.8) | (prob_1 < 0.15)

        Y_unl = (prob_1 > 0.5).astype(int)
        X_new = np.concatenate([X, X_unl[selected]])
        y_new = np.concatenate([y, Y_unl[selected]])
        sample_weights = np.ones(X_new.shape[0])
        return X_new, y_new, sample_weights


class RejectSpreading(RejectInference):
    """
    Label Spreading strategy for reject inference.
    
    A semi-supervised, graph-based technique assuming nearby samples are likely 
    to have the same labels. Rejected samples receive a temporary label of -1 
    and are concatenated with accepted samples. The Label Spreading algorithm is 
    fitted to this dataset to assign new labels to the rejected samples based on 
    their localization on the graph.
    """
    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([y, -1 * np.ones(X_unl.shape[0])])

        # LabelSpreading is purely array-based and usually doesn't throw feature 
        # names warnings, so standard ndarrays are fine here.
        label_spreading = LabelSpreading(kernel="knn")
        label_spreading.fit(X_concat, y_concat)
        
        y_new = label_spreading.transduction_
        sample_weights = np.ones(X_concat.shape[0])
        return X_concat, y_new, sample_weights