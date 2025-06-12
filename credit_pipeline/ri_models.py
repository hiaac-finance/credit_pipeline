import numpy as np
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading as SkLabelSpreading
from sklearn.linear_model import LogisticRegression


class RejectInferenceAlg:
    """Base class for reject inference algorithms.

    Parameters
    ----------
    clf : Sklearn-like classifier
        Classifier to be used for the reject inference algorithm.
    accept_clf : Sklearn-like classifier, optional
        Classifier to be used for acceptance inference, by default None. If None, it will use a LogisticRegression classifier.
    """

    def __init__(self, clf, accept_clf=None):
        self.clf = clf
        if accept_clf is None:
            self.accept_clf = LogisticRegression()
        else:
            self.accept_clf = accept_clf

    def fit(self, X, y):
        """Execute reject inference algorithm to update data and sample weights,
        then fit the classifier.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Features of the training data.
        y : np.ndarray or pd.Series
            Labels of the training data, where -1 indicates unlabeled data.

        Returns
        -------
        self : RejectInferenceAlg
            Fitted instance of the reject inference algorithm.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # separate X and X_unl based on y
        X, X_unl = X[y != -1], X[y == -1]
        y = y[y != -1]

        X, y, sample_weights = self.update_data(X, y, X_unl)
        self.clf.fit(X, y, sample_weight=sample_weights)

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def update_data(self, X, y, X_unl):
        """Update the training data with the unlabeled data.
        Should be implemented in subclasses.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Features of the labeled training data.
        y : np.ndarray or pd.Series
            Labels of the labeled training data.
        X_unl : np.ndarray or pd.DataFrame
            Features of the unlabeled training data.

        Returns
        -------
        X : np.ndarray
            Updated features of the training data.
        y : np.ndarray
            Updated labels of the training data.
        sample_weights : np.ndarray
            Sample weights for the training data.
        """
        return X, y, np.ones(X.shape[0])


class AugSoftCutoff(RejectInferenceAlg):
    def __init__(self, clf, accept_clf=None):
        super().__init__(clf, accept_clf)

    def update_data(self, X, y, X_unl):
        X_complete = np.concatenate([X, X_unl])
        y_complete = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        self.accept_clf.fit(X_complete, y_complete)

        # Get the probabilitie of being approved
        prob_accept = self.accept_clf.predict_proba(X_complete)[:, 1]

        n_scores_interv = 100

        # Sort the probabilities of being accepted
        prob_accept_asc = np.argsort(prob_accept)
        # #Split the probs in intervals
        score_interv = np.array_split(prob_accept_asc, n_scores_interv)

        # Create array for accepts weights
        sample_weights = np.ones(X.shape[0])
        for acceptees in score_interv:
            # remove the indices of rejects
            acceptees = np.array([i for i in acceptees if i < X.shape[0]])
            if len(acceptees) >= 1:
                # Augmentation Factor (Weight) for the split
                af = y_complete[acceptees].mean()  # A/(A+R)
                af = 1 / np.maximum(af, 1e-5)  # Avoid division by zero
                sample_weights[acceptees] = af

        return X, y, sample_weights


class AugUpDown(RejectInferenceAlg):
    def __init__(self, clf, accept_clf=None, method="up"):
        super().__init__(clf, accept_clf)
        assert method in ["up", "down"], "Method must be 'up' or 'down'."
        self.method = method

    def update_data(self, X, y, X_unl):
        X_complete = np.concatenate([X, X_unl])
        y_complete = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        self.accept_clf.fit(X_complete, y_complete)

        # Weights are the probabilitie of being approved
        weights = self.accept_clf.predict_proba(X_unl)[:, 1]

        if self.method == "up":
            # Upward: ŵ = w/p(A)
            sample_weights = 1 / weights[: X.shape[0]]
        elif self.method == "down":
            # Downward: ŵ = w * (1 - p(A))
            sample_weights = 1 * (1 - weights[: X.shape[0]])

        return X, y, sample_weights


class AugFuzzy(RejectInferenceAlg):
    def __init__(self, clf, accept_clf=None):
        super().__init__(clf, accept_clf)

    def update_data(self, X, y, X_unl):
        X_fuzzy = np.concatenate([X, X_unl, X_unl])
        y_fuzzy = np.concatenate([y, np.zeros(X_unl.shape[0]), np.ones(X_unl.shape[0])])

        # self.accept_clf.fit(X_fuzzy, y_fuzzy)
        self.accept_clf.fit(X, y)

        unl_0_weights = self.accept_clf.predict_proba(X_unl)[:, 0]
        unl_1_weights = self.accept_clf.predict_proba(X_unl)[:, 1]

        train_weights = np.ones(y.shape[0])
        sample_weights = np.concatenate([train_weights, unl_0_weights, unl_1_weights])

        return X_fuzzy, y_fuzzy, sample_weights


class Extrapolation(RejectInferenceAlg):
    def __init__(self, clf, accept_clf=None, augmentation_type="only_1"):
        super().__init__(clf, accept_clf)
        assert augmentation_type in [
            "only_1",
            "all",
            "confident",
        ], "Extrapolation type must be 'only_1', 'all' or 'confident'."
        self.augmentation_type = augmentation_type

    def update_data(self, X, y, X_unl):
        self.clf.fit(X, y)

        y_prob_unl = self.clf.predict_proba(X_unl)[:, 1]
        if self.augmentation_type == "only_1":
            X_combined = np.concatenate([X, X_unl[y_prob_unl >= 0.5]])
            n_new = (y_prob_unl >= 0.5).sum()
            y_combined = np.concatenate([y, np.ones(n_new)])
        elif self.augmentation_type == "all":
            X_combined = np.concatenate([X, X_unl])
            y_combined = np.concatenate([y, (y_prob_unl >= 0.5).astype(int)])
        elif self.augmentation_type == "confident":
            X_combined = np.concatenate(
                [X, X_unl[(y_prob_unl > 0.8) | (y_prob_unl < 0.15)]]
            )
            y_new = y_prob_unl[(y_prob_unl > 0.8) | (y_prob_unl < 0.15)] >= 0.5
            y_combined = np.concatenate([y, y_new.astype(int)])

        return X_combined, y_combined, np.ones(X_combined.shape[0])


class LabelSpreading(RejectInferenceAlg):
    def __init__(self, clf, accept_clf=None):
        super().__init__(clf, accept_clf)

    def update_data(self, X, y, X_unl):
        X_combined = np.concatenate([X, X_unl])
        y_unl = np.ones(X_unl.shape[0]) * -1  # Unlabeled data gets -1
        y_combined = np.concatenate([y, y_unl])

        ls = SkLabelSpreading(
            alpha=0.2,
            gamma=20,
            kernel="knn",
            max_iter=30,
            n_jobs=None,
            n_neighbors=7,
            tol=0.001,
        )
        ls.fit(X_combined, y_combined)
        predicted_labels = ls.transduction_

        # create new y with the original labels and the predicted labels for unlabeled data
        n_labels = y.shape[0]
        y_combined = np.concatenate([y, predicted_labels[n_labels:]])

        return X_combined, y_combined, np.ones(X_combined.shape[0])
