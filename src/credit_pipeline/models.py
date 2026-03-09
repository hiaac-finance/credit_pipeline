import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers.schedules import InverseTimeDecay
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """MLPClassifier in the Sklearn API using Keras.
    It mimics the MLPClassifier from Sklearn, but it uses Keras to train the model.
    The extra functionalities are the the possibility to use class weights and sample weights.

    Parameters
    ----------
    hidden_layer_sizes : tuple, optional
            List of hidden layer sizes as a tuple with has n_layers-2 elements, by default (100,)
        batch_size : int, optional
            Size of batch for training, by default 32
        learning_rate_init : float, optional
            Initial learning rate, by default 0.1
        learning_rate_decay_rate : float, optional
            Decay rate of learning rate, equal to 1 to constant learning rate, by default 0.1
        alpha : float, optional
            Weight of L2 regularization, by default 0.0001
        epochs : int, optional
            Number of epochs to train model, by default 100
        class_weight : string, optional
            If want to use class weights in the loss, pass the value "balanced", by default None
        random_state : int, optional
            Random seed, by default None
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        batch_size=32,
        learning_rate_init=0.1,
        learning_rate_decay_rate=0.1,
        alpha=0.0001,
        epochs=100,
        class_weight=None,
        random_state=None,
    ):
        self._random_state = random_state
        self._seed_everything(random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.alpha = alpha
        self.epochs = epochs
        self.class_weight = class_weight

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value
        self._seed_everything(value)

    def _seed_everything(self, value):
        if value is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            keras.utils.set_random_seed(self.random_state)
            tf.config.experimental.enable_op_determinism()

    def set_model(self, X):
        model = Sequential()
        model.add(
            Dense(
                self.hidden_layer_sizes[0],
                input_dim=X.shape[1],
                activation="relu",
                kernel_regularizer=l2(self.alpha),
            )
        )
        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(
                Dense(
                    layer_size,
                    activation="relu",
                    kernel_regularizer=l2(self.alpha),
                )
            )
        model.add(Dense(1, activation="sigmoid"))
        lr_schedule = InverseTimeDecay(
            self.learning_rate_init,
            decay_steps=self.epochs,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule),
            metrics=["AUC"],
        )
        return model

    def fit(self, X, y, sample_weight=None):
        if self.class_weight == "balanced":
            self.class_weight = {0: 1 / sum(y == 0), 1: 1 / sum(y == 1)}
        self.model = self.set_model(X)
        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            class_weight=self.class_weight if sample_weight is None else None,
            sample_weight=sample_weight,
            verbose=0,
        )

    def predict_proba(self, X):
        # prob = self.model.predict(X, verbose=0)
        prob = self.model(X.values, training=False)
        return np.concatenate([1 - prob, prob], axis=1)

    def predict(self, X):
        return self.model(X.values, training=False) > 0.5

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]
