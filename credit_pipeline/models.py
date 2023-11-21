from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """MLPClassifier in the Sklearn API using Keras.
    It mimics the MLPClassifier from Sklearn, but it uses Keras to train the model.
    The extra functionalities are the the possibility to use class weights and sample weights.

    :param hidden_layer_sizes: list of hidden layer sizes with n_layers-2 values, defaults to (100,)
    :type hidden_layer_sizes: tuple, optional
    :param batch_size: batch size used for training and inference, defaults to 32
    :type batch_size: int, optional
    :param init_learning_rate: initial learning rate used in Adam, defaults to 0.1
    :type learning_rate: float, optional
    :param learning_rate_decay_rate: learning rate decay rate, defaults to 0.1
    :type learning_rate_decay_rate: float, optional
    :param alpha: weight of L2 regularization, defaults to 0.0001
    :type alpha: float, optional
    :param epochs: number of epochs to train model, defaults to 100
    :type epochs: int, optional
    :param class_weight: if want to use class weights in the loss, pass the value "balanced", defaults to None
    :type class_weight: string or None, optional
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        batch_size=32,
        init_learning_rate=0.1,
        learning_rate_decay_rate=0.1,
        alpha=0.0001,
        epochs=100,
        class_weight=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.alpha = alpha
        self.epochs = epochs
        self.class_weight = class_weight

    def set_model(self, X):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                self.hidden_layer_sizes[0],
                input_dim=X.shape[1],
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(self.alpha),
            )
        )
        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(
                keras.layers.Dense(
                    layer_size,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(self.alpha),
                )
            )
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            self.init_learning_rate,
            decay_steps=self.epochs,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["AUC"],
        )
        return model

    def fit(self, X, y, sample_weight=None):
        if self.class_weight is None:
            self.class_weight = {0: 1, 1: 1}
        elif self.class_weight == "balanced":
            self.class_weight = {0: 1 / sum(y == 0), 1: 1 / sum(y == 1)}

        self.model = self.set_model(X)
        self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
            verbose=0,
        )

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X) > 0.5
    
    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]
        
        
