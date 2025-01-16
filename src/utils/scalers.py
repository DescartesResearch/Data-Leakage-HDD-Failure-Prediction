from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, depth=1):
        assert depth > 0
        self.offsets = [1] * depth
        self.min_max_scaler = MinMaxScaler()
        self.depth = depth

    def apply_log(self, X):
        for o in self.offsets:
            X = np.log(X + o)
        return X

    def inverse_log(self, X):
        for o in reversed(self.offsets):
            X = np.exp(X) - o

    def fit(self, X, y=None):
        self.min_max_scaler.fit(self.apply_log(X))
        self.n_features_in_ = self.min_max_scaler.n_features_in_
        return self

    def transform(self, X, y=None):
        return self.min_max_scaler.transform(self.apply_log(X))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X, y=None):
        return self.inverse_log(self.min_max_scaler.inverse_transform(X))
