import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except:
    raise Exception("Model not found. Please execute: 'pip install sklearn'")

from ezmodel.core.model import Model
from ezmodel.util.misc import discretize


class RandomForest(Model):

    def __init__(self, n_partitions=5, n_estimators=30, xl=None, xu=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.xl, self.xu = xl, xu
        self.n_partitions = n_partitions
        self.n_estimators = n_estimators

    def _fit(self, X, y, **kwargs):

        if self.xl is None:
            self.xl = X.min(axis=0)
        if self.xu is None:
            self.xu = X.max(axis=0)

        y = y[:, 0]

        X = discretize(X, self.n_partitions, self.xl, self.xu)

        D = {}
        for i, x in enumerate(X):
            s = str(x)
            if s not in D:
                D[s] = dict(X=x, y=y[i], n=1)
            else:
                _y, _n = D[s]['y'], D[s]['n']
                D[s] = dict(X=x, y=min(_y, y[i]), n=_n + 1)

        X = np.zeros((len(D), X.shape[1]))
        y = np.zeros(len(D))
        for i, e in enumerate(D.values()):
            X[i] = e["X"]
            y[i] = e["y"]

        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        rf.fit(X, y)

        self.model = rf

    def _predict(self, X, out):
        out["y"] = self.model.predict(discretize(X, self.n_partitions, self.xl, self.xu))[:, None]

    @classmethod
    def hyperparameters(cls):
        return {
            "n_partitions": [20, 50, 100],
            "n_estimators": [10, 100, 200]
        }
