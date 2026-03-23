import numpy as np


from ezmodel.core.model import Model
from ezmodel.util.dist import calc_dist


class KNN(Model):

    def __init__(self, n_nearest=5, p=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_nearest = n_nearest
        self.p = p

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, out):
        _X, _y = self.X, self.y[:, 0]

        D = calc_dist(X, _X)

        I = D.argsort(axis=1)[:, :self.n_nearest]

        _d = np.take_along_axis(D, I, axis=1)
        _d = (_d ** self.p)

        _d[_d == 0] = 1e-64

        _d = 1 / _d
        _d = _d / _d.sum(axis=1)[:, None]

        _y = np.take_along_axis(self.y, I, axis=0)

        y = (_d * _y).sum(axis=1)

        out["y"] = y[:, None]

    @classmethod
    def hyperparameters(cls):
        return {
            "n_nearest": [3, 5, 10],
            "p": [0.0, 1.0, 2.0, 5.0]
        }
