from ezmodel.core.model import Model
from ezmodel.experimental.custom.kernel import dist


class InverseDistanceWeighting(Model):

    def __init__(self, p=2.0, eps=1e-32, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p
        self.eps = eps

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, out, **kwargs):
        _X, _y = self.X, self.y[:, 0]

        D = dist(X, _X)
        D[D <= self.eps] = self.eps

        w = 1 / D ** self.p

        for k, d in enumerate(D):
            is_zero = (d <= self.eps)
            if is_zero.sum() > 0:
                w[k, ~is_zero] = 0
                w[k, is_zero] = 1

        w = w / w.sum(axis=1)[:, None]

        y = (_y * w).sum(axis=1)

        out["y"] = y[:, None]

    @classmethod
    def hyperparameters(cls):
        return {}
