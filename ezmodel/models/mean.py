import numpy as np

from ezmodel.core.model import Model


class SimpleMean(Model):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _fit(self, _, y, **kwargs):
        self.model = np.mean(y, axis=0)

    def _predict(self, X, out):
        out["y"] = np.full((len(X), 1), self.model)

    @classmethod
    def hyperparameters(cls):
        return {}


