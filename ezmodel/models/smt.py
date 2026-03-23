import numpy as np

try:
    from smt.surrogate_models import KRG
except:
    raise Exception("Model not found. Please execute: 'pip install smt'")

from ezmodel.core.model import Model


class smtKriging(Model):

    def __init__(self, theta0=1e-4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta0 = theta0

    def _fit(self, X, y, **kwargs):
        n_var = X.shape[1]

        sm = KRG(theta0=[self.theta0] * n_var)
        sm.set_training_values(X, y)
        sm.train()

        self.model = sm

    def _predict(self, X, out, **kwargs):

        if "y" in out:
            out["y"] = self.model.predict_values(X)

        if "sigma" in out:
            var = self.model.predict_variances(X)

            var[var <= 0] = 0
            out["var"] = var
            out["sigma"] = np.sqrt(var)

    @classmethod
    def hyperparameters(cls):
        return {

        }
