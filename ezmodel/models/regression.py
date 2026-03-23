import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except:
    raise Exception("Model not found. Please execute: 'pip install sklearn'")

from ezmodel.core.model import Model


class PolynomialRegression(Model):

    def __init__(self, degree=3, fail_if_not_enough_points=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.degree = degree
        self.model = None
        self.fail_if_not_enough_points = fail_if_not_enough_points

    def _fit(self, X, y, **kwargs):

        n_min_points = PolynomialFeatures(self.degree).fit_transform(X).shape[1]

        if not self.fail_if_not_enough_points or len(X) >= n_min_points:
            model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), LinearRegression())
            model.fit(X, y[:, 0])
            self.model = model
        else:
            raise Exception(f"For Polynomial Regression of degree {self.degree} at least {n_min_points} are necessary.")

    def _predict(self, X, out):
        out["y"] = self.model.predict(X)[:, None]

    @classmethod
    def hyperparameters(cls):
        return {
            "degree": np.arange(1, 4)
        }
