

try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR as _SVR
except:
    raise Exception("Model not found. Please execute: 'pip install sklearn'")


from ezmodel.core.model import Model


class SVR(Model):

    def __init__(self, kernel="rbf", eps=0.1, C=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.eps = eps
        self.C = C

    def _fit(self, X, y, **kwargs):
        svr = _SVR(kernel=self.kernel, epsilon=self.eps, C=self.C, gamma='scale', degree=3, tol=0.001, shrinking=True)
        regr = make_pipeline(StandardScaler(), svr)
        regr.fit(X, y[:, 0])
        self.model = regr

    def _predict(self, X, out):
        out["y"] = self.model.predict(X)[:, None]

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "eps": [0.1, 0.01],
            "C": [0.5, 1.0, 2.0]
        }
