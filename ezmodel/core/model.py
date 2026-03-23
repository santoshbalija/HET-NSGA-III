import time

import numpy as np

from ezmodel.core.transformation import NoNormalization
from ezmodel.util.misc import is_duplicate, at_least2d


class Model:

    def __init__(self,
                 norm_X=NoNormalization(),
                 norm_y=NoNormalization(),
                 active_dims=None,
                 filter_nan_and_inf=True,
                 eliminate_duplicates=False,
                 eliminate_duplicates_eps=1e-16,
                 raise_exception_while_fitting=True,
                 raise_exception_while_prediction=True,
                 verbose=False,
                 **kwargs):

        self.norm_X = norm_X
        self.norm_y = norm_y

        self.eliminate_duplicates = eliminate_duplicates
        self.eliminate_duplicates_eps = eliminate_duplicates_eps

        self.active_dims = active_dims
        self.filter_nan_and_inf = filter_nan_and_inf
        self.verbose = verbose

        self.time = None
        self.model = None
        self.X, self.y = None, None
        self._X, self._y = None, None
        self.success = None
        self.data = {}

        self.raise_exception_while_fitting = raise_exception_while_fitting
        self.raise_exception_while_prediction = raise_exception_while_prediction
        self.exception = None

        self.has_been_fitted = False

    def preprocess(self, X, y, **kwargs):

        if self.active_dims is not None:
            X = X[:, self.active_dims]

        if self.eliminate_duplicates:
            I = ~is_duplicate(X, eps=self.eliminate_duplicates_eps)
            X, y = X[I], y[I]

        if self.filter_nan_and_inf:
            X_I = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
            y_I = np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
            X, y = X[X_I & y_I], y[X_I & y_I]

        X, y = self.norm_X.forward(X), self.norm_y.forward(y)
        X, y = self._preprocess(X, y, **kwargs)

        return X, y

    def postprocess(self, out, **kwargs):
        if "y" in out:
            out["y"] = self.norm_y.backward(out["y"])
        out = self._postprocess(out, **kwargs)
        return out

    def fit(self, X, y, **kwargs):
        X, y = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        assert len(X) == len(y)
        self._X, self._y = X, y

        self.X, self.y = self.preprocess(X, y)

        start = time.time()

        try:

            # fit the model given the data
            self._fit(self.X, self.y, **kwargs)

            # do some parameter optimization if the model requires it
            self._optimize(**kwargs)

            # if no exception occurs set the model to be fitted successfully
            self.success = True

        except Exception as ex:
            self.success = False
            self.exception = ex
            if self.raise_exception_while_fitting:
                raise ex

        self.has_been_fitted = True
        self.time = time.time() - start

        return self

    def predict(self, X,
                return_values_of=["y"],
                return_as_dictionary=False,
                **kwargs):

        if not self.success:
            if self.raise_exception_while_fitting:
                raise Exception("There was an error while fitting the model.")
            else:
                return np.full(len(X), np.nan)

        if self.active_dims is not None:
            X = X[:, self.active_dims]

        # normalize the input
        X = self.norm_X.forward(at_least2d(X, expand="r"))

        # write in the output dictionary what should be returned
        out = {}
        for k in return_values_of:
            out[k] = None

        try:
            # get the prediction from the actual implementation
            self._predict(X, out, **kwargs)

            # do the post processing of the outputs
            out = self.postprocess(out, **kwargs)

        except Exception as e:
            if self.raise_exception_while_prediction:
                raise e
            else:
                out["y"] = np.full(len(X), np.inf)

        if return_as_dictionary:
            return out
        else:
            ret = tuple([out[v] for v in return_values_of])
            return ret if len(ret) > 1 else ret[0]

    @classmethod
    def hyperparameters(cls):
        return {}

    def _preprocess(self, X, y, **kwargs):
        return X, y

    def _postprocess(self, out, **kwargs):
        return out

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, out, **kwargs):
        pass

    def _optimize(self, **kwargs):
        pass
