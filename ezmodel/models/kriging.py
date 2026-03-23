import sys
from inspect import getmembers

import numpy as np
from pydacefit.dace import DACE

from ezmodel.core.model import Model


def get_corr(corr):
    for name, func in getmembers(sys.modules["pydacefit.corr"]):
        if name == "corr_" + corr:
            return func
    raise Exception("Correlation not found.")


def get_regr(corr):
    for name, func in getmembers(sys.modules["pydacefit.regr"]):
        if name == "regr_" + corr:
            return func
    raise Exception("Regression not found not found.")


class Kriging(Model):

    def __init__(self,
                 regr="linear",
                 corr="gauss",
                 ARD=False,
                 theta=1.0,
                 thetaL=0.00001,
                 thetaU=100.0,
                 **kwargs) -> None:

        super().__init__(eliminate_duplicates=True, **kwargs)
        self.regr = regr
        self.corr = corr
        self.ARD = ARD
        self.theta = theta
        self.thetaL = thetaL
        self.thetaU = thetaU

    def _fit(self, X, y, **kwargs):
        func_regr, func_corr = get_regr(self.regr), get_corr(self.corr)
        theta, thetaL, thetaU = self.theta, self.thetaL, self.thetaU

        if self.ARD:
            if self.thetaL is None or self.thetaU is None:
                pass
                # raise Exception("Bounds of theta must be given for Automatic Relevance Detection (ARD)!")
            else:
                _, m = X.shape
                theta = np.full(m, theta)
                thetaL = np.full(m, thetaL)
                thetaU = np.full(m, thetaU)

        self.model = DACE(regr=func_regr, corr=func_corr, theta=theta, thetaL=thetaL, thetaU=thetaU)
        self.model.fit(X, y)

    def _predict(self, X, out, **kwargs):
        calc_sigma = "sigma" in out
        calc_variance = "var" in out

        return_mse = calc_variance or calc_sigma






        ret = self.model.predict(X, return_mse=return_mse, **kwargs)

        if not return_mse:
            out["y"] = ret
        else:
            out["y"], var = ret
            var[var <= 0] = 0
            out["var"], out["sigma"] = var, np.sqrt(var)

    @classmethod
    def hyperparameters(cls):
        return {
            "regr": ["constant", "linear"],
            "corr": ["gauss", "cubic", "exp"],
            "thetaU": [20, 100],
            "ARD": [False, True]
        }
