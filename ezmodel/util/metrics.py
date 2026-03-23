import inspect
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

METRICS = ["mse", "mae", "r2", "spear"]


def calc_metric(metric, y, y_hat):
    check_equal_shape(y, y_hat)

    funcs = dict(inspect.getmembers(sys.modules[__name__]))

    if metric not in funcs:
        raise Exception("Metric is not known.")
    else:
        return funcs[metric](y, y_hat)


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def r2(y, y_hat):
    return r2_score(y, y_hat)


def spear(y, y_hat):
    return spearmanr(y, y_hat).correlation


# --------------------------------------------------------------
# Util
# --------------------------------------------------------------

def check_equal_shape(a, b):
    assert a.shape == b.shape

