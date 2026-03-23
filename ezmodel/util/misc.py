import numpy as np
from scipy.spatial.distance import cdist


def all_except(x, *args):
    if len(args) == 0:
        return x
    else:
        H = set(args) if len(args) > 5 else args
        I = [k for k in range(len(x)) if k not in H]
        return x[I]


def from_dict(D, *keys):
    return [D.get(k) for k in keys]


def is_duplicate(X, eps=1e-16):
    D = cdist(X, X)
    D[np.triu_indices(len(X))] = np.inf

    I = np.full(len(X), False)
    I[np.any(D < eps, axis=1)] = True
    return I


def to_list(v):
    if not isinstance(v, list):
        v = [v]
    return v


def at_least2d(x, expand="c"):
    if x.ndim == 1:
        if expand == "c":
            return x[:, None]
        elif expand == "r":
            return x[None, :]
    else:
        return x


def empty_dict_if_none(x):
    if x is None:
        return {}
    else:
        return x


def dict_to_str(vals, delim=",", sep="="):
    return delim.join(f'{key}{sep}{value}' for key, value in vals.items())


def discretize(X, n_partitions, xl=None, xu=None):
    if xl is None:
        xl = X.min(axis=0)

    if xu is None:
        xu = X.max(axis=0)

    thresholds = np.linspace(xl, xu, n_partitions + 1)[1:]
    ret = (X[..., None] < thresholds.T).argmax(axis=-1)
    return ret
