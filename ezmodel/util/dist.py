import numpy as np


def eucl_dist(u, v, alpha=2.0):
    return ((u - v) ** alpha).sum(axis=1)


def calc_dist(X, Z, func_dist=eucl_dist, **kwargs):

    # prepare the left and right for distances
    u = np.repeat(X, Z.shape[0], axis=0)
    v = np.tile(Z, (X.shape[0], 1))

    # calculate the distances and reshape to distance metric
    D = func_dist(u, v, **kwargs)
    D = np.reshape(D, (X.shape[0], Z.shape[0]))

    return D

