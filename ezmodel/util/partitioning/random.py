import math

import numpy as np

from ezmodel.core.partitioning import Partitioning


class RandomPartitioning(Partitioning):

    def __init__(self,
                 perc_train=0.7,
                 n_sets=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_sets = n_sets
        self.perc_train = perc_train

    def _do(self, X):
        n = len(X) if not isinstance(X, int) else X

        n_train = math.ceil(self.perc_train * n)

        partitions = [rnd(n, n_train) for _ in range(self.n_sets)]

        return partitions


def rnd(n, k):
    M = np.random.permutation(n)
    return M[:k], M[k:]
