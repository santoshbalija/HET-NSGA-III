import random

import numpy as np


class Partitioning:

    def __init__(self, seed=None, include_trn_in_test=False) -> None:
        super().__init__()
        self.seed = seed
        self.include_trn_in_test = include_trn_in_test

    def do(self, X):

        # set the random seed to ensure reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        ret = self._do(X)

        if self.include_trn_in_test:
            for k, (trn, tst) in enumerate(ret):
                ret[k] = (trn, trn + tst)

        return ret

    def _do(self, X):
        pass


def merge_and_partition(trn, tst):
    X, y = trn
    X_test, y_test = tst
    partitions = [(np.arange(len(X)), np.arange(len(X), len(X) + len(X_test)))]
    X, y = np.row_stack([X, X_test]), np.concatenate([y, y_test])
    return X, y, partitions
