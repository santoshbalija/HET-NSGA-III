import math
import random

from ezmodel.core.partitioning import Partitioning


class CrossvalidationPartitioning(Partitioning):

    def __init__(self,
                 k_folds=5,
                 randomize=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.randomize = randomize
        self.k_folds = k_folds

    def _do(self, X):
        n = len(X) if not isinstance(X, int) else X
        assert n > 1

        k_folds = min(self.k_folds, n)

        indices = list(range(n))
        if self.randomize:
            random.shuffle(indices)

        tst = [set() for _ in range(k_folds)]
        for k in range(n):
            tst[k % k_folds].add(indices[k])

        trn = [[j for j in indices if j not in tst[i]] for i in range(k_folds)]

        ret = []
        for _trn, _tst in zip(trn, tst):
            ret.append((_trn, list(_tst)))

        return ret
