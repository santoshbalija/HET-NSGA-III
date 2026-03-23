from ezmodel.util.aggregate.clearing import MinSelection
from ezmodel.util.misc import discretize


def aggregate_by_grid(X,
                      n_partitions,
                      selection=MinSelection,
                      return_cluster=False,
                      problem=None,
                      xl=None,
                      xu=None,
                      **kwargs):

    if problem is not None:
        xl, xu = problem.bounds()

    _X = discretize(X, n_partitions, xl=xl, xu=xu)

    D = {}
    for i, x in enumerate(_X):
        s = str(x)
        if s not in D:
            D[s] = [i]
        else:
            D[s] = D[s] + [i]

    ret = {}
    for group in D.values():
        i = selection(problem=problem, **kwargs).do(group)
        ret[i] = [e for e in group if e != i]

    I = list(ret.keys())

    if return_cluster:
        return I, ret
    else:
        return I
