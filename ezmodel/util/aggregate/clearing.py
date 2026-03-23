import numpy as np

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.util.clearing import EpsilonClearing
from pymoo.util.misc import vectorized_cdist, norm_eucl_dist_by_bounds
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Selection:

    def __init__(self, **kwargs) -> None:
        super().__init__()
        for k, v in kwargs.items():
            self.__dict__[k] = v


class MinSelection(Selection):

    def do(self, rem):
        return rem[self.F[rem].argmin()]


class RandomSelection(Selection):

    def do(self, rem):
        return rem[np.random.randint(len(rem))]


class MinMaxSelection(Selection):

    def __init__(self, min_eps=0.01, **kwargs) -> None:
        super().__init__(min_eps=min_eps, **kwargs)

    def do(self, rem):
        F, D = self.F[rem], self.D[rem][:, rem]

        _min = F.argmin()
        clearing = EpsilonClearing(D, self.min_eps)
        clearing.select(_min)

        _rem = clearing.remaining()
        if len(_rem) > 0:
            _max = _rem[F[_rem].argmax()]
            return [rem[_min], rem[_max]]
        else:
            return rem[_min]


class FrontwiseSelection(Selection):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        F = kwargs.get("F")
        G = kwargs.get("G")

        class MyProblem(Problem):
            def __init__(self, **kwargs):
                super().__init__(n_constr=0 if G is None else 1, **kwargs)

        pop = Population.new(index=np.arange(len(F)), F=F)

        if G is not None:
            pop.set("G", G)

        pop = RankAndCrowdingSurvival(nds=NonDominatedSorting()).do(MyProblem(), pop, n_survive=len(pop))

        self.rank = pop.get("rank")[pop.get("index")]
        self.crowding = pop.get("crowding")[pop.get("index")]

    def do(self, rem):
        _rank = self.rank[rem]
        _crowding = self.crowding[rem]
        I = np.lexsort([- _crowding, _rank])
        return rem[I[0]]


def aggregate_by_eps_clearing(X,
                              eps,
                              selection=MinSelection,
                              return_cluster=False,
                              func_dist=vectorized_cdist,
                              func_dist_by_bounds=norm_eucl_dist_by_bounds,
                              calc_distance=True,
                              problem=None,
                              xl=None,
                              xu=None,
                              **kwargs):
    if calc_distance:
        if problem is None:
            D = func_dist(X, X)
        else:
            if problem is not None:
                xl, xu = problem.bounds()
            D = func_dist_by_bounds(X, X, xl, xu)
    else:
        D = X

    clearing = EpsilonClearing(D, eps)

    sel = selection(problem=problem, X=X, D=D, **kwargs)

    D = {}

    while clearing.has_remaining():
        rem = clearing.remaining()
        S = sel.do(rem)

        if isinstance(S, list):
            for s in S:
                cleared = clearing.select(s)
                D[s] = cleared
        else:
            cleared = clearing.select(S)
            D[S] = cleared

    I = np.array(list(D.keys()))

    if return_cluster:
        return I, D
    else:
        return I
