from copy import deepcopy

import numpy as np
from ezmodel.core.benchmark import Benchmark
from ezmodel.core.partitioning import merge_and_partition
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning
from pymoo.core.population import Population
from pymoo.util.misc import from_dict
from pymoo.util.sliding_window import SlidingWindow

from pysamoo.core.indicator import INDICATORS
from ezmodel.util.transformation.plog import Plog


class Target:

    def __init__(self,
                 label,
                 models,
                 n_folds=5,
                 n_max_performances=5,
                 n_max_benchmarks=0,
                 indicators=INDICATORS):
        """

        Parameters
        ----------
        label : tuple
            This describes the type and what role it plays in the problem later on. For instance, if this target
            models the first constraint ('G', 0) would be the corresponding type.

        models : list
            A list of all models which are kept track of for this target.

        n_folds : int
            The number of folds if cross-validation is applied (happens only the first time)

        n_max_performances : int
            The number of maximum last performances to make a decision

        n_max_benchmarks : int
            The maximum number of benchmarks (only to review later)

        indicators : list
            A list of tuples (name, sign, func) defining what indicators should be calculated after
            each of the benchmarks.

        """

        self.label = label
        self.models = models
        self.n_folds = n_folds

        # the indicators to be calculated for this target
        self.indicators = indicators

        # this is the storage for the most recent benchmarks - full experiment
        self.benchmarks = SlidingWindow(size=n_max_benchmarks)

        # this keeps track of the past performances - each model gets one
        self.performances = {}
        for key in models.keys():
            self.performances[key] = SlidingWindow(size=n_max_performances)

        # the current name of the model which is used
        self.model = None

        # the actual model fitting the data points provided
        self.obj = None

        # the best model set by the last validation
        self.best = None

        self.plog = False

    def validate(self, trn, tst=None, find_best=True, **kwargs):

        # get the values to be predicted
        X, y = trn.get("X"), self._get_y(trn)

        # if np.mean(y) >1e1:
        #
        #     self.plog = True
        #     y = Plog().forward(y)
        #     print("Plog applied to the target.")
        #     # print("Plog", y)


        if tst is None:
            X, y, partitions = X, y, CrossvalidationPartitioning(self.n_folds).do(len(trn))
        else:
            _X, _y = tst.get("X"), self._get_y(tst)
            X, y, partitions = merge_and_partition((X, y), (_X, _y))

        # do the benchmark for the specific target given the partitions
        obj = Benchmark(self.models, n_threads=len(self.models), raise_exception=False).do(X, y, partitions=partitions)
        benchmark = obj.results(only_successful=False, as_list=False, include_metadata=True)
        self.benchmarks.append(benchmark)

        for model in self.models.keys():

            results = benchmark["results"][model]

            # for each of the run, execute all the performance indicators
            for run in results["runs"]:
                ret = self._indicators(benchmark, run)
                self.performances[model].append(ret)

        # if a list of indicators for selecting the best model is provided, do it
        if find_best:
            self.best = self.find_best(**kwargs)

    def validate_modified(self, trn, tst=None, find_best=True, **kwargs):

        # get the values to be predicted
        X, y = trn.get("X"), self._get_y(trn)


        # remove the nan values
        indices = np.argwhere(~np.isnan(y))
        index = [item[0] for item in indices]
        X = X[index]
        y = y[index]
        trn = trn[index]
        # if np.mean(y) > 1e1:
        #     self.plog = True
        #     y = Plog().forward(y)
        #     print("Plog applied to the target.")
        #     # print("Plog", y)


        if tst is None:
            X, y, partitions = X, y, CrossvalidationPartitioning(self.n_folds).do(len(trn))
        else:
            _X, _y = tst.get("X"), self._get_y(tst)
            # remove the nan values
            indices = np.argwhere(~np.isnan(_y))
            index = [item[0] for item in indices]
            _X = _X[index]
            _y = _y[index]
            tst = tst[index]

            # if there is no test set, use the training set and do cross-validation
            # else use the test set and do a single benchmark
            if len(tst) == 0:
                X, y, partitions = X, y, CrossvalidationPartitioning(self.n_folds).do(len(trn))
            else:
                X, y, partitions = merge_and_partition((X, y), (_X, _y))

        # do the benchmark for the specific target given the partitions
        obj = Benchmark(self.models, n_threads=len(self.models), raise_exception=False).do(X, y, partitions=partitions)
        benchmark = obj.results(only_successful=False, as_list=False, include_metadata=True)
        self.benchmarks.append(benchmark)

        for model in self.models.keys():

            results = benchmark["results"][model]

            # for each of the run, execute all the performance indicators
            for run in results["runs"]:
                ret = self._indicators(benchmark, run)
                self.performances[model].append(ret)

        # if a list of indicators for selecting the best model is provided, do it
        if find_best:
            self.best = self.find_best(**kwargs)

    def find_best(self, indicator=["kendall_tau", "mae"], exclude=[]):

        models = [model for model in self.models.keys() if model not in exclude]
        perf = self.performances

        # filter the models if one run as failed during benchmark
        models = np.array([m for m in models if None not in perf[m]])

        assert len(models) > 0, "Fitting each of the models has failed at least once in the benchmark."

        for entry in indicator:

            # get the performances from the the n_max_performance iterations
            v = np.array([self.performance(entry, model=model) for model in models])

            # multiply by the sign to consider minimization and maximization
            v *= self.indicators[entry]["sign"]

            # find the best model or models - if there is a tie
            models = models[v == np.min(v)]

            if len(models) == 1:
                break

        # finally find the best model using the indicator
        return np.random.choice(models)

    def fit(self, sols):
        assert self.best is not None, "You need to do one initial validation to find the best model for this target."

        # finally fit the model with the data
        obj = deepcopy(self.models[self.best])

        # get in and output
        X, y = sols.get("X"), self._get_y(sols)
        # if np.mean(y) > 1e1:
        #     self.plog = True
        #     y = Plog().forward(y)
        #     print("Plog applied to the target.")
        #     # print("Plog", y)

        obj.fit(X, y)

        self.model = self.best

        self.obj = obj

    def performance(self, indicator, model=None, func=np.mean):
        if model is None:
            model = self.best

        assert model is not None, "Either provide a model or set one through validation."

        v = np.array([e[indicator] for e in self.performances[model]])

        if func:
            v = func(v)

        return v

    def predict(self, X, out):
        assert self.obj is not None, "The target has not been fitted yet."
        #print("Fitted model", self.model)
        v, s = self.obj.predict(X, return_values_of=["y", "sigma"])
        # if self.plog:
        #     v = Plog().backward(v)
        #     s = Plog().backward(s)
        #     print("Plog reverse applied to the target.")
            # print("Plog backward", v)
        #print("Predicted", v.shape)
        key, index = self.label
        out.get(key)[:, [index]] = v
        out.get(key + "_sigma")[:, [index]] = s


    def _indicators(self, benchmark, run):

        X, y, partitions = from_dict(benchmark, "X", "y", "partitions")

        if run["success"]:
            ret = {}

            partition = run["partition"]
            trn, tst = partitions[partition]
            #print(len(trn), len(tst))

            y_true = y[tst]
            y_hat = run["y_hat"]

            for name, v in self.indicators.items():
                ret[name] = v["func"](y_true, y_hat, trn_y=y[trn])

            return ret

    def _get_y(self, sols):
        """
        Parameters
        ----------
        sols : Population
            A set of solutions.

        Returns
        -------
        Y : np.array
            The Y values to be modeled by the surrogate

        """
        key, index = self.label
        return sols.get(key)[:, index]

    def __repr__(self) -> str:
        return f"({self.label[0]}, {self.label[1]}) : {super(Target, self).__repr__()}"
