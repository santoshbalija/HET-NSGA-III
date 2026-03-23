from copy import copy

from ezmodel.core.benchmark import Benchmark
from ezmodel.core.model import Model


class ModelSelection(Model):

    def __init__(self,
                 benchmark_or_models,
                 refit=True,
                 max_trn_error=None,
                 sorted_by=None,  # for instance ("mse", "mean", True)
                 ) -> None:

        super().__init__()

        if not isinstance(benchmark_or_models, Benchmark):
            self.benchmark = Benchmark(benchmark_or_models)
        else:
            self.benchmark = benchmark_or_models

        self.max_trn_error = max_trn_error
        self.sorted_by = sorted_by

        self.refit = refit

        self.data = None
        self.best = None
        self.model = None

    def do(self, X, y, *args, return_result=False, **kwargs):

        if self.benchmark.data is None:
            self.benchmark.do(X, y, *args, **kwargs)

        R = self.benchmark.results(only_successful=True,
                                   as_list=True,
                                   sorted_by=self.sorted_by,
                                   max_trn_error=self.max_trn_error)

        if len(R) == 0:
            raise Exception("No model could have been fitted successfully!")

        best = R[0]

        if self.refit:
            model = copy(best["model"])
            model.fit(X, y)
        else:
            if best["n_runs"] > 1:
                raise Exception("refit is necessary of the benchmark has multiple partitions!")
            else:
                model = best["runs"][0]["model"]

        self.data = R
        self.best = best
        self.model = model

        if not return_result:
            return model
        else:
            return model, best

    def statistics(self):
        if self.data is None:
            return None
        else:
            import pandas as pd
            data = [(e["label"], e["metric"]) for e in self.data]
            df = pd.DataFrame(data, columns=["label", self.metric])
            return df

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.do(X, y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        return self.model

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)
