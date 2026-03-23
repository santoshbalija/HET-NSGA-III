import numpy as np

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 1000)

from ezmodel.core.benchmark import Benchmark
from ezmodel.core.factory import models_from_clazzes
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning

X = np.random.random((100, 3)) * 2 * np.pi
y = np.sin(X).sum(axis=1)

models = models_from_clazzes(RBF, Kriging)

# set up the benchmark and add the models to be used
benchmark = Benchmark(models, n_threads=4, verbose=True, raise_exception=True)

# create partitions to validate the performance of each model
partitions = CrossvalidationPartitioning(k_folds=5, seed=1).do(X)

# runs the experiment with the specified partitioning
benchmark.do(X, y, partitions=partitions)

# print out the benchmark results
print(benchmark.statistics("mae", sort="ascending"))
