import matplotlib.pyplot as plt
import numpy as np

from ezmodel.core.factory import models_from_clazzes
from ezmodel.core.selection import ModelSelection
from ezmodel.fit import fit
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.partitioning.random import RandomPartitioning
from ezmodel.util.sample_from_func import square_function

X, y, _X, _y = square_function(20, 200, n_var=1)

# select the best model from the given options
models = models_from_clazzes(RBF, Kriging)

model = fit(X, y, partitions=RandomPartitioning().do(X))

# predict the data using the model
y_hat = model.predict(_X)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat = model.predict(_X)

plt.scatter(X, y, label="Data_old_batch")
plt.plot(_X, y_hat, color="black", label="Regression")
plt.legend()
plt.show()
