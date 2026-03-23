import matplotlib.pyplot as plt
import numpy as np

from ezmodel.models.kriging import Kriging
from ezmodel.util.sample_from_func import square_function

model = Kriging(regr="linear",
                corr="gauss",
                ARD=False)

# create some data to test this model on
X, y, _X, _y = square_function(100, 20)

# let the model fit the data
model.fit(X, y)

# predict the data using the model
y_hat = model.predict(_X)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat = model.predict(_X)

plt.scatter(X, y, label="Data_old_batch")
plt.plot(_X, y_hat, color="black", label="RBF")
plt.legend()
plt.show()

