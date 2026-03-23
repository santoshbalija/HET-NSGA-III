import matplotlib.pyplot as plt
import numpy as np

from ezmodel.models.gpflow import GPFlow
from ezmodel.util.sample_from_func import sine_function

gp = GPFlow()

# create some data to test this model on
X, y, _X, _y = sine_function(4, 200)

# let the model fit the data
gp.fit(X, y)

# predict the data using the model
y_hat = gp.predict(_X)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat = gp.predict(_X)

plt.scatter(X, y, label="Data_old_batch")
plt.plot(_X, y_hat, color="black", label="GPY")
plt.legend()
plt.show()
