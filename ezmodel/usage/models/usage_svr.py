import matplotlib.pyplot as plt
import numpy as np

from ezmodel.models.mean import SimpleMean
from ezmodel.models.svr import SVR

from ezmodel.util.sample_from_func import sine_function

svm = SVR()

# create some data to test this model on
X, y, _X, _y = sine_function(20, 200)

# let the model fit the data
svm.fit(X, y)

# predict the data using the model
y_hat = svm.predict(_X)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat = svm.predict(_X)

plt.scatter(X, y, label="Data_old_batch")
plt.plot(_X, y_hat, color="black", label="GPy")
plt.legend()
plt.show()
