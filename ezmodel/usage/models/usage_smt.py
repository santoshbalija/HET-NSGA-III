
import matplotlib.pyplot as plt
import numpy as np
from ezmodel.models.smt import smtKriging
from ezmodel.util.metrics import calc_metric
from ezmodel.util.sample_from_func import sine_function

model = smtKriging()

# create some data to test this model on
X, y, _X, _y = sine_function(100, 20)

# let the model fit the data
model.fit(X, y)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat, y_var = model.predict(_X, return_values_of=["y", "var"])

# calculate the error and print it
mae = calc_metric("mae", _y, y_hat)
print("Mean Absolute Error", mae)

plt.scatter(X, y, label="Data_old_batch")
plt.plot(_X, y_hat, color="black", label="RBF")
plt.legend()
plt.show()

