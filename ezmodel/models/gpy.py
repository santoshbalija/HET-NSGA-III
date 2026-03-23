try:
    import GPy
except:
    raise Exception("Model not found. Please execute: 'pip install GPy'")

import numpy as np

from ezmodel.core.model import Model

kernels = {
    "matern32": GPy.kern.Matern32,
    "matern52": GPy.kern.Matern52,
    "cosine": GPy.kern.Cosine,
    "ratquad": GPy.kern.RatQuad,
    "exponential": GPy.kern.Exponential,
}


def get_kernel(name, n_var, ARD):
    if name not in kernels:
        raise Exception("Kernel not found.")
    else:
        return kernels[name](n_var, ARD=ARD)


class gpyGP(Model):

    def __init__(self, kernel="matern52", ARD=False, optimizer='lbfgs', **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.ARD = ARD
        self.optimizer = optimizer

    def _fit(self, X, y, **kwargs):
        kernel = get_kernel(self.kernel, X.shape[1], self.ARD)
        model = GPy.models.GPRegression(X, y, kernel, normalizer=True)

        model.constrain_positive('')
        (kern_variance, kern_lengthscale, gaussian_noise) = model.parameter_names()

        model[kern_variance].constrain_bounded(1e-6, 1e6, warning=False)
        model[kern_lengthscale].constrain_bounded(1e-6, 1e6, warning=False)
        model[gaussian_noise].constrain_fixed(1e-6, warning=False)

        if self.optimizer == 'lbfgs':
            model.optimize_restarts(optimizer='lbfgs',
                                    num_restarts=10,
                                    num_processes=1,
                                    verbose=False)
        elif self.optimizer == 'ga':
            model.optimize(optimizer='ga')
        else:
            raise Exception("Unknown Optimizer!")

        self.model = model

    def _predict(self, X, out):
        if "sigma" in out:
            out["y"], var = self.model.predict(X)

            var[var <= 0] = 0
            out["var"] = var
            out["sigma"] = np.sqrt(var)
        else:
            out["y"], _ = self.model.predict(X)

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ["matern52", "ratquad", "exponential"],
            "ARD": [False, True],
        }
