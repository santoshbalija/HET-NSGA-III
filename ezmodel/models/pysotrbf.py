
try:
    from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail, TPSKernel, ConstantTail
except:
    raise Exception("Model not found. Please execute: 'pip install pySOT'")





from ezmodel.core.model import Model


class pySOTRBF(Model):

    def __init__(self, kernel="cubic", tail="linear", eta=1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = kernel
        self.tail = tail
        self.eta = eta

    def _fit(self, X, y, **kwargs):
        n, m = X.shape
        kernel, tail = get_kernel(self.kernel), get_tail(self.tail, m)
        self.model = RBFInterpolant(dim=m, kernel=kernel, tail=tail, eta=self.eta)
        self.model.add_points(X, y)

    def _predict(self, X, out):
        out["y"] = self.model.predict(X)

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ["cubic", "tps"],
            "tail": ["linear"],
            "eta": [1e-12, 1e-6, 1e-4, 1e-2],
        }


def get_kernel(kernel):
    if kernel == "cubic":
        return CubicKernel()
    elif kernel == "tps":
        return TPSKernel()


def get_tail(tail, m):
    if tail == "linear":
        return LinearTail(m)
    elif tail == "constant":
        return ConstantTail(m)
