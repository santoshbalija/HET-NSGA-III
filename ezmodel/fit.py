from ezmodel.core.benchmark import Benchmark
from ezmodel.core.factory import models_from_clazzes
from ezmodel.core.partitioning import merge_and_partition
from ezmodel.core.selection import ModelSelection
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF


def fit(
        X,
        y,
        X_test=None,
        y_test=None,
        partitions=None,
        clazzes=[Kriging, RBF],
        defaults={},
        models=None,
        return_benchmark=False,
        return_selection=False,
        **kwargs):
    """

    Find the best out of a variety of surrogates regarding a metric.

    Parameters
    ----------
    X : np.array
        The design space data.

    y : np.array
        The target value to model

    X_test : optional, np.array
        If no partitioning but a test set should be used this shall be set

    y_test : optional, np.array
        If no partitioning but a test set should be used this shall be set

    partitions : list
        The partitions to be used to find the best surrogate

    clazzes : list
        class instances which are used to create the models

    defaults : dict
        If the classes should be instantiated by default values they should be set here

    models : list
        The models can also be provided directly. Then clazzes and defaults can be `None`

    return_benchmark : bool
        Whether the benchmark shall finally be return too or not

    Returns
    -------

    model : Model
        The best model found on the data.

    benchmark : Benchmark
        The benchmark used to find the best model.


    """

    if X_test is not None and y_test is not None:
        X, y, partitions = merge_and_partition((X, y), (X_test, y_test))
        refit = False
    else:

        # by default just do cross validation
        if partitions is None:
            raise Exception("Either provide partitions or a test set.")

        refit = True

    if models is None:
        models = models_from_clazzes(*clazzes, **defaults)

    benchmark = Benchmark(models)

    selection = ModelSelection(benchmark, refit=refit, **kwargs)
    model = selection.do(X, y, partitions)

    ret = [model]

    if return_benchmark:
        ret.append(benchmark)
    if return_selection:
        ret.append(selection)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
