import copy
from abc import abstractmethod

from ezmodel.util.misc import dict_to_str


class ModelFactory:

    @abstractmethod
    def do(self):
        pass


class ModelFactoryByClazz(ModelFactory):

    def __init__(self, clazz, params=None, create_instance=True, **kwargs) -> None:
        super().__init__()
        self.clazz = clazz
        self.kwargs = kwargs
        self.params = params
        self.create_instance = create_instance

    def do(self):
        params, clazz, kwargs = self.params, self.clazz, self.kwargs
        if params is None:
            params = self.clazz.hyperparameters()

        ret = {}
        for vals in dfs(params):
            label = f"{clazz.__name__}[{dict_to_str(vals)}]"

            obj = None
            if self.create_instance:
                all_kwargs = {**kwargs, **vals}
                obj = clazz(**all_kwargs)

            ret[label] = dict(label=label, clazz=clazz, params=vals, defaults=kwargs, model=obj)

        return ret


def models_from_clazzes(*clazzes, **defaults):
    models = {}
    for clazz in clazzes:
        models = {**models, **ModelFactoryByClazz(clazz, **defaults).do()}
    return models


# --------------------------------------------------------------
# Util
# --------------------------------------------------------------


def dfs(params):
    """

    Parameters
    ----------
    params : dict
    A dictionary where each key has a list of values.

    Returns
    -------
    comb : list
    All possible combinations of this dictionary having selected
    one entry from each key.

    """
    ret = []
    dfs_rec({}, params, ret)
    return ret


def dfs_rec(entry, params, ret):
    if len(params) == 0:
        ret.append(entry)
    else:
        _params = copy.deepcopy(params)
        key = list(params.keys())[0]
        del _params[key]

        for value in params[key]:
            _comb = copy.deepcopy(entry)
            _comb[key] = value

            dfs_rec(_comb, _params, ret)
