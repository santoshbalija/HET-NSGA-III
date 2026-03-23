import time


def time_in_s(t=None):
    if t is None:
        t = time.time()
    return t


def wait_until(t):
    while True:
        if time_in_s() >= t:
            break
        time.sleep(.0001)


def all_targets(n_obj, n_constr):
    desc = []
    desc.extend([('F', k) for k in range(n_obj)])
    desc.extend([('G', k) for k in range(n_constr)])
    return desc


def targets_to_pop(pop, out):
    for e in out.keys():

        if isinstance(e, tuple):
            (_target, _k) = e
            vals = out[e]

            V = pop.get(_target)

            if vals.ndim == 2:
                vals = vals[:, 0]

            V[:, _k] = vals
            pop.set(_target, V)
def targets_to_pop_index(pop, out,k):
    for e in out.keys():

        if isinstance(e, tuple):
            (_target, _k) = e
            vals = out[e]

            V = pop.get(_target)

            if vals.ndim == 2:
                vals = vals[:, 0]

            V[k, _k] = vals[k]
            pop.set(_target, V)



def pop_to_out(pop, targets):
    out = {}
    for target in targets:
        out[target] = get_target(pop, target)
    return out


def get_target(obj, target):
    _type, _k = target
    if _type == "F":
        return obj.get("F")[:, _k]
    elif _type == "G":
        return obj.get("G")[:, _k]


