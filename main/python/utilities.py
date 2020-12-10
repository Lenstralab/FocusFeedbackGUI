import os
import numpy as np
from threading import Thread
from parfor import parfor

threads = []
def thread(fun):
    """ decorator to run function in a separate thread to keep the gui responsive
    """
    def tfun(*args, **kwargs):
        T = Thread(target=fun, args=args, kwargs=kwargs)
        threads.append(T)
        T.start()
        # return T
    return tfun

def close_threads():
    print('Joining {} threads.'.format(len(threads)))
    for T in threads:
        T.join()

def mkdir(path):
    """ recursively make directory if it doesn't exist already
        wp@tl20190910
    """
    if not os.path.exists(path):
        mkdir(os.path.split(path)[0])
        os.mkdir(path)


def fixpar(N, fix):
    """ Returns a function which will add fixed parameters in fix into an array
        N: total length of array which will be input in the function
        fix: dictionary, {2: 5.6}: fix parameter[2] = 5.6

        see its use in functions.fitgauss

        wp@tl20190816
    """
    # indices with variable parameters
    idx = sorted(list(set(range(N)) - set(fix)))

    # put the fixed paramters in place
    f = np.zeros(N)
    for i, v in fix.items():
        f[i] = v

    # make array used to construct variable part
    P = np.zeros((N, len(idx)))
    for i, j in enumerate(idx):
        P[j, i] = 1

    return lambda par: np.dot(P, par) + f


def unfixpar(p, fix):
    """ reverse of fixpar, but just returns the array immediately instead of returning
        a function which will do it

        wp@tl20190816
    """
    p = list(p)
    [p.pop(i) for i in sorted(list(fix), reverse=True)]
    return np.array(p)


def maskpk(pk, mask):
    """ remove points in nx2 array which are located outside mask
        wp@tl20190709
    """
    pk = np.round(pk)
    idx = []
    for i in range(pk.shape[0]):
        if mask[tuple(pk[i, :])]:
            idx.append(i)
    return pk[idx, :]


def singletuple(x):
    """ Make sure x gets wrapped in a tuple as a whole, and not split up in the case of strings
        tuple('hi')       -> ('h', 'i')
        singletuple('hi') -> ('hi',)
        wp@tl20190321
    """
    a = [];
    a.append(x)
    return tuple(a)


def weightedmean(x, dx):
    s2 = 1 / (np.nansum(dx ** -2))
    return s2 * np.nansum(x / dx ** 2), np.sqrt(s2)


def circ_weightedmean(t, dt=1, T=2 * np.pi):
    t = (2 * np.pi / T) * t
    dt = (2 * np.pi / T) * dt

    S = np.sum(np.sin(t) / dt)
    C = np.sum(np.cos(t) / dt)

    return np.arctan2(S, C) * T / 2 / np.pi, np.sqrt(np.sum((C * np.cos(t) + S * np.sin(t)) ** 2)) / (
                S ** 2 + C ** 2) * T / 2 / np.pi


def rmnan(*a):
    a = list(a)
    idx = np.full(0, 0)
    for i in range(len(a)):
        idx = np.append(idx, np.where(~np.isfinite(a[i])))
    idx = list(np.unique(idx))
    if len(idx):
        for i in range(len(a)):
            if hasattr(a[i], "__getitem__"):
                for j in reversed(idx):
                    if isinstance(a[i], tuple):
                        a[i] = deltuple(a[i], j)
                    elif isinstance(a[i], np.ndarray):
                        a[i] = np.delete(a[i], j)
                    else:
                        del a[i][j]
    return tuple(a)


def outliers(D, keep=True):
    q2 = np.nanmedian(np.array(D).flatten())
    q1 = np.nanmedian(D[D < q2])
    q3 = np.nanmedian(D[D > q2])
    lb = 4 * q1 - 3 * q3
    ub = 4 * q3 - 3 * q1

    if keep:
        idx = np.where((D >= lb) & (D <= ub))
    else:
        idx = np.where(~((D >= lb) & (D <= ub)))
    return idx


def findrange(x, s):
    """ Finds the range (x-s/2;x+s/2) with the biggest number of points in x
        wp@tl20190301
    """
    l = len(x)
    t = np.zeros(2 * l)
    for i, y in enumerate(x):
        t[i] = len(x[(x >= y) & (x <= (y + s))])
        t[i + l] = len(x[(x >= (y - s)) & (x <= y)])
    i = np.argmax(t)
    if i < l:
        return x[i] + s / 2
    else:
        return x[i - l] - s / 2


def errwrap(fun, default, *args):
    """ Run a function fun, and when an error is caught return the default value
        wp@tl20190321
    """
    try:
        return fun(*args)
    except:
        return default