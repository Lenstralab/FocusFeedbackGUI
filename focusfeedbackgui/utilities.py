import numpy as np
from PyQt5 import QtCore
from traceback import format_exc
import yaml
import re
import os
from tllab_common.wimread import imread


def yamlload(f):
    # fix loading scientific notation without decimal separator
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return yaml.load(f, loader)


class qthread(QtCore.QThread):
    done = QtCore.pyqtSignal(object)

    def __init__(self, target, callback=None, *args, **kwargs):
        super().__init__()
        self.is_alive = True
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.done.connect(self.join)
        self.start()

    def run(self):
        try:
            self.done.emit((0, self.target(*self.args, **self.kwargs)))
        except Exception:
            self.done.emit((1, format_exc()))

    def join(self, state):
        self.quit()
        self.wait()
        self.is_alive = False
        state, args = state
        if not isinstance(args, tuple):
            args = (args,)
        if state:
            raise Exception(args)
        if self.callback is not None:
            self.callback(*args)


def errwrap(fun, default=None):
    def e(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            return default
    return e


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


def weightedmean(x, dx):
    s2 = 1 / (np.nansum(dx ** -2))
    return s2 * np.nansum(x / dx ** 2), np.sqrt(s2)


def circ_weightedmean(t, dt=1, T=2*np.pi):
    t *= 2 * np.pi / T
    dt = np.clip(dt, 1e-15, np.inf)
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
                    if isinstance(a[i], np.ndarray):
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


def warp(file, out=None, channel=None, zslice=None, time=None, split=False, force=True):
    if os.path.exists(file):
        with imread(file, transform=True) as im:
            if out is None:
                out = file[:-4] + '_transformed.tif'
            out = os.path.abspath(out)
            if not os.path.exists(os.path.dirname(out)):
                os.makedirs(os.path.dirname(out))
            if os.path.exists(out) and not force:
                print('File {} exists already, add the -f flag if you want to overwrite it.'.format(out))
            else:
                im.save_as_tiff(out, channel, zslice, time, split)
    else:
        print('File does not exist.')


def info(file):
    with imread(file) as im:
        print(im.summary)
