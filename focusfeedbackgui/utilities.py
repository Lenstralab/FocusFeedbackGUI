import yaml
import re
import os
import warnings
import numpy as np
from PySide2 import QtCore
from traceback import format_exc
from tllab_common.wimread import imread


def yaml_load(f):
    # fix loading scientific notation without decimal separator
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'''^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$''', re.X),
        list('-+0123456789.'))
    return yaml.load(f, loader)


class QThread(QtCore.QThread):
    done_signal = QtCore.Signal(object)

    def __init__(self, target, callback=None, *args, **kwargs):
        super().__init__()
        self.is_alive = True
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.done_signal.connect(self.join)
        self.start()

    def run(self):
        try:
            self.done_signal.emit((0, self.target(*self.args, **self.kwargs)))
        except Exception:
            warnings.warn(f'\n{format_exc()}')
            self.done_signal.emit((1, format_exc()))

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


def error_wrap(fun, default=None):
    def e(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception:
            warnings.warn(f'\n{format_exc()}')
            return default
    return e


def mask_pk(pk, mask):
    """ remove points in nx2 array which are located outside mask
        wp@tl20190709
    """
    pk = np.round(pk)
    idx = []
    for i in range(pk.shape[0]):
        if mask[tuple(pk[i, :])]:
            idx.append(i)
    return pk[idx, :]


def weighted_mean(x, dx):
    s2 = 1 / (np.nansum(dx ** -2))
    return s2 * np.nansum(x / dx ** 2), np.sqrt(s2)


def circ_weighted_mean(t, dt=1, period=2 * np.pi):
    t *= 2 * np.pi / period
    dt = np.clip(dt, 1e-15, np.inf)
    sin = np.sum(np.sin(t) / dt)
    cos = np.sum(np.cos(t) / dt)
    return np.arctan2(sin, cos) * period / 2 / np.pi, np.sqrt(np.sum((cos * np.cos(t) + sin * np.sin(t)) ** 2)) / (
                sin ** 2 + cos ** 2) * period / 2 / np.pi


def rm_nan(*a):
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


def outliers(data, keep=True):
    q2 = np.nanmedian(np.array(data).flatten())
    q1 = np.nanmedian(data[data < q2])
    q3 = np.nanmedian(data[data > q2])
    lb = 4 * q1 - 3 * q3
    ub = 4 * q3 - 3 * q1

    if keep:
        idx = np.where((data >= lb) & (data <= ub))
    else:
        idx = np.where(~((data >= lb) & (data <= ub)))
    return idx


def find_range(x, s):
    """ Finds the range (x-s/2;x+s/2) with the biggest number of points in x
        wp@tl20190301
    """
    length = len(x)
    t = np.zeros(2 * length)
    for i, y in enumerate(x):
        t[i] = len(x[(x >= y) & (x <= (y + s))])
        t[i + length] = len(x[(x >= (y - s)) & (x <= y)])
    i = np.argmax(t)
    if i < length:
        return x[i] + s / 2
    else:
        return x[i - length] - s / 2


def warp(file, out=None, channel=None, z_slice=None, time=None, split=False, force=True, transform_files=None):
    if transform_files is not None and transform_files[0].endswith('.yml'):
        beadfiles = None
        transform = transform_files[0]
    else:
        beadfiles = transform_files
        transform = True

    if os.path.exists(file):
        with imread(file, transform=transform, beadfile=beadfiles) as im:
            if out is None:
                out = file[:-4] + '_transformed.tif'
            out = os.path.abspath(out)
            if not os.path.exists(os.path.dirname(out)):
                os.makedirs(os.path.dirname(out))
            if os.path.exists(out) and not force:
                print('File {} exists already, add the -f flag if you want to overwrite it.'.format(out))
            else:
                im.save_as_tiff(out, channel, z_slice, time, split)
    else:
        print('File does not exist.')


def info(file):
    with imread(file) as im:
        print(im.summary)
