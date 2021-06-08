import numpy as np
import scipy.optimize
import scipy.ndimage
import functools
import multiprocessing
import trackpy
import pandas
import skimage.filters
from imread import imread
from parfor import parfor

if __package__ == '':
    import utilities
else:
    from . import utilities

def sliding_mean(X, data, sig, width, x):
    X = np.array(X)
    data = np.array(data)
    sig = np.array(sig)
    f = []
    for i in x:
        idx = (i - width / 2) < X
        idx *= (i + width / 2) > X
        idx = np.where(idx)
        d = data[idx]
        s = sig[idx]
        f.append((np.mean(d), np.std(d) / np.sqrt(len(d))))
        # f.append(ut.weightedmean(d, s))
    f = np.array(f)
    return f[:, 0], f[:, 1]


def data2cdf(data, maxSize=1e4):
    """ construct an nd cumulative probability density function
        inputs:
            data: 2d array with data in n columns
            maxSize: maximum points to consider

        returns: list with coordinates, and as last item cdf
            x, y = data2cdf(data) for 1d data
            x, y, z = data2cdf(data) for 2d data
            etc.
        wp@tl20191123
    """
    s = data.shape
    if np.ndim(data) == 1:
        data = np.expand_dims(data, 1)
    dim = data.shape[1]

    if data.shape[0] ** data.shape[1] > maxSize:
        bins = [np.linspace(np.nanmin(data[:, i]), np.nanmax(data[:, i]), maxSize) for i in range(data.shape[1])]
    else:
        bins = [sorted(np.unique(data[:, i])) for i in range(data.shape[1])]
    p, _ = np.histogramdd(data, [np.hstack((b, np.inf)) for b in bins])

    if dim == 2:
        p = p.T

    for i in range(dim):
        p = np.cumsum(p, i)

    out = [i.squeeze() for i in np.meshgrid(*bins)]
    out.append(p / data.shape[0])

    return out


def data2pdf(data, sig, x=None):
    """ construct a probability density function from data
        inputs:
            data: 1d array with data
            sig:  1d array with error estimates on data
            x:    1d array with points at which to evaluate the pdf

        returns:
            x:    1d array with points at which the pdf was evaluated
            f:    1d array with the pdf

        wp@tl20191119
    """
    if x is None:
        x = np.linspace(np.min(data - sig), np.max(data + sig))
    g = lambda mu, sigma, x: 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
    f = np.zeros(len(x))
    N = 0
    for z, dz in zip(data, sig):
        if np.isfinite(z) and np.isfinite(dz):
            f += g(z, dz, x)
            N += 1
    return x, f / N


def tpbatchpar(im, diameter):
    @parfor(range(len(im)), args=(im, diameter))
    def fun(i, im, diameter):
        f = trackpy.locate(im[i].squeeze(), diameter)
        f['frame'] = i
        if hasattr(im, 'czt'):
            f['C'], f['Z'], f['T'] = im.czt(i)
        else:
            f['C'], f['Z'], f['T'] = 0, 0, f['frame']
        return f

    q = pandas.concat(fun)
    q = q[q.x.notnull() & q.y.notnull()].reset_index(drop=True)
    # q.index = range(q.shape[0])
    return q


def locate(im, s_px=1.8, pxsize=0.1, deltaz=1, timeinterval=1):
    if isinstance(im, np.ndarray):
        im = imread(im)

    def dec(im, frame, c, z, t):
        if im.detector[c] % 2:
            l = .664  # JF646
        else:
            l = .51  # GFP
        s = l / 2 / im.NA / im.pxsize
        return gfilter(frame, s)

    tmpdec = im.frame_decorator
    im.frame_decorator = dec

    print('Finding interesting features.')
    f = tpbatchpar(im, 7)

    c = []
    z = []
    t = []
    for i in f['frame']:
        C, Z, T = im.czt(i)
        c.append(C)
        z.append(Z)
        t.append(T)
    f['C'], f['Z'], f['T'] = c, z, t

    h = []
    for c in range(im.shape[2]):
        h.append(f[f['c'] == c].copy())
        t = skimage.filters.threshold_triangle(h[c]['mass']) / 2
        origsize = h[c].shape[0]
        h[c] = h[c][h[c]['mass'] > t]
        print('Keeping {} of {} features ({:.2f}%) in channel {}.'.format(h[c].shape[0], origsize,
                                                                          100 * h[c].shape[0] / origsize, c))

    f = pandas.concat(h).copy()
    print('Localizing features.')
    # f = f.copy()

    f = tpsuperresseq(f, im, None)

    print('Filtering anomalous features')

    h = []
    for c in range(im.shape[2]):
        if im.detector[c] == 2:
            l = .51  # GFP
        else:
            l = .664  # JF646
        s = l / 2 / im.NA / im.pxsize
        h.append(f[f['c'] == c].copy())
        h[c] = h[c].query(
            '(x-x_ini)**2+(y-y_ini)**2<{} & dx<{} & dy<{} & s>{} & s<{}'.format(2 / im.pxsize, 0.15 / im.pxsize,
                                                                                0.15 / im.pxsize, s / 1.5, s * 3))
    f = pandas.concat(h).copy()
    pxsize = im.pxsize
    deltaz = im.deltaz
    timeinterval = im.timeinterval

    f.index = range(f.shape[0])
    f = f.copy()

    f['x_um'] = f['x'] * pxsize
    f['y_um'] = f['y'] * pxsize
    f['dx_um'] = f['dx'] * pxsize
    f['dy_um'] = f['dy'] * pxsize
    f['s_um'] = f['s'] * pxsize
    f['ds_um'] = f['ds'] * pxsize
    f['z_plane_um'] = f['z'] * deltaz
    f['time_s'] = f['t'] * timeinterval

    im.frame_decorator = tmpdec

    return f


def fitseries(im, c, xy, theta=True):
    """ fits all peaks at coordinates xy in image im, color c
        im: imopen object
        c: color channel number
        xy: peak coordinates
        q: table with: xc, yc, fwhm, area, offset, ellipticity, angle
        dq: errors on q
    """
    f = functools.partial(fitgauss, xy=xy, fix={6: theta})
    p = multiprocessing.Pool(36)
    q = p.map(f, [im.frame(c, 0, t) for t in range(im.shape[4])])
    p.close()
    p.join()
    p, dp, _ = zip(*q)
    q = np.array(p)
    dq = np.array(dp)
    return q, dq


def radiusGyration(im, c=None):
    """ gives the radius of gyration of the image im around center c
        which is an estimate of the peakwidth of a gaussian:
        g = sqrt(2) * s = fwhm / sqrt(ln(2))
    """
    if c is None:
        S = np.shape(im)
        c = np.unravel_index(np.argmax(im.T), (S[0], S[1]))
    x, y = np.meshgrid(range(im.shape[0]), range(im.shape[1]))
    r2 = (x - c[0]) ** 2 + (y - c[1]) ** 2
    return np.sqrt(np.nansum((im) * r2) / np.nansum(im))


def fminerr(fun, a, y, dy=None):
    """ Error estimation of a fit

        Inputs:
        fun: function which was fitted to data
        a:   function parameters
        y:   ydata
        dy:  errors on ydata

        Outputs:
        chisq: Chi^2
        da:    error estimates of the function parameters
        R2:    R^2

        Example:
        x = np.array((-3,-1,2,4,5))
        a = np.array((2,-3))
        y = (15,0,5,30,50)
        fun = lambda a: a[0]*x**2+a[1]
        chisq,dp,R2 = fminerr(fun,p,y)
    """
    # import pdb; pdb.set_trace()
    eps = np.spacing(1)
    a = np.array(a).flatten()
    y = np.array(y).flatten()
    if dy is None:
        dy = np.ones(np.shape(y))
    else:
        dy = np.array(dy).flatten()
    diffstep = 1e0
    nData = np.size(y)
    nPar = np.size(a)
    dy = 1 / (dy + eps)
    f0 = np.array(fun(a)).flatten()
    chisq = np.sum(((f0 - y) * dy) ** 2) / (nData - nPar)

    # calculate R^2
    sstot = np.sum((y - np.nanmean(y)) ** 2)
    ssres = np.sum((y - f0) ** 2)
    R2 = 1 - ssres / sstot

    # calculate derivatives
    deriv = np.zeros((nData, nPar))
    for i in range(nPar):
        ah = a.copy()
        ah[i] = a[i] * (1 + diffstep) + eps
        f = np.array(fun(ah)).flatten()
        deriv[:, i] = (f - f0) / (ah[i] - a[i]) * dy

    hesse = np.matmul(deriv.T, deriv)
    # hesse=deriv.T@deriv

    if np.linalg.matrix_rank(hesse) == np.shape(hesse)[0]:
        da = np.sqrt(chisq * np.diag(np.linalg.inv(hesse)))
    else:
        da = np.sqrt(chisq * np.diag(np.linalg.pinv(hesse)))
        # da = np.full(np.shape(a),np.nan)
        # print('Hessian not invertible, size: {0}, rank: {1}'.format(np.shape(hesse)[0],np.linalg.matrix_rank(hesse)))
    return chisq, da, R2


def cumdisfun(d):
    d = utilities.rmnan(d)[0]
    d = np.sort(d).flatten()
    y = np.arange(len(d))
    y = np.vstack((y, y + 1)).T.reshape(2 * y.size).astype('float')
    x = np.vstack((d, d)).T.reshape(2 * d.size).astype('float')
    return x, y


def distfit(d):
    if np.size(d) == 0:
        return np.nan, np.nan
    x, y = cumdisfun(d)
    g = lambda p: np.nansum(((scipy.special.erf((x - p[0]) / np.sqrt(2) / p[1]) + 1) / 2 - y / np.nanmax(y)) ** 2)
    r = scipy.optimize.minimize(g, (np.nanmean(d), np.nanstd(d)), options={'disp': False, 'maxiter': 1e5})
    mu = r.x[0]
    std = r.x[1]
    return mu, std