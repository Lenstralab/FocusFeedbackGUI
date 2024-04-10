import os
from glob import glob
from time import time

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.special

from focusfeedbackgui_rs.functions import *

np.seterr(invalid='ignore')


def fitgauss(im, theta=0, sigma=None, fastmode=False, err=False, xy=None):
    """ Fit gaussian function to image
        im:    2D array with image
        theta: Fixed theta to use
        fastmode: True: only moment analysis, False: moment analysis, then fitting
        q:  [x, y, fwhm, area, offset, ellipticity, angle towards x-axis]
    """
    if np.ndim(im) == 1:
        s = np.sqrt(len(im)).astype(int)
        im = np.reshape(im, (s, s))
    else:
        im = np.array(im)
    im = im.astype(float)
    if sigma is not None:
        im -= scipy.ndimage.gaussian_filter(im, sigma * 1.1)
        im = scipy.ndimage.gaussian_filter(im, sigma / 1.1)

    if xy is None:
        xy = np.array(np.unravel_index(np.nanargmax(im.T), np.shape(im)))
    r = 5
    cr = np.round(((xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))).astype('int')
    jm = crop(im, *cr)
    p = fitgaussint(jm, theta)
    if fastmode:  # Only use moment analysis
        q = p
        cs = cr
        S = np.shape(jm)
        xv, yv = meshgrid(np.arange(S[1], dtype=float), np.arange(S[0], dtype=float))
    else:  # Full fitting
        p[0:2] += cr[:, 0]
        s = 2 * np.ceil(p[2])
        cs = np.round(((p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))).astype('int')
        jm = crop(im, *cs)
        S = np.shape(jm)
        p[0:2] -= cs[:, 0]
        xv, yv = meshgrid(np.arange(S[1], dtype=float), np.arange(S[0], dtype=float))
        if theta is None:  # theta free
            def g(pf):
                return np.sum((jm - gaussian7grid(pf, xv, yv)) ** 2)
        else:  # theta fixed
            def g(pf):
                return np.sum((jm - gaussian7grid(np.append(pf, theta), xv, yv))**2)
            p = p[:6]
        r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
        q = r.x
        if theta is not None:
            q = np.append(q, theta)
    q[2] = np.abs(q[2])
    q[5] = np.abs(q[5])
    r2 = 1 - np.nansum((jm-gaussian7grid(q, xv, yv))**2) / np.nansum((jm-np.nanmean(jm))**2)
    dq = fminerr(lambda p0: gaussian7grid(p0, xv, yv), q, jm)[1]
    q[0:2] += cs[:, 0]
    if err:
        return q, dq, r2
    else:
        return q, r2


def fitgaussint(im, theta=None):
    """ Initial guess for gaussfit
        im: 2D array with image
        q:  [x, y, fwhm, area, offset, ellipticity, angle]
    """
    if theta is None:
        theta = 0
    S = np.shape(im)
    q = np.zeros(7, 'float')

    x, y = np.meshgrid(range(S[0]), range(S[1]))
    q[4] = np.nanmin(im)
    jm = im-q[4]
    q[3] = np.nansum(jm)
    q[0] = np.nansum(x*jm)/q[3]
    q[1] = np.nansum(y*jm)/q[3]
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = cos*(x-q[0])-(y-q[1])*sin, cos*(y-q[1])+(x-q[0])*sin

    s2 = np.nansum(jm**2)
    sx = np.sqrt(np.nansum((x*jm)**2)/s2)
    sy = np.sqrt(np.nansum((y*jm)**2)/s2)

    q[2] = np.sqrt(sx*sy)*4*np.sqrt(np.log(2))
    q[5] = np.sqrt(sx/sy)
    q[6] = 0 if theta is None else theta
    return q


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
    n_data = np.size(y)
    n_parameters = np.size(a)
    dy = 1 / (dy + eps)
    f0 = np.array(fun(a)).flatten()
    chi_squared = np.sum(((f0 - y) * dy) ** 2) / (n_data - n_parameters)

    # calculate R^2
    sstot = np.sum((y - np.nanmean(y)) ** 2)
    ssres = np.sum((y - f0) ** 2)
    r_squared = 1 - ssres / sstot

    # calculate derivatives
    deriv = np.zeros((n_data, n_parameters))
    for i in range(n_parameters):
        ah = a.copy()
        ah[i] = a[i] * (1 + diffstep) + eps
        f = np.array(fun(ah)).flatten()
        deriv[:, i] = (f - f0) / (ah[i] - a[i]) * dy

    hesse = np.matmul(deriv.T, deriv)

    if np.linalg.matrix_rank(hesse) == np.shape(hesse)[0]:
        invhesse = np.diag(np.linalg.inv(hesse))
    else:
        invhesse = np.diag(np.linalg.pinv(hesse))
    da = np.sqrt(chi_squared * invhesse)
    return chi_squared, da, r_squared


def crop(im, x, y, m=np.nan):
    """ crops image im, limits defined by min(x)..max(y), when these limits are
        outside im the resulting pixels will be filled with mean(im)
        wp@tl20181129
    """
    try:
        x = np.array(x).astype(int)
        y = np.array(y).astype(int)
        S = np.array(np.shape(im))
        R = np.array([[min(y), max(y)], [min(x), max(x)]]).astype(int)
        r = R.copy()
        r[R[:, 0] < 1, 0] = 1
        r[R[:, 1] > S, 1] = S[R[:, 1] > S]
        jm = im[r[0, 0]:r[0, 1], r[1, 0]:r[1, 1]]
        jm = np.concatenate((np.full((r[0, 0] - R[0, 0], np.shape(jm)[1]), m),
                             jm, np.full((R[0, 1] - r[0, 1], np.shape(jm)[1]), m)), 0)
        return np.concatenate((np.full((np.shape(jm)[0], r[1, 0] - R[1, 0]), m),
                               jm, np.full((np.shape(jm)[0], R[1, 1] - r[1, 1]), m)), 1)
    except Exception:
        return np.zeros((max(y)-min(y), max(x)-min(x)))


def disk(s, dim=2):
    """ make a disk-shaped structural element to be used with
        morphological functions
        wp@tl20190709
    """
    d = np.zeros((s,)*dim)
    c = (s-1)/2
    mg = np.meshgrid(*(range(s),)*dim)
    d2 = np.sum([(i-c)**2 for i in mg], 0)
    d[d2 < s**2/4] = 1
    return d


def clip_rectangle(frame_size, x, y, width, height):
    width, height = int(width + 0.5), int(height + 0.5)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    left = np.clip(x + frame_size[0] / 2 - width / 2 + 1, 1, frame_size[0] - 1)
    right = np.clip(x + frame_size[0] / 2 + width / 2, 2, frame_size[0])
    bottom = np.clip(y + frame_size[1] / 2 - height / 2 + 1, 1, frame_size[1] - 1)
    top = np.clip(y + frame_size[1] / 2 + height / 2, 2, frame_size[1])
    width, height = right - left + 1, top - bottom + 1
    if width % 2 and height % 2:
        return clip_rectangle(frame_size, x, y, width - 2, height - 2)
    elif width % 2:
        return clip_rectangle(frame_size, x, y, width - 2, height)
    elif height % 2:
        return clip_rectangle(frame_size, x, y, width, height - 2)
    else:
        return float((right+left-1)/2), float((top+bottom-1)/2), float(width), float(height)


def last_czi_file(folder=r'd:\data', t=np.inf):
    """ finds last created czi file in folder created not more than t seconds ago
        wp@tl20191218
    """
    files = glob(os.path.join(folder, '**', '*.czi'), recursive=True)
    tm = [os.path.getctime(file) for file in files]
    if not tm:
        return ''
    else:
        t_newest = np.max(tm)
        if time() - t_newest > t:
            return ''
        else:
            return files[np.argmax(tm)]
