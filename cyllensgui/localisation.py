import numpy as np
import scipy
import skimage
import pandas
from . import utilities
from . import functions2 as functions
import nbhelpers as nb
import skimage.feature
from parfor import pmap
import matplotlib.pyplot as plt


def crop(im, x, y=None, z=None, m=np.nan):
    """ crops image im, limits defined by min(x)..max(y), when these limits are
        outside im the resulting pixels will be filled with mean(im)
        wp@tl20181129
    """
    if isinstance(x, np.ndarray) and x.shape == (3, 2):
        z = x[2, :].copy().astype('int')
        y = x[1, :].copy().astype('int')
        x = x[0, :].copy().astype('int')
    elif isinstance(x, np.ndarray) and x.shape == (2, 2):
        y = x[1, :].copy().astype('int')
        x = x[0, :].copy().astype('int')
    else:
        x = np.array(x).astype('int')
        y = np.array(y).astype('int')
    if not z is None:  # 3D
        z = np.array(z).astype('int')
        S = np.array(np.shape(im))
        R = np.array([[min(y), max(y)], [min(x), max(x)], [min(z), max(z)]]).astype('int')
        r = R.copy()
        r[R[:, 0] < 0, 0] = 1
        r[R[:, 1] > S, 1] = S[R[:, 1] > S]
        jm = im[r[0, 0]:r[0, 1], r[1, 0]:r[1, 1], r[2, 0]:r[2, 1]]
        jm = np.concatenate((np.full((r[0, 0] - R[0, 0], jm.shape[1], jm.shape[2]), m), jm,
                             np.full((R[0, 1] - r[0, 1], jm.shape[1], jm.shape[2]), m)), 0)
        jm = np.concatenate((np.full((jm.shape[0], r[1, 0] - R[1, 0], jm.shape[2]), m), jm,
                             np.full((jm.shape[0], R[1, 1] - r[1, 1], jm.shape[2]), m)), 1)
        return np.concatenate((np.full((jm.shape[0], jm.shape[1], r[2, 0] - R[2, 0]), m), jm,
                               np.full((jm.shape[0], jm.shape[1], R[2, 1] - r[2, 1]), m)), 2)
    else:  # 2D
        S = np.array(np.shape(im))
        R = np.array([[min(y), max(y)], [min(x), max(x)]]).astype(int)
        r = R.copy()
        r[R[:, 0] < 1, 0] = 1
        r[R[:, 1] > S, 1] = S[R[:, 1] > S]
        jm = im[r[0, 0]:r[0, 1], r[1, 0]:r[1, 1]]
        jm = np.concatenate(
            (np.full((r[0, 0] - R[0, 0], np.shape(jm)[1]), m), jm, np.full((R[0, 1] - r[0, 1], np.shape(jm)[1]), m)), 0)
        return np.concatenate(
            (np.full((np.shape(jm)[0], r[1, 0] - R[1, 0]), m), jm, np.full((np.shape(jm)[0], R[1, 1] - r[1, 1]), m)), 1)


def disk(s, dim=2):
    """ make a diskshaped structural element to be used with
        morphological functions
        wp@tl20190709
    """
    d = np.zeros((s,)*dim)
    c = (s-1)/2
    mg = np.meshgrid(*(range(s),)*dim)
    d2 = np.sum([(i-c)**2 for i in mg], 0)
    d[d2<s**2/4] = 1
    return d


def detect_points_sf(im, sigma, mask=None, footprint=15):
    """ Find interesting spots to which try to fit gaussians in a single frame
        im:      2D or 3D array
        sigma:   theoretical width of the psf (isotropic)
        mask:    logical 2D or 3D array masking the area in which points are to be found

        wp@tl201908
    """
    dim = np.ndim(im)

    # pk = skimage.feature.blob_log(im, 1, 3, 21)
    # pk = pk[1<pk[:,2]]
    # pk = pk[pk[:,2]<3]

    if sigma > 0:
        # c = images.gfilter(im, sigma)
        c = scipy.ndimage.gaussian_laplace(im, sigma)
        c = scipy.ndimage.gaussian_filter(-c, sigma)
    else:
        c = im

    pk = skimage.feature.peak_local_max(c, footprint=disk(footprint, dim))
    if not mask is None:
        pk = utilities.maskpk(pk, mask)

    # plt.imshow(c)
    f = pandas.DataFrame({'y_ini': pk[:, 0], 'x_ini': pk[:, 1]})
    if dim == 3:
        f['z_ini'] = pk[:, 2]

    # plt.plot(f['x_ini'], f['y_ini'], 'ro', markersize=15, markerfacecolor='None')

    p = []
    R2 = []
    for i in range(len(f)):
        g = f.loc[i]
        if dim == 3:
            jm = crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8], g['z_ini'] + [-8, 8])
        else:
            jm = crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8])
        p.append(fitgaussint(jm))
        R2.append(1 - np.sum((gaussian(p[-1], 16, 16) - jm) ** 2) / np.sum((jm - np.mean(jm)) ** 2))
    p = np.array(p)
    if len(f):
        f['y'] = p[:, 0] + f['y_ini'] - 8
        f['x'] = p[:, 1] + f['x_ini'] - 8
        f['R2'] = R2
        if dim == 2:
            f['s_ini'] = p[:, 2] / 2 / np.sqrt(2 * np.log(2))
            f['i_ini'] = p[:, 3]
            f['o_ini'] = p[:, 4]
            f['e_ini'] = p[:, 5]
            f['theta_ini'] = p[:, 6]
        else:
            f['z'] = p[:, 2] + f['z_ini'] - 8
            f['s_ini'] = p[:, 3] / 2 / np.sqrt(2 * np.log(2))
            f['sz_ini'] = p[:, 4] / 2 / np.sqrt(2 * np.log(2))
            f['i_ini'] = p[:, 5]
            f['o_ini'] = p[:, 6]
    else:
        f['y'] = []
        f['x'] = []
        if dim == 2:
            f['e_ini'] = []
            f['theta_ini'] = []
        else:
            f['z'] = []
            f['sz_ini'] = []
        f['s_ini'] = []
        f['i_ini'] = []
        f['o_ini'] = []
        f['R2'] = []

    f = f.dropna()
    if len(f) > 2:
        th = skimage.filters.threshold_otsu(f['i_ini'])
        f = f.query('i_ini>{}'.format(th))
    return f


def fun(c, sigma, keep, fix, ell, tilt, k, filter):
    f = c[0][1].copy()
    jm = c[1].copy()
    if filter:
        #jm = gfilter(jm, sigma[int(f['C'])])
        jm = gfilter(jm, 1.6)
    fwhm = sigma[int(f['C'])] * 2 * np.sqrt(2 * np.log(2))

    for j in keep:
        if j in k:
            if j == 's':
                fix[k.index(j)] = float(2 * np.sqrt(2 * np.log(2)) * f[j])
                fwhm = fix[k.index(j)]
            else:
                fix[k.index(j)] = float(f[j])

    q, dq, s = fitgauss(jm, np.array(f[['x', 'y']]), ell, tilt, fwhm, fix, pl=False)

    f['y_ini'] = f['y']
    f['x_ini'] = f['x']
    f['y'] = q[1]
    f['x'] = q[0]
    f['dy'] = dq[1]
    f['dx'] = dq[0]
    f['s'] = q[2] / (2 * np.sqrt(2 * np.log(2)))
    f['ds'] = dq[2] / (2 * np.sqrt(2 * np.log(2)))
    f['i'] = q[3]
    f['di'] = dq[3]
    f['o'] = q[4]
    f['do'] = dq[4]
    f['e'] = q[5]
    f['de'] = dq[5]
    f['theta'] = q[6]
    f['dtheta'] = dq[6]
    f['tiltx'] = q[7]
    f['dtiltx'] = dq[7]
    f['tilty'] = q[8]
    f['dtilty'] = dq[8]
    f['i_peak'] = f['i'] / (2 * np.pi * f['s'] ** 2)
    f['di_peak'] = f['i_peak'] * np.sqrt(4 * (f['ds'] / f['s']) ** 2 + (f['di'] / f['i']) ** 2)
    f['X2'] = s[0]
    f['R2'] = s[1]
    f['sn'] = s[2]
    return pandas.DataFrame(f).transpose()


def gfilter(im, sigma, r=1.1):
    """ Bandpass filter an image using gaussian filters
        im:    2d array
        sigma: feature size to keep
        r:     lb, ub = sigma/r, sigma*r

        wp@tl2019
    """
    jm = im.copy()
    jm -= scipy.ndimage.gaussian_filter(jm, sigma * r)
    return scipy.ndimage.gaussian_filter(jm, sigma / r)


def tpsuperresseq(f, im, theta=True, tilt=True, keep=[], filter=True):
    fix = {}
    if theta is True:
        ell = True
    elif theta is None or theta is False:
        ell = False
    else:
        ell = True
        fix[6] = theta
    if not tilt is True:
        fix[7] = 0
        fix[8] = 0

    k = ['x', 'y', 's', 'i', 'o', 'e', 'theta', 'tiltx', 'tilty']

    frames = f['frame'].astype('int').tolist()
    c = ((i, im(j)) for i, j in zip(f.iterrows(), frames))
    s = [im.sigma(i) for i in range(int(f['C'].max()) + 1)]
    q = pmap(fun, c, (s, keep, fix, ell, tilt, k, filter), desc='Fitting localisations', length=len(frames))
    return pandas.concat(q, sort=True)


def gaussian_nonumba(p, X, Y):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,x-tilt,y-tilt]
        default ellipticity & angle: 1 resp. 0
        X,Y: size of image
    """
    efac = np.sqrt(np.log(2)) / p[2]
    if np.size(p) < 6:
        dx = efac
        dy = efac
    else:
        dx = efac / p[5]
        dy = efac * p[5]
    xv, yv = np.meshgrid(np.arange(Y) - p[0], np.arange(X) - p[1])
    if np.size(p) < 7:
        x = 2 * dx * xv
        y = 2 * dy * yv
    else:
        cos, sin = np.cos(p[6]), np.sin(p[6])
        x = 2 * dx * (cos * xv - yv * sin)
        y = 2 * dy * (cos * yv + xv * sin)
    erf = scipy.special.erf
    if np.size(p) < 8:
        O = p[4]
    else:
        O = p[4] + xv * p[7] + yv * p[8]
    return p[3] / 4 * (erf(x + dx) - erf(x - dx)) * (erf(y + dy) - erf(y - dy)) + O


def gaussian(p, X, Y, Z=None):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
    default ellipticity & angle: 1 resp. 0
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf for numba
    """
    p = [float(i) for i in p]
    if not Z is None:
        return gaussian3d(p, X, Y, Z)
    if len(p) == 5:
        return nb.gaussian5(p, X, Y)
    elif len(p) == 6:
        return nb.gaussian6(p, X, Y)
    elif len(p) == 7:
        return nb.gaussian7(p, X, Y)
    else:
        return nb.gaussian9(p, X, Y)


def fitgauss(im, xy=None, ell=False, tilt=False, fwhm=None, fix=None, pl=False):
    """ Fit gaussian function to image
        im:    2D array with image
        xy:    Initial guess for x, y, optional, default: pos of max in im
        ell:   Fit with ellipicity if True
        fwhm:  fwhm of the peak, used for boundary conditions
        fix:   dictionary describing which parameter to fix, to fix theta: fix={6: theta}
        q:  [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
        dq: errors (std) on q

        wp@tl2019
    """

    # print('xy:   ', xy)
    # print('ell:  ', ell)
    # print('tilt: ', tilt)
    # print('fwhm: ', fwhm)
    # print('fix:  ', fix)

    if not fwhm is None:
        fwhm = np.round(fwhm, 2)

    # handle input options
    if xy is None:
        # filter to throw away any background and approximate position of peak
        fm = (im - scipy.ndimage.gaussian_filter(im, 0.2))[1:-1, 1:-1]
        xy = [i + 1 for i in np.unravel_index(np.nanargmax(fm.T), np.shape(fm))]
    else:
        xy = [int(np.round(i)) for i in xy]
    if fix is None:
        fix = {}
    if ell is False:
        if 5 not in fix:
            fix[5] = 1
        if 6 not in fix:
            fix[6] = 0
    if tilt is False:
        if 7 not in fix:
            fix[7] = 0
        if 8 not in fix:
            fix[8] = 0

    xy = np.array(xy)
    for i in range(2):
        if i in fix:
            xy[i] = int(np.round(fix[i]))

    # size initial crop around peak
    if fwhm is None:
        r = 10
    else:
        r = 2.5 * fwhm

    # find tilt parameters from area around initial crop
    if tilt:
        cc = np.round(((xy[0] - 2 * r, xy[0] + 2 * r + 1), (xy[1] - 2 * r, xy[1] + 2 * r + 1))).astype('int')
        km = crop(im, cc)
        K = [i / 2 for i in km.shape]
        km[int(np.ceil(K[0] - r)):int(np.floor(K[0] + r + 1)),
        int(np.ceil(K[1] - r)):int(np.floor(K[1] + r + 1))] = np.nan
        t = fit_tilted_plane(km)
    else:
        t = [0, 0, 0]

    # find other initial parameters from initial crop with tilt subtracted
    cc = np.round(((xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))).astype('int')
    jm = crop(im, cc)
    xv, yv = nb.meshgrid(*map(np.arange, jm.shape))

    if 6 in fix:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv, theta=fix[6])
    else:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv)
    p[0:2] += cc[:, 0]
    if pl: print('initial q: ', p)

    for i in range(2):
        if i in fix:
            p[i] = xy[i]

    if fwhm is None:
        fwhm = p[2]
    else:
        p[2] = fwhm

    # just give up in some cases
    if not 1 < p[2] < 2 * fwhm or p[3] < 0.1:
        q = np.full(9, np.nan)
        dq = np.full(9, np.nan)
        return q, dq, (np.nan, np.nan, np.nan)

    s = fwhm / np.sqrt(2)  # new crop size

    cc = np.round(((p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))).astype('int')
    jm = crop(im, cc)
    S = np.shape(jm)

    bnds = [(0, S[0] - 1), (0, S[1] - 1), (fwhm / 2, fwhm * 2), (1e2, None), (0, None), (0.5, 2), (None, None),
            (None, None), (None, None)]
    xv, yv = nb.meshgrid(*map(np.arange, S[::-1]))

    # move fixed x and/or y with the crop
    for i in range(2):
        if i in fix:
            fix[i] -= cc[i, 0]
            xy[i] = p[i]

    # find tilt from area around new crop
    cd = np.round(((p[0] - 2 * s, p[0] + 2 * s + 1), (p[1] - 2 * s, p[1] + 2 * s + 1))).astype('int')
    km = crop(im, cd)
    K = [i / 2 for i in km.shape]
    km[int(np.ceil(K[0] - s)):int(np.floor(K[0] + s + 1)), int(np.ceil(K[1] - s)):int(np.floor(K[1] + s + 1))] = np.nan
    t = fit_tilted_plane(km)

    # update parameters to new crop
    p[0:2] -= cc[:, 0]
    p = np.append(p, (t[1], t[2]))
    # p = np.append(p, (1, 0, t[1], t[2]))
    p[4] = t[0] + t[1] * (p[0] + s) + t[2] * (p[1] + s)

    # remove fixed parameters and bounds from lists of initial parameters and bounds
    p = utilities.unfixpar(p, fix)
    [bnds.pop(i) for i in sorted(list(fix), reverse=True)]

    # define function to remove fixed parameters from list, then define function to be minimized
    fp = utilities.fixpar(9, fix)
    g = lambda a: np.nansum((jm - nb.gaussian9grid(fp(a), xv, yv)) ** 2)

    # make sure the initial parameters are within bounds
    for i, b in zip(p, bnds):
        i = utilities.errwrap(np.clip, i, i, b[0], b[1])

    nPar = len(p)

    # fit and find error predictions
    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})

    q = r.x
    dq = np.sqrt(r.fun / (np.size(jm) - np.size(q)) * np.diag(r.hess_inv))
    if pl:
        q0 = fp(q)
        print('q after first fit: ', q0[:2] + cc[:, 0], q0[2:])
        print('nfev:', r.nfev)

    # Check boundary conditions, maybe try to fit again
    refitted = False
    for i, b in zip(q, bnds):
        try:
            if not b[0] < i < b[1] and not refitted:
                r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e7}, bounds=bnds)
                q = r.x
                dq = functions.fminerr(lambda p: nb.gaussian9grid(fp(p), xv, yv), q, jm)[1]
                if pl: print('bounds: {} < {} < {}\nq after refit: {}\nnfev: {}'.format(b[0], i, b[1], q, r.nfev))
                refitted = True
        except:
            pass

    if pl:
        print('Refitted: ', refitted)
        a, b = np.min(jm), np.max(jm)
        plt.figure()
        plt.imshow(jm, vmin=a, vmax=b)
        plt.plot(q0[0], q0[1], 'or')
        plt.figure()
        plt.imshow(nb.gaussian9grid(fp(q), xv, yv), vmin=a, vmax=b)
        plt.figure()
        plt.imshow(np.abs(nb.gaussian9grid(fp(q), xv, yv) - jm), vmin=0, vmax=b - a)

    # reinsert fixed parameters
    q = fp(q)
    for i in sorted(fix):
        if i > len(dq):
            dq = np.append(dq, 0)
        else:
            dq = np.insert(dq, i, 0)

    # de-degenerate parameters and recalculate position from crop to frame
    q[2] = np.abs(q[2])
    q[0:2] += cc[:, 0]
    q[5] = np.abs(q[5])
    # q[6] %= np.pi
    q[6] = (q[6] + np.pi / 2) % np.pi - np.pi / 2

    # Chi-squared, R-squared, signal to noise ratio
    chisq = r.fun / (S[0] * S[1] - nPar)
    R2 = 1 - r.fun / np.nansum((jm - np.nanmean(jm)) ** 2)
    sn = q[3] / np.sqrt(r.fun / (S[0] * S[1])) / 2 / np.pi / q[2] ** 2

    return q, dq, (chisq, R2, sn)


def fitgaussint(im, xy=None, theta=None, mesh=None):
    """ finds initial parameters for a 2d Gaussian fit
        q = (x, y, fwhm, area, offset, ellipticity, angle) if 2D
        q = (x, y, z, fwhm, fwhmz, area, offset) if 3D
        wp@tl20191010
    """

    dim = np.ndim(im)
    S = np.shape(im)
    q = np.full(7, 0).astype('float')

    if dim == 2:
        if mesh is None:
            x, y = np.meshgrid(range(S[0]), range(S[1]))
        else:
            x, y = mesh
        q[4] = np.nanmin(im)
        q[3] = np.nansum((im - q[4]))

        if xy is None:
            q[0] = np.nansum(x * (im - q[4])) / q[3]
            q[1] = np.nansum(y * (im - q[4])) / q[3]
        else:
            q[:2] = xy

        if theta is None:
            tries = 10
            e = []
            t = np.delete(np.linspace(0, np.pi, tries + 1), tries)
            for th in t:
                e.append(fitgaussint(im, xy, th, (x, y))[5])
            q[6] = (fitcosint(2 * t, e)[2] / 2 + np.pi / 2) % np.pi - np.pi / 2
        else:
            q[6] = theta

        cos, sin = np.cos(q[6]), np.sin(q[6])
        x, y = cos * (x - q[0]) - (y - q[1]) * sin, cos * (y - q[1]) + (x - q[0]) * sin

        s2 = np.nansum((im - q[4]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[4])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[4])) ** 2) / s2)

        q[2] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[5] = np.sqrt(sx / sy)
    else:
        if mesh is None:
            x, y, z = np.meshgrid(range(S[0]), range(S[1]), range(S[2]))
        else:
            x, y, z = mesh
        q[6] = np.nanmin(im)
        q[5] = np.nansum((im - q[6]))

        if xy is None:
            q[0] = np.nansum(x * (im - q[6])) / q[5]
            q[1] = np.nansum(y * (im - q[6])) / q[5]
            q[2] = np.nansum(z * (im - q[6])) / q[5]
        else:
            q[:3] = xy

        x, y, z = x - q[0], y - q[1], z - q[2]

        s2 = np.nansum((im - q[6]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[6])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[6])) ** 2) / s2)
        sz = np.sqrt(np.nansum((z * (im - q[6])) ** 2) / s2)

        q[3] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[4] = sz * 4 * np.sqrt(np.log(2))
    return q


def fitcosint(theta, y):
    """ Finds parameters to y=a*cos(theta-psi)+b
        wp@tl20191010
    """
    b = np.trapz(y, theta) / np.mean(theta) / 2
    a = np.trapz(np.abs(y - b), theta) / 4

    t = np.sin(theta)
    s = np.cos(theta)

    T = np.sum(t)
    S = np.sum(s)
    A = np.sum(y * t)
    B = np.sum(y * s)
    C = np.sum(t ** 2)
    D = np.sum(t * s)
    E = np.sum(s ** 2)

    q = np.dot(np.linalg.inv(((C, D), (D, E))) / a, (A - b * T, B - b * S))

    psi = (np.arctan2(*q)) % (2 * np.pi)
    if q[1] < 0:
        a *= -1
        psi -= np.pi
    psi = (psi + np.pi) % (2 * np.pi) - np.pi

    return np.array((a, b, psi))


def fit_tilted_plane(im):
    """ Linear regression to determine z0, a, b in z = z0 + a*x + b*y

        nans and infs are filtered out

        input: 2d array containing z (x and y will be the pixel numbers)
        output: array [z0, a, b]

        wp@tl20190819
    """
    S = im.shape
    im = im.flatten()

    # vector [1, x_ij, y_ij] and filter nans and infs
    xv, yv = np.meshgrid(*map(range, S))
    v = [i.flatten()[np.isfinite(im)] for i in [np.ones(im.shape), xv, yv]]
    # construct matrix for the regression
    Q = [[np.sum(i * j) for i in v] for j in v]
    return np.dot(np.linalg.inv(Q), [np.sum(im[np.isfinite(im)] * i) for i in v])
