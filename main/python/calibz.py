# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import scipy, os, pandas, re, pickle
import trackpy as tp
from tqdm.auto import trange
import warnings
from imread import imread
import functions, utilities, localisation
import functools
import multiprocessing
import skimage.filters


def zhuang_fun(z, p):
    """ p: [sigma0,A,B,c,d]
        wp@tl2019
    """
    X = (z - p[3]) / p[4]
    return p[0] * np.sqrt(1 + X ** 2 + p[1] * X ** 3 + p[2] * X ** 4)


def zhuang_funA(z, p):
    # p: [sigma0,A,B,c,d]
    X = (z - p[3]) / p[4]
    r = (-3 * p[1] + np.sqrt(9 * p[1] ** 2 - 32 * p[2])) / 8 / p[2]
    l = (-3 * p[1] - np.sqrt(9 * p[1] ** 2 - 32 * p[2])) / 8 / p[2]
    l, r = min(l, r), max(l, r)
    X[X < l] = l
    X[X > r] = r
    return p[0] * np.sqrt(1 + X ** 2 + p[1] * X ** 3 + p[2] * X ** 4)


def fitzhuangint(z, s):
    p = np.zeros((5, 1))
    p[0] = np.nanmin(s)
    p[3] = z[np.nanargmin(s)]
    with np.errstate(divide='ignore'):
        d = np.sqrt((z - p[3]) ** 2 / ((s / p[0]) ** 2 - 1))
    d[np.isinf(d)] = np.nan
    p[4] = np.nanmean(d)
    # Z = np.linspace(np.nanmin(z),np.nanmax(z))
    # plt.plot(z,s,'.',Z,zhuang_fun(Z,p),'--')
    # plt.plot(Z,zhuang_fun(Z,p),'--')
    return p


def fitzhuang(z, s, ds, rec=True):
    z, s, ds = utilities.rmnan(z, s, ds)
    if z.size == 0:
        return np.full(5, np.nan), np.full(5, np.nan), (np.nan, np.nan)
    p = fitzhuangint(z, s)
    g = lambda pf: np.nansum((s - zhuang_fun(z, pf)) ** 2 / ds ** 2)
    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x
    if q.size >= z.size:
        return np.full(5, np.nan), np.full(5, np.nan), (np.nan, np.nan)
    else:
        dq = np.sqrt(r.fun / (z.size - q.size) * np.diag(r.hess_inv))
        X2 = r.fun / (z.size - p.size)
        R2 = 1 - np.nansum((s - zhuang_fun(z, q)) ** 2) / np.sum((s - np.mean(s)) ** 2)
    # Z = np.linspace(np.nanmin(z),np.nanmax(z))
    # plt.plot(Z,zhuang_fun(Z,q))
    if rec and np.isnan(zhuang_fun(0, q.T)):
        return fitzhuang(z, s, 1, False)
    return q.T, dq.T, (X2, R2)


def filterp(p, dp, pxsize):
    a = np.ones((p.shape[0], 1))

    # sigma
    a[p[:, 2] / 2 / np.sqrt(2 * np.log(2)) < 0.150 / pxsize] = 0
    a[p[:, 2] / 2 / np.sqrt(2 * np.log(2)) > 0.600 / pxsize] = 0

    # intensity
    a[p[:, 3] < 10000] = 0
    # a[p[:,3]>1e9] = 0

    # ellipticity
    a[p[:, 5] < 3 / 5] = 0
    a[p[:, 5] > 5 / 3] = 0

    # dx, dy
    a[dp[:, 0] > 0.03 / pxsize] = 0
    a[dp[:, 1] > 0.03 / pxsize] = 0

    idx = np.where(a)[0]
    return p[idx, :], dp[idx, :], idx


def filterq(q, dq):
    q = q[50:60, :]
    dq = dq[50:60, :]
    dx = 0.5
    m = np.min(dq[:, 0:2], 1)
    dq = dq[m < dx]
    q = q[m < dx]
    dq = dq[q[:, 2] < 20]
    q = q[q[:, 2] < 20]
    return q, dq


def findcandidates(im, c):
    """ finds peaks in a max intensity projection
        im: imopen object
        c: color channel number
        output: dataframe with x, y locations and angle of psf in xy plane
        wp@tl20190200
    """
    warnings.filterwarnings('ignore')
    mx = np.squeeze(np.max(im.block(c), 3))
    f = tp.locate(mx, 11)
    f = f[f['mass'] > skimage.filters.threshold_triangle(f['mass']) / np.sqrt(2)]
    th = np.zeros(f.shape[0])
    dth = np.zeros(f.shape[0])
    for i in trange(f.shape[0]):
        idx = f.index[i]
        q, dq = fitstack(im, c, (f['x'][idx], f['y'][idx]))
        q, dq, _ = filterp(q, dq, im.pxsize)
        th[i], dth[i] = utilities.circ_weightedmean(q[:, -1], dq[:, -1], np.pi / 2)

        # or maybe should use this:
        # idx = utilities.outliers(q[:,-1])
        # theta = q[:,-1][idx]
        # dtheta = dq[:,-1][idx]
        # th[i], dth[i] = functions.distfit(theta)

    f['theta'] = th
    f['dtheta'] = dth
    return f


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


def calibz(im):
    if not isinstance(im, imread):
        im = imread(im)

    pdfpath = os.path.splitext(im.path)[0] + '_Cyllens_calib.pdf'
    os.makedirs(os.path.split(pdfpath)[0], exist_ok=True)
    pdf = PdfPages(pdfpath)

    a = localisation.detect_points_sf(im.maxz(1), .35 / im.NA / im.pxsize)

    a = a.query('R2>0').copy()
    a['particle'] = range(len(a))

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 3, figure=fig)

    fig.add_subplot(gs[:2, :2])
    plt.imshow(im.maxz(1))
    plt.plot(a['x'], a['y'], 'or', markerfacecolor='none')
    for i in a.index:
        plt.text(a.loc[i, 'x'], a.loc[i, 'y'], a.loc[i, 'particle'], color='w')

    a['particle'] = range(len(a))

    d = pandas.DataFrame()
    C = im.masterch

    zmax = np.round(15 / im.deltaz).astype('int')
    zmax = min(zmax, im.shape[3])

    for T in range(im.shape[4]):
        for Z in range(zmax):
            b = a.copy()
            b['C'] = C
            b['Z'] = Z
            b['T'] = T
            b['z_um'] = Z * im.deltaz
            b['frame'] = im.czt2n(C, Z, T)
            d = pandas.concat((d, b))
    d = d.reset_index(drop=True)

    sbu = im.sigma
    # im.sigma = lambda x: 1.6
    #im.frame_decorator = lambda im, frame, c, z, t: gfilter(frame, im.sigma(c))

    a = localisation.tpsuperresseq(d, im, theta=True, tilt=False, filter=False)

    fig.add_subplot(gs[0, 2])
    plt.hist((a.query('R2>0.8')['theta'] + np.pi / 4) % (np.pi / 2) - np.pi / 4, 100);
    plt.xlabel('theta')
    pdf.savefig(fig)

    theta, dtheta = utilities.circ_weightedmean(a.query('R2>0.8')['theta'], a.query('R2>0.8')['dtheta'], np.pi / 2)
    #theta *= -1
    print('θ = {} ± {}'.format(theta, dtheta))

    a = localisation.tpsuperresseq(d, im, theta=theta, tilt=False, filter=True)
    a['particle'] = d['particle']
    a['s_um'] = a['s'] * im.pxsize
    a['ds_um'] = a['ds'] * im.pxsize
    a['dx_um'] = a['dx'] * im.pxsize
    a['dy_um'] = a['dy'] * im.pxsize

    im.sigma = sbu
    im.frame_decorator = None

    # individual Zhuang fits
    nColumns = 3

    a0 = a.query('R2>0.9 & 0.1<s_um<0.6 & 2/3<e<3/2 & dx_um<0.05 & dy_um<0.05 & de<0.2 & ds_um<0.2').copy()

    particles = set(a0['particle'])
    lp = len(particles)
    fig = plt.figure(figsize=(15, lp))
    gs = GridSpec(int(np.ceil(lp / nColumns)), nColumns, figure=fig)

    pr, px, dpx, py, dpy, X2x, X2y, R2x, R2y, z, sx, dsx, sy, dsy, Nx, Ny = ([] for i in range(16))

    for i, p in enumerate(particles):
        b = a0.query('particle=={}'.format(p)).copy()

        pr.append(int(p))
        z.append(np.array(b['z_um']))
        sx.append(np.array(b['s_um'] * np.sqrt(b['e'])))
        sy.append(np.array(b['s_um'] / np.sqrt(b['e'])))
        dsx.append(np.array(np.sqrt(b['ds_um'] ** 2 * b['e'] + b['s_um'] ** 2 * b['de'] ** 2 / 4 / b['e'])))
        dsy.append(np.array(dsx[-1] / b['e']))

        rs = 2
        Z = utilities.findrange(z[-1], rs)
        z[-1][z[-1] < (Z - rs / 2)] = np.nan
        z[-1][z[-1] > (Z + rs / 2)] = np.nan

        k = fitzhuang(z[-1], sx[-1], dsx[-1])
        px.append(k[0])
        dpx.append(k[1])
        X2x.append(k[2][0])
        R2x.append(k[2][1])
        k = fitzhuang(z[-1], sy[-1], dsy[-1])
        py.append(k[0])
        dpy.append(k[1])
        X2y.append(k[2][0])
        R2y.append(k[2][1])

        fig.add_subplot(gs[int(i / nColumns), int(i % nColumns)])
        plt.errorbar(z[-1], sx[-1], yerr=dsx[-1], fmt='.r', )
        plt.errorbar(z[-1], sy[-1], yerr=dsy[-1], fmt='.g', )
        # plt.title('X2={:.2f}; {:.2f}'.format(X2x[i], X2y[i]))
        if not np.isnan(px[-1]).all():
            Z = np.linspace(np.nanmin(z[-1]), np.nanmax(z[-1]))
            plt.plot(Z, zhuang_fun(Z, px[-1]), '-r', Z, zhuang_fun(Z, py[-1]), '-g')
            # plt.plot(Z, zhuang_fun(Z, px[i])/zhuang_fun(Z, py[i]), '-r')
            plt.xlabel('z (um)')
            plt.ylabel('s (um)')
        # plt.ylim(-0.5, 1.5)
        plt.ylim(0, 0.5)
    plt.tight_layout()
    pdf.savefig(fig)

    px = np.array(px)
    dpx = np.array(dpx)
    py = np.array(py)
    dpy = np.array(dpy)

    g = a.query('R2>0.9 & 0.15<s_um<0.6 & 2/3<e<3/2 & dx_um<0.05 & dy_um<0.05 & de<0.2 & ds_um<0.2').copy()
    x = [g.query('particle=={}'.format(i))['x'].mean() for i in pr]
    y = [g.query('particle=={}'.format(i))['y'].mean() for i in pr]

    f = pandas.DataFrame({'x': x, 'y': y, 'particle': pr})

    f['X2x'] = X2x
    f['R2x'] = R2x
    f['sigmax'] = px[:, 0]
    f['dsigmax'] = dpx[:, 0]
    f['Ax'] = px[:, 1]
    f['dAx'] = dpx[:, 1]
    f['Bx'] = px[:, 2]
    f['dBx'] = dpx[:, 2]
    f['cx'] = px[:, 3]
    f['dcx'] = dpx[:, 3]
    f['dx'] = px[:, 4]
    f['ddx'] = dpx[:, 4]
    f['X2y'] = X2y
    f['R2y'] = R2y
    f['sigmay'] = py[:, 0]
    f['dsigmay'] = dpy[:, 0]
    f['Ay'] = py[:, 1]
    f['dAy'] = dpy[:, 1]
    f['By'] = py[:, 2]
    f['dBy'] = dpy[:, 2]
    f['cy'] = py[:, 3]
    f['dcy'] = dpy[:, 3]
    f['dy'] = py[:, 4]
    f['ddy'] = dpy[:, 4]
    f['dc'] = px[:, 3] - py[:, 3]
    f['ddc'] = np.sqrt(dpx[:, 3] ** 2 + dpy[:, 3] ** 2)

    f0 = f.copy()
    #print(f['dc'])

    dc = [i/(im.immersionN**2/1.33**2) for i in (0.1, 0.6)]
    #dc = (0.1, 0.6)

    f = f.query('0.1<sigmax<0.5 & 0.1<sigmay<0.5 & abs(Ax)<10 & abs(Bx)<10 & abs(Ay)<10 & abs(By)<10 & X2x<20 & X2y<20 & {}<dc<{} & cx<10 & cy<10'.format(*dc))

    Zx = np.array(())
    Zy = np.array(())
    Sx = np.array(())
    Sy = np.array(())
    dSx = np.array(())
    dSy = np.array(())
    E = np.array(())
    Ze = np.array(())

    fig = plt.figure(figsize=(15, 3))
    gs = GridSpec(1, 3)

    ### x

    fig.add_subplot(gs[0, 0])
    for i in range(f.shape[0]):
        idx = f.index[i]
        Zx = np.append(Zx, z[idx] - f['cx'][idx])
        Sx = np.append(Sx, sx[idx])
        dSx = np.append(dSx, dsx[idx])
        plt.plot(z[idx] - f['cx'][idx], sx[idx], '.')

    # Zx[Zx<-1] = np.nan
    # Zx[Zx> 1] = np.nan
    qx, dqx, X2x = fitzhuang(Zx, Sx, dSx)
    if not np.isnan(qx).all():
        Z = np.linspace(np.nanmin(Zx), np.nanmax(Zx))
        plt.plot(Z, zhuang_fun(Z, qx), 'r-')
    plt.xlabel('z (um)')
    plt.ylabel('sigma_x (um)')

    # plt.xlim(-1,1)
    plt.ylim(0, .5)

    ### y

    fig.add_subplot(gs[0, 1])
    for i in range(f.shape[0]):
        idx = f.index[i]
        Zy = np.append(Zy, z[idx] - f['cy'][idx])
        Sy = np.append(Sy, sy[idx])
        dSy = np.append(dSy, dsy[idx])
        plt.plot(z[idx] - f['cy'][idx], sy[idx], '.')

    # Zx[Zx<-1] = np.nan
    # Zx[Zx> 1] = np.nan
    qy, dqy, X2y = fitzhuang(Zy, Sy, dSy)
    if not np.isnan(qy).all():
        Z = np.linspace(np.nanmin(Zy), np.nanmax(Zy))
        plt.plot(Z, zhuang_fun(Z, qy), 'r-')
    plt.xlabel('z (um)')
    plt.ylabel('sigma_y (um)')

    # plt.xlim(-1,1)
    plt.ylim(0, .5)

    dc, ddc = utilities.weightedmean(f['dc'], f['ddc'])

    dc += qx[3] - qy[3]
    ddc = np.sqrt(ddc ** 2 + dqx[3] ** 2 + dqy[3] ** 2)

    qx[3] = dc / 2
    qy[3] = -dc / 2
    dqx[3] = ddc / 2
    dqy[3] = ddc / 2

    Z = np.linspace(-1, 1)
    E = np.array(())
    Ze = np.array(())

    fig.add_subplot(gs[0, 2])

    for i in range(f.shape[0]):
        idx = f.index[i]
        plt.plot(z[idx] - (f['cx'][idx] + f['cy'][idx]) / 2, sx[idx] / sy[idx], '.')
        E = np.append(E, sx[idx] / sy[idx])
        Ze = np.append(Ze, z[idx] - (f['cx'][idx] + f['cy'][idx]) / 2)
    q = np.hstack(
        (np.sqrt(qx[0] / qy[0]), (qx[3] + qy[3]) / 2, (qx[3] - qy[3]) / 2, qx[1], qx[2], qx[4], qy[1], qy[2], qy[4]))
    rl, = plt.plot(Z, zhuangell(Z, q), 'r-')
    E[E > 1.5] = np.nan
    E[E < 0.6] = np.nan
    Ze[Ze < -1] = np.nan
    Ze[Ze > 1] = np.nan
    E[utilities.outliers(E, False)] = np.nan
    q, dq, X2 = fitzhuangell(Ze, E, q)
    gl, = plt.plot(Z, zhuangell(Z, q), 'g-')
    plt.xlabel('z (um)')
    plt.ylabel('ellipticity (um)')
    plt.legend((rl, gl), ('sx/sy', 'fit'))
    # plt.xlim(-1,1)
    plt.ylim(0.5, 1.5)

    plt.tight_layout()
    pdf.savefig(fig)

    txt = 'θ = {} ± {}\ne0: {} ± {}\nz0: {} ± {}\nc : {} ± {}\nAx: {} ± {}\nBx: {} ± {}\ndx: {} ± {}\nAy: {} ± {}\nBy: {} ± {}\ndy: {} ± {}'.format(
        theta, dtheta, q[0], dq[0], q[1], dq[1], q[2], dq[2], q[3], dq[3], q[4], dq[4], q[5], dq[5], q[6], dq[6], q[7],
        dq[7], q[8], dq[8])

    fig = plt.figure(figsize=(15, 3))
    plt.text(0.05, 0.5, txt, va='center');
    plt.axis('off');
    pdf.savefig(fig)
    pdf.close()

    cyllens = 'A'
    r = re.search('(?<=\s)\d+x', im.objective)
    if r:
        m = r.group(0)
    else:
        m = ''

    s = '{}{}{:.0f}:'.format(cyllens, m, im.optovar[0] * 10)
    s += '\n  q: [{}, {}, {}, {}, {}, {}, {}, {}, {}]'.format(*q)
    s += '\n  theta: {}'.format(theta)

    print('To put in CylLensGUI config file:')
    print(s)

    res = {'confstr': s, 'q': q, 'dq': dq, 'X2': X2, 'theta': theta, 'dtheta': dtheta, 'dc': dc, 'ddc': ddc, 'f': f,
           'f0': f0, 'a': a, 'Zx': Zx, 'Zy': Zy, 'Sx': Sx, 'Sy': Sy, 'dSx': dSx, 'dSy': dSy, 'Ze': Ze, 'E': E, 'qx': qx,
           'dqx': dqx, 'X2x': X2x, 'qy': qy, 'dqy': dqy, 'X2y': X2y}

    try:
        im.close()
    except:
        pass

    #with open(pdfpath.replace('.pdf', '.pkl'), 'wb') as file:
    #    pickle.dump(res, file)
    return theta, q


def zhuangell(z, q):
    """ Ellipticity as function of z, with parameter q
        q: [sx/sy,z0,c,Ax,Bx,dx,Ay,By,dy]
        wp@tl20190227
    """
    X = (z - q[2] - q[1]) / q[5]
    Y = (z + q[2] - q[1]) / q[8]
    return q[0] * np.sqrt((1 + X ** 2 + q[3] * X ** 3 + q[4] * X ** 4) / (1 + Y ** 2 + q[6] * Y ** 3 + q[7] * Y ** 4))


def fitzhuangell(z, e, p):
    """ Find calibration parameter q
        q:    [sx/sy,z0,c,Ax,Bx,dx,Ay,By,dy]
        z, e: data of ellipticity vs z
        p:    initial guess of q
        wp@tl20190227
    """
    z, e = utilities.rmnan(z, e)
    de = 1
    if z.size == 0:
        return np.full((1, 9), np.nan), np.full((1, 9), np.nan), np.nan
    g = lambda pf: np.nansum((e - zhuangell(z, pf)) ** 2)
    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x
    if q.size >= z.size:
        dq = np.full(q.size, np.nan)
    else:
        dq = np.sqrt(r.fun / (z.size - q.size) * np.diag(r.hess_inv))
    X2 = r.fun / (z.size - p.size)
    return q, dq, X2


def findzhuangrange(q):
    """ Finds the usable range of z in the zhuang function with parameter q,
        ie. the range between peak and valey around z=0
        wp@tl20190227
    """
    Lx = np.repeat((q[4] / q[5] ** 4, q[3] / q[5] ** 3, 1 / q[5] ** 2, 0, 1), 5).reshape((5, 5)).T
    Ly = np.repeat((q[7] / q[8] ** 4, q[6] / q[8] ** 3, 1 / q[8] ** 2, 0, 1), 5).reshape((5, 5)).T
    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    N1 = scipy.special.binom(T.T, T)
    N2 = scipy.special.binom(T.T - 1, T)
    N2[np.isnan(N2)] = 0
    S = (-np.ones((5, 5))) ** (T.T - T)
    za = np.sum(-N1 * (q[1] - q[2]) ** (T.T - T) * Ly * S, 1)
    zb = np.sum(T.T * N2 * (q[1] + q[2]) ** (T.T - T - 1) * Lx * S, 1)
    zc = np.sum(T.T * N2 * (q[1] - q[2]) ** (T.T - T - 1) * Ly * S, 1)
    zd = np.sum(N1 * (q[1] + q[2]) ** (T.T - T) * Lx * S, 1)

    q = np.full(8, np.nan)
    q[0] = za[0] * zb[1] + zc[1] * zd[0]
    q[1] = np.matmul(za[::-1][3:5], zb[1:3]) + np.matmul(zc[::-1][2:4], zd[0:2])
    q[2] = np.matmul(za[::-1][2:5], zb[1:4]) + np.matmul(zc[::-1][1:4], zd[0:3])
    q[3] = np.matmul(za[::-1][1:5], zb[1:5]) + np.matmul(zc[::-1][0:4], zd[0:4])
    q[4] = np.matmul(za[::-1][0:4], zb[1:5]) + np.matmul(zc[::-1][0:4], zd[1:5])
    q[5] = np.matmul(za[::-1][0:3], zb[2:5]) + np.matmul(zc[::-1][0:3], zd[2:5])
    q[6] = np.matmul(za[::-1][0:2], zb[3:5]) + np.matmul(zc[::-1][0:2], zd[3:5])
    q[7] = za[4] * zb[4] + zc[4] * zd[4]

    rts = np.roots(q)
    rts = rts[np.isreal(rts)]

    return np.real(np.array((np.max(rts[rts < 0]), np.min(rts[rts > 0]))))


def findz(ell, q, dell=None, dq=None):
    """ Find the z position of a psf with known ellipticity.
        ell:    ellipticity of the psf, ell = sigma_x/sigma_y
        q:
         e0:     ellipticity of psf in focus
         z0:     z when psf in focus
         c:      offset of x focus above, and y focus below z0
         Ax, Bx: 3rd and 4th order calibration parameters
         dx:     focal depth
         Ay, By, dy: ^
        wp@tl20190227
    """
    R2 = (ell / q[0]) ** 2
    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    Lx = np.repeat((q[4] / q[5] ** 4, q[3] / q[5] ** 3, 1 / q[5] ** 2, 0, 1), 5).reshape((5, 5)).T
    Ly = np.repeat((q[7] / q[8] ** 4, q[6] / q[8] ** 3, 1 / q[8] ** 2, 0, 1), 5).reshape((5, 5)).T
    N = scipy.special.binom(T.T, T)
    S = (-np.ones((5, 5))) ** (T.T - T)

    P = np.sum((R2 * Ly * (q[1] - q[2]) ** (T.T - T) - Lx * (q[1] + q[2]) ** (T.T - T)) * S * N, 1)
    if np.any(np.isnan(P)):
        if not dell is None:
            return (np.nan, np.nan)
        else:
            return np.nan

    rts = np.roots(P)
    rts = rts[np.isreal(rts)]
    rng = findzhuangrange(q)

    if (np.size(rts) == 0) | (not np.any((rts >= rng[0]) & (rts <= rng[1]))):
        if not dell is None:
            return (np.nan, np.nan)
        else:
            return np.nan

    z = np.real(rts[(rts >= rng[0]) & (rts <= rng[1])][0])

    if not dell is None:
        # attempt at error estimation
        dedz = scipy.misc.derivative(lambda x: zhuangell(z, q), z)
        s = 0
        for i in range(len(q)):
            s += (scipy.misc.derivative(lambda x: zhuangell(z, np.hstack((q[:i], x, q[(i + 1):]))), q[i]) * dq[i]) ** 2
        dz = np.sqrt((dell ** 2 + s) / dedz ** 2)
        z = (z, dz)

    return z


### Functions used for testing how good the calibration and/or bead z-stack is

def errfun(x, p):
    if len(p) == 2:
        return p[0] * np.sin(x) * np.exp(-(x / p[1]) ** 2)
    else:
        return p[0] * np.sin(x - p[2]) * np.exp(-((x - p[2]) / p[1]) ** 2)


def errfit(x, y, offset=False):
    g = lambda p: np.nansum((y - errfun(x, p)) ** 2)
    p = np.array((1, 1))
    if offset:
        if np.all(np.isnan(y)) | np.all(np.isnan(x)) | (len(y) < 2):
            p = np.append(p, 0)
        else:
            gr = np.gradient(y, x)
            if np.all(np.isnan(gr)):
                p = np.append(p, 0)
            else:
                p = np.append(p, x[np.nanargmax(gr)])
    return scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5}).x


def errfun_pk(p):
    #     p = p[1] (from errfun)
    g = lambda z: (1 / np.tan(z) - 2 * z / p ** 2) ** 2
    return scipy.optimize.minimize(g, 2 * p / 3, options={'disp': False, 'maxiter': 1e5}).x


def teststack(im, c, q, theta):
    import trackpy as tp
    mx = np.squeeze(np.max(im.block(c), 3))
    f = tp.locate(mx, 11)
    f = f[f['mass'] > skimage.filters.threshold_triangle(f['mass']) / np.sqrt(2)]
    X = np.arange(im.shape[3]) * im.deltaz
    d = list()
    x = list()
    plt.figure(figsize=(8, 8))
    tp.annotate(f, mx)
    plt.figure(figsize=(8, 3.5 * (f.shape[0] + 1)))
    plt.subplot(f.shape[0] + 1, 1, 1, title='image')

    for i in trange(f.shape[0]):
        ind = f.index[i]
        p, dp = fitstack(im, c, (f['x'][ind], f['y'][ind]), theta)
        p, dp, idx = filterp(p, dp, im.pxsize)

        if len(idx) == 0:
            x.append(np.nan)
            d.append(np.nan)
            continue

        # disregard peaks where ell~=1 most of the time
        m = scipy.stats.mode(p[:, 5])[0][0]
        marg = 0.01
        if (m > 1 - marg) & (m < 1 + marg):
            x.append(np.nan)
            d.append(np.nan)
            continue
        z = [findz(e, q) for e in p[:, 5]]
        r = errfit(X[idx], z, True)[2]
        x.append(X[idx] - r)
        plt.subplot(f.shape[0] + 1, 1, i + 1, title='{}'.format(i))
        d.append(z - x[i])
        plt.plot(x[i], z, x[i], x[i], x[i], np.abs(z - x[i]))
        plt.xlim((-1, 1))
        plt.ylim((-0.5, 0.5))
    plt.subplot(f.shape[0] + 1, 1, f.shape[0] + 1, title='errors')
    Z = []
    D = []
    for i in range(f.shape[0]):
        Z = np.append(Z, x[i])
        D = np.append(D, d[i])
        plt.plot(x[i], d[i], '.')
    plt.xlim((-1, 1))
    plt.ylim((-0.5, 0.5))

    return Z, D


def errplot(x, d):
    l = 2

    x, d = utilities.rmnan(x, d)
    if len(x) == 0:
        return

    plt.figure(figsize=(8, 8))

    Z = np.linspace(-l, l)
    p = errfit(x, d + x)

    r = (2 * errfun_pk(p[1]) / 3)[0]
    plt.plot(x, d, '.b', Z, -Z, '--g')
    plt.plot(Z, errfun(Z, p) - Z, '-r', Z, np.zeros(np.shape(Z)), '--k')
    plt.plot(r, errfun(r, p) - r, 'yo', -r, errfun(-r, p) + r, 'yo')
    plt.plot(np.full(np.shape(Z), r), Z, '--y', np.full(np.shape(Z), -r), Z, '--y')
    plt.xlim(-l, l)
    plt.ylim(-l, l)
    plt.xlabel('z (um)')
    plt.ylabel('error (um)')

    idx = np.where(np.abs(x) < r)
    xr = x[idx]
    dr = d[idx]

    print("Usable range: {} nm\nStandard deviation of the error: {} nm".format(int(2000 * r), int(1000 * np.std(dr))))


### Old functions below here

def findzxy(ell, qx, qy, dell=None, dqx=None, dqy=None):
    """ Find the z position of a psf with known ellipticity.
        ell:    ellipticity of the psf, ell = sigma_x/sigma_y
        qx, qy:
        sigma0: sigma of psf in focus
        c:      offset of x focus above, and y focus below z=0
        d:      focal depth
        A, B:   3rd and 4th order calibration parameters
    """
    R2 = (ell * qy[0] / qx[0]) ** 2
    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    Lx = np.repeat((qx[2] / qx[4] ** 4, qx[1] / qx[4] ** 3, 1 / qx[4] ** 2, 0, 1), 5).reshape((5, 5)).T
    Ly = np.repeat((qy[2] / qy[4] ** 4, qy[1] / qy[4] ** 3, 1 / qy[4] ** 2, 0, 1), 5).reshape((5, 5)).T
    N = scipy.special.binom(T.T, T)
    S = (-np.ones((5, 5))) ** (T.T - T)

    rts = np.roots(np.sum((R2 * Ly * qy[3] ** (T.T - T) - Lx * qx[3] ** (T.T - T)) * S * N, 1))
    rts = rts[np.isreal(rts)]
    rng = findzhuangrangexy(qx, qy)

    if (np.size(rts) == 0) | (not np.any((rts >= rng[0]) & (rts <= rng[1]))):
        z = np.nan
    else:
        z = np.real(rts[(rts >= rng[0]) & (rts <= rng[1])][0])

    if not dell is None:
        # attempt at error estimation
        q = np.append(qx, qy)
        dq = np.append(dqx, dqy)
        e = lambda z, q: zhuang_fun(z, q[:5]) / zhuang_fun(z, q[5:])
        dedz = scipy.misc.derivative(lambda x: e(x, q), z)
        s = 0
        for i in range(len(q)):
            s += (scipy.misc.derivative(lambda x: e(z, np.hstack((q[:i], x, q[(i + 1):]))), q[i]) * dq[i]) ** 2
        dz = np.sqrt((dell ** 2 + s) / dedz ** 2)
        z = (z, dz)
    else:
        dz = np.nan
    return z


def findzhuangrangexy(qx, qy):
    Lx = np.repeat((qx[2] / qx[4] ** 4, qx[1] / qx[4] ** 3, 1 / qx[4] ** 2, 0, 1), 5).reshape((5, 5)).T
    Ly = np.repeat((qy[2] / qy[4] ** 4, qy[1] / qy[4] ** 3, 1 / qy[4] ** 2, 0, 1), 5).reshape((5, 5)).T
    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    N1 = scipy.special.binom(T.T, T)
    N2 = scipy.special.binom(T.T - 1, T)
    N2[np.isnan(N2)] = 0
    S = (-np.ones((5, 5))) ** (T.T - T)
    za = np.sum(-N1 * qy[3] ** (T.T - T) * Ly * S, 1)
    zb = np.sum(T.T * N2 * qx[3] ** (T.T - T - 1) * Lx * S, 1)
    zc = np.sum(T.T * N2 * qy[3] ** (T.T - T - 1) * Ly * S, 1)
    zd = np.sum(N1 * qx[3] ** (T.T - T) * Lx * S, 1)

    q = np.full(8, np.nan)
    q[0] = za[0] * zb[1] + zc[1] * zd[0]
    q[1] = np.matmul(za[::-1][3:5], zb[1:3]) + np.matmul(zc[::-1][2:4], zd[0:2])
    q[2] = np.matmul(za[::-1][2:5], zb[1:4]) + np.matmul(zc[::-1][1:4], zd[0:3])
    q[3] = np.matmul(za[::-1][1:5], zb[1:5]) + np.matmul(zc[::-1][0:4], zd[0:4])
    q[4] = np.matmul(za[::-1][0:4], zb[1:5]) + np.matmul(zc[::-1][0:4], zd[1:5])
    q[5] = np.matmul(za[::-1][0:3], zb[2:5]) + np.matmul(zc[::-1][0:3], zd[2:5])
    q[6] = np.matmul(za[::-1][0:2], zb[3:5]) + np.matmul(zc[::-1][0:2], zd[3:5])
    q[7] = za[4] * zb[4] + zc[4] * zd[4]

    rts = np.roots(q)
    rts = rts[np.isreal(rts)]

    return np.real(np.array((np.max(rts[rts < 0]), np.min(rts[rts > 0]))))


def findz_ellinit(ell, qx, qy):
    """ Use 1st order Taylor expansion to find initial z from ellipticity
    """
    eps = 1e-6
    e = lambda z: zhuang_fun(z, qx) / zhuang_fun(z, qy)
    d = (e(eps) - e(-eps)) / 2 / eps
    return (ell - e(0)) / d


def findz_ell(ell, qx, qy):
    """ Find the z position of a psf with known ellipticity.
        ell:    ellipticity of the psf, ell = sigma_x/sigma_y
        qx, qy:
        sigma0: sigma of psf in focus
        c:      offset of x focus above, and y focus below z=0
        d:      focal depth
        A, B:   3rd and 4th order calibration parameters
    """
    e = lambda z: zhuang_fun(z, qx) / zhuang_fun(z, qy)
    g = lambda z: (ell - e(z)) ** 2
    z0 = findz_ellinit(ell, qx, qy)
    r = scipy.optimize.minimize(g, z0, options={'disp': False, 'maxiter': 1000})
    z = r.x
    dz = np.sqrt(r.fun * np.diag(r.hess_inv))
    # print(e(z))
    return z[0], dz[0]


def findz_ell_sig(sigma, ell, qx, qy):
    """ Find the z position of a psf with known sigma and ellipticity.
        sigma:  sigma of the psf, sigma = sqrt(sigma_x*sigma_y)
        ell:    ellipticity of the psf, ell = sigma_x/sigma_y
        sigma0: sigma of psf in focus
        c:      offset of x focus above, and y focus below z=0
        d:      focal depth
        A, B:   3rd and 4th order calibration parameters
    """
    e = lambda z: zhuang_fun(z, qx) / zhuang_fun(z, qy)
    s = lambda z: np.sqrt(zhuang_fun(z, qx) * zhuang_fun(z, qy))
    g = lambda z: np.sqrt((ell - e(z)) ** 2 + (np.sqrt(sigma) - np.sqrt(s(z))) ** 2)
    z0 = (2 * (ell > 1) - 1) * np.sqrt(
        np.max((c / d) ** 2 - 1 + np.array((-1, 1)) * np.sqrt((sigma / sigma0) ** 4 - 4 * (c / d) ** 2)))
    r = scipy.optimize.minimize(g, z0, options={'disp': False, 'maxiter': 1000})
    z = r.x
    dz = np.sqrt(r.fun * np.diag(r.hess_inv))
    return z[0], dz[0]
