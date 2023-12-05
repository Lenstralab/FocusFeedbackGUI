import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
import skimage.filters
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from ndbioimage import Imread
from parfor import parfor

from focusfeedbackgui import functions as fn
from focusfeedbackgui import utilities

warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread will likely fail.')
A4 = (8.27, 11.69)


def find_z(ell, q, dell=None, dq=None):
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
    r_squared = (ell/q[0])**2
    t = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    lx = np.repeat((q[4]/q[5]**4, q[3]/q[5]**3, 1/q[5]**2, 0, 1), 5).reshape((5, 5)).T
    ly = np.repeat((q[7]/q[8]**4, q[6]/q[8]**3, 1/q[8]**2, 0, 1), 5).reshape((5, 5)).T
    n = scipy.special.binom(t.T, t)
    s = (-np.ones((5, 5)))**(t.T-t)
    
    p = np.sum((r_squared*ly*(q[1]-q[2])**(t.T-t) - lx*(q[1]+q[2])**(t.T-t))*s*n, 1)
    if np.any(np.isnan(p)):
        if dell is not None:
            return np.nan, np.nan
        else:
            return np.nan
    
    rts = np.roots(p)
    rts = rts[np.isreal(rts)]
    rng = find_zhuang_range(q)
    
    if (np.size(rts) == 0) | (not np.any((rts >= rng[0]) & (rts <= rng[1]))):
        if dell is not None:
            return np.nan, np.nan
        else:
            return np.nan

    z = np.real(rts[(rts >= rng[0]) & (rts <= rng[1])][0])
    
    if dell is not None:
        # attempt at error estimation
        dedz = scipy.misc.derivative(lambda x: zhuang_ell(z, q), z)
        s = 0
        for i in range(len(q)):
            s += (scipy.misc.derivative(lambda x: zhuang_ell(z, np.hstack((q[:i], x, q[(i + 1):]))), q[i]) * dq[i]) ** 2
        dz = np.sqrt((dell**2+s)/dedz**2)
        z = (z, dz)

    return z


def find_zhuang_range(q):
    """ Finds the usable range of z in the zhuang function with parameter q,
        i.e. the range between peak and valley around z=0
        wp@tl20190227
    """
    lx = np.repeat((q[4]/q[5]**4, q[3]/q[5]**3, 1/q[5]**2, 0, 1), 5).reshape((5, 5)).T
    ly = np.repeat((q[7]/q[8]**4, q[6]/q[8]**3, 1/q[8]**2, 0, 1), 5).reshape((5, 5)).T
    t = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    n1 = scipy.special.binom(t.T, t)
    n2 = scipy.special.binom(t.T-1, t)
    n2[np.isnan(n2)] = 0
    s = (-np.ones((5, 5)))**(t.T-t)
    za = np.sum(-n1*(q[1]-q[2])**(t.T-t)*ly*s, 1)
    zb = np.sum(t.T*n2*(q[1]+q[2])**(t.T-t-1)*lx*s, 1)
    zc = np.sum(t.T*n2*(q[1]-q[2])**(t.T-t-1)*ly*s, 1)
    zd = np.sum(n1*(q[1]+q[2])**(t.T-t)*lx*s, 1)
    
    q = np.full(8, np.nan)
    q[0] = za[0]*zb[1] + zc[1]*zd[0]
    q[1] = za[::-1][3:5]@zb[1:3] + zc[::-1][2:4]@zd[0:2]
    q[2] = za[::-1][2:5]@zb[1:4] + zc[::-1][1:4]@zd[0:3]
    q[3] = za[::-1][1:5]@zb[1:5] + zc[::-1][0:4]@zd[0:4]
    q[4] = za[::-1][0:4]@zb[1:5] + zc[::-1][0:4]@zd[1:5]
    q[5] = za[::-1][0:3]@zb[2:5] + zc[::-1][0:3]@zd[2:5]
    q[6] = za[::-1][0:2]@zb[3:5] + zc[::-1][0:2]@zd[3:5]
    q[7] = za[4]*zb[4] + zc[4]*zd[4]
    
    rts = np.roots(q)
    rts = rts[np.isreal(rts)]
    
    return np.real(np.array((np.max(rts[rts < 0], initial=-10), np.min(rts[rts > 0], initial=10))))


def zhuang_fun(z, p):
    """ p: [sigma0, A, B, c, d]
        wp@tl2019
    """
    x = (z - p[3]) / p[4]
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[0] * np.sqrt(1 + x ** 2 + p[1] * x ** 3 + p[2] * x ** 4)


def fit_zhuang_int(z, s):
    p = np.zeros((5,))
    p[0] = np.nanmin(s)
    p[3] = z[np.nanargmin(s)]
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.sqrt((z - p[3]) ** 2 / ((s / p[0]) ** 2 - 1))
    d[np.isinf(d)] = np.nan
    if np.any(np.isfinite(d)):
        p[4] = np.nanmean(d)
    else:
        p[4] = np.nan
    # Z = np.linspace(np.nanmin(z),np.nanmax(z))
    # plt.plot(z,s,'.',Z,zhuang_fun(Z,p),'--')
    # plt.plot(Z,zhuang_fun(Z,p),'--')
    return p


def fit_zhuang(z, s, ds, rec=True):
    z, s, ds = utilities.rm_nan(z, s, ds)
    if z.size == 0:
        return np.full(5, np.nan), np.full(5, np.nan), (np.nan, np.nan)
    p = fit_zhuang_int(z, s)

    def g(pf):
        return np.nansum((s - zhuang_fun(z, pf)) ** 2 / ds ** 2)

    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x
    if q.size >= z.size:
        return np.full(5, np.nan), np.full(5, np.nan), (np.nan, np.nan)
    else:
        dq = np.sqrt(r.fun / (z.size - q.size) * np.diag(r.hess_inv))
        chi_squared = r.fun / (z.size - p.size)
        r_squared = 1 - np.nansum((s - zhuang_fun(z, q)) ** 2) / np.sum((s - np.mean(s)) ** 2)
    if rec and np.isnan(zhuang_fun(0, q.T)):
        return fit_zhuang(z, s, 1, False)
    return q.T, dq.T, (chi_squared, r_squared)


# functions for calibration

def detect_points(im, sigma, mask=None, footprint=15, filter=True):
    """ Find interesting spots to which try to fit gaussians in a single frame
        im:      2D or 3D array
        sigma:   theoretical width of the psf (isotropic)
        mask:    logical 2D or 3D array masking the area in which points are to be found

        wp@tl201908
    """
    dim = np.ndim(im)
    im = im.astype(float)

    if sigma > 0:
        c = scipy.ndimage.gaussian_laplace(im, sigma)
        c = scipy.ndimage.gaussian_filter(-c, sigma)
    else:
        c = im

    pk = skimage.feature.peak_local_max(c, footprint=fn.disk(footprint, dim))
    if mask is not None:
        pk = utilities.mask_pk(pk, mask)

    f = pandas.DataFrame({'y_ini': pk[:, 0], 'x_ini': pk[:, 1]})
    if dim == 3:
        f['z_ini'] = pk[:, 2]

    p = []
    r_squared = []
    for i in range(len(f)):
        g = f.loc[i]
        if dim == 3:
            jm = fn.crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8], g['z_ini'] + [-8, 8])
        else:
            jm = fn.crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8])
        p.append(fn.fitgaussint(jm))
        r_squared.append(1 - np.sum((fn.gaussian(p[-1], 16, 16) - jm) ** 2) / np.sum((jm - np.mean(jm)) ** 2))
    p = np.array(p)
    if len(f):
        f['y'] = p[:, 0] + f['y_ini'] - 8
        f['x'] = p[:, 1] + f['x_ini'] - 8
        f['R2'] = r_squared
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

    if filter:
        f = f.dropna()
        if len(f) > 100:
            th = skimage.filters.threshold_otsu(f['i_ini'].to_numpy())
            f = f.nlargest(100, 'i_ini') if len(g := f.query('i_ini>{}'.format(th))) < 100 else g
    return f


def localize(f, im, theta=None, fast_mode=False, progress=True):
    sigma = [im.sigma[i] for i in range(int(f['C'].max()) + 1)]

    @parfor(f.iterrows(), desc='Fitting localisations', length=len(f), bar=progress)
    def fun(row):
        h = row[1]
        q, dq, r_squared = fn.fitgauss(im[int(h['C']), int(h['Z'])], theta, sigma[int(h['C'])], fast_mode, True,
                                       h[['x_ini', 'y_ini']].to_numpy())

        h['y_ini'] = h['y']
        h['x_ini'] = h['x']
        h['y'] = q[1]
        h['x'] = q[0]
        h['dy'] = dq[1]
        h['dx'] = dq[0]
        h['s'] = q[2] / (2 * np.sqrt(2 * np.log(2)))
        h['ds'] = dq[2] / (2 * np.sqrt(2 * np.log(2)))
        h['i'] = q[3]
        h['di'] = dq[3]
        h['o'] = q[4]
        h['do'] = dq[4]
        h['e'] = q[5]
        h['de'] = dq[5]
        h['theta'] = q[6]
        h['dtheta'] = dq[6]
        h['i_peak'] = h['i'] / (2 * np.pi * h['s'] ** 2)
        h['di_peak'] = h['i_peak'] * np.sqrt(4 * (h['ds'] / h['s']) ** 2 + (h['di'] / h['i']) ** 2)
        h['R2'] = r_squared
        return pandas.DataFrame(h).transpose()
    return pandas.concat(fun, sort=True)


class Progress:
    def __init__(self, total, callback):
        self.total = total
        self.callback = callback
        self.perc = 0
        self.half = 0

    def __call__(self, n):
        p = 50 * n / self.total + 50 * self.half
        if int(p) > self.perc:
            self.callback(p)
            self.perc = int(p)


def group(it, n):
    for i in range(int(np.ceil(len(it)/n))):
        yield it[n*i:n*i+n]


def calibrate_z(file, em_lambdas, channels, cyllens=None, progress=None, elim=None, r2lim=None, path=None, theta=None):
    if elim is None:
        elim = 0.4, 2.5
    if r2lim is None:
        r2lim = 0.6, 0.75
    if path is None:
        path = os.path.splitext(file)[0]
    with Imread(file, axes='czxy') as im, PdfPages(f'{path}_Cyllens_calib.pdf') as pdf:
        if em_lambdas is not None:
            if np.isscalar(em_lambdas):
                em_lambdas = [em_lambdas] * im.shape['c']
            im.sigma = [i / 2 / im.NA / im.pxsize_um / 1000 for i in em_lambdas]
        if cyllens is not None:
            im.cyllens = cyllens
        print(f'Using channels {channels}, sigma: {[im.sigma[channel] for channel in channels]} px')
        if np.any([im.sigma[channel] < 1 for channel in channels]):
            warnings.warn('sigma is smaller than 1 px, spot fitting, tracking and calibration will likely fail')

        detections = []
        for channel in channels:
            detected = detect_points(im[channel].max('z'), im.sigma[channel])
            detected['C'] = channel
            detections.append(detected)
        detections = pandas.concat(detections, ignore_index=True)

        detections['particle'] = range(len(detections))

        fig = plt.figure(figsize=A4)
        gs = GridSpec(3, 5, figure=fig)

        fig.add_subplot(gs[:2, :5])
        im = np.hstack([im[channel].max('z') for channel in channels])
        plt.imshow(im, vmax=np.percentile(im, 99))
        for i, channel in enumerate(channels):
            b = detections.query(f'C=={channel}')
            plt.plot(b['x'] + i * im.shape['y'], b['y'], 'or', markerfacecolor='none')
            for _, row in b.iterrows():
                plt.text(row['x'] + i * im.shape['y'], row['y'], f"{row['particle']:.0f}", color='w')

        d = pandas.DataFrame()

        zmax = np.round(15 / im.deltaz_um).astype('int')
        zmax = min(zmax, im.shape['z'])

        for T in range(im.shape['t']):
            for Z in range(zmax):
                b = detections.copy()
                b['Z'] = Z
                b['T'] = T
                b['z_um'] = Z * im.deltaz_um
                # b['frame'] = [im.czt2n(channel, Z, T) for channel in b['C']]
                d = pandas.concat((d, b))
        d = d.reset_index(drop=True)
        if callable(progress):
            pr = Progress(len(d), progress)
        else:
            pr = True
        if theta is None:
            detections = localize(d, im, theta=None, progress=pr)
            if callable(progress):
                pr.half = 1
            b = detections.copy().dropna(subset=['theta']).query('R2>0.3')

            fig.add_subplot(gs[2, 1:4])
            plt.hist((b['theta'] + np.pi / 4) % (np.pi / 2) - np.pi / 4, 100)
            plt.xlabel('theta')
            pdf.savefig(fig)

            theta, dtheta = utilities.circ_weighted_mean(b['theta'], b['dtheta'], np.pi / 2)
            print('θ = {} ± {}'.format(theta, dtheta))
        else:
            dtheta = 0

        detections = localize(d, im, theta=theta, progress=pr)
        detections['particle'] = d['particle']
        detections['s_um'] = detections['s'] * im.pxsize_um
        detections['ds_um'] = detections['ds'] * im.pxsize_um
        detections['dx_um'] = detections['dx'] * im.pxsize_um
        detections['dy_um'] = detections['dy'] * im.pxsize_um

        # individual Zhuang fits
        n_columns = 3
        n_rows = 5

        filtered = detections.query(f'R2>{r2lim[0]} & 0.1<s_um<0.6 & {elim[0]}<e<{elim[1]} &'
                                    ' dx_um<0.2 & dy_um<0.2 & de<0.2 & ds_um<0.2').copy()

        pr, px, dpx, py, dpy, X2x, X2y, R2x, R2y, z, sx, dsx, sy, dsy, Nx, Ny = ([] for _ in range(16))

        if not len(filtered):
            raise Exception('Filters have excluded all beads, retry with a better stack.')

        for particles in group(list(set(filtered['particle'])), n_rows * n_columns):
            fig = plt.figure(figsize=A4)
            gs = GridSpec(n_rows, n_columns, figure=fig)
            for i, p in enumerate(particles):
                b = filtered.query('particle=={}'.format(p)).copy()

                pr.append(int(p))
                z.append(np.array(b['z_um']))
                sx.append(np.array(b['s_um'] * np.sqrt(b['e'])))
                sy.append(np.array(b['s_um'] / np.sqrt(b['e'])))
                dsx.append(np.array(np.sqrt(b['ds_um'] ** 2 * b['e'] + b['s_um'] ** 2 * b['de'] ** 2 / 4 / b['e'])))
                dsy.append(np.array(dsx[-1] / b['e']))

                rs = 2
                Z = utilities.find_range(z[-1], rs)
                z[-1][z[-1] < (Z - rs / 2)] = np.nan
                z[-1][z[-1] > (Z + rs / 2)] = np.nan

                k = fit_zhuang(z[-1], sx[-1], dsx[-1])
                px.append(k[0])
                dpx.append(k[1])
                X2x.append(k[2][0])
                R2x.append(k[2][1])
                k = fit_zhuang(z[-1], sy[-1], dsy[-1])
                py.append(k[0])
                dpy.append(k[1])
                X2y.append(k[2][0])
                R2y.append(k[2][1])

                fig.add_subplot(gs[i // n_columns, i % n_columns])
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

        g = detections.query(f'R2>{r2lim[1]} & 0.1<s_um<0.6 & {elim[0]}<e<{elim[1]} & dx_um<0.2 & dy_um<0.2 & de<0.2'
                             ' & ds_um<0.2').copy()
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

        f = f.query('0.1<sigmax<0.5 & 0.1<sigmay<0.5 & abs(Ax)<10 & abs(Bx)<10 & abs(Ay)<10 & abs(By)<10 & X2x<20'
                    f' & X2y<20 & cx<10 & cy<10').dropna()

        Zx = np.array(())
        Zy = np.array(())
        Sx = np.array(())
        Sy = np.array(())
        dSx = np.array(())
        dSy = np.array(())

        fig = plt.figure(figsize=A4)
        gs = GridSpec(4, 1)

        fig.add_subplot(gs[0, 0])
        for i in range(f.shape[0]):
            idx = f.index[i]
            Zx = np.append(Zx, z[idx] - f['cx'][idx])
            Sx = np.append(Sx, sx[idx])
            dSx = np.append(dSx, dsx[idx])
            plt.plot(z[idx] - f['cx'][idx], sx[idx], '.')

        qx, dqx, X2x = fit_zhuang(Zx, Sx, dSx)
        if not np.isnan(qx).all():
            Z = np.linspace(np.nanmin(Zx), np.nanmax(Zx))
            plt.plot(Z, zhuang_fun(Z, qx), 'r-')
        plt.xlabel('z (um)')
        plt.ylabel('sigma_x (um)')
        plt.ylim(0, .5)

        fig.add_subplot(gs[1, 0])
        for i in range(f.shape[0]):
            idx = f.index[i]
            Zy = np.append(Zy, z[idx] - f['cy'][idx])
            Sy = np.append(Sy, sy[idx])
            dSy = np.append(dSy, dsy[idx])
            plt.plot(z[idx] - f['cy'][idx], sy[idx], '.')

        qy, dqy, X2y = fit_zhuang(Zy, Sy, dSy)
        if not np.isnan(qy).all():
            Z = np.linspace(np.nanmin(Zy), np.nanmax(Zy))
            plt.plot(Z, zhuang_fun(Z, qy), 'r-')
        plt.xlabel('z (um)')
        plt.ylabel('sigma_y (um)')

        plt.ylim(0, .5)

        dc, ddc = utilities.weighted_mean(f['dc'], f['ddc'])

        dc += qx[3] - qy[3]
        ddc = np.sqrt(ddc ** 2 + dqx[3] ** 2 + dqy[3] ** 2)

        qx[3] = dc / 2
        qy[3] = -dc / 2
        dqx[3] = ddc / 2
        dqy[3] = ddc / 2

        Z = np.linspace(-1, 1)
        E = np.array(())
        Ze = np.array(())

        fig.add_subplot(gs[2, 0])

        for i in range(f.shape[0]):
            idx = f.index[i]
            plt.plot(z[idx] - (f['cx'][idx] + f['cy'][idx]) / 2, sx[idx] / sy[idx], '.')
            E = np.append(E, sx[idx] / sy[idx])
            Ze = np.append(Ze, z[idx] - (f['cx'][idx] + f['cy'][idx]) / 2)
        q = np.hstack((np.sqrt(qx[0] / qy[0]), (qx[3] + qy[3]) / 2, (qx[3] - qy[3]) / 2,
                       qx[1], qx[2], qx[4], qy[1], qy[2], qy[4]))
        rl, = plt.plot(Z, zhuang_ell(Z, q), 'r-')
        E[E > elim[1]] = np.nan
        E[E < elim[0]] = np.nan
        Ze[Ze < -1] = np.nan
        Ze[Ze > 1] = np.nan
        E[utilities.outliers(E, False)] = np.nan
        q, dq, X2 = fit_zhuang_ell(Ze, E, q)
        gl, = plt.plot(Z, zhuang_ell(Z, q), 'g-')
        plt.xlabel('z (um)')
        plt.ylabel('ellipticity (um)')
        plt.legend((rl, gl), ('sx/sy', 'fit'))
        # plt.xlim(-1,1)
        plt.ylim(elim[0]/1.1, elim[1]*1.1)

        txt = f'θ = {theta} ± {dtheta}\ne0: {q[0]} ± {dq[0]}\nz0: {q[1]} ± {dq[1]}\nc : {q[2]} ± {dq[2]}\n'\
              f'Ax: {q[3]} ± {dq[3]}\nBx: {q[4]} ± {dq[4]}\ndx: {q[5]} ± {dq[5]}\nAy: {q[6]} ± {dq[6]}\n'\
              f'By: {q[7]} ± {dq[7]}\ndy: {q[8]} ± {dq[8]}'

        fig.add_subplot(gs[3, 0])
        plt.text(0.05, 0.5, txt, va='center')
        plt.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)

        obj_mag = im.objective.nominal_magnification
        tl_mag = im.tubelens.nominal_magnification
        s = f'[cyllens_name]{obj_mag:.0f}x{10 * tl_mag:.0f}:'
        s += '\n  q: [{}, {}, {}, {}, {}, {}, {}, {}, {}]'.format(*q)
        s += '\n  theta: {}'.format(theta)

        print('To put in CylLensGUI config file:')
        print(s)
        magnification_str = f'{obj_mag:.0f}x{10 * tl_mag:.0f}'

    return {'channels': channels, 'magnification_str': magnification_str, 'theta': theta, 'q': q,
            'detections': detections, 'filtered': filtered, 'Ze': Ze, 'E': E, 'z': z, 'sx': sx, 'sy': sy,
            'dsx': dsx, 'dsy': dsy}


def zhuang_ell(z, q):
    """ Ellipticity as function of z, with parameter q
        q: [sx/sy, z0, c, Ax, Bx, dx, Ay, By, dy]
        wp@tl20190227
    """
    x = (z - q[2] - q[1]) / q[5]
    y = (z + q[2] - q[1]) / q[8]
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[0] * np.sqrt((1 + x**2 + q[3] * x**3 + q[4] * x**4) / (1 + y**2 + q[6] * y**3 + q[7] * y**4))


def fit_zhuang_ell(z, e, p):
    """ Find calibration parameter q
        q:    [sx/sy,z0,c,Ax,Bx,dx,Ay,By,dy]
        z, e: data of ellipticity vs z
        p:    initial guess of q
        wp@tl20190227
    """
    z, e = utilities.rm_nan(z, e)
    if z.size == 0:
        return np.full(9, np.nan), np.full(9, np.nan), np.nan

    def g(pf):
        return np.nansum((e - zhuang_ell(z, pf)) ** 2)

    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x
    if q.size >= z.size:
        dq = np.full(q.size, np.nan)
    else:
        dq = np.sqrt(r.fun / (z.size - q.size) * np.diag(r.hess_inv))
    chi_squared = r.fun / (z.size - p.size)
    return q, dq, chi_squared


# Functions used for testing how good the calibration and/or bead z-stack is

def error_fun(x, p):
    if len(p) == 2:
        return p[0] * np.sin(x) * np.exp(-(x / p[1]) ** 2)
    else:
        return p[0] * np.sin(x - p[2]) * np.exp(-((x - p[2]) / p[1]) ** 2)


def error_fit(x, y, offset=False):
    def g(q):
        return np.nansum((y - error_fun(x, q)) ** 2)

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


def error_fun_pk(p):
    def g(z):
        return (1 / np.tan(z) - 2 * z / p ** 2) ** 2
    return scipy.optimize.minimize(g, 2 * p / 3, options={'disp': False, 'maxiter': 1e5}).x


def error_plot(x, d):
    xlim = 1

    x, d = utilities.rm_nan(x, d)
    if len(x) == 0:
        return

    Z = np.linspace(-xlim, xlim)
    p = error_fit(x, d + x)

    r = np.clip((2 * error_fun_pk(p[1]) / 3)[0], None, 1/2)
    plt.plot(x, d, '.b', Z, -Z, '--g')
    plt.plot(Z, error_fun(Z, p) - Z, '-r', Z, np.zeros(np.shape(Z)), '--k')
    plt.plot(r, error_fun(r, p) - r, 'yo', -r, error_fun(-r, p) + r, 'yo')
    plt.plot(np.full(np.shape(Z), r), Z, '--y', np.full(np.shape(Z), -r), Z, '--y')
    plt.xlim(-xlim, xlim)
    plt.ylim(-xlim, xlim)
    plt.xlabel('z (um)')
    plt.ylabel('error (um)')

    dr = d[np.abs(x) < r]

    plt.text(0.5, 0.95, f'range: {2000 * r:.0f} nm', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.95, 0.55, f'std(error): {1000 * np.std(dr):.0f} nm', transform=plt.gca().transAxes,
             horizontalalignment='right')

    print(f'Usable range: {2000 * r:.0f} nm\nStandard deviation of the error: {1000 * np.std(dr):.0f} nm')
    return 2000 * r, 1000 * np.std(dr)
