import os
import re
import numpy as np
import scipy
import pandas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import skimage.filters
from parfor import parfor
import warnings
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread will likely fail.')


if __package__ == '':
    import functions as fn
    import utilities
    from imread import imread
else:
    from . import functions as fn
    from . import utilities
    from .imread import imread


A4 = (8.27, 11.69)


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
    R2 = (ell/q[0])**2
    T = np.flip(np.repeat(range(5),5).reshape((5,5)))
    Lx = np.repeat((q[4]/q[5]**4,q[3]/q[5]**3,1/q[5]**2,0,1),5).reshape((5,5)).T
    Ly = np.repeat((q[7]/q[8]**4,q[6]/q[8]**3,1/q[8]**2,0,1),5).reshape((5,5)).T
    N = scipy.special.binom(T.T,T)
    S = (-np.ones((5,5)))**(T.T-T)
    
    P = np.sum( (R2*Ly*(q[1]-q[2])**(T.T-T) - Lx*(q[1]+q[2])**(T.T-T))*S*N ,1)
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
            return np.nan, np.nan
        else:
            return np.nan

    z = np.real(rts[(rts >= rng[0]) & (rts <= rng[1])][0])
    
    if not dell is None:
        #attempt at error estimation
        dedz = scipy.misc.derivative(lambda x: zhuangell(z,q),z)
        s = 0
        for i in range(len(q)):
            s += (scipy.misc.derivative(lambda x: zhuangell(z,np.hstack((q[:i],x,q[(i+1):]))),q[i])*dq[i])**2
        dz = np.sqrt((dell**2+s)/dedz**2)
        z = (z,dz)

    return z
    
def zhuangell(z,q):
    """ Ellipticity as function of z, with parameter q
        q: [sx/sy, z0, c, Ax, Bx, dx, Ay, By, dy]
        wp@tl20190227
    """
    X = (z-q[2]-q[1])/q[5]
    Y = (z+q[2]-q[1])/q[8]
    return q[0]*np.sqrt((1+X**2+q[3]*X**3+q[4]*X**4)/(1+Y**2+q[6]*Y**3+q[7]*Y**4))
    
def findzhuangrange(q):
    """ Finds the usable range of z in the zhuang function with parameter q,
        ie. the range between peak and valey around z=0
        wp@tl20190227
    """
    Lx = np.repeat((q[4]/q[5]**4,q[3]/q[5]**3,1/q[5]**2,0,1),5).reshape((5,5)).T
    Ly = np.repeat((q[7]/q[8]**4,q[6]/q[8]**3,1/q[8]**2,0,1),5).reshape((5,5)).T
    T = np.flip(np.repeat(range(5),5).reshape((5,5)))
    N1 = scipy.special.binom(T.T,T)
    N2 = scipy.special.binom(T.T-1,T)
    N2[np.isnan(N2)] = 0
    S = (-np.ones((5,5)))**(T.T-T)
    za = np.sum(   -N1*(q[1]-q[2])**(T.T-T  )*Ly*S,1)
    zb = np.sum(T.T*N2*(q[1]+q[2])**(T.T-T-1)*Lx*S,1)
    zc = np.sum(T.T*N2*(q[1]-q[2])**(T.T-T-1)*Ly*S,1)
    zd = np.sum(    N1*(q[1]+q[2])**(T.T-T  )*Lx*S,1)
    
    q = np.full(8,np.nan)
    q[0] = za[0]        *zb[1]   + zc[1]        *zd[0]
    q[1] = za[::-1][3:5]@zb[1:3] + zc[::-1][2:4]@zd[0:2]
    q[2] = za[::-1][2:5]@zb[1:4] + zc[::-1][1:4]@zd[0:3]
    q[3] = za[::-1][1:5]@zb[1:5] + zc[::-1][0:4]@zd[0:4]
    q[4] = za[::-1][0:4]@zb[1:5] + zc[::-1][0:4]@zd[1:5]
    q[5] = za[::-1][0:3]@zb[2:5] + zc[::-1][0:3]@zd[2:5]
    q[6] = za[::-1][0:2]@zb[3:5] + zc[::-1][0:2]@zd[3:5]
    q[7] = za[4]        *zb[4]   + zc[4]        *zd[4]
    
    rts = np.roots(q)
    rts = rts[np.isreal(rts)]
    
    return np.real(np.array((np.max(rts[rts<0]),np.min(rts[rts>0]))))


def zhuang_fun(z, p):
    """ p: [sigma0, A, B, c, d]
        wp@tl2019
    """
    X = (z - p[3]) / p[4]
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[0] * np.sqrt(1 + X ** 2 + p[1] * X ** 3 + p[2] * X ** 4)


def fitzhuangint(z, s):
    p = np.zeros((5, 1))
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


# functions for calibration

def detect_points(im, sigma, mask=None, footprint=15):
    """ Find interesting spots to which try to fit gaussians in a single frame
        im:      2D or 3D array
        sigma:   theoretical width of the psf (isotropic)
        mask:    logical 2D or 3D array masking the area in which points are to be found

        wp@tl201908
    """
    dim = np.ndim(im)

    if sigma > 0:
        c = scipy.ndimage.gaussian_laplace(im, sigma)
        c = scipy.ndimage.gaussian_filter(-c, sigma)
    else:
        c = im

    pk = skimage.feature.peak_local_max(c, footprint=fn.disk(footprint, dim))
    if not mask is None:
        pk = utilities.maskpk(pk, mask)

    f = pandas.DataFrame({'y_ini': pk[:, 0], 'x_ini': pk[:, 1]})
    if dim == 3:
        f['z_ini'] = pk[:, 2]

    p = []
    R2 = []
    for i in range(len(f)):
        g = f.loc[i]
        if dim == 3:
            jm = fn.crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8], g['z_ini'] + [-8, 8])
        else:
            jm = fn.crop(im, g['x_ini'] + [-8, 8], g['y_ini'] + [-8, 8])
        p.append(fn.fitgaussint(jm))
        R2.append(1 - np.sum((fn.gaussian(p[-1], 16, 16) - jm) ** 2) / np.sum((jm - np.mean(jm)) ** 2))
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


def tpsuperresseq(f, im, theta=None, fastmode=False, progress=True):
    frames = f['frame'].astype('int').tolist()
    c = ((i, im(j)) for i, j in zip(f.iterrows(), frames))
    sigma = [im.sigma[i] for i in range(int(f['C'].max()) + 1)]
    @parfor(c, (theta, sigma, fastmode), desc='Fitting localisations', length=len(frames), bar=progress)
    def Q(c, theta, sigma, fastmode):
        f = c[0][1].copy()
        jm = c[1].copy()

        q, dq, R2 = fn.fitgauss(jm, theta, sigma[int(f['C'])], fastmode, True, f[['x_ini', 'y_ini']].to_numpy())

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
        f['i_peak'] = f['i'] / (2 * np.pi * f['s'] ** 2)
        f['di_peak'] = f['i_peak'] * np.sqrt(4 * (f['ds'] / f['s']) ** 2 + (f['di'] / f['i']) ** 2)
        f['R2'] = R2
        return pandas.DataFrame(f).transpose()
    return pandas.concat(Q, sort=True)


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


def calibz(file, em_lambda=None, masterchannel=None, cyllens=None, progress=None):
    with imread(file) as im, PdfPages(os.path.splitext(file)[0] + '_Cyllens_calib.pdf') as pdf:
        if not em_lambda is None:
            if np.isscalar(em_lambda):
                em_lambda = [em_lambda] * im.shape[2]
            im.sigma = [i / 2 / im.NA / im.pxsize / 1000 for i in em_lambda]
        if not masterchannel is None:
            im.masterch = masterchannel
        if cyllens is not None:
            im.cyllens = cyllens
        print(f'Using channel {im.masterch}, sigma: {im.sigma[im.masterch]}')

        a = detect_points(im.maxz(im.masterch), im.sigma[im.masterch])

        a['particle'] = range(len(a))

        fig = plt.figure(figsize=A4)
        gs = GridSpec(3, 5, figure=fig)

        fig.add_subplot(gs[:2, :5])
        plt.imshow(im.maxz(im.masterch))
        plt.plot(a['x'], a['y'], 'or', markerfacecolor='none')
        for i in a.index:
            plt.text(a.loc[i, 'x'], a.loc[i, 'y'], a.loc[i, 'particle'], color='w')

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
        if callable(progress):
            pr = Progress(len(d), progress)
        else:
            pr = True
        a = tpsuperresseq(d, im, theta=None, progress=pr)
        if callable(progress):
            pr.half = 1
        b = a.copy().dropna(subset=['theta']).query('R2>0.3')

        fig.add_subplot(gs[2, 1:4])
        plt.hist((b['theta'] + np.pi / 4) % (np.pi / 2) - np.pi / 4, 100)
        plt.xlabel('theta')
        pdf.savefig(fig)

        theta, dtheta = utilities.circ_weightedmean(b['theta'], b['dtheta'], np.pi / 2)
        #theta *= -1
        print('θ = {} ± {}'.format(theta, dtheta))

        a = tpsuperresseq(d, im, theta=theta, progress=pr)
        a['particle'] = d['particle']
        a['s_um'] = a['s'] * im.pxsize
        a['ds_um'] = a['ds'] * im.pxsize
        a['dx_um'] = a['dx'] * im.pxsize
        a['dy_um'] = a['dy'] * im.pxsize

        # individual Zhuang fits
        nColumns = 3
        nRows = 5

        a0 = a.query('R2>0.6 & 0.1<s_um<0.6 & 2/3<e<3/2 & dx_um<0.05 & dy_um<0.05 & de<0.2 & ds_um<0.2').copy()

        # particles = set(a0['particle'])

        pr, px, dpx, py, dpy, X2x, X2y, R2x, R2y, z, sx, dsx, sy, dsy, Nx, Ny = ([] for _ in range(16))

        for particles in group(list(set(a0['particle'])), nRows * nColumns):
            lp = len(particles)
            fig = plt.figure(figsize=A4)
            gs = GridSpec(nRows, nColumns, figure=fig)
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

                fig.add_subplot(gs[i // nColumns, i % nColumns])
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
        dc = [i/(im.immersionN**2/1.33**2) for i in (0.1, 0.6)]
        f = f.query('&'.join(('0.1<sigmax<0.5', '0.1<sigmay<0.5', 'abs(Ax)<10', 'abs(Bx)<10', 'abs(Ay)<10',
                              'abs(By)<10', 'X2x<20', 'X2y<20', '{}<dc<{}'.format(*dc), 'cx<10', 'cy<10')))

        Zx = np.array(())
        Zy = np.array(())
        Sx = np.array(())
        Sy = np.array(())
        dSx = np.array(())
        dSy = np.array(())
        E = np.array(())
        Ze = np.array(())

        fig = plt.figure(figsize=A4)
        gs = GridSpec(4, 1)

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

        fig.add_subplot(gs[1, 0])
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

        fig.add_subplot(gs[2, 0])

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

        txt = 'θ = {} ± {}\ne0: {} ± {}\nz0: {} ± {}\nc : {} ± {}\nAx: {} ± {}\nBx: {} ± {}\ndx: {} ± {}\nAy: {} ± {}\nBy: {} ± {}\ndy: {} ± {}'.format(
            theta, dtheta, q[0], dq[0], q[1], dq[1], q[2], dq[2], q[3], dq[3], q[4], dq[4], q[5], dq[5], q[6], dq[6], q[7],
            dq[7], q[8], dq[8])

        fig.add_subplot(gs[3, 0])
        plt.text(0.05, 0.5, txt, va='center');
        plt.axis('off');
        plt.tight_layout()
        pdf.savefig(fig)

        r = re.search('(?<=\s)\d+x', im.objective)
        if r:
            m = r.group(0)
        else:
            m = ''

        s = '{}{}{:.0f}:'.format(im.cyllens[C], m, im.optovar[0] * 10)
        s += '\n  q: [{}, {}, {}, {}, {}, {}, {}, {}, {}]'.format(*q)
        s += '\n  theta: {}'.format(theta)

        print('To put in CylLensGUI config file:')
        print(s)
        MagStr = f'{im.magnification}x{10*im.optovar[0]:.0f}'

        fig = plt.figure(figsize=A4)
        errplot(Ze, np.array([findz(e, q) for e in E]) - Ze)
        pdf.savefig(fig)

    return C, MagStr, theta, q


def zhuangell(z, q):
    """ Ellipticity as function of z, with parameter q
        q: [sx/sy,z0,c,Ax,Bx,dx,Ay,By,dy]
        wp@tl20190227
    """
    X = (z - q[2] - q[1]) / q[5]
    Y = (z + q[2] - q[1]) / q[8]
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[0] * np.sqrt((1 + X**2 + q[3] * X**3 + q[4] * X**4) / (1 + Y**2 + q[6] * Y**3 + q[7] * Y**4))


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
    g = lambda z: (1 / np.tan(z) - 2 * z / p ** 2) ** 2
    return scipy.optimize.minimize(g, 2 * p / 3, options={'disp': False, 'maxiter': 1e5}).x


def errplot(x, d):
    l = 1

    x, d = utilities.rmnan(x, d)
    if len(x) == 0:
        return

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

    dr = d[np.abs(x) < r]

    plt.text(0.5, 0.95, f'range: {2000 * r:.0f} nm', transform=plt.gca().transAxes, horizontalalignment='center')
    plt.text(0.95, 0.55, f'std(error): {2000 * np.std(dr):.0f} nm', transform=plt.gca().transAxes,
             horizontalalignment='right')

    print("Usable range: {} nm\nStandard deviation of the error: {} nm".format(int(2000 * r), int(2000 * np.std(dr))))
