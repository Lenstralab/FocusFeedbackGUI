import numpy as np
import scipy.optimize
import scipy.special
import scipy.ndimage
from numba import jit

@jit(nopython=True, nogil=True)
def meshgrid(x, y):
    s = (len(y), len(x))
    xv = np.zeros(s)
    yv = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            xv[i,j] = x[j]
            yv[i,j] = y[i]
    return xv, yv

@jit(nopython=True, nogil=True)
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

@jit(nopython=True, nogil=True)
def erf2(x):
    s = x.shape
    y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            y[i,j] = erf(x[i,j])
    return y

@jit(nopython=True, nogil=True)
def gaussian7grid(p, xv, yv):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
        xv, yv = meshgrid(np.arange(Y),np.arange(X))
            calculation of meshgrid is done outside, so it doesn't
            have to be done each time this function is run
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac/p[5]
    dy = efac*p[5]
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2*dx*(cos*(xv-p[0])-(yv-p[1])*sin)
    y = 2*dy*(cos*(yv-p[1])+(xv-p[0])*sin)
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]

def fitgauss(im, theta=0):
    """ Fit gaussian function to image
        im:    2D array with image
        theta: Fixed theta to use
        q:  [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
        dq: errors (std) on q
    """
    xy = np.array(np.unravel_index(np.nanargmax(im.T), np.shape(im)))
    r = 5
    jm = crop(im, (xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))
    p = fitgaussint(jm)
    if (p[2] > 8) | (p[3] < 0.1):
        q = np.ones(7)
        return q
    p[0:2] += (xy - r)
    s = 2 * np.ceil(p[2])
    jm = crop(im, (p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))
    S = np.shape(jm)
    p[0:2] -= (xy - s)
    xv, yv = meshgrid(np.arange(S[1]), np.arange(S[0]))
    p = np.append(p, 1)
    g = lambda pf: np.sum((jm - gaussian7grid(np.append(pf, theta), xv, yv)) ** 2)

    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    q = r.x

    q[2] = np.abs(q[2])
    q[0:2] += (xy - s)
    q = np.append(q, theta)
    q[5] = np.abs(q[5])

    return q

def fg(im, Theta, f):
    if np.ndim(im) == 1:
        s = np.sqrt(len(im)).astype(int)
        im = np.reshape(im, (s,s))
    else:
        im = np.array(im)
    im -= scipy.ndimage.gaussian_filter(im, f * 1.1)
    im = scipy.ndimage.gaussian_filter(im, f / 1.1)
    return fitgauss(im, Theta)

def fitgaussint(im):
    """ Initial guess for gaussfit
        im: 2D array with image
        q:  [x,y,fwhm,area,offset]
    """
    S = np.shape(im)
    q = np.full(5, np.nan)
    q[:2] = np.unravel_index(np.argmax(im.T),(S[0],S[1]))
    q[4] = np.min(im)
    q[3] = np.sum(im-q[4])
    f = np.mean((np.max(im),q[4]))
    jm = np.zeros(S)
    jm[im<f] = 0
    jm[im>f] = 1
    q[2] = np.sqrt(np.sum(jm))
    return q

def crop(im, x, y, m=np.nan):
    """ crops image im, limits defined by min(x)..max(y), when these limits are
        outside im the resulting pixels will be filled with mean(im)
        wp@tl20181129
    """
    x = np.array(x).astype(int)
    y = np.array(y).astype(int)
    S = np.array(np.shape(im))
    R = np.array([[min(y),max(y)],[min(x),max(x)]]).astype(int)
    r = R.copy()
    r[R[:,0]<1,0] = 1
    r[R[:,1]>S,1] = S[R[:,1]>S]
    jm = im[r[0,0]:r[0,1],r[1,0]:r[1,1]]
    jm =   np.concatenate((np.full((r[0,0]-R[0,0],np.shape(jm)[1]),m),jm,np.full((R[0,1]-r[0,1],np.shape(jm)[1]),m)),0)
    return np.concatenate((np.full((np.shape(jm)[0],r[1,0]-R[1,0]),m),jm,np.full((np.shape(jm)[0],R[1,1]-r[1,1]),m)),1)