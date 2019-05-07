import numpy as np
import scipy.optimize
import scipy.special
import scipy.ndimage

def gaussian(p,X,Y):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
        default ellipticity & angle: 1 resp. 0
        X,Y: size of image
    """
    efac = np.sqrt(np.log(2))/p[2]
    if np.size(p)<6:
        dx = efac
        dy = efac
    else:
        dx = efac/p[5]
        dy = efac*p[5]
    xv, yv = np.meshgrid(np.arange(Y)-p[0],np.arange(X)-p[1])
    if np.size(p)<7:
        x = 2*dx*xv
        y = 2*dy*yv
    else:
        cos, sin = np.cos(p[6]), np.sin(p[6])
        x = 2*dx*(cos*xv-yv*sin)
        y = 2*dy*(cos*yv+xv*sin)
    erf = scipy.special.erf
    return p[3]/4*(erf(x+dx)-erf(x-dx))*(erf(y+dy)-erf(y-dy))+p[4]

def fitgauss(im,xy=None,theta=None):
    """ Fit gaussian function to image
        im:    2D array with image
        xy:    Initial guess for x, y, optional, default: pos of max in im
        theta: angle of ellipse gaussian towards x-axis
            any number: fit with theta fixed
            True:       fit theta
            False/None: no ellipticity
        q:  [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
        dq: errors (std) on q
    """
    if xy is None:
        xy = np.unravel_index(np.nanargmax(im.T),np.shape(im))
    xy = np.array(xy)
    r = 5
    jm = crop(im,(xy[0]-r,xy[0]+r+1),(xy[1]-r,xy[1]+r+1))
    p = fitgaussint(jm)
    if (p[2]>8) | (p[3]<0.1):
        q = np.full(7, np.nan)
        dq = np.full(7, np.nan)
        return q, dq
    p[0:2] += (xy-r)
    s = 2*np.ceil(p[2])
    jm = crop(im,(p[0]-s,p[0]+s+1),(p[1]-s,p[1]+s+1))
    S = np.shape(jm)
    p[0:2] -= (xy-s)
    if theta is True:
        p = np.append(p,(1,0))
        g = lambda pf: np.sum((jm-gaussian(pf,S[0],S[1]))**2)
    elif (theta is False) or (theta is None):
        g = lambda pf: np.sum((jm-gaussian(np.append(pf,(1,0)),S[0],S[1]))**2)
    else:
        p = np.append(p,1)
        g = lambda pf: np.sum((jm-gaussian(np.append(pf,theta),S[0],S[1]))**2)
    r  = scipy.optimize.minimize(g,p,options={'disp': False, 'maxiter': 1e5})
    q  = r.x
    q[2] = np.abs(q[2])
    q[0:2] += (xy-s)
    dq = np.sqrt(r.fun/(np.size(jm)-np.size(q))*np.diag(r.hess_inv))
    if len(q) == 5:
        q = np.append(q, (1,0))
        dq = np.append(dq, (0,0))
    elif len(q) == 6:
        q = np.append(q, theta)
        dq = np.append(dq, 0)
    q[5] = np.abs(q[5])
    return q, dq

def fg(im, Theta, f):
    if np.ndim(im) == 1:
        s = np.sqrt(len(im)).astype(int)
        im = np.reshape(im, (s,s))
    else:
        im = np.array(im)
    im -= scipy.ndimage.gaussian_filter(im, f * 1.1)
    im = scipy.ndimage.gaussian_filter(im, f / 1.1)
    q, _ = fitgauss(im, None, Theta)
    q[np.isnan(q)] = 1
    return q

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

def crop(im,x,y,m=np.nan):
    """ crops image im, limits defined by min(x)..max(y), when these limits are
        outside im the resulting pixels will be filled with mean(im)
        wp@tl20181129
    """
    x=np.array(x).astype(int)
    y=np.array(y).astype(int)
    S=np.array(np.shape(im))
    R=np.array([[min(y),max(y)],[min(x),max(x)]]).astype(int)
    r=R.copy()
    r[R[:,0]<1,0]=1
    r[R[:,1]>S,1]=S[R[:,1]>S]
    jm=np.zeros(S)
    jm=im[r[0,0]:r[0,1],r[1,0]:r[1,1]]
    jm =   np.concatenate((np.full((r[0,0]-R[0,0],np.shape(jm)[1]),m),jm,np.full((R[0,1]-r[0,1],np.shape(jm)[1]),m)),0)
    return np.concatenate((np.full((np.shape(jm)[0],r[1,0]-R[1,0]),m),jm,np.full((np.shape(jm)[0],R[1,1]-r[1,1]),m)),1)