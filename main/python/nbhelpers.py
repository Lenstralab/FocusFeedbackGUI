from numba import jit
import numpy as np

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
def gaussian5(p,X,Y):
    """ p: [x,y,fwhm,area,offset]
        X,Y: size of image
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac
    dy = efac
    xv, yv = meshgrid(np.arange(Y)-p[0],np.arange(X)-p[1])
    x = 2*dx*xv
    y = 2*dy*yv
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]

@jit(nopython=True, nogil=True)
def gaussian6(p,X,Y):
    """ p: [x,y,fwhm,area,offset,ellipticity]
        X,Y: size of image
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac/p[5]
    dy = efac*p[5]
    xv, yv = meshgrid(np.arange(Y)-p[0],np.arange(X)-p[1])
    x = 2*dx*xv
    y = 2*dy*yv
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]

@jit(nopython=True, nogil=True)
def gaussian7(p,X,Y):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
        X,Y: size of image
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac/p[5]
    dy = efac*p[5]
    xv, yv = meshgrid(np.arange(Y)-p[0],np.arange(X)-p[1])
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2*dx*(cos*xv-yv*sin)
    y = 2*dy*(cos*yv+xv*sin)
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]

@jit(nopython=True, nogil=True)
def gaussian9(p,X,Y):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
        X,Y: size of image
        reimplemented for numba, small deviations from true result
            possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2))/p[2]
    dx = efac/p[5]
    dy = efac*p[5]
    xv, yv = meshgrid(np.arange(Y)-p[0],np.arange(X)-p[1])
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2*dx*(cos*xv-yv*sin)
    y = 2*dy*(cos*yv+xv*sin)
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]+p[7]*xv+p[8]*yv

@jit(nopython=True, nogil=True)
def gaussian7grid(p,xv,yv):
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

@jit(nopython=True, nogil=True)
def gaussian9grid(p,xv,yv):
    """ p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
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
    return p[3]/4*(erf2(x+dx)-erf2(x-dx))*(erf2(y+dy)-erf2(y-dy))+p[4]+p[7]*xv+p[8]*yv-p[7]*p[0]-p[8]*p[1]
