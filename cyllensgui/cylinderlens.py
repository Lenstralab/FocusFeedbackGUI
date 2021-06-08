import numpy as np
from scipy import special, misc

def findz(ell,q,dell=None,dq=None):
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
    N = special.binom(T.T,T)
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
            return (np.nan, np.nan)
        else:
            return np.nan

    z = np.real(rts[(rts >= rng[0]) & (rts <= rng[1])][0])
    
    if not dell is None:
        #attempt at error estimation
        dedz = misc.derivative(lambda x: zhuangell(z,q),z)
        s = 0
        for i in range(len(q)):
            s += (misc.derivative(lambda x: zhuangell(z,np.hstack((q[:i],x,q[(i+1):]))),q[i])*dq[i])**2
        dz = np.sqrt((dell**2+s)/dedz**2)
        z = (z,dz)

    return z
    
def zhuangell(z,q):
    """ Ellipticity as function of z, with parameter q
        q: [sx/sy,z0,c,Ax,Bx,dx,Ay,By,dy]
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
    N1 = special.binom(T.T,T)
    N2 = special.binom(T.T-1,T)
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