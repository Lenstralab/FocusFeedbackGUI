import numpy as np
import scipy.optimize
import scipy.special
import scipy.ndimage
from numba import jit
import os, re
from time import time
from inspect import stack
from numbers import Number

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
    """
    xy = np.array(np.unravel_index(np.nanargmax(im.T), np.shape(im)))
    r = 5
    jm = crop(im, (xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))
    p = fitgaussint(jm, theta)
    if (p[2] > 8) | (p[3] < 0.1):
        return np.full(7, np.nan), np.nan
    p[0:2] += (xy - r)
    s = 2 * np.ceil(p[2])
    jm = crop(im, (p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))
    S = np.shape(jm)
    p[0:2] -= (xy - s)
    xv, yv = meshgrid(np.arange(S[1]), np.arange(S[0]))
    g = lambda pf: np.sum((jm - gaussian7grid(np.append(pf, theta), xv, yv)) ** 2)

    r = scipy.optimize.minimize(g, p, options={'disp': False, 'maxiter': 1e5})
    r2 = 1 - (r.fun/np.sum((im-np.mean(im))**2))
    q = r.x

    q[2] = np.abs(q[2])
    q[0:2] += (xy - s)
    q = np.append(q, theta)
    q[5] = np.abs(q[5])

    return q, r2

def fg(im, Theta, f):
    if np.ndim(im) == 1:
        s = np.sqrt(len(im)).astype(int)
        im = np.reshape(im, (s,s))
    else:
        im = np.array(im)
    im -= scipy.ndimage.gaussian_filter(im, f * 1.1)
    im = scipy.ndimage.gaussian_filter(im, f / 1.1)
    q, r2 = fitgauss(im, Theta)
    return np.hstack((q, r2))

def fitgaussint(im, theta):
    """ Initial guess for gaussfit
        im: 2D array with image
        q:  [x,y,fwhm,area,offset,ellipticity,angle]
    """
    S = np.shape(im)
    q = np.full(6, 0).astype('float')

    x, y = np.meshgrid(range(S[0]), range(S[1]))
    q[4] = np.nanmin(im)
    jm = im-q[4]
    q[3] = np.nansum(jm)
    q[0] = np.nansum(x*jm)/q[3]
    q[1] = np.nansum(y*jm)/q[3]
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = cos*(x-q[0])-(y-q[1])*sin, cos*(y-q[1])+(x-q[0])*sin

    s2 = np.nansum((im-q[4])**2)
    sx = np.sqrt(np.nansum((x*jm)**2)/s2)
    sy = np.sqrt(np.nansum((y*jm)**2)/s2)

    q[2] = np.sqrt(sx*sy)*4*np.sqrt(np.log(2))
    q[5] = np.sqrt(sx/sy)
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

def cliprect(FS, X, Y, Sx, Sy):
    if Sx%2: Sx += 1
    if Sy%2: Sy += 1
    l = np.clip(X+FS[0]/2-Sx/2+1, 1, FS[0]-1)
    r = np.clip(X+FS[0]/2+Sx/2, 2, FS[0])
    b = np.clip(Y+FS[1]/2-Sy/2+1, 1, FS[1]-1)
    t = np.clip(Y+FS[1]/2+Sy/2, 2, FS[1])
    Sx, Sy = r-l+1, t-b+1
    if Sx%2 and Sy%2:
        return cliprect(FS, X, Y, Sx-2, Sy-2)
    elif Sx%2:
        return cliprect(FS, X, Y, Sx-2, Sy)
    elif Sy%2:
        return cliprect(FS, X, Y, Sx, Sy-2)
    else:
        return float((r+l-1)/2), float((t+b-1)/2), float(Sx), float(Sy)

def last_czi_file(folder='d:\data', t=60):
    """ finds last created czi file in folder created not more than t seconds ago
        wp@tl20191218
    """
    fname = ffind('.*\.czi$', folder, 5)
    tm = [os.path.getctime(f) for f in fname]
    t_newest = np.max(tm)
    if time()-t_newest>t:
        return ''
    else:
        return fname[np.argmax(tm)]

def ffind(expr, *args, **kwargs):
    """
    --------------------------------------------------------------------------
    % usage: fnames=ffind(expr,folder,rec,'once','directory')
    %
    % finds files that match regular expression 'expr' in 'folder' or
    % subdirectories
    %
    % inputs:
    %   folder: startfolder path, (optional, default: current working dirextory)
    %   expr:   string: regular expression to match
    %               example: to search for doc files do: '\.doc$'
    %           list or tuple: ffind will look for the folder in expr[0] starting
    %           from 'folder', 'rec' deep, then from that folder it will continue
    %           to navigate down to expr{end-1} and find the file (or folder) in
    %           expr[-1]
    %               example: ffind(('M_FvL','','^.*\.m'),'/home')
    %   rec:    recursive (optional, default: 3), also search in
    %           subdirectories rec deep (keep it low to avoid eternal loops)
    %   once:   optional flag, if enabled ffind will only output the first
    %           match it encounters as a string, use only if the existence of
    %           only one match is certain
    %   directory: optional flag: ffind only finds directories instead of files
    %
    % output:
    %   fnames: list containing all matches
    %
    % date: Aug 2014
    % author: wp
    % version: <01.00> (wp) <20140812.0000>
    %          <02.00>      <20180214.0000> Add once and directory flags, rec
    %                                       now signifies the folder-depth.
    %          <03.00>      <20190326.0000> Python implementation.
    %--------------------------------------------------------------------------
    """

    # argument parsing
    for i in args:
        if isinstance(i, Number):
            rec = i
        elif i == 'once':
            once = True
        elif i == 'directory':
            directory = True
        elif isinstance(i, str):
            folder = i

    for key, value in kwargs.items():
        if key is 'once':
            once = value
        elif key is 'directory':
            directory = value

    if not 'rec' in vars().keys():
        rec = 3
    if not 'once' in vars().keys():
        once = False
    if not 'directory' in vars().keys():
        directory = False
    if not 'folder' in vars().keys():
        folder = os.getcwd()

    if not folder[-1] == os.sep:
        folder = folder + os.sep

    # print('rec: {}, folder: {}'.format(rec, folder))

    if isinstance(expr, tuple):
        expr = list(expr)

    # search for the path in expr if needed
    if isinstance(expr, (list, tuple)):
        if len(expr) > 1:
            folder = ffind(expr[0], folder, rec, 'directory')
            for e in expr[1:-1]:
                folder_tmp = []
                for f in folder:
                    folder_tmp.extend(ffind(e, f, 0, 'directory'))
                folder = folder_tmp
            fnames = []
            for f in folder:
                fnames.extend(ffind(expr[-1], f, 0, once=once, directory=directory))
                if len(fnames) and once:
                    fnames = fnames[0]
                    break
            if not len(fnames) and once:
                fnames = ''
            return fnames
        expr = expr[0]

    # an empty expression should match everything
    if not isinstance(expr, re.Pattern):
        if not len(expr) > 0:
            expr = '.*'
        expr = re.compile(expr, re.IGNORECASE)

    lst = os.listdir(folder)

    fnames = []
    dirnames = []
    for l in lst:
        if re.search('^\.', l) != None:  # don't search in/for hidden things
            continue
        if (os.path.isdir(folder + l) == directory) & (re.search(expr, l) != None):
            fnames.append(folder + l)
            if once:
                break
        if rec and os.path.isdir(
                folder + l):  # list all folders, but don't go in them yet, faster when the target is in the current folder and once=True
            dirnames.append(folder + l)
    if not once or not len(fnames):  # recursively go through all subfolders
        for d in dirnames:
            fnames.extend(ffind(expr, d, rec - 1, once=once, directory=directory))
            if once and len(fnames):
                break

    if once and stack()[1][3] != 'ffind':
        if len(fnames):
            fnames = fnames[0]
        else:
            fnames = ''
    else:
        fnames.sort()
    return fnames
