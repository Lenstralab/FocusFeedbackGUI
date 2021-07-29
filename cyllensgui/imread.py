# -*- coding: utf-8 -*-

import untangle
import os
import re
import pandas
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import czifile
import yaml
from itertools import product
from collections import OrderedDict

if __package__ == '':
    from transforms import Transform
    from tiffwrite import IJTiffWriter
else:
    from .transforms import Transform
    from .tiffwrite import IJTiffWriter


def getConfig(file):
    """ Open a yml parameter file
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(file, 'r') as f:
        return yaml.load(f, loader)


class deque_dict(OrderedDict):
    def __init__(self, maxlen=None, *args, **kwargs):
        self.maxlen = maxlen
        super(deque_dict, self).__init__(*args, **kwargs)

    def __truncate__(self):
        while len(self) > self.maxlen:
            self.popitem(False)

    def __setitem__(self, *args, **kwargs):
        super(deque_dict, self).__setitem__(*args, **kwargs)
        self.__truncate__()

    def update(self, *args, **kwargs):
        super(deque_dict, self).update(*args, **kwargs)
        self.__truncate__()


def tolist(item):
    if isinstance(item, xmldata):
        return [item]
    elif hasattr(item, 'items'):
        return item
    elif isinstance(item, str):
        return [item]
    try:
        iter(item)
        return list(item)
    except TypeError:
        return list((item,))


class xmldata(dict):
    def __init__(self, elem):
        super(xmldata, self).__init__()
        if elem:
            if isinstance(elem, dict):
                self.update(elem)
            else:
                self.update(xmldata._todict(elem)[1])

    def re_search(self, reg, default=None, *args, **kwargs):
        return tolist(xmldata._output(xmldata._search(self, reg, True, default, *args, **kwargs)[1]))

    def search(self, key, default=None):
        return tolist(xmldata._output(xmldata._search(self, key, False, default)[1]))

    def re_search_all(self, reg, *args, **kwargs):
        K, V = xmldata._search_all(self, reg, True, *args, **kwargs)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    def search_all(self, key):
        K, V = xmldata._search_all(self, key, False)
        return {k: xmldata._output(v) for k, v in zip(K, V)}

    @staticmethod
    def _search(d, key, regex=False, default=None, *args, **kwargs):
        if isinstance(key, (list, tuple)):
            if len(key)==1:
                key = key[0]
            else:
                for v in xmldata._search_all(d, key[0], regex, *args, **kwargs)[1]:
                    found, value = xmldata._search(v, key[1:], regex, default, *args, **kwargs)
                    if found:
                        return True, value
                return False, default

        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        return True, v
                    elif isinstance(v, dict):
                        found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                        if found:
                            return True, value
                    elif isinstance(v, (list, tuple)):
                        for w in v:
                            found, value = xmldata._search(w, key, regex, default, *args, **kwargs)
                            if found:
                                return True, value
                else:
                    found, value = xmldata._search(v, key, regex, default, *args, **kwargs)
                    if found:
                        return True, value
        return False, default

    @staticmethod
    def _search_all(d, key, regex=False, *args, **kwargs):
        K = []
        V = []
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(k, str):
                    if (not regex and k == key) or (regex and re.findall(key, k, *args, **kwargs)):
                        K.append(k)
                        V.append(v)
                    elif isinstance(v, dict):
                        q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                        K.extend([str(k) + '|' + i for i in q])
                        V.extend(w)
                    elif isinstance(v, (list, tuple)):
                        for j, val in enumerate(v):
                            q, w = xmldata._search_all(val, key, regex, *args, **kwargs)
                            K.extend([str(k) + '|' + str(j) + '|' + i for i in q])
                            V.extend(w)
                else:
                    q, w = xmldata._search_all(v, key, regex, *args, **kwargs)
                    K.extend([str(k) + '|' + i for i in q])
                    V.extend(w)
        return K, V

    @staticmethod
    def _enumdict(d):
        d2 = {}
        for k, v in d.items():
            idx = [int(i) for i in re.findall('(?<=:)\d+$', k)]
            if idx:
                key = re.findall('^.*(?=:\d+$)', k)[0]
                if not key in d2:
                    d2[key] = {}
                d2[key][idx[0]] = d['{}:{}'.format(key, idx[0])]
            else:
                d2[k] = v
        rec = False
        for k, v in d2.items():
            if [int(i) for i in re.findall('(?<=:)\d+$', k)]:
                rec = True
                break
        if rec:
            return xmldata._enumdict(d2)
        else:
            return d2

    @staticmethod
    def _unique_children(l):
        if l:
            keys, values = zip(*l)
            d = {}
            for k in set(keys):
                value = [v for m, v in zip(keys, values) if k == m]
                if len(value) == 1:
                    d[k] = value[0]
                else:
                    d[k] = value
            return d
        else:
            return {}

    @staticmethod
    def _todict(elem):
        d = {}
        if hasattr(elem, 'Key') and hasattr(elem, 'Value'):
            name = elem.Key.cdata
            d = elem.Value.cdata
            return name, d

        if hasattr(elem, '_attributes') and not elem._attributes is None and 'ID' in elem._attributes:
            name = elem._attributes['ID']
            elem._attributes.pop('ID')
        elif hasattr(elem, '_name'):
            name = elem._name
        else:
            name = 'none'

        if name == 'Value':
            if hasattr(elem, 'children') and len(elem.children):
                return xmldata._todict(elem.children[0])

        if hasattr(elem, 'children'):
            children = [xmldata._todict(child) for child in elem.children]
            children = xmldata._unique_children(children)
            if children:
                d = dict(d, **children)
        if hasattr(elem, '_attributes'):
            children = elem._attributes
            if children:
                d = dict(d, **children)
        if not len(d.keys()) and hasattr(elem, 'cdata'):
            return name, elem.cdata

        return name, xmldata._enumdict(d)

    @staticmethod
    def _output(s):
        if isinstance(s, dict):
            return xmldata(s)
        elif isinstance(s, (tuple, list)):
            return [xmldata._output(i) for i in s]
        elif not isinstance(s, str):
            return s
        elif len(s) > 1 and s[0] == '[' and s[-1] == ']':
            return [xmldata._output(i) for i in s[1:-1].split(', ')]
        elif re.search('^[-+]?\d+$', s):
            return int(s)
        elif re.search('^[-+]?\d?\d*\.?\d+([eE][-+]?\d+)?$', s):
            return float(s)
        elif s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        elif s.lower() == 'none':
            return None
        else:
            return s


class imread:
    ''' class to read image files, while taking good care of important metadata,
            currently optimized for .czi files, but can open anything that bioformats can handle
        path: path to the image file
        optional:
        series: in case multiple experiments are saved in one file, like in .lif files
        transform: automatically correct warping between channels, need transforms.py among others
        meta: define metadata, used for pickle-ing
        beadfile: image file with beads which can be used for correcting warp

        NOTE: run imread.kill_vm() at the end of your script/program, otherwise python will not terminate

        modify images on the fly with a decorator function:
            define a function which takes an instance of this object, one image frame,
            and the coordinates c, z, t as arguments, and one image frame as return
            >> imread.frame_decorator = fun
            then use imread as usually

        Examples:
            >> im = imread('/DATA/lenstra_lab/w.pomp/data/20190913/01_YTL639_JF646_DefiniteFocus.czi')
            >> im
             << shows summary
            >> im.shape
             << (256, 256, 2, 1, 600)
            >> plt.imshow(im(1, 0, 100))
             << plots frame at position c=1, z=0, t=100 (python type indexing), note: round brackets; always 2d array with 1 frame
            >> data = im[:,:,0,0,:25]
             << retrieves 5d numpy array containing first 25 frames at c=0, z=0, note: square brackets; always 5d array
            >> plt.imshow(im.maxz(0, 0))
             << plots max-z projection at c=0, t=0
            >> len(im)
             << total number of frames
            >> im.pxsize
             << 0.09708737864077668 image-plane pixel size in um
            >> im.laserwavelengths
             << [642, 488]
            >> im.laserpowers
             << [0.02, 0.0005] in %

            See __init__ and other functions for more ideas.

        wp@tl2019
    '''

    def __init__(self, path, series=0, transform=False, beadfile=None, dtype=None, meta=None):
        if isinstance(path, np.ndarray):
            self.path = path
            self.filetype = 'ndarray'
        else:
            if isinstance(path, type(self)):
                path = path.path
            self.path = os.path.abspath(path)
            self.filetype = os.path.splitext(path)[1]
            if path == '' and not meta is None and 'filetype' in meta:
                self.filetype = meta['filetype']
        self.beadfile = beadfile
        self.dtype = dtype

        self.shape = (0, 0, 0, 0, 0)
        self.series = series
        self.pxsize = 1e-1
        self.settimeinterval = 0
        self.pxsizecam = 0
        self.magnification = 0
        if self.filetype == 'ndarray':
            self.title = 'numpy array'
            self.acquisitiondate = 'now'
        else:
            self.title = os.path.splitext(os.path.basename(self.path))[0]
            self.acquisitiondate = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime('%y-%m-%dT%H:%M:%S')
        self.exposuretime = (0,)
        self.deltaz = 1
        self.pcf = (1, 1)
        self.timeseries = False
        self.zstack = False
        self.laserwavelengths = ()
        self.laserpowers = ()
        self.powermode = 'normal'
        self.optovar = (1,)
        self.binning = 1
        self.collimator = (1,)
        self.tirfangle = (0,)
        self.gain = (100, 100)
        self.objective = 'unknown'
        self.filter = 'unknown'
        self.NA = 1
        self.cyllens = ['None', 'A']
        self.duolink = '488/640'
        self.detector = [0, 1]
        self.metadata = {}

        self.cache = deque_dict(16)
        self._frame_decorator = None

        # how far is the center of the frame removed from the center of the sensor
        self.frameoffset = (self.shape[0] / 2, self.shape[1] / 2)

        if self.filetype == '':
            self.seqread()
        elif self.filetype == 'ndarray':
            self.ndarray()
        elif self.filetype == '.czi':
            self.cziread()
        elif self.filetype in ('.tif', '.tiff'):
            self.tiffread()
        elif self.filetype:
            self.bfread()

        if not hasattr(self, 'cnamelist'):
            self.cnamelist = 'abcdefghijklmnopqrstuvwxyz'[:self.shape[2]]

        if not meta is None:
            for key, item in meta.items():
                self.__dict__[key] = item

        if 'None' in self.cyllens:
            self.slavech = self.cyllens.index('None')
            self.masterch = 1 - self.slavech  # channel with cyllens
        else:
            self.masterch, self.slavech = 1, 0

        m = self.extrametadata
        if not m is None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                self.masterch = m['FeedbackChannel']
                self.slavech = 1 - self.masterch
            except:
                pass

        self.zstack = self.shape[3] > 1

        parameter = np.zeros((3, 5))
        parameter[0,] = self.shape
        parameter[1,] = (2, 1, 3, 4, 5)
        parameter[2,] = (self.pxsize, self.pxsize, 0, 0, self.timeinterval)
        self.parameter = parameter

        # handle transforms
        if transform is False:
            self.dotransform = False
        else:
            self.dotransform = True
            if isinstance(transform, Transform):
                self.transform = transform
                self.transform.adapt(self.frameoffset, self.shape)
            else:
                self.transformFromBeads()
                self.transform.adapt(self.frameoffset, self.shape)
            self.__framet__ = lambda c, z, t: self.transform.frame(self.__frame__(c, z, t))

        # self.xmeta = xmldata(self.omedata)

    def cziread(self):
        #TODO: make sure frame function still works when a subblock has data from more than one frame
        self.reader = czifile.CziFile(self.path)
        self.close = self.reader.close
        self.reader.asarray()
        self.shape = tuple([self.reader.shape[self.reader.axes.index(directory_entry)] for directory_entry in 'XYCZT'])
        self.timeseries = self.shape[4] > 1
        self.zstack = self.shape[3] > 1

        def get_index(directory_entry, start):
            return [(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, directory_entry.shape)]

        def frame(c=0, z=0, t=0):
            f = np.zeros(self.shape[:2], self.dtype)
            for directory_entry in self.filedict[(c, z, t)]:
                subblock = directory_entry.data_segment()
                tile = subblock.data(resize=True, order=0)
                index = [slice(i - j, i - j + k) for i, j, k in zip(directory_entry.start, self.reader.start, tile.shape)]
                index = tuple([index[self.reader.axes.index(i)] for i in 'XY'])
                f[index] = tile.squeeze()
            return f

        filedict = {}
        for directory_entry in self.reader.filtered_subblock_directory:
            idx = get_index(directory_entry, self.reader.start)
            for c in range(*idx[self.reader.axes.index('C')]):
                for z in range(*idx[self.reader.axes.index('Z')]):
                    for t in range(*idx[self.reader.axes.index('T')]):
                        if (c, z, t) in filedict:
                            filedict[(c, z, t)].append(directory_entry)
                        else:
                            filedict[(c, z, t)] = [directory_entry]
        self.filedict = filedict
        self.__frame__ = frame

        self.metadata = xmldata(untangle.parse(self.reader.metadata()))

        image = list(self.metadata.search_all('Image').values())
        if len(image) and self.series in image[0]:
            image = xmldata(image[0][self.series])
        else:
            image = self.metadata

        pxsize = image.search('ScalingX')[0]
        if not pxsize is None:
            self.pxsize = pxsize * 1e6
        if self.zstack:
            deltaz = image.search('ScalingZ')[0]
            if not deltaz is None:
                self.deltaz = deltaz * 1e6

        self.title = self.metadata.re_search(('Information', 'Document', 'Name'), self.title)[0]
        self.acquisitiondate = self.metadata.re_search(('Information', 'Document', 'CreationDate'),
                                                       self.acquisitiondate)[0]
        self.exposuretime = self.metadata.re_search(('TrackSetup', 'CameraIntegrationTime'), self.exposuretime)
        if self.timeseries:
            self.settimeinterval = self.metadata.re_search(('Interval', 'TimeSpan', 'Value'),
                                                           self.settimeinterval * 1e3)[0] / 1000
            if not self.settimeinterval:
                self.settimeinterval = self.exposuretime[0]
        self.pxsizecam = self.metadata.re_search(('AcquisitionModeSetup', 'PixelPeriod'), self.pxsizecam)
        self.magnification = self.metadata.re_search('NominalMagnification', self.magnification)[0]
        attenuator = self.metadata.search('Attenuator')
        self.laserwavelengths = [1e9 * float(i['Wavelength']) for i in attenuator]
        self.laserpowers = [float(i['Transmission']) for i in attenuator]
        self.collimator = self.metadata.re_search(('Collimator', 'Position'))
        detector = self.metadata.search(('Instrument', 'Detector'))
        self.gain = [int(i['AmplificationGain']) for i in detector]
        self.powermode = self.metadata.re_search(('TrackSetup', 'FWFOVPosition'))[0]
        optovar = self.metadata.re_search(('TrackSetup', 'TubeLensPosition'), '1x')
        self.optovar = []
        for o in optovar:
            a = re.search('\d?\d*[,\.]?\d+(?=x$)', o)
            if hasattr(a, 'group'):
                self.optovar.append(float(a.group(0).replace(',', '.')))
        self.pcf = [2 ** self.metadata.re_search(('Image', 'ComponentBitCount'), 14)[0] / i \
                    for i in self.metadata.re_search(('Channel', 'PhotonConversionFactor'), 1)]
        self.binning = self.metadata.re_search(('AcquisitionModeSetup', 'CameraBinning'), 1)[0]
        self.objective = self.metadata.re_search(('AcquisitionModeSetup', 'Objective'))[0]
        self.NA = self.metadata.re_search(('Instrument', 'Objective', 'LensNA'))[0]
        self.filter = self.metadata.re_search(('TrackSetup', 'BeamSplitter', 'Filter'))[0]
        self.tirfangle = [50 * i for i in self.metadata.re_search(('TrackSetup', 'TirfAngle'), 0)]
        self.frameoffset = [self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetX'))[0],
                            self.metadata.re_search(('AcquisitionModeSetup', 'CameraFrameOffsetY'))[0]]
        d = self.metadata.re_search(('Instrument', 'Detector'))
        if not d is None:
            self.detector = [int(i[-1]) for i in d[0].search_all('Id').values()]
            if len(self.detector) == 0:
                self.detector = [0]
        else:
            self.detector = [0]
        # self.detector = [int(i[-1]) for i in self.metadata.re_search(('Instrument', 'Detector', 'Id'), [[0]])]

        if 1.5 < self.NA < 1.6:
            self.immersionN = 1.661
        elif 1.3 < self.NA < 1.5:
            self.immersionN = 1.518
        else:
            self.immersionN = 1.33

    @property
    def frame_decorator(self):
        return self._frame_decorator

    @frame_decorator.setter
    def frame_decorator(self, decorator):
        if self.filetype == 'ndarray':
            if not 'origcache' in self:
                self.origcache = self.cache
            if decorator is None:
                self.cache = self.origcache
            else:
                for k, v in self.origcache.items():
                    self.cache[k] = decorator(self, v, *k)
        else:
            self._frame_decorator = decorator
            self.cache = deque_dict(self.cache.maxlen)

    def __iter__(self):
        self.index = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            res = self(self.index)
            self.index += 1
            return res

    def __repr__(self):
        """ gives a helpfull summary of the recorded experiment
        """
        s = '##########################################################################################################\n'
        s += 'path/filename: {}\n'.format(self.path)
        s += 'shape (xyczt): {} x {} x {} x {} x {}\n'.format(*self.shape)
        s += 'pixelsize:     {:.2f} nm\n'.format(self.pxsize * 1000)
        if self.zstack:
            s += 'z-interval:    {:.2f} nm\n'.format(self.deltaz * 1000)
        s += 'Exposuretime:  ' + ('{:.2f} ' * len(self.exposuretime)).format(
            *(np.array(self.exposuretime) * 1000)) + 'ms\n'
        if self.timeseries:
            if self.timeval and np.diff(self.timeval).shape[0]:
                s += 't-interval:    {:.3f} ± {:.3f} s\n'.format(
                    np.diff(self.timeval).mean(),
                    np.diff(self.timeval).std())
            else:
                s += 't-interval:    {:.2f} s\n'.format(self.settimeinterval)
        s += 'binning:       {}x{}\n'.format(self.binning, self.binning)
        s += 'laser colors:  ' + ('{:.0f} ' * len(self.laserwavelengths)).format(*self.laserwavelengths) + 'nm\n'
        s += 'laser powers:  ' + ('{} ' * len(self.laserpowers)).format(*(np.array(self.laserpowers) * 100)) + '%\n'
        s += 'objective:     {}\n'.format(self.objective)
        s += 'magnification: {}x\n'.format(self.magnification)
        s += 'optovar:      ' + (' {}' * len(self.optovar)).format(*self.optovar) + 'x\n'
        s += 'filterset:     {}\n'.format(self.filter)
        s += 'powermode:     {}\n'.format(self.powermode)
        s += 'collimator:   ' + (' {}' * len(self.collimator)).format(*self.collimator) + '\n'
        s += 'TIRF angle:   ' + (' {:.2f}°' * len(self.tirfangle)).format(*self.tirfangle) + '\n'
        s += 'gain:         ' + (' {:.0f}' * len(self.gain)).format(*self.gain) + '\n'
        s += 'pcf:          ' + (' {:.2f}' * len(self.pcf)).format(*self.pcf)
        return s

    def __str__(self):
        return self.path

    def __len__(self):
        return self.shape[2] * self.shape[3] * self.shape[4]

    def __call__(self, *n):
        """ returns single 2D frame
            im(n):     index linearly in czt order
            im(c,z):   return im(c,z,t=0)
            im(c,z,t): return im(c,z,t)
        """
        if len(n) == 1:
            n = self.get_channel(n[0])
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return self.frame(c, z, t)
        return self.frame(*n)

    def __getitem__(self, n):
        """ returns sliced 5D block
            im[n]:     index linearly in czt order
            im[c,z]:   return im(c,z,t=0)
            im[c,z,t]: return im(c,z,t)
            RESULT IS ALWAYS 5D!
        """
        if isinstance(n, type(Ellipsis)):
            return self.block()
        if not isinstance(n, tuple):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return self.block(None, None, c, z, t)
        n = list(n)

        ell = [i for i, e in enumerate(n) if isinstance(e, type(Ellipsis))]
        if len(ell)>1:
            raise IndexError("an index can only have a single ellipsis ('Ellipsis')")
        if len(ell):
            if len(n)>5:
                n.remove(Ellipsis)
            else:
                n[ell[0]] = slice(0, -1, 1)
                for i in range(5-len(n)):
                    n.insert(ell[0], slice(0, -1, 1))
        if len(n) in (2, 4):
            n.append(slice(0, -1, 1))
        for _ in range(5-len(n)):
            n.insert(0, slice(0, -1, 1))
        for i, e in enumerate(n):
            if isinstance(e, slice):
                a = [e.start, e.stop, e.step]
                if a[0] is None:
                    a[0] = 0
                if a[1] is None:
                    a[1] = -1
                if a[2] is None:
                    a[2] = 1
                for j in range(2):
                    if a[j] < 0:
                        a[j] %= self.shape[i]
                        a[j] += 1
                n[i] = np.arange(*a)
        n = [np.array(i) for i in n]
        if len(n) == 3:
            return self.block(None, None, *n)
        if len(n) == 5:
            return self.block(*n)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if hasattr(self, 'close'):
            self.close()

    def __reduce__(self):
        if self.filetype=='ndarray':
            return (self.__class__, (self[Ellipsis], self.series, self.dotransform, self.beadfile, self.dtype))
        else:
            return (self.__class__, (self.path, self.series, self.dotransform, self.beadfile, self.dtype))

    @property
    def sigma(self):
        """ gives the sigma of the theoretical psf in in the two channels
            assume typical stokes-shift is 22 nm
            Do not blindly rely on this to give the correct answer.
        """
        if len(self.laserwavelengths) == 1:
            return [(self.laserwavelengths[0] + 22) / 2 / self.NA / self.pxsize / 1000] * self.shape[2]
        else:
            return [(self.laserwavelengths[self.detector[self.czt(n)[0]]] + 22) / 2 / self.NA / self.pxsize / 1000 for n
                    in range(self.shape[2])]

    def czt(self, n):
        """ returns indices c, z, t used when calling im(n)
        """
        if not isinstance(n, tuple):
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
            return (c, z, t)
        n = list(n)
        if len(n) == 2 or len(n) == 4:
            n.append(slice(0, -1, 1))
        if len(n) == 3:
            n = list(n)
            for i, e in enumerate(n):
                if isinstance(e, slice):
                    a = [e.start, e.stop, e.step]
                    if a[0] is None:
                        a[0] = 0
                    if a[1] is None:
                        a[1] = -1
                    if a[2] is None:
                        a[2] = 1
                    for j in range(2):
                        if a[j] < 0:
                            a[j] %= self.shape[2 + i]
                            a[j] += 1
                    n[i] = np.arange(*a)
            n = [np.array(i) for i in n]
            return tuple(n)
        if len(n) == 5:
            return tuple(n[2:5])

    def czt2n(self, c, z, t):
        return c + z * self.shape[2] + t * self.shape[2] * self.shape[3]

    def transform_frame(self, frame, c, *args):
        if self.dotransform and self.detector[c] == self.masterch:
            self.transform.frame(frame)
        else:
            return frame

    def get_czt(self, c, z, t):
        if c is None:
            c = range(self.shape[2])
        if z is None:
            z = range(self.shape[3])
        if t is None:
            t = range(self.shape[4])
        c = tolist(c)
        z = tolist(z)
        t = tolist(t)
        c = [self.get_channel(ic) for ic in c]
        return c, z, t

    def min(self, c=None, z=None, t=None):
        c, z, t = self.get_czt(c, z, t)
        T = np.full(self.shape[:2], np.inf, self.dtype)
        for ic in c:
            m = np.full(self.shape[:2], np.inf, self.dtype)
            for iz, it in product(z, t):
                m = np.nanmin((m, self.__frame__(ic, iz, it)), 0)
            T = np.nanmin((T, self.transform_frame(m, ic)), 0)
        return T.astype(self.dtype)

    def max(self, c=None, z=None, t=None):
        c, z, t = self.get_czt(c, z, t)
        T = np.full(self.shape[:2], -np.inf, self.dtype)
        for ic in c:
            m = np.full(self.shape[:2], -np.inf, self.dtype)
            for iz, it in product(z, t):
                m = np.nanmax((m, self.__frame__(ic, iz, it)), 0)
            T = np.nanmax((T, self.transform_frame(m, ic)), 0)
        return T.astype(self.dtype)

    def maxz(self, c=0, t=0):
        # deprecated, use max() instead
        return self.max(c, None, t)

    def mean(self, c=None, z=None, t=None):
        # TODO: handle nans correctly
        c, z, t = self.get_czt(c, z, t)
        n = len(c) * len(z) * len(t)
        T = np.zeros(self.shape[:2], float)
        for ic in c:
            m = np.zeros(self.shape[:2], float)
            for iz, it in product(z, t):
                m += self.__frame__(ic, iz, it).astype(float) / n
            T += self.transform_frame(m, ic)
        return T

    def sum(self, c=None, z=None, t=None):
        c, z, t = self.get_czt(c, z, t)
        T = []
        for ic in c:
            m = np.zeros(self.shape[:2], self.dtype)
            for iz, it in product(z, t):
                m = np.nansum((m, self.__frame__(ic, iz, it)), 0)
            T.append(self.transform_frame(m, ic))
        return np.nansum(T, 0)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = np.dtype(value)

    def get_channel(self, channel_name):
        if not isinstance(channel_name, str):
            return channel_name
        else:
            c = [i for i, c in enumerate(self.cnamelist) if c.lower().startswith(channel_name.lower())]
            assert len(c)>0, 'Channel {} not found in {}'.format(c, self.cnamelist)
            assert len(c)<2, 'Channel {} not unique in {}'.format(c, self.cnamelist)
            return c[0]

    def frame(self, c=0, z=0, t=0):
        """ returns single 2D frame
        """
        c = self.get_channel(c)
        c %= self.shape[2]
        z %= self.shape[3]
        t %= self.shape[4]

        # cache last n (default 16) frames in memory for speed (~250x faster)
        if (c, z, t) in self.cache:
            self.cache.move_to_end((c, z, t))
            f = self.cache[(c, z, t)]
        else:
            if self.dotransform and self.detector[c] == self.masterch:
                f = self.__framet__(c, z, t)
            else:
                f = self.__frame__(c, z, t)
            if self.frame_decorator is not None:
                f = self.frame_decorator(self, f, c, z, t)
            self.cache[(c, z, t)] = f
        if not self.dtype is None:
            return f.copy().astype(self.dtype)
        else:
            return f.copy()

    def data(self, c=0, z=0, t=0):
        """ returns 3D stack of frames
        """
        if np.any(c == None):
            c = range(self.shape[2])
        if np.any(z == None):
            z = range(self.shape[3])
        if np.any(t == None):
            t = range(self.shape[4])
        c = tolist(c)
        z = tolist(z)
        t = tolist(t)
        s = len(c) * len(z) * len(t)
        d = np.full((self.shape[0], self.shape[1], s), np.nan)
        t, z, c = np.meshgrid(t, z, c)
        c = c.flatten()
        z = z.flatten()
        t = t.flatten()
        for i in range(s):
            d[:, :, i] = self.frame(c[i], z[i], t[i])
        return d

    def block(self, x=None, y=None, c=None, z=None, t=None):
        """ returns 5D block of frames
        """
        x, y, c, z, t = [tolist(range(self.shape[i])) if e is None else tolist(e) for i, e in enumerate((x, y, c, z, t))]
        s = len(c) * len(z) * len(t)
        d = np.full((len(x), len(y), len(c), len(z), len(t)), np.nan)
        t, z, c = np.meshgrid(t, z, c)
        c = c.flatten()
        z = z.flatten()
        t = t.flatten()
        C = c - min(c)
        Z = z - min(z)
        T = t - min(t)
        for i in range(s):
            d[:, :, C[i], Z[i], T[i]] = self.frame(c[i], z[i], t[i])[x][:, y]
        return d

    @property
    def timeval(self):
        try:
            if self.filetype == '.czi':
                tval = np.unique(list(filter(lambda x: x.attachment_entry.filename.startswith('TimeStamp'),
                                     self.reader.attachments()))[0].data())
                return sorted(tval[tval>0])[:self.shape[4]]
        except:
            if hasattr(self, 'metadata'):
                image = self.metadata.search('Image')
                if (isinstance(image, dict) and self.series in image) or (isinstance(image, list) and len(image)):
                    image = xmldata(image[0])
                return sorted(np.unique(image.search_all('DeltaT').values()))[:self.shape[4]]
            else:
                return np.arange(self.shape[4]) * self.timeinterval

    @property
    def timeinterval(self):
        try:
            if hasattr(self, 'timeval'):
                if len(self.timeval) > 1:
                    return np.diff(self.timeval).mean()
                else:
                    return self.settimeinterval
            else:
                return self.settimeinterval
        except:
            return self.settimeinterval

    @property
    def piezoval(self):
        """ gives the height of the piezo and focus motor, only available when CylLensGUI was used
        """

        def upack(idx):
            time = list()
            val = list()
            if len(idx) == 0:
                return time, val
            for i in idx:
                time.append(int(re.search('\d+', n[i]).group(0)))
                val.append(w[i])
            return zip(*sorted(zip(time, val)))

        # Maybe the values are stored in the metadata
        n = self.metadata.search('LsmTag|Name')[0]
        w = self.metadata.search('LsmTag')[0]
        if not n is None:
            # n = self.metadata['LsmTag|Name'][1:-1].split(', ')
            # w = str2float(self.metadata['LsmTag'][1:-1].split(', '))

            pidx = np.where([re.search('^Piezo\s\d+$', x) is not None for x in n])[0]
            sidx = np.where([re.search('^Zstage\s\d+$', x) is not None for x in n])[0]

            ptime, pval = upack(pidx)
            stime, sval = upack(sidx)

        # Or maybe in an extra '.pzl' file
        else:
            m = self.extrametadata
            if not m is None and 'p' in m:
                q = np.array(m['p'])
                if not len(q.shape):
                    q = np.zeros((1, 3))

                ptime = [int(i) for i in q[:, 0]]
                pval = [float(i) for i in q[:, 1]]
                sval = [float(i) for i in q[:, 2]]

            else:
                ptime = []
                pval = []
                sval = []

        df = pandas.DataFrame(columns=['frame', 'piezoZ', 'stageZ'])
        df['frame'] = ptime
        df['piezoZ'] = pval
        df['stageZ'] = np.array(sval) - np.array(pval) - \
                       self.metadata.re_search('AcquisitionModeSetup\|ReferenceZ', 0)[0] * 1e6

        # remove duplicates
        df = df[~df.duplicated('frame', 'last')]
        return df

    @property
    def extrametadata(self):
        if len(self.path) > 3:
            if os.path.isfile(self.path[:-3] + 'pzl2'):
                pname = self.path[:-3] + 'pzl2'
            elif os.path.isfile(self.path[:-3] + 'pzl'):
                pname = self.path[:-3] + 'pzl'
            else:
                return
            try:
                return getConfig(pname)
            except:
                return
        return

    def get_bead_files(self):
        if self.path.endswith('Pos0'):
            path = os.path.dirname(os.path.dirname(self.path))
        else:
            path = os.path.dirname(self.path)
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().startswith('beads')])
        if not files:
            raise Exception('No bead file found!')
        Files = []
        for file in files:
            try:
                if os.path.isdir(file):
                    file = os.path.join(file, 'Pos0')
                with imread(file) as im:  # check for errors opening the file
                    pass
                Files.append(file)
            except:
                continue
        if not Files:
            raise Exception('No bead file found!')
        return Files

    def transformFromBeads(self):
        if self.path.endswith('Pos0'):
            path = os.path.dirname(os.path.dirname(self.path))
        else:
            path = os.path.dirname(self.path)

        ymlpath = os.path.join(path, 'transform.yml')
        tifpath = os.path.join(path, 'transform.tif')
        if os.path.isfile(ymlpath):
            self.transform = Transform(ymlpath)
        else:
            print('No transform file found, trying to generate one.')
            if self.beadfile is None:
                self.beadfile = self.get_bead_files()
            files = self.beadfile
            if isinstance(files, str):
                files = (files,)

            with IJTiffWriter(tifpath, (2, 1, len(files))) as tif:
                T = []
                for s, file in enumerate(files):
                    print(f'Using {file} to calculate a transform.')
                    with imread(file) as im:
                        if im.shape[2] > 1:
                            imr = self.max(self.detector.index(self.slavech))
                            img = self.max(self.detector.index(self.masterch))
                            t = Transform(imr, img)
                            print(f'parameters: {t.parameters}')
                            tif.save(np.hstack((imr, imr)), 0, 0, s)
                            tif.save(np.hstack((img, t.frame(img))), 1, 0, s)
                            T.append(t)
            print(f'Saving transform in {ymlpath}.')
            print(f'Please check the transform in {tifpath}.')
            self.transform = Transform()
            self.transform.shape = T[0].shape
            self.transform.parameters = np.mean([t.parameters for t in T], 0)
            self.transform.dparameters = (np.std([t.parameters for t in T], 0) / np.sqrt(len(T))).tolist()
            self.transform.save(ymlpath)

    def save_as_tiff(self, fname=None, c=None, z=None, t=None, split=False, bar=True, pixel_type='uint16'):
        """ saves the image as a tiff-file
            split: split channels into different files
        """
        if fname is None:
            fname = self.path[:-3] + 'tif'
        elif not fname[-3:] == 'tif':
            fname += '.tif'
        if split:
            for i in range(self.shape[2]):
                if self.timeseries:
                    self.save_as_tiff(fname[:-3] + 'C{:01d}.tif'.format(i), i, 0, None, False, bar, pixel_type)
                else:
                    self.save_as_tiff(fname[:-3] + 'C{:01d}.tif'.format(i), i, None, 0, False, bar, pixel_type)
        else:
            n = [c, z, t]
            for i in range(len(n)):
                if n[i] is None:
                    n[i] = range(self.shape[i + 2])
                elif not isinstance(n[i], (tuple, list)):
                    n[i] = (n[i],)

            shape = [len(i) for i in n]
            at_least_one = False
            with IJTiffWriter(fname, shape, pixel_type) as tif:
                for i, m in tqdm(zip(product(*[range(s) for s in shape]), product(*n)),
                                 total=np.prod(shape), desc='Saving tiff', disable=not bar):
                    if np.any(self(*m)) or not at_least_one:
                        tif.save(self(*m), *i)
                        at_least_one = True

    @property
    def summary(self):
        print(self.__repr__())
