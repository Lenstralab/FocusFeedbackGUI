import os
import numpy as np
from czifile import CziFile
from yaml import full_load as load
from re import findall

class imread(CziFile):
    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.frame_decorator = None
        super().__init__(path, *args, **kwargs)

        T = [self.axes.find(i) for i in 'YXCZT']
        E = [i for i, j in enumerate(T) if j < 0]
        T = [i for i in T if i >= 0]
        T.extend(set(range(len(self.axes))) - set(T))
        self.data = np.transpose(self.asarray(), T)
        for e in E:
            self.data = np.expand_dims(self.data, e)
        self.data = np.squeeze(self.data, tuple(range(5, self.data.ndim)))

        self.axes = 'YXCZT'
        self.shape = self.data.shape

        self.objective = \
        self.metadata(False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock'][
            'AcquisitionModeSetup']['Objective']
        self.NA = \
        self.metadata(False)['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][
            'LensNA']
        if 1.5<self.NA<1.6:
            self.immersionN = 1.661
        elif 1.3<self.NA<1.5:
            self.immersionN = 1.518
        else:
            self.immersionN = 1.33

        self.pxsize = 1e6 * self.metadata(False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks'][
            'AcquisitionBlock']['AcquisitionModeSetup']['ScalingX']
        self.deltaz = 1e6 * self.metadata(False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks'][
            'AcquisitionBlock']['AcquisitionModeSetup']['ScalingZ'] / (self.immersionN**2/1.33**2)
        self.laserwavelengths = [int(1e9 * i['Wavelength']) for i in
                                 self.metadata(False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks'][
                                     'AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Attenuators']['Attenuator']]
        self.detector = [int(findall('(?<=Detector:\d:)\d', i['Id'])[0]) for i in
                         self.metadata(False)['ImageDocument']['Metadata']['Information']['Instrument']['Detectors'][
                             'Detector']]
        self.optovar = [float(findall('\d,\d', self.metadata(False)['ImageDocument']['Metadata']['Experiment'][
            'ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['TubeLensPosition'])[0].replace(
            ',', '.'))]

        # should get this from somewhere else
        m = self.extrametadata
        self.cyllens = '[None, A]'
        self.masterch = 1
        self.slavech = 0
        if not m is None:
            try:
                self.cyllens = m['CylLens']
                self.duolink = m['DLFilterSet'].split(' & ')[m['DLFilterChannel']]
                self.masterch = m['FeedbackChannel']
                self.slavech = 1 - self.masterch
            except:
                pass

    def __call__(self, *n):
        if len(n) == 1:
            n = n[0]
            c = n % self.shape[2]
            z = (n // self.shape[2]) % self.shape[3]
            t = (n // (self.shape[2] * self.shape[3])) % self.shape[4]
        else:
            c, z, t = n
        frame = self.data[:, :, c, z, t].squeeze().astype(float)
        if self.frame_decorator is None:
            return frame
        else:
            return self.frame_decorator(self, frame, c, z, t)

    def maxz(self, c=0, t=0):
        """ returns max-z projection at color c and time t
        """
        T = np.full(self.shape[:2], -np.inf)
        for z in range(self.shape[3]):
            T = np.max(np.dstack((T, self(c, z, t))), 2)
        return T

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

    def sigma(self, n):
        """ gives the sigma of the theoretical psf in frame n or frame c, z, t
            assume typical stokes-shift is 22 nm
        """
        if len(self.laserwavelengths) == 1:
            return (self.laserwavelengths[0] + 22) / 2 / self.NA / self.pxsize / 1000
        else:
            return (self.laserwavelengths[
                        self.detector[self.czt(n)[0]] % self.shape[2]] + 22) / 2 / self.NA / self.pxsize / 1000

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
                with open(pname) as f:
                    config = load(f)
                return config
            except:
                return
        return
