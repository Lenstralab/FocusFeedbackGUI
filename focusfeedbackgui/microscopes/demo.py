import os
from tllab_common.wimread import imread
from focusfeedbackgui.microscopes import MicroscopeClass


class Microscope(MicroscopeClass):
    def __init__(self, *args, **kwargs):
        self.im = imread(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'files',
                                      'YTL639_2020_08_05__16_11_45.czi'))
        self.frame_size = self.im.shape[:2]
        self.objective_magnification = self.im.magnification
        self.optovar_magnification = self.im.optovar[0]
        self.objective_na = self.im.NA
        self.pxsize = self.im.pxsize
        self.time_interval = self.im.timeinterval
        self.filename = self.im.path

    def close(self, *args, **kwargs):
        self.im.close()
