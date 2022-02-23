""" Shadows zen.py and events.py to provide a demo on systems without ZEN installed. """

import os
import numpy as np
from re import search
from tllab_common.wimread import imread


def Events(*args, **kwargs):
    pass


class zen:
    DLFilter = 1
    StagePos = [0, 0]
    PiezoPos = 0
    ChannelNames = ('TV1', 'TV2')
    ChannelColorsInt = [255, 65280]

    def __init__(self):
        self.im = imread(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files',
                                      'YTL639_2020_08_05__16_11_45.czi'))

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.im.close()

    @staticmethod
    def wait(_, callback):
        callback(0)

    @property
    def ready(self):
        return True

    @property
    def ExperimentRunning(self):
        return False

    @property
    def FrameSize(self):
        return self.im.shape[:2]

    @property
    def MagStr(self):
        return f'{self.im.magnification}x{10*self.im.optovar[0]:.0f}'

    @property
    def ObjectiveNA(self):
        return self.im.NA

    @property
    def pxsize(self):
        return self.im.pxsize

    @property
    def TimeInterval(self):
        return self.im.timeinterval

    @property
    def FileName(self):
        return self.im.path

    @staticmethod
    def CameraFromChannelName(ChannelName):
        return int(search('(?<=TV)\d', ChannelName).group(0)) - 1

    @property
    def ChannelColorsHex(self):
        color = []
        for cci in self.ChannelColorsInt:
            h = np.base_repr(cci, 16, 6)[-6:]
            color.append('#' + h[4:] + h[2:4] + h[:2])
        return color

    @property
    def ChannelColorsRGB(self):
        return [[int(hcolor[2 * i + 1:2 * (i + 1) + 1], 16) / 255 for i in range(3)] for hcolor in
                self.ChannelColorsHex]

    def DrawRectangle(self, *args, **kwargs):
        return

    def DrawEllipse(self, *args, **kwargs):
        return

    def RemoveDrawing(self, *args, **kwargs):
        return

    def RemoveDrawings(self, *args, **kwargs):
        return

    def EnableEvent(self, *args, **kwargs):
        return
