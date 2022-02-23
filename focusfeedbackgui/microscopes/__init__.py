import numpy as np
from re import search
from abc import ABCMeta
from warnings import warn
from traceback import format_exc


class MicroscopeClass(metaclass=ABCMeta):
    def __new__(cls, microscope, *args, **kwargs):
        from focusfeedbackgui.microscopes.demo import Microscope
        try:
            if microscope.lower() == 'zen_black':
                from focusfeedbackgui.microscopes.zen_black import Microscope
        except ImportError:
            warn(f'Importing {microscope} failed, importing demo instead.\n{format_exc()}')
        return super().__new__(Microscope)

    def __init__(self, *args, **kwargs):
        self.ready = True
        self.frame_size = 256, 256
        self.objective_magnification = 100
        self.optovar_magnification = 1.6
        self.objective_na = 1.57
        self.pxsize = 97.07
        self.time_interval = 1
        self.filename = ''
        self.channel_colors_int = 255, 65280
        self.channel_names = 'TV1', 'TV2'
        self.piezo_pos = 0
        self.focus_pos = 0
        self.stage_pos = 0, 0
        self.duolink_filter = 1
        self.time = 0
        self.is_experiment_running = False

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self, *args, **kwargs):
        pass

    @staticmethod
    def wait(_, callback):
        callback(0)

    @staticmethod
    def events(*args, **kwargs):
        pass

    @staticmethod
    def get_camera(channel_name):
        return int(search(r'(?<=TV)\d', channel_name).group(0)) - 1

    @property
    def channel_colors_hex(self):
        color = []
        for cci in self.channel_colors_int:
            h = np.base_repr(cci, 16, 6)[-6:]
            color.append('#' + h[4:] + h[2:4] + h[:2])
        return color

    @property
    def channel_colors_rgb(self):
        return [[int(hcolor[2 * i + 1:2 * (i + 1) + 1], 16) / 255 for i in range(3)] for hcolor in
                self.channel_colors_hex]

    def draw_rectangle(self, *args, **kwargs):
        return

    def draw_ellipse(self, *args, **kwargs):
        return

    def remove_drawing(self, *args, **kwargs):
        return

    def remove_drawings(self, *args, **kwargs):
        return

    def enable_event(self, *args, **kwargs):
        return

    def move_piezo_relative(self, dz):
        # in um
        self.piezo_pos += dz
        return

    def move_stage_relative(self, dx=None, dy=None):
        # in um
        x, y = self.stage_pos
        if x is not None:
            x += dx
        if y is not None:
            y += dy
        self.stage_pos = (x, y)
        return

    @property
    def magnification_str(self):
        return '{:.0f}x{:.0f}'.format(self.objective_magnification, 10 * self.optovar_magnification)

    def get_frame(self, channel=1, x=None, y=None, width=32, height=32):
        return np.zeros((width, height))
