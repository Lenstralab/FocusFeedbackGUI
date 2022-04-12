import numpy as np
from re import search
from abc import ABCMeta
from warnings import warn
from traceback import format_exc
from importlib import import_module


class MicroscopeClass(metaclass=ABCMeta):
    # default values which should be overloaded
    ready = True  # is the Microscope software initialized
    frame_size = 256, 256
    objective_magnification = 100
    optovar_magnification = 1.6
    objective_na = 1.57
    pxsize = 97.07
    time_interval = 1
    filename = ''
    channel_colors_int = 255, 65280
    channel_names = 'TV1', 'TV2'
    piezo_pos = 0  # position of the focus piezo in um
    focus_pos = 0  # position of the focus motor (not the piezo) in um
    stage_pos = 0, 0  # position of the stage (x, y) in um
    duolink_filter = 1
    time = 0
    is_experiment_running = False

    def __new__(cls, microscope, *args, **kwargs):
        if isinstance(microscope, type) and issubclass(microscope, MicroscopeClass):
            microscope_subclass = microscope
        else:
            try:
                microscope_subclass = import_module(f'focusfeedbackgui.microscopes.{microscope.lower()}').Microscope
            except ImportError:
                warn(f'Importing {microscope} failed, using demo instead.\n\n{format_exc()}')
                microscope_subclass = Demo
        return super().__new__(microscope_subclass)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self, *args, **kwargs):
        pass

    @staticmethod
    def wait(_, callback):
        """ Wait until microscope is initialized. """
        callback(0)

    @staticmethod
    def events(app):
        """ Return an Event class which deals with events happening in the microscope (software)."""
        pass

    @staticmethod
    def get_camera(channel_name):
        """ Return the index of the camera used by channel channel_name. """
        return int(search(r'(?<=TV)\d', channel_name).group(0)) - 1

    @property
    def channel_colors_hex(self):
        """ Return a list of colors used in the microscope software in hex/html representation. """
        colors = []
        for cci in self.channel_colors_int:
            h = np.base_repr(cci, 16, 6)[-6:]
            colors.append('#' + h[4:] + h[2:4] + h[:2])
        return colors

    @property
    def channel_colors_rgb(self):
        """ Return a list of colors used in the microscope software in rgb list representation. """
        return [[int(hcolor[2 * i + 1:2 * (i + 1) + 1], 16) / 255 for i in range(3)] for hcolor in
                self.channel_colors_hex]

    def draw_rectangle(self, x, y, width, height, color=16777215, LineWidth=2, index=None):
        """ Draw a rectangle on the image in the microscope software and return its index. """
        pass

    def draw_ellipse(self, x, y, radius, ellipticity, theta, color=65025, LineWidth=2, index=None):
        """ Draw an ellipse on the image in the microscope software and return its index. """
        pass

    def remove_drawing(self, index):
        """ Remove drawing by index. """
        pass

    def remove_drawings(self):
        """ Remove all drawings. """
        pass

    def enable_event(self, event):
        """ Enable event in the microscope software by name event, example: 'LeftButtonDown'"""
        pass

    def move_piezo_relative(self, dz):
        """ Move the piezo dz um. """
        self.piezo_pos += dz

    def move_stage_relative(self, dx=None, dy=None):
        """ Move the stage dx, dy um. """
        x, y = self.stage_pos
        if x is not None:
            x += dx
        if y is not None:
            y += dy
        self.stage_pos = (x, y)

    @property
    def magnification_str(self):
        """ Return a string representing the microscope magnification: 100x16. """
        return '{:.0f}x{:.0f}'.format(self.objective_magnification, 10 * self.optovar_magnification)

    def get_frame(self, channel=1, x=None, y=None, width=32, height=32):
        """ Return a part of the current frame in channel channel. """
        return np.zeros((width, height))


class Demo(MicroscopeClass):
    pass
