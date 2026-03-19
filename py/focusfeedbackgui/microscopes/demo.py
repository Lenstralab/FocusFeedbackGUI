import sys
import threading
from multiprocessing import Process, Manager
from time import sleep

import numpy as np
from ndbioimage import Imread

from focusfeedbackgui import QGui
from focusfeedbackgui.microscopes import MicroscopeClass
from focusfeedbackgui.utilities import QThread

try:
    from PySide6 import QtCore
    from PySide6.QtGui import QAction, QGuiApplication
    from PySide6.QtWidgets import *

    pyside = 6
except ImportError:
    from PySide2 import QtCore
    from PySide2.QtWidgets import *

    pyside = 2


class Microscope(MicroscopeClass):
    # from multiprocessing import Manager

    # manager = Manager()
    # dict = manager.dict()

    def __init__(self, _name, namespace, _in_feedback_loop=False, *_args, **_kwargs):
        self.namespace = namespace
        self.namespace.frame = 0, np.zeros((2, 256, 256))
        self.namespace.is_experiment_running = False
        self.namespace.frame_size = 256, 256
        self.namespace.objective_magnification = 100
        self.namespace.optovar_magnification = 1.6
        self.namespace.objective_na = 1.57
        self.namespace.pxsize = 97.07
        self.namespace.timeinterval = 0.1
        self.namespace.filename = ""
        self.thread = Process(target=main, args=(self.namespace,))

    @property
    def is_experiment_running(self):
        return self.namespace.is_experiment_running

    @property
    def frame_size(self):
        return self.namespace.frame_size

    @property
    def objective_magnification(self):
        return self.namespace.objective_magnification

    @property
    def optovar_magnification(self):
        return self.namespace.optovar_magnification

    @property
    def objective_na(self):
        return self.namespace.objective_na

    @property
    def pxsize(self):
        return self.namespace.pxsize

    @property
    def timeinterval(self):
        return self.namespace.timeinterval

    @property
    def filename(self):
        return self.namespace.filename

    def get_frame(self, channel=1, x=None, y=None, width=32, height=32):
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        t, frame = self.namespace.frame
        width = 2 * ((width + 1) // 2)  # Ensure evenness
        height = 2 * ((height + 1) // 2)
        y_max, x_max = frame[channel].shape
        if x is None:
            x = (x_max - width) // 2
        else:
            x -= width // 2
        if y is None:
            y = (y_max - height) // 2
        else:
            y -= height // 2
        return frame[channel, y : y + height, x : x + width], t

    def draw_rectangle(self, x, y, width, height, color=16777215, LineWidth=2, index=None):
        if index is None:
            index = 0
            while True:
                if index not in self.namespace.patches:
                    break
                index += 1
        self.namespace.patches[index] = x, y, width, height, color, LineWidth
        return index

    def draw_ellipse(self, x, y, radius, ellipticity, theta, color=65025, LineWidth=2, index=None):
        if index is None:
            index = 0
            while True:
                if index not in self.namespace.patches:
                    break
                index += 1
        self.namespace.patches[index] = x, y, radius / np.sqrt(ellipticity), radius * np.sqrt(ellipticity), theta, color, LineWidth
        return index

    def remove_drawing(self, index):
        """Remove drawing by index."""
        if index in self.namespace.patches:
            self.namespace.patches.pop(index)

    def remove_drawings(self):
        """Remove all drawings."""
        self.namespace.patches.clear()


class UiMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "demo microscope"
        self.width = 640
        self.height = 640
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.layout = QVBoxLayout()

        self.open_btn = QPushButton("Open")
        self.open_btn.setToolTip("Open image")

        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Play image")
        self.play_btn.setEnabled(False)

        main_buttons = QHBoxLayout()
        main_buttons.addWidget(self.open_btn)
        main_buttons.addWidget(self.play_btn)

        self.plots = QGui.PlotCanvas()
        self.imshow = QGui.SubImPlot(self.plots, 111)

        self.layout.addLayout(main_buttons)
        self.layout.addWidget(self.plots)

        self.setLayout(self.layout)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()


class App(UiMainWindow):
    def __init__(self, namespace=None):
        super().__init__()
        self.quit = False
        self.play_thread = None
        self.patch_thread = QThread(self.patches_threaded)
        if namespace is None:
            self.namespace = Manager().Namespace()
        else:
            self.namespace = namespace
        self.open_btn.clicked.connect(self.open)
        self.play_btn.clicked.connect(self.play)
        self.image = None

    def patches_threaded(self):
        patches = dict(self.namespace.patches)
        while not self.quit:
            new = dict(self.namespace.patches)
            if not patches == new:
                patches = new
                self.imshow.set_patches(patches)
            sleep(0.025)

    @staticmethod
    def outliers(D, keep=True):
        q2 = np.nanmedian(np.array(D).flatten())
        q1 = np.nanmedian(D[D < q2])
        q3 = np.nanmedian(D[D > q2])
        lb = 4 * q1 - 3 * q3
        ub = 4 * q3 - 3 * q1

        if keep:
            idx = np.where((D >= lb) & (D <= ub))
        else:
            idx = np.where(~((D >= lb) & (D <= ub)))
        return idx

    def rgbshow(self, t):
        """display an rgb image using 1, 2 or 3 nxm arrays as input
        """
        im = self.image[t].max("z")
        self.namespace.frame = t, im
        jm = []
        for i in im:
            i = i.astype(float)
            j = i.flatten()
            h = j[self.outliers(j[j > 0])]
            hmin, hmax = np.nanpercentile(h, (1, 100))
            hmax *= 1.5
            jm.append(np.clip((i - hmin) / (hmax - hmin), 0, 1))
        while len(jm) < 3:
            jm.append(np.zeros(im[0].shape))
        self.imshow.update(np.stack(jm, 2))

    def open(self):
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(
            self,
            "microscope image file",
            "",
            "All files (*)",
            options=options,
        )
        self.image = Imread(file, axes="tczyx")
        self.namespace.frame_size = self.image.shape["yx"]
        self.namespace.objective_magnification = self.image.objective.nominal_magnification
        self.namespace.optovar_magnification = self.image.tubelens.nominal_magnification
        self.namespace.objective_na = self.image.objective.lens_na
        self.namespace.pxsize = 1000 * self.image.pxsize_um
        self.namespace.timeinterval = self.image.timeinterval
        self.namespace.filename = str(self.image.path)
        self.rgbshow(0)
        self.play_btn.setEnabled(self.image.timeseries)

    def play(self):
        if self.namespace.is_experiment_running:
            self.namespace.is_experiment_running = False
            self.play_btn.setText("Play")
        else:
            self.play_thread = QThread(self.play_threaded)
            self.play_btn.setText("Stop")

    def play_threaded(self):
        self.namespace.is_experiment_running = True
        for t in range(1, self.image.shape["t"]):
            sleep(self.image.timeinterval)
            if not self.namespace.is_experiment_running:
                break
            self.rgbshow(t)
        self.namespace.is_experiment_running = False

    def closeEvent(self, *args, **kwargs):
        self.namespace.is_experiment_running = False
        self.quit = True
        for i in range(50):  # wait up to 5 seconds for all threads to stop
            if all([not thread.is_alive() for thread in threading.enumerate()]):
                break
            sleep(0.1)


def main():
    app = QApplication([])
    _window = App()
    if pyside == 6:
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()