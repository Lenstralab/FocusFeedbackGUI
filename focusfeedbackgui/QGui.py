import warnings
from traceback import format_exc

import numpy as np
from matplotlib import pyplot, rcParams
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide2.QtWidgets import *

from focusfeedbackgui.utilities import error_wrap

rcParams.update({'figure.autolayout': True})
np.seterr(all='ignore')


class RadioButtons(QWidget):
    def __init__(self, txt, init_state=0):
        QWidget.__init__(self)
        self.layout = QGridLayout()
        self.radiobutton = []
        self.setLayout(self.layout)
        self.txt = txt
        self.state = txt[init_state]
        self.callback = None
        for i, t in enumerate(txt):
            self.radiobutton.append(QRadioButton(t))
            if i == init_state:
                self.radiobutton[-1].setChecked(True)
            self.radiobutton[-1].text = t
            self.radiobutton[-1].toggled.connect(self.onClicked)
            self.layout.addWidget(self.radiobutton[-1], 0, i)

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.state = radioButton.text
            if self.callback is not None:
                self.callback(radioButton.text)

    def changeState(self, state):
        for i, r in enumerate(self.radiobutton):
            if i == state:
                r.setChecked(True)
            else:
                r.setChecked(False)

    def connect(self, callback=None):
        self.callback = callback


class CheckBoxes(QWidget):
    def __init__(self, txt, init_state=None):
        QWidget.__init__(self)
        self.layout = QGridLayout()
        self.checkBoxes = []
        self.textBoxes = []
        self.textBoxValues = []
        self.setLayout(self.layout)
        self.state = [txt[i] for i in init_state] if init_state else []
        self.txt = txt
        self.callback = None
        self.callback_text = None
        for i, t in enumerate(txt):
            self.checkBoxes.append(QCheckBox(t))
            if init_state and i in init_state:
                self.checkBoxes[-1].setChecked(True)
            self.checkBoxes[-1].text = t
            self.checkBoxes[-1].toggled.connect(self.onClicked)
            self.layout.addWidget(self.checkBoxes[-1], i, 0)
            self.textBoxes.append(QLineEdit())
            self.textBoxes[-1].setText('500')
            self.textBoxes[-1].textChanged.connect(self.valChange(i))
            self.layout.addWidget(self.textBoxes[-1], i, 1)
            self.textBoxValues.append(500)

    def changeOptions(self, txt, colors, enabled):
        callback = False
        n = len(txt)
        for i, (cb, tb) in enumerate(zip(self.checkBoxes, self.textBoxes)):
            cb.setVisible(i < n)
            tb.setVisible(i < n)

        for cb, t, c, e in zip(self.checkBoxes, txt, colors, enabled):
            cb.setText(t)
            cb.setStyleSheet("QCheckBox\n{\nbackground-color: " + c + "\n}")
            if not e and cb.isChecked():
                cb.setChecked(False)
                callback = True
            cb.setEnabled(e)
        if callback and self.callback is not None:
            self.callback(self.state)

    def valChange(self, n):
        def fun(val):
            try:
                self.textBoxValues[n] = int(val)
            except Exception:
                warnings.warn(f'\n{format_exc()}')
            if self.callback_text:
                self.callback_text(n, val)
        return fun

    def setTextBoxValue(self, n, val):
        self.textBoxes[n].setText(str(val))
        self.textBoxValues[n] = val

    def getTextBoxValue(self, n):
        return self.textBoxValues[n]

    def onClicked(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.state.append(radio_button.text)
        else:
            self.state.remove(radio_button.text)
        if self.callback is not None:
            self.callback(self.state)

    def changeColor(self, n, color):
        try:
            self.checkBoxes[n].setStyleSheet("QRadioButton\n{\nbackground-color : " + color + "\n}")
        except Exception:
            warnings.warn(f'\n{format_exc()}')

    def connect(self, callback=None, callbacktext=None):
        self.callback = callback
        self.callback_text = callbacktext


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.subplot = []
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def remove_data(self):
        for subplot in self.subplot:
            subplot.remove_data()


class SubPlot:
    def __init__(self, canvas, position=111, linespec='-r', color=None, handle=0):
        if isinstance(position, tuple):
            self.ax = canvas.fig.add_subplot(*position)
        else:
            self.ax = canvas.fig.add_subplot(position)
        self.plt = {}
        self.append_plot(handle, linespec, color)
        canvas.subplot.append(self)
        self.canvas = canvas

    def __contains__(self, item):
        return item in self.plt

    def append_plot(self, handle, linespec='-b', color=None):
        if color is None:
            self.plt[handle] = self.ax.plot([], linespec)[0]
        else:
            self.plt[handle] = self.ax.plot([], linespec, color=color)[0]

    def append_data(self, x, y=None, handle=0):
        if y is None:
            y = x
            x = error_wrap(np.nanmax, -1)(self.plt[handle].get_xdata()) + np.arange(error_wrap(len, 1)(x)) + 1
        x = np.hstack((self.plt[handle].get_xdata(), x))
        y = np.hstack((self.plt[handle].get_ydata(), y))
        self.plt[handle].set_xdata(x)
        self.plt[handle].set_ydata(y)
        self.ax.relim()
        self.ax.autoscale_view()

    def range_data(self, x, y, lim=250, handle=0):
        margin = 0.05
        x = np.hstack((self.plt[handle].get_xdata(), x))
        y = np.hstack((self.plt[handle].get_ydata(), y))
        y = y[x > np.nanmax(x) - lim]
        x = x[x > np.nanmax(x) - lim]

        # x, y = zip(*[i for i in sorted(zip(x, y))])

        self.plt[handle].set_ydata(y)
        self.plt[handle].set_xdata(x)

        x = np.hstack([plt.get_xdata() for plt in self.plt.values()])
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        deltax = (xmax - xmin) * margin
        if np.isfinite(deltax) and not xmin == xmax:
            self.ax.set_xlim(xmin - deltax, xmax + deltax)

        y = np.hstack([plt.get_ydata() for plt in self.plt.values()])
        if np.any(np.isfinite(y)):
            ymin, ymax = np.nanmin(y), np.nanmax(y)
        else:
            ymin, ymax = np.nan, np.nan
        deltay = (ymax - ymin) * margin
        if np.isfinite(deltay) and not ymin == ymax:
            self.ax.set_ylim(ymin - deltay, ymax + deltay)
        self.ax.autoscale_view()

    def numel_data(self, x, y, lim=10000, handle=0):
        x = np.hstack((self.plt[handle].get_xdata(), x))
        y = np.hstack((self.plt[handle].get_ydata(), y))
        self.plt[handle].set_ydata(y[-lim:])
        self.plt[handle].set_xdata(x[-lim:])
        self.ax.relim()
        self.ax.autoscale_view()

    def remove_data(self):
        for plt in self.plt.values():
            plt.set_xdata([])
            plt.set_ydata([])

    def draw(self):
        self.canvas.draw()


class SubPatchPlot:
    def __init__(self, canvas, position=111, color='r'):
        if isinstance(position, tuple):
            self.ax = canvas.fig.add_subplot(*position)
        else:
            self.ax = canvas.fig.add_subplot(position)
        canvas.subplot.append(self)
        self.canvas = canvas
        self.rects = []
        self.color = color

    def numel_data(self, x, y, dx, dy=None, lim=1000):
        if self.rects:
            while len(self.rects) >= lim:
                self.rects.pop(0).remove()
        self.append_data(x, y, dx, dy)

    def append_data(self, x, y, dx, dy=None):
        if dy is None:
            dy = dx
        rect = pyplot.Rectangle((x-dx/2, y-dy/2), dx, dy, facecolor=self.color, alpha=1, edgecolor='g', zorder=1)
        self.ax.add_patch(rect)
        if self.rects:
            self.rects[-1].set_edgecolor(None)
        self.rects.append(rect)
        self.ax.relim()
        self.ax.autoscale_view()

    def append_data_docs(self, x, y, dx, dy=None):
        if dy is None:
            dy = dx
        rect = pyplot.Rectangle((x-dx/2, y-dy/2), dx, dy, fill=False, alpha=1, edgecolor='k', zorder=10)
        self.ax.add_patch(rect)
        if self.rects:
            self.rects[-1].set_edgecolor(None)
        self.ax.relim()
        self.ax.autoscale_view()

    def remove_data(self):
        if self.rects:
            while len(self.rects) > 0:
                self.rects.pop(0).remove()
        self.canvas.draw()

    def draw(self):
        self.canvas.draw()
