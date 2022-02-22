import os
import yaml
import re
import numpy as np
from sys import exit
from shutil import copyfile
from time import sleep, time
from datetime import datetime
from functools import partial
from multiprocessing import Process, Queue, Manager, freeze_support
from collections import deque
from PyQt5.QtWidgets import *
from PyQt5 import QtCore

if __package__ == '':
    import QGUI
    import cylinderlens as cyl
    import functions
    from utilities import qthread, yamlload
    from pid import pid
    from imread import warp
    try:
        from zen import zen, Events
    except Exception:
        from Fzen import zen, Events
else:
    from . import QGUI
    from . import cylinderlens as cyl
    from . import functions
    from .utilities import qthread, yamlload
    from .pid import pid
    from .imread import warp
    try:
        from .zen import zen, Events
    except Exception:
        from .Fzen import zen, Events

np.seterr(all='ignore')


def firstargonly(fun):
    """ decorator that only passes the first argument to a function
    """
    return lambda *args: fun(args[0])


def feedbackloop(Queue, NS):
    # this is run in a separate process
    Z = zen()
    _ = Z.PiezoPos

    def get_cm_str(channel):
        return NS.cyllens[Z.CameraFromChannelName(Z.ChannelNames[channel])] + Z.MagStr

    while not NS.quit:
        if NS.run:
            TimeInterval = Z.TimeInterval
            G = Z.PiezoPos
            FS = Z.FrameSize
            TimeMem = 0
            STimeMem = time()
            piezoTime = .5  # time piezo needs to settle in s
            detected = deque((True,) * 5, 5)
            zmem = {}

            if NS.feedback_mode == 'pid':
                P = pid(0, G, NS.max_step, TimeInterval, NS.gain)

            while Z.ExperimentRunning and (not NS.stop):
                ellipticity = {}
                PiezoPos = Z.PiezoPos
                FocusPos = Z.GetCurrentZ
                xyPos = {}

                for channel in NS.channels:
                    STime = time()
                    Frame, Time = Z.GetFrame(channel, *functions.cliprect(FS, *NS.roi_pos, NS.roi_size, NS.roi_size))

                    a = np.hstack(functions.fitgauss(Frame, NS.theta[get_cm_str(channel)], NS.sigma[channel],
                                                     NS.fast_mode))
                    if Time < TimeMem:
                        TTime = TimeMem + 1
                    else:
                        TTime = Time

                    # try to determine when something is detected by using a simple filter on R2, el and psf width
                    if not any(np.isnan(a)) and all([l[0] < n < l[1] for n, l in zip(a, NS.limits[channel])]):
                        detected.append(True)
                        if sum(detected)/len(detected) > 0.35:
                            Fitted = True
                            ellipticity[channel] = a[5]
                            xyPos[channel] = a[:2]
                        else:
                            Fitted = False
                    else:
                        Fitted = False
                        detected.append(False)
                    Queue.put((channel, TTime, Fitted, a[:8], PiezoPos, FocusPos, STime))

                STime = time()
                pzFactor = np.clip((STime - STimeMem) / piezoTime, 0.2, 1)

                # Update the piezo position:
                if NS.feedback_mode == 'pid':
                    if np.any(np.isfinite(list(ellipticity.values()))):
                        e = np.nanmean(list(ellipticity.values()))
                        F = -np.log(e)
                        if np.abs(F) > 1:
                            F = 0
                        Pz = P(F)
                        Z.PiezoPos = Pz

                        if Pz > (G + 5):
                            P = pid(0, G, NS.max_step, TimeInterval, NS.gain)
                else:  # Zhuang
                    dz = {channel: np.clip(cyl.findz(e, NS.q[get_cm_str(channel)]), -NS.max_step, NS.max_step)
                          for channel, e in ellipticity.items()}
                    cur_channel_idx = TTime % len(NS.channels)
                    next_channel_idx = (TTime + 1) % len(NS.channels)
                    if np.any(np.isfinite(list(dz.values()))):
                        if NS.mult_channel_mode == 0:  # Average
                            dz = np.nanmean(list(dz.values()))
                            if not np.isnan(dz):
                                Z.PiezoPos -= dz * pzFactor  # reduce if going faster than piezo, avoid oscillations
                        else:  # Alternate: save focus in current channel, apply focus to piezo for next channel
                            if np.isfinite(dz[NS.channels[cur_channel_idx]]):
                                zmem[NS.channels[cur_channel_idx]] = PiezoPos + dz[NS.channels[cur_channel_idx]] * pzFactor
                            elif not NS.channels[cur_channel_idx] in zmem:
                                zmem[NS.channels[cur_channel_idx]] = PiezoPos
                            if NS.channels[next_channel_idx] in zmem:
                                Z.PiezoPos = zmem[NS.channels[next_channel_idx]]

                if xyPos:  # Maybe move stage or ROI
                    xy = np.mean(list(xyPos.values()), 0)
                    if NS.feedback_mode_xy == 1:
                        NS.roi_pos = (NS.roi_pos + np.clip(xy - NS.roi_size / 2, -NS.max_step_xy, NS.max_step_xy)
                                      + 1 / 2).astype(int).tolist()
                    if NS.feedback_mode_xy == 2:
                        Z.MoveStageRel(*np.clip(xy - NS.roi_size / 2, -NS.max_step_xy, NS.max_step_xy) * Z.pxsize / 1e3)

                # Wait for next frame:
                while ((Z.GetTime - 1) == Time) and Z.ExperimentRunning and (not NS.stop) \
                        and (not NS.quit):
                    sleep(TimeInterval / 4)
                if Time < TimeMem:
                    break

                TimeMem = Time
                STimeMem = STime
                NS.run = False
        else:
            sleep(0.01)


class UiMainWindow(object):
    def setupUI(self, MainWindow):
        screen = QDesktopWidget().screenGeometry()
        self.title = 'Cylinder lens feedback GUI'
        self.width = 640
        self.height = 1024
        self.right = screen.width() - self.width
        self.top = 32
        self.color = '#454D62'
        self.textColor = '#FFFFFF'

        MainWindow.setWindowTitle(self.title)
        MainWindow.setMinimumSize(self.width, self.height)
        MainWindow.setGeometry(self.right, self.top, self.width, self.height)

        self.central_widget = QWidget(MainWindow)
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget(self.central_widget)
        self.tabs.setMaximumHeight(self.height - 50)

        # tab 1
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, 'Main')

        self.contrunchkbx = QCheckBox('Stay primed')
        self.contrunchkbx.setToolTip('Stay primed')
        self.contrunchkbx.setEnabled(False)

        self.centerbox = QCheckBox('Center on click')
        self.centerbox.setToolTip('Push, then click on image')
        self.centerbox.setEnabled(False)

        self.startbtn = QPushButton('Prime for experiment')
        self.startbtn.setToolTip('Prime for experiment')
        self.startbtn.setEnabled(False)

        self.stopbtn = QPushButton('Stop')
        self.stopbtn.setToolTip('Stop')
        self.stopbtn.setEnabled(False)

        self.rdb = QGUI.RadioButtons(('Zhuang', 'PID'))

        self.plot = QGUI.PlotCanvas()
        self.eplot = QGUI.SubPlot(self.plot, 611)
        self.eplot.append_plot('middle', ':', 'gray')
        self.eplot.append_plot('top', ':k')
        self.eplot.append_plot('bottom', ':k')
        self.eplot.ax.set_ylabel('Ellipticity')
        self.eplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.iplot = QGUI.SubPlot(self.plot, 612)
        self.iplot.ax.set_ylabel('Intensity')
        self.iplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.splot = QGUI.SubPlot(self.plot, 613)
        self.splot.ax.set_ylabel('Sigma (nm)')
        self.splot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.rplot = QGUI.SubPlot(self.plot, 614)
        self.rplot.ax.set_ylabel('R squared')
        self.rplot.append_plot('bottom', ':k')
        self.rplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.pplot = QGUI.SubPlot(self.plot, 615)
        self.pplot.append_plot('fb', '-k')
        self.pplot.ax.set_xlabel('Time (frames)')
        self.pplot.ax.set_ylabel('Piezo pos (µm)')

        self.xyplot = QGUI.SubPlot(self.plot, (6, 3, 16), '.r')
        self.xyplot.ax.set_xlabel('x (nm)')
        self.xyplot.ax.set_ylabel('y (nm)')
        self.xyplot.ax.invert_yaxis()
        self.xyplot.ax.set_aspect('equal', adjustable='datalim')

        self.xzplot = QGUI.SubPlot(self.plot, (6, 3, 17), '.r')
        self.xzplot.ax.set_xlabel('x (nm)')
        self.xzplot.ax.set_ylabel('z (nm)')
        self.xzplot.ax.set_aspect('equal', adjustable='datalim')

        self.yzplot = QGUI.SubPlot(self.plot, (6, 3, 18), '.r')
        self.yzplot.ax.set_xlabel('y (nm)')
        self.yzplot.ax.set_ylabel('z (nm)')
        self.yzplot.ax.set_aspect('equal', adjustable='datalim')

        self.buttons = QHBoxLayout()
        self.buttons.addWidget(self.contrunchkbx)
        self.buttons.addWidget(self.centerbox)
        self.buttons.addWidget(self.startbtn)
        self.buttons.addWidget(self.stopbtn)
        self.buttons.addWidget(self.rdb)

        self.tab1.layout = QVBoxLayout(self.tab1)
        self.tab1.layout.addLayout(self.buttons)
        self.tab1.layout.addWidget(self.plot)

        # tab 2
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, 'Configuration')

        self.grid = QGridLayout()
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(2, 3)

        r = 0
        self.grid.addWidget(QLabel('Feedback channel:'), r, 0)
        self.rdbch = QGUI.CheckBoxes(['{}'.format(i) for i in range(10)])
        for i, e in enumerate((664, 510, 583, 427)):
            self.rdbch.setTextBoxValue(i, e)
        self.grid.addWidget(self.rdbch, r, 1)

        r += 1
        self.grid.addWidget(QLabel('Feedback mode:'), r, 0)
        self.feedbackModeDrp = QComboBox()
        self.feedbackModeDrp.addItems(['Average', 'Alternate'])
        self.grid.addWidget(self.feedbackModeDrp, r, 1)

        r += 1
        self.cyllensdrp = []
        self.grid.addWidget(QLabel('Cylindrical lens back:'), r, 0)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None', 'A', 'B'])
        self.grid.addWidget(self.cyllensdrp[-1], r, 1)

        r += 1
        self.grid.addWidget(QLabel('Cylindrical lens front:'), r, 0)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None', 'A', 'B'])
        self.cyllensdrp[-1].setCurrentIndex(1)
        self.grid.addWidget(self.cyllensdrp[-1], r, 1)

        r += 1
        self.grid.addWidget(QLabel('Duolink filterset:'), r, 0)
        self.dlfs = QComboBox()
        self.dlfs.addItems(['488/_561_/640 & 488/_640_', '_561_/640 & empty'])
        self.grid.addWidget(self.dlfs, r, 1)

        r += 1
        self.grid.addWidget(QLabel('Duolink filter:'), r, 0)
        self.chdlf = QGUI.RadioButtons(('1', '2'))
        self.grid.addWidget(self.chdlf, r, 1)
        self.dlf = QLabel()
        self.grid.addWidget(self.dlf, r, 2)

        r += 1
        self.grid.addWidget(QLabel('XY feedback:'), r, 0)
        # self.xyrdb = QGUI.RadioButtons(('None', 'Move ROI', 'Move stage'))  # Stage not accurate enough
        self.xyrdb = QGUI.RadioButtons(('None', 'Move ROI'))
        self.grid.addWidget(self.xyrdb, r, 1)

        r += 1
        self.grid.addWidget(QLabel('ROI size:'), r, 0)
        self.ROISizefld = QLineEdit()
        self.grid.addWidget(self.ROISizefld, r, 1)
        self.grid.addWidget(QLabel('px'), r, 2)

        r += 1
        self.grid.addWidget(QLabel('ROI position:'), r, 0)
        self.ROIPosfld = QLineEdit()
        self.grid.addWidget(self.ROIPosfld, r, 1)
        self.grid.addWidget(QLabel('px'), r, 2)

        r += 1
        self.grid.addWidget(QLabel('Max stepsize x, y:'), r, 0)
        self.maxStepxyfld = QLineEdit()
        self.grid.addWidget(self.maxStepxyfld, r, 1)
        self.grid.addWidget(QLabel('px'), r, 2)

        r += 1
        self.grid.addWidget(QLabel('Max stepsize z:'), r, 0)
        self.maxStepfld = QLineEdit()
        self.grid.addWidget(self.maxStepfld, r, 1)
        self.grid.addWidget(QLabel('um'), r, 2)

        r += 1
        self.calibbtn = QPushButton('Calibrate with beads')
        self.calibbtn.setToolTip('Calibrate with beads')
        self.grid.addWidget(self.calibbtn, r, 1)
        self.calibbtn.setEnabled(False)

        self.tab2.setLayout(self.grid)

        # tab 3
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, 'Map')

        self.map = QGUI.SubPatchPlot(QGUI.PlotCanvas(), color=(0.6, 1, 0.8))
        self.map.ax.invert_xaxis()
        self.map.ax.invert_yaxis()
        self.map.ax.set_xlabel('x')
        self.map.ax.set_ylabel('y')
        self.map.ax.set_aspect('equal', adjustable='datalim')

        self.maprstbtn = QPushButton('Reset')

        self.tab3.layout = QVBoxLayout(self.tab3)
        self.tab3.layout.addWidget(self.map.canvas)
        self.tab3.layout.addWidget(self.maprstbtn)

        self.layout.addWidget(self.tabs)

        # menus
        mainMenu = QMenuBar(MainWindow)
        confMenu = mainMenu.addMenu('&Configuration')

        self.openAction = QAction('&Open', self)
        self.openAction.setShortcut('Ctrl+O')
        self.openAction.setStatusTip('Open configuration')

        self.saveAction = QAction('&Save', self)
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.setStatusTip('Save configuration')

        self.saveasAction = QAction('Save &As', self)
        self.saveasAction.setShortcut('Ctrl+Shift+S')
        self.saveasAction.setStatusTip('Save configuration as')

        confMenu.addAction(self.openAction)
        confMenu.addAction(self.saveAction)
        confMenu.addAction(self.saveasAction)

        warpMenu = mainMenu.addMenu('&Transform')

        self.warpAction = QAction('&Warp', self)
        self.warpAction.setShortcut('Ctrl+W')
        self.warpAction.setStatusTip('Save a copy of a file where the warp is corrected')

        warpMenu.addAction(self.warpAction)

        MainWindow.setCentralWidget(self.central_widget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class App(QMainWindow, UiMainWindow):
    def __init__(self):
        super().__init__()

        self.Queue = Queue()
        self.Manager = Manager()
        self.NS = self.Manager.Namespace()
        self.NS.q = self.Manager.dict()
        self.NS.feedback_mode = 'zhuang'
        self.NS.cyllens = self.Manager.list()
        self.NS.max_step = 1
        self.NS.max_step_xy = 5
        self.NS.theta = self.Manager.dict()
        self.NS.sigma = self.Manager.dict()
        self.NS.limits = self.Manager.dict()
        self.NS.mult_channel_mode = 0  # Average, 1: Alternate
        self.NS.channels = self.Manager.list()
        self.NS.feedback_mode_xy = 0
        self.NS.quit = False
        self.NS.stop = False
        self.NS.run = False

        self.stop = False  # Stop (waiting for) experiment
        self.quit = False  # Quit program

        self.zen = zen()
        self.conf_filename = os.path.join(os.path.dirname(__file__), 'conf.yml')

        self.calibrating = False

        self.ellipse = {}
        self.rectangle = None
        self.MagStr = self.zen.MagStr
        self.DLFilter = self.zen.DLFilter

        self.setupUI(self)

        self.contrunchkbx.toggled.connect(self.stayprimed)
        self.centerbox.toggled.connect(self.toggle_center_box)
        self.startbtn.clicked.connect(self.prime)
        self.stopbtn.clicked.connect(self.setstop)
        self.feedbackModeDrp.currentIndexChanged.connect(self.change_mult_channel_mode)
        for cyllensdrp in self.cyllensdrp:
            cyllensdrp.currentIndexChanged.connect(self.change_cyllens)
        self.dlfs.currentIndexChanged.connect(self.change_duolink_block)
        self.ROISizefld.textChanged.connect(self.change_roi_size)
        self.ROIPosfld.textChanged.connect(self.change_roi_pos)
        self.maxStepxyfld.textChanged.connect(self.change_max_step_xy)
        self.maxStepfld.textChanged.connect(self.change_max_step)
        self.calibbtn.clicked.connect(self.calibrate)
        self.maprstbtn.clicked.connect(self.resetmap)
        self.rdbch.connect(self.change_channel, self.change_wavelength)
        self.rdb.connect(self.change_feedback_mode)
        self.xyrdb.connect(self.change_xy_feedback_mode)
        self.chdlf.connect(self.change_duolink_filter)
        self.openAction.triggered.connect(self.confopen)
        self.saveAction.triggered.connect(partial(self.confsave, self.conf_filename))
        self.saveasAction.triggered.connect(self.confsave)
        self.warpAction.triggered.connect(self.warp)

        self.show()

        self.fblprocess = Process(target=feedbackloop, args=(self.Queue, self.NS))
        self.fblprocess.start()
        self.guithread = None

        self.zen.wait(self, self.zen_ready)

    def closeEvent(self, *args, **kwargs):
        self.setquit()

    def zen_ready(self, _):
        self.confopen(self.conf_filename)
        self.chdlf.changeState(self.zen.DLFilter)
        self.change_cyllens()
        self.change_color()
        self.centerbox.setEnabled(True)
        self.change_wavelengths()
        if len(self.NS.channels):
            self.contrunchkbx.setEnabled(True)
            self.startbtn.setEnabled(True)
        self.events = Events(self)

    def change_wavelengths(self):
        for channel, value in enumerate(self.rdbch.textBoxValues):
            self.change_wavelength(channel, value)

    def change_roi_size(self, val):
        self.NS.roi_size = float(val)

    def change_roi_pos(self, val):
        try:
            self.NS.roi_pos = [float(i) for i in re.findall('([-?\d\.]+)[^-\d\.]+([-?\d\.]+)', val)[0]]
        except Exception:
            pass

    def change_max_step_xy(self, val):
        self.NS.max_step_xy = float(val)

    def change_color(self):
        for channel in self.NS.channels:
            color = self.zen.ChannelColorsRGB[channel]
            for plot in (self.eplot, self.iplot, self.splot, self.rplot, self.pplot, self.xyplot, self.xzplot,
                         self.yzplot):
                if channel in plot:
                    plot.plt[channel].set_color(color)
            for tb in ('top', 'bottom'):
                cn = '{}{}'.format(tb, channel)
                if cn in self.splot:
                    self.splot.plt[cn].set_color(color)
        camera = [self.zen.CameraFromChannelName(i) for i in self.zen.ChannelNames]
        enabled = [False if self.NS.cyllens[c] == 'None' else True for c in camera]
        self.rdbch.changeOptions(self.zen.ChannelNames, self.zen.ChannelColorsHex, enabled)

    def change_wavelength(self, channel, val):
        wavelength = float(val)
        self.NS.sigma[channel] = wavelength / 2 / self.zen.ObjectiveNA / self.zen.pxsize
        fwhm = self.NS.sigma[channel] * 2 * np.sqrt(2 * np.log(2))
        self.NS.limits[channel] = self.Manager.list([[-np.inf, np.inf]] * 8)
        self.NS.limits[channel][2] = [fwhm / 2, fwhm * 2]  # fraction of fwhm
        self.NS.limits[channel][5] = [0.7, 1.3]  # ellipticity
        self.NS.limits[channel][7] = [0, np.inf]  # R2

    def change_cyllens(self):
        self.NS.cyllens = [i.currentText() for i in self.cyllensdrp]
        self.change_color()

    def calibrate(self, *args, **kwargs):
        if len(self.NS.channels) == 1:
            self.calibrating = True
            self.calibbtn.setEnabled(False)
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            file, _ = QFileDialog.getOpenFileName(self, "Beads for calibration", "", "CZI Files (*.czi);;All Files (*)",
                                                  options=options)
            if file:

                self.calibz_thread = qthread(cyl.calibz, self.calibrated, file, self.rdbch.textBoxValues,
                                             self.NS.channels[0], self.NS.cyllens, self.calibrate_progress)
                self.calibrate_progress(0)
            else:
                self.calibrating = False
                self.calibbtn.setEnabled(True)

    def calibrate_progress(self, progress):
        self.calibbtn.setText(f'Calibrating: {progress:.0f}%')

    def calibrated(self,channel, MagStr, theta, q):
        self.conf[self.get_cyllens(channel) + MagStr]['theta'] = float(theta)
        self.conf[self.get_cyllens(channel) + MagStr]['q'] = q.tolist()
        self.NS.theta[self.get_cm_str(channel)] = float(theta)
        self.NS.q[self.get_cyllens(channel) + MagStr] = q.tolist()
        self.calibrating = False
        self.calibbtn.setText('Calibrate with beads')
        self.calibbtn.setEnabled(True)

    def warp(self):
        options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        files, _ = QFileDialog.getOpenFileNames(self, "Image files", "", "CZI Files (*.czi);;All Files (*)",
                                              options=options)
        def warpfiles(files):
            for file in files:
                if os.path.isfile(file):
                    warp(file)
        self.warpthread = qthread(warpfiles, None, files)

    def resetmap(self):
        self.map.remove_data()

    def confsave(self, f):
        if not os.path.isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getSaveFileName(self, "Save config file", "", "YAML Files (*.yml);;All Files (*)",
                                               options=options)
        if f:
            self.conf_filename = f
            self.conf['maxStep'] = self.NS.max_step
            with open(f, 'w') as h:
                yaml.dump(self.conf, h, default_flow_style=None)

    def confopen(self, f):
        if not os.path.isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getOpenFileName(self, "Open config file", "", "YAML Files (*.yml);;All Files (*)",
                                               options=options)
        if f:
            with open(f, 'r') as h:
                self.conf = yamlload(h)
            self.conf_filename = f
            self.confload()

    def confload(self):
        if 'cyllenses' in self.conf:
            values = ['None']
            if isinstance(self.conf['cyllenses'], (list, tuple)):
                values.extend(self.conf['cyllenses'])
            else:
                values.extend(re.split('\s?[,;]\s?', self.conf['cyllenses']))
            for drp in self.cyllensdrp:
                if values != [drp.itemText(i) for i in range(drp.count())]:
                    drp.blockSignals(True)
                    drp.clear()
                    drp.addItems(values)
                    drp.blockSignals(False)
        q = {}
        theta = {}
        for key, value in self.conf.items():
            if isinstance(value, dict) and 'q' in value and 'theta' in value:
                self.NS.q[key] = value['q']
                self.NS.theta[key] = value['theta']
                q[key] = value['q']
                theta[key] = value['theta']

        self.NS.max_step = self.conf.get('maxStep', 0.1)
        self.maxStepfld.setText(f'{self.NS.max_step}')
        self.NS.max_step_xy = self.conf.get('maxStepxy', 5)
        self.maxStepxyfld.setText(f'{self.NS.max_step_xy}')
        self.NS.roi_size = self.conf.get('ROISize', 48)
        self.ROISizefld.setText(f'{self.NS.roi_size}')
        self.NS.roi_pos = self.conf.get('ROIPOS', [0, 0])
        self.ROIPosfld.setText(f'{self.NS.roi_pos}')
        self.NS.gain = self.conf.get('gain', 5e-3)
        self.NS.fast_mode = self.conf.get('fastMode', False)
        if 'style' in self.conf:
            self.setStyleSheet(self.conf['style'])

    def change_duolink_filter(self, val):
        if val == '1':
            self.zen.DLFilter = 0
        else:
            self.zen.DLFilter = 1
        self.change_duolink_block()

    def change_max_step(self, val):
        self.NS.max_step = float(val)

    def change_channel(self, val):
        if len(val):
            self.NS.channels = [int(v) for v in val]
            self.confopen(self.conf_filename)
            self.contrunchkbx.setEnabled(True)
            self.startbtn.setEnabled(True)
        else:
            self.contrunchkbx.setEnabled(False)
            self.startbtn.setEnabled(False)
        if not self.calibrating:
            self.calibbtn.setEnabled(len(self.NS.channels) == 1)

    def change_duolink_block(self, *args):
        self.dlf.setText(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])

    def change_feedback_mode(self, val):
        self.NS.feedback_mode = val.lower()

    def change_xy_feedback_mode(self, val):
        self.NS.feedback_mode_xy = self.xyrdb.txt.index(val)

    def change_mult_channel_mode(self, idx):
        self.NS.mult_channel_mode = idx

    def toggle_center_box(self):
        self.zen.EnableEvent('LeftButtonDown')

    def get_cyllens(self, channel):
        return self.NS.cyllens[self.zen.CameraFromChannelName(self.zen.ChannelNames[channel])]

    def get_cm_str(self, channel):
        return self.get_cyllens(channel) + self.zen.MagStr

    def prime(self):
        if self.guithread is None or not self.guithread.is_alive():
            self.NS.stop = False
            self.guithread = qthread(target=self.run)

    def stayprimed(self):
        if self.contrunchkbx.isChecked():
            self.prime()

    def run(self, *args, **kwargs):
        np.seterr(all='ignore');
        sleepTime = 0.02  # update interval

        self.startbtn.setEnabled(False)
        self.stopbtn.setEnabled(True)
        self.zen.RemoveDrawings()
        self.rectangle = self.zen.DrawRectangle(*functions.cliprect(self.zen.FrameSize, *self.NS.roi_pos,
                                                                    self.NS.roi_size,
                                                                    self.NS.roi_size), index=self.rectangle)
        roi_pos = self.NS.roi_pos
        while True:
            self.startbtn.setText('Wait for experiment to start')

            # First wait for the experiment to start:
            while (not self.zen.ExperimentRunning) and (not self.stop) and (not self.quit):
                self.rectangle = self.zen.DrawRectangle(*functions.cliprect(self.zen.FrameSize, *self.NS.roi_pos,
                                                                            self.NS.roi_size,
                                                                            self.NS.roi_size), index=self.rectangle)
                sleep(sleepTime)

            for channel in self.NS.channels:
                for tb in ('top', 'bottom'):
                    cn = '{}{}'.format(tb, channel)
                    if not cn in self.splot:
                        self.splot.append_plot(cn, ':', self.zen.ChannelColorsRGB[channel])

                for plot in (self.eplot, self.iplot, self.splot, self.rplot, self.pplot):
                    if not channel in plot:
                        plot.append_plot(channel, '-', self.zen.ChannelColorsRGB[channel])

                for plot in (self.xyplot, self.xzplot, self.yzplot):
                    if not channel in plot:
                        plot.append_plot(channel, '.', self.zen.ChannelColorsRGB[channel])

            # Experiment has started:
            self.NS.run = True
            self.rectangle = self.zen.DrawRectangle(*functions.cliprect(self.zen.FrameSize, *self.NS.roi_pos,
                                                                        self.NS.roi_size, self.NS.roi_size),
                                                    index=self.rectangle)

            cfilename = self.zen.FileName
            cfexists = os.path.isfile(cfilename) #whether ZEN already made the czi file (streaming)
            if cfexists:
                pfilename = os.path.splitext(cfilename)[0]+'.pzl'
            else:
                pfilename = os.path.splitext(self.conf.get('tmpPzlFile', 'd:\tmp\tmp.pzl'))[0] + \
                            datetime.now().strftime('_%Y%m%d-%H%M%S.pzl')
                if os.path.isfile(pfilename):
                    os.remove(pfilename)
            if not cfexists or (cfexists and not os.path.isfile(pfilename)):
                conf = {'version': 3, 'feedback_mode': self.NS.feedback_mode, 'FeedbackChannels': self.NS.channels,
                        'CylLens': self.NS.cyllens, 'mult_channel_mode': self.NS.mult_channel_mode,
                        'feedback_mode_xy': self.NS.feedback_mode_xy, 'DLFilterSet': self.dlfs.currentText(),
                        'DLFilterChannel': self.zen.DLFilter, 'maxStepxy': self.NS.max_step_xy,
                        'maxStep': self.NS.max_step, 'ROISize': self.NS.roi_size, 'ROIPos': self.NS.roi_pos,
                        'Columns': ['channel', 'frame', 'piezoPos', 'focusPos',
                                    'x', 'y', 'fwhm', 'i', 'o', 'e', 'time']}

                for key in self.NS.q.keys():
                    conf[key] = {'q': self.NS.q[key], 'theta': self.NS.theta[key]}

                with open(pfilename, 'w') as file:
                    yaml.dump(conf, file, default_flow_style=None)
                    file.write('p:\n')

                    self.plot.remove_data()

                    SizeX, SizeY = self.zen.FrameSize
                    z0 = None

                    self.startbtn.setText('Experiment started')

                    while (self.zen.ExperimentRunning or (not self.Queue.empty())) and (not self.stop) \
                            and (not self.quit):
                        pxsize = self.zen.pxsize

                        #Wait until feedbackloop analysed a new frame
                        while self.zen.ExperimentRunning and (not self.stop) and (not self.quit) and self.Queue.empty():
                            sleep(sleepTime)
                        if not self.Queue.empty():
                            Q = []
                            for i in range(20):
                                if not self.Queue.empty():
                                    Q.append(self.Queue.get())
                                    if z0 is None:
                                        z0 = Q[0][5]
                                else:
                                    break

                            for c, t, g, a, p, f, s in Q:
                                file.write('- [{},{},{},{},'.format(c, t, p, f))
                                file.write('{},{},{},{},{},{},'.format(*a))
                                file.write('{}]\n'.format(s))

                            channels = [q[0] for q in Q]
                            for channel in set(channels):
                                idx = [i for i, e in enumerate(channels) if e==channel]
                                Time, Fitted, a, PiezoPos, FocusPos = [[Q[i][j] for i in idx] for j in range(1, 6)]

                                a = np.array(a)
                                if a.ndim>1 and a.shape[1]:
                                    a[:,2][a[:,2] < self.NS.limits[channel][2][0]/2] = np.nan
                                    a[:,2][a[:,2] > self.NS.limits[channel][2][1]*2] = np.nan
                                    a[:,5][a[:,5] < self.NS.limits[channel][5][0]/2] = np.nan
                                    a[:,5][a[:,5] > self.NS.limits[channel][5][1]*2] = np.nan
                                    a[:,7][a[:,7] < self.NS.limits[channel][7][0]*2] = np.nan
                                    ridx = np.isnan(a[:,3]) | np.isnan(a[:,4])
                                    a[ridx,:] = np.nan

                                    zfit = np.array([-cyl.findz(e, self.NS.q[self.get_cm_str(channel)]) if f else np.nan
                                                     for e, f in zip(a[:, 5], Fitted)])
                                    z = 1000*(zfit + FocusPos - z0)

                                    self.eplot.range_data(Time, a[:,5], handle=channel)
                                    self.iplot.range_data(Time, a[:,3], handle=channel)
                                    self.splot.range_data(Time, a[:,2] / 2 / np.sqrt(2 * np.log(2)) * pxsize, handle=channel)
                                    self.splot.range_data(Time, [self.NS.limits[channel][2][0]/2/np.sqrt(2*np.log(2))*pxsize] * len(Time), handle=f'bottom{channel}')
                                    self.splot.range_data(Time, [self.NS.limits[channel][2][1]/2/np.sqrt(2*np.log(2))*pxsize] * len(Time), handle=f'top{channel}')
                                    self.rplot.range_data(Time, a[:,7], handle=channel)
                                    self.pplot.range_data(Time, zfit + FocusPos - z0, handle=channel)
                                    self.xyplot.append_data((a[:,0] - self.NS.roi_size / 2) * pxsize,
                                                            (a[:,1] - self.NS.roi_size / 2) * pxsize, channel)
                                    self.xzplot.append_data((a[:,0] - self.NS.roi_size / 2) * pxsize, z, channel)
                                    self.yzplot.append_data((a[:,1] - self.NS.roi_size / 2) * pxsize, z, channel)
                                    if channel==channels[0]: #Draw these only once
                                        self.eplot.range_data(Time, [1] * len(Time), handle='middle')
                                        self.eplot.range_data(Time, [self.NS.limits[channel][5][0]] * len(Time), handle='bottom')
                                        self.eplot.range_data(Time, [self.NS.limits[channel][5][1]] * len(Time), handle='top')
                                        self.rplot.range_data(Time, [self.NS.limits[channel][7][0]] * len(Time), handle='bottom')
                                        self.pplot.range_data(Time, np.array(FocusPos) - z0, handle='fb')
                                    self.plot.draw()

                                    if Fitted[-1]:
                                        X = float(a[-1,0] + (SizeX - self.NS.roi_size) / 2 + self.NS.roi_pos[0])
                                        Y = float(a[-1,1] + (SizeY - self.NS.roi_size) / 2 + self.NS.roi_pos[1])
                                        R = float(a[-1,2])
                                        E = float(a[-1,5])
                                        self.ellipse[channel] = self.zen.DrawEllipse(X, Y, R, E,
                                                 self.NS.theta[self.get_cm_str(channel)],
                                                 self.zen.ChannelColorsInt[channel], index=self.ellipse.get(channel))
                                        self.rectangle = self.zen.DrawRectangle(*functions.cliprect(self.zen.FrameSize,
                                                                     *self.NS.roi_pos, self.NS.roi_size, self.NS.roi_size),
                                                                                index=self.rectangle)
                                    else:
                                        self.zen.RemoveDrawing(self.ellipse.get(channel))

                #After the experiment:
                for ellipse in self.ellipse.values():
                    self.zen.RemoveDrawing(ellipse)
                if not cfexists:
                    for _ in range(5):
                        cfilename = functions.last_czi_file(self.conf.get('dataDir', 'd:\data'))
                        npfilename = os.path.splitext(cfilename)[0] + '.pzl'
                        if cfilename and not os.path.isfile(npfilename):
                            copyfile(pfilename, npfilename)
                            break
                        sleep(0.25)
                self.NS.roi_pos = roi_pos
            else:
                sleep(sleepTime)
            if not self.contrunchkbx.isChecked():
                break

        self.zen.RemoveDrawing(self.rectangle)
        self.stop = False
        self.startbtn.setText('Prime for experiment')
        self.stopbtn.setEnabled(False)
        self.startbtn.setEnabled(True)
        self.NS.run = False

    def setstop(self):
        # Stop being primed for an experiment
        self.stopbtn.setEnabled(False)
        self.contrunchkbx.setChecked(False)
        self.stop = True
        self.NS.stop = True

    def setquit(self):
        # Quit the whole program
        self.setstop()
        self.quit = True
        self.NS.quit = True
        self.fblprocess.join(5)
        self.fblprocess.terminate()


def main():
    freeze_support()  # to enable pyinstaller to work with multiprocessing
    app = QApplication([])
    window = App()
    exit(app.exec())

if __name__ == '__main__':
    main()
