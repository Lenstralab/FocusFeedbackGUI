from sys import exit
from os.path import isfile, splitext
from os import remove
from shutil import copyfile
from time import sleep, time
from datetime import datetime
from functools import partial
from multiprocessing import Process, Queue, Manager, freeze_support
from threading import Thread
from parfor import parpool
from PyQt5.QtWidgets import *

import numpy as np
from fbs_runtime.application_context.PyQt5 import ApplicationContext

import QGUI
import cylinderlens as cyl
import functions, config
from utilities import thread, close_threads
from events import events
from pid import pid
from zen import zen

np.seterr(all='ignore');

def firstargonly(fun):
    """ decorator that only passes the first argument to a function
    """
    return lambda *args: fun(args[0])

def feedbackloop(Queue, NameSpace):
    # this is run in a separate process
    np.seterr(all='ignore');
    Z = zen()
    _ = Z.PiezoPos
    while not NameSpace.quit:
        if NameSpace.run:
            mode = NameSpace.mode
            Size = NameSpace.Size
            Pos = NameSpace.Pos
            gain = NameSpace.gain
            channels = NameSpace.channels
            q = NameSpace.q
            theta = NameSpace.theta
            maxStep = NameSpace.maxStep
            fastMode = NameSpace.fastMode
            limits = NameSpace.limits
            feedbackMode = NameSpace.feedbackMode
            sigma = NameSpace.sigma
            TimeInterval = Z.TimeInterval
            G = Z.PiezoPos
            FS = Z.FrameSize
            TimeMem = 0
            detected = functions.truncated_list(5, (True,)*5)
            zmem = {}

            if mode == 'pid':
                P = pid(0, G, maxStep, TimeInterval, gain)

            def fitter(c, theta, sigma, fastMode):
                channel, Frame = c
                return functions.fg(Frame, theta[channel], sigma[channel], fastMode)

            with parpool(fitter, (theta, sigma, fastMode), nP=len(channels)) as pool:
                while Z.ExperimentRunning and (not NameSpace.stop):
                    ellipticity = {}
                    PiezoPos = Z.PiezoPos
                    FocusPos = Z.GetCurrentZ

                    Times = {}
                    for channel in channels:
                        STime = time()
                        Frame, Time = Z.GetFrame(channel, *functions.cliprect(FS, *Pos, Size, Size))
                        pool[channel] = (channel, Frame)
                        Times[channel] = (STime, Time)

                    for channel in channels:
                        STime, Time = Times[channel]
                        a = pool[channel]
                        if Time < TimeMem:
                            TTime = TimeMem + 1
                        else:
                            TTime = Time

                        #try to determine when something is detected by using a simple filter on R2, elipticity and psf width
                        if not any(np.isnan(a)) and all([l[0]<n<l[1] for n, l in zip(a, limits[channel])]):
                            if sum(detected)/len(detected) > 0.35:
                                Fitted = True
                                ellipticity[channel] = a[5]
                            else:
                                Fitted = False
                            detected.append(True)
                        else:
                            Fitted = False
                            detected.append(False)
                        Queue.put((channel, TTime, Fitted, a[:8], PiezoPos, FocusPos, STime))

                    # Update the piezo position:
                    if mode == 'pid':
                        if np.any(np.isfinite(list(ellipticity.values()))):
                            e = np.nanmean(list(ellipticity.values()))
                            F = -np.log(e)
                            if np.abs(F) > 1:
                                F = 0
                            Pz = P(F)
                            Z.PiezoPos = Pz

                            if Pz > (G + 5):
                                P = pid(0, G, maxStep, TimeInterval, gain)
                    else:
                        dz = {channel: np.clip(cyl.findz(e, q[channel]), -maxStep, maxStep) for channel, e in ellipticity.items()}
                        cur_channel_idx = TTime % len(channels)
                        next_channel_idx = (TTime + 1) % len(channels)
                        if np.any(np.isfinite(list(dz.values()))):
                            if feedbackMode == 0: #Average
                                dz = np.nanmean(list(dz.values()))
                                if not np.isnan(dz):
                                    # Z.MovePiezoRel(-dz)
                                    Z.PiezoPos -= dz
                            else: #Alternate: save focus in current channel, apply focus to piezo for next channel
                                if np.isfinite(dz[channels[cur_channel_idx]]):
                                    zmem[channels[cur_channel_idx]] = PiezoPos + dz[channels[cur_channel_idx]]
                                elif not channels[cur_channel_idx] in zmem:
                                    zmem[channels[cur_channel_idx]] = PiezoPos
                                if channels[next_channel_idx] in zmem:
                                    Z.PiezoPos = zmem[channels[next_channel_idx]]

                    # Wait for next frame:
                    while ((Z.GetTime - 1) == Time) and Z.ExperimentRunning and (not NameSpace.stop) and (not NameSpace.quit):
                        sleep(TimeInterval / 4)
                    if Time < TimeMem:
                        break

                    TimeMem = Time
                NameSpace.run = False
        else:
            sleep(0.01)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Cylinder lens feedback GUI'
        self.width = 640
        self.height = 1024
        self.right = 1920
        self.top = 32
        self.color = '#454D62'
        self.textColor = '#FFFFFF'

        self.stop = False # Stop (waiting for) experiment
        self.quit = False # Quit program
        self.conf = config.conf()

        self.zen = zen()

        self.q = {}
        self.maxStep = 1
        self.theta = {}

        self.feedbackMode = 0  # Average, 1: Alternate
        self.channels = [0]

        self.ellipse = {}
        self.rectangle = None
        self.MagStr = self.zen.MagStr
        self.DLFilter = self.zen.DLFilter

        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.setGeometry(self.right, self.top, self.width, self.height)

        self.central_widget = QWidget()
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget(self.central_widget)
        self.settab1()
        self.settab2()
        self.settab3()
        self.menus()
        self.layout.addWidget(self.tabs)

        self.Queue = Queue()
        self.NameSpace = Manager().Namespace()
        self.NameSpace.quit = False
        self.NameSpace.stop = False
        self.NameSpace.run = False
        self.fblprocess = Process(target=feedbackloop, args=(self.Queue, self.NameSpace))
        self.fblprocess.start()
        self.guithread = None

        events(self)

        self.setCentralWidget(self.central_widget)
        self.show()
        self.wait_for_zen()

    @thread
    def wait_for_zen(self):
        while not self.zen.ready and not self.stop and not self.quit:
            sleep(0.1)
        self.confopen(self.conf.filename)
        self.changeColor()
        self.contrunchkbx.setEnabled(True)
        self.centerbox.setEnabled(True)
        self.startbtn.setEnabled(True)

    def menus(self):
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&Configuration')

        openAction = QAction('&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open configuration')
        openAction.triggered.connect(self.confopen)

        saveAction = QAction('&Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save configuration')
        saveAction.triggered.connect(partial(self.confsave, self.conf.filename))

        saveasAction = QAction('Save &As', self)
        saveasAction.setShortcut('Ctrl+Shift+S')
        saveasAction.setStatusTip('Save configuration as')
        saveasAction.triggered.connect(self.confsave)

        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveasAction)

    def settab1(self):
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, 'Main')

        self.contrunchkbx = QCheckBox('Stay primed')
        self.contrunchkbx.setToolTip('Stay primed')
        self.contrunchkbx.toggled.connect(self.stayprimed)
        self.contrunchkbx.setEnabled(False)

        self.centerbox = QCheckBox('Center on click')
        self.centerbox.setToolTip('Push, then click on image')
        self.centerbox.toggled.connect(self.tglcenterbox)
        self.centerbox.setEnabled(False)

        self.startbtn = QPushButton('Prime for experiment')
        self.startbtn.setToolTip('Prime for experiment')
        self.startbtn.clicked.connect(self.prime)
        self.startbtn.setEnabled(False)

        self.stopbtn = QPushButton('Stop')
        self.stopbtn.setToolTip('Stop')
        self.stopbtn.clicked.connect(self.setstop)
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

    def settab2(self):
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, 'Configuration')

        self.grid = QGridLayout()
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(2, 3)

        r = 0
        self.grid.addWidget(QLabel('Feedback channel:'), r, 0)
        self.rdbch = QGUI.CheckBoxes(['{}'.format(i) for i in range(10)], init_state=self.channels, callback=self.changechannel)
        for i, e in enumerate((664, 510, 600)):
            self.rdbch.setTextBoxValue(i, e)
        self.grid.addWidget(self.rdbch, r, 1)

        r += 1
        self.grid.addWidget(QLabel('Feedback mode:'), r, 0)
        self.feedbackModeDrp = QComboBox()
        self.feedbackModeDrp.addItems(['Average', 'Alternate'])
        self.feedbackModeDrp.currentIndexChanged.connect(self.changeFeedbackMode)
        self.grid.addWidget(self.feedbackModeDrp, r, 1)

        r += 1
        self.cyllensdrp = []
        self.grid.addWidget(QLabel('Cylindrical lens back:'), r, 0)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None','A','B'])
        self.cyllensdrp[-1].currentIndexChanged.connect(self.changeCylLens)
        self.grid.addWidget(self.cyllensdrp[-1], r, 1)

        r += 1
        self.grid.addWidget(QLabel('Cylindrical lens front:'), r, 0)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None','A', 'B'])
        self.cyllensdrp[-1].setCurrentIndex(1)
        self.cyllensdrp[-1].currentIndexChanged.connect(self.changeCylLens)
        self.grid.addWidget(self.cyllensdrp[-1], r, 1)

        r += 1
        self.grid.addWidget(QLabel('Duolink filterset:'), r, 0)
        self.dlfs = QComboBox()
        self.dlfs.addItems(['488/_561_/640 & 488/_640_', '_561_/640 & empty'])
        self.dlfs.currentIndexChanged.connect(self.changeDL)
        self.grid.addWidget(self.dlfs, r, 1)

        r += 1
        self.grid.addWidget(QLabel('Duolink filter:'), r, 0)
        self.chdlf = QGUI.RadioButtons(('1', '2'), init_state=self.zen.DLFilter, callback=self.changeDLF)
        self.grid.addWidget(self.chdlf, r, 1)
        self.dlf = QLabel(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])
        self.grid.addWidget(self.dlf, r, 2)

        r += 1
        self.grid.addWidget(QLabel('Max stepsize:'), r, 0)
        self.maxStepfld = QLineEdit()
        self.maxStepfld.textChanged.connect(self.changemaxStep)
        self.grid.addWidget(self.maxStepfld, r, 1)
        self.grid.addWidget(QLabel('um'), r, 2)

        r += 1
        self.calibbtn = QPushButton('Calibrate with beads')
        self.calibbtn.setToolTip('Calibrate with beads')
        self.calibbtn.clicked.connect(self.calibrate)
        self.grid.addWidget(self.calibbtn, r, 1)

        self.tab2.setLayout(self.grid)

    def settab3(self):
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, 'Map')

        self.map = QGUI.SubPatchPlot(QGUI.PlotCanvas(), color=(0.6, 1, 0.8))
        self.map.ax.invert_xaxis()
        self.map.ax.invert_yaxis()
        self.map.ax.set_xlabel('x')
        self.map.ax.set_ylabel('y')
        self.map.ax.set_aspect('equal', adjustable='datalim')

        self.maprstbtn = QPushButton('Reset')
        #self.maprstbtn.setToolTip('Prime for experiment')
        self.maprstbtn.clicked.connect(self.resetmap)

        self.tab3.layout = QVBoxLayout(self.tab3)
        self.tab3.layout.addWidget(self.map.canvas)
        self.tab3.layout.addWidget(self.maprstbtn)
        LP = self.zen.StagePos
        try:
            FS = self.zen.FrameSize
            pxsize = self.zen.pxsize
            self.map.append_data(LP[0] / 1000, LP[1] / 1000, FS[0] * pxsize / 1e6, FS[1] * pxsize / 1e6)
            self.map.draw()
        except:
            pass

    def changeColor(self):
        for channel in self.channels:
            color = self.zen.ChannelColorsRGB[channel]
            for plot in (self.eplot, self.iplot, self.splot, self.rplot, self.pplot, self.xyplot, self.xzplot, self.yzplot):
                if channel in plot:
                    plot.plt[channel].set_color(color)
            for tb in ('top', 'bottom'):
                cn = '{}{}'.format(tb, channel)
                if cn in self.splot:
                    self.splot.plt[cn].set_color(color)
        camera = [self.zen.CameraFromChannelName(i) for i in self.zen.ChannelNames]
        enabled = [False if self.cyllensdrp[c].currentText()=='None' else True for c in camera]
        self.rdbch.changeOptions(self.zen.ChannelNames, self.zen.ChannelColorsHex, enabled)

    def changeCylLens(self):
        self.confopen(self.conf.filename)
        self.changeColor()

    def calibrate(self, *args, **kwargs):
        self.calibbtn.setEnabled(False)
        options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        file, _ = QFileDialog.getOpenFileName(self, "Beads for calibration", "", "CZI Files (*.czi);;All Files (*)", options=options)
        if file:
            self.calibrate_file(file)

    @thread
    def calibrate_file(self, file):
        from calibz import calibz
        np.seterr(all='ignore');
        self.theta, self.q = calibz(file)
        #self.NameSpace.q = self.q
        #self.NameSpace.theta = self.theta
        self.calibbtn.setEnabled(True)
        self.thetafld.setText('{}'.format(self.theta))
        for i in range(9):
            self.edt[i].setText('{}'.format(self.q[i]))

    def resetmap(self):
        self.map.remove_data()

    def confsave(self, f):
        if not isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getSaveFileName(self, "Save config file", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if f:
            self.conf.filename = f
            self.conf.maxStep = self.maxStep
            for channel in self.channels:
                self.conf[self.getCmstr(channel)].q = self.q[channel]
                self.conf[self.getCmstr(channel)].theta = self.theta[channel]

    def confopen(self, f):
        if not isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getOpenFileName(self, "Open config file", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if f:
            self.q = {}
            self.theta = {}
            self.conf.filename = f
            for channel in self.channels:
                if self.getCmstr(channel) in self.conf:
                    if 'q' in self.conf[self.getCmstr(channel)]:
                        self.q[channel] = self.conf[self.getCmstr(channel)].q
                    if 'theta' in self.conf[self.getCmstr(channel)]:
                        self.theta[channel] = self.conf[self.getCmstr(channel)].theta
            if 'maxStep' in self.conf:
                self.maxStep = self.conf.maxStep
                self.maxStepfld.setText('{}'.format(self.maxStep))
            if 'style' in self.conf:
                self.setStyleSheet(self.conf.style)

    def changeDLF(self, val):
        #Change the duolink filter
        if val=='1':
            self.zen.DLFilter = 0
        else:
            self.zen.DLFilter = 1
        self.changeDL()

    def changemaxStep(self, val):
        self.maxStep = float(val)

    def changechannel(self, val):
        if len(val):
            self.channels = [int(v) for v in val]
            self.confopen(self.conf.filename)

    def changeDL(self, *args):
        #Upon change of duolink filterblock
        self.dlf.setText(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])

    def changeFeedbackMode(self, idx):
        self.feedbackMode = idx

    def tglcenterbox(self):
        self.zen.EnableEvent('LeftButtonDown')

    def getCmstr(self, channel):
        camera = self.zen.CameraFromChannelName(self.zen.ChannelNames[channel])
        return self.cyllensdrp[camera].currentText()+self.zen.MagStr

    def closeEvent(self, *args, **kwargs):
        self.setquit()

    def prime(self):
        if self.guithread is None or not self.guithread.is_alive():
            self.NameSpace.stop = False
            self.guithread = Thread(target=self.run)
            self.guithread.start()

    def stayprimed(self):
        if self.contrunchkbx.isChecked():
            self.prime()

    def run(self, *args, **kwargs):
        np.seterr(all='ignore');
        sleepTime = 0.02 #update interval

        Size = self.conf.ROISize #Size of the ROI in which the psf is fitted
        Pos = self.conf.ROIPos
        fastMode = self.conf.fastMode

        gain = self.conf.gain

        self.startbtn.setEnabled(False)
        self.stopbtn.setEnabled(True)
        self.zen.RemoveDrawings()
        FS = self.zen.FrameSize
        self.rectangle = self.zen.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)
        while True:
            self.startbtn.setText('Wait for experiment to start')

            #First wait for the experiment to start:
            while (not self.zen.ExperimentRunning) and (not self.stop) and (not self.quit):
                FS = self.zen.FrameSize
                self.rectangle = self.zen.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)
                sleep(sleepTime)

            fwhm = {}
            sigma = {}
            limits = {}
            for channel in self.channels:
                wavelength = self.rdbch.textBoxValues[channel]
                sigma[channel] = wavelength / 2 / self.zen.ObjectiveNA / self.zen.pxsize
                fwhm[channel] = sigma[channel] * 2 * np.sqrt(2 * np.log(2))

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

                limits[channel] = [[-np.inf, np.inf]] * 8
                limits[channel][2] = [fwhm[channel]/2, fwhm[channel]*2] #fraction of fwhm
                limits[channel][5] = [0.7, 1.3]
                limits[channel][7] = [0, np.inf]

            #Experiment has started:
            self.NameSpace.mode = self.rdb.state.lower()
            self.NameSpace.Size = Size
            self.NameSpace.Pos = Pos
            self.NameSpace.gain = gain
            self.NameSpace.channels = self.channels
            self.NameSpace.q = self.q
            self.NameSpace.theta = self.theta
            self.NameSpace.maxStep = self.maxStep
            self.NameSpace.fastMode = fastMode
            self.NameSpace.limits = limits
            self.NameSpace.feedbackMode = self.feedbackMode
            self.NameSpace.sigma = sigma

            self.NameSpace.run = True

            FS = self.zen.FrameSize
            self.rectangle = self.zen.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)

            cfilename = self.zen.FileName
            cfexists = isfile(cfilename) #whether ZEN already made the czi file (streaming)
            if cfexists:
                pfilename = splitext(cfilename)[0]+'.pzl'
            else:
                pfilename = splitext(self.conf.tmpPzlFile)[0]+datetime.now().strftime('_%Y%m%d-%H%M%S.pzl')
                if isfile(pfilename):
                    remove(pfilename)
            if not cfexists or (cfexists and not isfile(pfilename)):
                metafile = config.conf(pfilename)
                metafile.mode = self.NameSpace.mode
                metafile.FeedbackChannels = self.NameSpace.channels
                metafile.CylLens = [self.cyllensdrp[i].currentText() for i in range(2)]
                metafile.DLFilterSet = self.dlfs.currentText()
                metafile.DLFilterChannel = self.zen.DLFilter
                metafile.q = self.NameSpace.q
                metafile.theta = self.NameSpace.theta
                metafile.maxStep = self.NameSpace.maxStep
                metafile.ROISize = Size
                metafile.ROIPos = Pos
                metafile.Columns = ['channel', 'frame', 'piezoPos', 'focusPos', 'x', 'y', 'fwhm', 'i', 'o', 'e', 'time']
                with open(pfilename, 'a+') as file:
                    file.write('p:\n')

                    self.plot.remove_data()

                    SizeX, SizeY = self.zen.FrameSize
                    z0 = None

                    self.startbtn.setText('Experiment started')

                    while (self.zen.ExperimentRunning or (not self.Queue.empty())) and (not self.stop) and (not self.quit):
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
                                    a[:,2][a[:,2] < limits[channel][2][0]/2] = np.nan
                                    a[:,2][a[:,2] > limits[channel][2][1]*2] = np.nan
                                    a[:,5][a[:,5] < limits[channel][5][0]/2] = np.nan
                                    a[:,5][a[:,5] > limits[channel][5][1]*2] = np.nan
                                    a[:,7][a[:,7] < limits[channel][7][0]*2] = np.nan
                                    ridx = np.isnan(a[:,3]) | np.isnan(a[:,4])
                                    a[ridx,:] = np.nan

                                    zfit = np.array([-cyl.findz(e, self.q[channel]) if f else np.nan for e, f in zip(a[:, 5], Fitted)])
                                    z = 1000*(zfit + FocusPos - z0)

                                    self.eplot.range_data(Time, a[:,5], handle=channel)
                                    self.iplot.range_data(Time, a[:,3], handle=channel)
                                    self.splot.range_data(Time, a[:,2] / 2 / np.sqrt(2 * np.log(2)) * pxsize, handle=channel)
                                    self.splot.range_data(Time, [limits[channel][2][0]/2/np.sqrt(2*np.log(2))*pxsize] * len(Time), handle='bottom{}'.format(channel))
                                    self.splot.range_data(Time, [limits[channel][2][1]/2/np.sqrt(2*np.log(2))*pxsize] * len(Time), handle='top{}'.format(channel))
                                    self.rplot.range_data(Time, a[:,7], handle=channel)
                                    self.pplot.range_data(Time, zfit + FocusPos - z0, handle=channel)
                                    self.xyplot.append_data((a[:,0] - Size / 2) * pxsize, (a[:,1] - Size / 2) * pxsize, channel)
                                    self.xzplot.append_data((a[:,0] - Size / 2) * pxsize, z, channel)
                                    self.yzplot.append_data((a[:,1] - Size / 2) * pxsize, z, channel)
                                    if channel==channels[0]: #Draw these only once
                                        self.eplot.range_data(Time, [1] * len(Time), handle='middle')
                                        self.eplot.range_data(Time, [limits[channel][5][0]] * len(Time), handle='bottom')
                                        self.eplot.range_data(Time, [limits[channel][5][1]] * len(Time), handle='top')
                                        self.rplot.range_data(Time, [limits[channel][7][0]] * len(Time), handle='bottom')
                                        self.pplot.range_data(Time, np.array(FocusPos) - z0, handle='fb')
                                    self.plot.draw()

                                    if Fitted[-1]:
                                        X = float(a[-1,0] + (SizeX - Size) / 2 + Pos[0])
                                        Y = float(a[-1,1] + (SizeY - Size) / 2 + Pos[1])
                                        R = float(a[-1,2])
                                        E = float(a[-1,5])
                                        self.ellipse[channel] = self.zen.DrawEllipse(X, Y, R, E, self.theta[channel],
                                                    self.zen.ChannelColorsInt[channel], index=self.ellipse.get(channel))
                                    else:
                                        self.zen.RemoveDrawing(self.ellipse.get(channel))

                #After the experiment:
                for ellipse in self.ellipse.values():
                    self.zen.RemoveDrawing(ellipse)
                if not cfexists:
                    for i in range(5):
                        cfilename = functions.last_czi_file(self.conf.dataDir)
                        npfilename = splitext(cfilename)[0] + '.pzl'
                        if cfilename and not isfile(npfilename):
                            copyfile(pfilename, npfilename)
                            break
                        sleep(0.25)
            else:
                sleep(sleepTime)
            if not self.contrunchkbx.isChecked():
                break

        self.zen.RemoveDrawing(self.rectangle)
        self.stop = False
        self.startbtn.setText('Prime for experiment')
        self.stopbtn.setEnabled(False)
        self.startbtn.setEnabled(True)
        self.NameSpace.run = False

    def setstop(self):
        # Stop being primed for an experiment
        self.stopbtn.setEnabled(False)
        self.contrunchkbx.setChecked(False)
        self.stop = True
        self.NameSpace.stop = True
        if not self.guithread is None and self.guithread.is_alive():
            self.guithread.join(5)

    def setquit(self):
        # Quit the whole program
        self.setstop()
        self.quit = True
        self.NameSpace.quit = True
        close_threads()
        self.fblprocess.join(5)
        self.fblprocess.terminate()

class AppContext(ApplicationContext):           # 1. Subclass ApplicationContext
    def run(self):                              # 2. Implement run()
        window = App()
        return self.app.exec_()

if __name__ == '__main__':
    freeze_support()                            # to enable fbs/pyinstaller to work with multiprocessing
    appctxt = AppContext()                      # 4. Instantiate the subclass
    exit_code = appctxt.run()                   # 5. Invoke run()
    exit(exit_code)