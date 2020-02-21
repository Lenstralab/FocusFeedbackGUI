from sys import exit
from os.path import isfile, splitext
from os import remove
from shutil import copyfile
from time import sleep, time
from datetime import datetime
from functools import partial
from threading import Thread
from multiprocessing import Process, Queue, Manager, freeze_support, queues

import numpy as np
from PyQt5.QtWidgets import *
from fbs_runtime.application_context import ApplicationContext
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams, pyplot
rcParams.update({'figure.autolayout': True})

import cylinderlens as cyl
import functions, config
from events import events
from pid import pid
from zen import zen

np.seterr(all='ignore');

def errwrap(fun, default, *args):
    """ returns either the result of fun(*args) or default when an error occurs
    """
    try:
        return fun(*args)
    except:
        return default

def firstargonly(fun):
    """ decorator that only passes the first argument to a function
    """
    return lambda *args: fun(args[0])

threads = []
def thread(fun):
    """ decorator to run function in a separate thread to keep the gui responsive
    """
    def tfun(*args, **kwargs):
        T = Thread(target=fun, args=args, kwargs=kwargs)
        threads.append(T)
        T.start()
        return T
    return tfun

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
            channel = NameSpace.channel
            q = NameSpace.q
            theta = NameSpace.theta
            maxStep = NameSpace.maxStep
            fastMode = NameSpace.fastMode
            TimeInterval = Z.TimeInterval
            G = Z.PiezoPos
            FS = Z.FrameSize
            TimeMem = 0
            detected = functions.truncated_list(5, (True,)*5)

            if channel == 0:
                wavelength = 646
            else:
                wavelength = 510
            f = wavelength / 2 / Z.ObjectiveNA / Z.pxsize
            fwhmlim = f * 2 * np.sqrt(2 * np.log(2))

            if mode == 'pid':
                P = pid(0, G, maxStep, TimeInterval, gain)

            while Z.ExperimentRunning and (not NameSpace.stop):
                STime = time()
                Frame, Time = Z.GetFrame(channel, *functions.cliprect(FS, *Pos, Size, Size))
                #Frame, Time = Z.GetFrame(channel, Xs=(Size, Size))
                PiezoPos = Z.PiezoPos
                FocusPos = Z.GetCurrentZ
                a = functions.fg(Frame, theta, f, fastMode)

                if Time < TimeMem:
                    TTime = TimeMem + 1
                else:
                    TTime = Time

                #try to determine when something is detected by using a simple filter
                if not any(np.isnan(a)) and a[7] > 0.3 and np.abs(a[5]-1) < 0.3 and np.abs(np.log(a[2] / fwhmlim)) < 0.7:
                    if sum(detected)/len(detected) > 0.75:
                        Queue.put((TTime, a[:8], PiezoPos, FocusPos, STime))

                        # Update the piezo position:
                        if mode == 'pid':
                            F = -np.log(a[5])
                            if np.abs(F) > 1:
                                F = 0
                            Pz = P(F)
                            Z.PiezoPos = Pz

                            if Pz > (G + 5):
                                P = pid(0, G, maxStep, TimeInterval, gain)
                        else:
                            z = np.clip(cyl.findz(a[5], q), -maxStep, maxStep)
                            #print(TTime, z)
                            if not np.isnan(z):
                                Z.MovePiezoRel(-z)
                        #print('Good:'+(' {:.2f}'*8).format(*a))
                    else:
                        Queue.put((TTime, np.full(np.shape(a), np.nan), PiezoPos, FocusPos, STime))
                    detected.append(True)
                else:
                    Queue.put((TTime, np.full(np.shape(a), np.nan), PiezoPos, FocusPos, STime))
                    detected.append(False)

                # Wait for next frame:
                while ((Z.GetTime - 1) == Time) and Z.ExperimentRunning and (not NameSpace.stop) and (not NameSpace.quit):
                    sleep(TimeInterval / 4)
                if Time < TimeMem:
                    break

                TimeMem = Time
            NameSpace.run = False
        else:
            sleep(0.01)
    Z.DisconnectZEN()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Cylinder lens feedback GUI'
        self.width = 640
        self.height = 1024
        self.stop = False # Stop (waiting for) experiment
        self.quit = False # Quit program
        self.conf = config.conf()
        self.curzentitle = ''

        self.zen = zen()

        self.q = []
        self.maxStep = 1
        self.theta = 0

        self.ellipse = None
        self.rectangle = None
        self.MagStr = self.zen.MagStr
        self.DLFilter = self.zen.DLFilter

        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
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
        z = zen()
        while not z.ready and not self.stop and not self.quit:
            sleep(0.1)
        z.DisconnectZEN()
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

        self.rdb = RadioButtons(('Zhuang', 'PID'))

        self.plot = PlotCanvas()
        self.eplot = SubPlot(self.plot, 611)
        self.eplot.append_plot('--b')
        #self.eplot.ax.set_yscale('log')
        self.eplot.ax.set_ylabel('Ellipticity')
        self.eplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.iplot = SubPlot(self.plot, 612)
        self.iplot.ax.set_ylabel('Intensity')
        self.iplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.splot = SubPlot(self.plot, 613)
        self.splot.ax.set_ylabel('Sigma (nm)')
        self.splot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.oplot = SubPlot(self.plot, 614)
        self.oplot.ax.set_ylabel('Offset')
        self.oplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.pplot = SubPlot(self.plot, 615)
        self.pplot.append_plot()
        self.pplot.ax.set_xlabel('Time (frames)')
        self.pplot.ax.set_ylabel('Piezo pos (µm)')

        self.xyplot = SubPlot(self.plot, (6, 3, 16), '.r')
        self.xyplot.ax.set_xlabel('x (nm)')
        self.xyplot.ax.set_ylabel('y (nm)')
        self.xyplot.ax.set_aspect('equal', adjustable='datalim')

        self.xzplot = SubPlot(self.plot, (6, 3, 17), '.r')
        self.xzplot.ax.set_xlabel('x (nm)')
        self.xzplot.ax.set_ylabel('z (nm)')
        self.xzplot.ax.set_aspect('equal', adjustable='datalim')

        self.yzplot = SubPlot(self.plot, (6, 3, 18), '.r')
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
        self.grid.setColumnStretch(3, 3)
        self.grid.setColumnStretch(6, 3)

        labelnames = ('e0:', 'z0:', 'c:', 'Ax:', 'Bx:', 'dx:', 'Ay:', 'By:', 'dy:')
        self.lbl = []
        for i, l in enumerate(labelnames):
            self.lbl.append(QLabel(l))
            self.grid.addWidget(self.lbl[-1], i, 1)

        self.edt = []
        for i in range(9):
            self.edt.append(QLineEdit())
            self.edt[-1].textChanged.connect(self.changeq(i))
            self.grid.addWidget(self.edt[-1], i, 2)

        unitnames = ('', 'um', 'um', 'um2', 'um3', 'um', 'um2', 'um3', 'um')
        self.unt = []
        for i, l in enumerate(unitnames):
            self.unt.append(QLabel(l))
            self.grid.addWidget(self.unt[-1], i, 3)

        self.channel = 1
        self.grid.addWidget(QLabel('Feedback channel:'), 0, 4)
        self.rdbch = RadioButtons(('R', 'G'), init_state=self.channel, callback=self.changechannel)
        self.grid.addWidget(self.rdbch, 0, 5)

        self.cyllensdrp = []
        self.grid.addWidget(QLabel('Cylindrical lens R:'), 1, 4)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None','A','B'])
        self.cyllensdrp[-1].currentIndexChanged.connect(partial(self.confopen, self.conf.filename))
        self.grid.addWidget(self.cyllensdrp[-1], 1, 5)

        self.grid.addWidget(QLabel('Cylindrical lens G:'), 2, 4)
        self.cyllensdrp.append(QComboBox())
        self.cyllensdrp[-1].addItems(['None','A', 'B'])
        self.cyllensdrp[-1].setCurrentIndex(1)
        self.cyllensdrp[-1].currentIndexChanged.connect(partial(self.confopen, self.conf.filename))
        self.grid.addWidget(self.cyllensdrp[-1], 2, 5)

        self.grid.addWidget(QLabel('Duolink filterset:'), 3, 4)
        self.dlfs = QComboBox()
        self.dlfs.addItems(['488/561 & 488/640', '561/640 & empty'])
        self.dlfs.currentIndexChanged.connect(self.changeDL)
        self.grid.addWidget(self.dlfs, 3, 5)

        self.grid.addWidget(QLabel('Duolink filter:'), 4, 4)
        self.chdlf = RadioButtons(('1', '2'), init_state=self.zen.DLFilter, callback=self.changeDLF)
        self.grid.addWidget(self.chdlf, 4, 5)
        self.dlf = QLabel(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])
        self.grid.addWidget(self.dlf, 4, 6)

        self.grid.addWidget(QLabel('theta:'), 5, 4)
        self.thetafld = QLineEdit()
        self.thetafld.textChanged.connect(self.changetheta)
        self.grid.addWidget(self.thetafld, 5, 5)
        self.grid.addWidget(QLabel('rad'), 5, 6)

        self.grid.addWidget(QLabel('Max stepsize:'), 6, 4)
        self.maxStepfld = QLineEdit()
        self.maxStepfld.textChanged.connect(self.changemaxStep)
        self.grid.addWidget(self.maxStepfld, 6, 5)
        self.grid.addWidget(QLabel('um'), 6, 6)

        self.calibbtn = QPushButton('Calibrate with beads')
        self.calibbtn.setToolTip('Calibrate with beads')
        self.calibbtn.clicked.connect(self.calibrate)
        self.grid.addWidget(self.calibbtn, 7, 5)

        self.tab2.setLayout(self.grid)

        self.confopen(self.conf.filename)

    def settab3(self):
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, 'Map')

        self.map = SubPatchPlot(PlotCanvas(), color=(0.6, 1, 0.8))
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
        print('Hello world!')
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
            self.conf[self.cmstr].q = self.q
            self.conf[self.cmstr].theta = self.theta
            self.conf.maxStep = self.maxStep

    def confopen(self, f):
        if not isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getOpenFileName(self, "Open config file", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if f:
            self.conf.filename = f
            if self.cmstr in self.conf:
                if 'q' in self.conf[self.cmstr]:
                    self.q = self.conf[self.cmstr].q
                    for i in range(9):
                        self.edt[i].setText('{}'.format(self.q[i]))

                if 'theta' in self.conf[self.cmstr]:
                    self.theta = self.conf[self.cmstr].theta
                    self.thetafld.setText('{}'.format(self.theta))

            if 'maxStep' in self.conf:
                self.maxStep = self.conf.maxStep
                self.maxStepfld.setText('{}'.format(self.maxStep))

    def changeDLF(self, val):
        #Change the duolink filter
        if val=='1':
            self.zen.DLFilter = 0
        else:
            self.zen.DLFilter = 1
        self.changeDL()

    def changeq(self, i):
        def fun(val):
            try:
                self.q[i] = float(val)
            except:
                pass
        return fun

    def changemaxStep(self, val):
        self.maxStep = float(val)

    def changetheta(self, val):
        self.theta = float(val)

    def changechannel(self, val):
        if val=='R':
            self.channel = 0
        else:
            self.channel = 1
        self.confopen(self.conf.filename)

    def changeDL(self, *args):
        #Upon change of duolink filterblock
        self.dlf.setText(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])

    def tglcenterbox(self):
        self.zen.EnableEvent('LeftButtonDown')

    @property
    def cmstr(self):
        return self.cyllensdrp[self.channel].currentText()+self.zen.MagStr

    def closeEvent(self, *args, **kwargs):
        self.setquit()

    def prime(self):
        if self.guithread is None or not self.guithread.is_alive():
            self.guithread = Thread(target=self.run)
            self.guithread.start()

    def stayprimed(self):
        if self.contrunchkbx.isChecked():
            self.prime()

    def run(self, *args, **kwargs):
        np.seterr(all='ignore');
        sleepTime = 0.02 #GUI update interval

        Size = self.conf.ROISize #Size of the ROI in which the psf is fitted
        Pos = self.conf.ROIPos
        fastMode = self.conf.fastMode

        gain = self.conf.gain

        self.startbtn.setEnabled(False)
        self.stopbtn.setEnabled(True)
        Z = zen()  #cannot move com-objects from one thread to another :(
        Z.RemoveDrawings()
        FS = Z.FrameSize
        self.rectangle = Z.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)
        while True:
            self.startbtn.setText('Wait for experiment to start')

            #First wait for the experiment to start:
            while (not Z.ExperimentRunning) and (not self.stop) and (not self.quit):
                FS = Z.FrameSize
                self.rectangle = Z.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)
                sleep(sleepTime)

            #Experiment has started:
            self.NameSpace.mode = self.rdb.state.lower()
            self.NameSpace.Size = Size
            self.NameSpace.Pos = Pos
            self.NameSpace.gain = gain
            self.NameSpace.channel = self.channel
            self.NameSpace.q = self.q
            self.NameSpace.theta = self.theta
            self.NameSpace.maxStep = self.maxStep
            self.NameSpace.fastMode = fastMode
            self.NameSpace.run = True

            FS = Z.FrameSize
            self.rectangle = Z.DrawRectangle(*functions.cliprect(FS, *Pos, Size, Size), index=self.rectangle)

            cfilename = Z.FileName
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
                metafile.FeedbackChannel = self.NameSpace.channel
                metafile.CylLens = [self.cyllensdrp[i].currentText() for i in range(2)]
                metafile.DLFilterSet = self.dlfs.currentText()
                metafile.DLFilterChannel = Z.DLFilter
                metafile.q = self.NameSpace.q
                metafile.theta = self.NameSpace.theta
                metafile.maxStep = self.NameSpace.maxStep
                metafile.ROISize = Size
                metafile.ROIPos = Pos
                with open(pfilename, 'a+') as file:
                    file.write('p:\n')

                    self.plot.remove_data()

                    SizeX, SizeY = Z.FrameSize
                    z0 = None

                    self.startbtn.setText('Experiment started')

                    while (Z.ExperimentRunning or (not self.Queue.empty())) and (not self.stop) and (not self.quit):
                        pxsize = Z.pxsize

                        #Wait until feedbackloop analysed a new frame
                        while Z.ExperimentRunning and (not self.stop) and (not self.quit) and self.Queue.empty():
                            sleep(sleepTime)
                        if not self.Queue.empty():
                            Time, a, PiezoPos, FocusPos, STime = [], [], [], [], []
                            for i in range(20):
                                if not self.Queue.empty():
                                    Q = self.Queue.get()
                                    Time.append(Q[0])
                                    a.append(Q[1])
                                    PiezoPos.append(Q[2])
                                    FocusPos.append(Q[3])
                                    STime.append(Q[4])
                                    if z0 is None:
                                        z0 = PiezoPos[0]+FocusPos[0]
                                else:
                                    break

                            for t, b, p, f, s in zip(Time, a, PiezoPos, FocusPos, STime):
                                file.write('- [{},{},{},'.format(t, p, f))
                                file.write('{},{},{},{},{},{},'.format(*b))
                                file.write('{}]\n'.format(s))

                            a = np.array(a)
                            if a.shape[1]==0:
                                continue
                            a[:,5][a[:,5]>1.3] = np.nan
                            a[:,5][a[:,5]<0.7] = np.nan
                            ridx = np.isnan(a[:,3]) | np.isnan(a[:,4])
                            a[ridx,:] = np.nan

                            zfit = np.array([-cyl.findz(e, self.q) for e in a[:, 5]])
                            z = 1000*(zfit - np.array(PiezoPos) - np.array(FocusPos) + z0)

                            self.eplot.range_data(Time, a[:,5])
                            self.eplot.range_data(Time, [1]*len(Time), N=1)
                            self.iplot.range_data(Time, a[:,3])
                            self.splot.range_data(Time, a[:,2] / 2 / np.sqrt(2 * np.log(2)) * pxsize)
                            self.oplot.range_data(Time, a[:,4])
                            self.pplot.range_data(Time, PiezoPos)
                            self.pplot.range_data(Time, PiezoPos+zfit, N=1)
                            self.xyplot.append_data((a[:,0] - Size / 2) * pxsize, (a[:,1] - Size / 2) * pxsize)
                            self.xzplot.append_data((a[:,0] - Size / 2) * pxsize, z)
                            self.yzplot.append_data((a[:,1] - Size / 2) * pxsize, z)
                            self.plot.draw()

                            X = float(a[-1,0] + (SizeX - Size) / 2 + Pos[0])
                            Y = float(a[-1,1] + (SizeY - Size) / 2 + Pos[1])
                            R = float(a[-1,2])
                            E = float(a[-1,5])

                            self.ellipse = Z.DrawEllipse(X, Y, R, E, self.theta, index=self.ellipse)

                #After the experiment:
                Z.RemoveDrawing(self.ellipse)
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

        Z.RemoveDrawing(self.rectangle)
        Z.DisconnectZEN()
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

        for T in threads:
            print(T)
            T.join()

        self.fblprocess.join(5)
        self.fblprocess.terminate()

class RadioButtons(QWidget):
    def __init__(self, txt, init_state=0, callback=None):
        QWidget.__init__(self)
        layout = QGridLayout()
        self.setLayout(layout)
        self.callback = callback
        self.state = txt[init_state]
        self.radiobutton = []
        for i, t in enumerate(txt):
            self.radiobutton.append(QRadioButton(t))
            if i == init_state:
                self.radiobutton[-1].setChecked(True)
            self.radiobutton[-1].text = t
            self.radiobutton[-1].toggled.connect(self.onClicked)
            layout.addWidget(self.radiobutton[-1], 0, i)

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.state = radioButton.text
            if not self.callback is None:
                self.callback(radioButton.text)

    def changeState(self, state):
        for i, r in enumerate(self.radiobutton):
            if i == state:
                r.setChecked(True)
            else:
                r.setChecked(False)

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
    def __init__(self, canvas, position=111, linespec='-r'):
        if isinstance(position, tuple):
            self.ax = canvas.fig.add_subplot(*position)
        else:
            self.ax = canvas.fig.add_subplot(position)
        self.plt = []
        self.plt.append(self.ax.plot([], linespec)[0])
        canvas.subplot.append(self)
        self.canvas = canvas

    def append_plot(self, linespec='-b'):
        self.plt.append(self.ax.plot([], linespec)[0])

    def append_data(self, x, y=None, N=0):
        if y is None:
            y = x
            x = errwrap(np.nanmax, -1, self.plt.get_xdata()) + np.arange(errwrap(len, 1, x)) + 1
        x = np.hstack((self.plt[N].get_xdata(), x))
        y = np.hstack((self.plt[N].get_ydata(), y))
        self.plt[N].set_xdata(x)
        self.plt[N].set_ydata(y)
        self.ax.relim()
        self.ax.autoscale_view()

    def range_data(self, x, y, range=250, N=0):
        margin = 0.05
        x = np.hstack((self.plt[N].get_xdata(), x))
        y = np.hstack((self.plt[N].get_ydata(), y))
        y = y[x > np.nanmax(x) - range]
        x = x[x > np.nanmax(x) - range]
        self.plt[N].set_ydata(y)
        self.plt[N].set_xdata(x)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        deltax = (xmax - xmin) * margin
        if np.isfinite(deltax) and not xmin==xmax:
            self.ax.set_xlim(xmin - deltax, xmax + deltax)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        deltay = (ymax - ymin) * margin
        if np.isfinite(deltay) and not ymin==ymax:
            self.ax.set_ylim(ymin - deltay, ymax + deltay)
        self.ax.autoscale_view()

    def numel_data(self, x, y, range=10000, N=0):
        x = np.hstack((self.plt.get_xdata(), x))
        y = np.hstack((self.plt.get_ydata(), y))
        self.plt[N].set_ydata(y[-range:])
        self.plt[N].set_xdata(x[-range:])
        self.ax.relim()
        self.ax.autoscale_view()

    def remove_data(self):
        for plt in self.plt:
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

    def numel_data(self, x, y, dx, dy=None, range=1000):
        if self.rects:
            while len(self.rects) >= range:
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

class AppContext(ApplicationContext):           # 1. Subclass ApplicationContext
    def run(self):                              # 2. Implement run()
        window = App()
        return self.app.exec_()

if __name__ == '__main__':
    freeze_support()                            # to enable fbs/pyinstaller to work with multiprocessing
    appctxt = AppContext()                      # 4. Instantiate the subclass
    exit_code = appctxt.run()                   # 5. Invoke run()
    exit(exit_code)