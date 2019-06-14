import sys
import os
import time
from functools import partial
from threading import Thread

import numpy as np
from PyQt5.QtWidgets import *
from fbs_runtime.application_context import ApplicationContext
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import cylinderlens as cyl
import functions
import config
from events import events
from pid import pid
from zen import zen

np.seterr(all='ignore');

def errwrap(fun,default,*args):
    """ returns either the result of fun(*args) or default when an error occurs
    """
    try:
        return fun(*args)
    except:
        return default

def thread(fun):
    """ decorator to run function in a separate thread to keep the gui responsive
    """
    return lambda *args: Thread(target=fun, args=args).start()

def firstargonly(fun):
    """ decorator that only passes the first argument to a function
    """
    return lambda *args: fun(args[0])

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 1500
        self.top = 50
        self.title = 'Cylinder lens feedback GUI'
        self.width = 640
        self.height = 400
        self.stop = False
        self.docenter = False
        self.conf = config.conf()

        self.zen = zen()

        self.q = list()
        self.maxStep = 1
        self.theta = 0

        self.ellipse = None
        self.rectangle = None
        self.MagStr = self.zen.MagStr
        self.DLFilter = self.zen.DLFilter

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.central_widget = QWidget()
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget(self.central_widget)
        self.settab1()
        self.settab2()
        self.settab3()
        self.menus()
        self.layout.addWidget(self.tabs)

        events(self)

        self.setCentralWidget(self.central_widget)
        self.show()

    def menus(self):
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')

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

        self.centerbtn = QPushButton('Center')
        self.centerbtn.setToolTip('Push, then click on image')
        self.centerbtn.clicked.connect(self.center)

        self.startbtn = QPushButton('Prime for experiment')
        self.startbtn.setToolTip('Prime for experiment')
        self.startbtn.clicked.connect(self.run)

        self.stopbtn = QPushButton('Stop')
        self.stopbtn.setToolTip('Stop')
        self.stopbtn.clicked.connect(self.setstop)
        self.stopbtn.setEnabled(False)

        self.rdb = RadioButtons(('Zhuang', 'PID'))

        self.eplot = PlotCanvas(None, 5, 1)
        self.eplot.ax.set_yscale('log')
        self.eplot.ax.set_xlabel('Time (frames)')
        self.eplot.ax.set_ylabel('Ellipticity')

        self.pplot = PlotCanvas(None, 5, 1)
        self.pplot.ax.set_xlabel('Time (frames)')
        self.pplot.ax.set_ylabel('Piezo pos (µm)')

        self.buttons = QHBoxLayout()
        self.buttons.addWidget(self.contrunchkbx)
        self.buttons.addWidget(self.centerbtn)
        self.buttons.addWidget(self.startbtn)
        self.buttons.addWidget(self.stopbtn)
        self.buttons.addWidget(self.rdb)

        self.tab1.layout = QVBoxLayout(self.tab1)
        self.tab1.layout.addLayout(self.buttons)
        self.tab1.layout.addWidget(self.eplot)
        self.tab1.layout.addWidget(self.pplot)

    def settab2(self):
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, 'Configuration')

        self.grid = QGridLayout()
        self.grid.setColumnStretch(3, 3)
        self.grid.setColumnStretch(6, 3)

        labelnames = ('e0:', 'z0:', 'c:', 'Ax:', 'Bx:', 'dx:', 'Ay:', 'By:', 'dy:')
        self.lbl = list()
        for i, l in enumerate(labelnames):
            self.lbl.append(QLabel(l))
            self.grid.addWidget(self.lbl[-1], i, 1)

        self.edt = list()
        for i in range(9):
            self.edt.append(QLineEdit())
            self.edt[-1].textChanged.connect(self.changeq(i))
            self.grid.addWidget(self.edt[-1], i, 2)

        unitnames = ('', 'um', 'um', 'um2', 'um3', 'um', 'um2', 'um3', 'um')
        self.unt = list()
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
        self.dlf = QLabel(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])
        self.grid.addWidget(self.dlf, 3, 6)

        self.grid.addWidget(QLabel('theta:'), 4, 4)
        self.thetafld = QLineEdit()
        self.thetafld.textChanged.connect(self.changetheta)
        self.grid.addWidget(self.thetafld, 4, 5)
        self.grid.addWidget(QLabel('rad'), 4, 6)

        self.grid.addWidget(QLabel('Max stepsize:'), 5, 4)
        self.maxStepfld = QLineEdit()
        self.maxStepfld.textChanged.connect(self.changemaxStep)
        self.grid.addWidget(self.maxStepfld, 5, 5)
        self.grid.addWidget(QLabel('um'), 5, 6)

        self.tab2.setLayout(self.grid)

        self.confopen(self.conf.filename)

    def settab3(self):
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, 'Map')

        self.map = PlotCanvas(None, 4, 4)
        self.map.ax.invert_xaxis()
        self.map.ax.invert_yaxis()
        self.map.ax.set_xlabel('x')
        self.map.ax.set_xlabel('y')
        self.tab3.layout = QVBoxLayout(self.tab3)
        self.tab3.layout.addWidget(self.map)

    def confsave(self, f):
        if not os.path.isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getSaveFileName(self, "Save config file", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if f:
            self.conf.filename = f
            self.conf[self.cmstr].q = self.q
            self.conf[self.cmstr].theta = self.theta
            self.conf.maxStep = self.maxStep

    def confopen(self, f):
        if not os.path.isfile(f):
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
        self.dlf.setText(self.dlfs.currentText().split(' & ')[self.zen.DLFilter])

    def center(self):
        if self.docenter == False:
            self.docenter = True
            self.centerbtn.setText('Waiting for click')
        else:
            self.docenter = False
            self.centerbtn.setText('Center')

    @property
    def cmstr(self):
        return self.cyllensdrp[self.channel].currentText()+self.zen.MagStr

    @firstargonly
    def closeEvent(self):
        self.setstop()

    def stayprimed(self):
        if self.contrunchkbx.isChecked():
            self.run()

    @thread
    @firstargonly
    def run(self):
        np.seterr(all='ignore');

        Size = 32 #Size of the ROI in which the psf is fitted
        SleepTime = 0.02
        gain = -5e-3

        self.startbtn.setEnabled(False)
        self.stopbtn.setEnabled(True)
        Z = zen()  #cannot move com-objects from one thread to another :(
        Z.RemoveDrawings()
        FS = Z.FrameSize
        self.rectangle = Z.DrawRectangle(FS[0]/2, FS[1]/2, Size, Size, index=self.rectangle)
        while True:
            mode = self.rdb.state.lower()
            self.startbtn.setText('Wait for experiment to start')

            #First wait for the experiment to start:
            while (not Z.ExperimentRunning) & (not self.stop):
                FS = Z.FrameSize
                self.rectangle = Z.DrawRectangle(FS[0] / 2, FS[1] / 2, Size, Size, index=self.rectangle)
                time.sleep(SleepTime)

            #Experiment has started:
            FS = Z.FrameSize
            self.rectangle = Z.DrawRectangle(FS[0] / 2, FS[1] / 2, Size, Size, index=self.rectangle)
            G = Z.PiezoPos
            pfilename = Z.FileName[:-3]+'pzl'
            if not os.path.isfile(pfilename):
                metafile = config.conf(pfilename)
                metafile.FeedbackChannel = self.channel
                metafile.CylLens = [self.cyllensdrp[i].currentText() for i in range(2)]
                metafile.DLFilterSet = self.dlfs.currentText()
                metafile.DLFilterChannel = Z.DLFilter
                metafile.q = self.q
                metafile.theta = self.theta
                metafile.maxStep = self.maxStep
                metafile.ROISize = Size
                file = open(pfilename, 'a+')

                if self.channel == 0:
                    wavelength = 646
                else:
                    wavelength = 510

                self.eplot.remove_data()
                self.pplot.remove_data()

                if mode == 'pid':
                    P = pid(0, G, self.maxStep, Z.TimeInterval, gain)

                f = wavelength / 2 / Z.ObjectiveNA / Z.pxsize
                fwhmlim = f * 2 * np.sqrt(2 * np.log(2))
                SizeX, SizeY = Z.FrameSize
                TimeInterval = Z.TimeInterval

                self.startbtn.setText('Experiment started')

                TimeMem = 0
                while (Z.ExperimentRunning) & (not self.stop):
                    Frame, Time = Z.GetFrameCenter(self.channel, Size)
                    if Time < TimeMem:
                        TTime = TimeMem + 1
                    else:
                        TTime = Time
                    file.write('{},{},{},'.format(TTime, Z.PiezoPos, Z.GetCurrentZ))
                    # Z.SaveDouble('Piezo {}'.format(Time), Z.GetPiezoPos())
                    # Z.SaveDouble('Zstage {}'.format(Time), Z.GetCurrentZ())

                    a = functions.fg(Frame, self.theta, f)
                    file.write('{},{},{},{},{},{}\n'.format(*a))

                    #Update the piezo position:
                    if mode == 'pid':
                        F = -np.log(a[5])
                        if np.abs(F) > 1:
                            F = 0
                        Pz = P(F)
                        Z.PiezoPos = Pz
                    else:
                        if np.abs(np.log(a[5]))>1 or a[2]<fwhmlim/4 or a[2]>fwhmlim*4:
                            z = 0
                        else:
                            z = np.clip(cyl.findz(a[5], self.q), -self.maxStep, self.maxStep)
                        Z.MovePiezoRel(-z)
                    if Time > TimeMem:
                        self.eplot.range_data(Time, a[5])
                        self.pplot.range_data(Time, Z.PiezoPos)

                    X = float(a[0] + (SizeX-Size) / 2 + 1)
                    Y = float(a[1] + (SizeY-Size) / 2 + 1)
                    R = float(a[2])
                    E = float(a[5])

                    self.ellipse = Z.DrawEllipse(X, Y, R, E, self.theta, index=self.ellipse)

                    if mode == 'pid':
                        if Pz > (G + 5):
                            P = pid(0, G, self.maxStep, TimeInterval, gain)

                    #Wait for next frame:
                    while ((Z.GetTime-1) == Time) & Z.ExperimentRunning & (not self.stop):
                        time.sleep(TimeInterval / 4)
                    if Time < TimeMem:
                        break

                    TimeMem = Time

                #After the experiment:
                file.close()
                Z.RemoveDrawing(self.ellipse)
            else:
                time.sleep((SleepTime))
            if not self.contrunchkbx.isChecked():
                break

        Z.RemoveDrawing(self.rectangle)
        Z.DisconnectZEN()
        self.stop = False
        self.startbtn.setText('Prime for experiment')
        self.stopbtn.setEnabled(False)
        self.startbtn.setEnabled(True)

    def setstop(self):
        self.contrunchkbx.setChecked(False)
        self.stop = True

class RadioButtons(QWidget):
    def __init__(self, txt, init_state=0, callback=None):
        QWidget.__init__(self)
        layout = QGridLayout()
        self.setLayout(layout)
        self.callback = callback
        self.state = txt[init_state]
        for i, t in enumerate(txt):
            radiobutton = QRadioButton(t)
            if i == init_state:
                radiobutton.setChecked(True)
            radiobutton.text = t
            radiobutton.toggled.connect(self.onClicked)
            layout.addWidget(radiobutton, 0, i)

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.state = radioButton.text
            if not self.callback is None:
                self.callback(radioButton.text)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
 
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plt, = self.ax.plot([], 'r-')
        self.draw()
        
    def append_data(self,x,y=None):
        if y is None:
            y = x
            x = errwrap(np.nanmax,-1,self.plt.get_xdata()) + np.arange(errwrap(len,1,x)) + 1
        x = np.hstack((self.plt.get_xdata(),x))
        y = np.hstack((self.plt.get_ydata(),y))
        self.plt.set_xdata(x)
        self.plt.set_ydata(y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()
        
    def range_data(self, x, y, range=100):
        x = np.hstack((self.plt.get_xdata(),x))
        y = np.hstack((self.plt.get_ydata(),y))
        self.plt.set_ydata(y[x > np.nanmax(x)-range])
        self.plt.set_xdata(x[x > np.nanmax(x)-range])
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

    def numel_data(self, x, y, range=10000):
        x = np.hstack((self.plt.get_xdata(), x))
        y = np.hstack((self.plt.get_ydata(), y))
        self.plt.set_ydata(y[-range:])
        self.plt.set_xdata(x[-range:])
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

    def remove_data(self):
        self.plt.set_xdata([])
        self.plt.set_ydata([])
        self.draw()
        
class AppContext(ApplicationContext):           # 1. Subclass ApplicationContext
    def run(self):                              # 2. Implement run()
        window = App()
        return self.app.exec_()

if __name__ == '__main__':
    appctxt = AppContext()                      # 4. Instantiate the subclass
    exit_code = appctxt.run()                   # 5. Invoke run()
    sys.exit(exit_code)