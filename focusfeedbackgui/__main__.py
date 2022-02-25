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
from focusfeedbackgui import QGui
from focusfeedbackgui import cylinderlens as cyl
from focusfeedbackgui import functions
from focusfeedbackgui.utilities import QThread, yaml_load, warp
from focusfeedbackgui.pid import Pid
from focusfeedbackgui.microscopes import MicroscopeClass


np.seterr(all='ignore')


def feedbackloop(queue, ns, microscope):
    # this is run in a separate process
    microscope = MicroscopeClass(microscope)
    _ = microscope.piezo_pos

    def get_cm_str(channel):
        return ns.cyllens[microscope.get_camera(microscope.channel_names[channel])] + microscope.magnification_str

    while not ns.quit:
        if ns.run:
            time_interval = microscope.time_interval
            piezo_pos = microscope.piezo_pos
            frame_size = microscope.frame_size
            time_n_prev = 0
            time_s_prev = time()
            piezo_time = .5  # time piezo needs to settle in s
            detected = deque((True,) * 5, 5)
            zmem = {}
            roi_pos = np.array(ns.roi_pos).astype(float)

            pid = Pid(0, piezo_pos, ns.max_step, time_interval, ns.gain)

            while microscope.is_experiment_running and (not ns.stop):
                ellipticity = {}
                piezo_pos = microscope.piezo_pos
                focus_pos = microscope.focus_pos
                xy_pos = {}

                for channel in ns.channels:
                    time_s_now = time()
                    frame, time_n_now = microscope.get_frame(channel,
                                                             *functions.clip_rectangle(frame_size, *ns.roi_pos,
                                                                                       ns.roi_size, ns.roi_size))
                    a = np.hstack(functions.fitgauss(frame, ns.theta[get_cm_str(channel)], ns.sigma[channel],
                                                     ns.fast_mode))
                    if time_n_now < time_n_prev:
                        time_n = time_n_prev + 1
                    else:
                        time_n = time_n_now

                    # try to determine when something is detected by using a simple filter on R2, el and psf width
                    if not any(np.isnan(a)) and all([l[0] < n < l[1] for n, l in zip(a, ns.limits[channel])]):
                        detected.append(True)
                        if sum(detected)/len(detected) > 0.35:
                            fitted = True
                            ellipticity[channel] = a[5]
                            xy_pos[channel] = a[:2]
                        else:
                            fitted = False
                    else:
                        fitted = False
                        detected.append(False)
                    queue.put((channel, time_n, fitted, a[:8], piezo_pos, focus_pos, time_s_now))
                else:
                    time_n, time_n_now = 0, 0

                time_s_now = time()
                piezo_factor = np.clip((time_s_now - time_s_prev) / piezo_time, 0.2, 1)

                # Update the piezo position:
                if ns.feedback_mode == 'pid':
                    if np.any(np.isfinite(list(ellipticity.values()))):
                        e = np.nanmean(list(ellipticity.values()))
                        log_e = -np.log(e)
                        if np.abs(log_e) > 1:
                            log_e = 0
                        piezo_pos_new = pid(log_e)
                        microscope.piezo_pos = piezo_pos_new

                        if piezo_pos_new > (piezo_pos + 5):
                            pid = Pid(0, piezo_pos, ns.max_step, time_interval, ns.gain)
                else:  # Zhuang
                    dz = {channel: np.clip(cyl.find_z(e, ns.q[get_cm_str(channel)]), -ns.max_step, ns.max_step)
                          for channel, e in ellipticity.items()}
                    cur_channel_idx = time_n % len(ns.channels)
                    next_channel_idx = (time_n + 1) % len(ns.channels)
                    if np.any(np.isfinite(list(dz.values()))):
                        if ns.mult_channel_mode == 0:  # Average
                            dz = np.nanmean(list(dz.values()))
                            if not np.isnan(dz):  # reduce if going faster than piezo, avoid oscillations
                                microscope.piezo_pos -= dz * piezo_factor
                        else:  # Alternate: save focus in current channel, apply focus to piezo for next channel
                            if np.isfinite(dz[ns.channels[cur_channel_idx]]):
                                zmem[ns.channels[cur_channel_idx]] = piezo_pos + dz[ns.channels[cur_channel_idx]] \
                                                                     * piezo_factor
                            elif not ns.channels[cur_channel_idx] in zmem:
                                zmem[ns.channels[cur_channel_idx]] = piezo_pos
                            if ns.channels[next_channel_idx] in zmem:
                                microscope.piezo_pos = zmem[ns.channels[next_channel_idx]]

                if xy_pos:  # Maybe move stage or ROI  # TODO: report pos wrt to frame in stead of roi which can move
                    xy = np.mean(list(xy_pos.values()), 0)
                    if ns.feedback_mode_xy == 1:
                        roi_pos += np.clip(xy - ns.roi_size / 2, -ns.max_step_xy, ns.max_step_xy)
                        ns.roi_pos = np.round(roi_pos).astype(int).tolist()

                    if ns.feedback_mode_xy == 2:
                        microscope.move_stage_relative(*np.clip(xy - ns.roi_size / 2, -ns.max_step_xy, ns.max_step_xy)
                                                       * microscope.pxsize / 1e3)

                # Wait for next frame:
                while microscope.time - 1 == time_n_now and microscope.is_experiment_running and not ns.stop \
                        and not ns.quit:
                    sleep(time_interval / 4)
                if time_n_now < time_n_prev:
                    break

                time_n_prev = time_n_now
                time_s_prev = time_s_now
                ns.run = False
        else:
            sleep(0.01)


class UiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        screen = QDesktopWidget().screenGeometry()
        self.title = 'Cylinder lens feedback GUI'
        self.width = 640
        self.height = 1024
        self.right = screen.width() - self.width
        self.top = 32

        with open(os.path.join(os.path.dirname(__file__), 'stylesheet.qss')) as style_sheet:
            self.setStyleSheet(style_sheet.read())

        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.setGeometry(self.right, self.top, self.width, self.height)

        self.central_widget = QWidget()
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget(self.central_widget)

        # tab 1
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, 'Main')

        self.stay_primed_box = QCheckBox('Stay primed')
        self.stay_primed_box.setToolTip('Stay primed')
        self.stay_primed_box.setEnabled(False)

        self.center_on_click_box = QCheckBox('Center on click')
        self.center_on_click_box.setToolTip('Push, then click on image')
        self.center_on_click_box.setEnabled(False)

        self.prime_btn = QPushButton('Prime for experiment')
        self.prime_btn.setToolTip('Prime for experiment')
        self.prime_btn.setEnabled(False)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setToolTip('Stop')
        self.stop_btn.setEnabled(False)

        self.feedback_mode_rdb = QGui.RadioButtons(('Zhuang', 'PID'))

        self.plots = QGui.PlotCanvas()
        self.eplot = QGui.SubPlot(self.plots, 611)
        self.eplot.append_plot('middle', ':', 'gray')
        self.eplot.append_plot('top', ':k')
        self.eplot.append_plot('bottom', ':k')
        self.eplot.ax.set_ylabel('Ellipticity')
        self.eplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.iplot = QGui.SubPlot(self.plots, 612)
        self.iplot.ax.set_ylabel('Intensity')
        self.iplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.splot = QGui.SubPlot(self.plots, 613)
        self.splot.ax.set_ylabel('Sigma (nm)')
        self.splot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.rplot = QGui.SubPlot(self.plots, 614)
        self.rplot.ax.set_ylabel('R squared')
        self.rplot.append_plot('bottom', ':k')
        self.rplot.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

        self.pplot = QGui.SubPlot(self.plots, 615)
        self.pplot.append_plot('fb', '-k')
        self.pplot.ax.set_xlabel('Time (frames)')
        self.pplot.ax.set_ylabel('Piezo pos (µm)')

        self.xyplot = QGui.SubPlot(self.plots, (6, 3, 16), '.r')
        self.xyplot.ax.set_xlabel('x (nm)')
        self.xyplot.ax.set_ylabel('y (nm)')
        self.xyplot.ax.invert_yaxis()
        self.xyplot.ax.set_aspect('equal', adjustable='datalim')

        self.xzplot = QGui.SubPlot(self.plots, (6, 3, 17), '.r')
        self.xzplot.ax.set_xlabel('x (nm)')
        self.xzplot.ax.set_ylabel('z (nm)')
        self.xzplot.ax.set_aspect('equal', adjustable='datalim')

        self.yzplot = QGui.SubPlot(self.plots, (6, 3, 18), '.r')
        self.yzplot.ax.set_xlabel('y (nm)')
        self.yzplot.ax.set_ylabel('z (nm)')
        self.yzplot.ax.set_aspect('equal', adjustable='datalim')

        main_buttons = QHBoxLayout()
        main_buttons.addWidget(self.stay_primed_box)
        main_buttons.addWidget(self.center_on_click_box)
        main_buttons.addWidget(self.prime_btn)
        main_buttons.addWidget(self.stop_btn)
        main_buttons.addWidget(self.feedback_mode_rdb)

        self.main_tab.layout = QVBoxLayout(self.main_tab)
        self.main_tab.layout.addLayout(main_buttons)
        self.main_tab.layout.addWidget(self.plots)

        # tab 2
        self.conf_tab = QWidget()
        self.tabs.addTab(self.conf_tab, 'Configuration')

        conf_grid = QGridLayout()
        conf_grid.setColumnStretch(0, 3)
        conf_grid.setColumnStretch(2, 3)

        r = 0
        conf_grid.addWidget(QLabel('Feedback channel:'), r, 0)
        self.channels_box = QGui.CheckBoxes(['{}'.format(i) for i in range(10)])
        for i, e in enumerate((664, 510, 583, 427)):
            self.channels_box.setTextBoxValue(i, e)
        conf_grid.addWidget(self.channels_box, r, 1)

        r += 1
        conf_grid.addWidget(QLabel('Feedback mode:'), r, 0)
        self.mult_channel_mode_drp = QComboBox()
        self.mult_channel_mode_drp.addItems(['Average', 'Alternate'])
        conf_grid.addWidget(self.mult_channel_mode_drp, r, 1)

        r += 1
        self.cyllens_drp = []
        conf_grid.addWidget(QLabel('Cylindrical lens back:'), r, 0)
        self.cyllens_drp.append(QComboBox())
        self.cyllens_drp[-1].addItems(['None', 'A', 'B'])
        conf_grid.addWidget(self.cyllens_drp[-1], r, 1)

        r += 1
        conf_grid.addWidget(QLabel('Cylindrical lens front:'), r, 0)
        self.cyllens_drp.append(QComboBox())
        self.cyllens_drp[-1].addItems(['None', 'A', 'B'])
        self.cyllens_drp[-1].setCurrentIndex(1)
        conf_grid.addWidget(self.cyllens_drp[-1], r, 1)

        r += 1
        conf_grid.addWidget(QLabel('Duolink filterset:'), r, 0)
        self.duolink_block_drp = QComboBox()
        self.duolink_block_drp.addItems(['488/_561_/640 & 488/_640_', '_561_/640 & empty'])
        conf_grid.addWidget(self.duolink_block_drp, r, 1)

        r += 1
        conf_grid.addWidget(QLabel('Duolink filter:'), r, 0)
        self.duolink_filter_rdb = QGui.RadioButtons(('1', '2'))
        conf_grid.addWidget(self.duolink_filter_rdb, r, 1)
        self.duolink_filter_lbl = QLabel()
        conf_grid.addWidget(self.duolink_filter_lbl, r, 2)

        r += 1
        conf_grid.addWidget(QLabel('XY feedback:'), r, 0)
        # self.xyrdb = QGui.RadioButtons(('None', 'Move ROI', 'Move stage'))  # Stage not accurate enough
        self.xy_feedback_mode_rdb = QGui.RadioButtons(('None', 'Move ROI'))
        conf_grid.addWidget(self.xy_feedback_mode_rdb, r, 1)

        r += 1
        conf_grid.addWidget(QLabel('ROI size:'), r, 0)
        self.roi_size_fld = QLineEdit()
        conf_grid.addWidget(self.roi_size_fld, r, 1)
        conf_grid.addWidget(QLabel('px'), r, 2)

        r += 1
        conf_grid.addWidget(QLabel('ROI position:'), r, 0)
        self.roi_pos_fld = QLineEdit()
        conf_grid.addWidget(self.roi_pos_fld, r, 1)
        conf_grid.addWidget(QLabel('px'), r, 2)

        r += 1
        conf_grid.addWidget(QLabel('Max stepsize x, y:'), r, 0)
        self.max_step_xy_fld = QLineEdit()
        conf_grid.addWidget(self.max_step_xy_fld, r, 1)
        conf_grid.addWidget(QLabel('px'), r, 2)

        r += 1
        conf_grid.addWidget(QLabel('Max stepsize z:'), r, 0)
        self.max_step_fld = QLineEdit()
        conf_grid.addWidget(self.max_step_fld, r, 1)
        conf_grid.addWidget(QLabel('um'), r, 2)

        r += 1
        self.calibrate_btn = QPushButton('Calibrate with beads')
        self.calibrate_btn.setToolTip('Calibrate with beads')
        conf_grid.addWidget(self.calibrate_btn, r, 1)
        self.calibrate_btn.setEnabled(False)

        r += 1
        self.warp_btn = QPushButton('Warp image file')
        self.warp_btn.setToolTip('Warp image channels so that colors overlap')
        conf_grid.addWidget(self.warp_btn, r, 1)

        self.conf_tab.setLayout(conf_grid)

        # tab 3
        self.map_tab = QWidget()
        self.tabs.addTab(self.map_tab, 'Map')

        self.map = QGui.SubPatchPlot(QGui.PlotCanvas(), color=(0.6, 1, 0.8))
        self.map.ax.invert_xaxis()
        self.map.ax.invert_yaxis()
        self.map.ax.set_xlabel('x')
        self.map.ax.set_ylabel('y')
        self.map.ax.set_aspect('equal', adjustable='datalim')

        self.reset_map_btn = QPushButton('Reset')

        self.map_tab.layout = QVBoxLayout(self.map_tab)
        self.map_tab.layout.addWidget(self.map.canvas)
        self.map_tab.layout.addWidget(self.reset_map_btn)

        self.layout.addWidget(self.tabs)

        # menus
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')

        self.open_action = QAction('&Open', self)
        self.open_action.setShortcut('Ctrl+O')
        self.open_action.setStatusTip('Open configuration')

        self.save_action = QAction('&Save', self)
        self.save_action.setShortcut('Ctrl+S')
        self.save_action.setStatusTip('Save configuration')

        self.saveas_action = QAction('Save &As', self)
        self.saveas_action.setShortcut('Ctrl+Shift+S')
        self.saveas_action.setStatusTip('Save configuration as')

        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.saveas_action)

        edit_menu = main_menu.addMenu('&Edit')

        self.calibrate_action = QAction('Calibrate', self)
        self.calibrate_action.setStatusTip('Calibrate feedback using a beadfile')
        self.calibrate_action.setEnabled(False)

        self.warp_action = QAction('Warp', self)
        self.warp_action.setStatusTip('Save a copy of a file where the warp is corrected')

        self.warp_with_action = QAction('Warp using file')
        self.warp_with_action.setStatusTip('Save a copy of a file where the warp is corrected and use a specific bead '
                                           'or transform file to do so')

        edit_menu.addAction(self.calibrate_action)
        edit_menu.addAction(self.warp_action)
        edit_menu.addAction(self.warp_with_action)

        self.setCentralWidget(self.central_widget)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()


class App(UiMainWindow):
    def __init__(self):
        self.queue = Queue()
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

        self.conf_filename = os.path.join(os.path.dirname(__file__), 'conf.yml')
        self.conf_open(self.conf_filename)
        self.microscope_class = self.conf.get('microscope', 'demo')
        self.microscope = MicroscopeClass(self.microscope_class)

        self.calibrating = False
        self.ellipse = {}
        self.rectangle = None
        self.magnification_str = self.microscope.magnification_str

        super().__init__()

        self.stay_primed_box.toggled.connect(self.stay_primed)
        self.center_on_click_box.toggled.connect(self.toggle_center_box)
        self.prime_btn.clicked.connect(self.prime)
        self.stop_btn.clicked.connect(self.set_stop)
        self.mult_channel_mode_drp.currentIndexChanged.connect(self.change_mult_channel_mode)
        for cyllensdrp in self.cyllens_drp:
            cyllensdrp.currentIndexChanged.connect(self.change_cyllens)
        self.duolink_block_drp.currentIndexChanged.connect(self.change_duolink_block)
        self.roi_size_fld.textChanged.connect(self.change_roi_size)
        self.roi_pos_fld.textChanged.connect(self.change_roi_pos)
        self.max_step_xy_fld.textChanged.connect(self.change_max_step_xy)
        self.max_step_fld.textChanged.connect(self.change_max_step)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.calibrate_action.triggered.connect(self.calibrate)
        self.warp_btn.clicked.connect(self.warp)
        self.reset_map_btn.clicked.connect(self.reset_map)
        self.channels_box.connect(self.change_channel, self.change_wavelength)
        self.feedback_mode_rdb.connect(self.change_feedback_mode)
        self.xy_feedback_mode_rdb.connect(self.change_xy_feedback_mode)
        self.duolink_filter_rdb.connect(self.change_duolink_filter)
        self.open_action.triggered.connect(self.conf_open)
        self.save_action.triggered.connect(partial(self.conf_save, self.conf_filename))
        self.saveas_action.triggered.connect(self.conf_save)
        self.warp_action.triggered.connect(self.warp)
        self.warp_with_action.triggered.connect(self.warp_using_file)

        self.fblprocess = Process(target=feedbackloop, args=(self.queue, self.NS, self.microscope_class))
        self.fblprocess.start()
        self.guithread = None

        self.microscope.wait(self, self.microscope_ready)

    def closeEvent(self, *args, **kwargs):
        self.set_quit()

    def microscope_ready(self, _):
        self.conf_load()
        self.duolink_filter_rdb.changeState(self.microscope.duolink_filter)
        self.change_cyllens()
        self.change_color()
        self.center_on_click_box.setEnabled(True)
        self.change_wavelengths()
        if len(self.NS.channels):
            self.stay_primed_box.setEnabled(True)
            self.prime_btn.setEnabled(True)
        self.events = self.microscope.events(self)

    def change_wavelengths(self):
        for channel, value in enumerate(self.channels_box.textBoxValues):
            self.change_wavelength(channel, value)

    def change_roi_size(self, val):
        try:
            self.NS.roi_size = float(val)
        except ValueError:
            pass

    def change_roi_pos(self, val):
        try:
            self.NS.roi_pos = [float(i) for i in re.findall(r'([-?\d.]+)[^-\d.]+([-?\d.]+)', val)[0]]
        except ValueError:
            pass

    def change_max_step_xy(self, val):
        try:
            self.NS.max_step_xy = float(val)
        except ValueError:
            pass

    def change_color(self):
        for channel, color in enumerate(self.microscope.channel_colors_rgb):
            if channel in self.NS.channels:
                for plot in (self.eplot, self.iplot, self.splot, self.rplot, self.pplot, self.xyplot, self.xzplot,
                             self.yzplot):
                    if channel in plot:
                        plot.plt[channel].set_color(color)
                for tb in ('top', 'bottom'):
                    cn = '{}{}'.format(tb, channel)
                    if cn in self.splot:
                        self.splot.plt[cn].set_color(color)
        camera = {name: self.microscope.get_camera(name) for name in self.microscope.channel_names}
        enabled = [False if self.NS.cyllens[c] == 'None' else True for c in camera.values()]
        self.channels_box.changeOptions(camera.keys(), self.microscope.channel_colors_hex, enabled)

    def change_wavelength(self, channel, val):
        try:
            wavelength = float(val)
            self.NS.sigma[channel] = wavelength / 2 / self.microscope.objective_na / self.microscope.pxsize
            fwhm = self.NS.sigma[channel] * 2 * np.sqrt(2 * np.log(2))
            self.NS.limits[channel] = self.Manager.list([[-np.inf, np.inf]] * 8)
            self.NS.limits[channel][2] = [fwhm / 2, fwhm * 2]  # fraction of fwhm
            self.NS.limits[channel][5] = [0.7, 1.3]  # ellipticity
            self.NS.limits[channel][7] = [0, np.inf]  # R2
        except ValueError:
            pass

    def change_cyllens(self):
        self.NS.cyllens = [i.currentText() for i in self.cyllens_drp]
        self.change_color()

    def calibrate(self, *args, **kwargs):
        if len(self.NS.channels) == 1:
            self.calibrating = True
            self.calibrate_btn.setEnabled(False)
            self.calibrate_action.setEnabled(False)
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            file, _ = QFileDialog.getOpenFileName(self, 'Beads for calibration', '',
                                                  'Carl Zeiss Image files (*.czi);;All files (*)', options=options)
            if file:

                self.calibz_thread = QThread(cyl.calibrate_z, self.calibrate_done, file,
                                             self.channels_box.textBoxValues, self.NS.channels[0], self.NS.cyllens,
                                             self.calibrate_progress)
                self.calibrate_progress(0)
            else:
                self.calibrating = False
                self.calibrate_btn.setEnabled(True)
                self.calibrate_action.setEnabled(True)

    def calibrate_progress(self, progress):
        self.calibrate_btn.setText(f'Calibrating: {progress:.0f}%')

    def calibrate_done(self, channel, magnification_str, theta, q):
        try:
            self.conf[self.get_cyllens(channel) + magnification_str]['theta'] = float(theta)
            self.conf[self.get_cyllens(channel) + magnification_str]['q'] = q.tolist()
            self.NS.theta[self.get_cm_str(channel)] = float(theta)
            self.NS.q[self.get_cyllens(channel) + magnification_str] = q.tolist()
            self.calibrating = False
            self.calibrate_btn.setText('Calibrate with beads')
            self.calibrate_btn.setEnabled(True)
            self.calibrate_action.setEnabled(True)
        except ValueError:
            pass

    def warp_using_file(self):
        options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        files, _ = QFileDialog.getOpenFileNames(self, 'Bead files or transform file', '',
                                                'Carl Zeiss Image files (*.czi);;Transform files (*.yml);;'
                                                'All files (*)', options=options)
        self.warp(transform_files=files)

    def warp(self, *, transform_files=None):
        options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        files, _ = QFileDialog.getOpenFileNames(self, 'Image files', '',
                                                'Carl Zeiss Image files (*.czi);;All files (*)', options=options)
        self.warp_btn.setEnabled(False)
        self.warp_action.setEnabled(False)
        self.warp_with_action.setEnabled(False)

        def warp_files(files):
            for file in files:
                if os.path.isfile(file):
                    warp(file, transform_files=transform_files)

        self.warp_thread = QThread(warp_files, self.warp_done, files)

    def warp_progress(self, progress):
        self.warp_btn.setText(f'Warping: {progress:.0f}%')

    def warp_done(self, *args, **kwargs):
        self.warp_btn.setText('Warp image file')
        self.warp_btn.setEnabled(True)
        self.warp_action.setEnabled(True)
        self.warp_with_action.setEnabled(True)

    def reset_map(self):
        self.map.remove_data()

    def conf_save(self, f):
        if not os.path.isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getSaveFileName(self, "Save config file", "", "Yaml files (*.yml);;All files (*)",
                                               options=options)
        if f:
            self.conf_filename = f
            self.conf['maxStep'] = self.NS.max_step
            self.conf['maxStepxy'] = self.NS.max_step_xy
            self.conf['ROISize'] = self.NS.roi_size
            self.conf['ROIPos'] = self.NS.roi_pos
            with open(f, 'w') as h:
                yaml.dump(self.conf, h, default_flow_style=None)

    def conf_open(self, f):
        if not os.path.isfile(f):
            options = (QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
            f, _ = QFileDialog.getOpenFileName(self, "Open config file", "", "Yaml files (*.yml);;All files (*)",
                                               options=options)
        if f:
            with open(f, 'r') as h:
                self.conf = yaml_load(h)
            self.conf_filename = f

    def conf_load(self):
        if 'cyllenses' in self.conf:
            values = ['None']
            if isinstance(self.conf['cyllenses'], (list, tuple)):
                values.extend(self.conf['cyllenses'])
            else:
                values.extend(re.split(r'\s?[,;]\s?', self.conf['cyllenses']))
            for drp in self.cyllens_drp:
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
        self.max_step_fld.setText(f'{self.NS.max_step}')
        self.NS.max_step_xy = self.conf.get('maxStepxy', 5)
        self.max_step_xy_fld.setText(f'{self.NS.max_step_xy}')
        self.NS.roi_size = self.conf.get('ROISize', 48)
        self.roi_size_fld.setText(f'{self.NS.roi_size}')
        self.NS.roi_pos = self.conf.get('ROIPos', [0, 0])
        self.roi_pos_fld.setText(f'{self.NS.roi_pos}')
        self.NS.gain = self.conf.get('gain', 5e-3)
        self.NS.fast_mode = self.conf.get('fastMode', False)

    def change_duolink_filter(self, val):
        if val == '1':
            self.microscope.duolink_filter = 0
        else:
            self.microscope.duolink_filter = 1
        self.change_duolink_block()

    def change_max_step(self, val):
        try:
            self.NS.max_step = float(val)
        except ValueError:
            pass

    def change_channel(self, val):
        if len(val):
            self.NS.channels = [int(v) for v in val]
            self.conf_load()
            self.stay_primed_box.setEnabled(True)
            self.prime_btn.setEnabled(True)
        else:
            self.NS.channels = []
            self.stay_primed_box.setEnabled(False)
            self.prime_btn.setEnabled(False)
        if not self.calibrating:
            self.calibrate_btn.setEnabled(len(self.NS.channels) == 1)

    def change_duolink_block(self, *args):
        self.duolink_filter_lbl.setText(
            self.duolink_block_drp.currentText().split(' & ')[self.microscope.duolink_filter])

    def change_feedback_mode(self, val):
        self.NS.feedback_mode = val.lower()

    def change_xy_feedback_mode(self, val):
        self.NS.feedback_mode_xy = self.xy_feedback_mode_rdb.txt.index(val)

    def change_mult_channel_mode(self, idx):
        self.NS.mult_channel_mode = idx

    def toggle_center_box(self):
        self.microscope.enable_event('LeftButtonDown')

    def get_cyllens(self, channel):
        return self.NS.cyllens[self.microscope.get_camera(self.microscope.channel_names[channel])]

    def get_cm_str(self, channel):
        return self.get_cyllens(channel) + self.microscope.magnification_str

    def prime(self):
        if self.guithread is None or not self.guithread.is_alive():
            self.NS.stop = False
            self.guithread = QThread(target=self.run)

    def stay_primed(self):
        if self.stay_primed_box.isChecked():
            self.prime()

    def run(self, *args, **kwargs):
        np.seterr(all='ignore')
        sleep_time = 0.02  # update interval

        self.prime_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.microscope.remove_drawings()
        self.rectangle = self.microscope.draw_rectangle(
            *functions.clip_rectangle(self.microscope.frame_size, *self.NS.roi_pos, self.NS.roi_size, self.NS.roi_size),
            index=self.rectangle)
        roi_pos = self.NS.roi_pos
        while True:
            self.prime_btn.setText('Wait for experiment to start')

            # First wait for the experiment to start:
            while (not self.microscope.is_experiment_running) and (not self.stop) and (not self.quit):
                self.rectangle = self.microscope.draw_rectangle(
                    *functions.clip_rectangle(self.microscope.frame_size, *self.NS.roi_pos, self.NS.roi_size,
                                              self.NS.roi_size), index=self.rectangle)
                sleep(sleep_time)

            for channel in self.NS.channels:
                for tb in ('top', 'bottom'):
                    cn = '{}{}'.format(tb, channel)
                    if cn not in self.splot:
                        self.splot.append_plot(cn, ':', self.microscope.channel_colors_rgb[channel])

                for plot in (self.eplot, self.iplot, self.splot, self.rplot, self.pplot):
                    if channel not in plot:
                        plot.append_plot(channel, '-', self.microscope.channel_colors_rgb[channel])

                for plot in (self.xyplot, self.xzplot, self.yzplot):
                    if channel not in plot:
                        plot.append_plot(channel, '.', self.microscope.channel_colors_rgb[channel])

            # Experiment has started:
            self.NS.run = True
            self.rectangle = self.microscope.draw_rectangle(
                *functions.clip_rectangle(self.microscope.frame_size, *self.NS.roi_pos, self.NS.roi_size,
                                          self.NS.roi_size),
                index=self.rectangle)

            cfilename = self.microscope.filename
            cfexists = os.path.isfile(cfilename)  # whether ZEN already made the czi file (streaming)
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
                        'feedback_mode_xy': self.NS.feedback_mode_xy,
                        'DLFilterSet': self.duolink_block_drp.currentText(),
                        'DLFilterChannel': self.microscope.duolink_filter, 'maxStepxy': self.NS.max_step_xy,
                        'maxStep': self.NS.max_step, 'ROISize': self.NS.roi_size, 'ROIPos': self.NS.roi_pos,
                        'Columns': ['channel', 'frame', 'piezoPos', 'focusPos',
                                    'x', 'y', 'fwhm', 'i', 'o', 'e', 'time']}

                for key in self.NS.q.keys():
                    conf[key] = {'q': self.NS.q[key], 'theta': self.NS.theta[key]}

                with open(pfilename, 'w') as file:
                    yaml.dump(conf, file, default_flow_style=None)
                    file.write('p:\n')

                    self.plots.remove_data()

                    width, height = self.microscope.frame_size
                    z0 = None

                    self.prime_btn.setText('Experiment started')

                    while (self.microscope.is_experiment_running or (not self.queue.empty())) and (not self.stop) \
                            and (not self.quit):
                        pxsize = self.microscope.pxsize

                        # Wait until feedbackloop analysed a new frame
                        while self.microscope.is_experiment_running and (not self.stop) and (not self.quit) \
                                and self.queue.empty():
                            sleep(sleep_time)
                        if not self.queue.empty():
                            Q = []
                            for i in range(20):
                                if not self.queue.empty():
                                    Q.append(self.queue.get())
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
                                idx = [i for i, e in enumerate(channels) if e == channel]
                                time_n, fitted, a, piezo_pos, focus_pos = [[Q[i][j] for i in idx] for j in range(1, 6)]

                                a = np.array(a)
                                if a.ndim > 1 and a.shape[1]:
                                    a[:, 2][a[:, 2] < self.NS.limits[channel][2][0]/2] = np.nan
                                    a[:, 2][a[:, 2] > self.NS.limits[channel][2][1]*2] = np.nan
                                    a[:, 5][a[:, 5] < self.NS.limits[channel][5][0]/2] = np.nan
                                    a[:, 5][a[:, 5] > self.NS.limits[channel][5][1]*2] = np.nan
                                    a[:, 7][a[:, 7] < self.NS.limits[channel][7][0]*2] = np.nan
                                    ridx = np.isnan(a[:, 3]) | np.isnan(a[:, 4])
                                    a[ridx, :] = np.nan

                                    zfit = np.array([-cyl.find_z(e, self.NS.q[self.get_cm_str(channel)])
                                                     if f else np.nan for e, f in zip(a[:, 5], fitted)])
                                    z = 1000 * (zfit + focus_pos - z0)

                                    self.eplot.range_data(time_n, a[:, 5], handle=channel)
                                    self.iplot.range_data(time_n, a[:, 3], handle=channel)
                                    self.splot.range_data(time_n, a[:, 2] / 2 / np.sqrt(2 * np.log(2)) * pxsize,
                                                          handle=channel)
                                    self.splot.range_data(time_n,
                                                          [self.NS.limits[channel][2][0]/2/np.sqrt(2*np.log(2))*pxsize]
                                                          * len(time_n), handle=f'bottom{channel}')
                                    self.splot.range_data(time_n,
                                                          [self.NS.limits[channel][2][1]/2/np.sqrt(2*np.log(2))*pxsize]
                                                          * len(time_n), handle=f'top{channel}')
                                    self.rplot.range_data(time_n, a[:, 7], handle=channel)
                                    self.pplot.range_data(time_n, zfit + focus_pos - z0, handle=channel)
                                    self.xyplot.append_data((a[:, 0] - self.NS.roi_size / 2) * pxsize,
                                                            (a[:, 1] - self.NS.roi_size / 2) * pxsize, channel)
                                    self.xzplot.append_data((a[:, 0] - self.NS.roi_size / 2) * pxsize, z, channel)
                                    self.yzplot.append_data((a[:, 1] - self.NS.roi_size / 2) * pxsize, z, channel)
                                    if channel == channels[0]:  # Draw these only once
                                        self.eplot.range_data(time_n, [1] * len(time_n), handle='middle')
                                        self.eplot.range_data(time_n, [self.NS.limits[channel][5][0]] * len(time_n),
                                                              handle='bottom')
                                        self.eplot.range_data(time_n, [self.NS.limits[channel][5][1]] * len(time_n),
                                                              handle='top')
                                        self.rplot.range_data(time_n, [self.NS.limits[channel][7][0]] * len(time_n),
                                                              handle='bottom')
                                        self.pplot.range_data(time_n, np.array(focus_pos) - z0, handle='fb')
                                    self.plots.draw()

                                    if fitted[-1]:
                                        x = float(a[-1, 0] + (width - self.NS.roi_size) / 2 + self.NS.roi_pos[0])
                                        y = float(a[-1, 1] + (height - self.NS.roi_size) / 2 + self.NS.roi_pos[1])
                                        radius = float(a[-1, 2])
                                        ellipticity = float(a[-1, 5])
                                        self.ellipse[channel] = self.microscope.draw_ellipse(
                                            x, y, radius, ellipticity, self.NS.theta[self.get_cm_str(channel)],
                                            self.microscope.channel_colors_int[channel],
                                            index=self.ellipse.get(channel))
                                        self.rectangle = self.microscope.draw_rectangle(
                                            *functions.clip_rectangle(self.microscope.frame_size, *self.NS.roi_pos,
                                                                      self.NS.roi_size, self.NS.roi_size),
                                            index=self.rectangle)
                                    else:
                                        self.microscope.remove_drawing(self.ellipse.get(channel))

                # After the experiment:
                for ellipse in self.ellipse.values():
                    self.microscope.remove_drawing(ellipse)
                if not cfexists:
                    for _ in range(5):
                        cfilename = functions.last_czi_file(self.conf.get('dataDir', r'd:\data'))
                        npfilename = os.path.splitext(cfilename)[0] + '.pzl'
                        if cfilename and not os.path.isfile(npfilename):
                            copyfile(pfilename, npfilename)
                            break
                        sleep(0.25)
                self.NS.roi_pos = roi_pos
            else:
                sleep(sleep_time)
            if not self.stay_primed_box.isChecked():
                break

        self.microscope.remove_drawing(self.rectangle)
        self.stop = False
        self.prime_btn.setText('Prime for experiment')
        self.stop_btn.setEnabled(False)
        self.prime_btn.setEnabled(True)
        self.NS.run = False

    def set_stop(self):
        # Stop being primed for an experiment
        self.stop_btn.setEnabled(False)
        self.stay_primed_box.setChecked(False)
        self.stop = True
        self.NS.stop = True

    def set_quit(self):
        # Quit the whole program
        self.set_stop()
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
