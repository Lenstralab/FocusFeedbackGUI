import win32com.client
import pythoncom
import numpy as np
from re import search
from time import sleep
from threading import get_ident
from functools import partial
from dataclasses import dataclass
from enum import Enum
from focusfeedbackgui.microscopes import MicroscopeClass
from focusfeedbackgui.utilities import QThread, error_wrap
from focusfeedbackgui.__main__ import App

# global property to keep track of indices of drawings in ZEN across threads
drawing_list = []
last_piezo_pos = None


class Cst(Enum):
    NoButtonMouseMove = 202
    LButtonMouseMove = 201
    LeftButtonDown = 203
    LeftButtonUp = 204
    RButtonMouseMove = 207
    RightButtonDown = 208
    RightButtonUp = 209

    Circle = 6
    ClosedArrow = 12
    ClosedBezier = 10
    ClosedPolyLine = 8
    Ellipse = 5
    Line = 2
    # None = 0
    OpenArrow = 11
    OpenBezier = 9
    OpenPolyLine = 7
    Palette = 14
    Rectangle = 4
    ScaleBar = 3
    Select = 1
    Text = 13


class Microscope(MicroscopeClass):
    def __init__(self, *args, event_handler=None, **kwargs):
        super().__init__()
        self.event_handler = event_handler
        self._zen_vba = {}
        self._id = {}
        self.progress_coord = self.vba.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates
        self.last_progress_coord = self.progress_coord()
        self.set_analog_mode(True)
        self.set_extended_range(False)

    def wait(self, app, callback):
        def fun():
            while not self.ready and not app.stop and not app.quit:
                sleep(0.1)
        self.wait_thread = QThread(fun, callback)

    def close(self):
        zen_id = get_ident()
        if zen_id in self._zen_vba:
            self._zen_vba.pop(zen_id)
        if zen_id in self._id:
            self._id.pop(zen_id)

    @staticmethod
    def events(app):
        def fun():
            with Microscope(event_handler=partial(event_handler_factory, app=app)) as z:
                m = Mem(z, app, [None, False], [None, False], False)
                z.enable_event('LeftButtonDown')
                while not app.quit:
                    sleep(.01)
                    pythoncom.PumpWaitingMessages()
                    m.checks()

        return QThread(fun)

    @property
    def zen(self):
        return self.reconnect()[0]

    @property
    def vba(self):
        return self.reconnect()[1]

    def reconnect(self):
        zen_id = get_ident()
        if zen_id not in self._zen_vba:
            pythoncom.CoInitialize()
            for zen_id, vbz_id in self._id.values():  # First try co-marshalling
                try:
                    zen = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(zen_id,
                                                                                            pythoncom.IID_IDispatch))
                    vba = win32com.client.Dispatch(pythoncom.CoGetInterfaceAndReleaseStream(vbz_id,
                                                                                            pythoncom.IID_IDispatch))
                    break
                except Exception:
                    continue
            else:
                zen = win32com.client.Dispatch('Zeiss.Micro.AIM.ApplicationInterface.ApplicationInterface')
                if self.event_handler is None:
                    vba = win32com.client.Dispatch('Lsm5Vba.Application')
                else:
                    vba = win32com.client.DispatchWithEvents('Lsm5Vba.Application', self.event_handler(self))
                self._id[zen_id] = (pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, zen),
                                    pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, vba))
            self._zen_vba[zen_id] = (zen, vba)
        return self._zen_vba[zen_id]

    @property
    def ready(self):
        return self.vba.Lsm5.DsRecordingActiveDocObject is not None

    def set_analog_mode(self, val=True):
        self.vba.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetAnalogMode(val)

    def set_extended_range(self, val=False):
        self.vba.Lsm5.ExternalCpObject().pHardwareObjects.pHighResFoc().bSetExtendedZRange(val)

    @property
    def drawing_list(self):
        global drawing_list
        return drawing_list

    @drawing_list.setter
    def drawing_list(self, val):
        global drawing_list
        drawing_list = val

    @property
    def current_doc(self):
        return self.vba.Lsm5.DsRecordingActiveDocObject

    def enable_event(self, event, doc=None):
        doc = doc or self.current_doc
        for i in range(25):
            if doc is not None:
                break
            sleep(0.2)
            doc = self.current_doc
        doc.EnableImageWindowEvent(Cst[event].value, True)

    def disable_event(self, event, doc=None):
        if not doc:
            doc = self.current_doc
        if doc:
            doc.EnableImageWindowEvent(Cst[event].value, False)

    @property
    def is_busy(self):
        return self.vba.Lsm5.DsRecordingActiveDocObject.is_busy() if self.ready else True

    @property
    def mouse_pos(self):
        if self.ready:
            x = self.vba.Lsm5.DsRecordingActiveDocObject.GetCurrentMousePosition()
            return x[5] + 1, x[4] + 1
        else:
            return 0, 0

    @property
    def objective_magnification(self):
        try:
            return float(search(r'\d+x', self.zen.GUI.Acquisition.AcquisitionMode.Objective.ByName).group()[:-1])
        except Exception:
            return 0

    @property
    def objective_na(self):
        try:
            return float(search(r'\d\.\d+', self.zen.GUI.Acquisition.AcquisitionMode.Objective.ByName).group())
        except Exception:
            return 1

    @property
    def optovar_magnification(self):
        try:
            return float(self.zen.GUI.Acquisition.LightPath.TubeLens.ByName[5:-1].replace(',', '.'))
        except Exception:
            return 0

    @property
    def pxsize(self):
        return 1e9 * self.zen.GUI.Document.DsRecordingDoc.VoxelSizeX

    @property
    def filename(self):
        return self.zen.GUI.Document.FullFileName.value

    @property
    def title(self):
        if self.ready:
            return self.vba.Lsm5.DsRecordingActiveDocObject.title()
        else:
            return ''

    @property
    def focus_pos(self):
        return self.vba.Lsm5.Hardware().CpFocus().Position

    @property
    def frame_size(self):
        if self.ready:
            scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
            return scan_doc.GetDimensionX(), scan_doc.GetDimensionY()
        else:
            return 0, 0

    def get_frame_centered(self, channel=1, size=32):
        size = 2 * int((size + 1) / 2)  # Ensure evenness
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        x_max = scan_doc.GetDimensionX()
        y_max = scan_doc.GetDimensionY()

        x0 = (x_max - size) / 2
        y0 = (y_max - size) / 2

        time = self.time
        if time > 0:
            time -= 1
        return np.reshape(scan_doc.GetSubregion(channel, x0, y0, 0, time, 1, 1, 1, 1, size, size, 1, 1, 2)[0],
                          (size, size)), time

    def get_frame(self, channel=1, x=None, y=None, width=32, height=32):
        width = 2 * int((width + 1) / 2)  # Ensure evenness
        height = 2 * int((height + 1) / 2)
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        x_max = scan_doc.GetDimensionX()
        y_max = scan_doc.GetDimensionY()
        if x is None:
            x = int((x_max - width) / 2)
        else:
            x -= int(width / 2)
        if y is None:
            y = int((y_max - height) / 2)
        else:
            y -= int(height / 2)

        time = self.time
        if time > 0:
            time -= 1
        return np.reshape(scan_doc.GetSubregion(channel, x, y, 0, time, 1, 1, 1, 1, width, height, 1, 1, 2)[0],
                          (width, height)), time

    @property
    def channel_colors_int(self):
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        return [scan_doc.ChannelColor(channel) for channel in range(self.n_channels)]

    @property
    def channel_names(self):
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        return [scan_doc.ChannelName(i) for i in range(self.n_channels)]

    @property
    def n_channels(self):
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        return scan_doc.GetDimensionChannels()

    @property
    def time(self):
        return self.vba.Lsm5.ExternalDsObject().ScanController().GetProgressCoordinates()[2]

    def wait_for_next_frame(self, timeout=5):
        sleep_time = 0.02
        wait_time = 0.0
        while (self.progress_coord() == self.last_progress_coord) & (
                timeout > wait_time) & self.is_experiment_running():
            wait_time += sleep_time
            sleep(sleep_time)
        self.last_progress_coord = self.progress_coord()

    @property
    def is_experiment_running(self):
        scan_doc = self.vba.Lsm5.DsRecordingActiveDocObject
        return self.zen.GUI.Acquisition.IsExperimentRunning.value \
               and scan_doc.GetDimensionTime() > 1 >= scan_doc.GetDimensionZ()

    @property
    def is_z_stack(self):
        return self.vba.Lsm5.DsRecordingActiveDocObject.GetDimensionZ() > 1

    @property
    def is_time_series(self):
        return self.vba.Lsm5.DsRecordingActiveDocObject.GetDimensionTime() > 1

    @property
    def time_interval(self):
        # in s
        time_interval = self.zen.GUI.Acquisition.TimeSeries.Interval.value
        unit = self.zen.GUI.Acquisition.TimeSeries.IntervalTimeUnit.ByIndex
        if unit == 0:
            time_interval *= 60
        elif unit == 2:
            time_interval /= 1000

        if time_interval == 0:
            self.zen.GUI.Acquisition.Channels.Track.ByIndex = 0
            self.zen.GUI.Acquisition.Channels.Track.Channel.ByIndex = 0
            time_interval = self.zen.GUI.Acquisition.Channels.Track.Channel.Camera.ExposureTime.value / 1000
        return time_interval

    def draw_ellipse(self, x, y, radius, ellipticity, theta, color=65025, LineWidth=2, index=None):
        if self.ready and np.all(np.isfinite((x, y, radius, ellipticity, theta))):
            # x, y, radius in pixels, theta in rad
            if ellipticity <= 0:
                ellipticity = 1
            overlay = self.vba.Lsm5.DsRecordingActiveDocObject.VectorOverlay()
            index = self.manipulate_drawing_list(overlay, index)
            overlay.LineWidth = LineWidth
            overlay.Color = color  # Green
            overlay.AddDrawingElement(5, 3, (x, x + radius * np.sqrt(ellipticity) * np.cos(theta),
                                             x + radius / np.sqrt(ellipticity) * np.sin(theta)),
                                      (y, y - radius * np.sqrt(ellipticity) * np.sin(theta),
                                       y + radius / np.sqrt(ellipticity) * np.cos(theta)))
            return index
        else:
            self.remove_drawing(index)
            return None

    def draw_rectangle(self, x, y, width, height, color=16777215, LineWidth=2, index=None):
        if self.ready and np.all(np.isfinite((x, y, width, height))):
            overlay = self.vba.Lsm5.DsRecordingActiveDocObject.VectorOverlay()
            index = self.manipulate_drawing_list(overlay, index)
            overlay.LineWidth = LineWidth
            overlay.Color = color  # Green
            overlay.AddDrawingElement(4, 2, (float(x - width / 2), float(x + width / 2)),
                                      (float(y - height / 2), float(y + height / 2)))
            return index
        else:
            self.remove_drawing(index)
            return None

    def remove_drawings(self):
        if self.ready:
            self.vba.Lsm5.DsRecordingActiveDocObject.VectorOverlay().RemoveAllDrawingElements()
        self.drawing_list = []

    def remove_drawing(self, index):
        if self.ready:
            overlay = self.vba.Lsm5.DsRecordingActiveDocObject.VectorOverlay()
            lst = self.drawing_list
            if overlay.GetNumberDrawingElements() != len(lst):
                lst = list(range(overlay.GetNumberDrawingElements()))
            if index in lst:
                overlay.RemoveDrawingElement(lst.index(index))
                lst.remove(index)
                self.drawing_list = lst
        else:
            self.drawing_list = []

    def manipulate_drawing_list(self, overlay, index):
        lst = self.drawing_list
        if overlay.GetNumberDrawingElements() != len(lst):
            lst = list(range(overlay.GetNumberDrawingElements()))
        if index is not None:
            if index in lst:
                overlay.RemoveDrawingElement(lst.index(index))
                lst.remove(index)
        else:
            if not lst:
                index = 0
            else:
                index = max(lst) + 1
        lst.append(index)
        self.drawing_list = lst
        return index

    def read_piezo_pos(self):
        # in um
        p = [self.vba.Lsm5.Hardware().CpHrz().Position]
        for i in range(200):
            sleep(0.01)
            p.append(self.vba.Lsm5.Hardware().CpHrz().Position)
        global last_piezo_pos
        last_piezo_pos = np.mean(np.unique(p))

    @property
    def piezo_pos(self):
        # in um
        global last_piezo_pos
        if last_piezo_pos is None:
            self.read_piezo_pos()
        return last_piezo_pos

    @piezo_pos.setter
    def piezo_pos(self, z):
        # in um
        if z > 250:
            z = 250
        elif z < -250:
            z = -250
        self.vba.Lsm5.Hardware().CpHrz().Position = z
        global last_piezo_pos
        last_piezo_pos = z
        return

    @property
    def stage_pos(self):
        # in um
        return self.vba.Lsm5.Hardware().CpStages().PositionX, self.vba.Lsm5.Hardware().CpStages().PositionY

    @stage_pos.setter
    def stage_pos(self, pos):
        x, y = pos
        if x is not None:
            lim = self.vba.Lsm5.Hardware().CpStages().UpperLimitX
            self.vba.Lsm5.Hardware().CpStages().PositionX = np.clip(x, -lim, lim)
        if y is not None:
            lim = self.vba.Lsm5.Hardware().CpStages().UpperLimitY
            self.vba.Lsm5.Hardware().CpStages().PositionY = np.clip(y, -lim, lim)

    @property
    def duolink_filter(self):
        filter_set = self.vba.Lsm5.Hardware().CpFilterSets()
        filter_set.Select('2C_FilterSlider')
        return filter_set.FilterSetPosition - 1

    @duolink_filter.setter
    def duolink_filter(self, position):
        filter_set = self.vba.Lsm5.Hardware().CpFilterSets()
        filter_set.Select('2C_FilterSlider')
        filter_set.FilterSetPosition = position + 1


class EventHandlerMetaClass(type):
    """
    A metaclass for event handlers that don't respond to all events.
    Without this an error would be raised by win32com when it tries
    to call an event handler method that isn't defined by the event
    handler instance.
    """
    @staticmethod
    def null_event_handler(event, *args, **kwargs):
        print(('Unhandled event {}'.format(event), args, kwargs))
        return None

    def __new__(mcs, name, bases, dictionary):
        # Construct the new class.
        cls = type.__new__(mcs, name, bases, dictionary)

        # Create dummy methods for any missing event handlers.
        cls._dispid_to_func_ = getattr(cls, "_dispid_to_func_", {})
        for dispid, name in cls._dispid_to_func_.items():
            func = getattr(cls, name, None)
            if func is None:
                setattr(cls, name, partial(EventHandlerMetaClass.null_event_handler, name))
        return cls


def event_handler_factory(zen, app):
    class EventHandlerCls(metaclass=EventHandlerMetaClass):
        def __init__(self):
            self.app = app
            self.zen = zen
            self.unique_events = set()

        @error_wrap
        def OnThrowEvent(self, *args):
            if args[0] == Cst.Text.value:
                return
            # print(('OnThrowEvent:', args))
            if args[0] == Cst.LeftButtonDown.value and self.app.center_on_click_box.isChecked():
                mouse_pos = self.zen.mouse_pos
                frame_size = self.zen.frame_size
                pxsize = self.zen.pxsize
                d = [(frame_size[i] / 2 - mouse_pos[i] + self.app.NS.roi_pos[i]) * pxsize / 1000 for i in range(2)]
                d[1] *= -1
                self.zen.move_stage_relative(*d)

        @error_wrap
        def OnThrowPropertyEvent(self, *args):
            # if args not in self.unique_events:
            #     self.unique_events.add(args)
            #     print(('OnThrowPropertyEvent:', args))
            if args[1] == '2C_FilterSlider' and self.app.duolink_filter != self.zen.duolink_filter:
                self.app.duolink_filter = self.zen.duolink_filter
                self.app.duolink_filter_lbl.setText(
                    self.app.duolink_block_drp.currentText().split(' & ')[self.zen.duolink_filter])
                self.app.duolink_filter_rdb.changeState(self.zen.duolink_filter)
            elif args[1] == 'Stage':
                last_pos = self.zen.stage_pos
                frame_size = self.zen.frame_size
                pxsize = self.zen.pxsize
                self.app.map.numel_data(last_pos[0] / 1000, last_pos[1] / 1000, frame_size[0] * pxsize / 1e6,
                                        frame_size[1] * pxsize / 1e6)
                self.app.map.draw()
            elif args[1] in ('DataColorPalette', 'FramesPerStack'):
                self.app.change_color()
            elif args[1] == 'TransmissionSpot':
                self.app.change_wavelengths()
    return EventHandlerCls


@dataclass
class Mem:
    zen: Microscope
    app: App
    previous_zen: list
    current_zen: list
    experiment_running: bool

    def title(self):
        return '' if self.current_zen[0] is None else self.current_zen[0].title()

    def checks(self):  # events that don't throw an event
        if self.title != self.zen.title:  # current document changed
            self.current_zen = [self.zen.current_doc, False]  # ActiveDocObject, LMB enabled
            self.zen.enable_event('LeftButtonDown')

        # draw a black rectangle in the map after an experiment started
        experiment_running, self.experiment_running = self.experiment_running, self.zen.is_experiment_running
        if not experiment_running and self.experiment_running:
            last_pos = self.zen.stage_pos
            frame_size = self.zen.frame_size
            pxsize = self.zen.pxsize
            self.app.map.append_data_docs(last_pos[0] / 1000, last_pos[1] / 1000, frame_size[0] * pxsize / 1e6,
                                          frame_size[1] * pxsize / 1e6)
            self.app.map.draw()

        # the LMB event is not enabled on the current doc
        if not self.current_zen[1] and self.app.centerbox.isChecked():
            if not self.previous_zen[0] is None:
                self.zen.disable_event('LeftButtonDown', self.previous_zen[0])
            if not self.current_zen[0] is None:
                self.zen.enable_event('LeftButtonDown', self.current_zen[0])
                self.current_zen[1] = True

        if self.app.magnification_str != self.zen.magnification_str:  # Some things in the optical path have changed
            self.app.magnification_str = self.zen.magnification_str
            self.app.confload()
